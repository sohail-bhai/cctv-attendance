from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import webbrowser
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd

from .config import (
    DEFAULT_DETECTION_SCORE,
    DEFAULT_FRAME_SKIP,
    DEFAULT_MAX_WIDTH,
    DEFAULT_NMS_THRESHOLD,
    DEFAULT_TOP_K,
)
from .diagnostics import json_safe, make_diagnostic_run_id
from .recall_analysis import (
    MARGIN_THRESHOLD,
    MATCH_THRESHOLD,
    PRESENT_CHECKPOINTS,
    canonical_roll,
)
from .session_identity import build_attendance_session_id, parse_attendance_session_id
from .tracklet_calibration import CalibrationCandidate, candidate_acceptance_mask
from .tracklet_review import (
    LABEL_COLUMNS,
    LabelValidationError,
    ReviewExportError,
    ReviewPackageResult,
    _sha256_file,
    _yes,
    export_review_package,
    load_review_package_data,
    load_student_mapping,
    validate_label_dataframe,
)


SHADOW_SCHEMA_VERSION = 1
EXPECTED_CANDIDATE_ID = "cal-5cd35b60dd83"
EXPECTED_CANDIDATE_CONFIG_SHA256 = (
    "60b065788d9a12c900c4259685d6c66d97bb1e731b715e9e96bb91b7afe9afa4"
)
EXPECTED_SUBJECT = "CVO"
EXPECTED_ROSTER_COUNT = 27
MIN_CORRECT_RECOVERIES = 3
MIN_CORRECT_IDENTITIES = 2
MIN_CORRECT_CHECKPOINTS = 2

REQUIRED_TRACKLET_COLUMNS = {
    "Session_ID",
    "Subject_Abbr",
    "Tracklet_Mode",
    "Tracklet_ID",
    "Checkpoint_ID",
    "Camera_ID",
    "Video",
    "Tracklet_Accepted",
    "Tracklet_Eligible",
    "Tracklet_Quality_Rejection",
    "Tracklet_Diagnostic_Reason",
    "Tracklet_Matcher_Reason",
    "Tracklet_Best_Roll",
    "Tracklet_Best_Score",
    "Tracklet_Second_Roll",
    "Tracklet_Second_Score",
    "Tracklet_Margin",
    "Dominant_Frame_Best_Roll",
    "Dominant_Frame_Best_Share_Pct",
    "Observation_Count",
    "Selected_Observation_Count",
    "Embedding_Count",
    "Consistent_Embedding_Count",
    "Inconsistent_Embedding_Count",
    "Pairwise_Similarity_Median",
    "Selected_Observation_IDs",
    "Zone_IDs",
    "Match_Threshold",
    "Margin_Threshold",
    "Aggregate_Mode",
    "Official_Attendance_Contribution",
}

REFERENCE_METADATA_KEYS = (
    "checkpoint_mode",
    "log_mode",
    "aggregate",
    "match_threshold",
    "margin_threshold",
    "sample_fps",
    "frame_skip",
    "zone_mode",
    "camera_zones",
    "zone_profile",
    "zone_merge_iou",
    "tracklet_mode",
    "tracklet_min_observations",
    "tracklet_max_selected",
    "tracklet_max_gap_seconds",
    "tracklet_min_iou",
    "tracklet_max_center_ratio",
    "tracklet_min_size_ratio",
    "tracklet_min_embedding_similarity",
    "official_recognition_unit",
)


class ShadowValidationError(ValueError):
    pass


@dataclass(frozen=True)
class FrozenCandidate:
    config_path: Path
    manifest_path: Path
    config_sha256: str
    payload: dict[str, Any]
    candidate: CalibrationCandidate


@dataclass(frozen=True)
class ShadowSessionPreflight:
    session_id: str
    session_fields: dict[str, str]
    subject_abbr: str
    video_root: Path
    videos: list[dict[str, Any]]
    student_map_path: Path
    roster_sha256: str
    subject_rolls: set[str]
    subject_roster: list[dict[str, Any]]
    frozen_candidate: FrozenCandidate
    reference_diagnostic_run: Path
    reference_configuration: dict[str, Any]


@dataclass(frozen=True)
class ShadowEvaluationResult:
    tracks: pd.DataFrame
    summary: dict[str, Any]
    camera_checkpoint_breakdown: pd.DataFrame


@dataclass(frozen=True)
class ShadowSessionRunResult:
    diagnostic_run: Path
    output_dir: Path
    review_package: ReviewPackageResult
    decisions: pd.DataFrame
    summary: dict[str, Any]
    recognition_rerun: bool
    runtime_seconds: float


def _text(value: Any) -> str:
    return str(value or "").strip()


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _single_file(directory: Path, pattern: str, label: str) -> Path:
    matches = sorted(Path(directory).glob(pattern))
    if len(matches) != 1:
        raise ShadowValidationError(
            f"Expected exactly one {label} matching {pattern} in {directory}; found {len(matches)}"
        )
    return matches[0]


def _require_exact_bool(payload: dict[str, Any], key: str, expected: bool) -> None:
    if payload.get(key) is not expected:
        expected_text = "true" if expected else "false"
        raise ShadowValidationError(f"Frozen candidate requires {key}={expected_text}")


def load_frozen_candidate(
    config_path: Path,
    manifest_path: Path,
    *,
    subject_abbr: str,
    expected_candidate_id: str = EXPECTED_CANDIDATE_ID,
    expected_config_sha256: str = EXPECTED_CANDIDATE_CONFIG_SHA256,
) -> FrozenCandidate:
    config_path = Path(config_path)
    manifest_path = Path(manifest_path)
    if not config_path.is_file():
        raise ShadowValidationError(f"Frozen candidate config not found: {config_path}")
    if not manifest_path.is_file():
        raise ShadowValidationError(f"Frozen candidate output manifest not found: {manifest_path}")
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ShadowValidationError(f"Could not read frozen candidate artifacts: {exc}") from exc

    config_hash = _sha256_file(config_path)
    if config_hash != expected_config_sha256:
        raise ShadowValidationError(
            "Frozen candidate config does not match the approved SHA-256: "
            f"expected {expected_config_sha256}, got {config_hash}"
        )
    recorded_hash = _text(manifest.get("files_sha256", {}).get(config_path.name))
    if not recorded_hash or recorded_hash != config_hash:
        raise ShadowValidationError(
            f"Frozen candidate config SHA-256 mismatch: expected {recorded_hash or 'missing'}, got {config_hash}"
        )
    if _text(manifest.get("candidate_id")) != expected_candidate_id:
        raise ShadowValidationError("Frozen candidate manifest candidate_id does not match the requested candidate")
    if manifest.get("production_changes") is not False:
        raise ShadowValidationError("Frozen candidate manifest must record production_changes=false")

    if payload.get("enabled") is not False:
        raise ShadowValidationError("Frozen candidate must remain disabled (enabled=false)")
    _require_exact_bool(payload, "production_approved", False)
    _require_exact_bool(payload, "requires_multisession_validation", True)
    _require_exact_bool(payload, "applies_only_to_baseline_rejected_tracklets", True)
    if _text(payload.get("mode")) != "shadow_only":
        raise ShadowValidationError("Frozen candidate mode must remain shadow_only")
    if _text(payload.get("subject_abbr")).upper() != _text(subject_abbr).upper():
        raise ShadowValidationError("Frozen candidate subject does not match the requested subject")
    if _text(payload.get("candidate_id")) != expected_candidate_id:
        raise ShadowValidationError("Frozen candidate config candidate_id does not match the requested candidate")

    try:
        candidate = CalibrationCandidate(**dict(payload.get("thresholds") or {}))
    except (TypeError, ValueError) as exc:
        raise ShadowValidationError(f"Frozen candidate threshold schema is invalid: {exc}") from exc
    if candidate.candidate_id != expected_candidate_id:
        raise ShadowValidationError(
            f"Frozen thresholds recompute to {candidate.candidate_id}, not {expected_candidate_id}"
        )
    if not candidate.require_dominant_agreement or not candidate.require_subject_roster_candidate:
        raise ShadowValidationError("Frozen candidate safety guards cannot be disabled")
    if candidate.min_selected_observations != 3 or candidate.min_consistent_embeddings != 3:
        raise ShadowValidationError("Frozen candidate minimum evidence guards have changed")

    guards = dict(payload.get("fixed_safety_guards") or {})
    expected_guards = {
        "candidate_must_be_in_authoritative_subject_roster": True,
        "aggregate_candidate_must_equal_dominant_frame_candidate": True,
        "minimum_selected_observations": 3,
        "minimum_consistent_embeddings": 3,
    }
    if guards != expected_guards:
        raise ShadowValidationError("Frozen candidate fixed_safety_guards do not match the approved snapshot")
    if _number(payload.get("official_match_threshold_unchanged"), -1) != MATCH_THRESHOLD:
        raise ShadowValidationError("Frozen candidate official match threshold is not 0.48")
    if _number(payload.get("official_margin_threshold_unchanged"), -1) != MARGIN_THRESHOLD:
        raise ShadowValidationError("Frozen candidate official margin threshold is not 0.08")
    if int(_number(payload.get("attendance_checkpoint_rule_unchanged"), -1)) != PRESENT_CHECKPOINTS:
        raise ShadowValidationError("Frozen candidate attendance checkpoint rule is not 3")
    return FrozenCandidate(
        config_path=config_path.resolve(),
        manifest_path=manifest_path.resolve(),
        config_sha256=config_hash,
        payload=payload,
        candidate=candidate,
    )


def _opencv_video_metadata(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ShadowValidationError(f"Unreadable video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise ShadowValidationError(f"Video metadata is incomplete or invalid: {path}")
    return {
        "fps": round(fps, 6),
        "frame_count": frame_count,
        "duration_seconds": round(frame_count / fps, 6),
        "width": width,
        "height": height,
    }


def discover_session_videos(
    video_root: Path,
    *,
    metadata_reader: Callable[[Path], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    video_root = Path(video_root)
    if not video_root.is_dir():
        raise ShadowValidationError(f"Prepared footage directory not found: {video_root}")
    checkpoint_dirs = sorted(path for path in video_root.iterdir() if path.is_dir())
    if len(checkpoint_dirs) != 5:
        raise ShadowValidationError(
            f"Session preflight requires exactly five checkpoint directories; found {len(checkpoint_dirs)}"
        )
    by_id: dict[str, Path] = {}
    for directory in checkpoint_dirs:
        prefix = directory.name.split("_", 1)[0].upper()
        if prefix not in {f"CP{number}" for number in range(1, 6)} or prefix in by_id:
            raise ShadowValidationError(f"Invalid or duplicate checkpoint directory: {directory.name}")
        by_id[prefix] = directory
    expected_ids = {f"CP{number}" for number in range(1, 6)}
    if set(by_id) != expected_ids:
        raise ShadowValidationError(
            "Checkpoint directories must resolve exactly to CP1-CP5; found " + ", ".join(sorted(by_id))
        )

    reader = metadata_reader or _opencv_video_metadata
    rows: list[dict[str, Any]] = []
    for checkpoint_id in sorted(expected_ids, key=lambda value: int(value[2:])):
        directory = by_id[checkpoint_id]
        files = sorted(path for path in directory.iterdir() if path.is_file())
        names = [path.name.lower() for path in files]
        if names != ["back.mp4", "front.mp4"]:
            raise ShadowValidationError(
                f"{directory.name} must contain exactly back.mp4 and front.mp4; found {', '.join(names) or 'none'}"
            )
        for path in files:
            size = path.stat().st_size
            if size <= 0:
                raise ShadowValidationError(f"Video file is empty: {path}")
            metadata = dict(reader(path))
            required_metadata = {"fps", "frame_count", "duration_seconds", "width", "height"}
            missing = sorted(required_metadata.difference(metadata))
            if missing:
                raise ShadowValidationError(
                    f"Video metadata missing {', '.join(missing)} for {path}"
                )
            rows.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_directory": directory.name,
                    "source_id": path.stem.lower(),
                    "source_file_name": path.name,
                    "source_path": str(path.resolve()),
                    "file_size_bytes": size,
                    "sha256": _sha256_file(path),
                    **metadata,
                }
            )
    if len(rows) != 10:
        raise ShadowValidationError(f"Session preflight requires ten videos; found {len(rows)}")
    return rows


def load_reference_configuration(reference_diagnostic_run: Path) -> dict[str, Any]:
    reference_diagnostic_run = Path(reference_diagnostic_run)
    if not reference_diagnostic_run.is_dir():
        raise ShadowValidationError(
            f"Reference diagnostic run not found: {reference_diagnostic_run}"
        )
    summary_path = _single_file(
        reference_diagnostic_run,
        "diagnostic_summary_*.json",
        "reference diagnostic summary",
    )
    tracklet_path = _single_file(
        reference_diagnostic_run,
        "tracklet_diagnostics_*.csv",
        "reference tracklet diagnostics",
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metadata = dict(summary.get("run_metadata") or {})
    missing_metadata = [key for key in REFERENCE_METADATA_KEYS if key not in metadata]
    if missing_metadata:
        raise ShadowValidationError(
            "Reference run lacks required historical parameters: " + ", ".join(missing_metadata)
        )
    expected_modes = {
        "checkpoint_mode": "clip-folders",
        "log_mode": "full",
        "aggregate": "top3",
        "zone_mode": "compare",
        "tracklet_mode": "compare",
        "official_recognition_unit": "frame_detection",
    }
    for key, expected in expected_modes.items():
        if _text(metadata.get(key)) != expected:
            raise ShadowValidationError(
                f"Reference run {key}={metadata.get(key)!r}; expected {expected!r}"
            )
    if _number(metadata.get("match_threshold"), -1) != MATCH_THRESHOLD:
        raise ShadowValidationError("Reference run match threshold is not 0.48")
    if _number(metadata.get("margin_threshold"), -1) != MARGIN_THRESHOLD:
        raise ShadowValidationError("Reference run margin threshold is not 0.08")
    reference_columns = set(pd.read_csv(tracklet_path, nrows=0).columns)
    missing_columns = sorted(REQUIRED_TRACKLET_COLUMNS.difference(reference_columns))
    if missing_columns:
        raise ShadowValidationError(
            "Reference tracklet feature schema is incomplete: " + ", ".join(missing_columns)
        )
    snapshot = {key: metadata[key] for key in REFERENCE_METADATA_KEYS}
    snapshot.update(
        {
            "detection_score": DEFAULT_DETECTION_SCORE,
            "nms_threshold": DEFAULT_NMS_THRESHOLD,
            "top_k": DEFAULT_TOP_K,
            "max_width": DEFAULT_MAX_WIDTH,
            "reference_diagnostic_run": str(reference_diagnostic_run.resolve()),
            "reference_summary_sha256": _sha256_file(summary_path),
            "reference_tracklet_schema_sha256": hashlib.sha256(
                "\n".join(sorted(reference_columns)).encode("utf-8")
            ).hexdigest(),
            "required_feature_columns": sorted(REQUIRED_TRACKLET_COLUMNS),
        }
    )
    return snapshot


def preflight_shadow_session(
    *,
    video_root: Path,
    session_id: str,
    subject_abbr: str,
    student_map_path: Path,
    candidate_config_path: Path,
    candidate_manifest_path: Path,
    reference_diagnostic_run: Path,
    expected_roster_count: int = EXPECTED_ROSTER_COUNT,
) -> ShadowSessionPreflight:
    try:
        session_fields = parse_attendance_session_id(session_id)
    except ValueError as exc:
        raise ShadowValidationError(str(exc)) from exc
    expected_session_id = build_attendance_session_id(
        session_fields["session_date"],
        session_fields["section"],
        session_fields["period"],
        subject_abbr,
    )
    if session_id != expected_session_id:
        raise ShadowValidationError(
            f"Session ID must resolve exactly to {expected_session_id}; got {session_id}"
        )
    if session_fields["period"] != "P2":
        raise ShadowValidationError("Phase 1.2H target period must be P2")
    videos = discover_session_videos(video_root)
    try:
        roster, _, subject_rolls_raw = load_student_mapping(Path(student_map_path), subject_abbr)
    except ReviewExportError as exc:
        raise ShadowValidationError(str(exc)) from exc
    subject_roster = [row for row in roster if row["in_subject_roster"]]
    subject_rolls = {canonical_roll(value) for value in subject_rolls_raw if canonical_roll(value)}
    if len(subject_rolls) != expected_roster_count or len(subject_roster) != expected_roster_count:
        raise ShadowValidationError(
            f"Authoritative {subject_abbr.upper()} roster must contain {expected_roster_count} students; "
            f"found {len(subject_rolls)}"
        )
    frozen_candidate = load_frozen_candidate(
        candidate_config_path,
        candidate_manifest_path,
        subject_abbr=subject_abbr,
    )
    reference_configuration = load_reference_configuration(reference_diagnostic_run)
    return ShadowSessionPreflight(
        session_id=session_id,
        session_fields=session_fields,
        subject_abbr=_text(subject_abbr).upper(),
        video_root=Path(video_root).resolve(),
        videos=videos,
        student_map_path=Path(student_map_path).resolve(),
        roster_sha256=_sha256_file(Path(student_map_path)),
        subject_rolls=subject_rolls,
        subject_roster=subject_roster,
        frozen_candidate=frozen_candidate,
        reference_diagnostic_run=Path(reference_diagnostic_run).resolve(),
        reference_configuration=reference_configuration,
    )


def _pass_fail(value: bool) -> str:
    return "Pass" if bool(value) else "Fail"


def build_shadow_decisions(
    tracklet_df: pd.DataFrame,
    frozen_candidate: FrozenCandidate,
    subject_rolls: set[str],
) -> pd.DataFrame:
    missing = sorted(REQUIRED_TRACKLET_COLUMNS.difference(tracklet_df.columns))
    if missing:
        raise ShadowValidationError(
            "Tracklet feature schema mismatch; missing: " + ", ".join(missing)
        )
    if tracklet_df["Tracklet_ID"].astype(str).duplicated().any():
        duplicates = sorted(
            tracklet_df.loc[
                tracklet_df["Tracklet_ID"].astype(str).duplicated(keep=False),
                "Tracklet_ID",
            ].astype(str).unique()
        )
        raise ShadowValidationError(
            "Tracklet diagnostics contain duplicate Tracklet_ID values: "
            + ", ".join(duplicates[:10])
        )
    frame = tracklet_df.copy().fillna("")
    if frame.empty:
        return frame.assign(
            Shadow_Decision_ID=pd.Series(dtype=str),
            Candidate_ID=pd.Series(dtype=str),
            Candidate_Config_SHA256=pd.Series(dtype=str),
            Baseline_Status=pd.Series(dtype=str),
            Shadow_Accepted=pd.Series(dtype=str),
            Shadow_Rejection_Reason=pd.Series(dtype=str),
        )

    if not frame["Tracklet_Mode"].astype(str).eq("compare").all():
        raise ShadowValidationError("Shadow validation requires tracklet_mode=compare diagnostics")
    if not frame["Aggregate_Mode"].astype(str).eq("top3").all():
        raise ShadowValidationError("Shadow validation requires aggregate=top3 diagnostics")
    if not pd.to_numeric(frame["Match_Threshold"], errors="coerce").eq(MATCH_THRESHOLD).all():
        raise ShadowValidationError("Tracklet diagnostics do not preserve match threshold 0.48")
    if not pd.to_numeric(frame["Margin_Threshold"], errors="coerce").eq(MARGIN_THRESHOLD).all():
        raise ShadowValidationError("Tracklet diagnostics do not preserve margin threshold 0.08")
    if frame["Official_Attendance_Contribution"].map(_yes).any():
        raise ShadowValidationError(
            "Shadow diagnostics contain an official attendance contribution; refusing candidate application"
        )

    normalized_roster = {canonical_roll(value) for value in subject_rolls if canonical_roll(value)}
    if not normalized_roster:
        raise ShadowValidationError("Authoritative subject roster is empty")
    frame["Baseline_Accepted"] = frame["Tracklet_Accepted"].map(_yes)
    frame["Tracklet_Eligible_Bool"] = frame["Tracklet_Eligible"].map(_yes)
    frame["Predicted_Roll"] = frame["Tracklet_Best_Roll"].map(canonical_roll)
    frame["Second_Roll"] = frame["Tracklet_Second_Roll"].map(canonical_roll)
    frame["Dominant_Frame_Canonical_Roll"] = frame["Dominant_Frame_Best_Roll"].map(
        canonical_roll
    )
    frame["Candidate_In_Subject_Roster"] = frame["Predicted_Roll"].isin(normalized_roster)
    frame["Dominant_Agreement"] = frame["Predicted_Roll"].eq(
        frame["Dominant_Frame_Canonical_Roll"]
    )
    numeric_columns = {
        "Best_Score_Value": "Tracklet_Best_Score",
        "Second_Score_Value": "Tracklet_Second_Score",
        "Margin_Value": "Tracklet_Margin",
        "Vote_Ratio_Pct_Value": "Dominant_Frame_Best_Share_Pct",
        "Observation_Count_Value": "Observation_Count",
        "Selected_Observation_Count_Value": "Selected_Observation_Count",
        "Embedding_Count_Value": "Embedding_Count",
        "Consistent_Embedding_Count_Value": "Consistent_Embedding_Count",
        "Inconsistent_Embedding_Count_Value": "Inconsistent_Embedding_Count",
        "Pairwise_Similarity_Median_Value": "Pairwise_Similarity_Median",
    }
    missing_defaults = {
        "Best_Score_Value": -1.0,
        "Second_Score_Value": -1.0,
        "Margin_Value": -1.0,
        "Vote_Ratio_Pct_Value": -1.0,
        "Observation_Count_Value": 0.0,
        "Selected_Observation_Count_Value": 0.0,
        "Embedding_Count_Value": 0.0,
        "Consistent_Embedding_Count_Value": 0.0,
        "Inconsistent_Embedding_Count_Value": 0.0,
        "Pairwise_Similarity_Median_Value": -1.0,
    }
    for output, source in numeric_columns.items():
        raw_values = frame[source].astype(str).str.strip()
        values = pd.to_numeric(raw_values, errors="coerce")
        malformed = values.isna() & raw_values.ne("")
        if malformed.any():
            example = _text(frame.loc[malformed, "Tracklet_ID"].iloc[0])
            raise ShadowValidationError(
                f"Tracklet feature schema has a non-numeric {source} value for {example}"
            )
        missing_on_eligible = values.isna() & frame["Tracklet_Eligible_Bool"]
        if missing_on_eligible.any():
            example = _text(frame.loc[missing_on_eligible, "Tracklet_ID"].iloc[0])
            raise ShadowValidationError(
                f"Eligible tracklet feature schema is missing {source} for {example}"
            )
        frame[output] = values.fillna(missing_defaults[output]).astype(float)
    denominator = frame["Embedding_Count_Value"].replace(0.0, np.nan)
    frame["Consistency_Ratio"] = (
        frame["Consistent_Embedding_Count_Value"] / denominator
    ).fillna(0.0)
    frame["Calibration_Eligible"] = (
        ~frame["Baseline_Accepted"] & frame["Tracklet_Eligible_Bool"]
    )

    candidate = frozen_candidate.candidate
    accepted_mask = candidate_acceptance_mask(frame, candidate)
    guards = {
        "Guard_Baseline_Rejected": ~frame["Baseline_Accepted"],
        "Guard_Tracklet_Eligible": frame["Tracklet_Eligible_Bool"],
        "Guard_Subject_Roster": frame["Candidate_In_Subject_Roster"],
        "Guard_Dominant_Agreement": frame["Dominant_Agreement"],
        "Guard_Best_Score": frame["Best_Score_Value"].ge(candidate.min_best_score),
        "Guard_Margin": frame["Margin_Value"].ge(candidate.min_margin),
        "Guard_Vote_Ratio": frame["Vote_Ratio_Pct_Value"].ge(candidate.min_vote_ratio_pct),
        "Guard_Observation_Count": frame["Observation_Count_Value"].ge(
            candidate.min_observation_count
        ),
        "Guard_Selected_Observations": frame["Selected_Observation_Count_Value"].ge(
            candidate.min_selected_observations
        ),
        "Guard_Consistent_Embeddings": frame["Consistent_Embedding_Count_Value"].ge(
            candidate.min_consistent_embeddings
        ),
        "Guard_Consistency_Ratio": frame["Consistency_Ratio"].ge(
            candidate.min_consistency_ratio
        ),
        "Guard_Pairwise_Similarity": frame["Pairwise_Similarity_Median_Value"].ge(
            candidate.min_pairwise_similarity_median
        ),
    }
    for column, values in guards.items():
        frame[column] = values.map(_pass_fail)
    frame["Candidate_ID"] = frozen_candidate.candidate.candidate_id
    frame["Candidate_Config_SHA256"] = frozen_candidate.config_sha256
    frame["Baseline_Status"] = np.where(frame["Baseline_Accepted"], "accepted", "rejected")
    frame["Baseline_Rejection_Reason"] = np.where(
        frame["Baseline_Accepted"], "", frame["Tracklet_Diagnostic_Reason"]
    )
    frame["Candidate_Roster_Valid"] = frame["Candidate_In_Subject_Roster"].map(
        lambda value: "Yes" if value else "No"
    )
    frame["Shadow_Accepted"] = accepted_mask.map(lambda value: "Yes" if value else "No")
    guard_columns = list(guards)

    def rejection_reason(row: pd.Series) -> str:
        failed = [column.removeprefix("Guard_") for column in guard_columns if row[column] == "Fail"]
        return "accepted_by_frozen_candidate" if not failed else "; ".join(failed)

    frame["Shadow_Rejection_Reason"] = frame.apply(rejection_reason, axis=1)
    frame["Shadow_Decision_ID"] = frame.apply(
        lambda row: "SHD-"
        + hashlib.sha256(
            "|".join(
                [
                    _text(row.get("Session_ID")),
                    _text(row.get("Tracklet_ID")),
                    frozen_candidate.candidate.candidate_id,
                    frozen_candidate.config_sha256,
                ]
            ).encode("utf-8")
        ).hexdigest()[:16],
        axis=1,
    )
    rejected = frame[~frame["Baseline_Accepted"]].copy()
    rejected["_checkpoint_order"] = rejected["Checkpoint_ID"].map(
        lambda value: int(_text(value)[2:]) if _text(value).startswith("CP") and _text(value)[2:].isdigit() else 999
    )
    rejected = rejected.sort_values(
        ["_checkpoint_order", "Camera_ID", "Tracklet_ID"],
        kind="mergesort",
    ).drop(columns=["_checkpoint_order"]).reset_index(drop=True)
    if rejected["Shadow_Decision_ID"].duplicated().any():
        raise ShadowValidationError("Shadow candidate decisions contain duplicate deterministic IDs")
    if rejected.loc[rejected["Shadow_Accepted"].eq("Yes"), "Tracklet_ID"].duplicated().any():
        raise ShadowValidationError("Shadow candidate produced duplicate recoveries")
    return rejected


SHADOW_PRIVATE_COLUMNS = {
    "Candidate_ID": "Candidate_ID",
    "Candidate_Config_SHA256": "Candidate_Config_SHA256",
    "Shadow_Decision_ID": "Shadow_Decision_ID",
    "Baseline_Status": "Baseline_Status",
    "Baseline_Rejection_Reason": "Baseline_Rejection_Reason",
    "Shadow_Accepted": "Shadow_Accepted",
    "Shadow_Rejection_Reason": "Shadow_Rejection_Reason",
    "Candidate_Roster_Valid": "Candidate_Roster_Valid",
    "Dominant_Frame_Best_Roll": "Dominant_Frame_Best_Roll",
    "Vote_Ratio_Pct": "Dominant_Frame_Best_Share_Pct",
    "Embedding_Count": "Embedding_Count",
    "Consistent_Embedding_Count": "Consistent_Embedding_Count",
    "Inconsistent_Embedding_Count": "Inconsistent_Embedding_Count",
    "Consistency_Ratio": "Consistency_Ratio",
    "Pairwise_Similarity_Median": "Pairwise_Similarity_Median",
    "Zone_IDs": "Zone_IDs",
    "Guard_Baseline_Rejected": "Guard_Baseline_Rejected",
    "Guard_Tracklet_Eligible": "Guard_Tracklet_Eligible",
    "Guard_Subject_Roster": "Guard_Subject_Roster",
    "Guard_Dominant_Agreement": "Guard_Dominant_Agreement",
    "Guard_Best_Score": "Guard_Best_Score",
    "Guard_Margin": "Guard_Margin",
    "Guard_Vote_Ratio": "Guard_Vote_Ratio",
    "Guard_Observation_Count": "Guard_Observation_Count",
    "Guard_Selected_Observations": "Guard_Selected_Observations",
    "Guard_Consistent_Embeddings": "Guard_Consistent_Embeddings",
    "Guard_Consistency_Ratio": "Guard_Consistency_Ratio",
    "Guard_Pairwise_Similarity": "Guard_Pairwise_Similarity",
}


def export_shadow_review_package(
    *,
    decisions: pd.DataFrame,
    observation_df: pd.DataFrame,
    video_root: Path,
    output_root: Path,
    student_map_path: Path,
    diagnostic_run_id: str,
    session_id: str,
    frozen_candidate: FrozenCandidate,
    source_files: dict[str, Path] | None = None,
    frame_loader: Callable[[Path, int], np.ndarray] | None = None,
) -> ReviewPackageResult:
    required = {"Tracklet_ID", "Baseline_Status", "Shadow_Accepted"}
    missing = sorted(required.difference(decisions.columns))
    if missing:
        raise ShadowValidationError(
            "Shadow decision export is missing columns: " + ", ".join(missing)
        )
    recovered = decisions[decisions["Shadow_Accepted"].eq("Yes")].copy()
    if recovered["Baseline_Status"].ne("rejected").any():
        raise ShadowValidationError("Baseline accepted tracks cannot enter the shadow reviewer")
    if recovered["Tracklet_ID"].astype(str).duplicated().any():
        raise ShadowValidationError("Shadow reviewer input contains duplicate tracklets")
    try:
        return export_review_package(
            tracklet_df=decisions,
            observation_df=observation_df,
            video_root=Path(video_root),
            output_root=Path(output_root),
            student_map_path=Path(student_map_path),
            subject_abbr=EXPECTED_SUBJECT,
            diagnostic_run_id=diagnostic_run_id,
            selected_tracklet_ids=recovered["Tracklet_ID"].astype(str).tolist(),
            review_kind="shadow_recovery",
            hidden_extra_columns=SHADOW_PRIVATE_COLUMNS,
            include_context=True,
            context_padding=2.0,
            package_prefix="shadow_recovery_review",
            subject_roster_only=True,
            allow_empty=True,
            session_id_override=session_id,
            private_metadata={
                "candidate_id": frozen_candidate.candidate.candidate_id,
                "candidate_config_sha256": frozen_candidate.config_sha256,
                "candidate_enabled": False,
                "production_approved": False,
                "production_changes": False,
                "official_attendance_changed": False,
            },
            source_files=source_files,
            frame_loader=frame_loader,
        )
    except ReviewExportError as exc:
        raise ShadowValidationError(str(exc)) from exc


def audit_blind_reviewer(review_package: Path) -> dict[str, Any]:
    review_package = Path(review_package)
    reviewer_dir = review_package / "reviewer"
    private_path = review_package / "private" / "hidden_predictions.csv"
    if not reviewer_dir.is_dir() or not private_path.is_file():
        raise ShadowValidationError("Blind reviewer package is incomplete")
    predictions = pd.read_csv(private_path, dtype=str, keep_default_na=False)
    text_files = sorted(
        path
        for path in reviewer_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".html", ".json", ".csv", ".txt", ".js", ".css"}
    )
    forbidden_tokens = {
        "Predicted_Identity_Raw",
        "Predicted_Roll",
        "Best_Score",
        "Second_Roll",
        "Second_Score",
        "Match_Threshold",
        "Margin_Threshold",
        "Tracklet_ID",
        "Baseline_Rejection_Reason",
        "Shadow_Rejection_Reason",
        "hidden_predictions.csv",
        "../private",
    }
    forbidden_tokens.update(
        _text(value) for value in predictions.get("Tracklet_ID", pd.Series(dtype=str)) if _text(value)
    )
    matches: list[str] = []
    for path in text_files:
        content = path.read_text(encoding="utf-8", errors="ignore")
        for token in sorted(forbidden_tokens):
            if token and token in content:
                matches.append(f"{path.relative_to(reviewer_dir).as_posix()}:{token}")
    manifest = json.loads((reviewer_dir / "review_manifest.json").read_text(encoding="utf-8"))
    public_ids = {_text(row.get("review_id")) for row in manifest.get("review_items", [])}
    private_ids = set(predictions.get("Review_ID", pd.Series(dtype=str)).astype(str))
    if public_ids != private_ids:
        raise ShadowValidationError("Reviewer and private prediction ID sets do not match")
    return {
        "review_items": len(public_ids),
        "private_predictions": len(private_ids),
        "forbidden_matches": matches,
        "privacy_passed": not matches,
        "private_mapping_loaded_by_browser": False,
    }


def write_output_manifest(
    root: Path,
    extra: dict[str, Any] | None = None,
    *,
    filename: str = "output_manifest.json",
) -> Path:
    root = Path(root)
    manifest_path = root / filename
    files = {
        path.relative_to(root).as_posix(): _sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != manifest_path
    }
    payload = {
        "schema_version": SHADOW_SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(root.resolve()),
        "files_sha256": files,
    }
    reserved = sorted(set(extra or {}).intersection(payload))
    if reserved:
        raise ShadowValidationError(
            "Output manifest metadata cannot override reserved fields: " + ", ".join(reserved)
        )
    payload.update(json_safe(extra or {}))
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    return manifest_path


def verify_output_manifest(root: Path, manifest_path: Path) -> dict[str, Any]:
    root = Path(root)
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise ShadowValidationError(f"Output manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for relative, expected_hash in dict(payload.get("files_sha256") or {}).items():
        path = root / relative
        if not path.is_file() or _sha256_file(path) != _text(expected_hash):
            failures.append(relative)
    if failures:
        raise ShadowValidationError(
            "Output artifact integrity check failed: " + ", ".join(failures[:10])
        )
    return {"verified_files": len(payload.get("files_sha256", {})), "manifest": payload}


class _QuietReviewHandler(SimpleHTTPRequestHandler):
    def log_message(self, _format: str, *_args: Any) -> None:
        return


def create_reviewer_server(
    review_package: Path,
    *,
    port: int = 0,
) -> tuple[ThreadingHTTPServer, str]:
    review_package = Path(review_package)
    reviewer_dir = review_package / "reviewer"
    if not (reviewer_dir / "index.html").is_file():
        raise ShadowValidationError(f"Reviewer index.html not found: {reviewer_dir}")
    if port < 0 or port > 65535:
        raise ShadowValidationError("Reviewer port must be between 0 and 65535")
    handler = partial(_QuietReviewHandler, directory=str(reviewer_dir.resolve()))
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    except OSError as exc:
        raise ShadowValidationError(f"Could not bind the local reviewer server: {exc}") from exc
    actual_port = int(server.server_address[1])
    return server, f"http://127.0.0.1:{actual_port}/"


def serve_review_package(
    review_package: Path,
    *,
    port: int = 0,
    open_browser: bool = True,
) -> None:
    server, url = create_reviewer_server(review_package, port=port)
    print(f"Blind reviewer URL: {url}")
    print(f"PID: {os.getpid()}")
    print("Server binding: 127.0.0.1 only")
    print("Clean shutdown: press Ctrl+C in this terminal")
    if open_browser and not webbrowser.open(url):
        print("Browser did not open automatically; open the URL shown above.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nReviewer server stopped.")
    finally:
        server.server_close()


def evaluate_shadow_predictions(
    prediction_df: pd.DataFrame,
    label_df: pd.DataFrame,
    *,
    known_rolls: set[str],
    package_id: str,
    candidate_id: str,
    candidate_config_sha256: str,
    session_id: str,
) -> ShadowEvaluationResult:
    required_predictions = {
        "Package_ID",
        "Review_ID",
        "Tracklet_ID",
        "Session_ID",
        "Subject_Abbr",
        "Checkpoint_ID",
        "Camera_ID",
        "Predicted_Roll",
        "Candidate_ID",
        "Candidate_Config_SHA256",
        "Evidence_SHA256",
    }
    missing = sorted(required_predictions.difference(prediction_df.columns))
    if missing:
        raise ShadowValidationError(
            "Shadow prediction package is missing columns: " + ", ".join(missing)
        )
    predictions = prediction_df.copy().fillna("")
    if predictions["Review_ID"].astype(str).duplicated().any():
        raise ShadowValidationError("Shadow prediction package contains duplicate Review_ID values")
    if predictions["Tracklet_ID"].astype(str).duplicated().any():
        raise ShadowValidationError("Shadow prediction package contains duplicate Tracklet_ID values")
    if not predictions.empty:
        if set(predictions["Package_ID"].astype(str)) != {package_id}:
            raise ShadowValidationError("Shadow prediction Package_ID does not match package metadata")
        if set(predictions["Candidate_ID"].astype(str)) != {candidate_id}:
            raise ShadowValidationError("Shadow prediction candidate ID does not match package metadata")
        if set(predictions["Candidate_Config_SHA256"].astype(str)) != {
            candidate_config_sha256
        }:
            raise ShadowValidationError("Shadow prediction candidate-config hash does not match")
        if set(predictions["Session_ID"].astype(str)) != {session_id}:
            raise ShadowValidationError("Shadow prediction package contains another session")

    normalized_known = {canonical_roll(value) for value in known_rolls if canonical_roll(value)}
    predictions["Predicted_Roll"] = predictions["Predicted_Roll"].map(canonical_roll)
    invalid_predictions = sorted(
        set(predictions["Predicted_Roll"].astype(str)).difference(normalized_known | {""})
    )
    if invalid_predictions:
        raise ShadowValidationError(
            "Shadow candidate predicted a roll outside the authoritative subject roster: "
            + ", ".join(invalid_predictions)
        )
    try:
        labels = validate_label_dataframe(
            label_df,
            set(predictions["Review_ID"].astype(str)),
            package_id,
            normalized_known,
        )
    except LabelValidationError as exc:
        raise ShadowValidationError(str(exc)) from exc
    prediction_ids = set(predictions["Review_ID"].astype(str))
    label_ids = set(labels["Review_ID"].astype(str))
    if label_ids != prediction_ids:
        missing_ids = sorted(prediction_ids.difference(label_ids))
        extra_ids = sorted(label_ids.difference(prediction_ids))
        raise ShadowValidationError(
            "Shadow label Review_ID set is incomplete or mismatched"
            + (f"; missing: {', '.join(missing_ids[:10])}" if missing_ids else "")
            + (f"; extra: {', '.join(extra_ids[:10])}" if extra_ids else "")
        )
    if labels["Review_Status"].eq("").any():
        pending = labels.loc[labels["Review_Status"].eq(""), "Review_ID"].tolist()
        raise ShadowValidationError(
            "All shadow candidates must be reviewed before evaluation; pending: "
            + ", ".join(pending[:10])
        )
    joined = predictions.merge(
        labels,
        on=["Package_ID", "Review_ID"],
        how="left",
        validate="one_to_one",
    )
    joined["Actual_Roll"] = joined["Actual_Roll"].map(canonical_roll)

    def classify(row: pd.Series) -> str:
        status = _text(row.get("Review_Status"))
        if status == "identified":
            return (
                "correct_recovery"
                if row["Actual_Roll"] == row["Predicted_Roll"]
                else "confirmed_false_identity"
            )
        if status == "not_in_mapping":
            return "unsafe_outsider_false_acceptance"
        if status == "mixed_track":
            return "unsafe_mixed_evidence"
        if status in {"unidentifiable", "uncertain"}:
            return "unverifiable"
        if status == "duplicate":
            return "duplicate"
        raise ShadowValidationError(f"Unsupported completed shadow review status: {status}")

    if joined.empty:
        joined["Shadow_Validation_Class"] = pd.Series(index=joined.index, dtype="object")
    else:
        joined["Shadow_Validation_Class"] = joined.apply(classify, axis=1)
    counts = joined["Shadow_Validation_Class"].value_counts().to_dict()
    false_identities = int(counts.get("confirmed_false_identity", 0))
    outsider = int(counts.get("unsafe_outsider_false_acceptance", 0))
    mixed = int(counts.get("unsafe_mixed_evidence", 0))
    unverifiable = int(counts.get("unverifiable", 0))
    duplicates = int(counts.get("duplicate", 0))
    correct = joined[joined["Shadow_Validation_Class"].eq("correct_recovery")]
    correct_count = int(len(correct))
    unique_identities = int(correct["Actual_Roll"].replace("", pd.NA).dropna().nunique())
    unique_checkpoints = int(correct["Checkpoint_ID"].replace("", pd.NA).dropna().nunique())

    if false_identities or outsider or mixed:
        decision = "reject_candidate"
        rationale = "At least one confirmed false identity, outsider acceptance, or mixed track exists."
    elif unverifiable:
        decision = "hold_unverifiable_sample"
        rationale = "No unsafe identity is confirmed, but one or more candidate recoveries are unverifiable."
    elif correct_count == 0:
        decision = "hold_no_recoveries"
        rationale = "The untouched session produced zero unique reviewable correct recoveries."
    elif (
        correct_count >= MIN_CORRECT_RECOVERIES
        and unique_identities >= MIN_CORRECT_IDENTITIES
        and unique_checkpoints >= MIN_CORRECT_CHECKPOINTS
    ):
        decision = "continue_multisession_shadow_validation"
        rationale = (
            "The complete blind review has zero unsafe accepts and meets the conservative "
            "multi-identity, multi-checkpoint evidence floor."
        )
    else:
        decision = "hold_insufficient_recovery_evidence"
        rationale = (
            "The complete blind review has no unsafe accepts but does not meet the minimum evidence floor."
        )

    breakdown_rows: list[dict[str, Any]] = []
    for (camera, checkpoint), group in joined.groupby(
        ["Camera_ID", "Checkpoint_ID"], sort=True, dropna=False
    ):
        group_counts = group["Shadow_Validation_Class"].value_counts()
        breakdown_rows.append(
            {
                "Camera_ID": _text(camera),
                "Checkpoint_ID": _text(checkpoint),
                "Reviewed_Tracks": int(len(group)),
                "Correct_Recoveries": int(group_counts.get("correct_recovery", 0)),
                "Confirmed_False_Identities": int(
                    group_counts.get("confirmed_false_identity", 0)
                ),
                "Unsafe_Outsider_Accepts": int(
                    group_counts.get("unsafe_outsider_false_acceptance", 0)
                ),
                "Unsafe_Mixed": int(group_counts.get("unsafe_mixed_evidence", 0)),
                "Unverifiable": int(group_counts.get("unverifiable", 0)),
                "Duplicates": int(group_counts.get("duplicate", 0)),
            }
        )
    breakdown = pd.DataFrame(
        breakdown_rows,
        columns=[
            "Camera_ID",
            "Checkpoint_ID",
            "Reviewed_Tracks",
            "Correct_Recoveries",
            "Confirmed_False_Identities",
            "Unsafe_Outsider_Accepts",
            "Unsafe_Mixed",
            "Unverifiable",
            "Duplicates",
        ],
    )
    summary = {
        "schema_version": SHADOW_SCHEMA_VERSION,
        "session_id": session_id,
        "subject_abbr": EXPECTED_SUBJECT,
        "package_id": package_id,
        "candidate_id": candidate_id,
        "candidate_config_sha256": candidate_config_sha256,
        "reviewed_shadow_tracks": int(len(joined)),
        "review_complete": True,
        "correct_recoveries": correct_count,
        "confirmed_false_identities": false_identities,
        "unsafe_outsider_false_acceptances": outsider,
        "unsafe_mixed_evidence": mixed,
        "unverifiable": unverifiable,
        "duplicates": duplicates,
        "unique_correct_identities": unique_identities,
        "unique_correct_checkpoints": unique_checkpoints,
        "decision": decision,
        "decision_rationale": rationale,
        "decision_rule": {
            "reject_on_any_confirmed_false_identity": True,
            "reject_on_any_not_in_mapping": True,
            "reject_on_any_mixed_track": True,
            "hold_on_any_unverifiable": True,
            "minimum_correct_unique_recoveries": MIN_CORRECT_RECOVERIES,
            "minimum_correct_identities": MIN_CORRECT_IDENTITIES,
            "minimum_correct_checkpoints": MIN_CORRECT_CHECKPOINTS,
        },
        "production_approved": False,
        "production_changes": False,
        "official_thresholds_changed": False,
        "attendance_rule_changed": False,
        "official_attendance_changed": False,
    }
    return ShadowEvaluationResult(joined, summary, breakdown)


def _shadow_evaluation_report(summary: dict[str, Any]) -> str:
    return (
        "# Phase 1.2H Untouched-Session Shadow Validation\n\n"
        f"- Session: {summary['session_id']}\n"
        f"- Candidate: {summary['candidate_id']}\n"
        f"- Reviewed shadow tracks: {summary['reviewed_shadow_tracks']}\n"
        f"- Correct recoveries: {summary['correct_recoveries']}\n"
        f"- Confirmed false identities: {summary['confirmed_false_identities']}\n"
        f"- Unsafe outsider accepts: {summary['unsafe_outsider_false_acceptances']}\n"
        f"- Unsafe mixed evidence: {summary['unsafe_mixed_evidence']}\n"
        f"- Unverifiable: {summary['unverifiable']}\n"
        f"- Duplicates: {summary['duplicates']}\n"
        f"- Decision: {summary['decision']}\n\n"
        f"{summary['decision_rationale']}\n\n"
        "This phase cannot approve production. Official thresholds, embeddings, the attendance rule, "
        "and official attendance remain unchanged.\n"
    )


def evaluate_shadow_review_package(
    review_package: Path,
    labels_path: Path,
    *,
    output_dir: Path | None = None,
    expected_candidate_id: str = EXPECTED_CANDIDATE_ID,
    expected_candidate_config_sha256: str = EXPECTED_CANDIDATE_CONFIG_SHA256,
    expected_roster_count: int = EXPECTED_ROSTER_COUNT,
) -> tuple[ShadowEvaluationResult, Path]:
    review_package = Path(review_package)
    labels_path = Path(labels_path)
    try:
        predictions, metadata, manifest = load_review_package_data(review_package)
    except LabelValidationError as exc:
        raise ShadowValidationError(str(exc)) from exc
    if metadata.get("review_kind") != "shadow_recovery" or manifest.get("review_kind") != "shadow_recovery":
        raise ShadowValidationError("The supplied package is not a shadow-recovery review package")
    candidate_id = _text(metadata.get("candidate_id"))
    candidate_hash = _text(metadata.get("candidate_config_sha256"))
    session_id = _text(metadata.get("session_id"))
    if not candidate_id or not candidate_hash or not session_id:
        raise ShadowValidationError("Shadow review metadata lacks candidate or session provenance")
    if candidate_id != expected_candidate_id:
        raise ShadowValidationError(
            f"Shadow review candidate is {candidate_id}, not approved frozen candidate {expected_candidate_id}"
        )
    if candidate_hash != expected_candidate_config_sha256:
        raise ShadowValidationError(
            "Shadow review candidate-config hash does not match the approved frozen candidate"
        )
    if (
        _text(metadata.get("subject_abbr")).upper() != EXPECTED_SUBJECT
        or _text(manifest.get("subject_abbr")).upper() != EXPECTED_SUBJECT
    ):
        raise ShadowValidationError("Shadow review package is not scoped to the authoritative CVO subject")
    if metadata.get("candidate_enabled") is not False or metadata.get("production_approved") is not False:
        raise ShadowValidationError("Shadow review metadata indicates an enabled or approved candidate")

    shadow_root = review_package.parent
    snapshot = shadow_root / "candidate_config_snapshot.json"
    output_manifest = shadow_root / "output_manifest.json"
    session_manifest_path = shadow_root / "session_input_manifest.json"
    if not snapshot.is_file() or not output_manifest.is_file() or not session_manifest_path.is_file():
        raise ShadowValidationError(
            "Shadow review package is missing its frozen config snapshot, session manifest, or output manifest"
        )
    verified_output = verify_output_manifest(shadow_root, output_manifest)["manifest"]
    if (
        _text(verified_output.get("candidate_id")) != candidate_id
        or _text(verified_output.get("candidate_config_sha256")) != candidate_hash
        or _text(verified_output.get("session_id")) != session_id
    ):
        raise ShadowValidationError("Shadow output manifest provenance does not match the review package")
    if _sha256_file(snapshot) != candidate_hash:
        raise ShadowValidationError("Candidate config snapshot hash does not match review metadata")
    session_manifest = json.loads(session_manifest_path.read_text(encoding="utf-8"))
    session_candidate = dict(session_manifest.get("candidate") or {})
    session_roster = dict(session_manifest.get("authoritative_roster") or {})
    if (
        _text(session_manifest.get("session_id")) != session_id
        or _text(session_manifest.get("subject_abbr")).upper() != EXPECTED_SUBJECT
        or _text(session_candidate.get("candidate_id")) != candidate_id
        or _text(session_candidate.get("config_sha256")) != candidate_hash
        or session_candidate.get("enabled") is not False
        or session_candidate.get("production_approved") is not False
    ):
        raise ShadowValidationError("Session manifest provenance does not match the shadow review package")

    student_map_path = Path(_text(metadata.get("student_map_path")))
    recorded_roster_hash = _text(
        dict(metadata.get("source_sha256") or {}).get("student_map")
    )
    if not student_map_path.is_file() or not recorded_roster_hash:
        raise ShadowValidationError("Shadow review metadata lacks its authoritative student-map source")
    current_roster_hash = _sha256_file(student_map_path)
    if (
        current_roster_hash != recorded_roster_hash
        or current_roster_hash != _text(session_roster.get("sha256"))
    ):
        raise ShadowValidationError(
            "Authoritative CVO roster changed after shadow export; review provenance must be reconciled"
        )
    try:
        _, _, current_subject_rolls_raw = load_student_mapping(
            student_map_path, EXPECTED_SUBJECT
        )
    except ReviewExportError as exc:
        raise ShadowValidationError(str(exc)) from exc
    current_subject_rolls = {
        canonical_roll(value) for value in current_subject_rolls_raw if canonical_roll(value)
    }
    metadata_subject_rolls = {
        canonical_roll(value)
        for value in (metadata.get("subject_rolls") or [])
        if canonical_roll(value)
    }
    session_subject_rolls = {
        canonical_roll(value)
        for value in (session_roster.get("rolls") or [])
        if canonical_roll(value)
    }
    if len(current_subject_rolls) != expected_roster_count:
        raise ShadowValidationError(
            f"Authoritative CVO roster must contain {expected_roster_count} students; "
            f"found {len(current_subject_rolls)}"
        )
    if current_subject_rolls != metadata_subject_rolls or current_subject_rolls != session_subject_rolls:
        raise ShadowValidationError("Shadow review roster membership does not match the authoritative CVO roster")
    privacy = audit_blind_reviewer(review_package)
    if not privacy["privacy_passed"]:
        raise ShadowValidationError(
            "Blind reviewer privacy audit failed before evaluation: "
            + ", ".join(privacy["forbidden_matches"][:10])
        )
    if not labels_path.is_file():
        raise ShadowValidationError(f"Completed shadow labels CSV not found: {labels_path}")
    labels = pd.read_csv(labels_path, dtype=str, keep_default_na=False)
    result = evaluate_shadow_predictions(
        predictions,
        labels,
        known_rolls=current_subject_rolls,
        package_id=_text(metadata.get("package_id")),
        candidate_id=candidate_id,
        candidate_config_sha256=candidate_hash,
        session_id=session_id,
    )
    if output_dir is None:
        output_dir = shadow_root / "evaluation" / make_diagnostic_run_id("shadow-evaluation")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    result.tracks.to_csv(output_dir / "reviewed_shadow_tracks.csv", index=False)
    unsafe_classes = {
        "confirmed_false_identity",
        "unsafe_outsider_false_acceptance",
        "unsafe_mixed_evidence",
    }
    result.tracks[
        result.tracks["Shadow_Validation_Class"].isin(unsafe_classes)
    ].to_csv(output_dir / "false_accept_audit.csv", index=False)
    result.camera_checkpoint_breakdown.to_csv(
        output_dir / "camera_checkpoint_breakdown.csv", index=False
    )
    summary = {
        **result.summary,
        "labels_path": str(labels_path.resolve()),
        "labels_sha256": _sha256_file(labels_path),
        "review_package": str(review_package.resolve()),
    }
    (output_dir / "shadow_validation_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=True), encoding="utf-8"
    )
    (output_dir / "decision.json").write_text(
        json.dumps(
            json_safe(
                {
                    "schema_version": SHADOW_SCHEMA_VERSION,
                    "candidate_id": candidate_id,
                    "decision": summary["decision"],
                    "rationale": summary["decision_rationale"],
                    "production_approved": False,
                }
            ),
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    (output_dir / "shadow_validation_report.md").write_text(
        _shadow_evaluation_report(summary), encoding="utf-8"
    )
    manifest_path = write_output_manifest(
        output_dir,
        {
            "candidate_id": candidate_id,
            "candidate_config_sha256": candidate_hash,
            "session_id": session_id,
            "decision": summary["decision"],
            "production_approved": False,
            "production_changes": False,
        },
        filename="evaluation_output_manifest.json",
    )
    return ShadowEvaluationResult(result.tracks, summary, result.camera_checkpoint_breakdown), output_dir


def build_baseline_command(
    preflight: ShadowSessionPreflight,
    *,
    repo_root: Path,
    diagnostic_dir: Path,
    diagnostic_run_id: str,
    timetable_path: Path,
    embeddings_path: Path,
    slot_id: str = "TUE_P2",
    python_executable: Path | None = None,
) -> list[str]:
    repo_root = Path(repo_root).resolve()
    camera_zones = Path(_text(preflight.reference_configuration["camera_zones"]))
    if not camera_zones.is_absolute():
        camera_zones = repo_root / camera_zones
    if not camera_zones.is_file():
        raise ShadowValidationError(f"Reference camera-zone config not found: {camera_zones}")
    if not Path(timetable_path).is_file():
        raise ShadowValidationError(f"Timetable not found: {timetable_path}")
    if not Path(embeddings_path).is_file():
        raise ShadowValidationError(f"Embedding database not found: {embeddings_path}")
    config = preflight.reference_configuration
    input_source = str(preflight.video_root)
    try:
        input_source = preflight.video_root.relative_to(repo_root).as_posix()
    except ValueError:
        pass
    executable = Path(python_executable or sys.executable).resolve()
    command = [
        str(executable),
        str((repo_root / "scripts" / "mark_attendance_checkpoints.py").resolve()),
        "--timetable",
        str(Path(timetable_path).resolve()),
        "--slot-id",
        slot_id,
        "--session-id",
        preflight.session_id,
        "--session-date",
        preflight.session_fields["session_date"],
        "--subject-abbr",
        preflight.subject_abbr,
        "--subject-name",
        "Computer Vision through OpenCV",
        "--course-code",
        "24AMLJ502",
        "--faculty-id",
        "vikas",
        "--faculty-name",
        "Mr. Vikas B",
        "--input-slot",
        slot_id,
        "--input-source-type",
        "prepared_slot",
        "--input-source-path",
        input_source,
        "--video-dir",
        str(preflight.video_root),
        "--embeddings",
        str(Path(embeddings_path).resolve()),
        "--det-score",
        str(DEFAULT_DETECTION_SCORE),
        "--match-threshold",
        str(MATCH_THRESHOLD),
        "--margin-threshold",
        str(MARGIN_THRESHOLD),
        "--frame-skip",
        str(int(_number(config["frame_skip"]))),
        "--sample-fps",
        str(_number(config["sample_fps"])),
        "--max-width",
        str(DEFAULT_MAX_WIDTH),
        "--aggregate",
        _text(config["aggregate"]),
        "--settle-minutes",
        "10.0",
        "--checkpoint-every-minutes",
        "10.0",
        "--checkpoint-clip-seconds",
        "20.0",
        "--checkpoint-mode",
        _text(config["checkpoint_mode"]),
        "--checkpoint-min-detections",
        "2",
        "--present-checkpoints",
        str(PRESENT_CHECKPOINTS),
        "--log-mode",
        _text(config["log_mode"]),
        "--diagnostic",
        "--diagnostic-only",
        "--diagnostic-dir",
        str(Path(diagnostic_dir).resolve()),
        "--diagnostic-run-id",
        diagnostic_run_id,
        "--zone-mode",
        _text(config["zone_mode"]),
        "--camera-zones",
        str(camera_zones.resolve()),
        "--zone-profile",
        _text(config["zone_profile"]),
        "--zone-merge-iou",
        str(_number(config["zone_merge_iou"])),
        "--tracklet-mode",
        _text(config["tracklet_mode"]),
        "--tracklet-min-observations",
        str(int(_number(config["tracklet_min_observations"]))),
        "--tracklet-max-selected",
        str(int(_number(config["tracklet_max_selected"]))),
        "--tracklet-max-gap-seconds",
        str(_number(config["tracklet_max_gap_seconds"])),
        "--tracklet-min-iou",
        str(_number(config["tracklet_min_iou"])),
        "--tracklet-max-center-ratio",
        str(_number(config["tracklet_max_center_ratio"])),
        "--tracklet-min-size-ratio",
        str(_number(config["tracklet_min_size_ratio"])),
        "--tracklet-min-embedding-similarity",
        str(_number(config["tracklet_min_embedding_similarity"])),
    ]
    return command


def validate_shadow_diagnostic_run(
    diagnostic_run: Path,
    preflight: ShadowSessionPreflight,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Path]]:
    diagnostic_run = Path(diagnostic_run)
    if not diagnostic_run.is_dir():
        raise ShadowValidationError(f"Shadow diagnostic run not found: {diagnostic_run}")
    paths = {
        "diagnostic_summary": _single_file(
            diagnostic_run, "diagnostic_summary_*.json", "diagnostic summary"
        ),
        "tracklet_diagnostics": _single_file(
            diagnostic_run, "tracklet_diagnostics_*.csv", "tracklet diagnostics"
        ),
        "tracklet_observations": _single_file(
            diagnostic_run, "tracklet_observations_*.csv", "tracklet observations"
        ),
    }
    summary = json.loads(paths["diagnostic_summary"].read_text(encoding="utf-8"))
    metadata = dict(summary.get("run_metadata") or {})
    if _text(metadata.get("session_id")) != preflight.session_id:
        raise ShadowValidationError("Diagnostic session ID does not match the shadow preflight")
    if _text(metadata.get("subject_abbr")).upper() != preflight.subject_abbr:
        raise ShadowValidationError("Diagnostic subject does not match the authoritative shadow subject")
    if metadata.get("diagnostic_only") is not True:
        raise ShadowValidationError("Diagnostic run is not marked diagnostic_only=true")
    if metadata.get("official_attendance_written") is not False:
        raise ShadowValidationError("Diagnostic run indicates that official attendance was written")
    if dict(summary.get("normal_output_files") or {}):
        raise ShadowValidationError("Diagnostic-only run unexpectedly contains normal attendance outputs")

    reference = preflight.reference_configuration
    exact_text_keys = (
        "checkpoint_mode",
        "log_mode",
        "aggregate",
        "zone_mode",
        "zone_profile",
        "tracklet_mode",
        "official_recognition_unit",
    )
    numeric_keys = (
        "match_threshold",
        "margin_threshold",
        "sample_fps",
        "frame_skip",
        "zone_merge_iou",
        "tracklet_min_observations",
        "tracklet_max_selected",
        "tracklet_max_gap_seconds",
        "tracklet_min_iou",
        "tracklet_max_center_ratio",
        "tracklet_min_size_ratio",
        "tracklet_min_embedding_similarity",
    )
    for key in exact_text_keys:
        if _text(metadata.get(key)) != _text(reference.get(key)):
            raise ShadowValidationError(
                f"Diagnostic setting {key} does not match the frozen reference run"
            )
    for key in numeric_keys:
        if abs(_number(metadata.get(key), -999) - _number(reference.get(key), -998)) > 1e-9:
            raise ShadowValidationError(
                f"Diagnostic setting {key} does not match the frozen reference run"
            )
    input_files = list(summary.get("input_files") or [])
    if len(input_files) != 10:
        raise ShadowValidationError(
            f"Diagnostic summary must contain ten source videos; found {len(input_files)}"
        )
    if not summary.get("checkpoint_extraction_valid"):
        raise ShadowValidationError("Diagnostic checkpoint extraction is not valid")

    tracklets = pd.read_csv(paths["tracklet_diagnostics"], dtype=str, keep_default_na=False)
    observations = pd.read_csv(paths["tracklet_observations"], dtype=str, keep_default_na=False)
    missing = sorted(REQUIRED_TRACKLET_COLUMNS.difference(tracklets.columns))
    if missing:
        raise ShadowValidationError(
            "TUE_P2 diagnostic feature schema is incomplete: " + ", ".join(missing)
        )
    if not tracklets.empty:
        if set(tracklets["Session_ID"].astype(str)) != {preflight.session_id}:
            raise ShadowValidationError("TUE_P2 tracklets contain another session ID")
        if set(tracklets["Subject_Abbr"].astype(str).str.upper()) != {preflight.subject_abbr}:
            raise ShadowValidationError("TUE_P2 tracklets contain another subject")
    required_observations = {
        "Tracklet_ID",
        "Observation_ID",
        "Checkpoint_ID",
        "Camera_ID",
        "Video",
        "Frame",
        "BBox_Original_Coordinates",
    }
    observation_missing = sorted(required_observations.difference(observations.columns))
    if observation_missing:
        raise ShadowValidationError(
            "TUE_P2 observation evidence schema is incomplete: "
            + ", ".join(observation_missing)
        )
    return tracklets, observations, summary, paths


def _resolved_video_manifest(
    preflight: ShadowSessionPreflight,
    diagnostic_summary: dict[str, Any],
    repo_root: Path,
) -> list[dict[str, Any]]:
    pipeline_ids: dict[tuple[str, str], str] = {}
    for item in diagnostic_summary.get("input_files") or []:
        source = Path(_text(item.get("source_file")))
        if not source.is_absolute():
            source = Path(repo_root) / source
        checkpoint = next(
            (
                part.split("_", 1)[0].upper()
                for part in source.parts
                if part.upper().startswith("CP") and part.split("_", 1)[0][2:].isdigit()
            ),
            "",
        )
        pipeline_ids[(checkpoint, _text(item.get("source_file_name")).lower())] = _text(
            item.get("camera_id")
        )
    resolved: list[dict[str, Any]] = []
    for row in preflight.videos:
        key = (row["checkpoint_id"], row["source_file_name"].lower())
        camera_id = pipeline_ids.get(key, "")
        if not camera_id:
            raise ShadowValidationError(
                f"Diagnostic summary did not preserve a camera ID for {key[0]}/{key[1]}"
            )
        resolved.append({**row, "pipeline_camera_id": camera_id})
    return resolved


def _powershell_quote(value: Any) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def run_shadow_session(
    *,
    repo_root: Path,
    video_root: Path,
    session_id: str,
    subject_abbr: str,
    student_map_path: Path,
    candidate_config_path: Path,
    candidate_manifest_path: Path,
    reference_diagnostic_run: Path,
    timetable_path: Path,
    embeddings_path: Path,
    diagnostic_run: Path | None = None,
    output_dir: Path | None = None,
    expected_roster_count: int = EXPECTED_ROSTER_COUNT,
    status_callback: Callable[[str], None] | None = print,
) -> ShadowSessionRunResult:
    started = time.perf_counter()
    repo_root = Path(repo_root).resolve()
    preflight = preflight_shadow_session(
        video_root=video_root,
        session_id=session_id,
        subject_abbr=subject_abbr,
        student_map_path=student_map_path,
        candidate_config_path=candidate_config_path,
        candidate_manifest_path=candidate_manifest_path,
        reference_diagnostic_run=reference_diagnostic_run,
        expected_roster_count=expected_roster_count,
    )
    if status_callback:
        status_callback(f"Resolved session: {preflight.session_id}")
        status_callback(f"Resolved footage: {preflight.video_root} ({len(preflight.videos)} videos)")
        status_callback(
            f"Resolved CVO roster: {preflight.student_map_path} ({len(preflight.subject_rolls)} students)"
        )
        status_callback(
            f"Frozen candidate: {preflight.frozen_candidate.candidate.candidate_id} "
            f"({preflight.frozen_candidate.config_sha256})"
        )

    candidate_hash_before = _sha256_file(preflight.frozen_candidate.config_path)
    roster_hash_before = _sha256_file(preflight.student_map_path)
    recognition_rerun = diagnostic_run is None
    command: list[str] = []
    if diagnostic_run is None:
        diagnostic_run_id = make_diagnostic_run_id(preflight.session_id)
        diagnostics_base = repo_root / "attendance_output" / "diagnostics"
        diagnostic_run = diagnostics_base / diagnostic_run_id
        if output_dir is None:
            output_dir = (
                diagnostic_run
                / "shadow_validation"
                / make_diagnostic_run_id("shadow-validation")
            )
        output_dir = Path(output_dir).resolve()
        if diagnostic_run.exists():
            raise ShadowValidationError(
                f"Refusing to overwrite an existing diagnostic run: {diagnostic_run}"
            )
        if output_dir.exists():
            raise ShadowValidationError(
                f"Refusing to overwrite an existing shadow output: {output_dir}"
            )
        command = build_baseline_command(
            preflight,
            repo_root=repo_root,
            diagnostic_dir=diagnostics_base,
            diagnostic_run_id=diagnostic_run_id,
            timetable_path=timetable_path,
            embeddings_path=embeddings_path,
            python_executable=Path(sys.executable),
        )
        if status_callback:
            status_callback("Starting diagnostic-only TUE_P2 recognition. No attendance files will be written.")
            status_callback("Command: " + subprocess.list2cmdline(command))
            sys.stdout.flush()
        try:
            subprocess.run(command, cwd=repo_root, check=True)
        except subprocess.CalledProcessError as exc:
            raise ShadowValidationError(
                f"TUE_P2 diagnostic command failed with exit code {exc.returncode}"
            ) from exc
    else:
        diagnostic_run = Path(diagnostic_run).resolve()
        if output_dir is None:
            output_dir = (
                diagnostic_run
                / "shadow_validation"
                / make_diagnostic_run_id("shadow-validation")
            )
        output_dir = Path(output_dir).resolve()
        if output_dir.exists():
            raise ShadowValidationError(
                f"Refusing to overwrite an existing shadow output: {output_dir}"
            )
        if status_callback:
            status_callback(f"Reusing compatible diagnostic run: {diagnostic_run}")

    if _sha256_file(preflight.frozen_candidate.config_path) != candidate_hash_before:
        raise ShadowValidationError("Frozen candidate config changed during the shadow run")
    if _sha256_file(preflight.student_map_path) != roster_hash_before:
        raise ShadowValidationError("Authoritative subject roster changed during the shadow run")

    tracklets, observations, diagnostic_summary, diagnostic_paths = validate_shadow_diagnostic_run(
        diagnostic_run, preflight
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    snapshot = output_dir / "candidate_config_snapshot.json"
    shutil.copyfile(preflight.frozen_candidate.config_path, snapshot)
    if _sha256_file(snapshot) != candidate_hash_before:
        raise ShadowValidationError("Frozen candidate snapshot differs from the verified source config")
    decisions = build_shadow_decisions(
        tracklets,
        preflight.frozen_candidate,
        preflight.subject_rolls,
    )
    recoveries = decisions[decisions["Shadow_Accepted"].eq("Yes")].copy()
    input_manifest = {
        "schema_version": SHADOW_SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "session_id": preflight.session_id,
        "session_fields": preflight.session_fields,
        "subject_abbr": preflight.subject_abbr,
        "authoritative_roster": {
            "path": str(preflight.student_map_path),
            "sha256": preflight.roster_sha256,
            "student_count": len(preflight.subject_rolls),
            "rolls": sorted(preflight.subject_rolls),
        },
        "candidate": {
            "candidate_id": preflight.frozen_candidate.candidate.candidate_id,
            "config_path": str(preflight.frozen_candidate.config_path),
            "config_sha256": preflight.frozen_candidate.config_sha256,
            "manifest_path": str(preflight.frozen_candidate.manifest_path),
            "enabled": False,
            "mode": "shadow_only",
            "production_approved": False,
        },
        "videos": _resolved_video_manifest(preflight, diagnostic_summary, repo_root),
        "recognition_and_tracklet_configuration": preflight.reference_configuration,
        "diagnostic_run": str(Path(diagnostic_run).resolve()),
        "recognition_rerun": recognition_rerun,
        "recognition_command": command,
        "official_match_threshold": MATCH_THRESHOLD,
        "official_margin_threshold": MARGIN_THRESHOLD,
        "attendance_checkpoint_rule": PRESENT_CHECKPOINTS,
        "production_changes": False,
        "official_attendance_written": False,
    }
    input_manifest_path = output_dir / "session_input_manifest.json"
    input_manifest_path.write_text(
        json.dumps(json_safe(input_manifest), indent=2, ensure_ascii=True), encoding="utf-8"
    )
    decisions_path = output_dir / "candidate_decisions.csv"
    recoveries_path = output_dir / "shadow_recoveries.csv"
    decisions.to_csv(decisions_path, index=False)
    recoveries.to_csv(recoveries_path, index=False)

    review_package = export_shadow_review_package(
        decisions=decisions,
        observation_df=observations,
        video_root=preflight.video_root,
        output_root=output_dir,
        student_map_path=preflight.student_map_path,
        diagnostic_run_id=Path(diagnostic_run).name,
        session_id=preflight.session_id,
        frozen_candidate=preflight.frozen_candidate,
        source_files={
            "tracklet_diagnostics": diagnostic_paths["tracklet_diagnostics"],
            "tracklet_observations": diagnostic_paths["tracklet_observations"],
            "candidate_decisions": decisions_path,
            "candidate_config_snapshot": snapshot,
            "session_input_manifest": input_manifest_path,
            "student_map": preflight.student_map_path,
        },
    )
    privacy = audit_blind_reviewer(review_package.root)
    if not privacy["privacy_passed"]:
        raise ShadowValidationError(
            "Blind reviewer privacy audit failed: " + ", ".join(privacy["forbidden_matches"][:10])
        )

    cli_script = repo_root / "scripts" / "validate_tracklet_ground_truth.py"
    launcher_command = (
        f"& {_powershell_quote(sys.executable)} {_powershell_quote(cli_script)} "
        f"open-shadow-review --review-package {_powershell_quote(review_package.root)}"
    )
    launcher_path = output_dir / "open_shadow_reviewer.ps1"
    launcher_path.write_text(launcher_command + "\n", encoding="utf-8")
    baseline_accepted = int(tracklets["Tracklet_Accepted"].map(_yes).sum())
    baseline_rejected = int(len(tracklets) - baseline_accepted)
    summary = {
        "schema_version": SHADOW_SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "session_id": preflight.session_id,
        "subject_abbr": preflight.subject_abbr,
        "candidate_id": preflight.frozen_candidate.candidate.candidate_id,
        "candidate_config_sha256": preflight.frozen_candidate.config_sha256,
        "candidate_enabled": False,
        "candidate_mode": "shadow_only",
        "production_approved": False,
        "baseline_tracks": int(len(tracklets)),
        "baseline_accepted": baseline_accepted,
        "baseline_rejected": baseline_rejected,
        "candidate_decision_rows": int(len(decisions)),
        "shadow_accepted": int(len(recoveries)),
        "review_status": "pending_human_review" if len(recoveries) else "not_required_zero_recoveries",
        "review_package": str(review_package.root.resolve()),
        "reviewer_launcher": str(launcher_path.resolve()),
        "privacy_audit": privacy,
        "recognition_rerun": recognition_rerun,
        "source_session": preflight.session_id,
        "authoritative_subject": preflight.subject_abbr,
        "production_changes": False,
        "official_thresholds_changed": False,
        "attendance_rule_changed": False,
        "embeddings_modified": False,
        "official_attendance_modified": False,
        "actual_present_roster_available": False,
        "attendance_recall_evaluated": False,
    }
    summary_path = output_dir / "shadow_run_summary.json"
    summary_path.write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=True), encoding="utf-8"
    )
    manifest_path = write_output_manifest(
        output_dir,
        {
            "candidate_id": preflight.frozen_candidate.candidate.candidate_id,
            "candidate_config_sha256": preflight.frozen_candidate.config_sha256,
            "session_id": preflight.session_id,
            "review_package_id": review_package.package_id,
            "production_changes": False,
            "official_attendance_modified": False,
        },
    )
    verify_output_manifest(output_dir, manifest_path)
    runtime_seconds = time.perf_counter() - started
    summary["runtime_seconds"] = round(runtime_seconds, 3)
    summary_path.write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=True), encoding="utf-8"
    )
    # Refresh the manifest after recording final runtime in the summary.
    write_output_manifest(
        output_dir,
        {
            "candidate_id": preflight.frozen_candidate.candidate.candidate_id,
            "candidate_config_sha256": preflight.frozen_candidate.config_sha256,
            "session_id": preflight.session_id,
            "review_package_id": review_package.package_id,
            "production_changes": False,
            "official_attendance_modified": False,
        },
    )
    verify_output_manifest(output_dir, manifest_path)
    return ShadowSessionRunResult(
        diagnostic_run=Path(diagnostic_run),
        output_dir=output_dir,
        review_package=review_package,
        decisions=decisions,
        summary=summary,
        recognition_rerun=recognition_rerun,
        runtime_seconds=runtime_seconds,
    )
