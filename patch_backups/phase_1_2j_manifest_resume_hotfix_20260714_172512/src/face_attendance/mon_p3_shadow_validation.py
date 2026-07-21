from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_DETECTION_SCORE,
    DEFAULT_MAX_WIDTH,
)
from .diagnostics import json_safe
from .embedding_version_family import EmbeddingFamilyError, verify_embedding_family
from .recall_analysis import MATCH_THRESHOLD, MARGIN_THRESHOLD, PRESENT_CHECKPOINTS, canonical_roll
from .session_identity import build_attendance_session_id, parse_attendance_session_id
from .shadow_validation import (
    REFERENCE_METADATA_KEYS,
    REQUIRED_TRACKLET_COLUMNS,
    ShadowValidationError,
    discover_session_videos,
    load_reference_configuration,
    verify_output_manifest,
    write_output_manifest,
)
from .tracklet_review import (
    LabelValidationError,
    ReviewExportError,
    export_review_package,
    load_review_package_data,
    load_student_mapping,
    validate_label_dataframe,
)


SCHEMA_VERSION = 1
POLICY_VERSION = "phase-1.2j-mon-p3-shadow-v1.1"
EXPECTED_SESSION_ID = "2026-06-22__B51__P3__CVO"
EXPECTED_SLOT_ID = "MON_P3"
EXPECTED_SUBJECT = "CVO"
EXPECTED_ROSTER_COUNT = 27
EXPECTED_FAMILY_DECISION = "built_unapproved_pending_mon_p3"
EXPECTED_PRODUCTION_EMBEDDINGS_SHA256 = (
    "c32ed31df10b7b9b43b8f19977a2adf0a82fb8e7a71f3e8bceb42c0565fedd49"
)
EXPECTED_PRODUCTION_SUMMARY_SHA256 = (
    "0df35c3bb9207e191aa49dad5536b8378e812d56463491bd7203885dca5ae9ee"
)
EXPECTED_RELATIVE_VIDEOS = (
    "CP1_1100\\back.mp4",
    "CP1_1100\\front.mp4",
    "CP2_1110\\back.mp4",
    "CP2_1110\\front.mp4",
    "CP3_1120\\back.mp4",
    "CP3_1120\\front.mp4",
    "CP4_1130\\back.mp4",
    "CP4_1130\\front.mp4",
    "CP5_1140\\back.mp4",
    "CP5_1140\\front.mp4",
)

REQUIRED_FREEZE_COLUMNS = {
    "RelativePath",
    "CanonicalPath",
    "CanonicalLength",
    "CanonicalSHA256",
    "MirrorPath",
    "MirrorLength",
    "MirrorSHA256",
    "ByteIdentical",
}

REQUIRED_OBSERVATION_COLUMNS = {
    "Tracklet_ID",
    "Observation_ID",
    "Checkpoint_ID",
    "Camera_ID",
    "Video",
    "Frame",
    "BBox_Original_Coordinates",
    "Zone_ID",
    "Frame_Best_Roll",
    "Frame_Best_Score",
    "Frame_Second_Roll",
    "Frame_Second_Score",
    "Frame_Margin",
    "Frame_Accepted",
    "Frame_Matcher_Reason",
    "Tracklet_Best_Roll",
    "Tracklet_Accepted",
}

REQUIRED_FACE_COLUMNS = {
    "Session_ID",
    "Subject_Abbr",
    "Checkpoint_ID",
    "Camera_ID",
    "Source_File_Name",
    "Frame",
    "Face_Index",
    "BBox_Original_Coordinates",
    "Detection_Source",
    "Zone_ID",
    "Selected_Source",
    "Accepted",
    "Best_Roll",
    "Best_Score",
    "Second_Roll",
    "Second_Score",
    "Margin",
    "Actual_Matcher_Reject_Reason",
}

TRACKLET_IDENTITY_COLUMNS = (
    "Tracklet_Best_Roll",
    "Tracklet_Best_Score",
    "Tracklet_Second_Roll",
    "Tracklet_Second_Score",
    "Tracklet_Margin",
    "Tracklet_Accepted",
    "Tracklet_Matcher_Reason",
)

TRACKLET_STRUCTURE_COLUMNS = (
    "Session_ID",
    "Subject_Abbr",
    "Tracklet_ID",
    "Checkpoint_ID",
    "Camera_ID",
    "Video",
    "Observation_Count",
    "Selected_Observation_Count",
    "Embedding_Count",
    "Consistent_Embedding_Count",
    "Inconsistent_Embedding_Count",
    "Selected_Observation_IDs",
    "Zone_IDs",
    "Match_Threshold",
    "Margin_Threshold",
    "Aggregate_Mode",
)

OBSERVATION_STRUCTURE_COLUMNS = (
    "Tracklet_ID",
    "Observation_ID",
    "Checkpoint_ID",
    "Camera_ID",
    "Video",
    "Frame",
    "BBox_Original_Coordinates",
    "Zone_ID",
    "Selected_For_Aggregation",
    "Embedding_Consistent",
    "Embedding_Extraction_Success",
    "Embedding_Dimension",
)

OBSERVATION_IDENTITY_COLUMNS = (
    "Frame_Best_Roll",
    "Frame_Best_Score",
    "Frame_Second_Roll",
    "Frame_Second_Score",
    "Frame_Margin",
    "Frame_Accepted",
    "Frame_Matcher_Reason",
    "Tracklet_Best_Roll",
    "Tracklet_Accepted",
)

FACE_STRUCTURE_COLUMNS = (
    "Session_ID",
    "Subject_Abbr",
    "Checkpoint_ID",
    "Camera_ID",
    "Source_File_Name",
    "Frame",
    "Face_Index",
    "BBox_Original_Coordinates",
    "Detection_Source",
    "Zone_ID",
    "Selected_Source",
)

FACE_IDENTITY_COLUMNS = (
    "Accepted",
    "Best_Roll",
    "Best_Score",
    "Second_Roll",
    "Second_Score",
    "Margin",
    "Actual_Matcher_Reject_Reason",
)


class MonP3ShadowError(ValueError):
    pass


@dataclass(frozen=True)
class FreezeVerification:
    json_path: Path
    csv_path: Path
    json_sha256: str
    csv_sha256: str
    canonical_root: Path
    mirror_root: Path
    rows: list[dict[str, Any]]


@dataclass(frozen=True)
class FamilyVerification:
    family_dir: Path
    family_id: str
    family_content_fingerprint: str
    family_output_manifest_sha256: str
    candidate_embeddings: Path
    candidate_embeddings_sha256: str
    candidate_variant_id: str
    candidate_embedding_records: int


@dataclass(frozen=True)
class MonP3Preflight:
    repo_root: Path
    session_id: str
    session_fields: dict[str, str]
    slot_id: str
    subject_abbr: str
    video_root: Path
    mirror_video_root: Path
    videos: list[dict[str, Any]]
    freeze: FreezeVerification
    family: FamilyVerification
    production_embeddings: Path
    production_summary: Path
    production_embeddings_sha256: str
    production_summary_sha256: str
    student_map: Path
    roster_sha256: str
    subject_rolls: set[str]
    timetable: Path
    timetable_sha256: str
    reference_diagnostic_run: Path
    reference_configuration: dict[str, Any]
    camera_zones: Path
    camera_zones_sha256: str
    fingerprint_sha256: str
    run_id: str
    output_root: Path
    output_dir: Path
    production_diagnostic_run: Path
    candidate_diagnostic_run: Path


@dataclass(frozen=True)
class DiagnosticArtifacts:
    root: Path
    summary_path: Path
    face_path: Path
    tracklet_path: Path
    observations_path: Path
    summary: dict[str, Any]
    faces: pd.DataFrame
    tracklets: pd.DataFrame
    observations: pd.DataFrame


@dataclass(frozen=True)
class ComparisonResult:
    tracklets: pd.DataFrame
    observations: pd.DataFrame
    faces: pd.DataFrame
    exception_tracks: pd.DataFrame
    frame_exception_rows: pd.DataFrame
    student_checkpoint: pd.DataFrame
    session_summary: dict[str, Any]


@dataclass(frozen=True)
class MonP3RunResult:
    output_dir: Path
    production_diagnostic_run: Path
    candidate_diagnostic_run: Path
    review_package: Path
    summary: dict[str, Any]
    idempotent_reuse: bool


def _text(value: Any) -> str:
    return str(value or "").strip()


def _canonical(value: Any) -> str:
    return canonical_roll(value)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _yes(value: Any) -> bool:
    return _text(value).lower() in {"yes", "true", "1"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_digest(payload: Any) -> str:
    encoded = json.dumps(json_safe(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _same_path(left: Path | str, right: Path | str) -> bool:
    left_text = os.path.normcase(os.path.normpath(str(Path(left).resolve())))
    right_text = os.path.normcase(os.path.normpath(str(Path(right).resolve())))
    return left_text == right_text


def _path_key(value: Path | str) -> str:
    return os.path.normcase(os.path.normpath(str(Path(value).resolve())))


def _checkpoint_id_from_relative_path(relative_path: str) -> str:
    folder = str(relative_path).replace("\\", "/").split("/", 1)[0]
    match = re.fullmatch(r"(CP\d+)(?:_.*)?", folder, flags=re.IGNORECASE)
    if match is None:
        raise MonP3ShadowError(
            f"Frozen MON_P3 relative path does not start with a checkpoint folder: {relative_path}"
        )
    return match.group(1).upper()


def _validate_diagnostic_source_layout(
    summary: dict[str, Any],
    preflight: MonP3Preflight,
) -> dict[str, Any]:
    expected_by_path = {
        _path_key(row["canonical_path"]): {
            "relative_path": row["relative_path"],
            "checkpoint_id": _checkpoint_id_from_relative_path(row["relative_path"]),
        }
        for row in preflight.freeze.rows
    }

    input_files = list(summary.get("input_files") or [])
    actual_input_paths = [_path_key(row.get("source_file", "")) for row in input_files]
    duplicate_input_paths = sorted(
        path
        for path in set(actual_input_paths)
        if actual_input_paths.count(path) > 1
    )
    missing_input_paths = sorted(set(expected_by_path).difference(actual_input_paths))
    unexpected_input_paths = sorted(set(actual_input_paths).difference(expected_by_path))
    if (
        len(input_files) != 10
        or duplicate_input_paths
        or missing_input_paths
        or unexpected_input_paths
    ):
        missing_relative = [
            expected_by_path[path]["relative_path"] for path in missing_input_paths
        ]
        raise MonP3ShadowError(
            "Diagnostic source layout does not match the frozen ten-video checkpoint set; "
            f"rows={len(input_files)}, duplicate_paths={len(duplicate_input_paths)}, "
            f"missing={missing_relative[:5]}, unexpected={unexpected_input_paths[:5]}"
        )

    timing_rows = list(summary.get("checkpoint_timing_map") or [])
    timing_paths = [_path_key(row.get("Source_File", "")) for row in timing_rows]
    duplicate_timing_paths = sorted(
        path
        for path in set(timing_paths)
        if timing_paths.count(path) > 1
    )
    missing_timing_paths = sorted(set(expected_by_path).difference(timing_paths))
    unexpected_timing_paths = sorted(set(timing_paths).difference(expected_by_path))
    if (
        len(timing_rows) != 10
        or duplicate_timing_paths
        or missing_timing_paths
        or unexpected_timing_paths
    ):
        missing_relative = [
            expected_by_path[path]["relative_path"] for path in missing_timing_paths
        ]
        raise MonP3ShadowError(
            "Diagnostic checkpoint timing map is not one-to-one with the frozen videos; "
            f"rows={len(timing_rows)}, duplicate_paths={len(duplicate_timing_paths)}, "
            f"missing={missing_relative[:5]}, unexpected={unexpected_timing_paths[:5]}"
        )

    mismatches: list[str] = []
    for row in timing_rows:
        path = _path_key(row.get("Source_File", ""))
        expected = expected_by_path[path]
        actual_checkpoint = _text(row.get("Checkpoint_ID")).upper()
        if actual_checkpoint != expected["checkpoint_id"]:
            mismatches.append(
                f"{expected['relative_path']} expected {expected['checkpoint_id']} "
                f"but diagnostic recorded {actual_checkpoint or '<blank>'}"
            )
    if mismatches:
        raise MonP3ShadowError(
            "Diagnostic checkpoint folder mapping is invalid: " + "; ".join(mismatches[:5])
        )

    return {
        "validated_video_count": len(input_files),
        "validated_timing_rows": len(timing_rows),
        "checkpoint_ids": sorted(
            {_text(row.get("Checkpoint_ID")).upper() for row in timing_rows}
        ),
        "one_to_one_checkpoint_source_mapping": True,
    }


def _load_json(path: Path, label: str) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise MonP3ShadowError(f"{label} not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MonP3ShadowError(f"Could not read {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise MonP3ShadowError(f"{label} must contain a JSON object")
    return payload


def _single_file(directory: Path, pattern: str, label: str) -> Path:
    matches = sorted(Path(directory).glob(pattern))
    if len(matches) != 1:
        raise MonP3ShadowError(
            f"Expected exactly one {label} matching {pattern} in {directory}; found {len(matches)}"
        )
    return matches[0]


def _read_csv_or_empty(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=sorted(set(columns)))
    except Exception as exc:
        raise MonP3ShadowError(f"Could not read CSV {path}: {exc}") from exc


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = _text(value).lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    raise MonP3ShadowError(f"Invalid Boolean value in source freeze: {value!r}")


def verify_source_freeze(
    *,
    freeze_json_path: Path,
    freeze_csv_path: Path,
    canonical_root: Path,
    mirror_root: Path,
    session_id: str = EXPECTED_SESSION_ID,
) -> FreezeVerification:
    canonical_root = Path(canonical_root).resolve()
    mirror_root = Path(mirror_root).resolve()
    payload = _load_json(Path(freeze_json_path), "MON_P3 source-freeze JSON")
    csv_path = Path(freeze_csv_path)
    if not csv_path.is_file():
        raise MonP3ShadowError(f"MON_P3 source-freeze CSV not found: {csv_path}")
    try:
        rows_df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except Exception as exc:
        raise MonP3ShadowError(f"Could not read MON_P3 source-freeze CSV: {exc}") from exc
    missing_columns = sorted(REQUIRED_FREEZE_COLUMNS.difference(rows_df.columns))
    if missing_columns:
        raise MonP3ShadowError(
            "MON_P3 source-freeze CSV is missing columns: " + ", ".join(missing_columns)
        )
    if int(payload.get("SchemaVersion", -1)) != 1 or _text(payload.get("Phase")) != "1.2J":
        raise MonP3ShadowError("MON_P3 source-freeze JSON schema or phase is invalid")
    if _text(payload.get("SessionID")) != session_id:
        raise MonP3ShadowError("MON_P3 source-freeze session ID does not match the requested session")
    required_false = ("RecognitionExecuted", "AttendanceWritten", "CandidatePromoted")
    if any(payload.get(key) is not False for key in required_false):
        raise MonP3ShadowError("MON_P3 source freeze claims recognition, attendance, or promotion occurred")
    required_true = (
        "ExpectedLayoutPassed",
        "TreesByteIdentical",
        "SourceFreezePassed",
        "CanonicalSourceOnly",
    )
    if any(payload.get(key) is not True for key in required_true):
        raise MonP3ShadowError("MON_P3 source freeze did not pass all required safeguards")
    if int(payload.get("CanonicalVideoCount", -1)) != 10 or int(payload.get("MirrorVideoCount", -1)) != 10:
        raise MonP3ShadowError("MON_P3 source freeze must contain exactly ten canonical and ten mirror videos")
    if not _same_path(payload.get("CanonicalVideoRoot", ""), canonical_root):
        raise MonP3ShadowError("MON_P3 source-freeze canonical root does not match the requested root")
    if not _same_path(payload.get("MirrorVideoRoot", ""), mirror_root):
        raise MonP3ShadowError("MON_P3 source-freeze mirror root does not match the requested root")
    recorded_csv = _text(payload.get("CSV"))
    if not recorded_csv or not _same_path(recorded_csv, csv_path):
        raise MonP3ShadowError("MON_P3 source-freeze JSON does not point to the supplied CSV")
    if len(rows_df) != 10:
        raise MonP3ShadowError(f"MON_P3 source-freeze CSV must contain ten rows; found {len(rows_df)}")
    actual_relative = sorted(rows_df["RelativePath"].astype(str).tolist())
    expected_relative = sorted(EXPECTED_RELATIVE_VIDEOS)
    if actual_relative != expected_relative:
        raise MonP3ShadowError(
            f"MON_P3 source-freeze path set is invalid; expected={expected_relative}, actual={actual_relative}"
        )
    if rows_df["RelativePath"].duplicated().any():
        raise MonP3ShadowError("MON_P3 source-freeze CSV contains duplicate relative paths")

    verified_rows: list[dict[str, Any]] = []
    for row in rows_df.to_dict("records"):
        relative = _text(row["RelativePath"])
        canonical_path = canonical_root / Path(relative.replace("\\", os.sep))
        mirror_path = mirror_root / Path(relative.replace("\\", os.sep))
        if not canonical_path.is_file() or not mirror_path.is_file():
            raise MonP3ShadowError(f"Frozen MON_P3 source is missing: {relative}")
        if not _same_path(row["CanonicalPath"], canonical_path):
            raise MonP3ShadowError(f"Canonical path changed since source freeze: {relative}")
        if not _same_path(row["MirrorPath"], mirror_path):
            raise MonP3ShadowError(f"Mirror path changed since source freeze: {relative}")
        canonical_length = canonical_path.stat().st_size
        mirror_length = mirror_path.stat().st_size
        canonical_hash = _sha256_file(canonical_path)
        mirror_hash = _sha256_file(mirror_path)
        recorded_canonical_length = _int(row["CanonicalLength"], -1)
        recorded_mirror_length = _int(row["MirrorLength"], -1)
        recorded_canonical_hash = _text(row["CanonicalSHA256"]).lower()
        recorded_mirror_hash = _text(row["MirrorSHA256"]).lower()
        if canonical_length != recorded_canonical_length or mirror_length != recorded_mirror_length:
            raise MonP3ShadowError(f"MON_P3 video length changed after source freeze: {relative}")
        if canonical_hash != recorded_canonical_hash or mirror_hash != recorded_mirror_hash:
            raise MonP3ShadowError(f"MON_P3 video hash changed after source freeze: {relative}")
        if not _parse_bool(row["ByteIdentical"]):
            raise MonP3ShadowError(f"MON_P3 source freeze marks a non-identical mirror: {relative}")
        if canonical_length != mirror_length or canonical_hash != mirror_hash:
            raise MonP3ShadowError(f"MON_P3 canonical and mirror files are not byte-identical: {relative}")
        verified_rows.append(
            {
                "relative_path": relative,
                "canonical_path": str(canonical_path),
                "mirror_path": str(mirror_path),
                "length": canonical_length,
                "sha256": canonical_hash,
            }
        )
    if len({row["sha256"] for row in verified_rows}) != len(verified_rows):
        raise MonP3ShadowError("MON_P3 source freeze contains duplicate video content")
    return FreezeVerification(
        json_path=Path(freeze_json_path).resolve(),
        csv_path=csv_path.resolve(),
        json_sha256=_sha256_file(Path(freeze_json_path)),
        csv_sha256=_sha256_file(csv_path),
        canonical_root=canonical_root,
        mirror_root=mirror_root,
        rows=verified_rows,
    )


def verify_candidate_family(
    *,
    family_dir: Path,
    production_embeddings: Path,
    production_summary: Path,
    expected_family_id: str = "",
) -> FamilyVerification:
    family_dir = Path(family_dir).resolve()
    try:
        verified = verify_embedding_family(
            family_dir,
            production_embeddings_path=Path(production_embeddings),
            production_summary_path=Path(production_summary),
        )
    except EmbeddingFamilyError as exc:
        raise MonP3ShadowError(str(exc)) from exc
    family_manifest = _load_json(family_dir / "family_manifest.json", "family manifest")
    evaluation_decision = _load_json(
        family_dir / "evaluation" / "evaluation_decision.json", "family evaluation decision"
    )
    family_output_manifest = family_dir / "output_manifest.json"
    if not family_output_manifest.is_file():
        raise MonP3ShadowError("Candidate family output manifest is missing")
    family_id = _text(family_manifest.get("family_id"))
    if expected_family_id and family_id != expected_family_id:
        raise MonP3ShadowError(
            f"Unexpected family ID. Expected {expected_family_id}, found {family_id}"
        )
    if _text(verified.get("decision")) != EXPECTED_FAMILY_DECISION:
        raise MonP3ShadowError(
            f"Candidate family must be {EXPECTED_FAMILY_DECISION}; found {verified.get('decision')}"
        )
    if _text(evaluation_decision.get("exact_next_step")) != "PHASE 1.2J - MON_P3 untouched shadow validation.":
        raise MonP3ShadowError("Candidate family does not explicitly authorize Phase 1.2J")
    if any(
        value is not False
        for value in (
            family_manifest.get("production_approved"),
            family_manifest.get("candidate_promoted"),
            family_manifest.get("mon_p3_processed"),
            evaluation_decision.get("production_approved"),
            evaluation_decision.get("candidate_promoted"),
            evaluation_decision.get("mon_p3_processed"),
        )
    ):
        raise MonP3ShadowError("Candidate family violates the unapproved/untouched contract")
    variant_manifest_path = family_dir / "variants" / "full_candidate" / "version_manifest.json"
    variant_manifest = _load_json(variant_manifest_path, "full-candidate variant manifest")
    candidate_embeddings = family_dir / "variants" / "full_candidate" / "student_embeddings.pkl"
    if not candidate_embeddings.is_file():
        raise MonP3ShadowError("Full-candidate embedding payload is missing")
    if _text(variant_manifest.get("status")) != "built_unapproved":
        raise MonP3ShadowError("Full candidate is not in built_unapproved state")
    if variant_manifest.get("availability") is not True:
        raise MonP3ShadowError("Full candidate is not available")
    if variant_manifest.get("production_promoted") is not False:
        raise MonP3ShadowError("Full candidate has already been promoted")
    if variant_manifest.get("training_contaminated_descriptive_only") is not True:
        raise MonP3ShadowError("Full candidate descriptive-only contract changed")
    if list(variant_manifest.get("evaluation_sessions_allowed") or []):
        raise MonP3ShadowError("Full candidate unexpectedly allows a training-session evaluation")
    candidate_variant_id = _text(variant_manifest.get("version_id"))
    candidate_embedding_records = _int(variant_manifest.get("embedding_records"), -1)
    family_content_fingerprint = _text(family_manifest.get("content_fingerprint_sha256"))
    if not candidate_variant_id or candidate_embedding_records <= 0:
        raise MonP3ShadowError("Full-candidate variant ID or embedding record count is invalid")
    if not family_content_fingerprint:
        raise MonP3ShadowError("Candidate family content fingerprint is missing")
    recorded_hash = _text(
        dict(variant_manifest.get("artifact_sha256") or {}).get("student_embeddings.pkl")
    )
    actual_hash = _sha256_file(candidate_embeddings)
    if not recorded_hash or recorded_hash != actual_hash:
        raise MonP3ShadowError("Full-candidate embedding hash does not match its manifest")
    return FamilyVerification(
        family_dir=family_dir,
        family_id=family_id,
        family_content_fingerprint=family_content_fingerprint,
        family_output_manifest_sha256=_sha256_file(family_output_manifest),
        candidate_embeddings=candidate_embeddings.resolve(),
        candidate_embeddings_sha256=actual_hash,
        candidate_variant_id=candidate_variant_id,
        candidate_embedding_records=candidate_embedding_records,
    )


def _validate_timetable(timetable: Path, slot_id: str, subject_abbr: str) -> None:
    try:
        frame = pd.read_csv(timetable, dtype=str, keep_default_na=False)
    except Exception as exc:
        raise MonP3ShadowError(f"Could not read timetable: {exc}") from exc
    if "slot_id" not in frame.columns:
        raise MonP3ShadowError("Timetable is missing slot_id")
    rows = frame[frame["slot_id"].astype(str).str.upper().eq(slot_id.upper())]
    if len(rows) != 1:
        raise MonP3ShadowError(f"Timetable must contain exactly one {slot_id} row; found {len(rows)}")
    row = rows.iloc[0]
    if _text(row.get("period")) not in {"3", "P3"}:
        raise MonP3ShadowError(f"{slot_id} is not period P3")
    if _text(row.get("section")).upper() != "B51":
        raise MonP3ShadowError(f"{slot_id} is not section B51")
    abbrs = {_text(value).upper() for value in _text(row.get("course_abbr")).replace("/", ",").split(",")}
    if subject_abbr.upper() not in abbrs:
        raise MonP3ShadowError(f"{slot_id} does not include subject {subject_abbr}")


def _resolve_camera_zones(repo_root: Path, reference_configuration: dict[str, Any]) -> Path:
    camera_zones = Path(_text(reference_configuration.get("camera_zones")))
    if not camera_zones.is_absolute():
        camera_zones = Path(repo_root) / camera_zones
    camera_zones = camera_zones.resolve()
    if not camera_zones.is_file():
        raise MonP3ShadowError(f"Reference camera-zone configuration not found: {camera_zones}")
    return camera_zones


def preflight_mon_p3_shadow(
    *,
    repo_root: Path,
    family_dir: Path,
    freeze_json: Path,
    freeze_csv: Path,
    canonical_video_root: Path,
    mirror_video_root: Path,
    reference_diagnostic_run: Path,
    student_map: Path,
    timetable: Path,
    production_embeddings: Path,
    production_summary: Path,
    output_root: Path,
    session_id: str = EXPECTED_SESSION_ID,
    slot_id: str = EXPECTED_SLOT_ID,
    subject_abbr: str = EXPECTED_SUBJECT,
    expected_family_id: str = "",
    metadata_reader: Callable[[Path], dict[str, Any]] | None = None,
) -> MonP3Preflight:
    repo_root = Path(repo_root).resolve()
    production_embeddings = Path(production_embeddings).resolve()
    production_summary = Path(production_summary).resolve()
    student_map = Path(student_map).resolve()
    timetable = Path(timetable).resolve()
    reference_diagnostic_run = Path(reference_diagnostic_run).resolve()
    output_root = Path(output_root).resolve()
    for label, path in (
        ("production embeddings", production_embeddings),
        ("production summary", production_summary),
        ("student mapping", student_map),
        ("timetable", timetable),
    ):
        if not path.is_file():
            raise MonP3ShadowError(f"Required {label} not found: {path}")
    production_embeddings_sha256 = _sha256_file(production_embeddings)
    production_summary_sha256 = _sha256_file(production_summary)
    if production_embeddings_sha256 != EXPECTED_PRODUCTION_EMBEDDINGS_SHA256:
        raise MonP3ShadowError(
            "Production embedding hash changed before MON_P3 shadow validation"
        )
    if production_summary_sha256 != EXPECTED_PRODUCTION_SUMMARY_SHA256:
        raise MonP3ShadowError("Production summary hash changed before MON_P3 shadow validation")
    try:
        session_fields = parse_attendance_session_id(session_id)
    except ValueError as exc:
        raise MonP3ShadowError(str(exc)) from exc
    expected_session = build_attendance_session_id(
        session_fields["session_date"],
        session_fields["section"],
        session_fields["period"],
        subject_abbr,
    )
    if session_id != expected_session:
        raise MonP3ShadowError(f"Session ID must resolve exactly to {expected_session}")
    if session_fields["session_date"] != "2026-06-22" or session_fields["period"] != "P3":
        raise MonP3ShadowError("Phase 1.2J is restricted to untouched 2026-06-22 B51 P3 CVO")
    if slot_id != EXPECTED_SLOT_ID or subject_abbr.upper() != EXPECTED_SUBJECT:
        raise MonP3ShadowError("Phase 1.2J slot or subject changed from MON_P3/CVO")

    freeze = verify_source_freeze(
        freeze_json_path=freeze_json,
        freeze_csv_path=freeze_csv,
        canonical_root=canonical_video_root,
        mirror_root=mirror_video_root,
        session_id=session_id,
    )
    family = verify_candidate_family(
        family_dir=family_dir,
        production_embeddings=production_embeddings,
        production_summary=production_summary,
        expected_family_id=expected_family_id,
    )
    try:
        videos = discover_session_videos(freeze.canonical_root, metadata_reader=metadata_reader)
    except ShadowValidationError as exc:
        raise MonP3ShadowError(str(exc)) from exc
    frozen_by_relative = {row["relative_path"].replace("\\", "/"): row for row in freeze.rows}
    for video in videos:
        relative = Path(video["source_path"]).relative_to(freeze.canonical_root).as_posix()
        frozen = frozen_by_relative.get(relative)
        if frozen is None or frozen["sha256"] != video["sha256"] or frozen["length"] != video["file_size_bytes"]:
            raise MonP3ShadowError(f"Discovered MON_P3 video does not match source freeze: {relative}")

    try:
        roster, _, subject_rolls_raw = load_student_mapping(student_map, subject_abbr)
    except ReviewExportError as exc:
        raise MonP3ShadowError(str(exc)) from exc
    subject_roster = [row for row in roster if row["in_subject_roster"]]
    subject_rolls = {_canonical(value) for value in subject_rolls_raw if _canonical(value)}
    if len(subject_rolls) != EXPECTED_ROSTER_COUNT or len(subject_roster) != EXPECTED_ROSTER_COUNT:
        raise MonP3ShadowError(
            f"Authoritative {subject_abbr} roster must contain {EXPECTED_ROSTER_COUNT} students; found {len(subject_rolls)}"
        )
    _validate_timetable(timetable, slot_id, subject_abbr)
    try:
        reference_configuration = load_reference_configuration(reference_diagnostic_run)
    except ShadowValidationError as exc:
        raise MonP3ShadowError(str(exc)) from exc
    camera_zones = _resolve_camera_zones(repo_root, reference_configuration)

    fingerprint_payload = {
        "policy_version": POLICY_VERSION,
        "session_id": session_id,
        "slot_id": slot_id,
        "subject_abbr": subject_abbr,
        "family_id": family.family_id,
        "family_content_fingerprint": family.family_content_fingerprint,
        "family_output_manifest_sha256": family.family_output_manifest_sha256,
        "candidate_embeddings_sha256": family.candidate_embeddings_sha256,
        "production_embeddings_sha256": production_embeddings_sha256,
        "production_summary_sha256": production_summary_sha256,
        "freeze_json_sha256": freeze.json_sha256,
        "freeze_csv_sha256": freeze.csv_sha256,
        "video_hashes": {row["relative_path"]: row["sha256"] for row in freeze.rows},
        "student_map_sha256": _sha256_file(student_map),
        "timetable_sha256": _sha256_file(timetable),
        "camera_zones_sha256": _sha256_file(camera_zones),
        "reference_configuration": reference_configuration,
    }
    fingerprint_sha256 = _stable_digest(fingerprint_payload)
    run_id = f"mon-p3-shadow-{fingerprint_sha256[:20]}"
    output_dir = output_root / run_id
    diagnostics_root = repo_root / "attendance_output" / "diagnostics"
    production_diagnostic_run = diagnostics_root / f"{session_id}_PHASE_12J_PRODUCTION_{fingerprint_sha256[:12]}"
    candidate_diagnostic_run = diagnostics_root / f"{session_id}_PHASE_12J_CANDIDATE_{fingerprint_sha256[:12]}"
    return MonP3Preflight(
        repo_root=repo_root,
        session_id=session_id,
        session_fields=session_fields,
        slot_id=slot_id,
        subject_abbr=subject_abbr.upper(),
        video_root=freeze.canonical_root,
        mirror_video_root=freeze.mirror_root,
        videos=videos,
        freeze=freeze,
        family=family,
        production_embeddings=production_embeddings,
        production_summary=production_summary,
        production_embeddings_sha256=production_embeddings_sha256,
        production_summary_sha256=production_summary_sha256,
        student_map=student_map,
        roster_sha256=_sha256_file(student_map),
        subject_rolls=subject_rolls,
        timetable=timetable,
        timetable_sha256=_sha256_file(timetable),
        reference_diagnostic_run=reference_diagnostic_run,
        reference_configuration=reference_configuration,
        camera_zones=camera_zones,
        camera_zones_sha256=_sha256_file(camera_zones),
        fingerprint_sha256=fingerprint_sha256,
        run_id=run_id,
        output_root=output_root,
        output_dir=output_dir,
        production_diagnostic_run=production_diagnostic_run,
        candidate_diagnostic_run=candidate_diagnostic_run,
    )


def build_diagnostic_command(
    preflight: MonP3Preflight,
    *,
    embeddings_path: Path,
    diagnostic_run_id: str,
    python_executable: Path | None = None,
) -> list[str]:
    config = preflight.reference_configuration
    input_source = str(preflight.video_root)
    try:
        input_source = preflight.video_root.relative_to(preflight.repo_root).as_posix()
    except ValueError:
        pass
    executable = Path(python_executable or sys.executable).resolve()
    return [
        str(executable),
        str((preflight.repo_root / "scripts" / "mark_attendance_checkpoints.py").resolve()),
        "--timetable",
        str(preflight.timetable),
        "--slot-id",
        preflight.slot_id,
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
        preflight.slot_id,
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
        str(_int(config["frame_skip"])),
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
        str((preflight.repo_root / "attendance_output" / "diagnostics").resolve()),
        "--diagnostic-run-id",
        diagnostic_run_id,
        "--zone-mode",
        _text(config["zone_mode"]),
        "--camera-zones",
        str(preflight.camera_zones),
        "--zone-profile",
        _text(config["zone_profile"]),
        "--zone-merge-iou",
        str(_number(config["zone_merge_iou"])),
        "--tracklet-mode",
        _text(config["tracklet_mode"]),
        "--tracklet-min-observations",
        str(_int(config["tracklet_min_observations"])),
        "--tracklet-max-selected",
        str(_int(config["tracklet_max_selected"])),
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


def _read_diagnostic_artifacts(
    diagnostic_run: Path,
    preflight: MonP3Preflight,
) -> DiagnosticArtifacts:
    diagnostic_run = Path(diagnostic_run).resolve()
    if not diagnostic_run.is_dir():
        raise MonP3ShadowError(f"Diagnostic run not found: {diagnostic_run}")
    summary_path = _single_file(diagnostic_run, "diagnostic_summary_*.json", "diagnostic summary")
    face_path = _single_file(diagnostic_run, "face_diagnostics_*.csv", "face diagnostics")
    tracklet_path = _single_file(diagnostic_run, "tracklet_diagnostics_*.csv", "tracklet diagnostics")
    observations_path = _single_file(
        diagnostic_run, "tracklet_observations_*.csv", "tracklet observations"
    )
    summary = _load_json(summary_path, "diagnostic summary")
    metadata = dict(summary.get("run_metadata") or {})
    if _text(metadata.get("session_id")) != preflight.session_id:
        raise MonP3ShadowError("Diagnostic session ID does not match Phase 1.2J preflight")
    if _text(metadata.get("subject_abbr")).upper() != preflight.subject_abbr:
        raise MonP3ShadowError("Diagnostic subject does not match Phase 1.2J preflight")
    if _text(metadata.get("slot_id")).upper() != preflight.slot_id:
        raise MonP3ShadowError("Diagnostic slot does not match MON_P3")
    if _text(metadata.get("period")).upper() != "P3":
        raise MonP3ShadowError("Diagnostic period does not match P3")
    if metadata.get("diagnostic_only") is not True:
        raise MonP3ShadowError("Diagnostic run is not marked diagnostic_only=true")
    if metadata.get("official_attendance_written") is not False:
        raise MonP3ShadowError("Diagnostic run indicates official attendance was written")
    if dict(summary.get("normal_output_files") or {}):
        raise MonP3ShadowError("Diagnostic-only run unexpectedly contains normal attendance outputs")
    if not summary.get("checkpoint_extraction_valid"):
        raise MonP3ShadowError("Diagnostic checkpoint extraction is invalid")
    reference = preflight.reference_configuration
    exact_keys = (
        "checkpoint_mode",
        "log_mode",
        "aggregate",
        "zone_mode",
        "zone_profile",
        "tracklet_mode",
        "official_recognition_unit",
    )
    numeric_keys = tuple(key for key in REFERENCE_METADATA_KEYS if key not in exact_keys and key != "camera_zones")
    for key in exact_keys:
        if _text(metadata.get(key)) != _text(reference.get(key)):
            raise MonP3ShadowError(f"Diagnostic setting {key} differs from frozen reference")
    for key in numeric_keys:
        if abs(_number(metadata.get(key), -999.0) - _number(reference.get(key), -998.0)) > 1e-9:
            raise MonP3ShadowError(f"Diagnostic setting {key} differs from frozen reference")
    _validate_diagnostic_source_layout(summary, preflight)

    faces = _read_csv_or_empty(
        face_path,
        set(REQUIRED_FACE_COLUMNS) | set(FACE_STRUCTURE_COLUMNS) | set(FACE_IDENTITY_COLUMNS),
    )
    tracklets = _read_csv_or_empty(
        tracklet_path,
        set(REQUIRED_TRACKLET_COLUMNS)
        | set(TRACKLET_STRUCTURE_COLUMNS)
        | set(TRACKLET_IDENTITY_COLUMNS),
    )
    observations = _read_csv_or_empty(
        observations_path,
        set(REQUIRED_OBSERVATION_COLUMNS)
        | set(OBSERVATION_STRUCTURE_COLUMNS)
        | set(OBSERVATION_IDENTITY_COLUMNS),
    )
    for label, frame, required in (
        ("face diagnostics", faces, REQUIRED_FACE_COLUMNS),
        ("tracklet diagnostics", tracklets, REQUIRED_TRACKLET_COLUMNS),
        ("tracklet observations", observations, REQUIRED_OBSERVATION_COLUMNS),
    ):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise MonP3ShadowError(f"{label} are missing columns: {', '.join(missing)}")
    if not tracklets.empty and tracklets["Tracklet_ID"].duplicated().any():
        raise MonP3ShadowError("Diagnostic tracklets contain duplicate Tracklet_ID values")
    if not observations.empty and observations["Observation_ID"].duplicated().any():
        raise MonP3ShadowError("Diagnostic observations contain duplicate Observation_ID values")
    return DiagnosticArtifacts(
        root=diagnostic_run,
        summary_path=summary_path,
        face_path=face_path,
        tracklet_path=tracklet_path,
        observations_path=observations_path,
        summary=summary,
        faces=faces,
        tracklets=tracklets,
        observations=observations,
    )


def _assert_same_structure(
    production: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    key: str,
    columns: Iterable[str],
    label: str,
) -> None:
    production_keys = set(production[key].astype(str))
    candidate_keys = set(candidate[key].astype(str))
    if production_keys != candidate_keys:
        raise MonP3ShadowError(
            f"{label} key set changed between production and candidate diagnostics; "
            f"production_only={sorted(production_keys - candidate_keys)[:5]}, "
            f"candidate_only={sorted(candidate_keys - production_keys)[:5]}"
        )
    left = production.set_index(key).sort_index()
    right = candidate.set_index(key).sort_index()
    for column in columns:
        if column == key:
            continue
        left_values = left[column].astype(str)
        right_values = right[column].astype(str)
        mismatch = left_values.ne(right_values)
        if mismatch.any():
            sample = mismatch[mismatch].index.astype(str).tolist()[:5]
            raise MonP3ShadowError(
                f"{label} structural column {column} changed between runs: {sample}"
            )


def _face_key(frame: pd.DataFrame) -> pd.Series:
    def normalize_bbox(value: Any) -> str:
        parts = [part.strip() for part in _text(value).split(",")]
        try:
            return ",".join(f"{float(part):.2f}" for part in parts[:4])
        except (TypeError, ValueError):
            return _text(value)

    return (
        frame["Checkpoint_ID"].astype(str)
        + "|"
        + frame["Camera_ID"].astype(str)
        + "|"
        + frame["Source_File_Name"].astype(str).str.lower()
        + "|"
        + frame["Frame"].astype(str)
        + "|"
        + frame["Face_Index"].astype(str)
        + "|"
        + frame["Detection_Source"].astype(str)
        + "|"
        + frame["Zone_ID"].astype(str)
        + "|"
        + frame["Selected_Source"].astype(str)
        + "|"
        + frame["BBox_Original_Coordinates"].map(normalize_bbox)
    )


def _build_tracklet_comparison(production: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    _assert_same_structure(
        production,
        candidate,
        key="Tracklet_ID",
        columns=TRACKLET_STRUCTURE_COLUMNS,
        label="tracklet diagnostics",
    )
    base = production.set_index("Tracklet_ID").sort_index()
    test = candidate.set_index("Tracklet_ID").sort_index()
    output = base[[column for column in TRACKLET_STRUCTURE_COLUMNS if column != "Tracklet_ID"]].copy()
    for column in TRACKLET_IDENTITY_COLUMNS:
        output[f"Production_{column}"] = base[column].astype(str)
        output[f"Candidate_{column}"] = test[column].astype(str)
    output = output.reset_index()
    prod_accepted = output["Production_Tracklet_Accepted"].map(_yes)
    cand_accepted = output["Candidate_Tracklet_Accepted"].map(_yes)
    prod_roll = output["Production_Tracklet_Best_Roll"].map(_canonical)
    cand_roll = output["Candidate_Tracklet_Best_Roll"].map(_canonical)
    conditions = [
        prod_accepted & cand_accepted & prod_roll.ne(cand_roll),
        ~prod_accepted & cand_accepted,
        prod_accepted & ~cand_accepted,
        prod_accepted & cand_accepted & prod_roll.eq(cand_roll),
    ]
    choices = [
        "accepted_identity_changed",
        "candidate_only_accept",
        "production_only_accept",
        "unchanged_accepted_same_identity",
    ]
    output["Tracklet_Delta_Type"] = np.select(conditions, choices, default="unchanged_rejected")
    output["Tracklet_Requires_Review"] = output["Tracklet_Delta_Type"].isin(
        {"accepted_identity_changed", "candidate_only_accept", "production_only_accept"}
    ).map(lambda value: "Yes" if value else "No")
    return output


def _build_observation_comparison(production: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    _assert_same_structure(
        production,
        candidate,
        key="Observation_ID",
        columns=OBSERVATION_STRUCTURE_COLUMNS,
        label="tracklet observations",
    )
    base = production.set_index("Observation_ID").sort_index()
    test = candidate.set_index("Observation_ID").sort_index()
    output = base[[column for column in OBSERVATION_STRUCTURE_COLUMNS if column != "Observation_ID"]].copy()
    for column in OBSERVATION_IDENTITY_COLUMNS:
        output[f"Production_{column}"] = base[column].astype(str)
        output[f"Candidate_{column}"] = test[column].astype(str)
    output = output.reset_index()
    prod_accepted = output["Production_Frame_Accepted"].map(_yes)
    cand_accepted = output["Candidate_Frame_Accepted"].map(_yes)
    prod_roll = output["Production_Frame_Best_Roll"].map(_canonical)
    cand_roll = output["Candidate_Frame_Best_Roll"].map(_canonical)
    output["Frame_Delta_Type"] = np.select(
        [
            prod_accepted & cand_accepted & prod_roll.ne(cand_roll),
            ~prod_accepted & cand_accepted,
            prod_accepted & ~cand_accepted,
            prod_accepted & cand_accepted & prod_roll.eq(cand_roll),
        ],
        [
            "accepted_identity_changed",
            "candidate_only_accept",
            "production_only_accept",
            "unchanged_accepted_same_identity",
        ],
        default="unchanged_rejected",
    )
    output["Frame_Requires_Review"] = output["Frame_Delta_Type"].isin(
        {"accepted_identity_changed", "candidate_only_accept", "production_only_accept"}
    ).map(lambda value: "Yes" if value else "No")
    return output


def _build_face_comparison(production: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    left = production.copy()
    right = candidate.copy()
    left["Face_Key"] = _face_key(left)
    right["Face_Key"] = _face_key(right)
    if left["Face_Key"].duplicated().any() or right["Face_Key"].duplicated().any():
        raise MonP3ShadowError("Face diagnostics contain duplicate stable geometry keys")
    _assert_same_structure(
        left,
        right,
        key="Face_Key",
        columns=FACE_STRUCTURE_COLUMNS,
        label="frame detections",
    )
    base = left.set_index("Face_Key").sort_index()
    test = right.set_index("Face_Key").sort_index()
    output = base[list(FACE_STRUCTURE_COLUMNS)].copy()
    for column in FACE_IDENTITY_COLUMNS:
        output[f"Production_{column}"] = base[column].astype(str)
        output[f"Candidate_{column}"] = test[column].astype(str)
    output = output.reset_index()
    prod_accepted = output["Production_Accepted"].map(_yes)
    cand_accepted = output["Candidate_Accepted"].map(_yes)
    prod_roll = output["Production_Best_Roll"].map(_canonical)
    cand_roll = output["Candidate_Best_Roll"].map(_canonical)
    output["Frame_Delta_Type"] = np.select(
        [
            prod_accepted & cand_accepted & prod_roll.ne(cand_roll),
            ~prod_accepted & cand_accepted,
            prod_accepted & ~cand_accepted,
            prod_accepted & cand_accepted & prod_roll.eq(cand_roll),
        ],
        [
            "accepted_identity_changed",
            "candidate_only_accept",
            "production_only_accept",
            "unchanged_accepted_same_identity",
        ],
        default="unchanged_rejected",
    )
    output["Frame_Requires_Review"] = output["Frame_Delta_Type"].isin(
        {"accepted_identity_changed", "candidate_only_accept", "production_only_accept"}
    ).map(lambda value: "Yes" if value else "No")
    return output


def _observation_geometry_key(frame: pd.DataFrame) -> pd.Series:
    def normalize_bbox(value: Any) -> str:
        parts = [part.strip() for part in _text(value).split(",")]
        try:
            return ",".join(f"{float(part):.2f}" for part in parts[:4])
        except (TypeError, ValueError):
            return _text(value)

    return (
        frame["Checkpoint_ID"].astype(str)
        + "|"
        + frame["Camera_ID"].astype(str)
        + "|"
        + frame["Video"].astype(str).str.lower()
        + "|"
        + frame["Frame"].astype(str)
        + "|"
        + frame["Zone_ID"].astype(str)
        + "|"
        + frame["BBox_Original_Coordinates"].map(normalize_bbox)
    )


def _parse_bbox(value: Any) -> tuple[float, float, float, float] | None:
    parts = [part.strip() for part in _text(value).split(",")]
    if len(parts) < 4:
        return None
    try:
        x, y, width, height = (float(part) for part in parts[:4])
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(number) for number in (x, y, width, height)) or width <= 0 or height <= 0:
        return None
    return x, y, width, height


def _bbox_iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    left_x2, left_y2 = lx + lw, ly + lh
    right_x2, right_y2 = rx + rw, ry + rh
    inter_w = max(0.0, min(left_x2, right_x2) - max(lx, rx))
    inter_h = max(0.0, min(left_y2, right_y2) - max(ly, ry))
    intersection = inter_w * inter_h
    union = (lw * lh) + (rw * rh) - intersection
    return intersection / union if union > 0 else 0.0


def _map_face_exceptions_to_tracklets(
    face_comparison: pd.DataFrame,
    candidate_observations: pd.DataFrame,
) -> tuple[pd.DataFrame, set[str]]:
    exceptions = face_comparison[face_comparison["Frame_Requires_Review"].eq("Yes")].copy()
    if exceptions.empty:
        exceptions["Mapped_Tracklet_ID"] = pd.Series(dtype=str)
        exceptions["Mapped_Observation_ID"] = pd.Series(dtype=str)
        exceptions["Mapped_IoU"] = pd.Series(dtype=float)
        exceptions["Mapping_Status"] = pd.Series(dtype=str)
        return exceptions, set()

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in candidate_observations.to_dict("records"):
        key = (
            _text(row.get("Checkpoint_ID")),
            _text(row.get("Camera_ID")),
            _text(row.get("Video")).lower(),
            _text(row.get("Frame")),
        )
        grouped.setdefault(key, []).append(row)

    mapped_tracklets: list[str] = []
    mapped_observations: list[str] = []
    mapped_ious: list[float] = []
    mapping_statuses: list[str] = []
    tracklet_ids: set[str] = set()
    for _, row in exceptions.iterrows():
        key = (
            _text(row.get("Checkpoint_ID")),
            _text(row.get("Camera_ID")),
            _text(row.get("Source_File_Name")).lower(),
            _text(row.get("Frame")),
        )
        face_bbox = _parse_bbox(row.get("BBox_Original_Coordinates"))
        if face_bbox is None:
            mapped_tracklets.append("")
            mapped_observations.append("")
            mapped_ious.append(0.0)
            mapping_statuses.append("invalid_face_bbox")
            continue
        scored: list[tuple[float, str, str]] = []
        for observation in grouped.get(key, []):
            observation_bbox = _parse_bbox(observation.get("BBox_Original_Coordinates"))
            if observation_bbox is None:
                continue
            scored.append(
                (
                    _bbox_iou(face_bbox, observation_bbox),
                    _text(observation.get("Tracklet_ID")),
                    _text(observation.get("Observation_ID")),
                )
            )
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        if not scored or scored[0][0] < 0.50:
            mapped_tracklets.append("")
            mapped_observations.append("")
            mapped_ious.append(round(scored[0][0], 6) if scored else 0.0)
            mapping_statuses.append("unmapped_no_overlapping_tracklet")
            continue
        best_iou = scored[0][0]
        best = [item for item in scored if abs(item[0] - best_iou) <= 1e-9]
        best_tracklets = {item[1] for item in best}
        if len(best_tracklets) != 1:
            mapped_tracklets.append("")
            mapped_observations.append("")
            mapped_ious.append(round(best_iou, 6))
            mapping_statuses.append("ambiguous_multiple_tracklets")
            continue
        _, tracklet_id, observation_id = best[0]
        mapped_tracklets.append(tracklet_id)
        mapped_observations.append(observation_id)
        mapped_ious.append(round(best_iou, 6))
        mapping_statuses.append("mapped")
        tracklet_ids.add(tracklet_id)
    exceptions["Mapped_Tracklet_ID"] = mapped_tracklets
    exceptions["Mapped_Observation_ID"] = mapped_observations
    exceptions["Mapped_IoU"] = mapped_ious
    exceptions["Mapping_Status"] = mapping_statuses
    return exceptions, tracklet_ids


def _build_student_checkpoint_comparison(tracklets: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in ("Production", "Candidate"):
        accepted = tracklets[tracklets[f"{model}_Tracklet_Accepted"].map(_yes)].copy()
        accepted["Roll"] = accepted[f"{model}_Tracklet_Best_Roll"].map(_canonical)
        accepted = accepted[accepted["Roll"].ne("")]
        for (roll, checkpoint), group in accepted.groupby(["Roll", "Checkpoint_ID"], dropna=False):
            rows.append(
                {
                    "Model": model.lower(),
                    "Roll": roll,
                    "Checkpoint_ID": checkpoint,
                    "Accepted_Tracklets": int(len(group)),
                }
            )
    long = pd.DataFrame(rows)
    if long.empty:
        return pd.DataFrame(
            columns=[
                "Roll",
                "Checkpoint_ID",
                "Production_Accepted_Tracklets",
                "Candidate_Accepted_Tracklets",
                "Checkpoint_Delta",
            ]
        )
    pivot = long.pivot_table(
        index=["Roll", "Checkpoint_ID"],
        columns="Model",
        values="Accepted_Tracklets",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    if "production" not in pivot:
        pivot["production"] = 0
    if "candidate" not in pivot:
        pivot["candidate"] = 0
    pivot = pivot.rename(
        columns={
            "production": "Production_Accepted_Tracklets",
            "candidate": "Candidate_Accepted_Tracklets",
        }
    )
    pivot["Checkpoint_Delta"] = np.select(
        [
            pivot["Production_Accepted_Tracklets"].eq(0)
            & pivot["Candidate_Accepted_Tracklets"].gt(0),
            pivot["Production_Accepted_Tracklets"].gt(0)
            & pivot["Candidate_Accepted_Tracklets"].eq(0),
            pivot["Production_Accepted_Tracklets"].ne(
                pivot["Candidate_Accepted_Tracklets"]
            ),
        ],
        ["candidate_only", "production_only", "count_changed"],
        default="unchanged",
    )
    return pivot.sort_values(["Roll", "Checkpoint_ID"], kind="mergesort").reset_index(drop=True)


def compare_diagnostic_runs(
    production: DiagnosticArtifacts,
    candidate: DiagnosticArtifacts,
) -> ComparisonResult:
    tracklets = _build_tracklet_comparison(production.tracklets, candidate.tracklets)
    observations = _build_observation_comparison(production.observations, candidate.observations)
    faces = _build_face_comparison(production.faces, candidate.faces)
    face_exceptions, frame_tracklet_ids = _map_face_exceptions_to_tracklets(
        faces, candidate.observations
    )
    tracklet_exception_ids = set(
        tracklets.loc[
            tracklets["Tracklet_Requires_Review"].eq("Yes"), "Tracklet_ID"
        ].astype(str)
    )
    observation_exception_ids = set(
        observations.loc[
            observations["Frame_Requires_Review"].eq("Yes"), "Tracklet_ID"
        ].astype(str)
    )
    exception_ids = tracklet_exception_ids | observation_exception_ids | frame_tracklet_ids
    exception_tracks = tracklets[tracklets["Tracklet_ID"].astype(str).isin(exception_ids)].copy()
    exception_tracks["Has_Tracklet_Delta"] = exception_tracks["Tracklet_ID"].astype(str).isin(
        tracklet_exception_ids
    ).map(lambda value: "Yes" if value else "No")
    exception_tracks["Has_Observation_Frame_Delta"] = exception_tracks["Tracklet_ID"].astype(str).isin(
        observation_exception_ids
    ).map(lambda value: "Yes" if value else "No")
    exception_tracks["Has_Official_Frame_Delta"] = exception_tracks["Tracklet_ID"].astype(str).isin(
        frame_tracklet_ids
    ).map(lambda value: "Yes" if value else "No")
    student_checkpoint = _build_student_checkpoint_comparison(tracklets)

    tracklet_delta_counts = tracklets["Tracklet_Delta_Type"].value_counts().sort_index().to_dict()
    observation_delta_counts = observations["Frame_Delta_Type"].value_counts().sort_index().to_dict()
    face_delta_counts = faces["Frame_Delta_Type"].value_counts().sort_index().to_dict()
    checkpoint_count = int(tracklets["Checkpoint_ID"].astype(str).nunique()) if not tracklets.empty else 0
    camera_count = int(tracklets["Camera_ID"].astype(str).nunique()) if not tracklets.empty else 0
    unmapped_frame_deltas = int(
        (~face_exceptions.get("Mapping_Status", pd.Series(dtype=str)).eq("mapped")).sum()
    ) if not face_exceptions.empty else 0
    evidence_sufficient = bool(
        len(faces) > 0 and len(tracklets) > 0 and len(observations) > 0 and checkpoint_count >= 2
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "production_tracklets": int(len(tracklets)),
        "candidate_tracklets": int(len(tracklets)),
        "production_observations": int(len(observations)),
        "candidate_observations": int(len(observations)),
        "production_face_detections": int(len(faces)),
        "candidate_face_detections": int(len(faces)),
        "checkpoint_count_with_tracklets": checkpoint_count,
        "camera_count_with_tracklets": camera_count,
        "evidence_sufficient": evidence_sufficient,
        "tracklet_delta_counts": {str(key): int(value) for key, value in tracklet_delta_counts.items()},
        "observation_delta_counts": {
            str(key): int(value) for key, value in observation_delta_counts.items()
        },
        "official_frame_delta_counts": {
            str(key): int(value) for key, value in face_delta_counts.items()
        },
        "exception_tracklets": int(len(exception_tracks)),
        "frame_exception_detections": int(len(face_exceptions)),
        "unmapped_official_frame_deltas": unmapped_frame_deltas,
        "all_material_frame_deltas_mapped_to_review_tracklets": unmapped_frame_deltas == 0,
        "diagnostic_structure_identical": True,
    }
    return ComparisonResult(
        tracklets=tracklets,
        observations=observations,
        faces=faces,
        exception_tracks=exception_tracks,
        frame_exception_rows=face_exceptions,
        student_checkpoint=student_checkpoint,
        session_summary=summary,
    )


def _snapshot_tree(root: Path, *, exclude_top_level: set[str] | None = None) -> dict[str, str]:
    root = Path(root)
    if not root.exists():
        return {}
    exclude = {value.lower() for value in (exclude_top_level or set())}
    records: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0].lower() in exclude:
            continue
        records[relative.as_posix()] = _sha256_file(path)
    return records


def snapshot_protected_state(preflight: MonP3Preflight) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "production_embeddings_sha256": _sha256_file(preflight.production_embeddings),
        "production_summary_sha256": _sha256_file(preflight.production_summary),
        "student_map_sha256": _sha256_file(preflight.student_map),
        "timetable_sha256": _sha256_file(preflight.timetable),
        "candidate_embeddings_sha256": _sha256_file(preflight.family.candidate_embeddings),
        "family_output_manifest_sha256": _sha256_file(
            preflight.family.family_dir / "output_manifest.json"
        ),
        "camera_zones_sha256": _sha256_file(preflight.camera_zones),
        "source_videos_sha256": {
            row["relative_path"]: _sha256_file(Path(row["canonical_path"]))
            for row in preflight.freeze.rows
        },
        "dataset_files": _snapshot_tree(preflight.repo_root / "dataset"),
        "augmented_dataset_files": _snapshot_tree(preflight.repo_root / "augmented_dataset"),
        "official_attendance_files": _snapshot_tree(
            preflight.repo_root / "attendance_output",
            exclude_top_level={"diagnostics", "shadow_validation", "embedding_forensics"},
        ),
    }


def compare_protected_state(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    changed = [key for key in sorted(before) if before.get(key) != after.get(key)]
    extra = [key for key in sorted(after) if key not in before]
    missing = [key for key in sorted(before) if key not in after]
    return {
        "schema_version": SCHEMA_VERSION,
        "unchanged": not changed and not extra and not missing,
        "changed_keys": changed,
        "extra_keys": extra,
        "missing_keys": missing,
    }


def _write_preflight_json(preflight: MonP3Preflight, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "status": "pass",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "session_id": preflight.session_id,
        "slot_id": preflight.slot_id,
        "subject_abbr": preflight.subject_abbr,
        "family_id": preflight.family.family_id,
        "candidate_variant_id": preflight.family.candidate_variant_id,
        "candidate_embeddings_sha256": preflight.family.candidate_embeddings_sha256,
        "production_embeddings_sha256": preflight.production_embeddings_sha256,
        "production_summary_sha256": preflight.production_summary_sha256,
        "source_freeze_json_sha256": preflight.freeze.json_sha256,
        "source_freeze_csv_sha256": preflight.freeze.csv_sha256,
        "canonical_video_root": str(preflight.video_root),
        "mirror_video_root": str(preflight.mirror_video_root),
        "video_count": len(preflight.videos),
        "mirror_byte_identical": True,
        "roster_count": len(preflight.subject_rolls),
        "fingerprint_sha256": preflight.fingerprint_sha256,
        "run_id": preflight.run_id,
        "planned_output": str(preflight.output_dir),
        "planned_production_diagnostic_run": str(preflight.production_diagnostic_run),
        "planned_candidate_diagnostic_run": str(preflight.candidate_diagnostic_run),
        "recognition_executed": False,
        "attendance_written": False,
        "manual_review_started": False,
        "candidate_promoted": False,
    }
    output_path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
    return output_path


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    status_callback: Callable[[str], None] | None = None,
) -> None:
    if status_callback:
        status_callback("Command: " + subprocess.list2cmdline(command))
    try:
        subprocess.run(command, cwd=cwd, check=True)
    except subprocess.CalledProcessError as exc:
        raise MonP3ShadowError(
            f"Diagnostic-only recognition failed with exit code {exc.returncode}"
        ) from exc


def _copy_input(path: Path, destination_dir: Path) -> Path:
    destination = destination_dir / Path(path).name
    shutil.copy2(path, destination)
    if _sha256_file(destination) != _sha256_file(path):
        raise MonP3ShadowError(f"Copied input hash mismatch: {path}")
    return destination


def _prepare_review_rows(
    comparison: ComparisonResult,
    candidate_artifacts: DiagnosticArtifacts,
) -> tuple[pd.DataFrame, dict[str, str]]:
    selected_ids = set(comparison.exception_tracks["Tracklet_ID"].astype(str))
    if not selected_ids:
        return candidate_artifacts.tracklets.copy(), {}
    review_rows = candidate_artifacts.tracklets.copy()
    details = comparison.exception_tracks.set_index("Tracklet_ID")
    extra_mapping: dict[str, str] = {}
    fields = {
        "Phase_1_2J_Delta_Type": "Tracklet_Delta_Type",
        "Phase_1_2J_Has_Tracklet_Delta": "Has_Tracklet_Delta",
        "Phase_1_2J_Has_Observation_Frame_Delta": "Has_Observation_Frame_Delta",
        "Phase_1_2J_Has_Official_Frame_Delta": "Has_Official_Frame_Delta",
        "Production_Tracklet_Accepted": "Production_Tracklet_Accepted",
        "Production_Tracklet_Best_Roll": "Production_Tracklet_Best_Roll",
        "Production_Tracklet_Best_Score": "Production_Tracklet_Best_Score",
        "Production_Tracklet_Margin": "Production_Tracklet_Margin",
        "Candidate_Tracklet_Accepted": "Candidate_Tracklet_Accepted",
        "Candidate_Tracklet_Best_Roll": "Candidate_Tracklet_Best_Roll",
        "Candidate_Tracklet_Best_Score": "Candidate_Tracklet_Best_Score",
        "Candidate_Tracklet_Margin": "Candidate_Tracklet_Margin",
    }
    for output_column, source_column in fields.items():
        values = review_rows["Tracklet_ID"].map(
            lambda value: details.at[str(value), source_column]
            if str(value) in details.index
            else ""
        )
        review_rows[output_column] = values
        extra_mapping[output_column] = output_column
    return review_rows, extra_mapping


def _verify_completed_output(preflight: MonP3Preflight) -> MonP3RunResult | None:
    if not preflight.output_dir.exists():
        return None
    manifest = preflight.output_dir / "output_manifest.json"
    summary_path = preflight.output_dir / "shadow_run_summary.json"
    if not manifest.is_file() or not summary_path.is_file():
        raise MonP3ShadowError(
            f"Existing Phase 1.2J output is incomplete; do not overwrite it: {preflight.output_dir}"
        )
    try:
        verify_output_manifest(preflight.output_dir, manifest)
    except ShadowValidationError as exc:
        raise MonP3ShadowError(str(exc)) from exc
    summary = _load_json(summary_path, "Phase 1.2J summary")
    if _text(summary.get("fingerprint_sha256")) != preflight.fingerprint_sha256:
        raise MonP3ShadowError("Existing Phase 1.2J output fingerprint does not match the current inputs")
    review_package = Path(_text(summary.get("review_package")))
    return MonP3RunResult(
        output_dir=preflight.output_dir,
        production_diagnostic_run=Path(_text(summary.get("production_diagnostic_run"))),
        candidate_diagnostic_run=Path(_text(summary.get("candidate_diagnostic_run"))),
        review_package=review_package,
        summary=summary,
        idempotent_reuse=True,
    )


def run_mon_p3_shadow(
    preflight: MonP3Preflight,
    *,
    python_executable: Path | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> MonP3RunResult:
    existing = _verify_completed_output(preflight)
    if existing is not None:
        return existing
    for path in (preflight.production_diagnostic_run, preflight.candidate_diagnostic_run):
        if path.exists():
            raise MonP3ShadowError(
                f"A Phase 1.2J diagnostic directory already exists without a complete final output: {path}"
            )
    preflight.output_root.mkdir(parents=True, exist_ok=True)
    temporary_output = preflight.output_root / f".{preflight.run_id}.tmp-{os.getpid()}"
    if temporary_output.exists():
        raise MonP3ShadowError(f"Temporary Phase 1.2J output already exists: {temporary_output}")
    started = time.perf_counter()
    protected_before = snapshot_protected_state(preflight)
    production_command = build_diagnostic_command(
        preflight,
        embeddings_path=preflight.production_embeddings,
        diagnostic_run_id=preflight.production_diagnostic_run.name,
        python_executable=python_executable,
    )
    candidate_command = build_diagnostic_command(
        preflight,
        embeddings_path=preflight.family.candidate_embeddings,
        diagnostic_run_id=preflight.candidate_diagnostic_run.name,
        python_executable=python_executable,
    )
    if status_callback:
        status_callback("Running production diagnostic-only MON_P3 pass. Official attendance remains disabled.")
    _run_command(production_command, cwd=preflight.repo_root, status_callback=status_callback)
    if status_callback:
        status_callback("Running candidate diagnostic-only MON_P3 pass. Candidate remains unpromoted.")
    _run_command(candidate_command, cwd=preflight.repo_root, status_callback=status_callback)

    production_artifacts = _read_diagnostic_artifacts(preflight.production_diagnostic_run, preflight)
    candidate_artifacts = _read_diagnostic_artifacts(preflight.candidate_diagnostic_run, preflight)
    comparison = compare_diagnostic_runs(production_artifacts, candidate_artifacts)
    protected_after = snapshot_protected_state(preflight)
    protected_comparison = compare_protected_state(protected_before, protected_after)
    if not protected_comparison["unchanged"]:
        raise MonP3ShadowError(
            "Protected production/dataset/attendance state changed during Phase 1.2J: "
            + ", ".join(protected_comparison["changed_keys"][:10])
        )

    temporary_output.mkdir(parents=True, exist_ok=False)
    input_dir = temporary_output / "frozen_inputs"
    input_dir.mkdir()
    copied_freeze_json = _copy_input(preflight.freeze.json_path, input_dir)
    copied_freeze_csv = _copy_input(preflight.freeze.csv_path, input_dir)
    _copy_input(preflight.family.family_dir / "family_manifest.json", input_dir)
    _copy_input(
        preflight.family.family_dir / "evaluation" / "evaluation_decision.json", input_dir
    )
    _copy_input(
        preflight.family.family_dir / "variants" / "full_candidate" / "version_manifest.json",
        input_dir,
    )

    comparison.tracklets.to_csv(temporary_output / "tracklet_comparison.csv", index=False)
    comparison.observations.to_csv(temporary_output / "observation_comparison.csv", index=False)
    comparison.faces.to_csv(temporary_output / "official_frame_comparison.csv", index=False)
    comparison.exception_tracks.to_csv(temporary_output / "exception_tracks.csv", index=False)
    comparison.frame_exception_rows.to_csv(
        temporary_output / "official_frame_exception_detections.csv", index=False
    )
    comparison.student_checkpoint.to_csv(
        temporary_output / "student_checkpoint_comparison.csv", index=False
    )
    (temporary_output / "protected_state_before.json").write_text(
        json.dumps(json_safe(protected_before), indent=2), encoding="utf-8"
    )
    (temporary_output / "protected_state_after.json").write_text(
        json.dumps(json_safe(protected_after), indent=2), encoding="utf-8"
    )
    (temporary_output / "protected_state_comparison.json").write_text(
        json.dumps(json_safe(protected_comparison), indent=2), encoding="utf-8"
    )

    input_manifest = {
        "schema_version": SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "fingerprint_sha256": preflight.fingerprint_sha256,
        "run_id": preflight.run_id,
        "session_id": preflight.session_id,
        "slot_id": preflight.slot_id,
        "subject_abbr": preflight.subject_abbr,
        "family": {
            "family_id": preflight.family.family_id,
            "family_content_fingerprint": preflight.family.family_content_fingerprint,
            "candidate_variant_id": preflight.family.candidate_variant_id,
            "candidate_embeddings_path": str(preflight.family.candidate_embeddings),
            "candidate_embeddings_sha256": preflight.family.candidate_embeddings_sha256,
            "candidate_embedding_records": preflight.family.candidate_embedding_records,
            "production_approved": False,
            "candidate_promoted": False,
        },
        "production": {
            "embeddings_path": str(preflight.production_embeddings),
            "embeddings_sha256": preflight.production_embeddings_sha256,
            "summary_path": str(preflight.production_summary),
            "summary_sha256": preflight.production_summary_sha256,
        },
        "source_freeze": {
            "json_path": str(preflight.freeze.json_path),
            "json_sha256": preflight.freeze.json_sha256,
            "csv_path": str(preflight.freeze.csv_path),
            "csv_sha256": preflight.freeze.csv_sha256,
            "canonical_video_root": str(preflight.video_root),
            "mirror_video_root": str(preflight.mirror_video_root),
            "canonical_source_only": True,
            "mirror_byte_identical": True,
            "videos": preflight.freeze.rows,
        },
        "authoritative_roster": {
            "student_map": str(preflight.student_map),
            "student_map_sha256": preflight.roster_sha256,
            "subject_student_count": len(preflight.subject_rolls),
            "subject_rolls": sorted(preflight.subject_rolls),
        },
        "reference_configuration": preflight.reference_configuration,
        "camera_zones": str(preflight.camera_zones),
        "camera_zones_sha256": preflight.camera_zones_sha256,
        "production_diagnostic_run": str(preflight.production_diagnostic_run),
        "candidate_diagnostic_run": str(preflight.candidate_diagnostic_run),
        "production_command": production_command,
        "candidate_command": candidate_command,
        "official_match_threshold": MATCH_THRESHOLD,
        "official_margin_threshold": MARGIN_THRESHOLD,
        "attendance_checkpoint_rule": PRESENT_CHECKPOINTS,
        "diagnostic_only": True,
        "official_attendance_written": False,
        "manual_review_scope": "material production-vs-candidate identity/acceptance deltas only",
        "production_promotion_allowed_in_phase": False,
    }
    input_manifest_path = temporary_output / "session_input_manifest.json"
    input_manifest_path.write_text(json.dumps(json_safe(input_manifest), indent=2), encoding="utf-8")

    review_rows, hidden_extra_columns = _prepare_review_rows(comparison, candidate_artifacts)
    selected_ids = set(comparison.exception_tracks["Tracklet_ID"].astype(str))
    try:
        review_package = export_review_package(
            tracklet_df=review_rows,
            observation_df=candidate_artifacts.observations,
            video_root=preflight.video_root,
            output_root=temporary_output,
            student_map_path=preflight.student_map,
            subject_abbr=preflight.subject_abbr,
            diagnostic_run_id=preflight.run_id,
            evidence_count=5,
            crop_padding=0.45,
            source_files={
                "production_tracklets": production_artifacts.tracklet_path,
                "candidate_tracklets": candidate_artifacts.tracklet_path,
                "production_observations": production_artifacts.observations_path,
                "candidate_observations": candidate_artifacts.observations_path,
                "tracklet_comparison": temporary_output / "tracklet_comparison.csv",
                "exception_tracks": temporary_output / "exception_tracks.csv",
                "session_input_manifest": input_manifest_path,
                "source_freeze_json": copied_freeze_json,
                "source_freeze_csv": copied_freeze_csv,
            },
            selected_tracklet_ids=selected_ids,
            review_kind="shadow_recovery",
            hidden_extra_columns=hidden_extra_columns,
            include_context=True,
            context_padding=2.0,
            package_prefix="mon_p3_candidate_delta_review",
            subject_roster_only=True,
            allow_empty=True,
            session_id_override=preflight.session_id,
            private_metadata={
                "phase": "1.2J",
                "policy_version": POLICY_VERSION,
                "fingerprint_sha256": preflight.fingerprint_sha256,
                "family_id": preflight.family.family_id,
                "candidate_variant_id": preflight.family.candidate_variant_id,
                "candidate_embeddings_sha256": preflight.family.candidate_embeddings_sha256,
                "production_embeddings_sha256": preflight.production_embeddings_sha256,
                "review_scope": "material_delta_tracklets_only",
                "candidate_promoted": False,
                "official_attendance_written": False,
            },
        )
    except ReviewExportError as exc:
        raise MonP3ShadowError(str(exc)) from exc

    exception_count = int(len(comparison.exception_tracks))
    unmapped_frame_deltas = int(comparison.session_summary["unmapped_official_frame_deltas"])
    evidence_sufficient = bool(comparison.session_summary["evidence_sufficient"])
    if unmapped_frame_deltas:
        decision = "hold_unmapped_official_frame_deltas"
    elif not evidence_sufficient:
        decision = "hold_insufficient_mon_p3_evidence"
    elif exception_count:
        decision = "pending_blind_exception_review"
    else:
        decision = "mon_p3_shadow_complete_no_material_deltas_pending_explicit_promotion"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "fingerprint_sha256": preflight.fingerprint_sha256,
        "run_id": preflight.run_id,
        "session_id": preflight.session_id,
        "family_id": preflight.family.family_id,
        "candidate_variant_id": preflight.family.candidate_variant_id,
        "decision": decision,
        "comparison": comparison.session_summary,
        "production_diagnostic_run": str(preflight.production_diagnostic_run),
        "candidate_diagnostic_run": str(preflight.candidate_diagnostic_run),
        "review_package": str(review_package.root.resolve()),
        "review_status": (
            "pending"
            if decision == "pending_blind_exception_review"
            else "blocked_unmapped_frame_deltas"
            if decision == "hold_unmapped_official_frame_deltas"
            else "blocked_insufficient_evidence"
            if decision == "hold_insufficient_mon_p3_evidence"
            else "not_required_zero_material_deltas"
        ),
        "review_tracklets": exception_count,
        "production_embeddings_changed": False,
        "production_summary_changed": False,
        "dataset_files_changed": False,
        "official_attendance_changed": False,
        "candidate_promoted": False,
        "production_approved": False,
        "mon_p3_processed": True,
        "canonical_source_only": True,
        "manual_review_scope": "exceptions_only",
        "broad_manual_review_required": False,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "exact_next_step": (
            "Complete the blind exception review and run Phase 1.2J evaluation."
            if decision == "pending_blind_exception_review"
            else "Inspect unmapped official frame deltas before any promotion workflow."
            if decision == "hold_unmapped_official_frame_deltas"
            else "Collect stronger untouched evidence before any promotion workflow."
            if decision == "hold_insufficient_mon_p3_evidence"
            else "Inspect the immutable Phase 1.2J report before any explicit promotion workflow."
        ),
    }
    (temporary_output / "shadow_run_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2), encoding="utf-8"
    )
    write_output_manifest(
        temporary_output,
        {
            "schema_version": SCHEMA_VERSION,
            "policy_version": POLICY_VERSION,
            "fingerprint_sha256": preflight.fingerprint_sha256,
            "run_id": preflight.run_id,
            "session_id": preflight.session_id,
            "family_id": preflight.family.family_id,
            "candidate_variant_id": preflight.family.candidate_variant_id,
            "decision": decision,
            "review_package_id": review_package.package_id,
            "production_changes": False,
            "official_attendance_modified": False,
            "candidate_promoted": False,
        },
    )
    try:
        verify_output_manifest(temporary_output, temporary_output / "output_manifest.json")
    except ShadowValidationError as exc:
        raise MonP3ShadowError(str(exc)) from exc
    temporary_output.rename(preflight.output_dir)
    final_review = preflight.output_dir / review_package.root.relative_to(temporary_output)
    summary["review_package"] = str(final_review.resolve())
    review_readme = (
        "PHASE 1.2J BLIND EXCEPTION REVIEW\n\n"
        "Give the reviewer only the reviewer directory. Keep private withheld until the labels CSV is final.\n"
        "This package contains only material production-vs-candidate delta tracklets; unchanged tracks are not reviewed again.\n"
        "After exporting the completed labels CSV, run scripts/run_phase_1_2j_mon_p3_shadow.py evaluate "
        "--output-dir <PHASE_1_2J_OUTPUT> --labels <COMPLETED_LABELS.csv>.\n"
        "Production embeddings, official attendance, and promotion remain unchanged.\n"
    )
    (final_review / "README.txt").write_text(review_readme, encoding="utf-8")
    launcher_path = preflight.output_dir / "open_mon_p3_exception_reviewer.ps1"
    launcher_command = (
        f"& '{str(Path(sys.executable)).replace(chr(39), chr(39) * 2)}' "
        f"'{str((preflight.repo_root / 'scripts' / 'validate_tracklet_ground_truth.py').resolve()).replace(chr(39), chr(39) * 2)}' "
        f"open-shadow-review --review-package '{str(final_review.resolve()).replace(chr(39), chr(39) * 2)}'\n"
    )
    launcher_path.write_text(launcher_command, encoding="utf-8")
    summary["reviewer_launcher"] = str(launcher_path.resolve())
    (preflight.output_dir / "shadow_run_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2), encoding="utf-8"
    )
    write_output_manifest(
        preflight.output_dir,
        {
            "schema_version": SCHEMA_VERSION,
            "policy_version": POLICY_VERSION,
            "fingerprint_sha256": preflight.fingerprint_sha256,
            "run_id": preflight.run_id,
            "session_id": preflight.session_id,
            "family_id": preflight.family.family_id,
            "candidate_variant_id": preflight.family.candidate_variant_id,
            "decision": decision,
            "review_package_id": review_package.package_id,
            "production_changes": False,
            "official_attendance_modified": False,
            "candidate_promoted": False,
        },
    )
    try:
        verify_output_manifest(preflight.output_dir, preflight.output_dir / "output_manifest.json")
    except ShadowValidationError as exc:
        raise MonP3ShadowError(str(exc)) from exc
    return MonP3RunResult(
        output_dir=preflight.output_dir,
        production_diagnostic_run=preflight.production_diagnostic_run,
        candidate_diagnostic_run=preflight.candidate_diagnostic_run,
        review_package=final_review,
        summary=summary,
        idempotent_reuse=False,
    )


def verify_mon_p3_output(
    *,
    output_dir: Path,
    production_embeddings: Path,
    production_summary: Path,
    family_dir: Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    if not output_dir.is_dir():
        raise MonP3ShadowError(f"Phase 1.2J output not found: {output_dir}")
    try:
        verified = verify_output_manifest(output_dir, output_dir / "output_manifest.json")
    except ShadowValidationError as exc:
        raise MonP3ShadowError(str(exc)) from exc
    summary = _load_json(output_dir / "shadow_run_summary.json", "Phase 1.2J summary")
    protected = _load_json(
        output_dir / "protected_state_comparison.json", "protected-state comparison"
    )
    if protected.get("unchanged") is not True:
        raise MonP3ShadowError("Phase 1.2J output records a protected-state change")
    if _sha256_file(Path(production_embeddings)) != EXPECTED_PRODUCTION_EMBEDDINGS_SHA256:
        raise MonP3ShadowError("Current production embeddings differ from Phase 1.2J baseline")
    if _sha256_file(Path(production_summary)) != EXPECTED_PRODUCTION_SUMMARY_SHA256:
        raise MonP3ShadowError("Current production summary differs from Phase 1.2J baseline")
    family = verify_candidate_family(
        family_dir=family_dir,
        production_embeddings=production_embeddings,
        production_summary=production_summary,
        expected_family_id=_text(summary.get("family_id")),
    )
    if family.candidate_variant_id != _text(summary.get("candidate_variant_id")):
        raise MonP3ShadowError("Phase 1.2J candidate variant changed")
    if summary.get("production_embeddings_changed") is not False:
        raise MonP3ShadowError("Phase 1.2J summary claims production embeddings changed")
    if summary.get("official_attendance_changed") is not False:
        raise MonP3ShadowError("Phase 1.2J summary claims official attendance changed")
    if summary.get("candidate_promoted") is not False:
        raise MonP3ShadowError("Phase 1.2J summary claims the candidate was promoted")
    return {
        "verified_files": verified["verified_files"],
        "run_id": summary["run_id"],
        "decision": summary["decision"],
        "review_tracklets": summary["review_tracklets"],
        "family_id": summary["family_id"],
        "candidate_variant_id": summary["candidate_variant_id"],
        "production_embeddings_changed": False,
        "official_attendance_changed": False,
        "candidate_promoted": False,
    }


def evaluate_mon_p3_review(
    *,
    output_dir: Path,
    labels_path: Path,
    student_map: Path,
    subject_abbr: str = EXPECTED_SUBJECT,
) -> tuple[dict[str, Any], Path]:
    output_dir = Path(output_dir).resolve()
    summary = _load_json(output_dir / "shadow_run_summary.json", "Phase 1.2J summary")
    try:
        verify_output_manifest(output_dir, output_dir / "output_manifest.json")
    except ShadowValidationError as exc:
        raise MonP3ShadowError(str(exc)) from exc
    review_package = Path(_text(summary.get("review_package")))
    try:
        hidden, metadata, manifest = load_review_package_data(review_package)
        _, known_rolls, _ = load_student_mapping(Path(student_map), subject_abbr)
    except (LabelValidationError, ReviewExportError) as exc:
        raise MonP3ShadowError(str(exc)) from exc
    if _text(metadata.get("phase")) != "1.2J" or _text(metadata.get("policy_version")) != POLICY_VERSION:
        raise MonP3ShadowError("Review package is not a Phase 1.2J exception package")
    if _text(metadata.get("fingerprint_sha256")) != _text(summary.get("fingerprint_sha256")):
        raise MonP3ShadowError("Review package fingerprint does not match the Phase 1.2J run")
    if not Path(labels_path).is_file():
        raise MonP3ShadowError(f"Completed Phase 1.2J labels not found: {labels_path}")
    labels = pd.read_csv(labels_path, dtype=str, keep_default_na=False)
    try:
        validated = validate_label_dataframe(
            labels,
            expected_review_ids=set(hidden["Review_ID"].astype(str)),
            package_id=_text(metadata.get("package_id")),
            known_rolls=set(known_rolls),
        )
    except LabelValidationError as exc:
        raise MonP3ShadowError(str(exc)) from exc
    if len(validated) != len(hidden):
        raise MonP3ShadowError("Phase 1.2J labels must contain every review item exactly once")
    if validated["Review_Status"].eq("").any():
        pending = validated.loc[validated["Review_Status"].eq(""), "Review_ID"].tolist()
        raise MonP3ShadowError(
            "Every Phase 1.2J exception must be reviewed before evaluation; pending: "
            + ", ".join(pending[:10])
        )
    joined = hidden.merge(validated, on=["Package_ID", "Review_ID"], how="left", validate="one_to_one")
    joined["Production_Accepted_Bool"] = joined["Production_Tracklet_Accepted"].map(_yes)
    joined["Candidate_Accepted_Bool"] = joined["Candidate_Tracklet_Accepted"].map(_yes)
    joined["Production_Roll"] = joined["Production_Tracklet_Best_Roll"].map(_canonical)
    joined["Candidate_Roll"] = joined["Candidate_Tracklet_Best_Roll"].map(_canonical)
    joined["Actual_Roll_Normalized"] = joined["Actual_Roll"].map(_canonical)

    result_codes: list[str] = []
    for row in joined.to_dict("records"):
        status = _text(row.get("Review_Status"))
        actual = _canonical(row.get("Actual_Roll_Normalized"))
        prod_accept = bool(row.get("Production_Accepted_Bool"))
        cand_accept = bool(row.get("Candidate_Accepted_Bool"))
        prod_roll = _canonical(row.get("Production_Roll"))
        cand_roll = _canonical(row.get("Candidate_Roll"))
        if status in {"unidentifiable", "uncertain", "duplicate"}:
            result_codes.append("unverifiable")
            continue
        if status in {"mixed_track", "not_in_mapping"}:
            result_codes.append("candidate_unsafe_nonstudent_or_mixed" if cand_accept else "production_unsafe_nonstudent_or_mixed")
            continue
        if status != "identified":
            result_codes.append("unverifiable")
            continue
        if cand_accept and actual != cand_roll:
            result_codes.append("candidate_false_identity")
        elif prod_accept and not cand_accept and actual == prod_roll:
            result_codes.append("candidate_lost_correct_production_accept")
        elif prod_accept and cand_accept and actual == cand_roll and actual != prod_roll:
            result_codes.append("candidate_corrected_production_identity")
        elif not prod_accept and cand_accept and actual == cand_roll:
            result_codes.append("candidate_correct_recovery")
        elif prod_accept and not cand_accept and actual != prod_roll:
            result_codes.append("candidate_removed_wrong_production_accept")
        elif prod_accept and cand_accept and actual == prod_roll == cand_roll:
            result_codes.append("both_correct_same_identity")
        else:
            result_codes.append("reviewed_other")
    joined["Phase_1_2J_Result"] = result_codes
    counts = joined["Phase_1_2J_Result"].value_counts().sort_index().to_dict()
    candidate_false = int(
        joined["Phase_1_2J_Result"].isin(
            {"candidate_false_identity", "candidate_unsafe_nonstudent_or_mixed"}
        ).sum()
    )
    retention_losses = int(
        joined["Phase_1_2J_Result"].eq("candidate_lost_correct_production_accept").sum()
    )
    unverifiable = int(joined["Phase_1_2J_Result"].eq("unverifiable").sum())
    correct_recoveries = int(
        joined["Phase_1_2J_Result"].eq("candidate_correct_recovery").sum()
    )
    corrected_identities = int(
        joined["Phase_1_2J_Result"].eq("candidate_corrected_production_identity").sum()
    )
    if candidate_false:
        decision = "mon_p3_shadow_failed_false_identity"
    elif retention_losses:
        decision = "mon_p3_shadow_failed_retention"
    elif unverifiable:
        decision = "mon_p3_shadow_hold_unverifiable_exceptions"
    else:
        decision = "mon_p3_shadow_passed_pending_explicit_promotion"
    evaluation_id = "mon-p3-evaluation-" + _stable_digest(
        {
            "run_fingerprint": summary["fingerprint_sha256"],
            "labels_sha256": _sha256_file(Path(labels_path)),
            "policy_version": POLICY_VERSION,
        }
    )[:20]
    evaluation_dir = output_dir / "evaluation" / evaluation_id
    if evaluation_dir.exists():
        try:
            verify_output_manifest(evaluation_dir, evaluation_dir / "output_manifest.json")
        except ShadowValidationError as exc:
            raise MonP3ShadowError(str(exc)) from exc
        existing = _load_json(
            evaluation_dir / "mon_p3_evaluation_summary.json", "existing Phase 1.2J evaluation"
        )
        return existing, evaluation_dir
    evaluation_dir.mkdir(parents=True, exist_ok=False)
    joined.to_csv(evaluation_dir / "reviewed_exception_tracks.csv", index=False)
    evaluation_summary = {
        "schema_version": SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "evaluation_id": evaluation_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": summary["run_id"],
        "fingerprint_sha256": summary["fingerprint_sha256"],
        "family_id": summary["family_id"],
        "candidate_variant_id": summary["candidate_variant_id"],
        "reviewed_exception_tracks": int(len(joined)),
        "result_counts": {str(key): int(value) for key, value in counts.items()},
        "candidate_false_identities_or_unsafe_accepts": candidate_false,
        "candidate_lost_correct_production_accepts": retention_losses,
        "unverifiable_exceptions": unverifiable,
        "candidate_correct_recoveries": correct_recoveries,
        "candidate_corrected_production_identities": corrected_identities,
        "decision": decision,
        "zero_false_identity_gate_passed": candidate_false == 0,
        "retention_gate_passed": retention_losses == 0,
        "evidence_complete_gate_passed": unverifiable == 0,
        "production_embeddings_changed": False,
        "official_attendance_changed": False,
        "candidate_promoted": False,
        "production_approved": False,
        "exact_next_step": (
            "Prepare a separate explicit promotion/rollback package only after final human approval."
            if decision == "mon_p3_shadow_passed_pending_explicit_promotion"
            else "Keep the candidate unpromoted and inspect the failed or held exception evidence."
        ),
    }
    (evaluation_dir / "mon_p3_evaluation_summary.json").write_text(
        json.dumps(json_safe(evaluation_summary), indent=2), encoding="utf-8"
    )
    report = (
        "# Phase 1.2J MON_P3 Shadow Evaluation\n\n"
        f"- Decision: {decision}\n"
        f"- Reviewed exception tracklets: {len(joined)}\n"
        f"- Candidate false identities / unsafe accepts: {candidate_false}\n"
        f"- Candidate lost correct production accepts: {retention_losses}\n"
        f"- Unverifiable exceptions: {unverifiable}\n"
        f"- Correct candidate recoveries: {correct_recoveries}\n"
        f"- Candidate-corrected production identities: {corrected_identities}\n\n"
        "Production embeddings, official attendance, and candidate promotion were not changed.\n"
    )
    (evaluation_dir / "mon_p3_evaluation_report.md").write_text(report, encoding="utf-8")
    write_output_manifest(
        evaluation_dir,
        {
            "schema_version": SCHEMA_VERSION,
            "policy_version": POLICY_VERSION,
            "evaluation_id": evaluation_id,
            "run_id": summary["run_id"],
            "family_id": summary["family_id"],
            "decision": decision,
            "candidate_promoted": False,
            "production_changes": False,
            "official_attendance_modified": False,
        },
    )
    try:
        verify_output_manifest(evaluation_dir, evaluation_dir / "output_manifest.json")
    except ShadowValidationError as exc:
        raise MonP3ShadowError(str(exc)) from exc
    return evaluation_summary, evaluation_dir


def write_preflight(preflight: MonP3Preflight, output_path: Path) -> Path:
    return _write_preflight_json(preflight, output_path)
