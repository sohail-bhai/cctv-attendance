from __future__ import annotations

import csv
import hashlib
import json
import math
import pickle
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import cv2
import numpy as np
import pandas as pd

from .config import (
    AUGMENTED_DATASET_DIR,
    DATASET_DIR,
    IMAGE_EXTENSIONS,
    SFACE_MODEL,
    YUNET_MODEL,
)
from .diagnostics import finite_float, json_safe, make_diagnostic_run_id
from .face_engine import FaceEngine
from .forensic_review import (
    ForensicReviewError,
    ForensicReviewItem,
    ForensicReviewPackage,
    export_forensic_review_package,
    write_reviewer_launcher,
)
from .recall_analysis import canonical_roll
from .shadow_validation import (
    EXPECTED_CANDIDATE_CONFIG_SHA256,
    EXPECTED_CANDIDATE_ID,
    ShadowValidationError,
    load_frozen_candidate,
    verify_output_manifest,
    write_output_manifest,
)
from .tracklet_review import (
    LabelValidationError,
    ReviewExportError,
    _face_crop,
    _parse_bbox,
    _resolve_video,
    _sha256_file,
    _video_index,
    _OpenCVFrameLoader,
    load_review_package_data,
    load_student_mapping,
    validate_label_dataframe,
)


FORENSIC_SCHEMA_VERSION = 1
BENCHMARK_SCHEMA_VERSION = 1
OFFICIAL_MATCH_THRESHOLD = 0.48
OFFICIAL_MARGIN_THRESHOLD = 0.08
OFFICIAL_ATTENDANCE_CHECKPOINTS = 3
EXPECTED_REVIEWED_TRACKS = 120
EXPECTED_SUBJECT = "CVO"
EXPECTED_ROSTER_COUNT = 27
MISSING_EMBEDDING_ROLLS = ("2401100CSE0268", "24011CSEAI0110")
REJECTED_CANDIDATE_STATUS = "rejected_multisession_false_accepts"
ENROLLMENT_ACTIONS = {
    "keep": "Keep for the next embedding candidate",
    "exclude_from_next_embedding_version": "Exclude from the next embedding candidate",
    "wrong_person_or_mislabeled": "Wrong person or mislabeled",
    "multiple_people": "Multiple people",
    "unusable_quality": "Unusable quality",
    "uncertain": "Uncertain",
    "duplicate_of_another_image": "Duplicate of another image",
}
CCTV_ACTIONS = {
    "approve_for_candidate_embedding_version": "Approve for a future candidate embedding version",
    "reject_wrong_or_unclear_identity": "Reject: wrong or unclear identity",
    "reject_quality": "Reject: insufficient quality",
    "reject_duplicate": "Reject: duplicate",
    "reject_mixed_person": "Reject: mixed person evidence",
    "uncertain": "Uncertain",
}
_ROLL_PATTERN = re.compile(r"^[A-Z0-9]+$")


class EmbeddingForensicsError(ValueError):
    pass


@dataclass(frozen=True)
class BenchmarkSource:
    source_kind: str
    package_dir: Path
    labels_path: Path
    diagnostic_run: Path
    expected_rows: int
    baseline_status: str
    expected_labels_sha256: str = ""
    trusted_manifest_path: Path | None = None


@dataclass(frozen=True)
class MultisessionBenchmark:
    frame: pd.DataFrame
    manifest: dict[str, Any]


@dataclass(frozen=True)
class ConfusionAnalysis:
    matrix: pd.DataFrame
    attractors: pd.DataFrame
    miss_patterns: pd.DataFrame
    pairs: pd.DataFrame
    summary: dict[str, Any]


@dataclass(frozen=True)
class EmbeddingGeometry:
    student_health: pd.DataFrame
    outliers: pd.DataFrame
    nearest_neighbors: pd.DataFrame
    duplicate_candidates: pd.DataFrame
    graph: dict[str, Any]
    summary: dict[str, Any]
    normalized_by_roll: dict[str, list[np.ndarray]]
    centroids: dict[str, np.ndarray]
    medoids: dict[str, np.ndarray]
    outlier_thresholds: dict[str, float]


@dataclass(frozen=True)
class DatasetAudit:
    inventory: pd.DataFrame
    quality: pd.DataFrame
    suspects: pd.DataFrame
    duplicates: pd.DataFrame
    summary: dict[str, Any]


@dataclass(frozen=True)
class CCTVSelection:
    inventory: pd.DataFrame
    selected: pd.DataFrame
    summary: pd.DataFrame


@dataclass(frozen=True)
class EmbeddingForensicsRun:
    output_dir: Path
    summary: dict[str, Any]
    enrollment_review: ForensicReviewPackage
    cctv_review: ForensicReviewPackage
    output_manifest: Path


def _text(value: Any) -> str:
    return str(value if value is not None else "").strip()


def _number(value: Any, default: float = 0.0) -> float:
    return finite_float(value, default)


def _canonical(value: Any) -> str:
    return canonical_roll(value).strip().upper()


def _yes(value: Any) -> bool:
    return _text(value).lower() in {"yes", "true", "1", "pass", "accepted"}


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _single_file(directory: Path, pattern: str, label: str) -> Path:
    matches = sorted(Path(directory).glob(pattern))
    if len(matches) != 1:
        raise EmbeddingForensicsError(
            f"Expected exactly one {label} in {directory}; found {len(matches)}"
        )
    return matches[0]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(json_safe(payload), indent=2, ensure_ascii=True), encoding="utf-8"
    )


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _stable_id(prefix: str, *values: Any, length: int = 16) -> str:
    digest = hashlib.sha256("\x1f".join(_text(value) for value in values).encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:length]}"


def _verify_manifest_file(manifest_path: Path, relative: str, source_path: Path) -> None:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    expected = _text(dict(payload.get("files_sha256") or {}).get(relative))
    if not expected:
        raise EmbeddingForensicsError(
            f"Trusted manifest does not record {relative}: {manifest_path}"
        )
    actual = _sha256_file(Path(source_path))
    if actual != expected:
        raise EmbeddingForensicsError(
            f"Source hash mismatch for {source_path}; expected {expected}, found {actual}"
        )


def locate_completed_shadow_evaluation(review_package: Path) -> Path:
    review_package = Path(review_package)
    metadata_path = review_package / "private" / "export_metadata.json"
    if not metadata_path.is_file():
        raise EmbeddingForensicsError(f"Shadow review metadata not found: {metadata_path}")
    review_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_package_id = _text(review_metadata.get("package_id"))
    if not expected_package_id:
        raise EmbeddingForensicsError("Shadow review package ID is missing")
    candidates: list[Path] = []
    for decision_path in sorted((review_package / "evaluation").glob("*/decision.json")):
        directory = decision_path.parent
        manifest_path = directory / "evaluation_output_manifest.json"
        summary_path = directory / "shadow_validation_summary.json"
        if not manifest_path.is_file() or not summary_path.is_file():
            continue
        try:
            verified = verify_output_manifest(directory, manifest_path)["manifest"]
        except ShadowValidationError:
            continue
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            _text(decision.get("candidate_id")) == EXPECTED_CANDIDATE_ID
            and _text(decision.get("decision")) == "reject_candidate"
            and decision.get("production_approved") is False
            and _text(summary.get("package_id")) == expected_package_id
            and _text(verified.get("package_id") or expected_package_id) == expected_package_id
            and _text(summary.get("candidate_config_sha256"))
            == EXPECTED_CANDIDATE_CONFIG_SHA256
        ):
            candidates.append(directory)
    if len(candidates) != 1:
        raise EmbeddingForensicsError(
            f"Expected one integrity-checked rejected shadow evaluation under {review_package}; "
            f"found {len(candidates)}"
        )
    return candidates[0]


def build_candidate_rejection_record(
    *,
    candidate_config_path: Path,
    calibration_dir: Path,
    shadow_review_package: Path,
    archived_at: str | None = None,
) -> dict[str, Any]:
    candidate_config_path = Path(candidate_config_path)
    calibration_dir = Path(calibration_dir)
    shadow_review_package = Path(shadow_review_package)
    candidate_manifest = calibration_dir / "output_manifest.json"
    try:
        frozen = load_frozen_candidate(
            candidate_config_path,
            candidate_manifest,
            subject_abbr=EXPECTED_SUBJECT,
        )
        calibration_manifest = verify_output_manifest(calibration_dir, candidate_manifest)["manifest"]
    except ShadowValidationError as exc:
        raise EmbeddingForensicsError(str(exc)) from exc
    if frozen.config_sha256 != EXPECTED_CANDIDATE_CONFIG_SHA256:
        raise EmbeddingForensicsError("Candidate-config SHA-256 does not match the frozen candidate")
    evaluation_dir = locate_completed_shadow_evaluation(shadow_review_package)
    evaluation_manifest_path = evaluation_dir / "evaluation_output_manifest.json"
    try:
        evaluation_manifest = verify_output_manifest(evaluation_dir, evaluation_manifest_path)["manifest"]
    except ShadowValidationError as exc:
        raise EmbeddingForensicsError(str(exc)) from exc
    decision_path = evaluation_dir / "decision.json"
    summary_path = evaluation_dir / "shadow_validation_summary.json"
    false_accept_path = evaluation_dir / "false_accept_audit.csv"
    reviewed_path = evaluation_dir / "reviewed_shadow_tracks.csv"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    false_accepts = pd.read_csv(false_accept_path, dtype=str, keep_default_na=False)
    class_counts = false_accepts["Shadow_Validation_Class"].value_counts().to_dict()
    correct = int(summary.get("correct_recoveries") or 0)
    confirmed_false = int(class_counts.get("confirmed_false_identity", 0))
    outsiders = int(class_counts.get("unsafe_outsider_false_acceptance", 0))
    if (
        _text(decision.get("candidate_id")) != EXPECTED_CANDIDATE_ID
        or _text(decision.get("decision")) != "reject_candidate"
        or decision.get("production_approved") is not False
        or correct != 13
        or confirmed_false != 5
        or outsiders != 2
    ):
        raise EmbeddingForensicsError("TUE_P2 rejection evidence does not match the audited decision")
    calibration_summary_path = calibration_dir / "calibration_summary.json"
    calibration_ground_truth = calibration_dir / "ground_truth_manifest.json"
    source_paths = {
        "candidate_config": candidate_config_path,
        "calibration_output_manifest": candidate_manifest,
        "calibration_summary": calibration_summary_path,
        "calibration_ground_truth_manifest": calibration_ground_truth,
        "shadow_review_manifest": shadow_review_package / "reviewer" / "review_manifest.json",
        "shadow_private_predictions": shadow_review_package / "private" / "hidden_predictions.csv",
        "shadow_export_metadata": shadow_review_package / "private" / "export_metadata.json",
        "shadow_evaluation_manifest": evaluation_manifest_path,
        "shadow_decision": decision_path,
        "shadow_summary": summary_path,
        "shadow_false_accept_audit": false_accept_path,
        "shadow_reviewed_tracks": reviewed_path,
    }
    missing = [name for name, path in source_paths.items() if not path.is_file()]
    if missing:
        raise EmbeddingForensicsError("Candidate rejection sources missing: " + ", ".join(missing))
    calibration_summary = json.loads(calibration_summary_path.read_text(encoding="utf-8"))
    if (
        int(dict(calibration_summary.get("benchmark") or {}).get("total_tracks") or 0) != 100
        or int(dict(calibration_summary.get("final_candidate") or {}).get("true_recoveries") or 0) != 8
        or int(dict(calibration_summary.get("final_candidate") or {}).get("false_accepts") or 0) != 0
    ):
        raise EmbeddingForensicsError("TUE_P1 calibration evidence does not match the frozen benchmark")
    return {
        "schema_version": FORENSIC_SCHEMA_VERSION,
        "candidate_id": EXPECTED_CANDIDATE_ID,
        "candidate_config_sha256": EXPECTED_CANDIDATE_CONFIG_SHA256,
        "status": REJECTED_CANDIDATE_STATUS,
        "final_decision": "reject_candidate",
        "production_approved": False,
        "enabled": False,
        "immutable": True,
        "archived_at": archived_at or datetime.now().isoformat(timespec="seconds"),
        "tue_p1_calibration_evidence": {
            "reviewed_tracks": 100,
            "out_of_fold_recoveries": int(
                dict(calibration_summary.get("out_of_fold") or {}).get("true_recoveries") or 0
            ),
            "out_of_fold_false_accepts": int(
                dict(calibration_summary.get("out_of_fold") or {}).get("false_accepts") or 0
            ),
            "full_benchmark_recoveries": 8,
            "full_benchmark_false_accepts": 0,
            "decision": _text(dict(calibration_summary.get("recommendation") or {}).get("decision")),
        },
        "tue_p2_evaluation_evidence": {
            "package_id": _text(summary.get("package_id")),
            "reviewed_tracks": int(summary.get("reviewed_shadow_tracks") or 0),
            "correct_recoveries": correct,
            "confirmed_false_identities": confirmed_false,
            "outsider_not_in_mapping_false_acceptances": outsiders,
            "decision": "reject_candidate",
        },
        "source_artifacts": {
            name: {"path": str(path.resolve()), "sha256": _sha256_file(path)}
            for name, path in source_paths.items()
        },
        "verified_manifests": {
            "calibration_files": len(calibration_manifest.get("files_sha256", {})),
            "evaluation_files": len(evaluation_manifest.get("files_sha256", {})),
        },
        "future_policy": {
            "may_be_enabled": False,
            "may_be_approved": False,
            "may_be_retuned_on_tue_p2": False,
            "tue_p2_is_untouched_validation": False,
        },
    }


def register_candidate_rejection(record: dict[str, Any], registry_dir: Path) -> tuple[Path, bool]:
    candidate_id = _text(record.get("candidate_id"))
    if not candidate_id or _text(record.get("status")) != REJECTED_CANDIDATE_STATUS:
        raise EmbeddingForensicsError("Only explicit rejected-candidate records may enter this registry")
    if record.get("production_approved") is not False or record.get("enabled") is not False:
        raise EmbeddingForensicsError("A rejected candidate cannot be enabled or production approved")
    registry_dir = Path(registry_dir)
    registry_dir.mkdir(parents=True, exist_ok=True)
    path = registry_dir / f"{candidate_id}.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        stable_existing = {key: value for key, value in existing.items() if key != "archived_at"}
        stable_new = {key: value for key, value in record.items() if key != "archived_at"}
        if stable_existing != stable_new:
            raise EmbeddingForensicsError(
                f"Conflicting rejection evidence already exists for candidate {candidate_id}"
            )
        return path, False
    with path.open("x", encoding="utf-8") as file_obj:
        json.dump(json_safe(record), file_obj, indent=2, ensure_ascii=True)
    return path, True


def assert_candidate_not_rejected_for_approval(candidate_id: str, registry_dir: Path) -> None:
    path = Path(registry_dir) / f"{_text(candidate_id)}.json"
    if not path.is_file():
        return
    record = json.loads(path.read_text(encoding="utf-8"))
    if _text(record.get("status")).startswith("rejected"):
        raise EmbeddingForensicsError(
            f"Candidate {candidate_id} is permanently rejected and cannot be approved"
        )


def _load_current_subject_roster(student_map_path: Path, subject_abbr: str) -> tuple[list[dict[str, Any]], set[str]]:
    try:
        roster, _, subject_rolls_raw = load_student_mapping(student_map_path, subject_abbr)
    except ReviewExportError as exc:
        raise EmbeddingForensicsError(str(exc)) from exc
    subject_rolls = {_canonical(value) for value in subject_rolls_raw if _canonical(value)}
    if len(subject_rolls) != EXPECTED_ROSTER_COUNT:
        raise EmbeddingForensicsError(
            f"Authoritative {subject_abbr} roster must contain {EXPECTED_ROSTER_COUNT} students; "
            f"found {len(subject_rolls)}"
        )
    canonical_rows = [_canonical(row.get("roll")) for row in roster if row.get("in_subject_roster")]
    if len(canonical_rows) != len(set(canonical_rows)):
        raise EmbeddingForensicsError("Authoritative subject roster contains canonical identity collisions")
    return roster, subject_rolls


def _verify_review_source_hashes(source: BenchmarkSource, metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    diagnostic = Path(source.diagnostic_run)
    tracklets = _single_file(diagnostic, "tracklet_diagnostics_*.csv", "tracklet diagnostics")
    observations = _single_file(diagnostic, "tracklet_observations_*.csv", "tracklet observations")
    recorded = dict(metadata.get("source_sha256") or {})
    for key, path in (("tracklet_diagnostics", tracklets), ("tracklet_observations", observations)):
        expected = _text(recorded.get(key))
        if not expected or _sha256_file(path) != expected:
            raise EmbeddingForensicsError(
                f"{source.source_kind} package source hash mismatch for {key}: {path}"
            )
    return {
        "tracklet_diagnostics": {"path": str(tracklets.resolve()), "sha256": _sha256_file(tracklets)},
        "tracklet_observations": {"path": str(observations.resolve()), "sha256": _sha256_file(observations)},
    }


def _diagnostic_features(diagnostic_run: Path) -> pd.DataFrame:
    path = _single_file(diagnostic_run, "tracklet_diagnostics_*.csv", "tracklet diagnostics")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "Tracklet_ID" not in frame.columns or frame["Tracklet_ID"].duplicated().any():
        raise EmbeddingForensicsError(f"Tracklet diagnostics have missing or duplicate Tracklet_ID values: {path}")
    keep = [
        "Tracklet_ID",
        "Tracklet_Eligible",
        "Tracklet_Quality_Rejection",
        "Observation_Count",
        "Embedding_Count",
        "Selected_Observation_Count",
        "Consistent_Embedding_Count",
        "Inconsistent_Embedding_Count",
        "Pairwise_Similarity_Median",
        "Zone_IDs",
        "Dominant_Frame_Best_Roll",
        "Dominant_Frame_Best_Share_Pct",
        "Quality_Weight_Median",
        "Face_Width_Median",
        "Blur_Median",
        "Detector_Score_Median",
        "Tracklet_Accepted",
        "Tracklet_Diagnostic_Reason",
    ]
    return frame[[column for column in keep if column in frame.columns]].copy()


def freeze_multisession_benchmark(
    sources: Sequence[BenchmarkSource],
    *,
    student_map_path: Path,
    subject_abbr: str = EXPECTED_SUBJECT,
    expected_total: int = EXPECTED_REVIEWED_TRACKS,
) -> MultisessionBenchmark:
    _, subject_rolls = _load_current_subject_roster(Path(student_map_path), subject_abbr)
    all_rows: list[pd.DataFrame] = []
    source_manifest: list[dict[str, Any]] = []
    for source in sources:
        package_dir = Path(source.package_dir)
        labels_path = Path(source.labels_path)
        if not labels_path.is_file():
            raise EmbeddingForensicsError(f"Completed human labels not found: {labels_path}")
        if source.expected_labels_sha256 and _sha256_file(labels_path) != source.expected_labels_sha256:
            raise EmbeddingForensicsError(
                f"Trusted label hash mismatch for {source.source_kind}: {labels_path}"
            )
        try:
            predictions, metadata, public_manifest = load_review_package_data(package_dir)
        except LabelValidationError as exc:
            raise EmbeddingForensicsError(str(exc)) from exc
        if predictions["Review_ID"].duplicated().any():
            raise EmbeddingForensicsError(f"{source.source_kind} package contains duplicate Review_ID values")
        labels = pd.read_csv(labels_path, dtype=str, keep_default_na=False)
        if "Review_ID" in labels and labels["Review_ID"].duplicated().any():
            raise EmbeddingForensicsError(f"{source.source_kind} labels contain duplicate Review_ID values")
        package_id = _text(metadata.get("package_id"))
        try:
            validated = validate_label_dataframe(
                labels,
                set(predictions["Review_ID"].astype(str)),
                package_id,
                subject_rolls,
            )
        except LabelValidationError as exc:
            raise EmbeddingForensicsError(str(exc)) from exc
        if len(predictions) != source.expected_rows or len(validated) != source.expected_rows:
            raise EmbeddingForensicsError(
                f"{source.source_kind} expected {source.expected_rows} reviewed rows; "
                f"found predictions={len(predictions)}, labels={len(validated)}"
            )
        if _text(metadata.get("subject_abbr")).upper() != subject_abbr.upper():
            raise EmbeddingForensicsError(f"{source.source_kind} package is not scoped to {subject_abbr}")
        joined = predictions.merge(
            validated,
            on=["Package_ID", "Review_ID"],
            how="inner",
            validate="one_to_one",
        )
        diagnostics = _diagnostic_features(source.diagnostic_run)
        joined = joined.merge(diagnostics, on="Tracklet_ID", how="left", suffixes=("", "_Diagnostic"), validate="many_to_one")
        if joined["Tracklet_Eligible"].eq("").all():
            raise EmbeddingForensicsError(f"{source.source_kind} tracks did not join to diagnostic features")
        joined["Actual_Roll"] = joined["Actual_Roll"].map(_canonical)
        joined["Predicted_Roll"] = joined["Predicted_Roll"].map(_canonical)
        joined["Second_Roll"] = joined["Second_Roll"].map(_canonical)
        invalid_actual = sorted(
            set(joined.loc[joined["Review_Status"].eq("identified"), "Actual_Roll"]) - subject_rolls
        )
        if invalid_actual:
            raise EmbeddingForensicsError(
                f"{source.source_kind} identified rolls are outside the authoritative roster: "
                + ", ".join(invalid_actual)
            )
        joined["Source_Kind"] = source.source_kind
        joined["Baseline_Status"] = source.baseline_status
        joined["Benchmark_Row_ID"] = [
            _stable_id("GT", package_id, review_id)
            for review_id in joined["Review_ID"].astype(str)
        ]
        joined["Top1_Correct"] = np.where(
            joined["Review_Status"].eq("identified"),
            np.where(joined["Predicted_Roll"].eq(joined["Actual_Roll"]), "Yes", "No"),
            "Not_Applicable",
        )
        joined["Second_Candidate_Correct"] = np.where(
            joined["Review_Status"].eq("identified") & joined["Second_Roll"].eq(joined["Actual_Roll"]),
            "Yes",
            "No",
        )
        if "Vote_Ratio_Pct" not in joined.columns:
            joined["Vote_Ratio_Pct"] = joined.get("Dominant_Frame_Best_Share_Pct", "")
        else:
            joined["Vote_Ratio_Pct"] = joined["Vote_Ratio_Pct"].where(
                joined["Vote_Ratio_Pct"].astype(str).str.strip().ne(""),
                joined.get("Dominant_Frame_Best_Share_Pct", ""),
            )
        if "Consistency_Ratio" not in joined.columns:
            embedding_counts = pd.to_numeric(joined.get("Embedding_Count", 0), errors="coerce").fillna(0)
            consistent_counts = pd.to_numeric(
                joined.get("Consistent_Embedding_Count", 0), errors="coerce"
            ).fillna(0)
            joined["Consistency_Ratio"] = np.where(
                embedding_counts.gt(0), consistent_counts / embedding_counts, 0.0
            )
        source_hashes = _verify_review_source_hashes(source, metadata)
        source_files = {
            "labels": {"path": str(labels_path.resolve()), "sha256": _sha256_file(labels_path)},
            "hidden_predictions": {
                "path": str((package_dir / "private" / "hidden_predictions.csv").resolve()),
                "sha256": _sha256_file(package_dir / "private" / "hidden_predictions.csv"),
            },
            "export_metadata": {
                "path": str((package_dir / "private" / "export_metadata.json").resolve()),
                "sha256": _sha256_file(package_dir / "private" / "export_metadata.json"),
            },
            "review_manifest": {
                "path": str((package_dir / "reviewer" / "review_manifest.json").resolve()),
                "sha256": _sha256_file(package_dir / "reviewer" / "review_manifest.json"),
            },
            **source_hashes,
        }
        if source.trusted_manifest_path:
            trusted = Path(source.trusted_manifest_path)
            if not trusted.is_file():
                raise EmbeddingForensicsError(f"Trusted source manifest not found: {trusted}")
            source_files["trusted_manifest"] = {
                "path": str(trusted.resolve()),
                "sha256": _sha256_file(trusted),
            }
        source_manifest.append(
            {
                "source_kind": source.source_kind,
                "package_id": package_id,
                "session_id": _text(metadata.get("session_id")),
                "reviewed_rows": len(joined),
                "baseline_status": source.baseline_status,
                "human_reviewed": True,
                "blind_review": bool(public_manifest.get("blind_review")),
                "files": source_files,
            }
        )
        all_rows.append(joined)
    if not all_rows:
        raise EmbeddingForensicsError("No human-reviewed benchmark sources were supplied")
    benchmark = pd.concat(all_rows, ignore_index=True, sort=False)
    if len(benchmark) != expected_total:
        raise EmbeddingForensicsError(
            f"Multisession benchmark expected {expected_total} reviewed rows; found {len(benchmark)}"
        )
    if benchmark["Benchmark_Row_ID"].duplicated().any():
        raise EmbeddingForensicsError("Multisession benchmark contains duplicate benchmark row IDs")
    benchmark = benchmark.sort_values(
        ["Session_ID", "Source_Kind", "Package_ID", "Review_ID"], kind="stable"
    ).reset_index(drop=True)
    source_hash_digest = hashlib.sha256(
        json.dumps(source_manifest, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    benchmark_id = f"multisession-gt-{source_hash_digest[:16]}"
    status_counts = {
        str(key): int(value)
        for key, value in benchmark["Review_Status"].value_counts().sort_index().items()
    }
    correct = int(benchmark["Top1_Correct"].eq("Yes").sum())
    wrong = int(benchmark["Top1_Correct"].eq("No").sum())
    manifest = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "subject_abbr": subject_abbr.upper(),
        "authoritative_roster_count": len(subject_rolls),
        "student_map": {
            "path": str(Path(student_map_path).resolve()),
            "sha256": _sha256_file(Path(student_map_path)),
        },
        "source_sessions": sorted(set(benchmark["Session_ID"].astype(str))),
        "source_package_ids": sorted(set(benchmark["Package_ID"].astype(str))),
        "sources": source_manifest,
        "row_counts": {
            "human_reviewed": int(len(benchmark)),
            "predicted_correct": correct,
            "predicted_wrong": wrong,
            "outsider_not_in_mapping": int(benchmark["Review_Status"].eq("not_in_mapping").sum()),
            "mixed_unverifiable": int(
                benchmark["Review_Status"].isin({"mixed_track", "uncertain", "unidentifiable", "duplicate"}).sum()
            ),
        },
        "status_counts": status_counts,
        "session_use_policy": {
            "2026-06-30__B51__P1__CVO": "reviewed_regression_and_possible_adaptation",
            "2026-06-30__B51__P2__CVO": "reviewed_regression_and_possible_adaptation",
            "MON_P3": "reserved_intended_untouched_validation_not_processed",
        },
        "sessions_prohibited_from_untouched_claim": [
            "2026-06-30__B51__P1__CVO",
            "2026-06-30__B51__P2__CVO",
        ],
        "contains_unreviewed_tue_p2_baseline_tracks": False,
        "predictions_used_as_ground_truth": False,
        "private_internal_identity_artifact": True,
        "source_manifest_sha256": source_hash_digest,
    }
    return MultisessionBenchmark(frame=benchmark, manifest=manifest)


def analyze_prediction_confusions(benchmark: pd.DataFrame) -> ConfusionAnalysis:
    required = {
        "Benchmark_Row_ID",
        "Source_Kind",
        "Session_ID",
        "Package_ID",
        "Review_ID",
        "Tracklet_ID",
        "Checkpoint_ID",
        "Camera_ID",
        "Predicted_Roll",
        "Actual_Roll",
        "Second_Roll",
        "Review_Status",
        "Top1_Correct",
    }
    missing = sorted(required.difference(benchmark.columns))
    if missing:
        raise EmbeddingForensicsError("Benchmark missing confusion columns: " + ", ".join(missing))
    rows: list[dict[str, Any]] = []
    for source in benchmark.to_dict("records"):
        status = _text(source.get("Review_Status"))
        actual = _canonical(source.get("Actual_Roll"))
        predicted = _canonical(source.get("Predicted_Roll"))
        second = _canonical(source.get("Second_Roll"))
        if status == "identified":
            result = "correct" if predicted == actual else "incorrect"
            actual_bucket = actual
        elif status == "not_in_mapping":
            result = "outsider_not_in_mapping"
            actual_bucket = "<NOT_IN_MAPPING>"
        else:
            result = status or "unverifiable"
            actual_bucket = f"<{(status or 'UNVERIFIABLE').upper()}>"
        embedding_count = int(round(_number(source.get("Embedding_Count"))))
        consistent_count = int(round(_number(source.get("Consistent_Embedding_Count"))))
        consistency = source.get("Consistency_Ratio")
        if _text(consistency) == "":
            consistency = consistent_count / embedding_count if embedding_count else 0.0
        rows.append(
            {
                "Benchmark_Row_ID": source.get("Benchmark_Row_ID"),
                "Source_Kind": source.get("Source_Kind"),
                "Baseline_Status": source.get("Baseline_Status"),
                "Session_ID": source.get("Session_ID"),
                "Package_ID": source.get("Package_ID"),
                "Review_ID": source.get("Review_ID"),
                "Tracklet_ID": source.get("Tracklet_ID"),
                "Checkpoint_ID": source.get("Checkpoint_ID"),
                "Camera_ID": source.get("Camera_ID"),
                "Zone_IDs": source.get("Zone_IDs", ""),
                "Predicted_Roll": predicted,
                "Actual_Roll": actual,
                "Actual_Bucket": actual_bucket,
                "Second_Roll": second,
                "Review_Status": status,
                "Prediction_Result": result,
                "Top1_Correct": "Yes" if result == "correct" else "No",
                "Second_Candidate_Correct": "Yes" if status == "identified" and second == actual else "No",
                "Best_Score": _number(source.get("Best_Score")),
                "Second_Score": _number(source.get("Second_Score")),
                "Margin": _number(source.get("Margin")),
                "Vote_Ratio_Pct": _number(source.get("Vote_Ratio_Pct")),
                "Consistency_Ratio": _number(consistency),
                "Pairwise_Similarity_Median": _number(source.get("Pairwise_Similarity_Median")),
                "Observation_Count": int(round(_number(source.get("Observation_Count")))),
            }
        )
    matrix = pd.DataFrame(rows).sort_values(
        ["Session_ID", "Source_Kind", "Review_ID"], kind="stable"
    ).reset_index(drop=True)
    unsafe = matrix[
        matrix["Prediction_Result"].isin({"incorrect", "outsider_not_in_mapping"})
    ].copy()
    attractor_rows: list[dict[str, Any]] = []
    for predicted, group in unsafe.groupby("Predicted_Roll", sort=True):
        known = group[group["Prediction_Result"].eq("incorrect")]
        attractor_rows.append(
            {
                "Predicted_Roll": predicted,
                "Confirmed_False_Accepts": int(len(group)),
                "Distinct_Actual_Identities_Attracted": int(known["Actual_Roll"].nunique()),
                "Outsider_Not_In_Mapping_Accepts": int(group["Prediction_Result"].eq("outsider_not_in_mapping").sum()),
                "Sessions": int(group["Session_ID"].nunique()),
                "Actual_Identities": "; ".join(sorted(set(known["Actual_Roll"]) - {""})),
                "Best_Score_Min": round(float(group["Best_Score"].min()), 4),
                "Best_Score_Median": round(float(group["Best_Score"].median()), 4),
                "Best_Score_Max": round(float(group["Best_Score"].max()), 4),
                "Margin_Min": round(float(group["Margin"].min()), 4),
                "Margin_Median": round(float(group["Margin"].median()), 4),
                "Margin_Max": round(float(group["Margin"].max()), 4),
                "Vote_Ratio_Median": round(float(group["Vote_Ratio_Pct"].median()), 4),
                "Consistency_Ratio_Median": round(float(group["Consistency_Ratio"].median()), 4),
                "Investigation_Flag": "review_needed_not_blacklisted",
            }
        )
    attractors = pd.DataFrame(attractor_rows)
    if not attractors.empty:
        attractors = attractors.sort_values(
            [
                "Confirmed_False_Accepts",
                "Distinct_Actual_Identities_Attracted",
                "Sessions",
                "Best_Score_Median",
            ],
            ascending=[False, False, False, False],
            kind="stable",
        ).reset_index(drop=True)
        attractors.insert(0, "Attractor_Rank", np.arange(1, len(attractors) + 1))
    miss_rows: list[dict[str, Any]] = []
    identified = matrix[matrix["Review_Status"].eq("identified")]
    for actual, group in identified.groupby("Actual_Roll", sort=True):
        missed = group[group["Prediction_Result"].eq("incorrect")]
        miss_rows.append(
            {
                "Actual_Roll": actual,
                "Reviewed_Tracks": int(len(group)),
                "Correct_Top1": int(group["Prediction_Result"].eq("correct").sum()),
                "Missed_Top1": int(len(missed)),
                "Top1_Accuracy": _ratio(int(group["Prediction_Result"].eq("correct").sum()), len(group)),
                "Second_Candidate_Correct": int(group["Second_Candidate_Correct"].eq("Yes").sum()),
                "Wrong_Predicted_Identities": "; ".join(sorted(set(missed["Predicted_Roll"]) - {""})),
                "Sessions_With_Miss": int(missed["Session_ID"].nunique()),
                "Checkpoints_With_Miss": int(missed["Checkpoint_ID"].nunique()),
            }
        )
    miss_patterns = pd.DataFrame(miss_rows).sort_values(
        ["Missed_Top1", "Reviewed_Tracks", "Actual_Roll"], ascending=[False, False, True], kind="stable"
    ).reset_index(drop=True)
    pair_rows: list[dict[str, Any]] = []
    for (predicted, actual_bucket), group in matrix.groupby(
        ["Predicted_Roll", "Actual_Bucket"], sort=True
    ):
        pair_rows.append(
            {
                "Predicted_Roll": predicted,
                "Actual_Roll_or_Status": actual_bucket,
                "Tracks": int(len(group)),
                "Correct": int(group["Prediction_Result"].eq("correct").sum()),
                "Incorrect": int(group["Prediction_Result"].eq("incorrect").sum()),
                "Outsider_Not_In_Mapping": int(group["Prediction_Result"].eq("outsider_not_in_mapping").sum()),
                "Sessions": int(group["Session_ID"].nunique()),
                "Checkpoints": int(group["Checkpoint_ID"].nunique()),
                "Best_Score_Median": round(float(group["Best_Score"].median()), 4),
                "Margin_Median": round(float(group["Margin"].median()), 4),
            }
        )
    pairs = pd.DataFrame(pair_rows).sort_values(
        ["Incorrect", "Outsider_Not_In_Mapping", "Tracks", "Predicted_Roll"],
        ascending=[False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    return ConfusionAnalysis(
        matrix=matrix,
        attractors=attractors,
        miss_patterns=miss_patterns,
        pairs=pairs,
        summary={
            "reviewed_tracks": int(len(matrix)),
            "correct_top1": int(matrix["Prediction_Result"].eq("correct").sum()),
            "incorrect_top1_identified": int(matrix["Prediction_Result"].eq("incorrect").sum()),
            "outsider_not_in_mapping": int(matrix["Prediction_Result"].eq("outsider_not_in_mapping").sum()),
            "second_candidate_correct": int(matrix["Second_Candidate_Correct"].eq("Yes").sum()),
            "attractor_identities_flagged": int(len(attractors)),
            "automatic_blacklists_created": 0,
            "prediction_used_as_ground_truth": False,
        },
    )


def _normalize_vector(value: Any) -> tuple[np.ndarray, bool, float]:
    try:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return np.empty(0, dtype=np.float64), False, float("nan")
    finite = bool(vector.size and np.isfinite(vector).all())
    norm = float(np.linalg.norm(vector)) if finite else float("nan")
    valid = bool(finite and norm > 1e-12)
    return (vector / norm if valid else vector), valid, norm


def _distribution(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {key: None for key in ("min", "p10", "median", "p90", "max")}
    return {
        "min": round(float(np.min(finite)), 6),
        "p10": round(float(np.quantile(finite, 0.10)), 6),
        "median": round(float(np.median(finite)), 6),
        "p90": round(float(np.quantile(finite, 0.90)), 6),
        "max": round(float(np.max(finite)), 6),
    }


def analyze_embedding_records(
    records: Sequence[dict[str, Any]],
    *,
    roster_rows: Sequence[dict[str, Any]] | None = None,
    attractor_rolls: set[str] | None = None,
    frequently_missed_rolls: set[str] | None = None,
    repo_root: Path | None = None,
) -> EmbeddingGeometry:
    attractor_rolls = {_canonical(value) for value in (attractor_rolls or set())}
    frequently_missed_rolls = {_canonical(value) for value in (frequently_missed_rolls or set())}
    parsed: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        raw_roll = _text(record.get("roll_no"))
        roll = _canonical(raw_roll)
        normalized, valid, norm = _normalize_vector(record.get("embedding"))
        source_path = _text(record.get("image_path"))
        source_exists = False
        if source_path and repo_root is not None:
            path = Path(source_path)
            source_exists = (path if path.is_absolute() else Path(repo_root) / path).is_file()
        parsed.append(
            {
                "record_index": index,
                "raw_roll": raw_roll,
                "roll": roll,
                "vector": normalized,
                "valid": valid,
                "norm": norm,
                "dimension": int(normalized.size),
                "image_path": source_path,
                "source_path_exists": source_exists,
                "raw_embedding": np.asarray(record.get("embedding", [])),
            }
        )
    by_roll: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in parsed:
        by_roll[row["roll"]].append(row)
    normalized_by_roll: dict[str, list[np.ndarray]] = {}
    centroids: dict[str, np.ndarray] = {}
    medoids: dict[str, np.ndarray] = {}
    outlier_thresholds: dict[str, float] = {}
    health_rows: list[dict[str, Any]] = []
    outlier_rows: list[dict[str, Any]] = []
    record_stats: dict[int, dict[str, Any]] = {}
    for roll, group in sorted(by_roll.items()):
        valid_rows = [row for row in group if row["valid"]]
        vectors = [row["vector"] for row in valid_rows]
        normalized_by_roll[roll] = vectors
        raw_keys = sorted({_text(row["raw_roll"]) for row in group})
        dimensions = sorted({int(row["dimension"]) for row in group})
        exact_duplicate_count = 0
        near_duplicate_pairs = 0
        intra_values = np.empty(0, dtype=np.float64)
        medoid_index = -1
        medoid_similarity = np.empty(0, dtype=np.float64)
        robust_scores = np.empty(0, dtype=np.float64)
        if vectors:
            matrix = np.stack(vectors)
            centroid = np.mean(matrix, axis=0)
            centroid_norm = float(np.linalg.norm(centroid))
            centroid = centroid / centroid_norm if centroid_norm > 1e-12 else centroid
            centroids[roll] = centroid
            similarities = matrix @ matrix.T
            medoid_index = int(np.argmax(similarities.mean(axis=1)))
            medoid = matrix[medoid_index]
            medoids[roll] = medoid
            medoid_similarity = matrix @ medoid
            distances = 1.0 - medoid_similarity
            distance_median = float(np.median(distances))
            mad = float(np.median(np.abs(distances - distance_median)))
            scale = max(1.4826 * mad, 0.01)
            robust_scores = (distances - distance_median) / scale
            outlier_thresholds[roll] = max(
                -1.0,
                1.0 - (distance_median + 3.5 * scale),
            )
            if len(matrix) > 1:
                intra_values = similarities[np.triu_indices(len(matrix), 1)]
                near_duplicate_pairs = int(np.sum(intra_values >= 0.9999))
            exact_counts = Counter(
                np.asarray(row["raw_embedding"]).tobytes() for row in valid_rows
            )
            exact_duplicate_count = int(sum(count - 1 for count in exact_counts.values() if count > 1))
            for position, row in enumerate(valid_rows):
                is_outlier = bool(robust_scores[position] >= 3.5)
                record_stats[row["record_index"]] = {
                    "Medoid_Similarity": round(float(medoid_similarity[position]), 6),
                    "Robust_Outlier_Score": round(float(robust_scores[position]), 6),
                    "Outlier_Flag": "Yes" if is_outlier else "No",
                    "Medoid_Record_Index": int(valid_rows[medoid_index]["record_index"]),
                }
                if is_outlier:
                    outlier_rows.append(
                        {
                            "Record_Index": int(row["record_index"]),
                            "Canonical_Roll": roll,
                            "Raw_Embedding_Key": row["raw_roll"],
                            "Image_Path": row["image_path"],
                            "Medoid_Similarity": round(float(medoid_similarity[position]), 6),
                            "Robust_Outlier_Score": round(float(robust_scores[position]), 6),
                            "Finding": "suspect_review_needed_not_auto_excluded",
                        }
                    )
        intra = _distribution(intra_values)
        health_rows.append(
            {
                "Canonical_Roll": roll,
                "Raw_Embedding_Keys": "; ".join(raw_keys),
                "Embedding_Count": len(group),
                "Valid_Embedding_Count": len(valid_rows),
                "Invalid_NaN_Count": len(group) - len(valid_rows),
                "Vector_Dimensions": "; ".join(str(value) for value in dimensions),
                "Normalized_Vector_Validity": "Pass"
                if valid_rows and all(abs(float(row["norm"]) - 1.0) <= 1e-3 for row in valid_rows)
                else ("No_Valid_Embeddings" if not valid_rows else "Needs_Normalization"),
                "Exact_Duplicate_Embedding_Count": exact_duplicate_count,
                "Near_Duplicate_Pair_Count": near_duplicate_pairs,
                "Intra_Similarity_Min": intra["min"],
                "Intra_Similarity_P10": intra["p10"],
                "Intra_Similarity_Median": intra["median"],
                "Intra_Similarity_P90": intra["p90"],
                "Intra_Similarity_Max": intra["max"],
                "Medoid_Record_Index": int(valid_rows[medoid_index]["record_index"])
                if medoid_index >= 0
                else "",
                "Robust_Outlier_Count": int(sum(score >= 3.5 for score in robust_scores)),
                "Canonical_Key_Collision": "Yes" if len(raw_keys) > 1 else "No",
                "Annotated_Embedding_Key": "Yes" if any(_canonical(key) != key.strip().upper() for key in raw_keys) else "No",
                "Malformed_Embedding_Key": "Yes" if not roll or not _ROLL_PATTERN.fullmatch(roll) else "No",
                "Attractor_In_Reviewed_CCTV": "Yes" if roll in attractor_rolls else "No",
                "Frequently_Missed_In_Reviewed_CCTV": "Yes" if roll in frequently_missed_rolls else "No",
                "Stored_Source_Paths_Missing": int(sum(not row["source_path_exists"] for row in group if row["image_path"])),
            }
        )
    roster_by_roll = {
        _canonical(row.get("roll")): row for row in (roster_rows or []) if _canonical(row.get("roll"))
    }
    existing_health = {row["Canonical_Roll"] for row in health_rows}
    for roll, roster_row in sorted(roster_by_roll.items()):
        if roll in existing_health or not roster_row.get("in_subject_roster"):
            continue
        health_rows.append(
            {
                "Canonical_Roll": roll,
                "Raw_Embedding_Keys": "",
                "Embedding_Count": 0,
                "Valid_Embedding_Count": 0,
                "Invalid_NaN_Count": 0,
                "Vector_Dimensions": "",
                "Normalized_Vector_Validity": "No_Valid_Embeddings",
                "Exact_Duplicate_Embedding_Count": 0,
                "Near_Duplicate_Pair_Count": 0,
                "Intra_Similarity_Min": None,
                "Intra_Similarity_P10": None,
                "Intra_Similarity_Median": None,
                "Intra_Similarity_P90": None,
                "Intra_Similarity_Max": None,
                "Medoid_Record_Index": "",
                "Robust_Outlier_Count": 0,
                "Canonical_Key_Collision": "No",
                "Annotated_Embedding_Key": "No",
                "Malformed_Embedding_Key": "No",
                "Attractor_In_Reviewed_CCTV": "Yes" if roll in attractor_rolls else "No",
                "Frequently_Missed_In_Reviewed_CCTV": "Yes" if roll in frequently_missed_rolls else "No",
                "Stored_Source_Paths_Missing": 0,
            }
        )
    nearest_rows: list[dict[str, Any]] = []
    rolls = sorted(centroids)
    nearest_by_roll: dict[str, list[tuple[float, str, float]]] = {}
    for roll in rolls:
        neighbors = sorted(
            (
                (
                    float(centroids[roll] @ centroids[other]),
                    other,
                    float(medoids[roll] @ medoids[other]),
                )
                for other in rolls
                if other != roll
            ),
            key=lambda value: (-value[0], value[1]),
        )
        nearest_by_roll[roll] = neighbors
        for rank, (centroid_similarity, other, medoid_similarity) in enumerate(neighbors[:5], start=1):
            nearest_rows.append(
                {
                    "Canonical_Roll": roll,
                    "Neighbor_Rank": rank,
                    "Neighbor_Roll": other,
                    "Centroid_Cosine_Similarity": round(centroid_similarity, 6),
                    "Medoid_Cosine_Similarity": round(medoid_similarity, 6),
                    "Inter_Student_Separation": round(1.0 - centroid_similarity, 6),
                    "Finding": "geometry_hint_not_ground_truth",
                }
            )
    health = pd.DataFrame(health_rows)
    for index, row in health.iterrows():
        roll = str(row["Canonical_Roll"])
        neighbors = nearest_by_roll.get(roll, [])
        if neighbors:
            health.loc[index, "Nearest_Other_Centroid_Roll"] = neighbors[0][1]
            health.loc[index, "Nearest_Other_Centroid_Similarity"] = round(neighbors[0][0], 6)
            health.loc[index, "Nearest_Other_Medoid_Roll"] = max(
                neighbors, key=lambda value: (value[2], value[1])
            )[1]
            health.loc[index, "Nearest_Other_Medoid_Similarity"] = round(
                max(value[2] for value in neighbors), 6
            )
            health.loc[index, "Top_Confusing_Neighbors"] = "; ".join(
                f"{value[1]}:{value[0]:.4f}" for value in neighbors[:5]
            )
        else:
            health.loc[index, "Nearest_Other_Centroid_Roll"] = ""
            health.loc[index, "Nearest_Other_Centroid_Similarity"] = np.nan
            health.loc[index, "Nearest_Other_Medoid_Roll"] = ""
            health.loc[index, "Nearest_Other_Medoid_Similarity"] = np.nan
            health.loc[index, "Top_Confusing_Neighbors"] = ""
        health.loc[index, "No_Valid_Embeddings"] = (
            "Yes" if int(row["Valid_Embedding_Count"]) == 0 else "No"
        )
    duplicate_rows: list[dict[str, Any]] = []
    valid_parsed = [row for row in parsed if row["valid"]]
    for left_index, left in enumerate(valid_parsed):
        for right in valid_parsed[left_index + 1 :]:
            if left["roll"] == right["roll"] or left["dimension"] != right["dimension"]:
                continue
            similarity = float(left["vector"] @ right["vector"])
            exact = np.asarray(left["raw_embedding"]).tobytes() == np.asarray(right["raw_embedding"]).tobytes()
            if exact or similarity >= 0.9995:
                duplicate_rows.append(
                    {
                        "Left_Record_Index": left["record_index"],
                        "Left_Roll": left["roll"],
                        "Left_Image_Path": left["image_path"],
                        "Right_Record_Index": right["record_index"],
                        "Right_Roll": right["roll"],
                        "Right_Image_Path": right["image_path"],
                        "Cosine_Similarity": round(similarity, 8),
                        "Exact_Vector_Duplicate": "Yes" if exact else "No",
                        "Finding": "suspect_review_needed_not_proof_of_mislabel",
                    }
                )
    health = health.sort_values("Canonical_Roll", kind="stable").reset_index(drop=True)
    outliers = pd.DataFrame(outlier_rows)
    if outliers.empty:
        outliers = pd.DataFrame(
            columns=[
                "Record_Index",
                "Canonical_Roll",
                "Raw_Embedding_Key",
                "Image_Path",
                "Medoid_Similarity",
                "Robust_Outlier_Score",
                "Finding",
            ]
        )
    else:
        outliers = outliers.sort_values(
            ["Robust_Outlier_Score", "Canonical_Roll", "Record_Index"],
            ascending=[False, True, True],
            kind="stable",
        ).reset_index(drop=True)
    nearest = pd.DataFrame(nearest_rows)
    duplicates = pd.DataFrame(duplicate_rows)
    if duplicates.empty:
        duplicates = pd.DataFrame(
            columns=[
                "Left_Record_Index",
                "Left_Roll",
                "Left_Image_Path",
                "Right_Record_Index",
                "Right_Roll",
                "Right_Image_Path",
                "Cosine_Similarity",
                "Exact_Vector_Duplicate",
                "Finding",
            ]
        )
    nodes = [
        {
            "roll": row["Canonical_Roll"],
            "embedding_count": int(row["Embedding_Count"]),
            "attractor": row["Attractor_In_Reviewed_CCTV"] == "Yes",
            "frequently_missed": row["Frequently_Missed_In_Reviewed_CCTV"] == "Yes",
            "no_valid_embeddings": row["No_Valid_Embeddings"] == "Yes",
        }
        for row in health.to_dict("records")
    ]
    edges = [
        {
            "source": row["Canonical_Roll"],
            "target": row["Neighbor_Roll"],
            "centroid_similarity": row["Centroid_Cosine_Similarity"],
            "medoid_similarity": row["Medoid_Cosine_Similarity"],
            "directed_rank": int(row["Neighbor_Rank"]),
            "interpretation": "model_geometry_hint_not_ground_truth",
        }
        for row in nearest.to_dict("records")
        if int(row["Neighbor_Rank"]) <= 3
    ]
    summary = {
        "schema_version": FORENSIC_SCHEMA_VERSION,
        "embedding_records": len(parsed),
        "canonical_embedding_students": len(by_roll),
        "valid_embeddings": int(sum(row["valid"] for row in parsed)),
        "invalid_nan_embeddings": int(sum(not row["valid"] for row in parsed)),
        "vector_dimensions": {
            str(key): int(value)
            for key, value in sorted(Counter(row["dimension"] for row in parsed).items())
        },
        "students_with_annotated_keys": int(health["Annotated_Embedding_Key"].eq("Yes").sum()),
        "canonical_key_collisions": int(health["Canonical_Key_Collision"].eq("Yes").sum()),
        "students_without_valid_embeddings": int(health["No_Valid_Embeddings"].eq("Yes").sum()),
        "robust_outlier_embeddings": int(len(outliers)),
        "cross_student_exact_or_near_duplicates": int(len(duplicates)),
        "automatic_deletions": 0,
        "automatic_blacklists": 0,
        "findings_are_review_hints_not_ground_truth": True,
    }
    return EmbeddingGeometry(
        student_health=health,
        outliers=outliers,
        nearest_neighbors=nearest,
        duplicate_candidates=duplicates,
        graph={"schema_version": FORENSIC_SCHEMA_VERSION, "nodes": nodes, "edges": edges},
        summary=summary,
        normalized_by_roll=normalized_by_roll,
        centroids=centroids,
        medoids=medoids,
        outlier_thresholds=outlier_thresholds,
    )


def analyze_embedding_database(
    embeddings_path: Path,
    **kwargs: Any,
) -> EmbeddingGeometry:
    embeddings_path = Path(embeddings_path)
    if not embeddings_path.is_file():
        raise EmbeddingForensicsError(f"Embedding database not found: {embeddings_path}")
    with embeddings_path.open("rb") as file_obj:
        payload = pickle.load(file_obj)
    records = payload.get("records")
    if not isinstance(records, list):
        raise EmbeddingForensicsError("Embedding database payload does not contain a records list")
    return analyze_embedding_records(records, **kwargs)


def load_embedding_payload(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise EmbeddingForensicsError(f"Embedding database not found: {path}")
    with path.open("rb") as file_obj:
        payload = pickle.load(file_obj)
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise EmbeddingForensicsError("Embedding database has an unsupported payload")
    return payload


def discover_enrollment_roots(
    *,
    repo_root: Path,
    embeddings_path: Path,
    include_augmented: bool = True,
) -> list[dict[str, Any]]:
    repo_root = Path(repo_root).resolve()
    payload = load_embedding_payload(embeddings_path)
    configured = _text(dict(payload.get("metadata") or {}).get("dataset")) or str(DATASET_DIR)
    configured_path = Path(configured)
    if not configured_path.is_absolute():
        configured_path = repo_root / configured_path
    candidates: list[tuple[str, Path, bool]] = [
        ("production_recorded_dataset", configured_path.resolve(), True)
    ]
    default_dataset = (repo_root / Path(DATASET_DIR).name).resolve()
    if default_dataset != configured_path.resolve():
        candidates.append(("configured_dataset", default_dataset, False))
    if include_augmented:
        augmented = (repo_root / Path(AUGMENTED_DATASET_DIR).name).resolve()
        candidates.append(("configured_augmented_dataset_nonproduction", augmented, False))
    seen: set[Path] = set()
    roots: list[dict[str, Any]] = []
    for role, path, production in candidates:
        if path in seen:
            continue
        seen.add(path)
        roots.append(
            {
                "role": role,
                "path": path,
                "exists": path.is_dir(),
                "production_recorded_input": production,
                "optional": role == "configured_augmented_dataset_nonproduction",
            }
        )
    if not roots[0]["exists"]:
        raise EmbeddingForensicsError(
            f"Production-recorded enrollment dataset not found: {roots[0]['path']}"
        )
    return roots


def _image_files(root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in Path(root).rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.as_posix().lower(),
    )


def _landmark_pose(face: np.ndarray, image_shape: tuple[int, ...]) -> tuple[float | None, float | None, float]:
    if len(face) < 14:
        return None, None, 1.0
    points = np.asarray(face[4:14], dtype=np.float64).reshape(5, 2)
    height, width = image_shape[:2]
    inside = np.array(
        [0 <= point[0] < width and 0 <= point[1] < height for point in points],
        dtype=bool,
    )
    eye_vector = points[1] - points[0]
    roll_degrees = math.degrees(math.atan2(float(eye_vector[1]), float(eye_vector[0])))
    eye_midpoint = (points[0] + points[1]) / 2.0
    eye_distance = max(float(np.linalg.norm(eye_vector)), 1.0)
    yaw_proxy = float((points[2][0] - eye_midpoint[0]) / eye_distance)
    occlusion_proxy = round(float(1.0 - inside.mean()), 4)
    return round(roll_degrees, 4), round(yaw_proxy, 4), occlusion_proxy


def _nearest_other_embedding(
    embedding: np.ndarray,
    own_roll: str,
    centroids: dict[str, np.ndarray],
) -> tuple[str, float]:
    candidates = sorted(
        (
            (float(embedding @ centroid), roll)
            for roll, centroid in centroids.items()
            if roll != own_roll and embedding.size == centroid.size
        ),
        key=lambda value: (-value[0], value[1]),
    )
    return (candidates[0][1], candidates[0][0]) if candidates else ("", float("nan"))


def _audit_one_dataset_image(
    *,
    path: Path,
    source_role: str,
    source_root: Path,
    engine: FaceEngine,
    geometry: EmbeddingGeometry,
    master_rolls: set[str],
    subject_rolls: set[str],
    enrollment_config: dict[str, Any],
) -> dict[str, Any]:
    from scripts.build_embeddings import add_padding_for_detection, face_quality_score

    relative = path.relative_to(source_root)
    owner_folder = relative.parts[0] if relative.parts else ""
    owner_roll = _canonical(owner_folder)
    flags: list[str] = []
    if owner_roll not in master_rolls:
        flags.append("folder_identity_absent_from_authoritative_registry")
    if owner_folder.strip().upper() != owner_roll:
        flags.append("canonical_naming_mismatch")
    image = cv2.imread(str(path))
    base: dict[str, Any] = {
        "Dataset_Source_Role": source_role,
        "Dataset_Root": str(source_root.resolve()),
        "Source_Path": str(path.resolve()),
        "Relative_Path": relative.as_posix(),
        "Owning_Folder": owner_folder,
        "Canonical_Roll": owner_roll,
        "In_Master_Registry": "Yes" if owner_roll in master_rolls else "No",
        "In_CVO_Roster": "Yes" if owner_roll in subject_rolls else "No",
        "File_SHA256": _sha256_file(path),
        "File_Size_Bytes": path.stat().st_size,
        "Decode_Success": "No",
        "Image_Width": 0,
        "Image_Height": 0,
        "Detected_Face_Count": 0,
        "Valid_Face_Count": 0,
        "Primary_Face_Width": None,
        "Primary_Face_Height": None,
        "Detector_Confidence": None,
        "Blur_Laplacian_Variance": None,
        "Brightness_Mean": None,
        "Pose_Roll_Degrees": None,
        "Pose_Yaw_Proxy": None,
        "Occlusion_Proxy": None,
        "Valid_SFace_Embedding": "No",
        "Embedding_Dimension": 0,
        "Own_Medoid_Similarity": None,
        "Nearest_Other_Roll": "",
        "Nearest_Other_Similarity": None,
        "Model_Similarity_Hint_Not_Ground_Truth": "",
    }
    if image is None or image.size == 0:
        flags.append("unreadable_image")
        base["Audit_Flags"] = "; ".join(flags)
        base["Suspect_For_Human_Review"] = "Yes"
        return base
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    base.update(
        {
            "Decode_Success": "Yes",
            "Image_Width": width,
            "Image_Height": height,
            "Brightness_Mean": round(brightness, 4),
        }
    )
    if brightness < 40.0 or brightness > 215.0:
        flags.append("severe_exposure_issue")
    padded, _ = add_padding_for_detection(image, float(enrollment_config["pad_percent"]))
    faces = engine.detect_faces_scaled(padded, max_width=int(enrollment_config["det_max_width"]))
    base["Detected_Face_Count"] = int(len(faces))
    if len(faces) == 0:
        flags.append("no_face")
        base["Audit_Flags"] = "; ".join(flags)
        base["Suspect_For_Human_Review"] = "Yes"
        return base
    if len(faces) > 1:
        flags.append("multiple_faces")
    candidates: list[dict[str, Any]] = []
    rejected_reasons: list[str] = []
    for face in faces:
        accepted, reason, rank = face_quality_score(
            face,
            padded.shape,
            int(enrollment_config["min_face_size"]),
            float(enrollment_config["min_area_ratio"]),
            int(enrollment_config["min_landmarks_inside"]),
            float(enrollment_config["min_eye_distance"]),
        )
        if not accepted:
            rejected_reasons.append(reason)
            continue
        feature = engine.extract_feature(padded, face)
        normalized, valid, _ = _normalize_vector(feature)
        box = tuple(float(value) for value in engine.face_box(face))
        crop = _face_crop(padded, box, 0.15)
        crop_blur = (
            float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            if crop is not None and crop.size
            else 0.0
        )
        roll_degrees, yaw_proxy, occlusion_proxy = _landmark_pose(face, padded.shape)
        candidates.append(
            {
                "face": face,
                "rank": float(rank),
                "box": box,
                "feature": normalized,
                "feature_valid": valid,
                "blur": crop_blur,
                "roll_degrees": roll_degrees,
                "yaw_proxy": yaw_proxy,
                "occlusion_proxy": occlusion_proxy,
            }
        )
    base["Valid_Face_Count"] = len(candidates)
    if not candidates:
        if any(reason in {"face_too_small", "face_area_too_small"} for reason in rejected_reasons):
            flags.append("tiny_face")
        flags.append("no_valid_face")
        base["Audit_Flags"] = "; ".join(dict.fromkeys(flags))
        base["Suspect_For_Human_Review"] = "Yes"
        return base
    best = max(candidates, key=lambda item: (item["rank"], item["box"]))
    box = best["box"]
    base.update(
        {
            "Primary_Face_Width": round(float(box[2]), 4),
            "Primary_Face_Height": round(float(box[3]), 4),
            "Detector_Confidence": round(float(engine.face_score(best["face"])), 6),
            "Blur_Laplacian_Variance": round(float(best["blur"]), 4),
            "Pose_Roll_Degrees": best["roll_degrees"],
            "Pose_Yaw_Proxy": best["yaw_proxy"],
            "Occlusion_Proxy": best["occlusion_proxy"],
            "Valid_SFace_Embedding": "Yes" if best["feature_valid"] else "No",
            "Embedding_Dimension": int(best["feature"].size),
        }
    )
    if min(float(box[2]), float(box[3])) < int(enrollment_config["min_face_size"]):
        flags.append("tiny_face")
    if best["blur"] < 45.0:
        flags.append("severe_blur")
    if not best["feature_valid"]:
        flags.append("embedding_extraction_failure")
    else:
        own_medoid = geometry.medoids.get(owner_roll)
        own_similarity = (
            float(best["feature"] @ own_medoid)
            if own_medoid is not None and own_medoid.size == best["feature"].size
            else float("nan")
        )
        nearest_roll, nearest_similarity = _nearest_other_embedding(
            best["feature"], owner_roll, geometry.centroids
        )
        base["Own_Medoid_Similarity"] = (
            round(own_similarity, 6) if math.isfinite(own_similarity) else None
        )
        base["Nearest_Other_Roll"] = nearest_roll
        base["Nearest_Other_Similarity"] = (
            round(nearest_similarity, 6) if math.isfinite(nearest_similarity) else None
        )
        if nearest_roll:
            base["Model_Similarity_Hint_Not_Ground_Truth"] = (
                f"{nearest_roll} at cosine {nearest_similarity:.4f}; model similarity hint - not ground truth"
            )
        threshold = geometry.outlier_thresholds.get(owner_roll)
        if threshold is not None and math.isfinite(own_similarity) and own_similarity < threshold:
            flags.append("own_class_outlier")
        if (
            math.isfinite(own_similarity)
            and math.isfinite(nearest_similarity)
            and nearest_similarity > own_similarity + 0.02
        ):
            flags.append("closer_to_another_student_than_own_medoid")
    base["Audit_Flags"] = "; ".join(dict.fromkeys(flags))
    base["Suspect_For_Human_Review"] = "Yes" if flags else "No"
    return base


def audit_enrollment_datasets(
    roots: Sequence[dict[str, Any]],
    *,
    geometry: EmbeddingGeometry,
    student_map_path: Path,
    subject_abbr: str = EXPECTED_SUBJECT,
    embeddings_path: Path,
    yunet_model: Path = YUNET_MODEL,
    sface_model: Path = SFACE_MODEL,
    progress: Callable[[str], None] | None = print,
    analyzer: Callable[..., dict[str, Any]] | None = None,
) -> DatasetAudit:
    roster, subject_rolls = _load_current_subject_roster(student_map_path, subject_abbr)
    master_rolls = {_canonical(row.get("roll")) for row in roster if _canonical(row.get("roll"))}
    payload = load_embedding_payload(embeddings_path)
    metadata = dict(payload.get("metadata") or {})
    enrollment_config = {
        "detection_score": float(metadata.get("detection_score", 0.60)),
        "det_max_width": int(metadata.get("det_max_width", 1280)),
        "min_face_size": int(metadata.get("min_face_size", 50)),
        "min_area_ratio": float(metadata.get("min_area_ratio", 0.003)),
        "pad_percent": float(metadata.get("pad_percent", 0.25)),
        "min_landmarks_inside": int(metadata.get("min_landmarks_inside", 4)),
        "min_eye_distance": float(metadata.get("min_eye_distance", 8.0)),
    }
    engine = None
    if analyzer is None:
        if not Path(yunet_model).is_file() or not Path(sface_model).is_file():
            raise EmbeddingForensicsError("YuNet or SFace model is missing for the dataset audit")
        engine = FaceEngine(
            Path(yunet_model),
            Path(sface_model),
            detection_score=enrollment_config["detection_score"],
        )
    available_roots = [root for root in roots if bool(root.get("exists"))]
    missing_roots = [root for root in roots if not bool(root.get("exists"))]
    files: list[tuple[dict[str, Any], Path]] = []
    for root in available_roots:
        files.extend((root, path) for path in _image_files(Path(root["path"])))
    rows: list[dict[str, Any]] = []
    total = len(files)
    for index, (root, path) in enumerate(files, start=1):
        if progress and (index == 1 or index % 25 == 0 or index == total):
            progress(f"Dataset audit progress: {index}/{total} images")
        if analyzer is None:
            row = _audit_one_dataset_image(
                path=path,
                source_role=_text(root.get("role")),
                source_root=Path(root["path"]),
                engine=engine,
                geometry=geometry,
                master_rolls=master_rolls,
                subject_rolls=subject_rolls,
                enrollment_config=enrollment_config,
            )
        else:
            row = analyzer(
                path=path,
                source_role=_text(root.get("role")),
                source_root=Path(root["path"]),
                geometry=geometry,
                master_rolls=master_rolls,
                subject_rolls=subject_rolls,
                enrollment_config=enrollment_config,
            )
        rows.append(row)
    quality = pd.DataFrame(rows)
    if quality.empty:
        quality = pd.DataFrame(
            columns=[
                "Dataset_Source_Role",
                "Dataset_Root",
                "Source_Path",
                "Relative_Path",
                "Owning_Folder",
                "Canonical_Roll",
                "File_SHA256",
                "Audit_Flags",
                "Suspect_For_Human_Review",
            ]
        )
    duplicate_rows: list[dict[str, Any]] = []
    duplicate_group_by_path: dict[str, str] = {}
    for file_hash, group in quality.groupby("File_SHA256", sort=True):
        if not file_hash or len(group) <= 1:
            continue
        group_id = _stable_id("DUP", file_hash, length=12)
        owners = sorted(set(group["Canonical_Roll"].astype(str)) - {""})
        roles = sorted(set(group["Dataset_Source_Role"].astype(str)) - {""})
        for row in group.to_dict("records"):
            duplicate_group_by_path[_text(row.get("Source_Path"))] = group_id
            duplicate_rows.append(
                {
                    "Duplicate_Group_ID": group_id,
                    "File_SHA256": file_hash,
                    "Source_Path": row.get("Source_Path"),
                    "Dataset_Source_Role": row.get("Dataset_Source_Role"),
                    "Canonical_Roll": row.get("Canonical_Roll"),
                    "Group_File_Count": int(len(group)),
                    "Distinct_Student_Count": len(owners),
                    "Cross_Student_Duplicate": "Yes" if len(owners) > 1 else "No",
                    "Cross_Dataset_Role_Duplicate": "Yes" if len(roles) > 1 else "No",
                    "Finding": "suspect_review_needed_not_auto_deleted",
                }
            )
    if len(quality):
        quality["Duplicate_Group_ID"] = quality["Source_Path"].map(duplicate_group_by_path).fillna("")
        duplicate_mask = quality["Duplicate_Group_ID"].ne("")
        quality.loc[duplicate_mask, "Audit_Flags"] = quality.loc[duplicate_mask, "Audit_Flags"].map(
            lambda value: "; ".join(dict.fromkeys([part.strip() for part in f"{value}; duplicated_file_hash".split(";") if part.strip()]))
        )
        quality.loc[duplicate_mask, "Suspect_For_Human_Review"] = "Yes"
    inventory_columns = [
        "Dataset_Source_Role",
        "Dataset_Root",
        "Source_Path",
        "Relative_Path",
        "Owning_Folder",
        "Canonical_Roll",
        "In_Master_Registry",
        "In_CVO_Roster",
        "File_SHA256",
        "File_Size_Bytes",
        "Decode_Success",
        "Image_Width",
        "Image_Height",
    ]
    inventory = quality[[column for column in inventory_columns if column in quality.columns]].copy()
    suspects = quality[quality["Suspect_For_Human_Review"].eq("Yes")].copy()
    duplicates = pd.DataFrame(duplicate_rows)
    if duplicates.empty:
        duplicates = pd.DataFrame(
            columns=[
                "Duplicate_Group_ID",
                "File_SHA256",
                "Source_Path",
                "Dataset_Source_Role",
                "Canonical_Roll",
                "Group_File_Count",
                "Distinct_Student_Count",
                "Cross_Student_Duplicate",
                "Cross_Dataset_Role_Duplicate",
                "Finding",
            ]
        )
    flag_counts = Counter(
        part.strip()
        for value in quality.get("Audit_Flags", pd.Series(dtype=str)).astype(str)
        for part in value.split(";")
        if part.strip()
    )
    summary = {
        "schema_version": FORENSIC_SCHEMA_VERSION,
        "dataset_roots": [
            {
                **{key: value for key, value in root.items() if key != "path"},
                "path": str(Path(root["path"]).resolve()),
                "image_count": len(_image_files(Path(root["path"]))) if root.get("exists") else 0,
            }
            for root in roots
        ],
        "images_discovered": int(len(quality)),
        "decoded_images": int(quality.get("Decode_Success", pd.Series(dtype=str)).eq("Yes").sum()),
        "valid_face_embeddings_computed": int(
            quality.get("Valid_SFace_Embedding", pd.Series(dtype=str)).eq("Yes").sum()
        ),
        "suspect_images": int(len(suspects)),
        "duplicate_hash_groups": int(duplicates["Duplicate_Group_ID"].nunique()) if len(duplicates) else 0,
        "duplicate_file_excess": int(
            sum(max(0, len(group) - 1) for _, group in quality.groupby("File_SHA256"))
        ),
        "cross_student_duplicate_groups": int(
            duplicates.loc[duplicates["Cross_Student_Duplicate"].eq("Yes"), "Duplicate_Group_ID"].nunique()
        )
        if len(duplicates)
        else 0,
        "flag_counts": {str(key): int(value) for key, value in sorted(flag_counts.items())},
        "missing_optional_roots": [str(Path(root["path"]).resolve()) for root in missing_roots],
        "dataset_files_modified": False,
        "automatic_deletions": 0,
        "automatic_moves": 0,
        "automatic_relabels": 0,
    }
    return DatasetAudit(
        inventory=inventory,
        quality=quality,
        suspects=suspects,
        duplicates=duplicates,
        summary=summary,
    )


def audit_missing_embeddings(
    *,
    target_rolls: Sequence[str],
    student_map_path: Path,
    subject_abbr: str,
    roots: Sequence[dict[str, Any]],
    dataset_audit: DatasetAudit,
    geometry: EmbeddingGeometry,
) -> pd.DataFrame:
    roster, subject_rolls = _load_current_subject_roster(student_map_path, subject_abbr)
    master_rolls = {_canonical(row.get("roll")) for row in roster if _canonical(row.get("roll"))}
    root_folders: list[tuple[dict[str, Any], Path]] = []
    for root in roots:
        path = Path(root["path"])
        if not path.is_dir():
            continue
        root_folders.extend((root, folder) for folder in sorted(path.iterdir()) if folder.is_dir())
    health_by_roll = {
        _canonical(row.get("Canonical_Roll")): row
        for row in geometry.student_health.to_dict("records")
    }
    rows: list[dict[str, Any]] = []
    for requested in target_rolls:
        roll = _canonical(requested)
        aliases = [
            (root, folder)
            for root, folder in root_folders
            if _canonical(folder.name) == roll
        ]
        exact = [(root, folder) for root, folder in aliases if folder.name.strip().upper() == roll]
        source_paths = {str(folder.resolve()) for _, folder in aliases}
        audit_rows = dataset_audit.quality[
            dataset_audit.quality.get("Canonical_Roll", pd.Series(dtype=str)).astype(str).eq(roll)
        ]
        images = int(len(audit_rows))
        readable = int(audit_rows.get("Decode_Success", pd.Series(dtype=str)).eq("Yes").sum())
        valid_faces = int(audit_rows.get("Valid_SFace_Embedding", pd.Series(dtype=str)).eq("Yes").sum())
        embedding_count = int(health_by_roll.get(roll, {}).get("Valid_Embedding_Count") or 0)
        if embedding_count > 0:
            reason = "not_missing"
        elif not aliases:
            reason = "dataset_missing"
        elif images == 0:
            reason = "dataset_empty"
        elif not exact:
            reason = "folder_alias_mismatch"
        elif readable == 0:
            reason = "image_decode_failure"
        elif valid_faces == 0:
            reason = "no_valid_face"
        elif valid_faces > 0:
            reason = "embedding_build_omission"
        else:
            reason = "other"
        rows.append(
            {
                "Canonical_Roll": roll,
                "Present_In_Authoritative_CVO_Roster": "Yes" if roll in subject_rolls else "No",
                "Present_In_Master_Student_Registry": "Yes" if roll in master_rolls else "No",
                "Dataset_Folder_Exists": "Yes" if aliases else "No",
                "Exact_Canonical_Folder_Exists": "Yes" if exact else "No",
                "Candidate_Folder_Aliases": "; ".join(sorted(source_paths)),
                "Readable_Images": readable,
                "Available_Images": images,
                "Valid_Face_Images": valid_faces,
                "Valid_SFace_Embeddings_Can_Be_Computed": "Yes" if valid_faces > 0 else "No",
                "Current_Embedding_Count": embedding_count,
                "Missing_Embedding_Root_Cause": reason,
                "Production_Embedding_Database_Changed": "No",
            }
        )
    return pd.DataFrame(rows).sort_values("Canonical_Roll", kind="stable").reset_index(drop=True)


def _aggregate_file_state(repo_root: Path, files: Iterable[Path]) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    rows: list[dict[str, Any]] = []
    for path in sorted({Path(value).resolve() for value in files}, key=lambda value: str(value).lower()):
        if not path.is_file():
            continue
        try:
            display_path = path.relative_to(repo_root).as_posix()
        except ValueError:
            display_path = str(path)
        rows.append(
            {
                "path": display_path,
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
        )
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            f"{row['path']}\0{row['size_bytes']}\0{row['sha256']}\n".encode("utf-8")
        )
    return {
        "file_count": len(rows),
        "total_bytes": int(sum(row["size_bytes"] for row in rows)),
        "aggregate_sha256": digest.hexdigest(),
        "files": rows,
    }


def snapshot_protected_state(
    *,
    repo_root: Path,
    candidate_config_path: Path,
    embeddings_path: Path | None = None,
    embedding_summary_path: Path | None = None,
    student_map_path: Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    embeddings_path = Path(embeddings_path or repo_root / "models" / "student_embeddings.pkl")
    embedding_summary_path = Path(
        embedding_summary_path or repo_root / "models" / "embedding_summary.csv"
    )
    student_map_path = Path(student_map_path or repo_root / "data" / "student_faculty_map.json")
    dataset_dir = repo_root / "dataset"
    augmented_dir = repo_root / "augmented_dataset"
    attendance_root = repo_root / "attendance_output"
    official_attendance_files = [
        path
        for path in attendance_root.rglob("*")
        if path.is_file()
        and path.relative_to(attendance_root).parts
        and path.relative_to(attendance_root).parts[0].lower()
        not in {"diagnostics", "embedding_forensics"}
    ] if attendance_root.is_dir() else []
    categories = {
        "dataset": _aggregate_file_state(
            repo_root, (path for path in dataset_dir.rglob("*") if path.is_file())
        ) if dataset_dir.is_dir() else _aggregate_file_state(repo_root, []),
        "augmented_dataset": _aggregate_file_state(
            repo_root, (path for path in augmented_dir.rglob("*") if path.is_file())
        ) if augmented_dir.is_dir() else _aggregate_file_state(repo_root, []),
        "production_model_and_roster": _aggregate_file_state(
            repo_root, [embeddings_path, embedding_summary_path, student_map_path]
        ),
        "official_attendance_outside_diagnostics": _aggregate_file_state(
            repo_root, official_attendance_files
        ),
        "frozen_candidate_config": _aggregate_file_state(repo_root, [candidate_config_path]),
    }
    return {
        "schema_version": FORENSIC_SCHEMA_VERSION,
        "categories": categories,
        "protected_paths_only": True,
    }


def compare_protected_state(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    before_categories = dict(before.get("categories") or {})
    after_categories = dict(after.get("categories") or {})
    for category in sorted(set(before_categories) | set(after_categories)):
        old = dict(before_categories.get(category) or {})
        new = dict(after_categories.get(category) or {})
        if (
            old.get("file_count"),
            old.get("total_bytes"),
            old.get("aggregate_sha256"),
        ) != (
            new.get("file_count"),
            new.get("total_bytes"),
            new.get("aggregate_sha256"),
        ):
            changes.append(
                {
                    "category": category,
                    "before": {
                        "file_count": old.get("file_count"),
                        "total_bytes": old.get("total_bytes"),
                        "aggregate_sha256": old.get("aggregate_sha256"),
                    },
                    "after": {
                        "file_count": new.get("file_count"),
                        "total_bytes": new.get("total_bytes"),
                        "aggregate_sha256": new.get("aggregate_sha256"),
                    },
                }
            )
    return changes


def _roster_name_map(student_map_path: Path, subject_abbr: str) -> dict[str, str]:
    roster, _ = _load_current_subject_roster(student_map_path, subject_abbr)
    return {
        _canonical(row.get("roll")): _text(row.get("name"))
        for row in roster
        if _canonical(row.get("roll"))
    }


def export_enrollment_audit_review(
    *,
    dataset_audit: DatasetAudit,
    output_dir: Path,
    run_id: str,
    student_map_path: Path,
    subject_abbr: str = EXPECTED_SUBJECT,
    package_revision: str = "reviewer-v1",
) -> ForensicReviewPackage:
    suspects = dataset_audit.suspects.sort_values(
        ["Canonical_Roll", "Dataset_Source_Role", "Relative_Path"], kind="stable"
    ).reset_index(drop=True)
    names = _roster_name_map(student_map_path, subject_abbr)
    digest = hashlib.sha256(
        "\n".join(
            f"{_text(row.get('Source_Path'))}\0{_text(row.get('File_SHA256'))}"
            for row in suspects.to_dict("records")
        ).encode("utf-8")
    ).hexdigest()
    package_digest = hashlib.sha256(f"{digest}\0{package_revision}".encode("utf-8")).hexdigest()
    package_id = f"enrollment-audit-{package_digest[:16]}"
    items: list[ForensicReviewItem] = []
    for row in suspects.to_dict("records"):
        source_path = Path(_text(row.get("Source_Path")))
        expected_roll = _canonical(row.get("Canonical_Roll"))
        item_id = _stable_id(
            "EA",
            package_id,
            row.get("Dataset_Source_Role"),
            row.get("Relative_Path"),
            row.get("File_SHA256"),
            length=18,
        )
        public = {
            "Expected_Roll": expected_roll,
            "Expected_Student_Name": names.get(expected_roll, "Not in authoritative registry"),
            "Dataset_Role": row.get("Dataset_Source_Role"),
            "Dataset_Folder": row.get("Owning_Folder"),
            "Source_Filename": source_path.name,
            "Quality_Warnings": row.get("Audit_Flags"),
            "Selection_Reason": "Flagged by deterministic enrollment dataset audit",
            "Face_Count": row.get("Detected_Face_Count"),
            "Face_Size": (
                f"{_text(row.get('Primary_Face_Width'))} x {_text(row.get('Primary_Face_Height'))} px"
                if _text(row.get("Primary_Face_Width"))
                else "Unavailable"
            ),
            "Blur_Metric": row.get("Blur_Laplacian_Variance"),
            "Brightness": row.get("Brightness_Mean"),
            "Model_Similarity_Hint_Not_Ground_Truth": row.get(
                "Model_Similarity_Hint_Not_Ground_Truth"
            ),
        }
        private = {
            "expected_roll": expected_roll,
            "expected_name": names.get(expected_roll, ""),
            "dataset_root": row.get("Dataset_Root"),
            "dataset_role": row.get("Dataset_Source_Role"),
            "relative_path": row.get("Relative_Path"),
            "source_file_sha256": row.get("File_SHA256"),
            "duplicate_group_id": row.get("Duplicate_Group_ID"),
            "selection_flags": row.get("Audit_Flags"),
            "identity_source": "enrollment_folder_expected_identity_for_human_audit",
            "alternative_identity_is_ground_truth": False,
        }
        items.append(
            ForensicReviewItem(
                item_id=item_id,
                evidence_source=source_path,
                public=public,
                private=private,
                preserve_resolution=False,
            )
        )
    return export_forensic_review_package(
        output_dir=Path(output_dir),
        package_id=package_id,
        package_kind="enrollment_audit",
        title="Enrollment Image Audit",
        instructions=(
            "Review every flagged enrollment image against the expected student. Findings are "
            "proposals for a future version only; this reviewer never changes source files."
        ),
        actions=ENROLLMENT_ACTIONS,
        items=items,
        metadata={
            "forensic_run_id": run_id,
            "reviewer_bundle_revision": package_revision,
            "subject_abbr": subject_abbr,
            "selected_suspect_images": len(items),
            "dataset_audit_summary": dataset_audit.summary,
            "student_map_sha256": _sha256_file(student_map_path),
            "expected_identity_visible_for_enrollment_audit": True,
            "alternative_model_identity_is_ground_truth": False,
            "automatic_source_changes": False,
        },
    )


def _observation_rank(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        1 if _yes(row.get("Selected_For_Aggregation")) else 0,
        1 if _yes(row.get("Embedding_Extraction_Success")) else 0,
        _number(row.get("Quality_Weight")),
        _number(row.get("Detector_Score")),
        min(_number(row.get("Blur_Laplacian_Variance")), 2500.0) / 2500.0,
        _number(row.get("Face_Width")),
        -int(round(_number(row.get("Frame")))),
        _text(row.get("Observation_ID")),
    )


def _shortlist_track_observations(frame: pd.DataFrame, limit: int = 4) -> list[dict[str, Any]]:
    ordered = sorted(frame.to_dict("records"), key=_observation_rank, reverse=True)
    selected: list[dict[str, Any]] = []
    for row in ordered:
        frame_index = int(round(_number(row.get("Frame"))))
        if any(
            _text(existing.get("Video")) == _text(row.get("Video"))
            and abs(int(round(_number(existing.get("Frame")))) - frame_index) < 10
            for existing in selected
        ):
            continue
        selected.append(row)
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        known = {_text(row.get("Observation_ID")) for row in selected}
        selected.extend(row for row in ordered if _text(row.get("Observation_ID")) not in known)
    return selected[:limit]


def _select_diverse_cctv_candidates(
    candidates: list[dict[str, Any]],
    *,
    maximum: int = 5,
) -> list[dict[str, Any]]:
    remaining = list(candidates)
    selected: list[dict[str, Any]] = []
    while remaining and len(selected) < min(maximum, len(candidates)):
        ranked: list[tuple[tuple[Any, ...], dict[str, Any], float]] = []
        for candidate in remaining:
            similarities = [
                float(candidate["_embedding"] @ item["_embedding"])
                for item in selected
                if candidate["_embedding"].size == item["_embedding"].size
            ]
            nearest = max(similarities) if similarities else -1.0
            checkpoint_bonus = 0.35 if candidate["Checkpoint_ID"] not in {
                item["Checkpoint_ID"] for item in selected
            } else 0.0
            camera_bonus = 0.20 if candidate["Camera_ID"] not in {
                item["Camera_ID"] for item in selected
            } else 0.0
            track_bonus = 0.25 if candidate["Tracklet_ID"] not in {
                item["Tracklet_ID"] for item in selected
            } else 0.0
            similarity_bonus = min(0.4, max(0.0, (1.0 - nearest) * 1.5)) if selected else 0.4
            score = float(candidate["Selection_Quality_Score"]) + checkpoint_bonus + camera_bonus + track_bonus + similarity_bonus
            ranked.append(
                (
                    (
                        round(score, 8),
                        -round(nearest, 8),
                        _text(candidate.get("Observation_ID")),
                    ),
                    candidate,
                    nearest,
                )
            )
        ranked.sort(key=lambda value: value[0], reverse=True)
        _, chosen, nearest = ranked[0]
        chosen["Diversity_Nearest_Selected_Cosine"] = (
            round(nearest, 6) if selected else None
        )
        chosen["Diversity_New_Checkpoint"] = "Yes" if chosen["Checkpoint_ID"] not in {
            item["Checkpoint_ID"] for item in selected
        } else "No"
        chosen["Diversity_New_Camera"] = "Yes" if chosen["Camera_ID"] not in {
            item["Camera_ID"] for item in selected
        } else "No"
        selected.append(chosen)
        remaining = [
            candidate for candidate in remaining if candidate["Item_ID"] != chosen["Item_ID"]
        ]
    return selected


def propose_verified_cctv_crops(
    *,
    benchmark: pd.DataFrame,
    benchmark_sources: Sequence[BenchmarkSource],
    output_crop_dir: Path,
    repo_root: Path,
    student_map_path: Path,
    subject_abbr: str = EXPECTED_SUBJECT,
    yunet_model: Path = YUNET_MODEL,
    sface_model: Path = SFACE_MODEL,
    progress: Callable[[str], None] | None = print,
    maximum_per_identity: int = 5,
) -> CCTVSelection:
    _, subject_rolls = _load_current_subject_roster(student_map_path, subject_abbr)
    names = _roster_name_map(student_map_path, subject_abbr)
    eligible = benchmark[
        benchmark["Review_Status"].eq("identified")
        & benchmark["Actual_Roll"].map(_canonical).isin(subject_rolls)
        & benchmark["Actual_Roll"].map(_canonical).ne("")
    ].copy()
    if eligible.empty:
        raise EmbeddingForensicsError("No human-identified CVO tracks are eligible for CCTV crop proposals")
    source_by_kind = {source.source_kind: source for source in benchmark_sources}
    unknown_sources = sorted(set(eligible["Source_Kind"].astype(str)) - set(source_by_kind))
    if unknown_sources:
        raise EmbeddingForensicsError(
            "CCTV benchmark source mapping is incomplete: " + ", ".join(unknown_sources)
        )
    engine = FaceEngine(Path(yunet_model), Path(sface_model), detection_score=0.60)
    frame_loader = _OpenCVFrameLoader()
    video_hashes: dict[Path, str] = {}
    candidates: list[dict[str, Any]] = []
    skipped = Counter()
    processed_tracks = 0
    try:
        for source_kind in sorted(set(eligible["Source_Kind"].astype(str))):
            source = source_by_kind[source_kind]
            metadata_path = Path(source.package_dir) / "private" / "export_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            source_hashes = _verify_review_source_hashes(source, metadata)
            observations_path = Path(source_hashes["tracklet_observations"]["path"])
            observations = pd.read_csv(observations_path, dtype=str, keep_default_na=False)
            if observations["Observation_ID"].duplicated().any():
                raise EmbeddingForensicsError(
                    f"Diagnostic observations contain duplicate IDs: {observations_path}"
                )
            video_root = Path(_text(metadata.get("video_root")))
            if not video_root.is_absolute():
                video_root = Path(repo_root) / video_root
            video_index = _video_index(video_root)
            source_tracks = eligible[eligible["Source_Kind"].eq(source_kind)].sort_values(
                ["Actual_Roll", "Session_ID", "Checkpoint_ID", "Camera_ID", "Review_ID"],
                kind="stable",
            )
            for benchmark_row in source_tracks.to_dict("records"):
                processed_tracks += 1
                if progress and (processed_tracks == 1 or processed_tracks % 20 == 0):
                    progress(
                        f"CCTV crop extraction progress: {processed_tracks}/{len(eligible)} human-identified tracks"
                    )
                track_rows = observations[
                    observations["Tracklet_ID"].astype(str).eq(_text(benchmark_row.get("Tracklet_ID")))
                ]
                if track_rows.empty:
                    skipped["track_without_verified_observations"] += 1
                    continue
                for observation in _shortlist_track_observations(track_rows, limit=4):
                    try:
                        video_path = _resolve_video(
                            video_index,
                            _text(observation.get("Checkpoint_ID") or benchmark_row.get("Checkpoint_ID")),
                            _text(observation.get("Video") or benchmark_row.get("Video")),
                        ).resolve()
                        video_hash = video_hashes.setdefault(video_path, _sha256_file(video_path))
                        frame_index = int(round(_number(observation.get("Frame"))))
                        frame = frame_loader(video_path, frame_index)
                        bbox = _parse_bbox(observation.get("BBox_Original_Coordinates"))
                        crop = _face_crop(frame, bbox, 0.35)
                    except (OSError, ReviewExportError, ValueError):
                        skipped["source_decode_or_crop_failure"] += 1
                        continue
                    if crop is None or crop.size == 0 or min(crop.shape[:2]) < 36:
                        skipped["tiny_or_empty_crop"] += 1
                        continue
                    faces = engine.detect_faces(crop)
                    if len(faces) != 1:
                        skipped["crop_face_count_not_one"] += 1
                        continue
                    feature = engine.extract_feature(crop, faces[0])
                    normalized, valid, _ = _normalize_vector(feature)
                    if not valid:
                        skipped["crop_embedding_failure"] += 1
                        continue
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    brightness = float(np.mean(gray))
                    pose_roll, yaw_proxy, occlusion_proxy = _landmark_pose(faces[0], crop.shape)
                    quality_score = (
                        _number(observation.get("Quality_Weight"))
                        + 0.35 * _number(observation.get("Detector_Score"))
                        + 0.20 * min(1.0, blur / 400.0)
                        + 0.15 * min(1.0, _number(observation.get("Face_Width")) / 140.0)
                    )
                    actual_roll = _canonical(benchmark_row.get("Actual_Roll"))
                    item_id = _stable_id(
                        "CC",
                        actual_roll,
                        benchmark_row.get("Benchmark_Row_ID"),
                        observation.get("Observation_ID"),
                        video_hash,
                        length=18,
                    )
                    candidates.append(
                        {
                            "Item_ID": item_id,
                            "Actual_Roll": actual_roll,
                            "Actual_Student_Name": names.get(actual_roll, ""),
                            "Identity_Source": "human_review_actual_roll",
                            "Source_Kind": source_kind,
                            "Session_ID": benchmark_row.get("Session_ID"),
                            "Package_ID": benchmark_row.get("Package_ID"),
                            "Review_ID": benchmark_row.get("Review_ID"),
                            "Benchmark_Row_ID": benchmark_row.get("Benchmark_Row_ID"),
                            "Tracklet_ID": benchmark_row.get("Tracklet_ID"),
                            "Observation_ID": observation.get("Observation_ID"),
                            "Checkpoint_ID": observation.get("Checkpoint_ID") or benchmark_row.get("Checkpoint_ID"),
                            "Camera_ID": observation.get("Camera_ID") or benchmark_row.get("Camera_ID"),
                            "Video": observation.get("Video"),
                            "Source_Video_Path": str(video_path),
                            "Source_Video_SHA256": video_hash,
                            "Frame": frame_index,
                            "BBox_Original_Coordinates": observation.get("BBox_Original_Coordinates"),
                            "Crop_Width": int(crop.shape[1]),
                            "Crop_Height": int(crop.shape[0]),
                            "Detector_Score": round(_number(observation.get("Detector_Score")), 6),
                            "Source_Quality_Weight": round(_number(observation.get("Quality_Weight")), 6),
                            "Crop_Blur_Laplacian_Variance": round(blur, 4),
                            "Crop_Brightness_Mean": round(brightness, 4),
                            "Pose_Roll_Degrees": pose_roll,
                            "Pose_Yaw_Proxy": yaw_proxy,
                            "Occlusion_Proxy": occlusion_proxy,
                            "Selection_Quality_Score": round(quality_score, 8),
                            "Human_Eligible": "Yes",
                            "Model_Prediction_Used_As_Identity": "No",
                            "_crop": crop,
                            "_embedding": normalized,
                        }
                    )
    finally:
        frame_loader.close()
    if progress:
        progress(
            f"CCTV crop extraction complete: {len(candidates)} valid proposals from {processed_tracks} tracks"
        )
    selected: list[dict[str, Any]] = []
    for actual_roll in sorted({row["Actual_Roll"] for row in candidates}):
        identity_candidates = [row for row in candidates if row["Actual_Roll"] == actual_roll]
        selected.extend(
            _select_diverse_cctv_candidates(
                identity_candidates,
                maximum=max(3, min(8, int(maximum_per_identity))),
            )
        )
    output_crop_dir = Path(output_crop_dir)
    output_crop_dir.mkdir(parents=True, exist_ok=False)
    selected_ids = {row["Item_ID"] for row in selected}
    for row in selected:
        crop_path = output_crop_dir / f"{row['Item_ID']}.png"
        if not cv2.imwrite(str(crop_path), row["_crop"], [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise EmbeddingForensicsError(f"Could not write candidate CCTV crop: {crop_path}")
        row["Candidate_Crop_Path"] = str(crop_path.resolve())
        row["Candidate_Crop_SHA256"] = _sha256_file(crop_path)
        row["Selected_For_Review"] = "Yes"
    inventory_rows: list[dict[str, Any]] = []
    for row in candidates:
        public_row = {key: value for key, value in row.items() if not key.startswith("_")}
        public_row.setdefault("Candidate_Crop_Path", "")
        public_row.setdefault("Candidate_Crop_SHA256", "")
        public_row["Selected_For_Review"] = "Yes" if row["Item_ID"] in selected_ids else "No"
        inventory_rows.append(public_row)
    inventory = pd.DataFrame(inventory_rows)
    selected_frame = pd.DataFrame(
        [{key: value for key, value in row.items() if not key.startswith("_")} for row in selected]
    )
    summary_rows: list[dict[str, Any]] = []
    for actual_roll, group in eligible.groupby(eligible["Actual_Roll"].map(_canonical), sort=True):
        roll_candidates = inventory[inventory.get("Actual_Roll", pd.Series(dtype=str)).eq(actual_roll)]
        roll_selected = selected_frame[
            selected_frame.get("Actual_Roll", pd.Series(dtype=str)).eq(actual_roll)
        ]
        summary_rows.append(
            {
                "Actual_Roll": actual_roll,
                "Student_Name": names.get(actual_roll, ""),
                "Human_Identified_Tracks": int(len(group)),
                "Valid_Candidate_Observations": int(len(roll_candidates)),
                "Selected_Crops_For_Human_Approval": int(len(roll_selected)),
                "Selected_Checkpoints": int(roll_selected.get("Checkpoint_ID", pd.Series(dtype=str)).nunique()),
                "Selected_Cameras": int(roll_selected.get("Camera_ID", pd.Series(dtype=str)).nunique()),
                "Source_Sessions": "; ".join(sorted(set(roll_selected.get("Session_ID", pd.Series(dtype=str))))),
                "Identity_Source": "human_review_actual_roll",
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.attrs["selection_metadata"] = {
        "benchmark_rows": int(len(benchmark)),
        "eligible_human_identified_tracks": int(len(eligible)),
        "excluded_nonidentified_tracks": int(len(benchmark) - len(eligible)),
        "valid_candidate_observations": int(len(candidates)),
        "selected_crops": int(len(selected)),
        "selected_identities": int(selected_frame.get("Actual_Roll", pd.Series(dtype=str)).nunique()),
        "skipped_reason_counts": {str(key): int(value) for key, value in sorted(skipped.items())},
        "identity_source": "human_review_actual_roll",
        "prediction_used_as_ground_truth": False,
        "live_dataset_modified": False,
        "mon_p3_processed": False,
    }
    return CCTVSelection(inventory=inventory, selected=selected_frame, summary=summary)


def export_verified_cctv_review(
    *,
    selection: CCTVSelection,
    output_dir: Path,
    run_id: str,
    benchmark_id: str,
    student_map_path: Path,
    subject_abbr: str = EXPECTED_SUBJECT,
    package_revision: str = "reviewer-v1",
) -> ForensicReviewPackage:
    selected = selection.selected.sort_values(
        ["Actual_Roll", "Session_ID", "Checkpoint_ID", "Camera_ID", "Item_ID"],
        kind="stable",
    )
    digest = hashlib.sha256(
        "\n".join(
            f"{row['Item_ID']}\0{row['Candidate_Crop_SHA256']}"
            for row in selected.to_dict("records")
        ).encode("utf-8")
    ).hexdigest()
    package_digest = hashlib.sha256(f"{digest}\0{package_revision}".encode("utf-8")).hexdigest()
    package_id = f"verified-cctv-{package_digest[:16]}"
    items: list[ForensicReviewItem] = []
    for row in selected.to_dict("records"):
        public = {
            "Actual_Student_Roll": row.get("Actual_Roll"),
            "Actual_Student_Name": row.get("Actual_Student_Name"),
            "Identity_Basis": "Completed human review (Actual_Roll)",
            "Source_Session": row.get("Session_ID"),
            "Checkpoint": row.get("Checkpoint_ID"),
            "Camera": row.get("Camera_ID"),
            "Source_Frame": row.get("Frame"),
            "Crop_Size": f"{row.get('Crop_Width')} x {row.get('Crop_Height')} px",
            "Detector_Confidence": row.get("Detector_Score"),
            "Blur_Metric": row.get("Crop_Blur_Laplacian_Variance"),
            "Brightness": row.get("Crop_Brightness_Mean"),
            "Pose_Roll": row.get("Pose_Roll_Degrees"),
            "Pose_Yaw_Proxy": row.get("Pose_Yaw_Proxy"),
            "Diversity_Nearest_Selected_Cosine": row.get(
                "Diversity_Nearest_Selected_Cosine"
            ),
            "Diversity_New_Checkpoint": row.get("Diversity_New_Checkpoint"),
            "Diversity_New_Camera": row.get("Diversity_New_Camera"),
        }
        private = {
            "actual_roll": row.get("Actual_Roll"),
            "identity_source": "human_review_actual_roll",
            "benchmark_id": benchmark_id,
            "benchmark_row_id": row.get("Benchmark_Row_ID"),
            "source_kind": row.get("Source_Kind"),
            "source_package_id": row.get("Package_ID"),
            "review_id": row.get("Review_ID"),
            "tracklet_id": row.get("Tracklet_ID"),
            "observation_id": row.get("Observation_ID"),
            "source_video_path": row.get("Source_Video_Path"),
            "source_video_sha256": row.get("Source_Video_SHA256"),
            "frame": row.get("Frame"),
            "bbox_original_coordinates": row.get("BBox_Original_Coordinates"),
            "candidate_crop_sha256": row.get("Candidate_Crop_SHA256"),
            "model_prediction_used_as_identity": False,
            "adaptation_session_if_approved": True,
        }
        items.append(
            ForensicReviewItem(
                item_id=_text(row.get("Item_ID")),
                evidence_source=Path(_text(row.get("Candidate_Crop_Path"))),
                public=public,
                private=private,
                preserve_resolution=True,
            )
        )
    return export_forensic_review_package(
        output_dir=Path(output_dir),
        package_id=package_id,
        package_kind="verified_cctv_enrollment",
        title="Verified CCTV Enrollment Crop Proposals",
        instructions=(
            "Approve only a clearly identifiable, single-person crop suitable for a future "
            "candidate embedding version. Approval never copies a crop into the live dataset."
        ),
        actions=CCTV_ACTIONS,
        items=items,
        metadata={
            "forensic_run_id": run_id,
            "reviewer_bundle_revision": package_revision,
            "benchmark_id": benchmark_id,
            "subject_abbr": subject_abbr,
            "selected_crops": len(items),
            "selection_metadata": selection.summary.attrs.get("selection_metadata", {}),
            "student_map_sha256": _sha256_file(student_map_path),
            "identity_source": "human_review_actual_roll",
            "model_prediction_used_as_ground_truth": False,
            "approved_crops_are_future_adaptation_data": True,
            "mon_p3_remains_untouched": True,
            "automatic_dataset_copy": False,
        },
    )


def _forensic_report_markdown(summary: dict[str, Any], confusion: ConfusionAnalysis) -> str:
    attractor_lines = []
    for row in confusion.attractors.head(10).to_dict("records"):
        attractor_lines.append(
            "- {roll}: {false_accepts} reviewed false accepts, {identities} distinct known "
            "actual identities, {outsiders} outsider/not-in-mapping accepts".format(
                roll=row.get("Predicted_Roll"),
                false_accepts=row.get("Confirmed_False_Accepts"),
                identities=row.get("Distinct_Actual_Identities_Attracted"),
                outsiders=row.get("Outsider_Not_In_Mapping_Accepts"),
            )
        )
    missing_lines = [
        f"- {row['roll']}: {row['root_cause']}"
        for row in summary.get("missing_embedding_status", [])
    ]
    return "\n".join(
        [
            "# Phase 1.2I-A Embedding Forensics",
            "",
            "## Safety outcome",
            "",
            f"- Rejected candidate: `{summary['candidate_id']}` remains disabled and unapproved.",
            "- Production embeddings changed: no.",
            "- Enrollment datasets changed: no.",
            "- Official thresholds, attendance rule, and official attendance changed: no.",
            "- Candidate embedding version built or promoted: no.",
            "",
            "## Reviewed benchmark",
            "",
            f"- Tracks: {summary['benchmark_reviewed_tracks']}",
            f"- Sessions: {', '.join(summary['source_sessions'])}",
            f"- Status distribution: {json.dumps(summary['benchmark_status_counts'], sort_keys=True)}",
            f"- Top-1 correct: {summary['identity_confusion']['correct_top1']}",
            f"- Top-1 wrong for identified people: {summary['identity_confusion']['incorrect_top1_identified']}",
            f"- Outsider/not-in-mapping absorptions: {summary['identity_confusion']['outsider_not_in_mapping']}",
            "- TUE_P1 and TUE_P2 are reviewed regression/possible-adaptation data, not untouched validation.",
            "- MON_P3 remains reserved and was not processed.",
            "",
            "## Attractors for investigation",
            "",
            *(attractor_lines or ["- None flagged."]),
            "",
            "These are reviewed-evidence investigation flags, not automatic blacklists or proof of bad enrollment.",
            "",
            "## Enrollment and embedding audit",
            "",
            f"- Embedding records: {summary['embedding_counts']['records']}",
            f"- Canonical identities in embedding database: {summary['embedding_counts']['canonical_students']}",
            f"- Dataset images audited: {summary['dataset_image_counts']['audited']}",
            f"- Suspect enrollment images requiring review: {summary['suspect_enrollment_images']}",
            f"- Candidate CCTV crops requiring review: {summary['candidate_cctv_crops']}",
            "",
            "## Missing embeddings",
            "",
            *(missing_lines or ["- No requested missing identities were audited."]),
            "",
            "## Human actions required",
            "",
            "1. Complete every item in the enrollment-image reviewer and return its exported approval CSV.",
            "2. Complete every item in the CCTV-crop reviewer and return its exported approval CSV.",
            "3. Do not run a versioned embedding build until both CSVs pass validation.",
            "4. Preserve MON_P3 for untouched validation before any future promotion.",
            "",
            "## Review packages",
            "",
            f"- Enrollment audit: `{summary['review_packages']['enrollment_audit']}`",
            f"- Verified CCTV proposals: `{summary['review_packages']['verified_cctv']}`",
            "",
        ]
    )


def _require_forensic_inputs(
    *,
    candidate_config_path: Path,
    calibration_dir: Path,
    shadow_review_package: Path,
    benchmark_sources: Sequence[BenchmarkSource],
    student_map_path: Path,
    embeddings_path: Path,
    embedding_summary_path: Path,
    yunet_model: Path,
    sface_model: Path,
) -> None:
    required_files = {
        "candidate config": Path(candidate_config_path),
        "calibration output manifest": Path(calibration_dir) / "output_manifest.json",
        "student map": Path(student_map_path),
        "production embeddings": Path(embeddings_path),
        "production embedding summary": Path(embedding_summary_path),
        "YuNet model": Path(yunet_model),
        "SFace model": Path(sface_model),
    }
    for source in benchmark_sources:
        required_files[f"{source.source_kind} labels"] = Path(source.labels_path)
        required_files[f"{source.source_kind} package metadata"] = (
            Path(source.package_dir) / "private" / "export_metadata.json"
        )
        required_files[f"{source.source_kind} package predictions"] = (
            Path(source.package_dir) / "private" / "hidden_predictions.csv"
        )
        required_files[f"{source.source_kind} tracklet diagnostics"] = _single_file(
            Path(source.diagnostic_run), "tracklet_diagnostics_*.csv", "tracklet diagnostics"
        )
        required_files[f"{source.source_kind} tracklet observations"] = _single_file(
            Path(source.diagnostic_run), "tracklet_observations_*.csv", "tracklet observations"
        )
    missing = [label for label, path in required_files.items() if not path.is_file()]
    if not Path(shadow_review_package).is_dir():
        missing.append("TUE_P2 shadow review package")
    if missing:
        raise EmbeddingForensicsError("Required forensic inputs are missing: " + ", ".join(missing))


def run_embedding_forensics(
    *,
    repo_root: Path,
    candidate_config_path: Path,
    calibration_dir: Path,
    shadow_review_package: Path,
    benchmark_sources: Sequence[BenchmarkSource],
    student_map_path: Path,
    embeddings_path: Path,
    embedding_summary_path: Path,
    output_dir: Path | None = None,
    subject_abbr: str = EXPECTED_SUBJECT,
    include_augmented_dataset: bool = True,
    yunet_model: Path = YUNET_MODEL,
    sface_model: Path = SFACE_MODEL,
    progress: Callable[[str], None] | None = print,
) -> EmbeddingForensicsRun:
    started = time.perf_counter()
    repo_root = Path(repo_root).resolve()
    candidate_config_path = Path(candidate_config_path).resolve()
    calibration_dir = Path(calibration_dir).resolve()
    shadow_review_package = Path(shadow_review_package).resolve()
    student_map_path = Path(student_map_path).resolve()
    embeddings_path = Path(embeddings_path).resolve()
    embedding_summary_path = Path(embedding_summary_path).resolve()
    _require_forensic_inputs(
        candidate_config_path=candidate_config_path,
        calibration_dir=calibration_dir,
        shadow_review_package=shadow_review_package,
        benchmark_sources=benchmark_sources,
        student_map_path=student_map_path,
        embeddings_path=embeddings_path,
        embedding_summary_path=embedding_summary_path,
        yunet_model=Path(yunet_model),
        sface_model=Path(sface_model),
    )
    run_id = make_diagnostic_run_id("embedding_forensics")
    output_dir = Path(
        output_dir
        or repo_root / "attendance_output" / "embedding_forensics" / run_id
    ).resolve()
    if output_dir.exists():
        raise EmbeddingForensicsError(f"Refusing to overwrite forensic output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    if progress:
        progress(f"Forensic output: {output_dir}")
    try:
        protected_before = snapshot_protected_state(
            repo_root=repo_root,
            candidate_config_path=candidate_config_path,
            embeddings_path=embeddings_path,
            embedding_summary_path=embedding_summary_path,
            student_map_path=student_map_path,
        )
        _write_json(output_dir / "protected_state_before.json", protected_before)

        rejection_record = build_candidate_rejection_record(
            candidate_config_path=candidate_config_path,
            calibration_dir=calibration_dir,
            shadow_review_package=shadow_review_package,
        )
        rejection_registry_dir = repo_root / "models" / "candidate_registry" / "rejected"
        rejection_registry_path, rejection_created = register_candidate_rejection(
            rejection_record, rejection_registry_dir
        )
        shutil.copy2(rejection_registry_path, output_dir / "candidate_rejection_record.json")
        if progress:
            progress(
                "Rejected-candidate registry: "
                + ("created" if rejection_created else "verified existing identical record")
            )

        frozen = freeze_multisession_benchmark(
            benchmark_sources,
            student_map_path=student_map_path,
            subject_abbr=subject_abbr,
        )
        benchmark_path = output_dir / "multisession_ground_truth.csv"
        _write_csv(benchmark_path, frozen.frame)
        benchmark_manifest = dict(frozen.manifest)
        benchmark_manifest["benchmark_artifact"] = {
            "path": str(benchmark_path.resolve()),
            "sha256": _sha256_file(benchmark_path),
        }
        _write_json(output_dir / "multisession_ground_truth_manifest.json", benchmark_manifest)
        if progress:
            progress(f"Frozen human-reviewed benchmark: {len(frozen.frame)} tracks")

        confusion = analyze_prediction_confusions(frozen.frame)
        _write_csv(output_dir / "prediction_confusion_matrix.csv", confusion.matrix)
        _write_csv(output_dir / "predicted_identity_attractors.csv", confusion.attractors)
        _write_csv(output_dir / "actual_identity_miss_patterns.csv", confusion.miss_patterns)
        _write_csv(output_dir / "confusion_pairs.csv", confusion.pairs)

        roster_rows, _ = _load_current_subject_roster(student_map_path, subject_abbr)
        attractor_rolls = set(confusion.attractors.get("Predicted_Roll", pd.Series(dtype=str)).astype(str))
        missed = confusion.miss_patterns[
            pd.to_numeric(
                confusion.miss_patterns.get("Missed_Top1", pd.Series(dtype=float)),
                errors="coerce",
            ).fillna(0).ge(2)
        ]
        geometry = analyze_embedding_database(
            embeddings_path,
            roster_rows=roster_rows,
            attractor_rolls=attractor_rolls,
            frequently_missed_rolls=set(missed.get("Actual_Roll", pd.Series(dtype=str)).astype(str)),
            repo_root=repo_root,
        )
        _write_csv(output_dir / "student_embedding_health.csv", geometry.student_health)
        _write_csv(output_dir / "embedding_outliers.csv", geometry.outliers)
        _write_csv(output_dir / "cross_student_nearest_neighbors.csv", geometry.nearest_neighbors)
        _write_csv(output_dir / "cross_student_duplicate_candidates.csv", geometry.duplicate_candidates)
        _write_json(output_dir / "embedding_confusion_graph.json", geometry.graph)
        embedding_health_summary = {
            **geometry.summary,
            "production_embeddings": {
                "path": str(embeddings_path),
                "sha256": _sha256_file(embeddings_path),
            },
            "embedding_summary": {
                "path": str(embedding_summary_path),
                "sha256": _sha256_file(embedding_summary_path),
            },
            "yunet_model": {"path": str(Path(yunet_model).resolve()), "sha256": _sha256_file(Path(yunet_model))},
            "sface_model": {"path": str(Path(sface_model).resolve()), "sha256": _sha256_file(Path(sface_model))},
            "production_embeddings_modified": False,
        }
        _write_json(output_dir / "embedding_health_summary.json", embedding_health_summary)

        roots = discover_enrollment_roots(
            repo_root=repo_root,
            embeddings_path=embeddings_path,
            include_augmented=include_augmented_dataset,
        )
        dataset_audit = audit_enrollment_datasets(
            roots,
            geometry=geometry,
            student_map_path=student_map_path,
            subject_abbr=subject_abbr,
            embeddings_path=embeddings_path,
            yunet_model=Path(yunet_model),
            sface_model=Path(sface_model),
            progress=progress,
        )
        _write_csv(output_dir / "dataset_image_inventory.csv", dataset_audit.inventory)
        _write_csv(output_dir / "dataset_quality_audit.csv", dataset_audit.quality)
        _write_csv(output_dir / "dataset_suspect_images.csv", dataset_audit.suspects)
        _write_csv(output_dir / "dataset_duplicate_files.csv", dataset_audit.duplicates)
        dataset_summary = {
            **dataset_audit.summary,
            "root_manifests": [
                {
                    "role": root["role"],
                    "path": str(Path(root["path"]).resolve()),
                    "exists": bool(root["exists"]),
                    "file_state": _aggregate_file_state(
                        repo_root,
                        (path for path in Path(root["path"]).rglob("*") if path.is_file())
                        if Path(root["path"]).is_dir()
                        else [],
                    ),
                }
                for root in roots
            ],
        }
        _write_json(output_dir / "dataset_audit_summary.json", dataset_summary)

        missing_audit = audit_missing_embeddings(
            target_rolls=MISSING_EMBEDDING_ROLLS,
            student_map_path=student_map_path,
            subject_abbr=subject_abbr,
            roots=roots,
            dataset_audit=dataset_audit,
            geometry=geometry,
        )
        _write_csv(output_dir / "missing_embedding_audit.csv", missing_audit)

        enrollment_review = export_enrollment_audit_review(
            dataset_audit=dataset_audit,
            output_dir=output_dir / "suspicious_enrollment_image_review",
            run_id=run_id,
            student_map_path=student_map_path,
            subject_abbr=subject_abbr,
        )
        cctv_selection = propose_verified_cctv_crops(
            benchmark=frozen.frame,
            benchmark_sources=benchmark_sources,
            output_crop_dir=output_dir / "verified_cctv_candidate_crops",
            repo_root=repo_root,
            student_map_path=student_map_path,
            subject_abbr=subject_abbr,
            yunet_model=Path(yunet_model),
            sface_model=Path(sface_model),
            progress=progress,
        )
        _write_csv(output_dir / "verified_cctv_candidate_inventory.csv", cctv_selection.inventory)
        _write_csv(output_dir / "verified_cctv_selection_summary.csv", cctv_selection.summary)
        cctv_review = export_verified_cctv_review(
            selection=cctv_selection,
            output_dir=output_dir / "verified_cctv_enrollment_review",
            run_id=run_id,
            benchmark_id=frozen.manifest["benchmark_id"],
            student_map_path=student_map_path,
            subject_abbr=subject_abbr,
        )
        cli_script = repo_root / "scripts" / "validate_tracklet_ground_truth.py"
        enrollment_launcher = write_reviewer_launcher(
            launcher_path=output_dir / "open_enrollment_audit_review.ps1",
            python_executable=Path(sys.executable),
            cli_script=cli_script,
            command="open-enrollment-audit-review",
            review_package=enrollment_review.root,
        )
        cctv_launcher = write_reviewer_launcher(
            launcher_path=output_dir / "open_cctv_enrollment_review.ps1",
            python_executable=Path(sys.executable),
            cli_script=cli_script,
            command="open-cctv-enrollment-review",
            review_package=cctv_review.root,
        )

        protected_after = snapshot_protected_state(
            repo_root=repo_root,
            candidate_config_path=candidate_config_path,
            embeddings_path=embeddings_path,
            embedding_summary_path=embedding_summary_path,
            student_map_path=student_map_path,
        )
        _write_json(output_dir / "protected_state_after.json", protected_after)
        protected_changes = compare_protected_state(protected_before, protected_after)
        if protected_changes:
            _write_json(
                output_dir / "protected_state_violation.json",
                {"protected_state_unchanged": False, "changes": protected_changes},
            )
            raise EmbeddingForensicsError(
                "Protected production state changed during forensics: "
                + ", ".join(change["category"] for change in protected_changes)
            )

        missing_status = [
            {
                "roll": row.get("Canonical_Roll"),
                "root_cause": row.get("Missing_Embedding_Root_Cause"),
                "dataset_folder_exists": row.get("Dataset_Folder_Exists"),
                "valid_face_images": int(row.get("Valid_Face_Images") or 0),
            }
            for row in missing_audit.to_dict("records")
        ]
        selection_metadata = cctv_selection.summary.attrs.get("selection_metadata", {})
        summary = {
            "schema_version": FORENSIC_SCHEMA_VERSION,
            "forensic_run_id": run_id,
            "output_dir": str(output_dir),
            "candidate_id": EXPECTED_CANDIDATE_ID,
            "candidate_status": REJECTED_CANDIDATE_STATUS,
            "candidate_final_decision": "reject_candidate",
            "production_candidate_enabled": False,
            "production_candidate_approved": False,
            "candidate_rejection_registry": str(rejection_registry_path.resolve()),
            "candidate_rejection_registry_created": rejection_created,
            "benchmark_id": frozen.manifest["benchmark_id"],
            "benchmark_reviewed_tracks": int(len(frozen.frame)),
            "benchmark_status_counts": frozen.manifest["status_counts"],
            "source_sessions": frozen.manifest["source_sessions"],
            "session_use_policy": frozen.manifest["session_use_policy"],
            "identity_confusion": confusion.summary,
            "embedding_counts": {
                "records": geometry.summary["embedding_records"],
                "canonical_students": geometry.summary["canonical_embedding_students"],
                "valid": geometry.summary["valid_embeddings"],
                "invalid_nan": geometry.summary["invalid_nan_embeddings"],
                "robust_outliers": geometry.summary["robust_outlier_embeddings"],
                "cross_student_exact_or_near_duplicates": geometry.summary[
                    "cross_student_exact_or_near_duplicates"
                ],
            },
            "dataset_image_counts": {
                "audited": dataset_audit.summary["images_discovered"],
                "decoded": dataset_audit.summary["decoded_images"],
                "valid_face_embeddings": dataset_audit.summary[
                    "valid_face_embeddings_computed"
                ],
                "duplicate_hash_groups": dataset_audit.summary["duplicate_hash_groups"],
            },
            "missing_embedding_status": missing_status,
            "suspect_enrollment_images": int(len(dataset_audit.suspects)),
            "candidate_cctv_crops": int(len(cctv_selection.selected)),
            "eligible_human_identified_tracks_for_cctv": int(
                selection_metadata.get("eligible_human_identified_tracks") or 0
            ),
            "review_packages": {
                "enrollment_audit": str(enrollment_review.root.resolve()),
                "verified_cctv": str(cctv_review.root.resolve()),
            },
            "reviewer_launchers": {
                "enrollment_audit": str(enrollment_launcher.resolve()),
                "verified_cctv": str(cctv_launcher.resolve()),
            },
            "required_approval_csv_filenames": {
                "enrollment_audit": f"forensic_review_approvals_{enrollment_review.package_id}.csv",
                "verified_cctv": f"forensic_review_approvals_{cctv_review.package_id}.csv",
            },
            "exact_next_human_actions": [
                "Complete and export every enrollment-audit review action.",
                "Complete and export every verified-CCTV review action.",
                "Return both exported approval CSVs for validation before any versioned build.",
                "Keep MON_P3 untouched for future pre-promotion validation.",
            ],
            "protected_state_unchanged": True,
            "current_embeddings_modified": False,
            "current_datasets_modified": False,
            "official_thresholds_modified": False,
            "official_match_threshold": OFFICIAL_MATCH_THRESHOLD,
            "official_margin_threshold": OFFICIAL_MARGIN_THRESHOLD,
            "attendance_rule_modified": False,
            "official_attendance_checkpoints": OFFICIAL_ATTENDANCE_CHECKPOINTS,
            "official_attendance_modified": False,
            "candidate_embedding_version_built": False,
            "candidate_embedding_version_promoted": False,
            "mon_p3_processed": False,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        }
        _write_json(output_dir / "forensic_summary.json", summary)
        (output_dir / "forensic_report.md").write_text(
            _forensic_report_markdown(summary, confusion), encoding="utf-8"
        )
        artifact_hashes = {
            path.relative_to(output_dir).as_posix(): _sha256_file(path)
            for path in sorted(output_dir.rglob("*"))
            if path.is_file()
            and path.name not in {"forensic_run_manifest.json", "output_manifest.json"}
        }
        run_manifest = {
            "schema_version": FORENSIC_SCHEMA_VERSION,
            "forensic_run_id": run_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "candidate_id": EXPECTED_CANDIDATE_ID,
            "candidate_config_sha256": _sha256_file(candidate_config_path),
            "benchmark_id": frozen.manifest["benchmark_id"],
            "source_inputs": {
                "candidate_config": {"path": str(candidate_config_path), "sha256": _sha256_file(candidate_config_path)},
                "student_map": {"path": str(student_map_path), "sha256": _sha256_file(student_map_path)},
                "production_embeddings": {"path": str(embeddings_path), "sha256": _sha256_file(embeddings_path)},
                "embedding_summary": {"path": str(embedding_summary_path), "sha256": _sha256_file(embedding_summary_path)},
                "yunet_model": {"path": str(Path(yunet_model).resolve()), "sha256": _sha256_file(Path(yunet_model))},
                "sface_model": {"path": str(Path(sface_model).resolve()), "sha256": _sha256_file(Path(sface_model))},
            },
            "artifact_sha256": artifact_hashes,
            "protected_state_before": protected_before,
            "protected_state_after": protected_after,
            "protected_state_unchanged": True,
            "production_changes": False,
            "candidate_embedding_built": False,
            "candidate_embedding_promoted": False,
        }
        _write_json(output_dir / "forensic_run_manifest.json", run_manifest)
        output_manifest = write_output_manifest(
            output_dir,
            {
                "forensic_run_id": run_id,
                "benchmark_id": frozen.manifest["benchmark_id"],
                "candidate_id": EXPECTED_CANDIDATE_ID,
                "candidate_status": REJECTED_CANDIDATE_STATUS,
                "protected_state_unchanged": True,
                "production_embeddings_changed": False,
                "datasets_changed": False,
                "official_attendance_changed": False,
                "candidate_embedding_built": False,
                "candidate_embedding_promoted": False,
            },
        )
        verify_output_manifest(output_dir, output_manifest)
        if progress:
            progress(f"Embedding forensics complete in {time.perf_counter() - started:.2f} seconds")
        return EmbeddingForensicsRun(
            output_dir=output_dir,
            summary=summary,
            enrollment_review=enrollment_review,
            cctv_review=cctv_review,
            output_manifest=output_manifest,
        )
    except Exception as exc:
        failure_path = output_dir / "forensic_failure.json"
        if not failure_path.exists():
            _write_json(
                failure_path,
                {
                    "schema_version": FORENSIC_SCHEMA_VERSION,
                    "forensic_run_id": run_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "completed": False,
                    "production_success_claimed": False,
                },
            )
        try:
            write_output_manifest(
                output_dir,
                {
                    "forensic_run_id": run_id,
                    "completed": False,
                    "partial_output": True,
                    "production_success_claimed": False,
                },
                filename="partial_output_manifest.json",
            )
        except (OSError, ShadowValidationError):
            pass
        raise


def verify_reusable_forensic_run(
    *,
    output_dir: Path,
    repo_root: Path,
    candidate_config_path: Path,
    embeddings_path: Path,
    embedding_summary_path: Path,
    student_map_path: Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    try:
        verify_output_manifest(output_dir, output_dir / "output_manifest.json")
    except ShadowValidationError as exc:
        raise EmbeddingForensicsError(str(exc)) from exc
    run_manifest_path = output_dir / "forensic_run_manifest.json"
    summary_path = output_dir / "forensic_summary.json"
    if not run_manifest_path.is_file() or not summary_path.is_file():
        raise EmbeddingForensicsError(f"Reusable forensic run is incomplete: {output_dir}")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    expected_after = dict(run_manifest.get("protected_state_after") or {})
    current = snapshot_protected_state(
        repo_root=repo_root,
        candidate_config_path=candidate_config_path,
        embeddings_path=embeddings_path,
        embedding_summary_path=embedding_summary_path,
        student_map_path=student_map_path,
    )
    changes = compare_protected_state(expected_after, current)
    if changes:
        raise EmbeddingForensicsError(
            "Refusing to reuse stale forensic output; protected inputs changed: "
            + ", ".join(change["category"] for change in changes)
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("protected_state_unchanged") is not True:
        raise EmbeddingForensicsError("Reusable forensic run does not carry a clean safety result")
    return summary


def regenerate_forensic_review_packages(
    *,
    source_forensic_run: Path,
    repo_root: Path,
    candidate_config_path: Path,
    embeddings_path: Path,
    embedding_summary_path: Path,
    student_map_path: Path,
    output_dir: Path | None = None,
    subject_abbr: str = EXPECTED_SUBJECT,
    package_revision: str = "reviewer-v5-final-runtime-fix",
    progress: Callable[[str], None] | None = print,
) -> EmbeddingForensicsRun:
    started = time.perf_counter()
    source_forensic_run = Path(source_forensic_run).resolve()
    repo_root = Path(repo_root).resolve()
    source_summary = verify_reusable_forensic_run(
        output_dir=source_forensic_run,
        repo_root=repo_root,
        candidate_config_path=candidate_config_path,
        embeddings_path=embeddings_path,
        embedding_summary_path=embedding_summary_path,
        student_map_path=student_map_path,
    )
    run_id = make_diagnostic_run_id("embedding_forensics_reviewfix")
    output_dir = Path(
        output_dir
        or repo_root / "attendance_output" / "embedding_forensics" / run_id
    ).resolve()
    if output_dir.exists():
        raise EmbeddingForensicsError(f"Refusing to overwrite forensic output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    skip_names = {
        "suspicious_enrollment_image_review",
        "verified_cctv_enrollment_review",
        "open_enrollment_audit_review.ps1",
        "open_cctv_enrollment_review.ps1",
        "forensic_summary.json",
        "forensic_report.md",
        "forensic_run_manifest.json",
        "output_manifest.json",
        "partial_output_manifest.json",
    }
    try:
        for source in sorted(source_forensic_run.iterdir(), key=lambda path: path.name.lower()):
            if source.name in skip_names:
                continue
            destination = output_dir / source.name
            if source.is_dir():
                shutil.copytree(source, destination)
            elif source.is_file():
                shutil.copy2(source, destination)
        if progress:
            progress(f"Reused integrity-checked forensic analyses from: {source_forensic_run}")

        dataset_audit = DatasetAudit(
            inventory=pd.read_csv(
                output_dir / "dataset_image_inventory.csv", dtype=str, keep_default_na=False
            ),
            quality=pd.read_csv(
                output_dir / "dataset_quality_audit.csv", dtype=str, keep_default_na=False
            ),
            suspects=pd.read_csv(
                output_dir / "dataset_suspect_images.csv", dtype=str, keep_default_na=False
            ),
            duplicates=pd.read_csv(
                output_dir / "dataset_duplicate_files.csv", dtype=str, keep_default_na=False
            ),
            summary=json.loads(
                (output_dir / "dataset_audit_summary.json").read_text(encoding="utf-8")
            ),
        )
        enrollment_review = export_enrollment_audit_review(
            dataset_audit=dataset_audit,
            output_dir=output_dir / "suspicious_enrollment_image_review",
            run_id=run_id,
            student_map_path=Path(student_map_path),
            subject_abbr=subject_abbr,
            package_revision=package_revision,
        )

        cctv_inventory = pd.read_csv(
            output_dir / "verified_cctv_candidate_inventory.csv",
            dtype=str,
            keep_default_na=False,
        )
        old_prefix = str(source_forensic_run)
        new_prefix = str(output_dir)
        cctv_inventory["Candidate_Crop_Path"] = cctv_inventory["Candidate_Crop_Path"].map(
            lambda value: value.replace(old_prefix, new_prefix, 1)
            if str(value).startswith(old_prefix)
            else value
        )
        _write_csv(output_dir / "verified_cctv_candidate_inventory.csv", cctv_inventory)
        selected = cctv_inventory[cctv_inventory["Selected_For_Review"].eq("Yes")].copy()
        cctv_summary = pd.read_csv(
            output_dir / "verified_cctv_selection_summary.csv",
            dtype=str,
            keep_default_na=False,
        )
        old_cctv_package = Path(source_summary["review_packages"]["verified_cctv"])
        old_cctv_metadata = json.loads(
            (old_cctv_package / "private" / "export_metadata.json").read_text(encoding="utf-8")
        )
        cctv_summary.attrs["selection_metadata"] = dict(
            old_cctv_metadata.get("selection_metadata") or {}
        )
        cctv_selection = CCTVSelection(
            inventory=cctv_inventory,
            selected=selected,
            summary=cctv_summary,
        )
        cctv_review = export_verified_cctv_review(
            selection=cctv_selection,
            output_dir=output_dir / "verified_cctv_enrollment_review",
            run_id=run_id,
            benchmark_id=source_summary["benchmark_id"],
            student_map_path=Path(student_map_path),
            subject_abbr=subject_abbr,
            package_revision=package_revision,
        )
        cli_script = repo_root / "scripts" / "validate_tracklet_ground_truth.py"
        enrollment_launcher = write_reviewer_launcher(
            launcher_path=output_dir / "open_enrollment_audit_review.ps1",
            python_executable=Path(sys.executable),
            cli_script=cli_script,
            command="open-enrollment-audit-review",
            review_package=enrollment_review.root,
        )
        cctv_launcher = write_reviewer_launcher(
            launcher_path=output_dir / "open_cctv_enrollment_review.ps1",
            python_executable=Path(sys.executable),
            cli_script=cli_script,
            command="open-cctv-enrollment-review",
            review_package=cctv_review.root,
        )
        summary = dict(source_summary)
        summary.update(
            {
                "forensic_run_id": run_id,
                "output_dir": str(output_dir),
                "review_packages": {
                    "enrollment_audit": str(enrollment_review.root.resolve()),
                    "verified_cctv": str(cctv_review.root.resolve()),
                },
                "reviewer_launchers": {
                    "enrollment_audit": str(enrollment_launcher.resolve()),
                    "verified_cctv": str(cctv_launcher.resolve()),
                },
                "required_approval_csv_filenames": {
                    "enrollment_audit": f"forensic_review_approvals_{enrollment_review.package_id}.csv",
                    "verified_cctv": f"forensic_review_approvals_{cctv_review.package_id}.csv",
                },
                "analysis_reused_without_model_rerun": True,
                "source_forensic_run": str(source_forensic_run),
                "source_forensic_output_manifest_sha256": _sha256_file(
                    source_forensic_run / "output_manifest.json"
                ),
                "superseded_review_packages": source_summary["review_packages"],
                "reviewer_bundle_revision": package_revision,
                "review_package_regeneration_elapsed_seconds": round(
                    time.perf_counter() - started, 2
                ),
            }
        )
        _write_json(output_dir / "forensic_summary.json", summary)
        confusion = ConfusionAnalysis(
            matrix=pd.read_csv(
                output_dir / "prediction_confusion_matrix.csv", dtype=str, keep_default_na=False
            ),
            attractors=pd.read_csv(
                output_dir / "predicted_identity_attractors.csv", dtype=str, keep_default_na=False
            ),
            miss_patterns=pd.read_csv(
                output_dir / "actual_identity_miss_patterns.csv", dtype=str, keep_default_na=False
            ),
            pairs=pd.read_csv(output_dir / "confusion_pairs.csv", dtype=str, keep_default_na=False),
            summary=dict(summary["identity_confusion"]),
        )
        (output_dir / "forensic_report.md").write_text(
            _forensic_report_markdown(summary, confusion), encoding="utf-8"
        )
        source_run_manifest = json.loads(
            (source_forensic_run / "forensic_run_manifest.json").read_text(encoding="utf-8")
        )
        artifact_hashes = {
            path.relative_to(output_dir).as_posix(): _sha256_file(path)
            for path in sorted(output_dir.rglob("*"))
            if path.is_file()
            and path.name not in {"forensic_run_manifest.json", "output_manifest.json"}
        }
        run_manifest = {
            **{
                key: value
                for key, value in source_run_manifest.items()
                if key not in {"forensic_run_id", "created_at", "artifact_sha256"}
            },
            "forensic_run_id": run_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "artifact_sha256": artifact_hashes,
            "analysis_reused_without_model_rerun": True,
            "source_forensic_run": {
                "path": str(source_forensic_run),
                "output_manifest_sha256": _sha256_file(
                    source_forensic_run / "output_manifest.json"
                ),
            },
            "reviewer_bundle_revision": package_revision,
        }
        _write_json(output_dir / "forensic_run_manifest.json", run_manifest)
        output_manifest = write_output_manifest(
            output_dir,
            {
                "forensic_run_id": run_id,
                "benchmark_id": summary["benchmark_id"],
                "candidate_id": EXPECTED_CANDIDATE_ID,
                "candidate_status": REJECTED_CANDIDATE_STATUS,
                "derived_from_forensic_run": str(source_forensic_run),
                "analysis_reused_without_model_rerun": True,
                "reviewer_bundle_revision": package_revision,
                "protected_state_unchanged": True,
                "production_embeddings_changed": False,
                "datasets_changed": False,
                "official_attendance_changed": False,
                "candidate_embedding_built": False,
                "candidate_embedding_promoted": False,
            },
        )
        verify_output_manifest(output_dir, output_manifest)
        if progress:
            progress(f"Regenerated review packages without model rerun: {output_dir}")
        return EmbeddingForensicsRun(
            output_dir=output_dir,
            summary=summary,
            enrollment_review=enrollment_review,
            cctv_review=cctv_review,
            output_manifest=output_manifest,
        )
    except Exception:
        failure_path = output_dir / "forensic_failure.json"
        if not failure_path.exists():
            _write_json(
                failure_path,
                {
                    "schema_version": FORENSIC_SCHEMA_VERSION,
                    "forensic_run_id": run_id,
                    "completed": False,
                    "production_success_claimed": False,
                    "review_package_regeneration_failed": True,
                },
            )
        raise
