from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .diagnostics import json_safe, make_diagnostic_run_id
from .embedding_forensics import (
    CCTV_ACTIONS,
    ENROLLMENT_ACTIONS,
    EmbeddingForensicsError,
    OFFICIAL_ATTENDANCE_CHECKPOINTS,
    OFFICIAL_MARGIN_THRESHOLD,
    OFFICIAL_MATCH_THRESHOLD,
    REJECTED_CANDIDATE_STATUS,
    _roster_name_map,
    compare_protected_state,
    snapshot_protected_state,
    verify_reusable_forensic_run,
)
from .forensic_review import (
    ForensicReviewError,
    ForensicReviewItem,
    ForensicReviewPackage,
    audit_forensic_review_package,
    export_forensic_review_package,
    load_forensic_review_package,
    write_reviewer_launcher,
)
from .shadow_validation import (
    ShadowValidationError,
    verify_output_manifest,
    write_output_manifest,
)
from .tracklet_review import _sha256_file


MAX_COMPACT_ENROLLMENT_ITEMS = 60
MAX_COMPACT_CCTV_ITEMS = 40
DEFAULT_COMPACT_ENROLLMENT_ITEMS = 50
DEFAULT_COMPACT_CCTV_ITEMS = 36
COMPACT_REVIEW_SCHEMA_VERSION = 1
EXPECTED_BROAD_ENROLLMENT_PACKAGE_ID = "enrollment-audit-5430747113d0daeb"
EXPECTED_BROAD_CCTV_PACKAGE_ID = "verified-cctv-77507b2f5d348aad"
EXPECTED_BROAD_ENROLLMENT_COUNT = 855
EXPECTED_BROAD_CCTV_COUNT = 98
REJECTED_CANDIDATE_ID = "cal-5cd35b60dd83"
REQUIRED_RANKING_ARTIFACTS = (
    "dataset_suspect_images.csv",
    "dataset_duplicate_files.csv",
    "embedding_outliers.csv",
    "predicted_identity_attractors.csv",
    "actual_identity_miss_patterns.csv",
    "student_embedding_health.csv",
    "verified_cctv_candidate_inventory.csv",
    "forensic_summary.json",
    "forensic_run_manifest.json",
)


class CompactReviewError(ValueError):
    pass


@dataclass(frozen=True)
class CompactSelection:
    selected: pd.DataFrame
    deferred: pd.DataFrame
    summary: dict[str, Any]


@dataclass(frozen=True)
class ValidatedBroadPackage:
    root: Path
    public: dict[str, Any]
    metadata: dict[str, Any]
    mapping: pd.DataFrame
    manifest_sha256: str
    tree_state: dict[str, Any]


@dataclass(frozen=True)
class CompactSourceValidation:
    source_forensic_run: Path
    source_reviewfix_run: Path
    source_output_manifest_sha256: str
    reviewfix_output_manifest_sha256: str
    ranking_artifact_sha256: dict[str, str]
    enrollment_package: ValidatedBroadPackage
    cctv_package: ValidatedBroadPackage
    protected_state: dict[str, Any]
    production_embeddings_sha256: str
    production_summary_sha256: str
    rejected_candidate_registry_sha256: str
    summary: dict[str, Any]


@dataclass(frozen=True)
class CompactReviewRun:
    output_dir: Path
    enrollment_selection: CompactSelection
    cctv_selection: CompactSelection
    enrollment_review: ForensicReviewPackage
    cctv_review: ForensicReviewPackage
    summary: dict[str, Any]
    output_manifest: Path
    idempotent_reuse: bool = False


def _text(value: Any) -> str:
    return str(value or "").strip()


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _yes(value: Any) -> bool:
    return _text(value).lower() in {"1", "true", "yes", "y"}


def _flags(value: Any) -> set[str]:
    return {
        item.strip()
        for item in _text(value).replace(",", ";").split(";")
        if item.strip()
    }


def _path_key(value: Any) -> str:
    return _text(value).replace("/", "\\").lower()


def _row_value(row: dict[str, Any], key: str, default: Any = "") -> Any:
    value = row.get(key, default)
    return default if value is None else value


def _require_unique_ids(frame: pd.DataFrame, column: str, label: str) -> None:
    if column not in frame.columns:
        raise CompactReviewError(f"{label} is missing required column: {column}")
    values = frame[column].map(_text)
    if values.eq("").any() or values.duplicated().any():
        raise CompactReviewError(f"{label} {column} values must be non-empty and unique")


def _validate_limit(value: int, hard_maximum: int, label: str) -> None:
    if value < 1 or value > hard_maximum:
        raise CompactReviewError(f"{label} maximum must be between 1 and {hard_maximum}")


def _attractor_map(frame: pd.DataFrame) -> tuple[dict[str, dict[str, Any]], set[str]]:
    records: dict[str, dict[str, Any]] = {}
    actual_identities: set[str] = set()
    for row in frame.to_dict("records") if not frame.empty else []:
        roll = _text(row.get("Predicted_Roll"))
        if not roll:
            continue
        actuals = sorted(_flags(row.get("Actual_Identities")))
        actual_identities.update(actuals)
        records[roll] = {
            "false_accepts": _integer(row.get("Confirmed_False_Accepts")),
            "distinct_actuals": _integer(row.get("Distinct_Actual_Identities_Attracted")),
            "outsiders": _integer(row.get("Outsider_Not_In_Mapping_Accepts")),
            "actuals": actuals,
        }
    return records, actual_identities


def _miss_map(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {
        _text(row.get("Actual_Roll")): {
            "missed": _integer(row.get("Missed_Top1")),
            "reviewed": _integer(row.get("Reviewed_Tracks")),
            "accuracy": _number(row.get("Top1_Accuracy"), 1.0),
            "wrong_predictions": sorted(_flags(row.get("Wrong_Predicted_Identities"))),
        }
        for row in frame.to_dict("records") if not frame.empty
        if _text(row.get("Actual_Roll"))
    }


def _match_outlier(source_path: str, outliers: list[dict[str, Any]]) -> dict[str, Any] | None:
    source_key = _path_key(source_path)
    matches = [
        row
        for row in outliers
        if source_key == _path_key(row.get("Image_Path"))
        or source_key.endswith("\\" + _path_key(row.get("Image_Path")).lstrip("\\"))
    ]
    if not matches:
        return None
    return max(matches, key=lambda row: _number(row.get("Robust_Outlier_Score")))


def _enrollment_risk(
    row: dict[str, Any],
    *,
    outlier_rows: list[dict[str, Any]],
    attractor_by_roll: dict[str, dict[str, Any]],
    actual_confusion_rolls: set[str],
    miss_by_roll: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    flags = _flags(row.get("Audit_Flags"))
    roll = _text(row.get("Canonical_Roll"))
    role = _text(row.get("Dataset_Source_Role"))
    own = _number(row.get("Own_Medoid_Similarity"), float("nan"))
    other = _number(row.get("Nearest_Other_Similarity"), float("nan"))
    margin = other - own if math.isfinite(own) and math.isfinite(other) else 0.0
    attractor = attractor_by_roll.get(roll, {})
    miss = miss_by_roll.get(roll, {})
    outlier = _match_outlier(_text(row.get("Source_Path")), outlier_rows)
    folder = _text(row.get("Owning_Folder"))
    folder_mismatch = bool(folder and roll and roll not in folder.upper().replace(" ", ""))
    invalid_ownership = (
        _text(row.get("In_Master_Registry")).lower() == "no"
        or "folder_identity_absent_from_authoritative_registry" in flags
        or folder_mismatch
    )
    closer_other = "closer_to_another_student_than_own_medoid" in flags and margin > 0.0
    multiple_faces = "multiple_faces" in flags or _integer(row.get("Detected_Face_Count")) > 1
    cross_student_duplicate = _yes(row.get("Cross_Student_Duplicate"))
    exact_outlier = outlier is not None
    own_outlier = "own_class_outlier" in flags
    unreadable = not _yes(row.get("Decode_Success"))
    no_face = "no_face" in flags or "no_valid_face" in flags
    tiny_face = "tiny_face" in flags
    severe_quality = bool(flags.intersection({"severe_blur", "severe_exposure_issue"}))
    production_contributor = role == "production_recorded_dataset" and _yes(
        row.get("Valid_SFace_Embedding")
    )

    reasons: list[str] = []
    score = 0.0
    tier = 4

    def add(reason: str, points: float, reason_tier: int) -> None:
        nonlocal score, tier
        reasons.append(reason)
        score += points
        tier = min(tier, reason_tier)

    if cross_student_duplicate:
        add("cross-student duplicate ownership risk", 190.0, 1)
    if invalid_ownership:
        add("folder or authoritative-identity ownership mismatch", 175.0, 1)
    if closer_other:
        add("embedding is closer to another student than its own medoid", 130.0 + min(40.0, margin * 80.0), 1)
    if multiple_faces:
        add("multiple faces detected in one enrollment image", 92.0, 1)
    if attractor:
        add(
            "owning identity is a confirmed reviewed false-accept attractor",
            24.0
            + 11.0 * attractor["false_accepts"]
            + 7.0 * attractor["distinct_actuals"]
            + 7.0 * attractor["outsiders"],
            1,
        )
    if roll in actual_confusion_rolls:
        add("owning identity appears in reviewed confusion pairs", 28.0, 1)
    if exact_outlier:
        add(
            "source image produced one of the strongest stored embedding outliers",
            108.0 + min(60.0, _number(outlier.get("Robust_Outlier_Score")) * 4.0),
            2,
        )
    elif own_outlier:
        add("source embedding is an own-class outlier", 70.0 + max(0.0, 1.0 - own) * 25.0, 2)
    if miss.get("missed", 0):
        add("owning identity is missed in the reviewed benchmark", 8.0 * miss["missed"], 2)
    if unreadable:
        add("source image is unreadable", 70.0, 3)
    if no_face:
        add("no usable face was detected", 58.0, 3)
    if tiny_face and (production_contributor or attractor or exact_outlier):
        add("face is extremely small for enrollment use", 28.0, 3)
    if severe_quality and production_contributor and (attractor or exact_outlier or own_outlier):
        add("severe quality issue affects a contributing high-risk image", 22.0, 3)

    issue_types = sum(
        bool(value)
        for value in (
            cross_student_duplicate,
            invalid_ownership,
            closer_other,
            multiple_faces,
            bool(attractor),
            roll in actual_confusion_rolls,
            exact_outlier or own_outlier,
            unreadable or no_face or tiny_face,
        )
    )
    score += issue_types * 3.0
    if role == "production_recorded_dataset":
        score += 3.0
    eligible = bool(reasons)
    exception_cap = bool(attractor) or issue_types >= 2
    involvement = []
    if attractor:
        involvement.append(
            f"attractor: {attractor['false_accepts']} false accepts, "
            f"{attractor['distinct_actuals']} distinct known identities, {attractor['outsiders']} outsiders"
        )
    if roll in actual_confusion_rolls:
        involvement.append("actual identity in reviewed confusion evidence")
    if miss.get("missed", 0):
        involvement.append(f"reviewed top-1 misses: {miss['missed']}")
    return {
        "eligible": eligible,
        "score": round(score, 6),
        "tier": tier if eligible else 4,
        "reasons": reasons,
        "exception_cap": exception_cap,
        "outlier": exact_outlier,
        "involvement": "; ".join(involvement) or "none confirmed",
    }


def select_compact_enrollment_items(
    broad_items: pd.DataFrame,
    embedding_outliers: pd.DataFrame,
    attractor_rows: pd.DataFrame,
    miss_rows: pd.DataFrame,
    *,
    maximum: int = MAX_COMPACT_ENROLLMENT_ITEMS,
    preferred: int = DEFAULT_COMPACT_ENROLLMENT_ITEMS,
) -> CompactSelection:
    _validate_limit(maximum, MAX_COMPACT_ENROLLMENT_ITEMS, "Enrollment")
    target = min(maximum, max(1, preferred))
    if broad_items.empty:
        return CompactSelection(pd.DataFrame(), pd.DataFrame(), {
            "source_broad_items": 0,
            "selected_items": 0,
            "deferred_items": 0,
            "low_priority_padding_used": False,
            "embedding_outliers_considered": len(embedding_outliers),
            "embedding_outliers_available_in_broad": 0,
            "embedding_outliers_selected": 0,
            "embedding_outliers_unavailable_from_broad": [
                {
                    "canonical_roll": _text(row.get("Canonical_Roll")),
                    "image_path": _text(row.get("Image_Path")),
                    "robust_outlier_score": _number(row.get("Robust_Outlier_Score")),
                    "reason": "not_present_in_validated_broad_enrollment_package",
                }
                for row in embedding_outliers.to_dict("records")
            ],
            "model_similarity_used_as_ground_truth": False,
        })
    _require_unique_ids(broad_items, "Source_Broad_Item_ID", "Enrollment broad review")
    attractor_by_roll, actual_confusion_rolls = _attractor_map(attractor_rows)
    miss_by_roll = _miss_map(miss_rows)
    outlier_records = embedding_outliers.to_dict("records") if not embedding_outliers.empty else []
    unavailable_outliers = [
        {
            "canonical_roll": _text(outlier.get("Canonical_Roll")),
            "image_path": _text(outlier.get("Image_Path")),
            "robust_outlier_score": _number(outlier.get("Robust_Outlier_Score")),
            "reason": "not_present_in_validated_broad_enrollment_package",
        }
        for outlier in outlier_records
        if not any(
            _match_outlier(_text(source.get("Source_Path")), [outlier]) is not None
            for source in broad_items.to_dict("records")
        )
    ]
    candidates: list[dict[str, Any]] = []
    for source in broad_items.to_dict("records"):
        risk = _enrollment_risk(
            source,
            outlier_rows=outlier_records,
            attractor_by_roll=attractor_by_roll,
            actual_confusion_rolls=actual_confusion_rolls,
            miss_by_roll=miss_by_roll,
        )
        candidates.append({**source, "_risk": risk})
    candidates.sort(
        key=lambda row: (
            row["_risk"]["tier"],
            -row["_risk"]["score"],
            _text(row.get("Canonical_Roll")),
            _text(row.get("File_SHA256")),
            _path_key(row.get("Source_Path")),
            _text(row.get("Source_Broad_Item_ID")),
        )
    )
    selected_source: list[dict[str, Any]] = []
    selected_ids_internal: set[str] = set()
    reason_by_id: dict[str, str] = {}
    selected_by_hash: dict[str, str] = {}
    student_counts: dict[str, int] = {}

    def try_select(row: dict[str, Any]) -> None:
        item_id = _text(row.get("Source_Broad_Item_ID"))
        if item_id in selected_ids_internal:
            return
        risk = row["_risk"]
        file_hash = _text(row.get("File_SHA256"))
        roll = _text(row.get("Canonical_Roll"))
        if file_hash in selected_by_hash:
            reason_by_id[item_id] = "exact_source_hash_redundant"
            return
        if not risk["eligible"]:
            reason_by_id[item_id] = "low_priority_quality_only"
            return
        cap = 5 if risk["exception_cap"] else 3
        if student_counts.get(roll, 0) >= cap:
            reason_by_id[item_id] = "per_student_selection_cap"
            return
        if len(selected_source) >= target:
            reason_by_id[item_id] = "compact_target_reached"
            return
        selected_source.append(row)
        selected_ids_internal.add(item_id)
        selected_by_hash[file_hash] = item_id
        student_counts[roll] = student_counts.get(roll, 0) + 1

    for row in candidates:
        if row["_risk"]["outlier"]:
            try_select(row)
    for row in candidates:
        try_select(row)
    selected_source.sort(
        key=lambda row: (
            row["_risk"]["tier"],
            -row["_risk"]["score"],
            _text(row.get("Canonical_Roll")),
            _text(row.get("File_SHA256")),
            _text(row.get("Source_Broad_Item_ID")),
        )
    )

    selected_ids = {_text(row.get("Source_Broad_Item_ID")) for row in selected_source}
    selected_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(selected_source, start=1):
        risk = row["_risk"]
        reasons = list(risk["reasons"])
        selected_rows.append(
            {
                "Compact_Rank": rank,
                "Source_Broad_Item_ID": _text(row.get("Source_Broad_Item_ID")),
                "Source_File": _text(row.get("Source_Path")),
                "Source_File_SHA256": _text(row.get("File_SHA256")),
                "Expected_Roll": _text(row.get("Canonical_Roll")),
                "Expected_Name": _text(row.get("Expected_Name")),
                "Priority_Tier": f"Tier {risk['tier']}",
                "Issue_Flags": _text(row.get("Audit_Flags")),
                "Primary_Selection_Reason": reasons[0] if reasons else "",
                "Secondary_Reasons": "; ".join(reasons[1:]),
                "Deterministic_Risk_Score": risk["score"],
                "Student_Level_Selection_Count": student_counts.get(
                    _text(row.get("Canonical_Roll")), 0
                ),
                "Duplicate_Group_ID": _text(row.get("Duplicate_Group_ID")),
                "Attractor_Confusion_Involvement": risk["involvement"],
                "Embedding_Outlier": "Yes" if risk["outlier"] else "No",
                "Dataset_Source_Role": _text(row.get("Dataset_Source_Role")),
                "Dataset_Folder": _text(row.get("Owning_Folder")),
                "Source_Relative_Path": _text(row.get("Relative_Path")),
                "Detected_Face_Count": _text(row.get("Detected_Face_Count")),
                "Valid_SFace_Embedding": _text(row.get("Valid_SFace_Embedding")),
                "Own_Medoid_Similarity": _text(row.get("Own_Medoid_Similarity")),
                "Nearest_Other_Roll": _text(row.get("Nearest_Other_Roll")),
                "Nearest_Other_Similarity": _text(row.get("Nearest_Other_Similarity")),
                "Model_Similarity_Hint_Not_Ground_Truth": _text(
                    row.get("Model_Similarity_Hint_Not_Ground_Truth")
                ),
                "Blur_Laplacian_Variance": _text(row.get("Blur_Laplacian_Variance")),
                "Brightness_Mean": _text(row.get("Brightness_Mean")),
                "Source_Forensic_Provenance": (
                    "Phase 1.2I-A dataset_suspect_images.csv and integrity-checked broad enrollment package"
                ),
            }
        )

    deferred_rows: list[dict[str, Any]] = []
    for row in broad_items.to_dict("records"):
        item_id = _text(row.get("Source_Broad_Item_ID"))
        if item_id in selected_ids:
            continue
        file_hash = _text(row.get("File_SHA256"))
        related = selected_by_hash.get(file_hash, "")
        risk = next(
            candidate["_risk"]
            for candidate in candidates
            if _text(candidate.get("Source_Broad_Item_ID")) == item_id
        )
        deferred_rows.append(
            {
                "Source_Broad_Item_ID": item_id,
                "Source_File": _text(row.get("Source_Path")),
                "Source_File_SHA256": file_hash,
                "Expected_Roll": _text(row.get("Canonical_Roll")),
                "Deferred_Reason": reason_by_id.get(item_id, "not_selected"),
                "Priority_Tier": f"Tier {risk['tier']}" if risk["eligible"] else "Deferred low priority",
                "Eligible_For_Later_Review": "Yes",
                "Redundant_With_Compact_Selected_Item": "Yes" if related else "No",
                "Related_Selected_Item_ID": related,
                "Deterministic_Risk_Score": risk["score"],
                "Issue_Flags": _text(row.get("Audit_Flags")),
            }
        )
    selected = pd.DataFrame(selected_rows)
    deferred = pd.DataFrame(deferred_rows).sort_values(
        ["Deterministic_Risk_Score", "Source_Broad_Item_ID"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True) if deferred_rows else pd.DataFrame()
    summary = {
        "source_broad_items": int(len(broad_items)),
        "selected_items": int(len(selected)),
        "deferred_items": int(len(deferred)),
        "preferred_items": int(preferred),
        "hard_maximum": int(maximum),
        "selected_students": sorted(student_counts),
        "per_student_counts": dict(sorted(student_counts.items())),
        "embedding_outliers_considered": int(len(embedding_outliers)),
        "embedding_outliers_available_in_broad": int(
            len(embedding_outliers) - len(unavailable_outliers)
        ),
        "embedding_outliers_selected": int(selected["Embedding_Outlier"].eq("Yes").sum()) if len(selected) else 0,
        "embedding_outliers_unavailable_from_broad": unavailable_outliers,
        "duplicate_source_hashes_selected": int(selected["Source_File_SHA256"].duplicated().sum()) if len(selected) else 0,
        "low_priority_padding_used": False,
        "model_similarity_used_as_ground_truth": False,
    }
    if len(selected) + len(deferred) != len(broad_items):
        raise CompactReviewError("Enrollment selected/deferred accounting is incomplete")
    return CompactSelection(selected, deferred, summary)


def _embedding_health_map(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {
        _text(row.get("Canonical_Roll")): row
        for row in frame.to_dict("records") if not frame.empty
        if _text(row.get("Canonical_Roll"))
    }


def _cctv_priority(
    row: dict[str, Any],
    attractors: dict[str, dict[str, Any]],
    misses_by_roll: dict[str, dict[str, Any]],
    health_by_roll: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    roll = _text(row.get("Actual_Roll"))
    attractor = attractors.get(roll, {})
    miss = misses_by_roll.get(roll, {})
    health = health_by_roll.get(roll, {})
    reasons: list[str] = []
    score = 0.0
    if attractor:
        reasons.append("identity is a confirmed reviewed false-accept attractor")
        score += (
            16.0 * attractor["false_accepts"]
            + 9.0 * attractor["distinct_actuals"]
            + 8.0 * attractor["outsiders"]
        )
    if miss.get("missed", 0):
        reasons.append(f"identity has {miss['missed']} reviewed top-1 miss(es)")
        score += 18.0 * miss["missed"] + max(0.0, 1.0 - miss["accuracy"]) * 20.0
    embedding_count = _integer(health.get("Valid_Embedding_Count"), 999)
    if embedding_count < 10:
        reasons.append(f"current enrollment has only {embedding_count} valid stored embeddings")
        score += max(0, 10 - embedding_count) * 3.0
    if _integer(health.get("Robust_Outlier_Count")):
        reasons.append("current enrollment contains robust embedding outliers")
        score += 8.0 * _integer(health.get("Robust_Outlier_Count"))
    if _yes(health.get("Frequently_Missed_In_Reviewed_CCTV")):
        reasons.append("embedding health audit marks the identity frequently missed")
        score += 20.0
    quality = _number(row.get("Selection_Quality_Score"))
    detector = _number(row.get("Detector_Score"))
    nearest = _number(row.get("Diversity_Nearest_Selected_Cosine"), 0.0)
    score += quality * 12.0 + detector * 4.0
    score += 5.0 if _yes(row.get("Diversity_New_Checkpoint")) else 0.0
    score += 4.0 if _yes(row.get("Diversity_New_Camera")) else 0.0
    score -= max(0.0, nearest - 0.65) * 55.0
    if not reasons:
        reasons.append("strong human-confirmed CCTV-domain evidence")
    return {
        "score": round(score, 6),
        "reasons": reasons,
        "high_priority": bool(attractor) or bool(miss.get("missed", 0)),
        "priority_text": "; ".join(reasons),
    }


def _cctv_ineligible_reason(row: dict[str, Any]) -> str:
    status = _text(row.get("Actual_Status") or "identified").lower()
    if status not in {"identified", "correct_identity_known", "different_identity"}:
        return f"excluded_actual_status_{status or 'missing'}"
    if not _yes(row.get("Human_Eligible")):
        return "not_human_eligible"
    if _text(row.get("Identity_Source")) != "human_review_actual_roll":
        return "identity_not_controlled_by_human_actual_roll"
    if _yes(row.get("Model_Prediction_Used_As_Identity")):
        return "model_prediction_used_as_identity"
    if not _text(row.get("Actual_Roll")):
        return "missing_actual_roll"
    if _text(row.get("Selected_For_Review")) and not _yes(row.get("Selected_For_Review")):
        return "not_in_source_broad_review"
    if not _text(row.get("Candidate_Crop_Path")) or not _text(row.get("Candidate_Crop_SHA256")):
        return "broken_evidence_reference"
    verdict = _text(row.get("Review_Verdict")).lower()
    if verdict in {"uncertain", "mixed", "mixed_track", "unidentifiable", "not_in_mapping", "duplicate"}:
        return f"excluded_review_verdict_{verdict}"
    return ""


def select_compact_cctv_items(
    broad_items: pd.DataFrame,
    attractor_rows: pd.DataFrame,
    miss_rows: pd.DataFrame,
    embedding_health: pd.DataFrame,
    *,
    maximum: int = MAX_COMPACT_CCTV_ITEMS,
    preferred: int = DEFAULT_COMPACT_CCTV_ITEMS,
    authoritative_rolls: set[str] | None = None,
) -> CompactSelection:
    _validate_limit(maximum, MAX_COMPACT_CCTV_ITEMS, "CCTV")
    target = min(maximum, max(1, preferred))
    if broad_items.empty:
        return CompactSelection(pd.DataFrame(), pd.DataFrame(), {
            "source_broad_items": 0,
            "selected_items": 0,
            "deferred_items": 0,
            "model_prediction_used_as_identity": False,
        })
    _require_unique_ids(broad_items, "Source_Broad_Item_ID", "CCTV broad review")
    attractor_by_roll, _ = _attractor_map(attractor_rows)
    misses_by_roll = _miss_map(miss_rows)
    health_by_roll = _embedding_health_map(embedding_health)
    authoritative = set(authoritative_rolls or ())
    candidates: list[dict[str, Any]] = []
    ineligible: dict[str, str] = {}
    for source in broad_items.to_dict("records"):
        item_id = _text(source.get("Source_Broad_Item_ID"))
        reason = _cctv_ineligible_reason(source)
        if not reason and authoritative and _text(source.get("Actual_Roll")) not in authoritative:
            reason = "actual_roll_not_in_authoritative_roster"
        if reason:
            ineligible[item_id] = reason
            continue
        priority = _cctv_priority(source, attractor_by_roll, misses_by_roll, health_by_roll)
        candidates.append({**source, "_priority": priority})

    candidates.sort(
        key=lambda row: (
            -row["_priority"]["score"],
            _text(row.get("Actual_Roll")),
            _text(row.get("Checkpoint_ID")),
            _text(row.get("Camera_ID")),
            _text(row.get("Tracklet_ID")),
            _text(row.get("Source_Broad_Item_ID")),
        )
    )
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_hashes: dict[str, str] = {}
    selected_tracks: dict[str, str] = {}
    student_counts: dict[str, int] = {}
    student_checkpoints: dict[str, set[str]] = {}
    student_cameras: dict[str, set[str]] = {}
    remaining = list(candidates)
    while remaining and len(selected) < target:
        ranked: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        for row in remaining:
            roll = _text(row.get("Actual_Roll"))
            cap = 4 if row["_priority"]["high_priority"] else 3
            if student_counts.get(roll, 0) >= cap:
                continue
            checkpoint = _text(row.get("Checkpoint_ID"))
            camera = _text(row.get("Camera_ID"))
            dynamic = row["_priority"]["score"] - student_counts.get(roll, 0) * 22.0
            dynamic += 17.0 if checkpoint not in student_checkpoints.get(roll, set()) else -5.0
            dynamic += 11.0 if camera not in student_cameras.get(roll, set()) else -3.0
            nearest = _number(row.get("Diversity_Nearest_Selected_Cosine"), 0.0)
            dynamic -= max(0.0, nearest - 0.65) * 70.0
            ranked.append(
                (
                    (
                        -round(dynamic, 6),
                        _text(row.get("Actual_Roll")),
                        _text(row.get("Checkpoint_ID")),
                        _text(row.get("Camera_ID")),
                        _text(row.get("Source_Broad_Item_ID")),
                    ),
                    row,
                )
            )
        if not ranked:
            break
        _, chosen = min(ranked, key=lambda entry: entry[0])
        remaining.remove(chosen)
        item_id = _text(chosen.get("Source_Broad_Item_ID"))
        evidence_hash = _text(chosen.get("Candidate_Crop_SHA256"))
        track = _text(chosen.get("Tracklet_ID"))
        if evidence_hash in selected_hashes or track in selected_tracks:
            continue
        roll = _text(chosen.get("Actual_Roll"))
        selected.append(chosen)
        selected_ids.add(item_id)
        selected_hashes[evidence_hash] = item_id
        selected_tracks[track] = item_id
        student_counts[roll] = student_counts.get(roll, 0) + 1
        student_checkpoints.setdefault(roll, set()).add(_text(chosen.get("Checkpoint_ID")))
        student_cameras.setdefault(roll, set()).add(_text(chosen.get("Camera_ID")))

    selected_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(selected, start=1):
        priority = row["_priority"]
        selected_rows.append(
            {
                "Compact_Rank": rank,
                "Source_Broad_Item_ID": _text(row.get("Source_Broad_Item_ID")),
                "Actual_Roll": _text(row.get("Actual_Roll")),
                "Student_Name": _text(row.get("Actual_Student_Name")),
                "Identity_Source": "human_review_actual_roll",
                "Source_Session": _text(row.get("Session_ID")),
                "Checkpoint": _text(row.get("Checkpoint_ID")),
                "Camera": _text(row.get("Camera_ID")),
                "Track_ID": _text(row.get("Tracklet_ID")),
                "Observation_ID": _text(row.get("Observation_ID")),
                "Source_Broad_Package_ID": _text(
                    row.get("Broad_Package_ID") or row.get("Package_ID")
                ),
                "Crop_Evidence_Path": _text(row.get("Candidate_Crop_Path")),
                "Evidence_SHA256": _text(row.get("Candidate_Crop_SHA256")),
                "Crop_Width": _text(row.get("Crop_Width")),
                "Crop_Height": _text(row.get("Crop_Height")),
                "Detector_Score": _text(row.get("Detector_Score")),
                "Blur_Laplacian_Variance": _text(row.get("Crop_Blur_Laplacian_Variance")),
                "Brightness_Mean": _text(row.get("Crop_Brightness_Mean")),
                "Pose_Roll_Degrees": _text(row.get("Pose_Roll_Degrees")),
                "Pose_Yaw_Proxy": _text(row.get("Pose_Yaw_Proxy")),
                "Occlusion_Proxy": _text(row.get("Occlusion_Proxy")),
                "Selection_Quality_Score": _text(row.get("Selection_Quality_Score")),
                "Nearest_Selected_Crop_Similarity": _text(
                    row.get("Diversity_Nearest_Selected_Cosine")
                ),
                "Diversity_New_Checkpoint": _text(row.get("Diversity_New_Checkpoint")),
                "Diversity_New_Camera": _text(row.get("Diversity_New_Camera")),
                "Primary_Selection_Reason": priority["reasons"][0],
                "Confusion_Miss_Priority": priority["priority_text"],
                "Deterministic_Risk_Score": priority["score"],
                "Student_Level_Selection_Count": student_counts.get(
                    _text(row.get("Actual_Roll")), 0
                ),
                "Source_Forensic_Provenance": (
                    "Phase 1.2I-A verified_cctv_candidate_inventory.csv and human Actual_Roll"
                ),
            }
        )
    selected_frame = pd.DataFrame(selected_rows)
    deferred_rows: list[dict[str, Any]] = []
    for row in broad_items.to_dict("records"):
        item_id = _text(row.get("Source_Broad_Item_ID"))
        if item_id in selected_ids:
            continue
        evidence_hash = _text(row.get("Candidate_Crop_SHA256"))
        track = _text(row.get("Tracklet_ID"))
        related = selected_hashes.get(evidence_hash) or selected_tracks.get(track) or ""
        if item_id in ineligible:
            reason = ineligible[item_id]
            eligible_later = "No"
        elif evidence_hash in selected_hashes:
            reason = "duplicate_crop_evidence"
            eligible_later = "Yes"
        elif track in selected_tracks:
            reason = "same_track_redundant"
            eligible_later = "Yes"
        elif student_counts.get(_text(row.get("Actual_Roll")), 0) >= (
            4 if _cctv_priority(row, attractor_by_roll, misses_by_roll, health_by_roll)["high_priority"] else 3
        ):
            reason = "per_student_crop_cap"
            eligible_later = "Yes"
        else:
            reason = "compact_target_reached"
            eligible_later = "Yes"
        deferred_rows.append(
            {
                "Source_Broad_Item_ID": item_id,
                "Actual_Roll": _text(row.get("Actual_Roll")),
                "Crop_Evidence_Path": _text(row.get("Candidate_Crop_Path")),
                "Evidence_SHA256": evidence_hash,
                "Deferred_Reason": reason,
                "Priority_Tier": "Excluded" if item_id in ineligible else "Deferred eligible",
                "Eligible_For_Later_Review": eligible_later,
                "Redundant_With_Compact_Selected_Item": "Yes" if related else "No",
                "Related_Selected_Item_ID": related,
            }
        )
    deferred_frame = pd.DataFrame(deferred_rows).sort_values(
        ["Actual_Roll", "Source_Broad_Item_ID"], kind="stable"
    ).reset_index(drop=True) if deferred_rows else pd.DataFrame()
    summary = {
        "source_broad_items": int(len(broad_items)),
        "selected_items": int(len(selected_frame)),
        "deferred_items": int(len(deferred_frame)),
        "preferred_items": int(preferred),
        "hard_maximum": int(maximum),
        "selected_students": sorted(student_counts),
        "per_student_counts": dict(sorted(student_counts.items())),
        "selected_checkpoints": {
            roll: sorted(values) for roll, values in sorted(student_checkpoints.items())
        },
        "selected_cameras": {
            roll: sorted(values) for roll, values in sorted(student_cameras.items())
        },
        "model_prediction_used_as_identity": False,
        "identity_source": "human_review_actual_roll",
    }
    if len(selected_frame) + len(deferred_frame) != len(broad_items):
        raise CompactReviewError("CCTV selected/deferred accounting is incomplete")
    return CompactSelection(selected_frame, deferred_frame, summary)


def _stable_digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            json_safe(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(json_safe(payload), indent=2, ensure_ascii=True), encoding="utf-8"
    )


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _tree_state(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    if not root.is_dir():
        raise CompactReviewError(f"Directory not found for integrity snapshot: {root}")
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    return {
        "file_count": len(rows),
        "total_bytes": int(sum(row["size_bytes"] for row in rows)),
        "aggregate_sha256": _stable_digest(rows),
    }


def validate_hashed_source_artifacts(
    root: Path,
    required_relative_paths: tuple[str, ...] | list[str] = (),
) -> dict[str, Any]:
    root = Path(root).resolve()
    if not root.is_dir():
        raise CompactReviewError(f"Source forensic run not found: {root}")
    manifest_path = root / "output_manifest.json"
    try:
        result = verify_output_manifest(root, manifest_path)
    except (ShadowValidationError, OSError, json.JSONDecodeError) as exc:
        raise CompactReviewError(str(exc)) from exc
    manifest = dict(result["manifest"])
    listed = dict(manifest.get("files_sha256") or {})
    missing = [relative for relative in required_relative_paths if relative not in listed]
    if missing:
        raise CompactReviewError(
            "Source output manifest does not cover required compact-ranking artifacts: "
            + ", ".join(missing)
        )
    hashes = {
        relative: _sha256_file(root / relative)
        for relative in required_relative_paths
    }
    return {
        "root": root,
        "manifest": manifest,
        "manifest_sha256": _sha256_file(manifest_path),
        "verified_files": int(result["verified_files"]),
        "required_artifact_sha256": hashes,
    }


def validate_broad_review_package(
    package_dir: Path,
    *,
    expected_package_id: str,
    expected_count: int,
    expected_kind: str,
) -> ValidatedBroadPackage:
    package_dir = Path(package_dir).resolve()
    if not package_dir.is_dir():
        raise CompactReviewError(f"Broad review package not found: {package_dir}")
    try:
        public, metadata, mapping = load_forensic_review_package(package_dir)
        privacy = audit_forensic_review_package(package_dir)
    except ForensicReviewError as exc:
        raise CompactReviewError(str(exc)) from exc
    package_id = _text(public.get("package_id"))
    package_kind = _text(public.get("package_kind"))
    item_count = _integer(public.get("item_count"), -1)
    if package_id != expected_package_id:
        raise CompactReviewError(
            f"Wrong broad-package ID: expected {expected_package_id}, found {package_id}"
        )
    if package_kind != expected_kind:
        raise CompactReviewError(
            f"Wrong broad-package kind: expected {expected_kind}, found {package_kind}"
        )
    if item_count != expected_count or len(mapping) != expected_count:
        raise CompactReviewError(
            f"Wrong broad-package count for {package_id}: expected {expected_count}, "
            f"found public={item_count}, private={len(mapping)}"
        )
    if privacy.get("privacy_passed") is not True:
        raise CompactReviewError(f"Broad review package privacy audit failed: {package_dir}")
    return ValidatedBroadPackage(
        root=package_dir,
        public=public,
        metadata=metadata,
        mapping=mapping,
        manifest_sha256=_sha256_file(package_dir / "package_manifest.json"),
        tree_state=_tree_state(package_dir),
    )


def _assert_clean_forensic_summary(summary: dict[str, Any]) -> None:
    expected_false = {
        "production_candidate_enabled": False,
        "production_candidate_approved": False,
        "current_embeddings_modified": False,
        "current_datasets_modified": False,
        "official_thresholds_modified": False,
        "attendance_rule_modified": False,
        "official_attendance_modified": False,
        "candidate_embedding_version_built": False,
        "candidate_embedding_version_promoted": False,
        "mon_p3_processed": False,
    }
    failures = [key for key, expected in expected_false.items() if summary.get(key) is not expected]
    if failures:
        raise CompactReviewError(
            "Source forensic safety state is not clean: " + ", ".join(failures)
        )
    if summary.get("protected_state_unchanged") is not True:
        raise CompactReviewError("Source forensic protected state is not unchanged")
    if (
        _text(summary.get("candidate_id")) != REJECTED_CANDIDATE_ID
        or _text(summary.get("candidate_status")) != REJECTED_CANDIDATE_STATUS
        or _text(summary.get("candidate_final_decision")) != "reject_candidate"
    ):
        raise CompactReviewError("Rejected candidate identity/status does not match Phase 1.2I-A")
    if (
        _number(summary.get("official_match_threshold"), -1.0) != OFFICIAL_MATCH_THRESHOLD
        or _number(summary.get("official_margin_threshold"), -1.0) != OFFICIAL_MARGIN_THRESHOLD
        or _integer(summary.get("official_attendance_checkpoints"), -1)
        != OFFICIAL_ATTENDANCE_CHECKPOINTS
    ):
        raise CompactReviewError("Source forensic output does not preserve official thresholds/rules")


def validate_compact_review_sources(
    *,
    repo_root: Path,
    source_forensic_run: Path,
    source_reviewfix_run: Path,
    candidate_config_path: Path,
    embeddings_path: Path,
    embedding_summary_path: Path,
    student_map_path: Path,
    candidate_registry_path: Path,
    expected_enrollment_package_id: str = EXPECTED_BROAD_ENROLLMENT_PACKAGE_ID,
    expected_cctv_package_id: str = EXPECTED_BROAD_CCTV_PACKAGE_ID,
    expected_enrollment_count: int = EXPECTED_BROAD_ENROLLMENT_COUNT,
    expected_cctv_count: int = EXPECTED_BROAD_CCTV_COUNT,
) -> CompactSourceValidation:
    repo_root = Path(repo_root).resolve()
    source_forensic_run = Path(source_forensic_run).resolve()
    source_reviewfix_run = Path(source_reviewfix_run).resolve()
    candidate_config_path = Path(candidate_config_path).resolve()
    embeddings_path = Path(embeddings_path).resolve()
    embedding_summary_path = Path(embedding_summary_path).resolve()
    student_map_path = Path(student_map_path).resolve()
    candidate_registry_path = Path(candidate_registry_path).resolve()

    source_validation = validate_hashed_source_artifacts(source_forensic_run)
    reviewfix_validation = validate_hashed_source_artifacts(
        source_reviewfix_run,
        [
            *REQUIRED_RANKING_ARTIFACTS,
            "suspicious_enrollment_image_review/package_manifest.json",
            "verified_cctv_enrollment_review/package_manifest.json",
        ],
    )
    source_manifest = source_validation["manifest"]
    reviewfix_manifest = reviewfix_validation["manifest"]
    derived_from = Path(_text(reviewfix_manifest.get("derived_from_forensic_run"))).resolve()
    if derived_from != source_forensic_run:
        raise CompactReviewError(
            "Reviewfix output does not derive from the requested primary forensic run"
        )
    if reviewfix_manifest.get("analysis_reused_without_model_rerun") is not True:
        raise CompactReviewError("Reviewfix output does not prove model-free analysis reuse")
    if _text(source_manifest.get("benchmark_id")) != _text(reviewfix_manifest.get("benchmark_id")):
        raise CompactReviewError("Primary and reviewfix benchmark IDs do not match")

    summary_path = source_reviewfix_run / "forensic_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _assert_clean_forensic_summary(summary)
    expected_enrollment_path = (
        source_reviewfix_run / "suspicious_enrollment_image_review"
    ).resolve()
    expected_cctv_path = (
        source_reviewfix_run / "verified_cctv_enrollment_review"
    ).resolve()
    summary_packages = dict(summary.get("review_packages") or {})
    if (
        Path(_text(summary_packages.get("enrollment_audit"))).resolve()
        != expected_enrollment_path
        or Path(_text(summary_packages.get("verified_cctv"))).resolve()
        != expected_cctv_path
    ):
        raise CompactReviewError("Reviewfix summary points to unexpected broad package paths")

    enrollment_package = validate_broad_review_package(
        expected_enrollment_path,
        expected_package_id=expected_enrollment_package_id,
        expected_count=expected_enrollment_count,
        expected_kind="enrollment_audit",
    )
    cctv_package = validate_broad_review_package(
        expected_cctv_path,
        expected_package_id=expected_cctv_package_id,
        expected_count=expected_cctv_count,
        expected_kind="verified_cctv_enrollment",
    )

    if not candidate_registry_path.is_file():
        raise CompactReviewError(
            f"Rejected candidate registry record not found: {candidate_registry_path}"
        )
    registry = json.loads(candidate_registry_path.read_text(encoding="utf-8"))
    if (
        _text(registry.get("candidate_id")) != REJECTED_CANDIDATE_ID
        or _text(registry.get("status")) != REJECTED_CANDIDATE_STATUS
        or registry.get("production_approved") is not False
        or registry.get("enabled") is not False
        or registry.get("immutable") is not True
    ):
        raise CompactReviewError("Rejected candidate registry safety contract failed")
    versions_root = repo_root / "models" / "versions"
    if versions_root.exists() and any(versions_root.iterdir()):
        raise CompactReviewError("Candidate embedding versions exist; compact phase requires none")

    try:
        verify_reusable_forensic_run(
            output_dir=source_reviewfix_run,
            repo_root=repo_root,
            candidate_config_path=candidate_config_path,
            embeddings_path=embeddings_path,
            embedding_summary_path=embedding_summary_path,
            student_map_path=student_map_path,
        )
    except EmbeddingForensicsError as exc:
        raise CompactReviewError(str(exc)) from exc
    run_manifest = json.loads(
        (source_reviewfix_run / "forensic_run_manifest.json").read_text(encoding="utf-8")
    )
    protected_state = dict(run_manifest.get("protected_state_after") or {})
    if not protected_state:
        raise CompactReviewError("Reviewfix forensic manifest lacks protected-state evidence")

    return CompactSourceValidation(
        source_forensic_run=source_forensic_run,
        source_reviewfix_run=source_reviewfix_run,
        source_output_manifest_sha256=source_validation["manifest_sha256"],
        reviewfix_output_manifest_sha256=reviewfix_validation["manifest_sha256"],
        ranking_artifact_sha256=reviewfix_validation["required_artifact_sha256"],
        enrollment_package=enrollment_package,
        cctv_package=cctv_package,
        protected_state=protected_state,
        production_embeddings_sha256=_sha256_file(embeddings_path),
        production_summary_sha256=_sha256_file(embedding_summary_path),
        rejected_candidate_registry_sha256=_sha256_file(candidate_registry_path),
        summary={
            "source_primary_verified_files": source_validation["verified_files"],
            "source_reviewfix_verified_files": reviewfix_validation["verified_files"],
            "source_broad_enrollment_count": expected_enrollment_count,
            "source_broad_cctv_count": expected_cctv_count,
            "source_broad_enrollment_package_id": expected_enrollment_package_id,
            "source_broad_cctv_package_id": expected_cctv_package_id,
            "candidate_id": REJECTED_CANDIDATE_ID,
            "candidate_status": REJECTED_CANDIDATE_STATUS,
            "candidate_embedding_built": False,
            "candidate_embedding_promoted": False,
            "protected_sources_match": True,
        },
    )


def _source_join_key(path: Any, sha256: Any) -> str:
    return f"{_path_key(path)}|{_text(sha256).lower()}"


def _load_enrollment_broad_items(
    validation: CompactSourceValidation,
    *,
    student_map_path: Path,
    subject_abbr: str,
) -> pd.DataFrame:
    suspects = pd.read_csv(
        validation.source_reviewfix_run / "dataset_suspect_images.csv",
        dtype=str,
        keep_default_na=False,
    )
    duplicates = pd.read_csv(
        validation.source_reviewfix_run / "dataset_duplicate_files.csv",
        dtype=str,
        keep_default_na=False,
    )
    mapping = validation.enrollment_package.mapping.copy()
    mapping["_join_key"] = [
        _source_join_key(path, sha)
        for path, sha in zip(mapping["Source_Path"], mapping["Source_SHA256"])
    ]
    suspects["_join_key"] = [
        _source_join_key(path, sha)
        for path, sha in zip(suspects["Source_Path"], suspects["File_SHA256"])
    ]
    if mapping["_join_key"].duplicated().any() or suspects["_join_key"].duplicated().any():
        raise CompactReviewError("Enrollment source package contains duplicate path/hash identities")
    merged = suspects.merge(
        mapping[["Item_ID", "Package_ID", "_join_key"]],
        on="_join_key",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(mapping) or len(merged) != len(suspects):
        raise CompactReviewError(
            "Broad enrollment reviewer and forensic suspect inventory do not match one-to-one"
        )
    if not duplicates.empty:
        duplicate_lookup = duplicates.copy()
        duplicate_lookup["_join_key"] = [
            _source_join_key(path, sha)
            for path, sha in zip(
                duplicate_lookup["Source_Path"], duplicate_lookup["File_SHA256"]
            )
        ]
        duplicate_lookup = duplicate_lookup.sort_values(
            ["_join_key", "Duplicate_Group_ID"], kind="stable"
        ).drop_duplicates("_join_key", keep="first")
        merged = merged.merge(
            duplicate_lookup[
                [
                    "_join_key",
                    "Cross_Student_Duplicate",
                    "Cross_Dataset_Role_Duplicate",
                    "Distinct_Student_Count",
                ]
            ],
            on="_join_key",
            how="left",
            validate="one_to_one",
        )
    else:
        merged["Cross_Student_Duplicate"] = "No"
        merged["Cross_Dataset_Role_Duplicate"] = "No"
        merged["Distinct_Student_Count"] = "0"
    names = _roster_name_map(Path(student_map_path), subject_abbr)
    merged["Source_Broad_Item_ID"] = merged["Item_ID"]
    merged["Broad_Package_ID"] = validation.enrollment_package.public["package_id"]
    merged["Expected_Name"] = merged["Canonical_Roll"].map(names).fillna("")
    return merged.drop(columns=["_join_key"])


def _load_cctv_broad_items(
    validation: CompactSourceValidation,
) -> pd.DataFrame:
    inventory = pd.read_csv(
        validation.source_reviewfix_run / "verified_cctv_candidate_inventory.csv",
        dtype=str,
        keep_default_na=False,
    )
    inventory = inventory[inventory["Selected_For_Review"].eq("Yes")].copy()
    mapping = validation.cctv_package.mapping.copy()
    if len(inventory) != len(mapping):
        raise CompactReviewError(
            f"Broad CCTV inventory/package count mismatch: {len(inventory)} vs {len(mapping)}"
        )
    if set(inventory["Item_ID"]) != set(mapping["Item_ID"]):
        raise CompactReviewError("Broad CCTV reviewer and candidate inventory item IDs differ")
    source_by_item = mapping.set_index("Item_ID").to_dict("index")
    failures: list[str] = []
    for row in inventory.to_dict("records"):
        source = source_by_item[_text(row.get("Item_ID"))]
        if (
            _path_key(source.get("Source_Path")) != _path_key(row.get("Candidate_Crop_Path"))
            or _text(source.get("Source_SHA256")) != _text(row.get("Candidate_Crop_SHA256"))
        ):
            failures.append(_text(row.get("Item_ID")))
    if failures:
        raise CompactReviewError(
            "Broad CCTV source provenance does not match inventory: " + ", ".join(failures[:10])
        )
    inventory["Source_Broad_Item_ID"] = inventory["Item_ID"]
    inventory["Broad_Package_ID"] = validation.cctv_package.public["package_id"]
    inventory["Actual_Status"] = "identified"
    return inventory


def _selection_fingerprint(
    *,
    package_kind: str,
    source_package_id: str,
    source_manifest_sha256: str,
    package_revision: str,
    selected: pd.DataFrame,
) -> str:
    records = []
    for row in selected.to_dict("records"):
        records.append(
            {
                key: row.get(key)
                for key in (
                    "Compact_Rank",
                    "Source_Broad_Item_ID",
                    "Source_File_SHA256",
                    "Evidence_SHA256",
                    "Expected_Roll",
                    "Actual_Roll",
                    "Deterministic_Risk_Score",
                )
                if key in row
            }
        )
    return _stable_digest(
        {
            "schema_version": COMPACT_REVIEW_SCHEMA_VERSION,
            "package_kind": package_kind,
            "source_package_id": source_package_id,
            "source_manifest_sha256": source_manifest_sha256,
            "package_revision": package_revision,
            "selected": records,
        }
    )


def _compact_item_id(prefix: str, source_package_id: str, source_item_id: str) -> str:
    return f"{prefix}-{_stable_digest([source_package_id, source_item_id])[:18]}"


def _quality_warning_text(issue_flags: Any) -> str:
    quality_flags = {
        "severe_blur",
        "severe_exposure_issue",
        "tiny_face",
        "no_face",
        "no_valid_face",
        "multiple_faces",
    }
    return "; ".join(sorted(_flags(issue_flags).intersection(quality_flags))) or "none"


def export_compact_enrollment_review(
    *,
    selected: pd.DataFrame,
    output_dir: Path,
    source_package_id: str,
    source_reviewfix_run: Path,
    source_reviewfix_manifest_sha256: str,
    selection_csv_sha256: str,
    package_revision: str,
) -> ForensicReviewPackage:
    fingerprint = _selection_fingerprint(
        package_kind="compact_enrollment_audit",
        source_package_id=source_package_id,
        source_manifest_sha256=source_reviewfix_manifest_sha256,
        package_revision=package_revision,
        selected=selected,
    )
    package_id = f"compact-enrollment-audit-{fingerprint[:16]}"
    items: list[ForensicReviewItem] = []
    for row in selected.to_dict("records"):
        source_item_id = _text(row.get("Source_Broad_Item_ID"))
        item_id = _compact_item_id("CEA", source_package_id, source_item_id)
        duplicate_context = _text(row.get("Duplicate_Group_ID")) or "No exact duplicate group"
        items.append(
            ForensicReviewItem(
                item_id=item_id,
                evidence_source=Path(_text(row.get("Source_File"))),
                public={
                    "Compact_Rank": _text(row.get("Compact_Rank")),
                    "Expected_Roll": _text(row.get("Expected_Roll")),
                    "Expected_Student_Name": _text(row.get("Expected_Name")),
                    "Source_Filename": Path(_text(row.get("Source_File"))).name,
                    "Dataset_Folder": _text(row.get("Dataset_Folder")),
                    "Dataset_Role": _text(row.get("Dataset_Source_Role")),
                    "Priority_Tier": _text(row.get("Priority_Tier")),
                    "Primary_Selection_Reason": _text(row.get("Primary_Selection_Reason")),
                    "Secondary_Reasons": _text(row.get("Secondary_Reasons")),
                    "Selected_Issue_Reasons": _text(row.get("Issue_Flags")),
                    "Quality_Warnings": _quality_warning_text(row.get("Issue_Flags")),
                    "Duplicate_Group_Context": duplicate_context,
                    "Confusion_Involvement": _text(
                        row.get("Attractor_Confusion_Involvement")
                    ),
                    "Model_Similarity_Hint_Not_Ground_Truth": _text(
                        row.get("Model_Similarity_Hint_Not_Ground_Truth")
                    ),
                },
                private={
                    "source_broad_package_id": source_package_id,
                    "source_broad_item_id": source_item_id,
                    "source_reviewfix_run": str(Path(source_reviewfix_run).resolve()),
                    "source_file_sha256": _text(row.get("Source_File_SHA256")),
                    "expected_roll": _text(row.get("Expected_Roll")),
                    "expected_name": _text(row.get("Expected_Name")),
                    "identity_source": "enrollment_folder_expected_identity_for_human_audit",
                    "model_similarity_is_ground_truth": False,
                    "compact_rank": _integer(row.get("Compact_Rank")),
                    "deterministic_risk_score": _number(row.get("Deterministic_Risk_Score")),
                    "issue_flags": _text(row.get("Issue_Flags")),
                    "duplicate_group_id": _text(row.get("Duplicate_Group_ID")),
                },
            )
        )
    return export_forensic_review_package(
        output_dir=output_dir,
        package_id=package_id,
        package_kind="compact_enrollment_audit",
        title="Critical Enrollment Identity Review",
        instructions=(
            "Review only the person and enrollment usefulness shown in each source image. "
            "Ranking signals identify risk for human inspection; model similarity is diagnostic, not ground truth. "
            "No action changes a dataset or embedding file."
        ),
        actions=ENROLLMENT_ACTIONS,
        items=items,
        metadata={
            "compact_review_schema_version": COMPACT_REVIEW_SCHEMA_VERSION,
            "package_revision": package_revision,
            "source_broad_package_id": source_package_id,
            "source_reviewfix_run": str(Path(source_reviewfix_run).resolve()),
            "source_reviewfix_manifest_sha256": source_reviewfix_manifest_sha256,
            "compact_selection_csv_sha256": selection_csv_sha256,
            "model_similarity_used_as_ground_truth": False,
            "automatic_approvals": 0,
        },
    )


def export_compact_cctv_review(
    *,
    selected: pd.DataFrame,
    output_dir: Path,
    source_package_id: str,
    source_reviewfix_run: Path,
    source_reviewfix_manifest_sha256: str,
    selection_csv_sha256: str,
    package_revision: str,
) -> ForensicReviewPackage:
    fingerprint = _selection_fingerprint(
        package_kind="compact_verified_cctv",
        source_package_id=source_package_id,
        source_manifest_sha256=source_reviewfix_manifest_sha256,
        package_revision=package_revision,
        selected=selected,
    )
    package_id = f"compact-verified-cctv-{fingerprint[:16]}"
    items: list[ForensicReviewItem] = []
    for row in selected.to_dict("records"):
        source_item_id = _text(row.get("Source_Broad_Item_ID"))
        item_id = _compact_item_id("CCC", source_package_id, source_item_id)
        items.append(
            ForensicReviewItem(
                item_id=item_id,
                evidence_source=Path(_text(row.get("Crop_Evidence_Path"))),
                preserve_resolution=True,
                public={
                    "Compact_Rank": _text(row.get("Compact_Rank")),
                    "Human_Confirmed_Actual_Roll": _text(row.get("Actual_Roll")),
                    "Student_Name": _text(row.get("Student_Name")),
                    "Identity_Basis": "Completed human review (Actual_Roll)",
                    "Source_Session": _text(row.get("Source_Session")),
                    "Checkpoint": _text(row.get("Checkpoint")),
                    "Camera": _text(row.get("Camera")),
                    "Crop_Size": (
                        f"{_text(row.get('Crop_Width'))} x {_text(row.get('Crop_Height'))} px"
                    ),
                    "Detector_Confidence": _text(row.get("Detector_Score")),
                    "Blur_Metric": _text(row.get("Blur_Laplacian_Variance")),
                    "Brightness": _text(row.get("Brightness_Mean")),
                    "Pose_Roll": _text(row.get("Pose_Roll_Degrees")),
                    "Pose_Yaw_Proxy": _text(row.get("Pose_Yaw_Proxy")),
                    "Nearest_Selected_Crop_Similarity": _text(
                        row.get("Nearest_Selected_Crop_Similarity")
                    ),
                    "Diversity_New_Checkpoint": _text(row.get("Diversity_New_Checkpoint")),
                    "Diversity_New_Camera": _text(row.get("Diversity_New_Camera")),
                    "Primary_Selection_Reason": _text(row.get("Primary_Selection_Reason")),
                    "Confusion_And_Miss_Priority": _text(
                        row.get("Confusion_Miss_Priority")
                    ),
                    "Provenance": "Human-identified CCTV evidence from the frozen reviewed benchmark",
                },
                private={
                    "source_broad_package_id": source_package_id,
                    "source_broad_item_id": source_item_id,
                    "source_reviewfix_run": str(Path(source_reviewfix_run).resolve()),
                    "source_crop_sha256": _text(row.get("Evidence_SHA256")),
                    "actual_roll": _text(row.get("Actual_Roll")),
                    "identity_source": "human_review_actual_roll",
                    "model_prediction_used_as_identity": False,
                    "source_session": _text(row.get("Source_Session")),
                    "checkpoint": _text(row.get("Checkpoint")),
                    "camera": _text(row.get("Camera")),
                    "tracklet_id": _text(row.get("Track_ID")),
                    "observation_id": _text(row.get("Observation_ID")),
                    "compact_rank": _integer(row.get("Compact_Rank")),
                },
            )
        )
    return export_forensic_review_package(
        output_dir=output_dir,
        package_id=package_id,
        package_kind="compact_verified_cctv",
        title="Priority Human-Confirmed CCTV Crop Review",
        instructions=(
            "Confirm whether each original-resolution crop is safe and useful for a future candidate "
            "embedding version. Actual_Roll comes only from completed human review. No crop is copied "
            "into a dataset and no approval is automatic."
        ),
        actions=CCTV_ACTIONS,
        items=items,
        metadata={
            "compact_review_schema_version": COMPACT_REVIEW_SCHEMA_VERSION,
            "package_revision": package_revision,
            "source_broad_package_id": source_package_id,
            "source_reviewfix_run": str(Path(source_reviewfix_run).resolve()),
            "source_reviewfix_manifest_sha256": source_reviewfix_manifest_sha256,
            "compact_selection_csv_sha256": selection_csv_sha256,
            "identity_source": "human_review_actual_roll",
            "model_prediction_used_as_identity": False,
            "automatic_approvals": 0,
        },
    )


def _flag_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    if column not in frame.columns:
        return counts
    for value in frame[column]:
        for flag in sorted(_flags(value)):
            counts[flag] = counts.get(flag, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns or frame.empty:
        return {}
    return {
        str(key): int(value)
        for key, value in frame[column].value_counts().sort_index().items()
    }


def _protected_summary(state: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {
            "file_count": value.get("file_count"),
            "total_bytes": value.get("total_bytes"),
            "aggregate_sha256": value.get("aggregate_sha256"),
        }
        for name, value in dict(state.get("categories") or {}).items()
    }


def _enrollment_ranking_explanation(summary: dict[str, Any]) -> str:
    lines = [
            "# Compact Enrollment Ranking",
            "",
            "This package is a deterministic subset of the immutable 855-item broad review.",
            "It does not classify an image as wrong and does not modify an enrollment source.",
            "",
            "## Priority order",
            "",
            "1. Identity-integrity signals: ownership/folder mismatch, cross-student duplication, "
            "meaningful other-vs-own medoid disagreement, multiple faces, and reviewed false-accept involvement.",
            "2. Embedding-integrity signals: the 13 stored embedding outliers, own-class outliers, "
            "and severe disagreement with the owning identity's enrollment set.",
            "3. Material usability blockers: unreadable/no-face/tiny-face conditions and severe quality "
            "issues only when the image contributes or belongs to an already high-risk identity.",
            "",
            "## Deterministic controls",
            "",
            "- Risk points are additive by confirmed issue type; model similarity contributes only when "
            "the forensic audit already marked an own-vs-other disagreement.",
            "- Model similarity is never identity ground truth.",
            "- Exact source hashes are deduplicated.",
            "- Ordinary identities are capped at 3; reviewed attractors or identities with multiple "
            "critical issue types are capped at 5.",
            "- Blur/brightness-only rows are deferred rather than used as package padding.",
            "",
            f"Selected: {summary['selected_items']}",
            f"Deferred: {summary['deferred_items']}",
            f"Embedding outliers considered: {summary['embedding_outliers_considered']}",
            f"Embedding outliers available in broad source: "
            f"{summary['embedding_outliers_available_in_broad']}",
            f"Embedding outliers selected: {summary['embedding_outliers_selected']}",
            "",
        ]
    unavailable = summary.get("embedding_outliers_unavailable_from_broad", [])
    if unavailable:
        lines.extend(
            [
                "## Source availability exception",
                "",
                "The following stored outlier is not present in the validated 855-item broad package. "
                "It was considered but cannot be added without breaking immutable-source subset accounting:",
                "",
            ]
        )
        lines.extend(
            f"- `{row['image_path']}` ({row['canonical_roll']}; "
            f"robust score {row['robust_outlier_score']})"
            for row in unavailable
        )
        lines.append("")
    return "\n".join(lines)


def _cctv_ranking_explanation(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Compact CCTV Crop Ranking",
            "",
            "This package is a deterministic subset of the immutable 98-item human-identified CCTV review.",
            "Actual_Roll from completed human review is the only identity authority.",
            "",
            "## Priority order",
            "",
            "1. Reviewed false-accept attractors and identities with repeated top-1 misses.",
            "2. Identities with weak current embedding coverage or robust enrollment outliers.",
            "3. Strong CCTV-domain crop quality with checkpoint, camera, pose, and distance diversity.",
            "",
            "## Deterministic controls",
            "",
            "- Only human-eligible, identified rows with identity_source=human_review_actual_roll are eligible.",
            "- Mixed, not-in-mapping, uncertain, unidentifiable, invalid-roll, and broken-evidence rows are excluded.",
            "- At most one crop per track and source hash is selected.",
            "- Ordinary identities are capped at 3; confusion/miss priorities are capped at 4.",
            "- Repeated checkpoints/cameras and high nearest-crop similarity receive deterministic penalties.",
            "- No crop is copied into a dataset during compact review generation.",
            "",
            f"Selected: {summary['selected_items']}",
            f"Deferred: {summary['deferred_items']}",
            f"Selected students: {len(summary['selected_students'])}",
            "",
        ]
    )


def generate_compact_forensic_reviews(
    *,
    repo_root: Path,
    source_forensic_run: Path,
    source_reviewfix_run: Path,
    candidate_config_path: Path,
    embeddings_path: Path,
    embedding_summary_path: Path,
    student_map_path: Path,
    candidate_registry_path: Path,
    enrollment_maximum: int = MAX_COMPACT_ENROLLMENT_ITEMS,
    cctv_maximum: int = MAX_COMPACT_CCTV_ITEMS,
    output_dir: Path | None = None,
    subject_abbr: str = "CVO",
    package_revision: str = "compact-priority-review-v1",
    progress: Callable[[str], None] | None = print,
) -> CompactReviewRun:
    started = time.perf_counter()
    _validate_limit(enrollment_maximum, MAX_COMPACT_ENROLLMENT_ITEMS, "Enrollment")
    _validate_limit(cctv_maximum, MAX_COMPACT_CCTV_ITEMS, "CCTV")
    repo_root = Path(repo_root).resolve()
    candidate_config_path = Path(candidate_config_path).resolve()
    embeddings_path = Path(embeddings_path).resolve()
    embedding_summary_path = Path(embedding_summary_path).resolve()
    student_map_path = Path(student_map_path).resolve()
    candidate_registry_path = Path(candidate_registry_path).resolve()
    requested_output_dir = Path(output_dir).resolve() if output_dir else None
    if requested_output_dir is not None and requested_output_dir.exists():
        raise CompactReviewError(
            f"Refusing to overwrite compact forensic output: {requested_output_dir}"
        )
    validation = validate_compact_review_sources(
        repo_root=repo_root,
        source_forensic_run=source_forensic_run,
        source_reviewfix_run=source_reviewfix_run,
        candidate_config_path=candidate_config_path,
        embeddings_path=embeddings_path,
        embedding_summary_path=embedding_summary_path,
        student_map_path=student_map_path,
        candidate_registry_path=candidate_registry_path,
    )
    if progress:
        progress(
            "Validated source forensic manifests and broad packages: "
            f"{validation.summary['source_broad_enrollment_count']} enrollment, "
            f"{validation.summary['source_broad_cctv_count']} CCTV"
        )

    enrollment_items = _load_enrollment_broad_items(
        validation,
        student_map_path=student_map_path,
        subject_abbr=subject_abbr,
    )
    cctv_items = _load_cctv_broad_items(validation)
    outliers = pd.read_csv(
        validation.source_reviewfix_run / "embedding_outliers.csv",
        dtype=str,
        keep_default_na=False,
    )
    attractors = pd.read_csv(
        validation.source_reviewfix_run / "predicted_identity_attractors.csv",
        dtype=str,
        keep_default_na=False,
    )
    misses = pd.read_csv(
        validation.source_reviewfix_run / "actual_identity_miss_patterns.csv",
        dtype=str,
        keep_default_na=False,
    )
    embedding_health = pd.read_csv(
        validation.source_reviewfix_run / "student_embedding_health.csv",
        dtype=str,
        keep_default_na=False,
    )
    names = _roster_name_map(student_map_path, subject_abbr)
    enrollment_selection = select_compact_enrollment_items(
        enrollment_items,
        outliers,
        attractors,
        misses,
        maximum=enrollment_maximum,
        preferred=min(DEFAULT_COMPACT_ENROLLMENT_ITEMS, enrollment_maximum),
    )
    cctv_selection = select_compact_cctv_items(
        cctv_items,
        attractors,
        misses,
        embedding_health,
        maximum=cctv_maximum,
        preferred=min(DEFAULT_COMPACT_CCTV_ITEMS, cctv_maximum),
        authoritative_rolls=set(names),
    )

    run_id = make_diagnostic_run_id("embedding_forensics_compact_review")
    output_dir = Path(
        requested_output_dir
        or repo_root / "attendance_output" / "embedding_forensics" / run_id
    ).resolve()
    if output_dir.exists():
        raise CompactReviewError(f"Refusing to overwrite compact forensic output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        enrollment_selection_path = output_dir / "compact_enrollment_selection.csv"
        enrollment_deferred_path = output_dir / "compact_enrollment_deferred.csv"
        cctv_selection_path = output_dir / "compact_cctv_selection.csv"
        cctv_deferred_path = output_dir / "compact_cctv_deferred.csv"
        _write_csv(enrollment_selection_path, enrollment_selection.selected)
        _write_csv(enrollment_deferred_path, enrollment_selection.deferred)
        _write_csv(cctv_selection_path, cctv_selection.selected)
        _write_csv(cctv_deferred_path, cctv_selection.deferred)
        enrollment_summary = {
            **enrollment_selection.summary,
            "selected_issue_categories": _flag_counts(
                enrollment_selection.selected, "Issue_Flags"
            ),
            "primary_selection_reasons": _value_counts(
                enrollment_selection.selected, "Primary_Selection_Reason"
            ),
            "source_broad_package_id": validation.enrollment_package.public["package_id"],
            "source_broad_package_manifest_sha256": validation.enrollment_package.manifest_sha256,
        }
        cctv_summary = {
            **cctv_selection.summary,
            "primary_selection_reasons": _value_counts(
                cctv_selection.selected, "Primary_Selection_Reason"
            ),
            "source_broad_package_id": validation.cctv_package.public["package_id"],
            "source_broad_package_manifest_sha256": validation.cctv_package.manifest_sha256,
        }
        _write_json(output_dir / "compact_enrollment_selection_summary.json", enrollment_summary)
        _write_json(output_dir / "compact_cctv_selection_summary.json", cctv_summary)
        (output_dir / "compact_enrollment_ranking_explanation.md").write_text(
            _enrollment_ranking_explanation(enrollment_summary), encoding="utf-8"
        )
        (output_dir / "compact_cctv_ranking_explanation.md").write_text(
            _cctv_ranking_explanation(cctv_summary), encoding="utf-8"
        )

        enrollment_review = export_compact_enrollment_review(
            selected=enrollment_selection.selected,
            output_dir=output_dir / "compact_enrollment_review",
            source_package_id=validation.enrollment_package.public["package_id"],
            source_reviewfix_run=validation.source_reviewfix_run,
            source_reviewfix_manifest_sha256=validation.reviewfix_output_manifest_sha256,
            selection_csv_sha256=_sha256_file(enrollment_selection_path),
            package_revision=package_revision,
        )
        cctv_review = export_compact_cctv_review(
            selected=cctv_selection.selected,
            output_dir=output_dir / "compact_cctv_review",
            source_package_id=validation.cctv_package.public["package_id"],
            source_reviewfix_run=validation.source_reviewfix_run,
            source_reviewfix_manifest_sha256=validation.reviewfix_output_manifest_sha256,
            selection_csv_sha256=_sha256_file(cctv_selection_path),
            package_revision=package_revision,
        )
        cli_script = repo_root / "scripts" / "validate_tracklet_ground_truth.py"
        enrollment_launcher = write_reviewer_launcher(
            launcher_path=output_dir / "open_compact_enrollment_review.ps1",
            python_executable=Path(sys.executable),
            cli_script=cli_script,
            command="open-compact-enrollment-review",
            review_package=enrollment_review.root,
        )
        cctv_launcher = write_reviewer_launcher(
            launcher_path=output_dir / "open_compact_cctv_review.ps1",
            python_executable=Path(sys.executable),
            cli_script=cli_script,
            command="open-compact-cctv-review",
            review_package=cctv_review.root,
        )

        protected_after = snapshot_protected_state(
            repo_root=repo_root,
            candidate_config_path=candidate_config_path,
            embeddings_path=embeddings_path,
            embedding_summary_path=embedding_summary_path,
            student_map_path=student_map_path,
        )
        protected_changes = compare_protected_state(validation.protected_state, protected_after)
        enrollment_tree_after = _tree_state(validation.enrollment_package.root)
        cctv_tree_after = _tree_state(validation.cctv_package.root)
        broad_packages_unchanged = (
            enrollment_tree_after == validation.enrollment_package.tree_state
            and cctv_tree_after == validation.cctv_package.tree_state
        )
        production_unchanged = (
            _sha256_file(embeddings_path) == validation.production_embeddings_sha256
            and _sha256_file(embedding_summary_path) == validation.production_summary_sha256
        )
        if protected_changes or not broad_packages_unchanged or not production_unchanged:
            raise CompactReviewError(
                "Compact generation changed protected inputs or broad review evidence"
            )

        elapsed = round(time.perf_counter() - started, 2)
        summary = {
            "schema_version": COMPACT_REVIEW_SCHEMA_VERSION,
            "compact_review_run_id": run_id,
            "output_dir": str(output_dir),
            "package_revision": package_revision,
            "source_primary_forensic_run": str(validation.source_forensic_run),
            "source_reviewfix_run": str(validation.source_reviewfix_run),
            "source_primary_output_manifest_sha256": validation.source_output_manifest_sha256,
            "source_reviewfix_output_manifest_sha256": validation.reviewfix_output_manifest_sha256,
            "source_broad_enrollment_count": len(enrollment_items),
            "source_broad_cctv_count": len(cctv_items),
            "compact_enrollment_selected_count": len(enrollment_selection.selected),
            "compact_enrollment_deferred_count": len(enrollment_selection.deferred),
            "compact_cctv_selected_count": len(cctv_selection.selected),
            "compact_cctv_deferred_count": len(cctv_selection.deferred),
            "compact_enrollment_selected_students": enrollment_selection.summary[
                "selected_students"
            ],
            "compact_enrollment_per_student_counts": enrollment_selection.summary[
                "per_student_counts"
            ],
            "compact_cctv_selected_students": cctv_selection.summary["selected_students"],
            "compact_cctv_per_student_counts": cctv_selection.summary[
                "per_student_counts"
            ],
            "selected_issue_categories": enrollment_summary["selected_issue_categories"],
            "review_packages": {
                "compact_enrollment": str(enrollment_review.root.resolve()),
                "compact_cctv": str(cctv_review.root.resolve()),
            },
            "review_package_ids": {
                "compact_enrollment": enrollment_review.package_id,
                "compact_cctv": cctv_review.package_id,
            },
            "reviewer_launchers": {
                "compact_enrollment": str(enrollment_launcher.resolve()),
                "compact_cctv": str(cctv_launcher.resolve()),
            },
            "required_approval_csv_filenames": {
                "compact_enrollment": f"forensic_review_approvals_{enrollment_review.package_id}.csv",
                "compact_cctv": f"forensic_review_approvals_{cctv_review.package_id}.csv",
            },
            "production_embeddings_changed": False,
            "datasets_changed": False,
            "broad_review_packages_changed": False,
            "recognition_rerun": False,
            "video_processing_rerun": False,
            "full_dataset_embedding_extraction_rerun": False,
            "candidate_embedding_built": False,
            "candidate_embedding_promoted": False,
            "mon_p3_processed": False,
            "official_thresholds_changed": False,
            "official_match_threshold": OFFICIAL_MATCH_THRESHOLD,
            "official_margin_threshold": OFFICIAL_MARGIN_THRESHOLD,
            "attendance_rule_changed": False,
            "official_attendance_checkpoints": OFFICIAL_ATTENDANCE_CHECKPOINTS,
            "official_attendance_changed": False,
            "automatic_image_approvals": 0,
            "model_prediction_used_as_identity": False,
            "protected_state_unchanged": True,
            "elapsed_seconds": elapsed,
        }
        _write_json(output_dir / "compact_review_summary.json", summary)
        compact_manifest = {
            "schema_version": COMPACT_REVIEW_SCHEMA_VERSION,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "compact_review_run_id": run_id,
            "source_validation": {
                **validation.summary,
                "source_primary_output_manifest_sha256": validation.source_output_manifest_sha256,
                "source_reviewfix_output_manifest_sha256": validation.reviewfix_output_manifest_sha256,
                "ranking_artifact_sha256": validation.ranking_artifact_sha256,
                "production_embeddings_sha256": validation.production_embeddings_sha256,
                "production_summary_sha256": validation.production_summary_sha256,
                "rejected_candidate_registry_sha256": validation.rejected_candidate_registry_sha256,
                "broad_enrollment_manifest_sha256": validation.enrollment_package.manifest_sha256,
                "broad_cctv_manifest_sha256": validation.cctv_package.manifest_sha256,
                "broad_enrollment_tree_before": validation.enrollment_package.tree_state,
                "broad_enrollment_tree_after": enrollment_tree_after,
                "broad_cctv_tree_before": validation.cctv_package.tree_state,
                "broad_cctv_tree_after": cctv_tree_after,
            },
            "protected_state_before": _protected_summary(validation.protected_state),
            "protected_state_after": _protected_summary(protected_after),
            "selection_artifacts": {
                "compact_enrollment_selection.csv": _sha256_file(enrollment_selection_path),
                "compact_enrollment_deferred.csv": _sha256_file(enrollment_deferred_path),
                "compact_cctv_selection.csv": _sha256_file(cctv_selection_path),
                "compact_cctv_deferred.csv": _sha256_file(cctv_deferred_path),
            },
            "review_package_ids": summary["review_package_ids"],
            "safety": {
                key: summary[key]
                for key in (
                    "production_embeddings_changed",
                    "datasets_changed",
                    "broad_review_packages_changed",
                    "recognition_rerun",
                    "candidate_embedding_built",
                    "candidate_embedding_promoted",
                    "mon_p3_processed",
                    "official_thresholds_changed",
                    "attendance_rule_changed",
                    "official_attendance_changed",
                )
            },
        }
        _write_json(output_dir / "compact_review_manifest.json", compact_manifest)
        output_manifest = write_output_manifest(
            output_dir,
            {
                "compact_review_run_id": run_id,
                "source_reviewfix_output_manifest_sha256": validation.reviewfix_output_manifest_sha256,
                "compact_enrollment_package_id": enrollment_review.package_id,
                "compact_cctv_package_id": cctv_review.package_id,
                "compact_enrollment_selected_count": len(enrollment_selection.selected),
                "compact_cctv_selected_count": len(cctv_selection.selected),
                "protected_state_unchanged": True,
                "production_embeddings_changed": False,
                "datasets_changed": False,
                "broad_packages_changed": False,
                "recognition_rerun": False,
                "candidate_embedding_built": False,
                "candidate_embedding_promoted": False,
                "mon_p3_processed": False,
            },
        )
        verify_output_manifest(output_dir, output_manifest)
        if progress:
            progress(
                f"Compact enrollment selected/deferred: {len(enrollment_selection.selected)}/"
                f"{len(enrollment_selection.deferred)}"
            )
            progress(
                f"Compact CCTV selected/deferred: {len(cctv_selection.selected)}/"
                f"{len(cctv_selection.deferred)}"
            )
            progress(f"Compact review generation completed in {elapsed:.2f} seconds")
        return CompactReviewRun(
            output_dir=output_dir,
            enrollment_selection=enrollment_selection,
            cctv_selection=cctv_selection,
            enrollment_review=enrollment_review,
            cctv_review=cctv_review,
            summary=summary,
            output_manifest=output_manifest,
        )
    except Exception as exc:
        try:
            _write_json(
                output_dir / "compact_review_failure.json",
                {
                    "schema_version": COMPACT_REVIEW_SCHEMA_VERSION,
                    "completed": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "production_success_claimed": False,
                    "recognition_rerun": False,
                    "candidate_embedding_built": False,
                    "candidate_embedding_promoted": False,
                },
            )
            write_output_manifest(
                output_dir,
                {
                    "compact_review_run_id": run_id,
                    "completed": False,
                    "partial_output": True,
                    "production_success_claimed": False,
                },
                filename="partial_output_manifest.json",
            )
        except (OSError, ShadowValidationError):
            pass
        raise


def verify_reusable_compact_review(
    *,
    output_dir: Path,
    repo_root: Path,
    source_forensic_run: Path,
    source_reviewfix_run: Path,
    candidate_config_path: Path,
    embeddings_path: Path,
    embedding_summary_path: Path,
    student_map_path: Path,
    candidate_registry_path: Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    try:
        verify_output_manifest(output_dir, output_dir / "output_manifest.json")
    except ShadowValidationError as exc:
        raise CompactReviewError(str(exc)) from exc
    manifest_path = output_dir / "compact_review_manifest.json"
    summary_path = output_dir / "compact_review_summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        raise CompactReviewError(f"Reusable compact review is incomplete: {output_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validation = validate_compact_review_sources(
        repo_root=repo_root,
        source_forensic_run=source_forensic_run,
        source_reviewfix_run=source_reviewfix_run,
        candidate_config_path=candidate_config_path,
        embeddings_path=embeddings_path,
        embedding_summary_path=embedding_summary_path,
        student_map_path=student_map_path,
        candidate_registry_path=candidate_registry_path,
    )
    source_validation = dict(manifest.get("source_validation") or {})
    expected_hashes = {
        "source_primary_output_manifest_sha256": validation.source_output_manifest_sha256,
        "source_reviewfix_output_manifest_sha256": validation.reviewfix_output_manifest_sha256,
        "production_embeddings_sha256": validation.production_embeddings_sha256,
        "production_summary_sha256": validation.production_summary_sha256,
        "broad_enrollment_manifest_sha256": validation.enrollment_package.manifest_sha256,
        "broad_cctv_manifest_sha256": validation.cctv_package.manifest_sha256,
    }
    mismatches = [
        key for key, value in expected_hashes.items()
        if _text(source_validation.get(key)) != value
    ]
    if mismatches:
        raise CompactReviewError(
            "Reusable compact output source hashes are stale: " + ", ".join(mismatches)
        )
    enrollment_package = validate_broad_review_package(
        output_dir / "compact_enrollment_review",
        expected_package_id=summary["review_package_ids"]["compact_enrollment"],
        expected_count=_integer(summary["compact_enrollment_selected_count"]),
        expected_kind="compact_enrollment_audit",
    )
    cctv_package = validate_broad_review_package(
        output_dir / "compact_cctv_review",
        expected_package_id=summary["review_package_ids"]["compact_cctv"],
        expected_count=_integer(summary["compact_cctv_selected_count"]),
        expected_kind="compact_verified_cctv",
    )
    if (
        len(pd.read_csv(output_dir / "compact_enrollment_selection.csv"))
        != enrollment_package.public["item_count"]
        or len(pd.read_csv(output_dir / "compact_cctv_selection.csv"))
        != cctv_package.public["item_count"]
    ):
        raise CompactReviewError("Reusable compact selection/package counts do not match")
    unsafe_true = [
        key
        for key in (
            "production_embeddings_changed",
            "datasets_changed",
            "broad_review_packages_changed",
            "recognition_rerun",
            "candidate_embedding_built",
            "candidate_embedding_promoted",
            "mon_p3_processed",
            "official_thresholds_changed",
            "attendance_rule_changed",
            "official_attendance_changed",
        )
        if summary.get(key) is not False
    ]
    if unsafe_true or summary.get("protected_state_unchanged") is not True:
        raise CompactReviewError(
            "Reusable compact summary violates the safety contract: " + ", ".join(unsafe_true)
        )
    return summary
