from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .diagnostics import finite_float, json_safe
from .embedding_db import StudentEmbeddingDB
from .tracklet_review import (
    LabelValidationError,
    ReviewPackageResult,
    _yes,
    export_review_package,
    load_actual_present,
    load_review_package_data,
    load_student_mapping,
    normalize_roll,
    validate_label_dataframe,
)


MATCH_THRESHOLD = 0.48
MARGIN_THRESHOLD = 0.08
PRESENT_CHECKPOINTS = 3


class RecallAnalysisError(ValueError):
    pass


@dataclass(frozen=True)
class UnresolvedSelectionConfig:
    target_tracks: int = 70
    candidate_cap: int = 6
    checkpoint_cap: int | None = None
    camera_cap: int | None = None
    zone_cap: int | None = None

    def resolved_caps(self) -> tuple[int, int, int]:
        if not 1 <= self.target_tracks <= 200:
            raise RecallAnalysisError("target_tracks must be between 1 and 200")
        if self.candidate_cap < 1:
            raise RecallAnalysisError("candidate_cap must be positive")
        checkpoint_cap = self.checkpoint_cap or max(1, math.ceil(self.target_tracks * 0.30))
        camera_cap = self.camera_cap or max(1, math.ceil(self.target_tracks * 0.60))
        zone_cap = self.zone_cap or max(1, math.ceil(self.target_tracks * 0.25))
        return checkpoint_cap, camera_cap, zone_cap


@dataclass(frozen=True)
class UnresolvedValidationResult:
    tracks: pd.DataFrame
    student_metrics: pd.DataFrame
    summary: dict[str, Any]
    camera_metrics: pd.DataFrame
    checkpoint_metrics: pd.DataFrame
    zone_metrics: pd.DataFrame
    quality_metrics: pd.DataFrame


@dataclass(frozen=True)
class RecallAnalysisRunResult:
    output_dir: Path
    review_package: ReviewPackageResult
    student_recall_gap: pd.DataFrame
    embedding_coverage: pd.DataFrame
    unresolved_selection: pd.DataFrame
    summary: dict[str, Any]


def canonical_roll(value: Any) -> str:
    raw = normalize_roll(value)
    match = re.match(r"^([A-Z0-9]+)", raw)
    if not match:
        return raw
    token = match.group(1)
    if any(character.isalpha() for character in token) and any(character.isdigit() for character in token):
        return token
    return raw


def _text(value: Any) -> str:
    return str(value or "").strip()


def _number(value: Any, default: float = 0.0) -> float:
    return finite_float(value, default)


def _int(value: Any) -> int:
    return int(round(_number(value)))


def _split_values(value: Any) -> list[str]:
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def _joined(values: Iterable[Any]) -> str:
    return "; ".join(sorted({_text(value) for value in values if _text(value)}))


def load_embedding_counts(path: Path) -> dict[str, int]:
    database = StudentEmbeddingDB.load(Path(path))
    return {key: len(values) for key, values in database.by_roll.items()}


def build_embedding_roster_coverage(
    roster: list[dict[str, Any]],
    embedding_summary_df: pd.DataFrame,
    *,
    embedding_counts: dict[str, int] | None = None,
) -> pd.DataFrame:
    required = {"Roll_Number", "Faces_Used"}
    missing = sorted(required.difference(embedding_summary_df.columns))
    if missing:
        raise RecallAnalysisError(f"Embedding summary missing columns: {', '.join(missing)}")

    summary_by_key = {
        _text(row["Roll_Number"]): _int(row["Faces_Used"])
        for row in embedding_summary_df.fillna("").to_dict("records")
        if _text(row.get("Roll_Number"))
    }
    counts = dict(embedding_counts or summary_by_key)
    all_keys = sorted(set(summary_by_key) | set(counts))
    keys_by_canonical: dict[str, list[str]] = defaultdict(list)
    for key in all_keys:
        keys_by_canonical[canonical_roll(key)].append(key)

    rows: list[dict[str, Any]] = []
    for student in roster:
        original_roll = _text(student.get("roll"))
        canonical = canonical_roll(original_roll)
        matching_keys = sorted(keys_by_canonical.get(canonical, []))
        record_count = sum(max(0, int(counts.get(key, 0))) for key in matching_keys)
        summary_faces = sum(max(0, int(summary_by_key.get(key, 0))) for key in matching_keys)
        collision = len(matching_keys) > 1
        exact = any(normalize_roll(key) == normalize_roll(original_roll) for key in matching_keys)
        annotated = bool(matching_keys) and not exact
        if not matching_keys:
            mapping_status = "missing"
        elif collision:
            mapping_status = "collision"
        elif annotated:
            mapping_status = "canonicalized_annotation"
        else:
            mapping_status = "exact"
        available = bool(matching_keys and record_count > 0)
        rows.append({
            "Original_Roster_Roll": original_roll,
            "Canonical_Roster_Roll": canonical,
            "Student_Name": _text(student.get("name")),
            "Matching_Embedding_Keys": _joined(matching_keys),
            "Canonical_Embedding_Rolls": _joined(canonical_roll(key) for key in matching_keys),
            "Embedding_Record_Count": record_count,
            "Summary_Faces_Used": summary_faces,
            "Embedding_Available": "Yes" if available else "No",
            "Valid_Status": "valid" if available else "invalid_or_missing",
            "Canonical_Mapping_Status": mapping_status,
            "Collision_Status": "collision" if collision else "none",
            "Missing_Status": "missing" if not matching_keys else "present",
        })
    return pd.DataFrame(rows)


TRACKLET_NUMERIC_COLUMNS = (
    "Tracklet_Best_Score",
    "Tracklet_Second_Score",
    "Tracklet_Margin",
    "Dominant_Frame_Best_Share_Pct",
    "Observation_Count",
    "Embedding_Count",
    "Selected_Observation_Count",
    "Consistent_Embedding_Count",
    "Inconsistent_Embedding_Count",
    "Usable_Observation_Count",
    "Marginal_Observation_Count",
    "Unusable_Observation_Count",
    "Quality_Weight_Median",
    "Face_Width_Median",
    "Pairwise_Similarity_Median",
)


def prepare_tracklet_diagnostics(tracklet_df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "Tracklet_ID",
        "Tracklet_Accepted",
        "Tracklet_Eligible",
        "Tracklet_Best_Roll",
        "Checkpoint_ID",
        "Camera_ID",
        "Selected_Observation_IDs",
    }
    missing = sorted(required.difference(tracklet_df.columns))
    if missing:
        raise RecallAnalysisError(f"Tracklet diagnostics missing columns: {', '.join(missing)}")
    frame = tracklet_df.copy().fillna("")
    for column in TRACKLET_NUMERIC_COLUMNS:
        if column not in frame:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    for column in (
        "Tracklet_Second_Roll",
        "Dominant_Frame_Best_Roll",
        "Zone_IDs",
        "Tracklet_Diagnostic_Reason",
        "Tracklet_Quality_Rejection",
    ):
        if column not in frame:
            frame[column] = ""
    frame["Candidate_Canonical_Roll"] = frame["Tracklet_Best_Roll"].map(canonical_roll)
    frame["Second_Canonical_Roll"] = frame["Tracklet_Second_Roll"].map(canonical_roll)
    frame["Dominant_Frame_Canonical_Roll"] = frame["Dominant_Frame_Best_Roll"].map(canonical_roll)
    return frame


def _quality_label(rows: pd.DataFrame) -> str:
    if rows.empty:
        return "none"
    if rows["Usable_Observation_Count"].max() > 0:
        return "usable_reference"
    if rows["Marginal_Observation_Count"].max() > 0:
        return "marginal_reference"
    return "unusable_reference"


def _closest_track(rows: pd.DataFrame) -> pd.Series | None:
    if rows.empty:
        return None
    ranked = rows.copy()
    ranked["_score_gap"] = (MATCH_THRESHOLD - ranked["Tracklet_Best_Score"]).clip(lower=0)
    ranked["_margin_gap"] = (MARGIN_THRESHOLD - ranked["Tracklet_Margin"]).clip(lower=0)
    ranked["_maturity_penalty"] = ranked["Tracklet_Eligible"].map(lambda value: 0.0 if _yes(value) else 1.0)
    ranked["_distance"] = ranked["_score_gap"] + ranked["_margin_gap"] + ranked["_maturity_penalty"]
    ranked = ranked.sort_values(
        ["_distance", "Quality_Weight_Median", "Observation_Count", "Tracklet_ID"],
        ascending=[True, False, False, True],
    )
    return ranked.iloc[0]


def _false_negative_category(
    accepted_checkpoints: set[str],
    coverage: dict[str, Any],
    candidate_rows: pd.DataFrame,
    mature_rows: pd.DataFrame,
) -> str:
    if len(accepted_checkpoints) >= PRESENT_CHECKPOINTS:
        return "present_confirmed"
    if _text(coverage.get("Collision_Status")) == "collision":
        return "canonical_roll_mismatch"
    if _text(coverage.get("Embedding_Available")) != "Yes":
        return "embedding_missing"
    if accepted_checkpoints:
        return "insufficient_checkpoint_coverage"
    if candidate_rows.empty:
        return "no_candidate_evidence"
    if candidate_rows["Usable_Observation_Count"].max() <= 0 and candidate_rows["Marginal_Observation_Count"].max() <= 0:
        return "no_usable_crop"
    if mature_rows.empty:
        return "no_mature_track"
    closest = _closest_track(mature_rows)
    if closest is None:
        return "unresolved_other"
    reason = _text(closest.get("Tracklet_Diagnostic_Reason"))
    if closest["Tracklet_Best_Score"] < MATCH_THRESHOLD:
        return "correct_candidate_score_below_threshold"
    if closest["Tracklet_Margin"] < MARGIN_THRESHOLD:
        return "correct_candidate_margin_below_threshold"
    if closest["Dominant_Frame_Best_Share_Pct"] < 50:
        return "correct_candidate_vote_inconsistent"
    if reason and reason not in {"accepted", "score_below_threshold", "margin_below_threshold", "score_and_margin_below_threshold"}:
        return "mature_track_rejected_by_conflict"
    return "unresolved_other"


def _corrective_direction(category: str) -> str:
    return {
        "present_confirmed": "none",
        "embedding_missing": "embedding_or_roster_repair",
        "canonical_roll_mismatch": "canonical_roll_mapping",
        "roster_embedding_mismatch": "embedding_or_roster_repair",
        "insufficient_checkpoint_coverage": "zone_camera_coverage_or_sampling_pending_review",
        "correct_candidate_score_below_threshold": "cctv_domain_dataset_adaptation_candidate_pending_review",
        "correct_candidate_margin_below_threshold": "tracklet_acceptance_calibration_candidate_pending_review",
        "correct_candidate_vote_inconsistent": "tracklet_acceptance_calibration_candidate_pending_review",
        "mature_track_rejected_by_conflict": "tracklet_acceptance_calibration_candidate_pending_review",
        "no_mature_track": "sampling_or_association",
        "no_usable_crop": "hardware_or_camera_placement",
        "no_candidate_evidence": "unresolved_blind_review_required",
        "camera_or_zone_visibility_gap": "zone_camera_coverage_or_sampling",
        "unresolved_other": "unresolved_blind_review_required",
    }.get(category, "unresolved_blind_review_required")


def build_student_recall_gap(
    *,
    roster: list[dict[str, Any]],
    actual_present_rolls: set[str],
    embedding_coverage_df: pd.DataFrame,
    tracklet_df: pd.DataFrame,
    accepted_label_evidence_df: pd.DataFrame,
) -> pd.DataFrame:
    tracks = prepare_tracklet_diagnostics(tracklet_df)
    coverage_by_roll = {
        canonical_roll(row["Canonical_Roster_Roll"]): row
        for row in embedding_coverage_df.fillna("").to_dict("records")
    }
    roster_by_roll = {canonical_roll(row.get("roll")): row for row in roster}
    accepted = accepted_label_evidence_df.copy().fillna("")
    for column in ("Actual_Roll", "Checkpoint_ID", "Camera_ID"):
        if column not in accepted:
            accepted[column] = ""
    accepted["Canonical_Actual_Roll"] = accepted["Actual_Roll"].map(canonical_roll)

    rows: list[dict[str, Any]] = []
    for roll in sorted({canonical_roll(value) for value in actual_present_rolls if canonical_roll(value)}):
        roster_row = roster_by_roll.get(roll, {})
        coverage = coverage_by_roll.get(roll, {
            "Matching_Embedding_Keys": "",
            "Embedding_Available": "No",
            "Canonical_Mapping_Status": "missing",
            "Collision_Status": "none",
        })
        accepted_rows = accepted[accepted["Canonical_Actual_Roll"].eq(roll)]
        accepted_checkpoints = {_text(value) for value in accepted_rows["Checkpoint_ID"] if _text(value)}
        accepted_cameras = {_text(value) for value in accepted_rows["Camera_ID"] if _text(value)}
        unresolved = tracks[~tracks["Tracklet_Accepted"].map(_yes)]
        best_rows = unresolved[unresolved["Candidate_Canonical_Roll"].eq(roll)]
        dominant_rows = unresolved[unresolved["Dominant_Frame_Canonical_Roll"].eq(roll)]
        candidate_rows = pd.concat([best_rows, dominant_rows], ignore_index=True).drop_duplicates("Tracklet_ID")
        mature_rows = best_rows[best_rows["Tracklet_Eligible"].map(_yes)]
        closest = _closest_track(candidate_rows)
        rejection_counts = Counter(
            _text(value) or "unspecified"
            for value in candidate_rows["Tracklet_Diagnostic_Reason"]
        )
        candidate_checkpoints = {_text(value) for value in candidate_rows["Checkpoint_ID"] if _text(value)}
        candidate_cameras = {_text(value) for value in candidate_rows["Camera_ID"] if _text(value)}
        candidate_zones = {
            zone
            for value in candidate_rows["Zone_IDs"]
            for zone in _split_values(value)
        }
        category = _false_negative_category(accepted_checkpoints, coverage, candidate_rows, mature_rows)
        secondary: list[str] = []
        if len(accepted_checkpoints) < PRESENT_CHECKPOINTS:
            secondary.append(f"accepted_checkpoint_count:{len(accepted_checkpoints)}")
        secondary.append(f"candidate_checkpoint_count:{len(candidate_checkpoints)}")
        if not candidate_rows.empty and candidate_rows["Usable_Observation_Count"].max() <= 0:
            secondary.append("no_usable_candidate_crop")
        if not mature_rows.empty:
            secondary.append(f"mature_candidate_tracks:{len(mature_rows)}")
        rows.append({
            "Canonical_Roll": roll,
            "Student_Name": _text(roster_row.get("name")),
            "Roster_Membership": "Yes" if roll in roster_by_roll else "No",
            "Embedding_Keys": _text(coverage.get("Matching_Embedding_Keys")),
            "Embedding_Available": _text(coverage.get("Embedding_Available")) or "No",
            "Canonical_Mapping_Status": _text(coverage.get("Canonical_Mapping_Status")) or "missing",
            "Accepted_Track_Count": int(len(accepted_rows)),
            "Accepted_Checkpoints": _joined(accepted_checkpoints),
            "Accepted_Cameras": _joined(accepted_cameras),
            "Eligible_Unresolved_Best_Candidate_Tracks": int(len(mature_rows)),
            "Mature_Rejected_Track_Count": int(len(mature_rows)),
            "Best_Aggregate_Score": round(float(candidate_rows["Tracklet_Best_Score"].max()), 4) if not candidate_rows.empty else "",
            "Best_Aggregate_Margin": round(float(candidate_rows["Tracklet_Margin"].max()), 4) if not candidate_rows.empty else "",
            "Best_Vote_Ratio": round(float(candidate_rows["Dominant_Frame_Best_Share_Pct"].max()) / 100.0, 4) if not candidate_rows.empty else "",
            "Best_Crop_Quality": _quality_label(candidate_rows),
            "Candidate_Checkpoints": _joined(candidate_checkpoints),
            "Candidate_Cameras": _joined(candidate_cameras),
            "Candidate_Zones": _joined(candidate_zones),
            "Rejection_Reason_Counts": json.dumps(dict(sorted(rejection_counts.items())), sort_keys=True),
            "Closest_To_Acceptance_Track_ID": _text(closest.get("Tracklet_ID")) if closest is not None else "",
            "Attendance_Evidence_Count": len(accepted_checkpoints),
            "Current_Outcome": "present_confirmed" if len(accepted_checkpoints) >= PRESENT_CHECKPOINTS else "false_negative",
            "False_Negative_Category": category,
            "Secondary_Contributing_Reasons": "; ".join(secondary),
            "Recommended_Corrective_Direction": _corrective_direction(category),
        })
    return pd.DataFrame(rows)


def _primary_zone(value: Any) -> str:
    zones = _split_values(value)
    return sorted(zones)[0] if zones else "full_frame"


def _track_quality_band(row: pd.Series | dict[str, Any]) -> str:
    usable = _number(row.get("Usable_Observation_Count"))
    marginal = _number(row.get("Marginal_Observation_Count"))
    weight = _number(row.get("Quality_Weight_Median"))
    width = _number(row.get("Face_Width_Median"))
    if usable > 0 and weight >= 0.45 and width >= 40:
        return "high"
    if usable + marginal >= 2 and width >= 28:
        return "medium"
    return "low"


def _selection_reasons(row: pd.Series, false_negative_rolls: set[str]) -> list[str]:
    reasons: list[str] = []
    score = float(row["Tracklet_Best_Score"])
    margin = float(row["Tracklet_Margin"])
    vote = float(row["Dominant_Frame_Best_Share_Pct"])
    if abs(score - MATCH_THRESHOLD) <= 0.08:
        reasons.append("near_match_threshold")
    if score < MATCH_THRESHOLD and vote >= 60:
        reasons.append("strong_vote_low_score")
    if score >= MATCH_THRESHOLD and margin < MARGIN_THRESHOLD:
        reasons.append("score_pass_margin_fail")
    if (
        row["Candidate_Canonical_Roll"] != row["Dominant_Frame_Canonical_Roll"]
        or row["Inconsistent_Embedding_Count"] > 0
    ):
        reasons.append("aggregate_vote_disagreement")
    if row["Candidate_Canonical_Roll"] in false_negative_rolls:
        reasons.append("missed_student_best_candidate")
    if row["Second_Canonical_Roll"] in false_negative_rolls:
        reasons.append("missed_student_second_candidate")
    if row["Quality_Band"] == "high":
        reasons.append("high_quality_evidence")
    if not _yes(row["Tracklet_Eligible"]):
        reasons.append("maturity_or_consistency_rejection")
    return reasons or ["representative_unresolved"]


def _selection_score(row: pd.Series, reasons: list[str]) -> float:
    score = float(row["Tracklet_Best_Score"])
    margin = float(row["Tracklet_Margin"])
    vote = min(100.0, max(0.0, float(row["Dominant_Frame_Best_Share_Pct"]))) / 100.0
    quality = min(1.0, max(0.0, float(row["Quality_Weight_Median"])))
    closeness = max(0.0, 1.0 - abs(score - MATCH_THRESHOLD) / 0.20)
    value = closeness * 35.0 + vote * 20.0 + quality * 15.0
    value += min(10.0, float(row["Observation_Count"]))
    value += 10.0 if "score_pass_margin_fail" in reasons else 0.0
    value += 8.0 if "strong_vote_low_score" in reasons else 0.0
    value += 8.0 if "missed_student_best_candidate" in reasons else 0.0
    value += 4.0 if "missed_student_second_candidate" in reasons else 0.0
    value += 5.0 if "aggregate_vote_disagreement" in reasons else 0.0
    if margin >= MARGIN_THRESHOLD:
        value += 3.0
    return round(value, 6)


def select_unresolved_tracks(
    tracklet_df: pd.DataFrame,
    *,
    false_negative_rolls: set[str],
    config: UnresolvedSelectionConfig | None = None,
) -> pd.DataFrame:
    config = config or UnresolvedSelectionConfig()
    checkpoint_cap, camera_cap, zone_cap = config.resolved_caps()
    tracks = prepare_tracklet_diagnostics(tracklet_df)
    pool = tracks[~tracks["Tracklet_Accepted"].map(_yes)].copy()
    pool["Quality_Band"] = pool.apply(_track_quality_band, axis=1)
    pool["Primary_Zone"] = pool["Zone_IDs"].map(_primary_zone)
    pool["Evidence_Fingerprint"] = pool["Selected_Observation_IDs"].map(
        lambda value: "|".join(sorted(_split_values(value)))
    )
    pool = pool[
        pool["Selected_Observation_Count"].ge(2)
        & pool["Embedding_Count"].ge(2)
        & pool["Observation_Count"].ge(2)
        & pool["Evidence_Fingerprint"].ne("")
        & pool["Quality_Band"].ne("low")
    ].copy()
    if pool.empty:
        raise RecallAnalysisError("No useful unresolved tracks remain after evidence-quality filtering")

    normalized_false_negatives = {canonical_roll(value) for value in false_negative_rolls}
    pool["Selection_Reasons"] = pool.apply(
        lambda row: "; ".join(_selection_reasons(row, normalized_false_negatives)), axis=1
    )
    pool["Selection_Score"] = pool.apply(
        lambda row: _selection_score(row, _split_values(row["Selection_Reasons"])), axis=1
    )
    pool = pool.sort_values(
        ["Selection_Score", "Quality_Weight_Median", "Observation_Count", "Tracklet_ID"],
        ascending=[False, False, False, True],
    ).drop_duplicates("Evidence_Fingerprint", keep="first")

    selected_ids: list[str] = []
    selected_set: set[str] = set()
    candidate_counts: Counter[str] = Counter()
    checkpoint_counts: Counter[str] = Counter()
    camera_counts: Counter[str] = Counter()
    zone_counts: Counter[str] = Counter()

    def try_add(row: pd.Series) -> bool:
        tracklet_id = _text(row["Tracklet_ID"])
        candidate = _text(row["Candidate_Canonical_Roll"]) or "no_candidate"
        checkpoint = _text(row["Checkpoint_ID"]) or "unknown"
        camera = _text(row["Camera_ID"]) or "unknown"
        zone = _text(row["Primary_Zone"]) or "full_frame"
        if tracklet_id in selected_set:
            return False
        if candidate_counts[candidate] >= config.candidate_cap:
            return False
        if checkpoint_counts[checkpoint] >= checkpoint_cap:
            return False
        if camera_counts[camera] >= camera_cap:
            return False
        if zone_counts[zone] >= zone_cap:
            return False
        selected_ids.append(tracklet_id)
        selected_set.add(tracklet_id)
        candidate_counts[candidate] += 1
        checkpoint_counts[checkpoint] += 1
        camera_counts[camera] += 1
        zone_counts[zone] += 1
        return True

    seed_columns = ["Candidate_Canonical_Roll", "Checkpoint_ID", "Camera_ID", "Primary_Zone", "Tracklet_Diagnostic_Reason"]
    for column in seed_columns:
        for _, group in pool.groupby(column, sort=True, dropna=False):
            try_add(group.iloc[0])
            if len(selected_ids) >= config.target_tracks:
                break
        if len(selected_ids) >= config.target_tracks:
            break
    if len(selected_ids) < config.target_tracks:
        for _, row in pool.iterrows():
            try_add(row)
            if len(selected_ids) >= config.target_tracks:
                break

    selected = pool[pool["Tracklet_ID"].astype(str).isin(selected_set)].copy()
    selected = selected.sort_values(
        ["Selection_Score", "Tracklet_ID"], ascending=[False, True]
    ).reset_index(drop=True)
    selected.insert(0, "Selection_Rank", range(1, len(selected) + 1))
    return selected


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _distribution(values: Iterable[Any]) -> dict[str, Any]:
    numbers = pd.to_numeric(pd.Series(list(values), dtype="object"), errors="coerce").dropna()
    if numbers.empty:
        return {"count": 0, "min": None, "median": None, "p90": None, "max": None}
    return {
        "count": int(len(numbers)),
        "min": round(float(numbers.min()), 4),
        "median": round(float(numbers.median()), 4),
        "p90": round(float(numbers.quantile(0.90)), 4),
        "max": round(float(numbers.max()), 4),
    }


def _identifiability_by(joined: pd.DataFrame, column: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for value, group in joined.groupby(column, sort=True, dropna=False):
        reviewed = group[group["Review_Status"].ne("")]
        identifiable = reviewed[reviewed["Review_Status"].eq("identified")]
        rows.append({
            column: _text(value),
            "Tracks": int(len(group)),
            "Reviewed": int(len(reviewed)),
            "Identifiable": int(len(identifiable)),
            "Unidentifiable": int(reviewed["Review_Status"].eq("unidentifiable").sum()),
            "Mixed": int(reviewed["Review_Status"].eq("mixed_track").sum()),
            "Uncertain": int(reviewed["Review_Status"].eq("uncertain").sum()),
            "Duplicate": int(reviewed["Review_Status"].eq("duplicate").sum()),
            "Identifiable_Rate": _ratio(len(identifiable), len(reviewed)),
        })
    return pd.DataFrame(rows)


def _score_band(value: Any) -> str:
    score = _number(value)
    if score < 0.35:
        return "<0.35"
    if score < 0.40:
        return "0.35-0.40"
    if score < 0.44:
        return "0.40-0.44"
    if score < MATCH_THRESHOLD:
        return "0.44-0.48"
    return ">=0.48"


def _margin_band(value: Any) -> str:
    margin = _number(value)
    if margin < 0.03:
        return "<0.03"
    if margin < 0.05:
        return "0.03-0.05"
    if margin < MARGIN_THRESHOLD:
        return "0.05-0.08"
    return ">=0.08"


def _threshold_regions(identified: pd.DataFrame) -> list[dict[str, Any]]:
    if identified.empty:
        return []
    regions = identified.copy()
    regions["Score_Band"] = regions["Best_Score"].map(_score_band)
    regions["Margin_Band"] = regions["Margin"].map(_margin_band)
    rows: list[dict[str, Any]] = []
    for (score_band, margin_band), group in regions.groupby(["Score_Band", "Margin_Band"], sort=True):
        best_correct = int(group["Candidate_Result"].eq("best_candidate_correct").sum())
        best_wrong = int((~group["Candidate_Result"].eq("best_candidate_correct")).sum())
        rows.append({
            "score_band": score_band,
            "margin_band": margin_band,
            "identified_tracks": int(len(group)),
            "best_candidate_correct": best_correct,
            "best_candidate_wrong": best_wrong,
            "region_assessment": "unsafe_mixed" if best_correct and best_wrong else "correct_only_observed" if best_correct else "wrong_only_observed",
        })
    return rows


def _candidate_accuracy_by_rejection(identified: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for reason, group in identified.groupby("Rejection_Reason", sort=True, dropna=False):
        best = int(group["Candidate_Result"].eq("best_candidate_correct").sum())
        second = int(group["Candidate_Result"].eq("second_candidate_correct").sum())
        neither = int(group["Candidate_Result"].eq("neither_candidate_correct").sum())
        rows.append({
            "rejection_reason": _text(reason) or "unspecified",
            "identified_tracks": int(len(group)),
            "best_candidate_correct": best,
            "second_candidate_correct": second,
            "neither_candidate_correct": neither,
            "best_candidate_accuracy": _ratio(best, len(group)),
        })
    return rows


def evaluate_unresolved_predictions(
    *,
    prediction_df: pd.DataFrame,
    label_df: pd.DataFrame,
    known_rolls: set[str],
    false_negative_rolls: set[str],
    accepted_label_evidence_df: pd.DataFrame,
) -> UnresolvedValidationResult:
    required = {
        "Package_ID",
        "Review_ID",
        "Predicted_Roll",
        "Second_Roll",
        "Best_Score",
        "Second_Score",
        "Margin",
        "Checkpoint_ID",
        "Camera_ID",
    }
    missing = sorted(required.difference(prediction_df.columns))
    if missing:
        raise LabelValidationError(f"Unresolved predictions missing columns: {', '.join(missing)}")
    predictions = prediction_df.copy().fillna("")
    package_ids = set(predictions["Package_ID"].astype(str))
    if len(package_ids) != 1:
        raise LabelValidationError("Unresolved predictions must contain one Package_ID")
    labels = validate_label_dataframe(
        label_df,
        set(predictions["Review_ID"].astype(str)),
        next(iter(package_ids)),
        known_rolls,
    )
    joined = predictions.merge(labels, on=["Package_ID", "Review_ID"], how="left", validate="one_to_one")
    for column in ("Review_Status", "Actual_Roll", "Reviewer_Notes", "Zone_IDs", "Rejection_Reason"):
        if column not in joined:
            joined[column] = ""
        joined[column] = joined[column].fillna("").astype(str)
    for column in ("Best_Score", "Second_Score", "Margin", "Vote_Ratio_Pct", "Face_Width_Median", "Quality_Weight_Median"):
        if column not in joined:
            joined[column] = 0.0
        joined[column] = pd.to_numeric(joined[column], errors="coerce").fillna(0.0)
    joined["Predicted_Roll"] = joined["Predicted_Roll"].map(canonical_roll)
    joined["Second_Roll"] = joined["Second_Roll"].map(canonical_roll)
    joined["Actual_Roll"] = joined["Actual_Roll"].map(canonical_roll)
    joined["Quality_Band"] = joined.apply(_track_quality_band, axis=1)

    def candidate_result(row: pd.Series) -> str:
        if row["Review_Status"] != "identified":
            return "not_evaluable"
        if row["Actual_Roll"] == row["Predicted_Roll"]:
            return "best_candidate_correct"
        if row["Actual_Roll"] == row["Second_Roll"]:
            return "second_candidate_correct"
        return "neither_candidate_correct"

    joined["Candidate_Result"] = joined.apply(candidate_result, axis=1)
    joined["Correct_Candidate_Score"] = joined.apply(
        lambda row: row["Best_Score"] if row["Candidate_Result"] == "best_candidate_correct"
        else row["Second_Score"] if row["Candidate_Result"] == "second_candidate_correct"
        else "",
        axis=1,
    )
    joined["Correct_Candidate_Margin"] = joined.apply(
        lambda row: row["Margin"] if row["Candidate_Result"] == "best_candidate_correct"
        else row["Second_Score"] - row["Best_Score"] if row["Candidate_Result"] == "second_candidate_correct"
        else "",
        axis=1,
    )

    reviewed = joined[joined["Review_Status"].ne("")]
    identified = joined[joined["Review_Status"].eq("identified")]
    best_correct = identified[identified["Candidate_Result"].eq("best_candidate_correct")]
    second_correct = identified[identified["Candidate_Result"].eq("second_candidate_correct")]
    neither = identified[identified["Candidate_Result"].eq("neither_candidate_correct")]
    correct_rows = identified[identified["Candidate_Result"].isin({"best_candidate_correct", "second_candidate_correct"})]
    correct_below = int(pd.to_numeric(correct_rows["Correct_Candidate_Score"], errors="coerce").lt(MATCH_THRESHOLD).sum())
    passed_score_failed_margin = int(
        best_correct["Best_Score"].ge(MATCH_THRESHOLD).mul(best_correct["Margin"].lt(MARGIN_THRESHOLD)).sum()
    )
    track_logic_opportunity = int(
        best_correct["Vote_Ratio_Pct"].ge(60).mul(best_correct["Quality_Band"].isin({"high", "medium"})).sum()
    )
    wrong_near_threshold = int(
        identified[~identified["Candidate_Result"].eq("best_candidate_correct")]["Best_Score"].ge(0.40).sum()
    )

    accepted = accepted_label_evidence_df.copy().fillna("")
    for column in ("Actual_Roll", "Checkpoint_ID", "Camera_ID"):
        if column not in accepted:
            accepted[column] = ""
    accepted["Actual_Roll"] = accepted["Actual_Roll"].map(canonical_roll)
    normalized_false_negatives = {canonical_roll(roll) for roll in false_negative_rolls}
    student_rows: list[dict[str, Any]] = []
    review_complete = len(reviewed) == len(joined)
    for roll in sorted(normalized_false_negatives):
        verified = identified[identified["Actual_Roll"].eq(roll)]
        accepted_student = accepted[accepted["Actual_Roll"].eq(roll)]
        verified_checkpoints = {_text(value) for value in verified["Checkpoint_ID"] if _text(value)}
        accepted_checkpoints = {_text(value) for value in accepted_student["Checkpoint_ID"] if _text(value)}
        all_checkpoints = verified_checkpoints | accepted_checkpoints
        student_rows.append({
            "Canonical_Roll": roll,
            "Verified_Unresolved_Tracks": int(len(verified)),
            "Verified_Unresolved_Checkpoints": _joined(verified_checkpoints),
            "Verified_Unresolved_Cameras": _joined(verified["Camera_ID"]),
            "Accepted_Checkpoints": _joined(accepted_checkpoints),
            "Potential_Checkpoint_Count": len(all_checkpoints),
            "Could_Reach_Three_Checkpoints": (
                "Yes" if len(all_checkpoints) >= PRESENT_CHECKPOINTS else "No"
            ) if review_complete else "Unknown_Partial_Review",
            "CCTV_Domain_Examples_Likely_Required": (
                "Yes" if not verified.empty and verified["Candidate_Result"].eq("neither_candidate_correct").mean() >= 0.5
                else "No" if not verified.empty else "Unknown"
            ),
            "No_Visual_Evidence": (
                "No" if not verified.empty
                else "Not_Established_Bounded_Sample" if review_complete
                else "Unknown_Partial_Review"
            ),
        })

    summary = {
        "schema_version": 1,
        "review": {
            "tracks": int(len(joined)),
            "reviewed_tracks": int(len(reviewed)),
            "unreviewed_tracks": int(len(joined) - len(reviewed)),
            "completion_rate": _ratio(len(reviewed), len(joined)),
            "identifiable": int(len(identified)),
            "unidentifiable": int(reviewed["Review_Status"].eq("unidentifiable").sum()),
            "mixed": int(reviewed["Review_Status"].eq("mixed_track").sum()),
            "uncertain": int(reviewed["Review_Status"].eq("uncertain").sum()),
            "not_in_mapping": int(reviewed["Review_Status"].eq("not_in_mapping").sum()),
            "duplicate": int(reviewed["Review_Status"].eq("duplicate").sum()),
            "identifiable_rate": _ratio(len(identified), len(reviewed)),
        },
        "correct_candidate_analysis": {
            "best_candidate_correct": int(len(best_correct)),
            "second_candidate_correct": int(len(second_correct)),
            "neither_candidate_correct": int(len(neither)),
            "correct_candidate_score": _distribution(correct_rows["Correct_Candidate_Score"]),
            "correct_candidate_margin": _distribution(correct_rows["Correct_Candidate_Margin"]),
            "correct_best_vote_ratio_pct": _distribution(best_correct["Vote_Ratio_Pct"]),
            "false_candidate_dominance": int(identified[~identified["Candidate_Result"].eq("best_candidate_correct")]["Vote_Ratio_Pct"].ge(60).sum()),
            "accuracy_by_rejection_reason": _candidate_accuracy_by_rejection(identified),
        },
        "threshold_opportunity": {
            "correct_candidate_below_match_threshold": correct_below,
            "correct_best_passed_score_failed_margin": passed_score_failed_margin,
            "track_level_logic_opportunity": track_logic_opportunity,
            "known_wrong_best_candidates_at_or_above_0_40": wrong_near_threshold,
            "score_margin_regions": _threshold_regions(identified),
            "thresholds_changed": False,
        },
        "recommendation": "pending_human_labels" if not review_complete else "requires_decision_logic",
    }
    return UnresolvedValidationResult(
        tracks=joined,
        student_metrics=pd.DataFrame(student_rows),
        summary=summary,
        camera_metrics=_identifiability_by(joined, "Camera_ID"),
        checkpoint_metrics=_identifiability_by(joined, "Checkpoint_ID"),
        zone_metrics=_identifiability_by(joined, "Zone_IDs"),
        quality_metrics=_identifiability_by(joined, "Quality_Band"),
    )


def load_accepted_label_evidence(
    review_package: Path,
    labels_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    predictions, metadata, _ = load_review_package_data(Path(review_package))
    labels = pd.read_csv(labels_path, dtype=str, keep_default_na=False)
    validated = validate_label_dataframe(
        labels,
        set(predictions["Review_ID"].astype(str)),
        _text(metadata.get("package_id")),
        set(metadata.get("known_rolls", [])),
    )
    joined = predictions.merge(validated, on=["Package_ID", "Review_ID"], how="left", validate="one_to_one")
    joined = joined[joined["Review_Status"].eq("identified")].copy()
    joined["Actual_Roll"] = joined["Actual_Roll"].map(canonical_roll)
    return joined, metadata


def _single_file(directory: Path, pattern: str, label: str) -> Path:
    matches = sorted(Path(directory).glob(pattern))
    if len(matches) != 1:
        raise RecallAnalysisError(f"Expected exactly one {label} in {directory}; found {len(matches)}")
    return matches[0]


def _selection_breakdown(selection: pd.DataFrame) -> dict[str, Any]:
    def counts(column: str) -> dict[str, int]:
        return {str(key): int(value) for key, value in selection[column].value_counts().sort_index().items()}

    return {
        "selected_tracks": int(len(selection)),
        "by_camera": counts("Camera_ID"),
        "by_checkpoint": counts("Checkpoint_ID"),
        "by_rejection_reason": counts("Tracklet_Diagnostic_Reason"),
        "by_candidate": counts("Candidate_Canonical_Roll"),
        "by_zone": counts("Primary_Zone"),
        "by_quality_band": counts("Quality_Band"),
    }


def _recall_report(summary: dict[str, Any], gap: pd.DataFrame) -> str:
    categories = summary["false_negative_categories"]
    coverage = summary["embedding_roster_coverage"]
    selection = summary["unresolved_selection"]
    category_lines = "\n".join(f"- {name}: {count}" for name, count in categories.items())
    return (
        "# Phase 1.2F Recall-Gap Analysis\n\n"
        "## Scope\n\n"
        "This report analyzes existing Phase 1.2D/1.2E artifacts. Recognition thresholds, the three-checkpoint "
        "attendance rule, and official frame-mode attendance were not changed. Initial model-derived categories "
        "remain provisional until the unresolved blind review is complete.\n\n"
        "## Embedding Coverage\n\n"
        f"- CVO roster students: {coverage['roster_students']}\n"
        f"- Embedding available: {coverage['embedding_available']}\n"
        f"- Missing embeddings: {coverage['missing_count']} ({', '.join(coverage['missing_rolls']) or 'none'})\n"
        f"- Annotated keys canonicalized: {coverage['annotated_key_count']}\n"
        f"- Canonical collisions: {coverage['collision_count']}\n\n"
        "## Current Student Outcomes\n\n"
        f"- Actual present: {len(gap)}\n"
        f"- Present under three-checkpoint evidence: {int(gap['False_Negative_Category'].eq('present_confirmed').sum())}\n"
        f"- False negatives: {int(gap['Current_Outcome'].eq('false_negative').sum())}\n\n"
        f"{category_lines}\n\n"
        "## Unresolved Review Selection\n\n"
        f"- Selected tracks: {selection['selected_tracks']}\n"
        f"- Camera distribution: {json.dumps(selection['by_camera'], sort_keys=True)}\n"
        f"- Checkpoint distribution: {json.dumps(selection['by_checkpoint'], sort_keys=True)}\n"
        f"- Rejection distribution: {json.dumps(selection['by_rejection_reason'], sort_keys=True)}\n\n"
        "## Decision\n\n"
        "No final A-E engineering recommendation is made yet. Complete the unresolved blind review, then run "
        "the unresolved evaluator.\n"
    )


UNRESOLVED_PRIVATE_COLUMNS = {
    "Rejection_Reason": "Tracklet_Diagnostic_Reason",
    "Matcher_Reason": "Tracklet_Matcher_Reason",
    "Tracklet_Eligible": "Tracklet_Eligible",
    "Tracklet_Quality_Rejection": "Tracklet_Quality_Rejection",
    "Embedding_Count": "Embedding_Count",
    "Consistent_Embedding_Count": "Consistent_Embedding_Count",
    "Inconsistent_Embedding_Count": "Inconsistent_Embedding_Count",
    "Vote_Ratio_Pct": "Dominant_Frame_Best_Share_Pct",
    "Dominant_Frame_Best_Roll": "Dominant_Frame_Best_Roll",
    "Quality_Weight_Median": "Quality_Weight_Median",
    "Face_Width_Median": "Face_Width_Median",
    "Usable_Observation_Count": "Usable_Observation_Count",
    "Marginal_Observation_Count": "Marginal_Observation_Count",
    "Unusable_Observation_Count": "Unusable_Observation_Count",
    "Zone_IDs": "Zone_IDs",
    "Pairwise_Similarity_Median": "Pairwise_Similarity_Median",
    "Selection_Rank": "Selection_Rank",
    "Selection_Score": "Selection_Score",
    "Selection_Reasons": "Selection_Reasons",
    "Quality_Band": "Quality_Band",
    "Primary_Zone": "Primary_Zone",
}


def run_recall_gap_analysis(
    *,
    diagnostic_run: Path,
    accepted_review_package: Path,
    accepted_labels: Path,
    actual_present_path: Path,
    student_map_path: Path,
    embedding_summary_path: Path,
    embeddings_path: Path,
    output_dir: Path | None = None,
    video_root: Path | None = None,
    subject_abbr: str = "CVO",
    selection_config: UnresolvedSelectionConfig | None = None,
) -> RecallAnalysisRunResult:
    diagnostic_run = Path(diagnostic_run)
    tracklet_path = _single_file(diagnostic_run, "tracklet_diagnostics_*.csv", "tracklet diagnostics CSV")
    observation_path = _single_file(diagnostic_run, "tracklet_observations_*.csv", "tracklet observations CSV")
    tracklets = pd.read_csv(tracklet_path, dtype=str, keep_default_na=False)
    observations = pd.read_csv(observation_path, dtype=str, keep_default_na=False)
    roster, _, _ = load_student_mapping(Path(student_map_path), subject_abbr)
    subject_roster = [row for row in roster if row["in_subject_roster"]]
    actual_present = load_actual_present(Path(actual_present_path))
    accepted_evidence, _ = load_accepted_label_evidence(accepted_review_package, accepted_labels)

    embedding_summary = pd.read_csv(embedding_summary_path, dtype=str, keep_default_na=False)
    embedding_counts = load_embedding_counts(embeddings_path)
    coverage = build_embedding_roster_coverage(
        subject_roster,
        embedding_summary,
        embedding_counts=embedding_counts,
    )
    gap = build_student_recall_gap(
        roster=subject_roster,
        actual_present_rolls=actual_present,
        embedding_coverage_df=coverage,
        tracklet_df=tracklets,
        accepted_label_evidence_df=accepted_evidence,
    )
    false_negative_rolls = set(
        gap.loc[gap["Current_Outcome"].eq("false_negative"), "Canonical_Roll"].astype(str)
    )
    selection = select_unresolved_tracks(
        tracklets,
        false_negative_rolls=false_negative_rolls,
        config=selection_config,
    )

    if output_dir is None:
        output_dir = diagnostic_run / datetime.now().strftime("recall_analysis_%Y%m%d_%H%M%S_%f")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    gap.to_csv(output_dir / "student_recall_gap.csv", index=False)
    coverage.to_csv(output_dir / "embedding_roster_coverage.csv", index=False)
    selection.to_csv(output_dir / "unresolved_review_selection.csv", index=False)

    if video_root is None:
        source = _text(tracklets.iloc[0].get("Input_Source_Path")) if not tracklets.empty else ""
        if not source:
            raise RecallAnalysisError("Video root is missing; pass --video-dir")
        video_root = Path(source)
        if not video_root.is_absolute():
            video_root = Path(__file__).resolve().parents[2] / video_root

    review_package = export_review_package(
        tracklet_df=selection,
        observation_df=observations,
        video_root=video_root,
        output_root=output_dir,
        student_map_path=student_map_path,
        subject_abbr=subject_abbr,
        diagnostic_run_id=diagnostic_run.name,
        evidence_count=5,
        crop_padding=0.45,
        selected_tracklet_ids=selection["Tracklet_ID"].astype(str).tolist(),
        review_kind="unresolved",
        hidden_extra_columns=UNRESOLVED_PRIVATE_COLUMNS,
        include_context=True,
        context_padding=2.0,
        package_prefix="unresolved_tracklet_review",
        source_files={
            "tracklet_diagnostics": tracklet_path,
            "tracklet_observations": observation_path,
            "accepted_labels": Path(accepted_labels),
            "actual_present": Path(actual_present_path),
            "embedding_summary": Path(embedding_summary_path),
            "student_map": Path(student_map_path),
        },
    )

    missing_rows = coverage[coverage["Embedding_Available"].eq("No")]
    categories = {
        str(key): int(value)
        for key, value in gap["False_Negative_Category"].value_counts().sort_index().items()
    }
    summary = {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "session_id": _text(tracklets.iloc[0].get("Session_ID")) if not tracklets.empty else "",
        "diagnostic_run": str(diagnostic_run.resolve()),
        "recognition_rerun_required": False,
        "official_attendance_changed": False,
        "match_threshold": MATCH_THRESHOLD,
        "margin_threshold": MARGIN_THRESHOLD,
        "present_checkpoint_rule": PRESENT_CHECKPOINTS,
        "embedding_roster_coverage": {
            "roster_students": int(len(coverage)),
            "embedding_available": int(coverage["Embedding_Available"].eq("Yes").sum()),
            "missing_count": int(len(missing_rows)),
            "missing_rolls": sorted(missing_rows["Canonical_Roster_Roll"].astype(str).tolist()),
            "annotated_key_count": int(coverage["Canonical_Mapping_Status"].eq("canonicalized_annotation").sum()),
            "collision_count": int(coverage["Collision_Status"].eq("collision").sum()),
        },
        "false_negative_categories": categories,
        "actual_present_students": int(len(gap)),
        "present_confirmed": int(gap["False_Negative_Category"].eq("present_confirmed").sum()),
        "false_negatives": int(gap["Current_Outcome"].eq("false_negative").sum()),
        "unresolved_selection": _selection_breakdown(selection),
        "unresolved_review_package": str(review_package.root.resolve()),
        "next_recommendation": "pending_unresolved_human_labels",
    }
    (output_dir / "false_negative_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=True), encoding="utf-8"
    )
    (output_dir / "recall_gap_report.md").write_text(_recall_report(summary, gap), encoding="utf-8")
    return RecallAnalysisRunResult(
        output_dir=output_dir,
        review_package=review_package,
        student_recall_gap=gap,
        embedding_coverage=coverage,
        unresolved_selection=selection,
        summary=summary,
    )


def _completed_recommendation(
    result: UnresolvedValidationResult,
    *,
    embedding_missing_false_negatives: int,
    total_false_negatives: int,
) -> str:
    review = result.summary["review"]
    if review["reviewed_tracks"] < review["tracks"]:
        return "pending_human_labels"
    meaningful_missing = max(1, math.ceil(total_false_negatives * 0.10))
    if embedding_missing_false_negatives >= meaningful_missing:
        return "E_embedding_roster_repair"
    identifiable_rate = review.get("identifiable_rate") or 0.0
    if identifiable_rate < 0.40:
        return "D_hardware_camera_pilot"
    candidate = result.summary["correct_candidate_analysis"]
    threshold = result.summary["threshold_opportunity"]
    identifiable = max(1, review["identifiable"])
    best_share = candidate["best_candidate_correct"] / identifiable
    neither_share = candidate["neither_candidate_correct"] / identifiable
    if (
        best_share >= 0.60
        and threshold["track_level_logic_opportunity"] >= max(3, math.ceil(identifiable * 0.30))
        and threshold["known_wrong_best_candidates_at_or_above_0_40"] <= max(1, math.floor(identifiable * 0.10))
    ):
        return "B_tracklet_acceptance_calibration"
    if neither_share >= 0.35 or threshold["correct_candidate_below_match_threshold"] >= math.ceil(identifiable * 0.50):
        return "A_cctv_domain_dataset_adaptation"
    return "C_zone_sampling_association_improvements"


def _recovery_report(summary: dict[str, Any]) -> str:
    review = summary["review"]
    candidate = summary["correct_candidate_analysis"]
    threshold = summary["threshold_opportunity"]
    context = summary.get("evaluation_context", {})
    context_lines = ""
    if context:
        context_lines = (
            f"- Roster context: {context.get('mode', 'unknown')}\n"
            f"- Authoritative roster students: {context.get('roster_students', 'not refreshed')}\n"
            f"- Actual-present students: {context.get('actual_present_students', 'not refreshed')}\n"
            f"- Embedding coverage: {context.get('embedding_available', 'not refreshed')}/"
            f"{context.get('roster_students', 'not refreshed')}\n"
        )
    return (
        "# Phase 1.2F Unresolved-Track Validation\n\n"
        f"- Reviewed: {review['reviewed_tracks']}/{review['tracks']}\n"
        f"- Identifiable: {review['identifiable']} ({review['identifiable_rate']})\n"
        f"- Best candidate correct: {candidate['best_candidate_correct']}\n"
        f"- Second candidate correct: {candidate['second_candidate_correct']}\n"
        f"- Neither candidate correct: {candidate['neither_candidate_correct']}\n"
        f"- Correct candidate below 0.48: {threshold['correct_candidate_below_match_threshold']}\n"
        f"- Correct best candidate passed score but failed 0.08 margin: {threshold['correct_best_passed_score_failed_margin']}\n"
        f"{context_lines}"
        f"- Recommendation: {summary['recommendation']}\n\n"
        "Thresholds, official attendance, and the three-checkpoint rule were not changed.\n"
    )



def _refresh_authoritative_recall_context(
    *,
    recall_analysis_dir: Path,
    accepted_label_evidence_df: pd.DataFrame,
    student_map_path: Path,
    subject_abbr: str,
    actual_present_path: Path,
    embedding_summary_path: Path,
    embeddings_path: Path,
) -> tuple[set[str], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Rebuild roster-sensitive recall context without rerunning recognition.

    Historical blind-review packages intentionally preserve the mapping snapshot that
    existed when they were exported. After an authoritative roster repair, that
    snapshot may be stale. This helper validates labels against the current subject
    roster and recomputes embedding coverage / false-negative context from the
    existing tracklet diagnostics only. No review evidence, recognition threshold,
    embedding database, or attendance output is modified.
    """
    recall_analysis_dir = Path(recall_analysis_dir)
    diagnostic_run = recall_analysis_dir.parent
    tracklet_path = _single_file(
        diagnostic_run,
        "tracklet_diagnostics_*.csv",
        "tracklet diagnostics CSV",
    )
    tracklets = pd.read_csv(tracklet_path, dtype=str, keep_default_na=False)

    roster, _, subject_rolls_raw = load_student_mapping(Path(student_map_path), subject_abbr)
    subject_roster = [row for row in roster if row.get("in_subject_roster")]
    subject_rolls = {canonical_roll(value) for value in subject_rolls_raw if canonical_roll(value)}
    if not subject_roster or not subject_rolls:
        raise RecallAnalysisError(
            f"Authoritative subject roster is empty for {str(subject_abbr).strip().upper() or 'subject'}"
        )

    actual_present = {
        canonical_roll(value)
        for value in load_actual_present(Path(actual_present_path))
        if canonical_roll(value)
    }
    unknown_present = sorted(actual_present.difference(subject_rolls))
    if unknown_present:
        raise RecallAnalysisError(
            "Actual-present roster contains students outside the authoritative subject roster: "
            + ", ".join(unknown_present)
        )

    canonical_roster_rows = {canonical_roll(row.get("roll")) for row in subject_roster}
    if canonical_roster_rows != subject_rolls:
        raise RecallAnalysisError(
            "Authoritative subject roster canonicalization is inconsistent between roster rows and roll set"
        )

    embedding_summary = pd.read_csv(embedding_summary_path, dtype=str, keep_default_na=False)
    embedding_counts = load_embedding_counts(embeddings_path)
    coverage = build_embedding_roster_coverage(
        subject_roster,
        embedding_summary,
        embedding_counts=embedding_counts,
    )
    gap = build_student_recall_gap(
        roster=subject_roster,
        actual_present_rolls=actual_present,
        embedding_coverage_df=coverage,
        tracklet_df=tracklets,
        accepted_label_evidence_df=accepted_label_evidence_df,
    )

    context = {
        "mode": "authoritative_current_roster",
        "subject_abbr": str(subject_abbr).strip().upper(),
        "diagnostic_run": str(diagnostic_run.resolve()),
        "tracklet_diagnostics": str(tracklet_path.resolve()),
        "student_map": str(Path(student_map_path).resolve()),
        "actual_present": str(Path(actual_present_path).resolve()),
        "embedding_summary": str(Path(embedding_summary_path).resolve()),
        "embeddings": str(Path(embeddings_path).resolve()),
        "roster_students": len(subject_rolls),
        "actual_present_students": len(actual_present),
        "embedding_available": int(coverage["Embedding_Available"].eq("Yes").sum()),
        "missing_embedding_rolls": sorted(
            coverage.loc[coverage["Embedding_Available"].ne("Yes"), "Canonical_Roster_Roll"]
            .astype(str)
            .map(canonical_roll)
            .tolist()
        ),
        "recognition_rerun_required": False,
        "thresholds_changed": False,
        "attendance_rule_changed": False,
    }
    return subject_rolls, coverage, gap, context

def evaluate_unresolved_review_package(
    *,
    review_package: Path,
    labels_path: Path,
    recall_analysis_dir: Path,
    accepted_review_package: Path,
    accepted_labels: Path,
    output_dir: Path | None = None,
    student_map_path: Path | None = None,
    subject_abbr: str = "CVO",
    actual_present_path: Path | None = None,
    embedding_summary_path: Path | None = None,
    embeddings_path: Path | None = None,
) -> tuple[UnresolvedValidationResult, Path]:
    predictions, metadata, manifest = load_review_package_data(Path(review_package))
    if metadata.get("review_kind") != "unresolved" or manifest.get("review_kind") != "unresolved":
        raise LabelValidationError("The supplied package is not an unresolved-track review package")
    labels = pd.read_csv(labels_path, dtype=str, keep_default_na=False)
    accepted_evidence, _ = load_accepted_label_evidence(accepted_review_package, accepted_labels)

    authoritative_inputs = (
        student_map_path,
        actual_present_path,
        embedding_summary_path,
        embeddings_path,
    )
    authoritative_count = sum(value is not None for value in authoritative_inputs)
    if authoritative_count not in {0, len(authoritative_inputs)}:
        raise RecallAnalysisError(
            "Authoritative unresolved evaluation requires student map, actual-present roster, "
            "embedding summary, and embedding database together"
        )

    historical_known_rolls = {canonical_roll(value) for value in metadata.get("known_rolls", [])}
    if authoritative_count:
        known_rolls, coverage, gap, evaluation_context = _refresh_authoritative_recall_context(
            recall_analysis_dir=Path(recall_analysis_dir),
            accepted_label_evidence_df=accepted_evidence,
            student_map_path=Path(student_map_path),
            subject_abbr=subject_abbr,
            actual_present_path=Path(actual_present_path),
            embedding_summary_path=Path(embedding_summary_path),
            embeddings_path=Path(embeddings_path),
        )
        evaluation_context["historical_package_known_rolls"] = len(historical_known_rolls)
        evaluation_context["authoritative_known_rolls"] = len(known_rolls)
        evaluation_context["added_since_package_export"] = sorted(known_rolls.difference(historical_known_rolls))
        evaluation_context["removed_since_package_export"] = sorted(historical_known_rolls.difference(known_rolls))
        false_negative_rows = gap[gap["Current_Outcome"].eq("false_negative")]
        false_negative_rolls = set(false_negative_rows["Canonical_Roll"].astype(str))
        embedding_missing = int(false_negative_rows["False_Negative_Category"].eq("embedding_missing").sum())
    else:
        gap_path = Path(recall_analysis_dir) / "student_recall_gap.csv"
        if not gap_path.is_file():
            raise RecallAnalysisError(f"Recall analysis is missing student_recall_gap.csv: {recall_analysis_dir}")
        gap = pd.read_csv(gap_path, dtype=str, keep_default_na=False)
        coverage_path = Path(recall_analysis_dir) / "embedding_roster_coverage.csv"
        coverage = (
            pd.read_csv(coverage_path, dtype=str, keep_default_na=False)
            if coverage_path.is_file()
            else pd.DataFrame()
        )
        false_negative_rows = gap[gap["Current_Outcome"].eq("false_negative")]
        false_negative_rolls = set(false_negative_rows["Canonical_Roll"].astype(str))
        embedding_missing = int(false_negative_rows["False_Negative_Category"].eq("embedding_missing").sum())
        known_rolls = historical_known_rolls
        evaluation_context = {
            "mode": "historical_package_snapshot",
            "historical_package_known_rolls": len(historical_known_rolls),
            "warning": (
                "Roster-sensitive metrics use the package-era mapping. Pass authoritative roster inputs "
                "after any roster correction."
            ),
            "recognition_rerun_required": False,
            "thresholds_changed": False,
            "attendance_rule_changed": False,
        }

    result = evaluate_unresolved_predictions(
        prediction_df=predictions,
        label_df=labels,
        known_rolls=known_rolls,
        false_negative_rolls=false_negative_rolls,
        accepted_label_evidence_df=accepted_evidence,
    )
    result.summary["human_identifiability_by_camera"] = result.camera_metrics.to_dict("records")
    result.summary["human_identifiability_by_checkpoint"] = result.checkpoint_metrics.to_dict("records")
    result.summary["human_identifiability_by_zone"] = result.zone_metrics.to_dict("records")
    result.summary["human_identifiability_by_quality_band"] = result.quality_metrics.to_dict("records")
    result.summary["embedding_missing_false_negatives"] = embedding_missing
    result.summary["evaluation_context"] = evaluation_context
    result.summary["recommendation"] = _completed_recommendation(
        result,
        embedding_missing_false_negatives=embedding_missing,
        total_false_negatives=len(false_negative_rows),
    )

    if output_dir is None:
        output_dir = Path(review_package) / "evaluation" / datetime.now().strftime("unresolved_validation_%Y%m%d_%H%M%S_%f")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    result.tracks.to_csv(output_dir / "unresolved_validation_tracks.csv", index=False)
    result.student_metrics.to_csv(output_dir / "unresolved_validation_students.csv", index=False)
    gap.to_csv(output_dir / "evaluation_student_recall_gap.csv", index=False)
    if not coverage.empty:
        coverage.to_csv(output_dir / "evaluation_embedding_roster_coverage.csv", index=False)
    (output_dir / "evaluation_context.json").write_text(
        json.dumps(json_safe(evaluation_context), indent=2, ensure_ascii=True), encoding="utf-8"
    )
    (output_dir / "unresolved_validation_summary.json").write_text(
        json.dumps(json_safe(result.summary), indent=2, ensure_ascii=True), encoding="utf-8"
    )
    (output_dir / "recall_recovery_report.md").write_text(_recovery_report(result.summary), encoding="utf-8")
    return result, output_dir
