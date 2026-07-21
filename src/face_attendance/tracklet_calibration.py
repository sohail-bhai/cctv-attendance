from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .diagnostics import finite_float, json_safe
from .recall_analysis import canonical_roll
from .tracklet_review import (
    ReviewExportError,
    load_student_mapping,
    validate_label_dataframe,
)


CALIBRATION_SCHEMA_VERSION = 1
DEFAULT_FOLDS = 5
DEFAULT_MIN_OOF_RECOVERIES = 3
DEFAULT_MIN_RECOVERY_FOLDS = 3

# These are safety floors, not fitted parameters. The recovery rule is only allowed
# to act on multi-frame tracklets with coherent embedding evidence and agreement
# between the aggregate top candidate and the dominant frame-level vote.
MIN_SELECTED_OBSERVATIONS = 3
MIN_CONSISTENT_EMBEDDINGS = 3
REQUIRE_DOMINANT_AGREEMENT = True
REQUIRE_SUBJECT_ROSTER_CANDIDATE = True

SCORE_GRID = (0.38, 0.40, 0.42, 0.44, 0.46, 0.48)
MARGIN_GRID = (0.03, 0.05, 0.08, 0.10)
VOTE_GRID = (50.0, 60.0, 70.0, 80.0, 90.0)
OBSERVATION_GRID = (3, 5, 8)
CONSISTENCY_GRID = (0.50, 0.60, 0.75, 0.90)
PAIRWISE_GRID = (0.45, 0.60, 0.70)


class TrackletCalibrationError(ValueError):
    pass


@dataclass(frozen=True)
class CalibrationCandidate:
    min_best_score: float
    min_margin: float
    min_vote_ratio_pct: float
    min_observation_count: int
    min_consistency_ratio: float
    min_pairwise_similarity_median: float
    min_selected_observations: int = MIN_SELECTED_OBSERVATIONS
    min_consistent_embeddings: int = MIN_CONSISTENT_EMBEDDINGS
    require_dominant_agreement: bool = REQUIRE_DOMINANT_AGREEMENT
    require_subject_roster_candidate: bool = REQUIRE_SUBJECT_ROSTER_CANDIDATE

    @property
    def candidate_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return "cal-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class CalibrationRunResult:
    output_dir: Path
    summary: dict[str, Any]
    candidate: CalibrationCandidate
    benchmark: pd.DataFrame
    out_of_fold: pd.DataFrame


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value: Any) -> str:
    return str(value or "").strip()


def _yes(value: Any) -> bool:
    return _text(value).lower() in {"yes", "true", "1"}


def _number(value: Any, default: float = 0.0) -> float:
    return finite_float(value, default)


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _single_file(directory: Path, pattern: str, label: str) -> Path:
    matches = sorted(Path(directory).glob(pattern))
    if len(matches) != 1:
        raise TrackletCalibrationError(
            f"Expected exactly one {label} matching {pattern} in {directory}, found {len(matches)}"
        )
    return matches[0]


def _load_package_snapshot(
    package_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load package metadata without requiring archived evidence images.

    The calibration benchmark is built from already completed, validated labels.
    Hidden predictions, manifest IDs, package IDs, and any evidence images that are
    present are still integrity-checked. Missing evidence files are reported in the
    manifest instead of blocking analysis of a deliberately trimmed project archive.
    """
    package_dir = Path(package_dir)
    predictions_path = package_dir / "private" / "hidden_predictions.csv"
    metadata_path = package_dir / "private" / "export_metadata.json"
    manifest_path = package_dir / "reviewer" / "review_manifest.json"
    required = (predictions_path, metadata_path, manifest_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise TrackletCalibrationError(
            "Review package is missing required snapshot files: " + ", ".join(missing)
        )

    predictions = pd.read_csv(predictions_path, dtype=str, keep_default_na=False)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    package_id = _text(metadata.get("package_id"))
    if not package_id:
        raise TrackletCalibrationError(f"Review package metadata has no package_id: {package_dir}")
    prediction_ids = set(predictions.get("Package_ID", pd.Series(dtype=str)).astype(str))
    if prediction_ids != {package_id} or _text(manifest.get("package_id")) != package_id:
        raise TrackletCalibrationError(f"Review package identifiers do not match: {package_dir}")
    manifest_ids = {
        _text(item.get("review_id")) for item in manifest.get("review_items", []) if _text(item.get("review_id"))
    }
    hidden_ids = set(predictions.get("Review_ID", pd.Series(dtype=str)).astype(str))
    if manifest_ids != hidden_ids:
        raise TrackletCalibrationError(
            f"Review manifest IDs do not match hidden prediction IDs: {package_dir}"
        )
    if predictions["Review_ID"].duplicated().any():
        raise TrackletCalibrationError(f"Review package contains duplicate Review_ID values: {package_dir}")
    if predictions.get("Tracklet_ID", pd.Series(dtype=str)).duplicated().any():
        raise TrackletCalibrationError(f"Review package contains duplicate Tracklet_ID values: {package_dir}")

    evidence_dir = package_dir / "reviewer" / "evidence"
    present = 0
    missing_count = 0
    verified = 0
    for row in predictions.to_dict("records"):
        expected_hash = _text(row.get("Evidence_SHA256"))
        if not expected_hash:
            raise TrackletCalibrationError(
                f"{package_dir.name}/{row.get('Review_ID')} has no evidence fingerprint"
            )
        evidence_path = evidence_dir / f"{row['Review_ID']}.jpg"
        if not evidence_path.is_file():
            missing_count += 1
            continue
        present += 1
        if _sha256_file(evidence_path) != expected_hash:
            raise TrackletCalibrationError(
                f"Evidence integrity check failed for {package_dir.name}/{row['Review_ID']}"
            )
        verified += 1

    integrity = {
        "package_id": package_id,
        "package_dir": str(package_dir.resolve()),
        "hidden_predictions_sha256": _sha256_file(predictions_path),
        "export_metadata_sha256": _sha256_file(metadata_path),
        "review_manifest_sha256": _sha256_file(manifest_path),
        "evidence_expected": int(len(predictions)),
        "evidence_present": present,
        "evidence_verified": verified,
        "evidence_missing_from_archive": missing_count,
        "evidence_integrity": "verified_all_present" if verified == len(predictions) else "verified_present_files",
    }
    return predictions, metadata, manifest, integrity


def _coalesce_numeric(frame: pd.DataFrame, hidden: str, diagnostic: str) -> pd.Series:
    hidden_values = pd.to_numeric(frame.get(hidden, pd.Series(index=frame.index, dtype=float)), errors="coerce")
    diagnostic_values = pd.to_numeric(
        frame.get(diagnostic, pd.Series(index=frame.index, dtype=float)), errors="coerce"
    )
    return hidden_values.where(hidden_values.notna(), diagnostic_values).fillna(0.0)


def _assert_close_columns(
    frame: pd.DataFrame,
    hidden_column: str,
    diagnostic_column: str,
    *,
    tolerance: float = 0.00011,
) -> None:
    if hidden_column not in frame or diagnostic_column not in frame:
        return
    hidden = pd.to_numeric(frame[hidden_column], errors="coerce")
    diagnostic = pd.to_numeric(frame[diagnostic_column], errors="coerce")
    comparable = hidden.notna() & diagnostic.notna()
    if not comparable.any():
        return
    mismatch = (hidden[comparable] - diagnostic[comparable]).abs().gt(tolerance)
    if mismatch.any():
        example = frame.loc[mismatch[mismatch].index[0], "Tracklet_ID"]
        raise TrackletCalibrationError(
            f"Hidden prediction and tracklet diagnostics disagree for {example}: "
            f"{hidden_column} vs {diagnostic_column}"
        )


def _review_status_class(status: str, top1_correct: bool, second_correct: bool) -> str:
    status = _text(status)
    if status == "identified":
        if top1_correct:
            return "correct_top1"
        if second_correct:
            return "correct_second_only"
        return "wrong_top_two"
    return status or "unreviewed"


def build_calibration_benchmark(
    *,
    diagnostic_run: Path,
    accepted_review_package: Path,
    accepted_labels_path: Path,
    unresolved_review_package: Path,
    unresolved_labels_path: Path,
    student_map_path: Path,
    subject_abbr: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    diagnostic_run = Path(diagnostic_run)
    if not diagnostic_run.is_dir():
        raise TrackletCalibrationError(f"Diagnostic run directory not found: {diagnostic_run}")
    tracklet_path = _single_file(diagnostic_run, "tracklet_diagnostics_*.csv", "tracklet diagnostics CSV")
    diagnostics = pd.read_csv(tracklet_path, dtype=str, keep_default_na=False)
    if diagnostics.get("Tracklet_ID", pd.Series(dtype=str)).duplicated().any():
        raise TrackletCalibrationError("Tracklet diagnostics contains duplicate Tracklet_ID values")

    try:
        _, _, subject_rolls_raw = load_student_mapping(Path(student_map_path), subject_abbr)
    except ReviewExportError as exc:
        raise TrackletCalibrationError(str(exc)) from exc
    subject_rolls = {canonical_roll(value) for value in subject_rolls_raw if canonical_roll(value)}
    if not subject_rolls:
        raise TrackletCalibrationError(
            f"Authoritative subject roster is empty for {_text(subject_abbr).upper() or 'subject'}"
        )

    accepted_predictions, accepted_metadata, _, accepted_integrity = _load_package_snapshot(
        Path(accepted_review_package)
    )
    unresolved_predictions, unresolved_metadata, _, unresolved_integrity = _load_package_snapshot(
        Path(unresolved_review_package)
    )
    expected_session = _text(accepted_metadata.get("session_id"))
    if not expected_session or _text(unresolved_metadata.get("session_id")) != expected_session:
        raise TrackletCalibrationError("Accepted and unresolved review packages are from different sessions")
    expected_subject = _text(accepted_metadata.get("subject_abbr")).upper()
    if expected_subject != _text(subject_abbr).upper() or _text(unresolved_metadata.get("subject_abbr")).upper() != expected_subject:
        raise TrackletCalibrationError("Review packages do not match the requested subject")

    package_specs = [
        ("accepted", Path(accepted_labels_path), accepted_predictions, accepted_metadata),
        ("unresolved", Path(unresolved_labels_path), unresolved_predictions, unresolved_metadata),
    ]
    joined_packages: list[pd.DataFrame] = []
    label_hashes: dict[str, str] = {}
    for source_kind, labels_path, predictions, metadata in package_specs:
        if not labels_path.is_file():
            raise TrackletCalibrationError(f"Completed labels CSV not found: {labels_path}")
        labels_raw = pd.read_csv(labels_path, dtype=str, keep_default_na=False)
        labels = validate_label_dataframe(
            labels_raw,
            set(predictions["Review_ID"].astype(str)),
            _text(metadata.get("package_id")),
            subject_rolls,
        )
        if len(labels) != len(predictions):
            raise TrackletCalibrationError(
                f"{source_kind} labels are incomplete: {len(labels)}/{len(predictions)} rows"
            )
        package = predictions.merge(
            labels,
            on=["Package_ID", "Review_ID"],
            how="left",
            validate="one_to_one",
        )
        package["Source_Kind"] = source_kind
        joined_packages.append(package)
        label_hashes[source_kind] = _sha256_file(labels_path)

    benchmark = pd.concat(joined_packages, ignore_index=True, sort=False)
    if benchmark["Tracklet_ID"].duplicated().any():
        duplicates = sorted(benchmark.loc[benchmark["Tracklet_ID"].duplicated(False), "Tracklet_ID"].unique())
        raise TrackletCalibrationError(
            "Accepted and unresolved packages overlap on Tracklet_ID: " + ", ".join(duplicates[:10])
        )
    benchmark = benchmark.merge(
        diagnostics,
        on="Tracklet_ID",
        how="left",
        validate="one_to_one",
        suffixes=("", "_Diagnostic"),
        indicator=True,
    )
    missing_diagnostics = benchmark.loc[benchmark["_merge"].ne("both"), "Tracklet_ID"].tolist()
    if missing_diagnostics:
        raise TrackletCalibrationError(
            "Reviewed tracklets are missing from tracklet diagnostics: " + ", ".join(missing_diagnostics[:10])
        )
    benchmark = benchmark.drop(columns=["_merge"])

    _assert_close_columns(benchmark, "Best_Score", "Tracklet_Best_Score")
    _assert_close_columns(benchmark, "Second_Score", "Tracklet_Second_Score")
    _assert_close_columns(benchmark, "Margin", "Tracklet_Margin")

    benchmark["Predicted_Roll"] = benchmark["Predicted_Roll"].map(canonical_roll)
    benchmark["Second_Roll"] = benchmark["Second_Roll"].map(canonical_roll)
    benchmark["Actual_Roll"] = benchmark["Actual_Roll"].map(canonical_roll)
    benchmark["Dominant_Frame_Canonical_Roll"] = benchmark["Dominant_Frame_Best_Roll"].map(canonical_roll)
    benchmark["Candidate_In_Subject_Roster"] = benchmark["Predicted_Roll"].isin(subject_rolls)
    benchmark["Dominant_Agreement"] = benchmark["Predicted_Roll"].eq(
        benchmark["Dominant_Frame_Canonical_Roll"]
    )
    benchmark["Top1_Correct"] = benchmark["Review_Status"].eq("identified") & benchmark[
        "Predicted_Roll"
    ].eq(benchmark["Actual_Roll"])
    benchmark["Second_Correct"] = benchmark["Review_Status"].eq("identified") & benchmark[
        "Second_Roll"
    ].eq(benchmark["Actual_Roll"])
    benchmark["Safety_Class"] = benchmark.apply(
        lambda row: _review_status_class(
            row.get("Review_Status", ""), bool(row["Top1_Correct"]), bool(row["Second_Correct"])
        ),
        axis=1,
    )
    benchmark["Baseline_Accepted"] = benchmark["Source_Kind"].eq("accepted")
    benchmark["Tracklet_Eligible_Bool"] = benchmark["Tracklet_Eligible"].map(_yes)

    numeric_sources = {
        "Best_Score_Value": ("Best_Score", "Tracklet_Best_Score"),
        "Second_Score_Value": ("Second_Score", "Tracklet_Second_Score"),
        "Margin_Value": ("Margin", "Tracklet_Margin"),
        "Vote_Ratio_Pct_Value": ("Vote_Ratio_Pct", "Dominant_Frame_Best_Share_Pct"),
        "Observation_Count_Value": ("Observation_Count", "Observation_Count_Diagnostic"),
        "Selected_Observation_Count_Value": (
            "Selected_Observation_Count",
            "Selected_Observation_Count_Diagnostic",
        ),
        "Embedding_Count_Value": ("Embedding_Count", "Embedding_Count_Diagnostic"),
        "Consistent_Embedding_Count_Value": (
            "Consistent_Embedding_Count",
            "Consistent_Embedding_Count_Diagnostic",
        ),
        "Inconsistent_Embedding_Count_Value": (
            "Inconsistent_Embedding_Count",
            "Inconsistent_Embedding_Count_Diagnostic",
        ),
        "Pairwise_Similarity_Median_Value": (
            "Pairwise_Similarity_Median",
            "Pairwise_Similarity_Median_Diagnostic",
        ),
        "Pairwise_Similarity_Min_Value": (
            "Pairwise_Similarity_Min",
            "Pairwise_Similarity_Min_Diagnostic",
        ),
        "Quality_Weight_Median_Value": (
            "Quality_Weight_Median",
            "Quality_Weight_Median_Diagnostic",
        ),
        "Face_Width_Median_Value": ("Face_Width_Median", "Face_Width_Median_Diagnostic"),
        "Tracklet_Duration_Seconds_Value": (
            "Tracklet_Duration_Seconds",
            "Tracklet_Duration_Seconds_Diagnostic",
        ),
    }
    for output_column, (hidden_column, diagnostic_column) in numeric_sources.items():
        benchmark[output_column] = _coalesce_numeric(benchmark, hidden_column, diagnostic_column)
    embedding_denominator = benchmark["Embedding_Count_Value"].replace(0.0, np.nan)
    benchmark["Consistency_Ratio"] = (
        benchmark["Consistent_Embedding_Count_Value"] / embedding_denominator
    ).fillna(0.0)

    accepted_rows = benchmark[benchmark["Source_Kind"].eq("accepted")]
    unresolved_rows = benchmark[benchmark["Source_Kind"].eq("unresolved")]
    if not accepted_rows["Tracklet_Accepted"].map(_yes).all():
        raise TrackletCalibrationError("Accepted review package contains a tracklet not marked accepted")
    if unresolved_rows["Tracklet_Accepted"].map(_yes).any():
        raise TrackletCalibrationError("Unresolved review package contains a tracklet marked accepted")
    if not accepted_rows["Review_Status"].eq("identified").all() or not accepted_rows["Top1_Correct"].all():
        raise TrackletCalibrationError(
            "Baseline accepted-track benchmark contains a non-identified or incorrect top-1 result; "
            "calibration must not proceed"
        )

    benchmark["Calibration_Eligible"] = (
        benchmark["Source_Kind"].eq("unresolved")
        & benchmark["Tracklet_Eligible_Bool"]
        & benchmark["Candidate_In_Subject_Roster"]
    )
    benchmark["Calibration_Target"] = np.where(
        benchmark["Top1_Correct"], "recoverable_correct_top1", "safety_negative"
    )
    benchmark["Benchmark_Row_ID"] = benchmark["Source_Kind"].str.slice(0, 3).str.upper() + ":" + benchmark[
        "Review_ID"
    ].astype(str)
    benchmark["Row_Fingerprint"] = benchmark.apply(
        lambda row: hashlib.sha256(
            "|".join(
                [
                    _text(row.get("Package_ID")),
                    _text(row.get("Review_ID")),
                    _text(row.get("Tracklet_ID")),
                    _text(row.get("Evidence_SHA256")),
                    _text(row.get("Review_Status")),
                    _text(row.get("Actual_Roll")),
                    _text(row.get("Predicted_Roll")),
                    f"{_number(row.get('Best_Score_Value')):.6f}",
                    f"{_number(row.get('Margin_Value')):.6f}",
                ]
            ).encode("utf-8")
        ).hexdigest(),
        axis=1,
    )

    selected_columns = [
        "Benchmark_Row_ID",
        "Source_Kind",
        "Package_ID",
        "Review_ID",
        "Tracklet_ID",
        "Session_ID",
        "Subject_Abbr",
        "Checkpoint_ID",
        "Camera_ID",
        "Video",
        "Review_Status",
        "Actual_Roll",
        "Predicted_Roll",
        "Second_Roll",
        "Safety_Class",
        "Top1_Correct",
        "Second_Correct",
        "Baseline_Accepted",
        "Calibration_Eligible",
        "Calibration_Target",
        "Candidate_In_Subject_Roster",
        "Tracklet_Eligible_Bool",
        "Dominant_Frame_Canonical_Roll",
        "Dominant_Agreement",
        "Best_Score_Value",
        "Second_Score_Value",
        "Margin_Value",
        "Vote_Ratio_Pct_Value",
        "Observation_Count_Value",
        "Selected_Observation_Count_Value",
        "Embedding_Count_Value",
        "Consistent_Embedding_Count_Value",
        "Inconsistent_Embedding_Count_Value",
        "Consistency_Ratio",
        "Pairwise_Similarity_Min_Value",
        "Pairwise_Similarity_Median_Value",
        "Quality_Weight_Median_Value",
        "Face_Width_Median_Value",
        "Tracklet_Duration_Seconds_Value",
        "Quality_Band",
        "Zone_IDs",
        "Evidence_SHA256",
        "Row_Fingerprint",
    ]
    for column in selected_columns:
        if column not in benchmark:
            benchmark[column] = ""
    benchmark = benchmark[selected_columns].copy()

    source_manifest = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "offline_shadow_calibration",
        "session_id": expected_session,
        "subject_abbr": expected_subject,
        "source_hashes": {
            "tracklet_diagnostics": _sha256_file(tracklet_path),
            "accepted_labels": label_hashes["accepted"],
            "unresolved_labels": label_hashes["unresolved"],
            "student_map": _sha256_file(Path(student_map_path)),
        },
        "accepted_package": accepted_integrity,
        "unresolved_package": unresolved_integrity,
        "rows": {
            "total": int(len(benchmark)),
            "accepted_reviewed": int(benchmark["Source_Kind"].eq("accepted").sum()),
            "unresolved_reviewed": int(benchmark["Source_Kind"].eq("unresolved").sum()),
            "identified": int(benchmark["Review_Status"].eq("identified").sum()),
            "top1_correct": int(benchmark["Top1_Correct"].sum()),
            "safety_negatives": int((~benchmark["Top1_Correct"]).sum()),
        },
        "authoritative_subject_roster_count": len(subject_rolls),
        "production_changes": False,
        "recognition_rerun": False,
        "thresholds_changed": False,
        "attendance_rule_changed": False,
    }
    return benchmark, source_manifest


def _candidate_grid() -> list[CalibrationCandidate]:
    candidates: list[CalibrationCandidate] = []
    for score in SCORE_GRID:
        for margin in MARGIN_GRID:
            for vote in VOTE_GRID:
                for observations in OBSERVATION_GRID:
                    for consistency in CONSISTENCY_GRID:
                        for pairwise in PAIRWISE_GRID:
                            candidates.append(
                                CalibrationCandidate(
                                    min_best_score=score,
                                    min_margin=margin,
                                    min_vote_ratio_pct=vote,
                                    min_observation_count=observations,
                                    min_consistency_ratio=consistency,
                                    min_pairwise_similarity_median=pairwise,
                                )
                            )
    return candidates


def candidate_acceptance_mask(
    frame: pd.DataFrame,
    candidate: CalibrationCandidate,
) -> pd.Series:
    mask = frame["Calibration_Eligible"].astype(bool)
    if candidate.require_subject_roster_candidate:
        mask &= frame["Candidate_In_Subject_Roster"].astype(bool)
    if candidate.require_dominant_agreement:
        mask &= frame["Dominant_Agreement"].astype(bool)
    mask &= frame["Best_Score_Value"].astype(float).ge(candidate.min_best_score)
    mask &= frame["Margin_Value"].astype(float).ge(candidate.min_margin)
    mask &= frame["Vote_Ratio_Pct_Value"].astype(float).ge(candidate.min_vote_ratio_pct)
    mask &= frame["Observation_Count_Value"].astype(float).ge(candidate.min_observation_count)
    mask &= frame["Selected_Observation_Count_Value"].astype(float).ge(
        candidate.min_selected_observations
    )
    mask &= frame["Consistent_Embedding_Count_Value"].astype(float).ge(
        candidate.min_consistent_embeddings
    )
    mask &= frame["Consistency_Ratio"].astype(float).ge(candidate.min_consistency_ratio)
    mask &= frame["Pairwise_Similarity_Median_Value"].astype(float).ge(
        candidate.min_pairwise_similarity_median
    )
    return mask


def _candidate_metrics(frame: pd.DataFrame, candidate: CalibrationCandidate) -> dict[str, Any]:
    accepted = candidate_acceptance_mask(frame, candidate)
    correct = frame["Top1_Correct"].astype(bool)
    true_recoveries = int((accepted & correct).sum())
    false_accepts = int((accepted & ~correct).sum())
    eligible_correct = int((frame["Calibration_Eligible"].astype(bool) & correct).sum())
    safety_negatives = int((frame["Calibration_Eligible"].astype(bool) & ~correct).sum())
    return {
        "candidate_id": candidate.candidate_id,
        "true_recoveries": true_recoveries,
        "false_accepts": false_accepts,
        "accepted_tracks": int(accepted.sum()),
        "eligible_correct_tracks": eligible_correct,
        "eligible_safety_negatives": safety_negatives,
        "recovery_rate": _ratio(true_recoveries, eligible_correct),
        "shadow_precision": _ratio(true_recoveries, true_recoveries + false_accepts),
    }


def _candidate_rank(candidate: CalibrationCandidate, metrics: dict[str, Any]) -> tuple[Any, ...]:
    # Zero false identity acceptance dominates every other objective. Among equally
    # safe candidates, maximize recoveries and then select the stricter rule.
    return (
        int(metrics["false_accepts"]),
        -int(metrics["true_recoveries"]),
        -candidate.min_best_score,
        -candidate.min_margin,
        -candidate.min_vote_ratio_pct,
        -candidate.min_observation_count,
        -candidate.min_consistency_ratio,
        -candidate.min_pairwise_similarity_median,
        candidate.candidate_id,
    )


def search_candidates(
    frame: pd.DataFrame,
    candidates: Iterable[CalibrationCandidate] | None = None,
) -> tuple[CalibrationCandidate, pd.DataFrame]:
    candidates = list(candidates or _candidate_grid())
    if not candidates:
        raise TrackletCalibrationError("Calibration candidate grid is empty")
    rows: list[dict[str, Any]] = []
    best_candidate: CalibrationCandidate | None = None
    best_rank: tuple[Any, ...] | None = None
    for candidate in candidates:
        metrics = _candidate_metrics(frame, candidate)
        row = {**asdict(candidate), **metrics}
        rows.append(row)
        rank = _candidate_rank(candidate, metrics)
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_candidate = candidate
    assert best_candidate is not None
    search = pd.DataFrame(rows).sort_values(
        [
            "false_accepts",
            "true_recoveries",
            "min_best_score",
            "min_margin",
            "min_vote_ratio_pct",
            "min_observation_count",
            "min_consistency_ratio",
            "min_pairwise_similarity_median",
        ],
        ascending=[True, False, False, False, False, False, False, False],
        kind="mergesort",
    )
    search.insert(0, "rank", range(1, len(search) + 1))
    return best_candidate, search


def assign_group_folds(frame: pd.DataFrame, folds: int = DEFAULT_FOLDS) -> pd.DataFrame:
    if not 3 <= folds <= 10:
        raise TrackletCalibrationError("folds must be between 3 and 10")
    calibration = frame[frame["Source_Kind"].eq("unresolved")].copy()
    if calibration.empty:
        raise TrackletCalibrationError("No unresolved benchmark rows are available for calibration")
    calibration["Group_Key"] = np.where(
        calibration["Review_Status"].eq("identified"),
        "identity:" + calibration["Actual_Roll"].astype(str),
        "safety:" + calibration["Review_Status"].astype(str) + ":" + calibration["Tracklet_ID"].astype(str),
    )
    group_stats: list[tuple[str, np.ndarray, str]] = []
    for group_key, group in calibration.groupby("Group_Key", sort=True):
        positive = int(group["Top1_Correct"].astype(bool).sum())
        negative = int(len(group) - positive)
        fingerprint = hashlib.sha256(group_key.encode("utf-8")).hexdigest()
        group_stats.append((group_key, np.array([negative, positive], dtype=int), fingerprint))
    if len(group_stats) < folds:
        raise TrackletCalibrationError(
            f"Not enough identity/safety groups for {folds} holdout folds: {len(group_stats)} groups"
        )

    # Deterministic approximation of stratified group K-fold. Entire human identities
    # remain in one fold, preventing leakage across repeated tracklets of a student.
    group_stats.sort(
        key=lambda item: (
            -float(np.std(item[1] / max(int(item[1].sum()), 1))),
            -int(item[1].sum()),
            item[2],
        )
    )
    class_totals = np.array(
        [
            int((~calibration["Top1_Correct"].astype(bool)).sum()),
            int(calibration["Top1_Correct"].astype(bool).sum()),
        ],
        dtype=float,
    )
    if np.any(class_totals <= 0):
        raise TrackletCalibrationError("Calibration benchmark needs both correct top-1 and safety-negative rows")
    fold_class_counts = np.zeros((folds, 2), dtype=int)
    fold_sizes = np.zeros(folds, dtype=int)
    assignments: dict[str, int] = {}
    for index, (group_key, class_counts, _) in enumerate(group_stats):
        if index < folds:
            selected_fold = index
        else:
            best: tuple[tuple[float, float, int, int], int] | None = None
            for fold in range(folds):
                fold_class_counts[fold] += class_counts
                fold_sizes[fold] += int(class_counts.sum())
                class_balance = float(np.mean(np.std(fold_class_counts / class_totals, axis=0)))
                size_balance = float(np.std(fold_sizes / max(len(calibration), 1)))
                key = (class_balance, size_balance, int(fold_sizes[fold]), fold)
                fold_class_counts[fold] -= class_counts
                fold_sizes[fold] -= int(class_counts.sum())
                if best is None or key < best[0]:
                    best = (key, fold)
            assert best is not None
            selected_fold = best[1]
        assignments[group_key] = selected_fold
        fold_class_counts[selected_fold] += class_counts
        fold_sizes[selected_fold] += int(class_counts.sum())

    calibration["Fold"] = calibration["Group_Key"].map(assignments).astype(int)
    fold_summary = calibration.groupby("Fold").agg(
        Tracks=("Top1_Correct", "size"),
        Correct_Top1=("Top1_Correct", "sum"),
    )
    fold_summary["Safety_Negatives"] = fold_summary["Tracks"] - fold_summary["Correct_Top1"]
    if len(fold_summary) != folds:
        raise TrackletCalibrationError("Fold assignment produced an empty holdout fold")
    if (fold_summary[["Correct_Top1", "Safety_Negatives"]].min(axis=1) <= 0).any():
        raise TrackletCalibrationError(
            "Identity-grouped fold assignment could not place both classes in every fold; "
            "collect more labeled ground truth or use fewer folds"
        )
    return calibration


def run_cross_validated_search(
    calibration: pd.DataFrame,
    *,
    folds: int = DEFAULT_FOLDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "Fold" not in calibration:
        raise TrackletCalibrationError("Calibration frame has no Fold assignments")
    candidates = _candidate_grid()
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    for fold in range(folds):
        training = calibration[calibration["Fold"].ne(fold)]
        holdout = calibration[calibration["Fold"].eq(fold)]
        if training.empty or holdout.empty:
            raise TrackletCalibrationError(f"Fold {fold} has an empty train or holdout partition")
        candidate, _ = search_candidates(training, candidates)
        training_metrics = _candidate_metrics(training, candidate)
        accepted = candidate_acceptance_mask(holdout, candidate)
        holdout_rows = holdout.copy()
        holdout_rows["OOF_Accepted"] = accepted.astype(bool)
        holdout_rows["OOF_Candidate_ID"] = candidate.candidate_id
        holdout_rows["OOF_Outcome"] = np.select(
            [
                holdout_rows["OOF_Accepted"] & holdout_rows["Top1_Correct"],
                holdout_rows["OOF_Accepted"] & ~holdout_rows["Top1_Correct"],
            ],
            ["recovered_correct", "false_accept"],
            default="remained_unresolved",
        )
        prediction_rows.append(holdout_rows)
        holdout_metrics = _candidate_metrics(holdout, candidate)
        fold_rows.append(
            {
                "fold": fold,
                **asdict(candidate),
                "candidate_id": candidate.candidate_id,
                "training_tracks": int(len(training)),
                "training_true_recoveries": training_metrics["true_recoveries"],
                "training_false_accepts": training_metrics["false_accepts"],
                "holdout_tracks": int(len(holdout)),
                "holdout_correct_top1": int(holdout["Top1_Correct"].sum()),
                "holdout_safety_negatives": int((~holdout["Top1_Correct"]).sum()),
                "holdout_true_recoveries": holdout_metrics["true_recoveries"],
                "holdout_false_accepts": holdout_metrics["false_accepts"],
                "holdout_shadow_precision": holdout_metrics["shadow_precision"],
            }
        )
    out_of_fold = pd.concat(prediction_rows, ignore_index=True).sort_values(
        ["Fold", "Benchmark_Row_ID"], kind="mergesort"
    )
    if len(out_of_fold) != len(calibration) or out_of_fold["Benchmark_Row_ID"].nunique() != len(calibration):
        raise TrackletCalibrationError("Out-of-fold predictions do not cover every unresolved benchmark row exactly once")
    return out_of_fold, pd.DataFrame(fold_rows)


def _breakdown(
    frame: pd.DataFrame,
    final_mask: pd.Series,
    oof_by_id: dict[str, bool],
    column: str,
    label: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value, group in frame.groupby(column, dropna=False, sort=True):
        group_final = final_mask.loc[group.index]
        group_oof = group["Benchmark_Row_ID"].map(oof_by_id).eq(True)
        correct = group["Top1_Correct"].astype(bool)
        rows.append(
            {
                "stratum_type": label,
                "stratum_value": _text(value) or "(blank)",
                "tracks": int(len(group)),
                "correct_top1": int(correct.sum()),
                "safety_negatives": int((~correct).sum()),
                "final_recovered": int((group_final & correct).sum()),
                "final_false_accepts": int((group_final & ~correct).sum()),
                "oof_recovered": int((group_oof & correct).sum()),
                "oof_false_accepts": int((group_oof & ~correct).sum()),
            }
        )
    return rows


def _calibration_report(summary: dict[str, Any]) -> str:
    benchmark = summary["benchmark"]
    oof = summary["out_of_fold"]
    final = summary["final_candidate"]
    recommendation = summary["recommendation"]
    candidate = summary["candidate_config"]
    return (
        "# Phase 1.2G Conservative Tracklet Acceptance Calibration\n\n"
        "## Safety boundary\n\n"
        "- Offline shadow analysis only\n"
        "- Candidate enabled: no\n"
        "- Production recognition changed: no\n"
        "- Match threshold changed: no\n"
        "- Margin threshold changed: no\n"
        "- Three-checkpoint attendance rule changed: no\n\n"
        "## Frozen reviewed benchmark\n\n"
        f"- Total reviewed tracklets: {benchmark['total_tracks']}\n"
        f"- Baseline accepted reviewed tracks: {benchmark['baseline_accepted_tracks']}\n"
        f"- Baseline accepted false identities: {benchmark['baseline_false_accepts']}\n"
        f"- Reviewed unresolved tracks: {benchmark['unresolved_tracks']}\n"
        f"- Recoverable correct top-1 unresolved tracks: {benchmark['recoverable_correct_top1']}\n"
        f"- Safety-negative unresolved tracks: {benchmark['safety_negatives']}\n\n"
        "## Identity-grouped holdout result\n\n"
        f"- Folds: {oof['folds']}\n"
        f"- Out-of-fold recovered tracks: {oof['true_recoveries']}\n"
        f"- Out-of-fold false accepts: {oof['false_accepts']}\n"
        f"- Out-of-fold shadow precision: {oof['shadow_precision']}\n"
        f"- Folds with at least one recovery: {oof['folds_with_recovery']}\n"
        f"- Folds with a false accept: {oof['folds_with_false_accept']}\n\n"
        "## Full-benchmark shadow candidate\n\n"
        f"- Candidate ID: {candidate['candidate_id']}\n"
        f"- Recovered unresolved tracks: {final['true_recoveries']}\n"
        f"- False accepts: {final['false_accepts']}\n"
        f"- Shadow precision: {final['shadow_precision']}\n"
        f"- Recovery rate among correct top-1 unresolved tracks: {final['recovery_rate']}\n\n"
        "## Decision\n\n"
        f"- Recommendation: {recommendation['decision']}\n"
        f"- Production decision: {recommendation['production_decision']}\n"
        f"- Next validation: {recommendation['next_step']}\n"
        f"- Rationale: {recommendation['rationale']}\n\n"
        "This is a single-session, reviewer-selected benchmark. A safe result here may only advance "
        "the candidate to an untouched-session shadow test; it is not production approval.\n"
    )


def run_tracklet_calibration(
    *,
    diagnostic_run: Path,
    accepted_review_package: Path,
    accepted_labels_path: Path,
    unresolved_review_package: Path,
    unresolved_labels_path: Path,
    student_map_path: Path,
    subject_abbr: str = "CVO",
    output_dir: Path | None = None,
    folds: int = DEFAULT_FOLDS,
    min_oof_recoveries: int = DEFAULT_MIN_OOF_RECOVERIES,
    min_recovery_folds: int = DEFAULT_MIN_RECOVERY_FOLDS,
) -> CalibrationRunResult:
    if min_oof_recoveries < 1:
        raise TrackletCalibrationError("min_oof_recoveries must be positive")
    if min_recovery_folds < 1 or min_recovery_folds > folds:
        raise TrackletCalibrationError("min_recovery_folds must be between 1 and folds")

    benchmark, source_manifest = build_calibration_benchmark(
        diagnostic_run=diagnostic_run,
        accepted_review_package=accepted_review_package,
        accepted_labels_path=accepted_labels_path,
        unresolved_review_package=unresolved_review_package,
        unresolved_labels_path=unresolved_labels_path,
        student_map_path=student_map_path,
        subject_abbr=subject_abbr,
    )
    calibration = assign_group_folds(benchmark, folds=folds)
    out_of_fold, fold_results = run_cross_validated_search(calibration, folds=folds)
    final_candidate, candidate_search = search_candidates(calibration)
    final_mask = candidate_acceptance_mask(calibration, final_candidate)

    oof_correct = out_of_fold["Top1_Correct"].astype(bool)
    oof_accepted = out_of_fold["OOF_Accepted"].astype(bool)
    oof_true = int((oof_accepted & oof_correct).sum())
    oof_false = int((oof_accepted & ~oof_correct).sum())
    folds_with_recovery = int(fold_results["holdout_true_recoveries"].gt(0).sum())
    folds_with_false_accept = int(fold_results["holdout_false_accepts"].gt(0).sum())
    final_true = int((final_mask & calibration["Top1_Correct"].astype(bool)).sum())
    final_false = int((final_mask & ~calibration["Top1_Correct"].astype(bool)).sum())

    if oof_false or final_false:
        decision = "reject_candidate"
        rationale = (
            "At least one human-reviewed safety-negative track was accepted in holdout or full-benchmark shadow analysis."
        )
        next_step = "Do not apply this calibration; improve embeddings or collect more discriminative evidence."
    elif oof_true < min_oof_recoveries or folds_with_recovery < min_recovery_folds:
        decision = "hold_for_more_ground_truth"
        rationale = (
            "No false accepts were observed, but the identity-grouped holdout recovery is too small or concentrated."
        )
        next_step = "Collect additional blind reviewed tracklets before choosing a candidate configuration."
    else:
        decision = "promote_to_multisession_shadow_validation"
        rationale = (
            "Zero false accepts were observed across identity-grouped holdouts and the full benchmark, "
            "with recoveries distributed across multiple folds."
        )
        next_step = (
            "Run the disabled candidate on an untouched session such as TUE_P2 or MON_P3 and blind-review every "
            "newly accepted recovery before considering any production integration."
        )

    baseline_accepted = benchmark["Baseline_Accepted"].astype(bool)
    baseline_correct = benchmark["Top1_Correct"].astype(bool)
    total_correct_top1 = int(benchmark["Top1_Correct"].sum())
    oof_by_id = dict(zip(out_of_fold["Benchmark_Row_ID"], out_of_fold["OOF_Accepted"].astype(bool)))
    final_by_id = dict(zip(calibration["Benchmark_Row_ID"], final_mask.astype(bool)))
    benchmark["Fold"] = benchmark["Benchmark_Row_ID"].map(
        dict(zip(calibration["Benchmark_Row_ID"], calibration["Fold"]))
    )
    benchmark["OOF_Accepted"] = benchmark["Benchmark_Row_ID"].map(oof_by_id).eq(True)
    benchmark["Final_Candidate_Accepted"] = benchmark["Benchmark_Row_ID"].map(final_by_id).eq(True)

    baseline_vs_candidate = pd.DataFrame(
        [
            {
                "mode": "baseline_reviewed",
                "automatic_accepts": int(baseline_accepted.sum()),
                "correct_accepts": int((baseline_accepted & baseline_correct).sum()),
                "false_accepts": int((baseline_accepted & ~baseline_correct).sum()),
                "precision": _ratio(
                    int((baseline_accepted & baseline_correct).sum()), int(baseline_accepted.sum())
                ),
                "coverage_of_reviewed_correct_top1": _ratio(
                    int((baseline_accepted & baseline_correct).sum()), total_correct_top1
                ),
            },
            {
                "mode": "baseline_plus_oof_shadow",
                "automatic_accepts": int(baseline_accepted.sum()) + int(oof_accepted.sum()),
                "correct_accepts": int((baseline_accepted & baseline_correct).sum()) + oof_true,
                "false_accepts": int((baseline_accepted & ~baseline_correct).sum()) + oof_false,
                "precision": _ratio(
                    int((baseline_accepted & baseline_correct).sum()) + oof_true,
                    int(baseline_accepted.sum()) + int(oof_accepted.sum()),
                ),
                "coverage_of_reviewed_correct_top1": _ratio(
                    int((baseline_accepted & baseline_correct).sum()) + oof_true, total_correct_top1
                ),
            },
            {
                "mode": "baseline_plus_final_candidate_benchmark",
                "automatic_accepts": int(baseline_accepted.sum()) + int(final_mask.sum()),
                "correct_accepts": int((baseline_accepted & baseline_correct).sum()) + final_true,
                "false_accepts": int((baseline_accepted & ~baseline_correct).sum()) + final_false,
                "precision": _ratio(
                    int((baseline_accepted & baseline_correct).sum()) + final_true,
                    int(baseline_accepted.sum()) + int(final_mask.sum()),
                ),
                "coverage_of_reviewed_correct_top1": _ratio(
                    int((baseline_accepted & baseline_correct).sum()) + final_true, total_correct_top1
                ),
            },
        ]
    )

    final_candidate_payload = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "candidate_id": final_candidate.candidate_id,
        "enabled": False,
        "mode": "shadow_only",
        "production_approved": False,
        "requires_multisession_validation": True,
        "applies_only_to_baseline_rejected_tracklets": True,
        "subject_abbr": _text(subject_abbr).upper(),
        "thresholds": asdict(final_candidate),
        "fixed_safety_guards": {
            "candidate_must_be_in_authoritative_subject_roster": True,
            "aggregate_candidate_must_equal_dominant_frame_candidate": True,
            "minimum_selected_observations": MIN_SELECTED_OBSERVATIONS,
            "minimum_consistent_embeddings": MIN_CONSISTENT_EMBEDDINGS,
        },
        "official_match_threshold_unchanged": 0.48,
        "official_margin_threshold_unchanged": 0.08,
        "attendance_checkpoint_rule_unchanged": 3,
    }

    summary = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "mode": "offline_shadow_calibration",
        "benchmark": {
            "total_tracks": int(len(benchmark)),
            "baseline_accepted_tracks": int(baseline_accepted.sum()),
            "baseline_false_accepts": int((baseline_accepted & ~baseline_correct).sum()),
            "unresolved_tracks": int(benchmark["Source_Kind"].eq("unresolved").sum()),
            "recoverable_correct_top1": int(calibration["Top1_Correct"].sum()),
            "safety_negatives": int((~calibration["Top1_Correct"]).sum()),
            "second_candidate_only_correct": int(calibration["Safety_Class"].eq("correct_second_only").sum()),
            "neither_top_two_correct": int(calibration["Safety_Class"].eq("wrong_top_two").sum()),
        },
        "cross_validation": {
            "strategy": "deterministic_identity_grouped_stratified_holdout",
            "folds": folds,
            "identity_leakage_prevented": True,
            "selection_objective": "minimize_false_accepts_then_maximize_recovery",
        },
        "out_of_fold": {
            "folds": folds,
            "true_recoveries": oof_true,
            "false_accepts": oof_false,
            "accepted_tracks": int(oof_accepted.sum()),
            "shadow_precision": _ratio(oof_true, oof_true + oof_false),
            "recovery_rate": _ratio(oof_true, int(calibration["Top1_Correct"].sum())),
            "folds_with_recovery": folds_with_recovery,
            "folds_with_false_accept": folds_with_false_accept,
            "unique_selected_candidate_ids": int(fold_results["candidate_id"].nunique()),
        },
        "final_candidate": {
            **_candidate_metrics(calibration, final_candidate),
            "candidate_id": final_candidate.candidate_id,
        },
        "candidate_config": final_candidate_payload,
        "recommendation": {
            "decision": decision,
            "production_decision": "keep_disabled",
            "rationale": rationale,
            "next_step": next_step,
        },
        "limitations": [
            "Only one CCTV session is represented.",
            "The unresolved package is reviewer-selection biased and is not a random sample of all rejected tracklets.",
            "Full-benchmark candidate metrics are descriptive; identity-grouped out-of-fold metrics are the primary evidence.",
            "No candidate may affect official attendance before untouched-session validation and blind review.",
        ],
        "production_changes": False,
        "recognition_rerun": False,
        "thresholds_changed": False,
        "attendance_rule_changed": False,
    }

    if output_dir is None:
        output_dir = Path(diagnostic_run) / "calibration" / datetime.now().strftime(
            "tracklet_calibration_%Y%m%d_%H%M%S_%f"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    benchmark.to_csv(output_dir / "tracklet_calibration_ground_truth.csv", index=False)
    (output_dir / "ground_truth_manifest.json").write_text(
        json.dumps(json_safe(source_manifest), indent=2, ensure_ascii=True), encoding="utf-8"
    )
    candidate_search.to_csv(output_dir / "candidate_search.csv", index=False)
    fold_results.to_csv(output_dir / "fold_configurations.csv", index=False)
    out_of_fold.to_csv(output_dir / "out_of_fold_predictions.csv", index=False)
    baseline_vs_candidate.to_csv(output_dir / "baseline_vs_candidate.csv", index=False)

    recovered = calibration.loc[final_mask].copy()
    recovered["OOF_Accepted"] = recovered["Benchmark_Row_ID"].map(oof_by_id).eq(True)
    recovered.to_csv(output_dir / "recovered_tracks.csv", index=False)
    safety_audit = calibration.loc[~calibration["Top1_Correct"].astype(bool)].copy()
    safety_audit["Final_Candidate_Accepted"] = safety_audit["Benchmark_Row_ID"].map(final_by_id).eq(True)
    safety_audit["OOF_Accepted"] = safety_audit["Benchmark_Row_ID"].map(oof_by_id).eq(True)
    safety_audit.to_csv(output_dir / "false_accept_audit.csv", index=False)

    breakdown_rows = []
    breakdown_rows.extend(_breakdown(calibration, final_mask, oof_by_id, "Camera_ID", "camera"))
    breakdown_rows.extend(_breakdown(calibration, final_mask, oof_by_id, "Checkpoint_ID", "checkpoint"))
    pd.DataFrame(breakdown_rows).to_csv(output_dir / "camera_checkpoint_breakdown.csv", index=False)

    (output_dir / "candidate_config.json").write_text(
        json.dumps(json_safe(final_candidate_payload), indent=2, ensure_ascii=True), encoding="utf-8"
    )
    (output_dir / "calibration_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=True), encoding="utf-8"
    )
    (output_dir / "calibration_report.md").write_text(_calibration_report(summary), encoding="utf-8")

    output_hashes = {
        path.name: _sha256_file(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "output_manifest.json"
    }
    output_manifest = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir.resolve()),
        "candidate_id": final_candidate.candidate_id,
        "decision": decision,
        "files_sha256": output_hashes,
        "production_changes": False,
    }
    (output_dir / "output_manifest.json").write_text(
        json.dumps(json_safe(output_manifest), indent=2, ensure_ascii=True), encoding="utf-8"
    )
    return CalibrationRunResult(
        output_dir=output_dir,
        summary=summary,
        candidate=final_candidate,
        benchmark=benchmark,
        out_of_fold=out_of_fold,
    )
