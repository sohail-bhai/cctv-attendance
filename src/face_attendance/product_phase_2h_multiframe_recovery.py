from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .mon_p4_shadow_validation import verify_source_freeze
from .tracklet_review import (
    ReviewExportError,
    _sha256_file,
    export_review_package,
    load_student_mapping,
    normalize_roll,
    validate_label_dataframe,
)

POLICY_VERSION = "product-phase-2h-multiframe-recall-recovery-v2.1"
RUN_ID = "product-2g-2026-06-22__B51__P4__CVO-7a239045"
SESSION_ID = "2026-06-22__B51__P4__CVO"
SUBJECT_ABBR = "CVO"
PRESENT_CHECKPOINTS = 3
REVIEW_CHECKPOINTS = 1
EXPECTED_TRACKLETS_TOTAL = 308
EXPECTED_ELIGIBLE_TRACKLETS = 250
EXPECTED_ACCEPTED_TRACKLETS = 31
EXPECTED_UNIQUE_ACCEPTED_IDENTITIES = 12
EXPECTED_SELECTED_PAIRS = 31
EXPECTED_REUSED_LABELS = 3
EXPECTED_PENDING_REVIEW = 28

RECOVERY_RULE_ID = "mfrr-v1-score040-margin005-vote70-obs5-pair060-repeat2"
RECOVERY_MIN_SCORE = 0.40
RECOVERY_MIN_MARGIN = 0.05
RECOVERY_MIN_DOMINANT_SHARE_PCT = 70.0
RECOVERY_MIN_OBSERVATIONS = 5
RECOVERY_MIN_SELECTED_OBSERVATIONS = 3
RECOVERY_MIN_CONSISTENT_EMBEDDINGS = 3
RECOVERY_MIN_PAIRWISE_MEDIAN = 0.60
RECOVERY_MIN_CHECKPOINT_REPETITION = 2
CURRENT_MODEL_ID = "ABL-16-ee3a4ff6"
PHASE_1_2N_SHADOW_ID = "mon-p4-shadow-695277fbd0dcd9d431e1"
PHASE_1_2N_FREEZE_ID = "source-freeze-06578150c42a1253d835"
SOURCE_FREEZE_JSON_SHA256 = "6d60252b7e70941a2466bf7a43996418939a5853d8a8123ccc8b549583396e07"
SOURCE_FREEZE_CSV_SHA256 = "5fc22e2f67b11c009b7f7af4484932635061523ca523e1acbb1554c9e3fd168d"

DIAGNOSTIC_HASHES: dict[str, str] = {
    f"checkpoint_diagnostics_{RUN_ID}.csv": "57eee323e824f7de78d03b1a1f29172b795ae3a1c52ff581652cfda8533cc771",
    f"diagnostic_summary_{RUN_ID}.json": "8e7384b64038277cb7e63dc5688c0b6a3cad488353f3457b11a22e922196cc88",
    f"face_diagnostics_{RUN_ID}.csv": "95ae091ee7e7c2874117804ef5885a73a8f9462466be436d1becf086c684b678",
    f"rejection_summary_{RUN_ID}.csv": "88b0e9e11dcdca48d7e6d2f24f1d4b24e3f1627be0e0bd0500db27a3b50aa05e",
    f"tracklet_checkpoint_comparison_{RUN_ID}.csv": "937d54838a2470274b875c8d3d00e2381237be3ce7ac818b1cf6a5acf8648c19",
    f"tracklet_comparison_{RUN_ID}.json": "36b1623ff1d4e029d18296702d8fe9c305bc22b9fae8cff6a3bdf08e4cbc77fc",
    f"tracklet_diagnostics_{RUN_ID}.csv": "18c16ef7048e86566ca33ab23b0f4fc599538938009897be739c1c60c3dbd8a1",
    f"tracklet_observations_{RUN_ID}.csv": "a7272aac31edf0c8eb1b82c14484e4f1124e6d9628571600716dd7b465662fa2",
    f"tracklet_window_comparison_{RUN_ID}.csv": "3d7ccb932142902024b531e9cdfcf521f37d90f3cf24f5ce5f5afec7e6c354c7",
    f"zone_candidate_comparison_{RUN_ID}.csv": "cc448c951bc187ead1f0481bad8e23c8e5049e8b5d05652efb9a962980c21291",
    f"zone_checkpoint_comparison_{RUN_ID}.csv": "11aba552d93c3ee8e3766547ae629d1dae89943e579bc59d94823be91dfaa4f6",
    f"zone_comparison_{RUN_ID}.json": "7d450799b1c54ac37054b04e75b714ffd19bf53fd64fb0dd010da4a0d14b0e33",
    f"zone_frame_comparison_{RUN_ID}.csv": "9167c5511aeb8d21d3e33643e7e07ee4c3b7b0b954e6f05db6fbb662a5350c49",
}

HISTORICAL_HASHES: dict[str, str] = {
    "benchmark_leakage_safe_subset_predictions.csv": "7dce1a058e5fca8309887fd16c28afe42e7bfc38852333c6a3770377d8ce6da1",
    "tue_p1_tracklet_diagnostics.csv": "0ec9539a7a7ac9be7f4eb0a51e8309dfb694529fbd34fc91a396887c56d01a77",
    "tue_p2_tracklet_diagnostics.csv": "e8d4aea76cc281bd52efe26bab8f1da50bbe7e83c6167dc1054af7cf97aeeb25",
}

EXPECTED_HISTORICAL_RECOVERIES: dict[str, str] = {
    "CP2-cam10-front-TRK00007": "24011CSEAI0103",
    "CP3-cam5-back-TRK00003": "24011CSEAI0007",
    "CP5-cam5-back-TRK00001": "2401100CSE0019",
    "CP4-cam10-front-TRK00012": "2401100CSE0209",
    "CP1-cam5-back-TRK00004": "24011CSEAI0007",
    "CP2-cam10-front-TRK00013": "24011CSEAI0051",
    "CP4-cam5-back-TRK00030": "2401100CSE0111",
    "CP4-cam5-back-TRK00024": "2401100CSE0044",
}

EXPECTED_MATERIAL_REPRESENTATIVES: dict[tuple[str, str], tuple[str, str]] = {
    ("2401100CSE0016", "CP1"): ("CP1-cam5-back-TRK00003", "guarded_recovery"),
    ("2401100CSE0016", "CP2"): ("CP2-cam5-back-TRK00034", "guarded_recovery"),
    ("2401100CSE0016", "CP3"): ("CP3-cam5-back-TRK00002", "guarded_recovery"),
    ("2401100CSE0016", "CP5"): ("CP5-cam5-back-TRK00002", "guarded_recovery"),
    ("2401100CSE0019", "CP2"): ("CP2-cam5-back-TRK00016", "strict_accepted"),
    ("2401100CSE0019", "CP3"): ("CP3-cam5-back-TRK00019", "guarded_recovery"),
    ("2401100CSE0019", "CP4"): ("CP4-cam5-back-TRK00007", "guarded_recovery"),
    ("2401100CSE0019", "CP5"): ("CP5-cam5-back-TRK00008", "strict_accepted"),
    ("2401100CSE0028", "CP2"): ("CP2-cam5-back-TRK00019", "strict_accepted"),
    ("2401100CSE0028", "CP5"): ("CP5-cam5-back-TRK00040", "strict_accepted"),
    ("2401100CSE0044", "CP1"): ("CP1-cam10-front-TRK00007", "strict_accepted"),
    ("2401100CSE0044", "CP5"): ("CP5-cam10-front-TRK00003", "strict_accepted"),
    ("2401100CSE0050", "CP1"): ("CP1-cam10-front-TRK00013", "strict_accepted"),
    ("2401100CSE0050", "CP2"): ("CP2-cam10-front-TRK00005", "strict_accepted"),
    ("2401100CSE0050", "CP3"): ("CP3-cam10-front-TRK00010", "strict_accepted"),
    ("2401100CSE0050", "CP4"): ("CP4-cam10-front-TRK00008", "strict_accepted"),
    ("2401100CSE0060", "CP1"): ("CP1-cam5-back-TRK00002", "strict_accepted"),
    ("2401100CSE0060", "CP2"): ("CP2-cam5-back-TRK00006", "strict_accepted"),
    ("2401100CSE0060", "CP3"): ("CP3-cam5-back-TRK00018", "guarded_recovery"),
    ("2401100CSE0060", "CP4"): ("CP4-cam5-back-TRK00014", "guarded_recovery"),
    ("2401100CSE0124", "CP2"): ("CP2-cam5-back-TRK00007", "strict_accepted"),
    ("2401100CSE0124", "CP3"): ("CP3-cam5-back-TRK00005", "strict_accepted"),
    ("2401100CSE0140", "CP2"): ("CP2-cam5-back-TRK00021", "strict_accepted"),
    ("2401100CSE0140", "CP3"): ("CP3-cam5-back-TRK00003", "strict_accepted"),
    ("2401100CSE0140", "CP4"): ("CP4-cam5-back-TRK00015", "strict_accepted"),
    ("24011CSEAI0007", "CP2"): ("CP2-cam5-back-TRK00012", "guarded_recovery"),
    ("24011CSEAI0007", "CP3"): ("CP3-cam5-back-TRK00006", "strict_accepted"),
    ("24011CSEAI0007", "CP4"): ("CP4-cam5-back-TRK00004", "guarded_recovery"),
    ("24011CSEAI0051", "CP1"): ("CP1-cam5-back-TRK00005", "guarded_recovery"),
    ("24011CSEAI0051", "CP3"): ("CP3-cam5-back-TRK00004", "guarded_recovery"),
    ("24011CSEAI0051", "CP5"): ("CP5-cam5-back-TRK00001", "strict_accepted"),
}

REUSED_REVIEW_TRACKS: dict[str, str] = {
    "CP2-cam5-back-TRK00021": "2401100CSE0140",
    "CP3-cam5-back-TRK00003": "2401100CSE0140",
    "CP4-cam5-back-TRK00015": "2401100CSE0140",
}


class ProductPhase2HError(RuntimeError):
    pass


@dataclass(frozen=True)
class Inputs:
    repo_root: Path
    diagnostic_root: Path
    tracklet_csv: Path
    observations_csv: Path
    summary_json: Path
    comparison_json: Path
    prior_reviewed_csv: Path
    student_map: Path
    freeze_json: Path
    freeze_csv: Path
    canonical_video_root: Path
    mirror_video_root: Path
    output_root: Path
    historical_benchmark_predictions: Path
    historical_tue_p1_tracklets: Path
    historical_tue_p2_tracklets: Path


@dataclass(frozen=True)
class PreflightResult:
    selected: pd.DataFrame
    pending: pd.DataFrame
    reused: pd.DataFrame
    tracklets: pd.DataFrame
    observations: pd.DataFrame
    historical_validation: pd.DataFrame
    output_id: str


def _text(value: Any) -> str:
    return str(value or "").strip()


def _yes(value: Any) -> bool:
    return _text(value).lower() in {"yes", "true", "1"}


def _json(path: Path, label: str) -> dict[str, Any]:
    if not Path(path).is_file():
        raise ProductPhase2HError(f"{label} not found: {path}")
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise ProductPhase2HError(f"Could not read {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProductPhase2HError(f"{label} must contain a JSON object")
    return payload


def _csv(path: Path, label: str) -> pd.DataFrame:
    if not Path(path).is_file():
        raise ProductPhase2HError(f"{label} not found: {path}")
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except Exception as exc:
        raise ProductPhase2HError(f"Could not read {label}: {exc}") from exc


def _verify_frozen_source_copy(inputs: Inputs) -> None:
    """Verify the immutable Phase 1.2N frozen-input copies and all ten videos.

    Phase 1.2N's JSON records the path of the original source-freeze CSV. The
    shadow run intentionally preserves byte-identical copies under
    ``frozen_inputs``. Those copies remain authoritative even when the original
    source-freeze directory is absent. We hash-pin the copies, then use the
    existing strict verifier with a temporary JSON whose only change is the CSV
    location. No project file is modified.
    """

    freeze_json = Path(inputs.freeze_json)
    freeze_csv = Path(inputs.freeze_csv)
    if not freeze_json.is_file():
        raise ProductPhase2HError(
            "Immutable Phase 1.2N frozen-input JSON not found: " + str(freeze_json)
        )
    if not freeze_csv.is_file():
        raise ProductPhase2HError(
            "Immutable Phase 1.2N frozen-input CSV not found: " + str(freeze_csv)
        )
    json_hash = _sha256_file(freeze_json)
    csv_hash = _sha256_file(freeze_csv)
    if json_hash != SOURCE_FREEZE_JSON_SHA256:
        raise ProductPhase2HError(
            f"Phase 1.2N frozen-input JSON hash changed: expected {SOURCE_FREEZE_JSON_SHA256}, got {json_hash}"
        )
    if csv_hash != SOURCE_FREEZE_CSV_SHA256:
        raise ProductPhase2HError(
            f"Phase 1.2N frozen-input CSV hash changed: expected {SOURCE_FREEZE_CSV_SHA256}, got {csv_hash}"
        )

    payload = _json(freeze_json, "Phase 1.2N frozen-input source-freeze JSON")
    if _text(payload.get("FreezeID")) != PHASE_1_2N_FREEZE_ID:
        raise ProductPhase2HError("Phase 1.2N source-freeze ID changed")
    recorded_csv_name = Path(_text(payload.get("CSV")).replace("\\", "/")).name
    if recorded_csv_name != freeze_csv.name:
        raise ProductPhase2HError("Phase 1.2N source-freeze JSON references an unexpected CSV")

    with tempfile.TemporaryDirectory(prefix="phase_2h_freeze_verify_") as temp_dir:
        temp_root = Path(temp_dir)
        temp_csv = temp_root / freeze_csv.name
        temp_json = temp_root / freeze_json.name
        shutil.copy2(freeze_csv, temp_csv)
        relocated = dict(payload)
        relocated["CSV"] = str(temp_csv.resolve())
        temp_json.write_text(
            json.dumps(relocated, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        try:
            verify_source_freeze(
                freeze_json_path=temp_json,
                freeze_csv_path=temp_csv,
                canonical_root=inputs.canonical_video_root,
                mirror_root=inputs.mirror_video_root,
                session_id=SESSION_ID,
            )
        except Exception as exc:
            raise ProductPhase2HError(
                f"Phase 1.2N frozen source/video verification failed: {exc}"
            ) from exc


def default_inputs(repo_root: Path) -> Inputs:
    repo = Path(repo_root).resolve()
    diagnostic_root = repo / "attendance_output" / "diagnostics" / RUN_ID
    phase_1_2n = repo / "attendance_output" / "shadow_validation" / "phase_1_2n"
    shadow_root = phase_1_2n / "shadow_runs" / PHASE_1_2N_SHADOW_ID
    prior_reviewed = (
        shadow_root / "evaluation" / "mon-p4-evaluation-302c6707c95029a017df"
        / "reviewed_exception_tracks.csv"
    )
    # Phase 1.2N copied its immutable source-freeze inputs into the shadow run.
    # The original source-freeze directory is historical and may be moved or
    # intentionally omitted from a compact project snapshot, so Phase 2H
    # consumes the hash-pinned frozen_inputs copies instead of guessing a flat
    # source_freeze path.
    freeze_root = shadow_root / "frozen_inputs"
    phase_1_2l = (
        repo / "attendance_output" / "embedding_forensics" / "phase_1_2l"
        / "source-ablation-da12b41f85bc813a6de1"
    )
    p1_root = repo / "attendance_output" / "diagnostics" / "2026-06-30__B51__P1__CVO_20260712_000753_991890"
    p2_root = repo / "attendance_output" / "diagnostics" / "2026-06-30__B51__P2__CVO_20260712_221105_286649"
    return Inputs(
        repo_root=repo,
        diagnostic_root=diagnostic_root,
        tracklet_csv=diagnostic_root / f"tracklet_diagnostics_{RUN_ID}.csv",
        observations_csv=diagnostic_root / f"tracklet_observations_{RUN_ID}.csv",
        summary_json=diagnostic_root / f"diagnostic_summary_{RUN_ID}.json",
        comparison_json=diagnostic_root / f"tracklet_comparison_{RUN_ID}.json",
        prior_reviewed_csv=prior_reviewed,
        student_map=repo / "data" / "student_faculty_map.json",
        freeze_json=freeze_root / "phase_1_2n_mon_p4_source_freeze.json",
        freeze_csv=freeze_root / "phase_1_2n_mon_p4_source_freeze.csv",
        canonical_video_root=repo / "cctv_videos" / "prepared_slots" / "2026-06-22" / "MON_P4",
        mirror_video_root=repo / "cctv_videos" / "MON_P4",
        output_root=repo / "attendance_output" / "product_workflow" / "phase_2h_multiframe_recovery",
        historical_benchmark_predictions=phase_1_2l / "benchmark_leakage_safe_subset_predictions.csv",
        historical_tue_p1_tracklets=p1_root / "tracklet_diagnostics_2026-06-30__B51__P1__CVO_20260712_000753_991890.csv",
        historical_tue_p2_tracklets=p2_root / "tracklet_diagnostics_2026-06-30__B51__P2__CVO_20260712_221105_286649.csv",
    )


def verify_diagnostic_hashes(inputs: Inputs) -> None:
    for filename, expected in DIAGNOSTIC_HASHES.items():
        path = inputs.diagnostic_root / filename
        if not path.is_file():
            raise ProductPhase2HError(f"Required Phase 2G diagnostic file is missing: {path}")
        actual = _sha256_file(path)
        if actual != expected:
            raise ProductPhase2HError(
                f"Phase 2G diagnostic hash changed for {filename}. Expected {expected}, got {actual}."
            )


def _numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    copy = frame.copy()
    for column in columns:
        copy[column] = pd.to_numeric(copy[column], errors="coerce")
    return copy


def _prepare_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in [
        "Tracklet_Best_Score", "Tracklet_Margin", "Dominant_Frame_Best_Share_Pct",
        "Observation_Count", "Selected_Observation_Count", "Consistent_Embedding_Count",
        "Pairwise_Similarity_Median",
    ]:
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    return frame


def _guarded_recovery_candidates(tracklet_df: pd.DataFrame) -> pd.DataFrame:
    frame = _prepare_numeric(tracklet_df)
    eligible = frame["Tracklet_Eligible"].map(_yes)
    accepted = frame["Tracklet_Accepted"].map(_yes)
    agreement = frame["Tracklet_Best_Roll"].map(normalize_roll).eq(
        frame["Dominant_Frame_Best_Roll"].map(normalize_roll)
    )
    candidates = frame[
        eligible & ~accepted & agreement
        & frame["Tracklet_Best_Score"].ge(RECOVERY_MIN_SCORE)
        & frame["Tracklet_Margin"].ge(RECOVERY_MIN_MARGIN)
        & frame["Dominant_Frame_Best_Share_Pct"].ge(RECOVERY_MIN_DOMINANT_SHARE_PCT)
        & frame["Observation_Count"].ge(RECOVERY_MIN_OBSERVATIONS)
        & frame["Selected_Observation_Count"].ge(RECOVERY_MIN_SELECTED_OBSERVATIONS)
        & frame["Consistent_Embedding_Count"].ge(RECOVERY_MIN_CONSISTENT_EMBEDDINGS)
        & frame["Pairwise_Similarity_Median"].ge(RECOVERY_MIN_PAIRWISE_MEDIAN)
    ].copy()
    candidates["Tracklet_Best_Roll"] = candidates["Tracklet_Best_Roll"].map(normalize_roll)
    repeat_counts = candidates.groupby("Tracklet_Best_Roll")["Checkpoint_ID"].nunique()
    repeated = set(repeat_counts[repeat_counts.ge(RECOVERY_MIN_CHECKPOINT_REPETITION)].index)
    candidates = candidates[candidates["Tracklet_Best_Roll"].isin(repeated)].copy()
    candidates = candidates.sort_values(
        ["Tracklet_Best_Roll", "Checkpoint_ID", "Tracklet_Best_Score", "Tracklet_Margin",
         "Pairwise_Similarity_Median", "Observation_Count", "Tracklet_ID"],
        ascending=[True, True, False, False, False, False, True],
        kind="mergesort",
    )
    candidates = candidates.groupby(["Tracklet_Best_Roll", "Checkpoint_ID"], as_index=False).head(1)
    candidates["Evidence_Tier"] = "guarded_recovery"
    candidates["Recovery_Rule_ID"] = RECOVERY_RULE_ID
    return candidates


def select_material_representatives(tracklet_df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "Tracklet_ID", "Session_ID", "Subject_Abbr", "Checkpoint_ID", "Camera_ID", "Video",
        "Tracklet_Eligible", "Tracklet_Accepted", "Tracklet_Best_Roll", "Tracklet_Best_Score",
        "Tracklet_Second_Roll", "Tracklet_Second_Score", "Tracklet_Margin",
        "Dominant_Frame_Best_Roll", "Dominant_Frame_Best_Share_Pct", "Observation_Count",
        "Selected_Observation_Count", "Consistent_Embedding_Count", "Pairwise_Similarity_Median",
        "Aggregate_Mode", "Match_Threshold", "Margin_Threshold",
    }
    missing = sorted(required.difference(tracklet_df.columns))
    if missing:
        raise ProductPhase2HError("Tracklet diagnostics missing columns: " + ", ".join(missing))
    if tracklet_df["Tracklet_ID"].astype(str).duplicated().any():
        raise ProductPhase2HError("Tracklet diagnostics contain duplicate Tracklet_ID values")

    frame = _prepare_numeric(tracklet_df)
    strict = frame[frame["Tracklet_Accepted"].map(_yes)].copy()
    if len(strict) != EXPECTED_ACCEPTED_TRACKLETS:
        raise ProductPhase2HError(
            f"Expected {EXPECTED_ACCEPTED_TRACKLETS} accepted tracklets; found {len(strict)}"
        )
    strict["Tracklet_Best_Roll"] = strict["Tracklet_Best_Roll"].map(normalize_roll)
    strict = strict.sort_values(
        ["Tracklet_Best_Roll", "Checkpoint_ID", "Tracklet_Best_Score", "Tracklet_Margin",
         "Pairwise_Similarity_Median", "Observation_Count", "Tracklet_ID"],
        ascending=[True, True, False, False, False, False, True],
        kind="mergesort",
    )
    strict = strict.groupby(["Tracklet_Best_Roll", "Checkpoint_ID"], as_index=False).head(1)
    strict["Evidence_Tier"] = "strict_accepted"
    strict["Recovery_Rule_ID"] = "strict_0.48_0.08"

    recovery = _guarded_recovery_candidates(frame)
    combined = pd.concat([strict, recovery], ignore_index=True, sort=False)
    combined["Tier_Order"] = combined["Evidence_Tier"].map({"strict_accepted": 0, "guarded_recovery": 1})
    combined = combined.sort_values(
        ["Tracklet_Best_Roll", "Checkpoint_ID", "Tier_Order", "Tracklet_Best_Score", "Tracklet_ID"],
        ascending=[True, True, True, False, True], kind="mergesort",
    ).drop_duplicates(["Tracklet_Best_Roll", "Checkpoint_ID"], keep="first")
    checkpoint_counts = combined.groupby("Tracklet_Best_Roll")["Checkpoint_ID"].nunique()
    material_rolls = set(checkpoint_counts[checkpoint_counts.ge(2)].index)
    selected = combined[combined["Tracklet_Best_Roll"].isin(material_rolls)].copy()
    selected["Potential_Checkpoint_Count"] = selected["Tracklet_Best_Roll"].map(checkpoint_counts).astype(int)
    selected["Potential_Status"] = selected["Potential_Checkpoint_Count"].map(
        lambda count: "Present candidate" if count >= PRESENT_CHECKPOINTS else "Needs Review candidate"
    )
    actual = {
        (normalize_roll(row["Tracklet_Best_Roll"]), _text(row["Checkpoint_ID"]).upper()):
        (_text(row["Tracklet_ID"]), _text(row["Evidence_Tier"]))
        for row in selected.to_dict("records")
    }
    if actual != EXPECTED_MATERIAL_REPRESENTATIVES:
        raise ProductPhase2HError(
            "Material multi-frame representative selection changed. "
            f"Expected={EXPECTED_MATERIAL_REPRESENTATIVES}, actual={actual}"
        )
    if len(selected) != EXPECTED_SELECTED_PAIRS:
        raise ProductPhase2HError(
            f"Expected {EXPECTED_SELECTED_PAIRS} material identity/checkpoint pairs; found {len(selected)}"
        )
    return selected.sort_values(
        ["Tracklet_Best_Roll", "Checkpoint_ID", "Evidence_Tier", "Tracklet_ID"]
    ).reset_index(drop=True)


def validate_historical_recovery_rule(inputs: Inputs) -> pd.DataFrame:
    expected_files = {
        inputs.historical_benchmark_predictions: HISTORICAL_HASHES["benchmark_leakage_safe_subset_predictions.csv"],
        inputs.historical_tue_p1_tracklets: HISTORICAL_HASHES["tue_p1_tracklet_diagnostics.csv"],
        inputs.historical_tue_p2_tracklets: HISTORICAL_HASHES["tue_p2_tracklet_diagnostics.csv"],
    }
    for path, expected_hash in expected_files.items():
        if not path.is_file():
            raise ProductPhase2HError(f"Required frozen recovery-validation evidence is missing: {path}")
        actual_hash = _sha256_file(path)
        if actual_hash != expected_hash:
            raise ProductPhase2HError(
                f"Frozen recovery-validation evidence changed: {path.name}. Expected {expected_hash}, got {actual_hash}."
            )
    benchmark = _csv(inputs.historical_benchmark_predictions, "Phase 1.2L leakage-safe benchmark predictions")
    benchmark = benchmark[benchmark["Model_ID"].astype(str).eq(CURRENT_MODEL_ID)].copy()
    if len(benchmark) != 120:
        raise ProductPhase2HError(f"Expected 120 frozen benchmark rows for {CURRENT_MODEL_ID}; found {len(benchmark)}")
    historical = pd.concat([
        _csv(inputs.historical_tue_p1_tracklets, "TUE_P1 tracklet diagnostics"),
        _csv(inputs.historical_tue_p2_tracklets, "TUE_P2 tracklet diagnostics"),
    ], ignore_index=True)
    join_columns = ["Tracklet_ID", "Session_ID", "Checkpoint_ID", "Camera_ID"]
    diagnostic_columns = join_columns + [
        "Tracklet_Eligible", "Observation_Count", "Selected_Observation_Count",
        "Consistent_Embedding_Count", "Pairwise_Similarity_Median",
        "Dominant_Frame_Best_Roll", "Dominant_Frame_Best_Share_Pct",
    ]
    joined = benchmark.merge(
        historical[diagnostic_columns], on=join_columns, how="left", validate="one_to_one"
    )
    if joined["Observation_Count"].isna().any():
        raise ProductPhase2HError("Frozen benchmark rows could not be joined to exact tracklet diagnostics")
    for column in [
        "Best_Score", "Margin", "Observation_Count", "Selected_Observation_Count",
        "Consistent_Embedding_Count", "Pairwise_Similarity_Median",
        "Dominant_Frame_Best_Share_Pct",
    ]:
        joined[column] = pd.to_numeric(joined[column], errors="coerce")
    rule = (
        ~joined["Accepted"].map(_yes)
        & joined["Tracklet_Eligible"].map(_yes)
        & joined["Best_Roll"].map(normalize_roll).eq(joined["Dominant_Frame_Best_Roll"].map(normalize_roll))
        & joined["Best_Score"].ge(RECOVERY_MIN_SCORE)
        & joined["Margin"].ge(RECOVERY_MIN_MARGIN)
        & joined["Dominant_Frame_Best_Share_Pct"].ge(RECOVERY_MIN_DOMINANT_SHARE_PCT)
        & joined["Observation_Count"].ge(RECOVERY_MIN_OBSERVATIONS)
        & joined["Selected_Observation_Count"].ge(RECOVERY_MIN_SELECTED_OBSERVATIONS)
        & joined["Consistent_Embedding_Count"].ge(RECOVERY_MIN_CONSISTENT_EMBEDDINGS)
        & joined["Pairwise_Similarity_Median"].ge(RECOVERY_MIN_PAIRWISE_MEDIAN)
    )
    candidates = joined[rule].copy()
    candidates["Best_Roll"] = candidates["Best_Roll"].map(normalize_roll)
    candidates["Actual_Roll"] = candidates["Actual_Roll"].map(normalize_roll)
    actual = dict(zip(candidates["Tracklet_ID"].astype(str), candidates["Best_Roll"].astype(str)))
    if actual != EXPECTED_HISTORICAL_RECOVERIES:
        raise ProductPhase2HError(
            f"Frozen leakage-safe recovery candidate set changed. Expected={EXPECTED_HISTORICAL_RECOVERIES}, actual={actual}"
        )
    candidates["Identity_Correct"] = (
        candidates["Review_Status"].eq("identified")
        & candidates["Best_Roll"].eq(candidates["Actual_Roll"])
    )
    candidates["Unsafe"] = (
        (candidates["Review_Status"].eq("identified") & ~candidates["Identity_Correct"])
        | candidates["Review_Status"].isin({"not_in_mapping", "mixed_track"})
    )
    candidates["Unverifiable"] = candidates["Review_Status"].isin({"unidentifiable", "uncertain"})
    if int(candidates["Identity_Correct"].sum()) != 8 or candidates["Unsafe"].any() or candidates["Unverifiable"].any():
        raise ProductPhase2HError("Frozen leakage-safe validation did not reproduce 8 correct / 0 unsafe / 0 unverifiable")
    candidates["Recovery_Rule_ID"] = RECOVERY_RULE_ID
    candidates["Validation_Result"] = "correct_human_reviewed_recovery"
    return candidates.sort_values(["Session_ID", "Tracklet_ID"]).reset_index(drop=True)


def load_reused_ground_truth(prior_reviewed_csv: Path, selected: pd.DataFrame) -> pd.DataFrame:
    prior = _csv(prior_reviewed_csv, "Phase 1.2N reviewed exception tracks")
    required = {"Tracklet_ID", "Review_Status", "Actual_Roll", "Predicted_Roll", "Session_ID"}
    missing = sorted(required.difference(prior.columns))
    if missing:
        raise ProductPhase2HError("Prior reviewed evidence missing columns: " + ", ".join(missing))
    rows: list[dict[str, Any]] = []
    selected_ids = set(selected["Tracklet_ID"].astype(str))
    for tracklet_id, expected_roll in REUSED_REVIEW_TRACKS.items():
        match = prior[prior["Tracklet_ID"].astype(str).eq(tracklet_id)]
        if len(match) != 1:
            raise ProductPhase2HError(f"Expected exactly one frozen review row for {tracklet_id}; found {len(match)}")
        row = match.iloc[0].to_dict()
        actual = normalize_roll(row.get("Actual_Roll"))
        predicted = normalize_roll(row.get("Predicted_Roll"))
        if _text(row.get("Review_Status")) != "identified" or actual != expected_roll or predicted != expected_roll:
            raise ProductPhase2HError(f"Frozen reviewed identity changed for {tracklet_id}")
        if _text(row.get("Session_ID")) != SESSION_ID:
            raise ProductPhase2HError(f"Frozen reviewed session changed for {tracklet_id}")
        if tracklet_id not in selected_ids:
            raise ProductPhase2HError(f"Frozen reviewed track is not part of the material selection: {tracklet_id}")
        rows.append({
            "Tracklet_ID": tracklet_id,
            "Review_Status": "identified",
            "Actual_Roll": actual,
            "Predicted_Roll": predicted,
            "Ground_Truth_Source": "phase_1_2n_blind_review",
            "Source_SHA256": _sha256_file(Path(prior_reviewed_csv)),
        })
    return pd.DataFrame(rows).sort_values("Tracklet_ID").reset_index(drop=True)


def _output_id(inputs: Inputs, selected: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(POLICY_VERSION.encode())
    digest.update(RECOVERY_RULE_ID.encode())
    digest.update(_sha256_file(inputs.tracklet_csv).encode())
    digest.update(_sha256_file(inputs.prior_reviewed_csv).encode())
    digest.update(_sha256_file(inputs.historical_benchmark_predictions).encode())
    for row in selected.sort_values("Tracklet_ID").to_dict("records"):
        digest.update(_text(row.get("Tracklet_ID")).encode())
        digest.update(_text(row.get("Evidence_Tier")).encode())
    return "multiframe-recovery-" + digest.hexdigest()[:20]


def preflight(inputs: Inputs, *, verify_videos: bool = True) -> PreflightResult:
    verify_diagnostic_hashes(inputs)
    summary = _json(inputs.summary_json, "Phase 2G diagnostic summary")
    comparison = _json(inputs.comparison_json, "Phase 2G tracklet comparison")
    metadata = summary.get("run_metadata", {})
    if _text(metadata.get("diagnostic_run_id")) != RUN_ID:
        raise ProductPhase2HError("Phase 2G diagnostic run ID changed")
    if _text(metadata.get("session_id")) != SESSION_ID:
        raise ProductPhase2HError("Phase 2G session ID changed")
    if _text(metadata.get("subject_abbr")).upper() != SUBJECT_ABBR:
        raise ProductPhase2HError("Phase 2G subject changed")
    if _text(metadata.get("official_recognition_unit")) != "frame_detection":
        raise ProductPhase2HError("Phase 2G official recognition authority is not frame_detection")
    if _text(metadata.get("zone_mode")) != "compare" or _text(metadata.get("tracklet_mode")) != "compare":
        raise ProductPhase2HError("Phase 2G zone/tracklet compare contract changed")
    if comparison.get("compare_mode_changed_official_attendance") is not False:
        raise ProductPhase2HError("Phase 2G compare mode unexpectedly changed official attendance")
    if int(comparison.get("tracklets_total", -1)) != EXPECTED_TRACKLETS_TOTAL:
        raise ProductPhase2HError("Phase 2G total tracklet count changed")
    if int(comparison.get("eligible_tracklets", -1)) != EXPECTED_ELIGIBLE_TRACKLETS:
        raise ProductPhase2HError("Phase 2G eligible tracklet count changed")
    if int(comparison.get("accepted_tracklets", -1)) != EXPECTED_ACCEPTED_TRACKLETS:
        raise ProductPhase2HError("Phase 2G accepted tracklet count changed")
    if int(comparison.get("unique_accepted_identities", -1)) != EXPECTED_UNIQUE_ACCEPTED_IDENTITIES:
        raise ProductPhase2HError("Phase 2G accepted identity count changed")

    historical_validation = validate_historical_recovery_rule(inputs)
    tracklets = _csv(inputs.tracklet_csv, "Phase 2G tracklet diagnostics")
    observations = _csv(inputs.observations_csv, "Phase 2G tracklet observations")
    selected = select_material_representatives(tracklets)
    reused = load_reused_ground_truth(inputs.prior_reviewed_csv, selected)
    pending = selected[~selected["Tracklet_ID"].isin(set(reused["Tracklet_ID"]))].copy()
    if len(selected) != EXPECTED_SELECTED_PAIRS or len(reused) != EXPECTED_REUSED_LABELS or len(pending) != EXPECTED_PENDING_REVIEW:
        raise ProductPhase2HError(
            f"Expected selected/reused/pending counts {EXPECTED_SELECTED_PAIRS}/{EXPECTED_REUSED_LABELS}/{EXPECTED_PENDING_REVIEW}; "
            f"found {len(selected)}/{len(reused)}/{len(pending)}"
        )
    roster, _, subject_rolls = load_student_mapping(inputs.student_map, SUBJECT_ABBR)
    if len(subject_rolls) != 27:
        raise ProductPhase2HError(f"Expected authoritative CVO roster of 27; found {len(subject_rolls)}")
    if verify_videos:
        _verify_frozen_source_copy(inputs)
    return PreflightResult(
        selected=selected,
        pending=pending,
        reused=reused,
        tracklets=tracklets,
        observations=observations,
        historical_validation=historical_validation,
        output_id=_output_id(inputs, selected),
    )


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _manifest_files(root: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "output_manifest.json":
            continue
        relative = path.relative_to(root).as_posix()
        files[relative] = {"sha256": _sha256_file(path), "size_bytes": path.stat().st_size}
    return files


def verify_output(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    manifest_path = output_dir / "output_manifest.json"
    manifest = _json(manifest_path, "Phase 2H output manifest")
    if manifest.get("policy_version") != POLICY_VERSION:
        raise ProductPhase2HError("Phase 2H output policy version changed")
    if manifest.get("output_id") != output_dir.name:
        raise ProductPhase2HError("Phase 2H output ID does not match directory name")
    recorded = manifest.get("files")
    if not isinstance(recorded, dict) or not recorded:
        raise ProductPhase2HError("Phase 2H output manifest has no files")
    actual_paths = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file()
        and path.name != "output_manifest.json"
        and path.relative_to(output_dir).parts[0] != "evaluation"
    }
    if actual_paths != set(recorded):
        raise ProductPhase2HError("Phase 2H output file set changed")
    for relative, metadata in recorded.items():
        path = output_dir / Path(relative)
        if _sha256_file(path) != metadata.get("sha256"):
            raise ProductPhase2HError(f"Phase 2H output hash changed: {relative}")
        if path.stat().st_size != int(metadata.get("size_bytes", -1)):
            raise ProductPhase2HError(f"Phase 2H output size changed: {relative}")
    return manifest


def export_recovery_review(inputs: Inputs) -> tuple[Path, Path, bool]:
    result = preflight(inputs, verify_videos=True)
    output_dir = inputs.output_root / result.output_id
    if output_dir.exists():
        manifest = verify_output(output_dir)
        reviewer = output_dir / _text(manifest.get("review_package")) / "reviewer" / "index.html"
        if not reviewer.is_file():
            raise ProductPhase2HError("Verified Phase 2H output is missing its reviewer")
        return output_dir, reviewer, True

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / (output_dir.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        _write_csv(staging / "selected_multiframe_checkpoint_evidence.csv", result.selected)
        _write_csv(staging / "frozen_leakage_safe_recovery_validation.csv", result.historical_validation)
        _write_csv(staging / "reused_phase_1_2n_ground_truth.csv", result.reused)
        _write_csv(staging / "pending_multiframe_blind_review_tracks.csv", result.pending)

        preview_rows = []
        for roll, group in result.selected.groupby("Tracklet_Best_Roll"):
            checkpoints = sorted(set(group["Checkpoint_ID"].astype(str)))
            count = len(checkpoints)
            status = "Present candidate" if count >= PRESENT_CHECKPOINTS else "Needs Review candidate"
            preview_rows.append({
                "Predicted_Roll": roll,
                "Predicted_Checkpoints": "; ".join(checkpoints),
                "Predicted_Checkpoint_Count": count,
                "Predicted_Shadow_Status": status,
                "Human_Validation_Status": "pending",
            })
        _write_csv(staging / "multiframe_recovery_preview.csv", pd.DataFrame(preview_rows))

        review_package = export_review_package(
            tracklet_df=result.tracklets,
            observation_df=result.observations,
            video_root=inputs.canonical_video_root,
            output_root=staging,
            student_map_path=inputs.student_map,
            subject_abbr=SUBJECT_ABBR,
            diagnostic_run_id=RUN_ID,
            evidence_count=5,
            crop_padding=0.45,
            package_id=result.output_id,
            selected_tracklet_ids=result.pending["Tracklet_ID"].astype(str).tolist(),
            review_kind="shadow_recovery",
            hidden_extra_columns={
                "Evidence_Tier": "Evidence_Tier",
                "Recovery_Rule_ID": "Recovery_Rule_ID",
                "Potential_Checkpoint_Count": "Potential_Checkpoint_Count",
            },
            include_context=True,
            context_padding=1.4,
            package_prefix="multiframe_recovery_review",
            subject_roster_only=True,
            session_id_override=SESSION_ID,
            source_files={
                "tracklet_diagnostics": inputs.tracklet_csv,
                "tracklet_observations": inputs.observations_csv,
                "phase_1_2n_reviewed_exceptions": inputs.prior_reviewed_csv,
            },
            private_metadata={
                "policy_version": POLICY_VERSION,
                "selection_scope": "strict accepted plus guarded near-threshold representatives for identities with at least two candidate checkpoints",
                "material_selected_count": len(result.selected),
                "recovery_rule_id": RECOVERY_RULE_ID,
                "frozen_validation_correct": len(result.historical_validation),
                "frozen_validation_unsafe": 0,
                "reused_ground_truth_count": len(result.reused),
                "new_blind_review_count": len(result.pending),
                "official_attendance_changed": False,
                "recognition_repeated": False,
            },
        )
        summary = {
            "schema_version": 1,
            "policy_version": POLICY_VERSION,
            "output_id": result.output_id,
            "session_id": SESSION_ID,
            "diagnostic_run_id": RUN_ID,
            "frame_official_result": {"present": 0, "needs_review": 1, "absent": 26},
            "multiframe_raw_result_before_human_validation": {
                "present_candidates_if_all_selected_evidence_is_correct": ["2401100CSE0016", "2401100CSE0019", "2401100CSE0050", "2401100CSE0060", "2401100CSE0140", "24011CSEAI0007", "24011CSEAI0051"],
                "needs_review_candidates_if_all_selected_evidence_is_correct": ["2401100CSE0028", "2401100CSE0044", "2401100CSE0124"],
                "material_identity_checkpoint_pairs": len(result.selected),
            },
            "reused_phase_1_2n_labels": len(result.reused),
            "new_blind_review_items": len(result.pending),
            "review_package": review_package.root.name,
            "decision": "pending_multiframe_recovery_blind_review",
            "recognition_repeated": False,
            "official_attendance_changed": False,
            "tracklets_activated": False,
        }
        (staging / "shadow_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (staging / "README.md").write_text(
            "# Product Phase 2H — Multi-frame Recall Recovery\n\n"
            "This output reuses the completed Phase 2G diagnostics, validates the guarded recovery rule "
            "against eight leakage-safe human-reviewed benchmark tracks, and reuses three exact "
            "Phase 1.2N blind labels. It exports twenty-eight new blind review cards. No recognition "
            "was repeated and official attendance was not changed.\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "policy_version": POLICY_VERSION,
            "output_id": result.output_id,
            "review_package": review_package.root.name,
            "files": _manifest_files(staging),
        }
        (staging / "output_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        staging.replace(output_dir)
        verify_output(output_dir)
        return output_dir, output_dir / review_package.root.name / "reviewer" / "index.html", False
    except ReviewExportError as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise ProductPhase2HError(f"Could not export the Phase 2H blind reviewer: {exc}") from exc
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise



def summarize_reviewed_material(
    combined: pd.DataFrame,
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, int, int]:
    required = {"Tracklet_ID", "Checkpoint_ID", "Predicted_Roll", "Review_Status", "Actual_Roll"}
    missing = sorted(required.difference(combined.columns))
    if missing:
        raise ProductPhase2HError(
            "Combined reviewed material missing columns: " + ", ".join(missing)
        )
    reviewed = combined.copy()
    reviewed["Predicted_Roll"] = reviewed["Predicted_Roll"].map(normalize_roll)
    reviewed["Actual_Roll"] = reviewed["Actual_Roll"].map(normalize_roll)
    reviewed["Identity_Correct"] = (
        reviewed["Review_Status"].eq("identified")
        & reviewed["Actual_Roll"].eq(reviewed["Predicted_Roll"])
    )
    reviewed["Unsafe_Identity_Error"] = (
        reviewed["Review_Status"].isin({"identified", "mixed_track", "not_in_mapping"})
        & ~reviewed["Identity_Correct"]
    )
    reviewed["Unverifiable"] = reviewed["Review_Status"].isin(
        {"unidentifiable", "uncertain", ""}
    )
    unsafe = int(reviewed["Unsafe_Identity_Error"].sum())
    unverifiable = int(reviewed["Unverifiable"].sum())
    confirmed = reviewed[reviewed["Identity_Correct"]]

    rows: list[dict[str, Any]] = []
    grouped = selected.copy()
    grouped["Normalized_Roll"] = grouped["Tracklet_Best_Roll"].map(normalize_roll)
    for roll, group in grouped.groupby("Normalized_Roll"):
        predicted_checkpoints = sorted(set(group["Checkpoint_ID"].astype(str)))
        actual_group = confirmed[confirmed["Predicted_Roll"].eq(roll)]
        confirmed_checkpoints = sorted(set(actual_group["Checkpoint_ID"].astype(str)))
        count = len(confirmed_checkpoints)
        status = (
            "Present"
            if count >= PRESENT_CHECKPOINTS
            else "Needs Review"
            if count >= REVIEW_CHECKPOINTS
            else "Unconfirmed"
        )
        rows.append(
            {
                "Roll": roll,
                "Predicted_Checkpoints": "; ".join(predicted_checkpoints),
                "Predicted_Checkpoint_Count": len(predicted_checkpoints),
                "Confirmed_Checkpoints": "; ".join(confirmed_checkpoints),
                "Confirmed_Checkpoint_Count": count,
                "Shadow_Status": status,
            }
        )
    return pd.DataFrame(rows).sort_values("Roll").reset_index(drop=True), unsafe, unverifiable


def _load_new_review(output_dir: Path, labels_path: Path, student_map: Path) -> pd.DataFrame:
    manifest = verify_output(output_dir)
    package_dir = output_dir / _text(manifest.get("review_package"))
    hidden = _csv(package_dir / "private" / "hidden_predictions.csv", "Phase 2H hidden predictions")
    labels = _csv(labels_path, "Phase 2H completed blind labels")
    _, known_rolls, _ = load_student_mapping(student_map, SUBJECT_ABBR)
    try:
        validated = validate_label_dataframe(
            labels,
            set(hidden["Review_ID"].astype(str)),
            _text(hidden.iloc[0]["Package_ID"]),
            known_rolls,
        )
    except ReviewExportError as exc:
        raise ProductPhase2HError(f"Phase 2H blind labels failed validation: {exc}") from exc
    joined = hidden.merge(validated, on=["Package_ID", "Review_ID"], how="left", validate="one_to_one")
    joined["Predicted_Roll"] = joined["Predicted_Roll"].map(normalize_roll)
    joined["Actual_Roll"] = joined["Actual_Roll"].map(normalize_roll)
    joined["Ground_Truth_Source"] = "product_phase_2h_blind_review"
    return joined


def evaluate_recovery(inputs: Inputs, output_dir: Path, labels_path: Path) -> Path:
    result = preflight(inputs, verify_videos=False)
    output_dir = Path(output_dir).resolve()
    manifest = verify_output(output_dir)
    if output_dir.name != result.output_id:
        raise ProductPhase2HError("Phase 2H output does not match current verified inputs")
    evaluation_dir = output_dir / "evaluation"
    if evaluation_dir.exists():
        evaluation_manifest = evaluation_dir / "evaluation_manifest.json"
        if evaluation_manifest.is_file():
            payload = _json(evaluation_manifest, "Phase 2H evaluation manifest")
            if payload.get("labels_sha256") == _sha256_file(Path(labels_path)):
                return evaluation_dir
        raise ProductPhase2HError("A different or incomplete Phase 2H evaluation already exists")

    new_review = _load_new_review(output_dir, labels_path, inputs.student_map)
    reused = result.reused.copy()
    reused["Checkpoint_ID"] = reused["Tracklet_ID"].map(
        dict(zip(result.selected["Tracklet_ID"], result.selected["Checkpoint_ID"]))
    )
    reused["Package_ID"] = "phase_1_2n_reused"
    reused["Review_ID"] = reused["Tracklet_ID"]
    reused["Reviewer_Notes"] = ""
    tier_by_track = dict(zip(result.selected["Tracklet_ID"], result.selected["Evidence_Tier"]))
    rule_by_track = dict(zip(result.selected["Tracklet_ID"], result.selected["Recovery_Rule_ID"]))
    reused["Evidence_Tier"] = reused["Tracklet_ID"].map(tier_by_track)
    reused["Recovery_Rule_ID"] = reused["Tracklet_ID"].map(rule_by_track)

    new_rows = new_review[[
        "Tracklet_ID", "Checkpoint_ID", "Predicted_Roll", "Review_Status", "Actual_Roll",
        "Ground_Truth_Source", "Package_ID", "Review_ID", "Reviewer_Notes",
        "Evidence_Tier", "Recovery_Rule_ID",
    ]].copy()
    combined = pd.concat([new_rows, reused[new_rows.columns]], ignore_index=True)
    if len(combined) != EXPECTED_SELECTED_PAIRS or combined["Tracklet_ID"].duplicated().any():
        raise ProductPhase2HError("Combined multi-frame review evidence is incomplete or duplicated")
    status_df, unsafe, unverifiable = summarize_reviewed_material(
        combined,
        result.selected,
    )
    combined["Predicted_Roll"] = combined["Predicted_Roll"].map(normalize_roll)
    combined["Actual_Roll"] = combined["Actual_Roll"].map(normalize_roll)
    combined["Identity_Correct"] = (
        combined["Review_Status"].eq("identified")
        & combined["Actual_Roll"].eq(combined["Predicted_Roll"])
    )
    combined["Unsafe_Identity_Error"] = (
        combined["Review_Status"].isin({"identified", "mixed_track", "not_in_mapping"})
        & ~combined["Identity_Correct"]
    )
    combined["Unverifiable"] = combined["Review_Status"].isin(
        {"unidentifiable", "uncertain", ""}
    )
    status_rows = status_df.to_dict("records")

    roster, _, subject_rolls = load_student_mapping(inputs.student_map, SUBJECT_ABBR)
    subject_roster = [row for row in roster if row["in_subject_roster"]]
    status_lookup = {row["Roll"]: row for row in status_rows}
    roster_rows = []
    for row in sorted(subject_roster, key=lambda item: normalize_roll(item["roll"])):
        roll = normalize_roll(row["roll"])
        evidence = status_lookup.get(roll, {})
        roster_rows.append({
            "Roll": roll,
            "Name": row.get("name", ""),
            "Checkpoint_Count": evidence.get("Confirmed_Checkpoint_Count", 0),
            "Checkpoints": evidence.get("Confirmed_Checkpoints", ""),
            "Tracklet_Shadow_Status": evidence.get("Shadow_Status", "Unconfirmed"),
            "Evidence_Interpretation": (
                "Confirmed by reviewed multi-frame evidence"
                if evidence.get("Confirmed_Checkpoint_Count", 0)
                else "Insufficient evidence; not proven absent"
            ),
            "Official_Attendance_Changed": "No",
        })
    if len(roster_rows) != 27:
        raise ProductPhase2HError(f"Expected 27 roster-first shadow rows; found {len(roster_rows)}")

    complete = len(new_review) == EXPECTED_PENDING_REVIEW and not new_review["Review_Status"].eq("").any()
    strict_unsafe = int(combined.loc[combined["Evidence_Tier"].eq("strict_accepted"), "Unsafe_Identity_Error"].sum())
    recovery_unsafe = int(combined.loc[combined["Evidence_Tier"].eq("guarded_recovery"), "Unsafe_Identity_Error"].sum())
    strict_unverifiable = int(combined.loc[combined["Evidence_Tier"].eq("strict_accepted"), "Unverifiable"].sum())
    recovery_unverifiable = int(combined.loc[combined["Evidence_Tier"].eq("guarded_recovery"), "Unverifiable"].sum())
    if not complete:
        decision = "hold_incomplete_multiframe_review"
    elif strict_unsafe:
        decision = "tracklet_authority_rejected_strict_identity_error"
    elif recovery_unsafe:
        decision = "multiframe_recovery_rule_rejected_identity_error"
    elif strict_unverifiable or recovery_unverifiable:
        decision = "hold_unverifiable_multiframe_evidence"
    else:
        decision = "multiframe_recovery_passed_pending_explicit_authority_patch"

    staging = output_dir / "evaluation.staging"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    _write_csv(staging / "joined_multiframe_recovery_review.csv", combined)
    _write_csv(staging / "identity_checkpoint_validation.csv", status_df)
    _write_csv(staging / "multiframe_shadow_roster_report.csv", pd.DataFrame(roster_rows))
    summary = {
        "schema_version": 1,
        "policy_version": POLICY_VERSION,
        "decision": decision,
        "reviewed_material_pairs": len(combined),
        "recovery_rule_id": RECOVERY_RULE_ID,
        "frozen_leakage_safe_validation": {"correct": 8, "unsafe": 0, "unverifiable": 0},
        "reused_phase_1_2n_pairs": len(reused),
        "new_blind_review_pairs": len(new_review),
        "unsafe_identity_errors": unsafe,
        "strict_accepted_unsafe_errors": strict_unsafe,
        "guarded_recovery_unsafe_errors": recovery_unsafe,
        "unverifiable_pairs": unverifiable,
        "strict_accepted_unverifiable": strict_unverifiable,
        "guarded_recovery_unverifiable": recovery_unverifiable,
        "present_rolls": sorted(status_df.loc[status_df["Shadow_Status"].eq("Present"), "Roll"].tolist()),
        "needs_review_rolls": sorted(status_df.loc[status_df["Shadow_Status"].eq("Needs Review"), "Roll"].tolist()),
        "frame_official_result": {"present": 0, "needs_review": 1, "absent": 26},
        "multiframe_shadow_result": {
            "present": int(status_df["Shadow_Status"].eq("Present").sum()),
            "needs_review": int(status_df["Shadow_Status"].eq("Needs Review").sum()),
            "unconfirmed": 27 - int(status_df["Shadow_Status"].isin(["Present", "Needs Review"]).sum()),
            "absent": 0,
        },
        "recognition_repeated": False,
        "official_attendance_changed": False,
        "tracklets_activated": False,
    }
    (staging / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    evaluation_manifest = {
        "schema_version": 1,
        "policy_version": POLICY_VERSION,
        "labels_sha256": _sha256_file(Path(labels_path)),
        "files": _manifest_files(staging),
    }
    (staging / "evaluation_manifest.json").write_text(json.dumps(evaluation_manifest, indent=2), encoding="utf-8")
    staging.replace(evaluation_dir)
    return evaluation_dir
