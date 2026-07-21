from __future__ import annotations

import json
import math
import os
import pickle
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from .config import DEFAULT_DETECTION_SCORE, DEFAULT_MAX_WIDTH, SFACE_MODEL, YUNET_MODEL
from .embedding_version_family import EmbeddingFamilyError, verify_embedding_family
from .face_engine import FaceEngine
from .mon_p3_retention_attribution import (
    EXPECTED_EVALUATION_ID,
    EXPECTED_FAMILY_ID,
    EXPECTED_PRODUCTION_EMBEDDINGS_SHA256,
    EXPECTED_PRODUCTION_SUMMARY_SHA256,
    EXPECTED_RUN_ID,
    RetentionAttributionError,
    _aggregate_track,
    _canonical,
    _float,
    _load_json,
    _required_int,
    _sha256_file,
    _stable_digest,
    _text,
    _video_map,
    resolve_inputs as resolve_retention_inputs,
    verify_attribution_output,
)
from .recall_analysis import MATCH_THRESHOLD, MARGIN_THRESHOLD
from .shadow_validation import ShadowValidationError, verify_output_manifest, write_output_manifest
from .utils import normalize_embedding
from .zones import load_camera_zone_config

SCHEMA_VERSION = 1
POLICY_VERSION = "phase-1.2l-source-subset-ablation-v1"
EXPECTED_ATTRIBUTION_ID = "retention-attribution-3f10a2ef843d58e92ced"
EXPECTED_ATTRIBUTION_DECISION = "family_rejected_mon_p3_retention_attributed"
EXPECTED_BENCHMARK_ID = "multisession-gt-34581599118b530c"
EXPECTED_BENCHMARK_ROWS = 120
EXPECTED_MON_P3_ROWS = 15
EXPECTED_IMPLICATED_SOURCES = 5
EXPECTED_SUBSET_COUNT = 32
TUE_P1_SESSION = "2026-06-30__B51__P1__CVO"
TUE_P2_SESSION = "2026-06-30__B51__P2__CVO"
MON_P3_SESSION = "2026-06-22__B51__P3__CVO"
FULL_VARIANT_LABEL = "D_full_candidate"
SCORE_TOLERANCE = 0.00001


class SourceAblationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ImplicatedSource:
    source_id: str
    canonical_roll: str
    source_kind: str
    source_session: str
    short_label: str
    expected_image_suffix: str


IMPLICATED_SOURCES: tuple[ImplicatedSource, ...] = (
    ImplicatedSource(
        source_id="CCTV-CCC-16b0462d0f283edb9f",
        canonical_roll="2401100CSE0121",
        source_kind="human_approved_cctv_crop",
        source_session=TUE_P1_SESSION,
        short_label="0121_p1_crop_16b046",
        expected_image_suffix="candidate_sources/verified_cctv/tue_p1/CCC-16b0462d0f283edb9f.png",
    ),
    ImplicatedSource(
        source_id="CCTV-CCC-492ad8ac8e05bd2cac",
        canonical_roll="2401100CSE0121",
        source_kind="human_approved_cctv_crop",
        source_session=TUE_P1_SESSION,
        short_label="0121_p1_crop_492ad8",
        expected_image_suffix="candidate_sources/verified_cctv/tue_p1/CCC-492ad8ac8e05bd2cac.png",
    ),
    ImplicatedSource(
        source_id="CCTV-CCC-9852dcae08d8e171db",
        canonical_roll="2401100CSE0121",
        source_kind="human_approved_cctv_crop",
        source_session=TUE_P1_SESSION,
        short_label="0121_p1_crop_9852dc",
        expected_image_suffix="candidate_sources/verified_cctv/tue_p1/CCC-9852dcae08d8e171db.png",
    ),
    ImplicatedSource(
        source_id=(
            "RECOVERY-cctv-recovery-69e11ea931f79330626d-"
            "CCC-40a4e971227bc88cd6"
        ),
        canonical_roll="2401100CSE0140",
        source_kind="same_track_recovered_cctv_medoid",
        source_session=TUE_P2_SESSION,
        short_label="0140_p2_cp2_medoid_40a4e9",
        expected_image_suffix=(
            "candidate_sources/same_track_recovered_cctv/tue_p2/"
            "CCC-40a4e971227bc88cd6__CP2_cam5_back.mp4_f372_zone_back_middle_left_4.png"
        ),
    ),
    ImplicatedSource(
        source_id=(
            "RECOVERY-cctv-recovery-69e11ea931f79330626d-"
            "CCC-debcd075bb2e58209a"
        ),
        canonical_roll="2401100CSE0140",
        source_kind="same_track_recovered_cctv_medoid",
        source_session=TUE_P2_SESSION,
        short_label="0140_p2_cp1_medoid_debcd0",
        expected_image_suffix=(
            "candidate_sources/same_track_recovered_cctv/tue_p2/"
            "CCC-debcd075bb2e58209a__CP1_cam5_back.mp4_f468_zone_back_middle_left_2.png"
        ),
    ),
)


@dataclass(frozen=True)
class SourceAblationInputs:
    repo_root: Path
    family_dir: Path
    shadow_output: Path
    evaluation_dir: Path
    attribution_dir: Path
    production_embeddings: Path
    production_summary: Path
    candidate_diagnostic: Path
    candidate_summary: Path
    candidate_observations: Path
    reviewed_tracks: Path
    full_variant_embeddings: Path
    full_variant_source_records: Path
    benchmark_feature_cache: Path
    benchmark_feature_manifest: Path
    production_benchmark_predictions: Path
    full_candidate_benchmark_predictions: Path
    camera_zones: Path
    fingerprint_sha256: str
    ablation_id: str
    output_dir: Path


@dataclass(frozen=True)
class VectorizedTop3Index:
    rolls: tuple[str, ...]
    record_matrix: np.ndarray
    roll_record_indices: tuple[np.ndarray, ...]

    @classmethod
    def from_records(cls, records: Sequence[dict[str, Any]]) -> "VectorizedTop3Index":
        if not records:
            raise SourceAblationError("Embedding records are empty")
        vectors: list[np.ndarray] = []
        by_roll: dict[str, list[int]] = {}
        for index, record in enumerate(records):
            roll = _canonical(record.get("roll_no"))
            if not roll:
                raise SourceAblationError(f"Embedding record {index} has no roll number")
            vector = normalize_embedding(np.asarray(record.get("embedding"), dtype=np.float32))
            if vector.size != 128 or not np.all(np.isfinite(vector)):
                raise SourceAblationError(f"Embedding record {index} is invalid")
            vectors.append(vector)
            by_roll.setdefault(roll, []).append(index)
        rolls = tuple(sorted(by_roll))
        return cls(
            rolls=rolls,
            record_matrix=np.stack(vectors).astype(np.float32, copy=False),
            roll_record_indices=tuple(
                np.asarray(by_roll[roll], dtype=np.int64) for roll in rolls
            ),
        )

    def score(
        self,
        queries: np.ndarray,
        *,
        active_record_mask: np.ndarray | None = None,
        match_threshold: float = MATCH_THRESHOLD,
        margin_threshold: float = MARGIN_THRESHOLD,
    ) -> pd.DataFrame:
        matrix = np.asarray(queries, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape[1] != 128:
            raise SourceAblationError(
                f"Query matrix must have shape (N, 128); found {matrix.shape}"
            )
        matrix = np.stack([normalize_embedding(row) for row in matrix]).astype(
            np.float32, copy=False
        )
        if active_record_mask is None:
            active = np.ones(len(self.record_matrix), dtype=bool)
        else:
            active = np.asarray(active_record_mask, dtype=bool).reshape(-1)
            if active.size != len(self.record_matrix):
                raise SourceAblationError(
                    "Active-record mask length does not match embedding records"
                )
        similarities = matrix @ self.record_matrix.T
        student_scores = np.full(
            (len(matrix), len(self.rolls)), -np.inf, dtype=np.float32
        )
        for roll_index, record_indices in enumerate(self.roll_record_indices):
            selected_indices = record_indices[active[record_indices]]
            if not len(selected_indices):
                continue
            values = similarities[:, selected_indices]
            count = min(3, len(selected_indices))
            if len(selected_indices) > count:
                top = np.partition(values, len(selected_indices) - count, axis=1)[
                    :, -count:
                ]
            else:
                top = values
            student_scores[:, roll_index] = np.mean(
                top, axis=1, dtype=np.float32
            )
        if np.any(np.all(~np.isfinite(student_scores), axis=1)):
            raise SourceAblationError("A query has no active student embeddings")

        order = np.argsort(-student_scores, axis=1, kind="stable")
        best_indices = order[:, 0]
        second_indices = order[:, 1] if len(self.rolls) > 1 else best_indices
        rows: list[dict[str, Any]] = []
        for row_index, (best_index, second_index) in enumerate(
            zip(best_indices, second_indices)
        ):
            best_score = float(student_scores[row_index, best_index])
            if len(self.rolls) > 1:
                second_score = float(student_scores[row_index, second_index])
                second_roll = self.rolls[int(second_index)]
            else:
                second_score = -1.0
                second_roll = ""
            margin = best_score - second_score if second_roll else best_score
            if best_score < match_threshold:
                accepted = False
                reason = "score_below_threshold"
            elif margin < margin_threshold:
                accepted = False
                reason = "margin_too_small"
            else:
                accepted = True
                reason = "accepted"
            rows.append(
                {
                    "Accepted": accepted,
                    "Best_Roll": self.rolls[int(best_index)],
                    "Best_Score": best_score,
                    "Second_Roll": second_roll,
                    "Second_Score": second_score,
                    "Margin": margin,
                    "Reason": reason,
                }
            )
        return pd.DataFrame(rows)



def _required_metric(payload: dict[str, Any], key: str) -> int:
    try:
        return _required_int(payload, key)
    except RetentionAttributionError as exc:
        raise SourceAblationError(str(exc)) from exc


def _required_bool(value: Any, *, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = _text(value).lower()
    if text in {"true", "yes", "1", "y"}:
        return True
    if text in {"false", "no", "0", "n", ""}:
        return False
    raise SourceAblationError(f"Invalid boolean for {label}: {value}")


def _load_pickle_dict(path: Path, *, label: str) -> dict[str, Any]:
    if not Path(path).is_file():
        raise SourceAblationError(f"{label} not found: {path}")
    try:
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
    except (OSError, pickle.UnpicklingError, EOFError) as exc:
        raise SourceAblationError(f"Unable to read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise SourceAblationError(f"{label} has an invalid pickle payload")
    return payload


def _load_embedding_payload(path: Path, *, label: str) -> dict[str, Any]:
    payload = _load_pickle_dict(path, label=label)
    if not isinstance(payload.get("records"), list):
        raise SourceAblationError(f"{label} has an invalid embedding payload")
    return payload


def _ordered_source_records(path: Path) -> pd.DataFrame:
    if not Path(path).is_file():
        raise SourceAblationError(f"Source records not found: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {
        "Record_Order",
        "Source_ID",
        "Canonical_Roll",
        "Source_Kind",
        "Source_Session",
        "Image_Path",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise SourceAblationError(
            "Source-record schema is incomplete: " + ", ".join(missing)
        )
    try:
        order = frame["Record_Order"].astype(int)
    except ValueError as exc:
        raise SourceAblationError("Source-record order contains a non-integer") from exc
    if order.duplicated().any() or sorted(order.tolist()) != list(range(len(frame))):
        raise SourceAblationError("Source-record order must be a complete zero-based sequence")
    frame = frame.assign(_order=order).sort_values("_order", kind="stable")
    return frame.drop(columns="_order").reset_index(drop=True)


def validate_record_alignment(
    source_records: pd.DataFrame, records: Sequence[dict[str, Any]]
) -> None:
    if len(source_records) != len(records):
        raise SourceAblationError(
            f"Embedding/source-record length mismatch: {len(records)} vs {len(source_records)}"
        )
    for index, (source, record) in enumerate(
        zip(source_records.to_dict("records"), records)
    ):
        expected_roll = _canonical(source.get("Canonical_Roll"))
        actual_roll = _canonical(record.get("roll_no"))
        if expected_roll != actual_roll:
            raise SourceAblationError(
                f"Embedding/source roll mismatch at record {index}: {actual_roll} != {expected_roll}"
            )
        if _text(source.get("Image_Path")).replace("\\", "/") != _text(
            record.get("image_path")
        ).replace("\\", "/"):
            raise SourceAblationError(
                f"Embedding/source image mismatch at record {index}"
            )
        vector = normalize_embedding(np.asarray(record.get("embedding"), dtype=np.float32))
        if vector.size != 128 or not np.all(np.isfinite(vector)):
            raise SourceAblationError(f"Invalid embedding at record {index}")


def validate_implicated_sources(source_records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bit_index, spec in enumerate(IMPLICATED_SOURCES):
        matches = source_records[source_records["Source_ID"].eq(spec.source_id)]
        if len(matches) != 1:
            raise SourceAblationError(
                f"Expected exactly one implicated source {spec.source_id}; found {len(matches)}"
            )
        row = matches.iloc[0].to_dict()
        failures: list[str] = []
        if _canonical(row.get("Canonical_Roll")) != spec.canonical_roll:
            failures.append("roll")
        if _text(row.get("Source_Kind")) != spec.source_kind:
            failures.append("source kind")
        if _text(row.get("Source_Session")) != spec.source_session:
            failures.append("source session")
        image = _text(row.get("Image_Path")).replace("\\", "/")
        if not image.endswith(spec.expected_image_suffix):
            failures.append("image path")
        if failures:
            raise SourceAblationError(
                f"Implicated source contract changed for {spec.source_id}: "
                + ", ".join(failures)
            )
        rows.append(
            {
                "Bit_Index": bit_index,
                "Bit_Value": 1 << bit_index,
                "Short_Label": spec.short_label,
                "Source_ID": spec.source_id,
                "Canonical_Roll": spec.canonical_roll,
                "Source_Kind": spec.source_kind,
                "Source_Session": spec.source_session,
                "Record_Order": int(row["Record_Order"]),
                "Image_Path": image,
            }
        )
    result = pd.DataFrame(rows).sort_values("Bit_Index", kind="stable")
    if len(result) != EXPECTED_IMPLICATED_SOURCES:
        raise SourceAblationError("Implicated-source inventory count changed")
    return result.reset_index(drop=True)


def enumerate_source_subsets() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    all_ids = [spec.source_id for spec in IMPLICATED_SOURCES]
    all_labels = [spec.short_label for spec in IMPLICATED_SOURCES]
    for mask in range(1 << len(IMPLICATED_SOURCES)):
        kept_ids = [source_id for index, source_id in enumerate(all_ids) if mask & (1 << index)]
        removed_ids = [source_id for source_id in all_ids if source_id not in kept_ids]
        kept_labels = [label for index, label in enumerate(all_labels) if mask & (1 << index)]
        digest = _stable_digest({"mask": mask, "kept_source_ids": kept_ids})[:8]
        rows.append(
            {
                "Config_ID": f"ABL-{mask:02d}-{digest}",
                "Mask": mask,
                "Kept_Source_Count": len(kept_ids),
                "Removed_Source_Count": len(removed_ids),
                "Kept_0121_P1_Crop_Count": sum(
                    1
                    for index, spec in enumerate(IMPLICATED_SOURCES)
                    if spec.canonical_roll == "2401100CSE0121" and mask & (1 << index)
                ),
                "Kept_0140_P2_Medoid_Count": sum(
                    1
                    for index, spec in enumerate(IMPLICATED_SOURCES)
                    if spec.canonical_roll == "2401100CSE0140" and mask & (1 << index)
                ),
                "Kept_Source_Labels": "|".join(kept_labels),
                "Kept_Source_IDs": "|".join(kept_ids),
                "Removed_Source_IDs": "|".join(removed_ids),
            }
        )
    if len(rows) != EXPECTED_SUBSET_COUNT:
        raise SourceAblationError("Source-subset enumeration count changed")
    return rows


def _active_mask_for_config(
    source_records: pd.DataFrame,
    *,
    kept_source_ids: set[str],
    evaluation_mode: str,
) -> np.ndarray:
    active = np.ones(len(source_records), dtype=bool)
    implicated = {spec.source_id for spec in IMPLICATED_SOURCES}
    for index, row in enumerate(source_records.to_dict("records")):
        source_id = _text(row.get("Source_ID"))
        if source_id in implicated and source_id not in kept_source_ids:
            active[index] = False
        source_kind = _text(row.get("Source_Kind"))
        source_session = _text(row.get("Source_Session"))
        if evaluation_mode == "tue_p1_leakage_safe":
            if source_kind == "human_approved_cctv_crop" and source_session == TUE_P1_SESSION:
                active[index] = False
        elif evaluation_mode == "tue_p2_leakage_safe":
            if (
                source_kind == "same_track_recovered_cctv_medoid"
                and source_session == TUE_P2_SESSION
            ):
                active[index] = False
        elif evaluation_mode != "full_composition":
            raise SourceAblationError(f"Unknown evaluation mode: {evaluation_mode}")
    return active


def _assert_score_reproduction(
    *,
    scored: pd.DataFrame,
    recorded: pd.DataFrame,
    id_column: str,
    recorded_prefix: str,
    label: str,
) -> None:
    required = {
        id_column,
        f"{recorded_prefix}Accepted",
        f"{recorded_prefix}Best_Roll",
        f"{recorded_prefix}Best_Score",
        f"{recorded_prefix}Second_Roll",
        f"{recorded_prefix}Second_Score",
        f"{recorded_prefix}Margin",
        f"{recorded_prefix}Reason",
    }
    missing = sorted(required.difference(recorded.columns))
    if missing:
        raise SourceAblationError(
            f"{label} recorded reproduction schema is incomplete: " + ", ".join(missing)
        )
    if scored[id_column].duplicated().any() or recorded[id_column].duplicated().any():
        raise SourceAblationError(f"{label} reproduction IDs are not unique")
    joined = scored.merge(
        recorded[list(required)], on=id_column, how="left", validate="one_to_one"
    )
    if len(joined) != len(scored) or joined[f"{recorded_prefix}Best_Roll"].isna().any():
        raise SourceAblationError(f"{label} reproduction join failed")
    failures: list[str] = []
    for row in joined.to_dict("records"):
        row_id = _text(row[id_column])
        actual_accepted = _required_bool(row["Accepted"], label=f"{label} accepted")
        expected_accepted = _required_bool(
            row[f"{recorded_prefix}Accepted"], label=f"{label} recorded accepted"
        )
        actual_best = _canonical(row["Best_Roll"])
        expected_best = _canonical(row[f"{recorded_prefix}Best_Roll"])
        actual_second = _canonical(row["Second_Roll"])
        expected_second = _canonical(row[f"{recorded_prefix}Second_Roll"])
        actual_reason = _text(row["Reason"])
        expected_reason = _text(row[f"{recorded_prefix}Reason"])
        if (
            actual_accepted != expected_accepted
            or actual_best != expected_best
            or actual_second != expected_second
            or actual_reason != expected_reason
        ):
            failures.append(row_id)
            continue
        for field in ("Best_Score", "Second_Score", "Margin"):
            observed = _float(row[field], math.nan)
            expected = _float(row[f"{recorded_prefix}{field}"], math.nan)
            if not math.isfinite(observed) or abs(observed - expected) > SCORE_TOLERANCE:
                failures.append(row_id)
                break
    if failures:
        raise SourceAblationError(
            f"{label} exact score reproduction failed: " + ", ".join(failures[:10])
        )


def _benchmark_recorded_frame(path: Path, *, model: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {
        "Benchmark_Row_ID",
        "Package_ID",
        "Review_ID",
        "Tracklet_ID",
        "Session_ID",
        "Source_Kind",
        "Baseline_Status",
        "Checkpoint_ID",
        "Camera_ID",
        "Review_Status",
        "Actual_Class",
        "Actual_Roll",
        "Accepted",
        "Top1_Roll",
        "Top2_Roll",
        "Top1_Score",
        "Top2_Score",
        "Margin",
        "Matcher_Reason",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise SourceAblationError(
            f"{model} benchmark predictions are incomplete: " + ", ".join(missing)
        )
    if len(frame) != EXPECTED_BENCHMARK_ROWS or frame["Benchmark_Row_ID"].duplicated().any():
        raise SourceAblationError(f"{model} benchmark prediction row count changed")
    frame = frame.copy()
    frame["Actual_Roll"] = frame["Actual_Roll"].map(_canonical)
    return frame.sort_values("Benchmark_Row_ID", kind="stable").reset_index(drop=True)


def _recorded_for_reproduction(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(
        columns={
            "Accepted": "Recorded_Accepted",
            "Top1_Roll": "Recorded_Best_Roll",
            "Top1_Score": "Recorded_Best_Score",
            "Top2_Roll": "Recorded_Second_Roll",
            "Top2_Score": "Recorded_Second_Score",
            "Margin": "Recorded_Margin",
            "Matcher_Reason": "Recorded_Reason",
        }
    )


def _load_benchmark_features(path: Path, manifest_path: Path) -> tuple[list[str], np.ndarray]:
    manifest = _load_json(manifest_path, "benchmark feature manifest")
    if _text(manifest.get("benchmark_id")) != EXPECTED_BENCHMARK_ID:
        raise SourceAblationError("Frozen benchmark ID changed")
    if int(manifest.get("feature_rows") or 0) != EXPECTED_BENCHMARK_ROWS:
        raise SourceAblationError("Frozen benchmark feature row count changed")
    payload = _load_pickle_dict(path, label="benchmark feature cache")
    if _text(payload.get("benchmark_id")) != EXPECTED_BENCHMARK_ID:
        raise SourceAblationError("Benchmark cache ID changed")
    features = payload.get("features")
    if not isinstance(features, list) or len(features) != EXPECTED_BENCHMARK_ROWS:
        raise SourceAblationError("Benchmark feature cache is incomplete")
    rows: list[tuple[str, np.ndarray]] = []
    for item in features:
        row_id = _text(item.get("benchmark_row_id"))
        vector = normalize_embedding(np.asarray(item.get("embedding"), dtype=np.float32))
        if not row_id or vector.size != 128 or not np.all(np.isfinite(vector)):
            raise SourceAblationError("Benchmark feature cache contains an invalid row")
        rows.append((row_id, vector))
    rows.sort(key=lambda item: item[0])
    ids = [row_id for row_id, _ in rows]
    if len(set(ids)) != EXPECTED_BENCHMARK_ROWS:
        raise SourceAblationError("Benchmark feature cache contains duplicate IDs")
    return ids, np.stack([vector for _, vector in rows]).astype(np.float32)


def _attach_ids(scored: pd.DataFrame, ids: Sequence[str], id_column: str) -> pd.DataFrame:
    if len(scored) != len(ids):
        raise SourceAblationError("Score output length does not match query IDs")
    result = scored.copy()
    result.insert(0, id_column, list(ids))
    return result


def _prediction_frame(
    metadata: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    id_column: str,
    model_id: str,
    evidence_mode: str,
    promotion_evidence_eligible: bool,
) -> pd.DataFrame:
    model_output_columns = {
        "Model_ID",
        "Feature_Available",
        "Top1_Roll",
        "Top2_Roll",
        "Top1_Score",
        "Top2_Score",
        "Margin",
        "Match_Threshold",
        "Margin_Threshold",
        "Aggregate_Mode",
        "Accepted",
        "Accepted_Roll",
        "Predicted_Identity_In_Authoritative_Roster",
        "Correct_Accepted",
        "Wrong_Accepted",
        "Outsider_Absorption",
        "Mixed_Acceptance",
        "Unverifiable_Acceptance",
        "Correct_Unresolved",
        "Incorrect_Unresolved",
        "Raw_Top1_Correct",
        "Matcher_Reason",
        "Evidence_Mode",
        "Training_Contaminated_Descriptive",
        "Promotion_Evidence_Eligible",
        "Best_Roll",
        "Best_Score",
        "Second_Roll",
        "Second_Score",
        "Reason",
    }
    clean_metadata = metadata.drop(
        columns=[column for column in model_output_columns if column in metadata.columns]
    )
    joined = clean_metadata.merge(
        scores, on=id_column, how="left", validate="one_to_one"
    )
    if len(joined) != len(metadata) or joined["Best_Roll"].isna().any():
        raise SourceAblationError(f"Prediction join failed for {model_id}")
    actual_class = joined["Actual_Class"].astype(str)
    actual_roll = joined["Actual_Roll"].map(_canonical)
    accepted = joined["Accepted"].map(bool)
    accepted_roll = joined["Best_Roll"].where(accepted, "").map(_canonical)
    known = actual_class.eq("known_student")
    result = joined.copy()
    result.insert(0, "Model_ID", model_id)
    result["Accepted_Roll"] = accepted_roll
    result["Correct_Accepted"] = accepted & known & accepted_roll.eq(actual_roll)
    result["Wrong_Accepted"] = accepted & known & ~accepted_roll.eq(actual_roll)
    result["Outsider_Absorption"] = accepted & actual_class.eq("not_in_mapping")
    result["Mixed_Acceptance"] = accepted & actual_class.eq("mixed")
    result["Unverifiable_Acceptance"] = accepted & actual_class.eq("unverifiable")
    result["Correct_Unresolved"] = ~accepted & ~known
    result["Incorrect_Unresolved"] = ~accepted & known
    result["Raw_Top1_Correct"] = known & result["Best_Roll"].map(_canonical).eq(actual_roll)
    result["Evidence_Mode"] = evidence_mode
    result["Promotion_Evidence_Eligible"] = promotion_evidence_eligible
    return result


def _prediction_metrics(frame: pd.DataFrame) -> dict[str, int]:
    unsafe = (
        frame["Wrong_Accepted"].map(bool)
        | frame["Outsider_Absorption"].map(bool)
        | frame["Mixed_Acceptance"].map(bool)
    )
    return {
        "rows": int(len(frame)),
        "accepted": int(frame["Accepted"].map(bool).sum()),
        "correct_accepted": int(frame["Correct_Accepted"].map(bool).sum()),
        "wrong_accepted": int(frame["Wrong_Accepted"].map(bool).sum()),
        "outsider_absorption": int(frame["Outsider_Absorption"].map(bool).sum()),
        "mixed_acceptance": int(frame["Mixed_Acceptance"].map(bool).sum()),
        "unverifiable_acceptance": int(
            frame["Unverifiable_Acceptance"].map(bool).sum()
        ),
        "unsafe_confirmed_acceptances": int(unsafe.sum()),
        "raw_top1_correct": int(frame["Raw_Top1_Correct"].map(bool).sum()),
    }


def _comparison_metrics(
    production: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    id_column: str,
) -> tuple[dict[str, int], pd.DataFrame]:
    fields = [
        id_column,
        "Actual_Class",
        "Actual_Roll",
        "Accepted",
        "Accepted_Roll",
        "Correct_Accepted",
        "Wrong_Accepted",
        "Outsider_Absorption",
        "Mixed_Acceptance",
        "Unverifiable_Acceptance",
        "Best_Roll",
        "Second_Roll",
        "Best_Score",
        "Second_Score",
        "Margin",
        "Reason",
    ]
    joined = production[fields].merge(
        candidate[fields],
        on=id_column,
        how="inner",
        validate="one_to_one",
        suffixes=("_Production", "_Candidate"),
    )
    if len(joined) != len(production) or len(joined) != len(candidate):
        raise SourceAblationError("Production/candidate comparison row count changed")
    joined["Lost_Correct_Production_Accept"] = (
        joined["Correct_Accepted_Production"].map(bool)
        & ~joined["Correct_Accepted_Candidate"].map(bool)
    )
    joined["Recovered_Correct_Accept"] = (
        ~joined["Correct_Accepted_Production"].map(bool)
        & joined["Correct_Accepted_Candidate"].map(bool)
    )
    joined["New_Wrong_Accept"] = (
        ~joined["Wrong_Accepted_Production"].map(bool)
        & joined["Wrong_Accepted_Candidate"].map(bool)
    )
    joined["New_Outsider_Absorption"] = (
        ~joined["Outsider_Absorption_Production"].map(bool)
        & joined["Outsider_Absorption_Candidate"].map(bool)
    )
    joined["New_Mixed_Acceptance"] = (
        ~joined["Mixed_Acceptance_Production"].map(bool)
        & joined["Mixed_Acceptance_Candidate"].map(bool)
    )
    joined["New_Unverifiable_Acceptance"] = (
        ~joined["Unverifiable_Acceptance_Production"].map(bool)
        & joined["Unverifiable_Acceptance_Candidate"].map(bool)
    )
    metrics = {
        "lost_correct_production_accepts": int(
            joined["Lost_Correct_Production_Accept"].sum()
        ),
        "recovered_correct_accepts": int(joined["Recovered_Correct_Accept"].sum()),
        "new_wrong_accepts": int(joined["New_Wrong_Accept"].sum()),
        "new_outsider_absorptions": int(joined["New_Outsider_Absorption"].sum()),
        "new_mixed_acceptances": int(joined["New_Mixed_Acceptance"].sum()),
        "new_unverifiable_acceptances": int(
            joined["New_Unverifiable_Acceptance"].sum()
        ),
    }
    return metrics, joined


def _pareto_frontier(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    objectives_max = [
        "MON_P3_Preserved_Prior_Candidate_Recoveries",
        "MON_P3_Recovered_Correct_vs_Production",
        "Benchmark_Safe_Recovered_Correct_vs_Production",
        "Benchmark_Safe_Correct_Accepted",
    ]
    objective_min = "Kept_Source_Count"
    rows = frame.to_dict("records")
    keep: list[dict[str, Any]] = []
    for candidate in rows:
        dominated = False
        for other in rows:
            if other["Config_ID"] == candidate["Config_ID"]:
                continue
            no_worse = all(other[key] >= candidate[key] for key in objectives_max) and (
                other[objective_min] <= candidate[objective_min]
            )
            strictly_better = any(
                other[key] > candidate[key] for key in objectives_max
            ) or other[objective_min] < candidate[objective_min]
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            keep.append(candidate)
    return pd.DataFrame(keep).sort_values(
        [
            "MON_P3_Preserved_Prior_Candidate_Recoveries",
            "MON_P3_Recovered_Correct_vs_Production",
            "Benchmark_Safe_Recovered_Correct_vs_Production",
            "Benchmark_Safe_Correct_Accepted",
            "Kept_Source_Count",
            "Mask",
        ],
        ascending=[False, False, False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)


def rank_configurations(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "Hard_Gates_Passed",
        "MON_P3_Preserved_Prior_Candidate_Recoveries",
        "MON_P3_Recovered_Correct_vs_Production",
        "Benchmark_Safe_Recovered_Correct_vs_Production",
        "Benchmark_Safe_Correct_Accepted",
        "Benchmark_Descriptive_Correct_Accepted",
        "Kept_Source_Count",
        "Mask",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise SourceAblationError(
            "Configuration ranking schema is incomplete: " + ", ".join(missing)
        )
    result = frame.sort_values(
        [
            "Hard_Gates_Passed",
            "MON_P3_Preserved_Prior_Candidate_Recoveries",
            "MON_P3_Recovered_Correct_vs_Production",
            "Benchmark_Safe_Recovered_Correct_vs_Production",
            "Benchmark_Safe_Correct_Accepted",
            "Benchmark_Descriptive_Correct_Accepted",
            "Kept_Source_Count",
            "Mask",
        ],
        ascending=[False, False, False, False, False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    result.insert(0, "Rank", np.arange(1, len(result) + 1))
    return result


def _source_inventory_fingerprint() -> list[dict[str, Any]]:
    return [
        {
            "source_id": spec.source_id,
            "canonical_roll": spec.canonical_roll,
            "source_kind": spec.source_kind,
            "source_session": spec.source_session,
            "short_label": spec.short_label,
            "expected_image_suffix": spec.expected_image_suffix,
        }
        for spec in IMPLICATED_SOURCES
    ]


def resolve_inputs(
    *,
    repo_root: Path,
    family_dir: Path,
    shadow_output: Path,
    evaluation_dir: Path,
    attribution_dir: Path,
    output_root: Path | None = None,
) -> SourceAblationInputs:
    repo_root = Path(repo_root).resolve()
    family_dir = Path(family_dir).resolve()
    shadow_output = Path(shadow_output).resolve()
    evaluation_dir = Path(evaluation_dir).resolve()
    attribution_dir = Path(attribution_dir).resolve()
    try:
        retention_inputs = resolve_retention_inputs(
            repo_root=repo_root,
            family_dir=family_dir,
            shadow_output=shadow_output,
            evaluation_dir=evaluation_dir,
        )
        attribution_verification = verify_attribution_output(attribution_dir)
        family_verification = verify_embedding_family(family_dir)
    except (RetentionAttributionError, EmbeddingFamilyError) as exc:
        raise SourceAblationError(str(exc)) from exc
    if retention_inputs.attribution_id != EXPECTED_ATTRIBUTION_ID:
        raise SourceAblationError("Phase 1.2K attribution fingerprint changed")
    if _text(attribution_verification.get("decision")) != EXPECTED_ATTRIBUTION_DECISION:
        raise SourceAblationError("Phase 1.2K attribution decision changed")
    if _text(family_verification.get("family_id")) != EXPECTED_FAMILY_ID:
        raise SourceAblationError("Embedding family ID changed")

    attribution_summary = _load_json(
        attribution_dir / "attribution_summary.json", "Phase 1.2K attribution summary"
    )
    if _text(attribution_summary.get("attribution_id")) != EXPECTED_ATTRIBUTION_ID:
        raise SourceAblationError("Unexpected Phase 1.2K attribution ID")
    if _required_metric(attribution_summary, "retention_losses") != 6:
        raise SourceAblationError("Phase 1.2K retention-loss count changed")
    if _required_metric(attribution_summary, "correct_candidate_recoveries") != 4:
        raise SourceAblationError("Phase 1.2K recovery count changed")
    if _required_metric(attribution_summary, "false_identities_or_unsafe_accepts") != 0:
        raise SourceAblationError("Phase 1.2K unsafe-identity count changed")

    full_variant = family_dir / "variants" / "full_candidate"
    evaluation = family_dir / "evaluation"
    full_variant_embeddings = full_variant / "student_embeddings.pkl"
    full_variant_source_records = full_variant / "source_records.csv"
    benchmark_feature_cache = evaluation / "benchmark_feature_cache.pkl"
    benchmark_feature_manifest = evaluation / "benchmark_feature_manifest.json"
    production_benchmark_predictions = evaluation / "production_baseline_predictions.csv"
    full_candidate_benchmark_predictions = evaluation / "full_candidate_descriptive_predictions.csv"
    required_paths = (
        full_variant_embeddings,
        full_variant_source_records,
        benchmark_feature_cache,
        benchmark_feature_manifest,
        production_benchmark_predictions,
        full_candidate_benchmark_predictions,
        attribution_dir / "reviewed_track_variant_scores.csv",
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise SourceAblationError("Required Phase 1.2L input missing: " + ", ".join(missing))

    payload = _load_embedding_payload(full_variant_embeddings, label="full candidate embeddings")
    source_records = _ordered_source_records(full_variant_source_records)
    validate_record_alignment(source_records, payload["records"])
    validate_implicated_sources(source_records)
    _load_benchmark_features(benchmark_feature_cache, benchmark_feature_manifest)
    _benchmark_recorded_frame(production_benchmark_predictions, model="production")
    _benchmark_recorded_frame(full_candidate_benchmark_predictions, model="full candidate")

    production_embeddings = repo_root / "models" / "student_embeddings.pkl"
    production_summary = repo_root / "models" / "embedding_summary.csv"
    if _sha256_file(production_embeddings) != EXPECTED_PRODUCTION_EMBEDDINGS_SHA256:
        raise SourceAblationError("Production embeddings changed before Phase 1.2L")
    if _sha256_file(production_summary) != EXPECTED_PRODUCTION_SUMMARY_SHA256:
        raise SourceAblationError("Production summary changed before Phase 1.2L")

    fingerprint = {
        "policy_version": POLICY_VERSION,
        "family_id": EXPECTED_FAMILY_ID,
        "family_output_manifest_sha256": _sha256_file(family_dir / "output_manifest.json"),
        "phase_1_2j_run_id": EXPECTED_RUN_ID,
        "phase_1_2j_output_manifest_sha256": _sha256_file(
            shadow_output / "output_manifest.json"
        ),
        "phase_1_2j_evaluation_id": EXPECTED_EVALUATION_ID,
        "phase_1_2j_evaluation_manifest_sha256": _sha256_file(
            evaluation_dir / "output_manifest.json"
        ),
        "phase_1_2k_attribution_id": EXPECTED_ATTRIBUTION_ID,
        "phase_1_2k_output_manifest_sha256": _sha256_file(
            attribution_dir / "output_manifest.json"
        ),
        "phase_1_2k_scores_sha256": _sha256_file(
            attribution_dir / "reviewed_track_variant_scores.csv"
        ),
        "production_embeddings_sha256": _sha256_file(production_embeddings),
        "production_summary_sha256": _sha256_file(production_summary),
        "full_candidate_embeddings_sha256": _sha256_file(full_variant_embeddings),
        "full_candidate_source_records_sha256": _sha256_file(full_variant_source_records),
        "benchmark_feature_cache_sha256": _sha256_file(benchmark_feature_cache),
        "benchmark_feature_manifest_sha256": _sha256_file(benchmark_feature_manifest),
        "production_benchmark_predictions_sha256": _sha256_file(
            production_benchmark_predictions
        ),
        "full_candidate_benchmark_predictions_sha256": _sha256_file(
            full_candidate_benchmark_predictions
        ),
        "candidate_summary_sha256": _sha256_file(retention_inputs.candidate_summary),
        "candidate_observations_sha256": _sha256_file(
            retention_inputs.candidate_observations
        ),
        "camera_zones_sha256": _sha256_file(retention_inputs.camera_zones),
        "yunet_sha256": _sha256_file(Path(YUNET_MODEL)),
        "sface_sha256": _sha256_file(Path(SFACE_MODEL)),
        "implicated_sources": _source_inventory_fingerprint(),
        "subset_count": EXPECTED_SUBSET_COUNT,
    }
    fingerprint_sha256 = _stable_digest(fingerprint)
    ablation_id = "source-ablation-" + fingerprint_sha256[:20]
    root = Path(
        output_root
        or repo_root
        / "attendance_output"
        / "embedding_forensics"
        / "phase_1_2l"
    ).resolve()
    return SourceAblationInputs(
        repo_root=repo_root,
        family_dir=family_dir,
        shadow_output=shadow_output,
        evaluation_dir=evaluation_dir,
        attribution_dir=attribution_dir,
        production_embeddings=production_embeddings,
        production_summary=production_summary,
        candidate_diagnostic=retention_inputs.candidate_diagnostic,
        candidate_summary=retention_inputs.candidate_summary,
        candidate_observations=retention_inputs.candidate_observations,
        reviewed_tracks=retention_inputs.reviewed_tracks,
        full_variant_embeddings=full_variant_embeddings,
        full_variant_source_records=full_variant_source_records,
        benchmark_feature_cache=benchmark_feature_cache,
        benchmark_feature_manifest=benchmark_feature_manifest,
        production_benchmark_predictions=production_benchmark_predictions,
        full_candidate_benchmark_predictions=full_candidate_benchmark_predictions,
        camera_zones=retention_inputs.camera_zones,
        fingerprint_sha256=fingerprint_sha256,
        ablation_id=ablation_id,
        output_dir=root / ablation_id,
    )


def _reproduce_mon_p3_features(
    inputs: SourceAblationInputs,
    *,
    yunet_model: Path,
    sface_model: Path,
    status: Callable[[str], None] | None,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    reviewed = pd.read_csv(inputs.reviewed_tracks, dtype=str, keep_default_na=False)
    if len(reviewed) != EXPECTED_MON_P3_ROWS or reviewed["Review_ID"].nunique() != EXPECTED_MON_P3_ROWS:
        raise SourceAblationError("Expected the completed 15-track MON_P3 review")
    if set(reviewed["Review_Status"].str.lower()) != {"identified"}:
        raise SourceAblationError("All Phase 1.2L MON_P3 tracks must be human identified")
    observations = pd.read_csv(
        inputs.candidate_observations, dtype=str, keep_default_na=False
    )
    summary = _load_json(inputs.candidate_summary, "candidate diagnostic summary")
    try:
        video_map = _video_map(summary)
    except RetentionAttributionError as exc:
        raise SourceAblationError(str(exc)) from exc
    zone_config = load_camera_zone_config(inputs.camera_zones)
    engine = FaceEngine(
        Path(yunet_model),
        Path(sface_model),
        detection_score=DEFAULT_DETECTION_SCORE,
    )
    review_rows = reviewed.sort_values("Review_ID", kind="stable").to_dict("records")
    vectors: list[np.ndarray] = []
    audit_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    for ordinal, row in enumerate(review_rows, start=1):
        track_id = _text(row.get("Tracklet_ID"))
        if status:
            status(f"MON_P3 reviewed-track feature {ordinal}/{EXPECTED_MON_P3_ROWS}: {track_id}")
        try:
            aggregate, audit = _aggregate_track(
                track_id=track_id,
                observations=observations,
                video_map=video_map,
                zone_config=zone_config,
                engine=engine,
                max_width=DEFAULT_MAX_WIDTH,
                status=None,
            )
        except RetentionAttributionError as exc:
            raise SourceAblationError(str(exc)) from exc
        vectors.append(aggregate)
        audit_rows.extend(audit)
        metadata_rows.append(
            {
                "Review_ID": _text(row.get("Review_ID")),
                "Tracklet_ID": track_id,
                "Session_ID": MON_P3_SESSION,
                "Package_ID": _text(row.get("Package_ID")),
                "Checkpoint_ID": _text(row.get("Checkpoint_ID")),
                "Camera_ID": _text(row.get("Camera_ID")),
                "Source_Kind": "mon_p3_reviewed_exception_track",
                "Baseline_Status": _text(row.get("Phase_1_2J_Result")),
                "Review_Status": "identified",
                "Actual_Class": "known_student",
                "Actual_Roll": _canonical(row.get("Actual_Roll_Normalized")),
                "Phase_1_2J_Result": _text(row.get("Phase_1_2J_Result")),
            }
        )
    return (
        pd.DataFrame(metadata_rows),
        np.stack(vectors).astype(np.float32),
        pd.DataFrame(audit_rows),
    )


def _mon_recorded_reproduction_frame(inputs: SourceAblationInputs) -> pd.DataFrame:
    frame = pd.read_csv(
        inputs.attribution_dir / "reviewed_track_variant_scores.csv",
        dtype=str,
        keep_default_na=False,
    )
    if len(frame) != EXPECTED_MON_P3_ROWS or frame["Review_ID"].duplicated().any():
        raise SourceAblationError("Phase 1.2K reviewed-track scores changed")
    production = frame[
        [
            "Review_ID",
            "production_Accepted",
            "production_Best_Roll",
            "production_Best_Score",
            "production_Second_Roll",
            "production_Second_Score",
            "production_Margin",
            "production_Reason",
        ]
    ].rename(
        columns={
            "production_Accepted": "Production_Accepted",
            "production_Best_Roll": "Production_Best_Roll",
            "production_Best_Score": "Production_Best_Score",
            "production_Second_Roll": "Production_Second_Roll",
            "production_Second_Score": "Production_Second_Score",
            "production_Margin": "Production_Margin",
            "production_Reason": "Production_Reason",
        }
    )
    candidate = frame[
        [
            "Review_ID",
            "D_full_candidate_Accepted",
            "D_full_candidate_Best_Roll",
            "D_full_candidate_Best_Score",
            "D_full_candidate_Second_Roll",
            "D_full_candidate_Second_Score",
            "D_full_candidate_Margin",
            "D_full_candidate_Reason",
        ]
    ].rename(
        columns={
            "D_full_candidate_Accepted": "Candidate_Accepted",
            "D_full_candidate_Best_Roll": "Candidate_Best_Roll",
            "D_full_candidate_Best_Score": "Candidate_Best_Score",
            "D_full_candidate_Second_Roll": "Candidate_Second_Roll",
            "D_full_candidate_Second_Score": "Candidate_Second_Score",
            "D_full_candidate_Margin": "Candidate_Margin",
            "D_full_candidate_Reason": "Candidate_Reason",
        }
    )
    return production.merge(candidate, on="Review_ID", validate="one_to_one")


def _protected_input_hashes(
    inputs: SourceAblationInputs,
    *,
    yunet_model: Path,
    sface_model: Path,
) -> dict[str, str]:
    protected = {
        "production_embeddings": inputs.production_embeddings,
        "production_summary": inputs.production_summary,
        "full_variant_embeddings": inputs.full_variant_embeddings,
        "full_variant_source_records": inputs.full_variant_source_records,
        "benchmark_feature_cache": inputs.benchmark_feature_cache,
        "benchmark_feature_manifest": inputs.benchmark_feature_manifest,
        "production_benchmark_predictions": inputs.production_benchmark_predictions,
        "full_candidate_benchmark_predictions": (
            inputs.full_candidate_benchmark_predictions
        ),
        "phase_1_2k_output_manifest": inputs.attribution_dir / "output_manifest.json",
        "phase_1_2k_reviewed_track_scores": (
            inputs.attribution_dir / "reviewed_track_variant_scores.csv"
        ),
        "candidate_diagnostic_summary": inputs.candidate_summary,
        "candidate_observations": inputs.candidate_observations,
        "reviewed_mon_p3_tracks": inputs.reviewed_tracks,
        "camera_zones": inputs.camera_zones,
        "yunet_model": Path(yunet_model),
        "sface_model": Path(sface_model),
    }
    return {label: _sha256_file(path) for label, path in protected.items()}


def run_source_ablation(
    inputs: SourceAblationInputs,
    *,
    yunet_model: Path = YUNET_MODEL,
    sface_model: Path = SFACE_MODEL,
    status: Callable[[str], None] | None = print,
) -> tuple[dict[str, Any], Path, bool]:
    if _sha256_file(Path(yunet_model)) != _sha256_file(Path(YUNET_MODEL)):
        raise SourceAblationError(
            "Phase 1.2L YuNet model differs from the fingerprinted project model"
        )
    if _sha256_file(Path(sface_model)) != _sha256_file(Path(SFACE_MODEL)):
        raise SourceAblationError(
            "Phase 1.2L SFace model differs from the fingerprinted project model"
        )

    if inputs.output_dir.exists():
        verified = verify_source_ablation_output(inputs.output_dir)
        existing_summary = _load_json(
            inputs.output_dir / "source_ablation_summary.json",
            "existing source ablation summary",
        )
        expected_hashes = existing_summary.get("protected_input_hashes_after")
        if not isinstance(expected_hashes, dict):
            raise SourceAblationError(
                "Existing Phase 1.2L output has no protected-input hash register"
            )
        current_hashes = _protected_input_hashes(
            inputs, yunet_model=yunet_model, sface_model=sface_model
        )
        changed = sorted(
            label
            for label, expected in expected_hashes.items()
            if current_hashes.get(label) != expected
        )
        if changed or set(current_hashes) != set(expected_hashes):
            raise SourceAblationError(
                "Current protected inputs differ from the existing Phase 1.2L output: "
                + ", ".join(changed or ["hash-register keys changed"])
            )
        return verified, inputs.output_dir, True

    protected_hashes_before = _protected_input_hashes(
        inputs, yunet_model=yunet_model, sface_model=sface_model
    )
    production_hash_before = protected_hashes_before["production_embeddings"]
    production_summary_hash_before = protected_hashes_before["production_summary"]
    payload = _load_embedding_payload(
        inputs.full_variant_embeddings, label="full candidate embeddings"
    )
    source_records = _ordered_source_records(inputs.full_variant_source_records)
    records = payload["records"]
    validate_record_alignment(source_records, records)
    implicated_inventory = validate_implicated_sources(source_records)
    candidate_index = VectorizedTop3Index.from_records(records)
    production_payload = _load_embedding_payload(
        inputs.production_embeddings, label="production embeddings"
    )
    production_index = VectorizedTop3Index.from_records(production_payload["records"])

    benchmark_ids, benchmark_features = _load_benchmark_features(
        inputs.benchmark_feature_cache, inputs.benchmark_feature_manifest
    )
    benchmark_metadata = _benchmark_recorded_frame(
        inputs.production_benchmark_predictions, model="production"
    )
    if benchmark_ids != benchmark_metadata["Benchmark_Row_ID"].tolist():
        raise SourceAblationError("Benchmark feature/prediction ID order changed")
    full_recorded = _benchmark_recorded_frame(
        inputs.full_candidate_benchmark_predictions, model="full candidate"
    )

    production_benchmark_scores = _attach_ids(
        production_index.score(benchmark_features), benchmark_ids, "Benchmark_Row_ID"
    )
    full_benchmark_scores = _attach_ids(
        candidate_index.score(benchmark_features), benchmark_ids, "Benchmark_Row_ID"
    )
    _assert_score_reproduction(
        scored=production_benchmark_scores,
        recorded=_recorded_for_reproduction(benchmark_metadata),
        id_column="Benchmark_Row_ID",
        recorded_prefix="Recorded_",
        label="120-track production benchmark",
    )
    _assert_score_reproduction(
        scored=full_benchmark_scores,
        recorded=_recorded_for_reproduction(full_recorded),
        id_column="Benchmark_Row_ID",
        recorded_prefix="Recorded_",
        label="120-track full-candidate benchmark",
    )
    production_benchmark = _prediction_frame(
        benchmark_metadata,
        production_benchmark_scores,
        id_column="Benchmark_Row_ID",
        model_id="production",
        evidence_mode="frozen_120_track_reviewer_feature_cache",
        promotion_evidence_eligible=True,
    )

    mon_metadata, mon_features, mon_audit = _reproduce_mon_p3_features(
        inputs,
        yunet_model=yunet_model,
        sface_model=sface_model,
        status=status,
    )
    mon_ids = mon_metadata["Review_ID"].tolist()
    production_mon_scores = _attach_ids(
        production_index.score(mon_features), mon_ids, "Review_ID"
    )
    full_mon_scores = _attach_ids(
        candidate_index.score(mon_features), mon_ids, "Review_ID"
    )
    mon_recorded = _mon_recorded_reproduction_frame(inputs)
    _assert_score_reproduction(
        scored=production_mon_scores,
        recorded=mon_recorded,
        id_column="Review_ID",
        recorded_prefix="Production_",
        label="15-track MON_P3 production",
    )
    _assert_score_reproduction(
        scored=full_mon_scores,
        recorded=mon_recorded,
        id_column="Review_ID",
        recorded_prefix="Candidate_",
        label="15-track MON_P3 full candidate",
    )
    production_mon = _prediction_frame(
        mon_metadata,
        production_mon_scores,
        id_column="Review_ID",
        model_id="production",
        evidence_mode="exact_reviewed_track_video_reproduction",
        promotion_evidence_eligible=False,
    )

    configs = enumerate_source_subsets()
    benchmark_safe_predictions: list[pd.DataFrame] = []
    benchmark_descriptive_predictions: list[pd.DataFrame] = []
    mon_predictions: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    p1_indices = np.flatnonzero(benchmark_metadata["Session_ID"].eq(TUE_P1_SESSION))
    p2_indices = np.flatnonzero(benchmark_metadata["Session_ID"].eq(TUE_P2_SESSION))
    if len(p1_indices) != 100 or len(p2_indices) != 20:
        raise SourceAblationError("Frozen benchmark session row counts changed")
    prior_recovery_ids = set(
        mon_metadata.loc[
            mon_metadata["Phase_1_2J_Result"].eq("candidate_correct_recovery"),
            "Review_ID",
        ]
    )
    if len(prior_recovery_ids) != 4:
        raise SourceAblationError("Expected four prior MON_P3 candidate recoveries")

    for ordinal, config in enumerate(configs, start=1):
        if status:
            status(
                f"Source subset {ordinal}/{EXPECTED_SUBSET_COUNT}: {config['Config_ID']} "
                f"({config['Kept_Source_Count']} implicated sources kept)"
            )
        kept_ids = {
            value for value in _text(config["Kept_Source_IDs"]).split("|") if value
        }
        full_mask = _active_mask_for_config(
            source_records,
            kept_source_ids=kept_ids,
            evaluation_mode="full_composition",
        )
        p1_safe_mask = _active_mask_for_config(
            source_records,
            kept_source_ids=kept_ids,
            evaluation_mode="tue_p1_leakage_safe",
        )
        p2_safe_mask = _active_mask_for_config(
            source_records,
            kept_source_ids=kept_ids,
            evaluation_mode="tue_p2_leakage_safe",
        )

        mon_scores = _attach_ids(
            candidate_index.score(mon_features, active_record_mask=full_mask),
            mon_ids,
            "Review_ID",
        )
        mon_prediction = _prediction_frame(
            mon_metadata,
            mon_scores,
            id_column="Review_ID",
            model_id=config["Config_ID"],
            evidence_mode="phase_1_2l_mon_p3_reviewed_track_ablation",
            promotion_evidence_eligible=False,
        )
        mon_prediction.insert(1, "Mask", config["Mask"])
        mon_prediction.insert(2, "Kept_Source_Count", config["Kept_Source_Count"])
        mon_metrics = _prediction_metrics(mon_prediction)
        mon_comparison_metrics, mon_comparison = _comparison_metrics(
            production_mon, mon_prediction, id_column="Review_ID"
        )
        mon_prediction = mon_prediction.merge(
            mon_comparison[
                [
                    "Review_ID",
                    "Lost_Correct_Production_Accept",
                    "Recovered_Correct_Accept",
                    "New_Wrong_Accept",
                ]
            ],
            on="Review_ID",
            validate="one_to_one",
        )
        preserved_prior = int(
            mon_prediction[
                mon_prediction["Review_ID"].isin(prior_recovery_ids)
                & mon_prediction["Correct_Accepted"].map(bool)
            ]["Review_ID"].nunique()
        )
        mon_predictions.append(mon_prediction)

        descriptive_scores = _attach_ids(
            candidate_index.score(
                benchmark_features, active_record_mask=full_mask
            ),
            benchmark_ids,
            "Benchmark_Row_ID",
        )
        descriptive_prediction = _prediction_frame(
            benchmark_metadata,
            descriptive_scores,
            id_column="Benchmark_Row_ID",
            model_id=config["Config_ID"],
            evidence_mode="phase_1_2l_full_composition_descriptive_120_track",
            promotion_evidence_eligible=False,
        )
        descriptive_prediction.insert(1, "Mask", config["Mask"])
        descriptive_prediction.insert(
            2, "Kept_Source_Count", config["Kept_Source_Count"]
        )
        descriptive_metrics = _prediction_metrics(descriptive_prediction)
        descriptive_comparison, _ = _comparison_metrics(
            production_benchmark,
            descriptive_prediction,
            id_column="Benchmark_Row_ID",
        )
        benchmark_descriptive_predictions.append(descriptive_prediction)

        p1_scores = candidate_index.score(
            benchmark_features[p1_indices], active_record_mask=p1_safe_mask
        )
        p2_scores = candidate_index.score(
            benchmark_features[p2_indices], active_record_mask=p2_safe_mask
        )
        safe_score_frame = pd.concat(
            [
                _attach_ids(
                    p1_scores,
                    [benchmark_ids[index] for index in p1_indices],
                    "Benchmark_Row_ID",
                ),
                _attach_ids(
                    p2_scores,
                    [benchmark_ids[index] for index in p2_indices],
                    "Benchmark_Row_ID",
                ),
            ],
            ignore_index=True,
        ).sort_values("Benchmark_Row_ID", kind="stable").reset_index(drop=True)
        safe_prediction = _prediction_frame(
            benchmark_metadata,
            safe_score_frame,
            id_column="Benchmark_Row_ID",
            model_id=config["Config_ID"],
            evidence_mode="phase_1_2l_leakage_safe_cross_session_120_track",
            promotion_evidence_eligible=False,
        )
        safe_prediction.insert(1, "Mask", config["Mask"])
        safe_prediction.insert(2, "Kept_Source_Count", config["Kept_Source_Count"])
        safe_metrics = _prediction_metrics(safe_prediction)
        safe_comparison_metrics, safe_comparison = _comparison_metrics(
            production_benchmark, safe_prediction, id_column="Benchmark_Row_ID"
        )
        safe_prediction = safe_prediction.merge(
            safe_comparison[
                [
                    "Benchmark_Row_ID",
                    "Lost_Correct_Production_Accept",
                    "Recovered_Correct_Accept",
                    "New_Wrong_Accept",
                    "New_Outsider_Absorption",
                    "New_Mixed_Acceptance",
                    "New_Unverifiable_Acceptance",
                ]
            ],
            on="Benchmark_Row_ID",
            validate="one_to_one",
        )
        benchmark_safe_predictions.append(safe_prediction)

        gate_checks = {
            "benchmark_safe_unsafe_accepts": (
                safe_metrics["unsafe_confirmed_acceptances"] == 0
            ),
            "benchmark_safe_unverifiable_accepts": (
                safe_metrics["unverifiable_acceptance"] == 0
            ),
            "benchmark_safe_lost_production_accepts": (
                safe_comparison_metrics["lost_correct_production_accepts"] == 0
            ),
            "benchmark_safe_new_wrong_accepts": (
                safe_comparison_metrics["new_wrong_accepts"] == 0
            ),
            "benchmark_safe_new_outsider_absorptions": (
                safe_comparison_metrics["new_outsider_absorptions"] == 0
            ),
            "benchmark_safe_new_mixed_acceptances": (
                safe_comparison_metrics["new_mixed_acceptances"] == 0
            ),
            "benchmark_safe_new_unverifiable_acceptances": (
                safe_comparison_metrics["new_unverifiable_acceptances"] == 0
            ),
            "benchmark_descriptive_unsafe_accepts": (
                descriptive_metrics["unsafe_confirmed_acceptances"] == 0
            ),
            "benchmark_descriptive_unverifiable_accepts": (
                descriptive_metrics["unverifiable_acceptance"] == 0
            ),
            "benchmark_descriptive_lost_production_accepts": (
                descriptive_comparison["lost_correct_production_accepts"] == 0
            ),
            "benchmark_descriptive_new_wrong_accepts": (
                descriptive_comparison["new_wrong_accepts"] == 0
            ),
            "benchmark_descriptive_new_outsider_absorptions": (
                descriptive_comparison["new_outsider_absorptions"] == 0
            ),
            "benchmark_descriptive_new_mixed_acceptances": (
                descriptive_comparison["new_mixed_acceptances"] == 0
            ),
            "benchmark_descriptive_new_unverifiable_acceptances": (
                descriptive_comparison["new_unverifiable_acceptances"] == 0
            ),
            "mon_p3_unsafe_accepts": (
                mon_metrics["unsafe_confirmed_acceptances"] == 0
            ),
            "mon_p3_unverifiable_accepts": (
                mon_metrics["unverifiable_acceptance"] == 0
            ),
            "mon_p3_lost_production_accepts": (
                mon_comparison_metrics["lost_correct_production_accepts"] == 0
            ),
            "mon_p3_new_wrong_accepts": (
                mon_comparison_metrics["new_wrong_accepts"] == 0
            ),
            "mon_p3_new_outsider_absorptions": (
                mon_comparison_metrics["new_outsider_absorptions"] == 0
            ),
            "mon_p3_new_mixed_acceptances": (
                mon_comparison_metrics["new_mixed_acceptances"] == 0
            ),
            "mon_p3_new_unverifiable_acceptances": (
                mon_comparison_metrics["new_unverifiable_acceptances"] == 0
            ),
        }
        gate_failures = [
            label for label, passed in gate_checks.items() if not passed
        ]
        hard_gates = not gate_failures
        summary_rows.append(
            {
                **config,
                "Hard_Gates_Passed": hard_gates,
                "Hard_Gate_Failures": "|".join(gate_failures),
                "MON_P3_Correct_Accepted": mon_metrics["correct_accepted"],
                "MON_P3_Unsafe_Accepts": mon_metrics[
                    "unsafe_confirmed_acceptances"
                ],
                "MON_P3_Lost_Correct_Production_Accepts": mon_comparison_metrics[
                    "lost_correct_production_accepts"
                ],
                "MON_P3_Recovered_Correct_vs_Production": mon_comparison_metrics[
                    "recovered_correct_accepts"
                ],
                "MON_P3_Preserved_Prior_Candidate_Recoveries": preserved_prior,
                "Benchmark_Safe_Correct_Accepted": safe_metrics[
                    "correct_accepted"
                ],
                "Benchmark_Safe_Unsafe_Accepts": safe_metrics[
                    "unsafe_confirmed_acceptances"
                ],
                "Benchmark_Safe_Unverifiable_Accepts": safe_metrics[
                    "unverifiable_acceptance"
                ],
                "Benchmark_Safe_Lost_Correct_Production_Accepts": safe_comparison_metrics[
                    "lost_correct_production_accepts"
                ],
                "Benchmark_Safe_Recovered_Correct_vs_Production": safe_comparison_metrics[
                    "recovered_correct_accepts"
                ],
                "Benchmark_Descriptive_Correct_Accepted": descriptive_metrics[
                    "correct_accepted"
                ],
                "Benchmark_Descriptive_Unsafe_Accepts": descriptive_metrics[
                    "unsafe_confirmed_acceptances"
                ],
                "Benchmark_Descriptive_Lost_Correct_Production_Accepts": descriptive_comparison[
                    "lost_correct_production_accepts"
                ],
                "Benchmark_Descriptive_Recovered_Correct_vs_Production": descriptive_comparison[
                    "recovered_correct_accepts"
                ],
                "Benchmark_Descriptive_New_Wrong_Accepts": descriptive_comparison[
                    "new_wrong_accepts"
                ],
                "Benchmark_Descriptive_New_Outsider_Absorptions": descriptive_comparison[
                    "new_outsider_absorptions"
                ],
                "Benchmark_Descriptive_New_Mixed_Acceptances": descriptive_comparison[
                    "new_mixed_acceptances"
                ],
                "Benchmark_Descriptive_New_Unverifiable_Acceptances": descriptive_comparison[
                    "new_unverifiable_acceptances"
                ],
            }
        )

    ranked = rank_configurations(pd.DataFrame(summary_rows))
    hard_gate_candidates = ranked[ranked["Hard_Gates_Passed"].map(bool)].copy()
    frontier = _pareto_frontier(hard_gate_candidates)
    recommended = hard_gate_candidates.iloc[0].to_dict() if len(hard_gate_candidates) else None
    decision = (
        "source_subset_ablation_completed_build_candidate_identified"
        if recommended is not None
        else "source_subset_ablation_completed_no_hard_gate_candidate"
    )

    temporary = inputs.output_dir.with_name(inputs.output_dir.name + ".tmp")
    if temporary.exists():
        quarantine = temporary.with_name(
            temporary.name + ".failed-" + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        if quarantine.exists():
            raise SourceAblationError(
                f"Unable to preserve prior incomplete output because quarantine exists: {quarantine}"
            )
        os.replace(temporary, quarantine)
        if status:
            status(f"Preserved prior incomplete output: {quarantine}")
    temporary.mkdir(parents=True, exist_ok=False)
    implicated_inventory.to_csv(temporary / "implicated_source_inventory.csv", index=False)
    ranked.to_csv(temporary / "subset_experiment_summary.csv", index=False)
    frontier.to_csv(temporary / "safe_candidate_frontier.csv", index=False)
    pd.concat(mon_predictions, ignore_index=True).to_csv(
        temporary / "mon_p3_subset_predictions.csv", index=False
    )
    pd.concat(benchmark_safe_predictions, ignore_index=True).to_csv(
        temporary / "benchmark_leakage_safe_subset_predictions.csv", index=False
    )
    pd.concat(benchmark_descriptive_predictions, ignore_index=True).to_csv(
        temporary / "benchmark_descriptive_subset_predictions.csv", index=False
    )
    mon_audit.to_csv(temporary / "mon_p3_feature_reproduction_audit.csv", index=False)
    np.savez_compressed(
        temporary / "mon_p3_reviewed_track_feature_cache.npz",
        review_ids=np.asarray(mon_ids, dtype="U16"),
        tracklet_ids=np.asarray(mon_metadata["Tracklet_ID"].tolist(), dtype="U64"),
        actual_rolls=np.asarray(mon_metadata["Actual_Roll"].tolist(), dtype="U32"),
        embeddings=mon_features.astype(np.float32),
    )

    recommendation_payload = {
        "schema_version": SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "ablation_id": inputs.ablation_id,
        "decision": decision,
        "selection_is_build_design_only": True,
        "production_approved": False,
        "candidate_promoted": False,
        "mon_p3_is_no_longer_untouched_for_future_promotion": True,
        "recommended_configuration": recommended,
        "frontier_config_ids": frontier["Config_ID"].tolist() if len(frontier) else [],
        "hard_gate_candidate_count": int(len(hard_gate_candidates)),
        "next_candidate_must_be_new_immutable_family": True,
        "final_promotion_requires_new_untouched_session": True,
    }
    (temporary / "recommended_composition.json").write_text(
        json.dumps(recommendation_payload, indent=2), encoding="utf-8"
    )

    protected_hashes_after = _protected_input_hashes(
        inputs, yunet_model=yunet_model, sface_model=sface_model
    )
    changed_inputs = sorted(
        label
        for label, before in protected_hashes_before.items()
        if protected_hashes_after.get(label) != before
    )
    if changed_inputs:
        raise SourceAblationError(
            "Protected inputs changed during Phase 1.2L: " + ", ".join(changed_inputs)
        )
    production_hash_after = protected_hashes_after["production_embeddings"]
    production_summary_hash_after = protected_hashes_after["production_summary"]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "ablation_id": inputs.ablation_id,
        "fingerprint_sha256": inputs.fingerprint_sha256,
        "decision": decision,
        "family_id": EXPECTED_FAMILY_ID,
        "phase_1_2j_run_id": EXPECTED_RUN_ID,
        "phase_1_2j_evaluation_id": EXPECTED_EVALUATION_ID,
        "phase_1_2k_attribution_id": EXPECTED_ATTRIBUTION_ID,
        "implicated_sources": EXPECTED_IMPLICATED_SOURCES,
        "source_subsets_evaluated": EXPECTED_SUBSET_COUNT,
        "benchmark_rows": EXPECTED_BENCHMARK_ROWS,
        "mon_p3_reviewed_tracks": EXPECTED_MON_P3_ROWS,
        "hard_gate_candidate_count": int(len(hard_gate_candidates)),
        "frontier_candidate_count": int(len(frontier)),
        "recommended_config_id": (
            _text(recommended.get("Config_ID")) if recommended is not None else ""
        ),
        "recommended_kept_source_count": (
            int(recommended.get("Kept_Source_Count")) if recommended is not None else None
        ),
        "recommended_removed_source_count": (
            int(recommended.get("Removed_Source_Count"))
            if recommended is not None
            else None
        ),
        "production_embeddings_before_sha256": production_hash_before,
        "production_embeddings_after_sha256": production_hash_after,
        "production_summary_before_sha256": production_summary_hash_before,
        "production_summary_after_sha256": production_summary_hash_after,
        "production_embeddings_changed": False,
        "production_summary_changed": False,
        "protected_input_hashes_before": protected_hashes_before,
        "protected_input_hashes_after": protected_hashes_after,
        "protected_inputs_changed": False,
        "dataset_files_changed": False,
        "official_attendance_changed": False,
        "candidate_family_built": False,
        "candidate_promoted": False,
        "full_mon_p3_recognition_repeated": False,
        "manual_review_repeated": False,
        "mon_p3_used_for_candidate_design": True,
        "mon_p3_eligible_as_future_untouched_promotion_session": False,
        "exact_next_step": (
            "Review the recommended source composition, then build a new immutable family "
            "in a separate phase. Final promotion requires a newly reserved untouched session."
            if recommended is not None
            else "Do not build a new family from these five-source subsets; inspect the ablation frontier and design a broader source-balancing policy."
        ),
    }
    (temporary / "source_ablation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    report = [
        "# Phase 1.2L Source-Subset Ablation",
        "",
        f"- Decision: `{decision}`",
        f"- Implicated sources tested: {EXPECTED_IMPLICATED_SOURCES}",
        f"- Exhaustive subsets evaluated: {EXPECTED_SUBSET_COUNT}",
        f"- Leakage-safe frozen benchmark rows: {EXPECTED_BENCHMARK_ROWS}",
        f"- Exact reviewed MON_P3 tracks: {EXPECTED_MON_P3_ROWS}",
        f"- Hard-gate candidates: {len(hard_gate_candidates)}",
        "- Production embeddings changed: no",
        "- Official attendance changed: no",
        "- New embedding family built: no",
        "- Candidate promoted: no",
        "- Full MON_P3 recognition repeated: no",
        "- Manual review repeated: no",
        "",
        "## Selection boundary",
        "",
        "MON_P3 was reused for bounded candidate-source design. It must not be treated as an untouched promotion session for the next family.",
        "",
    ]
    if recommended is not None:
        report += [
            "## Recommended build-design composition",
            "",
            f"- Config: `{recommended['Config_ID']}`",
            f"- Kept implicated sources: {recommended['Kept_Source_Count']}",
            f"- Removed implicated sources: {recommended['Removed_Source_Count']}",
            f"- Kept source labels: {recommended['Kept_Source_Labels'] or '(none)'}",
            f"- Removed source IDs: {recommended['Removed_Source_IDs'] or '(none)'}",
            f"- MON_P3 prior recoveries preserved: {recommended['MON_P3_Preserved_Prior_Candidate_Recoveries']}/4",
            f"- MON_P3 production accepts lost: {recommended['MON_P3_Lost_Correct_Production_Accepts']}",
            f"- Leakage-safe benchmark production accepts lost: {recommended['Benchmark_Safe_Lost_Correct_Production_Accepts']}",
            "",
            "This is a build-design recommendation only. A new immutable family and a new untouched validation session are still required.",
        ]
    else:
        report += [
            "## Result",
            "",
            "No tested source subset passed every hard gate. Do not build or promote a new family from this ablation alone.",
        ]
    (temporary / "source_ablation_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    write_output_manifest(
        temporary,
        {
            "policy_version": POLICY_VERSION,
            "ablation_id": inputs.ablation_id,
            "decision": decision,
            "source_subsets_evaluated": EXPECTED_SUBSET_COUNT,
            "production_changes": False,
            "official_attendance_modified": False,
            "candidate_family_built": False,
            "candidate_promoted": False,
            "full_mon_p3_recognition_repeated": False,
            "manual_review_repeated": False,
        },
    )
    try:
        verify_output_manifest(temporary, temporary / "output_manifest.json")
    except ShadowValidationError as exc:
        raise SourceAblationError(str(exc)) from exc
    inputs.output_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, inputs.output_dir)
    verified = verify_source_ablation_output(inputs.output_dir)
    return verified, inputs.output_dir, False


def verify_source_ablation_output(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    try:
        verified = verify_output_manifest(output_dir, output_dir / "output_manifest.json")
    except ShadowValidationError as exc:
        raise SourceAblationError(str(exc)) from exc
    summary = _load_json(output_dir / "source_ablation_summary.json", "source ablation summary")
    if _text(summary.get("policy_version")) != POLICY_VERSION:
        raise SourceAblationError("Source-ablation policy version changed")
    if int(summary.get("source_subsets_evaluated") or 0) != EXPECTED_SUBSET_COUNT:
        raise SourceAblationError("Source-ablation output does not contain 32 subsets")
    if int(summary.get("mon_p3_reviewed_tracks") or 0) != EXPECTED_MON_P3_ROWS:
        raise SourceAblationError("Source-ablation output does not contain 15 MON_P3 tracks")
    if int(summary.get("benchmark_rows") or 0) != EXPECTED_BENCHMARK_ROWS:
        raise SourceAblationError("Source-ablation output does not contain 120 benchmark rows")
    if summary.get("production_embeddings_changed") is not False:
        raise SourceAblationError("Source-ablation output claims production embeddings changed")
    if summary.get("candidate_promoted") is not False:
        raise SourceAblationError("Source-ablation output claims a candidate was promoted")
    if summary.get("protected_inputs_changed") is not False:
        raise SourceAblationError("Source-ablation output claims protected inputs changed")
    before_hashes = summary.get("protected_input_hashes_before")
    after_hashes = summary.get("protected_input_hashes_after")
    if not isinstance(before_hashes, dict) or before_hashes != after_hashes:
        raise SourceAblationError(
            "Source-ablation protected-input hash register is incomplete or changed"
        )
    experiments = pd.read_csv(
        output_dir / "subset_experiment_summary.csv", dtype=str, keep_default_na=False
    )
    if len(experiments) != EXPECTED_SUBSET_COUNT or experiments["Config_ID"].duplicated().any():
        raise SourceAblationError("Source-ablation experiment summary is incomplete")
    with np.load(output_dir / "mon_p3_reviewed_track_feature_cache.npz") as feature_cache:
        if feature_cache["embeddings"].shape != (EXPECTED_MON_P3_ROWS, 128):
            raise SourceAblationError(
                "MON_P3 reviewed-track feature cache has the wrong shape"
            )
    return {
        "status": "PASS",
        "output_dir": str(output_dir),
        "verified_files": int(verified.get("verified_files", 0)),
        "decision": _text(summary.get("decision")),
        "source_subsets_evaluated": EXPECTED_SUBSET_COUNT,
        "hard_gate_candidate_count": int(summary.get("hard_gate_candidate_count") or 0),
        "recommended_config_id": _text(summary.get("recommended_config_id")),
        "production_embeddings_changed": False,
        "official_attendance_changed": False,
        "candidate_family_built": False,
        "candidate_promoted": False,
        "full_mon_p3_recognition_repeated": False,
        "manual_review_repeated": False,
    }
