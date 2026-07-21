from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .diagnostics import json_safe
from .embedding_forensics import _canonical, load_embedding_payload
from .embedding_version_family import (
    CctvSourceSelectionContract,
    EmbeddingFamilyError,
    FamilyBuildResult,
    TUE_P1_SESSION,
    TUE_P2_SESSION,
    build_and_evaluate_embedding_family,
    preflight_embedding_family_inputs,
    verify_embedding_family,
)
from .mon_p3_source_ablation import (
    SourceAblationError,
    VectorizedTop3Index,
    _load_benchmark_features,
    verify_source_ablation_output,
    verify_source_ablation_selection_audit,
)
from .shadow_validation import ShadowValidationError, verify_output_manifest, write_output_manifest
from .tracklet_review import _sha256_file


SCHEMA_VERSION = 1
POLICY_VERSION = "phase-1.2m-corrected-source-family-v1"
PARENT_FAMILY_ID = "embfam-7bc431a3ad762398d4e9"
SOURCE_ABLATION_ID = "source-ablation-da12b41f85bc813a6de1"
SELECTION_AUDIT_ID = "selection-audit-bacb195efc6751ca2c22"
CORRECTED_CONFIG_ID = "ABL-16-ee3a4ff6"
SELECTION_POLICY_VERSION = "phase-1.2l1-promotion-evidence-parsimony-v1"
EXPECTED_SELECTION_OUTPUT_MANIFEST_SHA256 = (
    "7baa68fb901e8af8e3bd420282b1d84d96e79382bb43a24a65ea3a91058ac410"
)
EXPECTED_SELECTION_DECISION_SHA256 = (
    "97de993665df3c39dc59f38c7fca30009be1b8f4ae86c06f5ebf9b6b98ceb7d8"
)
EXPECTED_SOURCE_ABLATION_OUTPUT_MANIFEST_SHA256 = (
    "2a34c5c107e4966fdfafed8a2a2e5d5e2bae1e5ab2a914116c97eacad82c54a3"
)
KEPT_SOURCE_IDS = (
    "RECOVERY-cctv-recovery-69e11ea931f79330626d-CCC-debcd075bb2e58209a",
)
REMOVED_SOURCE_IDS = (
    "CCTV-CCC-16b0462d0f283edb9f",
    "CCTV-CCC-492ad8ac8e05bd2cac",
    "CCTV-CCC-9852dcae08d8e171db",
    "RECOVERY-cctv-recovery-69e11ea931f79330626d-CCC-40a4e971227bc88cd6",
)
EXPECTED_VARIANT_RECORD_COUNTS = {"A": 347, "B": 351, "C": 350, "D": 354}
SCORE_TOLERANCE = 1e-6


class Phase12MError(ValueError):
    pass


@dataclass(frozen=True)
class Phase12MInputs:
    repo_root: Path
    forensic_run_dir: Path
    enrollment_review_package: Path
    enrollment_approvals_path: Path
    cctv_review_package: Path
    cctv_approvals_path: Path
    production_embeddings_path: Path
    production_summary_path: Path
    student_map_path: Path
    dataset_root: Path
    augmented_dataset_root: Path
    versions_root: Path
    recovery_dir: Path
    source_ablation_dir: Path
    selection_audit_dir: Path
    parent_family_dir: Path


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise Phase12MError(f"{label} not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase12MError(f"Invalid {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise Phase12MError(f"{label} must be a JSON object: {path}")
    return payload


def _pipe_set(value: Any) -> set[str]:
    return {token for token in _text(value).split("|") if token}


def _expected_contract() -> CctvSourceSelectionContract:
    return CctvSourceSelectionContract(
        selection_audit_id=SELECTION_AUDIT_ID,
        selection_policy_version=SELECTION_POLICY_VERSION,
        source_ablation_id=SOURCE_ABLATION_ID,
        corrected_config_id=CORRECTED_CONFIG_ID,
        kept_source_ids=KEPT_SOURCE_IDS,
        removed_source_ids=REMOVED_SOURCE_IDS,
        selection_audit_output_manifest_sha256=EXPECTED_SELECTION_OUTPUT_MANIFEST_SHA256,
        selection_decision_sha256=EXPECTED_SELECTION_DECISION_SHA256,
        source_ablation_output_manifest_sha256=(
            EXPECTED_SOURCE_ABLATION_OUTPUT_MANIFEST_SHA256
        ),
        final_promotion_requires_new_untouched_session=True,
    )


def _variant_dir(family_dir: Path, key: str) -> Path:
    names = {
        "A": "cleaned_enrollment_only",
        "B": "cross_session_for_tue_p1",
        "C": "cross_session_for_tue_p2",
        "D": "full_candidate",
    }
    return Path(family_dir) / "variants" / names[key]


def _variant_records(family_dir: Path, key: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    directory = _variant_dir(family_dir, key)
    source_records = pd.read_csv(
        directory / "source_records.csv", dtype=str, keep_default_na=False
    )
    required = {
        "Record_Order",
        "Source_ID",
        "Canonical_Roll",
        "Source_Kind",
        "Source_Session",
        "Image_Path",
    }
    missing = sorted(required.difference(source_records.columns))
    if missing:
        raise Phase12MError(
            f"Variant {key} source records are missing: {', '.join(missing)}"
        )
    source_records["Record_Order"] = pd.to_numeric(
        source_records["Record_Order"], errors="raise"
    ).astype(int)
    source_records = source_records.sort_values("Record_Order", kind="stable").reset_index(
        drop=True
    )
    if source_records["Source_ID"].duplicated().any():
        raise Phase12MError(f"Variant {key} has duplicate source IDs")
    payload = load_embedding_payload(directory / "student_embeddings.pkl")
    records = list(payload["records"])
    if len(records) != len(source_records):
        raise Phase12MError(
            f"Variant {key} source records do not align with its embedding payload"
        )
    return source_records, records


def _source_vector_map(family_dir: Path, key: str) -> dict[str, dict[str, Any]]:
    source_records, records = _variant_records(family_dir, key)
    result: dict[str, dict[str, Any]] = {}
    for source, record in zip(source_records.to_dict("records"), records):
        source_id = _text(source["Source_ID"])
        vector = np.asarray(record.get("embedding"), dtype=np.float32).reshape(-1)
        if vector.size != 128 or not np.all(np.isfinite(vector)):
            raise Phase12MError(f"Variant {key} source {source_id} has an invalid vector")
        result[source_id] = {
            "source": source,
            "record": record,
            "vector": vector,
        }
    return result


def verify_phase_1_2m_evidence(inputs: Phase12MInputs) -> tuple[dict[str, Any], CctvSourceSelectionContract]:
    try:
        source_ablation = verify_source_ablation_output(inputs.source_ablation_dir)
        selection = verify_source_ablation_selection_audit(inputs.selection_audit_dir)
    except (SourceAblationError, OSError, ValueError) as exc:
        raise Phase12MError(str(exc)) from exc

    if _sha256_file(inputs.source_ablation_dir / "output_manifest.json") != (
        EXPECTED_SOURCE_ABLATION_OUTPUT_MANIFEST_SHA256
    ):
        raise Phase12MError("Phase 1.2L output manifest hash changed")
    if _sha256_file(inputs.selection_audit_dir / "output_manifest.json") != (
        EXPECTED_SELECTION_OUTPUT_MANIFEST_SHA256
    ):
        raise Phase12MError("Phase 1.2L.1 output manifest hash changed")
    if _sha256_file(inputs.selection_audit_dir / "selection_decision.json") != (
        EXPECTED_SELECTION_DECISION_SHA256
    ):
        raise Phase12MError("Phase 1.2L.1 selection decision hash changed")

    decision = _load_json(
        inputs.selection_audit_dir / "selection_decision.json", "selection decision"
    )
    if (
        _text(decision.get("audit_id")) != SELECTION_AUDIT_ID
        or _text(decision.get("selection_policy_version"))
        != SELECTION_POLICY_VERSION
        or _text(decision.get("source_ablation_id")) != SOURCE_ABLATION_ID
        or _text(decision.get("corrected_recommended_config_id"))
        != CORRECTED_CONFIG_ID
        or _pipe_set(decision.get("corrected_kept_source_ids"))
        != set(KEPT_SOURCE_IDS)
        or _pipe_set(decision.get("corrected_removed_source_ids"))
        != set(REMOVED_SOURCE_IDS)
        or decision.get("corrected_config_is_on_safe_frontier") is not True
        or decision.get("promotion_eligible_metrics_equal_to_legacy") is not True
        or int(decision.get("mon_p3_unsafe_accepts", -1)) != 0
        or int(decision.get("mon_p3_lost_correct_production_accepts", -1)) != 0
        or int(decision.get("mon_p3_recovered_correct_vs_production", -1)) != 4
        or int(decision.get("benchmark_safe_unsafe_accepts", -1)) != 0
        or int(decision.get("benchmark_safe_lost_correct_production_accepts", -1))
        != 0
        or int(decision.get("benchmark_safe_recovered_correct_vs_production", -1))
        != 1
        or decision.get("selection_is_build_design_only") is not True
        or decision.get("final_promotion_requires_new_untouched_session") is not True
    ):
        raise Phase12MError("Phase 1.2L.1 corrected-selection contract changed")

    subset = pd.read_csv(
        inputs.source_ablation_dir / "subset_experiment_summary.csv",
        dtype=str,
        keep_default_na=False,
    )
    row = subset[subset["Config_ID"].eq(CORRECTED_CONFIG_ID)]
    if len(row) != 1:
        raise Phase12MError("Corrected ABL-16 configuration is missing or duplicated")
    selected = row.iloc[0]
    if (
        _pipe_set(selected["Kept_Source_IDs"]) != set(KEPT_SOURCE_IDS)
        or _pipe_set(selected["Removed_Source_IDs"]) != set(REMOVED_SOURCE_IDS)
        or _text(selected["Hard_Gates_Passed"]).lower() not in {"true", "1", "yes"}
        or int(selected["MON_P3_Unsafe_Accepts"]) != 0
        or int(selected["MON_P3_Lost_Correct_Production_Accepts"]) != 0
        or int(selected["MON_P3_Recovered_Correct_vs_Production"]) != 4
        or int(selected["Benchmark_Safe_Unsafe_Accepts"]) != 0
        or int(selected["Benchmark_Safe_Lost_Correct_Production_Accepts"]) != 0
        or int(selected["Benchmark_Safe_Recovered_Correct_vs_Production"]) != 1
    ):
        raise Phase12MError("Phase 1.2L ABL-16 metrics or sources changed")

    try:
        parent_verified = verify_embedding_family(
            inputs.parent_family_dir,
            production_embeddings_path=inputs.production_embeddings_path,
            production_summary_path=inputs.production_summary_path,
        )
    except EmbeddingFamilyError as exc:
        raise Phase12MError(str(exc)) from exc
    if (
        _text(parent_verified.get("family_id")) != PARENT_FAMILY_ID
        or _text(parent_verified.get("decision"))
        != "built_unapproved_pending_mon_p3"
    ):
        raise Phase12MError("Parent recovered family identity or decision changed")

    parent_full, _ = _variant_records(inputs.parent_family_dir, "D")
    parent_ids = set(parent_full["Source_ID"].map(_text))
    implicated = set(KEPT_SOURCE_IDS) | set(REMOVED_SOURCE_IDS)
    if not implicated.issubset(parent_ids):
        raise Phase12MError("Parent family no longer contains all five implicated sources")
    inventory = pd.read_csv(
        inputs.source_ablation_dir / "implicated_source_inventory.csv",
        dtype=str,
        keep_default_na=False,
    )
    if set(inventory["Source_ID"].map(_text)) != implicated:
        raise Phase12MError("Phase 1.2L implicated-source inventory changed")

    contract = _expected_contract()
    normalized = contract.normalized()
    if set(normalized["kept_source_ids"]) != set(KEPT_SOURCE_IDS):
        raise Phase12MError("Internal kept-source contract changed")

    return (
        {
            "schema_version": SCHEMA_VERSION,
            "policy_version": POLICY_VERSION,
            "status": "passed",
            "read_only": True,
            "source_ablation": source_ablation,
            "selection_audit": selection,
            "parent_family": parent_verified,
            "selection_contract": normalized,
            "expected_variant_record_counts": EXPECTED_VARIANT_RECORD_COUNTS,
            "new_untouched_session_required": True,
            "mon_p3_reuse_for_promotion_allowed": False,
            "candidate_family_built": False,
            "candidate_promoted": False,
            "production_embeddings_changed": False,
            "official_attendance_changed": False,
        },
        contract,
    )


def preflight_phase_1_2m(
    *,
    inputs: Phase12MInputs,
) -> dict[str, Any]:
    evidence, _contract = verify_phase_1_2m_evidence(inputs)
    try:
        base = preflight_embedding_family_inputs(
            repo_root=inputs.repo_root,
            forensic_run_dir=inputs.forensic_run_dir,
            enrollment_review_package=inputs.enrollment_review_package,
            enrollment_approvals_path=inputs.enrollment_approvals_path,
            cctv_review_package=inputs.cctv_review_package,
            cctv_approvals_path=inputs.cctv_approvals_path,
            production_embeddings_path=inputs.production_embeddings_path,
            production_summary_path=inputs.production_summary_path,
            student_map_path=inputs.student_map_path,
            dataset_root=inputs.dataset_root,
            augmented_dataset_root=inputs.augmented_dataset_root,
            versions_root=inputs.versions_root,
            recovery_dir=inputs.recovery_dir,
        )
    except EmbeddingFamilyError as exc:
        raise Phase12MError(str(exc)) from exc
    return {
        **evidence,
        "base_family_preflight": base,
        "preflight_fingerprint_sha256": hashlib.sha256(
            json.dumps(
                json_safe({"evidence": evidence, "base": base}),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest(),
    }


def _compare_source_composition(
    *, parent_family_dir: Path, family_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    vector_rows: list[dict[str, Any]] = []
    removed = set(REMOVED_SOURCE_IDS)
    for key in ("A", "B", "C", "D"):
        parent = _source_vector_map(parent_family_dir, key)
        child = _source_vector_map(family_dir, key)
        expected_ids = set(parent).difference(removed)
        if set(child) != expected_ids:
            raise Phase12MError(
                f"Variant {key} source composition differs from parent-minus-selection"
            )
        if len(child) != EXPECTED_VARIANT_RECORD_COUNTS[key]:
            raise Phase12MError(
                f"Variant {key} expected {EXPECTED_VARIANT_RECORD_COUNTS[key]} records, "
                f"found {len(child)}"
            )
        for source_id in sorted(set(parent) | set(child)):
            in_parent = source_id in parent
            in_child = source_id in child
            expected_active = source_id not in removed
            rows.append(
                {
                    "Variant": key,
                    "Source_ID": source_id,
                    "Present_In_Parent": in_parent,
                    "Expected_In_New_Family": expected_active,
                    "Present_In_New_Family": in_child,
                    "Selection_Result": (
                        "removed_as_selected"
                        if in_parent and not in_child
                        else "preserved"
                    ),
                }
            )
            if in_child:
                parent_row = parent[source_id]
                child_row = child[source_id]
                if _canonical(parent_row["record"].get("roll_no")) != _canonical(
                    child_row["record"].get("roll_no")
                ):
                    raise Phase12MError(
                        f"Variant {key} roll ownership changed for {source_id}"
                    )
                max_abs = float(
                    np.max(np.abs(parent_row["vector"] - child_row["vector"]))
                )
                if max_abs > SCORE_TOLERANCE:
                    raise Phase12MError(
                        f"Variant {key} embedding drifted for {source_id}: {max_abs}"
                    )
                vector_rows.append(
                    {
                        "Variant": key,
                        "Source_ID": source_id,
                        "Canonical_Roll": _canonical(
                            child_row["record"].get("roll_no")
                        ),
                        "Max_Absolute_Vector_Difference": max_abs,
                        "Vector_Reproduced": True,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(vector_rows)


def _score_expected(
    *,
    database_path: Path,
    features: np.ndarray,
    ids: list[str],
    expected: pd.DataFrame,
    id_column: str,
    label: str,
) -> pd.DataFrame:
    payload = load_embedding_payload(database_path)
    index = VectorizedTop3Index.from_records(list(payload["records"]))
    actual = index.score(features)
    actual.insert(0, id_column, ids)
    required = {
        id_column,
        "Accepted",
        "Best_Roll",
        "Best_Score",
        "Second_Roll",
        "Second_Score",
        "Margin",
        "Reason",
    }
    missing = sorted(required.difference(expected.columns))
    if missing:
        raise Phase12MError(f"{label} expected predictions missing: {', '.join(missing)}")
    expected = expected[list(required)].copy()
    if actual[id_column].duplicated().any() or expected[id_column].duplicated().any():
        raise Phase12MError(f"{label} IDs are not unique")
    joined = actual.merge(
        expected,
        on=id_column,
        suffixes=("_Actual", "_Expected"),
        validate="one_to_one",
    )
    if len(joined) != len(actual):
        raise Phase12MError(f"{label} prediction join is incomplete")
    failures: list[str] = []
    for row in joined.to_dict("records"):
        row_id = _text(row[id_column])
        actual_accepted = bool(row["Accepted_Actual"])
        expected_accepted = _text(row["Accepted_Expected"]).lower() in {
            "true",
            "1",
            "yes",
        }
        if (
            actual_accepted != expected_accepted
            or _canonical(row["Best_Roll_Actual"])
            != _canonical(row["Best_Roll_Expected"])
            or _canonical(row["Second_Roll_Actual"])
            != _canonical(row["Second_Roll_Expected"])
            or _text(row["Reason_Actual"]) != _text(row["Reason_Expected"])
        ):
            failures.append(row_id)
            continue
        for field in ("Best_Score", "Second_Score", "Margin"):
            actual_value = float(row[f"{field}_Actual"])
            expected_value = float(row[f"{field}_Expected"])
            if abs(actual_value - expected_value) > SCORE_TOLERANCE:
                failures.append(row_id)
                break
    if failures:
        raise Phase12MError(
            f"{label} does not reproduce ABL-16 exactly: {', '.join(failures[:10])}"
        )
    joined.insert(1, "Reproduction_Passed", True)
    return joined


def validate_built_family_against_abl16(
    *,
    inputs: Phase12MInputs,
    family_dir: Path,
    output_root: Path | None = None,
) -> tuple[dict[str, Any], Path, bool]:
    family_dir = Path(family_dir).resolve()
    try:
        verified = verify_embedding_family(
            family_dir,
            production_embeddings_path=inputs.production_embeddings_path,
            production_summary_path=inputs.production_summary_path,
        )
    except EmbeddingFamilyError as exc:
        raise Phase12MError(str(exc)) from exc
    if _text(verified.get("decision")) != "built_unapproved_pending_new_untouched_session":
        raise Phase12MError("Phase 1.2M family has the wrong lifecycle decision")

    composition, vectors = _compare_source_composition(
        parent_family_dir=inputs.parent_family_dir, family_dir=family_dir
    )
    source_dir = inputs.source_ablation_dir
    config_id = CORRECTED_CONFIG_ID

    with np.load(
        source_dir / "mon_p3_reviewed_track_feature_cache.npz", allow_pickle=False
    ) as cache:
        mon_ids = [_text(value) for value in cache["review_ids"]]
        mon_features = np.asarray(cache["embeddings"], dtype=np.float32)
    mon_expected_all = pd.read_csv(
        source_dir / "mon_p3_subset_predictions.csv", dtype=str, keep_default_na=False
    )
    mon_expected = mon_expected_all[mon_expected_all["Model_ID"].eq(config_id)].copy()
    mon_reproduction = _score_expected(
        database_path=_variant_dir(family_dir, "D") / "student_embeddings.pkl",
        features=mon_features,
        ids=mon_ids,
        expected=mon_expected,
        id_column="Review_ID",
        label="15-track MON_P3 design-evidence reproduction",
    )

    benchmark_ids, benchmark_features = _load_benchmark_features(
        inputs.parent_family_dir / "evaluation" / "benchmark_feature_cache.pkl",
        inputs.parent_family_dir / "evaluation" / "benchmark_feature_manifest.json",
    )
    safe_all = pd.read_csv(
        source_dir / "benchmark_leakage_safe_subset_predictions.csv",
        dtype=str,
        keep_default_na=False,
    )
    safe_expected = safe_all[safe_all["Model_ID"].eq(config_id)].copy()
    safe_by_id = safe_expected.set_index("Benchmark_Row_ID", drop=False)
    p1_ids = [
        row_id
        for row_id in benchmark_ids
        if _text(safe_by_id.loc[row_id, "Session_ID"]) == TUE_P1_SESSION
    ]
    p2_ids = [
        row_id
        for row_id in benchmark_ids
        if _text(safe_by_id.loc[row_id, "Session_ID"]) == TUE_P2_SESSION
    ]
    index_by_id = {row_id: index for index, row_id in enumerate(benchmark_ids)}
    p1_features = benchmark_features[[index_by_id[row_id] for row_id in p1_ids]]
    p2_features = benchmark_features[[index_by_id[row_id] for row_id in p2_ids]]
    safe_p1 = _score_expected(
        database_path=_variant_dir(family_dir, "B") / "student_embeddings.pkl",
        features=p1_features,
        ids=p1_ids,
        expected=safe_expected[safe_expected["Session_ID"].eq(TUE_P1_SESSION)],
        id_column="Benchmark_Row_ID",
        label="TUE_P1 leakage-safe ABL-16 reproduction",
    )
    safe_p2 = _score_expected(
        database_path=_variant_dir(family_dir, "C") / "student_embeddings.pkl",
        features=p2_features,
        ids=p2_ids,
        expected=safe_expected[safe_expected["Session_ID"].eq(TUE_P2_SESSION)],
        id_column="Benchmark_Row_ID",
        label="TUE_P2 leakage-safe ABL-16 reproduction",
    )
    safe_reproduction = pd.concat([safe_p1, safe_p2], ignore_index=True).sort_values(
        "Benchmark_Row_ID", kind="stable"
    )

    descriptive_all = pd.read_csv(
        source_dir / "benchmark_descriptive_subset_predictions.csv",
        dtype=str,
        keep_default_na=False,
    )
    descriptive_expected = descriptive_all[
        descriptive_all["Model_ID"].eq(config_id)
    ].copy()
    descriptive_reproduction = _score_expected(
        database_path=_variant_dir(family_dir, "D") / "student_embeddings.pkl",
        features=benchmark_features,
        ids=benchmark_ids,
        expected=descriptive_expected,
        id_column="Benchmark_Row_ID",
        label="120-track descriptive ABL-16 reproduction",
    )

    family_manifest_hash = _sha256_file(family_dir / "output_manifest.json")
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "policy_version": POLICY_VERSION,
                "family_id": family_dir.name,
                "family_output_manifest_sha256": family_manifest_hash,
                "selection_output_manifest_sha256": (
                    EXPECTED_SELECTION_OUTPUT_MANIFEST_SHA256
                ),
                "source_ablation_output_manifest_sha256": (
                    EXPECTED_SOURCE_ABLATION_OUTPUT_MANIFEST_SHA256
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    validation_id = f"family-validation-{fingerprint[:20]}"
    output_root = Path(
        output_root
        or inputs.repo_root
        / "attendance_output"
        / "embedding_forensics"
        / "phase_1_2m"
    ).resolve()
    final_dir = output_root / validation_id
    if final_dir.exists():
        try:
            manifest = verify_output_manifest(final_dir, final_dir / "output_manifest.json")
        except ShadowValidationError as exc:
            raise Phase12MError(str(exc)) from exc
        summary = _load_json(final_dir / "validation_summary.json", "validation summary")
        if _text(summary.get("fingerprint_sha256")) != fingerprint:
            raise Phase12MError("Existing Phase 1.2M validation has different inputs")
        return summary, final_dir, True

    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / f".{validation_id}.building"
    if temporary.exists():
        raise Phase12MError(f"Stale Phase 1.2M validation directory exists: {temporary}")
    temporary.mkdir()
    composition.to_csv(temporary / "variant_source_composition.csv", index=False)
    vectors.to_csv(temporary / "source_vector_reproduction.csv", index=False)
    mon_reproduction.to_csv(temporary / "mon_p3_abl16_reproduction.csv", index=False)
    safe_reproduction.to_csv(
        temporary / "benchmark_leakage_safe_abl16_reproduction.csv", index=False
    )
    descriptive_reproduction.to_csv(
        temporary / "benchmark_descriptive_abl16_reproduction.csv", index=False
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "validation_id": validation_id,
        "fingerprint_sha256": fingerprint,
        "decision": "family_built_unapproved_exact_abl16_reproduction_passed",
        "family_id": family_dir.name,
        "family_output_manifest_sha256": family_manifest_hash,
        "parent_family_id": PARENT_FAMILY_ID,
        "source_ablation_id": SOURCE_ABLATION_ID,
        "selection_audit_id": SELECTION_AUDIT_ID,
        "corrected_config_id": CORRECTED_CONFIG_ID,
        "kept_source_ids": list(KEPT_SOURCE_IDS),
        "removed_source_ids": list(REMOVED_SOURCE_IDS),
        "variant_record_counts": EXPECTED_VARIANT_RECORD_COUNTS,
        "source_vectors_reproduced": len(vectors),
        "mon_p3_design_tracks_reproduced": len(mon_reproduction),
        "benchmark_leakage_safe_rows_reproduced": len(safe_reproduction),
        "benchmark_descriptive_rows_reproduced": len(descriptive_reproduction),
        "new_untouched_session_required": True,
        "mon_p3_reuse_for_promotion_allowed": False,
        "production_embeddings_changed": False,
        "official_attendance_changed": False,
        "full_mon_p3_recognition_repeated": False,
        "manual_review_repeated": False,
        "candidate_promoted": False,
        "exact_next_step": (
            "Reserve a different untouched CVO session and run a new shadow validation."
        ),
    }
    (temporary / "validation_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=True), encoding="utf-8"
    )
    (temporary / "validation_report.md").write_text(
        "\n".join(
            [
                "# Phase 1.2M Corrected Candidate Family Validation",
                "",
                f"- Family: `{family_dir.name}`",
                f"- Corrected source composition: `{CORRECTED_CONFIG_ID}`",
                f"- Variant records: `{EXPECTED_VARIANT_RECORD_COUNTS}`",
                "- MON_P3 design tracks reproduced: 15/15",
                "- Leakage-safe benchmark rows reproduced: 120/120",
                "- Descriptive benchmark rows reproduced: 120/120",
                "- Production changed: no",
                "- Candidate promoted: no",
                "- Final promotion requires a different untouched session: yes",
                "",
                "MON_P3 is retained as build-design evidence and must not be reused as the final promotion session.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_output_manifest(
        temporary,
        {
            "phase": "1.2M",
            "validation_id": validation_id,
            "decision": summary["decision"],
            "family_id": family_dir.name,
            "candidate_promoted": False,
            "production_embeddings_changed": False,
            "official_attendance_changed": False,
            "full_mon_p3_recognition_repeated": False,
            "manual_review_repeated": False,
        },
    )
    try:
        verify_output_manifest(temporary, temporary / "output_manifest.json")
    except ShadowValidationError as exc:
        raise Phase12MError(str(exc)) from exc
    temporary.replace(final_dir)
    verify_output_manifest(final_dir, final_dir / "output_manifest.json")
    return summary, final_dir, False


def build_phase_1_2m(
    *,
    inputs: Phase12MInputs,
    family_id: str = "",
    launcher_source: Path | None = None,
    progress: Callable[[str], None] | None = print,
) -> tuple[FamilyBuildResult, dict[str, Any], Path, bool]:
    _preflight = preflight_phase_1_2m(inputs=inputs)
    _evidence, contract = verify_phase_1_2m_evidence(inputs)
    try:
        result = build_and_evaluate_embedding_family(
            repo_root=inputs.repo_root,
            forensic_run_dir=inputs.forensic_run_dir,
            enrollment_review_package=inputs.enrollment_review_package,
            enrollment_approvals_path=inputs.enrollment_approvals_path,
            cctv_review_package=inputs.cctv_review_package,
            cctv_approvals_path=inputs.cctv_approvals_path,
            production_embeddings_path=inputs.production_embeddings_path,
            production_summary_path=inputs.production_summary_path,
            student_map_path=inputs.student_map_path,
            dataset_root=inputs.dataset_root,
            augmented_dataset_root=inputs.augmented_dataset_root,
            versions_root=inputs.versions_root,
            family_id=family_id,
            launcher_source=launcher_source,
            recovery_dir=inputs.recovery_dir,
            cctv_source_selection=contract,
            progress=progress,
        )
    except EmbeddingFamilyError as exc:
        raise Phase12MError(str(exc)) from exc
    summary, validation_dir, reused = validate_built_family_against_abl16(
        inputs=inputs, family_dir=result.family_dir
    )
    return result, summary, validation_dir, reused
