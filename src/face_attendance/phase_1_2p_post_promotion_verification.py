from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .embedding_forensics import EmbeddingForensicsError, load_embedding_payload
from .embedding_version_family import EmbeddingFamilyError, verify_embedding_family
from .mon_p3_source_ablation import (
    SourceAblationError,
    _load_benchmark_features,
    verify_source_ablation_output,
)
from .phase_1_2m_candidate_family import (
    CORRECTED_CONFIG_ID,
    Phase12MError,
    _score_expected,
)
from .phase_1_2o_explicit_promotion import (
    DEFAULT_CONTRACT,
    Phase12OPromotionError,
    PromotionContract,
    verify_promoted_state,
)
from .shadow_validation import (
    ShadowValidationError,
    verify_output_manifest,
    write_output_manifest,
)
from .tracklet_review import _sha256_file

SCHEMA_VERSION = 1
PHASE = "1.2P"
POLICY_VERSION = "phase-1.2p-post-promotion-operational-verification-v1.1"
PROMOTION_ID = "promotion-7cc01070e9bc644380c5"
SOURCE_ABLATION_ID = "source-ablation-da12b41f85bc813a6de1"
PARENT_FAMILY_ID = "embfam-7bc431a3ad762398d4e9"
PHASE_1_2M_VALIDATION_ID = "family-validation-fda6be48b23be13eb233"
EXPECTED_RECORDS = 354
EXPECTED_DIMENSION = 128
EXPECTED_BENCHMARK_ROWS = 120
EXPECTED_MON_P3_ROWS = 15


class Phase12PError(RuntimeError):
    """Raised when the promoted production state violates its frozen contract."""


@dataclass(frozen=True)
class Phase12PInputs:
    repo_root: Path
    promotion_dir: Path
    family_dir: Path
    production_embeddings: Path
    production_summary: Path
    current_pointer: Path
    source_ablation_dir: Path
    parent_family_dir: Path
    phase_1_2m_validation_dir: Path

    @classmethod
    def for_repo(cls, repo_root: Path) -> "Phase12PInputs":
        root = Path(repo_root).resolve()
        return cls(
            repo_root=root,
            promotion_dir=root / "models" / "promotion_history" / PROMOTION_ID,
            family_dir=root / "models" / "versions" / DEFAULT_CONTRACT.family_id,
            production_embeddings=root / "models" / "student_embeddings.pkl",
            production_summary=root / "models" / "embedding_summary.csv",
            current_pointer=root / "models" / "current_embedding_version.json",
            source_ablation_dir=(
                root
                / "attendance_output"
                / "embedding_forensics"
                / "phase_1_2l"
                / SOURCE_ABLATION_ID
            ),
            parent_family_dir=root / "models" / "versions" / PARENT_FAMILY_ID,
            phase_1_2m_validation_dir=(
                root
                / "attendance_output"
                / "embedding_forensics"
                / "phase_1_2m"
                / PHASE_1_2M_VALIDATION_ID
            ),
        )


@dataclass(frozen=True)
class Phase12PPreflight:
    verification_id: str
    promotion_id: str
    family_id: str
    candidate_variant_id: str
    production_embeddings_sha256: str
    production_summary_sha256: str
    parent_embeddings_backup_sha256: str
    parent_summary_backup_sha256: str
    production_embedding_records: int
    production_embedding_dimension: int
    family_verified_files: int
    promotion_verified_files: int
    source_ablation_verified_files: int
    phase_1_2m_validation_verified_files: int
    benchmark_rows_planned: int
    mon_p3_rows_planned: int
    rollback_command: str
    input_fingerprint_sha256: str
    production_changes: bool = False
    attendance_changes: bool = False
    recognition_execution: bool = False


@dataclass(frozen=True)
class Phase12PResult:
    verification_id: str
    output_dir: Path
    production_embedding_records: int
    benchmark_rows_reproduced: int
    mon_p3_rows_reproduced: int
    rollback_command: str
    idempotent_reuse: bool


def _text(value: Any) -> str:
    return str(value or "").strip()


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise Phase12PError(f"{label} not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase12PError(f"Unable to read {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise Phase12PError(f"{label} must contain a JSON object: {path}")
    return payload


def _stable_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


_ROLLBACK_COMMAND_RE = re.compile(
    r'^\s*&\s+"(?P<runner>[^"]+)"\s+'
    r'-Rollback\s+'
    r'-PromotionDir\s+"(?P<promotion_dir>[^"]+)"\s+'
    r'-ConfirmRollback\s+"(?P<promotion_id>[^"]+)"\s*$',
    flags=re.IGNORECASE,
)


def _canonical_path_key(value: str | Path) -> str:
    """Return an OS-aware key for a path without weakening path identity checks."""
    path = Path(value).expanduser()
    try:
        path = path.resolve(strict=False)
    except (OSError, RuntimeError):
        path = path.absolute()
    return os.path.normcase(os.path.normpath(str(path)))


def _parse_rollback_command(command: Any, *, label: str) -> tuple[str, str, str]:
    text = _text(command)
    match = _ROLLBACK_COMMAND_RE.fullmatch(text)
    if match is None:
        raise Phase12PError(f"{label} rollback command changed")
    return (
        match.group("runner"),
        match.group("promotion_dir"),
        match.group("promotion_id"),
    )


def _verify_rollback_command(
    command: Any,
    *,
    inputs: Phase12PInputs,
    promotion_id: str,
    label: str,
) -> None:
    runner, promotion_dir, parsed_id = _parse_rollback_command(command, label=label)
    expected_runner = inputs.repo_root / "scripts" / "run_phase_1_2o_explicit_promotion.ps1"
    checks = {
        "runner": _canonical_path_key(runner) == _canonical_path_key(expected_runner),
        "promotion directory": (
            _canonical_path_key(promotion_dir) == _canonical_path_key(inputs.promotion_dir)
        ),
        "promotion ID": parsed_id == promotion_id,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise Phase12PError(
            f"{label} rollback command changed: " + ", ".join(failed)
        )


def _expected_rollback_command(inputs: Phase12PInputs, promotion_id: str) -> str:
    runner = inputs.repo_root / "scripts" / "run_phase_1_2o_explicit_promotion.ps1"
    return (
        f'& "{runner}" -Rollback -PromotionDir "{inputs.promotion_dir}" '
        f'-ConfirmRollback "{promotion_id}"'
    )


def _verify_pointer_record_transaction(
    inputs: Phase12PInputs,
    contract: PromotionContract,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    pointer = _load_json(inputs.current_pointer, "current embedding pointer")
    record_path = inputs.promotion_dir / "promotion_record.json"
    record = _load_json(record_path, "promotion record")
    transaction = _load_json(inputs.promotion_dir / "transaction_state.json", "transaction state")

    checks = {
        "pointer status": _text(pointer.get("status")) == "promoted",
        "pointer promotion ID": _text(pointer.get("promotion_id")) == PROMOTION_ID,
        "pointer family ID": _text(pointer.get("family_id")) == contract.family_id,
        "pointer variant ID": _text(pointer.get("variant_id")) == contract.candidate_variant_id,
        "pointer embedding hash": (
            _text(pointer.get("production_embeddings_sha256"))
            == contract.candidate_embeddings_sha256
        ),
        "pointer summary hash": (
            _text(pointer.get("production_summary_sha256"))
            == contract.candidate_summary_sha256
        ),
        "record status": _text(record.get("status")) == "promoted",
        "record promotion ID": _text(record.get("promotion_id")) == PROMOTION_ID,
        "record family ID": _text(record.get("family_id")) == contract.family_id,
        "record variant ID": (
            _text(record.get("candidate_variant_id")) == contract.candidate_variant_id
        ),
        "record embedding hash": (
            _text(record.get("production_embeddings_sha256"))
            == contract.candidate_embeddings_sha256
        ),
        "record summary hash": (
            _text(record.get("production_summary_sha256"))
            == contract.candidate_summary_sha256
        ),
        "record parent embedding hash": (
            _text(record.get("parent_embeddings_sha256"))
            == contract.production_embeddings_sha256
        ),
        "record parent summary hash": (
            _text(record.get("parent_summary_sha256"))
            == contract.production_summary_sha256
        ),
        "record attendance unchanged": record.get("official_attendance_changed") is False,
        "record recognition not executed": record.get("recognition_executed") is False,
        "record datasets unchanged": record.get("datasets_changed") is False,
        "transaction status": _text(transaction.get("status")) == "promoted",
        "transaction promotion ID": _text(transaction.get("promotion_id")) == PROMOTION_ID,
        "transaction restoration flag": (
            transaction.get("automatic_restoration_required") is False
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise Phase12PError("Promoted lifecycle state failed: " + ", ".join(failed))

    pointer_record_path = Path(_text(pointer.get("promotion_record"))).resolve()
    if pointer_record_path != record_path.resolve():
        raise Phase12PError("Current pointer promotion-record path changed")
    pointer_hash = _sha256_file(inputs.current_pointer)
    if _text(record.get("current_pointer_sha256")) != pointer_hash:
        raise Phase12PError("Promotion record no longer matches the current pointer hash")

    expected_rollback = _expected_rollback_command(inputs, PROMOTION_ID)
    _verify_rollback_command(
        pointer.get("rollback_command"),
        inputs=inputs,
        promotion_id=PROMOTION_ID,
        label="Current pointer",
    )
    _verify_rollback_command(
        record.get("rollback_command"),
        inputs=inputs,
        promotion_id=PROMOTION_ID,
        label="Promotion record",
    )
    return pointer, record, transaction, expected_rollback


def _verify_production_payload(inputs: Phase12PInputs, contract: PromotionContract) -> tuple[int, int]:
    candidate_dir = inputs.family_dir / "variants" / "full_candidate"
    candidate_embeddings = candidate_dir / "student_embeddings.pkl"
    candidate_summary = candidate_dir / "embedding_summary.csv"
    if inputs.production_embeddings.read_bytes() != candidate_embeddings.read_bytes():
        raise Phase12PError("Production embeddings are not byte-identical to promoted Variant D")
    if inputs.production_summary.read_bytes() != candidate_summary.read_bytes():
        raise Phase12PError("Production summary is not byte-identical to promoted Variant D")
    if _sha256_file(inputs.production_embeddings) != contract.candidate_embeddings_sha256:
        raise Phase12PError("Production embedding hash changed after promotion")
    if _sha256_file(inputs.production_summary) != contract.candidate_summary_sha256:
        raise Phase12PError("Production summary hash changed after promotion")

    try:
        payload = load_embedding_payload(inputs.production_embeddings)
    except EmbeddingForensicsError as exc:
        raise Phase12PError(str(exc)) from exc
    records = list(payload.get("records") or [])
    if len(records) != EXPECTED_RECORDS:
        raise Phase12PError(
            f"Promoted production embedding count changed; expected {EXPECTED_RECORDS}, found {len(records)}"
        )
    dimensions: set[int] = set()
    for record in records:
        vector = np.asarray(record.get("embedding"), dtype=np.float32).reshape(-1)
        dimensions.add(int(vector.size))
        if not np.isfinite(vector).all():
            raise Phase12PError("Promoted production embeddings contain a non-finite vector")
    if dimensions != {EXPECTED_DIMENSION}:
        raise Phase12PError(
            f"Promoted production embedding dimensions changed: {sorted(dimensions)}"
        )
    return len(records), EXPECTED_DIMENSION


def _verify_phase_1_2m_validation(inputs: Phase12PInputs) -> int:
    directory = inputs.phase_1_2m_validation_dir
    try:
        verified = verify_output_manifest(directory, directory / "output_manifest.json")
    except ShadowValidationError as exc:
        raise Phase12PError(str(exc)) from exc
    summary = _load_json(directory / "validation_summary.json", "Phase 1.2M validation summary")
    checks = {
        "validation ID": _text(summary.get("validation_id")) == PHASE_1_2M_VALIDATION_ID,
        "family ID": _text(summary.get("family_id")) == DEFAULT_CONTRACT.family_id,
        "decision": (
            _text(summary.get("decision"))
            == "family_built_unapproved_exact_abl16_reproduction_passed"
        ),
        "config": _text(summary.get("corrected_config_id")) == CORRECTED_CONFIG_ID,
        "MON_P3 rows": int(summary.get("mon_p3_design_tracks_reproduced", -1)) == EXPECTED_MON_P3_ROWS,
        "benchmark rows": (
            int(summary.get("benchmark_descriptive_rows_reproduced", -1))
            == EXPECTED_BENCHMARK_ROWS
        ),
        "candidate unpromoted at build": summary.get("candidate_promoted") is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise Phase12PError("Phase 1.2M validation contract changed: " + ", ".join(failed))
    return int(verified.get("verified_files", 0))


def preflight_phase_1_2p(
    inputs: Phase12PInputs,
    contract: PromotionContract = DEFAULT_CONTRACT,
) -> Phase12PPreflight:
    if inputs.promotion_dir.name != PROMOTION_ID:
        raise Phase12PError("Unexpected promotion directory")
    try:
        promoted = verify_promoted_state(
            repo_root=inputs.repo_root,
            promotion_dir=inputs.promotion_dir,
            contract=contract,
        )
    except Phase12OPromotionError as exc:
        raise Phase12PError(str(exc)) from exc
    try:
        family_verified = verify_embedding_family(inputs.family_dir)
    except EmbeddingFamilyError as exc:
        raise Phase12PError(str(exc)) from exc
    try:
        source_verified = verify_source_ablation_output(inputs.source_ablation_dir)
    except SourceAblationError as exc:
        raise Phase12PError(str(exc)) from exc

    pointer, record, transaction, rollback_command = _verify_pointer_record_transaction(
        inputs, contract
    )
    records, dimension = _verify_production_payload(inputs, contract)
    validation_verified_files = _verify_phase_1_2m_validation(inputs)
    promotion_manifest = verify_output_manifest(
        inputs.promotion_dir,
        inputs.promotion_dir / "promotion_output_manifest.json",
    )

    backup_dir = inputs.promotion_dir / "production_before_promotion"
    backup_embeddings_hash = _sha256_file(backup_dir / "student_embeddings.pkl")
    backup_summary_hash = _sha256_file(backup_dir / "embedding_summary.csv")
    if backup_embeddings_hash != contract.production_embeddings_sha256:
        raise Phase12PError("Rollback embedding backup hash changed")
    if backup_summary_hash != contract.production_summary_sha256:
        raise Phase12PError("Rollback summary backup hash changed")

    input_identity = {
        "policy_version": POLICY_VERSION,
        "promotion_id": promoted.promotion_id,
        "family_id": promoted.family_id,
        "candidate_variant_id": promoted.candidate_variant_id,
        "production_embeddings_sha256": _sha256_file(inputs.production_embeddings),
        "production_summary_sha256": _sha256_file(inputs.production_summary),
        "current_pointer_sha256": _sha256_file(inputs.current_pointer),
        "promotion_record_sha256": _sha256_file(inputs.promotion_dir / "promotion_record.json"),
        "transaction_state_sha256": _sha256_file(inputs.promotion_dir / "transaction_state.json"),
        "promotion_output_manifest_sha256": _sha256_file(
            inputs.promotion_dir / "promotion_output_manifest.json"
        ),
        "family_output_manifest_sha256": _sha256_file(inputs.family_dir / "output_manifest.json"),
        "source_ablation_output_manifest_sha256": _sha256_file(
            inputs.source_ablation_dir / "output_manifest.json"
        ),
        "phase_1_2m_validation_output_manifest_sha256": _sha256_file(
            inputs.phase_1_2m_validation_dir / "output_manifest.json"
        ),
        "rollback_embeddings_sha256": backup_embeddings_hash,
        "rollback_summary_sha256": backup_summary_hash,
    }
    fingerprint = _stable_digest(input_identity)
    verification_id = f"post-promotion-verification-{fingerprint[:20]}"
    return Phase12PPreflight(
        verification_id=verification_id,
        promotion_id=promoted.promotion_id,
        family_id=promoted.family_id,
        candidate_variant_id=promoted.candidate_variant_id,
        production_embeddings_sha256=contract.candidate_embeddings_sha256,
        production_summary_sha256=contract.candidate_summary_sha256,
        parent_embeddings_backup_sha256=backup_embeddings_hash,
        parent_summary_backup_sha256=backup_summary_hash,
        production_embedding_records=records,
        production_embedding_dimension=dimension,
        family_verified_files=int(family_verified.get("verified_files", 0)),
        promotion_verified_files=int(promotion_manifest.get("verified_files", 0)),
        source_ablation_verified_files=int(source_verified.get("verified_files", 0)),
        phase_1_2m_validation_verified_files=validation_verified_files,
        benchmark_rows_planned=EXPECTED_BENCHMARK_ROWS,
        mon_p3_rows_planned=EXPECTED_MON_P3_ROWS,
        rollback_command=rollback_command,
        input_fingerprint_sha256=fingerprint,
    )


def write_preflight(preflight: Phase12PPreflight, output_path: Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "policy_version": POLICY_VERSION,
        "created_at": _now(),
        **asdict(preflight),
        "verification_executed": False,
        "production_embeddings_changed": False,
        "production_summary_changed": False,
        "official_attendance_changed": False,
        "recognition_executed": False,
        "exact_next_step": "Run the Phase 1.2P verification command; it is read-only.",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path.resolve()


def _reproduce_frozen_predictions(
    inputs: Phase12PInputs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_dir = inputs.source_ablation_dir
    benchmark_ids, benchmark_features = _load_benchmark_features(
        inputs.parent_family_dir / "evaluation" / "benchmark_feature_cache.pkl",
        inputs.parent_family_dir / "evaluation" / "benchmark_feature_manifest.json",
    )
    benchmark_expected_all = pd.read_csv(
        source_dir / "benchmark_descriptive_subset_predictions.csv",
        dtype=str,
        keep_default_na=False,
    )
    benchmark_expected = benchmark_expected_all[
        benchmark_expected_all["Model_ID"].eq(CORRECTED_CONFIG_ID)
    ].copy()
    benchmark_reproduction = _score_expected(
        database_path=inputs.production_embeddings,
        features=benchmark_features,
        ids=benchmark_ids,
        expected=benchmark_expected,
        id_column="Benchmark_Row_ID",
        label="Promoted production 120-track ABL-16 reproduction",
    )

    with np.load(
        source_dir / "mon_p3_reviewed_track_feature_cache.npz",
        allow_pickle=False,
    ) as cache:
        mon_ids = [_text(value) for value in cache["review_ids"]]
        mon_features = np.asarray(cache["embeddings"], dtype=np.float32)
    mon_expected_all = pd.read_csv(
        source_dir / "mon_p3_subset_predictions.csv",
        dtype=str,
        keep_default_na=False,
    )
    mon_expected = mon_expected_all[
        mon_expected_all["Model_ID"].eq(CORRECTED_CONFIG_ID)
    ].copy()
    mon_reproduction = _score_expected(
        database_path=inputs.production_embeddings,
        features=mon_features,
        ids=mon_ids,
        expected=mon_expected,
        id_column="Review_ID",
        label="Promoted production 15-track MON_P3 ABL-16 reproduction",
    )
    if len(benchmark_reproduction) != EXPECTED_BENCHMARK_ROWS:
        raise Phase12PError("Promoted production benchmark reproduction row count changed")
    if len(mon_reproduction) != EXPECTED_MON_P3_ROWS:
        raise Phase12PError("Promoted production MON_P3 reproduction row count changed")
    return benchmark_reproduction, mon_reproduction


def run_phase_1_2p(
    *,
    inputs: Phase12PInputs,
    output_root: Path | None = None,
    contract: PromotionContract = DEFAULT_CONTRACT,
) -> Phase12PResult:
    preflight = preflight_phase_1_2p(inputs, contract)
    output_root = Path(
        output_root
        or inputs.repo_root
        / "attendance_output"
        / "embedding_forensics"
        / "phase_1_2p"
    ).resolve()
    final_dir = output_root / preflight.verification_id
    if final_dir.exists():
        try:
            verify_output_manifest(final_dir, final_dir / "output_manifest.json")
        except ShadowValidationError as exc:
            raise Phase12PError(str(exc)) from exc
        summary = _load_json(final_dir / "verification_summary.json", "verification summary")
        if _text(summary.get("input_fingerprint_sha256")) != preflight.input_fingerprint_sha256:
            raise Phase12PError("Existing Phase 1.2P output has different inputs")
        return Phase12PResult(
            verification_id=preflight.verification_id,
            output_dir=final_dir,
            production_embedding_records=int(summary["production_embedding_records"]),
            benchmark_rows_reproduced=int(summary["benchmark_rows_reproduced"]),
            mon_p3_rows_reproduced=int(summary["mon_p3_rows_reproduced"]),
            rollback_command=_text(summary.get("rollback_command")),
            idempotent_reuse=True,
        )

    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / f".{preflight.verification_id}.building"
    if temporary.exists():
        raise Phase12PError(f"Stale Phase 1.2P build directory exists: {temporary}")
    temporary.mkdir()
    try:
        benchmark, mon_p3 = _reproduce_frozen_predictions(inputs)
        benchmark.to_csv(temporary / "production_benchmark_reproduction.csv", index=False)
        mon_p3.to_csv(temporary / "production_mon_p3_reproduction.csv", index=False)

        checks = [
            ("production_matches_promoted_variant_d", True, contract.candidate_embeddings_sha256),
            ("production_summary_matches_promoted_variant_d", True, contract.candidate_summary_sha256),
            ("current_pointer_matches_promotion_record", True, PROMOTION_ID),
            ("promotion_transaction_status", True, "promoted"),
            ("rollback_embedding_backup_verified", True, contract.production_embeddings_sha256),
            ("rollback_summary_backup_verified", True, contract.production_summary_sha256),
            ("family_manifest_verified", True, str(preflight.family_verified_files)),
            ("promotion_manifest_verified", True, str(preflight.promotion_verified_files)),
            ("source_ablation_manifest_verified", True, str(preflight.source_ablation_verified_files)),
            ("phase_1_2m_validation_verified", True, str(preflight.phase_1_2m_validation_verified_files)),
            ("production_embedding_payload_load", True, f"{preflight.production_embedding_records}x{preflight.production_embedding_dimension}"),
            ("production_benchmark_exact_reproduction", True, f"{len(benchmark)}/{EXPECTED_BENCHMARK_ROWS}"),
            ("production_mon_p3_exact_reproduction", True, f"{len(mon_p3)}/{EXPECTED_MON_P3_ROWS}"),
            ("official_attendance_changed", False, "no"),
            ("recognition_executed", False, "no"),
        ]
        pd.DataFrame(checks, columns=["Check", "Value", "Detail"]).to_csv(
            temporary / "operational_integrity_checks.csv", index=False
        )
        (temporary / "rollback_command.txt").write_text(
            preflight.rollback_command + "\n", encoding="utf-8"
        )
        summary = {
            "schema_version": SCHEMA_VERSION,
            "phase": PHASE,
            "policy_version": POLICY_VERSION,
            "verification_id": preflight.verification_id,
            "created_at": _now(),
            "decision": "promoted_production_operational_verification_passed",
            "promotion_id": preflight.promotion_id,
            "family_id": preflight.family_id,
            "candidate_variant_id": preflight.candidate_variant_id,
            "input_fingerprint_sha256": preflight.input_fingerprint_sha256,
            "production_embeddings_sha256": preflight.production_embeddings_sha256,
            "production_summary_sha256": preflight.production_summary_sha256,
            "parent_embeddings_backup_sha256": preflight.parent_embeddings_backup_sha256,
            "parent_summary_backup_sha256": preflight.parent_summary_backup_sha256,
            "production_embedding_records": preflight.production_embedding_records,
            "production_embedding_dimension": preflight.production_embedding_dimension,
            "benchmark_rows_reproduced": len(benchmark),
            "mon_p3_rows_reproduced": len(mon_p3),
            "rollback_command": preflight.rollback_command,
            "rollback_resolvable": True,
            "production_embeddings_changed_during_verification": False,
            "production_summary_changed_during_verification": False,
            "official_attendance_changed": False,
            "recognition_executed": False,
            "datasets_changed": False,
            "next_product_stage": "Return to dashboard, classes, reports, review, students, and HOD workflows.",
        }
        (temporary / "verification_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        report = "\n".join(
            [
                "# Phase 1.2P Post-Promotion Operational Verification",
                "",
                f"- Production family: `{preflight.family_id}`",
                f"- Promotion: `{preflight.promotion_id}`",
                f"- Production embedding records: {preflight.production_embedding_records}",
                f"- Frozen benchmark reproduction: {len(benchmark)}/{EXPECTED_BENCHMARK_ROWS}",
                f"- MON_P3 fingerprint reproduction: {len(mon_p3)}/{EXPECTED_MON_P3_ROWS}",
                "- Production files match the promoted candidate byte-for-byte: yes",
                "- Rollback backup verified and resolvable: yes",
                "- Attendance changed during verification: no",
                "- Recognition executed during verification: no",
                "",
                "## Professor-ready summary",
                "",
                "The new embedding family was promoted only after reusable benchmark testing and a separate untouched CCTV validation session. Post-promotion verification confirms that the live production files exactly match the validated candidate, reproduce the expected recognition fingerprints, and retain a verified rollback copy of the previous production model.",
                "",
                "## Next stage",
                "",
                "Return to product workflows: Dashboard, My Classes, My Reports, Review Students, My Students, and HOD pages. Live Demo remains last.",
            ]
        ) + "\n"
        (temporary / "verification_report.md").write_text(report, encoding="utf-8")
        write_output_manifest(
            temporary,
            {
                "phase": PHASE,
                "policy_version": POLICY_VERSION,
                "verification_id": preflight.verification_id,
                "decision": summary["decision"],
                "promotion_id": preflight.promotion_id,
                "family_id": preflight.family_id,
                "production_embeddings_changed": False,
                "production_summary_changed": False,
                "official_attendance_changed": False,
                "recognition_executed": False,
            },
        )
        verify_output_manifest(temporary, temporary / "output_manifest.json")
        temporary.replace(final_dir)
        verify_output_manifest(final_dir, final_dir / "output_manifest.json")
    except (
        Phase12PError,
        Phase12MError,
        SourceAblationError,
        ShadowValidationError,
        EmbeddingForensicsError,
    ) as exc:
        raise Phase12PError(str(exc)) from exc
    except Exception:
        raise
    finally:
        if temporary.exists():
            import shutil

            shutil.rmtree(temporary, ignore_errors=True)

    return Phase12PResult(
        verification_id=preflight.verification_id,
        output_dir=final_dir,
        production_embedding_records=preflight.production_embedding_records,
        benchmark_rows_reproduced=EXPECTED_BENCHMARK_ROWS,
        mon_p3_rows_reproduced=EXPECTED_MON_P3_ROWS,
        rollback_command=preflight.rollback_command,
        idempotent_reuse=False,
    )
