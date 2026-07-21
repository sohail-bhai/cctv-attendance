from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1
PHASE = "1.2O"
POLICY_VERSION = "phase-1.2o-explicit-promotion-rollback-v1"


class Phase12OPromotionError(RuntimeError):
    """Raised when a Phase 1.2O safety contract is violated."""


@dataclass(frozen=True)
class PromotionContract:
    family_id: str = "embfam-274b5207b8b71294ff75"
    candidate_variant_id: str = "embfam-274b5207b8b71294ff75-d"
    expected_family_decision: str = "built_unapproved_pending_new_untouched_session"
    session_id: str = "2026-06-22__B51__P4__CVO"
    run_id: str = "mon-p4-shadow-695277fbd0dcd9d431e1"
    evaluation_id: str = "mon-p4-evaluation-302c6707c95029a017df"
    evaluation_decision: str = "mon_p4_shadow_passed_pending_explicit_promotion"
    reviewed_exception_tracks: int = 5
    candidate_correct_recoveries: int = 3
    candidate_false_identities_or_unsafe_accepts: int = 0
    candidate_lost_correct_production_accepts: int = 0
    unverifiable_exceptions: int = 0
    production_embeddings_sha256: str = (
        "c32ed31df10b7b9b43b8f19977a2adf0a82fb8e7a71f3e8bceb42c0565fedd49"
    )
    production_summary_sha256: str = (
        "0df35c3bb9207e191aa49dad5536b8378e812d56463491bd7203885dca5ae9ee"
    )
    candidate_embeddings_sha256: str = (
        "f088d827adc548ee95f46566d758fd71fc304d042c43f1ecffc6526b60bcd832"
    )
    candidate_summary_sha256: str = (
        "63885588c374c37f4da9bf85294f240bdf0f28cb585e77b894ab516139ca46ae"
    )
    frozen_family_manifest_sha256: str = (
        "66c67c433847d171dfa7ed0ee829b156f9c502e069450e5edc7aa7ddc6ad91e0"
    )
    frozen_version_manifest_sha256: str = (
        "341e0211e4d5ae669b2f991825a1d7ff724b93dba8031f1c8a621d88a687fcd5"
    )
    source_freeze_json_sha256: str = (
        "6d60252b7e70941a2466bf7a43996418939a5853d8a8123ccc8b549583396e07"
    )
    shadow_output_manifest_sha256: str = (
        "b00db94aa2f90d777619f5c3991bf12f81d3e37049c29f0b2e134ce043b3c08d"
    )
    shadow_summary_sha256: str = (
        "e96d341cb2b68de54e3b6dbf06af85c0f52a5fd0c702f35313ffac4cc60daa18"
    )
    protected_state_comparison_sha256: str = (
        "136eb33f35d5c466ba0a8a4e716f0aababd4b8deb75c881b6ab71adad71501b5"
    )
    evaluation_output_manifest_sha256: str = (
        "5fc93d3e60f190aaf02bda4842124a9663011cf2612d7ef3ef81c1fa663ba111"
    )
    evaluation_summary_sha256: str = (
        "07f5725b55e3f54925ca7717b19297de4bd59c06883fa396b1381e23f432c738"
    )
    reviewed_exception_tracks_sha256: str = (
        "4996bb47ad39da5ba8a908e3af4ada89eda07609fb425aa3db3fb4caa91c10dc"
    )
    submitted_labels_sha256: str = (
        "a256a30f1c27934be9dda896e3451db608d0dcc60270ccd4efbec9cf313906fe"
    )


DEFAULT_CONTRACT = PromotionContract()


@dataclass(frozen=True)
class PromotionInputs:
    repo_root: Path
    family_dir: Path
    shadow_dir: Path
    evaluation_dir: Path
    labels_path: Path
    production_embeddings: Path
    production_summary: Path

    @classmethod
    def for_repo(cls, repo_root: Path, labels_path: Path) -> "PromotionInputs":
        root = Path(repo_root).resolve()
        family = root / "models" / "versions" / DEFAULT_CONTRACT.family_id
        shadow = (
            root
            / "attendance_output"
            / "shadow_validation"
            / "phase_1_2n"
            / "shadow_runs"
            / DEFAULT_CONTRACT.run_id
        )
        evaluation = shadow / "evaluation" / DEFAULT_CONTRACT.evaluation_id
        return cls(
            repo_root=root,
            family_dir=family,
            shadow_dir=shadow,
            evaluation_dir=evaluation,
            labels_path=Path(labels_path).resolve(),
            production_embeddings=root / "models" / "student_embeddings.pkl",
            production_summary=root / "models" / "embedding_summary.csv",
        )


@dataclass(frozen=True)
class PromotionPreflight:
    promotion_id: str
    family_id: str
    candidate_variant_id: str
    session_id: str
    run_id: str
    evaluation_id: str
    evaluation_decision: str
    candidate_correct_recoveries: int
    candidate_false_identities_or_unsafe_accepts: int
    candidate_lost_correct_production_accepts: int
    unverifiable_exceptions: int
    reviewed_exception_tracks: int
    production_embeddings_sha256: str
    production_summary_sha256: str
    candidate_embeddings_sha256: str
    candidate_summary_sha256: str
    labels_sha256: str
    evidence_fingerprint_sha256: str
    family_output_verified_files: int
    shadow_output_verified_files: int
    evaluation_output_verified_files: int
    production_changes: bool = False
    attendance_changes: bool = False
    candidate_promoted: bool = False


@dataclass(frozen=True)
class PromotionResult:
    promotion_id: str
    promotion_dir: Path
    family_id: str
    candidate_variant_id: str
    production_embeddings_sha256: str
    production_summary_sha256: str
    rollback_command: str
    idempotent_reuse: bool


@dataclass(frozen=True)
class RollbackResult:
    promotion_id: str
    promotion_dir: Path
    restored_embeddings_sha256: str
    restored_summary_sha256: str
    idempotent_reuse: bool


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _text(value: Any) -> str:
    return str(value or "").strip()


def _sha256_file(path: Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise Phase12OPromotionError(f"Required file not found: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise Phase12OPromotionError(f"{label} not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase12OPromotionError(f"Unable to read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Phase12OPromotionError(f"{label} must contain a JSON object: {path}")
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise Phase12OPromotionError(f"Stale JSON temporary file exists: {temporary}")
    data = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _safe_manifest_path(root: Path, relative: str) -> Path:
    relative_path = Path(str(relative).replace("\\", "/"))
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise Phase12OPromotionError(f"Unsafe manifest path: {relative}")
    root = Path(root).resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise Phase12OPromotionError(f"Manifest path escapes its output directory: {relative}") from exc
    return candidate


def verify_output_manifest(directory: Path, manifest_path: Path) -> dict[str, Any]:
    directory = Path(directory).resolve()
    manifest = _load_json(manifest_path, "output manifest")
    hashes = manifest.get("files_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise Phase12OPromotionError(f"Output manifest has no files_sha256 mapping: {manifest_path}")
    verified = 0
    for relative, expected in sorted(hashes.items()):
        expected_hash = _text(expected).lower()
        if len(expected_hash) != 64:
            raise Phase12OPromotionError(f"Invalid SHA-256 in manifest for {relative}")
        path = _safe_manifest_path(directory, _text(relative))
        actual = _sha256_file(path)
        if actual != expected_hash:
            raise Phase12OPromotionError(
                f"Output manifest hash mismatch for {path}; expected {expected_hash}, found {actual}"
            )
        verified += 1
    return {"verified_files": verified, "manifest": manifest}


def _assert_hash(path: Path, expected: str, label: str) -> str:
    actual = _sha256_file(path)
    if actual != expected.lower():
        raise Phase12OPromotionError(
            f"{label} hash mismatch; expected {expected.lower()}, found {actual}: {path}"
        )
    return actual


def _required_int(payload: dict[str, Any], key: str) -> int:
    if key not in payload:
        raise Phase12OPromotionError(f"Required integer is missing: {key}")
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise Phase12OPromotionError(f"Required integer must be an integer: {key}={value!r}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.is_file():
        raise Phase12OPromotionError(f"CSV file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _verify_family(inputs: PromotionInputs, contract: PromotionContract) -> int:
    family_dir = Path(inputs.family_dir).resolve()
    if family_dir.name != contract.family_id:
        raise Phase12OPromotionError(
            f"Unexpected family directory. Expected {contract.family_id}, found {family_dir.name}"
        )
    family_manifest_path = family_dir / "family_manifest.json"
    variant_dir = family_dir / "variants" / "full_candidate"
    version_manifest_path = variant_dir / "version_manifest.json"
    candidate_embeddings = variant_dir / "student_embeddings.pkl"
    candidate_summary = variant_dir / "embedding_summary.csv"
    _assert_hash(
        family_manifest_path,
        contract.frozen_family_manifest_sha256,
        "Candidate family manifest",
    )
    _assert_hash(
        version_manifest_path,
        contract.frozen_version_manifest_sha256,
        "Candidate version manifest",
    )
    _assert_hash(
        candidate_embeddings,
        contract.candidate_embeddings_sha256,
        "Candidate embeddings",
    )
    _assert_hash(
        candidate_summary,
        contract.candidate_summary_sha256,
        "Candidate summary",
    )
    family_manifest = _load_json(family_manifest_path, "candidate family manifest")
    version_manifest = _load_json(version_manifest_path, "candidate version manifest")
    if _text(family_manifest.get("family_id")) != contract.family_id:
        raise Phase12OPromotionError("Candidate family ID changed")
    if _text(family_manifest.get("final_decision")) != contract.expected_family_decision:
        raise Phase12OPromotionError("Candidate family decision changed")
    if family_manifest.get("candidate_promoted") is not False:
        raise Phase12OPromotionError("Candidate family manifest already reports promotion")
    if family_manifest.get("production_approved") is not False:
        raise Phase12OPromotionError("Candidate family manifest unexpectedly reports approval")
    if _text(version_manifest.get("version_id")) != contract.candidate_variant_id:
        raise Phase12OPromotionError("Candidate variant ID changed")
    if _text(version_manifest.get("status")) != "built_unapproved":
        raise Phase12OPromotionError("Candidate variant is not built_unapproved")
    if version_manifest.get("availability") is not True:
        raise Phase12OPromotionError("Candidate variant is unavailable")
    if version_manifest.get("production_promoted") is not False:
        raise Phase12OPromotionError("Candidate variant already reports production promotion")
    artifacts = dict(version_manifest.get("artifact_sha256") or {})
    if _text(artifacts.get("student_embeddings.pkl")) != contract.candidate_embeddings_sha256:
        raise Phase12OPromotionError("Candidate embedding hash changed in version manifest")
    if _text(artifacts.get("embedding_summary.csv")) != contract.candidate_summary_sha256:
        raise Phase12OPromotionError("Candidate summary hash changed in version manifest")
    family_output = verify_output_manifest(family_dir, family_dir / "output_manifest.json")
    return int(family_output["verified_files"])




def _expected_result_counts(contract: PromotionContract) -> Counter[str]:
    return Counter(
        {
            "both_correct_same_identity": 1,
            "candidate_correct_recovery": contract.candidate_correct_recoveries,
            "reviewed_other": 1,
        }
    )


def _evidence_identity_payload(
    contract: PromotionContract,
    labels_sha256: str,
    result_counts: Counter[str] | dict[str, int],
) -> dict[str, Any]:
    return {
        "policy_version": POLICY_VERSION,
        "family_id": contract.family_id,
        "candidate_variant_id": contract.candidate_variant_id,
        "candidate_embeddings_sha256": contract.candidate_embeddings_sha256,
        "candidate_summary_sha256": contract.candidate_summary_sha256,
        "parent_embeddings_sha256": contract.production_embeddings_sha256,
        "parent_summary_sha256": contract.production_summary_sha256,
        "run_id": contract.run_id,
        "evaluation_id": contract.evaluation_id,
        "evaluation_summary_sha256": contract.evaluation_summary_sha256,
        "reviewed_exception_tracks_sha256": contract.reviewed_exception_tracks_sha256,
        "labels_sha256": labels_sha256,
        "result_counts": dict(sorted(dict(result_counts).items())),
    }


def expected_promotion_id(contract: PromotionContract = DEFAULT_CONTRACT) -> str:
    fingerprint = _stable_digest(
        _evidence_identity_payload(
            contract,
            contract.submitted_labels_sha256,
            _expected_result_counts(contract),
        )
    )
    return "promotion-" + fingerprint[:20]


def _verify_review_alignment(
    labels_path: Path,
    reviewed_tracks_path: Path,
    contract: PromotionContract,
) -> tuple[int, Counter[str]]:
    labels = _read_csv(labels_path)
    reviewed = _read_csv(reviewed_tracks_path)
    if len(labels) != contract.reviewed_exception_tracks:
        raise Phase12OPromotionError(
            f"Expected {contract.reviewed_exception_tracks} submitted labels; found {len(labels)}"
        )
    if len(reviewed) != contract.reviewed_exception_tracks:
        raise Phase12OPromotionError(
            f"Expected {contract.reviewed_exception_tracks} reviewed exception rows; found {len(reviewed)}"
        )
    label_by_id: dict[str, dict[str, str]] = {}
    for row in labels:
        review_id = _text(row.get("Review_ID"))
        if not review_id or review_id in label_by_id:
            raise Phase12OPromotionError("Submitted labels contain a missing or duplicate Review_ID")
        label_by_id[review_id] = row
    result_counts: Counter[str] = Counter()
    reviewed_ids: set[str] = set()
    for row in reviewed:
        review_id = _text(row.get("Review_ID"))
        if not review_id or review_id in reviewed_ids:
            raise Phase12OPromotionError("Reviewed exceptions contain a missing or duplicate Review_ID")
        reviewed_ids.add(review_id)
        if review_id not in label_by_id:
            raise Phase12OPromotionError(f"Reviewed exception has no submitted label: {review_id}")
        submitted = label_by_id[review_id]
        if _text(submitted.get("Package_ID")) != _text(row.get("Package_ID")):
            raise Phase12OPromotionError(f"Package ID mismatch for {review_id}")
        if _text(submitted.get("Review_Status")) != _text(row.get("Review_Status")):
            raise Phase12OPromotionError(f"Review status mismatch for {review_id}")
        if _text(submitted.get("Actual_Roll")) != _text(row.get("Actual_Roll")):
            raise Phase12OPromotionError(f"Actual roll mismatch for {review_id}")
        result = _text(row.get("Phase_1_2N_Result"))
        if not result:
            raise Phase12OPromotionError(f"Reviewed exception result missing for {review_id}")
        result_counts[result] += 1
    if set(label_by_id) != reviewed_ids:
        raise Phase12OPromotionError("Submitted labels and reviewed exception IDs do not match exactly")
    unsafe_codes = {
        "candidate_false_identity",
        "candidate_unsafe_nonstudent_or_mixed",
        "candidate_lost_correct_production_accept",
        "unverifiable",
    }
    found_unsafe = {code: result_counts[code] for code in unsafe_codes if result_counts[code]}
    if found_unsafe:
        raise Phase12OPromotionError(f"Promotion evidence contains unsafe results: {found_unsafe}")
    expected_counts = _expected_result_counts(contract)
    if result_counts != expected_counts:
        raise Phase12OPromotionError(
            "Independent reviewed-track results changed; "
            f"expected {dict(expected_counts)}, found {dict(result_counts)}"
        )
    return len(reviewed), result_counts


def verify_promotion_evidence(
    inputs: PromotionInputs,
    contract: PromotionContract = DEFAULT_CONTRACT,
) -> PromotionPreflight:
    for path, label in (
        (inputs.repo_root, "repository root"),
        (inputs.family_dir, "candidate family directory"),
        (inputs.shadow_dir, "Phase 1.2N shadow directory"),
        (inputs.evaluation_dir, "Phase 1.2N evaluation directory"),
    ):
        if not Path(path).is_dir():
            raise Phase12OPromotionError(f"{label} not found: {path}")

    production_embeddings_hash = _assert_hash(
        inputs.production_embeddings,
        contract.production_embeddings_sha256,
        "Current production embeddings",
    )
    production_summary_hash = _assert_hash(
        inputs.production_summary,
        contract.production_summary_sha256,
        "Current production summary",
    )
    labels_hash = _assert_hash(
        inputs.labels_path,
        contract.submitted_labels_sha256,
        "Submitted blind-review labels",
    )

    family_verified_files = _verify_family(inputs, contract)

    shadow_output_manifest_path = inputs.shadow_dir / "output_manifest.json"
    _assert_hash(
        shadow_output_manifest_path,
        contract.shadow_output_manifest_sha256,
        "Phase 1.2N output manifest",
    )
    shadow_verified = verify_output_manifest(inputs.shadow_dir, shadow_output_manifest_path)
    _assert_hash(
        inputs.shadow_dir / "shadow_run_summary.json",
        contract.shadow_summary_sha256,
        "Phase 1.2N shadow summary",
    )
    _assert_hash(
        inputs.shadow_dir / "protected_state_comparison.json",
        contract.protected_state_comparison_sha256,
        "Phase 1.2N protected-state comparison",
    )
    _assert_hash(
        inputs.shadow_dir / "frozen_inputs" / "family_manifest.json",
        contract.frozen_family_manifest_sha256,
        "Frozen candidate family manifest",
    )
    _assert_hash(
        inputs.shadow_dir / "frozen_inputs" / "version_manifest.json",
        contract.frozen_version_manifest_sha256,
        "Frozen candidate version manifest",
    )
    _assert_hash(
        inputs.shadow_dir / "frozen_inputs" / "phase_1_2n_mon_p4_source_freeze.json",
        contract.source_freeze_json_sha256,
        "MON_P4 source-freeze JSON",
    )

    shadow_summary = _load_json(inputs.shadow_dir / "shadow_run_summary.json", "shadow summary")
    protected = _load_json(
        inputs.shadow_dir / "protected_state_comparison.json",
        "protected-state comparison",
    )
    source_freeze = _load_json(
        inputs.shadow_dir / "frozen_inputs" / "phase_1_2n_mon_p4_source_freeze.json",
        "MON_P4 source freeze",
    )
    if protected.get("unchanged") is not True:
        raise Phase12OPromotionError("Phase 1.2N protected-state comparison is not unchanged")
    if source_freeze.get("SourceFreezePassed") is not True:
        raise Phase12OPromotionError("MON_P4 source freeze did not pass")
    if source_freeze.get("UntouchedContractPassed") is not True:
        raise Phase12OPromotionError("MON_P4 untouched-session contract did not pass")
    if _text(source_freeze.get("SessionID")) != contract.session_id:
        raise Phase12OPromotionError("MON_P4 source-freeze session changed")
    if _text(shadow_summary.get("run_id")) != contract.run_id:
        raise Phase12OPromotionError("Phase 1.2N run ID changed")
    if _text(shadow_summary.get("session_id")) != contract.session_id:
        raise Phase12OPromotionError("Phase 1.2N session ID changed")
    if _text(shadow_summary.get("family_id")) != contract.family_id:
        raise Phase12OPromotionError("Phase 1.2N family ID changed")
    if _text(shadow_summary.get("candidate_variant_id")) != contract.candidate_variant_id:
        raise Phase12OPromotionError("Phase 1.2N candidate variant changed")
    if shadow_summary.get("production_embeddings_changed") is not False:
        raise Phase12OPromotionError("Phase 1.2N claims production embeddings changed")
    if shadow_summary.get("official_attendance_changed") is not False:
        raise Phase12OPromotionError("Phase 1.2N claims attendance changed")
    if shadow_summary.get("candidate_promoted") is not False:
        raise Phase12OPromotionError("Phase 1.2N claims the candidate was already promoted")

    evaluation_manifest_path = inputs.evaluation_dir / "output_manifest.json"
    evaluation_summary_path = inputs.evaluation_dir / "mon_p4_evaluation_summary.json"
    reviewed_tracks_path = inputs.evaluation_dir / "reviewed_exception_tracks.csv"
    _assert_hash(
        evaluation_manifest_path,
        contract.evaluation_output_manifest_sha256,
        "Phase 1.2N evaluation output manifest",
    )
    evaluation_verified = verify_output_manifest(inputs.evaluation_dir, evaluation_manifest_path)
    _assert_hash(
        evaluation_summary_path,
        contract.evaluation_summary_sha256,
        "Phase 1.2N evaluation summary",
    )
    _assert_hash(
        reviewed_tracks_path,
        contract.reviewed_exception_tracks_sha256,
        "Reviewed exception tracks",
    )
    evaluation = _load_json(evaluation_summary_path, "Phase 1.2N evaluation summary")
    if _text(evaluation.get("evaluation_id")) != contract.evaluation_id:
        raise Phase12OPromotionError("Phase 1.2N evaluation ID changed")
    if _text(evaluation.get("run_id")) != contract.run_id:
        raise Phase12OPromotionError("Phase 1.2N evaluation run ID changed")
    if _text(evaluation.get("family_id")) != contract.family_id:
        raise Phase12OPromotionError("Phase 1.2N evaluation family ID changed")
    if _text(evaluation.get("candidate_variant_id")) != contract.candidate_variant_id:
        raise Phase12OPromotionError("Phase 1.2N evaluation candidate variant changed")
    if _text(evaluation.get("decision")) != contract.evaluation_decision:
        raise Phase12OPromotionError("Phase 1.2N evaluation is not promotion-eligible")
    checks = {
        "reviewed_exception_tracks": contract.reviewed_exception_tracks,
        "candidate_correct_recoveries": contract.candidate_correct_recoveries,
        "candidate_false_identities_or_unsafe_accepts": (
            contract.candidate_false_identities_or_unsafe_accepts
        ),
        "candidate_lost_correct_production_accepts": (
            contract.candidate_lost_correct_production_accepts
        ),
        "unverifiable_exceptions": contract.unverifiable_exceptions,
    }
    for key, expected in checks.items():
        actual = _required_int(evaluation, key)
        if actual != expected:
            raise Phase12OPromotionError(
                f"Phase 1.2N evaluation metric changed: {key}; expected {expected}, found {actual}"
            )
    if evaluation.get("zero_false_identity_gate_passed") is not True:
        raise Phase12OPromotionError("Zero-false-identity gate did not pass")
    if evaluation.get("retention_gate_passed") is not True:
        raise Phase12OPromotionError("Retention gate did not pass")
    if evaluation.get("evidence_complete_gate_passed") is not True:
        raise Phase12OPromotionError("Evidence-complete gate did not pass")
    if evaluation.get("candidate_promoted") is not False:
        raise Phase12OPromotionError("Evaluation unexpectedly reports promotion")
    if evaluation.get("production_approved") is not False:
        raise Phase12OPromotionError("Evaluation unexpectedly reports production approval")

    reviewed_count, result_counts = _verify_review_alignment(
        inputs.labels_path,
        reviewed_tracks_path,
        contract,
    )
    evidence_payload = _evidence_identity_payload(
        contract,
        labels_hash,
        result_counts,
    )
    evidence_fingerprint = _stable_digest(evidence_payload)
    promotion_id = "promotion-" + evidence_fingerprint[:20]
    pointer_path = _promotion_pointer_path(inputs)
    promotion_dir = _promotion_history_root(inputs) / promotion_id
    if pointer_path.exists():
        raise Phase12OPromotionError(
            "Current embedding version pointer already exists; promotion preflight is ambiguous"
        )
    if promotion_dir.exists():
        raise Phase12OPromotionError(
            f"Promotion history directory already exists before promotion: {promotion_dir}"
        )
    return PromotionPreflight(
        promotion_id=promotion_id,
        family_id=contract.family_id,
        candidate_variant_id=contract.candidate_variant_id,
        session_id=contract.session_id,
        run_id=contract.run_id,
        evaluation_id=contract.evaluation_id,
        evaluation_decision=contract.evaluation_decision,
        candidate_correct_recoveries=contract.candidate_correct_recoveries,
        candidate_false_identities_or_unsafe_accepts=(
            contract.candidate_false_identities_or_unsafe_accepts
        ),
        candidate_lost_correct_production_accepts=(
            contract.candidate_lost_correct_production_accepts
        ),
        unverifiable_exceptions=contract.unverifiable_exceptions,
        reviewed_exception_tracks=reviewed_count,
        production_embeddings_sha256=production_embeddings_hash,
        production_summary_sha256=production_summary_hash,
        candidate_embeddings_sha256=contract.candidate_embeddings_sha256,
        candidate_summary_sha256=contract.candidate_summary_sha256,
        labels_sha256=labels_hash,
        evidence_fingerprint_sha256=evidence_fingerprint,
        family_output_verified_files=family_verified_files,
        shadow_output_verified_files=int(shadow_verified["verified_files"]),
        evaluation_output_verified_files=int(evaluation_verified["verified_files"]),
    )


def write_preflight(preflight: PromotionPreflight, output_path: Path) -> Path:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "policy_version": POLICY_VERSION,
        "created_at": _now(),
        **asdict(preflight),
        "promotion_executed": False,
        "production_embeddings_changed": False,
        "production_summary_changed": False,
        "official_attendance_changed": False,
        "candidate_promoted": False,
        "exact_next_step": (
            "Run the explicit promotion command with the exact family ID only after reviewing this preflight."
        ),
    }
    _write_json(Path(output_path), payload)
    return Path(output_path).resolve()


def _copy_verified(source: Path, target: Path, expected_hash: str | None = None) -> str:
    source = Path(source)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise Phase12OPromotionError(f"Refusing to overwrite prepared evidence: {target}")
    shutil.copy2(source, target)
    actual = _sha256_file(target)
    source_hash = _sha256_file(source)
    if actual != source_hash:
        target.unlink(missing_ok=True)
        raise Phase12OPromotionError(f"Copied file hash mismatch: {target}")
    if expected_hash and actual != expected_hash:
        target.unlink(missing_ok=True)
        raise Phase12OPromotionError(
            f"Copied file does not match expected hash: {target}; expected {expected_hash}, found {actual}"
        )
    return actual


def _atomic_replace_from(source: Path, target: Path, suffix: str) -> None:
    source = Path(source)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{suffix}-{os.getpid()}")
    if temporary.exists():
        raise Phase12OPromotionError(f"Stale atomic replacement file exists: {temporary}")
    with source.open("rb") as source_handle, temporary.open("wb") as target_handle:
        shutil.copyfileobj(source_handle, target_handle, length=1024 * 1024)
        target_handle.flush()
        os.fsync(target_handle.fileno())
    shutil.copystat(source, temporary)
    if _sha256_file(temporary) != _sha256_file(source):
        temporary.unlink(missing_ok=True)
        raise Phase12OPromotionError(f"Atomic replacement copy hash mismatch: {target}")
    os.replace(temporary, target)


def _snapshot_hashes(root: Path, excluded_names: Iterable[str] = ()) -> dict[str, str]:
    root = Path(root).resolve()
    excluded = set(excluded_names)
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in excluded:
            continue
        hashes[path.relative_to(root).as_posix()] = _sha256_file(path)
    return hashes


def _promotion_pointer_path(inputs: PromotionInputs) -> Path:
    return inputs.repo_root / "models" / "current_embedding_version.json"


def _promotion_history_root(inputs: PromotionInputs) -> Path:
    return inputs.repo_root / "models" / "promotion_history"


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not Path(path).is_file():
        return None
    return _load_json(path, "JSON file")


def _verify_existing_promoted_state(
    inputs: PromotionInputs,
    preflight: PromotionPreflight,
    promotion_dir: Path,
    contract: PromotionContract,
) -> PromotionResult | None:
    record_path = promotion_dir / "promotion_record.json"
    pointer_path = _promotion_pointer_path(inputs)
    if not promotion_dir.exists():
        return None
    if not record_path.is_file() or not pointer_path.is_file():
        raise Phase12OPromotionError(
            f"Promotion directory exists without a complete promoted state: {promotion_dir}"
        )
    record = _load_json(record_path, "promotion record")
    pointer = _load_json(pointer_path, "current embedding pointer")
    if _text(record.get("status")) != "promoted":
        raise Phase12OPromotionError("Existing promotion record is not promoted")
    if _text(record.get("promotion_id")) != preflight.promotion_id:
        raise Phase12OPromotionError("Existing promotion record ID mismatch")
    if _text(pointer.get("promotion_id")) != preflight.promotion_id:
        raise Phase12OPromotionError("Current embedding pointer references a different promotion")
    if _text(pointer.get("status")) != "promoted":
        raise Phase12OPromotionError("Current embedding pointer is not in promoted state")
    if _sha256_file(inputs.production_embeddings) != contract.candidate_embeddings_sha256:
        raise Phase12OPromotionError("Existing promoted production embeddings do not match the candidate")
    if _sha256_file(inputs.production_summary) != contract.candidate_summary_sha256:
        raise Phase12OPromotionError("Existing promoted production summary does not match the candidate")
    return PromotionResult(
        promotion_id=preflight.promotion_id,
        promotion_dir=promotion_dir.resolve(),
        family_id=contract.family_id,
        candidate_variant_id=contract.candidate_variant_id,
        production_embeddings_sha256=contract.candidate_embeddings_sha256,
        production_summary_sha256=contract.candidate_summary_sha256,
        rollback_command=_text(record.get("rollback_command")),
        idempotent_reuse=True,
    )


def promote_candidate_family(
    inputs: PromotionInputs,
    *,
    confirm_family_id: str,
    contract: PromotionContract = DEFAULT_CONTRACT,
) -> PromotionResult:
    if _text(confirm_family_id) != contract.family_id:
        raise Phase12OPromotionError(
            f"Explicit promotion confirmation must exactly equal {contract.family_id}"
        )
    promotion_root = _promotion_history_root(inputs)
    deterministic_id = expected_promotion_id(contract)
    promotion_dir = promotion_root / deterministic_id
    if (
        promotion_dir.exists()
        and _sha256_file(inputs.production_embeddings) == contract.candidate_embeddings_sha256
        and _sha256_file(inputs.production_summary) == contract.candidate_summary_sha256
    ):
        return verify_promoted_state(
            repo_root=inputs.repo_root,
            promotion_dir=promotion_dir,
            contract=contract,
        )
    preflight = verify_promotion_evidence(inputs, contract)
    if preflight.promotion_id != deterministic_id:
        raise Phase12OPromotionError("Promotion ID does not match the frozen evidence contract")
    pointer_path = _promotion_pointer_path(inputs)
    if pointer_path.exists():
        raise Phase12OPromotionError(
            "A current embedding version pointer already exists; refusing an ambiguous first promotion"
        )
    promotion_root.mkdir(parents=True, exist_ok=True)
    promotion_dir.mkdir(parents=False, exist_ok=False)
    backup_dir = promotion_dir / "production_before_promotion"
    evidence_dir = promotion_dir / "promotion_evidence"
    backup_dir.mkdir(parents=True, exist_ok=False)
    evidence_dir.mkdir(parents=True, exist_ok=False)
    candidate_dir = inputs.family_dir / "variants" / "full_candidate"
    candidate_embeddings = candidate_dir / "student_embeddings.pkl"
    candidate_summary = candidate_dir / "embedding_summary.csv"
    backup_embeddings = backup_dir / "student_embeddings.pkl"
    backup_summary = backup_dir / "embedding_summary.csv"
    transaction_path = promotion_dir / "transaction_state.json"
    rollback_command = (
        f'& "{inputs.repo_root / "scripts" / "run_phase_1_2o_explicit_promotion.ps1"}" '
        f'-Rollback -PromotionDir "{promotion_dir}" '
        f'-ConfirmRollback "{preflight.promotion_id}"'
    )
    try:
        _copy_verified(
            inputs.production_embeddings,
            backup_embeddings,
            contract.production_embeddings_sha256,
        )
        _copy_verified(
            inputs.production_summary,
            backup_summary,
            contract.production_summary_sha256,
        )
        evidence_files = {
            "phase_1_2n_output_manifest.json": inputs.shadow_dir / "output_manifest.json",
            "phase_1_2n_shadow_summary.json": inputs.shadow_dir / "shadow_run_summary.json",
            "phase_1_2n_source_freeze.json": (
                inputs.shadow_dir / "frozen_inputs" / "phase_1_2n_mon_p4_source_freeze.json"
            ),
            "phase_1_2n_evaluation_output_manifest.json": (
                inputs.evaluation_dir / "output_manifest.json"
            ),
            "phase_1_2n_evaluation_summary.json": (
                inputs.evaluation_dir / "mon_p4_evaluation_summary.json"
            ),
            "phase_1_2n_reviewed_exception_tracks.csv": (
                inputs.evaluation_dir / "reviewed_exception_tracks.csv"
            ),
            "submitted_blind_review_labels.csv": inputs.labels_path,
            "family_manifest.json": inputs.family_dir / "family_manifest.json",
            "candidate_version_manifest.json": candidate_dir / "version_manifest.json",
        }
        for name, source in evidence_files.items():
            _copy_verified(source, evidence_dir / name)
        plan = {
            "schema_version": SCHEMA_VERSION,
            "phase": PHASE,
            "policy_version": POLICY_VERSION,
            "promotion_id": preflight.promotion_id,
            "created_at": _now(),
            "family_id": contract.family_id,
            "candidate_variant_id": contract.candidate_variant_id,
            "candidate_embeddings_path": str(candidate_embeddings.resolve()),
            "candidate_summary_path": str(candidate_summary.resolve()),
            "production_embeddings_path": str(inputs.production_embeddings.resolve()),
            "production_summary_path": str(inputs.production_summary.resolve()),
            "parent_embeddings_sha256": contract.production_embeddings_sha256,
            "parent_summary_sha256": contract.production_summary_sha256,
            "candidate_embeddings_sha256": contract.candidate_embeddings_sha256,
            "candidate_summary_sha256": contract.candidate_summary_sha256,
            "evidence_fingerprint_sha256": preflight.evidence_fingerprint_sha256,
            "evaluation_decision": contract.evaluation_decision,
            "rollback_command": rollback_command,
            "attendance_changes_allowed": False,
            "recognition_execution_allowed": False,
        }
        _write_json(promotion_dir / "promotion_plan.json", plan)
        _write_json(
            transaction_path,
            {
                "schema_version": SCHEMA_VERSION,
                "promotion_id": preflight.promotion_id,
                "status": "prepared",
                "updated_at": _now(),
            },
        )
        _atomic_replace_from(candidate_embeddings, inputs.production_embeddings, "promoting")
        _atomic_replace_from(candidate_summary, inputs.production_summary, "promoting")
        if _sha256_file(inputs.production_embeddings) != contract.candidate_embeddings_sha256:
            raise Phase12OPromotionError("Post-promotion production embedding verification failed")
        if _sha256_file(inputs.production_summary) != contract.candidate_summary_sha256:
            raise Phase12OPromotionError("Post-promotion production summary verification failed")
        pointer = {
            "schema_version": SCHEMA_VERSION,
            "policy_version": POLICY_VERSION,
            "status": "promoted",
            "promotion_id": preflight.promotion_id,
            "family_id": contract.family_id,
            "variant_id": contract.candidate_variant_id,
            "promoted_at": _now(),
            "production_embeddings_sha256": contract.candidate_embeddings_sha256,
            "production_summary_sha256": contract.candidate_summary_sha256,
            "parent_embeddings_sha256": contract.production_embeddings_sha256,
            "parent_summary_sha256": contract.production_summary_sha256,
            "promotion_record": str((promotion_dir / "promotion_record.json").resolve()),
            "rollback_command": rollback_command,
        }
        _write_json(pointer_path, pointer)
        promotion_record = {
            "schema_version": SCHEMA_VERSION,
            "phase": PHASE,
            "policy_version": POLICY_VERSION,
            "promotion_id": preflight.promotion_id,
            "status": "promoted",
            "promoted_at": _now(),
            "family_id": contract.family_id,
            "candidate_variant_id": contract.candidate_variant_id,
            "evaluation_id": contract.evaluation_id,
            "evaluation_decision": contract.evaluation_decision,
            "reviewed_exception_tracks": contract.reviewed_exception_tracks,
            "candidate_correct_recoveries": contract.candidate_correct_recoveries,
            "candidate_false_identities_or_unsafe_accepts": 0,
            "candidate_lost_correct_production_accepts": 0,
            "unverifiable_exceptions": 0,
            "parent_embeddings_sha256": contract.production_embeddings_sha256,
            "parent_summary_sha256": contract.production_summary_sha256,
            "production_embeddings_sha256": contract.candidate_embeddings_sha256,
            "production_summary_sha256": contract.candidate_summary_sha256,
            "backup_embeddings_sha256": _sha256_file(backup_embeddings),
            "backup_summary_sha256": _sha256_file(backup_summary),
            "evidence_fingerprint_sha256": preflight.evidence_fingerprint_sha256,
            "current_pointer_sha256": _sha256_file(pointer_path),
            "rollback_command": rollback_command,
            "official_attendance_changed": False,
            "recognition_executed": False,
            "datasets_changed": False,
        }
        _write_json(promotion_dir / "promotion_record.json", promotion_record)
        manifest_payload = {
            "schema_version": SCHEMA_VERSION,
            "phase": PHASE,
            "policy_version": POLICY_VERSION,
            "promotion_id": preflight.promotion_id,
            "files_sha256": _snapshot_hashes(
                promotion_dir,
                excluded_names={
                    "promotion_output_manifest.json",
                    "transaction_state.json",
                    "rollback_record.json",
                    "rollback_output_manifest.json",
                },
            ),
        }
        _write_json(promotion_dir / "promotion_output_manifest.json", manifest_payload)
        verify_output_manifest(promotion_dir, promotion_dir / "promotion_output_manifest.json")
        _write_json(
            transaction_path,
            {
                "schema_version": SCHEMA_VERSION,
                "promotion_id": preflight.promotion_id,
                "status": "promoted",
                "updated_at": _now(),
                "automatic_restoration_required": False,
            },
        )
    except Exception as promotion_error:
        restoration_errors: list[str] = []
        if backup_embeddings.is_file() and backup_summary.is_file():
            for backup, target, label in (
                (backup_embeddings, inputs.production_embeddings, "embeddings"),
                (backup_summary, inputs.production_summary, "summary"),
            ):
                try:
                    _atomic_replace_from(backup, target, "restoring")
                except Exception as restore_error:  # pragma: no cover - hard failure path
                    restoration_errors.append(f"{label}: {restore_error}")
        pointer_path.unlink(missing_ok=True)
        try:
            _write_json(
                transaction_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "promotion_id": preflight.promotion_id,
                    "status": (
                        "failed_restoration_incomplete" if restoration_errors else "failed_restored"
                    ),
                    "updated_at": _now(),
                    "error": str(promotion_error),
                    "restoration_errors": restoration_errors,
                },
            )
        except Exception:
            pass
        if restoration_errors:
            raise Phase12OPromotionError(
                "Promotion failed and automatic restoration was incomplete: "
                + "; ".join(restoration_errors)
            ) from promotion_error
        if (
            _sha256_file(inputs.production_embeddings) != contract.production_embeddings_sha256
            or _sha256_file(inputs.production_summary) != contract.production_summary_sha256
        ):
            raise Phase12OPromotionError(
                "Promotion failed and the original production hashes were not restored"
            ) from promotion_error
        raise Phase12OPromotionError(
            f"Promotion failed; original production files were restored: {promotion_error}"
        ) from promotion_error
    return PromotionResult(
        promotion_id=preflight.promotion_id,
        promotion_dir=promotion_dir.resolve(),
        family_id=contract.family_id,
        candidate_variant_id=contract.candidate_variant_id,
        production_embeddings_sha256=contract.candidate_embeddings_sha256,
        production_summary_sha256=contract.candidate_summary_sha256,
        rollback_command=rollback_command,
        idempotent_reuse=False,
    )


def rollback_promotion(
    *,
    repo_root: Path,
    promotion_dir: Path,
    confirm_promotion_id: str,
    contract: PromotionContract = DEFAULT_CONTRACT,
) -> RollbackResult:
    repo_root = Path(repo_root).resolve()
    promotion_dir = Path(promotion_dir).resolve()
    promotion_id = promotion_dir.name
    if _text(confirm_promotion_id) != promotion_id:
        raise Phase12OPromotionError(
            f"Explicit rollback confirmation must exactly equal {promotion_id}"
        )
    record = _load_json(promotion_dir / "promotion_record.json", "promotion record")
    if _text(record.get("status")) != "promoted":
        raise Phase12OPromotionError("Rollback refused: promotion record is not promoted")
    if _text(record.get("promotion_id")) != promotion_id:
        raise Phase12OPromotionError("Rollback refused: promotion ID mismatch")
    pointer_path = repo_root / "models" / "current_embedding_version.json"
    production_embeddings = repo_root / "models" / "student_embeddings.pkl"
    production_summary = repo_root / "models" / "embedding_summary.csv"
    rollback_record_path = promotion_dir / "rollback_record.json"
    if rollback_record_path.is_file():
        rollback_record = _load_json(rollback_record_path, "rollback record")
        if (
            _text(rollback_record.get("status")) == "rolled_back"
            and _sha256_file(production_embeddings) == contract.production_embeddings_sha256
            and _sha256_file(production_summary) == contract.production_summary_sha256
        ):
            return RollbackResult(
                promotion_id=promotion_id,
                promotion_dir=promotion_dir,
                restored_embeddings_sha256=contract.production_embeddings_sha256,
                restored_summary_sha256=contract.production_summary_sha256,
                idempotent_reuse=True,
            )
        raise Phase12OPromotionError("Rollback record exists but production is not in the recorded state")
    pointer = _load_json(pointer_path, "current embedding pointer")
    if _text(pointer.get("promotion_id")) != promotion_id or _text(pointer.get("status")) != "promoted":
        raise Phase12OPromotionError("Rollback refused: current pointer does not match this promotion")
    if _sha256_file(production_embeddings) != contract.candidate_embeddings_sha256:
        raise Phase12OPromotionError("Rollback refused: production embeddings no longer match the promoted candidate")
    if _sha256_file(production_summary) != contract.candidate_summary_sha256:
        raise Phase12OPromotionError("Rollback refused: production summary no longer matches the promoted candidate")
    backup_dir = promotion_dir / "production_before_promotion"
    backup_embeddings = backup_dir / "student_embeddings.pkl"
    backup_summary = backup_dir / "embedding_summary.csv"
    _assert_hash(
        backup_embeddings,
        contract.production_embeddings_sha256,
        "Rollback embedding backup",
    )
    _assert_hash(
        backup_summary,
        contract.production_summary_sha256,
        "Rollback summary backup",
    )
    family_dir = repo_root / "models" / "versions" / contract.family_id / "variants" / "full_candidate"
    candidate_embeddings = family_dir / "student_embeddings.pkl"
    candidate_summary = family_dir / "embedding_summary.csv"
    _assert_hash(candidate_embeddings, contract.candidate_embeddings_sha256, "Candidate embeddings")
    _assert_hash(candidate_summary, contract.candidate_summary_sha256, "Candidate summary")
    try:
        _atomic_replace_from(backup_embeddings, production_embeddings, "rolling-back")
        _atomic_replace_from(backup_summary, production_summary, "rolling-back")
    except Exception as rollback_error:
        restoration_errors: list[str] = []
        for source, target, label in (
            (candidate_embeddings, production_embeddings, "embeddings"),
            (candidate_summary, production_summary, "summary"),
        ):
            try:
                _atomic_replace_from(source, target, "rollback-recovery")
            except Exception as restore_error:  # pragma: no cover - hard failure path
                restoration_errors.append(f"{label}: {restore_error}")
        if restoration_errors:
            raise Phase12OPromotionError(
                "Rollback failed and the promoted candidate could not be fully restored: "
                + "; ".join(restoration_errors)
            ) from rollback_error
        raise Phase12OPromotionError(
            "Rollback failed; the promoted candidate was restored"
        ) from rollback_error
    if _sha256_file(production_embeddings) != contract.production_embeddings_sha256:
        raise Phase12OPromotionError("Rollback embedding verification failed")
    if _sha256_file(production_summary) != contract.production_summary_sha256:
        raise Phase12OPromotionError("Rollback summary verification failed")
    rollback_record = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "policy_version": POLICY_VERSION,
        "promotion_id": promotion_id,
        "status": "rolled_back",
        "rolled_back_at": _now(),
        "restored_embeddings_sha256": contract.production_embeddings_sha256,
        "restored_summary_sha256": contract.production_summary_sha256,
        "promoted_embeddings_sha256": contract.candidate_embeddings_sha256,
        "promoted_summary_sha256": contract.candidate_summary_sha256,
        "official_attendance_changed": False,
        "recognition_executed": False,
        "datasets_changed": False,
    }
    _write_json(rollback_record_path, rollback_record)
    pointer_after = {
        "schema_version": SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "status": "rolled_back_to_parent",
        "last_promotion_id": promotion_id,
        "rolled_back_at": _now(),
        "current_production_embeddings_sha256": contract.production_embeddings_sha256,
        "current_production_summary_sha256": contract.production_summary_sha256,
        "rollback_record": str(rollback_record_path.resolve()),
        "candidate_family_preserved": contract.family_id,
    }
    _write_json(pointer_path, pointer_after)
    rollback_manifest = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "policy_version": POLICY_VERSION,
        "promotion_id": promotion_id,
        "files_sha256": {
            "rollback_record.json": _sha256_file(rollback_record_path),
            "current_embedding_version.json": _sha256_file(pointer_path),
        },
        "external_paths": {
            "current_embedding_version.json": str(pointer_path.resolve()),
        },
    }
    _write_json(promotion_dir / "rollback_output_manifest.json", rollback_manifest)
    return RollbackResult(
        promotion_id=promotion_id,
        promotion_dir=promotion_dir,
        restored_embeddings_sha256=contract.production_embeddings_sha256,
        restored_summary_sha256=contract.production_summary_sha256,
        idempotent_reuse=False,
    )


def verify_promoted_state(
    *,
    repo_root: Path,
    promotion_dir: Path,
    contract: PromotionContract = DEFAULT_CONTRACT,
) -> PromotionResult:
    repo_root = Path(repo_root).resolve()
    promotion_dir = Path(promotion_dir).resolve()
    record = _load_json(promotion_dir / "promotion_record.json", "promotion record")
    pointer_path = repo_root / "models" / "current_embedding_version.json"
    pointer = _load_json(pointer_path, "current embedding pointer")
    production_embeddings = repo_root / "models" / "student_embeddings.pkl"
    production_summary = repo_root / "models" / "embedding_summary.csv"
    if _text(record.get("status")) != "promoted":
        raise Phase12OPromotionError("Promotion record is not in promoted state")
    promotion_id = _text(record.get("promotion_id"))
    if not promotion_id or promotion_id != promotion_dir.name:
        raise Phase12OPromotionError("Promotion directory and promotion record IDs do not match")
    if _text(record.get("family_id")) != contract.family_id:
        raise Phase12OPromotionError("Promotion record family ID changed")
    if _text(record.get("candidate_variant_id")) != contract.candidate_variant_id:
        raise Phase12OPromotionError("Promotion record candidate variant changed")
    if _text(pointer.get("status")) != "promoted":
        raise Phase12OPromotionError("Current embedding pointer is not promoted")
    if _text(pointer.get("promotion_id")) != promotion_id:
        raise Phase12OPromotionError("Current embedding pointer references another promotion")
    if _text(pointer.get("family_id")) != contract.family_id:
        raise Phase12OPromotionError("Current embedding pointer family ID changed")
    _assert_hash(
        production_embeddings,
        contract.candidate_embeddings_sha256,
        "Promoted production embeddings",
    )
    _assert_hash(
        production_summary,
        contract.candidate_summary_sha256,
        "Promoted production summary",
    )
    _assert_hash(
        promotion_dir / "production_before_promotion" / "student_embeddings.pkl",
        contract.production_embeddings_sha256,
        "Pre-promotion embedding backup",
    )
    _assert_hash(
        promotion_dir / "production_before_promotion" / "embedding_summary.csv",
        contract.production_summary_sha256,
        "Pre-promotion summary backup",
    )
    verify_output_manifest(promotion_dir, promotion_dir / "promotion_output_manifest.json")
    return PromotionResult(
        promotion_id=promotion_id,
        promotion_dir=promotion_dir,
        family_id=contract.family_id,
        candidate_variant_id=contract.candidate_variant_id,
        production_embeddings_sha256=contract.candidate_embeddings_sha256,
        production_summary_sha256=contract.candidate_summary_sha256,
        rollback_command=_text(record.get("rollback_command")),
        idempotent_reuse=True,
    )
