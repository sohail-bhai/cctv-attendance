from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .processing_integration import (
    EXPECTED_CAMERA_STEMS,
    EXPECTED_CHECKPOINT_IDS,
    FIXED_POLICY,
    OFFICIAL_RECOGNITION_AUTHORITY,
    POLICY_VERSION as PROCESSING_POLICY_VERSION,
)


POLICY_VERSION = "product-phase-2l-a-current-policy-stabilization-v1"
FRONTEND_STATUS_CONTRACT_VERSION = "product-phase-2l-frontend-status-contract-v1"
OUTPUT_SCHEMA_VERSION = 1
SESSION_ID = "2026-06-22__B51__P4__CVO"
MIXED_TRACKLET_ID = "CP1-cam5-back-TRK00005"
MISSING_ENROLLMENT_ROLL = "2401100CSE0268"
PHASE_2K_C1_RUN_ID = "retrospective-audit-efe4cd29aa22738a079e"
PHASE_2K_C1_MANIFEST_SHA256 = "d9498d2268b99ea0acc1e6f412bd84c660004e3247032e1da87ca1ee6eca113e"
PHASE_2K_C1_RECOMMENDATION = "retain_current_policy_with_blockers"
EXPECTED_FAMILY = "embfam-274b5207b8b71294ff75"
EXPECTED_VARIANT = "embfam-274b5207b8b71294ff75-d"
EXPECTED_EMBEDDING_SHA256 = "f088d827adc548ee95f46566d758fd71fc304d042c43f1ecffc6526b60bcd832"
EXPECTED_SUMMARY_SHA256 = "63885588c374c37f4da9bf85294f240bdf0f28cb585e77b894ab516139ca46ae"
EXPECTED_POINTER_SHA256 = "7999b8ccf787dca9b8fb862f4e53ce7c1102eb729a3732c82fee76f3ba3ee05e"
EXPECTED_CANDIDATE_VERSION_MANIFEST_SHA256 = "341e0211e4d5ae669b2f991825a1d7ff724b93dba8031f1c8a621d88a687fcd5"
EXPECTED_OFFICIAL_REPORT_SHA256 = "0211ac1433831aceb4176ee199f64bf962dbc784765a99a00bed74c7b38d51e3"
EXPECTED_CANDIDATE_REPORT_SHA256 = "e28dc0278abfad034bd78bb576620a50f90100461f23b5e0b920e1e65e39c57f"
EXPECTED_CANDIDATE_MANIFEST_SHA256 = "e941fbbcdf70511a13ae49b773979f54f921f1b121a27128cfe84b312e0ffd17"
EXPECTED_REVIEW_REGISTRY_FINGERPRINT = "b2afa12c10104025f31c76f02ea91b9c33720cccb09773ca5639da39c1577a69"
EXPECTED_OFFICIAL_COUNTS = {
    "Present": 6,
    "Needs Review": 4,
    "Unconfirmed": 16,
    "Missing Enrollment": 1,
    "Absent": 0,
    "Unknown": 0,
    "Total": 27,
    "Unresolved": 21,
}
EXPECTED_CANDIDATE_COUNTS = {
    "Present": 2,
    "Needs Review": 8,
    "Unconfirmed": 16,
    "Missing Enrollment": 1,
    "Absent": 0,
    "Unknown": 0,
    "Total": 27,
    "Unresolved": 25,
}

PROTECTED_EXPECTED = {
    "data/attendance_status.json": "46fd2df55cc3f61c5fa03c893937eb2828b9cc2af064ea9e427276b79a9d0b46",
    "data/job_runtime.json": "5a23f09654d91fac09804cd97c2fa114bf4956a914f2e02ce2b7090f730141ed",
    "data/role_users.json": "77125140004294f2834fc869fd590e167d8ab78cff5dbeda1c074ab76c177517",
    "data/student_faculty_map.json": "0eca63ea58d12a73d3bd11e620b74be733a3219c9244e8a4a267d5b02acd3f91",
    "data/manual_overrides.json": "e9c6bd35c23c353795edb5d3c02e808793d7cbd90c7b0a87fdfc686f8fd5cdcf",
    "data/review_evidence_registry.json": "3b04e9aecfdfb1f64346c4a0e709fd9c36d7c56545bf816d6641b4c2a2e80841",
    "models/student_embeddings.pkl": EXPECTED_EMBEDDING_SHA256,
    "models/embedding_summary.csv": EXPECTED_SUMMARY_SHA256,
    "models/current_embedding_version.json": EXPECTED_POINTER_SHA256,
    "timetable_b51_2026_2027.csv": "10bbd578a859fbd4fa228c4eb4f1f92e7de5b92a1c4236d71a00337eecac9c57",
}

DOCUMENTATION_PATHS = (
    "docs/PHASE_2L_OPERATOR_GUIDE.md",
    "docs/PHASE_2L_PROFESSOR_DEMO_RUNBOOK.md",
    "docs/PHASE_2L_RECOVERY_AND_ROLLBACK.md",
)

ROLLBACK_PATHS = {
    "phase_2i_state_backup": "data/state_backups/phase_2i_before_2026-06-22__B51__P4__CVO_20260720_143351.json",
    "phase_2j_state_backup": "data/state_backups/phase_2j_before_revision_repair_20260721_043229.json",
    "production_embedding_backup": "models/promotion_history/promotion-7cc01070e9bc644380c5/production_before_promotion/student_embeddings.pkl",
    "production_summary_backup": "models/promotion_history/promotion-7cc01070e9bc644380c5/production_before_promotion/embedding_summary.csv",
    "promotion_record": "models/promotion_history/promotion-7cc01070e9bc644380c5/promotion_record.json",
}

ROLLBACK_HASHES = {
    "phase_2i_state_backup": "9fcec175ac46e19afb80492c61d6ffc2130b831d177819879ac84b3960e37a4a",
    "phase_2j_state_backup": "f3119b69f269365fd5987c3b166a17995fa24fbca06bfe0e1278a8b8ba620b3a",
    "production_embedding_backup": "c32ed31df10b7b9b43b8f19977a2adf0a82fb8e7a71f3e8bceb42c0565fedd49",
    "production_summary_backup": "0df35c3bb9207e191aa49dad5536b8378e812d56463491bd7203885dca5ae9ee",
    "promotion_record": "99ff588e0992e207b94dfeaf287c4a0c6c73be93f0fc2c2a7d126121232a4957",
}

REQUIRED_OUTPUT_FILES = (
    "stabilization_policy.json",
    "release_candidate_freeze.json",
    "frontend_contract_summary.json",
    "demo_readiness.json",
    "rollback_inventory.json",
    "documentation_inventory.json",
    "validation_summary.json",
    "no_recognition_declaration.json",
    "no_activation_declaration.json",
    "source_manifest.json",
    "evaluation_summary.json",
    "immutable_manifest.json",
)

REQUIRED_EXTERNAL_VALIDATIONS = (
    "python_compilation",
    "phase_2l_backend_preflight_tests",
    "frontend_logic_tests",
    "frontend_production_build",
    "phase_2k_c1_tests",
    "phase_2k_b_tests",
    "phase_2k_a_tests",
    "phase_2j_tests",
    "phase_2i_tests",
    "full_python_suite",
)


class StabilizationError(RuntimeError):
    """Raised when the Phase 2L release candidate cannot be proven safe."""


@dataclass(frozen=True)
class StabilizationContract:
    protected_hashes: Mapping[str, str] = field(default_factory=lambda: dict(PROTECTED_EXPECTED))
    phase_2k_c1_manifest_sha256: str = PHASE_2K_C1_MANIFEST_SHA256
    candidate_manifest_sha256: str = EXPECTED_CANDIDATE_MANIFEST_SHA256
    official_report_sha256: str = EXPECTED_OFFICIAL_REPORT_SHA256
    candidate_report_sha256: str = EXPECTED_CANDIDATE_REPORT_SHA256
    candidate_version_manifest_sha256: str = EXPECTED_CANDIDATE_VERSION_MANIFEST_SHA256
    expected_embedding_sha256: str = EXPECTED_EMBEDDING_SHA256
    expected_summary_sha256: str = EXPECTED_SUMMARY_SHA256
    expected_pointer_sha256: str = EXPECTED_POINTER_SHA256
    rollback_hashes: Mapping[str, str] = field(default_factory=lambda: dict(ROLLBACK_HASHES))


DEFAULT_CONTRACT = StabilizationContract()


@dataclass(frozen=True)
class StabilizationPreflight:
    repo_root: Path
    output_root: Path
    run_id: str
    run_fingerprint_sha256: str
    run_payload: dict[str, Any]
    protected_hashes: dict[str, str]
    production: dict[str, Any]
    policy: dict[str, Any]
    official_counts: dict[str, int]
    candidate_counts: dict[str, int]
    roster: dict[str, Any]
    review_registry: dict[str, Any]
    rollback_inventory: tuple[dict[str, Any], ...]
    documentation_inventory: tuple[dict[str, Any], ...]
    source_manifest: tuple[dict[str, Any], ...]
    test_validation: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_json_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StabilizationError(f"invalid_json:{path}") from exc
    if not isinstance(payload, dict):
        raise StabilizationError(f"json_object_required:{path}")
    return payload


def _verify_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file():
        raise StabilizationError(f"manifest_missing:{path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise StabilizationError(f"manifest_tampered:{path}:{actual}")
    payload = _load_json(path)
    files = payload.get("files") or {}
    if not isinstance(files, dict):
        raise StabilizationError(f"manifest_file_map_invalid:{path}")
    for relative, expected in files.items():
        if not isinstance(expected, dict):
            raise StabilizationError(f"manifest_entry_invalid:{path}:{relative}")
        target = path.parent / relative
        if not target.is_file():
            raise StabilizationError(f"manifest_file_missing:{target}")
        if sha256_file(target) != expected.get("sha256") or target.stat().st_size != expected.get("size_bytes"):
            raise StabilizationError(f"manifest_file_tampered:{target}")
    return payload


def _status_category(value: Any) -> str:
    status = str(value or "").strip().lower()
    if status.startswith("present") or status == "yes":
        return "Present"
    if "missing enrollment" in status:
        return "Missing Enrollment"
    if "unconfirmed" in status:
        return "Unconfirmed"
    if "review" in status:
        return "Needs Review"
    if status == "absent" or status == "no":
        return "Absent"
    return "Unknown"


def _report_counts(path: Path) -> dict[str, int]:
    counts = {label: 0 for label in ("Present", "Needs Review", "Unconfirmed", "Missing Enrollment", "Absent", "Unknown")}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        counts[_status_category(row.get("Status") or row.get("Final_Status") or row.get("Present"))] += 1
    counts["Total"] = len(rows)
    counts["Unresolved"] = counts["Needs Review"] + counts["Unconfirmed"] + counts["Missing Enrollment"] + counts["Unknown"]
    return counts


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _roster_rolls(student_map: Mapping[str, Any]) -> tuple[str, ...]:
    rows = (student_map.get("subject_students") or {}).get("CVO") or []
    rolls = tuple(str(row.get("roll") or "").strip().upper() for row in rows if str(row.get("roll") or "").strip())
    if len(rolls) != len(set(rolls)):
        raise StabilizationError("roster_duplicate_roll")
    return rolls


def validate_roster_and_enrollment(roster_rolls: Sequence[str], enrollment_rolls: Sequence[str]) -> dict[str, Any]:
    roster = tuple(roster_rolls)
    enrolled = set(enrollment_rolls)
    missing = sorted(set(roster) - enrolled)
    if len(roster) != 27:
        raise StabilizationError(f"roster_count_mismatch:{len(roster)}")
    if "24011CSEAI0110" not in roster or "2401100CSE0110" in roster:
        raise StabilizationError("roster_identity_constraint_mismatch:AI0110")
    if "24011CSEAI0061" in roster or "2401100CSE0237" not in roster:
        raise StabilizationError("roster_identity_constraint_mismatch:CVO")
    if missing != [MISSING_ENROLLMENT_ROLL]:
        raise StabilizationError(f"missing_enrollment_mismatch:{missing}")
    return {"count": len(roster), "missing_enrollment": missing, "identity_constraints_verified": True}


def validate_review_registry(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("reviewed_pairs") != 31 or payload.get("accepted_pairs") != 30 or payload.get("mixed_track_pairs") != 1:
        raise StabilizationError("review_registry_count_mismatch")
    if payload.get("registry_sha256") != EXPECTED_REVIEW_REGISTRY_FINGERPRINT:
        raise StabilizationError("review_registry_fingerprint_mismatch")
    evidence = payload.get("evidence") or []
    mixed = [row for row in evidence if row.get("tracklet_id") == MIXED_TRACKLET_ID]
    if len(mixed) != 1:
        raise StabilizationError("mixed_track_missing_from_quarantine")
    row = mixed[0]
    if row.get("review_status") != "mixed_track" or not row.get("mixed_track_quarantine") or row.get("accepted_for_carry_forward"):
        raise StabilizationError("mixed_track_quarantine_mismatch")
    return {
        "registry_id": payload.get("registry_id"),
        "registry_fingerprint_sha256": payload.get("registry_sha256"),
        "reviewed_pairs": 31,
        "accepted_pairs": 30,
        "mixed_track_pairs": 1,
        "mixed_tracklet_id": MIXED_TRACKLET_ID,
        "mixed_quarantined": True,
    }


def validate_recommendation(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("recommendation") != PHASE_2K_C1_RECOMMENDATION:
        raise StabilizationError("phase_2k_c1_recommendation_mismatch")
    if payload.get("independent_generalization_claimed") is not False:
        raise StabilizationError("independent_generalization_claim_present")
    if payload.get("guarded_recovery_must_remain_review_only") is not True:
        raise StabilizationError("guarded_recovery_review_only_claim_missing")
    return dict(payload)


def _run_command(command: Sequence[str], cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(
        list(command),
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        tail = "\n".join((completed.stdout + "\n" + completed.stderr).splitlines()[-30:])
        raise StabilizationError(f"validation_command_failed:{' '.join(command)}\n{tail}")
    return {"passed": True, "command": " ".join(command), "return_code": 0}


def run_release_contract_tests(repo_root: Path) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    frontend_tests = sorted((repo / "frontend/tests").glob("*.test.mjs"))
    frontend_command = ["node", "--test", *[str(path.relative_to(repo / "frontend")) for path in frontend_tests]]
    frontend = _run_command(frontend_command, repo / "frontend")
    backend_modules = [
        "tests.test_product_phase_2l_stabilization",
        "tests.test_product_phase_2d_reports_review",
        "tests.test_product_phase_2e_hod_control",
        "tests.test_product_phase_2i_frontend_contract",
        "tests.test_product_phase_2j_frontend_contract",
    ]
    backend_command = [sys.executable, "-m", "unittest", *backend_modules]
    backend = _run_command(backend_command, repo)
    return {
        "frontend_status_contract_tests": {**frontend, "test_file_count": len(frontend_tests)},
        "backend_relevant_tests": {**backend, "module_count": len(backend_modules)},
    }


def _source_row(repo: Path, relative: str) -> dict[str, Any]:
    path = repo / relative
    if not path.is_file():
        raise StabilizationError(f"required_source_missing:{relative}")
    return {"path": relative.replace("\\", "/"), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def preflight(
    repo_root: Path,
    output_root: Path | None = None,
    *,
    contract: StabilizationContract = DEFAULT_CONTRACT,
    policy_override: Mapping[str, Any] | None = None,
    run_tests: bool = False,
) -> StabilizationPreflight:
    repo = Path(repo_root).resolve()

    embedding_path = repo / "models/student_embeddings.pkl"
    summary_path = repo / "models/embedding_summary.csv"
    pointer_path = repo / "models/current_embedding_version.json"
    if sha256_file(embedding_path) != contract.expected_embedding_sha256:
        raise StabilizationError("embedding_hash_mismatch")
    if sha256_file(summary_path) != contract.expected_summary_sha256:
        raise StabilizationError("summary_hash_mismatch")
    if sha256_file(pointer_path) != contract.expected_pointer_sha256:
        raise StabilizationError("production_pointer_hash_mismatch")

    protected = {relative: sha256_file(repo / relative) for relative in contract.protected_hashes}
    drift = sorted(relative for relative, expected in contract.protected_hashes.items() if protected.get(relative) != expected)
    if drift:
        raise StabilizationError(f"protected_operational_hash_drift:{','.join(drift)}")

    audit_root = repo / f"attendance_output/product_workflow/phase_2k_retrospective_audit/{PHASE_2K_C1_RUN_ID}"
    audit_manifest = _verify_manifest(audit_root / "immutable_manifest.json", contract.phase_2k_c1_manifest_sha256)
    if audit_manifest.get("run_id") != PHASE_2K_C1_RUN_ID:
        raise StabilizationError("phase_2k_c1_run_id_mismatch")
    recommendation = validate_recommendation(_load_json(audit_root / "recommendation.json"))

    pointer = _load_json(pointer_path)
    if pointer.get("family_id") != EXPECTED_FAMILY or pointer.get("variant_id") != EXPECTED_VARIANT or pointer.get("status") != "promoted":
        raise StabilizationError("production_family_or_variant_mismatch")
    if pointer.get("production_embeddings_sha256") != contract.expected_embedding_sha256 or pointer.get("production_summary_sha256") != contract.expected_summary_sha256:
        raise StabilizationError("production_pointer_payload_mismatch")

    version_manifest_path = repo / "models/promotion_history/promotion-7cc01070e9bc644380c5/promotion_evidence/candidate_version_manifest.json"
    if sha256_file(version_manifest_path) != contract.candidate_version_manifest_sha256:
        raise StabilizationError("candidate_version_manifest_hash_mismatch")
    version_manifest = _load_json(version_manifest_path)
    recognition_config = version_manifest.get("official_recognition_configuration") or {}
    if version_manifest.get("version_id") != EXPECTED_VARIANT or version_manifest.get("embedding_dimension") != 128 or version_manifest.get("embedding_records") != 354:
        raise StabilizationError("production_embedding_metadata_mismatch")
    if recognition_config.get("aggregate") != "top3":
        raise StabilizationError("production_aggregation_mismatch")

    policy = dict(policy_override or FIXED_POLICY)
    if policy.get("match_threshold") != 0.48 or policy.get("margin_threshold") != 0.08:
        raise StabilizationError("threshold_mismatch")
    if policy.get("official_recognition_authority") != OFFICIAL_RECOGNITION_AUTHORITY:
        raise StabilizationError("authority_mismatch")
    if policy.get("aggregate") != "top3" or policy.get("checkpoint_mode") != "clip-folders":
        raise StabilizationError("processing_policy_mismatch")
    if tuple(EXPECTED_CHECKPOINT_IDS) != ("CP1", "CP2", "CP3", "CP4", "CP5") or tuple(EXPECTED_CAMERA_STEMS) != ("back", "front"):
        raise StabilizationError("checkpoint_front_back_contract_mismatch")

    attendance = _load_json(repo / "data/attendance_status.json")
    entry = attendance.get(SESSION_ID)
    if not isinstance(entry, dict):
        raise StabilizationError("official_session_missing")
    if entry.get("official_recognition_authority") != "reviewed_multiframe_tracklet_evidence":
        raise StabilizationError("reviewed_authority_mismatch")
    if entry.get("automatic_recognition_authority") != OFFICIAL_RECOGNITION_AUTHORITY:
        raise StabilizationError("automatic_authority_mismatch")
    if _truthy(entry.get("guarded_recovery_automatic")):
        raise StabilizationError("guarded_recovery_accidentally_automatic")
    if _truthy(entry.get("attendance_finalized")) or _truthy(entry.get("Attendance_Finalized")):
        raise StabilizationError("finalized_state_mismatch")

    official_path = repo / f"attendance_output/product_workflow/phase_2i_authority/authority-revision-2d83c4f679f839a6c784c636/corrected_attendance_{SESSION_ID}.csv"
    if sha256_file(official_path) != contract.official_report_sha256:
        raise StabilizationError("official_report_hash_mismatch")
    official_counts = _report_counts(official_path)
    if official_counts != EXPECTED_OFFICIAL_COUNTS:
        raise StabilizationError(f"official_totals_mismatch:{official_counts}")

    candidate_root = repo / f"attendance_output/product_workflow/report_revisions/{SESSION_ID}/automatic-candidate-99cc13934144bf7e1706b8be"
    candidate_manifest = _verify_manifest(candidate_root / "revision_manifest.json", contract.candidate_manifest_sha256)
    candidate_path = candidate_root / "candidate_attendance.csv"
    if sha256_file(candidate_path) != contract.candidate_report_sha256:
        raise StabilizationError("candidate_report_hash_mismatch")
    candidate_counts = _report_counts(candidate_path)
    if candidate_counts != EXPECTED_CANDIDATE_COUNTS:
        raise StabilizationError(f"candidate_totals_mismatch:{candidate_counts}")

    student_map = _load_json(repo / "data/student_faculty_map.json")
    roster_rolls = _roster_rolls(student_map)
    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        enrolled_rolls = [
            str(row.get("Roll_Number") or "").strip().upper()
            for row in csv.DictReader(handle)
            if int(float(row.get("Faces_Used") or 0)) > 0
        ]
    roster = validate_roster_and_enrollment(roster_rolls, enrolled_rolls)

    registry = validate_review_registry(_load_json(repo / "data/review_evidence_registry.json"))

    rollback_inventory: list[dict[str, Any]] = []
    for name, relative in ROLLBACK_PATHS.items():
        path = repo / relative
        if not path.is_file():
            raise StabilizationError(f"rollback_reference_missing:{name}")
        actual_hash = sha256_file(path)
        if actual_hash != contract.rollback_hashes[name]:
            raise StabilizationError(f"rollback_reference_hash_mismatch:{name}")
        rollback_inventory.append({"name": name, "path": relative, "sha256": actual_hash, "verified": True})
    promotion_record = _load_json(repo / ROLLBACK_PATHS["promotion_record"])
    if promotion_record.get("rollback_command") != pointer.get("rollback_command"):
        raise StabilizationError("production_rollback_command_mismatch")

    documentation_inventory = tuple(_source_row(repo, relative) for relative in DOCUMENTATION_PATHS)
    source_paths = (
        "src/face_attendance/product_phase_2l_stabilization.py",
        "src/face_attendance/processing_integration.py",
        "frontend/src/data/fallback.js",
        "frontend/src/data/users.js",
        "frontend/src/utils/consistency.js",
        "frontend/src/utils/csv.js",
        "frontend/src/utils/reportWorkflow.js",
        "frontend/src/pages/Reports.jsx",
        "frontend/src/pages/ManualReview.jsx",
        "frontend/src/pages/Help.jsx",
        "frontend/src/pages/HodControl.jsx",
        "frontend/src/pages/Login.jsx",
        "frontend/src/pages/Settings.jsx",
        "frontend/tests/consistency.test.mjs",
        "frontend/tests/reportWorkflow.test.mjs",
        "frontend/tests/phase2lContract.test.mjs",
        "app.py",
        "scripts/run_product_phase_2l_stabilization_preflight.py",
        "scripts/run_product_phase_2l_stabilization_preflight.ps1",
        "tests/test_product_phase_2d_reports_review.py",
        "tests/test_product_phase_2l_stabilization.py",
    )
    source_manifest = [_source_row(repo, relative) for relative in source_paths]
    source_manifest.extend(documentation_inventory)
    source_manifest.extend({"path": relative, "sha256": digest, "size_bytes": (repo / relative).stat().st_size} for relative, digest in sorted(protected.items()))
    source_manifest.extend([
        {"path": str((audit_root / "immutable_manifest.json").relative_to(repo)).replace("\\", "/"), "sha256": contract.phase_2k_c1_manifest_sha256, "size_bytes": (audit_root / "immutable_manifest.json").stat().st_size},
        {"path": str(official_path.relative_to(repo)).replace("\\", "/"), "sha256": contract.official_report_sha256, "size_bytes": official_path.stat().st_size},
        {"path": str((candidate_root / "revision_manifest.json").relative_to(repo)).replace("\\", "/"), "sha256": contract.candidate_manifest_sha256, "size_bytes": (candidate_root / "revision_manifest.json").stat().st_size},
        {"path": str(candidate_path.relative_to(repo)).replace("\\", "/"), "sha256": contract.candidate_report_sha256, "size_bytes": candidate_path.stat().st_size},
    ])
    source_manifest = sorted({row["path"]: row for row in source_manifest}.values(), key=lambda row: row["path"])

    test_validation = run_release_contract_tests(repo) if run_tests else {
        "frontend_status_contract_tests": {"passed": False, "status": "not_run"},
        "backend_relevant_tests": {"passed": False, "status": "not_run"},
    }
    if run_tests and not all(item.get("passed") for item in test_validation.values()):
        raise StabilizationError("release_contract_tests_failed")

    policy_fingerprint = canonical_json_hash({
        "processing_policy_version": PROCESSING_POLICY_VERSION,
        "authority": OFFICIAL_RECOGNITION_AUTHORITY,
        "guarded_recovery_automatic": False,
        "checkpoint_ids": list(EXPECTED_CHECKPOINT_IDS),
        "camera_stems": list(EXPECTED_CAMERA_STEMS),
        "match_threshold": 0.48,
        "margin_threshold": 0.08,
        "aggregate": "top3",
    })
    run_payload = {
        "phase_2k_c1_immutable_manifest_sha256": contract.phase_2k_c1_manifest_sha256,
        "production_pointer_sha256": contract.expected_pointer_sha256,
        "authority_policy_fingerprint_sha256": policy_fingerprint,
        "official_report_sha256": contract.official_report_sha256,
        "candidate_report_sha256": contract.candidate_report_sha256,
        "review_registry_fingerprint_sha256": registry["registry_fingerprint_sha256"],
        "frontend_status_contract_version": FRONTEND_STATUS_CONTRACT_VERSION,
        "stabilization_policy_version": POLICY_VERSION,
        "source_manifest_sha256": canonical_json_hash(source_manifest),
    }
    run_fingerprint = canonical_json_hash(run_payload)
    run_id = f"stabilization-{run_fingerprint[:20]}"
    output = Path(output_root).resolve() if output_root else repo / "attendance_output/product_workflow/phase_2l_stabilization"
    production = {
        "family_id": EXPECTED_FAMILY,
        "variant_id": EXPECTED_VARIANT,
        "embedding_sha256": contract.expected_embedding_sha256,
        "summary_sha256": contract.expected_summary_sha256,
        "pointer_sha256": contract.expected_pointer_sha256,
        "embedding_dimension": 128,
        "embedding_records": 354,
        "aggregation": "top3",
    }
    policy_summary = {
        "processing_policy_version": PROCESSING_POLICY_VERSION,
        "automatic_authority": OFFICIAL_RECOGNITION_AUTHORITY,
        "official_reviewed_authority": "reviewed_multiframe_tracklet_evidence",
        "guarded_recovery_automatic": False,
        "match_threshold": 0.48,
        "margin_threshold": 0.08,
        "checkpoints": list(EXPECTED_CHECKPOINT_IDS),
        "cameras_per_checkpoint": list(EXPECTED_CAMERA_STEMS),
    }
    return StabilizationPreflight(
        repo,
        output,
        run_id,
        run_fingerprint,
        run_payload,
        protected,
        production,
        policy_summary,
        official_counts,
        candidate_counts,
        roster,
        registry,
        tuple(rollback_inventory),
        documentation_inventory,
        tuple(source_manifest),
        test_validation,
    )


def no_recognition_declaration() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "recognition_ran": False,
        "video_decoded": False,
        "yunet_ran": False,
        "sface_ran": False,
        "session_reprocessed": False,
        "recognition_imported_or_invoked": False,
        "declaration": "Phase 2L reads persisted state and immutable evidence only.",
    }


def no_activation_declaration() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "attendance_changed": False,
        "report_revision_changed": False,
        "finalization_changed": False,
        "authority_changed": False,
        "guarded_recovery_promoted": False,
        "thresholds_changed": False,
        "embeddings_changed": False,
        "roster_changed": False,
        "timetable_changed": False,
        "review_registry_changed": False,
        "manual_overrides_changed": False,
        "jobs_changed": False,
        "HOD_configuration_changed": False,
    }


def build_output_files(plan: StabilizationPreflight, external_validations: Sequence[str]) -> dict[str, bytes]:
    validations = tuple(sorted(set(external_validations)))
    if validations != tuple(sorted(REQUIRED_EXTERNAL_VALIDATIONS)):
        missing = sorted(set(REQUIRED_EXTERNAL_VALIDATIONS) - set(validations))
        extra = sorted(set(validations) - set(REQUIRED_EXTERNAL_VALIDATIONS))
        raise StabilizationError(f"external_validation_record_incomplete:missing={missing}:extra={extra}")
    if not all(item.get("passed") for item in plan.test_validation.values()):
        raise StabilizationError("preflight_contract_tests_not_passed")

    stabilization_policy = {
        "schema_version": 1,
        "policy_version": POLICY_VERSION,
        "frontend_status_contract_version": FRONTEND_STATUS_CONTRACT_VERSION,
        "release_scope": "current_policy_stabilization_and_controlled_professor_demo",
        "automatic_authority": OFFICIAL_RECOGNITION_AUTHORITY,
        "guarded_recovery_automatic": False,
        "mutating_demo_operations_prohibited": True,
        "independent_generalization_claimed": False,
    }
    release_freeze = {
        "schema_version": 1,
        "run_id": plan.run_id,
        "release_candidate_status": "controlled_demo_ready_with_blockers",
        "recommendation": PHASE_2K_C1_RECOMMENDATION,
        "production": plan.production,
        "policy": plan.policy,
        "official_report": {"session_id": SESSION_ID, "revision_role": "official_reviewed", "counts": plan.official_counts, "sha256": EXPECTED_OFFICIAL_REPORT_SHA256, "finalized": False},
        "automatic_candidate": {"revision_role": "automatic_candidate", "counts": plan.candidate_counts, "sha256": EXPECTED_CANDIDATE_REPORT_SHA256, "archived": True},
        "review_registry": plan.review_registry,
        "independent_generalization_claimed": False,
        "guarded_recovery_production_safe_claimed": False,
    }
    frontend_contract = {
        "schema_version": 1,
        "contract_version": FRONTEND_STATUS_CONTRACT_VERSION,
        "statuses": ["Present", "Needs Review", "Unconfirmed", "Missing Enrollment", "Absent", "Unknown"],
        "unresolved_formula": "Needs Review + Unconfirmed + Missing Enrollment + Unknown",
        "unconfirmed_meaning": "insufficient camera evidence; not absence",
        "missing_enrollment_meaning": "no usable production embedding",
        "absent_meaning": "explicit resolved outcome",
        "unknown_behavior": "fail_closed_unresolved",
        "evidence_flagged_explicit_absent": "unresolved_until_saved_manual_resolution",
    }
    demo_readiness = {
        "schema_version": 1,
        "controlled_professor_demo_ready": True,
        "live_demo_deferred": True,
        "evidence_source": "preserved_MON_P4_official_and_archived_candidate",
        "safe_page_order": ["Login", "Dashboard / My Classes", "Reports", "Review Students", "HOD Control"],
        "prohibited_actions": ["Process Attendance", "Reprocess", "Edit attendance", "Finalize Attendance", "Change HOD configuration", "Live Demo"],
        "credentials_embedded": False,
    }
    validation = {
        "schema_version": 1,
        "release_preflight": "pass",
        "contract_test_execution": plan.test_validation,
        "ordered_external_validations": [{"name": name, "passed": True} for name in REQUIRED_EXTERNAL_VALIDATIONS],
        "documented_skip_preserved": True,
        "operational_hash_drift": False,
    }
    evaluation = {
        "schema_version": 1,
        "run_id": plan.run_id,
        "run_fingerprint_sha256": plan.run_fingerprint_sha256,
        "release_candidate_status": "controlled_demo_ready_with_blockers",
        "recommendation": PHASE_2K_C1_RECOMMENDATION,
        "frontend_contract_coherent": True,
        "rollback_inventory_verified": True,
        "documentation_verified": True,
        "mixed_track_quarantined": True,
        "guarded_recovery_review_only": True,
        "no_recognition": True,
        "no_activation": True,
        "remaining_blockers": [
            "all_existing_sessions_are_contaminated",
            "no_independent_generalization_evidence",
            "historical_guarded_wrong_person_and_outsider_candidates",
            "mandatory_mixed_guarded_candidate",
            "historical_out_of_roster_prediction",
        ],
    }
    payloads = {
        "stabilization_policy.json": stabilization_policy,
        "release_candidate_freeze.json": release_freeze,
        "frontend_contract_summary.json": frontend_contract,
        "demo_readiness.json": demo_readiness,
        "rollback_inventory.json": {"schema_version": 1, "items": list(plan.rollback_inventory)},
        "documentation_inventory.json": {"schema_version": 1, "items": list(plan.documentation_inventory)},
        "validation_summary.json": validation,
        "no_recognition_declaration.json": no_recognition_declaration(),
        "no_activation_declaration.json": no_activation_declaration(),
        "source_manifest.json": {"schema_version": 1, "run_payload": plan.run_payload, "sources": list(plan.source_manifest)},
        "evaluation_summary.json": evaluation,
    }
    return {name: canonical_json_bytes(payload) for name, payload in payloads.items()}


def _manifest_bytes(plan: StabilizationPreflight, files: Mapping[str, bytes]) -> bytes:
    manifest = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "run_id": plan.run_id,
        "run_fingerprint_sha256": plan.run_fingerprint_sha256,
        "immutable": True,
        "files": {
            name: {"sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)}
            for name, content in sorted(files.items())
        },
    }
    return canonical_json_bytes(manifest)


def verify_immutable_output(output_dir: Path) -> dict[str, Any]:
    root = Path(output_dir).resolve()
    actual_names = sorted(path.name for path in root.iterdir() if path.is_file()) if root.is_dir() else []
    if actual_names != sorted(REQUIRED_OUTPUT_FILES):
        raise StabilizationError(f"immutable_file_set_mismatch:{actual_names}")
    manifest = _load_json(root / "immutable_manifest.json")
    if manifest.get("policy_version") != POLICY_VERSION or manifest.get("immutable") is not True:
        raise StabilizationError("immutable_manifest_contract_mismatch")
    files = manifest.get("files") or {}
    if sorted(files) != sorted(name for name in REQUIRED_OUTPUT_FILES if name != "immutable_manifest.json"):
        raise StabilizationError("immutable_manifest_file_map_mismatch")
    for name, expected in files.items():
        path = root / name
        if sha256_file(path) != expected.get("sha256") or path.stat().st_size != expected.get("size_bytes"):
            raise StabilizationError(f"immutable_output_tampered:{name}")
    release = _load_json(root / "release_candidate_freeze.json")
    if release.get("independent_generalization_claimed") is not False or release.get("guarded_recovery_production_safe_claimed") is not False:
        raise StabilizationError("unsafe_release_metadata_claim")
    recognition = _load_json(root / "no_recognition_declaration.json")
    if any(recognition.get(key) for key in ("recognition_ran", "video_decoded", "yunet_ran", "sface_ran", "session_reprocessed")):
        raise StabilizationError("no_recognition_declaration_invalid")
    activation = _load_json(root / "no_activation_declaration.json")
    if any(value is not False for key, value in activation.items() if key != "schema_version"):
        raise StabilizationError("no_activation_declaration_invalid")
    source = _load_json(root / "source_manifest.json")
    expected_fingerprint = canonical_json_hash(source.get("run_payload") or {})
    if manifest.get("run_fingerprint_sha256") != expected_fingerprint or manifest.get("run_id") != f"stabilization-{expected_fingerprint[:20]}":
        raise StabilizationError("immutable_run_fingerprint_mismatch")
    return manifest


def write_immutable_output(plan: StabilizationPreflight, files: Mapping[str, bytes]) -> tuple[Path, bool]:
    output_root = plan.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / plan.run_id
    complete = dict(files)
    complete["immutable_manifest.json"] = _manifest_bytes(plan, files)
    if target.exists():
        verify_immutable_output(target)
        for name, expected in complete.items():
            if (target / name).read_bytes() != expected:
                raise StabilizationError(f"deterministic_id_collision:{name}")
        return target, True
    staging = output_root / f".{plan.run_id}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        for name, content in sorted(complete.items()):
            (staging / name).write_bytes(content)
        os.replace(staging, target)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    verify_immutable_output(target)
    return target, False


def materialize(plan: StabilizationPreflight, external_validations: Sequence[str]) -> tuple[Path, bool, dict[str, Any]]:
    files = build_output_files(plan, external_validations)
    output_dir, reused = write_immutable_output(plan, files)
    manifest = verify_immutable_output(output_dir)
    return output_dir, reused, manifest


def contract_with_hash(contract: StabilizationContract, relative: str, digest: str) -> StabilizationContract:
    protected = dict(contract.protected_hashes)
    protected[relative] = digest
    return replace(contract, protected_hashes=protected)
