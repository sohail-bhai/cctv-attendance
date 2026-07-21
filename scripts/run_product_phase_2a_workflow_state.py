from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.face_attendance.workflow_state import (  # noqa: E402
    atomic_write_json,
    load_job_registry,
    migrate_status_file,
    read_json_object,
    sanitize_status_store,
    serialize_job_registry,
)

STATUS_PATH = ROOT / "data" / "attendance_status.json"
JOB_PATH = ROOT / "data" / "job_runtime.json"
BACKUP_DIR = ROOT / "data" / "state_backups"
OUTPUT_ROOT = ROOT / "attendance_output" / "product_workflow" / "phase_2a"
PROTECTED_PATHS = (
    ROOT / "models" / "student_embeddings.pkl",
    ROOT / "models" / "embedding_summary.csv",
)
POLICY_VERSION = "product-phase-2a-workflow-state-foundation-v1"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str | None:
    return sha256_bytes(path.read_bytes()) if path.exists() else None


def stable_json_hash(payload) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256_bytes(raw)


def inspect_state() -> dict:
    raw_status = read_json_object(STATUS_PATH, missing_default={})
    clean_status, report = sanitize_status_store(raw_status)
    jobs, jobs_changed = load_job_registry(
        JOB_PATH,
        now_text="BACKEND_RESTART_PREFLIGHT",
        interrupted_message="Previous processing was interrupted by a backend restart. Reprocess this session safely.",
    ) if JOB_PATH.exists() else ({}, False)
    return {
        "policy_version": POLICY_VERSION,
        "status_path": str(STATUS_PATH),
        "status_exists": STATUS_PATH.exists(),
        "status_before_sha256": sha256_file(STATUS_PATH),
        "status_semantic_sha256": stable_json_hash(clean_status),
        "status_entries": len(clean_status),
        "status_sanitization": report.as_dict(),
        "status_before_bytes": STATUS_PATH.stat().st_size if STATUS_PATH.exists() else 0,
        "status_after_estimated_bytes": len((json.dumps(clean_status, indent=2, ensure_ascii=False) + "\n").encode("utf-8")),
        "job_registry_exists": JOB_PATH.exists(),
        "persisted_jobs": len(jobs),
        "interrupted_jobs_would_be_closed": bool(jobs_changed),
        "protected_hashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in PROTECTED_PATHS},
    }


def preflight() -> int:
    report = inspect_state()
    print("Product Phase 2A - workflow-state foundation preflight")
    print(f"Repository: {ROOT}")
    print(f"Status entries: {report['status_entries']}")
    print(f"Nested/transient fields to remove: {report['status_sanitization']['removed_fields']}")
    print(f"Status size: {report['status_before_bytes']} -> approximately {report['status_after_estimated_bytes']} bytes")
    print(f"Persisted jobs: {report['persisted_jobs']}")
    print("Recognition executed: no")
    print("Attendance results changed: no")
    print("Production embeddings changed: no")
    print("PREFLIGHT_STATUS=PASS")
    return 0


def run(confirm_backend_stopped: str) -> int:
    if confirm_backend_stopped != "BACKEND_STOPPED":
        raise SystemExit('Run requires --confirm-backend-stopped BACKEND_STOPPED')

    before = inspect_state()
    clean_before, _ = sanitize_status_store(read_json_object(STATUS_PATH, missing_default={}))
    migration = migrate_status_file(STATUS_PATH, BACKUP_DIR)

    jobs, jobs_changed = load_job_registry(
        JOB_PATH,
        now_text=datetime.now().strftime("%d-%m-%Y %I:%M:%S %p"),
        interrupted_message="Previous processing was interrupted by a backend restart. Reprocess this session safely.",
    ) if JOB_PATH.exists() else ({}, False)
    if jobs_changed:
        atomic_write_json(JOB_PATH, serialize_job_registry(jobs))

    after_status = read_json_object(STATUS_PATH, missing_default={})
    clean_after, after_report = sanitize_status_store(after_status)
    if after_report.changed:
        raise RuntimeError("Status migration did not produce a fully sanitized store.")
    if clean_after != clean_before:
        raise RuntimeError("Attendance status semantic evidence changed during sanitization.")

    protected_after = {str(path.relative_to(ROOT)): sha256_file(path) for path in PROTECTED_PATHS}
    if protected_after != before["protected_hashes"]:
        raise RuntimeError("Protected production model files changed during workflow-state migration.")

    output_id = "workflow-state-" + stable_json_hash(
        {
            "status_semantic_sha256": stable_json_hash(clean_after),
            "job_registry_sha256": sha256_file(JOB_PATH),
            "policy_version": POLICY_VERSION,
        }
    )[:20]
    output_dir = OUTPUT_ROOT / output_id
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "workflow_state_audit.json"

    audit = {
        "schema_version": 1,
        "phase": "Product Phase 2A",
        "policy_version": POLICY_VERSION,
        "output_id": output_id,
        "status_migration": migration,
        "status_semantic_sha256": stable_json_hash(clean_after),
        "status_after_sha256": sha256_file(STATUS_PATH),
        "status_after_bytes": STATUS_PATH.stat().st_size,
        "persisted_jobs": len(jobs),
        "interrupted_jobs_closed": bool(jobs_changed),
        "job_registry_sha256": sha256_file(JOB_PATH),
        "protected_hashes_before": before["protected_hashes"],
        "protected_hashes_after": protected_after,
        "recognition_executed": False,
        "official_attendance_changed": False,
        "production_embeddings_changed": False,
    }

    if audit_path.exists():
        existing = read_json_object(audit_path)
        stable_keys = (
            "policy_version",
            "output_id",
            "status_semantic_sha256",
            "status_after_sha256",
            "job_registry_sha256",
            "protected_hashes_after",
        )
        mismatches = [key for key in stable_keys if existing.get(key) != audit.get(key)]
        if mismatches:
            raise RuntimeError(
                f"Existing immutable audit conflicts with this run for {mismatches}: {audit_path}"
            )
        audit = existing
        reused = True
    else:
        atomic_write_json(audit_path, audit)
        reused = False

    print("Product Phase 2A workflow-state migration completed safely.")
    print(f"Output: {output_dir}")
    stored_migration = audit.get("status_migration") or migration
    print(f"Status backup: {stored_migration.get('backup_path') or 'not required'}")
    print(f"Removed nested/transient fields: {migration.get('removed_fields', 0)}")
    print(f"Persisted jobs: {len(jobs)}")
    print(f"Idempotent audit reuse: {'yes' if reused else 'no'}")
    print("Recognition executed: no")
    print("Official attendance changed: no")
    print("Production embeddings changed: no")
    print(f"PRODUCT_PHASE_2A_OUTPUT={output_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Product Phase 2A workflow-state migration and audit.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--confirm-backend-stopped", required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        return preflight()
    return run(args.confirm_backend_stopped)


if __name__ == "__main__":
    raise SystemExit(main())
