from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.face_attendance.config_governance import (
    ConfigGovernanceError,
    load_config_history,
    validate_role_users,
    validate_timetable_bytes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only Product Phase 2F guarded HOD configuration preflight."
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise ConfigGovernanceError(f"Could not parse {path}") from exc


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    users_path = root / "data" / "role_users.json"
    mapping_path = root / "data" / "student_faculty_map.json"
    timetable_path = root / "timetable_b51_2026_2027.csv"
    pointer_path = root / "data" / "config_current.json"
    backup_root = root / "data" / "config_backups"
    audit_root = root / "attendance_output" / "product_workflow" / "phase_2f"
    job_path = root / "data" / "job_runtime.json"
    status_path = root / "data" / "attendance_status.json"

    print("Product Phase 2F - guarded HOD configuration preflight")
    print(f"Repository: {root}")
    print("Configuration writes executed: no")
    print("Recognition executed: no")
    print("Attendance changes allowed: no")
    print("Roster changes allowed: no")
    print("Embedding/model changes allowed: no")

    for path in (users_path, mapping_path, timetable_path):
        if not path.is_file():
            raise ConfigGovernanceError(f"Required configuration file is missing: {path}")

    mapping = read_json(mapping_path)
    if not isinstance(mapping, dict) or not isinstance(mapping.get("subjects"), dict):
        raise ConfigGovernanceError("student_faculty_map.json has no authoritative subject catalog.")
    users = validate_role_users(read_json(users_path), mapping["subjects"])
    timetable = validate_timetable_bytes(timetable_path.read_bytes(), mapping["subjects"], users)
    if not timetable.valid:
        raise ConfigGovernanceError("Current timetable failed validation: " + "; ".join(timetable.errors[:8]))

    pending_backups = list(backup_root.glob(".pending-config-*")) if backup_root.is_dir() else []
    pending_audits = list(audit_root.glob(".pending-config-*")) if audit_root.is_dir() else []
    if pending_backups or pending_audits:
        raise ConfigGovernanceError("Incomplete pending configuration transaction directories were found.")

    active_jobs = 0
    if job_path.is_file():
        jobs = read_json(job_path)
        for row in jobs.get("jobs", []) if isinstance(jobs, dict) else []:
            if isinstance(row, dict) and str(row.get("status") or "") in {"Pending", "Processing"}:
                active_jobs += 1
    processing_statuses = 0
    if status_path.is_file():
        statuses = read_json(status_path)
        if isinstance(statuses, dict):
            processing_statuses = sum(
                isinstance(entry, dict) and str(entry.get("status") or "") == "Processing"
                for entry in statuses.values()
            )

    admin_count = sum(user.get("role") == "admin" for user in users)
    faculty_count = sum(user.get("role") == "faculty" for user in users)
    assigned = {}
    for user in users:
        if user.get("role") != "faculty":
            continue
        for subject in user.get("subjects") or []:
            if subject in assigned:
                raise ConfigGovernanceError(f"Subject {subject} is assigned to multiple faculty accounts.")
            assigned[subject] = user["id"]
    missing_assignments = sorted(set(mapping["subjects"]) - set(assigned))
    if missing_assignments:
        raise ConfigGovernanceError("Subjects without one faculty owner: " + ", ".join(missing_assignments))

    history = load_config_history(audit_root, limit=100)
    pointer_status = "not_created"
    if pointer_path.is_file():
        pointer = read_json(pointer_path)
        if not isinstance(pointer, dict):
            raise ConfigGovernanceError("Configuration pointer must be a JSON object.")
        pointer_status = str(pointer.get("status") or "unknown")
        if pointer_status not in {"applied", "rolled_back"}:
            raise ConfigGovernanceError(f"Unsupported configuration pointer status: {pointer_status}")

    print(f"Role users: {len(users)} ({admin_count} HOD, {faculty_count} faculty)")
    print(f"Subjects: {len(mapping['subjects'])}")
    print(f"Timetable rows: {timetable.row_count}")
    print(f"Timetable warnings: {len(timetable.warnings)}")
    print(f"Active jobs: {active_jobs}")
    print(f"Processing attendance states: {processing_statuses}")
    print(f"Existing configuration audits: {len(history)}")
    print(f"Configuration pointer: {pointer_status}")
    print("Current configuration files changed: no")
    print("PREFLIGHT_STATUS=PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConfigGovernanceError as exc:
        print(f"PREFLIGHT_STATUS=FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
