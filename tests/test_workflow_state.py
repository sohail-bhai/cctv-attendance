from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.face_attendance.workflow_state import (
    WorkflowStateError,
    active_job_for_session,
    atomic_write_json,
    compact_job_for_storage,
    load_job_registry,
    migrate_status_file,
    read_json_object,
    sanitize_status_store,
    serialize_job_registry,
)


class StatusSanitizationTests(unittest.TestCase):
    def test_recursive_runtime_fields_are_removed_without_losing_evidence(self):
        payload = {
            "legacy": {
                "status": "Completed",
                "attendance_data": [{"Roll_Number": "2401", "Status": "Present"}],
                "att_status": {
                    "status": "Failed",
                    "att_status": {"status": "Older"},
                },
                "frontend_state": {"expanded": True},
            }
        }
        clean, report = sanitize_status_store(payload)
        self.assertTrue(report.changed)
        self.assertEqual(report.entries_changed, 1)
        self.assertEqual(report.removed_fields, 2)
        self.assertNotIn("att_status", clean["legacy"])
        self.assertNotIn("frontend_state", clean["legacy"])
        self.assertEqual(clean["legacy"]["attendance_data"][0]["Status"], "Present")

    def test_non_object_store_fails_closed(self):
        with self.assertRaises(WorkflowStateError):
            sanitize_status_store([{"status": "Completed"}])

    def test_migration_creates_exact_backup_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            status_path = root / "attendance_status.json"
            backup_dir = root / "backups"
            original = {"legacy": {"status": "Completed", "att_status": {"status": "Failed"}}}
            status_path.write_text(json.dumps(original, indent=2), encoding="utf-8")
            original_bytes = status_path.read_bytes()

            first = migrate_status_file(status_path, backup_dir, timestamp="20260718_200000")
            self.assertTrue(first["changed"])
            backup_path = Path(first["backup_path"])
            self.assertEqual(backup_path.read_bytes(), original_bytes)
            self.assertNotIn("att_status", read_json_object(status_path)["legacy"])

            second = migrate_status_file(status_path, backup_dir, timestamp="20260718_200001")
            self.assertFalse(second["changed"])
            self.assertIsNone(second["backup_path"])
            self.assertEqual(len(list(backup_dir.glob("*.json"))), 1)

    def test_atomic_write_leaves_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state" / "value.json"
            atomic_write_json(path, {"value": 7, "path": Path("x/y")})
            self.assertEqual(read_json_object(path), {"value": 7, "path": str(Path("x/y"))})
            self.assertEqual(list(path.parent.glob("*.tmp")), [])


class JobRegistryTests(unittest.TestCase):
    def test_compact_job_removes_private_and_large_fields(self):
        job = {
            "job_id": "AB12",
            "status": "Completed",
            "command": "secret command",
            "stdout": "large output",
            "_total_videos": 10,
            "result": {
                "present_count": 20,
                "total_students": 27,
                "attendance_data": [{"Roll_Number": "x"}],
            },
        }
        compact = compact_job_for_storage(job)
        self.assertNotIn("command", compact)
        self.assertNotIn("stdout", compact)
        self.assertNotIn("_total_videos", compact)
        self.assertNotIn("attendance_data", compact["result"])
        self.assertEqual(compact["result"]["present_count"], 20)

    def test_interrupted_jobs_become_failed_after_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jobs.json"
            atomic_write_json(
                path,
                serialize_job_registry(
                    {
                        "A1": {
                            "job_id": "A1",
                            "session_id": "2026-06-30__B51__P1__CVO",
                            "status": "Processing",
                            "created_at": "old",
                        },
                        "B2": {
                            "job_id": "B2",
                            "session_id": "2026-06-30__B51__P2__CVO",
                            "status": "Completed",
                            "created_at": "new",
                        },
                    }
                ),
            )
            jobs, changed = load_job_registry(
                path,
                now_text="18-07-2026 08:00:00 PM",
                interrupted_message="Backend restarted",
            )
            self.assertTrue(changed)
            self.assertEqual(jobs["A1"]["status"], "Failed")
            self.assertEqual(jobs["A1"]["error"], "Backend restarted")
            self.assertEqual(jobs["B2"]["status"], "Completed")

    def test_active_job_selection_is_session_scoped_and_deterministic(self):
        jobs = {
            "OLD": {"job_id": "OLD", "session_id": "S1", "status": "Processing", "created_at": "1"},
            "NEW": {"job_id": "NEW", "session_id": "S1", "status": "Pending", "created_at": "2"},
            "OTHER": {"job_id": "OTHER", "session_id": "S2", "status": "Processing", "created_at": "3"},
        }
        self.assertEqual(active_job_for_session(jobs, "S1")["job_id"], "NEW")
        self.assertEqual(active_job_for_session(jobs, "S2")["job_id"], "OTHER")
        self.assertIsNone(active_job_for_session(jobs, "S3"))

    def test_duplicate_job_ids_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jobs.json"
            atomic_write_json(
                path,
                {
                    "schema_version": 1,
                    "jobs": [
                        {"job_id": "A1", "status": "Completed"},
                        {"job_id": "a1", "status": "Failed"},
                    ],
                },
            )
            with self.assertRaises(WorkflowStateError):
                load_job_registry(path, now_text="now", interrupted_message="restart")


if __name__ == "__main__":
    unittest.main()
