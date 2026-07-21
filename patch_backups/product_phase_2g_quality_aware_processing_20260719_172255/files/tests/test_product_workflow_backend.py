from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app as backend


class _ThreadStub:
    created = []

    def __init__(self, *, target, args, daemon):
        self.target = target
        self.args = args
        self.daemon = daemon
        self.started = False
        self.__class__.created.append(self)

    def start(self):
        self.started = True


class ProductWorkflowBackendTests(unittest.TestCase):
    def setUp(self):
        self.old_job_path = backend.JOB_RUNTIME_PATH
        self.old_jobs_hydrated = backend.JOBS_HYDRATED
        backend.JOBS.clear()
        backend.RUNNING_PROCESSES.clear()
        backend.CANCELLED_JOBS.clear()
        _ThreadStub.created.clear()

    def tearDown(self):
        backend.JOB_RUNTIME_PATH = self.old_job_path
        backend.JOBS_HYDRATED = self.old_jobs_hydrated
        backend.JOBS.clear()
        backend.RUNNING_PROCESSES.clear()
        backend.CANCELLED_JOBS.clear()

    def test_start_job_is_session_idempotent_and_preserves_actor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_dir = root / "videos"
            video_dir.mkdir()
            backend.JOB_RUNTIME_PATH = root / "job_runtime.json"
            backend.JOBS_HYDRATED = True
            session = {
                "session_id": "2026-06-30__B51__P1__CVO",
                "session_date": "2026-06-30",
                "day": "Tuesday",
                "period": "P1",
                "section": "B51",
                "subject_abbr": "CVO",
                "subject_name": "Computer Vision through OpenCV",
                "faculty_id": "vikas",
                "faculty_name": "Mr. Vikas B",
                "input_slot": "TUE_P1",
                "input_source_type": "prepared_slot",
                "input_source_path": "cctv_videos/prepared_slots/2026-06-30/TUE_P1",
            }
            actor = {
                "id": "vikas",
                "name": "Mr. Vikas B",
                "role": "faculty",
                "roleLabel": "Faculty - Computer Vision through OpenCV",
                "subjects": ["CVO"],
            }
            with patch("app.resolve_processing_context", return_value=({}, video_dir, session)), patch(
                "app.threading.Thread", _ThreadStub
            ):
                first_id, first_created = backend.start_job("Tuesday", "P1", {"subject_track": "CVO"}, user=actor)
                second_id, second_created = backend.start_job("Tuesday", "P1", {"subject_track": "CVO"}, user=actor)

            self.assertTrue(first_created)
            self.assertFalse(second_created)
            self.assertEqual(first_id, second_id)
            self.assertEqual(len(_ThreadStub.created), 1)
            self.assertTrue(_ThreadStub.created[0].started)
            self.assertEqual(backend.JOBS[first_id]["started_by_user_id"], "vikas")
            self.assertEqual(backend.JOBS[first_id]["controlled_by"], "Mr. Vikas B")
            registry = json.loads(backend.JOB_RUNTIME_PATH.read_text(encoding="utf-8"))
            self.assertEqual(len(registry["jobs"]), 1)

    def test_public_job_does_not_expose_command_stdout_or_attendance_rows(self):
        output = backend.public_job(
            {
                "job_id": "A1",
                "status": "Completed",
                "command": "python secret.py",
                "stdout": "large output",
                "_private": 1,
                "result": {"present_count": 20, "attendance_data": [{"Roll_Number": "1"}]},
            }
        )
        self.assertNotIn("command", output)
        self.assertNotIn("stdout", output)
        self.assertNotIn("_private", output)
        self.assertNotIn("attendance_data", output["result"])
        self.assertEqual(output["result"]["present_count"], 20)

    def test_save_statuses_never_persists_nested_runtime_status(self):
        old_status_path = backend.STATUS_PATH
        try:
            with tempfile.TemporaryDirectory() as tmp:
                backend.STATUS_PATH = Path(tmp) / "attendance_status.json"
                backend.save_statuses(
                    {
                        "session": {
                            "status": "Completed",
                            "attendance_data": [{"Roll_Number": "1", "Status": "Present"}],
                            "att_status": {"status": "Failed"},
                        }
                    }
                )
                stored = json.loads(backend.STATUS_PATH.read_text(encoding="utf-8"))
                self.assertNotIn("att_status", stored["session"])
                self.assertEqual(stored["session"]["attendance_data"][0]["Status"], "Present")
        finally:
            backend.STATUS_PATH = old_status_path


if __name__ == "__main__":
    unittest.main()
