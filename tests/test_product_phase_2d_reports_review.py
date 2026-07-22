from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app as backend


class ProductPhase2DReportsReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.old = {
            "STATUS_PATH": backend.STATUS_PATH,
            "OVERRIDES_PATH": backend.OVERRIDES_PATH,
            "OUTPUT_DIR": backend.OUTPUT_DIR,
            "WORKFLOW_STATE_INITIALIZED": backend.WORKFLOW_STATE_INITIALIZED,
        }
        backend.STATUS_PATH = root / "attendance_status.json"
        backend.OVERRIDES_PATH = root / "manual_overrides.json"
        backend.OUTPUT_DIR = root / "attendance_output"
        backend.OUTPUT_DIR.mkdir(parents=True)
        backend.WORKFLOW_STATE_INITIALIZED = True
        backend.write_json(backend.OVERRIDES_PATH, {})
        self.roster_patch = patch.object(
            backend,
            "load_student_map",
            return_value={
                "subject_students": {"CVO": [{"roll": "1"}, {"roll": "2"}]},
                "all_students": [{"roll": "1"}, {"roll": "2"}],
            },
        )
        self.roster_patch.start()
        self.test_hod = {
            "id": "phase2l-test-hod",
            "username": "phase2l-test-hod",
            "name": "Synthetic HOD",
            "role": "admin",
            "roleLabel": "Head of Department",
            "subjects": ["SWE", "CCM", "CVO"],
            "canSeeAll": True,
        }
        self.users_patch = patch.object(backend, "load_role_users", return_value=[self.test_hod])
        self.users_patch.start()
        self.client = backend.app.test_client()
        token, _ = backend.issue_auth_session(self.test_hod)
        self.client.environ_base["HTTP_AUTHORIZATION"] = f"Bearer {token}"
        self.client.environ_base["HTTP_X_USER_ID"] = self.test_hod["id"]

    def tearDown(self):
        backend.STATUS_PATH = self.old["STATUS_PATH"]
        backend.OVERRIDES_PATH = self.old["OVERRIDES_PATH"]
        backend.OUTPUT_DIR = self.old["OUTPUT_DIR"]
        backend.WORKFLOW_STATE_INITIALIZED = self.old["WORKFLOW_STATE_INITIALIZED"]
        self.users_patch.stop()
        self.roster_patch.stop()
        backend.AUTH_SESSIONS.clear()
        self.temp.cleanup()

    @staticmethod
    def _entry(rows, **extra):
        return {
            "session_id": "2026-06-30__B51__P1__CVO",
            "key": "2026-06-30__B51__P1__CVO",
            "session_date": "2026-06-30",
            "day": "Tuesday",
            "period": "P1",
            "subject": "CVO",
            "subject_abbr": "CVO",
            "course_name": "Computer Vision through OpenCV",
            "status": "Needs Review",
            "attendance_data": rows,
            **extra,
        }

    @staticmethod
    def _complete_rows(first: dict, second: dict | None = None):
        return [first, second or {"Roll_Number": "2", "Status": "Absent", "Flags": "", "Detection_Count": 0}]

    def _save_entry(self, entry):
        backend.write_json(backend.STATUS_PATH, {entry["session_id"]: entry})

    def test_saved_manual_decision_resolves_flagged_row(self):
        row = {
            "Roll_Number": "2401100CSE0052",
            "Status": "Absent",
            "Flags": "Single Camera Evidence",
            "Detection_Count": 5,
            "Manual_Override": True,
        }
        self.assertFalse(backend.review_row_requires_attention(row))
        self.assertEqual(backend.review_state_for_rows([row])["unresolved_count"], 0)

    def test_attendance_session_response_is_compact(self):
        entry = self._entry(self._complete_rows({"Roll_Number": "1", "Status": "Absent"}))
        self._save_entry(entry)
        response = self.client.get(f"/api/attendance/{entry['session_id']}")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertNotIn("attendance_data", payload["entry"])
        self.assertEqual(payload["attendance_data"][0]["Roll_Number"], "1")

    def test_unknown_status_fails_closed_and_is_counted_explicitly(self):
        row = {"Roll_Number": "1", "Status": "Future Status"}
        summary = backend.summarize_attendance([row])
        self.assertEqual(summary["unknown_count"], 1)
        self.assertEqual(summary["present_count"], 0)
        self.assertEqual(summary["absent_count"], 0)
        self.assertTrue(backend.review_row_requires_attention(row))

    def test_compact_session_exposes_unknown_and_authority_fields(self):
        entry = self._entry(
            self._complete_rows({"Roll_Number": "1", "Status": "Future Status"}),
            status="Needs Review",
            official_recognition_authority="reviewed_multiframe_tracklet_evidence",
            automatic_recognition_authority="strict_tracklet_aggregate_with_guarded_review_candidates",
            guarded_recovery_automatic=False,
            guarded_recovery_authority="review_only",
            source_report_quality_failed=True,
        )
        compact = backend.compact_attendance_session(
            entry["session_id"],
            entry,
            {"id": "admin", "role": "admin", "canSeeAll": True, "subjects": ["CVO"]},
        )
        self.assertEqual(compact["unknown_count"], 1)
        self.assertEqual(compact["unresolved_count"], 1)
        self.assertEqual(compact["official_recognition_authority"], "reviewed_multiframe_tracklet_evidence")
        self.assertEqual(compact["automatic_recognition_authority"], "strict_tracklet_aggregate_with_guarded_review_candidates")
        self.assertFalse(compact["guarded_recovery_automatic"])
        self.assertEqual(compact["guarded_recovery_authority"], "review_only")
        self.assertTrue(compact["source_report_quality_failed"])

    def test_annotated_roll_is_canonicalized_without_aliasing_distinct_students(self):
        rows = backend.apply_override_map([{"Roll_Number": "2401100CSE0016 (A) EESHA", "Status": "Present"}], {})
        self.assertEqual(rows[0]["Roll_Number"], "2401100CSE0016")
        self.assertEqual(rows[0]["Source_Roll_Label"], "2401100CSE0016 (A) EESHA")
        self.assertEqual(backend.canonical_student_roll("2401100CSE0110"), "2401100CSE0110")
        self.assertEqual(backend.canonical_student_roll("24011CSEAI0110"), "24011CSEAI0110")

    def test_edit_requires_reason_and_reopens_finalized_report(self):
        entry = self._entry(
            self._complete_rows({"Roll_Number": "1", "Status": "Absent", "Flags": "Single Camera Evidence", "Detection_Count": 5}),
            status="Completed",
            attendance_finalized=True,
            Attendance_Finalized="Yes",
            finalized_at="earlier",
            finalized_by="Faculty",
            final_attendance_csv="old-final.csv",
        )
        self._save_entry(entry)
        missing_reason = self.client.post(
            f"/api/attendance/{entry['session_id']}/edit",
            json={"changes": [{"roll": "1", "status": "Present", "reason": ""}]},
        )
        self.assertEqual(missing_reason.status_code, 400)

        response = self.client.post(
            f"/api/attendance/{entry['session_id']}/edit",
            json={"changes": [{"roll": "1", "status": "Present", "reason": "Verified from classroom evidence"}]},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["review_state"]["state"], "ready_to_finalize")
        self.assertFalse(payload["entry"]["attendance_finalized"])
        stored = json.loads(backend.STATUS_PATH.read_text(encoding="utf-8"))[entry["session_id"]]
        self.assertEqual(stored["status"], "Needs Review")
        self.assertEqual(stored["previous_finalized_at"], "earlier")
        self.assertEqual(stored["previous_final_attendance_csv"], "old-final.csv")
        self.assertEqual(stored["Attendance_Finalized"], "No")
        self.assertEqual(stored["finalization_history"][0]["final_attendance_csv"], "old-final.csv")
        self.assertEqual(stored["attendance_data"][0]["Status"], "Absent")

    def test_output_ref_is_stable_for_resolved_output_paths(self):
        nested = backend.OUTPUT_DIR / "2026-07-18" / "session" / "final.csv"
        nested.parent.mkdir(parents=True, exist_ok=True)
        nested.write_text("Roll_Number,Status\n1,Present\n", encoding="utf-8")
        expected = "2026-07-18/session/final.csv"
        self.assertEqual(backend.output_ref(nested), expected)
        self.assertEqual(backend.output_ref(nested.resolve()), expected)
        self.assertEqual(backend.output_ref(backend.safe_output_path(expected)), expected)

    def test_finalize_blocks_unresolved_then_passes_and_is_idempotent(self):
        entry = self._entry(
            self._complete_rows({"Roll_Number": "1", "Status": "Absent", "Flags": "Single Camera Evidence", "Detection_Count": 5})
        )
        self._save_entry(entry)
        blocked = self.client.post(f"/api/attendance/{entry['session_id']}/finalize", json={})
        self.assertEqual(blocked.status_code, 409)
        self.assertEqual(blocked.get_json()["review_state"]["unresolved_count"], 1)

        saved = self.client.post(
            f"/api/attendance/{entry['session_id']}/edit",
            json={"changes": [{"roll": "1", "status": "Absent", "reason": "Reviewed and confirmed absent"}]},
        )
        self.assertEqual(saved.status_code, 200)

        finalized = self.client.post(f"/api/attendance/{entry['session_id']}/finalize", json={})
        self.assertEqual(finalized.status_code, 200)
        payload = finalized.get_json()
        self.assertTrue(payload["review_state"]["attendance_finalized"])
        self.assertTrue(Path(backend.OUTPUT_DIR / payload["file"]).exists())
        stored = json.loads(backend.STATUS_PATH.read_text(encoding="utf-8"))[entry["session_id"]]
        self.assertEqual(stored["Attendance_Finalized"], "Yes")

        repeated = self.client.post(f"/api/attendance/{entry['session_id']}/finalize", json={})
        self.assertEqual(repeated.status_code, 200)
        self.assertTrue(repeated.get_json()["already_finalized"])
        self.assertEqual(repeated.get_json()["file"], payload["file"])

    def test_attendance_sessions_list_exposes_state_not_rows(self):
        complete = self._complete_rows({"Roll_Number": "1", "Status": "Absent"})
        ready = self._entry(complete, requires_manual_review=False)
        finalized = self._entry(
            self._complete_rows({"Roll_Number": "1", "Status": "Present"}, {"Roll_Number": "2", "Status": "Present"}),
            session_id="2026-06-30__B51__P2__CVO",
            key="2026-06-30__B51__P2__CVO",
            period="P2",
            status="Completed",
            attendance_finalized=True,
            final_attendance_csv="final.csv",
        )
        backend.write_json(backend.STATUS_PATH, {ready["session_id"]: ready, finalized["session_id"]: finalized})
        response = self.client.get("/api/attendance-sessions")
        self.assertEqual(response.status_code, 200)
        sessions = response.get_json()["sessions"]
        self.assertEqual(len(sessions), 2)
        self.assertTrue(all("attendance_data" not in item for item in sessions))
        states = {item["session_id"]: item["review_state"] for item in sessions}
        self.assertEqual(states[ready["session_id"]], "ready_to_finalize")
        self.assertEqual(states[finalized["session_id"]], "finalized")

    def test_roster_mismatch_is_visible_and_blocks_finalization(self):
        entry = self._entry([{"Roll_Number": "1", "Status": "Present"}, {"Roll_Number": "9", "Status": "Absent"}])
        self._save_entry(entry)
        listed = self.client.get("/api/attendance-sessions").get_json()["sessions"][0]
        self.assertEqual(listed["review_state"], "roster_mismatch")
        self.assertEqual(listed["missing_roster_rolls"], ["2"])
        self.assertEqual(listed["unexpected_roster_rolls"], ["9"])

        blocked = self.client.post(f"/api/attendance/{entry['session_id']}/finalize", json={})
        self.assertEqual(blocked.status_code, 409)
        self.assertEqual(blocked.get_json()["review_state"]["state"], "roster_mismatch")

    def test_lowercase_finalization_flag_overrides_legacy_uppercase_flag(self):
        self.assertFalse(backend.attendance_is_finalized({"attendance_finalized": False, "Attendance_Finalized": "Yes"}))
        self.assertTrue(backend.attendance_is_finalized({"Attendance_Finalized": "Yes"}))

    def test_edit_failure_restores_overrides_and_removes_draft_export(self):
        entry = self._entry(self._complete_rows({"Roll_Number": "1", "Status": "Absent", "Flags": "Single Camera Evidence", "Detection_Count": 5}))
        self._save_entry(entry)
        original_save = backend.save_statuses
        calls = {"count": 0}

        def fail_once(payload):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("simulated status write failure")
            return original_save(payload)

        with patch.object(backend, "save_statuses", side_effect=fail_once):
            response = self.client.post(
                f"/api/attendance/{entry['session_id']}/edit",
                json={"changes": [{"roll": "1", "status": "Present", "reason": "Verified"}]},
            )
        self.assertEqual(response.status_code, 500)
        self.assertEqual(json.loads(backend.OVERRIDES_PATH.read_text(encoding="utf-8")), {})
        stored = json.loads(backend.STATUS_PATH.read_text(encoding="utf-8"))[entry["session_id"]]
        self.assertEqual(stored["status"], "Needs Review")
        self.assertEqual(list(backend.OUTPUT_DIR.rglob("admin_reviewed_*.csv")), [])

    def test_finalize_failure_removes_uncommitted_final_export(self):
        entry = self._entry(self._complete_rows({"Roll_Number": "1", "Status": "Present"}))
        self._save_entry(entry)
        with patch.object(backend, "save_statuses", side_effect=RuntimeError("simulated finalization write failure")):
            response = self.client.post(f"/api/attendance/{entry['session_id']}/finalize", json={})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(list(backend.OUTPUT_DIR.rglob("finalized_attendance_*.csv")), [])
        stored = json.loads(backend.STATUS_PATH.read_text(encoding="utf-8"))[entry["session_id"]]
        self.assertFalse(backend.attendance_is_finalized(stored))


if __name__ == "__main__":
    unittest.main()
