import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import app as backend
from src.face_attendance.session_identity import (
    build_attendance_session_id,
    normalize_session_date,
    parse_attendance_session_id,
    strip_runtime_fields,
)


class SessionIdentityTests(unittest.TestCase):
    def test_simultaneous_courses_get_distinct_session_ids(self):
        cvo = build_attendance_session_id("2026-06-30", "B51", "P1", "CVO")
        ccm = build_attendance_session_id("2026-06-30", "B51", "P1", "CCM")
        self.assertNotEqual(cvo, ccm)
        self.assertEqual(cvo, "2026-06-30__B51__P1__CVO")
        self.assertEqual(ccm, "2026-06-30__B51__P1__CCM")

    def test_same_weekday_period_on_different_dates_are_distinct(self):
        first = build_attendance_session_id("2026-06-22", "B51", "P3", "CVO")
        second = build_attendance_session_id("2026-06-30", "B51", "P3", "CVO")
        self.assertNotEqual(first, second)

    def test_different_sections_are_distinct(self):
        b51 = build_attendance_session_id("2026-06-30", "B51", "P1", "CVO")
        b52 = build_attendance_session_id("2026-06-30", "B52", "P1", "CVO")
        self.assertNotEqual(b51, b52)

    def test_normalization_is_deterministic(self):
        one = build_attendance_session_id("2026-06-30", "b 51", "1", "cvo/lab")
        two = build_attendance_session_id("2026-06-30", "B_51", "P1", "CVO_LAB")
        self.assertEqual(one, two)

    def test_parse_round_trip(self):
        session_id = build_attendance_session_id("2026-06-30", "B51", "P2", "CCM")
        parsed = parse_attendance_session_id(session_id)
        self.assertEqual(parsed["session_date"], "2026-06-30")
        self.assertEqual(parsed["section"], "B51")
        self.assertEqual(parsed["period"], "P2")
        self.assertEqual(parsed["subject_abbr"], "CCM")

    def test_invalid_date_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_session_date("Tuesday")

    def test_two_logical_sessions_can_share_one_input_source(self):
        video_dir = backend.ROOT_DIR / "cctv_videos" / "TUE_P1"
        base = {
            "day": "Tuesday",
            "period": "P1",
            "slot_id": "TUE_P1",
            "section": "B51",
            "subject": "CVO",
        }
        cvo = backend.build_session_context(base, {"session_date": "2026-06-30", "subject_track": "CVO"}, video_dir)
        ccm = backend.build_session_context({**base, "subject": "CCM"}, {"session_date": "2026-06-30", "subject_track": "CCM"}, video_dir)
        self.assertNotEqual(cvo["session_id"], ccm["session_id"])
        self.assertEqual(cvo["input_source_path"], ccm["input_source_path"])

    def test_faculty_access_is_course_scoped(self):
        admin = {"role": "admin", "canSeeAll": True, "name": "Admin"}
        vikas = {"role": "faculty", "subjects": ["CVO"]}
        pranay = {"role": "faculty", "subjects": ["CCM"]}
        cvo_entry = {"subject_abbr": "CVO"}
        ccm_entry = {"subject_abbr": "CCM"}
        self.assertTrue(backend.user_can_access_session(vikas, cvo_entry))
        self.assertFalse(backend.user_can_access_session(vikas, ccm_entry))
        self.assertTrue(backend.user_can_access_session(pranay, ccm_entry))
        self.assertTrue(backend.user_can_access_session(admin, cvo_entry))
        self.assertTrue(backend.user_can_access_session(admin, ccm_entry))

    def test_manual_overrides_are_session_scoped_and_preserve_rows(self):
        cvo_id = "2026-06-30__B51__P1__CVO"
        ccm_id = "2026-06-30__B51__P1__CCM"
        rows = [
            {"Roll_Number": "101", "Status": "Absent"},
            {"Roll_Number": "102", "Status": "Present"},
        ]
        overrides = {cvo_id: {"101": {"manual_status": "Present", "reason": "Verified"}}}
        with patch("app.load_overrides", return_value=overrides):
            cvo_rows = backend.apply_overrides(cvo_id, rows)
            ccm_rows = backend.apply_overrides(ccm_id, rows)
        self.assertEqual(cvo_rows[0]["Status"], "Present")
        self.assertEqual(ccm_rows[0]["Status"], "Absent")
        self.assertEqual(len(cvo_rows), 2)

    def test_newest_created_ignores_old_historical_reports(self):
        old_output_dir = backend.OUTPUT_DIR
        with TemporaryDirectory() as tmp:
            backend.OUTPUT_DIR = Path(tmp)
            stale = backend.OUTPUT_DIR / "attendance_2026-06-30__B51__P1__CVO_old.csv"
            stale.write_text("Roll_Number,Status\n101,Present\n", encoding="utf-8")
            old_time = time.time() - 600
            stale.touch()
            self.assertIsNone(backend.newest_created("attendance_2026-06-30__B51__P1__CVO_*.csv", time.time()))
            fresh = backend.OUTPUT_DIR / "attendance_2026-06-30__B51__P1__CVO_new.csv"
            fresh.write_text("Roll_Number,Status\n101,Present\n", encoding="utf-8")
            self.assertEqual(backend.newest_created("attendance_2026-06-30__B51__P1__CVO_*.csv", old_time), fresh)
        backend.OUTPUT_DIR = old_output_dir

    def test_prepared_slot_date_mismatch_is_rejected(self):
        row = {"slot_id": "TUE_P1", "day": "Tuesday", "period": "P1"}
        source = backend.ROOT_DIR / "cctv_videos" / "prepared_slots" / "2026-06-30" / "TUE_P1"
        with self.assertRaisesRegex(ValueError, "Prepared-slot date mismatch"):
            backend.resolve_session_date(row, {"session_date": "2026-06-22"}, source)

    def test_direct_slot_without_reliable_date_cannot_use_today(self):
        row = {"slot_id": "MON_P1", "day": "Monday", "period": "P1"}
        source = backend.ROOT_DIR / "cctv_videos" / "MON_P1"
        with self.assertRaisesRegex(ValueError, "Attendance date is required"):
            backend.resolve_session_date(row, {}, source)


    def test_requested_session_id_mismatch_is_rejected(self):
        row = {
            "slot_id": "TUE_P1",
            "day": "Tuesday",
            "period": "P1",
            "section": "B51",
            "subject": "CVO",
        }
        with self.assertRaisesRegex(ValueError, "Session ID mismatch"):
            backend.build_session_context(
                row,
                {"session_date": "2026-06-30", "subject_track": "CVO", "session_id": "2026-06-30__B51__P1__CCM"},
                backend.ROOT_DIR / "cctv_videos" / "TUE_P1",
            )

    def test_ambiguous_period_requires_subject_track_for_admin(self):
        admin = {"role": "admin", "canSeeAll": True, "name": "Admin"}
        with patch("app.load_statuses", return_value={}), patch("app.repair_stale_processing_statuses", side_effect=lambda statuses: statuses):
            with self.assertRaisesRegex(ValueError, "Ambiguous CCM/CVO track"):
                backend.resolve_processing_context("Tuesday", "P1", {"session_date": "2026-06-30"}, admin)

    def test_prepared_slot_path_supplies_date(self):
        row = {"slot_id": "TUE_P1", "day": "Tuesday", "period": "P1"}
        source = backend.ROOT_DIR / "cctv_videos" / "prepared_slots" / "2026-06-30" / "TUE_P1"
        self.assertEqual(backend.resolve_session_date(row, {}, source), "2026-06-30")

    def test_known_demo_mapping_supplies_direct_slot_date(self):
        row = {"slot_id": "TUE_P2", "day": "Tuesday", "period": "P2"}
        source = backend.ROOT_DIR / "cctv_videos" / "TUE_P2"
        self.assertEqual(backend.resolve_session_date(row, {}, source), "2026-06-30")

    def test_known_backup_mapping_supplies_monday_date(self):
        row = {"slot_id": "MON_P3", "day": "Monday", "period": "P3"}
        source = backend.ROOT_DIR / "cctv_videos" / "MON_P3"
        self.assertEqual(backend.resolve_session_date(row, {}, source), "2026-06-22")

    def test_runtime_status_is_not_recursively_persisted(self):
        clean = strip_runtime_fields({"slot_id": "TUE_P1", "att_status": {"status": "Completed"}, "rt_status": "pending"})
        self.assertEqual(clean, {"slot_id": "TUE_P1"})


if __name__ == "__main__":
    unittest.main()
