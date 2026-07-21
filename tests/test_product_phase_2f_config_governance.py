from __future__ import annotations

import copy
import csv
import io
import json
import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime, timezone
from pathlib import Path

import app as backend
import src.face_attendance.config_governance as governance
from src.face_attendance.config_governance import (
    ConfigGovernanceError,
    apply_config_transaction,
    assign_subject_to_faculty,
    create_or_update_faculty,
    hash_password,
    json_bytes,
    rollback_latest_config_change,
    timetable_bytes,
    validate_timetable_bytes,
    verify_password,
)


def subject_catalog():
    return {
        "SWE": {"abbr": "SWE", "courseCode": "24CSEJ601", "courseName": "Software Engineering", "facultyId": "keerthi", "facultyName": "Dr. Keerthi G"},
        "CCM": {"abbr": "CCM", "courseCode": "24CSEJ602", "courseName": "Cloud Computing", "facultyId": "pranay", "facultyName": "Dr. Pranayaanath Reddy A"},
        "CVO": {"abbr": "CVO", "courseCode": "24AMLJ502", "courseName": "Computer Vision through OpenCV", "facultyId": "vikas", "facultyName": "Mr. Vikas B"},
    }


def role_users():
    return [
        {"id": "admin", "username": "admin", "password": "admin123", "name": "Main Admin", "role": "admin", "roleLabel": "Attendance Controller", "subjects": ["SWE", "CCM", "CVO"], "facultyName": "Main Admin", "canManageSystem": True, "canSeeAll": True},
        {"id": "keerthi", "username": "keerthi", "password": "swe123", "name": "Dr. Keerthi G", "role": "faculty", "roleLabel": "Faculty - Software Engineering", "subjects": ["SWE"], "facultyName": "Dr. Keerthi G", "canManageSystem": False, "canSeeAll": False},
        {"id": "vikas", "username": "vikas", "password": "cvo123", "name": "Mr. Vikas B", "role": "faculty", "roleLabel": "Faculty - Computer Vision through OpenCV", "subjects": ["CVO"], "facultyName": "Mr. Vikas B", "canManageSystem": False, "canSeeAll": False},
        {"id": "pranay", "username": "pranay", "password": "ccm123", "name": "Dr. Pranayaanath Reddy A", "role": "faculty", "roleLabel": "Faculty - Cloud Computing", "subjects": ["CCM"], "facultyName": "Dr. Pranayaanath Reddy A", "canManageSystem": False, "canSeeAll": False},
    ]


def mapping_payload():
    return {
        "subjects": subject_catalog(),
        "subject_students": {"SWE": [], "CCM": [], "CVO": []},
        "all_students": [],
    }


def timetable_rows():
    windows = {
        "1": ("09:00", "09:50"),
        "2": ("09:50", "10:40"),
        "3": ("10:50", "11:40"),
        "4": ("11:40", "12:30"),
        "6": ("13:20", "14:10"),
        "7": ("14:20", "15:10"),
        "8": ("15:10", "16:00"),
    }
    days = [("Monday", "MON"), ("Tuesday", "TUE"), ("Wednesday", "WED"), ("Thursday", "THU"), ("Friday", "FRI")]
    rows = []
    for day, prefix in days:
        for period, (start, end) in windows.items():
            if period in {"1", "2"}:
                abbr, code, name, instructor = "SWE", "24CSEJ601", "Software Engineering", "Dr. Keerthi G"
            elif period in {"3", "4"}:
                abbr, code, name, instructor = "CCM/CVO", "24CSEJ602/24AMLJ502", "Cloud Computing / Computer Vision through OpenCV", "Dr. Pranayaanath Reddy A / Mr. Vikas B"
            else:
                abbr, code, name, instructor = "CVO", "24AMLJ502", "Computer Vision through OpenCV", "Mr. Vikas B"
            rows.append({
                "slot_id": f"{prefix}_P{period}",
                "academic_year": "2026-2027",
                "year": "III",
                "semester": "V",
                "wef": "03/06/2026",
                "day": day,
                "period": period,
                "start_time": start,
                "end_time": end,
                "course_code": code,
                "course_abbr": abbr,
                "course_name": name,
                "instructor": instructor,
                "session_type": "Theory",
                "room": "1214/1215",
                "section": "B51",
                "camera_ids": "cam1",
                "attendance_required": "Yes",
                "notes": "",
            })
    return rows


class ConfigGovernancePureTests(unittest.TestCase):
    def test_new_passwords_are_hashed_and_legacy_login_still_works(self):
        users, change = create_or_update_faculty(
            role_users(),
            subject_catalog(),
            {
                "action": "create",
                "faculty_id": "newfaculty",
                "username": "new.faculty",
                "name": "Dr. New Faculty",
                "password": "strong-pass-123",
            },
        )
        created = next(row for row in users if row["id"] == "newfaculty")
        self.assertNotIn("password", created)
        self.assertTrue(created["password_hash"].startswith("pbkdf2_sha256$"))
        self.assertTrue(verify_password(created, "strong-pass-123"))
        self.assertTrue(verify_password(role_users()[1], "swe123"))
        self.assertTrue(change["password_changed"])

    def test_duplicate_username_and_hod_edit_fail_closed(self):
        with self.assertRaises(ConfigGovernanceError):
            create_or_update_faculty(
                role_users(),
                subject_catalog(),
                {"action": "create", "faculty_id": "other", "username": "vikas", "name": "Other", "password": "12345678"},
            )
        with self.assertRaises(ConfigGovernanceError):
            create_or_update_faculty(
                role_users(),
                subject_catalog(),
                {"action": "update", "faculty_id": "admin", "username": "admin", "name": "Changed"},
            )

    def test_exact_35_slot_timetable_passes_and_missing_slot_fails(self):
        payload = timetable_bytes(timetable_rows())
        result = validate_timetable_bytes(payload, subject_catalog(), role_users())
        self.assertTrue(result.valid, result.errors)
        self.assertEqual(result.row_count, 35)
        self.assertEqual(result.summary["days"]["Monday"], 7)

        broken = timetable_rows()[:-1]
        result = validate_timetable_bytes(timetable_bytes(broken), subject_catalog(), role_users())
        self.assertFalse(result.valid)
        self.assertTrue(any("Missing timetable slots" in error for error in result.errors))

    def test_subject_assignment_updates_roles_mapping_and_timetable_atomically_in_memory(self):
        users, mapping, rows, change = assign_subject_to_faculty(
            role_users(),
            mapping_payload(),
            timetable_rows(),
            subject="CVO",
            faculty_id="keerthi",
        )
        by_id = {row["id"]: row for row in users}
        self.assertIn("CVO", by_id["keerthi"]["subjects"])
        self.assertNotIn("CVO", by_id["vikas"]["subjects"])
        self.assertEqual(mapping["subjects"]["CVO"]["facultyId"], "keerthi")
        cvo_rows = [row for row in rows if "CVO" in row["course_abbr"].split("/")]
        self.assertTrue(cvo_rows)
        for row in cvo_rows:
            subjects = [part.strip() for part in row["course_abbr"].split("/")]
            instructors = [part.strip() for part in row["instructor"].split("/")]
            self.assertEqual(instructors[subjects.index("CVO")], "Dr. Keerthi G")
        self.assertEqual(change["previous_faculty_id"], "vikas")

    def test_transaction_writes_backup_audit_and_verified_rollback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data = root / "data"
            output = root / "attendance_output" / "product_workflow" / "phase_2f"
            data.mkdir(parents=True)
            users_path = data / "role_users.json"
            users_path.write_text('{"value": 1}\n', encoding="utf-8")
            result = apply_config_transaction(
                repo_root=root,
                targets={users_path: b'{"value": 2}\n'},
                backup_root=data / "config_backups",
                audit_root=output,
                pointer_path=data / "config_current.json",
                actor={"id": "admin", "name": "HOD"},
                change_type="test_change",
                reason="Testing guarded change",
                metadata={"safe": True},
                now=datetime(2026, 7, 19, 10, 0, tzinfo=timezone.utc),
            )
            self.assertEqual(users_path.read_text(encoding="utf-8"), '{"value": 2}\n')
            self.assertTrue(Path(result["backup_dir"]).is_dir())
            self.assertTrue(Path(result["audit_dir"], "audit_record.json").is_file())

            rollback = rollback_latest_config_change(
                repo_root=root,
                backup_root=data / "config_backups",
                audit_root=output,
                pointer_path=data / "config_current.json",
                actor={"id": "admin", "name": "HOD"},
                confirm_change_id=result["change_id"],
                reason="Restore previous config",
                now=datetime(2026, 7, 19, 11, 0, tzinfo=timezone.utc),
            )
            self.assertEqual(rollback["status"], "rolled_back")
            self.assertEqual(users_path.read_text(encoding="utf-8"), '{"value": 1}\n')

    def test_multi_file_write_failure_restores_every_original(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data = root / "data"
            data.mkdir(parents=True)
            first = data / "first.json"
            second = data / "second.json"
            first.write_bytes(b"first-before")
            second.write_bytes(b"second-before")
            original_write = governance._atomic_write_bytes

            def fail_second(path, payload):
                if Path(path) == second and payload == b"second-after":
                    raise OSError("simulated second-file failure")
                return original_write(path, payload)

            with patch.object(governance, "_atomic_write_bytes", side_effect=fail_second):
                with self.assertRaises(OSError):
                    apply_config_transaction(
                        repo_root=root,
                        targets={first: b"first-after", second: b"second-after"},
                        backup_root=data / "config_backups",
                        audit_root=root / "audit",
                        pointer_path=data / "config_current.json",
                        actor={"id": "admin", "name": "HOD"},
                        change_type="test_change",
                        reason="Testing atomic restoration",
                    )
            self.assertEqual(first.read_bytes(), b"first-before")
            self.assertEqual(second.read_bytes(), b"second-before")
            self.assertFalse((data / "config_current.json").exists())

    def test_rollback_refuses_current_file_drift(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data = root / "data"
            output = root / "attendance_output" / "product_workflow" / "phase_2f"
            data.mkdir(parents=True)
            path = data / "role_users.json"
            path.write_text("before", encoding="utf-8")
            result = apply_config_transaction(
                repo_root=root,
                targets={path: b"after"},
                backup_root=data / "config_backups",
                audit_root=output,
                pointer_path=data / "config_current.json",
                actor={"id": "admin", "name": "HOD"},
                change_type="test_change",
                reason="Testing guarded change",
            )
            path.write_text("drift", encoding="utf-8")
            with self.assertRaises(ConfigGovernanceError):
                rollback_latest_config_change(
                    repo_root=root,
                    backup_root=data / "config_backups",
                    audit_root=output,
                    pointer_path=data / "config_current.json",
                    actor={"id": "admin", "name": "HOD"},
                    confirm_change_id=result["change_id"],
                    reason="Restore previous config",
                )


class ProductPhase2FEndpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.root = root
        self.old = {
            "ROOT_DIR": backend.ROOT_DIR,
            "USERS_PATH": backend.USERS_PATH,
            "STUDENT_MAP_PATH": backend.STUDENT_MAP_PATH,
            "TIMETABLE_CANDIDATES": backend.TIMETABLE_CANDIDATES,
            "CONFIG_BACKUP_ROOT": backend.CONFIG_BACKUP_ROOT,
            "CONFIG_AUDIT_ROOT": backend.CONFIG_AUDIT_ROOT,
            "CONFIG_POINTER_PATH": backend.CONFIG_POINTER_PATH,
            "STATUS_PATH": backend.STATUS_PATH,
            "JOB_RUNTIME_PATH": backend.JOB_RUNTIME_PATH,
            "WORKFLOW_STATE_INITIALIZED": backend.WORKFLOW_STATE_INITIALIZED,
            "JOBS_HYDRATED": backend.JOBS_HYDRATED,
            "JOBS": copy.deepcopy(backend.JOBS),
            "AUTH_SESSIONS": copy.deepcopy(backend.AUTH_SESSIONS),
        }
        backend.ROOT_DIR = root
        backend.USERS_PATH = root / "data" / "role_users.json"
        backend.STUDENT_MAP_PATH = root / "data" / "student_faculty_map.json"
        self.timetable_path = root / "timetable_b51_2026_2027.csv"
        backend.TIMETABLE_CANDIDATES = [self.timetable_path]
        backend.CONFIG_BACKUP_ROOT = root / "data" / "config_backups"
        backend.CONFIG_AUDIT_ROOT = root / "attendance_output" / "product_workflow" / "phase_2f"
        backend.CONFIG_POINTER_PATH = root / "data" / "config_current.json"
        backend.STATUS_PATH = root / "data" / "attendance_status.json"
        backend.JOB_RUNTIME_PATH = root / "data" / "job_runtime.json"
        backend.WORKFLOW_STATE_INITIALIZED = True
        backend.JOBS_HYDRATED = True
        backend.JOBS.clear()
        backend.USERS_PATH.parent.mkdir(parents=True)
        backend.write_json(backend.USERS_PATH, role_users())
        backend.write_json(backend.STUDENT_MAP_PATH, mapping_payload())
        backend.write_json(backend.STATUS_PATH, {})
        self.timetable_path.write_bytes(timetable_bytes(timetable_rows()))
        backend.AUTH_SESSIONS.clear()
        self.client = backend.app.test_client()
        admin_login = self.client.post("/api/auth/login", json={"username": "admin", "password": "admin123"}).get_json()
        faculty_login = self.client.post("/api/auth/login", json={"username": "vikas", "password": "cvo123"}).get_json()
        self.admin_headers = {
            "X-User-Id": "admin",
            "Authorization": f"Bearer {admin_login['session_token']}",
        }
        self.faculty_headers = {
            "X-User-Id": "vikas",
            "Authorization": f"Bearer {faculty_login['session_token']}",
        }

    def tearDown(self):
        for key, value in self.old.items():
            if key == "JOBS":
                backend.JOBS.clear()
                backend.JOBS.update(value)
            elif key == "AUTH_SESSIONS":
                backend.AUTH_SESSIONS.clear()
                backend.AUTH_SESSIONS.update(value)
            else:
                setattr(backend, key, value)
        self.temp.cleanup()

    def test_write_endpoints_require_explicit_hod(self):
        no_identity = self.client.post("/api/hod/config/faculty", json={})
        self.assertEqual(no_identity.status_code, 401)
        spoofed_hod = self.client.post("/api/hod/config/faculty", headers={"X-User-Id": "admin"}, json={})
        self.assertEqual(spoofed_hod.status_code, 401)
        faculty = self.client.post("/api/hod/config/faculty", headers=self.faculty_headers, json={})
        self.assertEqual(faculty.status_code, 403)

    def test_create_faculty_never_returns_or_stores_plaintext_password(self):
        response = self.client.post(
            "/api/hod/config/faculty",
            headers=self.admin_headers,
            json={
                "action": "create",
                "faculty_id": "newfaculty",
                "username": "new.faculty",
                "name": "Dr. New Faculty",
                "password": "safe-password",
                "reason": "New faculty joined",
            },
        )
        self.assertEqual(response.status_code, 200, response.get_json())
        payload = response.get_json()
        self.assertNotIn("password", payload["faculty"])
        stored = json.loads(backend.USERS_PATH.read_text(encoding="utf-8"))
        created = next(row for row in stored if row["id"] == "newfaculty")
        self.assertNotIn("password", created)
        self.assertTrue(created["password_hash"].startswith("pbkdf2_sha256$"))
        login = self.client.post("/api/auth/login", json={"username": "new.faculty", "password": "safe-password"})
        self.assertEqual(login.status_code, 200)
        self.assertTrue(login.get_json().get("session_token"))
        self.assertNotIn("password_hash", login.get_json()["user"])
        self.assertTrue(backend.CONFIG_POINTER_PATH.is_file())

    def test_subject_assignment_updates_three_files_and_preserves_rosters(self):
        before_roster = json.loads(backend.STUDENT_MAP_PATH.read_text(encoding="utf-8"))["subject_students"]
        response = self.client.post(
            "/api/hod/config/subject-assignment",
            headers=self.admin_headers,
            json={"subject": "CVO", "faculty_id": "keerthi", "reason": "Faculty workload update"},
        )
        self.assertEqual(response.status_code, 200, response.get_json())
        mapping = json.loads(backend.STUDENT_MAP_PATH.read_text(encoding="utf-8"))
        self.assertEqual(mapping["subjects"]["CVO"]["facultyId"], "keerthi")
        self.assertEqual(mapping["subject_students"], before_roster)
        users = json.loads(backend.USERS_PATH.read_text(encoding="utf-8"))
        self.assertIn("CVO", next(row for row in users if row["id"] == "keerthi")["subjects"])
        self.assertNotIn("CVO", next(row for row in users if row["id"] == "vikas")["subjects"])

    def test_timetable_preview_does_not_write_and_apply_requires_same_hash(self):
        original = self.timetable_path.read_bytes()
        preview = self.client.post(
            "/api/hod/config/timetable/preview",
            headers=self.admin_headers,
            data={"timetable": (io.BytesIO(original), "timetable.csv")},
            content_type="multipart/form-data",
        )
        self.assertEqual(preview.status_code, 200, preview.get_json())
        preview_payload = preview.get_json()["preview"]
        self.assertTrue(preview_payload["valid"])
        self.assertEqual(self.timetable_path.read_bytes(), original)

        bad = self.client.post(
            "/api/hod/config/timetable/apply",
            headers=self.admin_headers,
            data={
                "timetable": (io.BytesIO(original), "timetable.csv"),
                "expected_sha256": "0" * 64,
                "reason": "New academic schedule",
                "confirmation": "REPLACE TIMETABLE",
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(bad.status_code, 409)
        self.assertEqual(self.timetable_path.read_bytes(), original)

    def test_config_payload_requires_token_and_exposes_no_credentials(self):
        denied = self.client.get("/api/hod/config", headers={"X-User-Id": "admin"})
        self.assertEqual(denied.status_code, 401)
        response = self.client.get("/api/hod/config", headers=self.admin_headers)
        self.assertEqual(response.status_code, 200, response.get_json())
        payload = response.get_json()
        self.assertTrue(payload["writes_enabled"])
        self.assertTrue(payload["timetable"]["valid"])
        serialized = json.dumps(payload)
        self.assertNotIn("admin123", serialized)
        self.assertNotIn("password_hash", serialized)

    def test_expired_bearer_token_cannot_fall_back_to_admin_header(self):
        token = self.admin_headers["Authorization"].split(" ", 1)[1]
        backend.AUTH_SESSIONS[token]["expires_at_epoch"] = 0
        response = self.client.post(
            "/api/hod/config/faculty",
            headers=self.admin_headers,
            json={
                "action": "create",
                "faculty_id": "expired",
                "username": "expired.user",
                "name": "Expired User",
                "password": "safe-password",
                "reason": "Should not be accepted",
            },
        )
        self.assertEqual(response.status_code, 401)

    def test_active_job_blocks_every_configuration_write(self):
        backend.JOBS["JOB-1"] = {"job_id": "JOB-1", "status": "Processing"}
        response = self.client.post(
            "/api/hod/config/faculty",
            headers=self.admin_headers,
            json={
                "action": "create",
                "faculty_id": "blocked",
                "username": "blocked.user",
                "name": "Blocked User",
                "password": "safe-password",
                "reason": "Should be blocked",
            },
        )
        self.assertEqual(response.status_code, 409)
        self.assertIn("active", response.get_json()["error"].lower())

    def test_latest_change_can_be_rolled_back_once(self):
        create = self.client.post(
            "/api/hod/config/faculty",
            headers=self.admin_headers,
            json={
                "action": "create",
                "faculty_id": "rollbackuser",
                "username": "rollback.user",
                "name": "Rollback User",
                "password": "safe-password",
                "reason": "Temporary faculty test",
            },
        )
        self.assertEqual(create.status_code, 200, create.get_json())
        change_id = create.get_json()["change"]["change_id"]
        rollback = self.client.post(
            "/api/hod/config/rollback",
            headers=self.admin_headers,
            json={"change_id": change_id, "reason": "Undo temporary test"},
        )
        self.assertEqual(rollback.status_code, 200, rollback.get_json())
        ids = {row["id"] for row in json.loads(backend.USERS_PATH.read_text(encoding="utf-8"))}
        self.assertNotIn("rollbackuser", ids)


if __name__ == "__main__":
    unittest.main()
