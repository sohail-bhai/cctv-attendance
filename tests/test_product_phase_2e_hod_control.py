from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app as backend


class ProductPhase2EHodControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.old = {
            "USERS_PATH": backend.USERS_PATH,
            "STUDENT_MAP_PATH": backend.STUDENT_MAP_PATH,
            "EMBEDDING_SUMMARY_PATH": backend.EMBEDDING_SUMMARY_PATH,
            "DATASET_ROOT": backend.DATASET_ROOT,
            "CURRENT_EMBEDDING_VERSION_PATH": backend.CURRENT_EMBEDDING_VERSION_PATH,
        }
        backend.USERS_PATH = root / "role_users.json"
        backend.STUDENT_MAP_PATH = root / "student_faculty_map.json"
        backend.EMBEDDING_SUMMARY_PATH = root / "models" / "embedding_summary.csv"
        backend.DATASET_ROOT = root / "dataset"
        backend.CURRENT_EMBEDDING_VERSION_PATH = root / "models" / "current_embedding_version.json"
        backend.EMBEDDING_SUMMARY_PATH.parent.mkdir(parents=True)
        backend.DATASET_ROOT.mkdir(parents=True)
        self.users = [
            {"id": "admin", "username": "admin", "password": "admin123", "name": "Main Admin", "role": "admin", "roleLabel": "Attendance Controller", "subjects": ["SWE", "CCM", "CVO"], "canSeeAll": True},
            {"id": "vikas", "username": "vikas", "password": "cvo123", "name": "Mr. Vikas B", "role": "faculty", "roleLabel": "Faculty - CVO", "subjects": ["CVO"], "canSeeAll": False},
        ]
        self.mapping = {
            "subjects": {"CVO": {"abbr": "CVO", "courseCode": "24AMLJ502", "courseName": "Computer Vision through OpenCV", "facultyId": "vikas", "facultyName": "Mr. Vikas B"}},
            "subject_students": {"CVO": [
                {"roll": "24011CSEAI0110", "name": "ALUWALA SURYA"},
                {"roll": "2401100CSE0110", "name": "DISTINCT STUDENT"},
                {"roll": "2401100CSE0268", "name": "PAKALAPATI NITHIN"},
            ]},
            "all_students": [
                {"roll": "24011CSEAI0110", "name": "ALUWALA SURYA", "subjects": ["CVO"]},
                {"roll": "2401100CSE0110", "name": "DISTINCT STUDENT", "subjects": ["CVO"]},
                {"roll": "2401100CSE0268", "name": "PAKALAPATI NITHIN", "subjects": ["CVO"]},
            ],
        }
        backend.write_json(backend.USERS_PATH, self.users)
        backend.write_json(backend.STUDENT_MAP_PATH, self.mapping)
        backend.write_json(backend.CURRENT_EMBEDDING_VERSION_PATH, {
            "status": "promoted",
            "family_id": "embfam-test",
            "variant_id": "full_candidate",
            "promotion_id": "promotion-test",
            "promoted_at": "2026-07-17T17:00:00",
        })
        with backend.EMBEDDING_SUMMARY_PATH.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["Roll_Number", "Images_Read", "Faces_Used", "No_Valid_Face", "Multiple_Faces", "Rejected_Crops", "Read_Fail"])
            writer.writeheader()
            writer.writerow({"Roll_Number": "24011CSEAI0110 (A) SURYA", "Images_Read": 12, "Faces_Used": 10, "No_Valid_Face": 2, "Multiple_Faces": 0, "Rejected_Crops": 1, "Read_Fail": 0})
        (backend.DATASET_ROOT / "24011CSEAI0110").mkdir()
        (backend.DATASET_ROOT / "2401100CSE0110").mkdir()
        self.client = backend.app.test_client()
        admin_token, _ = backend.issue_auth_session(self.users[0])
        faculty_token, _ = backend.issue_auth_session(self.users[1])
        self.admin_headers = {"Authorization": f"Bearer {admin_token}", "X-User-Id": "admin"}
        self.faculty_headers = {"Authorization": f"Bearer {faculty_token}", "X-User-Id": "vikas"}

    def tearDown(self):
        for key, value in self.old.items():
            setattr(backend, key, value)
        backend.AUTH_SESSIONS.clear()
        self.temp.cleanup()

    def test_unknown_explicit_identity_never_falls_back_to_hod(self):
        response = self.client.get("/api/auth/me", headers={"X-User-Id": "deleted-user"})
        self.assertEqual(response.status_code, 401)
        response = self.client.get("/api/hod/overview", headers={"X-User-Id": "deleted-user"})
        self.assertEqual(response.status_code, 401)

    def test_backend_login_is_authoritative_and_never_returns_password(self):
        response = self.client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
        self.assertEqual(response.status_code, 200)
        user = response.get_json()["user"]
        self.assertEqual(user["name"], "HOD")
        self.assertEqual(user["roleLabel"], "Head of Department")
        self.assertNotIn("password", user)
        bad = self.client.post("/api/auth/login", json={"username": "admin", "password": "wrong"})
        self.assertEqual(bad.status_code, 401)

    def test_hod_overview_is_forbidden_to_faculty(self):
        response = self.client.get("/api/hod/overview", headers=self.faculty_headers)
        self.assertEqual(response.status_code, 403)

    def test_coverage_keeps_ai0110_and_cse0110_distinct(self):
        with patch.object(backend, "load_timetable_rows", return_value=[{"subject": "CVO", "period": "P1"}]):
            payload = self.client.get("/api/hod/overview", headers=self.admin_headers).get_json()
        rows = {row["roll"]: row for row in payload["students"]}
        self.assertEqual(set(rows), {"24011CSEAI0110", "2401100CSE0110", "2401100CSE0268"})
        self.assertTrue(rows["24011CSEAI0110"]["embedding_available"])
        self.assertFalse(rows["2401100CSE0110"]["embedding_available"])
        self.assertTrue(rows["2401100CSE0110"]["dataset_available"])
        self.assertFalse(rows["2401100CSE0268"]["dataset_available"])
        self.assertEqual(payload["coverage"]["embedding_available"], 1)
        self.assertEqual(payload["production"]["family_id"], "embfam-test")

    def test_policy_matches_current_five_checkpoint_contract(self):
        with patch.object(backend, "load_timetable_rows", return_value=[]):
            policy = backend.hod_overview_payload()["policy"]
        self.assertEqual(policy["checkpoint_count"], 5)
        self.assertEqual(policy["present_checkpoints"], 3)
        self.assertEqual(policy["checkpoint_min_detections"], 2)
        self.assertEqual(policy["match_threshold"], 0.48)
        self.assertEqual(policy["margin_threshold"], 0.08)

    def test_students_endpoint_is_faculty_scoped_and_includes_coverage(self):
        response = self.client.get("/api/students", headers=self.faculty_headers)
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(len(payload["students"]), 3)
        self.assertEqual(payload["coverage"]["embedding_available"], 1)
        self.assertTrue(all(row["subjects"] == ["CVO"] for row in payload["students"]))

    def test_missing_coverage_sources_are_reported_as_unknown_not_missing(self):
        backend.EMBEDDING_SUMMARY_PATH.unlink()
        for path in list(backend.DATASET_ROOT.iterdir()):
            path.rmdir()
        backend.DATASET_ROOT.rmdir()
        rows, summary = backend.student_coverage_rows(self.mapping)
        self.assertFalse(summary["embedding_source_available"])
        self.assertFalse(summary["dataset_source_available"])
        self.assertTrue(all(row["coverage_status"] == "source_unavailable" for row in rows))
        self.assertTrue(all(row["embedding_available"] is None for row in rows))
        self.assertTrue(all(row["dataset_available"] is None for row in rows))


if __name__ == "__main__":
    unittest.main()
