from __future__ import annotations

import json
import secrets
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import app as backend
from src.face_attendance.product_phase_2l_authentication import (
    AuthenticationHardeningError,
    POLICY_VERSION,
    REQUIRED_EXTERNAL_VALIDATIONS,
    REQUIRED_OUTPUT_FILES,
    frontend_auth_contract,
    materialize,
    preflight,
    route_inventory,
    run_no_token_matrix,
    run_role_matrix,
    verify_immutable_output,
)


ROOT = Path(__file__).resolve().parents[1]


class ExplicitAuthenticationTests(unittest.TestCase):
    def setUp(self):
        backend.AUTH_SESSIONS.clear()
        self.client = backend.app.test_client()
        self.hod = {"id": "synthetic-hod", "name": "Synthetic HOD", "role": "admin", "subjects": ["SWE", "CCM", "CVO"], "canSeeAll": True}
        self.faculty = {"id": "synthetic-cvo", "name": "Synthetic Faculty", "role": "faculty", "subjects": ["CVO"], "canSeeAll": False}
        self.other_faculty = {"id": "synthetic-swe", "name": "Synthetic SWE Faculty", "role": "faculty", "subjects": ["SWE"], "canSeeAll": False}
        self.login_secret = secrets.token_urlsafe(24)
        self.hod["username"] = self.hod["id"]
        self.hod["password"] = self.login_secret
        self.users = [self.hod, self.faculty, self.other_faculty]
        self.users_patch = patch.object(backend, "load_role_users", return_value=self.users)
        self.users_patch.start()
        self.hod_token, _ = backend.issue_auth_session(self.hod)
        self.faculty_token, _ = backend.issue_auth_session(self.faculty)
        self.other_faculty_token, _ = backend.issue_auth_session(self.other_faculty)

    def tearDown(self):
        self.users_patch.stop()
        backend.AUTH_SESSIONS.clear()

    @staticmethod
    def headers(token: str, user_id: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}", "X-User-Id": user_id}

    def test_public_allowlist_is_minimal_and_health_has_no_private_paths(self):
        self.assertEqual(backend.AUTHENTICATION_POLICY_VERSION, POLICY_VERSION)
        self.assertEqual(backend.PUBLIC_ROUTE_ALLOWLIST, frozenset({
            ("GET", "/"),
            ("GET", "/api/health"),
            ("POST", "/api/auth/login"),
            ("GET", "/static/<path:filename>"),
        }))
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertNotIn("missing_required_files", payload)
        self.assertNotIn("workflow_state", payload)
        self.assertNotIn("processing_policy", payload)
        self.assertEqual(payload["authentication_policy_version"], POLICY_VERSION)
        self.assertEqual(self.client.get("/").status_code, 200)
        invalid = self.client.post("/api/auth/login", json={"username": "missing-synthetic-user", "password": secrets.token_urlsafe(24)})
        self.assertEqual(invalid.status_code, 401)
        self.assertEqual(invalid.get_json(), {"success": False, "error": "Invalid username or password."})

    def test_login_contract_logout_and_revocation(self):
        login = self.client.post("/api/auth/login", json={"username": self.hod["id"], "password": self.login_secret})
        self.assertEqual(login.status_code, 200)
        payload = login.get_json()
        self.assertTrue(payload.get("session_token"))
        token = payload["session_token"]
        headers = {"Authorization": f"Bearer {token}"}
        self.assertEqual(self.client.get("/api/auth/me", headers=headers).status_code, 200)
        self.assertEqual(self.client.post("/api/auth/logout", headers=headers).status_code, 200)
        self.assertEqual(self.client.get("/api/auth/me", headers=headers).status_code, 401)

    def test_missing_role_registry_fails_closed_without_creating_accounts(self):
        self.users_patch.stop()
        try:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                users_path = root / "missing-role-users.json"
                student_map_path = root / "student-map.json"
                student_map_path.write_text("{}\n", encoding="utf-8")
                with patch.object(backend, "USERS_PATH", users_path), patch.object(backend, "STUDENT_MAP_PATH", student_map_path):
                    self.assertEqual(backend.load_role_users(), [])
                    self.assertFalse(users_path.exists())
                    self.assertEqual(backend.timetable_for_day("Monday", None), [])
        finally:
            self.users_patch.start()

    def test_x_user_id_or_query_identity_alone_cannot_authenticate(self):
        for headers, path in [({"X-User-Id": "synthetic-hod"}, "/api/auth/me"), ({}, "/api/auth/me?user_id=synthetic-hod")]:
            with self.subTest(headers=headers, path=path):
                response = self.client.get(path, headers=headers)
                self.assertEqual(response.status_code, 401)
                self.assertEqual(response.get_json(), {"success": False, "error": "Authentication required."})

    def test_missing_invalid_malformed_and_empty_bearer_fail_401(self):
        headers = ({}, {"Authorization": "invalid"}, {"Authorization": "Bearer"}, {"Authorization": "Bearer "}, {"Authorization": "Bearer invalid-token"})
        for item in headers:
            with self.subTest(headers=item):
                response = self.client.get("/api/profile", headers=item)
                self.assertEqual(response.status_code, 401)
                self.assertEqual(set(response.get_json()), {"success", "error"})

    def test_bearer_succeeds_only_with_matching_optional_identity_hint(self):
        response = self.client.get("/api/auth/me", headers=self.headers(self.hod_token, self.hod["id"]))
        self.assertEqual(response.status_code, 200)
        bearer_only = self.client.get("/api/auth/me", headers={"Authorization": f"Bearer {self.hod_token}"})
        self.assertEqual(bearer_only.status_code, 200)
        mismatch = self.client.get("/api/auth/me", headers=self.headers(self.hod_token, self.faculty["id"]))
        self.assertEqual(mismatch.status_code, 401)

    def test_authenticated_faculty_is_forbidden_from_hod_inventory(self):
        response = self.client.get("/api/reports", headers=self.headers(self.faculty_token, self.faculty["id"]))
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.get_json(), {"success": False, "error": "Access denied."})

    def test_report_download_is_scoped_to_assigned_faculty_session(self):
        ref = "product_workflow/phase_2i_authority/authority-revision-2d83c4f679f839a6c784c636/corrected_attendance_2026-06-22__B51__P4__CVO.csv"
        assigned = self.client.get(f"/api/download/{ref}", headers=self.headers(self.faculty_token, self.faculty["id"]))
        try:
            self.assertEqual(assigned.status_code, 200)
        finally:
            assigned.close()
        denied = self.client.get(f"/api/download/{ref}", headers=self.headers(self.other_faculty_token, self.other_faculty["id"]))
        try:
            self.assertEqual(denied.status_code, 403)
            self.assertEqual(denied.get_json(), {"success": False, "error": "Access denied."})
        finally:
            denied.close()
        unknown = self.client.get("/api/download/product_workflow/nonexistent-CVO-report.csv", headers=self.headers(self.other_faculty_token, self.other_faculty["id"]))
        self.assertEqual(unknown.status_code, 403)
        self.assertEqual(unknown.get_json(), {"success": False, "error": "Access denied."})

    def test_options_is_protocol_only_without_authentication(self):
        response = self.client.open("/api/reports", method="OPTIONS")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_data(), b"")

    def test_every_registered_protected_route_rejects_no_token(self):
        inventory = route_inventory(backend)
        rows = run_no_token_matrix(backend, inventory)
        self.assertGreaterEqual(len(rows), 35)
        self.assertTrue(all(row["status"] == "401" and row["result"] == "pass" for row in rows))

    def test_synthetic_role_matrix_passes(self):
        rows = run_role_matrix(backend, ROOT)
        self.assertTrue(all(row["result"] == "pass" for row in rows))
        self.assertTrue(any(row["role"] == "Faculty-SWE" and row["actual_status"] == "403" for row in rows))


class AuthenticationEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = preflight(ROOT)

    def test_frontend_contract_has_no_identity_query_fallback(self):
        contract = frontend_auth_contract(ROOT)
        self.assertTrue(contract["passed"])
        self.assertTrue(contract["checks"]["no_identity_query_fallback"])

    def test_materialization_is_deterministic_and_exact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            build = root / "build"
            build.mkdir()
            (build / "index.html").write_text("synthetic frontend build", encoding="utf-8")
            plan = replace(self.plan, output_root=root / "output")
            first, reused, manifest = materialize(plan, REQUIRED_EXTERNAL_VALIDATIONS, build)
            self.assertFalse(reused)
            second, reused, repeated = materialize(plan, REQUIRED_EXTERNAL_VALIDATIONS, build)
            self.assertTrue(reused)
            self.assertEqual(first, second)
            self.assertEqual(manifest, repeated)
            self.assertEqual(sorted(path.name for path in first.iterdir()), sorted(REQUIRED_OUTPUT_FILES))
            self.assertEqual(verify_immutable_output(first)["run_id"], manifest["run_id"])

    def test_deterministic_collision_and_tamper_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            build = root / "build"
            build.mkdir()
            (build / "index.html").write_text("synthetic frontend build", encoding="utf-8")
            plan = replace(self.plan, output_root=root / "output")
            output, _, _ = materialize(plan, REQUIRED_EXTERNAL_VALIDATIONS, build)
            changed_contract = dict(plan.frontend_contract)
            changed_contract["synthetic_collision_probe"] = True
            changed_plan = replace(plan, frontend_contract=changed_contract)
            with self.assertRaisesRegex(AuthenticationHardeningError, "deterministic_id_collision"):
                materialize(changed_plan, REQUIRED_EXTERNAL_VALIDATIONS, build)
            policy = output / "authentication_policy.json"
            policy.write_text(json.dumps({"tampered": True}), encoding="utf-8")
            with self.assertRaisesRegex(AuthenticationHardeningError, "immutable_output_tampered"):
                verify_immutable_output(output)


if __name__ == "__main__":
    unittest.main()
