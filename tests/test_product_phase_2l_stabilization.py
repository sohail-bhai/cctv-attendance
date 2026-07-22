from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.face_attendance import product_phase_2l_stabilization as stabilization
from src.face_attendance.processing_integration import FIXED_POLICY
from src.face_attendance.product_phase_2l_stabilization import (
    DEFAULT_CONTRACT,
    DOCUMENTATION_PATHS,
    EXPECTED_CANDIDATE_COUNTS,
    EXPECTED_OFFICIAL_COUNTS,
    REQUIRED_EXTERNAL_VALIDATIONS,
    ROLLBACK_PATHS,
    StabilizationError,
    build_output_files,
    canonical_json_hash,
    contract_with_hash,
    materialize,
    no_activation_declaration,
    no_recognition_declaration,
    preflight,
    run_release_contract_tests,
    sha256_file,
    validate_recommendation,
    validate_review_registry,
    validate_roster_and_enrollment,
    verify_immutable_output,
    write_immutable_output,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SESSION_ID = stabilization.SESSION_ID
AUDIT_RELATIVE = Path(
    "attendance_output/product_workflow/phase_2k_retrospective_audit"
) / stabilization.PHASE_2K_C1_RUN_ID
OFFICIAL_RELATIVE = Path(
    "attendance_output/product_workflow/phase_2i_authority/"
    "authority-revision-2d83c4f679f839a6c784c636/"
    f"corrected_attendance_{SESSION_ID}.csv"
)
CANDIDATE_RELATIVE = Path(
    "attendance_output/product_workflow/report_revisions"
) / SESSION_ID / "automatic-candidate-99cc13934144bf7e1706b8be"
VERSION_MANIFEST_RELATIVE = Path(
    "models/promotion_history/promotion-7cc01070e9bc644380c5/"
    "promotion_evidence/candidate_version_manifest.json"
)
SOURCE_RELATIVES = (
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


def _copy_file(source_root: Path, target_root: Path, relative: str | Path) -> None:
    source = source_root / relative
    target = target_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mutate_first_status(path: Path) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    current = rows[0].get("Status") or rows[0].get("Final_Status") or rows[0].get("Present")
    replacement = "Absent" if current != "Absent" else "Present"
    if "Status" in rows[0]:
        rows[0]["Status"] = replacement
    elif "Final_Status" in rows[0]:
        rows[0]["Final_Status"] = replacement
    else:
        rows[0]["Present"] = replacement
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@contextmanager
def isolated_fixture():
    with tempfile.TemporaryDirectory(prefix="phase2l-fixture-") as temp:
        root = Path(temp)
        for relative in DEFAULT_CONTRACT.protected_hashes:
            _copy_file(REPO_ROOT, root, relative)
        shutil.copytree(REPO_ROOT / AUDIT_RELATIVE, root / AUDIT_RELATIVE)
        _copy_file(REPO_ROOT, root, OFFICIAL_RELATIVE)
        shutil.copytree(REPO_ROOT / CANDIDATE_RELATIVE, root / CANDIDATE_RELATIVE)
        _copy_file(REPO_ROOT, root, VERSION_MANIFEST_RELATIVE)
        for relative in ROLLBACK_PATHS.values():
            _copy_file(REPO_ROOT, root, relative)
        for relative in DOCUMENTATION_PATHS:
            _copy_file(REPO_ROOT, root, relative)
        for relative in SOURCE_RELATIVES:
            _copy_file(REPO_ROOT, root, relative)
        yield root


class Phase2LStabilizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = preflight(REPO_ROOT, run_tests=False)
        cls.passed_plan = replace(
            cls.plan,
            test_validation={
                "frontend_status_contract_tests": {"passed": True, "status": "test_fixture"},
                "backend_relevant_tests": {"passed": True, "status": "test_fixture"},
            },
        )

    def test_exact_current_preflight_passes(self) -> None:
        plan = preflight(REPO_ROOT, run_tests=False)
        self.assertEqual(plan.official_counts, EXPECTED_OFFICIAL_COUNTS)
        self.assertEqual(plan.candidate_counts, EXPECTED_CANDIDATE_COUNTS)
        self.assertEqual(plan.roster["count"], 27)
        self.assertTrue(plan.review_registry["mixed_quarantined"])
        self.assertEqual(len(plan.rollback_inventory), 5)
        self.assertEqual(plan.run_payload["source_manifest_sha256"], canonical_json_hash(list(plan.source_manifest)))

    def test_embedding_hash_mismatch_fails_closed(self) -> None:
        with isolated_fixture() as root:
            (root / "models/student_embeddings.pkl").write_bytes(b"tampered")
            with self.assertRaisesRegex(StabilizationError, "embedding_hash_mismatch"):
                preflight(root)

    def test_summary_hash_mismatch_fails_closed(self) -> None:
        with isolated_fixture() as root:
            (root / "models/embedding_summary.csv").write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(StabilizationError, "summary_hash_mismatch"):
                preflight(root)

    def test_threshold_mismatch_fails_closed(self) -> None:
        policy = dict(FIXED_POLICY)
        policy["match_threshold"] = 0.47
        with self.assertRaisesRegex(StabilizationError, "threshold_mismatch"):
            preflight(REPO_ROOT, policy_override=policy)

    def test_authority_mismatch_fails_closed(self) -> None:
        policy = dict(FIXED_POLICY)
        policy["official_recognition_authority"] = "unsafe_frame_level"
        with self.assertRaisesRegex(StabilizationError, "authority_mismatch"):
            preflight(REPO_ROOT, policy_override=policy)

    def test_guarded_recovery_cannot_be_automatic(self) -> None:
        relative = "data/attendance_status.json"
        with isolated_fixture() as root:
            path = root / relative
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload[SESSION_ID]["guarded_recovery_automatic"] = True
            _write_json(path, payload)
            contract = contract_with_hash(DEFAULT_CONTRACT, relative, sha256_file(path))
            with self.assertRaisesRegex(StabilizationError, "guarded_recovery_accidentally_automatic"):
                preflight(root, contract=contract)

    def test_mixed_track_must_remain_quarantined(self) -> None:
        relative = "data/review_evidence_registry.json"
        with isolated_fixture() as root:
            path = root / relative
            payload = json.loads(path.read_text(encoding="utf-8"))
            mixed = next(row for row in payload["evidence"] if row.get("tracklet_id") == stabilization.MIXED_TRACKLET_ID)
            mixed["tracklet_id"] = "not-the-required-mixed-track"
            _write_json(path, payload)
            contract = contract_with_hash(DEFAULT_CONTRACT, relative, sha256_file(path))
            with self.assertRaisesRegex(StabilizationError, "mixed_track_missing_from_quarantine"):
                preflight(root, contract=contract)

    def test_official_report_total_mismatch_fails_closed(self) -> None:
        with isolated_fixture() as root:
            path = root / OFFICIAL_RELATIVE
            _mutate_first_status(path)
            contract = replace(DEFAULT_CONTRACT, official_report_sha256=sha256_file(path))
            with self.assertRaisesRegex(StabilizationError, "official_totals_mismatch"):
                preflight(root, contract=contract)

    def test_candidate_report_total_mismatch_fails_closed(self) -> None:
        with isolated_fixture() as root:
            report = root / CANDIDATE_RELATIVE / "candidate_attendance.csv"
            manifest_path = root / CANDIDATE_RELATIVE / "revision_manifest.json"
            _mutate_first_status(report)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files"]["candidate_attendance.csv"] = {
                "sha256": sha256_file(report),
                "size_bytes": report.stat().st_size,
            }
            _write_json(manifest_path, manifest)
            contract = replace(
                DEFAULT_CONTRACT,
                candidate_report_sha256=sha256_file(report),
                candidate_manifest_sha256=sha256_file(manifest_path),
            )
            with self.assertRaisesRegex(StabilizationError, "candidate_totals_mismatch"):
                preflight(root, contract=contract)

    def test_roster_count_mismatch_fails_closed(self) -> None:
        relative = "data/student_faculty_map.json"
        with isolated_fixture() as root:
            path = root / relative
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["subject_students"]["CVO"].pop()
            _write_json(path, payload)
            contract = contract_with_hash(DEFAULT_CONTRACT, relative, sha256_file(path))
            with self.assertRaisesRegex(StabilizationError, "roster_count_mismatch"):
                preflight(root, contract=contract)

    def test_missing_enrollment_mismatch_fails_closed(self) -> None:
        student_map = json.loads((REPO_ROOT / "data/student_faculty_map.json").read_text(encoding="utf-8"))
        roster = [row["roll"] for row in student_map["subject_students"]["CVO"]]
        with (REPO_ROOT / "models/embedding_summary.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            enrollment = [row["Roll_Number"] for row in csv.DictReader(handle) if int(row["Faces_Used"]) > 0]
        enrollment.append(stabilization.MISSING_ENROLLMENT_ROLL)
        with self.assertRaisesRegex(StabilizationError, "missing_enrollment_mismatch"):
            validate_roster_and_enrollment(roster, enrollment)

    def test_review_registry_count_mismatch_fails_closed(self) -> None:
        payload = json.loads((REPO_ROOT / "data/review_evidence_registry.json").read_text(encoding="utf-8"))
        payload["accepted_pairs"] = 31
        with self.assertRaisesRegex(StabilizationError, "review_registry_count_mismatch"):
            validate_review_registry(payload)

    def test_finalized_state_mismatch_fails_closed(self) -> None:
        relative = "data/attendance_status.json"
        with isolated_fixture() as root:
            path = root / relative
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload[SESSION_ID]["attendance_finalized"] = True
            _write_json(path, payload)
            contract = contract_with_hash(DEFAULT_CONTRACT, relative, sha256_file(path))
            with self.assertRaisesRegex(StabilizationError, "finalized_state_mismatch"):
                preflight(root, contract=contract)

    def test_recommendation_mismatch_fails_closed(self) -> None:
        recommendation = json.loads((REPO_ROOT / AUDIT_RELATIVE / "recommendation.json").read_text(encoding="utf-8"))
        recommendation["recommendation"] = "promote"
        with self.assertRaisesRegex(StabilizationError, "phase_2k_c1_recommendation_mismatch"):
            validate_recommendation(recommendation)

    def test_tampered_phase_2k_c1_artifact_fails_closed(self) -> None:
        with isolated_fixture() as root:
            path = root / AUDIT_RELATIVE / "recommendation.json"
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(StabilizationError, "manifest_file_tampered"):
                preflight(root)

    def test_missing_rollback_reference_fails_closed(self) -> None:
        with isolated_fixture() as root:
            (root / ROLLBACK_PATHS["phase_2i_state_backup"]).unlink()
            with self.assertRaisesRegex(StabilizationError, "rollback_reference_missing:phase_2i_state_backup"):
                preflight(root)

    def test_preflight_leaves_protected_state_unchanged(self) -> None:
        before = {relative: sha256_file(REPO_ROOT / relative) for relative in DEFAULT_CONTRACT.protected_hashes}
        preflight(REPO_ROOT, run_tests=False)
        after = {relative: sha256_file(REPO_ROOT / relative) for relative in DEFAULT_CONTRACT.protected_hashes}
        self.assertEqual(after, before)

    def test_contract_test_commands_do_not_invoke_recognition_or_video(self) -> None:
        commands = []

        def fake_run(command, **_kwargs):
            commands.append(list(command))
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch.object(stabilization.subprocess, "run", side_effect=fake_run):
            result = run_release_contract_tests(REPO_ROOT)
        self.assertTrue(all(item["passed"] for item in result.values()))
        self.assertEqual([command[0] for command in commands], ["node", sys.executable])
        joined = " ".join(" ".join(command[1:]) for command in commands).lower()
        for forbidden in ("yunet", "sface", "videocapture", "process_session"):
            self.assertNotIn(forbidden, joined)
        module_source = (REPO_ROOT / "src/face_attendance/product_phase_2l_stabilization.py").read_text(encoding="utf-8")
        self.assertNotIn("import cv2", module_source)
        self.assertNotIn("FaceDetectorYN", module_source)
        self.assertNotIn("FaceRecognizerSF", module_source)

    def test_declarations_deny_recognition_and_activation(self) -> None:
        recognition = no_recognition_declaration()
        activation = no_activation_declaration()
        self.assertTrue(all(value is False for key, value in recognition.items() if key.endswith("_ran") or key in {"video_decoded", "session_reprocessed", "recognition_imported_or_invoked"}))
        self.assertTrue(all(value is False for key, value in activation.items() if key != "schema_version"))

    def test_output_bytes_are_deterministic(self) -> None:
        first = build_output_files(self.passed_plan, REQUIRED_EXTERNAL_VALIDATIONS)
        second = build_output_files(self.passed_plan, tuple(reversed(REQUIRED_EXTERNAL_VALIDATIONS)))
        self.assertEqual(first, second)

    def test_materialization_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase2l-output-") as temp:
            plan = replace(self.passed_plan, output_root=Path(temp))
            first_dir, first_reused, first_manifest = materialize(plan, REQUIRED_EXTERNAL_VALIDATIONS)
            second_dir, second_reused, second_manifest = materialize(plan, REQUIRED_EXTERNAL_VALIDATIONS)
            self.assertFalse(first_reused)
            self.assertTrue(second_reused)
            self.assertEqual(first_dir, second_dir)
            self.assertEqual(first_manifest, second_manifest)

    def test_manifest_tamper_is_detected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase2l-output-") as temp:
            plan = replace(self.passed_plan, output_root=Path(temp))
            output, _reused, _manifest = materialize(plan, REQUIRED_EXTERNAL_VALIDATIONS)
            path = output / "evaluation_summary.json"
            path.write_bytes(path.read_bytes() + b"tamper")
            with self.assertRaisesRegex(StabilizationError, "immutable_output_tampered:evaluation_summary.json"):
                verify_immutable_output(output)

    def test_deterministic_id_collision_is_detected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase2l-output-") as temp:
            plan = replace(self.passed_plan, output_root=Path(temp))
            files = build_output_files(plan, REQUIRED_EXTERNAL_VALIDATIONS)
            write_immutable_output(plan, files)
            changed = dict(files)
            changed["evaluation_summary.json"] = changed["evaluation_summary.json"] + b" "
            with self.assertRaisesRegex(StabilizationError, "deterministic_id_collision:evaluation_summary.json"):
                write_immutable_output(plan, changed)


if __name__ == "__main__":
    unittest.main()
