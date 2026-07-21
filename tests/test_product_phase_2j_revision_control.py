from __future__ import annotations

import base64
import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.face_attendance.product_phase_2i_authority import AUTHORITY_REVISION_ID, SESSION_ID
from src.face_attendance.report_revision_control import (
    ReportRevisionError,
    POLICY_VERSION,
    _entry_hash,
    attendance_decision_hash,
    is_protected_official,
    preserve_official_or_commit_candidate,
    rollback_mon_p4_repair,
    verify_revision_output,
)


def _rows(present: int, review: int, unconfirmed: int, missing: int, absent: int):
    rows = []
    counter = 1
    for status, count in [
        ("Present", present),
        ("Needs Review", review),
        ("Unconfirmed", unconfirmed),
        ("Missing Enrollment", missing),
        ("Absent", absent),
    ]:
        for _ in range(count):
            rows.append(
                {
                    "Roll_Number": f"R{counter:03d}",
                    "Final_Status": status,
                    "Status": status,
                    "Recognized_Checkpoints": "3" if status == "Present" else "2" if status == "Needs Review" else "0",
                }
            )
            counter += 1
    return rows


class ProductPhase2JRevisionControlTests(unittest.TestCase):
    def test_protected_official_detection(self):
        self.assertTrue(is_protected_official({"authority_revision_id": "reviewed-v1"}))
        self.assertTrue(is_protected_official({"attendance_finalized": True}))
        self.assertTrue(is_protected_official({"report_type": "Edited"}))
        self.assertFalse(is_protected_official({"report_type": "Reprocessed", "attendance_finalized": False}))

    def test_reprocess_candidate_is_archived_and_official_preserved(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            official = {
                "session_id": "S1",
                "authority_revision_id": "reviewed-v1",
                "revision_authority": "official_reviewed",
                "report_type": "Corrected Multi-frame Authority",
                "attendance_data": _rows(6, 4, 16, 1, 0),
            }
            candidate = {
                "session_id": "S1",
                "report_type": "Reprocessed",
                "official_recognition_authority": "strict_tracklet_aggregate_with_guarded_review_candidates",
                "completed_at": "2026-07-20T20:15:13",
                "job_id": "job-one",
                "attendance_data": _rows(2, 8, 16, 1, 0),
            }

            result = preserve_official_or_commit_candidate(root, "S1", official, candidate)
            self.assertTrue(result.official_preserved)
            self.assertEqual(result.disposition, "automatic_candidate_archived_official_preserved")
            self.assertEqual(result.current_entry["attendance_data"], official["attendance_data"])
            self.assertEqual(result.current_entry["pending_candidate_revision"]["present_count"], 2)
            self.assertEqual(result.current_entry["pending_candidate_revision"]["needs_review_count"], 8)
            output_dir = Path(result.current_entry["pending_candidate_revision"]["output_dir"])
            self.assertTrue(output_dir.is_dir())
            verify_revision_output(output_dir)

            rerun = dict(candidate)
            rerun["completed_at"] = "2026-07-20T21:15:13"
            rerun["job_id"] = "job-two"
            reused = preserve_official_or_commit_candidate(root, "S1", result.current_entry, rerun)
            self.assertEqual(reused.candidate_revision["revision_id"], result.candidate_revision["revision_id"])
            self.assertEqual(reused.current_entry["candidate_revision_count"], 1)

    def test_equivalent_reprocess_clears_pending_difference(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            official = {
                "session_id": "S1",
                "authority_revision_id": "reviewed-v1",
                "revision_authority": "official_reviewed",
                "attendance_data": _rows(6, 4, 16, 1, 0),
                "pending_candidate_revision": {"revision_id": "older-different"},
            }
            equivalent_rows = list(reversed(_rows(6, 4, 16, 1, 0)))
            # Incidental strict evidence on a row that remains Unconfirmed must
            # not create a new attendance decision or repeat the same review.
            next(row for row in equivalent_rows if row["Final_Status"] == "Unconfirmed")["Recognized_Checkpoints"] = "1"
            equivalent = {
                "session_id": "S1",
                "report_type": "Reprocessed",
                "official_recognition_authority": "strict_tracklet_aggregate_with_guarded_review_candidates",
                "attendance_data": equivalent_rows,
            }
            self.assertEqual(attendance_decision_hash(official), attendance_decision_hash(equivalent))
            result = preserve_official_or_commit_candidate(root, "S1", official, equivalent)
            self.assertEqual(result.disposition, "equivalent_candidate_archived_official_preserved")
            self.assertTrue(result.current_entry["latest_candidate_matches_official"])
            self.assertNotIn("pending_candidate_revision", result.current_entry)
            self.assertTrue(result.current_entry["latest_equivalent_candidate_revision"]["equivalent_to_official"])

    def test_present_checkpoint_change_is_not_equivalent(self):
        official = {"attendance_data": _rows(1, 0, 1, 0, 0)}
        changed_rows = _rows(1, 0, 1, 0, 0)
        next(row for row in changed_rows if row["Final_Status"] == "Present")["Recognized_Checkpoints"] = "4"
        self.assertNotEqual(
            attendance_decision_hash(official),
            attendance_decision_hash({"attendance_data": changed_rows}),
        )

    def test_unreviewed_candidate_can_become_current(self):
        with TemporaryDirectory() as temp_dir:
            candidate = {
                "session_id": "S2",
                "report_type": "AI",
                "attendance_data": _rows(1, 2, 3, 0, 4),
            }
            result = preserve_official_or_commit_candidate(Path(temp_dir), "S2", None, candidate)
            self.assertFalse(result.official_preserved)
            self.assertEqual(result.current_entry["revision_authority"], "automatic_current")
            self.assertIsNone(result.candidate_revision)

    def test_revision_manifest_detects_tampering(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            official = {
                "session_id": "S1",
                "authority_revision_id": "reviewed-v1",
                "attendance_data": _rows(6, 4, 16, 1, 0),
            }
            candidate = {
                "session_id": "S1",
                "report_type": "Reprocessed",
                "attendance_data": _rows(2, 8, 16, 1, 0),
            }
            result = preserve_official_or_commit_candidate(root, "S1", official, candidate)
            output_dir = Path(result.candidate_revision["output_dir"])
            (output_dir / "candidate_entry.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ReportRevisionError):
                verify_revision_output(output_dir)

    def test_rollback_restores_automatic_entry_and_previous_registry(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data = root / "data"
            backups = data / "state_backups"
            backups.mkdir(parents=True)

            source_entry = {
                "session_id": SESSION_ID,
                "official_recognition_authority": "strict_tracklet_aggregate_with_guarded_review_candidates",
                "attendance_data": _rows(2, 8, 16, 1, 0),
            }
            reviewed_entry = {
                "session_id": SESSION_ID,
                "authority_revision_id": AUTHORITY_REVISION_ID,
                "official_recognition_authority": "reviewed_multiframe_tracklet_evidence",
                "attendance_data": _rows(6, 4, 16, 1, 0),
            }
            (data / "attendance_status.json").write_text(
                json.dumps({SESSION_ID: reviewed_entry}),
                encoding="utf-8",
            )

            previous_registry = b'{"policy_version":"older-registry"}\n'
            installed_registry = (
                json.dumps(
                    {"policy_version": "product-phase-2j-exact-review-carry-forward-v2"},
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
            registry_path = data / "review_evidence_registry.json"
            registry_path.write_bytes(installed_registry)

            backup_path = backups / "phase_2j_before_revision_repair_test.json"
            backup_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "policy_version": POLICY_VERSION,
                        "session_id": SESSION_ID,
                        "source_entry_sha256": _entry_hash(source_entry),
                        "source_entry": source_entry,
                        "registry_existed_before_apply": True,
                        "registry_before_sha256": hashlib.sha256(previous_registry).hexdigest(),
                        "registry_before_base64": base64.b64encode(previous_registry).decode("ascii"),
                        "installed_registry_sha256": hashlib.sha256(installed_registry).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )

            result = rollback_mon_p4_repair(root, backup_path)
            self.assertEqual(result["status"], "rolled_back")
            self.assertEqual(result["registry_action"], "restored_previous_registry")
            restored = json.loads((data / "attendance_status.json").read_text(encoding="utf-8"))
            self.assertEqual(restored[SESSION_ID], source_entry)
            self.assertEqual(registry_path.read_bytes(), previous_registry)

    def test_rollback_rejects_changed_installed_registry(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data = root / "data"
            backups = data / "state_backups"
            backups.mkdir(parents=True)
            source_entry = {"session_id": SESSION_ID, "attendance_data": _rows(2, 8, 16, 1, 0)}
            reviewed_entry = {
                "session_id": SESSION_ID,
                "authority_revision_id": AUTHORITY_REVISION_ID,
                "attendance_data": _rows(6, 4, 16, 1, 0),
            }
            (data / "attendance_status.json").write_text(json.dumps({SESSION_ID: reviewed_entry}), encoding="utf-8")
            expected_registry = b'{"policy_version":"product-phase-2j-exact-review-carry-forward-v2"}\n'
            (data / "review_evidence_registry.json").write_bytes(b'{"policy_version":"tampered"}\n')
            backup_path = backups / "phase_2j_before_revision_repair_test.json"
            backup_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "policy_version": POLICY_VERSION,
                        "session_id": SESSION_ID,
                        "source_entry_sha256": _entry_hash(source_entry),
                        "source_entry": source_entry,
                        "registry_existed_before_apply": False,
                        "registry_before_sha256": None,
                        "registry_before_base64": None,
                        "installed_registry_sha256": hashlib.sha256(expected_registry).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ReportRevisionError):
                rollback_mon_p4_repair(root, backup_path)
            current = json.loads((data / "attendance_status.json").read_text(encoding="utf-8"))
            self.assertEqual(current[SESSION_ID], reviewed_entry)


if __name__ == "__main__":
    unittest.main()
