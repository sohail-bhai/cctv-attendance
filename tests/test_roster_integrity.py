import json
import tempfile
import unittest
from pathlib import Path

from src.face_attendance.roster_integrity import audit_subject_roster


ROOT = Path(__file__).resolve().parents[1]
PROMOTED_FAMILY_ID = "embfam-274b5207b8b71294ff75"
PROMOTION_ID = "promotion-7cc01070e9bc644380c5"
BASELINE_MISSING = ["2401100CSE0268", "24011CSEAI0110"]
PROMOTED_MISSING = ["2401100CSE0268"]


class RosterIntegrityTests(unittest.TestCase):
    def _expected_repository_embedding_coverage(self) -> tuple[int, list[str]]:
        """Resolve the exact valid coverage contract from the lifecycle pointer.

        The roster itself is immutable at 27 students. Embedding coverage is allowed
        to be either the protected parent state (25 covered) or the explicitly
        promoted Phase 1.2O state (26 covered). Any other pointer state fails closed.
        """

        pointer_path = ROOT / "models" / "current_embedding_version.json"
        if not pointer_path.is_file():
            return 25, BASELINE_MISSING

        pointer = json.loads(pointer_path.read_text(encoding="utf-8-sig"))
        status = str(pointer.get("status") or "").strip()
        promotion_id = str(pointer.get("promotion_id") or "").strip()

        if status == "promoted":
            self.assertEqual(pointer.get("family_id"), PROMOTED_FAMILY_ID)
            self.assertEqual(promotion_id, PROMOTION_ID)
            return 26, PROMOTED_MISSING

        if status == "rolled_back_to_parent":
            self.assertEqual(promotion_id, PROMOTION_ID)
            return 25, BASELINE_MISSING

        self.fail(f"Unsupported current embedding lifecycle status: {status!r}")

    def test_repository_cvo_roster_is_locked_to_confirmed_27_students(self):
        result = audit_subject_roster(
            ROOT / "data" / "student_faculty_map.json",
            "CVO",
            actual_present_path=ROOT / "actual_present" / "TUE_P1_actual_present_template.txt",
            embedding_summary_path=ROOT / "models" / "embedding_summary.csv",
            embeddings_path=ROOT / "models" / "student_embeddings.pkl",
        )
        summary = result.summary
        rolls = set(result.roster["Canonical_Roll"])
        expected_available, expected_missing = self._expected_repository_embedding_coverage()

        self.assertTrue(summary["integrity_passed"], summary["errors"])
        self.assertEqual(summary["roster_students"], 27)
        self.assertEqual(summary["actual_present_students"], 27)
        self.assertEqual(summary["embedding_available"], expected_available)
        self.assertEqual(summary["embedding_missing"], len(expected_missing))
        self.assertEqual(summary["missing_embedding_rolls"], expected_missing)

        coverage = result.embedding_coverage.set_index("Canonical_Roster_Roll")
        self.assertEqual(coverage.loc["2401100CSE0268", "Embedding_Available"], "No")
        self.assertEqual(
            coverage.loc["24011CSEAI0110", "Embedding_Available"],
            "Yes" if expected_available == 26 else "No",
        )

        self.assertIn("2401100CSE0184", rolls)
        self.assertIn("2401100CSE0237", rolls)
        self.assertIn("2401100CSE0268", rolls)
        self.assertIn("24011CSEAI0110", rolls)
        self.assertNotIn("2401100CSE0110", rolls)
        self.assertNotIn("24011CSEAI0061", rolls)

    def test_audit_reports_stale_membership_and_actual_present_drift(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mapping = root / "mapping.json"
            mapping.write_text(json.dumps({
                "subject_students": {
                    "CVO": [{"roll": "A1", "name": "Student One"}],
                },
                "all_students": [
                    {"roll": "A1", "name": "Student One", "subjects": []},
                    {"roll": "A2", "name": "Student Two", "subjects": ["CVO"]},
                ],
            }), encoding="utf-8")
            actual = root / "actual.txt"
            actual.write_text("A2\n", encoding="utf-8")

            result = audit_subject_roster(mapping, "CVO", actual_present_path=actual)
            self.assertFalse(result.summary["integrity_passed"])
            self.assertEqual(result.summary["missing_subject_membership"], ["A1"])
            self.assertEqual(result.summary["stale_subject_membership"], ["A2"])
            self.assertEqual(result.summary["missing_from_actual_present"], ["A1"])
            self.assertEqual(result.summary["actual_present_not_in_roster"], ["A2"])


if __name__ == "__main__":
    unittest.main()
