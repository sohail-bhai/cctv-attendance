import json
import tempfile
import unittest
from pathlib import Path

from src.face_attendance.roster_integrity import audit_subject_roster


ROOT = Path(__file__).resolve().parents[1]


class RosterIntegrityTests(unittest.TestCase):
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
        self.assertTrue(summary["integrity_passed"], summary["errors"])
        self.assertEqual(summary["roster_students"], 27)
        self.assertEqual(summary["actual_present_students"], 27)
        self.assertEqual(summary["embedding_available"], 25)
        self.assertEqual(
            summary["missing_embedding_rolls"],
            ["2401100CSE0268", "24011CSEAI0110"],
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
