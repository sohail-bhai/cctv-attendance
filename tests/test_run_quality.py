import unittest

from src.face_attendance.run_quality import (
    LOW_QUALITY_STATUS,
    VALID_STATUS,
    evaluate_run_quality,
    quality_from_slot_summary,
    quality_to_csv_fields,
    review_queue_reason,
)


class RunQualityTests(unittest.TestCase):
    def test_healthy_run_remains_valid(self):
        quality = evaluate_run_quality(
            total_students=26,
            total_checkpoints=5,
            detected_faces=180,
            accepted_recognitions=85,
            unique_students_recognized=18,
            poor_checkpoints=[],
            present_count=16,
            review_count=2,
            absent_count=8,
        )
        self.assertEqual(quality.status, VALID_STATUS)
        self.assertTrue(quality.attendance_finalized)
        self.assertFalse(quality.requires_manual_review)

    def test_many_faces_with_almost_no_recognition_needs_review(self):
        quality = evaluate_run_quality(
            total_students=26,
            total_checkpoints=5,
            detected_faces=191,
            accepted_recognitions=15,
            unique_students_recognized=1,
            poor_checkpoints=["CP1", "CP2", "CP3", "CP4"],
            present_count=0,
            review_count=0,
            absent_count=26,
            dominant_identity_roll="2401100CSE0052",
            dominant_identity_count=15,
        )
        self.assertEqual(quality.status, LOW_QUALITY_STATUS)
        self.assertTrue(quality.requires_manual_review)
        self.assertFalse(quality.attendance_finalized)
        self.assertIn("mass absence", quality.reason)

    def test_low_quality_metadata_is_reported_to_csv(self):
        quality = evaluate_run_quality(
            total_students=26,
            total_checkpoints=5,
            detected_faces=191,
            accepted_recognitions=15,
            unique_students_recognized=1,
            poor_checkpoints=["CP1", "CP2", "CP3", "CP4"],
            present_count=0,
            review_count=0,
            absent_count=26,
        )
        fields = quality_to_csv_fields(quality)
        self.assertEqual(fields["Requires_Manual_Review"], "Yes")
        self.assertEqual(fields["Attendance_Finalized"], "No")
        self.assertEqual(fields["Detected_Faces"], 191)

    def test_old_summary_without_quality_metadata_is_inferred(self):
        quality = quality_from_slot_summary({
            "Total_Students": 26,
            "Total_Checkpoints": 5,
            "Total_Face_Detections": 191,
            "Accepted_Recognitions": 15,
            "Students_With_At_Least_One_Detection": 1,
            "Students_Present": 0,
            "Students_Needs_Review": 0,
            "Students_Absent": 26,
            "Poor_Checkpoints": "CP1, CP2, CP3, CP4",
        })
        self.assertEqual(quality.status, LOW_QUALITY_STATUS)
        self.assertFalse(quality.attendance_finalized)

    def test_review_queue_reason_explains_flagged_absent_students(self):
        reason = review_queue_reason("Absent", "Single Camera Evidence", False)
        self.assertIn("Flagged for review despite raw outcome Absent", reason)


if __name__ == "__main__":
    unittest.main()
