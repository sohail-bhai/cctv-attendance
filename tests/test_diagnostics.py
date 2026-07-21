from datetime import datetime
import unittest

import numpy as np

from src.face_attendance.diagnostics import (
    checkpoint_window_warnings,
    crop_quality_metrics,
    diagnostic_quality_label,
    diagnostic_rejection_reason,
    face_geometry,
    make_diagnostic_run_id,
    percentile_fields,
    should_log_detection,
)


class DiagnosticHelperTests(unittest.TestCase):
    def test_full_log_mode_records_accepted_and_rejected_detections(self):
        self.assertTrue(should_log_detection("full", True, "accepted"))
        self.assertTrue(should_log_detection("full", False, "score_below_threshold"))

    def test_accepted_and_compact_modes_remain_backward_compatible(self):
        self.assertTrue(should_log_detection("accepted", True, "accepted"))
        self.assertFalse(should_log_detection("accepted", False, "score_below_threshold"))
        self.assertTrue(should_log_detection("compact", False, "margin_too_small"))
        self.assertFalse(should_log_detection("compact", False, "score_below_threshold"))

    def test_face_size_metrics_are_calculated_from_yunet_box(self):
        face = np.array([
            10, 20, 30, 40,
            15, 30, 35, 30, 25, 40, 17, 52, 33, 52,
            0.91,
        ], dtype=np.float32)
        geom = face_geometry(face, (100, 200, 3))
        self.assertEqual(geom["Face_Width"], 30)
        self.assertEqual(geom["Face_Height"], 40)
        self.assertEqual(geom["Face_Area"], 1200)
        self.assertAlmostEqual(geom["Face_Area_Ratio"], 1200 / 20000)
        self.assertEqual(geom["Landmark_Valid"], "Yes")
        self.assertEqual(geom["Crop_Partially_Outside_Frame"], "No")

    def test_outside_box_marks_partial_crop_and_invalid_landmark(self):
        face = np.array([
            -5, 5, 20, 20,
            -1, 10, 8, 10, 6, 15, 2, 20, 10, 20,
            0.88,
        ], dtype=np.float32)
        geom = face_geometry(face, (30, 30, 3))
        self.assertEqual(geom["Crop_Partially_Outside_Frame"], "Yes")
        self.assertEqual(geom["Landmark_Valid"], "No")

    def test_crop_quality_handles_normal_colour_crop(self):
        frame = np.zeros((80, 80, 3), dtype=np.uint8)
        frame[20:60, 20:60] = 180
        frame[30:50, 30:50] = 40
        metrics = crop_quality_metrics(frame, (20, 20, 40, 40))
        self.assertEqual(metrics["Crop_Valid"], "Yes")
        self.assertEqual(metrics["Crop_Width"], 40)
        self.assertEqual(metrics["Crop_Height"], 40)
        self.assertGreater(metrics["Contrast_StdDev"], 0)

    def test_crop_quality_handles_empty_tiny_grayscale_and_colour(self):
        empty = crop_quality_metrics(np.zeros((10, 10), dtype=np.uint8), (20, 20, 5, 5))
        self.assertEqual(empty["Crop_Valid"], "No")

        tiny_gray = crop_quality_metrics(np.ones((1, 1), dtype=np.uint8) * 127, (0, 0, 1, 1))
        self.assertEqual(tiny_gray["Crop_Valid"], "Yes")
        self.assertEqual(tiny_gray["Blur_Laplacian_Variance"], 0.0)

        colour = crop_quality_metrics(np.ones((4, 4, 3), dtype=np.uint8) * 240, (0, 0, 4, 4))
        self.assertEqual(colour["Crop_Valid"], "Yes")
        self.assertEqual(colour["Overexposed_Pct"], 100.0)

    def test_rejection_reasons_are_deterministic_and_distinct(self):
        self.assertEqual(
            diagnostic_rejection_reason(False, "score_below_threshold", True, 0.40, 0.12, 0.48, 0.08),
            "score_below_threshold",
        )
        self.assertEqual(
            diagnostic_rejection_reason(False, "margin_too_small", True, 0.50, 0.03, 0.48, 0.08),
            "margin_below_threshold",
        )
        self.assertEqual(
            diagnostic_rejection_reason(False, "score_below_threshold", True, 0.30, 0.02, 0.48, 0.08),
            "score_and_margin_below_threshold",
        )
        self.assertEqual(
            diagnostic_rejection_reason(False, "embedding_extraction_failure", False, 0, 0, 0.48, 0.08),
            "embedding_extraction_failure",
        )

    def test_diagnostic_reason_does_not_change_accepted_decision(self):
        reason = diagnostic_rejection_reason(True, "accepted", True, 0.49, 0.08, 0.48, 0.08)
        self.assertEqual(reason, "accepted")

    def test_quality_label_warns_without_operational_rejection(self):
        face = np.array([
            1, 1, 20, 20,
            3, 3, 12, 3, 8, 10, 4, 18, 14, 18,
            0.86,
        ], dtype=np.float32)
        geometry = face_geometry(face, (200, 200, 3))
        quality = crop_quality_metrics(np.ones((200, 200, 3), dtype=np.uint8) * 128, (1, 1, 20, 20))
        label, reasons = diagnostic_quality_label(geometry, quality, detector_score=0.86)
        self.assertEqual(label, "unusable_reference")
        self.assertIn("face_too_small_for_diagnostic_reference", reasons)
        self.assertIn("low_detector_confidence", reasons)

    def test_diagnostic_output_paths_are_unique(self):
        first = make_diagnostic_run_id("2026-06-30__B51__P1__CVO", datetime(2026, 7, 11, 10, 0, 0, 1))
        second = make_diagnostic_run_id("2026-06-30__B51__P1__CVO", datetime(2026, 7, 11, 10, 0, 0, 2))
        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith("2026-06-30__B51__P1__CVO_"))

    def test_short_clip_checkpoint_handling_is_explicit(self):
        valid, warnings = checkpoint_window_warnings(
            "clip-folders",
            source_duration=18.2,
            selected_start=0.0,
            selected_end=18.2,
            requested_start=600.0,
            requested_end=620.0,
        )
        self.assertTrue(valid)
        self.assertIn("clip_folder_uses_video_relative_window", warnings)
        self.assertIn("selected_window_clamped_to_source_duration", warnings)

    def test_invalid_checkpoint_windows_are_not_silent(self):
        valid, warnings = checkpoint_window_warnings("class-time", 12.0, 20.0, 22.0, 20.0, 22.0)
        self.assertFalse(valid)
        self.assertIn("selected_window_starts_after_source_duration", warnings)

    def test_cp5_end_of_video_clamping_is_deterministic(self):
        valid, warnings = checkpoint_window_warnings("class-time", 2990.0, 2980.0, 2990.0, 2980.0, 3000.0)
        self.assertTrue(valid)
        self.assertIn("selected_window_clamped_to_source_duration", warnings)

    def test_percentiles_have_stable_empty_and_nonempty_shapes(self):
        self.assertEqual(percentile_fields([], "Face_Width")["Face_Width_Median"], "")
        fields = percentile_fields([10, 20, 30], "Face_Width", decimals=1)
        self.assertEqual(fields["Face_Width_Min"], 10.0)
        self.assertEqual(fields["Face_Width_Median"], 20.0)


if __name__ == "__main__":
    unittest.main()
