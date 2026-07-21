import unittest

import numpy as np

from src.face_attendance.zones import (
    DetectionCandidate,
    ZoneConfigError,
    box_iou,
    map_resized_face_to_original,
    map_zone_face_to_original,
    merge_detection_candidates,
    parse_camera_zone_config,
    source_summary,
    upscale_zone_crop,
    zone_pixel_bounds,
)


def valid_config():
    return {
        "version": 1,
        "profiles": {
            "back": {
                "aliases": ["cam5", "back.mp4"],
                "expected_width": 2688,
                "expected_height": 1520,
                "zones": [
                    {
                        "id": "rear",
                        "label": "Rear seating",
                        "x": 0.1,
                        "y": 0.2,
                        "width": 0.3,
                        "height": 0.4,
                        "upscale": 2.0,
                        "enabled": True,
                    }
                ],
            }
        },
    }


def candidate(candidate_id, box, source="full_frame", zone_id="", rank=(1, 1, 100, 100, 0.9, 1)):
    face = np.array([
        box[0], box[1], box[2], box[3],
        box[0] + 2, box[1] + 2,
        box[0] + box[2] - 2, box[1] + 2,
        box[0] + box[2] / 2, box[1] + box[3] / 2,
        box[0] + 4, box[1] + box[3] - 2,
        box[0] + box[2] - 4, box[1] + box[3] - 2,
        0.9,
    ], dtype=np.float32)
    return DetectionCandidate(
        candidate_id=candidate_id,
        detection_source=source,
        face_original=face,
        recognition_face=face.copy(),
        recognition_frame=np.zeros((100, 100, 3), dtype=np.uint8),
        recognition_frame_kind="synthetic",
        detector_score=0.9,
        source_priority=2 if source == "zone" else 1,
        zone_id=zone_id,
        quality_rank=rank,
    )


class ZoneConfigTests(unittest.TestCase):
    def test_valid_normalized_zone_configuration_loads(self):
        config = parse_camera_zone_config(valid_config())
        self.assertEqual(config.version, 1)
        profile = config.resolve_profile("cam5", "back.mp4")
        self.assertIsNotNone(profile)
        self.assertEqual(profile.enabled_zones()[0].zone_id, "rear")

    def test_invalid_coordinates_are_rejected(self):
        payload = valid_config()
        payload["profiles"]["back"]["zones"][0]["x"] = 0.9
        payload["profiles"]["back"]["zones"][0]["width"] = 0.2
        with self.assertRaisesRegex(ZoneConfigError, "outside"):
            parse_camera_zone_config(payload)

    def test_duplicate_zone_ids_are_rejected(self):
        payload = valid_config()
        payload["profiles"]["back"]["zones"].append(dict(payload["profiles"]["back"]["zones"][0]))
        with self.assertRaisesRegex(ZoneConfigError, "Duplicate zone id"):
            parse_camera_zone_config(payload)

    def test_missing_profile_handling_is_deterministic(self):
        config = parse_camera_zone_config(valid_config())
        self.assertIsNone(config.resolve_profile("cam99", "unknown.mp4"))

    def test_zone_pixel_bounds_are_calculated_correctly(self):
        config = parse_camera_zone_config(valid_config())
        zone = config.profiles["back"].zones[0]
        self.assertEqual(zone_pixel_bounds(zone, (1520, 2688, 3)), (269, 304, 1075, 912))

    def test_zone_upscaling_preserves_dimensions(self):
        crop = np.zeros((20, 30, 3), dtype=np.uint8)
        upscaled = upscale_zone_crop(crop, 2.0)
        self.assertEqual(upscaled.shape[:2], (40, 60))

    def test_bounding_boxes_map_back_to_original_coordinates(self):
        face = np.array([20, 30, 40, 50, 22, 32, 58, 32, 40, 50, 24, 78, 56, 78, 0.91], dtype=np.float32)
        mapped = map_zone_face_to_original(face, (100, 200, 300, 500), 2.0)
        self.assertAlmostEqual(mapped[0], 110)
        self.assertAlmostEqual(mapped[1], 215)
        self.assertAlmostEqual(mapped[2], 20)
        self.assertAlmostEqual(mapped[3], 25)

    def test_landmarks_map_back_to_original_coordinates(self):
        face = np.array([20, 30, 40, 50, 22, 32, 58, 32, 40, 50, 24, 78, 56, 78, 0.91], dtype=np.float32)
        mapped = map_zone_face_to_original(face, (100, 200, 300, 500), 2.0)
        self.assertAlmostEqual(mapped[4], 111)
        self.assertAlmostEqual(mapped[5], 216)
        self.assertAlmostEqual(mapped[12], 128)
        self.assertAlmostEqual(mapped[13], 239)

    def test_resized_full_frame_face_maps_to_original_coordinates(self):
        face = np.array([10, 20, 30, 40, 15, 25, 35, 25, 25, 35, 18, 55, 32, 55, 0.9], dtype=np.float32)
        mapped = map_resized_face_to_original(face, 0.5)
        self.assertEqual(mapped[0], 20)
        self.assertEqual(mapped[2], 60)
        self.assertEqual(mapped[4], 30)


class ZoneMergeTests(unittest.TestCase):
    def test_overlapping_full_frame_and_zone_detections_merge_once(self):
        full = candidate("a", (10, 10, 40, 40), "full_frame", rank=(1, 1, 1600, 100, 0.9, 1))
        zone = candidate("b", (12, 12, 40, 40), "zone", "rear", rank=(1, 1, 1600, 100, 0.92, 2))
        merged = merge_detection_candidates([full, zone])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].selected_source, "full_frame_and_zone")
        self.assertIn("zone:rear", merged[0].contributing_sources)

    def test_overlapping_zones_do_not_double_count_the_same_face(self):
        first = candidate("a", (10, 10, 40, 40), "zone", "rear_left")
        second = candidate("b", (11, 11, 40, 40), "zone", "rear_right")
        merged = merge_detection_candidates([first, second])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].selected_source, "multiple_zones")

    def test_separate_faces_are_not_merged_incorrectly(self):
        first = candidate("a", (10, 10, 30, 30))
        second = candidate("b", (80, 80, 30, 30))
        self.assertEqual(len(merge_detection_candidates([first, second])), 2)

    def test_merge_order_is_deterministic(self):
        first = candidate("b", (12, 12, 40, 40), "zone", "rear")
        second = candidate("a", (10, 10, 40, 40), "full_frame")
        one = merge_detection_candidates([first, second])[0]
        two = merge_detection_candidates([second, first])[0]
        self.assertEqual(one.merge_group_id, two.merge_group_id)
        self.assertEqual(one.selected_source, two.selected_source)

    def test_highest_quality_representative_is_selected_predictably(self):
        weak = candidate("a", (10, 10, 40, 40), "full_frame", rank=(0, 0, 1600, 10, 0.9, 1))
        strong = candidate("b", (11, 11, 40, 40), "zone", "rear", rank=(1, 2, 1600, 200, 0.88, 2))
        merged = merge_detection_candidates([weak, strong])
        self.assertEqual(merged[0].candidate_id, "b")

    def test_provenance_survives_merging(self):
        full = candidate("a", (10, 10, 40, 40), "full_frame")
        zone = candidate("b", (12, 12, 40, 40), "zone", "rear")
        merged = merge_detection_candidates([full, zone])[0]
        self.assertEqual(merged.merged_detection_count, 2)
        self.assertIn("full_frame", merged.contributing_sources)
        self.assertIn("zone:rear", merged.contributing_sources)
        self.assertGreaterEqual(merged.duplicate_iou, 0.0)

    def test_source_summary_counts_sources(self):
        self.assertEqual(source_summary([
            candidate("a", (10, 10, 40, 40), "full_frame"),
            candidate("b", (80, 80, 40, 40), "zone", "rear"),
        ]), {"full_frame": 1, "zone:rear": 1})

    def test_box_iou_is_deterministic(self):
        first = np.array([10, 10, 40, 40], dtype=np.float32)
        second = np.array([20, 20, 40, 40], dtype=np.float32)
        self.assertAlmostEqual(box_iou(first, second), box_iou(second, first))


if __name__ == "__main__":
    unittest.main()
