import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.face_attendance.tracklet_review import (
    LabelValidationError,
    ReviewExportError,
    _OpenCVFrameLoader,
    evaluate_review_package,
    evaluate_predictions,
    export_label_correction_package,
    export_review_package,
    load_student_mapping,
    merge_label_corrections,
    validate_label_dataframe,
)


class TrackletReviewExportTests(unittest.TestCase):
    def test_frame_loader_falls_back_to_sequential_decode_when_random_seek_fails(self):
        failed_direct_seek = {"value": False}

        class FakeCapture:
            def __init__(self, _path):
                self.position = 0

            def isOpened(self):
                return True

            def set(self, _property, value):
                self.position = int(value)
                return True

            def read(self):
                if self.position == 5 and not failed_direct_seek["value"]:
                    failed_direct_seek["value"] = True
                    return False, None
                frame = np.full((2, 2, 3), self.position, dtype=np.uint8)
                self.position += 1
                return True, frame

            def release(self):
                return None

        with patch("src.face_attendance.tracklet_review.cv2.VideoCapture", FakeCapture):
            loader = _OpenCVFrameLoader()
            frame = loader(Path("tail.mp4"), 6)
            loader.close()

        self.assertTrue(failed_direct_seek["value"])
        self.assertEqual(int(frame[0, 0, 0]), 5)

    def test_export_is_blind_and_contains_only_accepted_tracklets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_path = root / "videos" / "CP1_0910" / "back.mp4"
            video_path.parent.mkdir(parents=True)
            video_path.write_bytes(b"test placeholder")
            student_map = root / "student_map.json"
            student_map.write_text(json.dumps({
                "subject_students": {"CVO": [{"roll": "ROLL-A", "name": "Student A"}]},
                "all_students": [
                    {"roll": "ROLL-A", "name": "Student A", "subjects": ["CVO"]},
                    {"roll": "ROLL-B", "name": "Student B", "subjects": ["SWE"]},
                ],
            }), encoding="utf-8")

            tracklets = pd.DataFrame([
                {
                    "Tracklet_ID": "private-tracklet-accepted",
                    "Tracklet_Accepted": "Yes",
                    "Tracklet_Best_Roll": "SECRET-PREDICTION",
                    "Tracklet_Best_Score": 0.58,
                    "Tracklet_Second_Roll": "ROLL-B",
                    "Tracklet_Second_Score": 0.41,
                    "Tracklet_Margin": 0.17,
                    "Checkpoint_ID": "CP1",
                    "Camera_ID": "cam5",
                    "Video": "back.mp4",
                    "Selected_Observation_IDs": "obs-1; obs-2",
                    "Observation_Count": 4,
                    "Selected_Observation_Count": 2,
                    "Session_ID": "session-1",
                    "Subject_Abbr": "CVO",
                    "Match_Threshold": 0.48,
                    "Margin_Threshold": 0.08,
                    "Aggregate_Mode": "top3",
                },
                {
                    "Tracklet_ID": "private-tracklet-rejected",
                    "Tracklet_Accepted": "No",
                    "Tracklet_Best_Roll": "ROLL-B",
                    "Checkpoint_ID": "CP1",
                    "Camera_ID": "cam5",
                    "Video": "back.mp4",
                    "Selected_Observation_IDs": "obs-3",
                },
            ])
            observations = pd.DataFrame([
                {
                    "Tracklet_ID": "private-tracklet-accepted",
                    "Observation_ID": "obs-1",
                    "Checkpoint_ID": "CP1",
                    "Camera_ID": "cam5",
                    "Video": "back.mp4",
                    "Frame": 1,
                    "Time_In_Video": "00:00.04",
                    "BBox_Original_Coordinates": "10,12,28,30",
                    "Quality_Weight": 0.8,
                    "Selected_For_Aggregation": "Yes",
                },
                {
                    "Tracklet_ID": "private-tracklet-accepted",
                    "Observation_ID": "obs-2",
                    "Checkpoint_ID": "CP1",
                    "Camera_ID": "cam5",
                    "Video": "back.mp4",
                    "Frame": 2,
                    "Time_In_Video": "00:00.08",
                    "BBox_Original_Coordinates": "12,13,28,30",
                    "Quality_Weight": 0.7,
                    "Selected_For_Aggregation": "Yes",
                },
            ])

            frame = np.full((80, 100, 3), 180, dtype=np.uint8)
            result = export_review_package(
                tracklet_df=tracklets,
                observation_df=observations,
                video_root=root / "videos",
                output_root=root / "output",
                student_map_path=student_map,
                subject_abbr="CVO",
                diagnostic_run_id="diagnostic-1",
                package_id="package-test",
                frame_loader=lambda _path, _frame_index: frame.copy(),
            )

            self.assertEqual(result.accepted_tracklets, 1)
            public_text = "\n".join(
                path.read_text(encoding="utf-8", errors="ignore")
                for path in result.reviewer_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in {".html", ".json", ".csv", ".txt"}
            )
            self.assertNotIn("SECRET-PREDICTION", public_text)
            self.assertNotIn("private-tracklet-accepted", public_text)
            self.assertNotIn("Tracklet_Best_Roll", public_text)
            self.assertNotIn("Tracklet_ID", public_text)

            reviewer_html = (result.reviewer_dir / "index.html").read_text(encoding="utf-8")
            self.assertIn("student-search", reviewer_html)
            self.assertIn("Search by roll or name", reviewer_html)
            self.assertIn("identityRequired", reviewer_html)
            self.assertIn(".join('\\r\\n')", reviewer_html)
            self.assertNotIn(".join('\r\n')", reviewer_html)
            element_ids = re.findall(r'\bid="([^"]+)"', reviewer_html)
            self.assertEqual(len(element_ids), len(set(element_ids)))

            hidden_text = (result.private_dir / "hidden_predictions.csv").read_text(encoding="utf-8")
            self.assertIn("SECRET-PREDICTION", hidden_text)
            self.assertIn("private-tracklet-accepted", hidden_text)
            evidence = list((result.reviewer_dir / "evidence").glob("*.jpg"))
            self.assertEqual(len(evidence), 1)

            labels = pd.read_csv(result.reviewer_dir / "labels_template.csv", dtype=str).fillna("")
            labels.loc[0, "Review_Status"] = "not_in_mapping"
            labels_path = root / "completed_labels.csv"
            labels.to_csv(labels_path, index=False)
            validation, validation_dir = evaluate_review_package(
                result.root,
                labels_path,
                output_dir=root / "validation",
            )
            self.assertEqual(validation.summary["identity_metrics"]["reviewed_tracklets"], 1)
            self.assertTrue((validation_dir / "joined_tracklet_review.csv").is_file())
            self.assertTrue((validation_dir / "validation_summary.json").is_file())

            evidence[0].write_bytes(b"tampered")
            with self.assertRaisesRegex(LabelValidationError, "integrity check failed"):
                evaluate_review_package(
                    result.root,
                    labels_path,
                    output_dir=root / "tampered-validation",
                )

    def test_empty_student_mapping_fails_instead_of_building_a_broken_reviewer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "empty_mapping.json"
            mapping_path.write_text(
                json.dumps({"subject_students": {"CVO": []}, "all_students": []}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ReviewExportError, "no students"):
                load_student_mapping(mapping_path, "CVO")

    def test_correction_only_export_and_safe_merge_preserve_original_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_path = root / "videos" / "CP1_0910" / "back.mp4"
            video_path.parent.mkdir(parents=True)
            video_path.write_bytes(b"placeholder")
            student_map = root / "student_map.json"
            student_map.write_text(json.dumps({
                "subject_students": {"CVO": [
                    {"roll": "ROLL-A", "name": "Student A"},
                    {"roll": "ROLL-SIRI", "name": "Student Siri"},
                ]},
                "all_students": [
                    {"roll": "ROLL-A", "name": "Student A", "subjects": ["CVO"]},
                    {"roll": "ROLL-SIRI", "name": "Student Siri", "subjects": ["CVO"]},
                    {"roll": "ROLL-OTHER", "name": "Other Student", "subjects": ["SWE"]},
                ],
            }), encoding="utf-8")
            tracklets = pd.DataFrame([
                {
                    "Tracklet_ID": "track-1", "Tracklet_Accepted": "Yes",
                    "Tracklet_Best_Roll": "ROLL-A", "Tracklet_Best_Score": 0.4,
                    "Tracklet_Second_Roll": "ROLL-OTHER", "Tracklet_Second_Score": 0.35,
                    "Tracklet_Margin": 0.05, "Checkpoint_ID": "CP1", "Camera_ID": "cam5",
                    "Video": "back.mp4", "Selected_Observation_IDs": "obs-1",
                    "Observation_Count": 2, "Session_ID": "session-1", "Subject_Abbr": "CVO",
                },
                {
                    "Tracklet_ID": "track-2", "Tracklet_Accepted": "Yes",
                    "Tracklet_Best_Roll": "ROLL-A", "Tracklet_Best_Score": 0.55,
                    "Tracklet_Second_Roll": "ROLL-OTHER", "Tracklet_Second_Score": 0.3,
                    "Tracklet_Margin": 0.25, "Checkpoint_ID": "CP1", "Camera_ID": "cam5",
                    "Video": "back.mp4", "Selected_Observation_IDs": "obs-2",
                    "Observation_Count": 2, "Session_ID": "session-1", "Subject_Abbr": "CVO",
                },
            ])
            observations = pd.DataFrame([
                {
                    "Tracklet_ID": "track-1", "Observation_ID": "obs-1", "Checkpoint_ID": "CP1",
                    "Camera_ID": "cam5", "Video": "back.mp4", "Frame": 1,
                    "BBox_Original_Coordinates": "10,10,30,30", "Quality_Weight": 0.8,
                    "Selected_For_Aggregation": "Yes",
                },
                {
                    "Tracklet_ID": "track-2", "Observation_ID": "obs-2", "Checkpoint_ID": "CP1",
                    "Camera_ID": "cam5", "Video": "back.mp4", "Frame": 2,
                    "BBox_Original_Coordinates": "12,12,30,30", "Quality_Weight": 0.8,
                    "Selected_For_Aggregation": "Yes",
                },
            ])
            frame = np.full((80, 100, 3), 180, dtype=np.uint8)
            source_package = export_review_package(
                tracklet_df=tracklets,
                observation_df=observations,
                video_root=root / "videos",
                output_root=root / "source",
                student_map_path=student_map,
                subject_abbr="CVO",
                diagnostic_run_id="diagnostic-1",
                package_id="source-package",
                frame_loader=lambda _path, _frame_index: frame.copy(),
            )
            source_labels = pd.read_csv(
                source_package.reviewer_dir / "labels_template.csv", dtype=str, keep_default_na=False
            )
            source_labels.loc[0, "Review_Status"] = "not_in_mapping"
            source_labels.loc[1, "Review_Status"] = "identified"
            source_labels.loc[1, "Actual_Roll"] = "ROLL-A"
            source_labels_path = root / "source_labels.csv"
            source_labels.to_csv(source_labels_path, index=False)
            original_bytes = source_labels_path.read_bytes()

            correction_package = export_label_correction_package(
                source_review_package=source_package.root,
                source_labels_path=source_labels_path,
                student_map_path=student_map,
                subject_abbr="CVO",
                output_root=root / "corrections",
                package_id="correction-package",
            )
            correction_manifest = json.loads(
                (correction_package.reviewer_dir / "review_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(correction_manifest["review_tracklets"], 1)
            self.assertEqual(correction_manifest["review_items"][0]["review_id"], source_labels.loc[0, "Review_ID"])
            reviewer_html = (correction_package.reviewer_dir / "index.html").read_text(encoding="utf-8")
            self.assertIn("ROLL-SIRI", reviewer_html)
            self.assertNotIn("ROLL-OTHER", reviewer_html)
            self.assertNotIn("Tracklet_Best_Roll", reviewer_html)

            corrections = pd.read_csv(
                correction_package.reviewer_dir / "labels_template.csv", dtype=str, keep_default_na=False
            )
            corrections.loc[0, "Review_Status"] = "identified"
            corrections.loc[0, "Actual_Roll"] = "ROLL-SIRI"
            corrections_path = root / "corrections.csv"
            corrections.to_csv(corrections_path, index=False)
            output_path = root / "merged_labels.csv"
            merged_result = merge_label_corrections(
                correction_package.root,
                corrections_path,
                source_labels_path,
                output_path,
            )
            merged = pd.read_csv(output_path, dtype=str, keep_default_na=False)
            corrected_id = source_labels.loc[0, "Review_ID"]
            untouched_id = source_labels.loc[1, "Review_ID"]
            self.assertEqual(
                merged.loc[merged["Review_ID"].eq(corrected_id), "Actual_Roll"].iloc[0],
                "ROLL-SIRI",
            )
            self.assertEqual(
                merged.loc[merged["Review_ID"].eq(untouched_id), "Actual_Roll"].iloc[0],
                "ROLL-A",
            )
            self.assertEqual(source_labels_path.read_bytes(), original_bytes)
            self.assertTrue(merged_result.audit_path.is_file())
            with self.assertRaisesRegex(LabelValidationError, "overwrite"):
                merge_label_corrections(
                    correction_package.root,
                    corrections_path,
                    source_labels_path,
                    source_labels_path,
                )


class TrackletReviewEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.predictions = pd.DataFrame([
            {"Package_ID": "pkg", "Review_ID": "R0001", "Predicted_Roll": "A", "Checkpoint_ID": "CP1", "Camera_ID": "front", "Best_Score": 0.62, "Margin": 0.20},
            {"Package_ID": "pkg", "Review_ID": "R0002", "Predicted_Roll": "A", "Checkpoint_ID": "CP2", "Camera_ID": "front", "Best_Score": 0.54, "Margin": 0.11},
            {"Package_ID": "pkg", "Review_ID": "R0003", "Predicted_Roll": "A", "Checkpoint_ID": "CP3", "Camera_ID": "back", "Best_Score": 0.50, "Margin": 0.09},
            {"Package_ID": "pkg", "Review_ID": "R0004", "Predicted_Roll": "B", "Checkpoint_ID": "CP1", "Camera_ID": "back", "Best_Score": 0.57, "Margin": 0.14},
        ])
        self.labels = pd.DataFrame([
            {"Package_ID": "pkg", "Review_ID": "R0001", "Review_Status": "identified", "Actual_Roll": "A", "Reviewer_Notes": ""},
            {"Package_ID": "pkg", "Review_ID": "R0002", "Review_Status": "identified", "Actual_Roll": "B", "Reviewer_Notes": ""},
            {"Package_ID": "pkg", "Review_ID": "R0003", "Review_Status": "unidentifiable", "Actual_Roll": "", "Reviewer_Notes": "blurred"},
            {"Package_ID": "pkg", "Review_ID": "R0004", "Review_Status": "mixed_track", "Actual_Roll": "", "Reviewer_Notes": "two people"},
        ])

    def test_metrics_separate_identity_attendance_and_review_coverage(self):
        result = evaluate_predictions(
            prediction_df=self.predictions,
            label_df=self.labels,
            known_rolls={"A", "B", "C"},
            subject_rolls={"A", "B", "C"},
            actual_present_rolls={"A", "C"},
            present_checkpoints=3,
        )

        identity = result.summary["identity_metrics"]
        self.assertEqual(identity["accepted_tracklets"], 4)
        self.assertEqual(identity["reviewed_tracklets"], 4)
        self.assertEqual(identity["identified_tracklets"], 2)
        self.assertEqual(identity["correct_identifications"], 1)
        self.assertEqual(identity["incorrect_identifications"], 1)
        self.assertEqual(identity["mixed_tracklets"], 1)
        self.assertEqual(identity["unidentifiable_tracklets"], 1)
        self.assertEqual(identity["identified_top1_accuracy"], 0.5)
        self.assertAlmostEqual(identity["reviewable_accepted_precision"], 1 / 3, places=4)

        attendance = result.summary["attendance_metrics"]["actual_present_roster"]
        self.assertEqual(attendance["status"], "available")
        self.assertEqual(attendance["predicted_present_rolls"], ["A"])
        self.assertEqual(attendance["true_positive_rolls"], ["A"])
        self.assertEqual(attendance["false_negative_rolls"], ["C"])
        self.assertEqual(attendance["precision"], 1.0)
        self.assertEqual(attendance["recall"], 0.5)
        self.assertAlmostEqual(attendance["f1"], 2 / 3, places=4)
        self.assertEqual(result.summary["recommendation"]["production_decision"], "hold")

    def test_label_validation_rejects_duplicate_ids_and_unknown_rolls(self):
        duplicate = pd.concat([self.labels.iloc[:1], self.labels.iloc[:1]], ignore_index=True)
        with self.assertRaisesRegex(LabelValidationError, "duplicate"):
            validate_label_dataframe(duplicate, {"R0001"}, "pkg", {"A", "B"})

        unknown = self.labels.iloc[:1].copy()
        unknown.loc[0, "Actual_Roll"] = "NOT-KNOWN"
        with self.assertRaisesRegex(LabelValidationError, "known student mapping"):
            validate_label_dataframe(unknown, {"R0001"}, "pkg", {"A", "B"})

    def test_partial_review_is_valid_but_cannot_recommend_promotion(self):
        partial = self.labels.iloc[:1].copy()
        result = evaluate_predictions(
            prediction_df=self.predictions,
            label_df=partial,
            known_rolls={"A", "B", "C"},
            subject_rolls={"A", "B", "C"},
            actual_present_rolls=None,
            present_checkpoints=3,
        )

        self.assertEqual(result.summary["identity_metrics"]["review_completion_rate"], 0.25)
        self.assertEqual(result.summary["attendance_metrics"]["actual_present_roster"]["status"], "unavailable")
        self.assertEqual(result.summary["recommendation"]["next_step"], "complete_blind_review")
        self.assertEqual(result.summary["recommendation"]["production_decision"], "hold")


if __name__ == "__main__":
    unittest.main()
