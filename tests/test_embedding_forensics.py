import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

from src.face_attendance.embedding_forensics import (
    BenchmarkSource,
    EmbeddingForensicsError,
    _audit_one_dataset_image,
    analyze_embedding_records,
    analyze_prediction_confusions,
    assert_candidate_not_rejected_for_approval,
    audit_enrollment_datasets,
    audit_missing_embeddings,
    compare_protected_state,
    build_candidate_rejection_record,
    freeze_multisession_benchmark,
    propose_verified_cctv_crops,
    register_candidate_rejection,
    snapshot_protected_state,
)
from src.face_attendance.tracklet_review import _sha256_file


ROOT = Path(__file__).resolve().parents[1]
STUDENT_MAP = ROOT / "data" / "student_faculty_map.json"


def normalized(values):
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def geometry_fixture():
    records = []
    for index, values in enumerate(
        ([1, 0, 0], [1, 0.01, 0], [1, -0.01, 0], [1, 0, 0.01], [1, 0, -0.01], [0, 1, 0])
    ):
        records.append(
            {
                "roll_no": "2401100CSE0016",
                "image_path": f"dataset/2401100CSE0016/{index}.jpg",
                "embedding": normalized(values),
            }
        )
    records.extend(
        [
            {
                "roll_no": "2401100CSE0016 (E) Student",
                "image_path": "dataset/2401100CSE0016/annotated.jpg",
                "embedding": normalized([1, 0.02, 0]),
            },
            {
                "roll_no": "2401100CSE0019",
                "image_path": "dataset/2401100CSE0019/exact.jpg",
                "embedding": normalized([1, 0, 0]),
            },
            {
                "roll_no": "2401100CSE0028",
                "image_path": "dataset/2401100CSE0028/bad.jpg",
                "embedding": np.array([np.nan, 0, 1], dtype=np.float32),
            },
        ]
    )
    return analyze_embedding_records(records, repo_root=ROOT)


class FakeNoFaceEngine:
    def detect_faces_scaled(self, _image, max_width=0):
        return np.empty((0, 15), dtype=np.float32)


class FakeMultiFaceEngine:
    @staticmethod
    def _face(offset=0):
        return np.array(
            [
                45 + offset,
                35,
                80,
                90,
                65,
                65,
                100,
                65,
                82,
                82,
                68,
                105,
                98,
                105,
                0.96,
            ],
            dtype=np.float32,
        )

    def detect_faces_scaled(self, _image, max_width=0):
        return np.stack([self._face(), self._face(4)])

    @staticmethod
    def extract_feature(_image, _face):
        return normalized([1, 0, 0])

    @staticmethod
    def face_box(face):
        return tuple(float(value) for value in face[:4])

    @staticmethod
    def face_score(face):
        return float(face[14])


class FakeCropEngine:
    def __init__(self, *_args, **_kwargs):
        pass

    @staticmethod
    def detect_faces(image):
        height, width = image.shape[:2]
        return np.asarray(
            [[
                2,
                2,
                max(20, width - 4),
                max(20, height - 4),
                12,
                15,
                max(22, width - 12),
                15,
                width / 2,
                height / 2,
                15,
                max(25, height - 12),
                max(25, width - 15),
                max(25, height - 12),
                0.97,
            ]],
            dtype=np.float32,
        )

    @staticmethod
    def extract_feature(image, _face):
        value = float(np.mean(image)) / 255.0
        return normalized([max(value, 0.01), 1.0 - value, 0.25])


class FakeFrameLoader:
    calls = []

    def __init__(self):
        self.calls = []

    def __call__(self, video_path, frame_index):
        self.calls.append((video_path, frame_index))
        image = np.full((100, 120, 3), 80 + frame_index, dtype=np.uint8)
        cv2.circle(image, (50, 50), 24, (180, 180, 180), -1)
        return image

    def close(self):
        return None


class EmbeddingForensicsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_embedding_geometry_detects_nan_collision_duplicates_and_outlier(self):
        result = geometry_fixture()
        self.assertEqual(result.summary["invalid_nan_embeddings"], 1)
        self.assertGreaterEqual(result.summary["canonical_key_collisions"], 1)
        self.assertGreaterEqual(result.summary["cross_student_exact_or_near_duplicates"], 1)
        self.assertGreaterEqual(result.summary["robust_outlier_embeddings"], 1)
        health = result.student_health.set_index("Canonical_Roll")
        self.assertEqual(health.loc["2401100CSE0016", "Annotated_Embedding_Key"], "Yes")

    def test_embedding_neighbor_and_outlier_results_are_deterministic(self):
        first = geometry_fixture()
        second = geometry_fixture()
        pd.testing.assert_frame_equal(first.student_health, second.student_health)
        pd.testing.assert_frame_equal(first.outliers, second.outliers)
        pd.testing.assert_frame_equal(first.nearest_neighbors, second.nearest_neighbors)

    def test_attractors_use_only_confirmed_human_reviewed_failures(self):
        rows = [
            ("A", "B", "identified", "C"),
            ("A", "C", "identified", "C"),
            ("A", "", "not_in_mapping", ""),
            ("A", "A", "identified", "B"),
            ("Z", "", "unidentifiable", ""),
        ]
        benchmark = pd.DataFrame(
            [
                {
                    "Benchmark_Row_ID": f"GT-{index}",
                    "Source_Kind": "reviewed",
                    "Baseline_Status": "baseline",
                    "Session_ID": "S1",
                    "Package_ID": "P1",
                    "Review_ID": f"R{index}",
                    "Tracklet_ID": f"T{index}",
                    "Checkpoint_ID": f"CP{index}",
                    "Camera_ID": "cam",
                    "Predicted_Roll": predicted,
                    "Actual_Roll": actual,
                    "Second_Roll": second,
                    "Review_Status": status,
                    "Top1_Correct": "Yes" if status == "identified" and predicted == actual else "No",
                    "Best_Score": 0.5,
                    "Second_Score": 0.4,
                    "Margin": 0.1,
                    "Vote_Ratio_Pct": 70,
                    "Consistency_Ratio": 0.8,
                    "Pairwise_Similarity_Median": 0.7,
                    "Observation_Count": 5,
                }
                for index, (predicted, actual, status, second) in enumerate(rows)
            ]
        )
        result = analyze_prediction_confusions(benchmark)
        attractor = result.attractors.iloc[0]
        self.assertEqual(attractor["Predicted_Roll"], "A")
        self.assertEqual(int(attractor["Confirmed_False_Accepts"]), 3)
        self.assertEqual(int(attractor["Distinct_Actual_Identities_Attracted"]), 2)
        self.assertEqual(int(attractor["Outsider_Not_In_Mapping_Accepts"]), 1)
        unidentifiable = result.matrix[result.matrix["Review_Status"].eq("unidentifiable")].iloc[0]
        self.assertEqual(unidentifiable["Actual_Roll"], "")
        self.assertFalse(result.summary["prediction_used_as_ground_truth"])

    def test_rejected_candidate_registry_is_idempotent_and_conflicts_fail(self):
        record = {
            "candidate_id": "cal-test",
            "status": "rejected_multisession_false_accepts",
            "production_approved": False,
            "enabled": False,
            "archived_at": "2026-01-01T00:00:00",
            "evidence": {"sha256": "a" * 64},
        }
        path, created = register_candidate_rejection(record, self.root / "registry")
        original = path.read_bytes()
        self.assertTrue(created)
        path_again, created_again = register_candidate_rejection(
            {**record, "archived_at": "2026-01-02T00:00:00"}, self.root / "registry"
        )
        self.assertFalse(created_again)
        self.assertEqual(path_again.read_bytes(), original)
        with self.assertRaisesRegex(EmbeddingForensicsError, "Conflicting"):
            register_candidate_rejection(
                {**record, "evidence": {"sha256": "b" * 64}}, self.root / "registry"
            )
        with self.assertRaisesRegex(EmbeddingForensicsError, "permanently rejected"):
            assert_candidate_not_rejected_for_approval("cal-test", self.root / "registry")

    def test_unreadable_and_no_face_images_are_flagged_without_source_changes(self):
        folder = self.root / "dataset" / "2401100CSE0016"
        folder.mkdir(parents=True)
        unreadable = folder / "bad.jpg"
        unreadable.write_bytes(b"not an image")
        geometry = geometry_fixture()
        config = {
            "pad_percent": 0.25,
            "det_max_width": 1280,
            "min_face_size": 50,
            "min_area_ratio": 0.003,
            "min_landmarks_inside": 4,
            "min_eye_distance": 8,
        }
        before = unreadable.read_bytes()
        row = _audit_one_dataset_image(
            path=unreadable,
            source_role="production",
            source_root=self.root / "dataset",
            engine=FakeNoFaceEngine(),
            geometry=geometry,
            master_rolls={"2401100CSE0016"},
            subject_rolls={"2401100CSE0016"},
            enrollment_config=config,
        )
        self.assertIn("unreadable_image", row["Audit_Flags"])
        self.assertEqual(unreadable.read_bytes(), before)
        readable = folder / "blank.jpg"
        self.assertTrue(cv2.imwrite(str(readable), np.full((200, 200, 3), 120, np.uint8)))
        row = _audit_one_dataset_image(
            path=readable,
            source_role="production",
            source_root=self.root / "dataset",
            engine=FakeNoFaceEngine(),
            geometry=geometry,
            master_rolls={"2401100CSE0016"},
            subject_rolls={"2401100CSE0016"},
            enrollment_config=config,
        )
        self.assertIn("no_face", row["Audit_Flags"])

    def test_multiple_faces_are_flagged_without_auto_relabel(self):
        folder = self.root / "dataset" / "2401100CSE0016"
        folder.mkdir(parents=True)
        image_path = folder / "multiple.jpg"
        self.assertTrue(cv2.imwrite(str(image_path), np.full((200, 200, 3), 130, np.uint8)))
        row = _audit_one_dataset_image(
            path=image_path,
            source_role="production",
            source_root=self.root / "dataset",
            engine=FakeMultiFaceEngine(),
            geometry=geometry_fixture(),
            master_rolls={"2401100CSE0016"},
            subject_rolls={"2401100CSE0016"},
            enrollment_config={
                "pad_percent": 0.0,
                "det_max_width": 1280,
                "min_face_size": 50,
                "min_area_ratio": 0.003,
                "min_landmarks_inside": 4,
                "min_eye_distance": 8,
            },
        )
        self.assertIn("multiple_faces", row["Audit_Flags"])
        self.assertEqual(row["Canonical_Roll"], "2401100CSE0016")

    def test_dataset_duplicate_hash_and_missing_optional_root_are_reported(self):
        dataset = self.root / "dataset"
        left = dataset / "2401100CSE0016" / "same.jpg"
        right = dataset / "2401100CSE0019" / "same.jpg"
        left.parent.mkdir(parents=True)
        right.parent.mkdir(parents=True)
        payload = b"same bytes"
        left.write_bytes(payload)
        right.write_bytes(payload)
        before = {left: left.read_bytes(), right: right.read_bytes()}

        def analyzer(**kwargs):
            path = kwargs["path"]
            roll = path.parent.name
            return {
                "Dataset_Source_Role": kwargs["source_role"],
                "Dataset_Root": str(kwargs["source_root"]),
                "Source_Path": str(path.resolve()),
                "Relative_Path": path.relative_to(kwargs["source_root"]).as_posix(),
                "Owning_Folder": roll,
                "Canonical_Roll": roll,
                "In_Master_Registry": "Yes",
                "In_CVO_Roster": "Yes",
                "File_SHA256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "File_Size_Bytes": path.stat().st_size,
                "Decode_Success": "No",
                "Image_Width": 0,
                "Image_Height": 0,
                "Valid_SFace_Embedding": "No",
                "Audit_Flags": "unreadable_image",
                "Suspect_For_Human_Review": "Yes",
            }

        embeddings = self.root / "embeddings.pkl"
        import pickle

        with embeddings.open("wb") as file_obj:
            pickle.dump({"records": [], "metadata": {}}, file_obj)
        result = audit_enrollment_datasets(
            [
                {"role": "production", "path": dataset, "exists": True, "optional": False},
                {"role": "optional", "path": self.root / "missing", "exists": False, "optional": True},
            ],
            geometry=geometry_fixture(),
            student_map_path=STUDENT_MAP,
            embeddings_path=embeddings,
            analyzer=analyzer,
            progress=None,
        )
        self.assertEqual(result.summary["cross_student_duplicate_groups"], 1)
        self.assertEqual(len(result.summary["missing_optional_roots"]), 1)
        self.assertEqual(left.read_bytes(), before[left])
        self.assertEqual(right.read_bytes(), before[right])

    def test_missing_embedding_root_cause_is_dataset_missing(self):
        empty_dataset = self.root / "dataset"
        empty_dataset.mkdir()
        embeddings = self.root / "embeddings.pkl"
        import pickle

        with embeddings.open("wb") as file_obj:
            pickle.dump({"records": [], "metadata": {}}, file_obj)
        audit = audit_enrollment_datasets(
            [{"role": "production", "path": empty_dataset, "exists": True, "optional": False}],
            geometry=geometry_fixture(),
            student_map_path=STUDENT_MAP,
            embeddings_path=embeddings,
            analyzer=lambda **_kwargs: {},
            progress=None,
        )
        missing = audit_missing_embeddings(
            target_rolls=["2401100CSE0268", "24011CSEAI0110"],
            student_map_path=STUDENT_MAP,
            subject_abbr="CVO",
            roots=[{"role": "production", "path": empty_dataset, "exists": True}],
            dataset_audit=audit,
            geometry=geometry_fixture(),
        )
        self.assertEqual(set(missing["Missing_Embedding_Root_Cause"]), {"dataset_missing"})

    def test_protected_state_snapshot_detects_only_real_byte_change(self):
        (self.root / "dataset").mkdir()
        (self.root / "augmented_dataset").mkdir()
        (self.root / "models").mkdir()
        (self.root / "data").mkdir()
        (self.root / "attendance_output").mkdir()
        candidate = self.root / "candidate.json"
        candidate.write_text("candidate", encoding="utf-8")
        embeddings = self.root / "models" / "student_embeddings.pkl"
        summary = self.root / "models" / "embedding_summary.csv"
        mapping = self.root / "data" / "student_faculty_map.json"
        embeddings.write_bytes(b"emb")
        summary.write_text("summary", encoding="utf-8")
        mapping.write_text("{}", encoding="utf-8")
        image = self.root / "dataset" / "a.jpg"
        image.write_bytes(b"image")
        before = snapshot_protected_state(
            repo_root=self.root,
            candidate_config_path=candidate,
            embeddings_path=embeddings,
            embedding_summary_path=summary,
            student_map_path=mapping,
        )
        same = snapshot_protected_state(
            repo_root=self.root,
            candidate_config_path=candidate,
            embeddings_path=embeddings,
            embedding_summary_path=summary,
            student_map_path=mapping,
        )
        self.assertEqual(compare_protected_state(before, same), [])
        image.write_bytes(b"changed")
        after = snapshot_protected_state(
            repo_root=self.root,
            candidate_config_path=candidate,
            embeddings_path=embeddings,
            embedding_summary_path=summary,
            student_map_path=mapping,
        )
        self.assertEqual(compare_protected_state(before, after)[0]["category"], "dataset")

    def test_cctv_proposals_use_actual_roll_and_exclude_nonidentified_statuses(self):
        package = self.root / "package"
        private = package / "private"
        diagnostic = self.root / "diagnostic"
        video_root = self.root / "videos"
        private.mkdir(parents=True)
        diagnostic.mkdir()
        video_root.mkdir()
        video = video_root / "cam.mp4"
        video.write_bytes(b"synthetic video provenance")
        tracks = pd.DataFrame([{"Tracklet_ID": "T1"}, {"Tracklet_ID": "T2"}])
        track_path = diagnostic / "tracklet_diagnostics_test.csv"
        tracks.to_csv(track_path, index=False)
        observations = []
        for track in ("T1", "T2"):
            for frame in (10, 25, 40, 55):
                observations.append(
                    {
                        "Tracklet_ID": track,
                        "Observation_ID": f"{track}-O{frame}",
                        "Checkpoint_ID": "CP1" if frame < 40 else "CP2",
                        "Camera_ID": "cam1",
                        "Video": "cam.mp4",
                        "Frame": frame,
                        "BBox_Original_Coordinates": "20,20,50,55",
                        "Selected_For_Aggregation": "Yes",
                        "Embedding_Extraction_Success": "Yes",
                        "Quality_Weight": 0.8,
                        "Detector_Score": 0.95,
                        "Blur_Laplacian_Variance": 200,
                        "Face_Width": 50,
                    }
                )
        observation_path = diagnostic / "tracklet_observations_test.csv"
        pd.DataFrame(observations).to_csv(observation_path, index=False)
        (private / "export_metadata.json").write_text(
            json.dumps(
                {
                    "package_id": "pkg",
                    "video_root": str(video_root),
                    "source_sha256": {
                        "tracklet_diagnostics": _sha256_file(track_path),
                        "tracklet_observations": _sha256_file(observation_path),
                    },
                }
            ),
            encoding="utf-8",
        )
        benchmark = pd.DataFrame(
            [
                {
                    "Review_Status": "identified",
                    "Actual_Roll": "2401100CSE0016",
                    "Predicted_Roll": "2401100CSE0050",
                    "Source_Kind": "source",
                    "Session_ID": "2026-06-30__B51__P1__CVO",
                    "Package_ID": "pkg",
                    "Review_ID": "R1",
                    "Benchmark_Row_ID": "GT-1",
                    "Tracklet_ID": "T1",
                    "Checkpoint_ID": "CP1",
                    "Camera_ID": "cam1",
                },
                {
                    "Review_Status": "not_in_mapping",
                    "Actual_Roll": "",
                    "Predicted_Roll": "2401100CSE0050",
                    "Source_Kind": "source",
                    "Session_ID": "2026-06-30__B51__P1__CVO",
                    "Package_ID": "pkg",
                    "Review_ID": "R2",
                    "Benchmark_Row_ID": "GT-2",
                    "Tracklet_ID": "T2",
                    "Checkpoint_ID": "CP1",
                    "Camera_ID": "cam1",
                },
            ]
        )
        source = BenchmarkSource("source", package, self.root / "labels.csv", diagnostic, 2, "baseline")
        with patch("src.face_attendance.embedding_forensics.FaceEngine", FakeCropEngine), patch(
            "src.face_attendance.embedding_forensics._OpenCVFrameLoader", FakeFrameLoader
        ):
            result = propose_verified_cctv_crops(
                benchmark=benchmark,
                benchmark_sources=[source],
                output_crop_dir=self.root / "crops",
                repo_root=self.root,
                student_map_path=STUDENT_MAP,
                progress=None,
            )
        self.assertGreaterEqual(len(result.selected), 3)
        self.assertEqual(set(result.selected["Actual_Roll"]), {"2401100CSE0016"})
        self.assertEqual(set(result.selected["Identity_Source"]), {"human_review_actual_roll"})
        self.assertNotIn("Predicted_Roll", result.inventory.columns)
        self.assertTrue(all(Path(path).is_file() for path in result.selected["Candidate_Crop_Path"]))

    def test_real_multisession_benchmark_freezes_exactly_120_reviewed_tracks(self):
        p1 = ROOT / "attendance_output" / "diagnostics" / "2026-06-30__B51__P1__CVO_20260712_000753_991890"
        p2 = ROOT / "attendance_output" / "diagnostics" / "2026-06-30__B51__P2__CVO_20260712_221105_286649"
        accepted = p1 / "tracklet_review_2026-06-30__B51__P1__CVO_20260712_000753_991890-20260712074431-763c4d"
        unresolved = p1 / "recall_analysis_20260712_124500_128763" / "unresolved_tracklet_review_2026-06-30__B51__P1__CVO_20260712_000753_991890-20260712124500-1f3271"
        shadow = p2 / "shadow_validation" / "shadow-validation_20260713_084319_043368" / "shadow_recovery_review_2026-06-30__B51__P2__CVO_20260712_221105_286649-20260713084319-55466e"
        p2_labels = Path(r"D:\Downloads\tracklet_review_labels_2026-06-30__B51__P2__CVO_20260712_221105_286649-20260713084319-55466e.csv")
        required = [accepted, unresolved, shadow, p2_labels]
        if not all(path.exists() for path in required):
            self.skipTest("Real Phase 1.2E/H review artifacts are not present")
        sources = [
            BenchmarkSource(
                "tue_p1_accepted_review",
                accepted,
                ROOT / "tracklet_labels_TUE_P1.csv",
                p1,
                37,
                "baseline_accepted_human_reviewed",
                "185c9d4e09ff97e5cc1cd99f04e08f090b08dd58efe6dd391ab4e5aa1a1eee4e",
            ),
            BenchmarkSource(
                "tue_p1_corrected_unresolved_review",
                unresolved,
                ROOT / "unresolved_labels_TUE_P1_corrected.csv",
                p1,
                63,
                "baseline_unresolved_human_reviewed",
                "ce850c74550e9f8cc82a9468ce553b35c2a5f401e940b6073e03f412779fbb03",
            ),
            BenchmarkSource(
                "tue_p2_shadow_recovery_review",
                shadow,
                p2_labels,
                p2,
                20,
                "baseline_rejected_shadow_proposal_human_reviewed",
                "cfd489d092256bbbff9822954c22cdafcce0aaa4823c7422957a44a3c7802ee8",
            ),
        ]
        result = freeze_multisession_benchmark(sources, student_map_path=STUDENT_MAP)
        self.assertEqual(len(result.frame), 120)
        self.assertEqual(result.manifest["status_counts"], {
            "identified": 105,
            "mixed_track": 1,
            "not_in_mapping": 12,
            "unidentifiable": 2,
        })
        self.assertEqual(result.manifest["row_counts"]["predicted_correct"], 83)
        self.assertEqual(result.manifest["row_counts"]["predicted_wrong"], 22)
        self.assertFalse(result.manifest["contains_unreviewed_tue_p2_baseline_tracks"])
        self.assertIn("MON_P3", result.manifest["session_use_policy"])

    def test_real_rejected_candidate_record_verifies_p1_and_p2_evidence(self):
        p1 = ROOT / "attendance_output" / "diagnostics" / "2026-06-30__B51__P1__CVO_20260712_000753_991890"
        calibration = p1 / "calibration" / "tracklet_calibration_20260712_175448"
        shadow = (
            ROOT
            / "attendance_output"
            / "diagnostics"
            / "2026-06-30__B51__P2__CVO_20260712_221105_286649"
            / "shadow_validation"
            / "shadow-validation_20260713_084319_043368"
            / "shadow_recovery_review_2026-06-30__B51__P2__CVO_20260712_221105_286649-20260713084319-55466e"
        )
        if not shadow.exists():
            self.skipTest("Real Phase 1.2H rejected-candidate evidence is not present")
        record = build_candidate_rejection_record(
            candidate_config_path=calibration / "candidate_config.json",
            calibration_dir=calibration,
            shadow_review_package=shadow,
            archived_at="2026-07-13T00:00:00",
        )
        self.assertEqual(record["final_decision"], "reject_candidate")
        self.assertFalse(record["production_approved"])
        self.assertEqual(record["tue_p2_evaluation_evidence"]["correct_recoveries"], 13)
        self.assertEqual(record["tue_p2_evaluation_evidence"]["confirmed_false_identities"], 5)
        self.assertEqual(
            record["tue_p2_evaluation_evidence"]["outsider_not_in_mapping_false_acceptances"], 2
        )


if __name__ == "__main__":
    unittest.main()
