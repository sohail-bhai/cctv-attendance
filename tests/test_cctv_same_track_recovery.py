from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

from src.face_attendance.cctv_same_track_recovery import (
    ObservationRecovery,
    SameTrackRecoveryConfig,
    SameTrackRecoveryError,
    _bbox_iou,
    _eligible_observations,
    _parse_bbox,
    _recover_observation,
    _select_consistent_recoveries,
    _video_map,
    verify_same_track_recovery,
)


from src.face_attendance.zones import parse_camera_zone_config

class SameTrackRecoveryConfigTests(unittest.TestCase):
    def test_default_policy_is_strict_and_bounded(self) -> None:
        config = SameTrackRecoveryConfig()
        config.validate()
        self.assertEqual(config.max_observations_per_track, 5)
        self.assertEqual(config.max_embeddings_per_track, 2)
        self.assertGreater(config.min_match_iou, 0.0)
        self.assertFalse(hasattr(config, "super_resolution"))

    def test_invalid_caps_fail_closed(self) -> None:
        with self.assertRaises(SameTrackRecoveryError):
            SameTrackRecoveryConfig(
                max_observations_per_track=1
            ).validate()
        with self.assertRaises(SameTrackRecoveryError):
            SameTrackRecoveryConfig(
                max_observations_per_track=2,
                max_embeddings_per_track=3,
            ).validate()


class GeometryTests(unittest.TestCase):
    def test_bbox_parser_and_iou_are_deterministic(self) -> None:
        bbox = _parse_bbox("10,20,30,40")
        self.assertEqual(bbox, (10.0, 20.0, 30.0, 40.0))
        self.assertAlmostEqual(_bbox_iou(bbox, bbox), 1.0)
        self.assertAlmostEqual(_bbox_iou(bbox, (100, 100, 10, 10)), 0.0)

    def test_invalid_bbox_fails_closed(self) -> None:
        for value in ("", "1,2,3", "1,2,-3,4", "x,2,3,4"):
            with self.subTest(value=value):
                with self.assertRaises(SameTrackRecoveryError):
                    _parse_bbox(value)


class ObservationSelectionTests(unittest.TestCase):
    def _diagnostics(self, **overrides: str) -> pd.DataFrame:
        row = {
            "Tracklet_ID": "T1",
            "Tracklet_Eligible": "Yes",
            "Inconsistent_Embedding_Count": "0",
            "Consistent_Embedding_Count": "3",
        }
        row.update(overrides)
        return pd.DataFrame([row])

    def _observations(self) -> pd.DataFrame:
        rows = []
        for ordinal, weight in enumerate((0.7, 0.9, 0.8, 0.95), start=1):
            rows.append(
                {
                    "Tracklet_ID": "T1",
                    "Session_ID": "S1",
                    "Observation_ID": f"O{ordinal}",
                    "Selected_For_Aggregation": "Yes" if ordinal < 4 else "No",
                    "Embedding_Consistent": "Yes" if ordinal < 4 else "No",
                    "Embedding_Extraction_Success": "Yes",
                    "Landmark_Valid": "Yes",
                    "Detection_Source": "zone",
                    "Zone_ID": "z1",
                    "Embedding_Dimension": "128",
                    "Quality_Weight": str(weight),
                    "Face_Width": str(30 + ordinal),
                    "Face_Height": str(40 + ordinal),
                    "Detector_Score": "0.9",
                    "Blur_Laplacian_Variance": "1000",
                }
            )
        return pd.DataFrame(rows)

    def test_only_selected_consistent_zone_observations_are_ranked(self) -> None:
        item = {"Track_ID": "T1", "Source_Session": "S1"}
        selected = _eligible_observations(
            item,
            self._observations(),
            self._diagnostics(),
            SameTrackRecoveryConfig(target_session="S1"),
        )
        self.assertEqual([row["Observation_ID"] for row in selected], ["O2", "O3", "O1"])

    def test_ineligible_or_inconsistent_track_fails_closed(self) -> None:
        item = {"Track_ID": "T1", "Source_Session": "S1"}
        with self.assertRaises(SameTrackRecoveryError):
            _eligible_observations(
                item,
                self._observations(),
                self._diagnostics(Tracklet_Eligible="No"),
                SameTrackRecoveryConfig(target_session="S1"),
            )
        with self.assertRaises(SameTrackRecoveryError):
            _eligible_observations(
                item,
                self._observations(),
                self._diagnostics(Inconsistent_Embedding_Count="1"),
                SameTrackRecoveryConfig(target_session="S1"),
            )

    def test_model_prediction_fields_do_not_participate_in_selection(self) -> None:
        observations = self._observations()
        observations["Frame_Best_Roll"] = ["WRONG", "WRONG", "WRONG", "WRONG"]
        observations["Tracklet_Best_Roll"] = ["WRONG", "WRONG", "WRONG", "WRONG"]
        item = {"Track_ID": "T1", "Source_Session": "S1"}
        selected = _eligible_observations(
            item,
            observations,
            self._diagnostics(),
            SameTrackRecoveryConfig(target_session="S1"),
        )
        self.assertEqual(len(selected), 3)


class ConsistencySelectionTests(unittest.TestCase):
    def _recovery(self, observation_id: str, vector: np.ndarray, quality: float) -> ObservationRecovery:
        return ObservationRecovery(
            item_id="I1",
            track_id="T1",
            observation_id=observation_id,
            candidate_roll="R1",
            source_session="S1",
            checkpoint="CP1",
            camera="cam5",
            video_path=Path("video.mp4"),
            video_sha256="a" * 64,
            frame_index=1,
            zone_profile="back",
            zone_id="z1",
            zone_upscale=2.0,
            recorded_bbox=(1, 1, 10, 10),
            matched_bbox=(1, 1, 10, 10),
            matched_iou=1.0,
            detector_score=0.9,
            quality_weight=quality,
            face_width=10,
            face_height=10,
            blur_variance=100,
            embedding=vector.astype(np.float32),
            audit_crop=np.ones((10, 10, 3), dtype=np.uint8),
        )

    def test_medoid_plus_one_consistent_observation_is_selected(self) -> None:
        base = np.zeros(128, dtype=np.float32)
        base[0] = 1.0
        close = base.copy()
        close[1] = 0.05
        outlier = np.zeros(128, dtype=np.float32)
        outlier[2] = 1.0
        selected, summary = _select_consistent_recoveries(
            [
                self._recovery("O1", base, 0.8),
                self._recovery("O2", close, 0.9),
                self._recovery("O3", outlier, 1.0),
            ],
            SameTrackRecoveryConfig(target_session="S1", min_recovered_similarity=0.25),
        )
        self.assertEqual(summary["status"], "recovered")
        self.assertEqual(len(selected), 2)
        self.assertNotIn("O3", {item.observation_id for item in selected})

    def test_single_reproduced_observation_is_not_enough(self) -> None:
        vector = np.zeros(128, dtype=np.float32)
        vector[0] = 1.0
        selected, summary = _select_consistent_recoveries(
            [self._recovery("O1", vector, 1.0)],
            SameTrackRecoveryConfig(target_session="S1"),
        )
        self.assertEqual(selected, [])
        self.assertEqual(summary["status"], "insufficient_reproduced_observations")


class ExactObservationReproductionTests(unittest.TestCase):
    class FakeEngine:
        def __init__(self, face: np.ndarray, vector: np.ndarray) -> None:
            self.face = face
            self.vector = vector

        def detect_faces(self, frame: np.ndarray) -> np.ndarray:
            return np.asarray([self.face], dtype=np.float32)

        @staticmethod
        def face_score(face: np.ndarray) -> float:
            return float(face[14])

        def extract_feature(self, frame: np.ndarray, face: np.ndarray) -> np.ndarray:
            return self.vector

    def _zone_config(self):
        return parse_camera_zone_config(
            {
                "version": 1,
                "profiles": {
                    "back": {
                        "aliases": ["cam5", "back.mp4"],
                        "expected_width": 100,
                        "expected_height": 100,
                        "zones": [
                            {
                                "id": "z1",
                                "label": "whole",
                                "x": 0,
                                "y": 0,
                                "width": 1,
                                "height": 1,
                                "upscale": 1,
                                "enabled": True,
                            }
                        ],
                    }
                },
            }
        )

    def _face(self, x: float = 10.0) -> np.ndarray:
        # YuNet row: box, five landmarks, score.
        return np.asarray(
            [x, 20, 30, 40, 16, 30, 30, 30, 23, 40, 18, 52, 29, 52, 0.95],
            dtype=np.float32,
        )

    def _observation(self) -> dict[str, str]:
        return {
            "Observation_ID": "CP1:cam5:back.mp4:f10:zone:z1:1",
            "Frame": "10",
            "Camera_ID": "cam5",
            "Video": "back.mp4",
            "Zone_ID": "z1",
            "BBox_Original_Coordinates": "10,20,30,40",
            "Checkpoint_ID": "CP1",
            "Quality_Weight": "0.8",
            "Face_Width": "30",
            "Face_Height": "40",
            "Blur_Laplacian_Variance": "900",
        }

    @patch(
        "src.face_attendance.cctv_same_track_recovery._read_frame_1_based",
        return_value=np.zeros((100, 100, 3), dtype=np.uint8),
    )
    def test_exact_zone_observation_reuses_operational_embedding_path(self, _mock_read) -> None:
        vector = np.zeros(128, dtype=np.float32)
        vector[0] = 1.0
        result = _recover_observation(
            item={
                "Item_ID": "I1",
                "Track_ID": "T1",
                "Candidate_Roll": "R1",
                "Source_Session": "S1",
            },
            observation=self._observation(),
            video_path=Path("video.mp4"),
            video_sha256="a" * 64,
            zone_config=self._zone_config(),
            engine=self.FakeEngine(self._face(), vector),
            config=SameTrackRecoveryConfig(target_session="S1"),
        )
        self.assertAlmostEqual(result.matched_iou, 1.0)
        self.assertEqual(result.embedding.size, 128)
        self.assertEqual(result.zone_id, "z1")

    @patch(
        "src.face_attendance.cctv_same_track_recovery._read_frame_1_based",
        return_value=np.zeros((100, 100, 3), dtype=np.uint8),
    )
    def test_bbox_mismatch_fails_closed_instead_of_using_nearby_face(self, _mock_read) -> None:
        vector = np.zeros(128, dtype=np.float32)
        vector[0] = 1.0
        with self.assertRaises(SameTrackRecoveryError):
            _recover_observation(
                item={
                    "Item_ID": "I1",
                    "Track_ID": "T1",
                    "Candidate_Roll": "R1",
                    "Source_Session": "S1",
                },
                observation=self._observation(),
                video_path=Path("video.mp4"),
                video_sha256="a" * 64,
                zone_config=self._zone_config(),
                engine=self.FakeEngine(self._face(x=70), vector),
                config=SameTrackRecoveryConfig(target_session="S1"),
            )


class VideoMapTests(unittest.TestCase):
    def test_video_map_is_checkpoint_camera_and_filename_scoped(self) -> None:
        summary = {
            "checkpoint_timing_map": [
                {
                    "Checkpoint_ID": "CP1",
                    "Camera_ID": "cam5",
                    "Source_File_Name": "back.mp4",
                    "Source_File": "C:/x/CP1/back.mp4",
                },
                {
                    "Checkpoint_ID": "CP2",
                    "Camera_ID": "cam5",
                    "Source_File_Name": "back.mp4",
                    "Source_File": "C:/x/CP2/back.mp4",
                },
            ]
        }
        mapping = _video_map(summary)
        self.assertEqual(mapping[("CP1", "cam5", "back.mp4")], Path("C:/x/CP1/back.mp4"))
        self.assertEqual(mapping[("CP2", "cam5", "back.mp4")], Path("C:/x/CP2/back.mp4"))

    def test_ambiguous_video_map_fails_closed(self) -> None:
        summary = {
            "checkpoint_timing_map": [
                {
                    "Checkpoint_ID": "CP1",
                    "Camera_ID": "cam5",
                    "Source_File_Name": "back.mp4",
                    "Source_File": "C:/x/a.mp4",
                },
                {
                    "Checkpoint_ID": "CP1",
                    "Camera_ID": "cam5",
                    "Source_File_Name": "back.mp4",
                    "Source_File": "C:/x/b.mp4",
                },
            ]
        }
        with self.assertRaises(SameTrackRecoveryError):
            _video_map(summary)


class RecoveryBundleIntegrityTests(unittest.TestCase):
    def test_recovery_bundle_vectors_and_hashes_are_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            vector = np.zeros(128, dtype=np.float32)
            vector[0] = 1.0
            np.savez_compressed(root / "recovered_embeddings.npz", k1=vector)
            embeddings_hash = __import__("hashlib").sha256(
                (root / "recovered_embeddings.npz").read_bytes()
            ).hexdigest()
            manifest = {
                "status": "recovery_available",
                "input_fingerprint_sha256": "f" * 64,
                "selected_recovered_embeddings": [{"embedding_key": "k1"}],
                "recovered_embeddings_sha256": embeddings_hash,
            }
            (root / "recovery_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            entries = []
            for name in ("recovered_embeddings.npz", "recovery_manifest.json"):
                path = root / name
                entries.append(
                    {
                        "path": name,
                        "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
                        "bytes": path.stat().st_size,
                    }
                )
            (root / "output_manifest.json").write_text(
                json.dumps({"files": entries}), encoding="utf-8"
            )
            verified = verify_same_track_recovery(root)
            self.assertEqual(verified["status"], "recovery_available")

    def test_tampered_vector_bundle_fails_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            vector = np.zeros(128, dtype=np.float32)
            np.savez_compressed(root / "recovered_embeddings.npz", k1=vector)
            manifest = {
                "selected_recovered_embeddings": [{"embedding_key": "k1"}],
                "recovered_embeddings_sha256": "0" * 64,
            }
            (root / "recovery_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "output_manifest.json").write_text(json.dumps({"files": []}), encoding="utf-8")
            with self.assertRaises(SameTrackRecoveryError):
                verify_same_track_recovery(root)


if __name__ == "__main__":
    unittest.main()
