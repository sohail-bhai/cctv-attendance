from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

from src.face_attendance.embedding_version_family import (
    EmbeddingFamilyError,
    RECOVERY_FAMILY_POLICY_VERSION,
    TUE_P1_SESSION,
    TUE_P2_SESSION,
    merge_recovery_into_cctv_staging,
    resolve_variant_availability,
    stage_recovery_consumption_plan,
    validate_same_track_recovery_for_family,
)


class RecoveryBundleFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.versions_root = root / "versions"
        self.source_family_id = "embfam-source1234567890"
        self.source_family_dir = self.versions_root / self.source_family_id
        self.recovery_dir = self.versions_root / "recovery_runs" / "cctv-recovery-test123456"
        self.production_embeddings = root / "student_embeddings.pkl"
        self.production_summary = root / "embedding_summary.csv"
        self.production_embeddings.write_bytes(b"production-embeddings")
        self.production_summary.write_bytes(b"production-summary")
        self.source_family_dir.mkdir(parents=True)
        self.recovery_dir.mkdir(parents=True)
        self.source_output_manifest = self.source_family_dir / "output_manifest.json"
        self.source_output_manifest.write_text("{}", encoding="utf-8")
        self.source_output_hash = self._sha(self.source_output_manifest)
        self.rolls = [
            "2401100CSE0016",
            "2401100CSE0044",
            "2401100CSE0067",
            "2401100CSE0140",
        ]
        self.item_ids = [f"CCC-test-{index}" for index in range(5)]
        self.track_ids = [f"TRACK-{index}" for index in range(5)]
        self.medoid_ids = [f"OBS-{index}-M" for index in range(5)]
        self.other_ids = [f"OBS-{index}-O" for index in range(5)]
        self._write_source_usability()
        self.manifest = self._write_recovery_bundle()

    @staticmethod
    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _write_source_usability(self) -> None:
        rows = []
        for index, (item_id, track_id) in enumerate(zip(self.item_ids, self.track_ids)):
            rows.append(
                {
                    "Item_ID": item_id,
                    "Track_ID": track_id,
                    "Observation_ID": f"ORIGINAL-{index}",
                    "Candidate_Roll": self.rolls[index % len(self.rolls)],
                    "Source_Session": TUE_P2_SESSION,
                    "Embedding_Usable": False,
                }
            )
        pd.DataFrame(rows).to_csv(
            self.source_family_dir / "cctv_embedding_usability.csv", index=False
        )

    def _write_recovery_bundle(self) -> dict:
        selected_metadata = []
        track_results = []
        audit_rows = []
        arrays = {}
        crops = self.recovery_dir / "recovered_crops"
        crops.mkdir()
        for index, (item_id, track_id, medoid_id, other_id) in enumerate(
            zip(self.item_ids, self.track_ids, self.medoid_ids, self.other_ids)
        ):
            roll = self.rolls[index % len(self.rolls)]
            track_results.append(
                {
                    "item_id": item_id,
                    "track_id": track_id,
                    "candidate_roll": roll,
                    "source_session": TUE_P2_SESSION,
                    "status": "recovered",
                    "medoid_observation_id": medoid_id,
                    "selected_observation_ids": [medoid_id, other_id],
                    "selected_pairwise_min": 0.65 - index * 0.02,
                }
            )
            for suffix, observation_id in (("medoid", medoid_id), ("other", other_id)):
                key = f"{item_id}__{suffix}"
                vector = np.zeros(128, dtype=np.float32)
                vector[index] = 1.0
                if suffix == "other":
                    vector[(index + 10) % 128] = 0.1
                    vector /= np.linalg.norm(vector)
                arrays[key] = vector
                crop = crops / f"{key}.png"
                self.assert_image_written(crop)
                crop_hash = self._sha(crop)
                selected_metadata.append(
                    {
                        "embedding_key": key,
                        "item_id": item_id,
                        "track_id": track_id,
                        "observation_id": observation_id,
                        "candidate_roll": roll,
                        "source_session": TUE_P2_SESSION,
                        "checkpoint": f"CP{index + 1}",
                        "camera": "cam5",
                        "frame_index": 100 + index,
                        "video_path": str(self.root / f"video-{index}.mp4"),
                        "video_sha256": "a" * 64,
                        "matched_iou": 0.99,
                        "detector_score": 0.90,
                        "identity_source": "human_reviewed_track_actual_roll",
                        "model_prediction_used_as_identity": False,
                        "manual_rereview_required": False,
                        "quality_gate_relaxed": False,
                        "super_resolution_used": False,
                    }
                )
                audit_rows.append(
                    {
                        "Item_ID": item_id,
                        "Track_ID": track_id,
                        "Observation_ID": observation_id,
                        "Candidate_Roll": roll,
                        "Source_Session": TUE_P2_SESSION,
                        "Selected_For_Recovery": True,
                        "Embedding_Key": key,
                        "Identity_Source": "human_reviewed_track_actual_roll",
                        "Model_Prediction_Used_As_Identity": False,
                        "Manual_Rereview_Required": False,
                        "Quality_Gate_Relaxed": False,
                        "Super_Resolution_Used": False,
                        "Audit_Crop_SHA256": crop_hash,
                        "Detector_Score": 0.90,
                        "Matched_IoU": 0.99,
                    }
                )
        np.savez_compressed(self.recovery_dir / "recovered_embeddings.npz", **arrays)
        pd.DataFrame(audit_rows).to_csv(
            self.recovery_dir / "same_track_recovery_audit.csv", index=False
        )
        (self.recovery_dir / "output_manifest.json").write_text(
            json.dumps({"schema_version": 1, "files": []}), encoding="utf-8"
        )
        manifest = {
            "status": "recovery_available",
            "recovery_id": "cctv-recovery-test123456",
            "source_family_id": self.source_family_id,
            "source_family_decision": "built_unapproved_insufficient_evidence",
            "target_session": TUE_P2_SESSION,
            "production_embeddings_before_sha256": self._sha(self.production_embeddings),
            "production_embeddings_after_sha256": self._sha(self.production_embeddings),
            "production_summary_before_sha256": self._sha(self.production_summary),
            "production_summary_after_sha256": self._sha(self.production_summary),
            "production_files_changed": False,
            "dataset_files_changed": False,
            "official_attendance_changed": False,
            "candidate_promoted": False,
            "mon_p3_processed": False,
            "manual_review_requested": False,
            "input_fingerprint": {
                "family_output_manifest_sha256": self.source_output_hash,
                "target_items": [
                    {
                        "item_id": item_id,
                        "track_id": track_id,
                        "observation_id": f"ORIGINAL-{index}",
                        "candidate_roll": self.rolls[index % len(self.rolls)],
                        "source_session": TUE_P2_SESSION,
                    }
                    for index, (item_id, track_id) in enumerate(
                        zip(self.item_ids, self.track_ids)
                    )
                ],
            },
            "track_results": track_results,
            "selected_recovered_embeddings": selected_metadata,
        }
        (self.recovery_dir / "recovery_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return manifest

    @staticmethod
    def assert_image_written(path: Path) -> None:
        image = np.full((32, 32, 3), 127, dtype=np.uint8)
        if not cv2.imwrite(str(path), image):
            raise AssertionError(f"Unable to write fixture crop: {path}")


class RecoveredFamilyConsumptionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.fixture = RecoveryBundleFixture(self.root)
        self.roster = set(self.fixture.rolls)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _plan(self):
        with patch(
            "src.face_attendance.cctv_same_track_recovery.verify_same_track_recovery",
            return_value=self.fixture.manifest,
        ), patch(
            "src.face_attendance.embedding_version_family.verify_embedding_family",
            return_value={
                "decision": "built_unapproved_insufficient_evidence",
                "verified_files": 102,
                "variants_verified": 4,
            },
        ):
            return validate_same_track_recovery_for_family(
                recovery_dir=self.fixture.recovery_dir,
                versions_root=self.fixture.versions_root,
                production_embeddings_path=self.fixture.production_embeddings,
                production_summary_path=self.fixture.production_summary,
                roster_rolls=self.roster,
            )

    def test_consumes_exactly_one_medoid_per_reviewed_track(self) -> None:
        plan = self._plan()
        self.assertEqual(len(plan.sources), 5)
        self.assertEqual(
            {row["observation_id"] for row in plan.sources},
            set(self.fixture.medoid_ids),
        )
        self.assertEqual(
            plan.summary["consumption_rule"],
            "exactly_one_medoid_embedding_per_human_reviewed_track",
        )
        self.assertEqual(plan.summary["candidate_consumed_medoid_embeddings"], 5)
        self.assertFalse(plan.summary["model_prediction_used_as_identity"])
        self.assertFalse(plan.summary["manual_rereview_required"])

    def test_stricter_consumption_similarity_floor_fails_closed(self) -> None:
        self.fixture.manifest["track_results"][0]["selected_pairwise_min"] = 0.39
        with self.assertRaisesRegex(EmbeddingFamilyError, "similarity floor"):
            self._plan()

    def test_model_identity_or_relaxed_quality_fails_closed(self) -> None:
        self.fixture.manifest["selected_recovered_embeddings"][0][
            "model_prediction_used_as_identity"
        ] = True
        with self.assertRaisesRegex(EmbeddingFamilyError, "safety contract"):
            self._plan()

    def test_source_family_item_set_must_match_exactly(self) -> None:
        usability = pd.read_csv(
            self.fixture.source_family_dir / "cctv_embedding_usability.csv"
        )
        usability.loc[0, "Item_ID"] = "CCC-unrelated"
        usability.to_csv(
            self.fixture.source_family_dir / "cctv_embedding_usability.csv", index=False
        )
        with self.assertRaisesRegex(EmbeddingFamilyError, "target items"):
            self._plan()

    def test_staging_copies_only_medoid_sources_and_preserves_audit_inputs(self) -> None:
        plan = self._plan()
        temporary = self.root / "building"
        final = self.root / "final-family"
        temporary.mkdir()
        staging, records, manifest = stage_recovery_consumption_plan(
            plan=plan,
            temporary_family_dir=temporary,
            final_family_dir=final,
            repo_root=self.root,
        )
        self.assertEqual(len(staging), 5)
        self.assertEqual(len(records), 5)
        self.assertTrue(staging["Embedding_Usable"].all())
        self.assertTrue(
            staging["Recovery_Consumption_Role"].eq("track_medoid_only").all()
        )
        self.assertEqual(manifest["policy_version"], RECOVERY_FAMILY_POLICY_VERSION)
        self.assertTrue((temporary / "recovery_consumption_manifest.json").is_file())
        self.assertTrue(
            (temporary / "recovery_inputs" / manifest["recovery_id"] / "recovered_embeddings.npz").is_file()
        )

    def test_recovery_makes_variant_b_available_without_changing_variant_c_contract(self) -> None:
        original_staging = pd.DataFrame(
            [
                {
                    "Source_ID": "P1",
                    "Candidate_Roll": self.fixture.rolls[0],
                    "Source_Session": TUE_P1_SESSION,
                    "Embedding_Usable": True,
                },
                {
                    "Source_ID": "P2-UNUSABLE",
                    "Candidate_Roll": self.fixture.rolls[1],
                    "Source_Session": TUE_P2_SESSION,
                    "Embedding_Usable": False,
                },
            ]
        )
        original_records = [
            {
                "roll_no": self.fixture.rolls[0],
                "_source_id": "P1",
                "_source_session": TUE_P1_SESSION,
            }
        ]
        recovery_staging = pd.DataFrame(
            [
                {
                    "Source_ID": "R1",
                    "Candidate_Roll": self.fixture.rolls[1],
                    "Source_Session": TUE_P2_SESSION,
                    "Embedding_Usable": True,
                }
            ]
        )
        recovery_records = [
            {
                "roll_no": self.fixture.rolls[1],
                "_source_id": "R1",
                "_source_session": TUE_P2_SESSION,
            }
        ]
        combined, records, manifest = merge_recovery_into_cctv_staging(
            original_staging=original_staging,
            original_records=original_records,
            original_manifest={
                "approved_rows": 2,
                "approved_but_embedding_unusable_crops": 1,
                "minimum_usable_evidence_policy": {},
            },
            recovery_staging=recovery_staging,
            recovery_records=recovery_records,
            recovery_manifest={"recovery_id": "R", "policy_version": RECOVERY_FAMILY_POLICY_VERSION},
        )
        availability = resolve_variant_availability(manifest)
        self.assertEqual(len(combined), 3)
        self.assertEqual(len(records), 2)
        self.assertTrue(availability["B"]["available"])
        self.assertTrue(availability["C"]["available"])
        self.assertEqual(availability["B"]["source_sessions_included"], [TUE_P2_SESSION])
        self.assertEqual(availability["C"]["source_sessions_included"], [TUE_P1_SESSION])


if __name__ == "__main__":
    unittest.main()
