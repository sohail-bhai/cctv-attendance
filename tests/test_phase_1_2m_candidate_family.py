from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from src.face_attendance.embedding_version_family import (
    CctvSourceSelectionContract,
    EmbeddingFamilyError,
    TUE_P1_SESSION,
    TUE_P2_SESSION,
    apply_cctv_source_selection,
)
from src.face_attendance.phase_1_2m_candidate_family import (
    CORRECTED_CONFIG_ID,
    EXPECTED_SELECTION_DECISION_SHA256,
    EXPECTED_SELECTION_OUTPUT_MANIFEST_SHA256,
    EXPECTED_SOURCE_ABLATION_OUTPUT_MANIFEST_SHA256,
    KEPT_SOURCE_IDS,
    REMOVED_SOURCE_IDS,
    Phase12MError,
    _compare_source_composition,
    _expected_contract,
    _pipe_set,
    _score_expected,
)


class SelectionContractTests(unittest.TestCase):
    def test_exact_phase_1_2m_contract_is_normalized(self) -> None:
        contract = _expected_contract().normalized()
        self.assertEqual(contract["corrected_config_id"], CORRECTED_CONFIG_ID)
        self.assertEqual(set(contract["kept_source_ids"]), set(KEPT_SOURCE_IDS))
        self.assertEqual(set(contract["removed_source_ids"]), set(REMOVED_SOURCE_IDS))
        self.assertTrue(contract["final_promotion_requires_new_untouched_session"])
        self.assertFalse(contract["candidate_promoted"])

    def test_contract_rejects_keep_remove_overlap(self) -> None:
        contract = CctvSourceSelectionContract(
            selection_audit_id="selection-audit-test",
            selection_policy_version="selection-v1",
            source_ablation_id="source-ablation-test",
            corrected_config_id="ABL-test",
            kept_source_ids=("source-a",),
            removed_source_ids=("source-a",),
            selection_audit_output_manifest_sha256="a" * 64,
            selection_decision_sha256="b" * 64,
            source_ablation_output_manifest_sha256="c" * 64,
        )
        with self.assertRaises(EmbeddingFamilyError):
            contract.normalized()

    def test_contract_rejects_invalid_hash(self) -> None:
        contract = CctvSourceSelectionContract(
            selection_audit_id="selection-audit-test",
            selection_policy_version="selection-v1",
            source_ablation_id="source-ablation-test",
            corrected_config_id="ABL-test",
            kept_source_ids=("source-a",),
            removed_source_ids=("source-b",),
            selection_audit_output_manifest_sha256="not-a-hash",
            selection_decision_sha256="b" * 64,
            source_ablation_output_manifest_sha256="c" * 64,
        )
        with self.assertRaises(EmbeddingFamilyError):
            contract.normalized()

    def test_evidence_hash_constants_are_sha256(self) -> None:
        for value in (
            EXPECTED_SELECTION_OUTPUT_MANIFEST_SHA256,
            EXPECTED_SELECTION_DECISION_SHA256,
            EXPECTED_SOURCE_ABLATION_OUTPUT_MANIFEST_SHA256,
        ):
            self.assertEqual(len(value), 64)
            int(value, 16)

    def test_pipe_set_ignores_empty_tokens(self) -> None:
        self.assertEqual(_pipe_set("a||b|"), {"a", "b"})


class SourceSelectionApplicationTests(unittest.TestCase):
    @staticmethod
    def _fixture():
        source_ids = ["keep", "remove", "p1-other", "p2-other"]
        sessions = [TUE_P2_SESSION, TUE_P1_SESSION, TUE_P1_SESSION, TUE_P2_SESSION]
        kinds = [
            "same_track_recovered_cctv_medoid",
            "human_approved_cctv_crop",
            "human_approved_cctv_crop",
            "same_track_recovered_cctv_medoid",
        ]
        staging = pd.DataFrame(
            {
                "Source_ID": source_ids + ["unusable"],
                "Candidate_Roll": ["r1", "r2", "r3", "r4", "r5"],
                "Source_Session": sessions + [TUE_P1_SESSION],
                "Embedding_Usable": [True, True, True, True, False],
            }
        )
        records = [
            {
                "roll_no": f"r{index}",
                "image_path": f"{source_id}.png",
                "detection_score": 0.9,
                "embedding": np.eye(4, 128, dtype=np.float32)[index - 1],
                "_source_id": source_id,
                "_source_kind": kind,
                "_source_session": session,
            }
            for index, (source_id, session, kind) in enumerate(
                zip(source_ids, sessions, kinds), start=1
            )
        ]
        manifest = {
            "embedding_usable_crops": 4,
            "embedding_usable_session_counts": {TUE_P1_SESSION: 2, TUE_P2_SESSION: 2},
            "same_track_recovered_medoid_crops": 2,
            "minimum_usable_evidence_policy": {},
        }
        contract = CctvSourceSelectionContract(
            selection_audit_id="selection-audit-test",
            selection_policy_version="selection-v1",
            source_ablation_id="source-ablation-test",
            corrected_config_id="ABL-test",
            kept_source_ids=("keep",),
            removed_source_ids=("remove",),
            selection_audit_output_manifest_sha256="a" * 64,
            selection_decision_sha256="b" * 64,
            source_ablation_output_manifest_sha256="c" * 64,
        )
        return staging, records, manifest, contract

    def test_selection_removes_only_explicit_source(self) -> None:
        staging, records, manifest, contract = self._fixture()
        selected, filtered, updated, summary, audit = apply_cctv_source_selection(
            staging=staging,
            records=records,
            staging_manifest=manifest,
            contract=contract,
        )
        active = {row["_source_id"] for row in filtered}
        self.assertEqual(active, {"keep", "p1-other", "p2-other"})
        self.assertEqual(summary["removed_cctv_embedding_records"], 1)
        self.assertEqual(updated["embedding_usable_crops"], 3)
        self.assertEqual(updated["embedding_usable_session_counts"][TUE_P1_SESSION], 1)
        self.assertEqual(updated["embedding_usable_session_counts"][TUE_P2_SESSION], 2)
        self.assertEqual(updated["same_track_recovered_medoid_crops"], 2)
        self.assertEqual(
            set(audit.loc[audit["Selected_For_Candidate"], "Source_ID"]), active
        )
        unusable = selected[selected["Source_ID"].eq("unusable")].iloc[0]
        self.assertFalse(bool(unusable["Selected_For_Candidate"]))

    def test_selection_preserves_staged_removed_source_for_audit(self) -> None:
        staging, records, manifest, contract = self._fixture()
        selected, _filtered, updated, _summary, _audit = apply_cctv_source_selection(
            staging=staging,
            records=records,
            staging_manifest=manifest,
            contract=contract,
        )
        removed = selected[selected["Source_ID"].eq("remove")].iloc[0]
        self.assertEqual(
            removed["Source_Selection_Decision"],
            "removed_by_phase_1_2l_1_selection",
        )
        self.assertFalse(bool(removed["Selected_For_Candidate"]))
        self.assertEqual(len(updated["records"]), len(staging))

    def test_missing_implicated_source_fails_closed(self) -> None:
        staging, records, manifest, contract = self._fixture()
        records = [row for row in records if row["_source_id"] != "remove"]
        with self.assertRaises(EmbeddingFamilyError):
            apply_cctv_source_selection(
                staging=staging,
                records=records,
                staging_manifest=manifest,
                contract=contract,
            )


class SourceCompositionTests(unittest.TestCase):
    @staticmethod
    def _write_variant(root: Path, key: str, rows: list[tuple[str, np.ndarray]]) -> None:
        names = {
            "A": "cleaned_enrollment_only",
            "B": "cross_session_for_tue_p1",
            "C": "cross_session_for_tue_p2",
            "D": "full_candidate",
        }
        directory = root / "variants" / names[key]
        directory.mkdir(parents=True)
        source_rows = []
        records = []
        for order, (source_id, vector) in enumerate(rows):
            source_rows.append(
                {
                    "Record_Order": order,
                    "Source_ID": source_id,
                    "Canonical_Roll": "roll-a",
                    "Source_Kind": "cleaned_enrollment",
                    "Source_Session": "",
                    "Image_Path": f"{source_id}.png",
                }
            )
            records.append(
                {
                    "roll_no": "roll-a",
                    "image_path": f"{source_id}.png",
                    "detection_score": 0.9,
                    "embedding": np.asarray(vector, dtype=np.float32),
                }
            )
        pd.DataFrame(source_rows).to_csv(directory / "source_records.csv", index=False)
        with (directory / "student_embeddings.pkl").open("wb") as file_obj:
            pickle.dump({"records": records, "metadata": {}}, file_obj)

    def test_parent_minus_removed_composition_and_vectors_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "parent"
            child = root / "child"
            keep = np.ones(128, dtype=np.float32)
            removed = np.zeros(128, dtype=np.float32)
            for key in ("A", "B", "C", "D"):
                self._write_variant(parent, key, [("keep", keep), (REMOVED_SOURCE_IDS[0], removed)])
                self._write_variant(child, key, [("keep", keep)])
            with mock.patch.dict(
                "src.face_attendance.phase_1_2m_candidate_family.EXPECTED_VARIANT_RECORD_COUNTS",
                {"A": 1, "B": 1, "C": 1, "D": 1},
                clear=True,
            ):
                composition, vectors = _compare_source_composition(
                    parent_family_dir=parent, family_dir=child
                )
            self.assertTrue(composition["Present_In_New_Family"].any())
            self.assertTrue(vectors["Vector_Reproduced"].all())

    def test_vector_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "parent"
            child = root / "child"
            parent_vector = np.ones(128, dtype=np.float32)
            child_vector = parent_vector.copy()
            child_vector[0] += 0.01
            for key in ("A", "B", "C", "D"):
                self._write_variant(parent, key, [("keep", parent_vector)])
                self._write_variant(child, key, [("keep", child_vector)])
            with mock.patch.dict(
                "src.face_attendance.phase_1_2m_candidate_family.EXPECTED_VARIANT_RECORD_COUNTS",
                {"A": 1, "B": 1, "C": 1, "D": 1},
                clear=True,
            ):
                with self.assertRaises(Phase12MError):
                    _compare_source_composition(
                        parent_family_dir=parent, family_dir=child
                    )


class PredictionReproductionTests(unittest.TestCase):
    @staticmethod
    def _database(path: Path) -> None:
        records = [
            {
                "roll_no": "roll-a",
                "image_path": "a.png",
                "detection_score": 1.0,
                "embedding": np.r_[1.0, np.zeros(127)].astype(np.float32),
            },
            {
                "roll_no": "roll-b",
                "image_path": "b.png",
                "detection_score": 1.0,
                "embedding": np.r_[0.0, 1.0, np.zeros(126)].astype(np.float32),
            },
        ]
        with path.open("wb") as file_obj:
            pickle.dump({"records": records, "metadata": {}}, file_obj)

    def test_exact_prediction_reproduction_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            database = Path(tmp) / "db.pkl"
            self._database(database)
            features = np.asarray([np.r_[1.0, np.zeros(127)]], dtype=np.float32)
            expected = pd.DataFrame(
                [
                    {
                        "Row_ID": "row-1",
                        "Accepted": True,
                        "Best_Roll": "roll-a",
                        "Best_Score": 1.0,
                        "Second_Roll": "roll-b",
                        "Second_Score": 0.0,
                        "Margin": 1.0,
                        "Reason": "accepted",
                    }
                ]
            )
            result = _score_expected(
                database_path=database,
                features=features,
                ids=["row-1"],
                expected=expected,
                id_column="Row_ID",
                label="unit",
            )
            self.assertTrue(result["Reproduction_Passed"].all())

    def test_prediction_score_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            database = Path(tmp) / "db.pkl"
            self._database(database)
            features = np.asarray([np.r_[1.0, np.zeros(127)]], dtype=np.float32)
            expected = pd.DataFrame(
                [
                    {
                        "Row_ID": "row-1",
                        "Accepted": True,
                        "Best_Roll": "roll-a",
                        "Best_Score": 0.9,
                        "Second_Roll": "roll-b",
                        "Second_Score": 0.0,
                        "Margin": 0.9,
                        "Reason": "accepted",
                    }
                ]
            )
            with self.assertRaises(Phase12MError):
                _score_expected(
                    database_path=database,
                    features=features,
                    ids=["row-1"],
                    expected=expected,
                    id_column="Row_ID",
                    label="unit",
                )


if __name__ == "__main__":
    unittest.main()
