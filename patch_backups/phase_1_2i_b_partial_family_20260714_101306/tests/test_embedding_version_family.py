from __future__ import annotations

import hashlib
from contextlib import ExitStack
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.face_attendance.embedding_version_family import (
    CCTV_APPROVE,
    ENROLLMENT_DUPLICATE,
    ENROLLMENT_KEEP,
    ENROLLMENT_MULTIPLE,
    ENROLLMENT_WRONG,
    SOURCE_IDENTITY,
    TARGET_IDENTITY,
    TUE_P1_SESSION,
    TUE_P2_SESSION,
    CompactApprovalBundle,
    EmbeddingFamilyError,
    IdentityCorrectionResult,
    _audit_single_cctv_embedding,
    _extract_single_cctv_embedding,
    _normalize_duplicate_references,
    apply_family_safety_gates,
    apply_identity_correction_to_approval_bundle,
    preflight_embedding_family_inputs,
    discover_enrollment_sources,
    select_balanced_sources,
    stage_approved_cctv_sources,
)
from src.face_attendance.enrollment_preprocessing import (
    EnrollmentBuildConfig,
    EnrollmentExtraction,
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _extraction(vector: list[float], rank: float = 1.0) -> EnrollmentExtraction:
    normalized = np.asarray(vector, dtype=np.float32)
    normalized /= np.linalg.norm(normalized)
    return EnrollmentExtraction(
        embedding=normalized,
        reason="accepted",
        detection_count=1,
        accepted_candidate_count=1,
        detector_score=0.9,
        rank_score=rank,
        image_width=100,
        image_height=100,
    )


def _correction(hashes: set[str], item_ids: set[str] | None = None) -> IdentityCorrectionResult:
    return IdentityCorrectionResult(
        manifest={
            "correction_id": "idcorr-test",
            "source_roll": SOURCE_IDENTITY,
            "target_roll": TARGET_IDENTITY,
        },
        items=pd.DataFrame(),
        affected_hashes=frozenset(hashes),
        affected_enrollment_item_ids=frozenset(item_ids or set()),
        affected_cctv_item_ids=frozenset(),
    )


def _approval_bundle(enrollment: pd.DataFrame) -> CompactApprovalBundle:
    cctv = pd.DataFrame(
        [
            {
                "Package_ID": "cctv-package",
                "Item_ID": "CCTV-1",
                "Review_Action": CCTV_APPROVE,
                "Candidate_Roll": "2401100CSE0001",
                "Actual_Roll": "2401100CSE0001",
                "Source_Session": TUE_P1_SESSION,
            }
        ]
    )
    return CompactApprovalBundle(
        enrollment=enrollment,
        cctv=cctv,
        summary={"corrected_enrollment_item_ids": []},
        input_manifest={},
    )


def _source_row(
    source_id: str,
    relative_path: str,
    *,
    source_hash: str,
    lineage: str,
    marker: str,
    rank_action: str = "not_selected_for_compact_review",
) -> dict[str, object]:
    return {
        "Source_ID": source_id,
        "Candidate_Owner": "2401100CSE0001",
        "Relative_Path": relative_path,
        "Source_SHA256": source_hash,
        "Review_Item_ID": "",
        "Review_Action": rank_action,
        "Policy_Eligible": True,
        "Policy_Reason": "deferred_unreviewed_preserved_by_baseline_policy",
        "Lineage_Stem": lineage,
        "Augmentation_Marker": marker,
        "Is_Augmentation": bool(marker and marker not in {"original", "orig"}),
        "Is_Explicit_Original": marker in {"original", "orig"},
    }


def _prediction_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    defaults = {
        "Package_ID": "pkg",
        "Review_ID": "review",
        "Tracklet_ID": "track",
        "Checkpoint_ID": "CP1",
        "Camera_ID": "front",
        "Baseline_Status": "",
        "Review_Status": "identified",
        "Actual_Class": "known_student",
        "Actual_Roll": "2401100CSE0001",
        "Accepted": False,
        "Accepted_Roll": "",
        "Correct_Accepted": False,
        "Wrong_Accepted": False,
        "Outsider_Absorption": False,
        "Mixed_Acceptance": False,
        "Unverifiable_Acceptance": False,
        "Incorrect_Unresolved": True,
        "Top1_Score": 0.0,
        "Margin": 0.0,
        "Feature_Available": True,
        "Source_Kind": "tue_p1_corrected_unresolved_review",
    }
    materialized = []
    for index, row in enumerate(rows):
        value = dict(defaults)
        value.update(row)
        value.setdefault("Benchmark_Row_ID", f"ROW-{index + 1}")
        value.setdefault("Session_ID", TUE_P1_SESSION)
        materialized.append(value)
    return pd.DataFrame(materialized)


def _variant_manifests(coverage: int = 1) -> dict[str, dict[str, object]]:
    return {
        "A": {
            "source_sessions_included": [],
            "evaluation_sessions_allowed": [TUE_P1_SESSION, TUE_P2_SESSION],
            "training_contaminated_descriptive_only": False,
            "canonical_identity_coverage": {"covered": coverage},
        },
        "B": {
            "source_sessions_included": [TUE_P2_SESSION],
            "evaluation_sessions_allowed": [TUE_P1_SESSION],
            "training_contaminated_descriptive_only": False,
            "canonical_identity_coverage": {"covered": coverage},
        },
        "C": {
            "source_sessions_included": [TUE_P1_SESSION],
            "evaluation_sessions_allowed": [TUE_P2_SESSION],
            "training_contaminated_descriptive_only": False,
            "canonical_identity_coverage": {"covered": coverage},
        },
        "D": {
            "source_sessions_included": [TUE_P1_SESSION, TUE_P2_SESSION],
            "evaluation_sessions_allowed": [],
            "training_contaminated_descriptive_only": True,
            "canonical_identity_coverage": {"covered": coverage},
        },
    }


class CompactDuplicateNormalizationTests(unittest.TestCase):
    def test_visible_compact_rank_resolves_to_kept_opaque_item_id(self) -> None:
        public = {
            "review_items": [
                {"item_id": "CEA-KEEP", "public": {"Compact_Rank": "7"}},
                {"item_id": "CEA-DUP", "public": {"Compact_Rank": "8"}},
            ]
        }
        approvals = pd.DataFrame(
            [
                {
                    "Item_ID": "CEA-KEEP",
                    "Review_Action": ENROLLMENT_KEEP,
                    "Duplicate_Of_Item_ID": "",
                },
                {
                    "Item_ID": "CEA-DUP",
                    "Review_Action": ENROLLMENT_DUPLICATE,
                    "Duplicate_Of_Item_ID": "7",
                },
            ]
        )
        result = _normalize_duplicate_references(approvals, public)
        duplicate = result[result["Item_ID"].eq("CEA-DUP")].iloc[0]
        self.assertEqual(duplicate["Duplicate_Of_Item_ID_Raw"], "7")
        self.assertEqual(duplicate["Duplicate_Of_Item_ID"], "CEA-KEEP")

    def test_duplicate_target_must_be_keep(self) -> None:
        public = {
            "review_items": [
                {"item_id": "CEA-A", "public": {"Compact_Rank": "1"}},
                {"item_id": "CEA-B", "public": {"Compact_Rank": "2"}},
            ]
        }
        approvals = pd.DataFrame(
            [
                {
                    "Item_ID": "CEA-A",
                    "Review_Action": ENROLLMENT_MULTIPLE,
                    "Duplicate_Of_Item_ID": "",
                },
                {
                    "Item_ID": "CEA-B",
                    "Review_Action": ENROLLMENT_DUPLICATE,
                    "Duplicate_Of_Item_ID": "1",
                },
            ]
        )
        with self.assertRaisesRegex(EmbeddingFamilyError, "must be marked keep"):
            _normalize_duplicate_references(approvals, public)

    def test_nonduplicate_row_cannot_smuggle_duplicate_reference(self) -> None:
        public = {
            "review_items": [
                {"item_id": "CEA-A", "public": {"Compact_Rank": "1"}}
            ]
        }
        approvals = pd.DataFrame(
            [
                {
                    "Item_ID": "CEA-A",
                    "Review_Action": ENROLLMENT_KEEP,
                    "Duplicate_Of_Item_ID": "1",
                }
            ]
        )
        with self.assertRaisesRegex(EmbeddingFamilyError, "without the duplicate action"):
            _normalize_duplicate_references(approvals, public)


class IdentityCorrectionPropagationTests(unittest.TestCase):
    def test_folder_level_correction_reassigns_all_review_actions_not_only_wrong_rows(self) -> None:
        hashes = {"h-keep", "h-duplicate", "h-multiple", "h-wrong"}
        enrollment = pd.DataFrame(
            [
                {
                    "Item_ID": "KEEP",
                    "Expected_Roll": SOURCE_IDENTITY,
                    "Candidate_Roll": SOURCE_IDENTITY,
                    "Source_SHA256": "h-keep",
                    "Review_Action": ENROLLMENT_KEEP,
                },
                {
                    "Item_ID": "DUP",
                    "Expected_Roll": SOURCE_IDENTITY,
                    "Candidate_Roll": SOURCE_IDENTITY,
                    "Source_SHA256": "h-duplicate",
                    "Review_Action": ENROLLMENT_DUPLICATE,
                },
                {
                    "Item_ID": "MULTI",
                    "Expected_Roll": SOURCE_IDENTITY,
                    "Candidate_Roll": SOURCE_IDENTITY,
                    "Source_SHA256": "h-multiple",
                    "Review_Action": ENROLLMENT_MULTIPLE,
                },
                {
                    "Item_ID": "WRONG",
                    "Expected_Roll": SOURCE_IDENTITY,
                    "Candidate_Roll": TARGET_IDENTITY,
                    "Source_SHA256": "h-wrong",
                    "Review_Action": ENROLLMENT_WRONG,
                },
                {
                    "Item_ID": "OTHER",
                    "Expected_Roll": "2401100CSE0002",
                    "Candidate_Roll": "2401100CSE0002",
                    "Source_SHA256": "h-other",
                    "Review_Action": ENROLLMENT_KEEP,
                },
            ]
        )
        result = apply_identity_correction_to_approval_bundle(
            _approval_bundle(enrollment), _correction(hashes, {"WRONG"})
        )
        affected = result.enrollment[
            result.enrollment["Source_SHA256"].isin(hashes)
        ]
        self.assertEqual(set(affected["Candidate_Roll"]), {TARGET_IDENTITY})
        self.assertTrue(affected["Identity_Correction_Applied"].all())
        other = result.enrollment[result.enrollment["Item_ID"].eq("OTHER")].iloc[0]
        self.assertEqual(other["Candidate_Roll"], "2401100CSE0002")
        self.assertFalse(bool(other["Identity_Correction_Applied"]))
        self.assertEqual(result.summary["reconciled_enrollment_review_rows"], 4)

    def test_discovery_honors_duplicate_and_multiple_people_after_folder_rename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset"
            target = dataset / TARGET_IDENTITY
            target.mkdir(parents=True)
            payloads = {
                "keep.jpg": b"keep",
                "duplicate.jpg": b"duplicate",
                "multiple.jpg": b"multiple",
                "wrong.jpg": b"wrong",
            }
            rows = []
            actions = {
                "keep.jpg": ENROLLMENT_KEEP,
                "duplicate.jpg": ENROLLMENT_DUPLICATE,
                "multiple.jpg": ENROLLMENT_MULTIPLE,
                "wrong.jpg": ENROLLMENT_WRONG,
            }
            hashes: set[str] = set()
            for index, (name, payload) in enumerate(payloads.items(), start=1):
                path = target / name
                path.write_bytes(payload)
                source_hash = _sha256_bytes(payload)
                hashes.add(source_hash)
                rows.append(
                    {
                        "Item_ID": f"ITEM-{index}",
                        "Expected_Roll": SOURCE_IDENTITY,
                        "Candidate_Roll": SOURCE_IDENTITY,
                        "Source_SHA256": source_hash,
                        "Review_Action": actions[name],
                        "Duplicate_Of_Item_ID": "",
                        "Duplicate_Target_Source_SHA256": "",
                    }
                )
            bundle = apply_identity_correction_to_approval_bundle(
                _approval_bundle(pd.DataFrame(rows)), _correction(hashes, {"ITEM-4"})
            )
            discovered = discover_enrollment_sources(
                dataset_root=dataset,
                approval_bundle=bundle,
                correction=_correction(hashes, {"ITEM-4"}),
            )
            by_name = {
                Path(row["Source_Path"]).name: row
                for row in discovered.to_dict("records")
            }
            self.assertTrue(bool(by_name["keep.jpg"]["Policy_Eligible"]))
            self.assertTrue(bool(by_name["wrong.jpg"]["Policy_Eligible"]))
            self.assertFalse(bool(by_name["duplicate.jpg"]["Policy_Eligible"]))
            self.assertEqual(
                by_name["duplicate.jpg"]["Policy_Reason"],
                "reviewer_confirmed_duplicate",
            )
            self.assertFalse(bool(by_name["multiple.jpg"]["Policy_Eligible"]))
            self.assertEqual(
                by_name["multiple.jpg"]["Policy_Reason"],
                "reviewer_confirmed_multiple_people",
            )
            self.assertEqual(
                {row["Candidate_Owner"] for row in by_name.values()},
                {TARGET_IDENTITY},
            )


class AugmentationBalancingTests(unittest.TestCase):
    def test_original_plus_only_one_meaningful_augmentation_is_selected(self) -> None:
        sources = pd.DataFrame(
            [
                _source_row(
                    "original",
                    "2401100CSE0001/face_original.jpg",
                    source_hash="hash-original",
                    lineage="face",
                    marker="original",
                    rank_action=ENROLLMENT_KEEP,
                ),
                _source_row(
                    "rot-left",
                    "2401100CSE0001/face_rot_left.jpg",
                    source_hash="hash-left",
                    lineage="face",
                    marker="rot_left",
                ),
                _source_row(
                    "rot-right",
                    "2401100CSE0001/face_rot_right.jpg",
                    source_hash="hash-right",
                    lineage="face",
                    marker="rot_right",
                ),
            ]
        )
        result = select_balanced_sources(
            sources,
            {
                "original": _extraction([1.0, 0.0, 0.0], 3.0),
                "rot-left": _extraction([0.8, 0.6, 0.0], 2.0),
                "rot-right": _extraction([0.75, 0.66, 0.0], 1.0),
            },
        )
        selected = result.sources[result.sources["Selected_For_Candidate"].map(bool)]
        self.assertEqual(len(selected), 2)
        self.assertIn("original", set(selected["Source_ID"]))
        self.assertEqual(int(selected["Is_Augmentation"].map(bool).sum()), 1)
        self.assertLessEqual(selected["Source_Group_ID"].value_counts().max(), 2)

    def test_independent_real_photos_are_not_collapsed(self) -> None:
        sources = pd.DataFrame(
            [
                _source_row(
                    "real-a",
                    "2401100CSE0001/photo_a.jpg",
                    source_hash="hash-a",
                    lineage="photo_a",
                    marker="",
                ),
                _source_row(
                    "real-b",
                    "2401100CSE0001/photo_b.jpg",
                    source_hash="hash-b",
                    lineage="photo_b",
                    marker="",
                ),
            ]
        )
        result = select_balanced_sources(
            sources,
            {
                "real-a": _extraction([1.0, 0.0], 1.0),
                "real-b": _extraction([0.0, 1.0], 1.0),
            },
        )
        self.assertEqual(set(result.selected_source_ids), {"real-a", "real-b"})
        self.assertEqual(result.sources["Source_Group_ID"].nunique(), 2)

    def test_embedding_inconsistent_augmentation_is_excluded(self) -> None:
        sources = pd.DataFrame(
            [
                _source_row(
                    "original",
                    "2401100CSE0001/face_original.jpg",
                    source_hash="hash-original",
                    lineage="face",
                    marker="original",
                ),
                _source_row(
                    "bad-aug",
                    "2401100CSE0001/face_rot_left.jpg",
                    source_hash="hash-aug",
                    lineage="face",
                    marker="rot_left",
                ),
            ]
        )
        result = select_balanced_sources(
            sources,
            {
                "original": _extraction([1.0, 0.0], 2.0),
                "bad-aug": _extraction([0.0, 1.0], 1.0),
            },
        )
        bad = result.sources[result.sources["Source_ID"].eq("bad-aug")].iloc[0]
        self.assertFalse(bool(bad["Selected_For_Candidate"]))
        self.assertEqual(bad["Selection_Reason"], "augmentation_embedding_inconsistent")


class SourceOwnershipIntegrityTests(unittest.TestCase):
    def test_exact_image_hash_cannot_be_owned_by_two_students(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset"
            first = dataset / "2401100CSE0001"
            second = dataset / "2401100CSE0002"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            (first / "face.jpg").write_bytes(b"same-person-file")
            (second / "face-copy.jpg").write_bytes(b"same-person-file")
            with self.assertRaisesRegex(
                EmbeddingFamilyError, "owned by multiple canonical identities"
            ):
                discover_enrollment_sources(
                    dataset_root=dataset,
                    approval_bundle=_approval_bundle(pd.DataFrame()),
                    correction=_correction(set()),
                )


class PreflightIsolationTests(unittest.TestCase):
    def test_preflight_is_read_only_and_does_not_initialize_face_models(self) -> None:
        roster = {TARGET_IDENTITY}
        roster.update(f"2401100CSE{index:04d}" for index in range(1, 27))
        roster.discard(SOURCE_IDENTITY)
        self.assertEqual(len(roster), 27)

        approvals = CompactApprovalBundle(
            enrollment=pd.DataFrame(
                [
                    {
                        "Item_ID": "ITEM-1",
                        "Identity_Correction_Applied": True,
                    }
                ]
            ),
            cctv=pd.DataFrame(
                [
                    {
                        "Review_Action": CCTV_APPROVE,
                        "Source_Session": TUE_P1_SESSION,
                    }
                ]
            ),
            summary={"enrollment_rows": 50, "cctv_rows": 36},
            input_manifest={},
        )
        correction = _correction({"hash-1"}, {"ITEM-1"})
        discovered = pd.DataFrame(
            [
                {
                    "Policy_Eligible": True,
                    "Policy_Reason": "reviewer_confirmed_keep",
                },
                {
                    "Policy_Eligible": False,
                    "Policy_Reason": "reviewer_confirmed_duplicate",
                },
            ]
        )
        snapshot = {"fixed_files": {}, "trees": {}}
        benchmark = pd.DataFrame(
            [{"Benchmark_Row_ID": "ROW-1"}, {"Benchmark_Row_ID": "ROW-2"}]
        )

        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "src.face_attendance.embedding_version_family._ensure_production_preflight",
                    return_value={
                        "production_embeddings": {"sha256": "emb"},
                        "production_summary": {"sha256": "sum"},
                    },
                )
            )
            stack.enter_context(
                patch(
                    "src.face_attendance.embedding_version_family.validate_compact_approval_bundle",
                    return_value=approvals,
                )
            )
            stack.enter_context(
                patch(
                    "src.face_attendance.embedding_version_family.reconcile_identity_correction",
                    return_value=correction,
                )
            )
            stack.enter_context(
                patch(
                    "src.face_attendance.embedding_version_family.apply_identity_correction_to_approval_bundle",
                    return_value=approvals,
                )
            )
            stack.enter_context(
                patch(
                    "src.face_attendance.embedding_version_family.reconcile_shruthi_source",
                    return_value={"status": "found_unique_renamed_folder_match"},
                )
            )
            stack.enter_context(
                patch(
                    "src.face_attendance.embedding_version_family.load_student_mapping",
                    return_value=([], {}, roster),
                )
            )
            stack.enter_context(
                patch(
                    "src.face_attendance.embedding_version_family.snapshot_phase_i_b_protected_state",
                    return_value=snapshot,
                )
            )
            stack.enter_context(
                patch(
                    "src.face_attendance.embedding_version_family.discover_enrollment_sources",
                    return_value=discovered,
                )
            )
            stack.enter_context(
                patch(
                    "src.face_attendance.embedding_version_family.load_frozen_benchmark_evidence",
                    return_value=(benchmark, {"benchmark_id": "bench"}),
                )
            )
            face_engine = stack.enter_context(
                patch("src.face_attendance.embedding_version_family.FaceEngine")
            )
            write_json = stack.enter_context(
                patch("src.face_attendance.embedding_version_family._write_json")
            )
            write_csv = stack.enter_context(
                patch("src.face_attendance.embedding_version_family._write_csv")
            )

            result = preflight_embedding_family_inputs(
                repo_root=Path("repo"),
                forensic_run_dir=Path("forensic"),
                enrollment_review_package=Path("enrollment-package"),
                enrollment_approvals_path=Path("enrollment.csv"),
                cctv_review_package=Path("cctv-package"),
                cctv_approvals_path=Path("cctv.csv"),
                production_embeddings_path=Path("student_embeddings.pkl"),
                production_summary_path=Path("embedding_summary.csv"),
                student_map_path=Path("student_map.json"),
                dataset_root=Path("dataset"),
                augmented_dataset_root=Path("augmented_dataset"),
                expected_historical_images=None,
            )

        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["read_only"])
        self.assertFalse(result["candidate_build_started"])
        self.assertFalse(result["face_extraction_started"])
        self.assertTrue(result["protected_state_unchanged"])
        scalability = result["review_scalability_policy"]
        self.assertTrue(
            scalability["current_manual_reviews_are_one_time_bootstrap_ground_truth"]
        )
        self.assertEqual(scalability["live_review_mode"], "exception_only")
        self.assertTrue(
            scalability["review_work_must_not_scale_linearly_with_frames_or_cameras"]
        )
        face_engine.assert_not_called()
        write_json.assert_not_called()
        write_csv.assert_not_called()


class CctvPreprocessingTests(unittest.TestCase):
    @staticmethod
    def _cctv_metrics(*, usable: bool, reason: str = "") -> dict[str, object]:
        return {
            "width": 80,
            "height": 90,
            "detector_score": 0.85 if usable else 0.0,
            "rank_score": 1.0 if usable else -999.0,
            "detection_count": 1,
            "accepted_candidate_count": 1 if usable else 0,
            "preprocessing_policy": "padded_photo_landmark_validation",
            "embedding_usable": usable,
            "embedding_status": (
                "embedding_usable" if usable else "approved_but_embedding_unusable"
            ),
            "embedding_exclusion_reason": "" if usable else reason,
            "embedding_dimension": 3 if usable else 0,
            "human_approval_preserved": True,
            "manual_rereview_required": False,
            "quality_gate_relaxed": False,
            "super_resolution_used": False,
        }

    @staticmethod
    def _approved_cctv_row(
        *, item_id: str, source_path: Path, session: str, roll: str
    ) -> dict[str, object]:
        return {
            "Package_ID": "compact-cctv-test",
            "Item_ID": item_id,
            "Review_Action": CCTV_APPROVE,
            "Resolved_Source_Path": str(source_path),
            "Source_SHA256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "Private_Provenance": {
                "actual_roll": roll,
                "identity_source": "human_review_actual_roll",
                "source_session": session,
                "source_broad_item_id": f"BROAD-{item_id}",
                "tracklet_id": f"TRACK-{item_id}",
                "observation_id": f"OBS-{item_id}",
                "checkpoint": "CP1",
                "camera": "cam1",
            },
        }

    def test_cctv_crop_reuses_production_compatible_single_face_preprocessing(self) -> None:
        extraction = _extraction([1.0, 0.0, 0.0])
        config = EnrollmentBuildConfig()
        with patch(
            "src.face_attendance.embedding_version_family.extract_enrollment_embedding",
            return_value=extraction,
        ) as mocked:
            vector, metrics = _extract_single_cctv_embedding(
                Path("approved.jpg"), object(), config
            )
        mocked.assert_called_once()
        self.assertTrue(mocked.call_args.kwargs["require_single_face"])
        self.assertEqual(vector.shape, (3,))
        self.assertEqual(metrics["preprocessing_policy"], config.validation_mode)

    def test_cctv_crop_fails_closed_when_single_face_policy_rejects(self) -> None:
        rejected = EnrollmentExtraction(
            embedding=None,
            reason="multiple_faces_not_unambiguous",
            detection_count=2,
            accepted_candidate_count=0,
            detector_score=0.0,
            rank_score=-999.0,
            image_width=100,
            image_height=100,
        )
        with patch(
            "src.face_attendance.embedding_version_family.extract_enrollment_embedding",
            return_value=rejected,
        ):
            with self.assertRaisesRegex(EmbeddingFamilyError, "single-face"):
                _extract_single_cctv_embedding(Path("approved.jpg"), object())

    def test_audit_preserves_human_approval_when_embedding_is_unusable(self) -> None:
        rejected = EnrollmentExtraction(
            embedding=None,
            reason="no_valid_face",
            detection_count=1,
            accepted_candidate_count=0,
            detector_score=0.0,
            rank_score=-999.0,
            image_width=68,
            image_height=82,
        )
        with patch(
            "src.face_attendance.embedding_version_family.extract_enrollment_embedding",
            return_value=rejected,
        ):
            vector, metrics = _audit_single_cctv_embedding(
                Path("human-approved.png"), object()
            )
        self.assertIsNone(vector)
        self.assertFalse(metrics["embedding_usable"])
        self.assertEqual(
            metrics["embedding_status"], "approved_but_embedding_unusable"
        )
        self.assertEqual(metrics["embedding_exclusion_reason"], "no_valid_face")
        self.assertTrue(metrics["human_approval_preserved"])
        self.assertFalse(metrics["manual_rereview_required"])
        self.assertFalse(metrics["quality_gate_relaxed"])

    def test_staging_audits_all_approved_crops_and_uses_only_safe_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = root / "sources"
            sources.mkdir()
            p1_bad = sources / "p1-bad.png"
            p1_good = sources / "p1-good.png"
            p2_good = sources / "p2-good.png"
            p1_bad.write_bytes(b"p1-bad")
            p1_good.write_bytes(b"p1-good")
            p2_good.write_bytes(b"p2-good")
            roll = "2401100CSE0001"
            cctv = pd.DataFrame(
                [
                    self._approved_cctv_row(
                        item_id="CCC-001",
                        source_path=p1_bad,
                        session=TUE_P1_SESSION,
                        roll=roll,
                    ),
                    self._approved_cctv_row(
                        item_id="CCC-002",
                        source_path=p1_good,
                        session=TUE_P1_SESSION,
                        roll=roll,
                    ),
                    self._approved_cctv_row(
                        item_id="CCC-003",
                        source_path=p2_good,
                        session=TUE_P2_SESSION,
                        roll=roll,
                    ),
                ]
            )
            bundle = CompactApprovalBundle(
                enrollment=pd.DataFrame(),
                cctv=cctv,
                summary={},
                input_manifest={},
            )

            def audit(path: Path, _engine: object):
                if Path(path).stem == "CCC-001":
                    return None, self._cctv_metrics(
                        usable=False, reason="no_valid_face"
                    )
                return np.asarray([1.0, 0.0, 0.0], dtype=np.float32), self._cctv_metrics(
                    usable=True
                )

            with patch(
                "src.face_attendance.embedding_version_family._audit_single_cctv_embedding",
                side_effect=audit,
            ) as mocked:
                staging, records, manifest = stage_approved_cctv_sources(
                    temporary_family_dir=root / "building",
                    final_family_dir=root / "final",
                    repo_root=root,
                    approval_bundle=bundle,
                    correction=_correction(set()),
                    engine=object(),
                    roster_rolls={roll},
                    expected_approved=3,
                    progress=None,
                )

            self.assertEqual(mocked.call_count, 3)
            self.assertEqual(len(staging), 3)
            self.assertEqual(len(records), 2)
            self.assertEqual(manifest["approved_rows"], 3)
            self.assertEqual(manifest["embedding_usable_crops"], 2)
            self.assertEqual(
                manifest["approved_but_embedding_unusable_crops"], 1
            )
            self.assertTrue(manifest["all_approved_crops_audited"])
            self.assertTrue(manifest["minimum_usable_evidence_passed"])
            bad = staging[staging["Item_ID"].eq("CCC-001")].iloc[0]
            self.assertFalse(bool(bad["Embedding_Usable"]))
            self.assertEqual(
                bad["Embedding_Status"], "approved_but_embedding_unusable"
            )
            self.assertEqual(bad["Embedding_Exclusion_Reason"], "no_valid_face")
            self.assertFalse(bool(bad["Manual_Rereview_Required"]))
            self.assertNotIn("CCTV-CCC-001", {row["_source_id"] for row in records})

    def test_staging_reports_missing_usable_session_after_auditing_every_crop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = root / "sources"
            sources.mkdir()
            p1_bad = sources / "p1-bad.png"
            p2_good = sources / "p2-good.png"
            p1_bad.write_bytes(b"p1-bad")
            p2_good.write_bytes(b"p2-good")
            roll = "2401100CSE0001"
            bundle = CompactApprovalBundle(
                enrollment=pd.DataFrame(),
                cctv=pd.DataFrame(
                    [
                        self._approved_cctv_row(
                            item_id="CCC-001",
                            source_path=p1_bad,
                            session=TUE_P1_SESSION,
                            roll=roll,
                        ),
                        self._approved_cctv_row(
                            item_id="CCC-002",
                            source_path=p2_good,
                            session=TUE_P2_SESSION,
                            roll=roll,
                        ),
                    ]
                ),
                summary={},
                input_manifest={},
            )

            def audit(path: Path, _engine: object):
                if Path(path).stem == "CCC-001":
                    return None, self._cctv_metrics(
                        usable=False, reason="no_valid_face"
                    )
                return np.asarray([1.0, 0.0, 0.0], dtype=np.float32), self._cctv_metrics(
                    usable=True
                )

            with patch(
                "src.face_attendance.embedding_version_family._audit_single_cctv_embedding",
                side_effect=audit,
            ) as mocked:
                staging, records, manifest = stage_approved_cctv_sources(
                    temporary_family_dir=root / "building",
                    final_family_dir=root / "final",
                    repo_root=root,
                    approval_bundle=bundle,
                    correction=_correction(set()),
                    engine=object(),
                    roster_rolls={roll},
                    expected_approved=2,
                    progress=None,
                )

            self.assertEqual(mocked.call_count, 2)
            self.assertEqual(len(staging), 2)
            self.assertEqual(len(records), 1)
            self.assertFalse(manifest["minimum_usable_evidence_passed"])
            self.assertEqual(manifest["missing_usable_sessions"], [TUE_P1_SESSION])
            self.assertFalse(manifest["manual_rereview_required"])
            self.assertFalse(manifest["quality_gate_relaxed"])


class SafetyGateTests(unittest.TestCase):
    def _safe_frames(self) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        baseline = _prediction_frame(
            [
                {
                    "Benchmark_Row_ID": "P1-ACCEPTED",
                    "Session_ID": TUE_P1_SESSION,
                    "Source_Kind": "tue_p1_accepted_review",
                    "Accepted": True,
                    "Accepted_Roll": "2401100CSE0001",
                    "Correct_Accepted": True,
                    "Incorrect_Unresolved": False,
                    "Top1_Score": 0.8,
                    "Margin": 0.2,
                },
                {
                    "Benchmark_Row_ID": "P2-UNKNOWN",
                    "Session_ID": TUE_P2_SESSION,
                    "Source_Kind": "tue_p2_shadow_recovery_review",
                    "Review_Status": "not_in_mapping",
                    "Actual_Class": "not_in_mapping",
                    "Actual_Roll": "",
                    "Accepted": False,
                    "Incorrect_Unresolved": False,
                },
            ]
        )
        a = baseline.copy()
        b = baseline[baseline["Session_ID"].eq(TUE_P1_SESSION)].copy()
        c = baseline[baseline["Session_ID"].eq(TUE_P2_SESSION)].copy()
        d = baseline.copy()
        return baseline, {"A": a, "B": b, "C": c, "D": d}

    def test_existing_outsider_acceptance_is_still_an_absolute_candidate_failure(self) -> None:
        baseline, variants = self._safe_frames()
        for frame in (baseline, variants["A"], variants["C"], variants["D"]):
            index = frame.index[frame["Benchmark_Row_ID"].eq("P2-UNKNOWN")][0]
            frame.at[index, "Accepted"] = True
            frame.at[index, "Accepted_Roll"] = "2401100CSE0001"
            frame.at[index, "Outsider_Absorption"] = True
        decision, _ = apply_family_safety_gates(
            baseline=baseline,
            variant_predictions=variants,
            variant_manifests=_variant_manifests(),
            protected_comparison={"unchanged": True},
            production_roster_coverage=1,
            benchmark_expected_rows=2,
            expected_tue_p1_accepted_rows=1,
        )
        self.assertEqual(decision["decision"], "regression_failed_false_identity")
        gate = decision["gates"]["zero_new_false_accepts"]
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["unique_unsafe_benchmark_rows"], 1)
        self.assertEqual(gate["unsafe_model_rows"], 2)

    def test_variant_b_is_decisive_for_tue_p1_retention_while_a_is_diagnostic(self) -> None:
        baseline, variants = self._safe_frames()
        a_index = variants["A"].index[
            variants["A"]["Benchmark_Row_ID"].eq("P1-ACCEPTED")
        ][0]
        variants["A"].at[a_index, "Accepted"] = False
        variants["A"].at[a_index, "Accepted_Roll"] = ""
        variants["A"].at[a_index, "Correct_Accepted"] = False
        variants["A"].at[a_index, "Incorrect_Unresolved"] = True
        decision, _ = apply_family_safety_gates(
            baseline=baseline,
            variant_predictions=variants,
            variant_manifests=_variant_manifests(),
            protected_comparison={"unchanged": True},
            production_roster_coverage=1,
            benchmark_expected_rows=2,
            expected_tue_p1_accepted_rows=1,
        )
        retention = decision["gates"]["reviewed_accepted_track_retention"]
        self.assertTrue(retention["passed"])
        self.assertEqual(retention["decisive_variant"], "B")
        self.assertEqual(
            retention["comparisons"]["A"]["lost_correct_acceptances"], 1
        )
        self.assertEqual(
            retention["comparisons"]["B"]["lost_correct_acceptances"], 0
        )

    def test_cross_session_leakage_fails_integrity_gate(self) -> None:
        baseline, variants = self._safe_frames()
        manifests = _variant_manifests()
        manifests["B"]["source_sessions_included"] = [TUE_P1_SESSION, TUE_P2_SESSION]
        decision, _ = apply_family_safety_gates(
            baseline=baseline,
            variant_predictions=variants,
            variant_manifests=manifests,
            protected_comparison={"unchanged": True},
            production_roster_coverage=1,
            benchmark_expected_rows=2,
            expected_tue_p1_accepted_rows=1,
        )
        self.assertEqual(decision["decision"], "regression_failed_integrity")
        self.assertFalse(decision["gates"]["leakage"]["passed"])

    def test_new_known_student_wrong_accept_fails_false_identity_gate(self) -> None:
        baseline, variants = self._safe_frames()
        index = variants["B"].index[
            variants["B"]["Benchmark_Row_ID"].eq("P1-ACCEPTED")
        ][0]
        variants["B"].at[index, "Accepted"] = True
        variants["B"].at[index, "Accepted_Roll"] = "2401100CSE9999"
        variants["B"].at[index, "Correct_Accepted"] = False
        variants["B"].at[index, "Wrong_Accepted"] = True
        variants["B"].at[index, "Incorrect_Unresolved"] = False
        decision, _ = apply_family_safety_gates(
            baseline=baseline,
            variant_predictions=variants,
            variant_manifests=_variant_manifests(),
            protected_comparison={"unchanged": True},
            production_roster_coverage=1,
            benchmark_expected_rows=2,
            expected_tue_p1_accepted_rows=1,
        )
        self.assertEqual(decision["decision"], "regression_failed_false_identity")
        self.assertFalse(decision["gates"]["zero_new_false_accepts"]["passed"])

    def test_missing_variant_prediction_row_fails_integrity_gate(self) -> None:
        baseline, variants = self._safe_frames()
        variants["A"] = variants["A"][
            ~variants["A"]["Benchmark_Row_ID"].eq("P2-UNKNOWN")
        ].copy()
        decision, _ = apply_family_safety_gates(
            baseline=baseline,
            variant_predictions=variants,
            variant_manifests=_variant_manifests(),
            protected_comparison={"unchanged": True},
            production_roster_coverage=1,
            benchmark_expected_rows=2,
            expected_tue_p1_accepted_rows=1,
        )
        self.assertEqual(decision["decision"], "regression_failed_integrity")
        structure = decision["gates"]["evaluation_structure"]
        self.assertFalse(structure["passed"])
        self.assertTrue(
            any("Variant A benchmark row set changed" in value for value in structure["failures"])
        )


if __name__ == "__main__":
    unittest.main()
