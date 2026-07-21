import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.face_attendance.compact_forensic_review import (
    CompactReviewError,
    export_compact_cctv_review,
    export_compact_enrollment_review,
    generate_compact_forensic_reviews,
    select_compact_cctv_items,
    select_compact_enrollment_items,
    validate_broad_review_package,
    validate_hashed_source_artifacts,
)
from src.face_attendance.forensic_review import (
    ForensicReviewError,
    ForensicReviewItem,
    export_forensic_review_package,
    load_forensic_review_package,
    validate_forensic_approvals,
    write_reviewer_launcher,
)
from src.face_attendance.shadow_validation import create_reviewer_server, write_output_manifest
from src.face_attendance.tracklet_review import _sha256_file


def enrollment_item(
    item_id,
    roll,
    flags,
    *,
    sha="",
    path="",
    role="production_recorded_dataset",
    duplicate_group="",
    own=0.70,
    other=0.20,
    other_roll="2401100CSE9999",
    valid_embedding="Yes",
):
    return {
        "Source_Broad_Item_ID": item_id,
        "Source_Path": path or f"C:/dataset/{roll}/{item_id}.jpg",
        "File_SHA256": sha or f"sha-{item_id}",
        "Canonical_Roll": roll,
        "Expected_Name": f"Student {roll[-4:]}",
        "Dataset_Source_Role": role,
        "Owning_Folder": roll,
        "Relative_Path": f"{roll}/{item_id}.jpg",
        "Audit_Flags": flags,
        "Duplicate_Group_ID": duplicate_group,
        "Decode_Success": "Yes",
        "Detected_Face_Count": "1",
        "Valid_Face_Count": "1",
        "Valid_SFace_Embedding": valid_embedding,
        "Primary_Face_Width": "90",
        "Primary_Face_Height": "110",
        "Blur_Laplacian_Variance": "100",
        "Brightness_Mean": "120",
        "Own_Medoid_Similarity": str(own),
        "Nearest_Other_Roll": other_roll,
        "Nearest_Other_Similarity": str(other),
        "Model_Similarity_Hint_Not_Ground_Truth": (
            f"{other_roll} at cosine {other}; model similarity hint - not ground truth"
        ),
        "In_Master_Registry": "Yes",
        "In_CVO_Roster": "Yes",
    }


def cctv_item(
    item_id,
    roll,
    *,
    checkpoint="CP1",
    camera="cam5",
    track="",
    eligible="Yes",
    actual_status="identified",
    identity_source="human_review_actual_roll",
    predicted_roll="",
    nearest="0.20",
    quality="1.0",
):
    return {
        "Source_Broad_Item_ID": item_id,
        "Item_ID": item_id,
        "Actual_Roll": roll,
        "Actual_Student_Name": f"Student {roll[-4:]}",
        "Actual_Status": actual_status,
        "Identity_Source": identity_source,
        "Human_Eligible": eligible,
        "Model_Prediction_Used_As_Identity": "No",
        "Predicted_Roll": predicted_roll,
        "Source_Kind": "tue_p1_corrected_unresolved_review",
        "Session_ID": "2026-06-30__B51__P1__CVO",
        "Package_ID": "source-package",
        "Review_ID": f"R-{item_id}",
        "Benchmark_Row_ID": f"GT-{item_id}",
        "Tracklet_ID": track or f"TRK-{item_id}",
        "Observation_ID": f"OBS-{item_id}",
        "Checkpoint_ID": checkpoint,
        "Camera_ID": camera,
        "Frame": "100",
        "Crop_Width": "80",
        "Crop_Height": "100",
        "Detector_Score": "0.95",
        "Crop_Blur_Laplacian_Variance": "400",
        "Crop_Brightness_Mean": "120",
        "Pose_Roll_Degrees": "2",
        "Pose_Yaw_Proxy": "0.15",
        "Occlusion_Proxy": "0.05",
        "Selection_Quality_Score": quality,
        "Diversity_Nearest_Selected_Cosine": nearest,
        "Diversity_New_Checkpoint": "Yes",
        "Diversity_New_Camera": "Yes" if camera == "cam10" else "No",
        "Candidate_Crop_Path": f"C:/crops/{item_id}.png",
        "Candidate_Crop_SHA256": f"crop-sha-{item_id}",
        "Selected_For_Review": "Yes",
    }


def attractors(*rolls):
    return pd.DataFrame(
        [
            {
                "Predicted_Roll": roll,
                "Confirmed_False_Accepts": "4",
                "Distinct_Actual_Identities_Attracted": "3",
                "Outsider_Not_In_Mapping_Accepts": "1",
                "Actual_Identities": "2401100CSE0100; 2401100CSE0101",
            }
            for roll in rolls
        ]
    )


def misses(*rolls):
    return pd.DataFrame(
        [
            {
                "Actual_Roll": roll,
                "Reviewed_Tracks": "5",
                "Missed_Top1": "3",
                "Top1_Accuracy": "0.4",
                "Wrong_Predicted_Identities": "2401100CSE0001",
            }
            for roll in rolls
        ]
    )


class CompactEnrollmentSelectionTests(unittest.TestCase):
    def test_identity_integrity_ranks_above_blur_and_low_priority_does_not_pad(self):
        rows = [
            enrollment_item(
                "critical",
                "2401100CSE0001",
                "closer_to_another_student_than_own_medoid",
                own=0.20,
                other=0.72,
            ),
            enrollment_item("multi", "2401100CSE0002", "multiple_faces"),
        ]
        rows.extend(
            enrollment_item(f"blur-{index}", f"2401100CSE{index + 10:04d}", "severe_blur")
            for index in range(20)
        )
        result = select_compact_enrollment_items(
            pd.DataFrame(rows),
            embedding_outliers=pd.DataFrame(),
            attractor_rows=pd.DataFrame(),
            miss_rows=pd.DataFrame(),
            maximum=60,
            preferred=50,
        )
        self.assertEqual(list(result.selected["Source_Broad_Item_ID"]), ["critical", "multi"])
        self.assertEqual(len(result.deferred), 20)
        self.assertFalse(result.summary["low_priority_padding_used"])

    def test_outliers_are_considered_and_selected_when_within_caps(self):
        rows = []
        outlier_rows = []
        for index in range(13):
            roll = f"2401100CSE{index + 1:04d}"
            path = f"C:/dataset/{roll}/outlier-{index}.jpg"
            rows.append(enrollment_item(f"outlier-{index}", roll, "own_class_outlier", path=path))
            outlier_rows.append(
                {
                    "Canonical_Roll": roll,
                    "Image_Path": path,
                    "Robust_Outlier_Score": str(20 - index),
                    "Medoid_Similarity": "0.1",
                }
            )
        result = select_compact_enrollment_items(
            pd.DataFrame(rows),
            embedding_outliers=pd.DataFrame(outlier_rows),
            attractor_rows=pd.DataFrame(),
            miss_rows=pd.DataFrame(),
        )
        self.assertEqual(result.summary["embedding_outliers_considered"], 13)
        self.assertEqual(result.summary["embedding_outliers_available_in_broad"], 13)
        self.assertEqual(result.summary["embedding_outliers_selected"], 13)
        self.assertEqual(result.summary["embedding_outliers_unavailable_from_broad"], [])

    def test_outlier_absent_from_broad_is_reported_without_fabricating_a_review_item(self):
        included_path = "C:/dataset/2401100CSE0001/included.jpg"
        result = select_compact_enrollment_items(
            pd.DataFrame(
                [
                    enrollment_item(
                        "included",
                        "2401100CSE0001",
                        "own_class_outlier",
                        path=included_path,
                    )
                ]
            ),
            embedding_outliers=pd.DataFrame(
                [
                    {
                        "Canonical_Roll": "2401100CSE0001",
                        "Image_Path": included_path,
                        "Robust_Outlier_Score": "8.0",
                    },
                    {
                        "Canonical_Roll": "2401100CSE0002",
                        "Image_Path": "C:/dataset/2401100CSE0002/absent.jpg",
                        "Robust_Outlier_Score": "7.0",
                    },
                ]
            ),
            attractor_rows=pd.DataFrame(),
            miss_rows=pd.DataFrame(),
        )
        self.assertEqual(result.summary["embedding_outliers_considered"], 2)
        self.assertEqual(result.summary["embedding_outliers_available_in_broad"], 1)
        self.assertEqual(result.summary["embedding_outliers_selected"], 1)
        self.assertEqual(
            result.summary["embedding_outliers_unavailable_from_broad"],
            [
                {
                    "canonical_roll": "2401100CSE0002",
                    "image_path": "C:/dataset/2401100CSE0002/absent.jpg",
                    "robust_outlier_score": 7.0,
                    "reason": "not_present_in_validated_broad_enrollment_package",
                }
            ],
        )
        self.assertEqual(list(result.selected["Source_Broad_Item_ID"]), ["included"])

    def test_source_hashes_are_deduplicated_and_duplicate_groups_are_minimal(self):
        rows = [
            enrollment_item("dup-a", "2401100CSE0001", "multiple_faces; duplicated_file_hash", sha="same", duplicate_group="DUP-1"),
            enrollment_item("dup-b", "2401100CSE0001", "multiple_faces; duplicated_file_hash", sha="same", duplicate_group="DUP-1"),
            enrollment_item("unique", "2401100CSE0002", "multiple_faces", sha="unique"),
        ]
        result = select_compact_enrollment_items(
            pd.DataFrame(rows), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        )
        self.assertEqual(result.selected["Source_File_SHA256"].nunique(), len(result.selected))
        self.assertEqual(sum(result.selected["Duplicate_Group_ID"].eq("DUP-1")), 1)
        deferred_dup = result.deferred[result.deferred["Source_Broad_Item_ID"].eq("dup-b")].iloc[0]
        self.assertEqual(deferred_dup["Deferred_Reason"], "exact_source_hash_redundant")
        self.assertTrue(deferred_dup["Related_Selected_Item_ID"])

    def test_student_caps_enforce_three_ordinary_and_five_high_priority(self):
        ordinary = [
            enrollment_item(f"ordinary-{index}", "2401100CSE0001", "multiple_faces", sha=f"o-{index}")
            for index in range(8)
        ]
        high = [
            enrollment_item(
                f"high-{index}",
                "2401100CSE0002",
                "multiple_faces; closer_to_another_student_than_own_medoid",
                sha=f"h-{index}",
                own=0.1,
                other=0.8,
            )
            for index in range(8)
        ]
        result = select_compact_enrollment_items(
            pd.DataFrame(ordinary + high),
            pd.DataFrame(),
            attractors("2401100CSE0002"),
            pd.DataFrame(),
        )
        counts = result.selected.groupby("Expected_Roll").size().to_dict()
        self.assertEqual(counts["2401100CSE0001"], 3)
        self.assertEqual(counts["2401100CSE0002"], 5)

    def test_selection_is_deterministic_bounded_and_fully_accounted(self):
        rows = [
            enrollment_item(
                f"item-{index}",
                f"2401100CSE{index:04d}",
                "closer_to_another_student_than_own_medoid",
                own=0.2,
                other=0.7,
            )
            for index in range(80)
        ]
        first = select_compact_enrollment_items(
            pd.DataFrame(rows), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), maximum=60
        )
        second = select_compact_enrollment_items(
            pd.DataFrame(list(reversed(rows))), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), maximum=60
        )
        self.assertLessEqual(len(first.selected), 60)
        self.assertEqual(len(first.selected) + len(first.deferred), 80)
        self.assertEqual(
            list(first.selected["Source_Broad_Item_ID"]),
            list(second.selected["Source_Broad_Item_ID"]),
        )

    def test_model_similarity_is_diagnostic_only_and_never_ground_truth(self):
        row = enrollment_item("hint-only", "2401100CSE0001", "", own=0.1, other=0.99)
        result = select_compact_enrollment_items(
            pd.DataFrame([row]), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        )
        self.assertTrue(result.selected.empty)
        self.assertFalse(result.summary["model_similarity_used_as_ground_truth"])

    def test_invalid_limits_fail_closed(self):
        with self.assertRaises(CompactReviewError):
            select_compact_enrollment_items(
                pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), maximum=61
            )


class CompactCctvSelectionTests(unittest.TestCase):
    def test_only_human_identified_eligible_rows_are_selected(self):
        rows = [
            cctv_item("good", "2401100CSE0001"),
            cctv_item("outsider", "2401100CSE0002", actual_status="not_in_mapping"),
            cctv_item("mixed", "2401100CSE0003", actual_status="mixed_track"),
            cctv_item("uncertain", "2401100CSE0004", actual_status="uncertain"),
            cctv_item("unknown", "2401100CSE0005", actual_status="unidentifiable"),
            cctv_item("ineligible", "2401100CSE0006", eligible="No"),
            cctv_item("prediction", "2401100CSE0007", identity_source="model_prediction"),
        ]
        result = select_compact_cctv_items(
            pd.DataFrame(rows), attractors("2401100CSE0001"), pd.DataFrame(), pd.DataFrame()
        )
        self.assertEqual(list(result.selected["Source_Broad_Item_ID"]), ["good"])
        self.assertEqual(len(result.selected) + len(result.deferred), len(rows))

    def test_actual_roll_is_authoritative_even_when_prediction_differs(self):
        row = cctv_item(
            "identity",
            "2401100CSE0124",
            predicted_roll="2401100CSE0050",
        )
        result = select_compact_cctv_items(
            pd.DataFrame([row]), attractors("2401100CSE0124"), pd.DataFrame(), pd.DataFrame()
        )
        self.assertEqual(result.selected.iloc[0]["Actual_Roll"], "2401100CSE0124")
        self.assertFalse(result.summary["model_prediction_used_as_identity"])

    def test_student_cap_and_same_track_deduplication_are_enforced(self):
        rows = [
            cctv_item(
                f"crop-{index}",
                "2401100CSE0001",
                checkpoint=f"CP{(index % 5) + 1}",
                camera="cam10" if index % 2 else "cam5",
                track="shared" if index < 2 else f"track-{index}",
                nearest=str(index / 10),
            )
            for index in range(8)
        ]
        result = select_compact_cctv_items(
            pd.DataFrame(rows), attractors("2401100CSE0001"), pd.DataFrame(), pd.DataFrame()
        )
        self.assertLessEqual(len(result.selected), 4)
        self.assertEqual(result.selected["Track_ID"].nunique(), len(result.selected))

    def test_diverse_checkpoint_camera_crop_beats_near_duplicate(self):
        rows = [
            cctv_item("base", "2401100CSE0001", checkpoint="CP1", camera="cam5", nearest="", quality="1.2"),
            cctv_item("near", "2401100CSE0001", checkpoint="CP1", camera="cam5", nearest="0.95", quality="1.3"),
            cctv_item("diverse", "2401100CSE0001", checkpoint="CP4", camera="cam10", nearest="0.20", quality="1.0"),
        ]
        result = select_compact_cctv_items(
            pd.DataFrame(rows), attractors("2401100CSE0001"), pd.DataFrame(), pd.DataFrame(), maximum=2, preferred=2
        )
        self.assertIn("diverse", set(result.selected["Source_Broad_Item_ID"]))
        self.assertNotIn("near", set(result.selected["Source_Broad_Item_ID"]))

    def test_cctv_selection_is_deterministic_bounded_and_fully_accounted(self):
        rows = [
            cctv_item(
                f"crop-{index}",
                f"2401100CSE{index:04d}",
                checkpoint=f"CP{(index % 5) + 1}",
                camera="cam10" if index % 3 == 0 else "cam5",
                nearest=str((index % 7) / 10),
            )
            for index in range(70)
        ]
        priority_rolls = tuple(row["Actual_Roll"] for row in rows)
        first = select_compact_cctv_items(
            pd.DataFrame(rows), attractors(*priority_rolls), pd.DataFrame(), pd.DataFrame(), maximum=40
        )
        second = select_compact_cctv_items(
            pd.DataFrame(list(reversed(rows))), attractors(*priority_rolls), pd.DataFrame(), pd.DataFrame(), maximum=40
        )
        self.assertLessEqual(len(first.selected), 40)
        self.assertEqual(len(first.selected) + len(first.deferred), 70)
        self.assertEqual(
            list(first.selected["Source_Broad_Item_ID"]),
            list(second.selected["Source_Broad_Item_ID"]),
        )

    def test_invalid_limits_fail_closed(self):
        with self.assertRaises(CompactReviewError):
            select_compact_cctv_items(
                pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), maximum=41
            )


class CompactSourceSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_source_manifest_hash_mismatch_fails_closed(self):
        run = self.root / "forensic-run"
        run.mkdir()
        source = run / "ranking.csv"
        source.write_text("id,value\n1,clean\n", encoding="utf-8")
        write_output_manifest(run, {"completed": True})
        validated = validate_hashed_source_artifacts(run, ["ranking.csv"])
        self.assertEqual(validated["verified_files"], 1)
        source.write_text("id,value\n1,tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(CompactReviewError, "integrity check failed"):
            validate_hashed_source_artifacts(run, ["ranking.csv"])

    def test_missing_source_run_and_unlisted_required_artifact_fail(self):
        with self.assertRaisesRegex(CompactReviewError, "not found"):
            validate_hashed_source_artifacts(self.root / "missing")
        run = self.root / "run"
        run.mkdir()
        (run / "listed.csv").write_text("id\n1\n", encoding="utf-8")
        write_output_manifest(run)
        with self.assertRaisesRegex(CompactReviewError, "does not cover"):
            validate_hashed_source_artifacts(run, ["unlisted.csv"])

    def test_wrong_broad_package_id_and_missing_package_fail(self):
        image = np.full((80, 80, 3), 120, dtype=np.uint8)
        source = self.root / "source.jpg"
        self.assertTrue(cv2.imwrite(str(source), image))
        package = export_forensic_review_package(
            output_dir=self.root / "broad",
            package_id="actual-package",
            package_kind="enrollment_audit",
            title="Broad",
            instructions="Review",
            actions={"keep": "Keep"},
            items=[ForensicReviewItem("EA-1", source, {"Expected_Roll": "R1"}, {})],
        )
        with self.assertRaisesRegex(CompactReviewError, "Wrong broad-package ID"):
            validate_broad_review_package(
                package.root,
                expected_package_id="expected-package",
                expected_count=1,
                expected_kind="enrollment_audit",
            )
        with self.assertRaisesRegex(CompactReviewError, "not found"):
            validate_broad_review_package(
                self.root / "missing",
                expected_package_id="expected-package",
                expected_count=1,
                expected_kind="enrollment_audit",
            )

    def test_existing_compact_output_is_refused_before_source_work(self):
        existing = self.root / "existing"
        existing.mkdir()
        with self.assertRaisesRegex(CompactReviewError, "Refusing to overwrite"):
            generate_compact_forensic_reviews(
                repo_root=self.root,
                source_forensic_run=self.root / "missing-source",
                source_reviewfix_run=self.root / "missing-reviewfix",
                candidate_config_path=self.root / "candidate.json",
                embeddings_path=self.root / "embeddings.pkl",
                embedding_summary_path=self.root / "summary.csv",
                student_map_path=self.root / "students.json",
                candidate_registry_path=self.root / "rejected.json",
                output_dir=existing,
                progress=None,
            )


class CompactReviewerPackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.images = []
        for index in range(3):
            path = self.root / f"source-{index}.png"
            image = np.full((90 + index * 5, 100, 3), 100 + index * 20, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(path), image))
            self.images.append(path)

    def tearDown(self):
        self.temp.cleanup()

    def enrollment_selection(self):
        rows = [
            enrollment_item(
                f"EA-{index}",
                f"2401100CSE{index + 1:04d}",
                "multiple_faces",
                path=str(self.images[index]),
                sha=_sha256_file(self.images[index]),
            )
            for index in range(2)
        ]
        result = select_compact_enrollment_items(
            pd.DataFrame(rows), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        )
        return result.selected

    def cctv_selection(self):
        rows = []
        for index in range(2):
            row = cctv_item(
                f"CC-{index}",
                f"2401100CSE{index + 1:04d}",
                checkpoint=f"CP{index + 1}",
                camera="cam10" if index else "cam5",
            )
            row["Candidate_Crop_Path"] = str(self.images[index])
            row["Candidate_Crop_SHA256"] = _sha256_file(self.images[index])
            rows.append(row)
        return select_compact_cctv_items(
            pd.DataFrame(rows),
            attractors(*(row["Actual_Roll"] for row in rows)),
            pd.DataFrame(),
            pd.DataFrame(),
        ).selected

    def test_reviewers_contain_exact_compact_selection_and_validate_hashes(self):
        enrollment_selected = self.enrollment_selection()
        cctv_selected = self.cctv_selection()
        enrollment = export_compact_enrollment_review(
            selected=enrollment_selected,
            output_dir=self.root / "compact-enrollment",
            source_package_id="broad-enrollment",
            source_reviewfix_run=self.root,
            source_reviewfix_manifest_sha256="a" * 64,
            selection_csv_sha256="b" * 64,
            package_revision="test-v1",
        )
        cctv = export_compact_cctv_review(
            selected=cctv_selected,
            output_dir=self.root / "compact-cctv",
            source_package_id="broad-cctv",
            source_reviewfix_run=self.root,
            source_reviewfix_manifest_sha256="a" * 64,
            selection_csv_sha256="c" * 64,
            package_revision="test-v1",
        )
        enrollment_public, _, enrollment_mapping = load_forensic_review_package(enrollment.root)
        cctv_public, _, cctv_mapping = load_forensic_review_package(cctv.root)
        self.assertEqual(enrollment_public["item_count"], len(enrollment_selected))
        self.assertEqual(cctv_public["item_count"], len(cctv_selected))
        enrollment_sources = {
            __import__("json").loads(value)["source_broad_item_id"]
            for value in enrollment_mapping["Private_Provenance_JSON"]
        }
        cctv_sources = {
            __import__("json").loads(value)["source_broad_item_id"]
            for value in cctv_mapping["Private_Provenance_JSON"]
        }
        self.assertEqual(enrollment_sources, set(enrollment_selected["Source_Broad_Item_ID"]))
        self.assertEqual(cctv_sources, set(cctv_selected["Source_Broad_Item_ID"]))
        cctv_reviewer = (cctv.reviewer_dir / "review_manifest.json").read_text(encoding="utf-8")
        self.assertNotIn("Predicted_Roll", cctv_reviewer)
        self.assertIn("Human_Confirmed_Actual_Roll", cctv_reviewer)

    def test_compact_export_schema_rejects_incomplete_approval(self):
        package = export_compact_enrollment_review(
            selected=self.enrollment_selection(),
            output_dir=self.root / "compact",
            source_package_id="broad",
            source_reviewfix_run=self.root,
            source_reviewfix_manifest_sha256="a" * 64,
            selection_csv_sha256="b" * 64,
            package_revision="test-v1",
        )
        approvals = self.root / "approvals.csv"
        frame = pd.read_csv(package.reviewer_dir / "approvals_template.csv", dtype=str, keep_default_na=False)
        frame.to_csv(approvals, index=False)
        with self.assertRaisesRegex(ForensicReviewError, "missing or invalid"):
            validate_forensic_approvals(
                package.root, approvals, expected_kind="compact_enrollment_audit"
            )
        frame["Review_Action"] = "keep"
        frame.to_csv(approvals, index=False)
        result = validate_forensic_approvals(
            package.root, approvals, expected_kind="compact_enrollment_audit"
        )
        self.assertTrue(result.summary["complete"])

    def test_long_path_launcher_and_loopback_server_need_no_browser(self):
        long_root = self.root / ("long-review-path-" + "x" * 110)
        package = export_compact_cctv_review(
            selected=self.cctv_selection(),
            output_dir=long_root / "compact-cctv",
            source_package_id="broad",
            source_reviewfix_run=self.root,
            source_reviewfix_manifest_sha256="a" * 64,
            selection_csv_sha256="b" * 64,
            package_revision="test-v1",
        )
        launcher = write_reviewer_launcher(
            launcher_path=long_root / "open.ps1",
            python_executable=Path("C:/venv/python.exe"),
            cli_script=Path("C:/repo/validate.py"),
            command="open-compact-cctv-review",
            review_package=package.root,
        )
        launcher_text = launcher.read_text(encoding="utf-8")
        self.assertIn("open-compact-cctv-review", launcher_text)
        self.assertNotIn("Start-Process", launcher_text)
        server, url = create_reviewer_server(package.root, port=0)
        try:
            self.assertEqual(server.server_address[0], "127.0.0.1")
            self.assertTrue(url.startswith("http://127.0.0.1:"))
        finally:
            server.server_close()

    def test_compact_export_does_not_modify_existing_broad_package(self):
        broad = export_forensic_review_package(
            output_dir=self.root / "broad",
            package_id="broad-evidence",
            package_kind="enrollment_audit",
            title="Broad",
            instructions="Review",
            actions={"keep": "Keep"},
            items=[ForensicReviewItem("EA-1", self.images[0], {}, {})],
        )
        before = _sha256_file(broad.manifest_path)
        export_compact_enrollment_review(
            selected=self.enrollment_selection(),
            output_dir=self.root / "compact",
            source_package_id="broad-evidence",
            source_reviewfix_run=self.root,
            source_reviewfix_manifest_sha256="a" * 64,
            selection_csv_sha256="b" * 64,
            package_revision="test-v1",
        )
        self.assertEqual(_sha256_file(broad.manifest_path), before)


if __name__ == "__main__":
    unittest.main()
