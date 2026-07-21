import hashlib
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

from src.face_attendance import embedding_lifecycle
from src.face_attendance.embedding_lifecycle import (
    EmbeddingLifecycleError,
    build_versioned_embeddings,
    evaluate_embedding_version,
    promote_embedding_version,
    rollback_embedding_version,
    validate_embedding_approval_bundle,
    verify_embedding_version,
)
from src.face_attendance.forensic_review import (
    ForensicReviewItem,
    export_forensic_review_package,
)
from src.face_attendance.shadow_validation import write_output_manifest
from src.face_attendance.tracklet_review import _sha256_file


ENROLLMENT_ACTIONS = {
    "keep": "Keep",
    "exclude_from_next_embedding_version": "Exclude",
}
CCTV_ACTIONS = {
    "approve_for_candidate_embedding_version": "Approve",
    "reject_quality": "Reject quality",
}


class EmbeddingLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.models = self.repo / "models"
        self.models.mkdir()
        self.source_image = self.repo / "dataset" / "2401100CSE0016" / "source.jpg"
        self.source_image.parent.mkdir(parents=True)
        image = np.full((120, 120, 3), 130, dtype=np.uint8)
        cv2.circle(image, (60, 55), 34, (210, 210, 210), -1)
        self.assertTrue(cv2.imwrite(str(self.source_image), image))
        self.cctv_crop = self.repo / "forensic_source" / "crop.png"
        self.cctv_crop.parent.mkdir()
        self.assertTrue(cv2.imwrite(str(self.cctv_crop), image))
        self.production_embeddings = self.models / "student_embeddings.pkl"
        self.production_summary = self.models / "embedding_summary.csv"
        with self.production_embeddings.open("wb") as file_obj:
            pickle.dump(
                {
                    "version": 1,
                    "created_at": "legacy",
                    "metadata": {"dataset": str(self.repo / "dataset")},
                    "records": [
                        {
                            "roll_no": "2401100CSE0016",
                            "image_path": str(self.source_image),
                            "detection_score": 0.95,
                            "embedding": np.array([1.0, 0.0, 0.0], dtype=np.float32),
                        }
                    ],
                },
                file_obj,
            )
        self.production_summary.write_text(
            "Roll_Number,Faces_Used\n2401100CSE0016,1\n", encoding="utf-8"
        )
        self.yunet = self.models / "yunet.onnx"
        self.sface = self.models / "sface.onnx"
        self.yunet.write_bytes(b"fake-yunet-model")
        self.sface.write_bytes(b"fake-sface-model")
        self.forensic_run = self.repo / "attendance_output" / "embedding_forensics" / "run"
        self.forensic_run.mkdir(parents=True)
        (self.forensic_run / "forensic_summary.json").write_text(
            json.dumps(
                {
                    "benchmark_reviewed_tracks": 120,
                    "protected_state_unchanged": True,
                    "candidate_embedding_version_built": False,
                }
            ),
            encoding="utf-8",
        )
        (self.forensic_run / "multisession_ground_truth_manifest.json").write_text(
            json.dumps({"benchmark_id": "benchmark-120", "row_counts": {"human_reviewed": 120}}),
            encoding="utf-8",
        )
        self.enrollment_package = export_forensic_review_package(
            output_dir=self.forensic_run / "enrollment-review",
            package_id="enrollment-pkg",
            package_kind="enrollment_audit",
            title="Enrollment",
            instructions="Review",
            actions=ENROLLMENT_ACTIONS,
            items=[
                ForensicReviewItem(
                    item_id="EA-1",
                    evidence_source=self.source_image,
                    public={"Expected_Roll": "2401100CSE0016"},
                    private={"expected_roll": "2401100CSE0016"},
                )
            ],
        )
        self.cctv_package = export_forensic_review_package(
            output_dir=self.forensic_run / "cctv-review",
            package_id="cctv-pkg",
            package_kind="verified_cctv_enrollment",
            title="CCTV",
            instructions="Review",
            actions=CCTV_ACTIONS,
            items=[
                ForensicReviewItem(
                    item_id="CC-1",
                    evidence_source=self.cctv_crop,
                    public={"Actual_Student_Roll": "2401100CSE0016"},
                    private={
                        "actual_roll": "2401100CSE0016",
                        "identity_source": "human_review_actual_roll",
                        "source_session": "2026-06-30__B51__P1__CVO",
                    },
                    preserve_resolution=True,
                )
            ],
        )
        write_output_manifest(self.forensic_run, {"completed": True})
        self.enrollment_approvals = self.root / "enrollment.csv"
        self.cctv_approvals = self.root / "cctv.csv"
        self.write_approvals("keep", "reject_quality")
        self.versions_root = self.models / "versions"
        self.registry = self.models / "candidate_registry" / "rejected"

    def tearDown(self):
        self.temp.cleanup()

    def write_approvals(self, enrollment_action, cctv_action):
        enrollment = pd.read_csv(
            self.enrollment_package.reviewer_dir / "approvals_template.csv",
            dtype=str,
            keep_default_na=False,
        )
        enrollment["Review_Action"] = enrollment_action
        enrollment.to_csv(self.enrollment_approvals, index=False, encoding="utf-8-sig")
        cctv = pd.read_csv(
            self.cctv_package.reviewer_dir / "approvals_template.csv",
            dtype=str,
            keep_default_na=False,
        )
        cctv["Review_Action"] = cctv_action
        cctv.to_csv(self.cctv_approvals, index=False, encoding="utf-8-sig")

    def build(self, **overrides):
        kwargs = {
            "repo_root": self.repo,
            "forensic_run_dir": self.forensic_run,
            "enrollment_review_package": self.enrollment_package.root,
            "enrollment_approvals_path": self.enrollment_approvals,
            "cctv_review_package": self.cctv_package.root,
            "cctv_approvals_path": self.cctv_approvals,
            "production_embeddings_path": self.production_embeddings,
            "production_summary_path": self.production_summary,
            "versions_root": self.versions_root,
            "yunet_model": self.yunet,
            "sface_model": self.sface,
            "candidate_registry_dir": self.registry,
        }
        kwargs.update(overrides)
        return build_versioned_embeddings(**kwargs)

    def test_approval_bundle_requires_both_complete_files(self):
        bundle = validate_embedding_approval_bundle(
            enrollment_review_package=self.enrollment_package.root,
            enrollment_approvals_path=self.enrollment_approvals,
            cctv_review_package=self.cctv_package.root,
            cctv_approvals_path=self.cctv_approvals,
        )
        self.assertTrue(bundle.summary["complete"])
        frame = pd.read_csv(self.cctv_approvals, dtype=str, keep_default_na=False)
        frame["Review_Action"] = ""
        frame.to_csv(self.cctv_approvals, index=False)
        with self.assertRaisesRegex(EmbeddingLifecycleError, "missing or invalid"):
            validate_embedding_approval_bundle(
                enrollment_review_package=self.enrollment_package.root,
                enrollment_approvals_path=self.enrollment_approvals,
                cctv_review_package=self.cctv_package.root,
                cctv_approvals_path=self.cctv_approvals,
            )

    def test_build_refuses_missing_approval_file(self):
        self.cctv_approvals.unlink()
        with self.assertRaisesRegex(EmbeddingLifecycleError, "Approval CSV not found"):
            self.build()
        self.assertFalse(self.versions_root.exists())

    def test_build_is_built_unapproved_idempotent_and_never_overwrites_production(self):
        before_embeddings = self.production_embeddings.read_bytes()
        before_summary = self.production_summary.read_bytes()
        result = self.build()
        self.assertEqual(result.manifest["status"], "built_unapproved")
        self.assertFalse(result.manifest["production_embeddings_overwritten_by_build"])
        self.assertEqual(self.production_embeddings.read_bytes(), before_embeddings)
        self.assertEqual(self.production_summary.read_bytes(), before_summary)
        self.assertTrue((result.version_dir / "rollback_metadata.json").is_file())
        self.assertTrue((result.version_dir / "source_manifest.json").is_file())
        self.assertTrue(result.manifest["approval_csv_hashes"]["enrollment"])
        repeated = self.build()
        self.assertTrue(repeated.idempotent_reuse)
        self.assertEqual(repeated.version_dir, result.version_dir)

    def test_enrollment_exclusion_changes_only_candidate_version(self):
        self.write_approvals("exclude_from_next_embedding_version", "reject_quality")
        production_hash = _sha256_file(self.production_embeddings)
        result = self.build(version_id="embv-exclusion-test")
        with (result.version_dir / "student_embeddings.pkl").open("rb") as file_obj:
            candidate = pickle.load(file_obj)
        self.assertEqual(candidate["records"], [])
        self.assertEqual(_sha256_file(self.production_embeddings), production_hash)
        diff = json.loads(
            (result.version_dir / "current_vs_candidate_diff.json").read_text(encoding="utf-8")
        )
        self.assertEqual(diff["excluded_parent_records"], 1)

    def test_rejected_source_candidate_is_blocked_before_build(self):
        self.registry.mkdir(parents=True)
        (self.registry / "cal-rejected.json").write_text(
            json.dumps(
                {
                    "candidate_id": "cal-rejected",
                    "status": "rejected_multisession_false_accepts",
                    "production_approved": False,
                    "enabled": False,
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(EmbeddingLifecycleError, "permanently rejected"):
            self.build(source_candidate_ids=["cal-rejected"])

    def test_version_integrity_detects_tampering(self):
        result = self.build()
        (result.version_dir / "student_embeddings.pkl").write_bytes(b"tampered")
        with self.assertRaisesRegex(EmbeddingLifecycleError, "integrity check failed"):
            verify_embedding_version(result.version_dir)

    def test_promotion_refuses_without_regression_and_without_untouched_validation(self):
        result = self.build()
        production_hash = _sha256_file(self.production_embeddings)
        with self.assertRaisesRegex(EmbeddingLifecycleError, "regression has not passed"):
            promote_embedding_version(
                version_dir=result.version_dir,
                production_embeddings_path=self.production_embeddings,
                production_summary_path=self.production_summary,
                candidate_registry_dir=self.registry,
            )
        status = json.loads(
            (result.version_dir / "regression_status.json").read_text(encoding="utf-8")
        )
        status["regression_passed"] = True
        (result.version_dir / "regression_status.json").write_text(json.dumps(status), encoding="utf-8")
        with self.assertRaisesRegex(EmbeddingLifecycleError, "MON_P3"):
            promote_embedding_version(
                version_dir=result.version_dir,
                production_embeddings_path=self.production_embeddings,
                production_summary_path=self.production_summary,
                candidate_registry_dir=self.registry,
            )
        self.assertEqual(_sha256_file(self.production_embeddings), production_hash)

    def test_evaluation_rejects_wrong_thresholds_and_wrong_benchmark(self):
        result = self.build()
        benchmark_hash = _sha256_file(
            self.forensic_run / "multisession_ground_truth_manifest.json"
        )
        evidence = self.root / "regression.json"
        payload = {
            "version_id": result.version_id,
            "benchmark_id": "wrong",
            "benchmark_manifest_sha256": benchmark_hash,
            "reviewed_tracks": 120,
            "regression_passed": True,
            "match_threshold": 0.48,
            "margin_threshold": 0.08,
            "attendance_checkpoints": 3,
        }
        evidence.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(EmbeddingLifecycleError, "does not match"):
            evaluate_embedding_version(
                version_dir=result.version_dir,
                forensic_run_dir=self.forensic_run,
                regression_results_path=evidence,
            )
        payload["benchmark_id"] = "benchmark-120"
        payload["match_threshold"] = 0.47
        evidence.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(EmbeddingLifecycleError, "changed official"):
            evaluate_embedding_version(
                version_dir=result.version_dir,
                forensic_run_dir=self.forensic_run,
                regression_results_path=evidence,
            )

    def test_passed_evaluation_can_promote_and_rollback_in_temp_repo(self):
        result = self.build()
        original_embeddings = self.production_embeddings.read_bytes()
        original_summary = self.production_summary.read_bytes()
        regression = self.root / "regression.json"
        regression.write_text(
            json.dumps(
                {
                    "version_id": result.version_id,
                    "benchmark_id": "benchmark-120",
                    "benchmark_manifest_sha256": _sha256_file(
                        self.forensic_run / "multisession_ground_truth_manifest.json"
                    ),
                    "reviewed_tracks": 120,
                    "regression_passed": True,
                    "match_threshold": 0.48,
                    "margin_threshold": 0.08,
                    "attendance_checkpoints": 3,
                }
            ),
            encoding="utf-8",
        )
        untouched = self.root / "mon_p3.json"
        untouched.write_text(
            json.dumps(
                {
                    "version_id": result.version_id,
                    "session_id": "MON_P3",
                    "untouched_before_validation": True,
                    "validation_passed": True,
                }
            ),
            encoding="utf-8",
        )
        status = evaluate_embedding_version(
            version_dir=result.version_dir,
            forensic_run_dir=self.forensic_run,
            regression_results_path=regression,
            untouched_validation_path=untouched,
        )
        self.assertTrue(status["promotion_allowed"])
        promotion = promote_embedding_version(
            version_dir=result.version_dir,
            production_embeddings_path=self.production_embeddings,
            production_summary_path=self.production_summary,
            candidate_registry_dir=self.registry,
        )
        self.assertEqual(promotion["status"], "promoted")
        self.assertNotEqual(self.production_embeddings.read_bytes(), original_embeddings)
        rollback = rollback_embedding_version(version_dir=result.version_dir)
        self.assertEqual(rollback["status"], "rolled_back")
        self.assertEqual(self.production_embeddings.read_bytes(), original_embeddings)
        self.assertEqual(self.production_summary.read_bytes(), original_summary)

    def test_partial_promotion_failure_restores_the_production_pair(self):
        result = self.build()
        original_embeddings = self.production_embeddings.read_bytes()
        original_summary = self.production_summary.read_bytes()
        status_path = result.version_dir / "regression_status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status.update(
            {
                "regression_passed": True,
                "untouched_session_validation_passed": True,
                "untouched_session_validation": {
                    "session_id": "MON_P3",
                    "untouched_before_validation": True,
                    "validation_passed": True,
                },
            }
        )
        status_path.write_text(json.dumps(status), encoding="utf-8")
        real_atomic_replace = embedding_lifecycle._atomic_replace_from
        calls = 0

        def fail_second_replace(source, target):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated summary replacement failure")
            real_atomic_replace(source, target)

        with patch.object(
            embedding_lifecycle, "_atomic_replace_from", side_effect=fail_second_replace
        ):
            with self.assertRaisesRegex(EmbeddingLifecycleError, "were restored"):
                promote_embedding_version(
                    version_dir=result.version_dir,
                    production_embeddings_path=self.production_embeddings,
                    production_summary_path=self.production_summary,
                    candidate_registry_dir=self.registry,
                )
        self.assertEqual(self.production_embeddings.read_bytes(), original_embeddings)
        self.assertEqual(self.production_summary.read_bytes(), original_summary)

    def test_approved_source_change_after_review_fails_closed(self):
        self.source_image.write_bytes(self.source_image.read_bytes() + b"changed")
        with self.assertRaisesRegex(EmbeddingLifecycleError, "source integrity"):
            self.build()


if __name__ == "__main__":
    unittest.main()
