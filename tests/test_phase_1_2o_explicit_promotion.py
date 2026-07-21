from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.face_attendance import phase_1_2o_explicit_promotion as mod


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(directory: Path, files: list[Path], **metadata: object) -> Path:
    mapping = {
        path.relative_to(directory).as_posix(): sha(path)
        for path in files
    }
    manifest = directory / "output_manifest.json"
    write_json(
        manifest,
        {
            "schema_version": 1,
            "files_sha256": mapping,
            **metadata,
        },
    )
    return manifest


class Fixture:
    def __init__(self, root: Path, *, decision: str = "pass") -> None:
        self.root = root
        self.repo = root / "repo"
        self.family_id = "embfam-test"
        self.variant_id = "embfam-test-d"
        self.run_id = "mon-p4-shadow-test"
        self.evaluation_id = "mon-p4-evaluation-test"
        self.session_id = "2026-06-22__B51__P4__CVO"
        self.family = self.repo / "models" / "versions" / self.family_id
        self.variant = self.family / "variants" / "full_candidate"
        self.shadow = (
            self.repo
            / "attendance_output"
            / "shadow_validation"
            / "phase_1_2n"
            / "shadow_runs"
            / self.run_id
        )
        self.evaluation = self.shadow / "evaluation" / self.evaluation_id
        self.labels = root / "labels.csv"
        self.production_embeddings = self.repo / "models" / "student_embeddings.pkl"
        self.production_summary = self.repo / "models" / "embedding_summary.csv"
        self.production_embeddings.parent.mkdir(parents=True, exist_ok=True)
        self.production_embeddings.write_bytes(b"production-embeddings")
        self.production_summary.write_text("production-summary\n", encoding="utf-8")
        self.variant.mkdir(parents=True, exist_ok=True)
        self.candidate_embeddings = self.variant / "student_embeddings.pkl"
        self.candidate_summary = self.variant / "embedding_summary.csv"
        self.candidate_embeddings.write_bytes(b"candidate-embeddings")
        self.candidate_summary.write_text("candidate-summary\n", encoding="utf-8")

        self.family_manifest = self.family / "family_manifest.json"
        write_json(
            self.family_manifest,
            {
                "schema_version": 1,
                "family_id": self.family_id,
                "final_decision": "built_unapproved_pending_new_untouched_session",
                "candidate_promoted": False,
                "production_approved": False,
            },
        )
        self.version_manifest = self.variant / "version_manifest.json"
        write_json(
            self.version_manifest,
            {
                "schema_version": 1,
                "version_id": self.variant_id,
                "status": "built_unapproved",
                "availability": True,
                "production_promoted": False,
                "artifact_sha256": {
                    "student_embeddings.pkl": sha(self.candidate_embeddings),
                    "embedding_summary.csv": sha(self.candidate_summary),
                },
            },
        )
        self.family_output_manifest = write_manifest(
            self.family,
            [
                self.family_manifest,
                self.version_manifest,
                self.candidate_embeddings,
                self.candidate_summary,
            ],
            family_id=self.family_id,
        )

        frozen = self.shadow / "frozen_inputs"
        frozen.mkdir(parents=True, exist_ok=True)
        self.frozen_family = frozen / "family_manifest.json"
        self.frozen_family.write_bytes(self.family_manifest.read_bytes())
        self.frozen_version = frozen / "version_manifest.json"
        self.frozen_version.write_bytes(self.version_manifest.read_bytes())
        self.source_freeze = frozen / "phase_1_2n_mon_p4_source_freeze.json"
        write_json(
            self.source_freeze,
            {
                "SchemaVersion": 1,
                "SessionID": self.session_id,
                "SourceFreezePassed": True,
                "UntouchedContractPassed": True,
            },
        )
        self.protected = self.shadow / "protected_state_comparison.json"
        write_json(self.protected, {"schema_version": 1, "unchanged": True})
        self.shadow_summary = self.shadow / "shadow_run_summary.json"
        write_json(
            self.shadow_summary,
            {
                "schema_version": 1,
                "run_id": self.run_id,
                "session_id": self.session_id,
                "family_id": self.family_id,
                "candidate_variant_id": self.variant_id,
                "production_embeddings_changed": False,
                "official_attendance_changed": False,
                "candidate_promoted": False,
            },
        )
        self.shadow_manifest = write_manifest(
            self.shadow,
            [
                self.frozen_family,
                self.frozen_version,
                self.source_freeze,
                self.protected,
                self.shadow_summary,
            ],
            run_id=self.run_id,
        )

        result_codes = [
            "both_correct_same_identity",
            "candidate_correct_recovery",
            "candidate_correct_recovery",
            "candidate_correct_recovery",
            "reviewed_other",
        ]
        labels_rows: list[dict[str, str]] = []
        reviewed_rows: list[dict[str, str]] = []
        for index, result_code in enumerate(result_codes, start=1):
            rid = f"R{index:04d}"
            roll = "2401100CSE0067" if index == 1 else "2401100CSE0140"
            package = "package-test"
            labels_rows.append(
                {
                    "Package_ID": package,
                    "Review_ID": rid,
                    "Review_Status": "identified",
                    "Actual_Roll": roll,
                    "Reviewer_Notes": "",
                }
            )
            reviewed_rows.append(
                {
                    "Package_ID": package,
                    "Review_ID": rid,
                    "Review_Status": "identified",
                    "Actual_Roll": roll,
                    "Phase_1_2N_Result": result_code,
                }
            )
        write_csv(self.labels, labels_rows)
        self.reviewed = self.evaluation / "reviewed_exception_tracks.csv"
        write_csv(self.reviewed, reviewed_rows)
        self.report = self.evaluation / "mon_p4_evaluation_report.md"
        self.report.write_text("test report\n", encoding="utf-8")
        self.evaluation_summary = self.evaluation / "mon_p4_evaluation_summary.json"
        evaluation_decision = (
            "mon_p4_shadow_passed_pending_explicit_promotion"
            if decision == "pass"
            else "mon_p4_shadow_failed_retention"
        )
        write_json(
            self.evaluation_summary,
            {
                "schema_version": 1,
                "evaluation_id": self.evaluation_id,
                "run_id": self.run_id,
                "family_id": self.family_id,
                "candidate_variant_id": self.variant_id,
                "reviewed_exception_tracks": 5,
                "candidate_correct_recoveries": 3,
                "candidate_false_identities_or_unsafe_accepts": 0,
                "candidate_lost_correct_production_accepts": 0,
                "unverifiable_exceptions": 0,
                "decision": evaluation_decision,
                "zero_false_identity_gate_passed": True,
                "retention_gate_passed": True,
                "evidence_complete_gate_passed": True,
                "candidate_promoted": False,
                "production_approved": False,
            },
        )
        self.evaluation_manifest = write_manifest(
            self.evaluation,
            [self.reviewed, self.report, self.evaluation_summary],
            evaluation_id=self.evaluation_id,
        )
        self.contract = mod.PromotionContract(
            family_id=self.family_id,
            candidate_variant_id=self.variant_id,
            expected_family_decision="built_unapproved_pending_new_untouched_session",
            session_id=self.session_id,
            run_id=self.run_id,
            evaluation_id=self.evaluation_id,
            evaluation_decision="mon_p4_shadow_passed_pending_explicit_promotion",
            reviewed_exception_tracks=5,
            candidate_correct_recoveries=3,
            candidate_false_identities_or_unsafe_accepts=0,
            candidate_lost_correct_production_accepts=0,
            unverifiable_exceptions=0,
            production_embeddings_sha256=sha(self.production_embeddings),
            production_summary_sha256=sha(self.production_summary),
            candidate_embeddings_sha256=sha(self.candidate_embeddings),
            candidate_summary_sha256=sha(self.candidate_summary),
            frozen_family_manifest_sha256=sha(self.frozen_family),
            frozen_version_manifest_sha256=sha(self.frozen_version),
            source_freeze_json_sha256=sha(self.source_freeze),
            shadow_output_manifest_sha256=sha(self.shadow_manifest),
            shadow_summary_sha256=sha(self.shadow_summary),
            protected_state_comparison_sha256=sha(self.protected),
            evaluation_output_manifest_sha256=sha(self.evaluation_manifest),
            evaluation_summary_sha256=sha(self.evaluation_summary),
            reviewed_exception_tracks_sha256=sha(self.reviewed),
            submitted_labels_sha256=sha(self.labels),
        )
        self.inputs = mod.PromotionInputs(
            repo_root=self.repo,
            family_dir=self.family,
            shadow_dir=self.shadow,
            evaluation_dir=self.evaluation,
            labels_path=self.labels,
            production_embeddings=self.production_embeddings,
            production_summary=self.production_summary,
        )


class EvidenceVerificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.fx = Fixture(Path(self.temp.name))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_valid_evidence_passes(self) -> None:
        result = mod.verify_promotion_evidence(self.fx.inputs, self.fx.contract)
        self.assertEqual(result.candidate_correct_recoveries, 3)
        self.assertEqual(result.candidate_false_identities_or_unsafe_accepts, 0)
        self.assertEqual(result.candidate_lost_correct_production_accepts, 0)
        self.assertEqual(result.unverifiable_exceptions, 0)
        self.assertTrue(result.promotion_id.startswith("promotion-"))

    def test_zero_metrics_are_not_treated_as_missing(self) -> None:
        result = mod.verify_promotion_evidence(self.fx.inputs, self.fx.contract)
        self.assertEqual(result.candidate_false_identities_or_unsafe_accepts, 0)

    def test_tampered_label_file_fails(self) -> None:
        self.fx.labels.write_text("tampered\n", encoding="utf-8")
        with self.assertRaises(mod.Phase12OPromotionError):
            mod.verify_promotion_evidence(self.fx.inputs, self.fx.contract)

    def test_non_passing_evaluation_fails(self) -> None:
        data = json.loads(self.fx.evaluation_summary.read_text(encoding="utf-8"))
        data["decision"] = "mon_p4_shadow_failed_retention"
        write_json(self.fx.evaluation_summary, data)
        contract = self.fx.contract.__class__(
            **{
                **self.fx.contract.__dict__,
                "evaluation_summary_sha256": sha(self.fx.evaluation_summary),
            }
        )
        write_manifest(
            self.fx.evaluation,
            [self.fx.reviewed, self.fx.report, self.fx.evaluation_summary],
            evaluation_id=self.fx.evaluation_id,
        )
        contract = contract.__class__(
            **{
                **contract.__dict__,
                "evaluation_output_manifest_sha256": sha(self.fx.evaluation_manifest),
            }
        )
        with self.assertRaises(mod.Phase12OPromotionError):
            mod.verify_promotion_evidence(self.fx.inputs, contract)

    def test_source_freeze_must_be_untouched(self) -> None:
        data = json.loads(self.fx.source_freeze.read_text(encoding="utf-8"))
        data["UntouchedContractPassed"] = False
        write_json(self.fx.source_freeze, data)
        write_manifest(
            self.fx.shadow,
            [
                self.fx.frozen_family,
                self.fx.frozen_version,
                self.fx.source_freeze,
                self.fx.protected,
                self.fx.shadow_summary,
            ],
            run_id=self.fx.run_id,
        )
        contract = self.fx.contract.__class__(
            **{
                **self.fx.contract.__dict__,
                "source_freeze_json_sha256": sha(self.fx.source_freeze),
                "shadow_output_manifest_sha256": sha(self.fx.shadow_manifest),
            }
        )
        with self.assertRaises(mod.Phase12OPromotionError):
            mod.verify_promotion_evidence(self.fx.inputs, contract)

    def test_family_payload_tamper_fails(self) -> None:
        self.fx.candidate_embeddings.write_bytes(b"tampered candidate")
        with self.assertRaises(mod.Phase12OPromotionError):
            mod.verify_promotion_evidence(self.fx.inputs, self.fx.contract)

    def test_reviewed_result_change_fails_even_when_safety_counts_are_zero(self) -> None:
        with self.fx.reviewed.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[-1]["Phase_1_2N_Result"] = "both_correct_same_identity"
        write_csv(self.fx.reviewed, rows)
        write_manifest(
            self.fx.evaluation,
            [self.fx.reviewed, self.fx.report, self.fx.evaluation_summary],
            evaluation_id=self.fx.evaluation_id,
        )
        contract = self.fx.contract.__class__(
            **{
                **self.fx.contract.__dict__,
                "reviewed_exception_tracks_sha256": sha(self.fx.reviewed),
                "evaluation_output_manifest_sha256": sha(self.fx.evaluation_manifest),
            }
        )
        with self.assertRaises(mod.Phase12OPromotionError):
            mod.verify_promotion_evidence(self.fx.inputs, contract)

    def test_manifest_path_traversal_is_rejected(self) -> None:
        bad = self.fx.root / "bad_manifest.json"
        write_json(bad, {"files_sha256": {"../outside": "0" * 64}})
        with self.assertRaises(mod.Phase12OPromotionError):
            mod.verify_output_manifest(self.fx.root, bad)

    def test_boolean_required_integer_is_rejected(self) -> None:
        data = json.loads(self.fx.evaluation_summary.read_text(encoding="utf-8"))
        data["candidate_false_identities_or_unsafe_accepts"] = False
        write_json(self.fx.evaluation_summary, data)
        write_manifest(
            self.fx.evaluation,
            [self.fx.reviewed, self.fx.report, self.fx.evaluation_summary],
            evaluation_id=self.fx.evaluation_id,
        )
        contract = self.fx.contract.__class__(
            **{
                **self.fx.contract.__dict__,
                "evaluation_summary_sha256": sha(self.fx.evaluation_summary),
                "evaluation_output_manifest_sha256": sha(self.fx.evaluation_manifest),
            }
        )
        with self.assertRaises(mod.Phase12OPromotionError):
            mod.verify_promotion_evidence(self.fx.inputs, contract)


class PromotionAndRollbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.fx = Fixture(Path(self.temp.name))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def promote(self) -> mod.PromotionResult:
        return mod.promote_candidate_family(
            self.fx.inputs,
            confirm_family_id=self.fx.family_id,
            contract=self.fx.contract,
        )

    def test_exact_confirmation_is_required(self) -> None:
        with self.assertRaises(mod.Phase12OPromotionError):
            mod.promote_candidate_family(
                self.fx.inputs,
                confirm_family_id="wrong-family",
                contract=self.fx.contract,
            )

    def test_promotion_replaces_both_files_and_writes_pointer(self) -> None:
        result = self.promote()
        self.assertEqual(sha(self.fx.production_embeddings), sha(self.fx.candidate_embeddings))
        self.assertEqual(sha(self.fx.production_summary), sha(self.fx.candidate_summary))
        pointer = self.fx.repo / "models" / "current_embedding_version.json"
        self.assertTrue(pointer.is_file())
        self.assertTrue((result.promotion_dir / "promotion_record.json").is_file())
        self.assertTrue((result.promotion_dir / "promotion_output_manifest.json").is_file())
        verified = mod.verify_promoted_state(
            repo_root=self.fx.repo,
            promotion_dir=result.promotion_dir,
            contract=self.fx.contract,
        )
        self.assertEqual(verified.promotion_id, result.promotion_id)

    def test_promotion_preserves_parent_backup(self) -> None:
        parent_embeddings_hash = self.fx.contract.production_embeddings_sha256
        parent_summary_hash = self.fx.contract.production_summary_sha256
        result = self.promote()
        backup = result.promotion_dir / "production_before_promotion"
        self.assertEqual(sha(backup / "student_embeddings.pkl"), parent_embeddings_hash)
        self.assertEqual(sha(backup / "embedding_summary.csv"), parent_summary_hash)

    def test_promotion_rerun_is_idempotent(self) -> None:
        first = self.promote()
        second = self.promote()
        self.assertEqual(first.promotion_id, second.promotion_id)
        self.assertTrue(second.idempotent_reuse)

    def test_existing_pointer_before_first_promotion_is_rejected(self) -> None:
        write_json(
            self.fx.repo / "models" / "current_embedding_version.json",
            {"status": "unknown"},
        )
        with self.assertRaises(mod.Phase12OPromotionError):
            self.promote()

    def test_failure_during_second_replace_restores_parent(self) -> None:
        original = mod._atomic_replace_from
        calls = {"count": 0}

        def flaky(source: Path, target: Path, suffix: str) -> None:
            calls["count"] += 1
            if calls["count"] == 2:
                raise OSError("simulated second replacement failure")
            original(source, target, suffix)

        with mock.patch.object(mod, "_atomic_replace_from", side_effect=flaky):
            with self.assertRaises(mod.Phase12OPromotionError):
                self.promote()
        self.assertEqual(sha(self.fx.production_embeddings), self.fx.contract.production_embeddings_sha256)
        self.assertEqual(sha(self.fx.production_summary), self.fx.contract.production_summary_sha256)
        self.assertFalse((self.fx.repo / "models" / "current_embedding_version.json").exists())

    def test_rollback_restores_parent_and_is_idempotent(self) -> None:
        result = self.promote()
        rolled = mod.rollback_promotion(
            repo_root=self.fx.repo,
            promotion_dir=result.promotion_dir,
            confirm_promotion_id=result.promotion_id,
            contract=self.fx.contract,
        )
        self.assertFalse(rolled.idempotent_reuse)
        self.assertEqual(sha(self.fx.production_embeddings), self.fx.contract.production_embeddings_sha256)
        self.assertEqual(sha(self.fx.production_summary), self.fx.contract.production_summary_sha256)
        again = mod.rollback_promotion(
            repo_root=self.fx.repo,
            promotion_dir=result.promotion_dir,
            confirm_promotion_id=result.promotion_id,
            contract=self.fx.contract,
        )
        self.assertTrue(again.idempotent_reuse)

    def test_rollback_requires_exact_confirmation(self) -> None:
        result = self.promote()
        with self.assertRaises(mod.Phase12OPromotionError):
            mod.rollback_promotion(
                repo_root=self.fx.repo,
                promotion_dir=result.promotion_dir,
                confirm_promotion_id="wrong",
                contract=self.fx.contract,
            )

    def test_rollback_refuses_production_drift(self) -> None:
        result = self.promote()
        self.fx.production_embeddings.write_bytes(b"drift")
        with self.assertRaises(mod.Phase12OPromotionError):
            mod.rollback_promotion(
                repo_root=self.fx.repo,
                promotion_dir=result.promotion_dir,
                confirm_promotion_id=result.promotion_id,
                contract=self.fx.contract,
            )


if __name__ == "__main__":
    unittest.main()
