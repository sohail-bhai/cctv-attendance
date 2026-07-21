from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from src.face_attendance import phase_1_2p_post_promotion_verification as mod
from tests.test_phase_1_2o_explicit_promotion import Fixture, sha, write_json


class PromotedLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.fx = Fixture(Path(self.temp.name))
        self.promoted = mod.verify_promoted_state  # imported dependency smoke
        result = __import__(
            "src.face_attendance.phase_1_2o_explicit_promotion",
            fromlist=["promote_candidate_family"],
        ).promote_candidate_family(
            self.fx.inputs,
            confirm_family_id=self.fx.family_id,
            contract=self.fx.contract,
        )
        self.inputs = mod.Phase12PInputs(
            repo_root=self.fx.repo,
            promotion_dir=result.promotion_dir,
            family_dir=self.fx.family,
            production_embeddings=self.fx.production_embeddings,
            production_summary=self.fx.production_summary,
            current_pointer=self.fx.repo / "models" / "current_embedding_version.json",
            source_ablation_dir=self.fx.root / "source-ablation",
            parent_family_dir=self.fx.root / "parent-family",
            phase_1_2m_validation_dir=self.fx.root / "phase-1.2m-validation",
        )
        self.inputs.source_ablation_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.inputs.source_ablation_dir / "output_manifest.json", {"files_sha256": {"dummy.txt": "0" * 64}})
        self.inputs.phase_1_2m_validation_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.inputs.phase_1_2m_validation_dir / "output_manifest.json", {"files_sha256": {"dummy.txt": "0" * 64}})

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _run_preflight(self) -> mod.Phase12PPreflight:
        with (
            mock.patch.object(mod, "PROMOTION_ID", self.promoted_id),
            mock.patch.object(mod, "verify_embedding_family", return_value={"verified_files": 116}),
            mock.patch.object(mod, "verify_source_ablation_output", return_value={"verified_files": 11}),
            mock.patch.object(mod, "_verify_phase_1_2m_validation", return_value=7),
            mock.patch.object(mod, "_verify_production_payload", return_value=(354, 128)),
        ):
            return mod.preflight_phase_1_2p(self.inputs, self.fx.contract)

    @property
    def promoted_id(self) -> str:
        return self.inputs.promotion_dir.name

    def test_valid_promoted_state_passes(self) -> None:
        result = self._run_preflight()
        self.assertEqual(result.family_id, self.fx.family_id)
        self.assertEqual(result.production_embedding_records, 354)
        self.assertEqual(result.benchmark_rows_planned, 120)
        self.assertTrue(result.verification_id.startswith("post-promotion-verification-"))

    def test_pointer_status_drift_fails(self) -> None:
        pointer = json.loads(self.inputs.current_pointer.read_text(encoding="utf-8"))
        pointer["status"] = "unknown"
        write_json(self.inputs.current_pointer, pointer)
        with self.assertRaises(mod.Phase12PError):
            self._run_preflight()

    def test_transaction_must_remain_promoted(self) -> None:
        transaction = self.inputs.promotion_dir / "transaction_state.json"
        payload = json.loads(transaction.read_text(encoding="utf-8"))
        payload["status"] = "prepared"
        write_json(transaction, payload)
        with self.assertRaises(mod.Phase12PError):
            self._run_preflight()

    def test_rollback_command_drift_fails(self) -> None:
        pointer = json.loads(self.inputs.current_pointer.read_text(encoding="utf-8"))
        pointer["rollback_command"] = "unsafe"
        write_json(self.inputs.current_pointer, pointer)
        record = self.inputs.promotion_dir / "promotion_record.json"
        record_payload = json.loads(record.read_text(encoding="utf-8"))
        record_payload["current_pointer_sha256"] = sha(self.inputs.current_pointer)
        write_json(record, record_payload)
        with self.assertRaises(mod.Phase12PError):
            self._run_preflight()

    def _write_rollback_command(self, command: str) -> None:
        pointer = json.loads(self.inputs.current_pointer.read_text(encoding="utf-8"))
        pointer["rollback_command"] = command
        write_json(self.inputs.current_pointer, pointer)
        record_path = self.inputs.promotion_dir / "promotion_record.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["rollback_command"] = command
        record["current_pointer_sha256"] = sha(self.inputs.current_pointer)
        write_json(record_path, record)
        manifest_path = self.inputs.promotion_dir / "promotion_output_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files_sha256"]["promotion_record.json"] = sha(record_path)
        write_json(manifest_path, manifest)

    def test_equivalent_resolved_rollback_paths_pass(self) -> None:
        runner = (
            self.inputs.repo_root
            / "scripts"
            / "path-normalization-check"
            / ".."
            / "run_phase_1_2o_explicit_promotion.ps1"
        )
        promotion_dir = (
            self.inputs.promotion_dir.parent
            / "path-normalization-check"
            / ".."
            / self.inputs.promotion_dir.name
        )
        command = (
            f'& "{runner}" -Rollback -PromotionDir "{promotion_dir}" '
            f'-ConfirmRollback "{self.promoted_id}"'
        )
        self._write_rollback_command(command)
        result = self._run_preflight()
        self.assertEqual(result.promotion_id, self.promoted_id)

    def test_rollback_command_extra_argument_fails(self) -> None:
        pointer = json.loads(self.inputs.current_pointer.read_text(encoding="utf-8"))
        command = str(pointer["rollback_command"]) + " -Force"
        self._write_rollback_command(command)
        with self.assertRaises(mod.Phase12PError):
            self._run_preflight()

    def test_production_drift_fails_closed(self) -> None:
        self.inputs.production_embeddings.write_bytes(b"drift")
        with self.assertRaises(mod.Phase12PError):
            self._run_preflight()


class ProductionPayloadTests(unittest.TestCase):
    def test_valid_payload_and_byte_identity_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            family = root / "family"
            variant = family / "variants" / "full_candidate"
            variant.mkdir(parents=True)
            records = [
                {
                    "roll": f"R{index:03d}",
                    "embedding": np.full(128, index + 1, dtype=np.float32),
                }
                for index in range(mod.EXPECTED_RECORDS)
            ]
            payload = {"records": records, "metadata": {}}
            candidate_embeddings = variant / "student_embeddings.pkl"
            production_embeddings = root / "student_embeddings.pkl"
            with candidate_embeddings.open("wb") as handle:
                pickle.dump(payload, handle)
            production_embeddings.write_bytes(candidate_embeddings.read_bytes())
            candidate_summary = variant / "embedding_summary.csv"
            production_summary = root / "embedding_summary.csv"
            candidate_summary.write_text("Roll,Count\n", encoding="utf-8")
            production_summary.write_bytes(candidate_summary.read_bytes())
            contract = mod.PromotionContract(
                **{
                    **mod.DEFAULT_CONTRACT.__dict__,
                    "candidate_embeddings_sha256": sha(candidate_embeddings),
                    "candidate_summary_sha256": sha(candidate_summary),
                }
            )
            inputs = mod.Phase12PInputs(
                repo_root=root,
                promotion_dir=root / "promotion",
                family_dir=family,
                production_embeddings=production_embeddings,
                production_summary=production_summary,
                current_pointer=root / "pointer.json",
                source_ablation_dir=root / "ablation",
                parent_family_dir=root / "parent",
                phase_1_2m_validation_dir=root / "validation",
            )
            count, dimension = mod._verify_production_payload(inputs, contract)
            self.assertEqual(count, mod.EXPECTED_RECORDS)
            self.assertEqual(dimension, 128)

    def test_non_finite_vector_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            family = root / "family"
            variant = family / "variants" / "full_candidate"
            variant.mkdir(parents=True)
            records = [
                {"roll": f"R{index:03d}", "embedding": np.ones(128, dtype=np.float32)}
                for index in range(mod.EXPECTED_RECORDS)
            ]
            records[0]["embedding"][0] = np.nan
            payload = {"records": records}
            candidate_embeddings = variant / "student_embeddings.pkl"
            production_embeddings = root / "student_embeddings.pkl"
            with candidate_embeddings.open("wb") as handle:
                pickle.dump(payload, handle)
            production_embeddings.write_bytes(candidate_embeddings.read_bytes())
            candidate_summary = variant / "embedding_summary.csv"
            production_summary = root / "embedding_summary.csv"
            candidate_summary.write_text("x\n", encoding="utf-8")
            production_summary.write_bytes(candidate_summary.read_bytes())
            contract = mod.PromotionContract(
                **{
                    **mod.DEFAULT_CONTRACT.__dict__,
                    "candidate_embeddings_sha256": sha(candidate_embeddings),
                    "candidate_summary_sha256": sha(candidate_summary),
                }
            )
            inputs = mod.Phase12PInputs(
                root,
                root / "promotion",
                family,
                production_embeddings,
                production_summary,
                root / "pointer",
                root / "ablation",
                root / "parent",
                root / "validation",
            )
            with self.assertRaises(mod.Phase12PError):
                mod._verify_production_payload(inputs, contract)


class OutputTests(unittest.TestCase):
    def test_run_writes_immutable_output_and_reuses_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = mod.Phase12PInputs.for_repo(root)
            preflight = mod.Phase12PPreflight(
                verification_id="post-promotion-verification-test",
                promotion_id=mod.PROMOTION_ID,
                family_id=mod.DEFAULT_CONTRACT.family_id,
                candidate_variant_id=mod.DEFAULT_CONTRACT.candidate_variant_id,
                production_embeddings_sha256=mod.DEFAULT_CONTRACT.candidate_embeddings_sha256,
                production_summary_sha256=mod.DEFAULT_CONTRACT.candidate_summary_sha256,
                parent_embeddings_backup_sha256=mod.DEFAULT_CONTRACT.production_embeddings_sha256,
                parent_summary_backup_sha256=mod.DEFAULT_CONTRACT.production_summary_sha256,
                production_embedding_records=354,
                production_embedding_dimension=128,
                family_verified_files=116,
                promotion_verified_files=17,
                source_ablation_verified_files=11,
                phase_1_2m_validation_verified_files=7,
                benchmark_rows_planned=120,
                mon_p3_rows_planned=15,
                rollback_command="rollback command",
                input_fingerprint_sha256="a" * 64,
            )
            benchmark = pd.DataFrame(
                {"Benchmark_Row_ID": [f"B{i:03d}" for i in range(120)], "Reproduction_Passed": True}
            )
            mon = pd.DataFrame(
                {"Review_ID": [f"R{i:03d}" for i in range(15)], "Reproduction_Passed": True}
            )
            output_root = root / "output"
            with (
                mock.patch.object(mod, "preflight_phase_1_2p", return_value=preflight),
                mock.patch.object(mod, "_reproduce_frozen_predictions", return_value=(benchmark, mon)),
            ):
                first = mod.run_phase_1_2p(inputs=inputs, output_root=output_root)
                second = mod.run_phase_1_2p(inputs=inputs, output_root=output_root)
            self.assertFalse(first.idempotent_reuse)
            self.assertTrue(second.idempotent_reuse)
            self.assertTrue((first.output_dir / "verification_summary.json").is_file())
            self.assertTrue((first.output_dir / "output_manifest.json").is_file())
            summary = json.loads(
                (first.output_dir / "verification_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["decision"], "promoted_production_operational_verification_passed")
            self.assertEqual(summary["benchmark_rows_reproduced"], 120)


if __name__ == "__main__":
    unittest.main()
