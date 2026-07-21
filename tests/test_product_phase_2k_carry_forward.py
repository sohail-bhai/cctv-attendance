from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.face_attendance.product_phase_2k_carry_forward import (
    CAPTURE_CONTRACT_VERSION,
    EXPECTED_PHASE_2K_A_MANIFEST_SHA256,
    MIXED_TRACKLET_ID,
    POLICY_VERSION,
    CarryForwardContractError,
    _actual_verification_rows,
    build_output_files,
    canonical_json_bytes,
    capture_contract,
    carry_forward_policy,
    evaluate_exact_carry_forward,
    normalize_provenance_path,
    preflight,
    synthetic_exact_bundle,
    synthetic_verification_matrix,
    validate_capture_manifest,
    verify_immutable_output,
    write_immutable_output,
)


ROOT = Path(__file__).resolve().parents[1]


class ExactCarryForwardContractTests(unittest.TestCase):
    def test_exact_full_match_is_eligible_deterministic_and_idempotent(self):
        expected = synthetic_exact_bundle()
        first = evaluate_exact_carry_forward(expected, copy.deepcopy(expected))
        second = evaluate_exact_carry_forward(expected, copy.deepcopy(expected))
        self.assertEqual(first, second)
        self.assertTrue(first["eligible"])
        self.assertEqual(first["result"], "exact_match_eligible")
        self.assertEqual(first["review_ids_carried"], ["R0001", "R0002"])

    def test_mandatory_fixture_matrix_covers_exact_mismatch_and_fail_closed_cases(self):
        rows = synthetic_verification_matrix()
        self.assertEqual(len(rows), 34)
        self.assertTrue(all(row["Passed"] for row in rows))
        by_name = {row["Scenario"]: row for row in rows}
        self.assertEqual(
            by_name["one_source_video_hash_mismatch"]["Actual_Result"],
            "ineligible_source_video_hash_changed",
        )
        self.assertEqual(by_name["mixed_parent_exact_hashes"]["Actual_Result"], "mixed_quarantined")
        self.assertEqual(by_name["child_evidence_missing"]["Actual_Result"], "incomplete_evidence")
        self.assertEqual(by_name["manifest_tampering"]["Actual_Result"], "invalid_or_tampered")

    def test_all_or_nothing_rejects_partial_review_set_and_never_inherits_to_children(self):
        expected = synthetic_exact_bundle()
        current = copy.deepcopy(expected)
        current["review"]["rows"] = current["review"]["rows"][:1]
        result = evaluate_exact_carry_forward(expected, current)
        self.assertFalse(result["eligible"])
        self.assertEqual(result["result"], "ineligible_partial_review_set_mismatch")
        self.assertFalse(result["partial_carry_forward"])
        self.assertFalse(result["inherited_by_children"])
        self.assertEqual(result["review_ids_carried"], [])

    def test_source_list_physical_order_is_canonicalized_but_order_binding_is_exact(self):
        expected = synthetic_exact_bundle()
        physically_reordered = copy.deepcopy(expected)
        physically_reordered["source"]["videos"].reverse()
        self.assertTrue(evaluate_exact_carry_forward(expected, physically_reordered)["eligible"])
        changed_binding = copy.deepcopy(expected)
        changed_binding["source"]["videos"][0]["canonical_order"] = 1
        changed_binding["source"]["videos"][1]["canonical_order"] = 0
        result = evaluate_exact_carry_forward(expected, changed_binding)
        self.assertEqual(result["result"], "ineligible_source_order_changed")

    def test_windows_path_normalization_changes_only_drive_spelling_and_separators(self):
        self.assertEqual(
            normalize_provenance_path("c:\\Class\\CP1\\back.mp4"),
            "C:/Class/CP1/back.mp4",
        )
        self.assertNotEqual(
            normalize_provenance_path("C:/Class/CP1/back.mp4"),
            normalize_provenance_path("C:/class/CP1/back.mp4"),
        )
        expected = synthetic_exact_bundle()
        current = copy.deepcopy(expected)
        expected["source"]["videos"][0]["path"] = "c:\\Class\\CP1\\back.mp4"
        current["source"]["videos"][0]["path"] = "C:/class/CP1/back.mp4"
        self.assertEqual(
            evaluate_exact_carry_forward(expected, current)["result"],
            "ineligible_source_path_changed",
        )

    def test_duplicate_signature_and_missing_provenance_are_invalid_or_incomplete(self):
        expected = synthetic_exact_bundle()
        duplicate = copy.deepcopy(expected)
        duplicate["review"]["rows"][1]["original_evidence_signature_sha256"] = duplicate[
            "review"
        ]["rows"][0]["original_evidence_signature_sha256"]
        self.assertEqual(
            evaluate_exact_carry_forward(expected, duplicate)["reason_code"],
            "duplicate_evidence_signature",
        )
        missing = copy.deepcopy(expected)
        del missing["parents"][0]["ordered_time_indices"]
        result = evaluate_exact_carry_forward(expected, missing)
        self.assertEqual(result["result"], "incomplete_evidence")
        self.assertIn("parents[0].ordered_time_indices", result["missing_fields"])


class ImmutableOutputTests(unittest.TestCase):
    def test_repeated_materialization_is_byte_identical_and_collision_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = {"carry_forward_policy.json": b"{}\n", "evaluation_summary.json": b"{}\n"}
            output, reused = write_immutable_output(output_root=root, run_id="carry-forward-test", files=files)
            self.assertFalse(reused)
            before = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
            same, reused = write_immutable_output(output_root=root, run_id="carry-forward-test", files=files)
            self.assertTrue(reused)
            self.assertEqual(output, same)
            after = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
            self.assertEqual(before, after)
            with self.assertRaisesRegex(CarryForwardContractError, "deterministic_id_collision"):
                write_immutable_output(
                    output_root=root,
                    run_id="carry-forward-test",
                    files={**files, "evaluation_summary.json": b'{"changed":true}\n'},
                )

    def test_tampering_is_detected_independently(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output, _ = write_immutable_output(
                output_root=root,
                run_id="carry-forward-tamper",
                files={"carry_forward_policy.json": b"{}\n"},
            )
            verify_immutable_output(output)
            (output / "carry_forward_policy.json").write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(CarryForwardContractError, "hash changed"):
                verify_immutable_output(output)


class CaptureContractTests(unittest.TestCase):
    def _manifest(self, root: Path) -> dict:
        sources = []
        order = 0
        for checkpoint in ("CP1", "CP2", "CP3", "CP4", "CP5"):
            for camera in ("back", "front"):
                relative = f"sources/{checkpoint}/{camera}.mp4"
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = f"fixture-{checkpoint}-{camera}".encode("utf-8")
                path.write_bytes(payload)
                sources.append(
                    {
                        "relative_path": relative,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "size_bytes": len(payload),
                        "checkpoint_id": checkpoint,
                        "camera_id": camera,
                        "canonical_order": order,
                        "duration_seconds": 20.0,
                        "fps": 25.0,
                        "frame_count": 500,
                        "capture_start_timestamp": f"2026-08-01T10:{order:02d}:00+05:30",
                    }
                )
                order += 1
        return {
            "capture_contract_version": CAPTURE_CONTRACT_VERSION,
            "session_id": "2026-08-01__B51__P4__CVO",
            "untouched_declaration": True,
            "prior_use": {
                "recognition_previously_run": False,
                "human_review_exists": False,
                "used_for_model_selection": False,
                "used_for_threshold_or_calibration": False,
                "used_for_source_ablation": False,
                "used_for_promotion_evidence": False,
                "used_for_retention_testing": False,
                "used_in_any_reviewed_benchmark": False,
            },
            "production_configuration": capture_contract(
                hashlib.sha256(canonical_json_bytes(carry_forward_policy())).hexdigest()
            )["fixed_production_configuration"],
            "sources": sources,
            "parameter_tuning_after_results_visible": False,
            "diagnostic_observation_schema_id": "product-phase-2k-c-diagnostic-observation-v1",
            "appearance_evidence_mode": "derived_pairwise_and_local_window_similarity_matrices",
            "blind_review_schema_id": "product-phase-2k-c-blind-review-v1",
        }

    def test_capture_fixture_validates_by_hash_without_decoding_or_recognition(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = validate_capture_manifest(self._manifest(root), package_root=root, verify_files=True)
            self.assertTrue(result["valid"])
            self.assertFalse(result["video_decoded"])
            self.assertFalse(result["recognition_ran"])

    def test_contaminated_or_tampered_capture_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self._manifest(root)
            manifest["prior_use"]["used_for_model_selection"] = True
            with self.assertRaisesRegex(CarryForwardContractError, "contamination"):
                validate_capture_manifest(manifest, package_root=root, verify_files=True)
            manifest = self._manifest(root)
            (root / manifest["sources"][0]["relative_path"]).write_bytes(b"changed")
            with self.assertRaisesRegex(CarryForwardContractError, "size changed|hash changed"):
                validate_capture_manifest(manifest, package_root=root, verify_files=True)


class RealPhase2KBArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = preflight(ROOT, verify_candidate_audit_manifests=False)

    def test_phase_2k_a_manifest_and_all_protected_inputs_verify(self):
        self.assertEqual(cls_value := self.result.phase_2k_a_manifest_sha256, EXPECTED_PHASE_2K_A_MANIFEST_SHA256)
        self.assertEqual(len(self.result.protected_hashes), 10)
        self.assertEqual(len(self.result.roster_rolls), 27)
        self.assertEqual(self.result.missing_enrollment_rolls, ("2401100CSE0268",))
        self.assertEqual(cls_value, EXPECTED_PHASE_2K_A_MANIFEST_SHA256)

    def test_mixed_track_is_rejected_and_safe_exact_rows_remain_diagnostically_eligible(self):
        rows, overall, mixed, _ = _actual_verification_rows(self.result)
        real_pairs = [row for row in rows if row["Scope"] == "verified_phase_2k_a_review_pair"]
        self.assertEqual(sum(1 for row in real_pairs if row["Eligible"]), 18)
        self.assertEqual(sum(1 for row in real_pairs if not row["Eligible"]), 13)
        self.assertEqual(overall["result"], "mixed_quarantined")
        self.assertEqual(mixed["tracklet_id"], MIXED_TRACKLET_ID)
        self.assertEqual(mixed["ordered_observation_count"], 39)
        self.assertEqual(mixed["carry_forward_result"], "mixed_quarantined")
        self.assertEqual(mixed["child_count"], 0)
        self.assertEqual(mixed["official_attendance_contribution"], 0)

    def test_candidate_inventory_has_no_false_untouched_claim(self):
        self.assertEqual(len(self.result.candidate_inventory), 4)
        self.assertTrue(all(not item["Truly_Untouched"] for item in self.result.candidate_inventory))
        self.assertTrue(all(item["Suitability_Result"].startswith("rejected") for item in self.result.candidate_inventory))
        self.assertTrue(all(item["Checkpoint_Count"] == 5 for item in self.result.candidate_inventory))
        self.assertTrue(all(item["Video_Count"] == 10 for item in self.result.candidate_inventory))

    def test_required_immutable_files_and_no_activation_declaration_are_complete(self):
        files, evaluation = build_output_files(self.result)
        self.assertEqual(
            set(files),
            {
                "carry_forward_policy.json", "exact_match_contract.json",
                "carry_forward_verification_matrix.csv", "carry_forward_failure_reasons.csv",
                "mixed_track_regression.json", "candidate_session_inventory.csv",
                "independent_session_recommendation.json", "independent_session_capture_contract.json",
                "diagnostic_observation_schema.json", "blind_review_schema.json",
                "acceptance_criteria.json", "retention_criteria.json", "no_activation_declaration.json",
                "source_manifest.json", "evaluation_summary.json",
            },
        )
        declaration = json.loads(files["no_activation_declaration.json"])
        for key in (
            "recognition_ran", "video_processing_ran", "official_attendance_changed",
            "candidate_attendance_changed", "authority_changed", "guarded_recovery_promoted",
            "embeddings_changed", "roster_changed", "thresholds_changed", "review_registry_changed",
        ):
            self.assertFalse(declaration[key])
        recommendation = json.loads(files["independent_session_recommendation.json"])
        self.assertEqual(recommendation["status"], "no_existing_untouched_session")
        self.assertEqual(evaluation["diagnostically_exact_match_eligible_pairs"], 18)


if __name__ == "__main__":
    unittest.main()
