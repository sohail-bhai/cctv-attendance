from __future__ import annotations

import ast
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.face_attendance.product_phase_2k_retrospective_audit import (
    CURRENT_AUTOMATIC_POLICY,
    CURRENT_EMBEDDING_SHA256,
    MIXED_PREDICTED_ROLL,
    MIXED_TRACKLET_ID,
    POLICY_VERSION,
    REQUIRED_OUTPUT_FILES,
    RetrospectiveAuditError,
    build_output_files,
    choose_recommendation,
    classify_sufficiency,
    contamination_declaration,
    evaluate_attendance_semantics,
    identity_fingerprint,
    materialize,
    no_activation_declaration,
    normalize_records,
    normalize_windows_path,
    preflight,
    retention_metrics,
    safety_metrics,
    sha256_file,
    validate_bindings,
    validate_roster,
    verify_immutable_output,
    verify_manifest,
    write_immutable_output,
)


ROOT = Path(__file__).resolve().parents[1]
CONTEXT = {
    "session_id": "2026-08-01__B51__P4__CVO",
    "source_fingerprint": "a" * 64,
    "embedding_family": "fixture-family",
    "embedding_hash": "b" * 64,
    "policy_version": CURRENT_AUTOMATIC_POLICY,
    "policy_compatible": True,
    "evidence_family": "fixture_current",
    "provenance_artifact": "C:\\fixture\\review.csv",
    "provenance_hash": "c" * 64,
}


def fixture_record(**overrides):
    row = {
        "session_id": CONTEXT["session_id"],
        "checkpoint": "CP1",
        "camera": "cam5",
        "parent_track_id": "CP1-cam5-back-TRK00001",
        "evidence_id": "E0001",
        "predicted_identity": "2401100CSE0050",
        "reviewed_identity": "2401100CSE0050",
        "reviewed_disposition": "identified",
        "strict_accepted": True,
        "guarded_candidate": False,
        "score": 0.6,
        "margin": 0.2,
        "observation_count": 10,
        "checkpoint_support": 1,
        "purity_result": "pure_unsplit",
        "quarantine_result": "not_quarantined",
        "carry_forward_result": "exact_match_eligible",
        "previously_correct": True,
        "retained": True,
        "recovered": False,
    }
    row.update(overrides)
    return row


def normalized(*rows, context=None):
    return normalize_records(rows, source_schema="canonical", context=context or CONTEXT)


def fake_output_files():
    return {name: (name + "\n").encode("utf-8") for name in REQUIRED_OUTPUT_FILES if name != "immutable_manifest.json"}


class FixtureFirstNormalizationAndSafetyTests(unittest.TestCase):
    def test_01_complete_reviewed_strict_safe_session(self):
        metrics = safety_metrics(normalized(fixture_record()), "strict_accepted")
        self.assertEqual(metrics["correct"], 1)
        self.assertEqual(metrics["wrong_person"], 0)

    def test_02_wrong_person_strict_acceptance(self):
        rows = normalized(fixture_record(reviewed_identity="2401100CSE0044"))
        self.assertEqual(safety_metrics(rows, "strict_accepted")["wrong_person"], 1)

    def test_03_outsider_absorption(self):
        rows = normalized(fixture_record(reviewed_identity="", reviewed_disposition="not_in_mapping"))
        self.assertEqual(safety_metrics(rows, "strict_accepted")["outsider"], 1)

    def test_04_mixed_track_strict_acceptance(self):
        rows = normalized(fixture_record(reviewed_identity="", reviewed_disposition="mixed_track"))
        self.assertEqual(safety_metrics(rows, "strict_accepted")["mixed_track"], 1)

    def test_05_mixed_track_correctly_quarantined(self):
        row = normalized(fixture_record(strict_accepted=False, guarded_candidate=True, reviewed_disposition="mixed_track", reviewed_identity="", purity_result="mixed_quarantined", quarantine_result="quarantined", carry_forward_result="mixed_track_quarantine"))[0]
        self.assertFalse(row["strict_accepted"])
        self.assertEqual(row["purity_result"], "mixed_quarantined")

    def test_06_correct_guarded_candidate(self):
        rows = normalized(fixture_record(strict_accepted=False, guarded_candidate=True, recovered=True))
        self.assertEqual(safety_metrics(rows, "guarded_candidate")["correct"], 1)

    def test_07_wrong_person_guarded_candidate(self):
        rows = normalized(fixture_record(strict_accepted=False, guarded_candidate=True, reviewed_identity="2401100CSE0044"))
        self.assertEqual(safety_metrics(rows, "guarded_candidate")["wrong_person"], 1)

    def test_08_partial_review_coverage(self):
        facts = {"manifest_verified": True, "source_provenance": True, "embedding_provenance": True, "track_identity": True, "human_truth": True, "review_join": False, "roster_context": True, "policy_provenance": True, "evaluation_complete": False, "current_policy_compatible": True}
        self.assertEqual(classify_sufficiency(facts), "partial_retrospective_evidence")

    def test_09_missing_private_join_is_partial_not_fabricated(self):
        facts = {"manifest_verified": True, "human_truth": True, "review_join": False}
        self.assertEqual(classify_sufficiency(facts), "partial_retrospective_evidence")

    def test_10_duplicate_conflicting_review_rows_rejected(self):
        first = fixture_record()
        second = fixture_record(reviewed_identity="2401100CSE0044")
        with self.assertRaisesRegex(RetrospectiveAuditError, "duplicate_conflicting"):
            normalized(first, second)

    def test_11_ambiguous_evidence_join_rejected(self):
        with self.assertRaisesRegex(RetrospectiveAuditError, "ambiguous_evidence_join"):
            normalized(fixture_record(evidence_id="", parent_track_id=""))

    def test_12_tampered_source_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            evidence = root / "evidence.csv"
            evidence.write_bytes(b"original\n")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"files_sha256": {"evidence.csv": hashlib.sha256(b"original\n").hexdigest()}}), encoding="utf-8")
            evidence.write_bytes(b"tampered\n")
            with self.assertRaisesRegex(RetrospectiveAuditError, "manifest_file_tampered"):
                verify_manifest(manifest)

    def test_13_source_fingerprint_mismatch_rejected(self):
        with self.assertRaisesRegex(RetrospectiveAuditError, "source_fingerprint_mismatch"):
            validate_bindings(normalized(fixture_record()), source_fingerprint="d" * 64, embedding_hash="b" * 64, policy_version=CURRENT_AUTOMATIC_POLICY)

    def test_14_embedding_hash_mismatch_rejected(self):
        with self.assertRaisesRegex(RetrospectiveAuditError, "embedding_hash_mismatch"):
            validate_bindings(normalized(fixture_record()), source_fingerprint="a" * 64, embedding_hash="d" * 64, policy_version=CURRENT_AUTOMATIC_POLICY)

    def test_15_policy_mismatch_rejected(self):
        with self.assertRaisesRegex(RetrospectiveAuditError, "policy_mismatch"):
            validate_bindings(normalized(fixture_record()), source_fingerprint="a" * 64, embedding_hash="b" * 64, policy_version="changed")

    def test_16_historical_schema_normalization(self):
        raw = {"Session_ID": CONTEXT["session_id"], "Tracklet_ID": "CP2-cam5-back-TRK00002", "Checkpoint_ID": "CP2", "Camera_ID": "cam5", "Candidate_Roll": "2401100CSE0050", "Actual_Roll_Normalized": "2401100CSE0050", "Review_Status": "identified", "Candidate_Accepted_Bool": "True", "Phase_1_2J_Result": "both_correct_same_identity", "Observation_Count": "8"}
        rows = normalize_records([raw], source_schema="phase_1_2j", context={**CONTEXT, "policy_compatible": False})
        self.assertTrue(rows[0]["strict_accepted"])
        self.assertFalse(rows[0]["policy_compatible"])

    def test_17_windows_path_normalization(self):
        self.assertEqual(normalize_windows_path("c:\\Audit\\MON_P4\\review.csv"), "C:/Audit/MON_P4/review.csv")
        with self.assertRaisesRegex(RetrospectiveAuditError, "traversal"):
            normalize_windows_path("C:\\Audit\\..\\secret.csv")

    def test_18_distinct_roll_numbers_preserved(self):
        ai = "24011CSEAI0110"
        cse = "2401100CSE0110"
        rows = normalized(fixture_record(predicted_identity=ai, reviewed_identity=ai), fixture_record(evidence_id="E0002", parent_track_id="CP2-cam5-back-TRK00002", predicted_identity=cse, reviewed_identity=cse))
        self.assertNotEqual(rows[0]["predicted_identity"], rows[1]["predicted_identity"])
        self.assertNotEqual(identity_fingerprint(ai), identity_fingerprint(cse))

    def test_19_out_of_roster_prediction_remains_visible_to_audit(self):
        row = normalized(fixture_record(predicted_identity="OUTSIDER-ID", reviewed_identity="", reviewed_disposition="not_in_mapping"))[0]
        self.assertEqual(row["predicted_identity"], "OUTSIDER-ID")
        self.assertEqual(row["identity_outcome"], "outsider")

    def test_20_missing_enrollment_semantics(self):
        result = evaluate_attendance_semantics(["Present", "Missing Enrollment"], roster_count=2, low_quality_run=False)
        self.assertEqual(result["counts"]["Missing Enrollment"], 1)
        self.assertTrue(result["missing_enrollment_distinct"])

    def test_21_low_quality_non_detection_remains_unconfirmed(self):
        result = evaluate_attendance_semantics(["Unconfirmed", "Unconfirmed"], roster_count=2, low_quality_run=True)
        self.assertTrue(result["low_quality_non_detection_safe"])
        self.assertFalse(result["false_mass_absence"])

    def test_22_false_automatic_absent_detected(self):
        result = evaluate_attendance_semantics(["Absent", "Unconfirmed"], roster_count=2, low_quality_run=True)
        self.assertTrue(result["false_mass_absence"])

    def test_23_roster_row_missing_detected(self):
        result = evaluate_attendance_semantics(["Present"], roster_count=2, low_quality_run=False)
        self.assertFalse(result["roster_complete"])

    def test_24_duplicate_roster_row_detected(self):
        result = validate_roster(["A", "A"])
        self.assertEqual(result["duplicate_rows"], 1)
        self.assertFalse(result["valid"])

    def test_25_previously_correct_evidence_retained(self):
        result = retention_metrics(normalized(fixture_record(previously_correct=True, retained=True)))
        self.assertEqual(result["retained"], 1)
        self.assertEqual(result["lost"], 0)

    def test_26_previously_correct_evidence_lost(self):
        result = retention_metrics(normalized(fixture_record(previously_correct=True, retained=False)))
        self.assertEqual(result["lost"], 1)

    def test_27_guarded_recovery_restores_correct_checkpoint(self):
        result = retention_metrics(normalized(fixture_record(strict_accepted=False, guarded_candidate=True, previously_correct=False, retained=False, recovered=True)))
        self.assertEqual(result["recovered_checkpoints"], 1)

    def test_28_mixed_quarantine_overrides_exact_carry_forward(self):
        row = normalized(fixture_record(reviewed_disposition="mixed_track", reviewed_identity="", strict_accepted=False, purity_result="mixed_quarantined", quarantine_result="quarantined", carry_forward_result="mixed_track_quarantine"))[0]
        self.assertNotEqual(row["carry_forward_result"], "exact_match_eligible")

    def test_29_partial_carry_forward_is_rejected(self):
        row = normalized(fixture_record(carry_forward_result="partial_review_set_mismatch"))[0]
        self.assertEqual(row["carry_forward_result"], "partial_review_set_mismatch")

    def test_30_metric_unavailable_not_silently_zero(self):
        self.assertEqual(safety_metrics([], "strict_accepted")["correct"], "unavailable")
        self.assertEqual(retention_metrics([])["lost"], "unavailable")
        unsupported_guarded = normalized(fixture_record(guarded_metric_available=False))
        self.assertEqual(safety_metrics(unsupported_guarded, "guarded_candidate")["material"], "unavailable")
        unsupported_retention = normalized(fixture_record(retention_baseline_available=False, recovery_metric_available=False))
        self.assertEqual(retention_metrics(unsupported_retention)["previously_correct"], "unavailable")
        self.assertEqual(retention_metrics(unsupported_retention)["recovered"], "unavailable")

    def test_31_incompatible_policy_metrics_remain_labeled(self):
        rows = normalized(fixture_record(), context={**CONTEXT, "policy_compatible": False, "policy_version": "historical"})
        self.assertFalse(rows[0]["policy_compatible"])
        self.assertEqual(rows[0]["policy_version"], "historical")

    def test_32_combined_aggregation_across_sessions(self):
        first = normalized(fixture_record())[0]
        second = normalized(fixture_record(session_id="2026-08-02__B51__P4__CVO", evidence_id="E0002", parent_track_id="CP2-cam5-back-TRK00002"), context={**CONTEXT, "session_id": "2026-08-02__B51__P4__CVO"})[0]
        self.assertEqual(safety_metrics([first, second], "strict_accepted")["correct"], 2)


class DeterminismAndPreservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before_hashes = {relative: sha256_file(ROOT / relative) for relative in (
            "data/attendance_status.json", "data/job_runtime.json", "data/role_users.json",
            "data/student_faculty_map.json", "data/manual_overrides.json",
            "data/review_evidence_registry.json", "models/student_embeddings.pkl",
            "models/embedding_summary.csv", "models/current_embedding_version.json",
            "timetable_b51_2026_2027.csv",
        )}
        cls.plan = preflight(ROOT)

    def test_33_deterministic_output_bytes(self):
        self.assertEqual(build_output_files(self.plan), build_output_files(self.plan))

    def test_34_idempotent_materialization(self):
        with tempfile.TemporaryDirectory() as temp:
            files = fake_output_files()
            first, reused = write_immutable_output(Path(temp), "retrospective-audit-fixture", "a" * 64, files)
            self.assertFalse(reused)
            before = {path.name: path.read_bytes() for path in first.iterdir()}
            second, reused = write_immutable_output(Path(temp), "retrospective-audit-fixture", "a" * 64, files)
            self.assertTrue(reused)
            self.assertEqual(first, second)
            self.assertEqual(before, {path.name: path.read_bytes() for path in second.iterdir()})

    def test_35_same_deterministic_id_with_different_bytes_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            files = fake_output_files()
            write_immutable_output(Path(temp), "retrospective-audit-collision", "a" * 64, files)
            changed = copy.deepcopy(files)
            changed["audit_policy.json"] = b"changed\n"
            with self.assertRaisesRegex(RetrospectiveAuditError, "deterministic_id_collision"):
                write_immutable_output(Path(temp), "retrospective-audit-collision", "a" * 64, changed)

    def test_36_immutable_manifest_tamper_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            output, _ = write_immutable_output(Path(temp), "retrospective-audit-tamper", "a" * 64, fake_output_files())
            (output / "audit_policy.json").write_bytes(b"tampered\n")
            with self.assertRaisesRegex(RetrospectiveAuditError, "immutable_file_hash_changed"):
                verify_immutable_output(output)

    def test_37_no_video_decoding_or_inference_import(self):
        source = (ROOT / "src/face_attendance/product_phase_2k_retrospective_audit.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        imports.update(node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom))
        self.assertFalse(any(name.startswith(("cv2", "face_recognition", "onnxruntime")) for name in imports))

    def test_38_protected_operational_hashes_unchanged(self):
        after = {relative: sha256_file(ROOT / relative) for relative in self.before_hashes}
        self.assertEqual(self.before_hashes, after)
        self.assertEqual(self.plan.protected_hashes, after)

    def test_39_mandatory_mixed_track_regression(self):
        rows = [row for row in self.plan.normalized_records if row["parent_track_id"] == MIXED_TRACKLET_ID and row["evidence_family"] == "phase_2h_current_authority"]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["predicted_identity"], MIXED_PREDICTED_ROLL)
        self.assertFalse(row["strict_accepted"])
        self.assertEqual(row["purity_result"], "mixed_quarantined")
        self.assertNotEqual(row["carry_forward_result"], "exact_match_eligible")

    def test_40_honest_contamination_and_claim_limits(self):
        declaration = contamination_declaration()
        self.assertTrue(declaration["sessions_contaminated"])
        self.assertFalse(declaration["independent_generalization_claimed"])
        self.assertFalse(declaration["guarded_recovery_promotion_authorized"])
        self.assertEqual(len(declaration["claims"]), 4)

    def test_41_complete_sufficiency_requires_all_contract_fields(self):
        facts = {key: True for key in ("manifest_verified", "source_provenance", "embedding_provenance", "track_identity", "human_truth", "review_join", "roster_context", "policy_provenance", "evaluation_complete", "current_policy_compatible")}
        self.assertEqual(classify_sufficiency(facts), "complete_retrospective_evidence")

    def test_42_no_human_truth_is_insufficient(self):
        self.assertEqual(classify_sufficiency({"manifest_verified": True, "human_truth": False}), "insufficient_for_identity_safety_evaluation")

    def test_43_real_sufficiency_has_one_exact_current_session(self):
        classes = {row["Session_ID"]: row["Classification"] for row in self.plan.sufficiency_rows}
        self.assertEqual(sum(value == "complete_retrospective_evidence" for value in classes.values()), 1)
        self.assertEqual(sum(value == "incompatible_policy_evidence" for value in classes.values()), 3)

    def test_44_recommendation_enum_is_fail_closed(self):
        self.assertEqual(choose_recommendation(artifact_verification_passed=False, exact_current_evidence_exists=True, unsafe_current_strict_accepts=0, all_sessions_complete_and_independent=False), "artifact_verification_failed")
        self.assertEqual(choose_recommendation(artifact_verification_passed=True, exact_current_evidence_exists=True, unsafe_current_strict_accepts=1, all_sessions_complete_and_independent=False), "strict_authority_unsafe")
        self.assertEqual(self.plan.analyses["recommendation"]["recommendation"], "retain_current_policy_with_blockers")

    def test_45_no_activation_declaration_is_complete_and_false(self):
        declaration = no_activation_declaration()
        state_values = [value for key, value in declaration.items() if key not in {"schema_version", "phase"}]
        self.assertEqual(len(state_values), 22)
        self.assertTrue(all(value is False for value in state_values))

    def test_46_required_output_file_contract(self):
        files = build_output_files(self.plan)
        self.assertEqual(set(files) | {"immutable_manifest.json"}, set(REQUIRED_OUTPUT_FILES))

    def test_47_private_join_identities_are_redacted_in_normalized_output(self):
        text = build_output_files(self.plan)["normalized_review_evidence.csv"].decode("utf-8")
        self.assertIn("restricted_fingerprints_only", text)
        self.assertNotIn("2401100CSE0050", text)
        self.assertIn(MIXED_PREDICTED_ROLL, text)

    def test_48_real_attendance_totals_are_preserved(self):
        self.assertEqual(self.plan.analyses["official_counts"], {"Present": 6, "Needs Review": 4, "Unconfirmed": 16, "Missing Enrollment": 1, "Absent": 0, "Total": 27})
        self.assertEqual(self.plan.analyses["candidate_counts"], {"Present": 2, "Needs Review": 8, "Unconfirmed": 16, "Missing Enrollment": 1, "Absent": 0, "Total": 27})

    def test_49_all_consumed_manifests_verified(self):
        self.assertGreaterEqual(len(self.plan.manifest_rows), 14)
        self.assertTrue(all(row["Verified"] for row in self.plan.manifest_rows))

    def test_50_real_output_id_and_fingerprint_are_deterministic(self):
        second = preflight(ROOT)
        self.assertEqual(self.plan.run_id, second.run_id)
        self.assertEqual(self.plan.run_fingerprint_sha256, second.run_fingerprint_sha256)

    def test_51_real_current_strict_slice_is_safe(self):
        current = [row for row in self.plan.normalized_records if row["policy_compatible"]]
        metrics = safety_metrics(current, "strict_accepted")
        self.assertEqual(metrics["correct"], 19)
        self.assertEqual(metrics["wrong_person"], 0)
        self.assertEqual(metrics["outsider"], 0)
        self.assertEqual(metrics["mixed_track"], 0)

    def test_52_historical_guarded_slice_exposes_known_risk(self):
        tue_p2 = [row for row in self.plan.normalized_records if row["session_id"] == "2026-06-30__B51__P2__CVO"]
        metrics = safety_metrics(tue_p2, "guarded_candidate")
        self.assertEqual(metrics["correct"], 13)
        self.assertEqual(metrics["wrong_person"], 5)
        self.assertEqual(metrics["outsider"], 2)

    def test_53_historical_out_of_roster_prediction_is_flagged_without_condemning_current_policy(self):
        rows = {row["Session_ID"]: row for row in self.plan.analyses["roster_rows"]}
        self.assertEqual(rows["2026-06-30__B51__P1__CVO"]["Out_Of_Roster_Predictions"], 1)
        self.assertTrue(rows["2026-06-30__B51__P1__CVO"]["Integrity_Violation"])
        self.assertFalse(rows["2026-06-30__B51__P1__CVO"]["Current_Policy_Integrity_Violation"])

    def test_54_unsupported_historical_guarded_metrics_are_unavailable(self):
        for session in ("2026-06-22__B51__P3__CVO", "2026-06-30__B51__P1__CVO"):
            rows = [row for row in self.plan.normalized_records if row["session_id"] == session]
            self.assertEqual(safety_metrics(rows, "guarded_candidate")["material"], "unavailable")

    def test_55_partial_retention_families_keep_unsupported_halves_unavailable(self):
        tue_p1 = [row for row in self.plan.normalized_records if row["session_id"] == "2026-06-30__B51__P1__CVO"]
        tue_p2 = [row for row in self.plan.normalized_records if row["session_id"] == "2026-06-30__B51__P2__CVO"]
        self.assertEqual(retention_metrics(tue_p1)["recovered"], "unavailable")
        self.assertEqual(retention_metrics(tue_p2)["previously_correct"], "unavailable")
        self.assertEqual(retention_metrics(tue_p2)["recovered"], 13)

    def test_56_session_inventory_explicitly_maps_all_evidence_components(self):
        self.assertEqual(len(self.plan.inventory_rows), 15)
        coverage = ";".join(str(row["Evidence_Coverage"]) for row in self.plan.inventory_rows)
        for required in ("source_freeze_manifest", "source_video_hashes", "tracklet_observations", "private_review_join", "human_labels", "authority_report", "tracklet_purity", "exact_carry_forward"):
            self.assertIn(required, coverage)
        self.assertTrue(all(row["Manifest_Verified"] for row in self.plan.inventory_rows))

    def test_57_mandatory_identity_visibility_is_scoped_to_session_and_track(self):
        text = build_output_files(self.plan)["normalized_review_evidence.csv"].decode("utf-8")
        self.assertEqual(text.count("mandatory_regression"), 1)
        same_track_other_session = [row for row in self.plan.normalized_records if row["parent_track_id"] == MIXED_TRACKLET_ID and row["session_id"] != "2026-06-22__B51__P4__CVO"]
        self.assertEqual(len(same_track_other_session), 1)


if __name__ == "__main__":
    unittest.main()
