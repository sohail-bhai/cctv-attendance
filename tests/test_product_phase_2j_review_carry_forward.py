from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from src.face_attendance.review_carry_forward import (
    EVIDENCE_SIGNATURE_FIELDS,
    POLICY_VERSION,
    REGISTRY_SCHEMA_VERSION,
    _canonical_json_hash,
    _evidence_signature,
    apply_exact_review_carry_forward,
    verify_registry,
)

ROOT = Path(__file__).resolve().parents[1]
EVALUATION = (
    ROOT
    / "attendance_output"
    / "product_workflow"
    / "phase_2h_multiframe_recovery"
    / "multiframe-recovery-7d9d3a539c0b32e33048"
    / "evaluation"
)


def _synthetic_tracklet_row(tracklet_id: str, checkpoint: str, predicted: str, index: int) -> dict[str, str]:
    row = {field: "" for field in EVIDENCE_SIGNATURE_FIELDS}
    row.update(
        {
            "Tracklet_ID": tracklet_id,
            "Checkpoint_ID": checkpoint,
            "Camera_ID": "cam5",
            "Video": f"{checkpoint.lower()}_cam5.mp4",
            "Start_Frame": str(100 + index * 10),
            "End_Frame": str(109 + index * 10),
            "Observation_Count": "8",
            "Embedding_Count": "8",
            "Selected_Observation_Count": "5",
            "Consistent_Embedding_Count": "5",
            "Medoid_Observation_ID": f"OBS-{index}-3",
            "Member_Observation_IDs": ";".join(f"OBS-{index}-{n}" for n in range(8)),
            "Selected_Observation_IDs": ";".join(f"OBS-{index}-{n}" for n in range(5)),
            "Consistent_Observation_IDs": ";".join(f"OBS-{index}-{n}" for n in range(5)),
            "Dominant_Frame_Best_Roll": predicted,
            "Dominant_Frame_Best_Count": "6",
            "Dominant_Frame_Best_Share_Pct": "75.0",
            "Tracklet_Best_Roll": predicted,
            "Tracklet_Best_Score": "0.455",
            "Tracklet_Second_Roll": "2401100CSE9999",
            "Tracklet_Second_Score": "0.350",
            "Tracklet_Margin": "0.105",
            "Pairwise_Similarity_Median": "0.720",
            "Aggregate_Mode": "medoid",
        }
    )
    return row


def _registry_and_tracklets():
    review = pd.read_csv(EVALUATION / "joined_multiframe_recovery_review.csv", dtype=str, keep_default_na=False)
    evidence = []
    tracklet_rows = []
    for index, (_, row) in enumerate(review.iterrows()):
        predicted = str(row["Predicted_Roll"]).strip()
        actual = str(row["Actual_Roll"]).strip()
        status = str(row["Review_Status"]).strip().lower()
        tracklet_row = _synthetic_tracklet_row(
            str(row["Tracklet_ID"]).strip(),
            str(row["Checkpoint_ID"]).strip(),
            predicted,
            index,
        )
        evidence.append(
            {
                "tracklet_id": tracklet_row["Tracklet_ID"],
                "checkpoint_id": tracklet_row["Checkpoint_ID"],
                "predicted_roll": predicted,
                "review_status": status,
                "actual_roll": actual,
                "accepted_for_carry_forward": status == "identified" and predicted == actual,
                "mixed_track_quarantine": status == "mixed_track",
                "ground_truth_source": str(row["Ground_Truth_Source"]),
                "review_id": str(row["Review_ID"]),
                "evidence_signature_sha256": _evidence_signature(tracklet_row),
            }
        )
        tracklet_rows.append(tracklet_row)
    fingerprint = {
        "session_id": "2026-06-22__B51__P4__CVO",
        "fingerprint_sha256": "exact-reviewed-source",
    }
    registry = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "registry_id": "review-registry-test",
        "session_id": "2026-06-22__B51__P4__CVO",
        "source_fingerprint": fingerprint,
        "review_source": "phase_2h",
        "recovery_rule_id": "rule",
        "reviewed_pairs": 31,
        "accepted_pairs": 30,
        "mixed_track_pairs": 1,
        "evidence": evidence,
    }
    registry["registry_sha256"] = _canonical_json_hash({k: v for k, v in registry.items() if k != "registry_sha256"})
    return verify_registry(registry), pd.DataFrame(tracklet_rows), fingerprint


def _base_rows():
    roster = pd.read_csv(EVALUATION / "multiframe_shadow_roster_report.csv", dtype=str, keep_default_na=False)
    rows = []
    for _, item in roster.iterrows():
        roll = str(item["Roll"]).strip()
        status = "Missing Enrollment" if roll == "2401100CSE0268" else "Unconfirmed"
        rows.append(
            {
                "Roll_Number": roll,
                "Status": status,
                "Final_Status": status,
                "Recognized_Checkpoints": "0",
                "Total_Checkpoints": "5",
                "Total_Accepted_Detections": "0",
            }
        )
    return rows


class ProductPhase2JReviewCarryForwardTests(unittest.TestCase):
    def test_exact_review_carry_forward_reproduces_reviewed_statuses(self):
        registry, tracklets, fingerprint = _registry_and_tracklets()
        rows, result = apply_exact_review_carry_forward(
            _base_rows(), tracklet_frame=tracklets, registry=registry, actual_source_fingerprint=fingerprint
        )
        self.assertTrue(result["applied"])
        counts = pd.Series([row["Final_Status"] for row in rows]).value_counts().to_dict()
        self.assertEqual(
            counts,
            {"Unconfirmed": 16, "Present": 6, "Needs Review": 4, "Missing Enrollment": 1},
        )
        by_roll = {row["Roll_Number"]: row for row in rows}
        self.assertEqual(by_roll["2401100CSE0016"]["Reviewed_Tracklet_Checkpoints"], "CP1; CP2; CP3; CP5")
        self.assertEqual(by_roll["24011CSEAI0051"]["Final_Status"], "Needs Review")
        self.assertEqual(by_roll["24011CSEAI0051"]["Mixed_Track_Checkpoints_Rejected"], "CP1")
        self.assertEqual(result["mixed_track_pairs_rejected"], 1)

    def test_source_fingerprint_change_disables_carry_forward(self):
        registry, tracklets, fingerprint = _registry_and_tracklets()
        changed = dict(fingerprint)
        changed["fingerprint_sha256"] = "different-source"
        rows, result = apply_exact_review_carry_forward(
            _base_rows(), tracklet_frame=tracklets, registry=registry, actual_source_fingerprint=changed
        )
        self.assertEqual(result, {"applied": False, "reason": "source_or_embedding_fingerprint_changed"})
        self.assertTrue(all(row["Final_Status"] in {"Unconfirmed", "Missing Enrollment"} for row in rows))

    def test_missing_reviewed_tracklet_fails_closed_without_partial_reuse(self):
        registry, tracklets, fingerprint = _registry_and_tracklets()
        tracklets = tracklets.iloc[1:].reset_index(drop=True)
        rows, result = apply_exact_review_carry_forward(
            _base_rows(), tracklet_frame=tracklets, registry=registry, actual_source_fingerprint=fingerprint
        )
        self.assertFalse(result["applied"])
        self.assertEqual(result["reason"], "reviewed_evidence_drift")
        self.assertEqual(result["missing_pair_count"], 1)
        self.assertTrue(all(row["Final_Status"] in {"Unconfirmed", "Missing Enrollment"} for row in rows))

    def test_changed_tracklet_evidence_signature_fails_closed(self):
        registry, tracklets, fingerprint = _registry_and_tracklets()
        tracklets.loc[0, "Start_Frame"] = "99999"
        rows, result = apply_exact_review_carry_forward(
            _base_rows(), tracklet_frame=tracklets, registry=registry, actual_source_fingerprint=fingerprint
        )
        self.assertFalse(result["applied"])
        self.assertEqual(result["reason"], "reviewed_evidence_signature_drift")
        self.assertEqual(result["changed_pair_count"], 1)
        self.assertTrue(all(row["Final_Status"] in {"Unconfirmed", "Missing Enrollment"} for row in rows))


if __name__ == "__main__":
    unittest.main()
