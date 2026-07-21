from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from src.face_attendance.embedding_db import MatchResult
from src.face_attendance.mon_p3_retention_attribution import (
    MIN_REPRODUCTION_IOU,
    RetentionAttributionError,
    _assert_reproduced_match,
    _bbox_iou,
    _causal_variant_classification,
    _canonical,
    _parse_bbox,
    _required_int,
    _source_group_for_roll,
    _stable_digest,
    _variant_additions,
    _video_map,
)


class GeometryTests(unittest.TestCase):
    def test_parse_bbox_accepts_valid_value(self) -> None:
        self.assertEqual(_parse_bbox("1,2,3,4"), (1.0, 2.0, 3.0, 4.0))

    def test_parse_bbox_rejects_bad_dimensions(self) -> None:
        with self.assertRaises(RetentionAttributionError):
            _parse_bbox("1,2,0,4")

    def test_bbox_iou_exact_match(self) -> None:
        self.assertAlmostEqual(_bbox_iou((1, 2, 3, 4), (1, 2, 3, 4)), 1.0)

    def test_bbox_iou_disjoint(self) -> None:
        self.assertEqual(_bbox_iou((0, 0, 2, 2), (5, 5, 2, 2)), 0.0)

    def test_reproduction_floor_is_fail_closed(self) -> None:
        self.assertGreaterEqual(MIN_REPRODUCTION_IOU, 0.35)


class SourceInventoryTests(unittest.TestCase):
    def setUp(self) -> None:
        base = pd.DataFrame(
            [
                {"Source_ID": "E1", "Canonical_Roll": "A", "Source_Kind": "cleaned_enrollment", "Source_Session": "", "Image_Path": "a"},
                {"Source_ID": "E2", "Canonical_Roll": "B", "Source_Kind": "cleaned_enrollment", "Source_Session": "", "Image_Path": "b"},
            ]
        )
        self.frames = {
            "A_cleaned_enrollment": base,
            "B_p2_recovered_for_p1": pd.concat(
                [base, pd.DataFrame([{"Source_ID": "R1", "Canonical_Roll": "0140", "Source_Kind": "same_track_recovered_cctv_medoid", "Source_Session": "P2", "Image_Path": "r1"}])],
                ignore_index=True,
            ),
            "C_p1_approved_for_p2": pd.concat(
                [base, pd.DataFrame([{"Source_ID": "C1", "Canonical_Roll": "0121", "Source_Kind": "human_approved_cctv_crop", "Source_Session": "P1", "Image_Path": "c1"}])],
                ignore_index=True,
            ),
            "D_full_candidate": pd.concat(
                [
                    base,
                    pd.DataFrame(
                        [
                            {"Source_ID": "R1", "Canonical_Roll": "0140", "Source_Kind": "same_track_recovered_cctv_medoid", "Source_Session": "P2", "Image_Path": "r1"},
                            {"Source_ID": "C1", "Canonical_Roll": "0121", "Source_Kind": "human_approved_cctv_crop", "Source_Session": "P1", "Image_Path": "c1"},
                        ]
                    ),
                ],
                ignore_index=True,
            ),
        }

    def test_variant_additions_exclude_cleaned_base(self) -> None:
        additions = _variant_additions(self.frames)
        self.assertNotIn("E1", set(additions["Source_ID"]))
        self.assertEqual(set(additions["Source_ID"]), {"R1", "C1"})

    def test_recovered_roll_group_is_identified(self) -> None:
        additions = _variant_additions(self.frames)
        self.assertEqual(
            _source_group_for_roll(additions, "0140"),
            "tue_p2_recovered_medoid_group",
        )

    def test_approved_roll_group_is_identified(self) -> None:
        additions = _variant_additions(self.frames)
        self.assertEqual(
            _source_group_for_roll(additions, "0121"),
            "tue_p1_approved_cctv_group",
        )

    def test_unknown_roll_falls_back_to_existing_sources(self) -> None:
        additions = _variant_additions(self.frames)
        self.assertEqual(
            _source_group_for_roll(additions, "9999"),
            "cleaned_enrollment_or_existing_sources",
        )


class CausalClassificationTests(unittest.TestCase):
    def row(self, prod: bool, a: bool, b: bool, c: bool, d: bool) -> dict:
        return {
            "production_Accepted": prod,
            "A_cleaned_enrollment_Accepted": a,
            "B_p2_recovered_for_p1_Accepted": b,
            "C_p1_approved_for_p2_Accepted": c,
            "D_full_candidate_Accepted": d,
        }

    def test_cleaned_enrollment_can_be_attributed(self) -> None:
        self.assertEqual(
            _causal_variant_classification(self.row(True, False, False, False, False)),
            "cleaned_enrollment_introduced_loss",
        )

    def test_p2_recovered_group_can_be_attributed(self) -> None:
        self.assertEqual(
            _causal_variant_classification(self.row(True, True, False, True, False)),
            "tue_p2_recovered_medoid_group_introduced_loss",
        )

    def test_p1_approved_group_can_be_attributed(self) -> None:
        self.assertEqual(
            _causal_variant_classification(self.row(True, True, True, False, False)),
            "tue_p1_approved_cctv_group_introduced_loss",
        )

    def test_interaction_can_be_attributed(self) -> None:
        self.assertEqual(
            _causal_variant_classification(self.row(True, True, True, True, False)),
            "cross_session_source_interaction_introduced_loss",
        )

    def test_both_groups_can_independently_trigger_loss(self) -> None:
        self.assertEqual(
            _causal_variant_classification(self.row(True, True, False, False, False)),
            "both_source_groups_independently_trigger_loss",
        )

    def test_non_loss_is_not_attributed(self) -> None:
        self.assertEqual(
            _causal_variant_classification(self.row(True, True, True, True, True)),
            "not_a_retention_loss",
        )


class ReproductionContractTests(unittest.TestCase):
    def recorded(self) -> dict:
        return {
            "Production_Tracklet_Best_Roll": "2401100CSE0019",
            "Production_Tracklet_Second_Roll": "2401100CSE0110",
            "Production_Tracklet_Best_Score": "0.4847",
            "Production_Tracklet_Second_Score": "0.3377",
            "Production_Tracklet_Margin": "0.1470",
            "Production_Tracklet_Accepted": "Yes",
            "Production_Tracklet_Matcher_Reason": "accepted",
            "Candidate_Tracklet_Best_Roll": "2401100CSE0019",
            "Candidate_Tracklet_Second_Roll": "2401100CSE0140",
            "Candidate_Tracklet_Best_Score": "0.4847",
            "Candidate_Tracklet_Second_Score": "0.4125",
            "Candidate_Tracklet_Margin": "0.0722",
            "Candidate_Tracklet_Accepted": "No",
            "Candidate_Tracklet_Matcher_Reason": "margin_too_small",
        }

    def test_matching_production_contract_passes(self) -> None:
        result = MatchResult(True, "2401100CSE0019", 0.4847, "2401100CSE0110", 0.3377, 0.1470, "accepted")
        _assert_reproduced_match(
            variant="production", result=result, recorded=self.recorded(), track_id="T1"
        )

    def test_matching_candidate_contract_passes(self) -> None:
        result = MatchResult(False, "2401100CSE0019", 0.4847, "2401100CSE0140", 0.4125, 0.0722, "margin_too_small")
        _assert_reproduced_match(
            variant="D_full_candidate", result=result, recorded=self.recorded(), track_id="T1"
        )

    def test_wrong_second_identity_fails_closed(self) -> None:
        result = MatchResult(False, "2401100CSE0019", 0.4847, "2401100CSE0121", 0.4125, 0.0722, "margin_too_small")
        with self.assertRaises(RetentionAttributionError):
            _assert_reproduced_match(
                variant="D_full_candidate", result=result, recorded=self.recorded(), track_id="T1"
            )

    def test_score_drift_fails_closed(self) -> None:
        result = MatchResult(False, "2401100CSE0019", 0.49, "2401100CSE0140", 0.4125, 0.0775, "margin_too_small")
        with self.assertRaises(RetentionAttributionError):
            _assert_reproduced_match(
                variant="D_full_candidate", result=result, recorded=self.recorded(), track_id="T1"
            )


class RequiredIntegerTests(unittest.TestCase):
    def test_zero_is_preserved_instead_of_treated_as_missing(self) -> None:
        self.assertEqual(_required_int({"count": 0}, "count"), 0)

    def test_frozen_evaluation_counts_accept_zero_false_identities(self) -> None:
        payload = {
            "candidate_lost_correct_production_accepts": 6,
            "candidate_correct_recoveries": 4,
            "candidate_false_identities_or_unsafe_accepts": 0,
        }
        self.assertEqual(_required_int(payload, "candidate_lost_correct_production_accepts"), 6)
        self.assertEqual(_required_int(payload, "candidate_correct_recoveries"), 4)
        self.assertEqual(_required_int(payload, "candidate_false_identities_or_unsafe_accepts"), 0)

    def test_missing_required_integer_fails_closed(self) -> None:
        with self.assertRaises(RetentionAttributionError):
            _required_int({}, "count")

    def test_boolean_and_fractional_values_fail_closed(self) -> None:
        with self.assertRaises(RetentionAttributionError):
            _required_int({"count": False}, "count")
        with self.assertRaises(RetentionAttributionError):
            _required_int({"count": 1.5}, "count")


class InputFingerprintTests(unittest.TestCase):
    def test_stable_digest_is_order_independent(self) -> None:
        self.assertEqual(_stable_digest({"a": 1, "b": 2}), _stable_digest({"b": 2, "a": 1}))

    def test_canonical_roll_strips_display_suffix(self) -> None:
        self.assertEqual(_canonical("2401100CSE0028 (A) Vishnu"), "2401100CSE0028")

    def test_video_map_requires_ten_unique_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for i in range(10):
                source = root / f"{i}.mp4"
                source.write_bytes(b"video")
                rows.append(
                    {
                        "Checkpoint_ID": f"CP{(i // 2) + 1}",
                        "Camera_ID": "cam5" if i % 2 == 0 else "cam10",
                        "Source_File_Name": "back.mp4" if i % 2 == 0 else "front.mp4",
                        "Source_File": str(source),
                    }
                )
            mapping = _video_map({"checkpoint_timing_map": rows})
            self.assertEqual(len(mapping), 10)

    def test_video_map_rejects_missing_checkpoint_rows(self) -> None:
        with self.assertRaises(RetentionAttributionError):
            _video_map({"checkpoint_timing_map": []})


if __name__ == "__main__":
    unittest.main()
