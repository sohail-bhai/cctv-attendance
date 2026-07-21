from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.face_attendance.product_phase_2i_authority import (
    AUTHORITY_REVISION_ID,
    AUTOMATIC_RECOGNITION_AUTHORITY,
    EXPECTED_COUNTS,
    MIXED_TRACKLET_ID,
    MISSING_ENROLLMENT_ROLL,
    ProductPhase2IError,
    apply_automatic_status_semantics,
    apply_authority_revision,
    create_authority_output,
    default_inputs,
    preflight,
    rollback_authority_revision,
    select_guarded_review_candidates,
    verify_authority_output,
)


class ProductPhase2IAuthorityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = Path(__file__).resolve().parents[1]

    def _copy_authority_fixture(self, destination: Path) -> Path:
        # Phase 2I originally superseded a frame-detection report. Tests must
        # use that frozen source fixture rather than mutable live attendance
        # state, which legitimately changes after Phase 2I is applied or a
        # later browser rerun completes.
        source_fixture = (
            self.repo
            / "tests"
            / "fixtures"
            / "product_phase_2i_frame_source_attendance_status.json"
        )
        status_dst = destination / "data" / "attendance_status.json"
        status_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_fixture, status_dst)

        for rel in (
            "data/student_faculty_map.json",
            "attendance_output/product_workflow/phase_2h_multiframe_recovery/"
            "multiframe-recovery-7d9d3a539c0b32e33048/evaluation",
        ):
            src = self.repo / rel
            dst = destination / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        return destination

    def test_exact_frozen_evaluation_builds_expected_roster_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._copy_authority_fixture(Path(tmp))
            result = preflight(default_inputs(root))
            self.assertEqual(result.summary, {**EXPECTED_COUNTS, "attendance_percentage": 22.2})
            statuses = {row["Roll_Number"]: row["Status"] for row in result.corrected_rows}
            self.assertEqual(statuses[MISSING_ENROLLMENT_ROLL], "Missing Enrollment")
            self.assertEqual(sum(status == "Present" for status in statuses.values()), 6)
            self.assertEqual(sum(status == "Needs Review" for status in statuses.values()), 4)
            self.assertEqual(sum(status == "Unconfirmed" for status in statuses.values()), 16)
            self.assertNotIn("Absent", statuses.values())

    def test_mixed_track_is_preserved_as_negative_regression_and_not_confirmed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._copy_authority_fixture(Path(tmp))
            inputs = default_inputs(root)
            result = preflight(inputs)
            ai0051 = next(row for row in result.corrected_rows if row["Roll_Number"] == "24011CSEAI0051")
            self.assertEqual(ai0051["Status"], "Needs Review")
            self.assertEqual(ai0051["Reviewed_Tracklet_Checkpoints"], "CP3; CP5")
            self.assertEqual(ai0051["CP1_Tracklet_Confirmed"], "No")
            joined = pd.read_csv(
                inputs.evaluation_dir / "joined_multiframe_recovery_review.csv",
                dtype=str,
                keep_default_na=False,
            )
            mixed = joined[joined["Tracklet_ID"].eq(MIXED_TRACKLET_ID)]
            self.assertEqual(len(mixed), 1)
            self.assertEqual(mixed.iloc[0]["Review_Status"], "mixed_track")

    def test_tampered_frozen_evaluation_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._copy_authority_fixture(Path(tmp))
            path = default_inputs(root).evaluation_dir / "evaluation_summary.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["decision"] = "tampered"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ProductPhase2IError):
                preflight(default_inputs(root))

    def test_authority_output_is_immutable_and_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._copy_authority_fixture(Path(tmp))
            inputs = default_inputs(root)
            result = preflight(inputs)
            first, reused_first = create_authority_output(inputs, result)
            second, reused_second = create_authority_output(inputs, result)
            self.assertFalse(reused_first)
            self.assertTrue(reused_second)
            self.assertEqual(first, second)
            manifest = verify_authority_output(first)
            self.assertEqual(manifest["authority_revision_id"], AUTHORITY_REVISION_ID)
            corrected = pd.read_csv(first / "corrected_attendance_2026-06-22__B51__P4__CVO.csv")
            self.assertEqual(len(corrected), 27)

    def test_explicit_apply_and_targeted_rollback_restore_exact_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._copy_authority_fixture(Path(tmp))
            inputs = default_inputs(root)
            before = json.loads(inputs.status_path.read_text(encoding="utf-8"))[
                "2026-06-22__B51__P4__CVO"
            ]
            applied = apply_authority_revision(inputs)
            state = json.loads(inputs.status_path.read_text(encoding="utf-8"))
            entry = state["2026-06-22__B51__P4__CVO"]
            self.assertEqual(entry["authority_revision_id"], AUTHORITY_REVISION_ID)
            self.assertEqual(entry["official_recognition_authority"], "reviewed_multiframe_tracklet_evidence")
            self.assertEqual(entry["automatic_recognition_authority"], AUTOMATIC_RECOGNITION_AUTHORITY)
            self.assertFalse(entry["guarded_recovery_automatic"])
            self.assertTrue(entry["source_report_superseded"])
            rollback_authority_revision(inputs, Path(applied["backup_path"]))
            restored = json.loads(inputs.status_path.read_text(encoding="utf-8"))[
                "2026-06-22__B51__P4__CVO"
            ]
            self.assertEqual(restored, before)

    def test_future_automatic_semantics_never_turn_guarded_candidate_present(self):
        rows = [
            {"Roll_Number": "A", "Final_Status": "Present Strong"},
            {"Roll_Number": "B", "Final_Status": "Needs Review"},
            {"Roll_Number": "C", "Final_Status": "Absent"},
            {"Roll_Number": "D", "Final_Status": "Absent"},
            {"Roll_Number": "E", "Final_Status": "Absent"},
        ]
        output = apply_automatic_status_semantics(
            rows,
            quality_requires_review=True,
            missing_embedding_rolls=["E"],
            guarded_candidates={"C": {"CP1", "CP3"}},
        )
        statuses = {row["Roll_Number"]: row["Status"] for row in output}
        self.assertEqual(statuses["A"], "Present")
        self.assertEqual(statuses["B"], "Needs Review")
        self.assertEqual(statuses["C"], "Needs Review")
        self.assertEqual(statuses["D"], "Unconfirmed")
        self.assertEqual(statuses["E"], "Missing Enrollment")
        self.assertNotEqual(statuses["C"], "Present")

    def test_quality_valid_session_can_produce_absent(self):
        output = apply_automatic_status_semantics(
            [{"Roll_Number": "A", "Final_Status": "Absent"}],
            quality_requires_review=False,
            missing_embedding_rolls=[],
        )
        self.assertEqual(output[0]["Status"], "Absent")

    def test_guarded_candidate_selection_requires_repeat_checkpoints(self):
        base = {
            "Tracklet_Eligible": "Yes",
            "Tracklet_Accepted": "No",
            "Tracklet_Best_Roll": "2401100CSE0016",
            "Tracklet_Best_Score": "0.44",
            "Tracklet_Margin": "0.07",
            "Dominant_Frame_Best_Roll": "2401100CSE0016",
            "Dominant_Frame_Best_Share_Pct": "80",
            "Observation_Count": "8",
            "Selected_Observation_Count": "5",
            "Consistent_Embedding_Count": "5",
            "Pairwise_Similarity_Median": "0.70",
        }
        frame = pd.DataFrame(
            [
                {**base, "Tracklet_ID": "T1", "Checkpoint_ID": "CP1"},
                {**base, "Tracklet_ID": "T2", "Checkpoint_ID": "CP3"},
                {
                    **base,
                    "Tracklet_ID": "T3",
                    "Checkpoint_ID": "CP2",
                    "Tracklet_Best_Roll": "2401100CSE0028",
                    "Dominant_Frame_Best_Roll": "2401100CSE0028",
                },
            ]
        )
        candidates = select_guarded_review_candidates(frame)
        self.assertEqual(candidates, {"2401100CSE0016": {"CP1", "CP3"}})


if __name__ == "__main__":
    unittest.main()
