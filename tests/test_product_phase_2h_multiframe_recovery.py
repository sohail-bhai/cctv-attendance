from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import pandas as pd

import src.face_attendance.product_phase_2h_multiframe_recovery as phase2h

from src.face_attendance.product_phase_2h_multiframe_recovery import (
    EXPECTED_MATERIAL_REPRESENTATIVES,
    EXPECTED_PENDING_REVIEW,
    POLICY_VERSION,
    EXPECTED_SELECTED_PAIRS,
    RECOVERY_RULE_ID,
    REUSED_REVIEW_TRACKS,
    ProductPhase2HError,
    load_reused_ground_truth,
    select_material_representatives,
    summarize_reviewed_material,
    verify_output,
    default_inputs,
    _verify_frozen_source_copy,
)


def _base_row() -> dict[str, object]:
    return {
        "Session_ID": "2026-06-22__B51__P4__CVO",
        "Subject_Abbr": "CVO",
        "Camera_ID": "cam5",
        "Video": "back.mp4",
        "Tracklet_Eligible": "Yes",
        "Tracklet_Accepted": "No",
        "Tracklet_Best_Score": 0.45,
        "Tracklet_Second_Roll": "2401100CSE0254",
        "Tracklet_Second_Score": 0.30,
        "Tracklet_Margin": 0.15,
        "Dominant_Frame_Best_Share_Pct": 90.0,
        "Observation_Count": 12,
        "Selected_Observation_Count": 5,
        "Consistent_Embedding_Count": 5,
        "Pairwise_Similarity_Median": 0.75,
        "Aggregate_Mode": "top3",
        "Match_Threshold": 0.48,
        "Margin_Threshold": 0.08,
    }


def make_tracklets() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    strict_rows: list[dict[str, object]] = []
    for (roll, checkpoint), (tracklet_id, tier) in EXPECTED_MATERIAL_REPRESENTATIVES.items():
        row = _base_row()
        row.update({
            "Tracklet_ID": tracklet_id,
            "Tracklet_Best_Roll": roll,
            "Dominant_Frame_Best_Roll": roll,
            "Checkpoint_ID": checkpoint,
            "Camera_ID": "cam10" if "cam10" in tracklet_id else "cam5",
            "Video": "front.mp4" if "front" in tracklet_id else "back.mp4",
        })
        if tier == "strict_accepted":
            row.update({
                "Tracklet_Accepted": "Yes",
                "Tracklet_Best_Score": 0.60,
                "Tracklet_Margin": 0.20,
            })
            strict_rows.append(row.copy())
        rows.append(row)

    # Raw diagnostics contain 31 strict accepted tracklets. Add lower-ranked duplicates
    # without changing the strongest identity/checkpoint representative.
    for index in range(31 - len(strict_rows)):
        source = strict_rows[index % len(strict_rows)].copy()
        source.update({
            "Tracklet_ID": f"STRICT-EXTRA-{index:02d}",
            "Tracklet_Best_Score": 0.49,
            "Tracklet_Margin": 0.09,
            "Pairwise_Similarity_Median": 0.40,
            "Observation_Count": 2,
            "Selected_Observation_Count": 2,
            "Consistent_Embedding_Count": 2,
        })
        rows.append(source)
    return pd.DataFrame(rows)


class SelectionTests(unittest.TestCase):
    def test_exact_combined_representatives_are_selected(self) -> None:
        selected = select_material_representatives(make_tracklets())
        actual = {
            (row["Tracklet_Best_Roll"], row["Checkpoint_ID"]):
            (row["Tracklet_ID"], row["Evidence_Tier"])
            for row in selected.to_dict("records")
        }
        self.assertEqual(actual, EXPECTED_MATERIAL_REPRESENTATIVES)
        self.assertEqual(len(selected), EXPECTED_SELECTED_PAIRS)
        self.assertEqual(int(selected["Evidence_Tier"].eq("strict_accepted").sum()), 19)
        self.assertEqual(int(selected["Evidence_Tier"].eq("guarded_recovery").sum()), 12)
        self.assertTrue(
            selected.loc[selected["Evidence_Tier"].eq("guarded_recovery"), "Recovery_Rule_ID"]
            .eq(RECOVERY_RULE_ID).all()
        )

    def test_raw_strict_accepted_count_change_fails_closed(self) -> None:
        frame = make_tracklets()
        frame = frame[~frame["Tracklet_ID"].eq("STRICT-EXTRA-00")].copy()
        with self.assertRaisesRegex(ProductPhase2HError, "Expected 31 accepted tracklets"):
            select_material_representatives(frame)

    def test_weakened_recovery_candidate_changes_contract_and_fails(self) -> None:
        frame = make_tracklets()
        target = frame["Tracklet_ID"].eq("CP1-cam5-back-TRK00003")
        frame.loc[target, "Tracklet_Best_Score"] = 0.39
        with self.assertRaisesRegex(ProductPhase2HError, "selection changed"):
            select_material_representatives(frame)

    def test_dominant_disagreement_blocks_recovery(self) -> None:
        frame = make_tracklets()
        target = frame["Tracklet_ID"].eq("CP1-cam5-back-TRK00003")
        frame.loc[target, "Dominant_Frame_Best_Roll"] = "2401100CSE0254"
        with self.assertRaisesRegex(ProductPhase2HError, "selection changed"):
            select_material_representatives(frame)


class FrozenGroundTruthTests(unittest.TestCase):
    def _write_prior(self, root: Path, *, altered: bool = False) -> Path:
        rows = []
        for index, (tracklet_id, roll) in enumerate(REUSED_REVIEW_TRACKS.items()):
            rows.append({
                "Tracklet_ID": tracklet_id,
                "Review_Status": "identified",
                "Actual_Roll": "2401100CSE0050" if altered and index == 0 else roll,
                "Predicted_Roll": roll,
                "Session_ID": "2026-06-22__B51__P4__CVO",
            })
        path = root / "reviewed_exception_tracks.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_three_phase_1_2n_labels_are_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            selected = select_material_representatives(make_tracklets())
            reused = load_reused_ground_truth(self._write_prior(Path(temp)), selected)
            self.assertEqual(len(reused), 3)
            self.assertEqual(set(reused["Actual_Roll"]), {"2401100CSE0140"})
            self.assertEqual(EXPECTED_SELECTED_PAIRS - len(reused), EXPECTED_PENDING_REVIEW)

    def test_changed_frozen_identity_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            selected = select_material_representatives(make_tracklets())
            with self.assertRaisesRegex(ProductPhase2HError, "identity changed"):
                load_reused_ground_truth(self._write_prior(Path(temp), altered=True), selected)


class ShadowEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.selected = select_material_representatives(make_tracklets())
        self.combined = pd.DataFrame([
            {
                "Tracklet_ID": row["Tracklet_ID"],
                "Checkpoint_ID": row["Checkpoint_ID"],
                "Predicted_Roll": row["Tracklet_Best_Roll"],
                "Review_Status": "identified",
                "Actual_Roll": row["Tracklet_Best_Roll"],
            }
            for row in self.selected.to_dict("records")
        ])

    def test_all_correct_yields_seven_present_three_review(self) -> None:
        status, unsafe, unverifiable = summarize_reviewed_material(self.combined, self.selected)
        self.assertEqual(unsafe, 0)
        self.assertEqual(unverifiable, 0)
        self.assertEqual(int(status["Shadow_Status"].eq("Present").sum()), 7)
        self.assertEqual(int(status["Shadow_Status"].eq("Needs Review").sum()), 3)
        self.assertEqual(int(status["Shadow_Status"].eq("Unconfirmed").sum()), 0)
        self.assertEqual(
            set(status.loc[status["Shadow_Status"].eq("Present"), "Roll"]),
            {
                "2401100CSE0016", "2401100CSE0019", "2401100CSE0050",
                "2401100CSE0060", "2401100CSE0140", "24011CSEAI0007",
                "24011CSEAI0051",
            },
        )

    def test_wrong_identity_is_unsafe_and_checkpoint_is_not_counted(self) -> None:
        changed = self.combined.copy()
        changed.loc[0, "Actual_Roll"] = "2401100CSE0254"
        status, unsafe, unverifiable = summarize_reviewed_material(changed, self.selected)
        self.assertEqual(unsafe, 1)
        self.assertEqual(unverifiable, 0)
        roll = changed.loc[0, "Predicted_Roll"]
        row = status[status["Roll"].eq(roll)].iloc[0]
        self.assertLess(row["Confirmed_Checkpoint_Count"], row["Predicted_Checkpoint_Count"])

    def test_unidentifiable_is_held_not_counted_as_absent(self) -> None:
        changed = self.combined.copy()
        target_roll = changed.loc[0, "Predicted_Roll"]
        changed.loc[changed["Predicted_Roll"].eq(target_roll), "Review_Status"] = "unidentifiable"
        changed.loc[changed["Predicted_Roll"].eq(target_roll), "Actual_Roll"] = ""
        status, unsafe, unverifiable = summarize_reviewed_material(changed, self.selected)
        self.assertEqual(unsafe, 0)
        self.assertGreater(unverifiable, 0)
        row = status[status["Roll"].eq(target_roll)].iloc[0]
        self.assertEqual(row["Shadow_Status"], "Unconfirmed")


class OutputIntegrityTests(unittest.TestCase):
    def test_verified_output_allows_later_evaluation_subdirectory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "multiframe-recovery-test"
            output.mkdir()
            artifact = output / "artifact.txt"
            artifact.write_text("frozen", encoding="utf-8")
            manifest = {
                "schema_version": 1,
                "policy_version": POLICY_VERSION,
                "output_id": output.name,
                "review_package": "review-package",
                "files": {
                    "artifact.txt": {
                        "sha256": hashlib.sha256(b"frozen").hexdigest(),
                        "size_bytes": 6,
                    }
                },
            }
            (output / "output_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            evaluation = output / "evaluation"
            evaluation.mkdir()
            (evaluation / "evaluation_summary.json").write_text("{}", encoding="utf-8")

            verified = verify_output(output)
            self.assertEqual(verified["output_id"], output.name)

            artifact.write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ProductPhase2HError, "output hash changed"):
                verify_output(output)



class FrozenSourceLocationTests(unittest.TestCase):
    def test_default_inputs_use_shadow_run_frozen_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            inputs = default_inputs(repo)
            expected_root = (
                repo.resolve()
                / "attendance_output" / "shadow_validation" / "phase_1_2n"
                / "shadow_runs" / phase2h.PHASE_1_2N_SHADOW_ID / "frozen_inputs"
            )
            self.assertEqual(inputs.freeze_json, expected_root / "phase_1_2n_mon_p4_source_freeze.json")
            self.assertEqual(inputs.freeze_csv, expected_root / "phase_1_2n_mon_p4_source_freeze.csv")
            self.assertNotIn("source_freeze", inputs.freeze_json.parts)

    def test_relocated_frozen_copy_is_hash_pinned_then_verified_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            inputs = default_inputs(repo)
            inputs.freeze_json.parent.mkdir(parents=True, exist_ok=True)
            inputs.freeze_csv.write_text("RelativePath\n", encoding="utf-8")
            payload = {
                "FreezeID": phase2h.PHASE_1_2N_FREEZE_ID,
                "CSV": r"F:\\historical\\phase_1_2n_mon_p4_source_freeze.csv",
            }
            inputs.freeze_json.write_text(json.dumps(payload), encoding="utf-8")
            json_hash = hashlib.sha256(inputs.freeze_json.read_bytes()).hexdigest()
            csv_hash = hashlib.sha256(inputs.freeze_csv.read_bytes()).hexdigest()

            captured: dict[str, object] = {}

            def fake_verify_source_freeze(**kwargs):
                captured.update(kwargs)
                temp_json = Path(kwargs["freeze_json_path"])
                temp_csv = Path(kwargs["freeze_csv_path"])
                relocated = json.loads(temp_json.read_text(encoding="utf-8"))
                self.assertEqual(Path(relocated["CSV"]), temp_csv.resolve())
                self.assertEqual(temp_csv.read_bytes(), inputs.freeze_csv.read_bytes())
                return object()

            with mock.patch.object(phase2h, "SOURCE_FREEZE_JSON_SHA256", json_hash), \
                 mock.patch.object(phase2h, "SOURCE_FREEZE_CSV_SHA256", csv_hash), \
                 mock.patch.object(phase2h, "verify_source_freeze", side_effect=fake_verify_source_freeze):
                _verify_frozen_source_copy(inputs)

            self.assertEqual(captured["canonical_root"], inputs.canonical_video_root)
            self.assertEqual(captured["mirror_root"], inputs.mirror_video_root)
            self.assertEqual(captured["session_id"], phase2h.SESSION_ID)
            self.assertEqual(json.loads(inputs.freeze_json.read_text(encoding="utf-8")), payload)

    def test_tampered_frozen_copy_fails_before_video_verifier(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            inputs = default_inputs(Path(temp))
            inputs.freeze_json.parent.mkdir(parents=True, exist_ok=True)
            inputs.freeze_json.write_text(
                json.dumps({
                    "FreezeID": phase2h.PHASE_1_2N_FREEZE_ID,
                    "CSV": "phase_1_2n_mon_p4_source_freeze.csv",
                }),
                encoding="utf-8",
            )
            inputs.freeze_csv.write_text("tampered", encoding="utf-8")
            with mock.patch.object(phase2h, "verify_source_freeze") as verifier:
                with self.assertRaisesRegex(ProductPhase2HError, "JSON hash changed"):
                    _verify_frozen_source_copy(inputs)
                verifier.assert_not_called()

if __name__ == "__main__":
    unittest.main()
