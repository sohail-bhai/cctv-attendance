import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.face_attendance.tracklet_calibration import (
    CalibrationCandidate,
    assign_group_folds,
    candidate_acceptance_mask,
    run_tracklet_calibration,
    search_candidates,
)


def calibration_row(
    row_id: str,
    *,
    actual_roll: str,
    predicted_roll: str,
    review_status: str = "identified",
    score: float = 0.42,
    margin: float = 0.06,
    vote: float = 75.0,
    observations: int = 5,
    selected: int = 5,
    embeddings: int = 5,
    consistent: int = 5,
    pairwise: float = 0.62,
    dominant_agreement: bool = True,
    candidate_in_roster: bool = True,
):
    top1_correct = review_status == "identified" and actual_roll == predicted_roll
    return {
        "Benchmark_Row_ID": row_id,
        "Source_Kind": "unresolved",
        "Review_ID": row_id,
        "Tracklet_ID": f"track-{row_id}",
        "Review_Status": review_status,
        "Actual_Roll": actual_roll,
        "Predicted_Roll": predicted_roll,
        "Top1_Correct": top1_correct,
        "Calibration_Eligible": True,
        "Candidate_In_Subject_Roster": candidate_in_roster,
        "Dominant_Agreement": dominant_agreement,
        "Best_Score_Value": score,
        "Margin_Value": margin,
        "Vote_Ratio_Pct_Value": vote,
        "Observation_Count_Value": observations,
        "Selected_Observation_Count_Value": selected,
        "Embedding_Count_Value": embeddings,
        "Consistent_Embedding_Count_Value": consistent,
        "Consistency_Ratio": consistent / embeddings if embeddings else 0.0,
        "Pairwise_Similarity_Median_Value": pairwise,
        "Camera_ID": "cam5",
        "Checkpoint_ID": "CP3",
    }


class CandidateSearchTests(unittest.TestCase):
    def test_search_prioritizes_zero_false_accepts_before_extra_recovery(self):
        frame = pd.DataFrame(
            [
                calibration_row("P1", actual_roll="A", predicted_roll="A", score=0.43),
                calibration_row("P2", actual_roll="B", predicted_roll="B", score=0.39),
                calibration_row(
                    "N1",
                    actual_roll="C",
                    predicted_roll="A",
                    score=0.41,
                ),
            ]
        )

        candidate, search = search_candidates(frame)
        best = search.iloc[0]

        self.assertEqual(int(best["false_accepts"]), 0)
        self.assertEqual(int(best["true_recoveries"]), 1)
        self.assertGreaterEqual(candidate.min_best_score, 0.42)

    def test_fixed_safety_guards_block_out_of_roster_and_disagreement_rows(self):
        frame = pd.DataFrame(
            [
                calibration_row("GOOD", actual_roll="A", predicted_roll="A"),
                calibration_row(
                    "OUTSIDE",
                    actual_roll="B",
                    predicted_roll="B",
                    candidate_in_roster=False,
                ),
                calibration_row(
                    "DISAGREE",
                    actual_roll="C",
                    predicted_roll="C",
                    dominant_agreement=False,
                ),
            ]
        )
        candidate = CalibrationCandidate(
            min_best_score=0.38,
            min_margin=0.03,
            min_vote_ratio_pct=50.0,
            min_observation_count=3,
            min_consistency_ratio=0.5,
            min_pairwise_similarity_median=0.45,
        )

        accepted = candidate_acceptance_mask(frame, candidate)

        self.assertEqual(frame.loc[accepted, "Benchmark_Row_ID"].tolist(), ["GOOD"])


class GroupedHoldoutTests(unittest.TestCase):
    def test_identity_groups_never_cross_folds_and_every_fold_has_both_classes(self):
        rows = []
        for index in range(7):
            rows.append(
                calibration_row(
                    f"P{index}-1",
                    actual_roll=f"S{index}",
                    predicted_roll=f"S{index}",
                )
            )
            rows.append(
                calibration_row(
                    f"P{index}-2",
                    actual_roll=f"S{index}",
                    predicted_roll=f"S{index}",
                )
            )
        for index in range(8):
            rows.append(
                calibration_row(
                    f"N{index}",
                    actual_roll="",
                    predicted_roll="S0",
                    review_status="not_in_mapping",
                    score=0.34,
                )
            )
        frame = pd.DataFrame(rows)

        folded = assign_group_folds(frame, folds=5)

        identified = folded[folded["Review_Status"].eq("identified")]
        self.assertTrue((identified.groupby("Actual_Roll")["Fold"].nunique() == 1).all())
        summary = folded.groupby("Fold")["Top1_Correct"].agg(["count", "sum"])
        self.assertEqual(len(summary), 5)
        self.assertTrue((summary["sum"] > 0).all())
        self.assertTrue(((summary["count"] - summary["sum"]) > 0).all())


class CalibrationIntegrationTests(unittest.TestCase):
    def _write_package(self, root: Path, package_id: str, rows: list[dict], labels: list[dict]):
        package = root / package_id
        (package / "private").mkdir(parents=True)
        (package / "reviewer").mkdir(parents=True)
        pd.DataFrame(rows).to_csv(package / "private" / "hidden_predictions.csv", index=False)
        (package / "private" / "export_metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "package_id": package_id,
                    "session_id": "session-1",
                    "subject_abbr": "CVO",
                }
            ),
            encoding="utf-8",
        )
        (package / "reviewer" / "review_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "package_id": package_id,
                    "review_items": [{"review_id": row["Review_ID"]} for row in rows],
                }
            ),
            encoding="utf-8",
        )
        labels_path = root / f"{package_id}_labels.csv"
        pd.DataFrame(labels).to_csv(labels_path, index=False)
        return package, labels_path

    def test_run_writes_disabled_candidate_and_identity_grouped_zero_false_holdout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            diagnostic = root / "diagnostic"
            diagnostic.mkdir()
            mapping = root / "student_map.json"
            rolls = [f"S{index}" for index in range(6)]
            mapping.write_text(
                json.dumps(
                    {
                        "subject_students": {
                            "CVO": [{"roll": roll, "name": roll} for roll in rolls]
                        },
                        "all_students": [
                            {"roll": roll, "name": roll, "subjects": ["CVO"]} for roll in rolls
                        ],
                    }
                ),
                encoding="utf-8",
            )

            accepted_predictions = []
            accepted_labels = []
            unresolved_predictions = []
            unresolved_labels = []
            diagnostic_rows = []

            def hidden(package_id, review_id, tracklet_id, predicted, score, margin):
                return {
                    "Package_ID": package_id,
                    "Review_ID": review_id,
                    "Tracklet_ID": tracklet_id,
                    "Session_ID": "session-1",
                    "Subject_Abbr": "CVO",
                    "Checkpoint_ID": "CP3",
                    "Camera_ID": "cam5",
                    "Video": "back.mp4",
                    "Predicted_Identity_Raw": predicted,
                    "Predicted_Roll": predicted,
                    "Best_Score": score,
                    "Second_Identity_Raw": "S5",
                    "Second_Roll": "S5",
                    "Second_Score": score - margin,
                    "Margin": margin,
                    "Match_Threshold": 0.48,
                    "Margin_Threshold": 0.08,
                    "Aggregate_Mode": "top3",
                    "Observation_Count": 5,
                    "Selected_Observation_Count": 5,
                    "Selected_Observation_IDs": "a; b; c; d; e",
                    "Evidence_SHA256": "a" * 64,
                }

            def diagnostic_row(tracklet_id, predicted, score, margin, accepted):
                return {
                    "Tracklet_ID": tracklet_id,
                    "Tracklet_Accepted": "Yes" if accepted else "No",
                    "Tracklet_Eligible": "Yes",
                    "Tracklet_Best_Roll": predicted,
                    "Tracklet_Best_Score": score,
                    "Tracklet_Second_Roll": "S5",
                    "Tracklet_Second_Score": score - margin,
                    "Tracklet_Margin": margin,
                    "Dominant_Frame_Best_Roll": predicted,
                    "Dominant_Frame_Best_Share_Pct": 80.0,
                    "Observation_Count": 5,
                    "Selected_Observation_Count": 5,
                    "Embedding_Count": 5,
                    "Consistent_Embedding_Count": 5,
                    "Inconsistent_Embedding_Count": 0,
                    "Pairwise_Similarity_Min": 0.50,
                    "Pairwise_Similarity_Median": 0.65,
                    "Quality_Weight_Median": 0.55,
                    "Face_Width_Median": 45.0,
                    "Tracklet_Duration_Seconds": 2.0,
                    "Checkpoint_ID": "CP3",
                    "Camera_ID": "cam5",
                    "Video": "back.mp4",
                    "Quality_Band": "medium",
                    "Zone_IDs": "back_middle",
                }

            accepted_package_id = "accepted-package"
            for index in range(3):
                roll = rolls[index]
                review_id = f"R{index + 1:04d}"
                tracklet_id = f"accepted-{index}"
                accepted_predictions.append(
                    hidden(accepted_package_id, review_id, tracklet_id, roll, 0.55, 0.12)
                )
                accepted_labels.append(
                    {
                        "Package_ID": accepted_package_id,
                        "Review_ID": review_id,
                        "Review_Status": "identified",
                        "Actual_Roll": roll,
                        "Reviewer_Notes": "",
                    }
                )
                diagnostic_rows.append(diagnostic_row(tracklet_id, roll, 0.55, 0.12, True))

            unresolved_package_id = "unresolved-package"
            for index in range(6):
                roll = rolls[index]
                review_id = f"R{index + 1:04d}"
                tracklet_id = f"positive-{index}"
                unresolved_predictions.append(
                    hidden(unresolved_package_id, review_id, tracklet_id, roll, 0.42, 0.06)
                )
                unresolved_labels.append(
                    {
                        "Package_ID": unresolved_package_id,
                        "Review_ID": review_id,
                        "Review_Status": "identified",
                        "Actual_Roll": roll,
                        "Reviewer_Notes": "",
                    }
                )
                diagnostic_rows.append(diagnostic_row(tracklet_id, roll, 0.42, 0.06, False))
            for index in range(6):
                review_id = f"R{index + 7:04d}"
                tracklet_id = f"negative-{index}"
                unresolved_predictions.append(
                    hidden(unresolved_package_id, review_id, tracklet_id, "S0", 0.34, 0.06)
                )
                unresolved_labels.append(
                    {
                        "Package_ID": unresolved_package_id,
                        "Review_ID": review_id,
                        "Review_Status": "not_in_mapping",
                        "Actual_Roll": "",
                        "Reviewer_Notes": "",
                    }
                )
                diagnostic_rows.append(diagnostic_row(tracklet_id, "S0", 0.34, 0.06, False))

            accepted_package, accepted_labels_path = self._write_package(
                root, accepted_package_id, accepted_predictions, accepted_labels
            )
            unresolved_package, unresolved_labels_path = self._write_package(
                root, unresolved_package_id, unresolved_predictions, unresolved_labels
            )
            pd.DataFrame(diagnostic_rows).to_csv(
                diagnostic / "tracklet_diagnostics_session.csv", index=False
            )
            output = root / "output"

            result = run_tracklet_calibration(
                diagnostic_run=diagnostic,
                accepted_review_package=accepted_package,
                accepted_labels_path=accepted_labels_path,
                unresolved_review_package=unresolved_package,
                unresolved_labels_path=unresolved_labels_path,
                student_map_path=mapping,
                subject_abbr="CVO",
                output_dir=output,
                folds=3,
                min_oof_recoveries=3,
                min_recovery_folds=3,
            )

            self.assertEqual(result.summary["out_of_fold"]["false_accepts"], 0)
            self.assertEqual(result.summary["out_of_fold"]["true_recoveries"], 6)
            self.assertEqual(
                result.summary["recommendation"]["decision"],
                "promote_to_multisession_shadow_validation",
            )
            config = json.loads((output / "candidate_config.json").read_text(encoding="utf-8"))
            self.assertFalse(config["enabled"])
            self.assertFalse(config["production_approved"])
            self.assertTrue((output / "ground_truth_manifest.json").is_file())
            self.assertTrue((output / "false_accept_audit.csv").is_file())
            self.assertTrue((output / "output_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
