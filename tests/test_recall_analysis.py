import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.face_attendance.recall_analysis import (
    UnresolvedSelectionConfig,
    build_embedding_roster_coverage,
    build_student_recall_gap,
    evaluate_unresolved_predictions,
    evaluate_unresolved_review_package,
    select_unresolved_tracks,
)
from src.face_attendance.tracklet_review import export_review_package


def track(
    tracklet_id,
    candidate,
    checkpoint,
    camera,
    *,
    score=0.45,
    margin=0.06,
    vote=75.0,
    accepted="No",
    eligible="Yes",
    selected_ids=None,
    zone="middle",
    reason="score_and_margin_below_threshold",
    quality=0.55,
):
    selected_ids = selected_ids or [f"{tracklet_id}-a", f"{tracklet_id}-b"]
    return {
        "Tracklet_ID": tracklet_id,
        "Tracklet_Accepted": accepted,
        "Tracklet_Eligible": eligible,
        "Tracklet_Best_Roll": candidate,
        "Tracklet_Second_Roll": "OTHER",
        "Tracklet_Best_Score": score,
        "Tracklet_Second_Score": max(0.0, score - margin),
        "Tracklet_Margin": margin,
        "Dominant_Frame_Best_Roll": candidate,
        "Dominant_Frame_Best_Share_Pct": vote,
        "Checkpoint_ID": checkpoint,
        "Camera_ID": camera,
        "Video": f"{camera}.mp4",
        "Zone_IDs": zone,
        "Tracklet_Diagnostic_Reason": reason,
        "Tracklet_Matcher_Reason": "score_below_threshold",
        "Tracklet_Quality_Rejection": "" if eligible == "Yes" else "insufficient_consistent_embeddings",
        "Observation_Count": 5,
        "Embedding_Count": 5,
        "Selected_Observation_Count": len(selected_ids),
        "Consistent_Embedding_Count": len(selected_ids),
        "Inconsistent_Embedding_Count": 0,
        "Selected_Observation_IDs": "; ".join(selected_ids),
        "Usable_Observation_Count": 2,
        "Marginal_Observation_Count": 3,
        "Unusable_Observation_Count": 0,
        "Quality_Weight_Median": quality,
        "Face_Width_Median": 48,
        "Pairwise_Similarity_Median": 0.62,
        "Match_Threshold": 0.48,
        "Margin_Threshold": 0.08,
        "Aggregate_Mode": "top3",
        "Session_ID": "session-1",
        "Subject_Abbr": "CVO",
    }


class EmbeddingCoverageTests(unittest.TestCase):
    def test_annotated_keys_are_canonicalized_and_missing_and_collisions_are_explicit(self):
        roster = [
            {"roll": "A001", "name": "Alpha"},
            {"roll": "B002", "name": "Beta"},
            {"roll": "C003", "name": "Gamma"},
        ]
        summary = pd.DataFrame([
            {"Roll_Number": "A001 (A) Alpha", "Faces_Used": 3},
            {"Roll_Number": "B002", "Faces_Used": 2},
            {"Roll_Number": "B002 (B) Beta", "Faces_Used": 1},
        ])

        coverage = build_embedding_roster_coverage(
            roster,
            summary,
            embedding_counts={"A001 (A) Alpha": 3, "B002": 2, "B002 (B) Beta": 1},
        ).set_index("Canonical_Roster_Roll")

        self.assertEqual(coverage.loc["A001", "Canonical_Mapping_Status"], "canonicalized_annotation")
        self.assertEqual(coverage.loc["A001", "Embedding_Record_Count"], 3)
        self.assertEqual(coverage.loc["B002", "Collision_Status"], "collision")
        self.assertEqual(coverage.loc["C003", "Missing_Status"], "missing")
        self.assertEqual(coverage.loc["C003", "Embedding_Available"], "No")


class RecallGapCategorizationTests(unittest.TestCase):
    def test_each_present_student_gets_a_deterministic_primary_category(self):
        roster = [
            {"roll": roll, "name": roll, "in_subject_roster": True}
            for roll in ["A", "B", "C", "D", "E"]
        ]
        coverage = pd.DataFrame([
            {
                "Canonical_Roster_Roll": roll,
                "Matching_Embedding_Keys": roll if roll != "B" else "",
                "Embedding_Available": "No" if roll == "B" else "Yes",
                "Canonical_Mapping_Status": "missing" if roll == "B" else "exact",
                "Collision_Status": "none",
            }
            for roll in ["A", "B", "C", "D", "E"]
        ])
        accepted = pd.DataFrame([
            {"Actual_Roll": "A", "Checkpoint_ID": cp, "Camera_ID": "front"}
            for cp in ["CP1", "CP2", "CP3"]
        ] + [{"Actual_Roll": "C", "Checkpoint_ID": "CP1", "Camera_ID": "back"}])
        tracks = pd.DataFrame([
            track("c-near", "C", "CP2", "back", score=0.46, margin=0.10),
            track("d-margin", "D", "CP1", "front", score=0.52, margin=0.04, reason="margin_below_threshold"),
            track("other", "OTHER", "CP1", "front", score=0.40),
        ])

        result = build_student_recall_gap(
            roster=roster,
            actual_present_rolls={"A", "B", "C", "D", "E"},
            embedding_coverage_df=coverage,
            tracklet_df=tracks,
            accepted_label_evidence_df=accepted,
        ).set_index("Canonical_Roll")

        self.assertEqual(result.loc["A", "False_Negative_Category"], "present_confirmed")
        self.assertEqual(result.loc["B", "False_Negative_Category"], "embedding_missing")
        self.assertEqual(result.loc["C", "False_Negative_Category"], "insufficient_checkpoint_coverage")
        self.assertEqual(result.loc["D", "False_Negative_Category"], "correct_candidate_margin_below_threshold")
        self.assertEqual(result.loc["E", "False_Negative_Category"], "no_candidate_evidence")
        self.assertEqual(len(result), 5)


class UnresolvedSelectionTests(unittest.TestCase):
    def test_ranking_balances_candidates_checkpoints_and_cameras_and_excludes_noise_duplicates(self):
        rows = []
        for index in range(18):
            rows.append(track(
                f"t{index:02d}",
                f"S{index % 3}",
                f"CP{index % 3 + 1}",
                "front" if index % 2 == 0 else "back",
                score=0.43 + (index % 5) * 0.012,
                margin=0.04 + (index % 4) * 0.015,
                zone=f"z{index % 4}",
            ))
        rows.append(track("duplicate", "S0", "CP1", "front", selected_ids=["t00-a", "t00-b"]))
        noise = track("noise", "S9", "CP1", "front", selected_ids=["noise-only"])
        noise.update({"Observation_Count": 1, "Embedding_Count": 1, "Selected_Observation_Count": 1})
        rows.append(noise)

        selected = select_unresolved_tracks(
            pd.DataFrame(rows),
            false_negative_rolls={"S0", "S1", "S2"},
            config=UnresolvedSelectionConfig(
                target_tracks=9,
                candidate_cap=3,
                checkpoint_cap=4,
                camera_cap=6,
                zone_cap=4,
            ),
        )

        self.assertEqual(len(selected), 9)
        self.assertLessEqual(selected["Candidate_Canonical_Roll"].value_counts().max(), 3)
        self.assertLessEqual(selected["Checkpoint_ID"].value_counts().max(), 4)
        self.assertLessEqual(selected["Camera_ID"].value_counts().max(), 6)
        self.assertEqual(selected["Evidence_Fingerprint"].nunique(), len(selected))
        self.assertNotIn("noise", set(selected["Tracklet_ID"]))
        self.assertEqual(set(selected["Camera_ID"]), {"front", "back"})
        self.assertEqual(set(selected["Checkpoint_ID"]), {"CP1", "CP2", "CP3"})


class UnresolvedBlindPackageTests(unittest.TestCase):
    def test_unresolved_export_keeps_model_data_private_and_includes_context_and_duplicate_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_path = root / "videos" / "CP1_0910" / "front.mp4"
            video_path.parent.mkdir(parents=True)
            video_path.write_bytes(b"placeholder")
            mapping = root / "mapping.json"
            mapping.write_text(json.dumps({
                "subject_students": {"CVO": [{"roll": "S1", "name": "Student One"}]},
                "all_students": [{"roll": "S1", "name": "Student One"}],
            }), encoding="utf-8")
            tracks = pd.DataFrame([track("private-unresolved", "SECRET", "CP1", "front")])
            observations = pd.DataFrame([
                {
                    "Tracklet_ID": "private-unresolved",
                    "Observation_ID": observation_id,
                    "Checkpoint_ID": "CP1",
                    "Video": "front.mp4",
                    "Frame": frame_index,
                    "BBox_Original_Coordinates": "20,20,30,30",
                    "Quality_Weight": 0.7,
                    "Selected_For_Aggregation": "Yes",
                }
                for frame_index, observation_id in [(1, "private-unresolved-a"), (2, "private-unresolved-b")]
            ])

            result = export_review_package(
                tracklet_df=tracks,
                observation_df=observations,
                video_root=root / "videos",
                output_root=root / "output",
                student_map_path=mapping,
                subject_abbr="CVO",
                diagnostic_run_id="run-1",
                package_id="unresolved-test",
                selected_tracklet_ids=["private-unresolved"],
                review_kind="unresolved",
                include_context=True,
                hidden_extra_columns={"Rejection_Reason": "Tracklet_Diagnostic_Reason"},
                frame_loader=lambda _path, _index: np.full((100, 120, 3), 170, dtype=np.uint8),
            )

            public_text = "\n".join(
                path.read_text(encoding="utf-8", errors="ignore")
                for path in result.reviewer_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in {".html", ".json", ".csv", ".txt"}
            )
            self.assertNotIn("SECRET", public_text)
            self.assertNotIn("private-unresolved", public_text)
            self.assertNotIn("Rejection_Reason", public_text)
            self.assertIn('value="duplicate"', public_text)
            self.assertIn("Context", public_text)
            private_text = (result.private_dir / "hidden_predictions.csv").read_text(encoding="utf-8")
            self.assertIn("SECRET", private_text)
            self.assertIn("score_and_margin_below_threshold", private_text)
            manifest = json.loads((result.reviewer_dir / "review_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["review_items"][0]["evidence_views"], 3)
            self.assertEqual(manifest["review_tracklets"], 1)
            self.assertEqual(manifest["accepted_tracklets"], 0)
            self.assertEqual(result.accepted_tracklets, 0)


class UnresolvedEvaluationTests(unittest.TestCase):
    def test_correct_second_neither_and_threshold_opportunities_are_reported_with_partial_review(self):
        predictions = pd.DataFrame([
            {"Package_ID": "pkg", "Review_ID": "R1", "Predicted_Roll": "A", "Second_Roll": "X", "Best_Score": 0.45, "Second_Score": 0.30, "Margin": 0.15, "Vote_Ratio_Pct": 80, "Checkpoint_ID": "CP1", "Camera_ID": "front", "Zone_IDs": "z1", "Rejection_Reason": "score_below_threshold", "Face_Width_Median": 50, "Quality_Weight_Median": 0.6},
            {"Package_ID": "pkg", "Review_ID": "R2", "Predicted_Roll": "A", "Second_Roll": "B", "Best_Score": 0.46, "Second_Score": 0.44, "Margin": 0.02, "Vote_Ratio_Pct": 70, "Checkpoint_ID": "CP2", "Camera_ID": "back", "Zone_IDs": "z2", "Rejection_Reason": "score_and_margin_below_threshold", "Face_Width_Median": 42, "Quality_Weight_Median": 0.5},
            {"Package_ID": "pkg", "Review_ID": "R3", "Predicted_Roll": "A", "Second_Roll": "B", "Best_Score": 0.50, "Second_Score": 0.45, "Margin": 0.05, "Vote_Ratio_Pct": 85, "Checkpoint_ID": "CP3", "Camera_ID": "front", "Zone_IDs": "z1", "Rejection_Reason": "margin_below_threshold", "Face_Width_Median": 38, "Quality_Weight_Median": 0.45},
            {"Package_ID": "pkg", "Review_ID": "R4", "Predicted_Roll": "D", "Second_Roll": "A", "Best_Score": 0.43, "Second_Score": 0.40, "Margin": 0.03, "Vote_Ratio_Pct": 55, "Checkpoint_ID": "CP4", "Camera_ID": "back", "Zone_IDs": "z3", "Rejection_Reason": "score_below_threshold", "Face_Width_Median": 28, "Quality_Weight_Median": 0.3},
        ])
        labels = pd.DataFrame([
            {"Package_ID": "pkg", "Review_ID": "R1", "Review_Status": "identified", "Actual_Roll": "A", "Reviewer_Notes": ""},
            {"Package_ID": "pkg", "Review_ID": "R2", "Review_Status": "identified", "Actual_Roll": "B", "Reviewer_Notes": ""},
            {"Package_ID": "pkg", "Review_ID": "R3", "Review_Status": "identified", "Actual_Roll": "C", "Reviewer_Notes": ""},
            {"Package_ID": "pkg", "Review_ID": "R4", "Review_Status": "", "Actual_Roll": "", "Reviewer_Notes": ""},
        ])

        result = evaluate_unresolved_predictions(
            prediction_df=predictions,
            label_df=labels,
            known_rolls={"A", "B", "C", "D"},
            false_negative_rolls={"A", "B", "C"},
            accepted_label_evidence_df=pd.DataFrame(columns=["Actual_Roll", "Checkpoint_ID", "Camera_ID"]),
        )

        candidate = result.summary["correct_candidate_analysis"]
        self.assertEqual(candidate["best_candidate_correct"], 1)
        self.assertEqual(candidate["second_candidate_correct"], 1)
        self.assertEqual(candidate["neither_candidate_correct"], 1)
        self.assertEqual(result.summary["threshold_opportunity"]["correct_candidate_below_match_threshold"], 2)
        self.assertEqual(result.summary["review"]["reviewed_tracks"], 3)
        self.assertEqual(result.summary["review"]["unreviewed_tracks"], 1)
        self.assertEqual(result.summary["recommendation"], "pending_human_labels")
        self.assertEqual(len(result.student_metrics), 3)

        completed_labels = labels.copy()
        completed_labels.loc[completed_labels["Review_ID"].eq("R4"), "Review_Status"] = "unidentifiable"
        completed = evaluate_unresolved_predictions(
            prediction_df=predictions,
            label_df=completed_labels,
            known_rolls={"A", "B", "C", "D"},
            false_negative_rolls={"A", "B", "C", "D"},
            accepted_label_evidence_df=pd.DataFrame(columns=["Actual_Roll", "Checkpoint_ID", "Camera_ID"]),
        )
        student_metrics = completed.student_metrics.set_index("Canonical_Roll")
        self.assertEqual(student_metrics.loc["D", "No_Visual_Evidence"], "Not_Established_Bounded_Sample")


class AuthoritativeRosterEvaluationTests(unittest.TestCase):
    def test_corrected_label_can_use_student_added_after_historical_package_export(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            diagnostic_run = root / "diagnostic_run"
            recall_analysis = diagnostic_run / "recall_analysis"
            recall_analysis.mkdir(parents=True)

            pd.DataFrame([
                track("siri-track", "SIRI", "CP3", "cam5", score=0.44, margin=0.05),
            ]).to_csv(diagnostic_run / "tracklet_diagnostics_session.csv", index=False)

            student_map = root / "student_map.json"
            student_map.write_text(json.dumps({
                "subject_students": {
                    "CVO": [
                        {"roll": "A001", "name": "Alpha"},
                        {"roll": "SIRI", "name": "Siri"},
                    ]
                },
                "all_students": [
                    {"roll": "A001", "name": "Alpha"},
                    {"roll": "SIRI", "name": "Siri"},
                ],
            }), encoding="utf-8")

            actual_present = root / "actual_present.txt"
            actual_present.write_text("A001\nSIRI\n", encoding="utf-8")
            embedding_summary = root / "embedding_summary.csv"
            pd.DataFrame([
                {"Roll_Number": "A001", "Faces_Used": 2},
                {"Roll_Number": "SIRI", "Faces_Used": 2},
            ]).to_csv(embedding_summary, index=False)
            embeddings = root / "student_embeddings.pkl"
            embeddings.write_bytes(b"placeholder")

            predictions = pd.DataFrame([{
                "Package_ID": "historical-package",
                "Review_ID": "R1",
                "Predicted_Roll": "A001",
                "Second_Roll": "SIRI",
                "Best_Score": 0.44,
                "Second_Score": 0.42,
                "Margin": 0.02,
                "Vote_Ratio_Pct": 70,
                "Checkpoint_ID": "CP3",
                "Camera_ID": "cam5",
                "Zone_IDs": "middle",
                "Rejection_Reason": "score_and_margin_below_threshold",
                "Face_Width_Median": 42,
                "Quality_Weight_Median": 0.5,
            }])
            metadata = {
                "package_id": "historical-package",
                "review_kind": "unresolved",
                "known_rolls": ["A001"],
            }
            manifest = {"review_kind": "unresolved"}
            labels = root / "corrected_labels.csv"
            pd.DataFrame([{
                "Package_ID": "historical-package",
                "Review_ID": "R1",
                "Review_Status": "identified",
                "Actual_Roll": "SIRI",
                "Reviewer_Notes": "added after roster repair",
            }]).to_csv(labels, index=False)

            output_dir = root / "evaluation"
            with (
                patch(
                    "src.face_attendance.recall_analysis.load_review_package_data",
                    return_value=(predictions, metadata, manifest),
                ),
                patch(
                    "src.face_attendance.recall_analysis.load_accepted_label_evidence",
                    return_value=(
                        pd.DataFrame(columns=["Actual_Roll", "Checkpoint_ID", "Camera_ID"]),
                        {},
                    ),
                ),
                patch(
                    "src.face_attendance.recall_analysis.load_embedding_counts",
                    return_value={"A001": 2, "SIRI": 2},
                ),
            ):
                result, created = evaluate_unresolved_review_package(
                    review_package=root / "historical_review",
                    labels_path=labels,
                    recall_analysis_dir=recall_analysis,
                    accepted_review_package=root / "accepted_review",
                    accepted_labels=root / "accepted_labels.csv",
                    output_dir=output_dir,
                    student_map_path=student_map,
                    subject_abbr="CVO",
                    actual_present_path=actual_present,
                    embedding_summary_path=embedding_summary,
                    embeddings_path=embeddings,
                )

            self.assertEqual(created, output_dir)
            self.assertEqual(result.summary["review"]["identifiable"], 1)
            context = result.summary["evaluation_context"]
            self.assertEqual(context["mode"], "authoritative_current_roster")
            self.assertEqual(context["added_since_package_export"], ["SIRI"])
            self.assertEqual(context["roster_students"], 2)
            self.assertTrue((output_dir / "evaluation_student_recall_gap.csv").is_file())
            self.assertTrue((output_dir / "evaluation_embedding_roster_coverage.csv").is_file())
            self.assertTrue((output_dir / "evaluation_context.json").is_file())


if __name__ == "__main__":
    unittest.main()
