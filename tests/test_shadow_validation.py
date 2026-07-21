import hashlib
import json
import shutil
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.face_attendance.shadow_validation import (
    EXPECTED_CANDIDATE_ID,
    ShadowValidationError,
    audit_blind_reviewer,
    build_baseline_command,
    build_shadow_decisions,
    create_reviewer_server,
    discover_session_videos,
    evaluate_shadow_predictions,
    evaluate_shadow_review_package,
    export_shadow_review_package,
    load_frozen_candidate,
    run_shadow_session,
    verify_output_manifest,
    write_output_manifest,
)


FROZEN_PAYLOAD = {
    "schema_version": 1,
    "candidate_id": EXPECTED_CANDIDATE_ID,
    "enabled": False,
    "mode": "shadow_only",
    "production_approved": False,
    "requires_multisession_validation": True,
    "applies_only_to_baseline_rejected_tracklets": True,
    "subject_abbr": "CVO",
    "thresholds": {
        "min_best_score": 0.38,
        "min_margin": 0.03,
        "min_vote_ratio_pct": 50.0,
        "min_observation_count": 3,
        "min_consistency_ratio": 0.5,
        "min_pairwise_similarity_median": 0.45,
        "min_selected_observations": 3,
        "min_consistent_embeddings": 3,
        "require_dominant_agreement": True,
        "require_subject_roster_candidate": True,
    },
    "fixed_safety_guards": {
        "candidate_must_be_in_authoritative_subject_roster": True,
        "aggregate_candidate_must_equal_dominant_frame_candidate": True,
        "minimum_selected_observations": 3,
        "minimum_consistent_embeddings": 3,
    },
    "official_match_threshold_unchanged": 0.48,
    "official_margin_threshold_unchanged": 0.08,
    "attendance_checkpoint_rule_unchanged": 3,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_candidate(root: Path, **overrides):
    payload = json.loads(json.dumps(FROZEN_PAYLOAD))
    payload.update(overrides)
    config = root / "candidate_config.json"
    config.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    manifest = root / "output_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidate_id": EXPECTED_CANDIDATE_ID,
                "files_sha256": {"candidate_config.json": sha256(config)},
                "production_changes": False,
            }
        ),
        encoding="utf-8",
    )
    return config, manifest


def load_test_candidate(config: Path, manifest: Path):
    return load_frozen_candidate(
        config,
        manifest,
        subject_abbr="CVO",
        expected_config_sha256=sha256(config),
    )


def track_row(
    tracklet_id: str,
    *,
    predicted: str = "S1",
    accepted: bool = False,
    eligible: bool = True,
    dominant: str | None = None,
    score: float = 0.42,
    margin: float = 0.06,
    vote: float = 75.0,
    observations: int = 5,
    selected: int = 5,
    embeddings: int = 5,
    consistent: int = 5,
    pairwise: float = 0.62,
    checkpoint: str = "CP1",
    camera: str = "front",
):
    dominant = predicted if dominant is None else dominant
    return {
        "Session_ID": "2026-06-30__B51__P2__CVO",
        "Subject_Abbr": "CVO",
        "Tracklet_Mode": "compare",
        "Tracklet_ID": tracklet_id,
        "Checkpoint_ID": checkpoint,
        "Camera_ID": camera,
        "Video": f"{camera}.mp4",
        "Tracklet_Accepted": "Yes" if accepted else "No",
        "Tracklet_Eligible": "Yes" if eligible else "No",
        "Tracklet_Quality_Rejection": "" if eligible else "insufficient_observations",
        "Tracklet_Diagnostic_Reason": "accepted" if accepted else "score_and_margin_below_threshold",
        "Tracklet_Matcher_Reason": "accepted" if accepted else "score_below_threshold",
        "Tracklet_Best_Roll": predicted,
        "Tracklet_Best_Score": score,
        "Tracklet_Second_Roll": "S2",
        "Tracklet_Second_Score": score - margin,
        "Tracklet_Margin": margin,
        "Dominant_Frame_Best_Roll": dominant,
        "Dominant_Frame_Best_Share_Pct": vote,
        "Observation_Count": observations,
        "Selected_Observation_Count": selected,
        "Embedding_Count": embeddings,
        "Consistent_Embedding_Count": consistent,
        "Inconsistent_Embedding_Count": max(0, embeddings - consistent),
        "Pairwise_Similarity_Median": pairwise,
        "Selected_Observation_IDs": "; ".join(f"{tracklet_id}-o{i}" for i in range(selected)),
        "Zone_IDs": "center",
        "Match_Threshold": 0.48,
        "Margin_Threshold": 0.08,
        "Aggregate_Mode": "top3",
        "Official_Attendance_Contribution": "No",
        "Quality_Weight_Median": 0.55,
        "Face_Width_Median": 44,
    }


def prediction_row(review_id: str, predicted: str, checkpoint: str = "CP1"):
    return {
        "Package_ID": "pkg",
        "Review_ID": review_id,
        "Tracklet_ID": f"private-{review_id}",
        "Session_ID": "2026-06-30__B51__P2__CVO",
        "Subject_Abbr": "CVO",
        "Checkpoint_ID": checkpoint,
        "Camera_ID": "front",
        "Predicted_Roll": predicted,
        "Candidate_ID": EXPECTED_CANDIDATE_ID,
        "Candidate_Config_SHA256": "a" * 64,
        "Evidence_SHA256": "b" * 64,
    }


def label_row(review_id: str, status: str, actual: str = ""):
    return {
        "Package_ID": "pkg",
        "Review_ID": review_id,
        "Review_Status": status,
        "Actual_Roll": actual,
        "Reviewer_Notes": "",
    }


class FrozenCandidateTests(unittest.TestCase):
    def test_candidate_must_remain_disabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config, manifest = write_candidate(Path(temp_dir), enabled=True)
            with self.assertRaisesRegex(ShadowValidationError, "disabled"):
                load_test_candidate(config, manifest)

    def test_production_approved_candidate_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config, manifest = write_candidate(Path(temp_dir), production_approved=True)
            with self.assertRaisesRegex(ShadowValidationError, "production_approved"):
                load_test_candidate(config, manifest)

    def test_candidate_config_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config, manifest = write_candidate(Path(temp_dir))
            config.write_text(config.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ShadowValidationError, "SHA-256"):
                load_test_candidate(config, manifest)

    def test_frozen_candidate_id_is_recomputed_from_thresholds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config, manifest = write_candidate(Path(temp_dir))
            frozen = load_test_candidate(config, manifest)
            self.assertEqual(frozen.candidate.candidate_id, EXPECTED_CANDIDATE_ID)

    def test_candidate_must_match_the_approved_config_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config, manifest = write_candidate(Path(temp_dir))
            with self.assertRaisesRegex(ShadowValidationError, "approved SHA-256"):
                load_frozen_candidate(
                    config,
                    manifest,
                    subject_abbr="CVO",
                    expected_config_sha256="0" * 64,
                )


class ShadowDecisionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        config, manifest = write_candidate(Path(self.temp_dir.name))
        self.candidate = load_test_candidate(config, manifest)

    def tearDown(self):
        self.temp_dir.cleanup()

    def decisions(self, rows, roster=None):
        return build_shadow_decisions(
            pd.DataFrame(rows), self.candidate, set(roster or {"S1", "S2"})
        )

    def test_candidate_applies_only_to_baseline_rejected_tracks(self):
        decisions = self.decisions([track_row("accepted", accepted=True), track_row("rejected")])
        self.assertEqual(decisions["Tracklet_ID"].tolist(), ["rejected"])

    def test_baseline_accepted_track_cannot_be_shadow_accepted(self):
        decisions = self.decisions([track_row("accepted", accepted=True)])
        self.assertTrue(decisions.empty)

    def test_candidate_roll_must_belong_to_authoritative_roster(self):
        decisions = self.decisions([track_row("outside", predicted="OUT")])
        self.assertEqual(decisions.iloc[0]["Guard_Subject_Roster"], "Fail")
        self.assertEqual(decisions.iloc[0]["Shadow_Accepted"], "No")

    def test_dominant_candidate_agreement_is_enforced(self):
        decisions = self.decisions([track_row("disagree", dominant="S2")])
        self.assertEqual(decisions.iloc[0]["Guard_Dominant_Agreement"], "Fail")
        self.assertEqual(decisions.iloc[0]["Shadow_Accepted"], "No")

    def test_minimum_observation_and_consistency_guards_are_enforced(self):
        decisions = self.decisions(
            [track_row("weak", observations=2, selected=2, embeddings=5, consistent=2)]
        )
        row = decisions.iloc[0]
        self.assertEqual(row["Guard_Observation_Count"], "Fail")
        self.assertEqual(row["Guard_Selected_Observations"], "Fail")
        self.assertEqual(row["Guard_Consistent_Embeddings"], "Fail")
        self.assertEqual(row["Guard_Consistency_Ratio"], "Fail")

    def test_feature_schema_mismatch_fails_closed(self):
        row = track_row("missing")
        del row["Pairwise_Similarity_Median"]
        with self.assertRaisesRegex(ShadowValidationError, "schema"):
            self.decisions([row])

    def test_missing_eligible_numeric_feature_fails_closed(self):
        with self.assertRaisesRegex(ShadowValidationError, "Eligible tracklet feature schema"):
            self.decisions([track_row("missing-eligible", pairwise="")])

    def test_blank_pairwise_for_ineligible_track_is_safely_rejected(self):
        decisions = self.decisions(
            [
                track_row(
                    "ineligible-single-frame",
                    eligible=False,
                    observations=1,
                    selected=1,
                    embeddings=1,
                    consistent=1,
                    pairwise="",
                )
            ]
        )
        row = decisions.iloc[0]
        self.assertEqual(row["Guard_Tracklet_Eligible"], "Fail")
        self.assertEqual(row["Guard_Pairwise_Similarity"], "Fail")
        self.assertEqual(row["Shadow_Accepted"], "No")

    def test_shadow_decisions_are_deterministic(self):
        rows = [track_row("b", checkpoint="CP2"), track_row("a", checkpoint="CP1")]
        first = self.decisions(rows)
        second = self.decisions(list(reversed(rows)))
        pd.testing.assert_frame_equal(first, second)

    def test_duplicate_tracklet_recoveries_are_rejected(self):
        with self.assertRaisesRegex(ShadowValidationError, "duplicate"):
            self.decisions([track_row("same"), track_row("same")])


class BlindReviewerTests(unittest.TestCase):
    def _fixture(self, root: Path, rows=None):
        config, manifest = write_candidate(root)
        candidate = load_test_candidate(config, manifest)
        mapping = root / "mapping.json"
        mapping.write_text(
            json.dumps(
                {
                    "subject_students": {
                        "CVO": [
                            {"roll": "S1", "name": "Student One"},
                            {"roll": "S2", "name": "Student Two"},
                        ]
                    },
                    "all_students": [
                        {"roll": "S1", "name": "Student One"},
                        {"roll": "S2", "name": "Student Two"},
                        {"roll": "OUT", "name": "Other Course"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        video = root / "videos" / "CP1_1000" / "front.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"video")
        rows = (
            [track_row("shadow-a"), track_row("shadow-b", predicted="S2")]
            if rows is None
            else rows
        )
        tracklets = pd.DataFrame(rows, columns=track_row("schema").keys())
        decisions = build_shadow_decisions(tracklets, candidate, {"S1", "S2"})
        observations = []
        for row in rows:
            for index in range(3):
                observations.append(
                    {
                        "Tracklet_ID": row["Tracklet_ID"],
                        "Observation_ID": f"{row['Tracklet_ID']}-o{index}",
                        "Checkpoint_ID": "CP1",
                        "Video": "front.mp4",
                        "Frame": index + 1,
                        "BBox_Original_Coordinates": "20,20,30,30",
                        "Selected_For_Aggregation": "Yes",
                        "Quality_Weight": 0.7,
                    }
                )
        package = export_shadow_review_package(
            decisions=decisions,
            observation_df=pd.DataFrame(
                observations,
                columns=[
                    "Tracklet_ID",
                    "Observation_ID",
                    "Checkpoint_ID",
                    "Video",
                    "Frame",
                    "BBox_Original_Coordinates",
                    "Selected_For_Aggregation",
                    "Quality_Weight",
                ],
            ),
            video_root=root / "videos",
            output_root=root / "shadow",
            student_map_path=mapping,
            diagnostic_run_id="p2-run",
            session_id="2026-06-30__B51__P2__CVO",
            frozen_candidate=candidate,
            source_files={"student_map": mapping},
            frame_loader=lambda _path, _frame: np.full((100, 120, 3), 160, dtype=np.uint8),
        )
        shutil.copy2(config, root / "shadow" / "candidate_config_snapshot.json")
        (root / "shadow" / "session_input_manifest.json").write_text(
            json.dumps(
                {
                    "session_id": "2026-06-30__B51__P2__CVO",
                    "subject_abbr": "CVO",
                    "candidate": {
                        "candidate_id": EXPECTED_CANDIDATE_ID,
                        "config_sha256": candidate.config_sha256,
                        "enabled": False,
                        "production_approved": False,
                    },
                    "authoritative_roster": {
                        "path": str(mapping.resolve()),
                        "sha256": sha256(mapping),
                        "student_count": 2,
                        "rolls": ["S1", "S2"],
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        write_output_manifest(
            root / "shadow",
            {
                "candidate_id": EXPECTED_CANDIDATE_ID,
                "candidate_config_sha256": candidate.config_sha256,
                "session_id": "2026-06-30__B51__P2__CVO",
            },
        )
        return package

    def test_reviewer_contains_exact_shadow_accepted_set(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            package = self._fixture(Path(temp_dir))
            private = pd.read_csv(package.private_dir / "hidden_predictions.csv", dtype=str)
            self.assertEqual(set(private["Tracklet_ID"]), {"shadow-a", "shadow-b"})
            self.assertEqual(len(private), 2)

    def test_public_reviewer_has_no_prediction_or_track_id_leak(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            package = self._fixture(Path(temp_dir))
            audit = audit_blind_reviewer(package.root)
            self.assertEqual(audit["forbidden_matches"], [])
            public = (package.reviewer_dir / "index.html").read_text(encoding="utf-8")
            self.assertNotIn("shadow-a", public)
            self.assertNotIn("Predicted_Roll", public)

    def test_private_mapping_retains_predicted_identities(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            package = self._fixture(Path(temp_dir))
            private = pd.read_csv(package.private_dir / "hidden_predictions.csv", dtype=str)
            self.assertEqual(set(private["Predicted_Roll"]), {"S1", "S2"})
            self.assertTrue(private["Candidate_ID"].eq(EXPECTED_CANDIDATE_ID).all())

    def test_reviewer_dropdown_contains_only_authoritative_subject_roster(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            package = self._fixture(Path(temp_dir))
            public = (package.reviewer_dir / "index.html").read_text(encoding="utf-8")
            self.assertIn("Student One", public)
            self.assertIn("Student Two", public)
            self.assertNotIn("Other Course", public)

    def test_exported_labels_validate_package_id_and_hashes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            package = self._fixture(root)
            labels = pd.read_csv(package.reviewer_dir / "labels_template.csv", dtype=str).fillna("")
            labels["Package_ID"] = "wrong-package"
            labels["Review_Status"] = "unidentifiable"
            labels_path = root / "labels.csv"
            labels.to_csv(labels_path, index=False)
            with self.assertRaisesRegex(ShadowValidationError, "Package_ID"):
                evaluate_shadow_review_package(
                    package.root,
                    labels_path,
                    output_dir=root / "evaluation",
                    expected_candidate_config_sha256=sha256(root / "candidate_config.json"),
                    expected_roster_count=2,
                )

    def test_zero_recovery_package_is_valid_and_evaluates_to_hold(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            package = self._fixture(root, rows=[])
            manifest = json.loads(
                (package.reviewer_dir / "review_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["review_items"], [])
            labels = package.reviewer_dir / "labels_template.csv"
            result, _ = evaluate_shadow_review_package(
                package.root,
                labels,
                output_dir=root / "evaluation",
                expected_candidate_config_sha256=sha256(root / "candidate_config.json"),
                expected_roster_count=2,
            )
            self.assertEqual(result.summary["decision"], "hold_no_recoveries")

    def test_evaluator_rejects_a_roster_changed_after_export(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            package = self._fixture(root)
            labels = pd.read_csv(
                package.reviewer_dir / "labels_template.csv", dtype=str
            ).fillna("")
            labels["Review_Status"] = "unidentifiable"
            labels_path = root / "labels.csv"
            labels.to_csv(labels_path, index=False)
            mapping = root / "mapping.json"
            mapping.write_text(
                mapping.read_text(encoding="utf-8") + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ShadowValidationError, "roster changed"):
                evaluate_shadow_review_package(
                    package.root,
                    labels_path,
                    output_dir=root / "evaluation",
                    expected_candidate_config_sha256=sha256(root / "candidate_config.json"),
                    expected_roster_count=2,
                )


class ShadowEvaluationTests(unittest.TestCase):
    def evaluate(self, predictions, labels):
        return evaluate_shadow_predictions(
            pd.DataFrame(predictions),
            pd.DataFrame(labels),
            known_rolls={"S1", "S2", "S3"},
            package_id="pkg",
            candidate_id=EXPECTED_CANDIDATE_ID,
            candidate_config_sha256="a" * 64,
            session_id="2026-06-30__B51__P2__CVO",
        )

    def test_one_wrong_identified_roll_rejects_candidate(self):
        result = self.evaluate([prediction_row("R1", "S1")], [label_row("R1", "identified", "S2")])
        self.assertEqual(result.summary["decision"], "reject_candidate")
        self.assertEqual(result.summary["confirmed_false_identities"], 1)

    def test_not_in_mapping_rejects_candidate(self):
        result = self.evaluate([prediction_row("R1", "S1")], [label_row("R1", "not_in_mapping")])
        self.assertEqual(result.summary["decision"], "reject_candidate")

    def test_mixed_track_rejects_candidate(self):
        result = self.evaluate([prediction_row("R1", "S1")], [label_row("R1", "mixed_track")])
        self.assertEqual(result.summary["decision"], "reject_candidate")

    def test_unidentifiable_holds_unverifiable_sample(self):
        result = self.evaluate([prediction_row("R1", "S1")], [label_row("R1", "unidentifiable")])
        self.assertEqual(result.summary["decision"], "hold_unverifiable_sample")

    def test_zero_recoveries_holds_without_fabricating_success(self):
        predictions = pd.DataFrame(columns=prediction_row("R1", "S1").keys())
        labels = pd.DataFrame(columns=label_row("R1", "").keys())
        result = evaluate_shadow_predictions(
            predictions,
            labels,
            known_rolls={"S1"},
            package_id="pkg",
            candidate_id=EXPECTED_CANDIDATE_ID,
            candidate_config_sha256="a" * 64,
            session_id="2026-06-30__B51__P2__CVO",
        )
        self.assertEqual(result.summary["decision"], "hold_no_recoveries")

    def test_multi_identity_multi_checkpoint_clean_sample_continues_shadow_validation(self):
        predictions = [
            prediction_row("R1", "S1", "CP1"),
            prediction_row("R2", "S2", "CP2"),
            prediction_row("R3", "S1", "CP3"),
        ]
        labels = [
            label_row("R1", "identified", "S1"),
            label_row("R2", "identified", "S2"),
            label_row("R3", "identified", "S1"),
        ]
        result = self.evaluate(predictions, labels)
        self.assertEqual(result.summary["decision"], "continue_multisession_shadow_validation")
        self.assertFalse(result.summary["production_approved"])


class IntegrityAndLauncherTests(unittest.TestCase):
    def test_fresh_runner_does_not_precreate_the_diagnostic_root(self):
        class StopAfterLaunchAssertion(RuntimeError):
            pass

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            candidate_config = root / "candidate_config.json"
            student_map = root / "student_map.json"
            candidate_config.write_text("candidate", encoding="utf-8")
            student_map.write_text("roster", encoding="utf-8")
            preflight = SimpleNamespace(
                session_id="2026-06-30__B51__P2__CVO",
                videos=[{}] * 10,
                student_map_path=student_map,
                subject_rolls={"S1"},
                frozen_candidate=SimpleNamespace(
                    config_path=candidate_config,
                    config_sha256=sha256(candidate_config),
                    candidate=SimpleNamespace(candidate_id=EXPECTED_CANDIDATE_ID),
                ),
            )
            diagnostic_root = root / "attendance_output" / "diagnostics" / "p2-diagnostic"

            def assert_launch_state(*_args, **_kwargs):
                self.assertFalse(diagnostic_root.exists())
                raise StopAfterLaunchAssertion

            with (
                patch(
                    "src.face_attendance.shadow_validation.preflight_shadow_session",
                    return_value=preflight,
                ),
                patch(
                    "src.face_attendance.shadow_validation.make_diagnostic_run_id",
                    side_effect=["p2-diagnostic", "shadow-output"],
                ),
                patch(
                    "src.face_attendance.shadow_validation.build_baseline_command",
                    return_value=["python", "diagnostic"],
                ),
                patch(
                    "src.face_attendance.shadow_validation.subprocess.run",
                    side_effect=assert_launch_state,
                ),
            ):
                with self.assertRaises(StopAfterLaunchAssertion):
                    run_shadow_session(
                        repo_root=root,
                        video_root=root / "videos",
                        session_id=preflight.session_id,
                        subject_abbr="CVO",
                        student_map_path=student_map,
                        candidate_config_path=candidate_config,
                        candidate_manifest_path=root / "candidate_manifest.json",
                        reference_diagnostic_run=root / "reference",
                        timetable_path=root / "timetable.csv",
                        embeddings_path=root / "embeddings.pkl",
                        expected_roster_count=1,
                        status_callback=None,
                    )

    def test_shadow_command_is_exactly_diagnostic_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_root = root / "videos"
            video_root.mkdir()
            camera_zones = root / "camera_zones.json"
            timetable = root / "timetable.csv"
            embeddings = root / "embeddings.pkl"
            for path in (camera_zones, timetable, embeddings):
                path.write_bytes(b"fixture")
            preflight = SimpleNamespace(
                video_root=video_root,
                session_id="2026-06-30__B51__P2__CVO",
                session_fields={"session_date": "2026-06-30"},
                subject_abbr="CVO",
                reference_configuration={
                    "camera_zones": str(camera_zones),
                    "frame_skip": 3,
                    "sample_fps": 2.0,
                    "aggregate": "top3",
                    "checkpoint_mode": "clip-folders",
                    "log_mode": "full",
                    "zone_mode": "compare",
                    "zone_profile": "auto",
                    "zone_merge_iou": 0.2,
                    "tracklet_mode": "compare",
                    "tracklet_min_observations": 2,
                    "tracklet_max_selected": 5,
                    "tracklet_max_gap_seconds": 1.5,
                    "tracklet_min_iou": 0.1,
                    "tracklet_max_center_ratio": 1.25,
                    "tracklet_min_size_ratio": 0.5,
                    "tracklet_min_embedding_similarity": 0.25,
                },
            )
            command = build_baseline_command(
                preflight,
                repo_root=root,
                diagnostic_dir=root / "diagnostics",
                diagnostic_run_id="p2-shadow-run",
                timetable_path=timetable,
                embeddings_path=embeddings,
                python_executable=root / "python.exe",
            )
            self.assertIn("--diagnostic", command)
            self.assertIn("--diagnostic-only", command)
            self.assertNotIn("--output-dir", command)
            self.assertNotIn("--save-unknown", command)
            self.assertEqual(command[command.index("--match-threshold") + 1], "0.48")
            self.assertEqual(command[command.index("--margin-threshold") + 1], "0.08")
            self.assertEqual(command[command.index("--present-checkpoints") + 1], "3")
            self.assertEqual(command[command.index("--tracklet-mode") + 1], "compare")

    def test_output_manifest_hashes_are_verified(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact = root / "artifact.txt"
            artifact.write_text("stable", encoding="utf-8")
            manifest = write_output_manifest(root, {"candidate_id": EXPECTED_CANDIDATE_ID})
            verified = verify_output_manifest(root, manifest)
            self.assertEqual(verified["verified_files"], 1)
            artifact.write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ShadowValidationError, "integrity"):
                verify_output_manifest(root, manifest)

    def test_session_video_discovery_requires_exact_five_by_two_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for number in range(1, 6):
                checkpoint = root / f"CP{number}_1000"
                checkpoint.mkdir()
                for camera in ("front", "back"):
                    (checkpoint / f"{camera}.mp4").write_bytes(b"video")
            metadata = lambda _path: {
                "fps": 25.0,
                "frame_count": 100,
                "duration_seconds": 4.0,
                "width": 640,
                "height": 480,
            }
            videos = discover_session_videos(root, metadata_reader=metadata)
            self.assertEqual(len(videos), 10)
            (root / "CP5_1000" / "front.mp4").unlink()
            with self.assertRaisesRegex(ShadowValidationError, "front.mp4"):
                discover_session_videos(root, metadata_reader=metadata)

    def test_long_path_local_server_binds_loopback_and_never_serves_private_sibling(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index in range(12):
                root = root / f"long_review_segment_{index:02d}"
            reviewer = root / "reviewer"
            private = root / "private"
            reviewer.mkdir(parents=True)
            private.mkdir()
            (reviewer / "index.html").write_text("shadow reviewer", encoding="utf-8")
            (private / "secret.txt").write_text("private prediction", encoding="utf-8")
            server, url = create_reviewer_server(root, port=0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                self.assertEqual(server.server_address[0], "127.0.0.1")
                self.assertIn("shadow reviewer", urllib.request.urlopen(url, timeout=5).read().decode())
                with self.assertRaises(urllib.error.HTTPError):
                    urllib.request.urlopen(url + "../private/secret.txt", timeout=5)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
