from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.face_attendance.product_phase_2k_tracklet_purity import (
    MIXED_TRACKLET_ID,
    analyze_preflight,
    build_shadow_files,
    default_inputs,
    preflight,
)
from src.face_attendance.tracklet_purity import (
    AMBIGUOUS_QUARANTINED,
    APPEARANCE_DISCONTINUITY_ONLY,
    INVALID_EMBEDDING_EVIDENCE,
    MIXED_QUARANTINED,
    PURE_CHILD_CANDIDATE,
    PURE_UNSPLIT,
    SPLIT_CANDIDATE,
    PurityObservation,
    TrackletPurityError,
    analyze_track_purity,
    canonical_relative_path,
    review_carry_forward_decision,
    verify_immutable_output,
    write_immutable_output,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FINGERPRINT = "1" * 64
EMBEDDING_FINGERPRINT = "2" * 64


def _observation(
    index: int,
    *,
    roll: str = "2401100CSE0001",
    embedding: tuple[float, ...] | None = (1.0, 0.0),
    bbox: tuple[float, float, float, float] | None = None,
    timestamp: float | None = None,
) -> PurityObservation:
    return PurityObservation(
        observation_id=f"OBS-{index:03d}",
        frame_index=index * 10,
        timestamp_seconds=index * 0.4 if timestamp is None else timestamp,
        bbox=bbox or (100.0 + index, 80.0, 40.0, 50.0),
        predicted_roll=roll,
        best_score=0.60,
        margin=0.20,
        accepted=True,
        selected=index < 5,
        quality_label="usable_reference",
        embedding_extraction_success=embedding is not None,
        embedding_consistent=True,
        embedding=None if embedding is None else np.asarray(embedding, dtype=np.float32),
    )


def _analyze(observations, **kwargs):
    return analyze_track_purity(
        parent_track_id=kwargs.pop("parent_track_id", "CP1-cam5-back-TRK90000"),
        checkpoint_id="CP1",
        camera_id="cam5",
        source_video=kwargs.pop("source_video", "CP1\\cam5\\back.mp4"),
        observations=tuple(observations),
        source_fingerprint_sha256=SOURCE_FINGERPRINT,
        production_embedding_sha256=EMBEDDING_FINGERPRINT,
        **kwargs,
    )


def _strict_evaluator(observations):
    counts = {}
    for item in observations:
        counts[item.predicted_roll] = counts.get(item.predicted_roll, 0) + 1
    roll = sorted(counts, key=lambda value: (-counts[value], value))[0]
    return {"predicted_roll": roll, "best_score": 0.60, "margin": 0.20, "accepted": True}


class TrackletPurityPolicyTests(unittest.TestCase):
    def test_stable_pure_track_remains_unsplit_and_deterministic(self):
        observations = [_observation(index) for index in range(6)]
        first = _analyze(observations)
        second = _analyze(observations)
        self.assertEqual(first.outcome, PURE_UNSPLIT)
        self.assertFalse(first.quarantined)
        self.assertEqual(first.children, ())
        self.assertEqual([item["observation_id"] for item in first.identity_vote_timeline], [item.observation_id for item in observations])
        self.assertEqual(first.parent_evidence_fingerprint_sha256, second.parent_evidence_fingerprint_sha256)

    def test_single_noisy_observation_never_creates_unsafe_children(self):
        observations = [_observation(index) for index in range(7)]
        observations[3] = _observation(3, embedding=(-1.0, 0.0))
        result = _analyze(observations)
        self.assertIn(result.outcome, {PURE_UNSPLIT, APPEARANCE_DISCONTINUITY_ONLY, AMBIGUOUS_QUARANTINED, MIXED_QUARANTINED})
        self.assertEqual(result.children, ())
        self.assertNotEqual(result.outcome, SPLIT_CANDIDATE)

    def test_clear_temporal_geometry_embedding_identity_boundary_splits_deterministically(self):
        observations = [
            _observation(index, roll="ROLL-A", embedding=(1.0, 0.0), bbox=(index, 0.0, 20.0, 20.0), timestamp=index * 0.4)
            for index in range(3)
        ]
        observations.extend(
            _observation(
                index,
                roll="ROLL-B",
                embedding=(0.0, 1.0),
                bbox=(100.0 + index, 100.0, 20.0, 20.0),
                timestamp=2.4 + (index - 3) * 0.4,
            )
            for index in range(3, 6)
        )
        first = _analyze(observations, strict_evaluator=_strict_evaluator)
        second = _analyze(observations, strict_evaluator=_strict_evaluator)
        self.assertEqual(first.outcome, SPLIT_CANDIDATE)
        self.assertTrue(first.quarantined)
        self.assertEqual(len(first.children), 2)
        self.assertTrue(all(child.outcome == PURE_CHILD_CANDIDATE for child in first.children))
        self.assertTrue(all(child.strict_gate_passed for child in first.children))
        self.assertTrue(all(child.shadow_only and not child.inherited_human_review for child in first.children))
        self.assertEqual(
            [child.child_track_id for child in first.children],
            [child.child_track_id for child in second.children],
        )
        self.assertNotEqual(first.children[0].evidence_fingerprint_sha256, first.children[1].evidence_fingerprint_sha256)
        self.assertEqual(first.children[0].observation_ids, tuple(item.observation_id for item in observations[:3]))
        self.assertEqual(first.children[1].observation_ids, tuple(item.observation_id for item in observations[3:]))

    def test_ambiguous_identity_transition_without_other_signals_fails_closed(self):
        observations = [
            _observation(index, roll="ROLL-A" if index < 3 else "ROLL-B", embedding=(1.0, 0.0))
            for index in range(6)
        ]
        result = _analyze(observations)
        self.assertEqual(result.outcome, AMBIGUOUS_QUARANTINED)
        self.assertTrue(result.quarantined)
        self.assertEqual(result.children, ())

    def test_too_few_observations_on_one_side_prevents_split(self):
        observations = [
            _observation(index, roll="ROLL-A", embedding=(1.0, 0.0), bbox=(index, 0.0, 20.0, 20.0))
            for index in range(2)
        ]
        observations.extend(
            _observation(
                index,
                roll="ROLL-B",
                embedding=(0.0, 1.0),
                bbox=(100.0 + index, 100.0, 20.0, 20.0),
                timestamp=2.0 + (index - 2) * 0.4,
            )
            for index in range(2, 6)
        )
        result = _analyze(observations)
        self.assertNotEqual(result.outcome, SPLIT_CANDIDATE)
        self.assertEqual(result.children, ())

    def test_multiple_significant_embedding_clusters_quarantine_parent_without_boundary(self):
        observations = []
        for index in range(8):
            embedding = (1.0, 0.0) if index % 2 == 0 else (0.0, 1.0)
            observations.append(_observation(index, roll="ROLL-A", embedding=embedding))
        result = _analyze(observations)
        self.assertEqual(result.outcome, MIXED_QUARANTINED)
        self.assertEqual(sorted(result.significant_embedding_cluster_sizes), [4, 4])
        self.assertEqual(result.children, ())

    def test_identity_vote_switch_alone_never_splits(self):
        observations = [
            _observation(index, roll="ROLL-A" if index < 4 else "ROLL-B", embedding=(1.0, 0.0))
            for index in range(8)
        ]
        result = _analyze(observations)
        self.assertNotEqual(result.outcome, SPLIT_CANDIDATE)
        self.assertEqual(result.children, ())

    def test_invalid_embedding_evidence_fails_closed(self):
        observations = [_observation(index, embedding=(float("nan"), 0.0)) for index in range(5)]
        result = _analyze(observations)
        self.assertEqual(result.outcome, INVALID_EMBEDDING_EVIDENCE)
        self.assertTrue(result.quarantined)

    def test_overlapping_observation_time_is_rejected(self):
        observations = [_observation(index) for index in range(4)]
        observations[2] = _observation(2, timestamp=observations[1].timestamp_seconds)
        with self.assertRaisesRegex(TrackletPurityError, "Overlapping or duplicated"):
            _analyze(observations)

    def test_review_carry_forward_requires_exact_evidence_and_unchanged_lineage(self):
        stable = _analyze([_observation(index) for index in range(6)])
        allowed = review_carry_forward_decision(
            registry_accepted=True,
            registry_mixed_quarantine=False,
            existing_evidence_signature_sha256="same",
            current_evidence_signature_sha256="same",
            purity_result=stable,
        )
        self.assertTrue(allowed["allowed"])
        changed_set = review_carry_forward_decision(
            registry_accepted=True,
            registry_mixed_quarantine=False,
            existing_evidence_signature_sha256="before",
            current_evidence_signature_sha256="after",
            purity_result=stable,
        )
        self.assertEqual(changed_set["reason"], "evidence_signature_changed")
        changed_fingerprint = review_carry_forward_decision(
            registry_accepted=True,
            registry_mixed_quarantine=False,
            existing_evidence_signature_sha256="same",
            current_evidence_signature_sha256="same",
            existing_purity_evidence_fingerprint_sha256="before",
            current_purity_evidence_fingerprint_sha256="after",
            purity_result=stable,
        )
        self.assertEqual(changed_fingerprint["reason"], "purity_evidence_fingerprint_changed")

        split_observations = [
            _observation(index, roll="A", embedding=(1.0, 0.0), bbox=(index, 0.0, 20.0, 20.0))
            for index in range(3)
        ] + [
            _observation(
                index,
                roll="B",
                embedding=(0.0, 1.0),
                bbox=(100.0 + index, 100.0, 20.0, 20.0),
                timestamp=2.4 + (index - 3) * 0.4,
            )
            for index in range(3, 6)
        ]
        split = _analyze(split_observations)
        lineage = review_carry_forward_decision(
            registry_accepted=True,
            registry_mixed_quarantine=False,
            existing_evidence_signature_sha256="same",
            current_evidence_signature_sha256="same",
            purity_result=split,
        )
        self.assertEqual(lineage["reason"], "lineage_or_purity_changed")
        self.assertFalse(lineage["partial_carry_forward"])
        self.assertFalse(lineage["inherited_by_children"])

    def test_windows_paths_are_canonical_and_do_not_change_fingerprint(self):
        observations = [_observation(index) for index in range(6)]
        windows = _analyze(observations, source_video="CP1\\cam5\\back.mp4")
        posix = _analyze(observations, source_video="CP1/cam5/back.mp4")
        self.assertEqual(canonical_relative_path("CP1\\cam5\\back.mp4"), "CP1/cam5/back.mp4")
        self.assertEqual(windows.parent_evidence_fingerprint_sha256, posix.parent_evidence_fingerprint_sha256)

    def test_immutable_output_is_idempotent_and_tampering_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = {"purity_policy.json": b"{}\n", "parent_track_summary.csv": b"a\n1\n"}
            output, reused = write_immutable_output(output_root=root, run_id="tracklet-purity-test", files=files)
            self.assertFalse(reused)
            same, reused = write_immutable_output(output_root=root, run_id="tracklet-purity-test", files=files)
            self.assertTrue(reused)
            self.assertEqual(output, same)
            verify_immutable_output(output)
            (output / "parent_track_summary.csv").write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(TrackletPurityError, "hash changed"):
                verify_immutable_output(output)


class ProductPhase2KRealArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.preflight = preflight(default_inputs(ROOT), verify_videos=False)
        cls.analyses, cls.review_impact = analyze_preflight(cls.preflight)
        cls.by_id = {item.parent_track_id: item for item in cls.analyses}

    def test_exact_mixed_track_parent_remains_quarantined_without_children_or_review(self):
        mixed = self.by_id[MIXED_TRACKLET_ID]
        impact = next(item for item in self.review_impact if item["Tracklet_ID"] == MIXED_TRACKLET_ID)
        self.assertEqual(mixed.observation_count, 39)
        self.assertEqual(mixed.outcome, MIXED_QUARANTINED)
        self.assertTrue(mixed.quarantined)
        self.assertEqual(mixed.children, ())
        self.assertEqual(mixed.parent_official_checkpoint_contribution, 0)
        self.assertFalse(mixed.review_carry_forward_allowed)
        self.assertFalse(impact["Review_Carry_Forward_Allowed"])
        self.assertEqual(impact["Review_Carry_Forward_Reason"], "mixed_track_quarantine")
        self.assertFalse(impact["Child_Review_Inheritance"])

    def test_reviewed_strict_historical_tracks_are_not_unnecessarily_split(self):
        strict_ids = set(
            self.preflight.phase_2h_selected[
                self.preflight.phase_2h_selected["Evidence_Tier"].astype(str).str.strip() == "strict_accepted"
            ]["Tracklet_ID"].astype(str)
        )
        self.assertGreater(len(strict_ids), 0)
        self.assertEqual(
            [track_id for track_id in sorted(strict_ids) if self.by_id[track_id].outcome == SPLIT_CANDIDATE],
            [],
        )

    def test_shadow_contract_contains_all_required_outputs_and_preservation_flags(self):
        files, summary = build_shadow_files(self.preflight)
        self.assertEqual(
            set(files),
            {
                "purity_policy.json",
                "source_manifest.json",
                "parent_track_summary.csv",
                "split_boundary_candidates.csv",
                "child_track_summary.csv",
                "quarantined_tracks.csv",
                "review_carry_forward_impact.csv",
                "attendance_impact_shadow.csv",
                "evaluation_summary.json",
            },
        )
        preservation = summary["preservation_assertions"]
        self.assertFalse(preservation["official_attendance_changed"])
        self.assertFalse(preservation["recognition_repeated"])
        self.assertFalse(preservation["video_reprocessed"])
        self.assertFalse(preservation["authority_changed"])
        self.assertEqual(summary["known_mixed_track_analysis"]["final_shadow_disposition"], MIXED_QUARANTINED)


if __name__ == "__main__":
    unittest.main()
