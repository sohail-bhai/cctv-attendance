import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts.mark_attendance_checkpoints import (
    Checkpoint,
    CheckpointAttendanceBook,
    build_tracklet_observation_rows,
    evaluate_finalized_tracklet,
    parse_args,
    select_tracklet_candidates,
    tracklet_config_from_args,
    validate_tracklet_review_args,
)
from src.face_attendance.embedding_db import MatchResult
from src.face_attendance.tracklets import (
    TrackletBuilder,
    TrackletConfig,
    TrackletObservation,
    observation_quality_weight,
)


def observation(
    observation_id,
    frame_index,
    timestamp,
    box,
    embedding=(1.0, 0.0, 0.0),
    quality_label="usable_reference",
    landmark_valid=True,
    blur=120.0,
    detector_score=0.95,
    source="full_frame",
    zone_id="",
):
    return TrackletObservation(
        observation_id=observation_id,
        frame_index=frame_index,
        timestamp_seconds=timestamp,
        bbox=tuple(float(value) for value in box),
        detector_score=detector_score,
        embedding=None if embedding is None else np.asarray(embedding, dtype=np.float32),
        quality_label=quality_label,
        quality_reasons="",
        landmark_valid=landmark_valid,
        blur_score=blur,
        face_width=float(box[2]),
        face_height=float(box[3]),
        detection_source=source,
        zone_id=zone_id,
    )


class TrackletAssociationTests(unittest.TestCase):
    def test_same_face_across_frames_forms_one_tracklet(self):
        builder = TrackletBuilder(TrackletConfig(), id_prefix="CP1-cam5")
        builder.update([observation("a", 1, 0.0, (10, 10, 40, 40))])
        builder.update([observation("b", 2, 0.5, (12, 11, 41, 40))])
        builder.update([observation("c", 3, 1.0, (14, 12, 40, 41))])

        tracklets = builder.finalize()

        self.assertEqual(len(tracklets), 1)
        self.assertEqual(tracklets[0].observation_count, 3)
        self.assertEqual(tracklets[0].member_ids, ("a", "b", "c"))
        self.assertTrue(tracklets[0].eligible)

    def test_spatially_separate_faces_are_not_merged(self):
        builder = TrackletBuilder(TrackletConfig(), id_prefix="CP1-cam5")
        builder.update([
            observation("left-1", 1, 0.0, (10, 10, 30, 30)),
            observation("right-1", 1, 0.0, (100, 10, 30, 30)),
        ])
        builder.update([
            observation("left-2", 2, 0.5, (12, 10, 30, 30)),
            observation("right-2", 2, 0.5, (98, 11, 30, 30)),
        ])

        tracklets = builder.finalize()

        self.assertEqual(len(tracklets), 2)
        self.assertEqual(sorted(tracklet.observation_count for tracklet in tracklets), [2, 2])

    def test_large_time_gap_starts_a_new_tracklet(self):
        config = TrackletConfig(max_gap_seconds=1.0)
        builder = TrackletBuilder(config, id_prefix="CP1-cam5")
        builder.update([observation("a", 1, 0.0, (10, 10, 40, 40))])
        builder.update([observation("b", 2, 1.5, (11, 10, 40, 40))])

        self.assertEqual(len(builder.finalize()), 2)

    def test_extreme_size_change_does_not_join_nearby_face(self):
        config = TrackletConfig(min_size_ratio=0.50)
        builder = TrackletBuilder(config, id_prefix="CP1-cam5")
        builder.update([observation("large", 1, 0.0, (10, 10, 60, 60))])
        builder.update([observation("tiny", 2, 0.5, (30, 30, 12, 12))])

        self.assertEqual(len(builder.finalize()), 2)

    def test_association_is_deterministic_when_detection_order_changes(self):
        def run(reverse):
            builder = TrackletBuilder(TrackletConfig(), id_prefix="CP1-cam5")
            first = [
                observation("a1", 1, 0.0, (10, 10, 30, 30)),
                observation("b1", 1, 0.0, (70, 10, 30, 30)),
            ]
            second = [
                observation("a2", 2, 0.5, (12, 10, 30, 30)),
                observation("b2", 2, 0.5, (68, 10, 30, 30)),
            ]
            builder.update(list(reversed(first)) if reverse else first)
            builder.update(list(reversed(second)) if reverse else second)
            return [(tracklet.tracklet_id, tracklet.member_ids) for tracklet in builder.finalize()]

        self.assertEqual(run(False), run(True))

    def test_builder_rejects_out_of_order_frames(self):
        builder = TrackletBuilder(TrackletConfig(), id_prefix="CP1-cam5")
        builder.update([observation("later", 2, 1.0, (10, 10, 40, 40))])
        with self.assertRaisesRegex(ValueError, "chronological"):
            builder.update([observation("earlier", 1, 0.5, (11, 10, 40, 40))])

    def test_equal_competing_assignments_split_instead_of_mixing_tracks(self):
        builder = TrackletBuilder(TrackletConfig(), id_prefix="CP1-cam5")
        builder.update([
            observation("left", 1, 0.0, (0, 0, 20, 20)),
            observation("right", 1, 0.0, (40, 0, 20, 20)),
        ])
        builder.update([observation("ambiguous", 2, 0.5, (20, 0, 20, 20))])

        tracklets = builder.finalize()

        self.assertEqual(len(tracklets), 3)
        self.assertEqual(sorted(tracklet.observation_count for tracklet in tracklets), [1, 1, 1])
        ambiguous_member = next(
            member
            for tracklet in tracklets
            for member in tracklet.members
            if member.observation.observation_id == "ambiguous"
        )
        self.assertEqual(ambiguous_member.association_reason, "ambiguous_association_split")


class TrackletAggregationTests(unittest.TestCase):
    def test_quality_ranking_keeps_stronger_observations(self):
        config = TrackletConfig(max_selected_observations=2, min_size_ratio=0.20)
        builder = TrackletBuilder(config, id_prefix="CP1-cam5")
        builder.update([observation(
            "weak", 1, 0.0, (10, 10, 22, 22),
            quality_label="unusable_reference", landmark_valid=False, blur=4.0, detector_score=0.86,
        )])
        builder.update([observation("strong", 2, 0.5, (11, 10, 45, 45), blur=180.0, detector_score=0.97)])
        builder.update([observation("middle", 3, 1.0, (12, 11, 38, 38), blur=80.0, detector_score=0.92)])

        tracklet = builder.finalize()[0]

        self.assertEqual(tracklet.selected_member_ids, ("strong", "middle"))
        self.assertNotIn("weak", tracklet.selected_member_ids)

    def test_aggregate_embedding_is_normalized(self):
        builder = TrackletBuilder(TrackletConfig(), id_prefix="CP1-cam5")
        builder.update([observation("a", 1, 0.0, (10, 10, 40, 40), embedding=(3.0, 0.0, 0.0))])
        builder.update([observation("b", 2, 0.5, (11, 10, 40, 40), embedding=(2.0, 0.1, 0.0))])

        aggregate = builder.finalize()[0].aggregate_embedding

        self.assertIsNotNone(aggregate)
        self.assertAlmostEqual(float(np.linalg.norm(aggregate)), 1.0, places=6)

    def test_embedding_outlier_is_excluded_from_aggregate(self):
        config = TrackletConfig(min_embedding_similarity=0.40)
        builder = TrackletBuilder(config, id_prefix="CP1-cam5")
        builder.update([observation("a", 1, 0.0, (10, 10, 40, 40), embedding=(1.0, 0.0, 0.0))])
        builder.update([observation("b", 2, 0.5, (11, 10, 40, 40), embedding=(0.98, 0.10, 0.0))])
        builder.update([observation("outlier", 3, 1.0, (12, 10, 40, 40), embedding=(0.0, 1.0, 0.0))])

        tracklet = builder.finalize()[0]

        self.assertTrue(tracklet.eligible)
        self.assertEqual(tracklet.consistent_member_ids, ("a", "b"))
        self.assertEqual(tracklet.inconsistent_member_ids, ("outlier",))
        self.assertGreater(float(tracklet.aggregate_embedding[0]), 0.99)

    def test_single_observation_tracklet_is_not_eligible(self):
        builder = TrackletBuilder(TrackletConfig(min_observations=2), id_prefix="CP1-cam5")
        builder.update([observation("a", 1, 0.0, (10, 10, 40, 40))])

        tracklet = builder.finalize()[0]

        self.assertFalse(tracklet.eligible)
        self.assertEqual(tracklet.rejection_reason, "insufficient_observations")
        self.assertEqual(tracklet.embedding_count, 1)
        self.assertIsNone(tracklet.aggregate_embedding)

    def test_missing_embeddings_are_reported_without_crashing(self):
        builder = TrackletBuilder(TrackletConfig(min_observations=2), id_prefix="CP1-cam5")
        builder.update([observation("a", 1, 0.0, (10, 10, 40, 40), embedding=None)])
        builder.update([observation("b", 2, 0.5, (11, 10, 40, 40), embedding=None)])

        tracklet = builder.finalize()[0]

        self.assertFalse(tracklet.eligible)
        self.assertEqual(tracklet.rejection_reason, "insufficient_embeddings")

    def test_quality_weight_penalizes_unusable_unaligned_crop(self):
        good = observation_quality_weight(observation("good", 1, 0.0, (0, 0, 48, 48)))
        weak = observation_quality_weight(observation(
            "weak", 1, 0.0, (0, 0, 20, 20), quality_label="unusable_reference",
            landmark_valid=False, blur=3.0, detector_score=0.86,
        ))
        self.assertGreater(good, weak)

    def test_configuration_validation_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "min_observations"):
            TrackletConfig(min_observations=1)
        with self.assertRaisesRegex(ValueError, "min_iou"):
            TrackletConfig(min_iou=1.5)
        with self.assertRaisesRegex(ValueError, "min_embedding_similarity"):
            TrackletConfig(min_embedding_similarity=-1.5)


class TrackletRunnerContractTests(unittest.TestCase):
    def test_cli_default_keeps_tracklets_off(self):
        with patch("sys.argv", ["mark_attendance_checkpoints.py", "--timetable", "timetable.csv"]):
            args = parse_args()
        self.assertEqual(args.tracklet_mode, "off")
        self.assertFalse(args.export_tracklet_review)

    def test_review_export_is_explicit_and_requires_diagnostic_tracklets(self):
        base = {
            "export_tracklet_review": True,
            "diagnostic": True,
            "tracklet_mode": "compare",
            "tracklet_review_evidence_count": 5,
            "tracklet_review_crop_padding": 0.45,
        }
        validate_tracklet_review_args(SimpleNamespace(**base))

        with self.assertRaisesRegex(SystemExit, "requires --diagnostic"):
            validate_tracklet_review_args(SimpleNamespace(**{**base, "diagnostic": False}))
        with self.assertRaisesRegex(SystemExit, "requires --tracklet-mode"):
            validate_tracklet_review_args(SimpleNamespace(**{**base, "tracklet_mode": "off"}))

    def test_off_mode_does_not_validate_or_construct_tracklet_config(self):
        args = SimpleNamespace(
            tracklet_mode="off",
            tracklet_min_observations=-50,
            tracklet_max_selected=-1,
            tracklet_max_gap_seconds=-1,
            tracklet_min_iou=8,
            tracklet_max_center_ratio=-1,
            tracklet_min_size_ratio=4,
            tracklet_min_embedding_similarity=4,
        )
        with patch.dict("sys.modules", {"src.face_attendance.tracklets": None}):
            self.assertIsNone(tracklet_config_from_args(args))

    def test_frame_attendance_output_remains_tracklet_schema_free(self):
        checkpoint = Checkpoint(
            cp_id="CP1",
            label="CP1_09_10",
            class_time="09:10",
            class_offset_sec=600.0,
            window_start_sec=600.0,
            window_end_sec=620.0,
        )
        book = CheckpointAttendanceBook(
            all_students=["student"],
            checkpoints=[checkpoint],
            checkpoint_min_detections=2,
            present_checkpoints=1,
            strong_checkpoints=2,
            review_checkpoints=1,
            low_confidence_score=0.51,
        )

        attendance = book.to_attendance_dataframe({})
        checkpoint_rows = book.to_student_checkpoint_dataframe({})

        self.assertFalse(any("Tracklet" in column for column in attendance.columns))
        self.assertNotIn("Tracklet_Confirmed", checkpoint_rows.columns)
        self.assertNotIn("Evidence_Unit", checkpoint_rows.columns)

    def test_compare_mode_uses_merged_zone_evidence_only_for_tracklets(self):
        full = [object()]
        merged = [object(), object()]

        source, selected = select_tracklet_candidates("compare", "compare", full, merged)

        self.assertEqual(source, "merged_full_frame_plus_zones")
        self.assertIs(selected, merged)

    def test_off_and_operational_source_selection_preserve_explicit_authority(self):
        full = [object()]
        merged = [object(), object()]
        self.assertEqual(select_tracklet_candidates("off", "zones", full, merged), ("off", []))
        self.assertIs(select_tracklet_candidates("tracklets", "compare", full, merged)[1], full)
        self.assertIs(select_tracklet_candidates("tracklets", "zones", full, merged)[1], merged)

    def test_aggregate_match_forwards_existing_thresholds_unchanged(self):
        builder = TrackletBuilder(TrackletConfig(), id_prefix="CP1-cam5")
        builder.update([observation("a", 1, 0.0, (10, 10, 40, 40))])
        builder.update([observation("b", 2, 0.5, (11, 10, 40, 40))])
        tracklet = builder.finalize()[0]

        class RecordingDB:
            def __init__(self):
                self.received = None

            def match(self, embedding, match_threshold, margin_threshold, aggregate):
                self.received = (match_threshold, margin_threshold, aggregate, float(np.linalg.norm(embedding)))
                return MatchResult(True, "student", 0.60, "other", 0.40, 0.20, "accepted")

        db = RecordingDB()
        match, reason = evaluate_finalized_tracklet(tracklet, db, 0.48, 0.08, "top3")

        self.assertTrue(match.accepted)
        self.assertEqual(reason, "accepted")
        self.assertEqual(db.received[:3], (0.48, 0.08, "top3"))
        self.assertAlmostEqual(db.received[3], 1.0, places=6)

    def test_observation_state_does_not_retain_raw_frames(self):
        value = observation("a", 1, 0.0, (10, 10, 40, 40))
        self.assertFalse(hasattr(value, "frame"))
        self.assertFalse(hasattr(value, "crop"))

    def test_serialized_observation_rows_do_not_expose_embedding_vectors(self):
        checkpoint = Checkpoint(
            cp_id="CP1",
            label="CP1_09_10",
            class_time="09:10",
            class_offset_sec=600.0,
            window_start_sec=600.0,
            window_end_sec=620.0,
        )
        builder = TrackletBuilder(TrackletConfig(), id_prefix="CP1-cam5")
        builder.update([observation("a", 1, 0.0, (10, 10, 40, 40))])
        builder.update([observation("b", 2, 0.5, (11, 10, 40, 40))])
        tracklet = builder.finalize()[0]
        match = MatchResult(True, "student", 0.60, "other", 0.40, 0.20, "accepted")

        rows = build_tracklet_observation_rows(
            slot_dict={},
            source_set="baseline_full_frame",
            tracklet=tracklet,
            match=match,
            cp=checkpoint,
            camera_id="cam5",
            video_path=Path("back.mp4"),
        )

        self.assertEqual(len(rows), 2)
        self.assertTrue(all("Embedding_Dimension" in row for row in rows))
        self.assertTrue(all("Embedding" not in row for row in rows))
        self.assertFalse(any(isinstance(value, np.ndarray) for row in rows for value in row.values()))

    def test_accepted_multiframe_tracklet_confirms_checkpoint_once(self):
        checkpoint = Checkpoint(
            cp_id="CP1",
            label="CP1_09_10",
            class_time="09:10",
            class_offset_sec=600.0,
            window_start_sec=600.0,
            window_end_sec=620.0,
        )
        book = CheckpointAttendanceBook(
            all_students=["student"],
            checkpoints=[checkpoint],
            checkpoint_min_detections=2,
            present_checkpoints=1,
            strong_checkpoints=2,
            review_checkpoints=1,
            low_confidence_score=0.51,
            tracklet_evidence_enabled=True,
        )

        book.add_tracklet(
            "student",
            "CP1",
            0.55,
            0.10,
            "back.mp4",
            "cam5",
            "CP1/09:10/cam5/00:05.00/TRK00001",
        )

        self.assertEqual(book.recognized_checkpoint_ids("student"), ["CP1"])
        self.assertEqual(book.cp_counts["student"]["CP1"], 1)
        attendance = book.to_attendance_dataframe({})
        self.assertEqual(attendance.loc[0, "Final_Status"], "Present")
        self.assertEqual(attendance.loc[0, "CP1_Tracklet_Confirmed"], "Yes")
        checkpoint_rows = book.to_student_checkpoint_dataframe({})
        self.assertEqual(checkpoint_rows.loc[0, "Recognized_In_Checkpoint"], "Yes")
        self.assertEqual(checkpoint_rows.loc[0, "Evidence_Unit"], "tracklet_aggregate")


if __name__ == "__main__":
    unittest.main()
