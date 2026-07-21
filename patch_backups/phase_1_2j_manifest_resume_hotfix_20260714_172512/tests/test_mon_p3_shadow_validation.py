from __future__ import annotations

import json
import tempfile
from dataclasses import replace
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from src.face_attendance.mon_p3_shadow_validation import (
    EXPECTED_RELATIVE_VIDEOS,
    EXPECTED_SESSION_ID,
    FamilyVerification,
    FreezeVerification,
    MonP3Preflight,
    MonP3ShadowError,
    _build_face_comparison,
    _build_observation_comparison,
    _build_student_checkpoint_comparison,
    _build_tracklet_comparison,
    _map_face_exceptions_to_tracklets,
    _read_csv_or_empty,
    _validate_diagnostic_source_layout,
    build_diagnostic_command,
    compare_protected_state,
    verify_candidate_family,
    verify_source_freeze,
)


def _write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def _hash(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _freeze_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    canonical = root / "canonical"
    mirror = root / "mirror"
    rows = []
    for index, relative in enumerate(EXPECTED_RELATIVE_VIDEOS, start=1):
        canonical_path = canonical / Path(relative.replace("\\", "/"))
        mirror_path = mirror / Path(relative.replace("\\", "/"))
        payload = (f"video-{index}" * 3).encode()
        _write_bytes(canonical_path, payload)
        _write_bytes(mirror_path, payload)
        rows.append(
            {
                "RelativePath": relative,
                "CanonicalPath": str(canonical_path.resolve()),
                "CanonicalLength": canonical_path.stat().st_size,
                "CanonicalSHA256": _hash(canonical_path),
                "MirrorPath": str(mirror_path.resolve()),
                "MirrorLength": mirror_path.stat().st_size,
                "MirrorSHA256": _hash(mirror_path),
                "ByteIdentical": True,
            }
        )
    csv_path = root / "freeze.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path = root / "freeze.json"
    json_path.write_text(
        json.dumps(
            {
                "SchemaVersion": 1,
                "Phase": "1.2J",
                "SessionID": EXPECTED_SESSION_ID,
                "CanonicalVideoRoot": str(canonical.resolve()),
                "MirrorVideoRoot": str(mirror.resolve()),
                "CanonicalVideoCount": 10,
                "MirrorVideoCount": 10,
                "ExpectedLayoutPassed": True,
                "TreesByteIdentical": True,
                "SourceFreezePassed": True,
                "RecognitionExecuted": False,
                "AttendanceWritten": False,
                "CandidatePromoted": False,
                "CanonicalSourceOnly": True,
                "CSV": str(csv_path.resolve()),
            }
        ),
        encoding="utf-8",
    )
    return canonical, mirror, json_path, csv_path


def _tracklet_row(track_id: str, *, accepted: str, roll: str, score: str = "0.60") -> dict:
    return {
        "Session_ID": EXPECTED_SESSION_ID,
        "Subject_Abbr": "CVO",
        "Tracklet_Mode": "compare",
        "Tracklet_ID": track_id,
        "Checkpoint_ID": "CP1",
        "Camera_ID": "cam1",
        "Video": "front.mp4",
        "Tracklet_Accepted": accepted,
        "Tracklet_Eligible": "Yes",
        "Tracklet_Quality_Rejection": "",
        "Tracklet_Diagnostic_Reason": "",
        "Tracklet_Matcher_Reason": "accepted" if accepted == "Yes" else "below_threshold",
        "Tracklet_Best_Roll": roll,
        "Tracklet_Best_Score": score,
        "Tracklet_Second_Roll": "S2",
        "Tracklet_Second_Score": "0.20",
        "Tracklet_Margin": "0.40",
        "Dominant_Frame_Best_Roll": roll,
        "Dominant_Frame_Best_Share_Pct": "100",
        "Observation_Count": "4",
        "Selected_Observation_Count": "4",
        "Embedding_Count": "4",
        "Consistent_Embedding_Count": "4",
        "Inconsistent_Embedding_Count": "0",
        "Pairwise_Similarity_Median": "0.80",
        "Selected_Observation_IDs": f"{track_id}-o1;{track_id}-o2",
        "Zone_IDs": "zone1",
        "Match_Threshold": "0.48",
        "Margin_Threshold": "0.08",
        "Aggregate_Mode": "top3",
        "Official_Attendance_Contribution": "No",
    }


def _observation_row(
    track_id: str,
    observation_id: str,
    *,
    accepted: str,
    roll: str,
    frame: int = 12,
) -> dict:
    return {
        "Tracklet_ID": track_id,
        "Observation_ID": observation_id,
        "Checkpoint_ID": "CP1",
        "Camera_ID": "cam1",
        "Video": "front.mp4",
        "Frame": str(frame),
        "BBox_Original_Coordinates": "10.00,20.00,30.00,40.00",
        "Zone_ID": "zone1",
        "Selected_For_Aggregation": "Yes",
        "Embedding_Consistent": "Yes",
        "Embedding_Extraction_Success": "Yes",
        "Embedding_Dimension": "128",
        "Frame_Best_Roll": roll,
        "Frame_Best_Score": "0.60",
        "Frame_Second_Roll": "S2",
        "Frame_Second_Score": "0.20",
        "Frame_Margin": "0.40",
        "Frame_Accepted": accepted,
        "Frame_Matcher_Reason": "accepted" if accepted == "Yes" else "below_threshold",
        "Tracklet_Best_Roll": roll,
        "Tracklet_Accepted": accepted,
    }


def _face_row(*, accepted: str, roll: str, frame: int = 12) -> dict:
    return {
        "Session_ID": EXPECTED_SESSION_ID,
        "Subject_Abbr": "CVO",
        "Checkpoint_ID": "CP1",
        "Camera_ID": "cam1",
        "Source_File_Name": "front.mp4",
        "Frame": str(frame),
        "Face_Index": "0",
        "BBox_Original_Coordinates": "10.00,20.00,30.00,40.00",
        "Detection_Source": "zone",
        "Zone_ID": "zone1",
        "Selected_Source": "zone",
        "Accepted": accepted,
        "Best_Roll": roll,
        "Best_Score": "0.60",
        "Second_Roll": "S2",
        "Second_Score": "0.20",
        "Margin": "0.40",
        "Actual_Matcher_Reject_Reason": "accepted" if accepted == "Yes" else "below_threshold",
    }


class CsvSafetyTests(unittest.TestCase):
    def test_headerless_empty_csv_becomes_a_typed_empty_frame(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "empty.csv"
            path.write_text("", encoding="utf-8")
            frame = _read_csv_or_empty(path, {"A", "B"})
            self.assertTrue(frame.empty)
            self.assertEqual(set(frame.columns), {"A", "B"})


class SourceFreezeTests(unittest.TestCase):
    def test_valid_freeze_rehashes_all_ten_video_pairs(self):
        with tempfile.TemporaryDirectory() as temp:
            canonical, mirror, json_path, csv_path = _freeze_fixture(Path(temp))
            result = verify_source_freeze(
                freeze_json_path=json_path,
                freeze_csv_path=csv_path,
                canonical_root=canonical,
                mirror_root=mirror,
            )
            self.assertEqual(len(result.rows), 10)
            self.assertTrue(all(row["sha256"] for row in result.rows))

    def test_canonical_video_mutation_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            canonical, mirror, json_path, csv_path = _freeze_fixture(Path(temp))
            (canonical / "CP1_1100" / "back.mp4").write_bytes(b"tampered")
            with self.assertRaisesRegex(MonP3ShadowError, "changed after source freeze"):
                verify_source_freeze(
                    freeze_json_path=json_path,
                    freeze_csv_path=csv_path,
                    canonical_root=canonical,
                    mirror_root=mirror,
                )

    def test_mirror_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            canonical, mirror, json_path, csv_path = _freeze_fixture(Path(temp))
            (mirror / "CP2_1110" / "front.mp4").write_bytes(b"different")
            with self.assertRaises(MonP3ShadowError):
                verify_source_freeze(
                    freeze_json_path=json_path,
                    freeze_csv_path=csv_path,
                    canonical_root=canonical,
                    mirror_root=mirror,
                )

    def test_freeze_claiming_recognition_was_executed_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            canonical, mirror, json_path, csv_path = _freeze_fixture(Path(temp))
            payload = json.loads(json_path.read_text())
            payload["RecognitionExecuted"] = True
            json_path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(MonP3ShadowError, "claims recognition"):
                verify_source_freeze(
                    freeze_json_path=json_path,
                    freeze_csv_path=csv_path,
                    canonical_root=canonical,
                    mirror_root=mirror,
                )


class CandidateFamilyTests(unittest.TestCase):
    def _family_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        family = root / "family"
        candidate = family / "variants" / "full_candidate" / "student_embeddings.pkl"
        candidate.parent.mkdir(parents=True)
        candidate.write_bytes(b"candidate-embeddings")
        production = root / "production.pkl"
        summary = root / "production_summary.csv"
        production.write_bytes(b"production")
        summary.write_text("roll,count\nS1,1\n", encoding="utf-8")
        (family / "evaluation").mkdir(parents=True)
        (family / "family_manifest.json").write_text(
            json.dumps(
                {
                    "family_id": "embfam-test",
                    "content_fingerprint_sha256": "f" * 64,
                    "production_approved": False,
                    "candidate_promoted": False,
                    "mon_p3_processed": False,
                }
            ),
            encoding="utf-8",
        )
        (family / "evaluation" / "evaluation_decision.json").write_text(
            json.dumps(
                {
                    "exact_next_step": "PHASE 1.2J - MON_P3 untouched shadow validation.",
                    "production_approved": False,
                    "candidate_promoted": False,
                    "mon_p3_processed": False,
                }
            ),
            encoding="utf-8",
        )
        (candidate.parent / "version_manifest.json").write_text(
            json.dumps(
                {
                    "version_id": "embfam-test-d",
                    "status": "built_unapproved",
                    "availability": True,
                    "production_promoted": False,
                    "training_contaminated_descriptive_only": True,
                    "evaluation_sessions_allowed": [],
                    "embedding_records": 10,
                    "artifact_sha256": {"student_embeddings.pkl": _hash(candidate)},
                }
            ),
            encoding="utf-8",
        )
        (family / "output_manifest.json").write_text("{}", encoding="utf-8")
        return family, production, summary

    def test_full_candidate_uses_version_id_and_positive_record_count(self):
        with tempfile.TemporaryDirectory() as temp:
            family, production, summary = self._family_fixture(Path(temp))
            with mock.patch(
                "src.face_attendance.mon_p3_shadow_validation.verify_embedding_family",
                return_value={"decision": "built_unapproved_pending_mon_p3"},
            ):
                verified = verify_candidate_family(
                    family_dir=family,
                    production_embeddings=production,
                    production_summary=summary,
                    expected_family_id="embfam-test",
                )
            self.assertEqual(verified.candidate_variant_id, "embfam-test-d")
            self.assertEqual(verified.candidate_embedding_records, 10)

    def test_full_candidate_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            family, production, summary = self._family_fixture(Path(temp))
            manifest_path = family / "variants" / "full_candidate" / "version_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["artifact_sha256"]["student_embeddings.pkl"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch(
                "src.face_attendance.mon_p3_shadow_validation.verify_embedding_family",
                return_value={"decision": "built_unapproved_pending_mon_p3"},
            ):
                with self.assertRaisesRegex(MonP3ShadowError, "hash does not match"):
                    verify_candidate_family(
                        family_dir=family,
                        production_embeddings=production,
                        production_summary=summary,
                        expected_family_id="embfam-test",
                    )


class ComparisonTests(unittest.TestCase):
    def test_candidate_only_tracklet_accept_is_material_delta(self):
        production = pd.DataFrame([_tracklet_row("T1", accepted="No", roll="S1")])
        candidate = pd.DataFrame([_tracklet_row("T1", accepted="Yes", roll="S1")])
        compared = _build_tracklet_comparison(production, candidate)
        self.assertEqual(compared.iloc[0]["Tracklet_Delta_Type"], "candidate_only_accept")
        self.assertEqual(compared.iloc[0]["Tracklet_Requires_Review"], "Yes")

    def test_same_accepted_identity_is_not_sent_to_review(self):
        production = pd.DataFrame([_tracklet_row("T1", accepted="Yes", roll="S1")])
        candidate = pd.DataFrame([_tracklet_row("T1", accepted="Yes", roll="S1", score="0.70")])
        compared = _build_tracklet_comparison(production, candidate)
        self.assertEqual(
            compared.iloc[0]["Tracklet_Delta_Type"], "unchanged_accepted_same_identity"
        )
        self.assertEqual(compared.iloc[0]["Tracklet_Requires_Review"], "No")

    def test_accepted_identity_change_is_material_delta(self):
        production = pd.DataFrame([_tracklet_row("T1", accepted="Yes", roll="S1")])
        candidate = pd.DataFrame([_tracklet_row("T1", accepted="Yes", roll="S3")])
        compared = _build_tracklet_comparison(production, candidate)
        self.assertEqual(compared.iloc[0]["Tracklet_Delta_Type"], "accepted_identity_changed")

    def test_tracklet_structure_change_fails_integrity(self):
        production = pd.DataFrame([_tracklet_row("T1", accepted="No", roll="S1")])
        candidate_row = _tracklet_row("T1", accepted="No", roll="S1")
        candidate_row["Checkpoint_ID"] = "CP2"
        with self.assertRaisesRegex(MonP3ShadowError, "structural column Checkpoint_ID"):
            _build_tracklet_comparison(production, pd.DataFrame([candidate_row]))

    def test_observation_frame_delta_is_detected(self):
        production = pd.DataFrame(
            [_observation_row("T1", "O1", accepted="No", roll="S1")]
        )
        candidate = pd.DataFrame(
            [_observation_row("T1", "O1", accepted="Yes", roll="S1")]
        )
        compared = _build_observation_comparison(production, candidate)
        self.assertEqual(compared.iloc[0]["Frame_Delta_Type"], "candidate_only_accept")

    def test_official_frame_delta_is_detected(self):
        production = pd.DataFrame([_face_row(accepted="No", roll="S1")])
        candidate = pd.DataFrame([_face_row(accepted="Yes", roll="S1")])
        compared = _build_face_comparison(production, candidate)
        self.assertEqual(compared.iloc[0]["Frame_Delta_Type"], "candidate_only_accept")

    def test_official_frame_delta_maps_to_exact_tracklet(self):
        production = pd.DataFrame([_face_row(accepted="No", roll="S1")])
        candidate = pd.DataFrame([_face_row(accepted="Yes", roll="S1")])
        compared = _build_face_comparison(production, candidate)
        observations = pd.DataFrame(
            [_observation_row("T1", "O1", accepted="Yes", roll="S1")]
        )
        mapped, tracklets = _map_face_exceptions_to_tracklets(compared, observations)
        self.assertEqual(tracklets, {"T1"})
        self.assertEqual(mapped.iloc[0]["Mapped_Tracklet_ID"], "T1")

    def test_unmapped_official_frame_delta_is_held_without_crashing(self):
        production = pd.DataFrame([_face_row(accepted="No", roll="S1")])
        candidate = pd.DataFrame([_face_row(accepted="Yes", roll="S1")])
        compared = _build_face_comparison(production, candidate)
        observations = pd.DataFrame(
            [_observation_row("T1", "O1", accepted="Yes", roll="S1", frame=99)]
        )
        mapped, tracklets = _map_face_exceptions_to_tracklets(compared, observations)
        self.assertEqual(tracklets, set())
        self.assertEqual(mapped.iloc[0]["Mapping_Status"], "unmapped_no_overlapping_tracklet")
        self.assertEqual(mapped.iloc[0]["Mapped_Tracklet_ID"], "")

    def test_student_checkpoint_comparison_is_diagnostic_only_and_deterministic(self):
        production = pd.DataFrame(
            [
                _tracklet_row("T1", accepted="No", roll="S1"),
                _tracklet_row("T2", accepted="Yes", roll="S2"),
            ]
        )
        candidate = pd.DataFrame(
            [
                _tracklet_row("T1", accepted="Yes", roll="S1"),
                _tracklet_row("T2", accepted="Yes", roll="S2"),
            ]
        )
        compared = _build_tracklet_comparison(production, candidate)
        checkpoints = _build_student_checkpoint_comparison(compared)
        s1 = checkpoints[checkpoints["Roll"].eq("S1")].iloc[0]
        self.assertEqual(s1["Checkpoint_Delta"], "candidate_only")


class ProtectedStateTests(unittest.TestCase):
    def test_identical_protected_snapshots_pass(self):
        compared = compare_protected_state({"a": "1", "tree": {"x": "2"}}, {"a": "1", "tree": {"x": "2"}})
        self.assertTrue(compared["unchanged"])

    def test_protected_snapshot_change_is_reported(self):
        compared = compare_protected_state({"a": "1"}, {"a": "2"})
        self.assertFalse(compared["unchanged"])
        self.assertEqual(compared["changed_keys"], ["a"])


class CommandContractTests(unittest.TestCase):
    def _preflight(self, root: Path) -> MonP3Preflight:
        camera_zones = root / "camera_zones.json"
        camera_zones.write_text("{}")
        freeze = FreezeVerification(
            json_path=root / "freeze.json",
            csv_path=root / "freeze.csv",
            json_sha256="j",
            csv_sha256="c",
            canonical_root=root / "videos",
            mirror_root=root / "mirror",
            rows=[],
        )
        family = FamilyVerification(
            family_dir=root / "family",
            family_id="embfam-test",
            family_content_fingerprint="f",
            family_output_manifest_sha256="m",
            candidate_embeddings=root / "candidate.pkl",
            candidate_embeddings_sha256="h",
            candidate_variant_id="embfam-test-d",
            candidate_embedding_records=10,
        )
        config = {
            "frame_skip": 1,
            "sample_fps": 2.0,
            "aggregate": "top3",
            "checkpoint_mode": "clip-folders",
            "log_mode": "full",
            "zone_mode": "compare",
            "zone_profile": "auto",
            "zone_merge_iou": 0.65,
            "tracklet_mode": "compare",
            "tracklet_min_observations": 3,
            "tracklet_max_selected": 5,
            "tracklet_max_gap_seconds": 2.0,
            "tracklet_min_iou": 0.2,
            "tracklet_max_center_ratio": 1.5,
            "tracklet_min_size_ratio": 0.5,
            "tracklet_min_embedding_similarity": 0.35,
        }
        return MonP3Preflight(
            repo_root=root,
            session_id=EXPECTED_SESSION_ID,
            session_fields={
                "session_date": "2026-06-22",
                "section": "B51",
                "period": "P3",
                "subject_abbr": "CVO",
            },
            slot_id="MON_P3",
            subject_abbr="CVO",
            video_root=root / "videos",
            mirror_video_root=root / "mirror",
            videos=[],
            freeze=freeze,
            family=family,
            production_embeddings=root / "production.pkl",
            production_summary=root / "summary.csv",
            production_embeddings_sha256="p",
            production_summary_sha256="s",
            student_map=root / "map.json",
            roster_sha256="r",
            subject_rolls=set(),
            timetable=root / "timetable.csv",
            timetable_sha256="t",
            reference_diagnostic_run=root / "reference",
            reference_configuration=config,
            camera_zones=camera_zones,
            camera_zones_sha256="z",
            fingerprint_sha256="x" * 64,
            run_id="mon-p3-shadow-test",
            output_root=root / "output",
            output_dir=root / "output" / "run",
            production_diagnostic_run=root / "diagnostics" / "production",
            candidate_diagnostic_run=root / "diagnostics" / "candidate",
        )

    def test_command_is_diagnostic_only_and_uses_mon_p3(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            preflight = self._preflight(root)
            command = build_diagnostic_command(
                preflight,
                embeddings_path=preflight.family.candidate_embeddings,
                diagnostic_run_id="candidate-run",
                python_executable=Path("/usr/bin/python3"),
            )
            joined = " ".join(command)
            self.assertIn("--diagnostic-only", command)
            self.assertIn("--diagnostic", command)
            self.assertIn("MON_P3", command)
            self.assertIn(EXPECTED_SESSION_ID, command)
            self.assertIn(str(preflight.family.candidate_embeddings.resolve()), command)
            self.assertNotIn("--save-unknown", command)
            self.assertNotIn("--tracklet-review-export", joined)

    def test_production_and_candidate_commands_differ_only_by_embeddings_and_run_id(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            preflight = self._preflight(root)
            production = build_diagnostic_command(
                preflight,
                embeddings_path=preflight.production_embeddings,
                diagnostic_run_id="production-run",
            )
            candidate = build_diagnostic_command(
                preflight,
                embeddings_path=preflight.family.candidate_embeddings,
                diagnostic_run_id="candidate-run",
            )
            prod_clean = [
                "<EMBEDDINGS>" if value == str(preflight.production_embeddings.resolve()) else "<RUN>" if value == "production-run" else value
                for value in production
            ]
            cand_clean = [
                "<EMBEDDINGS>" if value == str(preflight.family.candidate_embeddings.resolve()) else "<RUN>" if value == "candidate-run" else value
                for value in candidate
            ]
            self.assertEqual(prod_clean, cand_clean)


class DiagnosticSourceLayoutTests(unittest.TestCase):
    def _preflight_with_frozen_videos(self, root: Path) -> MonP3Preflight:
        base = CommandContractTests()._preflight(root)
        rows = []
        for relative in EXPECTED_RELATIVE_VIDEOS:
            source = root / "videos" / Path(relative.replace("\\", "/"))
            _write_bytes(source, relative.encode("utf-8"))
            rows.append(
                {
                    "relative_path": relative,
                    "canonical_path": str(source.resolve()),
                    "mirror_path": str((root / "mirror" / Path(relative.replace("\\", "/"))).resolve()),
                    "length": source.stat().st_size,
                    "sha256": _hash(source),
                }
            )
        freeze = replace(base.freeze, rows=rows)
        return replace(base, freeze=freeze, video_root=root / "videos")

    def _valid_summary(self, preflight: MonP3Preflight) -> dict:
        input_files = []
        timing_rows = []
        for row in preflight.freeze.rows:
            relative = row["relative_path"].replace("\\", "/")
            checkpoint = relative.split("/", 1)[0].split("_", 1)[0]
            source = row["canonical_path"]
            input_files.append(
                {
                    "source_file": source,
                    "source_file_name": Path(source).name,
                }
            )
            timing_rows.append(
                {
                    "Source_File": source,
                    "Source_File_Name": Path(source).name,
                    "Checkpoint_ID": checkpoint,
                }
            )
        return {
            "input_files": input_files,
            "checkpoint_timing_map": timing_rows,
        }

    def test_exact_five_checkpoint_two_camera_layout_passes(self):
        with tempfile.TemporaryDirectory() as temp:
            preflight = self._preflight_with_frozen_videos(Path(temp))
            result = _validate_diagnostic_source_layout(
                self._valid_summary(preflight),
                preflight,
            )
            self.assertEqual(result["validated_video_count"], 10)
            self.assertEqual(result["checkpoint_ids"], ["CP1", "CP2", "CP3", "CP4", "CP5"])

    def test_cp1_repeated_and_cp2_missing_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            preflight = self._preflight_with_frozen_videos(Path(temp))
            summary = self._valid_summary(preflight)
            cp1_rows = [
                row
                for row in summary["input_files"]
                if "CP1_1100" in row["source_file"]
            ]
            summary["input_files"] = cp1_rows + cp1_rows + [
                row
                for row in summary["input_files"]
                if "CP1_1100" not in row["source_file"]
                and "CP2_1110" not in row["source_file"]
            ]
            with self.assertRaisesRegex(MonP3ShadowError, "frozen ten-video checkpoint set"):
                _validate_diagnostic_source_layout(summary, preflight)

    def test_checkpoint_id_must_match_source_folder(self):
        with tempfile.TemporaryDirectory() as temp:
            preflight = self._preflight_with_frozen_videos(Path(temp))
            summary = self._valid_summary(preflight)
            summary["checkpoint_timing_map"][0]["Checkpoint_ID"] = "CP2"
            with self.assertRaisesRegex(MonP3ShadowError, "folder mapping is invalid"):
                _validate_diagnostic_source_layout(summary, preflight)



if __name__ == "__main__":
    unittest.main()
