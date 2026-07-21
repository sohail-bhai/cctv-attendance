from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.face_attendance.processing_integration import (
    OFFICIAL_RECOGNITION_AUTHORITY,
    POLICY_VERSION,
    PROCESSING_MODE,
    ProcessingIntegrationError,
    build_processing_command,
    build_processing_metadata,
    filter_embedding_records_to_roster,
    fixed_processing_policy,
    inspect_exact_checkpoint_layout,
    load_subject_roster,
)


class PolicyTests(unittest.TestCase):
    def test_fixed_policy_uses_strict_tracklet_authority(self):
        policy = fixed_processing_policy({"processing_mode": PROCESSING_MODE})
        self.assertEqual(policy["match_threshold"], 0.48)
        self.assertEqual(policy["margin_threshold"], 0.08)
        self.assertEqual(policy["present_checkpoints"], 3)
        self.assertEqual(policy["zone_mode"], "zones")
        self.assertEqual(policy["tracklet_mode"], "tracklets")
        self.assertEqual(policy["official_recognition_authority"], OFFICIAL_RECOGNITION_AUTHORITY)
        self.assertTrue(policy["tracklets_used_for_official_attendance"])

    def test_client_cannot_lower_threshold_or_checkpoint_rule(self):
        for payload in (
            {"processing_mode": PROCESSING_MODE, "match_threshold": 0.30},
            {"processing_mode": PROCESSING_MODE, "margin_threshold": 0.01},
            {"processing_mode": PROCESSING_MODE, "present_checkpoints": 1},
            {"processing_mode": PROCESSING_MODE, "tracklet_mode": "compare"},
            {"processing_mode": PROCESSING_MODE, "zone_mode": "compare"},
        ):
            with self.subTest(payload=payload), self.assertRaises(ProcessingIntegrationError):
                fixed_processing_policy(payload)

    def test_equivalent_explicit_values_are_allowed(self):
        policy = fixed_processing_policy(
            {
                "processing_mode": PROCESSING_MODE,
                "match_threshold": "0.48",
                "margin_threshold": 0.08,
                "save_unknown": False,
                "timeout_seconds": 1800,
            }
        )
        self.assertEqual(policy["timeout_seconds"], 1800)

    def test_unknown_or_unsafe_timeout_fails_closed(self):
        with self.assertRaises(ProcessingIntegrationError):
            fixed_processing_policy({"processing_mode": "demo"})
        with self.assertRaises(ProcessingIntegrationError):
            fixed_processing_policy({"processing_mode": PROCESSING_MODE, "timeout_seconds": 60})


class LayoutTests(unittest.TestCase):
    def _make_layout(self, root: Path, missing: str | None = None, extra_video: bool = False) -> None:
        for idx in range(1, 6):
            cp_id = f"CP{idx}"
            if cp_id == missing:
                continue
            folder = root / f"{cp_id}_{1140 + idx * 10:04d}"
            folder.mkdir(parents=True)
            (folder / "front.mp4").write_bytes(b"front")
            (folder / "back.mp4").write_bytes(b"back")
            if extra_video and idx == 1:
                (folder / "third.mp4").write_bytes(b"third")

    def test_exact_five_checkpoint_front_back_layout_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_layout(root)
            layout = inspect_exact_checkpoint_layout(root)
            self.assertEqual(layout.checkpoint_count, 5)
            self.assertEqual(layout.video_count, 10)
            self.assertTrue(layout.to_public_dict()["exact_front_back_layout"])

    def test_missing_checkpoint_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_layout(root, missing="CP2")
            with self.assertRaises(ProcessingIntegrationError):
                inspect_exact_checkpoint_layout(root)

    def test_extra_or_wrong_camera_video_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_layout(root, extra_video=True)
            with self.assertRaises(ProcessingIntegrationError):
                inspect_exact_checkpoint_layout(root)

    def test_ambiguous_checkpoint_name_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_layout(root)
            bad = root / "CP2-copy"
            bad.mkdir()
            (bad / "front.mp4").write_bytes(b"front")
            with self.assertRaises(ProcessingIntegrationError):
                inspect_exact_checkpoint_layout(root)


class RosterTests(unittest.TestCase):
    def _write_map(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "subject_students": {
                        "CVO": [
                            {"roll": "24011CSEAI0110", "name": "A"},
                            {"roll": "2401100CSE0268", "name": "B"},
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )

    def test_roster_filters_match_candidates_and_keeps_missing_student(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "map.json"
            self._write_map(path)
            roster = load_subject_roster(path, "CVO")
            records, missing = filter_embedding_records_to_roster(
                [
                    {"roll_no": "24011CSEAI0110", "embedding": [1.0, 0.0]},
                    {"roll_no": "OUTSIDER", "embedding": [0.0, 1.0]},
                ],
                roster.rolls,
            )
            self.assertEqual([row["roll_no"] for row in records], ["24011CSEAI0110"])
            self.assertEqual(missing, ["2401100CSE0268"])
            metadata = build_processing_metadata(
                roster=roster,
                embedding_record_count=1,
                embedding_student_count=1,
                missing_embedding_rolls=missing,
            )
            self.assertEqual(metadata["Authoritative_Roster_Count"], 2)
            self.assertEqual(metadata["Missing_Embedding_Rolls"], "2401100CSE0268")
            self.assertEqual(metadata["Processing_Contract_ID"], POLICY_VERSION)

    def test_distinct_rolls_are_not_aliased(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "map.json"
            path.write_text(
                json.dumps(
                    {
                        "subject_students": {
                            "CVO": [
                                {"roll": "2401100CSE0110"},
                                {"roll": "24011CSEAI0110"},
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            roster = load_subject_roster(path, "CVO")
            self.assertEqual(roster.rolls, ("2401100CSE0110", "24011CSEAI0110"))

    def test_duplicate_roster_roll_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "map.json"
            path.write_text(
                json.dumps(
                    {
                        "subject_students": {
                            "CVO": [{"roll": "24011CSEAI0110"}, {"roll": "24011CSEAI0110"}]
                        }
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ProcessingIntegrationError):
                load_subject_roster(path, "CVO")


class WiringContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.app_text = (cls.root / "app.py").read_text(encoding="utf-8")
        cls.runner_text = (cls.root / "scripts" / "mark_attendance_checkpoints.py").read_text(encoding="utf-8")
        cls.frontend_text = (cls.root / "frontend" / "src" / "pages" / "Timetable.jsx").read_text(encoding="utf-8")

    def test_browser_command_enables_strict_tracklet_authority(self):
        policy = fixed_processing_policy({"processing_mode": PROCESSING_MODE})
        command = build_processing_command(
            python_executable="python",
            script_path=Path("scripts/mark_attendance_checkpoints.py"),
            timetable_path=Path("timetable.csv"),
            slot_id="TUE_P1",
            video_dir=Path("videos"),
            embeddings_path=Path("models/student_embeddings.pkl"),
            student_map_path=Path("data/student_faculty_map.json"),
            output_dir=Path("attendance_output"),
            camera_zones_path=Path("data/camera_zones.json"),
            session={
                "session_id": "2026-06-30__B51__P1__CVO",
                "session_date": "2026-06-30",
                "subject_abbr": "CVO",
                "subject_name": "Computer Vision through OpenCV",
                "course_code": "24AMLJ502",
                "faculty_id": "vikas",
                "faculty_name": "Mr. Vikas B",
                "input_slot": "TUE_P1",
                "input_source_type": "prepared_slot",
                "input_source_path": "cctv_videos/prepared_slots/2026-06-30/TUE_P1",
            },
            policy=policy,
            diagnostic_run_id="product-2i-test",
        )
        joined = " ".join(command)
        self.assertIn("--require-authoritative-roster", command)
        self.assertIn("--diagnostic", command)
        self.assertIn("--zone-mode zones", joined)
        self.assertIn("--tracklet-mode tracklets", joined)
        self.assertIn("build_processing_command(", self.app_text)
        self.assertIn('"tracklets_used_for_official_attendance": True', self.app_text)

    def test_frontend_no_longer_controls_thresholds(self):
        self.assertIn("processing_mode: 'quality_aware_tracklet_authority_v1'", self.frontend_text)
        payload_block = self.frontend_text.split('const CONTROLLED_PROCESSING_PAYLOAD = {', 1)[1].split('};', 1)[0]
        self.assertNotIn("match_threshold", payload_block)
        self.assertNotIn("margin_threshold", payload_block)
        self.assertNotIn("present_checkpoints", payload_block)

    def test_processor_uses_authoritative_roster_for_report_population(self):
        self.assertIn("attendance_rolls = list(roster_contract.rolls)", self.runner_text)
        self.assertIn("CheckpointAttendanceBook(\n        attendance_rolls,", self.runner_text)
        self.assertIn("filtered_records", self.runner_text)
        self.assertIn("total_students=len(attendance_rolls)", self.runner_text)
        self.assertIn('"Total_Students": len(attendance_rolls)', self.runner_text)
        self.assertIn('Present: {present_count} / {len(attendance_rolls)}', self.runner_text)


if __name__ == "__main__":
    unittest.main()
