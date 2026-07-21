from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "mark_attendance_checkpoints.py"
SPEC = importlib.util.spec_from_file_location("mark_attendance_checkpoints_for_test", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load checkpoint script: {SCRIPT_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

Checkpoint = MODULE.Checkpoint
discover_checkpoint_clip_folders = MODULE.discover_checkpoint_clip_folders


def _checkpoints() -> list[Checkpoint]:
    return [
        Checkpoint("CP1", "CP1_11_00", "11:00", 600.0, 600.0, 620.0),
        Checkpoint("CP2", "CP2_11_10", "11:10", 1200.0, 1200.0, 1220.0),
        Checkpoint("CP3", "CP3_11_20", "11:20", 1800.0, 1800.0, 1820.0),
        Checkpoint("CP4", "CP4_11_30", "11:30", 2400.0, 2400.0, 2420.0),
        Checkpoint("CP5", "CP5_11_40", "11:40", 3000.0, 2980.0, 3000.0),
    ]


def _video_folder(root: Path, name: str) -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "front.mp4").write_bytes(b"fixture")
    return folder


class CheckpointClipFolderDiscoveryTests(unittest.TestCase):
    def test_mon_p3_folders_map_one_to_one(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            expected = {
                "CP1": "CP1_1100",
                "CP2": "CP2_1110",
                "CP3": "CP3_1120",
                "CP4": "CP4_1130",
                "CP5": "CP5_1140",
            }
            for folder_name in expected.values():
                _video_folder(root, folder_name)

            found = discover_checkpoint_clip_folders(root, _checkpoints())

            self.assertEqual(
                {cp_id: folder.name for cp_id, folder in found.items()},
                expected,
            )
            self.assertEqual(len({folder.resolve() for folder in found.values()}), 5)

    def test_cp1_1100_is_never_reused_as_cp2_1110(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cp1 = _video_folder(root, "CP1_1100")

            found = discover_checkpoint_clip_folders(root, _checkpoints())

            self.assertEqual(found, {"CP1": cp1})
            self.assertNotIn("CP2", found)

    def test_time_only_folder_uses_exact_time_token(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = _video_folder(root, "checkpoint_11_10")

            found = discover_checkpoint_clip_folders(root, _checkpoints())

            self.assertEqual(found, {"CP2": target})

    def test_explicit_checkpoint_and_conflicting_time_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _video_folder(root, "CP1_1110")

            with self.assertRaisesRegex(ValueError, "ID/time conflict"):
                discover_checkpoint_clip_folders(root, _checkpoints())

    def test_two_folders_for_one_checkpoint_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _video_folder(root, "CP1")
            _video_folder(root, "checkpoint_CP1")

            with self.assertRaisesRegex(ValueError, "Multiple clip folders map to CP1"):
                discover_checkpoint_clip_folders(root, _checkpoints())


if __name__ == "__main__":
    unittest.main()
