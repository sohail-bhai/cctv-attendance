from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .recall_analysis import canonical_roll


POLICY_VERSION = "product-phase-2g-quality-aware-processing-v1"
PROCESSING_MODE = "quality_aware_production_v1"
OFFICIAL_RECOGNITION_AUTHORITY = "frame_detection"
TRACKLET_MODE = "compare"
ZONE_MODE = "compare"
EXPECTED_CHECKPOINT_IDS = ("CP1", "CP2", "CP3", "CP4", "CP5")
EXPECTED_CAMERA_STEMS = ("back", "front")
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

FIXED_POLICY: dict[str, Any] = {
    "processing_mode": PROCESSING_MODE,
    "checkpoint_mode": "clip-folders",
    "checkpoint_min_detections": 2,
    "present_checkpoints": 3,
    "strong_checkpoints": 4,
    "review_checkpoints": 2,
    "match_threshold": 0.48,
    "margin_threshold": 0.08,
    "sample_fps": 2.0,
    "frame_skip": 3,
    "log_mode": "full",
    "output_layout": "organized",
    "aggregate": "top3",
    "save_unknown": False,
    "timeout_seconds": 30 * 60,
    "settle_minutes": 10.0,
    "checkpoint_every_minutes": 10.0,
    "checkpoint_clip_seconds": 20.0,
    "diagnostic": True,
    "zone_mode": ZONE_MODE,
    "zone_profile": "auto",
    "zone_merge_iou": 0.20,
    "tracklet_mode": TRACKLET_MODE,
    "tracklet_min_observations": 2,
    "tracklet_max_selected": 5,
    "tracklet_max_gap_seconds": 1.5,
    "tracklet_min_iou": 0.10,
    "tracklet_max_center_ratio": 1.25,
    "tracklet_min_size_ratio": 0.50,
    "tracklet_min_embedding_similarity": 0.25,
    "official_recognition_authority": OFFICIAL_RECOGNITION_AUTHORITY,
    "tracklets_used_for_official_attendance": False,
}

SAFETY_OVERRIDE_KEYS = frozenset(
    {
        "checkpoint_mode",
        "checkpoint_min_detections",
        "present_checkpoints",
        "strong_checkpoints",
        "review_checkpoints",
        "match_threshold",
        "margin_threshold",
        "sample_fps",
        "frame_skip",
        "log_mode",
        "output_layout",
        "aggregate",
        "save_unknown",
        "settle_minutes",
        "checkpoint_every_minutes",
        "checkpoint_clip_seconds",
        "zone_mode",
        "zone_profile",
        "zone_merge_iou",
        "tracklet_mode",
        "tracklet_min_observations",
        "tracklet_max_selected",
        "tracklet_max_gap_seconds",
        "tracklet_min_iou",
        "tracklet_max_center_ratio",
        "tracklet_min_size_ratio",
        "tracklet_min_embedding_similarity",
    }
)


class ProcessingIntegrationError(ValueError):
    pass


@dataclass(frozen=True)
class CheckpointSource:
    checkpoint_id: str
    directory: Path
    videos: tuple[Path, ...]


@dataclass(frozen=True)
class CheckpointLayout:
    root: Path
    checkpoints: tuple[CheckpointSource, ...]

    @property
    def checkpoint_count(self) -> int:
        return len(self.checkpoints)

    @property
    def video_count(self) -> int:
        return sum(len(item.videos) for item in self.checkpoints)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "video_dir_exists": self.root.is_dir(),
            "checkpoint_folder_count": self.checkpoint_count,
            "recursive_video_count": self.video_count,
            "checkpoint_folders": [item.directory.name for item in self.checkpoints],
            "has_full_checkpoint_set": self.checkpoint_count == 5,
            "is_demo_clip_set": False,
            "exact_front_back_layout": True,
            "checkpoint_video_counts": {
                item.checkpoint_id: len(item.videos) for item in self.checkpoints
            },
            "checkpoint_video_names": {
                item.checkpoint_id: [path.name for path in item.videos]
                for item in self.checkpoints
            },
        }


@dataclass(frozen=True)
class RosterContract:
    subject: str
    rolls: tuple[str, ...]
    source_path: Path

    @property
    def count(self) -> int:
        return len(self.rolls)


def _as_text(value: Any) -> str:
    return str(value or "").strip()


def _same_value(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        if isinstance(actual, bool):
            return actual is expected
        lowered = _as_text(actual).lower()
        return lowered in ({"true", "1", "yes", "on"} if expected else {"false", "0", "no", "off", ""})
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return abs(float(actual) - float(expected)) <= 1e-9
        except (TypeError, ValueError):
            return False
    return _as_text(actual).lower() == _as_text(expected).lower()


def fixed_processing_policy(payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the immutable browser-processing policy.

    The client may identify the processing mode and request a shorter timeout, but it
    cannot alter recognition thresholds, checkpoint voting, zone/tracklet modes, or
    evidence logging. Any attempted drift fails closed rather than being ignored.
    """

    payload = dict(payload or {})
    requested_mode = _as_text(payload.get("processing_mode") or PROCESSING_MODE)
    if requested_mode != PROCESSING_MODE:
        raise ProcessingIntegrationError(
            f"Unsupported processing mode {requested_mode!r}; expected {PROCESSING_MODE!r}."
        )

    drift = []
    for key in sorted(SAFETY_OVERRIDE_KEYS.intersection(payload)):
        if not _same_value(payload.get(key), FIXED_POLICY[key]):
            drift.append(f"{key}={payload.get(key)!r} (required {FIXED_POLICY[key]!r})")
    if drift:
        raise ProcessingIntegrationError(
            "Browser processing safety policy cannot be overridden: " + "; ".join(drift)
        )

    policy = dict(FIXED_POLICY)
    if "timeout_seconds" in payload:
        try:
            timeout_seconds = int(payload["timeout_seconds"])
        except (TypeError, ValueError) as exc:
            raise ProcessingIntegrationError("timeout_seconds must be an integer") from exc
        if not 10 * 60 <= timeout_seconds <= 45 * 60:
            raise ProcessingIntegrationError(
                "timeout_seconds must remain between 600 and 2700 seconds"
            )
        policy["timeout_seconds"] = timeout_seconds
    return policy


def inspect_exact_checkpoint_layout(
    video_dir: Path,
    *,
    expected_camera_stems: Sequence[str] = EXPECTED_CAMERA_STEMS,
) -> CheckpointLayout:
    root = Path(video_dir)
    if not root.is_dir():
        raise ProcessingIntegrationError(f"Footage folder not found: {root}")

    checkpoint_dirs: dict[str, Path] = {}
    extra_checkpoint_dirs: list[str] = []
    for child in root.iterdir():
        if not child.is_dir() or not child.name.lower().startswith("cp"):
            continue
        match = re.fullmatch(r"(CP[1-5])_(\d{4})", child.name.upper())
        if not match:
            extra_checkpoint_dirs.append(child.name)
            continue
        cp_id = match.group(1)
        if cp_id in checkpoint_dirs:
            raise ProcessingIntegrationError(
                f"Multiple folders resolve to {cp_id}: {checkpoint_dirs[cp_id].name}, {child.name}"
            )
        checkpoint_dirs[cp_id] = child

    if extra_checkpoint_dirs:
        raise ProcessingIntegrationError(
            "Unexpected checkpoint folder naming: " + ", ".join(sorted(extra_checkpoint_dirs))
        )
    missing = [cp_id for cp_id in EXPECTED_CHECKPOINT_IDS if cp_id not in checkpoint_dirs]
    if missing or len(checkpoint_dirs) != len(EXPECTED_CHECKPOINT_IDS):
        found = ", ".join(sorted(checkpoint_dirs)) or "none"
        raise ProcessingIntegrationError(
            f"Production attendance requires exactly CP1-CP5; missing={missing or 'none'}, found={found}."
        )

    expected_stems = {str(value).lower() for value in expected_camera_stems}
    sources: list[CheckpointSource] = []
    seen_hashless_paths: set[str] = set()
    for cp_id in EXPECTED_CHECKPOINT_IDS:
        directory = checkpoint_dirs[cp_id]
        videos = tuple(
            sorted(
                (
                    path
                    for path in directory.iterdir()
                    if path.is_file() and path.suffix.lower() in ALLOWED_VIDEO_EXTENSIONS
                ),
                key=lambda path: path.name.lower(),
            )
        )
        stems = {path.stem.lower() for path in videos}
        if len(videos) != len(expected_stems) or stems != expected_stems:
            raise ProcessingIntegrationError(
                f"{directory.name} must contain exactly {sorted(expected_stems)} videos; "
                f"found {[path.name for path in videos]}."
            )
        for path in videos:
            resolved = str(path.resolve()).lower()
            if resolved in seen_hashless_paths:
                raise ProcessingIntegrationError(f"Video source reused twice: {path}")
            seen_hashless_paths.add(resolved)
            if path.stat().st_size <= 0:
                raise ProcessingIntegrationError(f"Video file is empty: {path}")
        sources.append(CheckpointSource(cp_id, directory, videos))

    layout = CheckpointLayout(root=root, checkpoints=tuple(sources))
    if layout.video_count != 10:
        raise ProcessingIntegrationError(
            f"Production attendance requires ten front/back clips; found {layout.video_count}."
        )
    return layout


def load_subject_roster(student_map_path: Path, subject: str) -> RosterContract:
    path = Path(student_map_path)
    if not path.is_file():
        raise ProcessingIntegrationError(f"Authoritative student map not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProcessingIntegrationError(f"Authoritative student map is invalid: {path}") from exc

    subject_key = _as_text(subject).upper()
    rows = (payload.get("subject_students") or {}).get(subject_key)
    if not isinstance(rows, list) or not rows:
        raise ProcessingIntegrationError(
            f"No authoritative roster is configured for {subject_key}."
        )

    rolls: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ProcessingIntegrationError(f"Invalid {subject_key} roster row")
        roll = canonical_roll(row.get("roll"))
        if not roll:
            raise ProcessingIntegrationError(f"Blank roll number in {subject_key} roster")
        if roll in seen:
            raise ProcessingIntegrationError(f"Duplicate roll {roll} in {subject_key} roster")
        seen.add(roll)
        rolls.append(roll)
    return RosterContract(subject_key, tuple(rolls), path)


def filter_embedding_records_to_roster(
    records: Iterable[Mapping[str, Any]], roster_rolls: Iterable[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    roster = {canonical_roll(value) for value in roster_rolls if canonical_roll(value)}
    filtered: list[dict[str, Any]] = []
    covered: set[str] = set()
    for raw in records:
        roll = canonical_roll(raw.get("roll_no"))
        if not roll or roll not in roster:
            continue
        record = dict(raw)
        record["roll_no"] = roll
        filtered.append(record)
        covered.add(roll)
    missing = sorted(roster - covered)
    return filtered, missing


def build_processing_metadata(
    *,
    roster: RosterContract,
    embedding_record_count: int,
    embedding_student_count: int,
    missing_embedding_rolls: Sequence[str],
) -> dict[str, Any]:
    return {
        "Processing_Contract_ID": POLICY_VERSION,
        "Processing_Mode": PROCESSING_MODE,
        "Official_Recognition_Authority": OFFICIAL_RECOGNITION_AUTHORITY,
        "Zone_Mode": ZONE_MODE,
        "Tracklet_Mode": TRACKLET_MODE,
        "Tracklets_Used_For_Official_Attendance": "No",
        "Authoritative_Roster_Source": str(roster.source_path),
        "Authoritative_Roster_Subject": roster.subject,
        "Authoritative_Roster_Count": roster.count,
        "Embedding_Covered_Students": embedding_student_count,
        "Embedding_Record_Count": embedding_record_count,
        "Missing_Embedding_Count": len(missing_embedding_rolls),
        "Missing_Embedding_Rolls": "; ".join(missing_embedding_rolls),
    }


def build_processing_command(
    *,
    python_executable: str,
    script_path: Path,
    timetable_path: Path,
    slot_id: str,
    video_dir: Path,
    embeddings_path: Path,
    student_map_path: Path,
    output_dir: Path,
    camera_zones_path: Path,
    session: Mapping[str, Any],
    policy: Mapping[str, Any],
    diagnostic_run_id: str,
) -> list[str]:
    required_session = (
        "session_id",
        "session_date",
        "subject_abbr",
        "subject_name",
        "course_code",
        "faculty_id",
        "faculty_name",
        "input_slot",
        "input_source_type",
        "input_source_path",
    )
    missing = [key for key in required_session if key not in session]
    if missing:
        raise ProcessingIntegrationError(
            "Session metadata missing: " + ", ".join(missing)
        )
    if policy.get("processing_mode") != PROCESSING_MODE:
        raise ProcessingIntegrationError("Processing policy mode mismatch")
    if policy.get("official_recognition_authority") != OFFICIAL_RECOGNITION_AUTHORITY:
        raise ProcessingIntegrationError("Official recognition authority changed")
    if policy.get("tracklet_mode") != TRACKLET_MODE or policy.get("zone_mode") != ZONE_MODE:
        raise ProcessingIntegrationError("Validated evidence modes changed")
    if policy.get("tracklets_used_for_official_attendance") is not False:
        raise ProcessingIntegrationError("Tracklets cannot become official authority in Phase 2G")

    return [
        str(python_executable),
        "-u",
        str(Path(script_path)),
        "--timetable", str(Path(timetable_path)),
        "--slot-id", str(slot_id),
        "--video-dir", str(Path(video_dir)),
        "--embeddings", str(Path(embeddings_path)),
        "--student-map", str(Path(student_map_path)),
        "--require-authoritative-roster",
        "--processing-contract-id", POLICY_VERSION,
        "--output-dir", str(Path(output_dir)),
        "--match-threshold", str(policy["match_threshold"]),
        "--margin-threshold", str(policy["margin_threshold"]),
        "--checkpoint-min-detections", str(policy["checkpoint_min_detections"]),
        "--present-checkpoints", str(policy["present_checkpoints"]),
        "--strong-checkpoints", str(policy["strong_checkpoints"]),
        "--review-checkpoints", str(policy["review_checkpoints"]),
        "--settle-minutes", str(policy["settle_minutes"]),
        "--checkpoint-every-minutes", str(policy["checkpoint_every_minutes"]),
        "--checkpoint-clip-seconds", str(policy["checkpoint_clip_seconds"]),
        "--checkpoint-mode", str(policy["checkpoint_mode"]),
        "--frame-skip", str(policy["frame_skip"]),
        "--sample-fps", str(policy["sample_fps"]),
        "--log-mode", str(policy["log_mode"]),
        "--output-layout", str(policy["output_layout"]),
        "--aggregate", str(policy["aggregate"]),
        "--diagnostic",
        "--diagnostic-dir", str(Path(output_dir) / "diagnostics"),
        "--diagnostic-run-id", str(diagnostic_run_id),
        "--zone-mode", str(policy["zone_mode"]),
        "--camera-zones", str(Path(camera_zones_path)),
        "--zone-profile", str(policy["zone_profile"]),
        "--zone-merge-iou", str(policy["zone_merge_iou"]),
        "--tracklet-mode", str(policy["tracklet_mode"]),
        "--tracklet-min-observations", str(policy["tracklet_min_observations"]),
        "--tracklet-max-selected", str(policy["tracklet_max_selected"]),
        "--tracklet-max-gap-seconds", str(policy["tracklet_max_gap_seconds"]),
        "--tracklet-min-iou", str(policy["tracklet_min_iou"]),
        "--tracklet-max-center-ratio", str(policy["tracklet_max_center_ratio"]),
        "--tracklet-min-size-ratio", str(policy["tracklet_min_size_ratio"]),
        "--tracklet-min-embedding-similarity", str(policy["tracklet_min_embedding_similarity"]),
        "--session-id", str(session["session_id"]),
        "--session-date", str(session["session_date"]),
        "--subject-abbr", str(session["subject_abbr"]),
        "--subject-name", str(session.get("subject_name") or ""),
        "--course-code", str(session.get("course_code") or ""),
        "--faculty-id", str(session.get("faculty_id") or ""),
        "--faculty-name", str(session.get("faculty_name") or ""),
        "--input-slot", str(session.get("input_slot") or slot_id),
        "--input-source-type", str(session.get("input_source_type") or ""),
        "--input-source-path", str(session.get("input_source_path") or ""),
    ]
