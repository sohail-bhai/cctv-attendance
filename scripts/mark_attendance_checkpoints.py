from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.face_attendance.config import (
    DEFAULT_DETECTION_SCORE,
    DEFAULT_FRAME_SKIP,
    DEFAULT_MARGIN_THRESHOLD,
    DEFAULT_MATCH_THRESHOLD,
    DEFAULT_MAX_WIDTH,
    DEFAULT_NMS_THRESHOLD,
    DEFAULT_TOP_K,
    EMBEDDINGS_PATH,
    OUTPUT_DIR,
    SFACE_MODEL,
    UNKNOWN_DIR,
    VIDEO_DIR,
    VIDEO_EXTENSIONS,
    YUNET_MODEL,
)
from src.face_attendance.embedding_db import StudentEmbeddingDB
from src.face_attendance.face_engine import FaceEngine
from src.face_attendance.timetable import list_slots, load_timetable, parse_time, select_slot
from src.face_attendance.utils import clean_filename, ensure_dirs, iter_files, resize_keep_aspect, safe_crop


@dataclass(frozen=True)
class Checkpoint:
    cp_id: str
    label: str
    class_time: str
    class_offset_sec: float
    window_start_sec: float
    window_end_sec: float
    note: str = ""


@dataclass
class VideoMeta:
    path: Path
    fps: float
    frame_count: int
    duration_sec: float


class CheckpointAttendanceBook:
    """Collects student evidence per checkpoint, camera, and class slot."""

    def __init__(
        self,
        all_students: list[str],
        checkpoints: list[Checkpoint],
        checkpoint_min_detections: int,
        present_checkpoints: int,
        strong_checkpoints: int,
        review_checkpoints: int,
        low_confidence_score: float,
    ) -> None:
        self.all_students = sorted(all_students)
        self.checkpoints = checkpoints
        self.cp_ids = [cp.cp_id for cp in checkpoints]
        self.checkpoint_min_detections = int(checkpoint_min_detections)
        self.present_checkpoints = int(present_checkpoints)
        self.strong_checkpoints = int(strong_checkpoints)
        self.review_checkpoints = int(review_checkpoints)
        self.low_confidence_score = float(low_confidence_score)

        self.total_counts: Dict[str, int] = defaultdict(int)
        self.scores: Dict[str, List[float]] = defaultdict(list)
        self.margins: Dict[str, List[float]] = defaultdict(list)
        self.videos_seen: Dict[str, Set[str]] = defaultdict(set)
        self.cameras_seen: Dict[str, Set[str]] = defaultdict(set)
        self.first_seen: Dict[str, str] = {}
        self.last_seen: Dict[str, str] = {}

        self.cp_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.cp_scores: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self.cp_margins: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self.cp_cameras: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        self.cp_videos: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        self.camera_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def add_detection(
        self,
        roll_no: str,
        checkpoint_id: str,
        score: float,
        margin: float,
        video_name: str,
        camera_id: str,
        evidence_time: str,
    ) -> None:
        self.total_counts[roll_no] += 1
        self.scores[roll_no].append(float(score))
        self.margins[roll_no].append(float(margin))
        self.videos_seen[roll_no].add(video_name)
        self.cameras_seen[roll_no].add(camera_id)
        self.first_seen.setdefault(roll_no, evidence_time)
        self.last_seen[roll_no] = evidence_time

        self.cp_counts[roll_no][checkpoint_id] += 1
        self.cp_scores[roll_no][checkpoint_id].append(float(score))
        self.cp_margins[roll_no][checkpoint_id].append(float(margin))
        self.cp_cameras[roll_no][checkpoint_id].add(camera_id)
        self.cp_videos[roll_no][checkpoint_id].add(video_name)
        self.camera_counts[roll_no][camera_id] += 1

    def recognized_checkpoint_ids(self, roll_no: str) -> list[str]:
        recognized = []
        for cp_id in self.cp_ids:
            if self.cp_counts[roll_no].get(cp_id, 0) >= self.checkpoint_min_detections:
                recognized.append(cp_id)
        return recognized

    def checkpoint_symbols(self, roll_no: str) -> dict[str, str]:
        symbols = {}
        for cp_id in self.cp_ids:
            count = self.cp_counts[roll_no].get(cp_id, 0)
            if count >= self.checkpoint_min_detections:
                symbols[cp_id] = "Yes"
            elif count > 0:
                symbols[cp_id] = "Weak"
            else:
                symbols[cp_id] = "No"
        return symbols

    def status_and_flags_for(self, roll_no: str) -> tuple[str, str]:
        recognized = self.recognized_checkpoint_ids(roll_no)
        recognized_count = len(recognized)
        flags: list[str] = []

        if recognized_count >= self.strong_checkpoints:
            status = "Present Strong"
        elif recognized_count >= self.present_checkpoints:
            status = "Present"
        elif recognized_count >= self.review_checkpoints:
            status = "Needs Review"
        else:
            status = "Absent"

        cp_set = set(recognized)
        if len(self.cp_ids) >= 5:
            first_two = set(self.cp_ids[:2])
            last_three = set(self.cp_ids[2:5])
            first_three = set(self.cp_ids[:3])
            last_two = set(self.cp_ids[-2:])
            if cp_set.isdisjoint(first_two) and len(cp_set & last_three) >= 3:
                flags.append("Late Entry")
            if len(cp_set & first_three) >= 3 and cp_set.isdisjoint(last_two):
                flags.append("Left Early Flag")

        scores = self.scores.get(roll_no, [])
        if scores and (sum(scores) / len(scores)) < self.low_confidence_score:
            flags.append("Low Confidence")
        if self.total_counts.get(roll_no, 0) > 0 and recognized_count == 0:
            flags.append("Weak Evidence Only")
        if self.total_counts.get(roll_no, 0) > 0 and len(self.cameras_seen.get(roll_no, set())) == 1:
            flags.append("Single Camera Evidence")

        return status, "; ".join(flags)

    def to_attendance_dataframe(self, slot_dict: dict) -> pd.DataFrame:
        rows = []
        for roll in self.all_students:
            scores = self.scores.get(roll, [])
            margins = self.margins.get(roll, [])
            recognized = self.recognized_checkpoint_ids(roll)
            status, flags = self.status_and_flags_for(roll)
            row = {
                **slot_dict,
                "Roll_Number": roll,
                "Final_Status": status,
                "Present": "Yes" if status in {"Present", "Present Strong"} else "No",
                "Recognized_Checkpoints": len(recognized),
                "Total_Checkpoints": len(self.cp_ids),
                "Checkpoint_Attendance_Score": round((len(recognized) / len(self.cp_ids)) * 100, 1) if self.cp_ids else 0,
                "Total_Accepted_Detections": self.total_counts.get(roll, 0),
                "Best_Score": round(max(scores), 4) if scores else "",
                "Average_Score": round(sum(scores) / len(scores), 4) if scores else "",
                "Average_Margin": round(sum(margins) / len(margins), 4) if margins else "",
                "Videos_Seen": ", ".join(sorted(self.videos_seen.get(roll, []))),
                "Cameras_Seen": ", ".join(sorted(self.cameras_seen.get(roll, []))),
                "First_Seen_Evidence": self.first_seen.get(roll, ""),
                "Last_Seen_Evidence": self.last_seen.get(roll, ""),
                "Flags": flags,
            }
            for cp in self.checkpoints:
                cp_scores = self.cp_scores[roll].get(cp.cp_id, [])
                row[f"{cp.cp_id}_{clean_filename(cp.class_time)}"] = self.checkpoint_symbols(roll)[cp.cp_id]
                row[f"{cp.cp_id}_Detections"] = self.cp_counts[roll].get(cp.cp_id, 0)
                row[f"{cp.cp_id}_Best_Score"] = round(max(cp_scores), 4) if cp_scores else ""
                row[f"{cp.cp_id}_Cameras"] = ", ".join(sorted(self.cp_cameras[roll].get(cp.cp_id, set())))
            rows.append(row)
        return pd.DataFrame(rows)

    def to_student_checkpoint_dataframe(self, slot_dict: dict) -> pd.DataFrame:
        rows = []
        for roll in self.all_students:
            for cp in self.checkpoints:
                scores = self.cp_scores[roll].get(cp.cp_id, [])
                margins = self.cp_margins[roll].get(cp.cp_id, [])
                count = self.cp_counts[roll].get(cp.cp_id, 0)
                rows.append({
                    **slot_dict,
                    "Roll_Number": roll,
                    "Checkpoint_ID": cp.cp_id,
                    "Checkpoint_Label": cp.label,
                    "Checkpoint_Class_Time": cp.class_time,
                    "Recognized_In_Checkpoint": "Yes" if count >= self.checkpoint_min_detections else "No",
                    "Checkpoint_Detections": count,
                    "Checkpoint_Min_Detections": self.checkpoint_min_detections,
                    "Best_Score": round(max(scores), 4) if scores else "",
                    "Average_Score": round(sum(scores) / len(scores), 4) if scores else "",
                    "Average_Margin": round(sum(margins) / len(margins), 4) if margins else "",
                    "Cameras_Seen": ", ".join(sorted(self.cp_cameras[roll].get(cp.cp_id, set()))),
                    "Videos_Seen": ", ".join(sorted(self.cp_videos[roll].get(cp.cp_id, set()))),
                })
        return pd.DataFrame(rows)

    def camera_summary_dataframe(self, slot_dict: dict, camera_metrics: dict) -> pd.DataFrame:
        rows = []
        for key in sorted(camera_metrics):
            cp_id, camera_id, video_name = key
            metrics = camera_metrics[key]
            accepted = metrics.get("accepted", 0)
            detected = metrics.get("detected", 0)
            rows.append({
                **slot_dict,
                "Checkpoint_ID": cp_id,
                "Camera_ID": camera_id,
                "Video": video_name,
                "Frames_Processed": metrics.get("frames", 0),
                "Faces_Detected": detected,
                "Accepted_Recognitions": accepted,
                "Rejected_Or_Unknown": metrics.get("rejected", 0),
                "Unique_Students_Recognized": len(metrics.get("students", set())),
                "Recognition_Rate": round((accepted / detected) * 100, 1) if detected else 0,
            })
        return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="MVP 3 checkpoint-based CCTV attendance using timetable + YuNet + SFace.")

    parser.add_argument("--timetable", required=True, help="Path to timetable CSV/XLSX")
    parser.add_argument("--slot-id", help="Timetable slot id, for example MON_P1")
    parser.add_argument("--day", help="Day name, for example Monday")
    parser.add_argument("--period", help="Period number, for example 1")
    parser.add_argument("--class-time", help="Time inside the class slot, for example 09:10")
    parser.add_argument("--list-slots", action="store_true", help="List timetable slots and exit")

    parser.add_argument("--video-dir", default=str(VIDEO_DIR), help="Folder containing CCTV videos/camera angles for this class slot")
    parser.add_argument("--embeddings", default=str(EMBEDDINGS_PATH), help="student_embeddings.pkl path")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output attendance folder")
    parser.add_argument("--unknown-dir", default=str(UNKNOWN_DIR), help="Folder to save unknown/rejected faces")

    parser.add_argument("--det-score", type=float, default=DEFAULT_DETECTION_SCORE, help="YuNet detection threshold")
    parser.add_argument("--match-threshold", type=float, default=0.48, help="Cosine match threshold. Higher = stricter")
    parser.add_argument("--margin-threshold", type=float, default=0.08, help="Best score must beat second score by this amount")
    parser.add_argument("--frame-skip", type=int, default=DEFAULT_FRAME_SKIP, help="Fallback: process every Nth frame when --sample-fps is 0")
    parser.add_argument("--sample-fps", type=float, default=2.0, help="Fast processing: sample this many frames per second from each checkpoint window. Use 0 to disable and use --frame-skip.")
    parser.add_argument("--max-width", type=int, default=DEFAULT_MAX_WIDTH, help="Resize video frames to this max width. 0 = original size")
    parser.add_argument("--aggregate", choices=["centroid", "max", "top3"], default="top3", help="How to compare against enrolled images")

    parser.add_argument("--settle-minutes", type=float, default=10.0, help="Ignore first N minutes as settling time")
    parser.add_argument("--checkpoint-every-minutes", type=float, default=10.0, help="Checkpoint interval after settling time")
    parser.add_argument("--checkpoint-clip-seconds", type=float, default=20.0, help="Video seconds processed for each checkpoint")
    parser.add_argument("--checkpoint-mode", choices=["auto", "class-time", "split-video", "clip-folders"], default="auto", help="class-time uses real class offsets; split-video divides short demo videos into checkpoint windows; auto chooses safely")
    parser.add_argument("--checkpoint-min-detections", type=int, default=2, help="Accepted detections needed inside a checkpoint to count that checkpoint")
    parser.add_argument("--present-checkpoints", type=int, default=3, help="Checkpoints needed for Present")
    parser.add_argument("--strong-checkpoints", type=int, default=4, help="Checkpoints needed for Present Strong")
    parser.add_argument("--review-checkpoints", type=int, default=2, help="Checkpoints needed for Needs Review")
    parser.add_argument("--low-confidence-score", type=float, default=0.51, help="Average score below this adds Low Confidence flag")
    parser.add_argument("--quality-poor-rate", type=float, default=25.0, help="Checkpoint recognition rate below this percent is flagged Poor")

    parser.add_argument("--log-mode", choices=["full", "accepted", "compact"], default="accepted", help="Detection log size: full=all attempts, accepted=recognized only, compact=recognized plus margin-confusion rows")
    parser.add_argument("--output-layout", choices=["flat", "organized"], default="organized", help="flat saves directly in attendance_output; organized saves in attendance_output/YYYY-MM-DD/SLOT/")
    parser.add_argument("--save-unknown", action="store_true", help="Save rejected/unknown face crops")
    parser.add_argument("--max-unknown-per-camera-checkpoint", type=int, default=20, help="Storage safety limit for unknown crops")
    parser.add_argument("--display", action="store_true", help="Show live video window. Press q to stop")
    return parser.parse_args()


def seconds_to_text(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    m = int(seconds // 60)
    s = seconds - (m * 60)
    return f"{m:02d}:{s:05.2f}"


def time_plus_minutes(time_text: str, minutes: float) -> str:
    base = datetime.combine(datetime.today(), parse_time(time_text))
    return (base + timedelta(minutes=minutes)).strftime("%H:%M")


def duration_minutes(start_time: str, end_time: str) -> float:
    today = datetime.today()
    start_dt = datetime.combine(today, parse_time(start_time))
    end_dt = datetime.combine(today, parse_time(end_time))
    if end_dt <= start_dt:
        end_dt += timedelta(days=1)
    return (end_dt - start_dt).total_seconds() / 60.0


def build_checkpoints(slot, settle_minutes: float, every_minutes: float, clip_seconds: float) -> list[Checkpoint]:
    total_minutes = duration_minutes(slot.start_time, slot.end_time)
    if total_minutes <= 0:
        raise ValueError(f"Invalid slot duration: {slot.start_time}-{slot.end_time}")
    if settle_minutes >= total_minutes:
        settle_minutes = max(0.0, total_minutes / 4.0)

    offsets_min: list[float] = []
    current = float(settle_minutes)
    while current < total_minutes:
        offsets_min.append(current)
        current += float(every_minutes)

    if not offsets_min or not math.isclose(offsets_min[-1], total_minutes, abs_tol=0.01):
        offsets_min.append(total_minutes)

    checkpoints = []
    for idx, offset_min in enumerate(offsets_min, start=1):
        class_offset_sec = offset_min * 60.0
        if math.isclose(offset_min, total_minutes, abs_tol=0.01):
            # Final checkpoint ends exactly at class end to avoid next class movement.
            window_end = total_minutes * 60.0
            window_start = max(0.0, window_end - clip_seconds)
            note = "final_checkpoint_ends_at_class_end"
        else:
            window_start = class_offset_sec
            window_end = min(total_minutes * 60.0, window_start + clip_seconds)
            note = "regular_checkpoint"

        checkpoints.append(Checkpoint(
            cp_id=f"CP{idx}",
            label=f"CP{idx}_{clean_filename(time_plus_minutes(slot.start_time, offset_min))}",
            class_time=time_plus_minutes(slot.start_time, offset_min),
            class_offset_sec=class_offset_sec,
            window_start_sec=window_start,
            window_end_sec=window_end,
            note=note,
        ))
    return checkpoints


def draw_label(frame, box, text, color):
    x, y, w, h = box
    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    y_text = max(20, y1 - 8)
    cv2.putText(frame, text, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def infer_camera_id(video_path: Path, default_camera_ids: str) -> str:
    """Infer a stable camera id from file/folder names.

    Examples:
        cam1.mp4 -> cam1
        camera_2.mp4 -> camera_2
        video1.mp4 -> cam1
        angle3.mp4 -> cam3
    """
    stem = video_path.stem.lower()
    parent = video_path.parent.name.lower()
    for text in (stem, parent):
        clean = text.replace("-", "_").replace(" ", "_")
        tokens = clean.split("_")
        for token in tokens:
            if token.startswith("cam") or token.startswith("camera"):
                return token
        # Common demo naming: video1.mp4, video2.mp4, angle3.mp4
        for prefix in ("video", "angle", "view"):
            if clean.startswith(prefix):
                suffix = clean.replace(prefix, "", 1)
                if suffix.isdigit():
                    return f"cam{int(suffix)}"

    default_ids = [x.strip() for x in str(default_camera_ids).split(",") if x.strip()]
    if len(default_ids) > 1:
        return video_path.stem
    if len(default_ids) == 1 and video_path.stem.lower() not in {"video1", "video2", "video3"}:
        return default_ids[0]
    return video_path.stem


def build_camera_map(metas: list[VideoMeta], default_camera_ids: str) -> dict[str, str]:
    """Map every video to a camera id.

    Timetable CSV may say only `cam1`, but a folder can still contain
    video1.mp4, video2.mp4, video3.mp4 as three camera angles.
    This function prevents all three videos from being incorrectly labeled as cam1.
    """
    explicit_ids = [x.strip() for x in str(default_camera_ids).split(",") if x.strip()]
    sorted_metas = sorted(metas, key=lambda m: m.path.name.lower())
    mapping: dict[str, str] = {}
    used: set[str] = set()

    for idx, meta in enumerate(sorted_metas, start=1):
        inferred = infer_camera_id(meta.path, default_camera_ids)
        # If inference returns a generic file stem, prefer timetable ids or camN.
        generic = inferred == meta.path.stem
        if generic:
            if len(explicit_ids) >= idx:
                camera_id = explicit_ids[idx - 1]
            else:
                camera_id = f"cam{idx}"
        else:
            camera_id = inferred

        if camera_id in used:
            camera_id = f"cam{idx}"
        used.add(camera_id)
        mapping[meta.path.name] = camera_id
    return mapping


def build_slot_prefix(slot) -> str:
    pieces = [slot.slot_id, slot.course_abbr, slot.day, f"P{slot.period}"]
    return "_".join(clean_filename(str(piece).replace("/", "-")) for piece in pieces if str(piece).strip())



def checkpoint_match_keys(cp: Checkpoint) -> set[str]:
    """Return flexible folder-name keys for a checkpoint.

    Supports folders like:
        CP1/
        CP1_0910/
        0910/
        09_10/
        checkpoint_09_10/
    """
    time_raw = str(cp.class_time).strip()
    compact_time = time_raw.replace(":", "")
    underscored_time = time_raw.replace(":", "_")
    dashed_time = time_raw.replace(":", "-")
    return {
        cp.cp_id.lower(),
        cp.label.lower(),
        clean_filename(cp.label).lower(),
        clean_filename(time_raw).lower(),
        compact_time.lower(),
        underscored_time.lower(),
        dashed_time.lower(),
    }


def discover_checkpoint_clip_folders(video_dir: Path, checkpoints: list[Checkpoint]) -> dict[str, Path]:
    """Map checkpoint IDs to manually sampled clip folders.

    Expected testing/demo structure:
        cctv_videos/MON_P1/
            CP1_0910/cam1.mp4, cam2.mp4, cam3.mp4
            CP2_0920/cam1.mp4, cam2.mp4, cam3.mp4
            ...

    This prevents short 10-second clips from being split into fake checkpoints.
    """
    if not video_dir.exists():
        return {}
    child_dirs = [p for p in video_dir.iterdir() if p.is_dir()]
    folder_map: dict[str, Path] = {}
    for cp in checkpoints:
        keys = checkpoint_match_keys(cp)
        for folder in child_dirs:
            name = folder.name.lower().replace(" ", "_").replace("-", "_")
            compact_name = name.replace("_", "")
            has_videos = any(p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS for p in folder.iterdir())
            if not has_videos:
                continue
            matched = False
            for key in keys:
                norm_key = key.lower().replace(" ", "_").replace("-", "_")
                compact_key = norm_key.replace("_", "")
                if norm_key in name or compact_key in compact_name:
                    matched = True
                    break
            if matched:
                folder_map[cp.cp_id] = folder
                break
    return folder_map


def has_checkpoint_clip_folders(video_dir: Path, checkpoints: list[Checkpoint]) -> bool:
    return bool(discover_checkpoint_clip_folders(video_dir, checkpoints))

def video_metadata(video_path: Path) -> VideoMeta | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frames / fps if fps > 0 and frames > 0 else 0.0
    cap.release()
    return VideoMeta(video_path, fps, frames, duration)


def choose_checkpoint_mode(requested_mode: str, metas: list[VideoMeta], checkpoints: list[Checkpoint], video_dir: Path | None = None) -> str:
    if requested_mode != "auto":
        return requested_mode
    if video_dir is not None and has_checkpoint_clip_folders(video_dir, checkpoints):
        return "clip-folders"
    if not metas:
        return "class-time"
    required_end = max(cp.window_end_sec for cp in checkpoints)
    shortest = min(meta.duration_sec for meta in metas if meta.duration_sec > 0) if any(meta.duration_sec > 0 for meta in metas) else 0
    if shortest <= 0:
        return "class-time"
    if shortest + 1.0 < required_end:
        return "split-video"
    return "class-time"


def windows_for_video(meta: VideoMeta, checkpoints: list[Checkpoint], mode: str, fallback_clip_seconds: float) -> dict[str, tuple[float, float]]:
    if mode == "class-time" or meta.duration_sec <= 0:
        return {cp.cp_id: (cp.window_start_sec, cp.window_end_sec) for cp in checkpoints}

    # Split-video mode is mainly for demo/short clips. It divides each camera video into
    # the same number of logical checkpoint windows, while keeping the report labels as class times.
    n = len(checkpoints)
    windows: dict[str, tuple[float, float]] = {}
    if n <= 0:
        return windows
    if meta.duration_sec <= n:
        segment = meta.duration_sec / n
        for idx, cp in enumerate(checkpoints):
            start = idx * segment
            end = min(meta.duration_sec, (idx + 1) * segment)
            windows[cp.cp_id] = (start, end)
        return windows

    clip = min(float(fallback_clip_seconds), max(1.0, meta.duration_sec / n))
    if n == 1:
        starts = [max(0.0, (meta.duration_sec - clip) / 2.0)]
    else:
        starts = [(meta.duration_sec - clip) * (i / (n - 1)) for i in range(n)]
    for cp, start in zip(checkpoints, starts):
        windows[cp.cp_id] = (max(0.0, start), min(meta.duration_sec, start + clip))
    return windows


def seek_to_start(cap, fps: float, start_sec: float) -> int:
    frame_no = int(max(0, round(start_sec * fps))) if fps > 0 else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    return frame_no


def save_outputs(
    attendance_df: pd.DataFrame,
    detection_log_df: pd.DataFrame,
    slot_summary_df: pd.DataFrame,
    checkpoint_summary_df: pd.DataFrame,
    student_checkpoint_df: pd.DataFrame,
    camera_summary_df: pd.DataFrame,
    needs_review_df: pd.DataFrame,
    slot,
    output_dir: Path,
    output_layout: str = "organized",
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = build_slot_prefix(slot)

    if output_layout == "organized":
        run_dir = output_dir / datetime.now().strftime("%Y-%m-%d") / prefix
    else:
        run_dir = output_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "attendance_csv": run_dir / f"attendance_{prefix}_{timestamp}.csv",
        "detection_log_csv": run_dir / f"detection_log_{prefix}_{timestamp}.csv",
        "slot_summary_csv": run_dir / f"slot_summary_{prefix}_{timestamp}.csv",
        "checkpoint_summary_csv": run_dir / f"checkpoint_summary_{prefix}_{timestamp}.csv",
        "student_checkpoint_csv": run_dir / f"student_checkpoint_evidence_{prefix}_{timestamp}.csv",
        "camera_summary_csv": run_dir / f"camera_summary_{prefix}_{timestamp}.csv",
        "needs_review_csv": run_dir / f"needs_review_{prefix}_{timestamp}.csv",
        "excel_file": run_dir / f"attendance_{prefix}_{timestamp}.xlsx",
    }

    attendance_df.to_csv(paths["attendance_csv"], index=False)
    detection_log_df.to_csv(paths["detection_log_csv"], index=False)
    slot_summary_df.to_csv(paths["slot_summary_csv"], index=False)
    checkpoint_summary_df.to_csv(paths["checkpoint_summary_csv"], index=False)
    student_checkpoint_df.to_csv(paths["student_checkpoint_csv"], index=False)
    camera_summary_df.to_csv(paths["camera_summary_csv"], index=False)
    needs_review_df.to_csv(paths["needs_review_csv"], index=False)

    # Large detection logs make Excel very slow. Keep the CSV complete and cap the Excel sheet.
    excel_detection_log = detection_log_df
    if len(excel_detection_log) > 50000:
        excel_detection_log = excel_detection_log.head(50000).copy()
        excel_detection_log.loc[len(excel_detection_log)] = {
            col: "TRUNCATED_IN_EXCEL__SEE_FULL_DETECTION_LOG_CSV" for col in excel_detection_log.columns
        }

    with pd.ExcelWriter(paths["excel_file"], engine="openpyxl") as writer:
        slot_summary_df.to_excel(writer, sheet_name="Slot Summary", index=False)
        attendance_df.to_excel(writer, sheet_name="Final Attendance", index=False)
        needs_review_df.to_excel(writer, sheet_name="Needs Review", index=False)
        checkpoint_summary_df.to_excel(writer, sheet_name="Checkpoint Summary", index=False)
        student_checkpoint_df.to_excel(writer, sheet_name="Student Checkpoints", index=False)
        camera_summary_df.to_excel(writer, sheet_name="Camera Summary", index=False)
        excel_detection_log.to_excel(writer, sheet_name="Detection Log", index=False)

    return paths


def quality_label(recognition_rate: float, detected: int, coverage_rate: float = 0.0) -> str:
    """Label checkpoint quality.

    Accepted/detected rate alone is misleading in CCTV because the same faces are
    detected many times and strict thresholds intentionally reject many crops.
    Coverage rate = unique recognized students / enrolled students is a better
    checkpoint health signal for attendance.
    """
    if detected == 0:
        return "No Faces"
    if coverage_rate >= 55 or recognition_rate >= 35:
        return "Good"
    if coverage_rate >= 35 or recognition_rate >= 15:
        return "Medium"
    return "Poor"


def main() -> None:
    args = parse_args()
    run_start_perf = time.perf_counter()

    timetable_df = load_timetable(args.timetable)
    if args.list_slots:
        print(list_slots(timetable_df).to_string(index=False))
        return

    slot = select_slot(
        timetable_df,
        slot_id=args.slot_id,
        day=args.day,
        period=args.period,
        class_time=args.class_time,
    )

    video_dir = Path(args.video_dir)
    embeddings_path = Path(args.embeddings)
    output_dir = Path(args.output_dir)
    unknown_dir = Path(args.unknown_dir) / build_slot_prefix(slot)

    if not embeddings_path.exists():
        raise SystemExit(f"Embedding database not found: {embeddings_path}\nRun: python scripts/build_embeddings.py")
    if not video_dir.exists():
        raise SystemExit(f"Video folder not found: {video_dir}")

    ensure_dirs(output_dir)
    if args.save_unknown:
        ensure_dirs(unknown_dir)

    db = StudentEmbeddingDB.load(embeddings_path)
    engine = FaceEngine(
        YUNET_MODEL,
        SFACE_MODEL,
        detection_score=args.det_score,
        nms_threshold=DEFAULT_NMS_THRESHOLD,
        top_k=DEFAULT_TOP_K,
    )

    checkpoints = build_checkpoints(
        slot,
        settle_minutes=args.settle_minutes,
        every_minutes=args.checkpoint_every_minutes,
        clip_seconds=args.checkpoint_clip_seconds,
    )

    # MVP3 supports two kinds of input:
    # 1) full class camera recordings in the slot folder, or
    # 2) manually sampled checkpoint clip folders: CP1_0910/, CP2_0920/, ...
    checkpoint_clip_folders = discover_checkpoint_clip_folders(video_dir, checkpoints)

    root_video_files = iter_files(video_dir, VIDEO_EXTENSIONS)
    metas: list[VideoMeta] = []
    path_checkpoint_map: dict[str, str] = {}

    using_manual_clip_folders = args.checkpoint_mode == "clip-folders" or (args.checkpoint_mode == "auto" and checkpoint_clip_folders)

    if using_manual_clip_folders:
        if not checkpoint_clip_folders:
            raise SystemExit(
                "Expected structure: cctv_videos/<SLOT_ID>/CP1_0910/cam1.mp4 ... CP5_0950/cam3.mp4, "
                "or use --checkpoint-mode split-video for simple demo clips."
            )

        # Demo/testing support:
        # If only CP1_0910 exists, process only that available checkpoint instead of failing
        # or repeatedly warning about CP2-CP5. For full attendance, add all five CP folders.
        available_cp_ids = [cp.cp_id for cp in checkpoints if cp.cp_id in checkpoint_clip_folders]
        if 0 < len(available_cp_ids) < len(checkpoints):
            print(
                "Demo clip mode: using available checkpoint folders only: "
                + ", ".join(available_cp_ids)
                + ". Add CP1-CP5 folders for real 5-checkpoint attendance."
            )
            checkpoints = [cp for cp in checkpoints if cp.cp_id in checkpoint_clip_folders]
            if checkpoints:
                args.present_checkpoints = min(args.present_checkpoints, len(checkpoints))
                args.strong_checkpoints = min(args.strong_checkpoints, len(checkpoints))
                args.review_checkpoints = min(args.review_checkpoints, len(checkpoints))

        for cp in checkpoints:
            folder = checkpoint_clip_folders.get(cp.cp_id)
            if not folder:
                continue
            for video_path in iter_files(folder, VIDEO_EXTENSIONS):
                meta = video_metadata(video_path)
                if not meta:
                    print(f"Could not open video: {video_path}")
                    continue
                metas.append(meta)
                path_checkpoint_map[str(video_path.resolve())] = cp.cp_id
    else:
        if not root_video_files:
            raise SystemExit(f"No video files found in: {video_dir}")
        for video_path in root_video_files:
            meta = video_metadata(video_path)
            if not meta:
                print(f"Could not open video: {video_path}")
                continue
            metas.append(meta)

    if not metas:
        raise SystemExit("No readable video files found.")

    checkpoint_mode_used = choose_checkpoint_mode(args.checkpoint_mode, metas, checkpoints, video_dir=video_dir)
    video_camera_map = build_camera_map(metas, slot.camera_ids)

    attendance = CheckpointAttendanceBook(
        db.roll_numbers,
        checkpoints=checkpoints,
        checkpoint_min_detections=args.checkpoint_min_detections,
        present_checkpoints=args.present_checkpoints,
        strong_checkpoints=args.strong_checkpoints,
        review_checkpoints=args.review_checkpoints,
        low_confidence_score=args.low_confidence_score,
    )

    print("MVP 3 Checkpoint-Based YuNet + SFace Attendance Started")
    print(f"Slot: {slot.slot_id} | {slot.day} P{slot.period} | {slot.start_time}-{slot.end_time}")
    print(f"Subject: {slot.course_abbr} - {slot.course_name}")
    print(f"Instructor: {slot.instructor}")
    print(f"Room: {slot.room} | Section: {slot.section}")
    print(f"Students enrolled: {db.student_count}")
    print(f"Embeddings: {len(db)}")
    print(f"Camera videos: {len(metas)}")
    print("Camera mapping:")
    for video_name, camera_id in sorted(video_camera_map.items()):
        print(f"  {video_name} -> {camera_id}")
    print(f"Checkpoint mode: {checkpoint_mode_used}")
    print("Checkpoints:")
    for cp in checkpoints:
        print(f"  {cp.cp_id}: {cp.class_time} | class-window {seconds_to_text(cp.window_start_sec)} to {seconds_to_text(cp.window_end_sec)}")
    print(f"Match threshold: {args.match_threshold}")
    print(f"Margin threshold: {args.margin_threshold}")
    print(f"Checkpoint min detections: {args.checkpoint_min_detections}")
    print(f"Present rule: {args.present_checkpoints}/{len(checkpoints)} checkpoints")
    print(f"Sampling: {args.sample_fps} fps" if args.sample_fps and args.sample_fps > 0 else f"Frame skip: {args.frame_skip}")
    print(f"Detection log mode: {args.log_mode}")
    print()

    slot_dict = slot.to_dict()
    detection_log: list[dict] = []
    checkpoint_metrics: Dict[str, dict] = defaultdict(lambda: {"frames": 0, "detected": 0, "accepted": 0, "rejected": 0, "students": set()})
    camera_metrics: Dict[tuple, dict] = defaultdict(lambda: {"frames": 0, "detected": 0, "accepted": 0, "rejected": 0, "students": set()})
    unknown_saved_per_key: Dict[tuple, int] = defaultdict(int)
    unknown_saved = 0
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_processed_frames = 0
    total_skipped_windows = 0
    videos_processed: list[str] = []
    stop_requested = False

    for meta in metas:
        video_path = meta.path
        video_name = video_path.name
        camera_id = video_camera_map.get(video_name, infer_camera_id(video_path, slot.camera_ids))
        videos_processed.append(video_name)
        if checkpoint_mode_used == "clip-folders":
            mapped_cp_id = path_checkpoint_map.get(str(video_path.resolve()))
            cp_loop = [cp for cp in checkpoints if cp.cp_id == mapped_cp_id]
            clip_end = meta.duration_sec
            if args.checkpoint_clip_seconds and args.checkpoint_clip_seconds > 0 and meta.duration_sec > args.checkpoint_clip_seconds:
                clip_end = args.checkpoint_clip_seconds
            windows = {mapped_cp_id: (0.0, clip_end)} if mapped_cp_id else {}
        else:
            cp_loop = checkpoints
            windows = windows_for_video(meta, checkpoints, checkpoint_mode_used, args.checkpoint_clip_seconds)

        print(f"Processing camera angle: {video_name} | Camera: {camera_id} | Duration: {seconds_to_text(meta.duration_sec)}")

        for cp in cp_loop:
            start_sec, end_sec = windows.get(cp.cp_id, (cp.window_start_sec, cp.window_end_sec))
            if meta.duration_sec > 0 and start_sec >= meta.duration_sec:
                total_skipped_windows += 1
                print(f"  {cp.cp_id} skipped for {video_name}: window starts after video duration")
                continue
            if meta.duration_sec > 0:
                end_sec = min(end_sec, meta.duration_sec)
            if end_sec <= start_sec:
                total_skipped_windows += 1
                print(f"  {cp.cp_id} skipped for {video_name}: empty window")
                continue

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"  Could not open video during checkpoint: {video_path}")
                continue

            fps = meta.fps or cap.get(cv2.CAP_PROP_FPS) or 0.0
            current_frame = seek_to_start(cap, fps, start_sec)
            processed_frames_this_window = 0
            if args.sample_fps and args.sample_fps > 0 and fps > 0:
                effective_frame_step = max(1, int(round(fps / float(args.sample_fps))))
            else:
                effective_frame_step = max(1, int(args.frame_skip))

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                current_frame += 1
                time_sec = current_frame / fps if fps > 0 else 0.0
                if time_sec > end_sec:
                    break
                if effective_frame_step > 1 and current_frame % effective_frame_step != 0:
                    continue

                frame, _ = resize_keep_aspect(frame, args.max_width)
                processed_frames_this_window += 1
                total_processed_frames += 1
                checkpoint_metrics[cp.cp_id]["frames"] += 1
                camera_key = (cp.cp_id, camera_id, video_name)
                camera_metrics[camera_key]["frames"] += 1

                time_text = seconds_to_text(time_sec)
                detections = engine.detect_and_extract(frame)
                checkpoint_metrics[cp.cp_id]["detected"] += len(detections)
                camera_metrics[camera_key]["detected"] += len(detections)

                for face_idx, item in enumerate(detections, start=1):
                    box = item["box"]
                    face_score = item["score"]
                    match = db.match(
                        item["embedding"],
                        match_threshold=args.match_threshold,
                        margin_threshold=args.margin_threshold,
                        aggregate=args.aggregate,
                    )

                    accepted_roll = match.roll_no if match.accepted else "Unknown"
                    if match.accepted and match.roll_no:
                        evidence_time = f"{cp.cp_id}/{cp.class_time}/{camera_id}/{time_text}"
                        attendance.add_detection(match.roll_no, cp.cp_id, match.best_score, match.margin, video_name, camera_id, evidence_time)
                        checkpoint_metrics[cp.cp_id]["accepted"] += 1
                        checkpoint_metrics[cp.cp_id]["students"].add(match.roll_no)
                        camera_metrics[camera_key]["accepted"] += 1
                        camera_metrics[camera_key]["students"].add(match.roll_no)
                        color = (0, 255, 0)
                        text = f"{match.roll_no} {match.best_score:.2f} {cp.cp_id}"
                        count_for_roll = attendance.total_counts.get(match.roll_no, 0)
                    else:
                        checkpoint_metrics[cp.cp_id]["rejected"] += 1
                        camera_metrics[camera_key]["rejected"] += 1
                        color = (0, 0, 255)
                        text = f"Unknown {match.best_score:.2f} {cp.cp_id}"
                        count_for_roll = 0
                        if args.save_unknown:
                            save_key = (cp.cp_id, camera_id)
                            if unknown_saved_per_key[save_key] < args.max_unknown_per_camera_checkpoint:
                                crop = safe_crop(frame, box)
                                if crop is not None:
                                    unknown_saved_per_key[save_key] += 1
                                    unknown_saved += 1
                                    subdir = unknown_dir / cp.cp_id / clean_filename(camera_id)
                                    subdir.mkdir(parents=True, exist_ok=True)
                                    filename = f"unknown_{run_id}_{clean_filename(video_path.stem)}_f{current_frame}_d{face_idx}_{match.reason}.jpg"
                                    cv2.imwrite(str(subdir / filename), crop)

                    should_log = (
                        args.log_mode == "full"
                        or match.accepted
                        or (args.log_mode == "compact" and match.reason == "margin_too_small")
                    )
                    if should_log:
                        detection_log.append({
                            **slot_dict,
                            "Checkpoint_ID": cp.cp_id,
                            "Checkpoint_Label": cp.label,
                            "Checkpoint_Class_Time": cp.class_time,
                            "Checkpoint_Mode": checkpoint_mode_used,
                            "Video_Window_Start": seconds_to_text(start_sec),
                            "Video_Window_End": seconds_to_text(end_sec),
                            "Video": video_name,
                            "Camera_ID": camera_id,
                            "Frame": current_frame,
                            "Time_In_Video": time_text,
                            "Face_Index": face_idx,
                            "YuNet_Face_Score": round(face_score, 4),
                            "Accepted": "Yes" if match.accepted else "No",
                            "Predicted_Roll": accepted_roll,
                            "Best_Roll": match.roll_no or "",
                            "Best_Score": round(match.best_score, 4),
                            "Second_Roll": match.second_roll_no or "",
                            "Second_Score": round(match.second_score, 4),
                            "Margin": round(match.margin, 4),
                            "Reject_Reason": match.reason,
                            "Detection_Count_For_Roll": count_for_roll,
                        })

                    if args.display:
                        draw_label(frame, box, text, color)

                if args.display:
                    cv2.putText(frame, f"{slot.slot_id} | {slot.course_abbr} | {cp.cp_id} {cp.class_time}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
                    cv2.putText(frame, f"Camera: {camera_id} | Video: {video_name}", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
                    cv2.putText(frame, "Press q to stop", (20, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
                    cv2.imshow("MVP3 Checkpoint Attendance", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        stop_requested = True
                        break

            cap.release()
            print(f"  {cp.cp_id} {cp.class_time}: processed {processed_frames_this_window} frames")
            if stop_requested:
                break
        if stop_requested:
            break

    if args.display:
        cv2.destroyAllWindows()

    attendance_df = attendance.to_attendance_dataframe(slot_dict)
    student_checkpoint_df = attendance.to_student_checkpoint_dataframe(slot_dict)
    detection_log_df = pd.DataFrame(detection_log)
    camera_summary_df = attendance.camera_summary_dataframe(slot_dict, camera_metrics)
    needs_review_df = attendance_df[
        attendance_df["Final_Status"].eq("Needs Review")
        | attendance_df["Flags"].fillna("").astype(str).ne("")
    ].copy()

    checkpoint_rows = []
    for cp in checkpoints:
        metrics = checkpoint_metrics[cp.cp_id]
        detected = int(metrics.get("detected", 0))
        accepted = int(metrics.get("accepted", 0))
        rejected = int(metrics.get("rejected", 0))
        recognition_rate = round((accepted / detected) * 100, 1) if detected else 0
        unique_students = len(metrics.get("students", set()))
        coverage_rate = round((unique_students / db.student_count) * 100, 1) if db.student_count else 0
        checkpoint_rows.append({
            **slot_dict,
            "Checkpoint_ID": cp.cp_id,
            "Checkpoint_Label": cp.label,
            "Checkpoint_Class_Time": cp.class_time,
            "Checkpoint_Note": cp.note,
            "Checkpoint_Mode": checkpoint_mode_used,
            "Class_Window_Start": seconds_to_text(cp.window_start_sec),
            "Class_Window_End": seconds_to_text(cp.window_end_sec),
            "Frames_Processed": int(metrics.get("frames", 0)),
            "Faces_Detected": detected,
            "Accepted_Recognitions": accepted,
            "Rejected_Or_Unknown": rejected,
            "Unique_Students_Recognized": unique_students,
            "Recognition_Rate": recognition_rate,
            "Student_Coverage_Rate": coverage_rate,
            "Quality": quality_label(recognition_rate, detected, coverage_rate),
        })
    checkpoint_summary_df = pd.DataFrame(checkpoint_rows)

    present_count = int(attendance_df["Final_Status"].isin(["Present", "Present Strong"]).sum())
    strong_count = int((attendance_df["Final_Status"] == "Present Strong").sum())
    review_count = int((attendance_df["Final_Status"] == "Needs Review").sum())
    absent_count = int((attendance_df["Final_Status"] == "Absent").sum())
    total_detections = int(sum(int(m.get("detected", 0)) for m in checkpoint_metrics.values()))
    accepted_count = int(sum(int(m.get("accepted", 0)) for m in checkpoint_metrics.values()))
    rejected_count = int(sum(int(m.get("rejected", 0)) for m in checkpoint_metrics.values()))
    poor_checkpoints = checkpoint_summary_df[checkpoint_summary_df["Quality"] == "Poor"]["Checkpoint_ID"].tolist() if not checkpoint_summary_df.empty else []

    slot_summary_df = pd.DataFrame([{
        **slot_dict,
        "Run_Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Processing_Runtime_Seconds": round(time.perf_counter() - run_start_perf, 2),
        "MVP_Level": "MVP3_1_Optimized_Checkpoint_Voting",
        "Video_Dir": str(video_dir),
        "Videos_Processed": ", ".join(videos_processed),
        "Camera_Angles_Processed": len(metas),
        "Camera_Map": "; ".join(f"{video}->{cam}" for video, cam in sorted(video_camera_map.items())),
        "Checkpoint_Mode": checkpoint_mode_used,
        "Manual_Checkpoint_Clip_Folders": "; ".join(f"{cp_id}:{path.name}" for cp_id, path in sorted(checkpoint_clip_folders.items())),
        "Total_Checkpoints": len(checkpoints),
        "Checkpoint_Times": ", ".join(cp.class_time for cp in checkpoints),
        "Settle_Minutes": args.settle_minutes,
        "Checkpoint_Every_Minutes": args.checkpoint_every_minutes,
        "Checkpoint_Clip_Seconds": args.checkpoint_clip_seconds,
        "Total_Students": db.student_count,
        "Frames_Processed": total_processed_frames,
        "Skipped_Windows": total_skipped_windows,
        "Total_Face_Detections": total_detections,
        "Accepted_Recognitions": accepted_count,
        "Rejected_Or_Unknown": rejected_count,
        "Students_With_At_Least_One_Detection": int((attendance_df["Total_Accepted_Detections"] > 0).sum()),
        "Students_Present": present_count,
        "Students_Present_Strong": strong_count,
        "Students_Needs_Review": review_count,
        "Students_Absent": absent_count,
        "Poor_Checkpoints": ", ".join(poor_checkpoints),
        "Unknown_Faces_Saved": unknown_saved if args.save_unknown else 0,
        "Match_Threshold": args.match_threshold,
        "Margin_Threshold": args.margin_threshold,
        "Checkpoint_Min_Detections": args.checkpoint_min_detections,
        "Present_Checkpoints_Rule": args.present_checkpoints,
        "Strong_Checkpoints_Rule": args.strong_checkpoints,
        "Review_Checkpoints_Rule": args.review_checkpoints,
        "Frame_Skip": args.frame_skip,
        "Sample_FPS": args.sample_fps,
        "Effective_Log_Mode": args.log_mode,
        "Output_Layout": args.output_layout,
        "Aggregate_Mode": args.aggregate,
    }])

    paths = save_outputs(
        attendance_df=attendance_df,
        detection_log_df=detection_log_df,
        slot_summary_df=slot_summary_df,
        checkpoint_summary_df=checkpoint_summary_df,
        student_checkpoint_df=student_checkpoint_df,
        camera_summary_df=camera_summary_df,
        needs_review_df=needs_review_df,
        slot=slot,
        output_dir=output_dir,
        output_layout=args.output_layout,
    )

    print("\nMVP 3 checkpoint attendance completed")
    print(f"Subject: {slot.course_abbr} - {slot.course_name}")
    print(f"Slot: {slot.day} P{slot.period} | {slot.start_time}-{slot.end_time}")
    print(f"Present: {present_count} / {db.student_count}")
    print(f"Present Strong: {strong_count}")
    print(f"Needs Review: {review_count}")
    print(f"Absent: {absent_count}")
    print(f"Checkpoint mode used: {checkpoint_mode_used}")
    if poor_checkpoints:
        print(f"Poor checkpoints: {', '.join(poor_checkpoints)}")
    print(f"Runtime: {time.perf_counter() - run_start_perf:.2f} seconds")
    print(f"Unknown/rejected faces saved: {unknown_saved}" if args.save_unknown else "Unknown saving: off")
    for name, path in paths.items():
        print(f"Saved {name}: {path}")


if __name__ == "__main__":
    main()
