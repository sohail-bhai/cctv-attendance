from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Set, Tuple

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
from src.face_attendance.diagnostics import (
    checkpoint_window_warnings,
    crop_quality_metrics,
    diagnostic_quality_label,
    diagnostic_rejection_reason,
    face_geometry,
    json_safe,
    make_diagnostic_run_id,
    percentile_fields,
    should_log_detection,
)
from src.face_attendance.embedding_db import MatchResult, StudentEmbeddingDB
from src.face_attendance.face_engine import FaceEngine
from src.face_attendance.processing_integration import (
    POLICY_VERSION as PROCESSING_CONTRACT_POLICY_VERSION,
    ProcessingIntegrationError,
    build_processing_metadata,
    filter_embedding_records_to_roster,
    inspect_exact_checkpoint_layout,
    load_subject_roster,
)
from src.face_attendance.run_quality import evaluate_run_quality, quality_to_csv_fields, review_queue_reason
from src.face_attendance.timetable import list_slots, load_timetable, parse_time, select_slot
from src.face_attendance.utils import clean_filename, ensure_dirs, iter_files, resize_keep_aspect, safe_crop
from src.face_attendance.zones import (
    DetectionCandidate,
    ZoneConfigError,
    bounds_text,
    face_box_text,
    load_camera_zone_config,
    map_resized_face_to_original,
    map_zone_face_to_original,
    merge_detection_candidates,
    source_summary,
    upscale_zone_crop,
    zone_pixel_bounds,
)

if TYPE_CHECKING:
    from src.face_attendance.tracklets import FinalizedTracklet, TrackletConfig, TrackletObservation


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
        tracklet_evidence_enabled: bool = False,
    ) -> None:
        self.all_students = sorted(all_students)
        self.checkpoints = checkpoints
        self.cp_ids = [cp.cp_id for cp in checkpoints]
        self.checkpoint_min_detections = int(checkpoint_min_detections)
        self.present_checkpoints = int(present_checkpoints)
        self.strong_checkpoints = int(strong_checkpoints)
        self.review_checkpoints = int(review_checkpoints)
        self.low_confidence_score = float(low_confidence_score)
        self.tracklet_evidence_enabled = bool(tracklet_evidence_enabled)

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
        self.tracklet_confirmed: Dict[str, Set[str]] = defaultdict(set)

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

    def add_tracklet(
        self,
        roll_no: str,
        checkpoint_id: str,
        score: float,
        margin: float,
        video_name: str,
        camera_id: str,
        evidence_time: str,
    ) -> None:
        if not self.tracklet_evidence_enabled:
            raise RuntimeError("Tracklet evidence is not enabled for this attendance book")
        self.add_detection(roll_no, checkpoint_id, score, margin, video_name, camera_id, evidence_time)
        self.tracklet_confirmed[roll_no].add(checkpoint_id)

    def recognized_checkpoint_ids(self, roll_no: str) -> list[str]:
        recognized = []
        for cp_id in self.cp_ids:
            if (
                cp_id in self.tracklet_confirmed.get(roll_no, set())
                or self.cp_counts[roll_no].get(cp_id, 0) >= self.checkpoint_min_detections
            ):
                recognized.append(cp_id)
        return recognized

    def checkpoint_symbols(self, roll_no: str) -> dict[str, str]:
        symbols = {}
        for cp_id in self.cp_ids:
            count = self.cp_counts[roll_no].get(cp_id, 0)
            if cp_id in self.tracklet_confirmed.get(roll_no, set()) or count >= self.checkpoint_min_detections:
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
                if self.tracklet_evidence_enabled:
                    row[f"{cp.cp_id}_Tracklet_Confirmed"] = (
                        "Yes" if cp.cp_id in self.tracklet_confirmed.get(roll, set()) else "No"
                    )
            rows.append(row)
        return pd.DataFrame(rows)

    def to_student_checkpoint_dataframe(self, slot_dict: dict) -> pd.DataFrame:
        rows = []
        for roll in self.all_students:
            for cp in self.checkpoints:
                scores = self.cp_scores[roll].get(cp.cp_id, [])
                margins = self.cp_margins[roll].get(cp.cp_id, [])
                count = self.cp_counts[roll].get(cp.cp_id, 0)
                tracklet_confirmed = cp.cp_id in self.tracklet_confirmed.get(roll, set())
                row = {
                    **slot_dict,
                    "Roll_Number": roll,
                    "Checkpoint_ID": cp.cp_id,
                    "Checkpoint_Label": cp.label,
                    "Checkpoint_Class_Time": cp.class_time,
                    "Recognized_In_Checkpoint": "Yes" if tracklet_confirmed or count >= self.checkpoint_min_detections else "No",
                    "Checkpoint_Detections": count,
                    "Checkpoint_Min_Detections": self.checkpoint_min_detections,
                    "Best_Score": round(max(scores), 4) if scores else "",
                    "Average_Score": round(sum(scores) / len(scores), 4) if scores else "",
                    "Average_Margin": round(sum(margins) / len(margins), 4) if margins else "",
                    "Cameras_Seen": ", ".join(sorted(self.cp_cameras[roll].get(cp.cp_id, set()))),
                    "Videos_Seen": ", ".join(sorted(self.cp_videos[roll].get(cp.cp_id, set()))),
                }
                if self.tracklet_evidence_enabled:
                    row["Tracklet_Confirmed"] = "Yes" if tracklet_confirmed else "No"
                    row["Evidence_Unit"] = "tracklet_aggregate"
                rows.append(row)
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

    parser.add_argument("--session-id", default="", help="Logical attendance session id, e.g. 2026-06-30__B51__P1__CVO")
    parser.add_argument("--session-date", default="", help="Actual attendance date in YYYY-MM-DD")
    parser.add_argument("--subject-abbr", default="", help="Logical subject/course track for this run, e.g. CVO")
    parser.add_argument("--subject-name", default="", help="Logical subject/course name for this run")
    parser.add_argument("--course-code", default="", help="Logical course code for this run")
    parser.add_argument("--faculty-id", default="", help="Faculty id for this logical session")
    parser.add_argument("--faculty-name", default="", help="Faculty name for this logical session")
    parser.add_argument("--input-slot", default="", help="Physical input slot folder id, e.g. TUE_P1")
    parser.add_argument("--input-source-type", default="", help="Physical input source type")
    parser.add_argument("--input-source-path", default="", help="Physical input source path")
    parser.add_argument("--student-map", default=str(ROOT / "data" / "student_faculty_map.json"), help="Authoritative subject roster mapping JSON")
    parser.add_argument("--require-authoritative-roster", action="store_true", help="Fail closed unless the selected subject has an authoritative roster; restrict matching and report rows to that roster")
    parser.add_argument("--processing-contract-id", default="", help="Expected Product Phase 2G processing contract id")

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
    parser.add_argument("--diagnostic", action="store_true", help="Write per-face and checkpoint diagnostic evidence without changing recognition decisions")
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="Write diagnostics only; do not write attendance, report, or correction outputs",
    )
    parser.add_argument("--diagnostic-dir", default="", help="Diagnostic output folder. Default: <output-dir>/diagnostics")
    parser.add_argument(
        "--diagnostic-run-id",
        default="",
        help="Explicit unique diagnostic run id for an orchestrated shadow run",
    )
    parser.add_argument("--zone-mode", choices=["off", "compare", "zones"], default="off", help="Camera-zone processing: off=current full-frame path, compare=diagnostic comparison only, zones=use merged zone detections")
    parser.add_argument("--camera-zones", default=str(ROOT / "data" / "camera_zones.json"), help="Versioned camera zone configuration JSON")
    parser.add_argument("--zone-profile", default="auto", help="Zone profile id or auto to resolve from camera/video name")
    parser.add_argument("--zone-merge-iou", type=float, default=0.20, help="IoU threshold for spatial duplicate merging")
    parser.add_argument("--tracklet-mode", choices=["off", "compare", "tracklets"], default="off", help="Multi-frame processing: off=current frame path, compare=diagnostics only, tracklets=use aggregate tracklet matches")
    parser.add_argument("--tracklet-min-observations", type=int, default=2, help="Minimum consistent observations required for an eligible tracklet")
    parser.add_argument("--tracklet-max-selected", type=int, default=5, help="Maximum highest-quality embeddings aggregated per tracklet")
    parser.add_argument("--tracklet-max-gap-seconds", type=float, default=1.5, help="Maximum temporal gap between associated face observations")
    parser.add_argument("--tracklet-min-iou", type=float, default=0.10, help="Minimum IoU for direct tracklet association")
    parser.add_argument("--tracklet-max-center-ratio", type=float, default=1.25, help="Maximum center displacement relative to face size when IoU is low")
    parser.add_argument("--tracklet-min-size-ratio", type=float, default=0.50, help="Minimum face-area ratio allowed during association")
    parser.add_argument("--tracklet-min-embedding-similarity", type=float, default=0.25, help="Minimum within-track cosine similarity to the selected medoid")
    parser.add_argument("--export-tracklet-review", action="store_true", help="Export a blind human-review package for accepted tracklets")
    parser.add_argument("--tracklet-review-dir", default="", help="Review package parent directory. Default: current diagnostic run")
    parser.add_argument("--tracklet-review-student-map", default=str(ROOT / "data" / "student_faculty_map.json"), help="Student mapping used by the blind reviewer")
    parser.add_argument("--tracklet-review-evidence-count", type=int, default=5, help="Maximum selected evidence views per accepted tracklet")
    parser.add_argument("--tracklet-review-crop-padding", type=float, default=0.45, help="Face-box padding ratio for blind review crops")
    parser.add_argument("--output-layout", choices=["flat", "organized"], default="organized", help="flat saves directly in attendance_output; organized saves in attendance_output/YYYY-MM-DD/SLOT/")
    parser.add_argument("--save-unknown", action="store_true", help="Save rejected/unknown face crops")
    parser.add_argument("--max-unknown-per-camera-checkpoint", type=int, default=20, help="Storage safety limit for unknown crops")
    parser.add_argument("--display", action="store_true", help="Show live video window. Press q to stop")
    return parser.parse_args()


def validate_tracklet_review_args(args) -> None:
    if not args.export_tracklet_review:
        return
    if not args.diagnostic:
        raise SystemExit("--export-tracklet-review requires --diagnostic")
    if args.tracklet_mode == "off":
        raise SystemExit("--export-tracklet-review requires --tracklet-mode compare or tracklets")
    if not 1 <= args.tracklet_review_evidence_count <= 10:
        raise SystemExit("--tracklet-review-evidence-count must be between 1 and 10")
    if not 0 <= args.tracklet_review_crop_padding <= 2:
        raise SystemExit("--tracklet-review-crop-padding must be between 0 and 2")


def validate_diagnostic_only_args(args) -> None:
    if not args.diagnostic_only:
        return
    if not args.diagnostic:
        raise SystemExit("--diagnostic-only requires --diagnostic")
    if args.tracklet_mode != "compare":
        raise SystemExit("--diagnostic-only shadow runs require --tracklet-mode compare")
    if args.save_unknown:
        raise SystemExit("--diagnostic-only cannot save unknown faces")
    if args.export_tracklet_review:
        raise SystemExit("--diagnostic-only review export must use the separate shadow validator")
    if args.diagnostic_run_id and clean_filename(args.diagnostic_run_id) != args.diagnostic_run_id:
        raise SystemExit("--diagnostic-run-id must be a safe filename component")


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



_CHECKPOINT_FOLDER_ID_RE = re.compile(r"(?:^|[^a-z0-9])cp0*(\d+)(?=$|[^a-z0-9])", re.IGNORECASE)
_CHECKPOINT_FOLDER_TIME_RE = re.compile(r"(?:^|[^0-9])([0-2]\d[0-5]\d)(?=$|[^0-9])")


def checkpoint_match_keys(cp: Checkpoint) -> set[str]:
    """Return exact normalized aliases for a checkpoint folder.

    These aliases are metadata only. Folder discovery deliberately avoids
    substring matching because compact times can overlap. For example,
    ``1110`` is a substring of ``CP1_1100`` after punctuation removal and
    previously caused CP2 to reuse the CP1 folder.
    """
    time_raw = str(cp.class_time).strip()
    compact_time = re.sub(r"[^0-9]", "", time_raw)
    normalized_label = re.sub(r"[^a-z0-9]+", "_", str(cp.label).lower()).strip("_")
    return {
        str(cp.cp_id).upper(),
        normalized_label,
        compact_time,
    }


def _folder_checkpoint_ids(folder_name: str) -> set[str]:
    return {
        f"CP{int(match.group(1))}"
        for match in _CHECKPOINT_FOLDER_ID_RE.finditer(str(folder_name))
    }


def _folder_time_codes(folder_name: str) -> set[str]:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(folder_name).lower()).strip("_")
    codes = {match.group(1) for match in _CHECKPOINT_FOLDER_TIME_RE.finditer(normalized)}
    tokens = [token for token in normalized.split("_") if token]
    for left, right in zip(tokens, tokens[1:]):
        if len(left) == 2 and len(right) == 2 and left.isdigit() and right.isdigit():
            hour = int(left)
            minute = int(right)
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                codes.add(f"{hour:02d}{minute:02d}")
    return codes


def _checkpoint_time_code(cp: Checkpoint) -> str:
    code = re.sub(r"[^0-9]", "", str(cp.class_time))
    if len(code) != 4:
        raise ValueError(f"Checkpoint {cp.cp_id} has invalid class time: {cp.class_time!r}")
    return code


def discover_checkpoint_clip_folders(video_dir: Path, checkpoints: list[Checkpoint]) -> dict[str, Path]:
    """Map checkpoint IDs to unique manually sampled clip folders.

    Explicit ``CP<n>`` tokens take precedence. Time-only folders are matched
    using exact four-digit time tokens. A folder can never be silently reused
    for another checkpoint, and ambiguous or contradictory names fail closed.
    """
    if not video_dir.exists():
        return {}

    checkpoint_by_id = {str(cp.cp_id).upper(): cp for cp in checkpoints}
    checkpoint_by_time: dict[str, list[Checkpoint]] = defaultdict(list)
    for cp in checkpoints:
        checkpoint_by_time[_checkpoint_time_code(cp)].append(cp)

    child_dirs = sorted(
        (path for path in video_dir.iterdir() if path.is_dir()),
        key=lambda path: path.name.lower(),
    )
    folder_map: dict[str, Path] = {}
    claimed_folders: dict[Path, str] = {}

    for folder in child_dirs:
        if not any(
            path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
            for path in folder.iterdir()
        ):
            continue

        explicit_ids = _folder_checkpoint_ids(folder.name)
        known_explicit = sorted(explicit_ids.intersection(checkpoint_by_id))
        time_codes = _folder_time_codes(folder.name)

        matched: Checkpoint | None = None
        if known_explicit:
            if len(known_explicit) != 1:
                raise ValueError(
                    f"Checkpoint clip folder names multiple checkpoints: "
                    f"{folder.name} -> {known_explicit}"
                )
            matched = checkpoint_by_id[known_explicit[0]]
            expected_time = _checkpoint_time_code(matched)
            known_times = sorted(code for code in time_codes if code in checkpoint_by_time)
            if known_times and expected_time not in known_times:
                raise ValueError(
                    f"Checkpoint clip folder ID/time conflict: {folder.name} declares "
                    f"{matched.cp_id} but time token(s) {known_times}"
                )
        elif time_codes:
            candidates = {
                cp
                for code in time_codes
                for cp in checkpoint_by_time.get(code, [])
            }
            if len(candidates) > 1:
                raise ValueError(
                    f"Checkpoint clip folder time is ambiguous: {folder.name}"
                )
            if len(candidates) == 1:
                matched = next(iter(candidates))

        if matched is None:
            continue

        cp_id = str(matched.cp_id).upper()
        resolved_folder = folder.resolve()
        if cp_id in folder_map and folder_map[cp_id].resolve() != resolved_folder:
            raise ValueError(
                f"Multiple clip folders map to {cp_id}: "
                f"{folder_map[cp_id].name}, {folder.name}"
            )
        if resolved_folder in claimed_folders and claimed_folders[resolved_folder] != cp_id:
            raise ValueError(
                f"Clip folder {folder.name} was assigned to both "
                f"{claimed_folders[resolved_folder]} and {cp_id}"
            )
        folder_map[cp_id] = folder
        claimed_folders[resolved_folder] = cp_id

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
    output_prefix: str | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = clean_filename(output_prefix) if output_prefix else build_slot_prefix(slot)

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


def numeric_bucket(value: Any, bins: list[tuple[float, float, str]]) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if not math.isfinite(number):
        return "unknown"
    for low, high, label in bins:
        if low <= number < high:
            return label
    return bins[-1][2] if bins else "unknown"


def build_checkpoint_diagnostic_rows(slot_dict: dict, window_metrics: dict) -> list[dict]:
    rows = []
    for key in sorted(window_metrics):
        metrics = window_metrics[key]
        total_faces = int(metrics.get("Total_Faces", 0))
        accepted = int(metrics.get("Accepted", 0))
        row = {
            **slot_dict,
            "Diagnostic_Run_ID": metrics.get("Diagnostic_Run_ID", ""),
            "Checkpoint_ID": metrics.get("Checkpoint_ID", ""),
            "Checkpoint_Label": metrics.get("Checkpoint_Label", ""),
            "Checkpoint_Class_Time": metrics.get("Checkpoint_Class_Time", ""),
            "Checkpoint_Mode": metrics.get("Checkpoint_Mode", ""),
            "Source_File": metrics.get("Source_File", ""),
            "Source_File_Name": metrics.get("Source_File_Name", ""),
            "Camera_ID": metrics.get("Camera_ID", ""),
            "Source_Duration_Seconds": metrics.get("Source_Duration_Seconds", ""),
            "Source_FPS": metrics.get("Source_FPS", ""),
            "Source_Frame_Count": metrics.get("Source_Frame_Count", ""),
            "Requested_Class_Window_Start": metrics.get("Requested_Class_Window_Start", ""),
            "Requested_Class_Window_End": metrics.get("Requested_Class_Window_End", ""),
            "Selected_Video_Window_Start": metrics.get("Selected_Video_Window_Start", ""),
            "Selected_Video_Window_End": metrics.get("Selected_Video_Window_End", ""),
            "Actual_First_Processed_Timestamp": metrics.get("Actual_First_Processed_Timestamp", ""),
            "Actual_Last_Processed_Timestamp": metrics.get("Actual_Last_Processed_Timestamp", ""),
            "Sampled_Frames": int(metrics.get("Sampled_Frames", 0)),
            "Frames_With_Faces": int(metrics.get("Frames_With_Faces", 0)),
            "Total_Faces": total_faces,
            "Accepted_Recognitions": accepted,
            "Rejected_Or_Unknown": int(metrics.get("Rejected", 0)),
            "Recognition_Rate": round((accepted / total_faces) * 100, 1) if total_faces else 0,
            "Unique_Accepted_Identities": len(metrics.get("Unique_Accepted", set())),
            "Window_Valid": metrics.get("Window_Valid", ""),
            "Timing_Warnings": "; ".join(metrics.get("Timing_Warnings", [])),
        }
        row.update(percentile_fields(metrics.get("Face_Widths", []), "Face_Width", decimals=2))
        row.update(percentile_fields(metrics.get("Face_Heights", []), "Face_Height", decimals=2))
        row.update(percentile_fields(metrics.get("Face_Area_Ratios", []), "Face_Area_Ratio", decimals=8))
        row.update(percentile_fields(metrics.get("Blur_Scores", []), "Blur_Laplacian_Variance", decimals=2))
        row.update(percentile_fields(metrics.get("Detector_Scores", []), "Detector_Score", decimals=4))
        row.update(percentile_fields(metrics.get("Best_Scores", []), "Best_Score", decimals=4))
        row.update(percentile_fields(metrics.get("Margins", []), "Margin", decimals=4))
        rows.append(row)
    return rows


def build_rejection_summary(face_diagnostics_df: pd.DataFrame) -> pd.DataFrame:
    if face_diagnostics_df.empty:
        return pd.DataFrame()

    df = face_diagnostics_df.copy()
    df["Best_Score_Range"] = df["Best_Score"].apply(lambda v: numeric_bucket(v, [
        (-999.0, 0.30, "<0.30"),
        (0.30, 0.40, "0.30-0.40"),
        (0.40, 0.48, "0.40-0.48"),
        (0.48, 0.60, "0.48-0.60"),
        (0.60, 999.0, ">=0.60"),
    ]))
    df["Margin_Range"] = df["Margin"].apply(lambda v: numeric_bucket(v, [
        (-999.0, 0.02, "<0.02"),
        (0.02, 0.05, "0.02-0.05"),
        (0.05, 0.08, "0.05-0.08"),
        (0.08, 0.15, "0.08-0.15"),
        (0.15, 999.0, ">=0.15"),
    ]))
    df["Face_Size_Range"] = df["Face_Width"].apply(lambda v: numeric_bucket(v, [
        (-999.0, 24, "<24px"),
        (24, 32, "24-32px"),
        (32, 48, "32-48px"),
        (48, 80, "48-80px"),
        (80, 9999, ">=80px"),
    ]))
    df["Blur_Range"] = df["Blur_Laplacian_Variance"].apply(lambda v: numeric_bucket(v, [
        (-999.0, 25, "<25"),
        (25, 75, "25-75"),
        (75, 150, "75-150"),
        (150, 999999, ">=150"),
    ]))
    grouped = df.groupby(
        [
            "Diagnostic_Reject_Reason",
            "Checkpoint_ID",
            "Camera_ID",
            "Best_Score_Range",
            "Margin_Range",
            "Face_Size_Range",
            "Blur_Range",
        ],
        dropna=False,
    )
    rows = []
    for group_key, group_df in grouped:
        reason, cp_id, camera_id, score_range, margin_range, face_range, blur_range = group_key
        rows.append({
            "Diagnostic_Reject_Reason": reason,
            "Checkpoint_ID": cp_id,
            "Camera_ID": camera_id,
            "Best_Score_Range": score_range,
            "Margin_Range": margin_range,
            "Face_Size_Range": face_range,
            "Blur_Range": blur_range,
            "Count": len(group_df),
            "Accepted_Count": int(group_df["Accepted"].eq("Yes").sum()) if "Accepted" in group_df else 0,
            "Median_Best_Score": round(float(pd.to_numeric(group_df["Best_Score"], errors="coerce").median()), 4) if "Best_Score" in group_df else "",
            "Median_Margin": round(float(pd.to_numeric(group_df["Margin"], errors="coerce").median()), 4) if "Margin" in group_df else "",
            "Median_Face_Width": round(float(pd.to_numeric(group_df["Face_Width"], errors="coerce").median()), 2) if "Face_Width" in group_df else "",
            "Median_Blur": round(float(pd.to_numeric(group_df["Blur_Laplacian_Variance"], errors="coerce").median()), 2) if "Blur_Laplacian_Variance" in group_df else "",
        })
    return pd.DataFrame(rows).sort_values(["Count"], ascending=False)


def dataframe_records(df: pd.DataFrame, limit: int | None = None) -> list[dict]:
    if df.empty:
        return []
    records = df.head(limit).to_dict(orient="records") if limit else df.to_dict(orient="records")
    return json_safe(records)


def save_diagnostic_outputs(
    face_diagnostics_df: pd.DataFrame,
    checkpoint_diagnostics_df: pd.DataFrame,
    rejection_summary_df: pd.DataFrame,
    diagnostic_summary: dict,
    diagnostic_root: Path,
    diagnostic_run_id: str,
) -> dict:
    diagnostic_root.mkdir(parents=True, exist_ok=False)
    paths = {
        "face_diagnostics_csv": diagnostic_root / f"face_diagnostics_{diagnostic_run_id}.csv",
        "checkpoint_diagnostics_csv": diagnostic_root / f"checkpoint_diagnostics_{diagnostic_run_id}.csv",
        "rejection_summary_csv": diagnostic_root / f"rejection_summary_{diagnostic_run_id}.csv",
        "diagnostic_summary_json": diagnostic_root / f"diagnostic_summary_{diagnostic_run_id}.json",
    }
    face_diagnostics_df.to_csv(paths["face_diagnostics_csv"], index=False)
    checkpoint_diagnostics_df.to_csv(paths["checkpoint_diagnostics_csv"], index=False)
    rejection_summary_df.to_csv(paths["rejection_summary_csv"], index=False)
    with paths["diagnostic_summary_json"].open("w", encoding="utf-8") as f:
        json.dump(json_safe(diagnostic_summary), f, indent=2)
    return paths


def detection_quality_rank(face_original, raw_frame, detector_score: float, source_priority: int) -> tuple:
    geom = face_geometry(face_original, raw_frame.shape)
    crop_quality = crop_quality_metrics(raw_frame, tuple(float(v) for v in face_original[:4]))
    label, _ = diagnostic_quality_label(geom, crop_quality, detector_score)
    label_rank = {"usable_reference": 2, "marginal_reference": 1, "unusable_reference": 0}.get(label, 0)
    landmark_rank = 1 if geom.get("Landmark_Valid") == "Yes" else 0
    blur = float(crop_quality.get("Blur_Laplacian_Variance") or 0.0)
    return (
        landmark_rank,
        label_rank,
        float(geom.get("Face_Area") or 0.0),
        min(blur, 50000.0),
        float(detector_score),
        int(source_priority),
    )


def evaluate_detection_candidate(
    candidate: DetectionCandidate,
    engine: FaceEngine,
    db: StudentEmbeddingDB,
    match_threshold: float,
    margin_threshold: float,
    aggregate: str,
) -> dict:
    embedding = engine.extract_feature(candidate.recognition_frame, candidate.recognition_face)
    embedding_success = embedding is not None
    if embedding_success:
        match = db.match(
            embedding,
            match_threshold=match_threshold,
            margin_threshold=margin_threshold,
            aggregate=aggregate,
        )
    else:
        match = MatchResult(False, None, 0.0, None, 0.0, 0.0, "embedding_extraction_failure")
    diagnostic_reason = diagnostic_rejection_reason(
        match.accepted,
        match.reason,
        embedding_success,
        match.best_score,
        match.margin,
        match_threshold,
        margin_threshold,
    )
    return {
        "embedding": embedding,
        "embedding_success": embedding_success,
        "match": match,
        "diagnostic_reason": diagnostic_reason,
    }


def build_zone_candidate_row(
    slot_dict: dict,
    detection_set: str,
    candidate: DetectionCandidate,
    eval_result: dict,
    raw_frame,
    cp,
    camera_id: str,
    video_path: Path,
    current_frame: int,
    time_text: str,
    match_threshold: float,
    margin_threshold: float,
    aggregate: str,
) -> dict:
    geom = face_geometry(candidate.face_original, raw_frame.shape)
    crop_quality = crop_quality_metrics(raw_frame, tuple(float(v) for v in candidate.face_original[:4]))
    label, reasons = diagnostic_quality_label(geom, crop_quality, candidate.detector_score)
    match = eval_result["match"]
    embedding = eval_result["embedding"]
    embedding_success = bool(eval_result["embedding_success"])
    return {
        **slot_dict,
        "Detection_Set": detection_set,
        "Detection_Source": candidate.detection_source,
        "Zone_Profile": candidate.zone_profile,
        "Zone_ID": candidate.zone_id,
        "Zone_Label": candidate.zone_label,
        "Zone_Upscale_Factor": candidate.zone_upscale_factor,
        "Zone_Bounds_Normalized": candidate.zone_bounds_normalized,
        "Zone_Bounds_Pixels": candidate.zone_bounds_pixels,
        "Detector_Input_Width": candidate.detector_input_width,
        "Detector_Input_Height": candidate.detector_input_height,
        "BBox_Zone_Coordinates": candidate.bbox_zone_coordinates,
        "BBox_Original_Coordinates": candidate.bbox_original_coordinates,
        "Recognition_Frame_Kind": candidate.recognition_frame_kind,
        "Merge_Group_ID": candidate.merge_group_id,
        "Merged_Detection_Count": candidate.merged_detection_count,
        "Contributing_Sources": candidate.contributing_sources,
        "Selected_After_Merge": "Yes" if candidate.selected_after_merge else "No",
        "Selected_Source": candidate.selected_source,
        "Selected_Zone_ID": candidate.selected_zone_id,
        "Duplicate_IoU": candidate.duplicate_iou,
        "Merge_Reason": candidate.merge_reason,
        "Checkpoint_ID": cp.cp_id,
        "Checkpoint_Label": cp.label,
        "Checkpoint_Class_Time": cp.class_time,
        "Camera_ID": camera_id,
        "Video": video_path.name,
        "Frame": current_frame,
        "Time_In_Video": time_text,
        "Candidate_ID": candidate.candidate_id,
        "YuNet_Face_Score": round(candidate.detector_score, 4),
        **geom,
        "Original_Equivalent_Face_Width": geom.get("Face_Width", ""),
        "Original_Equivalent_Face_Height": geom.get("Face_Height", ""),
        **crop_quality,
        "Diagnostic_Quality_Label": label,
        "Diagnostic_Quality_Reasons": reasons,
        "Embedding_Extraction_Success": "Yes" if embedding_success else "No",
        "Embedding_Dimension": int(len(embedding)) if embedding_success else 0,
        "Best_Roll": match.roll_no or "",
        "Best_Score": round(match.best_score, 4),
        "Second_Roll": match.second_roll_no or "",
        "Second_Score": round(match.second_score, 4),
        "Margin": round(match.margin, 4),
        "Match_Threshold": match_threshold,
        "Margin_Threshold": margin_threshold,
        "Aggregate_Mode": aggregate,
        "Accepted": "Yes" if match.accepted else "No",
        "Actual_Matcher_Reject_Reason": match.reason,
        "Diagnostic_Reject_Reason": eval_result["diagnostic_reason"],
    }


def summarize_zone_comparison(zone_candidate_df: pd.DataFrame) -> pd.DataFrame:
    if zone_candidate_df.empty:
        return pd.DataFrame()
    rows = []
    for group_key, group_df in zone_candidate_df.groupby(["Detection_Set", "Checkpoint_ID", "Camera_ID"], dropna=False):
        detection_set, cp_id, camera_id = group_key
        accepted = int(group_df["Accepted"].eq("Yes").sum())
        unique_accepted = int(group_df.loc[group_df["Accepted"].eq("Yes"), "Best_Roll"].replace("", pd.NA).dropna().nunique())
        row = {
            "Detection_Set": detection_set,
            "Checkpoint_ID": cp_id,
            "Camera_ID": camera_id,
            "Detections": len(group_df),
            "Accepted_Recognitions": accepted,
            "Rejected_Or_Unknown": len(group_df) - accepted,
            "Recognition_Rate": round((accepted / len(group_df)) * 100, 1) if len(group_df) else 0,
            "Unique_Accepted_Identities": unique_accepted,
            "Embedding_Success_Rate": round(group_df["Embedding_Extraction_Success"].eq("Yes").mean() * 100, 1),
            "Usable_Count": int(group_df["Diagnostic_Quality_Label"].eq("usable_reference").sum()),
            "Marginal_Count": int(group_df["Diagnostic_Quality_Label"].eq("marginal_reference").sum()),
            "Unusable_Count": int(group_df["Diagnostic_Quality_Label"].eq("unusable_reference").sum()),
        }
        row.update(percentile_fields(group_df["Original_Equivalent_Face_Width"], "Face_Width", decimals=2))
        row.update(percentile_fields(group_df["Original_Equivalent_Face_Height"], "Face_Height", decimals=2))
        row.update(percentile_fields(group_df["Blur_Laplacian_Variance"], "Blur_Laplacian_Variance", decimals=2))
        row.update(percentile_fields(group_df["YuNet_Face_Score"], "Detector_Score", decimals=4))
        row.update(percentile_fields(group_df["Best_Score"], "Best_Score", decimals=4))
        row.update(percentile_fields(group_df["Margin"], "Margin", decimals=4))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["Detection_Set", "Checkpoint_ID", "Camera_ID"])


def tracklet_config_from_args(args) -> TrackletConfig | None:
    if args.tracklet_mode == "off":
        return None
    from src.face_attendance.tracklets import TrackletConfig

    try:
        return TrackletConfig(
            min_observations=args.tracklet_min_observations,
            max_selected_observations=args.tracklet_max_selected,
            max_gap_seconds=args.tracklet_max_gap_seconds,
            min_iou=args.tracklet_min_iou,
            max_center_ratio=args.tracklet_max_center_ratio,
            min_size_ratio=args.tracklet_min_size_ratio,
            min_embedding_similarity=args.tracklet_min_embedding_similarity,
        )
    except ValueError as exc:
        raise SystemExit(f"Tracklet configuration error: {exc}") from exc


def select_tracklet_candidates(
    tracklet_mode: str,
    zone_mode: str,
    full_candidates: list[DetectionCandidate],
    combined_candidates: list[DetectionCandidate],
) -> tuple[str, list[DetectionCandidate]]:
    if tracklet_mode == "off":
        return "off", []
    if zone_mode != "off" and (tracklet_mode == "compare" or zone_mode == "zones"):
        return "merged_full_frame_plus_zones", combined_candidates
    return "baseline_full_frame", full_candidates


def build_tracklet_observation(
    candidate: DetectionCandidate,
    eval_result: dict,
    raw_frame,
    current_frame: int,
    timestamp_seconds: float,
) -> TrackletObservation:
    from src.face_attendance.tracklets import TrackletObservation

    geom = face_geometry(candidate.face_original, raw_frame.shape)
    crop_quality = crop_quality_metrics(raw_frame, tuple(float(value) for value in candidate.face_original[:4]))
    quality_label_value, quality_reasons = diagnostic_quality_label(geom, crop_quality, candidate.detector_score)
    match = eval_result["match"]
    contributing_sources = candidate.contributing_sources or candidate.source_label
    selected_source = candidate.selected_source or candidate.detection_source
    return TrackletObservation(
        observation_id=candidate.candidate_id,
        frame_index=current_frame,
        timestamp_seconds=timestamp_seconds,
        bbox=tuple(float(value) for value in candidate.face_original[:4]),
        detector_score=float(candidate.detector_score),
        embedding=eval_result["embedding"],
        quality_label=quality_label_value,
        quality_reasons=quality_reasons,
        landmark_valid=geom.get("Landmark_Valid") == "Yes",
        blur_score=float(crop_quality.get("Blur_Laplacian_Variance") or 0.0),
        face_width=float(geom.get("Face_Width") or 0.0),
        face_height=float(geom.get("Face_Height") or 0.0),
        detection_source=candidate.detection_source,
        zone_id=candidate.zone_id,
        zone_profile=candidate.zone_profile,
        contributing_sources=contributing_sources,
        selected_source=selected_source,
        merged_detection_count=int(candidate.merged_detection_count),
        frame_best_roll=match.roll_no or "",
        frame_best_score=float(match.best_score),
        frame_second_roll=match.second_roll_no or "",
        frame_second_score=float(match.second_score),
        frame_margin=float(match.margin),
        frame_match_accepted=bool(match.accepted),
        frame_match_reason=match.reason,
    )


def evaluate_finalized_tracklet(
    tracklet: FinalizedTracklet,
    db: StudentEmbeddingDB,
    match_threshold: float,
    margin_threshold: float,
    aggregate: str,
) -> tuple[MatchResult, str]:
    if not tracklet.eligible or tracklet.aggregate_embedding is None:
        reason = tracklet.rejection_reason or "ineligible_tracklet"
        return MatchResult(False, None, 0.0, None, 0.0, 0.0, reason), reason
    match = db.match(
        tracklet.aggregate_embedding,
        match_threshold=match_threshold,
        margin_threshold=margin_threshold,
        aggregate=aggregate,
    )
    reason = diagnostic_rejection_reason(
        match.accepted,
        match.reason,
        True,
        match.best_score,
        match.margin,
        match_threshold,
        margin_threshold,
    )
    return match, reason


def build_tracklet_diagnostic_row(
    slot_dict: dict,
    tracklet_mode: str,
    source_set: str,
    tracklet: FinalizedTracklet,
    match: MatchResult,
    diagnostic_reason: str,
    cp: Checkpoint,
    camera_id: str,
    video_path: Path,
    match_threshold: float,
    margin_threshold: float,
    aggregate: str,
) -> dict:
    observations = [member.observation for member in tracklet.members]
    quality_counts = Counter(observation.quality_label for observation in observations)
    best_counts = Counter(observation.frame_best_roll for observation in observations if observation.frame_best_roll)
    accepted_counts = Counter(
        observation.frame_best_roll
        for observation in observations
        if observation.frame_match_accepted and observation.frame_best_roll
    )
    dominant_roll, dominant_count = best_counts.most_common(1)[0] if best_counts else ("", 0)
    row = {
        **slot_dict,
        "Tracklet_Mode": tracklet_mode,
        "Tracklet_Source_Set": source_set,
        "Tracklet_ID": tracklet.tracklet_id,
        "Checkpoint_ID": cp.cp_id,
        "Checkpoint_Label": cp.label,
        "Checkpoint_Class_Time": cp.class_time,
        "Camera_ID": camera_id,
        "Video": video_path.name,
        "Start_Frame": tracklet.start_frame,
        "End_Frame": tracklet.end_frame,
        "Start_Time_In_Video": seconds_to_text(tracklet.start_seconds),
        "End_Time_In_Video": seconds_to_text(tracklet.end_seconds),
        "Tracklet_Duration_Seconds": round(tracklet.end_seconds - tracklet.start_seconds, 4),
        "Observation_Count": tracklet.observation_count,
        "Embedding_Count": tracklet.embedding_count,
        "Selected_Observation_Count": len(tracklet.selected_member_ids),
        "Consistent_Embedding_Count": len(tracklet.consistent_member_ids),
        "Inconsistent_Embedding_Count": len(tracklet.inconsistent_member_ids),
        "Tracklet_Eligible": "Yes" if tracklet.eligible else "No",
        "Tracklet_Quality_Rejection": "" if tracklet.eligible else tracklet.rejection_reason,
        "Aggregate_Embedding_Dimension": int(len(tracklet.aggregate_embedding)) if tracklet.aggregate_embedding is not None else 0,
        "Medoid_Observation_ID": tracklet.medoid_member_id,
        "Member_Observation_IDs": "; ".join(tracklet.member_ids),
        "Selected_Observation_IDs": "; ".join(tracklet.selected_member_ids),
        "Consistent_Observation_IDs": "; ".join(tracklet.consistent_member_ids),
        "Inconsistent_Observation_IDs": "; ".join(tracklet.inconsistent_member_ids),
        "Pairwise_Similarity_Min": round(tracklet.pairwise_similarity_min, 4) if tracklet.pairwise_similarity_min is not None else "",
        "Pairwise_Similarity_Median": round(tracklet.pairwise_similarity_median, 4) if tracklet.pairwise_similarity_median is not None else "",
        "Usable_Observation_Count": int(quality_counts.get("usable_reference", 0)),
        "Marginal_Observation_Count": int(quality_counts.get("marginal_reference", 0)),
        "Unusable_Observation_Count": int(quality_counts.get("unusable_reference", 0)),
        "Contributing_Detection_Sources": "; ".join(sorted({observation.contributing_sources for observation in observations if observation.contributing_sources})),
        "Selected_Detection_Sources": "; ".join(sorted({observation.selected_source for observation in observations if observation.selected_source})),
        "Zone_Profiles": "; ".join(sorted({observation.zone_profile for observation in observations if observation.zone_profile})),
        "Zone_IDs": "; ".join(sorted({observation.zone_id for observation in observations if observation.zone_id})),
        "Merged_Detection_Count": int(sum(observation.merged_detection_count for observation in observations)),
        "Association_Reasons": "; ".join(sorted({member.association_reason for member in tracklet.members})),
        "Frame_Level_Accepted_Observations": int(sum(observation.frame_match_accepted for observation in observations)),
        "Frame_Level_Unique_Accepted": len(accepted_counts),
        "Dominant_Frame_Best_Roll": dominant_roll,
        "Dominant_Frame_Best_Count": dominant_count,
        "Dominant_Frame_Best_Share_Pct": round((dominant_count / len(observations)) * 100.0, 1) if observations else 0,
        "Tracklet_Best_Roll": match.roll_no or "",
        "Tracklet_Best_Score": round(match.best_score, 4),
        "Tracklet_Second_Roll": match.second_roll_no or "",
        "Tracklet_Second_Score": round(match.second_score, 4),
        "Tracklet_Margin": round(match.margin, 4),
        "Match_Threshold": match_threshold,
        "Margin_Threshold": margin_threshold,
        "Aggregate_Mode": aggregate,
        "Tracklet_Accepted": "Yes" if match.accepted else "No",
        "Tracklet_Matcher_Reason": match.reason,
        "Tracklet_Diagnostic_Reason": diagnostic_reason,
        "Official_Attendance_Contribution": "Yes" if tracklet_mode == "tracklets" and match.accepted else "No",
    }
    row.update(percentile_fields((observation.face_width for observation in observations), "Face_Width", decimals=2))
    row.update(percentile_fields((observation.face_height for observation in observations), "Face_Height", decimals=2))
    row.update(percentile_fields((observation.blur_score for observation in observations), "Blur", decimals=2))
    row.update(percentile_fields((observation.detector_score for observation in observations), "Detector_Score", decimals=4))
    row.update(percentile_fields((member.quality_weight for member in tracklet.members), "Quality_Weight", decimals=4))
    row.update(percentile_fields((observation.frame_best_score for observation in observations), "Frame_Best_Score", decimals=4))
    row.update(percentile_fields((observation.frame_margin for observation in observations), "Frame_Margin", decimals=4))
    return row


def build_tracklet_observation_rows(
    slot_dict: dict,
    source_set: str,
    tracklet: FinalizedTracklet,
    match: MatchResult,
    cp: Checkpoint,
    camera_id: str,
    video_path: Path,
) -> list[dict]:
    rows = []
    for member in tracklet.members:
        observation = member.observation
        embedding = observation.embedding
        rows.append({
            **slot_dict,
            "Tracklet_Source_Set": source_set,
            "Tracklet_ID": tracklet.tracklet_id,
            "Observation_ID": observation.observation_id,
            "Checkpoint_ID": cp.cp_id,
            "Camera_ID": camera_id,
            "Video": video_path.name,
            "Frame": observation.frame_index,
            "Time_In_Video": seconds_to_text(observation.timestamp_seconds),
            "BBox_Original_Coordinates": ",".join(f"{value:.2f}" for value in observation.bbox),
            "Face_Width": round(observation.face_width, 2),
            "Face_Height": round(observation.face_height, 2),
            "Detector_Score": round(observation.detector_score, 4),
            "Diagnostic_Quality_Label": observation.quality_label,
            "Diagnostic_Quality_Reasons": observation.quality_reasons,
            "Landmark_Valid": "Yes" if observation.landmark_valid else "No",
            "Blur_Laplacian_Variance": round(observation.blur_score, 4),
            "Detection_Source": observation.detection_source,
            "Selected_Source": observation.selected_source,
            "Contributing_Sources": observation.contributing_sources,
            "Zone_Profile": observation.zone_profile,
            "Zone_ID": observation.zone_id,
            "Merged_Detection_Count": observation.merged_detection_count,
            "Association_IoU": round(member.association_iou, 4),
            "Association_Center_Ratio": round(member.association_center_ratio, 4),
            "Association_Size_Ratio": round(member.association_size_ratio, 4),
            "Association_Reason": member.association_reason,
            "Quality_Weight": round(member.quality_weight, 6),
            "Selected_For_Aggregation": "Yes" if member.selected_for_aggregation else "No",
            "Embedding_Consistent": "Yes" if member.embedding_consistent else "No",
            "Embedding_Extraction_Success": "Yes" if embedding is not None else "No",
            "Embedding_Dimension": int(len(embedding)) if embedding is not None else 0,
            "Frame_Best_Roll": observation.frame_best_roll,
            "Frame_Best_Score": round(observation.frame_best_score, 4),
            "Frame_Second_Roll": observation.frame_second_roll,
            "Frame_Second_Score": round(observation.frame_second_score, 4),
            "Frame_Margin": round(observation.frame_margin, 4),
            "Frame_Accepted": "Yes" if observation.frame_match_accepted else "No",
            "Frame_Matcher_Reason": observation.frame_match_reason,
            "Tracklet_Best_Roll": match.roll_no or "",
            "Tracklet_Accepted": "Yes" if match.accepted else "No",
        })
    return rows


def summarize_tracklet_comparison(tracklet_df: pd.DataFrame) -> pd.DataFrame:
    if tracklet_df.empty:
        return pd.DataFrame()
    rows = []
    group_columns = ["Tracklet_Source_Set", "Checkpoint_ID", "Camera_ID"]
    for group_key, group_df in tracklet_df.groupby(group_columns, dropna=False):
        source_set, checkpoint_id, camera_id = group_key
        accepted_rows = group_df[group_df["Tracklet_Accepted"].eq("Yes")]
        observations = int(pd.to_numeric(group_df["Observation_Count"], errors="coerce").fillna(0).sum())
        tracklet_count = int(len(group_df))
        rows.append({
            "Tracklet_Source_Set": source_set,
            "Checkpoint_ID": checkpoint_id,
            "Camera_ID": camera_id,
            "Source_Observations": observations,
            "Tracklets": tracklet_count,
            "Observation_Reduction_Pct": round((1.0 - tracklet_count / observations) * 100.0, 1) if observations else 0,
            "Eligible_Tracklets": int(group_df["Tracklet_Eligible"].eq("Yes").sum()),
            "Accepted_Tracklets": int(len(accepted_rows)),
            "Unique_Accepted_Identities": int(accepted_rows["Tracklet_Best_Roll"].replace("", pd.NA).dropna().nunique()),
            "Mean_Observations_Per_Tracklet": round(float(pd.to_numeric(group_df["Observation_Count"], errors="coerce").mean()), 2),
            "Median_Observations_Per_Tracklet": round(float(pd.to_numeric(group_df["Observation_Count"], errors="coerce").median()), 2),
            "Quality_Rejections": json.dumps(group_df["Tracklet_Quality_Rejection"].replace("", pd.NA).dropna().value_counts().to_dict(), sort_keys=True),
        })
    return pd.DataFrame(rows).sort_values(group_columns)


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
    validate_tracklet_review_args(args)
    validate_diagnostic_only_args(args)
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

    if not args.diagnostic_only:
        ensure_dirs(output_dir)
    if args.save_unknown:
        ensure_dirs(unknown_dir)

    db = StudentEmbeddingDB.load(embeddings_path)
    roster_contract = None
    attendance_rolls = list(db.roll_numbers)
    missing_embedding_rolls: list[str] = []
    if args.require_authoritative_roster:
        if args.processing_contract_id != PROCESSING_CONTRACT_POLICY_VERSION:
            raise SystemExit(
                f"Processing contract mismatch: {args.processing_contract_id!r}; "
                f"expected {PROCESSING_CONTRACT_POLICY_VERSION!r}"
            )
        try:
            roster_contract = load_subject_roster(Path(args.student_map), args.subject_abbr or slot.course_abbr)
            filtered_records, missing_embedding_rolls = filter_embedding_records_to_roster(
                db.records, roster_contract.rolls
            )
            if not filtered_records:
                raise ProcessingIntegrationError(
                    f"No production embeddings belong to the {roster_contract.subject} roster"
                )
            db = StudentEmbeddingDB(
                filtered_records,
                metadata={**db.metadata, "authoritative_roster_subject": roster_contract.subject},
            )
            attendance_rolls = list(roster_contract.rolls)
            inspect_exact_checkpoint_layout(video_dir)
        except ProcessingIntegrationError as exc:
            raise SystemExit(f"Product Phase 2G processing contract failed: {exc}") from exc

    engine = FaceEngine(
        YUNET_MODEL,
        SFACE_MODEL,
        detection_score=args.det_score,
        nms_threshold=DEFAULT_NMS_THRESHOLD,
        top_k=DEFAULT_TOP_K,
    )
    zone_config = None
    zone_config_path = Path(args.camera_zones)
    if args.zone_mode != "off":
        try:
            zone_config = load_camera_zone_config(zone_config_path)
        except (OSError, json.JSONDecodeError, ZoneConfigError) as exc:
            raise SystemExit(f"Camera zone configuration error: {zone_config_path}: {exc}")
        if args.zone_profile != "auto" and args.zone_profile not in zone_config.profiles:
            raise SystemExit(f"Camera zone profile not found: {args.zone_profile}")
    if args.tracklet_mode == "compare" and not args.diagnostic:
        raise SystemExit("Tracklet compare mode requires --diagnostic so comparison evidence is not discarded")
    if args.tracklet_mode == "tracklets" and args.save_unknown:
        raise SystemExit("--save-unknown is not supported with official tracklet mode because raw frames are not retained")
    tracklet_config = tracklet_config_from_args(args)

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
        attendance_rolls,
        checkpoints=checkpoints,
        checkpoint_min_detections=args.checkpoint_min_detections,
        present_checkpoints=args.present_checkpoints,
        strong_checkpoints=args.strong_checkpoints,
        review_checkpoints=args.review_checkpoints,
        low_confidence_score=args.low_confidence_score,
        tracklet_evidence_enabled=args.tracklet_mode == "tracklets",
    )

    print("MVP 3 Checkpoint-Based YuNet + SFace Attendance Started")
    print(f"Slot: {slot.slot_id} | {slot.day} P{slot.period} | {slot.start_time}-{slot.end_time}")
    print(f"Subject: {slot.course_abbr} - {slot.course_name}")
    print(f"Instructor: {slot.instructor}")
    print(f"Room: {slot.room} | Section: {slot.section}")
    print(f"Authoritative roster students: {len(attendance_rolls)}")
    print(f"Embedding-covered roster students: {db.student_count}")
    print(f"Embeddings: {len(db)}")
    if missing_embedding_rolls:
        print("Missing roster embeddings: " + ", ".join(missing_embedding_rolls))
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
    print(f"Diagnostic mode: {'on' if args.diagnostic else 'off'}")
    print(f"Zone mode: {args.zone_mode}")
    if zone_config:
        print(f"Camera zones: {zone_config_path}")
    if tracklet_config:
        print(f"Tracklet mode: {args.tracklet_mode}")
        print(
            "Tracklets: "
            f"min observations={tracklet_config.min_observations}, "
            f"max selected={tracklet_config.max_selected_observations}, "
            f"max gap={tracklet_config.max_gap_seconds:.2f}s"
        )
    print()

    slot_dict = slot.to_dict()
    session_metadata = {
        "Session_ID": args.session_id,
        "Session_Date": args.session_date,
        "Subject_Abbr": args.subject_abbr,
        "Subject_Name": args.subject_name,
        "Course_Code": args.course_code,
        "Faculty_ID": args.faculty_id,
        "Faculty_Name": args.faculty_name,
        "Input_Slot": args.input_slot,
        "Input_Source_Type": args.input_source_type,
        "Input_Source_Path": args.input_source_path,
    }
    if roster_contract is not None:
        session_metadata.update(
            build_processing_metadata(
                roster=roster_contract,
                embedding_record_count=len(db),
                embedding_student_count=db.student_count,
                missing_embedding_rolls=missing_embedding_rolls,
            )
        )
    for key, value in session_metadata.items():
        if value:
            slot_dict[key] = value
    if args.subject_abbr:
        slot_dict["Course_Abbr"] = args.subject_abbr
    if args.subject_name:
        slot_dict["Course_Name"] = args.subject_name
    if args.course_code:
        slot_dict["Course_Code"] = args.course_code
    if args.faculty_name:
        slot_dict["Instructor"] = args.faculty_name
    detection_log: list[dict] = []
    checkpoint_metrics: Dict[str, dict] = defaultdict(lambda: {"frames": 0, "detected": 0, "accepted": 0, "rejected": 0, "students": set()})
    camera_metrics: Dict[tuple, dict] = defaultdict(lambda: {"frames": 0, "detected": 0, "accepted": 0, "rejected": 0, "students": set()})
    unknown_saved_per_key: Dict[tuple, int] = defaultdict(int)
    unknown_saved = 0
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    diagnostic_run_id = (
        args.diagnostic_run_id
        or make_diagnostic_run_id(args.session_id or build_slot_prefix(slot))
    ) if args.diagnostic else ""
    detailed_face_metrics = args.diagnostic or args.log_mode == "full"
    face_diagnostics: list[dict] = []
    zone_candidate_diagnostics: list[dict] = []
    zone_frame_comparison: list[dict] = []
    tracklet_diagnostics: list[dict] = []
    tracklet_observation_diagnostics: list[dict] = []
    tracklet_window_comparison: list[dict] = []
    window_metrics: dict[tuple[str, str, str], dict] = {}
    candidate_best_counts: Counter[str] = Counter()
    candidate_accepted_counts: Counter[str] = Counter()
    total_processed_frames = 0
    total_skipped_windows = 0
    videos_processed: list[str] = []
    stop_requested = False

    for meta in metas:
        video_path = meta.path
        video_name = video_path.name
        camera_id = video_camera_map.get(video_name, infer_camera_id(video_path, slot.camera_ids))
        if zone_config and args.zone_profile == "auto":
            active_zone_profile = zone_config.resolve_profile(camera_id, video_name)
        elif zone_config:
            active_zone_profile = zone_config.profiles.get(args.zone_profile)
        else:
            active_zone_profile = None
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
        if args.zone_mode != "off" and active_zone_profile is None:
            print(f"  Zone warning: no camera-zone profile matched camera={camera_id}, video={video_name}; using full-frame only")

        for cp in cp_loop:
            start_sec, end_sec = windows.get(cp.cp_id, (cp.window_start_sec, cp.window_end_sec))
            requested_start_sec = cp.window_start_sec
            requested_end_sec = cp.window_end_sec
            selected_end_for_warnings = min(end_sec, meta.duration_sec) if meta.duration_sec > 0 else end_sec
            window_valid, timing_warnings = checkpoint_window_warnings(
                checkpoint_mode_used,
                meta.duration_sec,
                start_sec,
                selected_end_for_warnings,
                requested_start_sec,
                requested_end_sec,
            )
            camera_key = (cp.cp_id, camera_id, video_name)
            window_metrics[camera_key] = {
                "Diagnostic_Run_ID": diagnostic_run_id,
                "Checkpoint_ID": cp.cp_id,
                "Checkpoint_Label": cp.label,
                "Checkpoint_Class_Time": cp.class_time,
                "Checkpoint_Mode": checkpoint_mode_used,
                "Source_File": str(video_path),
                "Source_File_Name": video_name,
                "Camera_ID": camera_id,
                "Source_Duration_Seconds": round(meta.duration_sec, 3),
                "Source_FPS": round(meta.fps, 3),
                "Source_Frame_Count": int(meta.frame_count),
                "Requested_Class_Window_Start": seconds_to_text(requested_start_sec),
                "Requested_Class_Window_End": seconds_to_text(requested_end_sec),
                "Selected_Video_Window_Start": seconds_to_text(start_sec),
                "Selected_Video_Window_End": seconds_to_text(selected_end_for_warnings),
                "Actual_First_Processed_Timestamp": "",
                "Actual_Last_Processed_Timestamp": "",
                "Sampled_Frames": 0,
                "Frames_With_Faces": 0,
                "Total_Faces": 0,
                "Accepted": 0,
                "Rejected": 0,
                "Unique_Accepted": set(),
                "Face_Widths": [],
                "Face_Heights": [],
                "Face_Area_Ratios": [],
                "Blur_Scores": [],
                "Detector_Scores": [],
                "Best_Scores": [],
                "Margins": [],
                "Window_Valid": "Yes" if window_valid else "No",
                "Timing_Warnings": timing_warnings,
            }
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

            tracklet_builder = None
            tracklet_source_set = "off"
            tracklet_source_observations = 0
            tracklet_source_frame_accepted = 0
            tracklet_source_frame_identities: set[str] = set()
            tracklet_build_seconds = 0.0
            if tracklet_config is not None:
                from src.face_attendance.tracklets import TrackletBuilder

                tracklet_prefix = clean_filename(f"{cp.cp_id}-{camera_id}-{video_path.stem}")
                tracklet_builder = TrackletBuilder(tracklet_config, id_prefix=tracklet_prefix)

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

                raw_frame = frame
                full_frame, full_frame_scale = resize_keep_aspect(raw_frame, args.max_width)
                processed_frames_this_window += 1
                total_processed_frames += 1
                checkpoint_metrics[cp.cp_id]["frames"] += 1
                camera_metrics[camera_key]["frames"] += 1
                window_metrics[camera_key]["Sampled_Frames"] += 1

                time_text = seconds_to_text(time_sec)
                if not window_metrics[camera_key]["Actual_First_Processed_Timestamp"]:
                    window_metrics[camera_key]["Actual_First_Processed_Timestamp"] = time_text
                window_metrics[camera_key]["Actual_Last_Processed_Timestamp"] = time_text
                full_detect_started = time.perf_counter()
                faces = engine.detect_faces(full_frame)
                full_detect_seconds = time.perf_counter() - full_detect_started
                full_candidates: list[DetectionCandidate] = []
                for face_idx, face in enumerate(faces, start=1):
                    face_original = map_resized_face_to_original(face, full_frame_scale)
                    candidate = DetectionCandidate(
                        candidate_id=f"{cp.cp_id}:{camera_id}:{video_name}:f{current_frame}:full:{face_idx}",
                        detection_source="full_frame",
                        face_original=face_original,
                        recognition_face=face,
                        recognition_frame=full_frame,
                        recognition_frame_kind="resized_full_frame" if abs(full_frame_scale - 1.0) > 1e-9 else "original_full_frame",
                        detector_score=engine.face_score(face),
                        source_priority=1,
                        detector_input_width=int(full_frame.shape[1]),
                        detector_input_height=int(full_frame.shape[0]),
                        bbox_original_coordinates=face_box_text(face_original),
                        quality_rank=(),
                    )
                    candidate.quality_rank = detection_quality_rank(candidate.face_original, raw_frame, candidate.detector_score, candidate.source_priority)
                    full_candidates.append(candidate)

                zone_candidates: list[DetectionCandidate] = []
                zone_detect_seconds = 0.0
                zone_profile_id = ""
                zone_warning = ""
                if args.zone_mode != "off" and active_zone_profile is not None:
                    zone_profile_id = active_zone_profile.profile_id
                    for zone in active_zone_profile.enabled_zones():
                        zone_bounds = zone_pixel_bounds(zone, raw_frame.shape)
                        x1, y1, x2, y2 = zone_bounds
                        zone_crop = raw_frame[y1:y2, x1:x2]
                        if zone_crop.size == 0:
                            continue
                        zone_input = upscale_zone_crop(zone_crop, zone.upscale)
                        zone_started = time.perf_counter()
                        zone_faces = engine.detect_faces(zone_input)
                        zone_detect_seconds += time.perf_counter() - zone_started
                        for zone_face_idx, zone_face in enumerate(zone_faces, start=1):
                            face_original = map_zone_face_to_original(zone_face, zone_bounds, zone.upscale)
                            candidate = DetectionCandidate(
                                candidate_id=f"{cp.cp_id}:{camera_id}:{video_name}:f{current_frame}:zone:{zone.zone_id}:{zone_face_idx}",
                                detection_source="zone",
                                face_original=face_original,
                                recognition_face=zone_face,
                                recognition_frame=zone_input,
                                recognition_frame_kind="upscaled_zone" if zone.upscale > 1.0 else "zone_crop",
                                detector_score=engine.face_score(zone_face),
                                source_priority=2,
                                zone_profile=active_zone_profile.profile_id,
                                zone_id=zone.zone_id,
                                zone_label=zone.label,
                                zone_upscale_factor=zone.upscale,
                                zone_bounds_normalized=zone.bounds_normalized_text,
                                zone_bounds_pixels=bounds_text(zone_bounds),
                                detector_input_width=int(zone_input.shape[1]),
                                detector_input_height=int(zone_input.shape[0]),
                                bbox_zone_coordinates=face_box_text(zone_face),
                                bbox_original_coordinates=face_box_text(face_original),
                                quality_rank=(),
                            )
                            candidate.quality_rank = detection_quality_rank(candidate.face_original, raw_frame, candidate.detector_score, candidate.source_priority)
                            zone_candidates.append(candidate)
                elif args.zone_mode != "off":
                    zone_warning = "no_matching_zone_profile"

                merge_started = time.perf_counter()
                zone_only_merged = merge_detection_candidates(zone_candidates, iou_threshold=args.zone_merge_iou) if zone_candidates else []
                combined_merged = merge_detection_candidates(full_candidates + zone_candidates, iou_threshold=args.zone_merge_iou) if (full_candidates or zone_candidates) else []
                merge_seconds = time.perf_counter() - merge_started
                frame_official_candidates = combined_merged if args.zone_mode == "zones" else full_candidates
                official_candidates = [] if args.tracklet_mode == "tracklets" else frame_official_candidates

                checkpoint_metrics[cp.cp_id]["detected"] += len(official_candidates)
                camera_metrics[camera_key]["detected"] += len(official_candidates)
                window_metrics[camera_key]["Total_Faces"] += len(official_candidates)
                if len(official_candidates) > 0:
                    window_metrics[camera_key]["Frames_With_Faces"] += 1

                eval_cache: dict[str, dict] = {}

                def candidate_eval(candidate: DetectionCandidate) -> dict:
                    if candidate.candidate_id not in eval_cache:
                        eval_cache[candidate.candidate_id] = evaluate_detection_candidate(
                            candidate,
                            engine,
                            db,
                            match_threshold=args.match_threshold,
                            margin_threshold=args.margin_threshold,
                            aggregate=args.aggregate,
                        )
                    return eval_cache[candidate.candidate_id]

                recognition_started = time.perf_counter()

                if args.zone_mode != "off" and args.diagnostic:
                    comparison_sets = [
                        ("baseline_full_frame", full_candidates),
                        ("zone_only_premerge", zone_candidates),
                        ("zone_only_merged", zone_only_merged),
                        ("merged_full_frame_plus_zones", combined_merged),
                    ]
                    for detection_set, candidates_for_set in comparison_sets:
                        for candidate in candidates_for_set:
                            zone_candidate_diagnostics.append(build_zone_candidate_row(
                                slot_dict=slot_dict,
                                detection_set=detection_set,
                                candidate=candidate,
                                eval_result=candidate_eval(candidate),
                                raw_frame=raw_frame,
                                cp=cp,
                                camera_id=camera_id,
                                video_path=video_path,
                                current_frame=current_frame,
                                time_text=time_text,
                                match_threshold=args.match_threshold,
                                margin_threshold=args.margin_threshold,
                                aggregate=args.aggregate,
                            ))

                if tracklet_builder is not None:
                    tracklet_source_set, tracklet_candidates = select_tracklet_candidates(
                        args.tracklet_mode,
                        args.zone_mode,
                        full_candidates,
                        combined_merged,
                    )
                    tracklet_started = time.perf_counter()
                    observations = []
                    for candidate in tracklet_candidates:
                        eval_result = candidate_eval(candidate)
                        match = eval_result["match"]
                        observations.append(build_tracklet_observation(
                            candidate=candidate,
                            eval_result=eval_result,
                            raw_frame=raw_frame,
                            current_frame=current_frame,
                            timestamp_seconds=time_sec,
                        ))
                        tracklet_source_observations += 1
                        if match.accepted:
                            tracklet_source_frame_accepted += 1
                            if match.roll_no:
                                tracklet_source_frame_identities.add(match.roll_no)
                    if observations:
                        tracklet_builder.update(observations)
                        if args.tracklet_mode == "tracklets":
                            window_metrics[camera_key]["Frames_With_Faces"] += 1
                    tracklet_build_seconds += time.perf_counter() - tracklet_started

                for face_idx, candidate in enumerate(official_candidates, start=1):
                    eval_result = candidate_eval(candidate)
                    match = eval_result["match"]
                    embedding = eval_result["embedding"]
                    embedding_success = bool(eval_result["embedding_success"])
                    diagnostic_reason = eval_result["diagnostic_reason"]
                    face_score = candidate.detector_score
                    if candidate.detection_source == "full_frame":
                        diag_face = candidate.recognition_face
                        diag_frame = candidate.recognition_frame
                        box = tuple(float(v) for v in candidate.recognition_face[:4])
                    else:
                        diag_face = candidate.face_original
                        diag_frame = raw_frame
                        box = tuple(float(v) for v in candidate.face_original[:4])
                    geom = face_geometry(diag_face, diag_frame.shape) if detailed_face_metrics else {}
                    original_geom = face_geometry(candidate.face_original, raw_frame.shape) if detailed_face_metrics else {}
                    crop_quality = crop_quality_metrics(diag_frame, box) if detailed_face_metrics else {}
                    quality_ref_label, quality_ref_reasons = diagnostic_quality_label(geom, crop_quality, face_score) if detailed_face_metrics else ("", "")

                    if match.roll_no:
                        candidate_best_counts[match.roll_no] += 1
                    window_metrics[camera_key]["Face_Widths"].append(geom.get("Face_Width", ""))
                    window_metrics[camera_key]["Face_Heights"].append(geom.get("Face_Height", ""))
                    window_metrics[camera_key]["Face_Area_Ratios"].append(geom.get("Face_Area_Ratio", ""))
                    window_metrics[camera_key]["Blur_Scores"].append(crop_quality.get("Blur_Laplacian_Variance", ""))
                    window_metrics[camera_key]["Detector_Scores"].append(face_score)
                    window_metrics[camera_key]["Best_Scores"].append(match.best_score)
                    window_metrics[camera_key]["Margins"].append(match.margin)

                    accepted_roll = match.roll_no if match.accepted else "Unknown"
                    if match.accepted and match.roll_no:
                        evidence_time = f"{cp.cp_id}/{cp.class_time}/{camera_id}/{time_text}"
                        attendance.add_detection(match.roll_no, cp.cp_id, match.best_score, match.margin, video_name, camera_id, evidence_time)
                        checkpoint_metrics[cp.cp_id]["accepted"] += 1
                        checkpoint_metrics[cp.cp_id]["students"].add(match.roll_no)
                        camera_metrics[camera_key]["accepted"] += 1
                        camera_metrics[camera_key]["students"].add(match.roll_no)
                        window_metrics[camera_key]["Accepted"] += 1
                        window_metrics[camera_key]["Unique_Accepted"].add(match.roll_no)
                        candidate_accepted_counts[match.roll_no] += 1
                        color = (0, 255, 0)
                        text = f"{match.roll_no} {match.best_score:.2f} {cp.cp_id}"
                        count_for_roll = attendance.total_counts.get(match.roll_no, 0)
                    else:
                        checkpoint_metrics[cp.cp_id]["rejected"] += 1
                        camera_metrics[camera_key]["rejected"] += 1
                        window_metrics[camera_key]["Rejected"] += 1
                        color = (0, 0, 255)
                        text = f"Unknown {match.best_score:.2f} {cp.cp_id}"
                        count_for_roll = 0
                        if args.save_unknown:
                            save_key = (cp.cp_id, camera_id)
                            if unknown_saved_per_key[save_key] < args.max_unknown_per_camera_checkpoint:
                                crop = safe_crop(diag_frame, box)
                                if crop is not None:
                                    unknown_saved_per_key[save_key] += 1
                                    unknown_saved += 1
                                    subdir = unknown_dir / cp.cp_id / clean_filename(camera_id)
                                    subdir.mkdir(parents=True, exist_ok=True)
                                    filename = f"unknown_{run_id}_{clean_filename(video_path.stem)}_f{current_frame}_d{face_idx}_{match.reason}.jpg"
                                    cv2.imwrite(str(subdir / filename), crop)

                    should_log = should_log_detection(args.log_mode, match.accepted, match.reason)
                    if should_log:
                        log_row = {
                            **slot_dict,
                            "Diagnostic_Run_ID": diagnostic_run_id,
                            "Checkpoint_ID": cp.cp_id,
                            "Checkpoint_Label": cp.label,
                            "Checkpoint_Class_Time": cp.class_time,
                            "Checkpoint_Mode": checkpoint_mode_used,
                            "Zone_Mode": args.zone_mode,
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
                            "Actual_Matcher_Reject_Reason": match.reason,
                            "Diagnostic_Reject_Reason": diagnostic_reason,
                            "Embedding_Extraction_Success": "Yes" if embedding_success else "No",
                            "Embedding_Dimension": int(len(embedding)) if embedding_success else 0,
                            "Match_Threshold": args.match_threshold,
                            "Margin_Threshold": args.margin_threshold,
                            "Aggregate_Mode": args.aggregate,
                            "Detection_Count_For_Roll": count_for_roll,
                            "Detection_Source": candidate.detection_source,
                            "Zone_Profile": candidate.zone_profile,
                            "Zone_ID": candidate.zone_id,
                            "Zone_Label": candidate.zone_label,
                            "Zone_Upscale_Factor": candidate.zone_upscale_factor,
                            "Zone_Bounds_Normalized": candidate.zone_bounds_normalized,
                            "Zone_Bounds_Pixels": candidate.zone_bounds_pixels,
                            "Detector_Input_Width": candidate.detector_input_width,
                            "Detector_Input_Height": candidate.detector_input_height,
                            "BBox_Zone_Coordinates": candidate.bbox_zone_coordinates,
                            "BBox_Original_Coordinates": candidate.bbox_original_coordinates,
                            "Merge_Group_ID": candidate.merge_group_id,
                            "Merged_Detection_Count": candidate.merged_detection_count,
                            "Contributing_Sources": candidate.contributing_sources,
                            "Selected_After_Merge": "Yes" if candidate.selected_after_merge else "No",
                            "Selected_Source": candidate.selected_source,
                            "Selected_Zone_ID": candidate.selected_zone_id,
                            "Duplicate_IoU": candidate.duplicate_iou,
                            "Merge_Reason": candidate.merge_reason,
                        }
                        if detailed_face_metrics:
                            log_row.update(geom)
                            log_row.update(crop_quality)
                            log_row["Original_Equivalent_Face_Width"] = original_geom.get("Face_Width", "")
                            log_row["Original_Equivalent_Face_Height"] = original_geom.get("Face_Height", "")
                            log_row["Diagnostic_Quality_Label"] = quality_ref_label
                            log_row["Diagnostic_Quality_Reasons"] = quality_ref_reasons
                            log_row["Timing_Warnings"] = "; ".join(timing_warnings)
                        detection_log.append(log_row)

                    if args.diagnostic:
                        diagnostic_row = {
                            **slot_dict,
                            "Diagnostic_Run_ID": diagnostic_run_id,
                            "Session_ID": args.session_id,
                            "Session_Date": args.session_date,
                            "Section": slot.section,
                            "Period": f"P{slot.period}",
                            "Subject_Abbr": args.subject_abbr or slot.course_abbr,
                            "Subject_Name": args.subject_name or slot.course_name,
                            "Course_Code": args.course_code,
                            "Checkpoint_ID": cp.cp_id,
                            "Checkpoint_Label": cp.label,
                            "Checkpoint_Class_Time": cp.class_time,
                            "Checkpoint_Mode": checkpoint_mode_used,
                            "Zone_Mode": args.zone_mode,
                            "Camera_ID": camera_id,
                            "Source_File": str(video_path),
                            "Source_File_Name": video_name,
                            "Frame": current_frame,
                            "Video_Relative_Timestamp": time_text,
                            "Classroom_Timestamp": cp.class_time,
                            "Detection_ID": f"{diagnostic_run_id}:{cp.cp_id}:{clean_filename(camera_id)}:{clean_filename(video_name)}:{current_frame}:{face_idx}",
                            "Face_Index": face_idx,
                            "YuNet_Face_Score": round(face_score, 4),
                            **geom,
                            "Original_Equivalent_Face_Width": original_geom.get("Face_Width", ""),
                            "Original_Equivalent_Face_Height": original_geom.get("Face_Height", ""),
                            **crop_quality,
                            "Diagnostic_Quality_Label": quality_ref_label,
                            "Diagnostic_Quality_Reasons": quality_ref_reasons,
                            "Embedding_Extraction_Success": "Yes" if embedding_success else "No",
                            "Embedding_Dimension": int(len(embedding)) if embedding_success else 0,
                            "Best_Roll": match.roll_no or "",
                            "Best_Score": round(match.best_score, 4),
                            "Second_Roll": match.second_roll_no or "",
                            "Second_Score": round(match.second_score, 4),
                            "Margin": round(match.margin, 4),
                            "Match_Threshold": args.match_threshold,
                            "Margin_Threshold": args.margin_threshold,
                            "Aggregate_Mode": args.aggregate,
                            "Accepted": "Yes" if match.accepted else "No",
                            "Predicted_Roll": accepted_roll,
                            "Actual_Matcher_Reject_Reason": match.reason,
                            "Diagnostic_Reject_Reason": diagnostic_reason,
                            "Source_Duration_Seconds": round(meta.duration_sec, 3),
                            "Requested_Class_Window_Start": seconds_to_text(requested_start_sec),
                            "Requested_Class_Window_End": seconds_to_text(requested_end_sec),
                            "Selected_Video_Window_Start": seconds_to_text(start_sec),
                            "Selected_Video_Window_End": seconds_to_text(end_sec),
                            "Window_Valid": "Yes" if window_valid else "No",
                            "Timing_Warnings": "; ".join(timing_warnings),
                            "Detection_Source": candidate.detection_source,
                            "Zone_Profile": candidate.zone_profile,
                            "Zone_ID": candidate.zone_id,
                            "Zone_Label": candidate.zone_label,
                            "Zone_Upscale_Factor": candidate.zone_upscale_factor,
                            "Zone_Bounds_Normalized": candidate.zone_bounds_normalized,
                            "Zone_Bounds_Pixels": candidate.zone_bounds_pixels,
                            "Detector_Input_Width": candidate.detector_input_width,
                            "Detector_Input_Height": candidate.detector_input_height,
                            "BBox_Zone_Coordinates": candidate.bbox_zone_coordinates,
                            "BBox_Original_Coordinates": candidate.bbox_original_coordinates,
                            "Merge_Group_ID": candidate.merge_group_id,
                            "Merged_Detection_Count": candidate.merged_detection_count,
                            "Contributing_Sources": candidate.contributing_sources,
                            "Selected_After_Merge": "Yes" if candidate.selected_after_merge else "No",
                            "Selected_Source": candidate.selected_source,
                            "Selected_Zone_ID": candidate.selected_zone_id,
                            "Duplicate_IoU": candidate.duplicate_iou,
                            "Merge_Reason": candidate.merge_reason,
                        }
                        face_diagnostics.append(diagnostic_row)

                    if args.display:
                        if candidate.detection_source == "full_frame":
                            draw_label(full_frame, box, text, color)

                recognition_seconds = time.perf_counter() - recognition_started

                if args.zone_mode != "off":
                    zone_frame_comparison.append({
                        **slot_dict,
                        "Diagnostic_Run_ID": diagnostic_run_id,
                        "Checkpoint_ID": cp.cp_id,
                        "Checkpoint_Label": cp.label,
                        "Checkpoint_Class_Time": cp.class_time,
                        "Camera_ID": camera_id,
                        "Video": video_name,
                        "Frame": current_frame,
                        "Time_In_Video": time_text,
                        "Zone_Mode": args.zone_mode,
                        "Zone_Profile": zone_profile_id,
                        "Zone_Warning": zone_warning,
                        "Full_Frame_Detections": len(full_candidates),
                        "Zone_Premerge_Detections": len(zone_candidates),
                        "Zone_Only_Merged_Detections": len(zone_only_merged),
                        "Merged_Full_Frame_And_Zone_Detections": len(combined_merged),
                        "Duplicates_Removed": max(0, len(full_candidates) + len(zone_candidates) - len(combined_merged)),
                        "Full_Frame_Source_Summary": json.dumps(source_summary(full_candidates), sort_keys=True),
                        "Zone_Source_Summary": json.dumps(source_summary(zone_candidates), sort_keys=True),
                        "Merged_Source_Summary": json.dumps(source_summary(combined_merged), sort_keys=True),
                        "Full_Detect_Seconds": round(full_detect_seconds, 5),
                        "Zone_Detect_Seconds": round(zone_detect_seconds, 5),
                        "Merge_Seconds": round(merge_seconds, 5),
                        "Recognition_Seconds": round(recognition_seconds, 5),
                    })

                if args.display:
                    cv2.putText(full_frame, f"{slot.slot_id} | {slot.course_abbr} | {cp.cp_id} {cp.class_time}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
                    cv2.putText(full_frame, f"Camera: {camera_id} | Video: {video_name}", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
                    cv2.putText(full_frame, "Press q to stop", (20, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
                    cv2.imshow("MVP3 Checkpoint Attendance", full_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        stop_requested = True
                        break

            if tracklet_builder is not None:
                tracklet_finalize_started = time.perf_counter()
                finalized_tracklets = tracklet_builder.finalize()
                tracklet_accepted = 0
                tracklet_accepted_identities: set[str] = set()
                tracklet_eligible = 0
                tracklet_rejections: Counter[str] = Counter()

                if args.tracklet_mode == "tracklets":
                    eligible_official_tracklets = [tracklet for tracklet in finalized_tracklets if tracklet.eligible]
                    checkpoint_metrics[cp.cp_id]["detected"] += len(eligible_official_tracklets)
                    camera_metrics[camera_key]["detected"] += len(eligible_official_tracklets)
                    window_metrics[camera_key]["Total_Faces"] += len(eligible_official_tracklets)

                for tracklet in finalized_tracklets:
                    match, tracklet_diagnostic_reason = evaluate_finalized_tracklet(
                        tracklet,
                        db,
                        match_threshold=args.match_threshold,
                        margin_threshold=args.margin_threshold,
                        aggregate=args.aggregate,
                    )
                    tracklet_row = build_tracklet_diagnostic_row(
                        slot_dict=slot_dict,
                        tracklet_mode=args.tracklet_mode,
                        source_set=tracklet_source_set,
                        tracklet=tracklet,
                        match=match,
                        diagnostic_reason=tracklet_diagnostic_reason,
                        cp=cp,
                        camera_id=camera_id,
                        video_path=video_path,
                        match_threshold=args.match_threshold,
                        margin_threshold=args.margin_threshold,
                        aggregate=args.aggregate,
                    )
                    tracklet_diagnostics.append(tracklet_row)
                    if args.diagnostic:
                        tracklet_observation_diagnostics.extend(build_tracklet_observation_rows(
                            slot_dict=slot_dict,
                            source_set=tracklet_source_set,
                            tracklet=tracklet,
                            match=match,
                            cp=cp,
                            camera_id=camera_id,
                            video_path=video_path,
                        ))

                    if tracklet.eligible:
                        tracklet_eligible += 1
                    else:
                        tracklet_rejections[tracklet.rejection_reason] += 1
                    if match.accepted and match.roll_no:
                        tracklet_accepted += 1
                        tracklet_accepted_identities.add(match.roll_no)

                    if args.tracklet_mode != "tracklets":
                        continue
                    if not tracklet.eligible:
                        continue

                    window_metrics[camera_key]["Face_Widths"].append(tracklet_row.get("Face_Width_Median", ""))
                    window_metrics[camera_key]["Face_Heights"].append(tracklet_row.get("Face_Height_Median", ""))
                    window_metrics[camera_key]["Blur_Scores"].append(tracklet_row.get("Blur_Median", ""))
                    window_metrics[camera_key]["Detector_Scores"].append(tracklet_row.get("Detector_Score_Median", ""))
                    window_metrics[camera_key]["Best_Scores"].append(match.best_score)
                    window_metrics[camera_key]["Margins"].append(match.margin)
                    if match.roll_no:
                        candidate_best_counts[match.roll_no] += 1

                    count_for_roll = 0
                    if match.accepted and match.roll_no:
                        evidence_time = f"{cp.cp_id}/{cp.class_time}/{camera_id}/{seconds_to_text(tracklet.end_seconds)}/{tracklet.tracklet_id}"
                        attendance.add_tracklet(
                            match.roll_no,
                            cp.cp_id,
                            match.best_score,
                            match.margin,
                            video_name,
                            camera_id,
                            evidence_time,
                        )
                        checkpoint_metrics[cp.cp_id]["accepted"] += 1
                        checkpoint_metrics[cp.cp_id]["students"].add(match.roll_no)
                        camera_metrics[camera_key]["accepted"] += 1
                        camera_metrics[camera_key]["students"].add(match.roll_no)
                        window_metrics[camera_key]["Accepted"] += 1
                        window_metrics[camera_key]["Unique_Accepted"].add(match.roll_no)
                        candidate_accepted_counts[match.roll_no] += 1
                        count_for_roll = attendance.total_counts.get(match.roll_no, 0)
                    else:
                        checkpoint_metrics[cp.cp_id]["rejected"] += 1
                        camera_metrics[camera_key]["rejected"] += 1
                        window_metrics[camera_key]["Rejected"] += 1

                    if should_log_detection(args.log_mode, match.accepted, match.reason):
                        detection_log.append({
                            **slot_dict,
                            "Diagnostic_Run_ID": diagnostic_run_id,
                            "Checkpoint_ID": cp.cp_id,
                            "Checkpoint_Label": cp.label,
                            "Checkpoint_Class_Time": cp.class_time,
                            "Checkpoint_Mode": checkpoint_mode_used,
                            "Zone_Mode": args.zone_mode,
                            "Tracklet_Mode": args.tracklet_mode,
                            "Video_Window_Start": seconds_to_text(start_sec),
                            "Video_Window_End": seconds_to_text(end_sec),
                            "Video": video_name,
                            "Camera_ID": camera_id,
                            "Frame": tracklet.end_frame,
                            "Time_In_Video": seconds_to_text(tracklet.end_seconds),
                            "Face_Index": tracklet.tracklet_id,
                            "YuNet_Face_Score": tracklet_row.get("Detector_Score_Max", ""),
                            "Accepted": "Yes" if match.accepted else "No",
                            "Predicted_Roll": match.roll_no if match.accepted and match.roll_no else "Unknown",
                            "Best_Roll": match.roll_no or "",
                            "Best_Score": round(match.best_score, 4),
                            "Second_Roll": match.second_roll_no or "",
                            "Second_Score": round(match.second_score, 4),
                            "Margin": round(match.margin, 4),
                            "Reject_Reason": match.reason,
                            "Actual_Matcher_Reject_Reason": match.reason,
                            "Diagnostic_Reject_Reason": tracklet_diagnostic_reason,
                            "Embedding_Extraction_Success": "Yes" if tracklet.aggregate_embedding is not None else "No",
                            "Embedding_Dimension": int(len(tracklet.aggregate_embedding)) if tracklet.aggregate_embedding is not None else 0,
                            "Match_Threshold": args.match_threshold,
                            "Margin_Threshold": args.margin_threshold,
                            "Aggregate_Mode": args.aggregate,
                            "Detection_Count_For_Roll": count_for_roll,
                            "Detection_Source": "tracklet_aggregate",
                            "Original_Equivalent_Face_Width": tracklet_row.get("Face_Width_Median", ""),
                            "Original_Equivalent_Face_Height": tracklet_row.get("Face_Height_Median", ""),
                            "Tracklet_ID": tracklet.tracklet_id,
                            "Tracklet_Source_Set": tracklet_source_set,
                            "Tracklet_Observation_Count": tracklet.observation_count,
                            "Tracklet_Selected_Count": len(tracklet.selected_member_ids),
                            "Tracklet_Consistent_Count": len(tracklet.consistent_member_ids),
                            "Tracklet_Eligible": "Yes" if tracklet.eligible else "No",
                            "Tracklet_Quality_Rejection": "" if tracklet.eligible else tracklet.rejection_reason,
                        })

                tracklet_runtime_seconds = tracklet_build_seconds + (time.perf_counter() - tracklet_finalize_started)
                tracklet_window_comparison.append({
                    **slot_dict,
                    "Diagnostic_Run_ID": diagnostic_run_id,
                    "Tracklet_Mode": args.tracklet_mode,
                    "Tracklet_Source_Set": tracklet_source_set,
                    "Checkpoint_ID": cp.cp_id,
                    "Checkpoint_Label": cp.label,
                    "Checkpoint_Class_Time": cp.class_time,
                    "Camera_ID": camera_id,
                    "Video": video_name,
                    "Source_Frame_Observations": tracklet_source_observations,
                    "Source_Frame_Accepted": tracklet_source_frame_accepted,
                    "Source_Frame_Unique_Accepted": len(tracklet_source_frame_identities),
                    "Tracklets": len(finalized_tracklets),
                    "Eligible_Tracklets": tracklet_eligible,
                    "Accepted_Tracklets": tracklet_accepted,
                    "Unique_Accepted_Tracklet_Identities": len(tracklet_accepted_identities),
                    "Observation_Reduction_Pct": round((1.0 - len(finalized_tracklets) / tracklet_source_observations) * 100.0, 1) if tracklet_source_observations else 0,
                    "Tracklet_Quality_Rejections": json.dumps(dict(tracklet_rejections), sort_keys=True),
                    "Tracklet_Runtime_Seconds": round(tracklet_runtime_seconds, 5),
                    "Official_Attendance_Authority": "tracklet_aggregate" if args.tracklet_mode == "tracklets" else "frame_level_baseline",
                    "Compare_Mode_Changed_Official_Attendance": "No" if args.tracklet_mode == "compare" else "Not_Applicable",
                })

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
    checkpoint_rows = []
    for cp in checkpoints:
        metrics = checkpoint_metrics[cp.cp_id]
        detected = int(metrics.get("detected", 0))
        accepted = int(metrics.get("accepted", 0))
        rejected = int(metrics.get("rejected", 0))
        recognition_rate = round((accepted / detected) * 100, 1) if detected else 0
        unique_students = len(metrics.get("students", set()))
        coverage_rate = round((unique_students / len(attendance_rolls)) * 100, 1) if attendance_rolls else 0
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
    dominant_identity_roll = ""
    dominant_identity_count = 0
    if not detection_log_df.empty and "Accepted" in detection_log_df.columns and "Best_Roll" in detection_log_df.columns:
        accepted_rows = detection_log_df[detection_log_df["Accepted"].eq("Yes")]
        if not accepted_rows.empty:
            counts = accepted_rows["Best_Roll"].astype(str).value_counts()
            dominant_identity_roll = str(counts.index[0])
            dominant_identity_count = int(counts.iloc[0])
    run_quality = evaluate_run_quality(
        total_students=len(attendance_rolls),
        total_checkpoints=len(checkpoints),
        detected_faces=total_detections,
        accepted_recognitions=accepted_count,
        unique_students_recognized=int((attendance_df["Total_Accepted_Detections"] > 0).sum()),
        poor_checkpoints=poor_checkpoints,
        present_count=present_count,
        review_count=review_count,
        absent_count=absent_count,
        dominant_identity_roll=dominant_identity_roll,
        dominant_identity_count=dominant_identity_count,
    )
    quality_fields = quality_to_csv_fields(run_quality)
    for key, value in quality_fields.items():
        attendance_df[key] = value
        student_checkpoint_df[key] = value
    attendance_df["Review_Queue_Reason"] = attendance_df.apply(
        lambda row: review_queue_reason(row.get("Final_Status"), row.get("Flags"), run_quality.requires_manual_review),
        axis=1,
    )
    needs_review_df = attendance_df[
        attendance_df["Final_Status"].eq("Needs Review")
        | attendance_df["Flags"].fillna("").astype(str).ne("")
        | bool(run_quality.requires_manual_review)
    ].copy()

    slot_summary_row = {
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
        "Total_Students": len(attendance_rolls),
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
        **quality_fields,
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
        "Diagnostic_Mode": "Yes" if args.diagnostic else "No",
        "Diagnostic_Run_ID": diagnostic_run_id,
        "Zone_Mode": args.zone_mode,
        "Camera_Zones_Config": str(zone_config_path) if args.zone_mode != "off" else "",
        "Zone_Profile": args.zone_profile,
        "Zone_Merge_IoU": args.zone_merge_iou,
        "Output_Layout": args.output_layout,
        "Aggregate_Mode": args.aggregate,
    }
    if tracklet_config is not None:
        slot_summary_row.update({
            "Tracklet_Mode": args.tracklet_mode,
            "Tracklet_Min_Observations": args.tracklet_min_observations,
            "Tracklet_Max_Selected": args.tracklet_max_selected,
            "Tracklet_Max_Gap_Seconds": args.tracklet_max_gap_seconds,
            "Tracklet_Min_IoU": args.tracklet_min_iou,
            "Tracklet_Max_Center_Ratio": args.tracklet_max_center_ratio,
            "Tracklet_Min_Size_Ratio": args.tracklet_min_size_ratio,
            "Tracklet_Min_Embedding_Similarity": args.tracklet_min_embedding_similarity,
            "Official_Recognition_Unit": "tracklet_aggregate" if args.tracklet_mode == "tracklets" else "frame_detection",
            "Tracklet_Checkpoint_Confirmation": "one accepted eligible tracklet confirms one checkpoint",
        })
    slot_summary_df = pd.DataFrame([slot_summary_row])

    paths: dict[str, Path] = {}
    if not args.diagnostic_only:
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
            output_prefix=args.session_id or None,
        )

    diagnostic_paths: dict[str, Path] = {}
    if args.diagnostic:
        face_diagnostics_df = pd.DataFrame(face_diagnostics)
        checkpoint_diagnostics_df = pd.DataFrame(build_checkpoint_diagnostic_rows(slot_dict, window_metrics))
        rejection_summary_df = build_rejection_summary(face_diagnostics_df)
        zone_candidate_df = pd.DataFrame(zone_candidate_diagnostics)
        zone_frame_df = pd.DataFrame(zone_frame_comparison)
        zone_checkpoint_df = summarize_zone_comparison(zone_candidate_df) if not zone_candidate_df.empty else pd.DataFrame()
        tracklet_df = pd.DataFrame(tracklet_diagnostics)
        tracklet_observation_df = pd.DataFrame(tracklet_observation_diagnostics)
        tracklet_window_df = pd.DataFrame(tracklet_window_comparison)
        tracklet_checkpoint_df = summarize_tracklet_comparison(tracklet_df) if not tracklet_df.empty else pd.DataFrame()

        rejection_counts = Counter()
        if not face_diagnostics_df.empty and "Diagnostic_Reject_Reason" in face_diagnostics_df.columns:
            rejection_counts.update(face_diagnostics_df["Diagnostic_Reject_Reason"].fillna("").astype(str))
        accepted_identity_total = sum(candidate_accepted_counts.values())
        dominant_roll = candidate_accepted_counts.most_common(1)[0][0] if accepted_identity_total else ""
        dominant_count = candidate_accepted_counts.most_common(1)[0][1] if accepted_identity_total else 0
        diagnostic_warnings = sorted({
            warning
            for metrics in window_metrics.values()
            for warning in metrics.get("Timing_Warnings", [])
            if warning
        })
        diagnostic_run_metadata = {
            "diagnostic_run_id": diagnostic_run_id,
            "session_id": args.session_id,
            "session_date": args.session_date,
            "slot_id": slot.slot_id,
            "input_slot": args.input_slot,
            "input_source_type": args.input_source_type,
            "input_source_path": args.input_source_path,
            "section": slot.section,
            "period": f"P{slot.period}",
            "subject_abbr": args.subject_abbr or slot.course_abbr,
            "subject_name": args.subject_name or slot.course_name,
            "faculty_id": args.faculty_id,
            "faculty_name": args.faculty_name,
            "checkpoint_mode": checkpoint_mode_used,
            "log_mode": args.log_mode,
            "aggregate": args.aggregate,
            "match_threshold": args.match_threshold,
            "margin_threshold": args.margin_threshold,
            "sample_fps": args.sample_fps,
            "frame_skip": args.frame_skip,
            "zone_mode": args.zone_mode,
            "camera_zones": str(zone_config_path) if args.zone_mode != "off" else "",
            "zone_profile": args.zone_profile,
            "zone_merge_iou": args.zone_merge_iou,
        }
        if tracklet_config is not None:
            diagnostic_run_metadata.update(
                {
                    "tracklet_mode": args.tracklet_mode,
                    "tracklet_min_observations": args.tracklet_min_observations,
                    "tracklet_max_selected": args.tracklet_max_selected,
                    "tracklet_max_gap_seconds": args.tracklet_max_gap_seconds,
                    "tracklet_min_iou": args.tracklet_min_iou,
                    "tracklet_max_center_ratio": args.tracklet_max_center_ratio,
                    "tracklet_min_size_ratio": args.tracklet_min_size_ratio,
                    "tracklet_min_embedding_similarity": args.tracklet_min_embedding_similarity,
                    "official_recognition_unit": (
                        "tracklet_aggregate"
                        if args.tracklet_mode == "tracklets"
                        else "frame_detection"
                    ),
                    "tracklet_checkpoint_confirmation": (
                        "one accepted eligible tracklet confirms one checkpoint"
                    ),
                    "diagnostic_only": bool(args.diagnostic_only),
                    "official_attendance_written": not bool(args.diagnostic_only),
                }
            )
        diagnostic_summary = {
            "run_metadata": diagnostic_run_metadata,
            "input_files": [
                {
                    "source_file": str(meta.path),
                    "source_file_name": meta.path.name,
                    "camera_id": video_camera_map.get(meta.path.name, infer_camera_id(meta.path, slot.camera_ids)),
                    "fps": round(meta.fps, 3),
                    "frame_count": int(meta.frame_count),
                    "duration_seconds": round(meta.duration_sec, 3),
                }
                for meta in metas
            ],
            "checkpoint_timing_map": dataframe_records(checkpoint_diagnostics_df),
            "overall_counts": {
                "frames_processed": total_processed_frames,
                "faces_detected": total_detections,
                "accepted_recognitions": accepted_count,
                "rejected_or_unknown": rejected_count,
                "unique_students_recognized": int((attendance_df["Total_Accepted_Detections"] > 0).sum()),
                "students_present": present_count,
                "students_needs_review": review_count,
                "students_absent": absent_count,
                "skipped_windows": total_skipped_windows,
            },
            "rejection_distribution": dict(rejection_counts),
            "dominant_rejection_reason": next((reason for reason, _ in rejection_counts.most_common() if reason != "accepted"), ""),
            "camera_comparison": dataframe_records(camera_summary_df),
            "checkpoint_comparison": dataframe_records(checkpoint_summary_df),
            "quality_distributions": {
                **percentile_fields(face_diagnostics_df["Face_Width"] if "Face_Width" in face_diagnostics_df else [], "Face_Width", decimals=2),
                **percentile_fields(face_diagnostics_df["Face_Height"] if "Face_Height" in face_diagnostics_df else [], "Face_Height", decimals=2),
                **percentile_fields(face_diagnostics_df["Face_Area_Ratio"] if "Face_Area_Ratio" in face_diagnostics_df else [], "Face_Area_Ratio", decimals=8),
                **percentile_fields(face_diagnostics_df["Blur_Laplacian_Variance"] if "Blur_Laplacian_Variance" in face_diagnostics_df else [], "Blur_Laplacian_Variance", decimals=2),
                **percentile_fields(face_diagnostics_df["YuNet_Face_Score"] if "YuNet_Face_Score" in face_diagnostics_df else [], "Detector_Score", decimals=4),
                **percentile_fields(face_diagnostics_df["Best_Score"] if "Best_Score" in face_diagnostics_df else [], "Best_Score", decimals=4),
                **percentile_fields(face_diagnostics_df["Margin"] if "Margin" in face_diagnostics_df else [], "Margin", decimals=4),
            },
            "candidate_dominance": {
                "best_candidate_counts": dict(candidate_best_counts.most_common(20)),
                "accepted_candidate_counts": dict(candidate_accepted_counts.most_common(20)),
                "dominant_accepted_roll": dominant_roll,
                "dominant_accepted_count": dominant_count,
                "dominant_accepted_share_pct": round((dominant_count / accepted_identity_total) * 100.0, 1) if accepted_identity_total else 0,
            },
            "warnings": diagnostic_warnings,
            "checkpoint_extraction_valid": all(metrics.get("Window_Valid") == "Yes" for metrics in window_metrics.values()),
            "normal_output_files": {name: str(path) for name, path in paths.items()},
        }
        if args.zone_mode != "off":
            zone_set_summary = {}
            if not zone_candidate_df.empty:
                for detection_set, group_df in zone_candidate_df.groupby("Detection_Set", dropna=False):
                    accepted_rows = group_df[group_df["Accepted"].eq("Yes")]
                    top_accepted = accepted_rows["Best_Roll"].astype(str).value_counts().head(10).to_dict() if not accepted_rows.empty else {}
                    zone_set_summary[str(detection_set)] = {
                        "detections": int(len(group_df)),
                        "accepted": int(group_df["Accepted"].eq("Yes").sum()),
                        "unique_accepted": int(accepted_rows["Best_Roll"].replace("", pd.NA).dropna().nunique()) if not accepted_rows.empty else 0,
                        "top_accepted_identities": top_accepted,
                        "quality_counts": group_df["Diagnostic_Quality_Label"].astype(str).value_counts().to_dict(),
                        "face_width": percentile_fields(group_df["Original_Equivalent_Face_Width"], "Face_Width", decimals=2),
                        "best_score": percentile_fields(group_df["Best_Score"], "Best_Score", decimals=4),
                        "margin": percentile_fields(group_df["Margin"], "Margin", decimals=4),
                    }
            zone_frame_summary = {}
            if not zone_frame_df.empty:
                numeric_cols = [
                    "Full_Frame_Detections",
                    "Zone_Premerge_Detections",
                    "Zone_Only_Merged_Detections",
                    "Merged_Full_Frame_And_Zone_Detections",
                    "Duplicates_Removed",
                    "Full_Detect_Seconds",
                    "Zone_Detect_Seconds",
                    "Merge_Seconds",
                    "Recognition_Seconds",
                ]
                for col in numeric_cols:
                    zone_frame_summary[col] = round(float(pd.to_numeric(zone_frame_df[col], errors="coerce").fillna(0).sum()), 4)
            diagnostic_summary["zone_comparison"] = {
                "set_summary": zone_set_summary,
                "frame_summary": zone_frame_summary,
                "checkpoint_rows": dataframe_records(zone_checkpoint_df),
            }

        if args.tracklet_mode != "off":
            eligible_tracklet_rows = (
                tracklet_df[tracklet_df["Tracklet_Eligible"].eq("Yes")]
                if not tracklet_df.empty
                else pd.DataFrame()
            )
            accepted_tracklet_rows = (
                tracklet_df[tracklet_df["Tracklet_Accepted"].eq("Yes")]
                if not tracklet_df.empty
                else pd.DataFrame()
            )
            accepted_tracklet_counts = (
                accepted_tracklet_rows["Tracklet_Best_Roll"].replace("", pd.NA).dropna().astype(str).value_counts()
                if not accepted_tracklet_rows.empty
                else pd.Series(dtype="int64")
            )
            tracklet_quality_rejections = (
                tracklet_df["Tracklet_Quality_Rejection"].replace("", pd.NA).dropna().astype(str).value_counts().to_dict()
                if not tracklet_df.empty
                else {}
            )
            tracklet_window_summary = {}
            if not tracklet_window_df.empty:
                for column in [
                    "Source_Frame_Observations",
                    "Source_Frame_Accepted",
                    "Tracklets",
                    "Eligible_Tracklets",
                    "Accepted_Tracklets",
                    "Tracklet_Runtime_Seconds",
                ]:
                    tracklet_window_summary[column] = round(
                        float(pd.to_numeric(tracklet_window_df[column], errors="coerce").fillna(0).sum()),
                        5,
                    )
            dominant_tracklet_roll = str(accepted_tracklet_counts.index[0]) if len(accepted_tracklet_counts) else ""
            dominant_tracklet_count = int(accepted_tracklet_counts.iloc[0]) if len(accepted_tracklet_counts) else 0
            accepted_tracklet_total = int(accepted_tracklet_counts.sum()) if len(accepted_tracklet_counts) else 0
            diagnostic_summary["tracklet_comparison"] = {
                "mode": args.tracklet_mode,
                "official_attendance_authority": "tracklet_aggregate" if args.tracklet_mode == "tracklets" else "frame_level_baseline",
                "compare_mode_changed_official_attendance": False if args.tracklet_mode == "compare" else None,
                "window_summary": tracklet_window_summary,
                "tracklets_total": int(len(tracklet_df)),
                "eligible_tracklets": int(tracklet_df["Tracklet_Eligible"].eq("Yes").sum()) if not tracklet_df.empty else 0,
                "accepted_tracklets": int(len(accepted_tracklet_rows)),
                "unique_accepted_identities": int(accepted_tracklet_rows["Tracklet_Best_Roll"].replace("", pd.NA).dropna().nunique()) if not accepted_tracklet_rows.empty else 0,
                "top_accepted_identities": accepted_tracklet_counts.head(20).to_dict(),
                "dominant_accepted_roll": dominant_tracklet_roll,
                "dominant_accepted_count": dominant_tracklet_count,
                "dominant_accepted_share_pct": round((dominant_tracklet_count / accepted_tracklet_total) * 100.0, 1) if accepted_tracklet_total else 0,
                "quality_rejections": tracklet_quality_rejections,
                "observation_count": percentile_fields(tracklet_df["Observation_Count"] if "Observation_Count" in tracklet_df else [], "Observation_Count", decimals=2),
                "recognition_distribution_scope": "eligible_tracklets_only",
                "best_score": percentile_fields(eligible_tracklet_rows["Tracklet_Best_Score"] if "Tracklet_Best_Score" in eligible_tracklet_rows else [], "Best_Score", decimals=4),
                "margin": percentile_fields(eligible_tracklet_rows["Tracklet_Margin"] if "Tracklet_Margin" in eligible_tracklet_rows else [], "Margin", decimals=4),
                "pairwise_similarity": percentile_fields(eligible_tracklet_rows["Pairwise_Similarity_Median"] if "Pairwise_Similarity_Median" in eligible_tracklet_rows else [], "Pairwise_Similarity", decimals=4),
                "eligible_acceptance_rate_pct": round((len(accepted_tracklet_rows) / len(eligible_tracklet_rows)) * 100.0, 1) if len(eligible_tracklet_rows) else 0,
                "checkpoint_rows": dataframe_records(tracklet_checkpoint_df),
                "window_rows": dataframe_records(tracklet_window_df),
            }

        diagnostic_base_dir = Path(args.diagnostic_dir) if args.diagnostic_dir else output_dir / "diagnostics"
        diagnostic_root = diagnostic_base_dir / diagnostic_run_id
        diagnostic_paths = save_diagnostic_outputs(
            face_diagnostics_df=face_diagnostics_df,
            checkpoint_diagnostics_df=checkpoint_diagnostics_df,
            rejection_summary_df=rejection_summary_df,
            diagnostic_summary=diagnostic_summary,
            diagnostic_root=diagnostic_root,
            diagnostic_run_id=diagnostic_run_id,
        )
        if args.zone_mode != "off":
            zone_paths = {
                "zone_candidate_comparison_csv": diagnostic_root / f"zone_candidate_comparison_{diagnostic_run_id}.csv",
                "zone_frame_comparison_csv": diagnostic_root / f"zone_frame_comparison_{diagnostic_run_id}.csv",
                "zone_checkpoint_comparison_csv": diagnostic_root / f"zone_checkpoint_comparison_{diagnostic_run_id}.csv",
                "zone_comparison_json": diagnostic_root / f"zone_comparison_{diagnostic_run_id}.json",
            }
            zone_candidate_df.to_csv(zone_paths["zone_candidate_comparison_csv"], index=False)
            zone_frame_df.to_csv(zone_paths["zone_frame_comparison_csv"], index=False)
            zone_checkpoint_df.to_csv(zone_paths["zone_checkpoint_comparison_csv"], index=False)
            with zone_paths["zone_comparison_json"].open("w", encoding="utf-8") as f:
                json.dump(json_safe(diagnostic_summary.get("zone_comparison", {})), f, indent=2)
            diagnostic_paths.update(zone_paths)
        if args.tracklet_mode != "off":
            tracklet_paths = {
                "tracklet_diagnostics_csv": diagnostic_root / f"tracklet_diagnostics_{diagnostic_run_id}.csv",
                "tracklet_observations_csv": diagnostic_root / f"tracklet_observations_{diagnostic_run_id}.csv",
                "tracklet_checkpoint_comparison_csv": diagnostic_root / f"tracklet_checkpoint_comparison_{diagnostic_run_id}.csv",
                "tracklet_window_comparison_csv": diagnostic_root / f"tracklet_window_comparison_{diagnostic_run_id}.csv",
                "tracklet_comparison_json": diagnostic_root / f"tracklet_comparison_{diagnostic_run_id}.json",
            }
            tracklet_df.to_csv(tracklet_paths["tracklet_diagnostics_csv"], index=False)
            tracklet_observation_df.to_csv(tracklet_paths["tracklet_observations_csv"], index=False)
            tracklet_checkpoint_df.to_csv(tracklet_paths["tracklet_checkpoint_comparison_csv"], index=False)
            tracklet_window_df.to_csv(tracklet_paths["tracklet_window_comparison_csv"], index=False)
            with tracklet_paths["tracklet_comparison_json"].open("w", encoding="utf-8") as f:
                json.dump(json_safe(diagnostic_summary.get("tracklet_comparison", {})), f, indent=2)
            diagnostic_paths.update(tracklet_paths)
            if args.export_tracklet_review:
                from src.face_attendance.tracklet_review import ReviewExportError, export_review_package

                review_parent = Path(args.tracklet_review_dir) if args.tracklet_review_dir else diagnostic_root
                try:
                    review_package = export_review_package(
                        tracklet_df=tracklet_df,
                        observation_df=tracklet_observation_df,
                        video_root=video_dir,
                        output_root=review_parent,
                        student_map_path=Path(args.tracklet_review_student_map),
                        subject_abbr=args.subject_abbr or slot.course_abbr,
                        diagnostic_run_id=diagnostic_run_id,
                        evidence_count=args.tracklet_review_evidence_count,
                        crop_padding=args.tracklet_review_crop_padding,
                        source_files={
                            "tracklet_diagnostics": tracklet_paths["tracklet_diagnostics_csv"],
                            "tracklet_observations": tracklet_paths["tracklet_observations_csv"],
                        },
                    )
                except (OSError, ReviewExportError) as exc:
                    raise SystemExit(f"Tracklet review export failed: {exc}") from exc
                diagnostic_paths["tracklet_review_package"] = review_package.root
                diagnostic_paths["blind_tracklet_reviewer"] = review_package.reviewer_dir

    print("\nMVP 3 checkpoint processing completed")
    print(f"Subject: {slot.course_abbr} - {slot.course_name}")
    print(f"Slot: {slot.day} P{slot.period} | {slot.start_time}-{slot.end_time}")
    if args.diagnostic_only:
        print("Mode: diagnostic-only shadow processing")
        print("Official attendance output: not written")
    else:
        print(f"Present: {present_count} / {len(attendance_rolls)}")
        print(f"Present Strong: {strong_count}")
        print(f"Needs Review: {review_count}")
        print(f"Absent: {absent_count}")
    print(f"Checkpoint mode used: {checkpoint_mode_used}")
    if poor_checkpoints:
        print(f"Poor checkpoints: {', '.join(poor_checkpoints)}")
    print(f"Run quality: {run_quality.status}")
    if run_quality.requires_manual_review:
        print(f"Quality reason: {run_quality.reason}")
    print(f"Runtime: {time.perf_counter() - run_start_perf:.2f} seconds")
    print(f"Unknown/rejected faces saved: {unknown_saved}" if args.save_unknown else "Unknown saving: off")
    for name, path in paths.items():
        print(f"Saved {name}: {path}")
    for name, path in diagnostic_paths.items():
        print(f"Saved {name}: {path}")


if __name__ == "__main__":
    main()
