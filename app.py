from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import signal
import queue
import re
import secrets
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from src.face_attendance.run_quality import quality_from_slot_summary, quality_to_status_fields
from src.face_attendance.session_identity import (
    build_attendance_session_id,
    clean_filename as clean_session_filename,
    normalize_session_date,
    parse_attendance_session_id,
    strip_runtime_fields,
)
from src.face_attendance.workflow_state import (
    WorkflowStateError,
    active_job_for_session,
    atomic_write_json,
    compact_job_for_storage,
    load_job_registry,
    migrate_status_file,
    read_json_object,
    sanitize_status_store,
    serialize_job_registry,
)
from src.face_attendance.processing_integration import (
    POLICY_VERSION as PROCESSING_CONTRACT_POLICY_VERSION,
    PROCESSING_MODE as QUALITY_AWARE_PROCESSING_MODE,
    ProcessingIntegrationError,
    build_processing_command,
    fixed_processing_policy,
    inspect_exact_checkpoint_layout,
)
from src.face_attendance.product_phase_2i_authority import (
    ProductPhase2IError,
    apply_automatic_status_semantics,
    select_guarded_review_candidates,
)
from src.face_attendance.report_revision_control import (
    ReportRevisionError,
    preserve_official_or_commit_candidate,
)
from src.face_attendance.review_carry_forward import (
    ReviewCarryForwardError,
    apply_exact_review_carry_forward,
    load_registry as load_review_carry_forward_registry,
    source_fingerprint as review_source_fingerprint,
)
from src.face_attendance.config_governance import (
    ConfigGovernanceError,
    apply_config_transaction,
    assign_subject_to_faculty,
    create_or_update_faculty,
    json_bytes as config_json_bytes,
    load_config_history,
    rollback_latest_config_change,
    timetable_bytes as config_timetable_bytes,
    validate_role_users,
    validate_timetable_bytes,
    verify_password,
)

ROOT_DIR = Path(__file__).resolve().parent
TIMETABLE_CANDIDATES = [
    ROOT_DIR / "timetable_b51_2026_2027.csv",
    ROOT_DIR / "timetable" / "class_timetable.csv",
    ROOT_DIR / "frontend" / "public" / "timetable_b51_2026_2027.csv",
]
OUTPUT_DIR = ROOT_DIR / "attendance_output"
VIDEO_DIR = ROOT_DIR / "cctv_videos"
DATA_DIR = ROOT_DIR / "data"
STATUS_PATH = DATA_DIR / "attendance_status.json"
OVERRIDES_PATH = DATA_DIR / "manual_overrides.json"
JOB_RUNTIME_PATH = DATA_DIR / "job_runtime.json"
STATE_BACKUP_DIR = DATA_DIR / "state_backups"
REVIEW_CARRY_FORWARD_REGISTRY_PATH = DATA_DIR / "review_evidence_registry.json"
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

DEFAULT_MATCH_THRESHOLD = 0.48
DEFAULT_MARGIN_THRESHOLD = 0.08
DEFAULT_MIN_DETECTIONS = 15  # kept for MVP2 compatibility
DEFAULT_REVIEW_MIN_DETECTIONS = 7  # kept for MVP2 compatibility
DEFAULT_FRAME_SKIP = 3
DEFAULT_SAMPLE_FPS = 2.0
DEFAULT_LOG_MODE = "accepted"
DEFAULT_OUTPUT_LAYOUT = "organized"
DEFAULT_CHECKPOINT_MIN_DETECTIONS = 2
DEFAULT_PRESENT_CHECKPOINTS = 3
DEFAULT_STRONG_CHECKPOINTS = 4
DEFAULT_REVIEW_CHECKPOINTS = 2
DEFAULT_SETTLE_MINUTES = 10
DEFAULT_CHECKPOINT_EVERY_MINUTES = 10
DEFAULT_CHECKPOINT_CLIP_SECONDS = 20
DEFAULT_CHECKPOINT_MODE = "auto"  # auto supports both full 5-checkpoint folders and current 3-demo-clip testing
DEFAULT_AGGREGATE = "top3"
SAVE_UNKNOWN_BY_DEFAULT = False  # keep storage safe from thousands of rejected crops
DEFAULT_JOB_TIMEOUT_SECONDS = 20 * 60
AUTH_SESSION_TTL_SECONDS = 12 * 60 * 60

# Reliable demo/session metadata. Slot folder names do not contain dates; these
# mappings are explicit project knowledge, not filesystem timestamp inference.
KNOWN_SLOT_SESSION_DATES = {
    "TUE_P1": "2026-06-30",
    "TUE_P2": "2026-06-30",
    "MON_P3": "2026-06-22",
    "MON_P4": "2026-06-22",
}
FRONTEND_POLL_SECONDS = 3

ADMIN_PROFILE = {
    "name": "HOD",
    "role": "Head of Department",
    "organization": "Sreenidhi University",
    "department": "Smart Attendance Cell",
    "email": "admin@sreenidhi.edu.in",
}


USERS_PATH = DATA_DIR / "role_users.json"
STUDENT_MAP_PATH = DATA_DIR / "student_faculty_map.json"
EMBEDDING_SUMMARY_PATH = ROOT_DIR / "models" / "embedding_summary.csv"
DATASET_ROOT = ROOT_DIR / "dataset"
CURRENT_EMBEDDING_VERSION_PATH = ROOT_DIR / "models" / "current_embedding_version.json"
CONFIG_BACKUP_ROOT = DATA_DIR / "config_backups"
CONFIG_AUDIT_ROOT = OUTPUT_DIR / "product_workflow" / "phase_2f"
CONFIG_POINTER_PATH = DATA_DIR / "config_current.json"
CONFIG_WRITE_LOCK = threading.RLock()

SUBJECT_INFO = {
    "SWE": {"abbr": "SWE", "course_code": "24CSEJ601", "course_name": "Software Engineering", "faculty_id": "keerthi", "faculty_name": "Dr. Keerthi G"},
    "CCM": {"abbr": "CCM", "course_code": "24CSEJ602", "course_name": "Cloud Computing", "faculty_id": "pranay", "faculty_name": "Dr. Pranayaanath Reddy A"},
    "CVO": {"abbr": "CVO", "course_code": "24AMLJ502", "course_name": "Computer Vision through OpenCV", "faculty_id": "vikas", "faculty_name": "Mr. Vikas B"},
}

DEFAULT_ROLE_USERS = [
    {"id": "admin", "username": "admin", "password": "admin123", "name": "Main Admin", "role": "admin", "roleLabel": "Attendance Controller", "subjects": ["SWE", "CCM", "CVO"], "facultyName": "Main Admin", "canManageSystem": True, "canSeeAll": True},
    {"id": "keerthi", "username": "keerthi", "password": "swe123", "name": "Dr. Keerthi G", "role": "faculty", "roleLabel": "Faculty - Software Engineering", "subjects": ["SWE"], "facultyName": "Dr. Keerthi G", "canManageSystem": False, "canSeeAll": False},
    {"id": "vikas", "username": "vikas", "password": "cvo123", "name": "Mr. Vikas B", "role": "faculty", "roleLabel": "Faculty - Computer Vision through OpenCV", "subjects": ["CVO"], "facultyName": "Mr. Vikas B", "canManageSystem": False, "canSeeAll": False},
    {"id": "pranay", "username": "pranay", "password": "ccm123", "name": "Dr. Pranayaanath Reddy A", "role": "faculty", "roleLabel": "Faculty - Cloud Computing", "subjects": ["CCM"], "facultyName": "Dr. Pranayaanath Reddy A", "canManageSystem": False, "canSeeAll": False},
]

app = Flask(__name__)
CORS(app)

JOBS: dict[str, dict] = {}
RUNNING_PROCESSES: dict[str, subprocess.Popen] = {}
AUTH_SESSIONS: dict[str, dict] = {}
AUTH_LOCK = threading.RLock()
CANCELLED_JOBS: set[str] = set()
JOBS_HYDRATED = False
WORKFLOW_STATE_REPORT: dict = {}
WORKFLOW_STATE_INITIALIZED = False

LOCK = threading.RLock()


class LiveDemoService:
    """Backend webcam service for the presentation demo page."""

    def __init__(self) -> None:
        from collections import deque

        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.running = False
        self.latest_jpeg: bytes | None = None
        self.events = deque(maxlen=80)
        self.recognized: dict[str, dict] = {}
        self.stats = self._blank_stats()
        self.error: str | None = None
        self.started_at: str | None = None
        self.started_by: str | None = None
        self.settings: dict = {}
        self._engine = None
        self._db = None
        self._last_faces: list[dict] = []

    @staticmethod
    def _blank_stats() -> dict:
        return {
            "frames": 0,
            "faces_detected": 0,
            "recognized_total": 0,
            "unknown_total": 0,
            "recognized_unique": 0,
            "fps": 0.0,
            "camera_index": 0,
            "actual_camera_index": None,
            "model": "YuNet + SFace",
            "last_frame_at": None,
        }

    @staticmethod
    def _safe_int(value, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
        try:
            parsed = int(value)
        except Exception:
            parsed = default
        if minimum is not None:
            parsed = max(minimum, parsed)
        if maximum is not None:
            parsed = min(maximum, parsed)
        return parsed

    @staticmethod
    def _safe_float(value, default: float, minimum: float | None = None, maximum: float | None = None) -> float:
        try:
            parsed = float(value)
        except Exception:
            parsed = default
        if minimum is not None:
            parsed = max(minimum, parsed)
        if maximum is not None:
            parsed = min(maximum, parsed)
        return parsed

    def _load_models(self, options: dict) -> None:
        from src.face_attendance.config import (
            DEFAULT_NMS_THRESHOLD,
            DEFAULT_TOP_K,
            EMBEDDINGS_PATH,
            SFACE_MODEL,
            YUNET_MODEL,
        )
        from src.face_attendance.embedding_db import StudentEmbeddingDB
        from src.face_attendance.face_engine import FaceEngine

        if not Path(EMBEDDINGS_PATH).exists():
            raise FileNotFoundError(f"Student embeddings not found: {EMBEDDINGS_PATH}. Run build_embeddings.py first.")
        if not Path(YUNET_MODEL).exists():
            raise FileNotFoundError(f"YuNet model not found: {YUNET_MODEL}.")
        if not Path(SFACE_MODEL).exists():
            raise FileNotFoundError(f"SFace model not found: {SFACE_MODEL}.")

        detection_score = self._safe_float(options.get("detection_score"), 0.78, 0.30, 0.99)
        self._db = StudentEmbeddingDB.load(Path(EMBEDDINGS_PATH))
        self._engine = FaceEngine(
            Path(YUNET_MODEL),
            Path(SFACE_MODEL),
            detection_score=detection_score,
            nms_threshold=DEFAULT_NMS_THRESHOLD,
            top_k=DEFAULT_TOP_K,
        )

    def _open_camera(self, cv2, preferred_index: int):
        candidates = []
        for value in [preferred_index, 0, 1, 2]:
            if value not in candidates:
                candidates.append(value)

        backend = cv2.CAP_DSHOW if os.name == "nt" and hasattr(cv2, "CAP_DSHOW") else 0
        for index in candidates:
            cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
            if cap is not None and cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap, index
            if cap is not None:
                cap.release()
        return None, None

    def start(self, payload: dict | None, user: dict | None = None) -> dict:
        payload = dict(payload or {})
        with self.lock:
            if self.running:
                return {"success": True, "message": "Live demo is already running.", "state": self.state()}

            options = {
                "camera_index": self._safe_int(payload.get("camera_index", payload.get("cameraIndex", 0)), 0, 0, 8),
                "match_threshold": self._safe_float(payload.get("match_threshold", payload.get("matchThreshold", DEFAULT_MATCH_THRESHOLD)), DEFAULT_MATCH_THRESHOLD, 0.10, 0.95),
                "margin_threshold": self._safe_float(payload.get("margin_threshold", payload.get("marginThreshold", DEFAULT_MARGIN_THRESHOLD)), DEFAULT_MARGIN_THRESHOLD, 0.00, 0.50),
                "detection_score": self._safe_float(payload.get("detection_score", payload.get("detectionScore", 0.78)), 0.78, 0.30, 0.99),
                "process_every_n": self._safe_int(payload.get("process_every_n", payload.get("processEveryN", 2)), 2, 1, 12),
                "display_width": self._safe_int(payload.get("display_width", payload.get("displayWidth", 1040)), 1040, 480, 1600),
                "process_width": self._safe_int(payload.get("process_width", payload.get("processWidth", 960)), 960, 320, 1600),
                "mirror": bool(payload.get("mirror", True)),
                "aggregate": payload.get("aggregate") if payload.get("aggregate") in {"centroid", "max", "top3"} else "top3",
            }

            self.stop_event.clear()
            self.latest_jpeg = None
            self.events.clear()
            self.recognized = {}
            self.stats = self._blank_stats()
            self.stats["camera_index"] = options["camera_index"]
            self.error = None
            self.started_at = datetime.now().isoformat(timespec="seconds")
            self.started_by = (user or {}).get("name") or "Demo User"
            self.settings = options
            self._last_faces = []

            try:
                self._load_models(options)
            except Exception as exc:
                self.error = str(exc)
                self.running = False
                return {"success": False, "error": self.error, "state": self.state()}

            self.running = True
            self.thread = threading.Thread(target=self._run_loop, name="live-demo-camera", daemon=True)
            self.thread.start()
            return {"success": True, "message": "Live demo started.", "state": self.state()}

    def stop(self) -> dict:
        with self.lock:
            was_running = self.running
            self.stop_event.set()
        thread = self.thread
        if thread and thread.is_alive():
            thread.join(timeout=3.0)
        with self.lock:
            self.running = False
            if not was_running and not self.error:
                self.error = None
        return {"success": True, "message": "Live demo stopped.", "state": self.state()}

    def _add_event(self, event: dict) -> None:
        event["id"] = f"evt_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
        self.events.appendleft(event)

    def _record_recognition(self, roll: str, event: dict) -> int:
        now = event["time"]
        current = self.recognized.get(roll) or {
            "roll": roll,
            "label": roll,
            "count": 0,
            "best_score": 0.0,
            "first_seen": now,
            "last_seen": now,
            "status": "Recognized",
        }
        current["count"] += 1
        current["best_score"] = max(float(current.get("best_score") or 0.0), float(event.get("score") or 0.0))
        current["last_seen"] = now
        self.recognized[roll] = current
        return int(current["count"])

    @staticmethod
    def _draw_label(cv2, frame, box, label: str, color: tuple[int, int, int], score: float, margin: float) -> None:
        x, y, w, h = [int(round(v)) for v in box]
        h_img, w_img = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_img - 1, x + max(1, w)), min(h_img - 1, y + max(1, h))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        text = f"{label}  {score:.2f}"
        detail = f"margin {margin:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        y0 = max(24, y1 - 12)
        cv2.rectangle(frame, (x1, y0 - th - 10), (min(w_img - 1, x1 + tw + 16), y0 + 8), color, -1)
        cv2.putText(frame, text, (x1 + 8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (8, 18, 28), 2, cv2.LINE_AA)
        cv2.putText(frame, detail, (x1, min(h_img - 12, y2 + 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

    @staticmethod
    def _draw_hud(cv2, frame, stats: dict, running: bool, error: str | None) -> None:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (18, 18), (min(w - 18, 590), 126), (5, 18, 34), -1)
        cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)
        status = "LIVE FACE AUTHENTICATION DEMO" if running else "DEMO PAUSED"
        cv2.putText(frame, status, (34, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 255, 190), 2, cv2.LINE_AA)
        cv2.putText(frame, "YuNet detection  +  SFace recognition  +  cosine matching", (34, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (235, 245, 255), 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"Faces: {stats.get('faces_detected', 0)}   Recognized: {stats.get('recognized_total', 0)}   Unknown: {stats.get('unknown_total', 0)}   FPS: {stats.get('fps', 0):.1f}",
            (34, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        if error:
            cv2.putText(frame, f"ERROR: {error[:70]}", (24, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 120, 255), 2, cv2.LINE_AA)

    @staticmethod
    def _resize_keep_aspect(cv2, frame, max_width: int):
        if not max_width or frame.shape[1] <= max_width:
            return frame
        scale = max_width / float(frame.shape[1])
        new_h = int(round(frame.shape[0] * scale))
        return cv2.resize(frame, (max_width, new_h), interpolation=cv2.INTER_AREA)

    def _run_loop(self) -> None:
        import cv2

        cap = None
        fps_window_start = time.time()
        fps_window_frames = 0
        try:
            with self.lock:
                options = dict(self.settings)
            cap, actual_index = self._open_camera(cv2, options["camera_index"])
            if cap is None:
                raise RuntimeError("Could not open camera. Try Camera ID 1, close other camera apps, then start again.")
            with self.lock:
                self.stats["actual_camera_index"] = actual_index

            frame_index = 0
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError("Camera frame could not be read. Restart the demo or try another camera ID.")

                frame_index += 1
                if options.get("mirror", True):
                    frame = cv2.flip(frame, 1)

                process_now = frame_index % max(1, int(options["process_every_n"])) == 0
                frame_faces: list[dict] = []
                if process_now and self._engine is not None and self._db is not None:
                    detections = self._engine.detect_and_extract(frame, max_width=int(options["process_width"]))
                    for item in detections:
                        match = self._db.match(
                            item["embedding"],
                            match_threshold=float(options["match_threshold"]),
                            margin_threshold=float(options["margin_threshold"]),
                            aggregate=options.get("aggregate", "top3"),
                        )
                        now = datetime.now().strftime("%H:%M:%S")
                        if match.accepted and match.roll_no:
                            label = str(match.roll_no)
                            event = {
                                "time": now,
                                "roll": label,
                                "label": label,
                                "status": "Recognized",
                                "score": round(float(match.best_score), 4),
                                "margin": round(float(match.margin), 4),
                                "face_score": round(float(item.get("score") or 0), 4),
                                "reason": "accepted",
                            }
                            count = self._record_recognition(label, event)
                            event["count"] = count
                            frame_faces.append({**event, "box": item["box"]})
                            with self.lock:
                                self.stats["recognized_total"] += 1
                        else:
                            best = match.roll_no or "Unknown"
                            event = {
                                "time": now,
                                "roll": "Unknown",
                                "label": "Unknown",
                                "status": "Unknown",
                                "score": round(float(match.best_score), 4),
                                "margin": round(float(match.margin), 4),
                                "face_score": round(float(item.get("score") or 0), 4),
                                "reason": match.reason or "not_matched",
                                "best_candidate": best,
                                "count": "-",
                            }
                            frame_faces.append({**event, "box": item["box"]})
                            with self.lock:
                                self.stats["unknown_total"] += 1
                        with self.lock:
                            self._add_event(event)
                            self.stats["faces_detected"] += 1
                    with self.lock:
                        self._last_faces = frame_faces
                else:
                    with self.lock:
                        frame_faces = list(self._last_faces)

                for face in frame_faces:
                    status = face.get("status")
                    color = (22, 234, 166) if status == "Recognized" else (52, 120, 255)
                    label = face.get("label") or "Unknown"
                    self._draw_label(cv2, frame, face["box"], label, color, float(face.get("score") or 0.0), float(face.get("margin") or 0.0))

                now_time = time.time()
                fps_window_frames += 1
                if now_time - fps_window_start >= 1.0:
                    with self.lock:
                        self.stats["fps"] = round(fps_window_frames / max(0.001, now_time - fps_window_start), 1)
                    fps_window_start = now_time
                    fps_window_frames = 0

                with self.lock:
                    self.stats["frames"] = frame_index
                    self.stats["recognized_unique"] = len(self.recognized)
                    self.stats["last_frame_at"] = datetime.now().isoformat(timespec="seconds")
                    stats_copy = dict(self.stats)
                    error_copy = self.error
                    running_copy = self.running

                self._draw_hud(cv2, frame, stats_copy, running_copy, error_copy)
                frame = self._resize_keep_aspect(cv2, frame, int(options["display_width"]))
                ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                if ok:
                    with self.lock:
                        self.latest_jpeg = jpeg.tobytes()

                time.sleep(0.01)
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
        finally:
            if cap is not None:
                cap.release()
            with self.lock:
                self.running = False

    def placeholder_jpeg(self, text: str | None = None) -> bytes:
        import cv2
        import numpy as np

        message = text or self.error or "Start the live demo to open the camera"
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        frame[:] = (8, 25, 43)
        cv2.rectangle(frame, (28, 28), (932, 512), (24, 68, 92), 2)
        cv2.putText(frame, "SREENIDHI SMART ATTENDANCE", (54, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.88, (20, 255, 190), 2, cv2.LINE_AA)
        cv2.putText(frame, "Live Face Recognition Demo", (54, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (240, 248, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, message[:80], (54, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (205, 226, 240), 1, cv2.LINE_AA)
        cv2.putText(frame, "Click Start Demo to begin webcam detection", (54, 282), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (52, 211, 153), 2, cv2.LINE_AA)
        ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        return jpeg.tobytes() if ok else b""

    def get_jpeg(self) -> bytes:
        with self.lock:
            frame = self.latest_jpeg
            error = self.error
            running = self.running
        if frame:
            return frame
        return self.placeholder_jpeg(error if error and not running else None)

    def state(self) -> dict:
        with self.lock:
            return {
                "running": self.running,
                "error": self.error,
                "started_at": self.started_at,
                "started_by": self.started_by,
                "settings": dict(self.settings),
                "stats": dict(self.stats),
                "events": list(self.events)[:30],
                "recognized": sorted(self.recognized.values(), key=lambda item: (-int(item.get("count") or 0), item.get("roll") or ""))[:30],
            }


LIVE_DEMO = LiveDemoService()
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
VIDEO_DIR.mkdir(exist_ok=True)


def now_text() -> str:
    return datetime.now().strftime("%d-%m-%Y %I:%M:%S %p")


def normalize_day(day: str) -> str:
    value = (day or "").strip().lower()
    mapping = {
        "mon": "Monday", "monday": "Monday",
        "tue": "Tuesday", "tuesday": "Tuesday",
        "wed": "Wednesday", "wednesday": "Wednesday",
        "thu": "Thursday", "thursday": "Thursday",
        "fri": "Friday", "friday": "Friday",
        "sat": "Saturday", "saturday": "Saturday",
        "sun": "Sunday", "sunday": "Sunday",
    }
    return mapping.get(value[:3], day.title() if day else "Monday")


def day_prefix(day: str) -> str:
    return normalize_day(day)[:3].upper()


def period_text(period: str | int) -> str:
    raw = str(period).strip().upper()
    return raw[1:] if raw.startswith("P") else raw


def period_display(period: str | int) -> str:
    return f"P{period_text(period)}"


def period_key(day: str, period: str | int) -> str:
    """Legacy key kept only for backward-compatible reads."""
    return f"{normalize_day(day)}_{period_display(period)}"


def slot_id_for(day: str, period: str | int) -> str:
    return f"{day_prefix(day)}_P{period_text(period)}"


def relative_to_root(path: Path | None) -> str | None:
    if not path:
        return None
    try:
        return path.relative_to(ROOT_DIR).as_posix()
    except Exception:
        return str(path)


def prepared_slot_date_from_path(path: Path | None) -> str | None:
    if not path:
        return None
    parts = list(path.resolve().parts)
    lowered = [p.lower() for p in parts]
    if "prepared_slots" not in lowered:
        return None
    idx = lowered.index("prepared_slots")
    if idx + 1 >= len(parts):
        return None
    try:
        return normalize_session_date(parts[idx + 1])
    except ValueError:
        return None


def classify_input_source(video_dir: Path, slot_id: str) -> str:
    rel = relative_to_root(video_dir) or str(video_dir)
    normalized = rel.replace("\\", "/").lower()
    if "/prepared_slots/" in f"/{normalized}/":
        return "prepared_slot"
    if "/real_cctv/" in f"/{normalized}/":
        return "real_cctv"
    if normalized.rstrip("/").endswith(f"cctv_videos/{str(slot_id).lower()}"):
        return "current_slot_folder"
    return "explicit_video_dir"


def resolve_session_date(row: dict, payload: dict | None = None, video_dir: Path | None = None) -> str:
    payload = payload or {}
    slot_id = str(row.get("slot_id") or slot_id_for(row.get("day"), row.get("period"))).upper()
    explicit = payload.get("session_date") or payload.get("attendance_date") or payload.get("date")
    prepared_date = prepared_slot_date_from_path(video_dir)
    if explicit:
        resolved = normalize_session_date(explicit)
        if prepared_date and prepared_date != resolved:
            raise ValueError(f"Prepared-slot date mismatch: requested {resolved}, source has {prepared_date}.")
        return resolved
    if prepared_date:
        return prepared_date
    known = KNOWN_SLOT_SESSION_DATES.get(slot_id)
    if known:
        return known
    raise ValueError("Attendance date is required for this footage source.")


def build_session_context(row: dict, payload: dict | None, video_dir: Path | None) -> dict:
    payload = payload or {}
    clean_row = strip_runtime_fields(row)
    subject = str(payload.get("subject_track") or clean_row.get("subject_track") or clean_row.get("subject") or clean_row.get("course_abbr") or "").upper().strip()
    if not subject or "/" in subject:
        raise ValueError("Ambiguous CCM/CVO track. Select a specific subject/course track.")
    if subject not in row_subjects(clean_row):
        raise ValueError(f"Subject/course {subject} is not valid for this timetable slot.")
    session_date = resolve_session_date(clean_row, payload, video_dir)
    section = payload.get("section") or clean_row.get("section") or "B51"
    period = period_display(clean_row.get("period") or payload.get("period") or payload.get("period_number"))
    info = SUBJECT_INFO.get(subject, {})
    slot_id = clean_row.get("slot_id") or slot_id_for(clean_row.get("day"), period)
    session_id = build_attendance_session_id(session_date, section, period, subject)
    requested_session_id = str(payload.get("session_id") or "").strip()
    if requested_session_id and requested_session_id != session_id:
        parsed = parse_attendance_session_id(requested_session_id)
        if parsed["session_id"] != session_id:
            raise ValueError(f"Session ID mismatch: requested {requested_session_id}, resolved {session_id}.")
    input_path = relative_to_root(video_dir) if video_dir else None
    return {
        **clean_row,
        "key": session_id,
        "session_id": session_id,
        "session_date": session_date,
        "day": normalize_day(clean_row.get("day")),
        "section": str(section),
        "period": period,
        "period_number": period_text(period),
        "subject": subject,
        "subject_abbr": subject,
        "subject_track": subject,
        "course_abbr": subject,
        "course_code": info.get("course_code", clean_row.get("course_code", "-")),
        "course_name": info.get("course_name", clean_row.get("course_name", subject)),
        "subject_name": info.get("course_name", clean_row.get("course_name", subject)),
        "faculty_id": info.get("faculty_id", clean_row.get("faculty_id", "")),
        "faculty_name": info.get("faculty_name", clean_row.get("teacher") or clean_row.get("instructor") or ""),
        "teacher": info.get("faculty_name", clean_row.get("teacher", "-")),
        "instructor": info.get("faculty_name", clean_row.get("instructor", "-")),
        "room": clean_row.get("room", "-"),
        "start_time": clean_row.get("start_time", ""),
        "end_time": clean_row.get("end_time", ""),
        "input_slot": slot_id,
        "input_source_type": classify_input_source(video_dir, slot_id) if video_dir else "unresolved",
        "input_source_path": input_path,
    }


def session_preview_for_row(row: dict) -> dict:
    clean_row = strip_runtime_fields(row)
    slot_id = clean_row.get("slot_id") or slot_id_for(clean_row.get("day"), clean_row.get("period"))
    session_date = KNOWN_SLOT_SESSION_DATES.get(str(slot_id).upper())
    if not session_date:
        return {"session_date_required": True, "input_slot": slot_id}
    subject = clean_row.get("subject_track") or clean_row.get("subject") or clean_row.get("course_abbr")
    try:
        session_id = build_attendance_session_id(session_date, clean_row.get("section") or "B51", clean_row.get("period"), subject)
    except ValueError:
        return {"session_date": session_date, "session_date_required": True, "input_slot": slot_id}
    return {
        "session_id": session_id,
        "session_date": session_date,
        "session_date_required": False,
        "input_slot": slot_id,
        "subject_abbr": str(subject).upper(),
    }


def legacy_status_for_row(statuses: dict, day: str, row: dict) -> dict | None:
    legacy_key = period_key(day, row.get("period"))
    entry = statuses.get(legacy_key)
    if not isinstance(entry, dict) or entry.get("session_id"):
        return None
    row_subject_list = row_subjects(row)
    entry_subjects = row_subjects(entry)
    row_subject = str(row.get("subject_track") or row.get("subject") or "").upper()
    if len(row_subject_list) == 1 and (not entry_subjects or entry_subjects == [row_subject]):
        safe_entry = dict(entry)
        safe_entry["legacy_record"] = True
        safe_entry["legacy_key"] = legacy_key
        safe_entry.setdefault("status", entry.get("status") or "Legacy record")
        return safe_entry
    return {
        "status": "Legacy record",
        "legacy_record": True,
        "legacy_key": legacy_key,
        "message": "Ambiguous legacy session needs migration before it can be assigned to a specific faculty/course.",
    }


def status_for_timetable_row(statuses: dict, day: str, row: dict) -> dict | None:
    session_id = row.get("session_id")
    if session_id and isinstance(statuses.get(session_id), dict):
        return statuses[session_id]
    return legacy_status_for_row(statuses, day, row)


def user_can_access_session(user: dict, entry: dict) -> bool:
    if is_admin_user(user):
        return True
    subject = entry.get("subject_abbr") or entry.get("subject_track") or entry.get("subject") or entry.get("course_abbr")
    return subject_allowed_for_user(user, str(subject))


def resolve_processing_context(day: str, period: str | int, payload: dict | None, user: dict | None = None) -> tuple[dict, Path, dict]:
    payload = payload or {}
    subject_track = str(payload.get("subject_track") or "").upper().strip()
    if not subject_track:
        wanted = period_display(period)
        matches = [
            row
            for row in timetable_for_day(day, user or load_role_users()[0])
            if row.get("period", "").upper() == wanted.upper()
        ]
        subjects = {str(row.get("subject_track") or row.get("subject") or "").upper() for row in matches if row.get("subject_track") or row.get("subject")}
        if len(subjects) > 1:
            raise ValueError("Ambiguous CCM/CVO track. Select a specific subject/course track.")
    row = get_period_row(day, period, user, subject_track)
    if not row:
        raise ValueError(f"Period {period_display(period)} not found for {normalize_day(day)} and subject {subject_track or 'selected user'}.")
    slot_id = row.get("slot_id") or slot_id_for(day, period)
    video_dir = resolve_video_dir(slot_id, payload)
    if not video_dir.exists():
        raise ValueError(f"Missing footage folder: {relative_to_root(video_dir) or video_dir}")
    if not any(p.is_file() and p.suffix.lower() in ALLOWED_VIDEO_EXTENSIONS for p in video_dir.rglob("*")):
        raise ValueError(f"Empty footage folder: {relative_to_root(video_dir) or video_dir}")
    session = build_session_context(row, payload, video_dir)
    return row, video_dir, session


def read_json(path: Path, default):
    if not path.exists():
        write_json(path, default)
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return default


def write_json(path: Path, data) -> None:
    atomic_write_json(path, data)


def ensure_role_files() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    if not USERS_PATH.exists():
        write_json(USERS_PATH, DEFAULT_ROLE_USERS)
    if not STUDENT_MAP_PATH.exists():
        write_json(STUDENT_MAP_PATH, {"subjects": SUBJECT_INFO, "subject_students": {}, "all_students": []})


def load_role_users() -> list[dict]:
    ensure_role_files()
    return read_json(USERS_PATH, DEFAULT_ROLE_USERS)


def public_user(user: dict | None) -> dict | None:
    if not user:
        return None
    clean = dict(user)
    clean.pop("password", None)
    clean.pop("password_hash", None)
    if is_admin_user(clean):
        clean["name"] = "HOD"
        clean["roleLabel"] = "Head of Department"
        clean["facultyName"] = "HOD"
    return clean


def _bearer_token() -> str:
    authorization = str(request.headers.get("Authorization") or "").strip()
    if not authorization.lower().startswith("bearer "):
        return ""
    return authorization[7:].strip()


def _prune_auth_sessions(now_epoch: float | None = None) -> None:
    now_epoch = float(now_epoch if now_epoch is not None else time.time())
    expired = [
        token for token, session in AUTH_SESSIONS.items()
        if float(session.get("expires_at_epoch") or 0) <= now_epoch
    ]
    for token in expired:
        AUTH_SESSIONS.pop(token, None)


def issue_auth_session(user: dict) -> tuple[str, str]:
    token = secrets.token_urlsafe(32)
    now_epoch = time.time()
    expires_at_epoch = now_epoch + AUTH_SESSION_TTL_SECONDS
    expires_at = datetime.fromtimestamp(expires_at_epoch, timezone.utc).isoformat()
    with AUTH_LOCK:
        _prune_auth_sessions(now_epoch)
        AUTH_SESSIONS[token] = {
            "user_id": str(user.get("id") or "").strip().lower(),
            "created_at_epoch": now_epoch,
            "expires_at_epoch": expires_at_epoch,
        }
    return token, expires_at


def _user_from_auth_session(token: str, users: list[dict] | None = None) -> dict | None:
    token = str(token or "").strip()
    if not token:
        return None
    with AUTH_LOCK:
        _prune_auth_sessions()
        session = AUTH_SESSIONS.get(token)
        if not session:
            return None
        user_id = str(session.get("user_id") or "").strip().lower()
    users = users or load_role_users()
    for user in users:
        if str(user.get("id") or "").strip().lower() == user_id:
            return user
    with AUTH_LOCK:
        AUTH_SESSIONS.pop(token, None)
    return None


def revoke_auth_session(token: str) -> None:
    with AUTH_LOCK:
        AUTH_SESSIONS.pop(str(token or "").strip(), None)


def get_current_user() -> dict | None:
    users = load_role_users()
    token = _bearer_token()
    if token:
        # A supplied but invalid/expired token must never fall back to a spoofable user id.
        return _user_from_auth_session(token, users)
    requested = request.headers.get("X-User-Id") or request.args.get("user_id")
    if requested:
        user_id = str(requested).strip().lower()
        for user in users:
            if str(user.get("id", "")).lower() == user_id or str(user.get("username", "")).lower() == user_id:
                return user
        # An explicit but stale/unknown identity must never fall back to HOD.
        return None
    return users[0] if users else None


def is_admin_user(user: dict | None) -> bool:
    return bool(user and (user.get("role") == "admin" or user.get("canSeeAll")))


def user_subjects(user: dict | None) -> set[str]:
    if is_admin_user(user):
        return set(SUBJECT_INFO.keys())
    return {str(s).upper() for s in (user or {}).get("subjects", [])}


def split_subjects(value: str) -> list[str]:
    cleaned = str(value or "").upper().replace("-", "/")
    subjects: list[str] = []
    for part in cleaned.split("/"):
        subject = part.replace("LAB", "").replace("THEORY", "").strip()
        if subject:
            subjects.append(subject)
    return subjects


def row_subjects(row: dict) -> list[str]:
    return split_subjects(row.get("subject") or row.get("course_abbr") or row.get("Course_Abbr") or row.get("Subject"))


def subject_allowed_for_user(user: dict, subject: str) -> bool:
    return is_admin_user(user) or str(subject).upper() in user_subjects(user)


def row_allowed_for_user(row: dict, user: dict) -> bool:
    return any(subject_allowed_for_user(user, subject) for subject in row_subjects(row))


def project_row_for_subject(row: dict, subject: str) -> dict:
    subject = str(subject).upper()
    info = SUBJECT_INFO.get(subject, {})
    item = dict(row)
    item["subject"] = subject
    item["course_abbr"] = subject
    item["course_code"] = info.get("course_code", item.get("course_code", "-"))
    item["course_name"] = info.get("course_name", item.get("course_name", subject))
    item["teacher"] = info.get("faculty_name", item.get("teacher", "-"))
    item["instructor"] = info.get("faculty_name", item.get("instructor", "-"))
    item["subject_track"] = subject
    item["is_simultaneous_track"] = "/" in str(row.get("subject") or row.get("course_abbr") or "")
    return item


def expand_rows_for_user(rows: list[dict], user: dict) -> list[dict]:
    expanded: list[dict] = []
    for row in rows:
        subjects = row_subjects(row) or [row.get("subject", "-")]
        for subject in subjects:
            if subject_allowed_for_user(user, subject):
                # Admin also sees split CCM and CVO rows so simultaneous classes are clear.
                expanded.append(project_row_for_subject(row, subject))
    return expanded


def subject_allowed_for_period(day: str, period: str | int, subject: str, user: dict) -> bool:
    wanted = period_display(period)
    for row in load_timetable_rows():
        if row.get("day") == normalize_day(day) and row.get("period", "").upper() == wanted.upper():
            return subject_allowed_for_user(user, subject) and (str(subject).upper() in row_subjects(row))
    return False


def load_student_map() -> dict:
    ensure_role_files()
    return read_json(STUDENT_MAP_PATH, {"subjects": SUBJECT_INFO, "subject_students": {}, "all_students": []})


def allowed_rolls_for_user(user: dict) -> set[str]:
    mapping = load_student_map()
    if is_admin_user(user):
        return {canonical_student_roll(s.get("roll")) for s in mapping.get("all_students", []) if canonical_student_roll(s.get("roll"))}
    allowed: set[str] = set()
    subject_students = mapping.get("subject_students", {})
    for subject in user_subjects(user):
        for student in subject_students.get(subject, []):
            roll = canonical_student_roll(student.get("roll"))
            if roll:
                allowed.add(roll)
    return allowed


def _int_or_zero(value) -> int:
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError):
        return 0


def load_embedding_coverage(path: Path | None = None) -> tuple[dict[str, dict], bool]:
    summary_path = Path(path or EMBEDDING_SUMMARY_PATH)
    if not summary_path.is_file():
        return {}, False
    records: dict[str, dict] = {}
    try:
        with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                roll = canonical_student_roll(row.get("Roll_Number") or row.get("roll"))
                if not roll:
                    continue
                faces_used = _int_or_zero(row.get("Faces_Used"))
                records[roll] = {
                    "roll": roll,
                    "embedding_available": faces_used > 0,
                    "faces_used": faces_used,
                    "images_read": _int_or_zero(row.get("Images_Read")),
                    "no_valid_face": _int_or_zero(row.get("No_Valid_Face")),
                    "multiple_faces": _int_or_zero(row.get("Multiple_Faces")),
                    "rejected_crops": _int_or_zero(row.get("Rejected_Crops")),
                    "read_fail": _int_or_zero(row.get("Read_Fail")),
                }
    except (OSError, csv.Error):
        return {}, False
    return records, True


def load_dataset_coverage(root: Path | None = None) -> tuple[set[str], bool]:
    dataset_root = Path(root or DATASET_ROOT)
    if not dataset_root.is_dir():
        return set(), False
    rolls = {
        canonical_student_roll(path.name)
        for path in dataset_root.iterdir()
        if path.is_dir() and canonical_student_roll(path.name)
    }
    return rolls, True


def load_production_version(path: Path | None = None) -> dict:
    pointer_path = Path(path or CURRENT_EMBEDDING_VERSION_PATH)
    if not pointer_path.is_file():
        return {"status": "legacy_or_unversioned", "family_id": None, "promotion_id": None, "promoted_at": None}
    payload = read_json(pointer_path, {})
    if not isinstance(payload, dict):
        return {"status": "invalid_pointer", "family_id": None, "promotion_id": None, "promoted_at": None}
    return {
        "status": payload.get("status") or "unknown",
        "family_id": payload.get("family_id"),
        "variant_id": payload.get("variant_id"),
        "promotion_id": payload.get("promotion_id"),
        "promoted_at": payload.get("promoted_at"),
    }


def student_coverage_rows(mapping: dict | None = None) -> tuple[list[dict], dict]:
    mapping = mapping or load_student_map()
    embedding_records, embedding_source_available = load_embedding_coverage()
    dataset_rolls, dataset_source_available = load_dataset_coverage()
    rows: list[dict] = []
    for raw in mapping.get("all_students", []):
        roll = canonical_student_roll(raw.get("roll"))
        if not roll:
            continue
        embedding = embedding_records.get(roll, {})
        embedding_available = bool(embedding.get("embedding_available")) if embedding_source_available else None
        dataset_available = roll in dataset_rolls if dataset_source_available else None
        if embedding_available is None or dataset_available is None:
            coverage_status = "source_unavailable"
        elif embedding_available and dataset_available:
            coverage_status = "ready"
        elif dataset_available and not embedding_available:
            coverage_status = "dataset_only"
        elif embedding_available and not dataset_available:
            coverage_status = "embedding_only"
        else:
            coverage_status = "missing_both"
        rows.append({
            "roll": roll,
            "name": str(raw.get("name") or "").strip(),
            "subjects": sorted({str(value).upper() for value in raw.get("subjects", []) if str(value).strip()}),
            "dataset_available": dataset_available,
            "embedding_available": embedding_available,
            "coverage_status": coverage_status,
            "faces_used": embedding.get("faces_used") if embedding_source_available else None,
            "images_read": embedding.get("images_read") if embedding_source_available else None,
        })
    summary = {
        "total_students": len(rows),
        "dataset_source_available": dataset_source_available,
        "embedding_source_available": embedding_source_available,
        "dataset_available": sum(row["dataset_available"] is True for row in rows),
        "embedding_available": sum(row["embedding_available"] is True for row in rows),
        "ready": sum(row["coverage_status"] == "ready" for row in rows),
        "issues": sum(row["coverage_status"] not in {"ready", "source_unavailable"} for row in rows),
    }
    return rows, summary


def subject_catalog(mapping: dict | None = None) -> dict[str, dict]:
    mapping = mapping or load_student_map()
    raw_subjects = mapping.get("subjects") if isinstance(mapping.get("subjects"), dict) else {}
    catalog: dict[str, dict] = {}
    for subject in sorted(set(SUBJECT_INFO) | {str(key).upper() for key in raw_subjects}):
        raw = raw_subjects.get(subject, {}) if isinstance(raw_subjects.get(subject, {}), dict) else {}
        fallback = SUBJECT_INFO.get(subject, {})
        catalog[subject] = {
            "abbr": subject,
            "course_code": raw.get("course_code") or raw.get("courseCode") or fallback.get("course_code"),
            "course_name": raw.get("course_name") or raw.get("courseName") or fallback.get("course_name") or subject,
            "faculty_id": raw.get("faculty_id") or raw.get("facultyId") or fallback.get("faculty_id"),
            "faculty_name": raw.get("faculty_name") or raw.get("facultyName") or fallback.get("faculty_name"),
        }
    return catalog


def hod_overview_payload() -> dict:
    mapping = load_student_map()
    coverage_rows, coverage_summary = student_coverage_rows(mapping)
    coverage_by_roll = {row["roll"]: row for row in coverage_rows}
    users = [public_user(user) for user in load_role_users()]
    timetable_rows = load_timetable_rows()
    subjects = []
    for abbr, info in subject_catalog(mapping).items():
        roster = mapping.get("subject_students", {}).get(abbr, [])
        roster_rolls = [canonical_student_roll(student.get("roll")) for student in roster]
        roster_rolls = [roll for roll in roster_rolls if roll]
        subjects.append({
            **info,
            "roster_count": len(roster_rolls),
            "embedding_available": sum(coverage_by_roll.get(roll, {}).get("embedding_available") is True for roll in roster_rolls),
            "dataset_available": sum(coverage_by_roll.get(roll, {}).get("dataset_available") is True for roll in roster_rolls),
            "missing_embeddings": [roll for roll in roster_rolls if coverage_by_roll.get(roll, {}).get("embedding_available") is False],
            "timetable_slots": sum(abbr in row_subjects(row) and not re.search(r"lunch", str(row.get("subject") or ""), re.I) for row in timetable_rows),
        })
    return {
        "success": True,
        "users": users,
        "faculty": [user for user in users if user and user.get("role") == "faculty"],
        "subjects": subjects,
        "students": coverage_rows,
        "coverage": coverage_summary,
        "production": load_production_version(),
        "policy": {
            "recognition_stack": "YuNet + SFace",
            "match_threshold": DEFAULT_MATCH_THRESHOLD,
            "margin_threshold": DEFAULT_MARGIN_THRESHOLD,
            "checkpoint_count": 5,
            "checkpoint_min_detections": DEFAULT_CHECKPOINT_MIN_DETECTIONS,
            "present_checkpoints": DEFAULT_PRESENT_CHECKPOINTS,
            "strong_checkpoints": DEFAULT_STRONG_CHECKPOINTS,
            "review_checkpoints": DEFAULT_REVIEW_CHECKPOINTS,
            "sample_fps": DEFAULT_SAMPLE_FPS,
            "live_demo_priority": "last",
        },
    }



def _require_explicit_hod() -> tuple[dict | None, tuple | None]:
    token = _bearer_token()
    if not token:
        return None, (jsonify({"success": False, "error": "A verified HOD login session is required for configuration writes."}), 401)
    user = _user_from_auth_session(token)
    if not user:
        return None, (jsonify({"success": False, "error": "The HOD login session is invalid or expired. Sign in again."}), 401)
    if not is_admin_user(user):
        return None, (jsonify({"success": False, "error": "HOD access is required."}), 403)
    return user, None


def _configuration_write_blocker() -> str | None:
    ensure_jobs_hydrated()
    active_jobs = [
        job for job in JOBS.values()
        if str(job.get("status") or "") in {"Pending", "Processing"}
    ]
    if active_jobs:
        return f"Configuration writes are blocked while {len(active_jobs)} attendance job(s) are active."
    statuses = load_statuses()
    processing = [
        session_id for session_id, entry in statuses.items()
        if isinstance(entry, dict) and str(entry.get("status") or "") == "Processing"
    ]
    if processing:
        return "Configuration writes are blocked while attendance status contains an active Processing session."
    return None


def _configuration_subject_catalog(mapping: dict | None = None) -> dict:
    mapping = mapping or load_student_map()
    catalog = mapping.get("subjects") if isinstance(mapping, dict) else None
    if not isinstance(catalog, dict) or not catalog:
        raise ConfigGovernanceError("The authoritative subject catalog is unavailable.")
    return catalog


def _configuration_timetable_validation(
    *,
    mapping: dict | None = None,
    users: list[dict] | None = None,
    timetable_path: Path | None = None,
):
    mapping = mapping or load_student_map()
    users = users or load_role_users()
    timetable_path = timetable_path or find_timetable_path()
    return validate_timetable_bytes(
        Path(timetable_path).read_bytes(),
        _configuration_subject_catalog(mapping),
        users,
    )


def _public_config_pointer() -> dict | None:
    if not CONFIG_POINTER_PATH.is_file():
        return None
    payload = read_json(CONFIG_POINTER_PATH, None)
    if not isinstance(payload, dict):
        raise ConfigGovernanceError("Configuration pointer is not a JSON object.")
    allowed = {
        "schema_version",
        "status",
        "change_id",
        "change_type",
        "created_at",
        "actor",
        "reason",
        "target_files",
        "rolled_back_at",
        "rolled_back_by",
        "rollback_reason",
    }
    return {key: payload.get(key) for key in allowed if key in payload}


def configuration_governance_payload() -> dict:
    mapping = load_student_map()
    catalog = _configuration_subject_catalog(mapping)
    users = validate_role_users(load_role_users(), catalog)
    validation = _configuration_timetable_validation(mapping=mapping, users=users)
    faculty = []
    for user in users:
        if user.get("role") != "faculty":
            continue
        public = public_user(user)
        public["credential_format"] = "hashed" if user.get("password_hash") else "legacy"
        faculty.append(public)
    history = load_config_history(CONFIG_AUDIT_ROOT, limit=20)
    return {
        "success": True,
        "writes_enabled": True,
        "faculty": faculty,
        "subjects": list(subject_catalog(mapping).values()),
        "timetable": validation.public_dict(include_rows=False),
        "current_change": _public_config_pointer(),
        "history": history,
        "safety": {
            "active_jobs_block_writes": True,
            "roster_writes_enabled": False,
            "embedding_writes_enabled": False,
            "model_writes_enabled": False,
            "historical_attendance_changed": False,
            "latest_change_rollback_available": bool(
                CONFIG_POINTER_PATH.is_file()
                and (_public_config_pointer() or {}).get("status") == "applied"
            ),
        },
    }


def _configuration_transaction(
    *,
    user: dict,
    targets: dict[Path, bytes],
    change_type: str,
    reason: str,
    metadata: dict,
) -> dict:
    blocker = _configuration_write_blocker()
    if blocker:
        raise ConfigGovernanceError(blocker)
    return apply_config_transaction(
        repo_root=ROOT_DIR,
        targets=targets,
        backup_root=CONFIG_BACKUP_ROOT,
        audit_root=CONFIG_AUDIT_ROOT,
        pointer_path=CONFIG_POINTER_PATH,
        actor=public_user(user),
        change_type=change_type,
        reason=reason,
        metadata=metadata,
    )


def filter_attendance_rows_for_user(rows: list[dict], user: dict) -> list[dict]:
    if is_admin_user(user):
        return rows
    allowed = allowed_rolls_for_user(user)
    return [row for row in rows if canonical_student_roll(row.get("Roll_Number") or row.get("roll")) in allowed]


def load_statuses() -> dict:
    ensure_workflow_state_initialized()
    payload = read_json_object(STATUS_PATH, missing_default={})
    clean, _ = sanitize_status_store(payload)
    return clean


def save_statuses(data: dict) -> None:
    clean, _ = sanitize_status_store(data)
    atomic_write_json(STATUS_PATH, clean)


def repair_stale_processing_statuses(statuses: dict) -> dict:
    """If Flask restarted or a script crashed, old Processing states can remain in JSON.
    Mark them as Failed so the frontend does not show infinite Processing.
    """
    ensure_jobs_hydrated()
    changed = False
    active_ids = {jid for jid, job in JOBS.items() if job.get("status") in ("Pending", "Processing")}
    for key, entry in list(statuses.items()):
        if isinstance(entry, dict) and entry.get("status") == "Processing":
            job_id = entry.get("job_id")
            if not job_id or job_id not in active_ids:
                entry["status"] = "Failed"
                entry["error"] = "Previous processing did not finish or backend restarted. You can safely Reprocess this slot."
                entry["completed_at"] = now_text()
                statuses[key] = entry
                changed = True
    if changed:
        save_statuses(statuses)
    return statuses


def load_overrides() -> dict:
    return read_json(OVERRIDES_PATH, {})


def save_overrides(data: dict) -> None:
    write_json(OVERRIDES_PATH, data)


def find_timetable_path() -> Path:
    for path in TIMETABLE_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("Timetable CSV not found. Place timetable_b51_2026_2027.csv in the project root.")


def load_timetable_rows() -> list[dict]:
    path = find_timetable_path()
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            day = normalize_day(row.get("day") or row.get("Day") or "")
            period = period_text(row.get("period") or row.get("Period") or "")
            if not day or not period:
                continue
            course_abbr = (row.get("course_abbr") or row.get("Course_Abbr") or row.get("subject") or row.get("Subject") or "-").strip()
            course_name = (row.get("course_name") or row.get("Course_Name") or row.get("subject_name") or row.get("Subject_Name") or course_abbr).strip()
            rows.append({
                "slot_id": (row.get("slot_id") or row.get("Slot_ID") or slot_id_for(day, period)).strip(),
                "day": day,
                "period": period_display(period),
                "period_number": period,
                "subject": course_abbr,
                "course_abbr": course_abbr,
                "course_code": (row.get("course_code") or row.get("Course_Code") or "-").strip(),
                "course_name": course_name,
                "teacher": (row.get("teacher") or row.get("instructor") or row.get("Instructor") or row.get("Course Instructor") or "-").strip(),
                "instructor": (row.get("teacher") or row.get("instructor") or row.get("Instructor") or row.get("Course Instructor") or "-").strip(),
                "start_time": (row.get("start_time") or row.get("Start_Time") or "").strip(),
                "end_time": (row.get("end_time") or row.get("End_Time") or "").strip(),
                "room": (row.get("room") or row.get("Room") or row.get("Room No.") or "-").strip(),
                "section": (row.get("section") or row.get("Section") or "B51").strip(),
                "camera_ids": (row.get("camera_ids") or row.get("Camera_IDs") or "cam1,cam2,cam3").strip(),
                "session_type": (row.get("session_type") or row.get("Session_Type") or "Theory").strip(),
            })
    return rows


def timetable_for_day(day: str, user: dict | None = None) -> list[dict]:
    day = normalize_day(day)
    user = user or load_role_users()[0]
    statuses = repair_stale_processing_statuses(load_statuses())
    base_rows = []
    for row in load_timetable_rows():
        if row["day"] != day:
            continue
        base_rows.append(row)
    rows = []
    for row in expand_rows_for_user(base_rows, user):
        item = dict(row)
        item.update(session_preview_for_row(item))
        item["att_status"] = status_for_timetable_row(statuses, day, item)
        item["rt_status"] = "pending"
        item["controller"] = public_user(user)["name"]
        rows.append(item)
    return rows


def get_period_row(day: str, period: str | int, user: dict | None = None, subject_track: str | None = None) -> dict | None:
    wanted = period_display(period)
    rows = timetable_for_day(day, user or load_role_users()[0])
    if subject_track:
        for row in rows:
            if row["period"].upper() == wanted.upper() and str(row.get("subject_track") or row.get("subject")).upper() == str(subject_track).upper():
                return row
    for row in rows:
        if row["period"].upper() == wanted.upper():
            return row
    return None




def inspect_video_layout(video_dir: Path) -> dict:
    """Detect whether the slot folder contains full checkpoint folders or only demo clips.

    Supported layouts:
        Full MVP3:
            MON_P1/CP1_0910/cam1.mp4 ... CP5_0950/cam3.mp4
        Demo testing:
            MON_P1/CP1_0910/video1.mp4, video2.mp4, video3.mp4
        Simple demo:
            MON_P1/video1.mp4, video2.mp4, video3.mp4
    """
    cp_folders = []
    root_videos = []
    recursive_videos = []
    if video_dir.exists():
        for child in video_dir.iterdir():
            if child.is_file() and child.suffix.lower() in ALLOWED_VIDEO_EXTENSIONS:
                root_videos.append(child)
            elif child.is_dir() and child.name.lower().startswith("cp"):
                vids = [p for p in child.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_VIDEO_EXTENSIONS]
                if vids:
                    cp_folders.append(child)
                    recursive_videos.extend(vids)
        if not recursive_videos:
            recursive_videos = [p for p in video_dir.rglob("*") if p.is_file() and p.suffix.lower() in ALLOWED_VIDEO_EXTENSIONS]
    cp_names = sorted(folder.name for folder in cp_folders)
    return {
        "video_dir_exists": video_dir.exists(),
        "root_video_count": len(root_videos),
        "checkpoint_folder_count": len(cp_folders),
        "recursive_video_count": len(recursive_videos),
        "checkpoint_folders": cp_names,
        "has_full_checkpoint_set": len(cp_folders) >= 5,
        "is_demo_clip_set": (0 < len(cp_folders) < 5) or (len(cp_folders) == 0 and len(root_videos) > 0),
    }


def build_effective_processing_options(video_dir: Path, payload: dict | None = None) -> dict:
    """Return the fixed quality-aware Product Phase 2I production policy.

    Browser attendance never silently downgrades to a one-checkpoint demo rule.
    It requires the exact five-checkpoint front/back layout and rejects any
    client attempt to change thresholds, checkpoint voting, or evidence modes.
    """
    try:
        policy = fixed_processing_policy(payload or {})
        layout_contract = inspect_exact_checkpoint_layout(video_dir)
    except ProcessingIntegrationError as exc:
        raise ValueError(str(exc)) from exc
    return {**policy, "processing_profile": QUALITY_AWARE_PROCESSING_MODE, "layout": layout_contract.to_public_dict()}

def resolve_video_dir(slot_id: str, payload: dict | None = None) -> Path:
    payload = dict(payload or {})
    explicit = payload.get("video_dir") or payload.get("videoDir")
    candidates: list[Path] = []
    if explicit:
        candidate = Path(explicit)
        candidates.append(candidate if candidate.is_absolute() else ROOT_DIR / candidate)
    else:
        raw_date = payload.get("session_date") or payload.get("attendance_date") or payload.get("date")
        if raw_date:
            session_date = normalize_session_date(raw_date)
            candidates.append(VIDEO_DIR / "prepared_slots" / session_date / slot_id)
        candidates.extend([VIDEO_DIR / slot_id, VIDEO_DIR / slot_id.lower(), VIDEO_DIR / slot_id.replace("_", "")])

    video_root = VIDEO_DIR.resolve()
    checked: list[str] = []
    for path in candidates:
        resolved = path.resolve()
        try:
            resolved.relative_to(video_root)
        except ValueError as exc:
            raise ValueError(f"Footage source must remain under {VIDEO_DIR}: {resolved}") from exc
        checked.append(relative_to_root(resolved) or str(resolved))
        if resolved.is_dir() and any(
            item.is_file() and item.suffix.lower() in ALLOWED_VIDEO_EXTENSIONS
            for item in resolved.rglob("*")
        ):
            return resolved
    raise ValueError(
        f"No exact footage folder was found for {slot_id}. Checked: {', '.join(checked) or 'none'}. "
        "The browser will not fall back to the shared cctv_videos root."
    )

def output_ref(path: Path | None) -> str | None:
    if not path:
        return None
    candidate = Path(path)
    try:
        # Normalize both sides before computing the public reference. On Windows,
        # safe_output_path() returns a resolved path while OUTPUT_DIR may retain
        # its original spelling; comparing the raw forms can incorrectly fall
        # back to a basename on an idempotent finalize request.
        return candidate.resolve().relative_to(OUTPUT_DIR.resolve()).as_posix()
    except Exception:
        return candidate.name


def newest(pattern: str, after: float = 0.0) -> Path | None:
    files = [p for p in OUTPUT_DIR.rglob(pattern) if p.is_file() and p.stat().st_mtime >= after]
    if not files:
        files = [p for p in OUTPUT_DIR.rglob(pattern) if p.is_file()]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def newest_created(pattern: str, after: float) -> Path | None:
    files = [p for p in OUTPUT_DIR.rglob(pattern) if p.is_file() and p.stat().st_mtime >= after]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def safe_output_path(filename: str) -> Path | None:
    # Supports both plain filenames and organized refs like 2026-06-23/MON_P1/file.csv.
    clean = str(filename or "").replace("\\", "/").strip().lstrip("/")
    if not clean or ".." in Path(clean).parts:
        return None
    candidate = (OUTPUT_DIR / clean).resolve()
    try:
        candidate.relative_to(OUTPUT_DIR.resolve())
    except Exception:
        return None
    if candidate.exists() and candidate.is_file():
        return candidate
    # Fallback: find by basename for old frontend/status records.
    base = Path(clean).name
    matches = [p for p in OUTPUT_DIR.rglob(base) if p.is_file()]
    return max(matches, key=lambda p: p.stat().st_mtime) if matches else None


def read_attendance_csv(path: Path | None) -> list[dict]:
    if not path or not path.exists():
        return []
    df = pd.read_csv(path)
    rows = []
    for record in df.fillna("").to_dict(orient="records"):
        status = record.get("Status") or record.get("Final_Status") or ("Present" if str(record.get("Present", "")).lower() == "yes" else "Absent")
        item = dict(record)
        item["Status"] = status
        item["Original_AI_Status"] = record.get("Original_AI_Status") or status
        rows.append(item)
    return rows


def read_slot_quality(path: Path | None) -> dict:
    if not path or not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if df.empty:
        return {}
    quality = quality_from_slot_summary(df.fillna("").to_dict(orient="records")[0])
    return quality_to_status_fields(quality)


def summarize_attendance(rows: list[dict]) -> dict:
    total = len(rows)
    present = 0
    review = 0
    unconfirmed = 0
    missing_enrollment = 0
    absent = 0
    for row in rows:
        status = str(row.get("Status") or row.get("Final_Status") or "").strip().lower()
        if status.startswith("present"):
            present += 1
        elif "missing enrollment" in status:
            missing_enrollment += 1
        elif "unconfirmed" in status:
            unconfirmed += 1
        elif "review" in status:
            review += 1
        elif status == "absent":
            absent += 1
    pct = round((present / total) * 100, 1) if total else 0
    return {
        "total_students": total,
        "present_count": present,
        "needs_review_count": review,
        "unconfirmed_count": unconfirmed,
        "missing_enrollment_count": missing_enrollment,
        "absent_count": absent,
        "attendance_percentage": pct,
    }


def _truthy(value) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def canonical_student_roll(value) -> str:
    raw = str(value or "").strip().upper()
    match = re.match(r"^([A-Z0-9]+)", raw)
    if not match:
        return raw
    token = match.group(1)
    if any(character.isalpha() for character in token) and any(character.isdigit() for character in token):
        return token
    return raw


def _row_detection_count(row: dict) -> int:
    for key in ("Total_Accepted_Detections", "Detection_Count", "Accepted_Detections", "Detections"):
        try:
            return max(0, int(float(row.get(key) or 0)))
        except (TypeError, ValueError):
            continue
    return 0


def review_row_requires_attention(row: dict) -> bool:
    status = str(row.get("Status") or row.get("Final_Status") or "Absent").strip()
    lowered = status.lower()
    saved_resolution = _truthy(row.get("Manual_Override")) and (
        lowered.startswith("present") or lowered == "absent"
    )
    if saved_resolution:
        return False
    flags = str(row.get("Flags") or row.get("Flag") or "").strip()
    flagged = bool(flags and flags != "—")
    unresolved_status = any(
        token in lowered
        for token in ("needs review", "unconfirmed", "missing enrollment")
    )
    return (
        unresolved_status
        or bool(re.search(r"review|weak|low confidence|late|early|insufficient|missing enrollment", f"{status} {flags}", re.IGNORECASE))
        or flagged
        or ("absent" in lowered and _row_detection_count(row) > 0)
    )


def roster_coverage_for_rows(entry: dict, rows: list[dict]) -> dict:
    subjects = row_subjects(entry)
    if len(subjects) != 1:
        return {
            "roster_known": False,
            "roster_complete": False,
            "subject": subjects[0] if len(subjects) == 1 else None,
            "expected_count": 0,
            "row_count": len(rows),
            "missing_rolls": [],
            "unexpected_rolls": [],
            "duplicate_rolls": [],
            "missing_count": 0,
            "unexpected_count": 0,
            "duplicate_count": 0,
            "reason": "A single authoritative subject roster could not be resolved for this session.",
        }

    subject = subjects[0]
    mapping = load_student_map()
    expected = {
        canonical_student_roll(student.get("roll"))
        for student in mapping.get("subject_students", {}).get(subject, [])
        if canonical_student_roll(student.get("roll"))
    }
    if not expected:
        return {
            "roster_known": False,
            "roster_complete": False,
            "subject": subject,
            "expected_count": 0,
            "row_count": len(rows),
            "missing_rolls": [],
            "unexpected_rolls": [],
            "duplicate_rolls": [],
            "missing_count": 0,
            "unexpected_count": 0,
            "duplicate_count": 0,
            "reason": f"No authoritative {subject} roster is configured.",
        }

    actual_rolls = [
        canonical_student_roll(row.get("Roll_Number") or row.get("roll"))
        for row in rows
        if canonical_student_roll(row.get("Roll_Number") or row.get("roll"))
    ]
    counts: dict[str, int] = {}
    for roll in actual_rolls:
        counts[roll] = counts.get(roll, 0) + 1
    actual = set(actual_rolls)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    duplicates = sorted(roll for roll, count in counts.items() if count > 1)
    complete = not missing and not unexpected and not duplicates and len(actual_rolls) == len(expected)
    reason = ""
    if not complete:
        parts = []
        if missing:
            parts.append(f"{len(missing)} roster student(s) are missing")
        if unexpected:
            parts.append(f"{len(unexpected)} out-of-roster row(s) are present")
        if duplicates:
            parts.append(f"{len(duplicates)} duplicate roll(s) are present")
        reason = "; ".join(parts) + "."
    return {
        "roster_known": True,
        "roster_complete": complete,
        "subject": subject,
        "expected_count": len(expected),
        "row_count": len(actual_rolls),
        "missing_rolls": missing,
        "unexpected_rolls": unexpected,
        "duplicate_rolls": duplicates,
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "duplicate_count": len(duplicates),
        "reason": reason,
    }


def review_state_for_rows(rows: list[dict], entry: dict | None = None) -> dict:
    unresolved = [
        canonical_student_roll(row.get("Roll_Number") or row.get("roll"))
        for row in rows
        if review_row_requires_attention(row)
    ]
    unresolved = [roll for roll in unresolved if roll]
    coverage = roster_coverage_for_rows(entry, rows) if entry is not None else {
        "roster_known": False,
        "roster_complete": True,
        "subject": None,
        "expected_count": len(rows),
        "row_count": len(rows),
        "missing_rolls": [],
        "unexpected_rolls": [],
        "duplicate_rolls": [],
        "missing_count": 0,
        "unexpected_count": 0,
        "duplicate_count": 0,
        "reason": "",
    }
    return {
        "unresolved_count": len(unresolved),
        "unresolved_rolls": unresolved,
        "ready_to_finalize": bool(rows) and not unresolved and coverage["roster_complete"],
        "total_students": len(rows),
        **coverage,
    }


def attendance_is_finalized(entry: dict) -> bool:
    if "attendance_finalized" in entry:
        return _truthy(entry.get("attendance_finalized"))
    return _truthy(entry.get("Attendance_Finalized"))


def review_state_name(entry: dict, review_state: dict) -> str:
    if attendance_is_finalized(entry):
        return "finalized"
    if not review_state.get("roster_complete", True):
        return "roster_mismatch"
    if review_state.get("unresolved_count"):
        return "needs_attention"
    return "ready_to_finalize"


def public_review_state(review_state: dict, user: dict) -> dict:
    public = dict(review_state)
    if not is_admin_user(user):
        public["unexpected_rolls"] = []
        public["duplicate_rolls"] = []
    return public


def public_attendance_entry(entry: dict) -> dict:
    public = dict(entry or {})
    public.pop("attendance_data", None)
    public.pop("att_status", None)
    return public


def compact_attendance_session(key: str, entry: dict, user: dict) -> dict | None:
    if not isinstance(entry, dict) or entry.get("status") not in ("Completed", "Needs Review") or not entry.get("attendance_data"):
        return None
    if not user_can_access_session(user, entry):
        return None
    all_rows = apply_overrides(key, entry.get("attendance_data", []))
    visible_rows = filter_attendance_rows_for_user(all_rows, user)
    summary = summarize_attendance(visible_rows)
    review_state = review_state_for_rows(visible_rows, entry)
    finalized = attendance_is_finalized(entry)
    final_file = entry.get("final_attendance_csv") or entry.get("admin_reviewed_csv")
    subject = str(entry.get("subject_abbr") or entry.get("subject_track") or entry.get("subject") or entry.get("course_abbr") or "").upper()
    session_id = str(entry.get("session_id") or key)
    state = review_state_name(entry, review_state)
    return {
        "session_id": session_id,
        "key": key,
        "session_date": entry.get("session_date") or entry.get("date"),
        "day": entry.get("day"),
        "period": entry.get("period") or entry.get("period_number"),
        "subject": subject,
        "course_code": entry.get("course_code"),
        "course_name": entry.get("course_name") or entry.get("subject_name"),
        "faculty_name": entry.get("faculty_name") or entry.get("instructor") or entry.get("teacher"),
        "room": entry.get("room"),
        "section": entry.get("section"),
        "start_time": entry.get("start_time"),
        "end_time": entry.get("end_time"),
        "status": entry.get("status"),
        "report_type": entry.get("report_type"),
        "review_state": state,
        "attendance_finalized": finalized,
        "requires_manual_review": bool(review_state["unresolved_count"] or not review_state["roster_complete"]),
        "unresolved_count": review_state["unresolved_count"],
        "roster_known": review_state["roster_known"],
        "roster_complete": review_state["roster_complete"],
        "roster_expected_count": review_state["expected_count"],
        "roster_row_count": review_state["row_count"],
        "missing_roster_rolls": review_state["missing_rolls"],
        "unexpected_roster_rolls": review_state["unexpected_rolls"],
        "duplicate_roster_rolls": review_state["duplicate_rolls"],
        "missing_roster_count": review_state["missing_count"],
        "unexpected_roster_count": review_state["unexpected_count"],
        "duplicate_roster_count": review_state["duplicate_count"],
        "roster_reason": review_state["reason"],
        "total_students": summary["total_students"],
        "present_count": summary["present_count"],
        "needs_review_count": summary["needs_review_count"],
        "unconfirmed_count": summary["unconfirmed_count"],
        "missing_enrollment_count": summary["missing_enrollment_count"],
        "absent_count": summary["absent_count"],
        "attendance_percentage": summary["attendance_percentage"],
        "official_recognition_authority": entry.get("official_recognition_authority"),
        "tracklets_used_for_official_attendance": entry.get("tracklets_used_for_official_attendance"),
        "source_report_superseded": bool(entry.get("source_report_superseded")),
        "authority_revision_id": entry.get("authority_revision_id"),
        "revision_authority": entry.get("revision_authority"),
        "official_report_preserved_after_reprocess": bool(entry.get("official_report_preserved_after_reprocess")),
        "pending_candidate_revision": entry.get("pending_candidate_revision"),
        "latest_automatic_candidate": entry.get("latest_automatic_candidate"),
        "latest_equivalent_candidate_revision": entry.get("latest_equivalent_candidate_revision"),
        "latest_candidate_matches_official": bool(entry.get("latest_candidate_matches_official")),
        "candidate_revision_count": int(entry.get("candidate_revision_count") or 0),
        "review_carry_forward_applied": bool(entry.get("review_carry_forward_applied")),
        "review_carry_forward_source": entry.get("review_carry_forward_source"),
        "completed_at": entry.get("completed_at"),
        "last_edited_at": entry.get("last_edited_at"),
        "finalized_at": entry.get("finalized_at"),
        "finalized_by": entry.get("finalized_by"),
        "download_url": f"/api/download/{final_file}" if final_file else None,
    }


def export_attendance_rows(entry: dict, rows: list[dict], prefix: str) -> Path:
    session_id = entry.get("session_id") or entry.get("key") or "legacy_session"
    safe_session = clean_session_filename(session_id)
    run_dir = OUTPUT_DIR / datetime.now().strftime("%Y-%m-%d") / safe_session
    run_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = run_dir / f"{prefix}_{safe_session}_{timestamp}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def export_reviewed_attendance(entry: dict, rows: list[dict]) -> Path:
    return export_attendance_rows(entry, rows, "admin_reviewed")


def export_finalized_attendance(entry: dict, rows: list[dict]) -> Path:
    return export_attendance_rows(entry, rows, "finalized_attendance")


def canonical_override_map(overrides: dict) -> dict:
    canonical: dict[str, dict] = {}
    for raw_roll, value in (overrides or {}).items():
        roll = canonical_student_roll(raw_roll)
        if not roll:
            continue
        if roll in canonical and canonical[roll] != value:
            raise WorkflowStateError(f"Conflicting manual override records canonicalize to {roll}.")
        canonical[roll] = value
    return canonical


def apply_override_map(rows: list[dict], overrides: dict) -> list[dict]:
    overrides = canonical_override_map(overrides)
    result = []
    for row in rows:
        item = dict(row)
        source_roll = str(item.get("Roll_Number") or item.get("roll") or "").strip().upper()
        roll = canonical_student_roll(source_roll)
        if roll:
            item["Roll_Number"] = roll
        if source_roll and source_roll != roll:
            item.setdefault("Source_Roll_Label", source_roll)
        if roll in overrides:
            ov = overrides[roll]
            item["Original_AI_Status"] = item.get("Original_AI_Status") or item.get("Status")
            item["Status"] = ov.get("manual_status", item.get("Status"))
            item["Manual_Override"] = True
            item["Override_Reason"] = ov.get("reason", "Admin correction")
            item["Edited_By"] = ov.get("edited_by", ADMIN_PROFILE["name"])
            item["Edited_At"] = ov.get("edited_at", "")
        result.append(item)
    return result


def apply_overrides(key: str, rows: list[dict]) -> list[dict]:
    return apply_override_map(rows, load_overrides().get(key, {}))


def public_job(job: dict) -> dict:
    """Return a compact JSON-safe job object without internal process details."""
    if not job:
        return {}
    return compact_job_for_storage(job)


def _persist_jobs_locked() -> None:
    atomic_write_json(JOB_RUNTIME_PATH, serialize_job_registry(JOBS))


def ensure_jobs_hydrated() -> None:
    global JOBS_HYDRATED
    with LOCK:
        if JOBS_HYDRATED:
            return
        loaded, changed = load_job_registry(
            JOB_RUNTIME_PATH,
            now_text=now_text(),
            interrupted_message="Previous processing was interrupted by a backend restart. Reprocess this session safely.",
        )
        JOBS.clear()
        JOBS.update(loaded)
        JOBS_HYDRATED = True
        if changed:
            _persist_jobs_locked()


def ensure_workflow_state_initialized() -> dict:
    global WORKFLOW_STATE_INITIALIZED, WORKFLOW_STATE_REPORT
    with LOCK:
        if WORKFLOW_STATE_INITIALIZED:
            return dict(WORKFLOW_STATE_REPORT)
        migration = migrate_status_file(STATUS_PATH, STATE_BACKUP_DIR)
        ensure_jobs_hydrated()
        WORKFLOW_STATE_REPORT = {
            "status": "ready",
            "status_migration": migration,
            "persisted_jobs": len(JOBS),
            "job_registry": str(JOB_RUNTIME_PATH),
        }
        WORKFLOW_STATE_INITIALIZED = True
        return dict(WORKFLOW_STATE_REPORT)


def set_job(job_id: str, **updates) -> None:
    ensure_jobs_hydrated()
    with LOCK:
        job = JOBS.get(job_id, {})
        job.update(updates)
        job["updated_at"] = now_text()
        JOBS[job_id] = job
        _persist_jobs_locked()


def append_job_log(job_id: str, line: str, keep: int = 40) -> None:
    line = (line or "").strip()
    if not line:
        return
    ensure_jobs_hydrated()

    def clamp_percent(value: int | float) -> int:
        try:
            return max(0, min(100, int(round(float(value)))))
        except Exception:
            return 0

    with LOCK:
        job = JOBS.get(job_id, {})
        logs = list(job.get("log_tail", []))
        logs.append(line)
        job["log_tail"] = logs[-keep:]

        lower = line.lower()
        layout = job.get("video_layout") or {}
        total_videos = int(job.get("_total_videos") or layout.get("recursive_video_count") or 0)
        total_checkpoints = int(job.get("_total_checkpoints") or 0)

        # Header / setup information from mark_attendance_checkpoints.py
        if line.startswith("Camera videos:"):
            try:
                total_videos = int(line.split(":", 1)[1].strip())
                job["_total_videos"] = total_videos
            except Exception:
                pass
            job["progress_stage"] = "checking_files"
            job["progress_step"] = 0
            job["progress_percent"] = max(int(job.get("progress_percent") or 0), 10)
            job["progress_text"] = f"Found {total_videos or layout.get('recursive_video_count', 0)} camera video(s). Preparing checkpoints..."

        elif line.startswith("Checkpoint mode:"):
            job["progress_stage"] = "checking_files"
            job["progress_step"] = 0
            job["progress_percent"] = max(int(job.get("progress_percent") or 0), 12)
            job["progress_text"] = line

        elif line.startswith("Checkpoints:"):
            job["_reading_checkpoint_list"] = True
            job["_total_checkpoints"] = 0
            job["progress_stage"] = "checking_files"
            job["progress_step"] = 0
            job["progress_percent"] = max(int(job.get("progress_percent") or 0), 14)
            job["progress_text"] = "Checkpoint windows loaded. Starting camera scan..."

        elif job.get("_reading_checkpoint_list") and line.startswith("CP") and ":" in line and "processed" not in lower and "skipped" not in lower:
            job["_total_checkpoints"] = int(job.get("_total_checkpoints") or 0) + 1
            job["progress_stage"] = "checking_files"
            job["progress_step"] = 0
            job["progress_percent"] = max(int(job.get("progress_percent") or 0), 16)
            job["progress_text"] = f"Loaded {job['_total_checkpoints']} checkpoint window(s)."

        elif line.startswith("Processing camera angle:"):
            job["_reading_checkpoint_list"] = False
            current_camera_index = int(job.get("_current_camera_index") or 0) + 1
            job["_current_camera_index"] = current_camera_index
            job["current_camera"] = line.split("Processing camera angle:", 1)[1].strip()
            total = max(1, int(job.get("_total_videos") or layout.get("recursive_video_count") or current_camera_index))
            # 20-50% while moving across camera/video clips.
            job["progress_stage"] = "processing_clips"
            job["progress_step"] = 1
            job["progress_percent"] = clamp_percent(20 + ((current_camera_index - 1) / total) * 30)
            job["progress_text"] = f"Camera {current_camera_index}/{total}: {job['current_camera']}"

        elif line.startswith("CP") and ("processed" in lower or "skipped" in lower):
            done = int(job.get("_checkpoint_events_done") or 0) + 1
            job["_checkpoint_events_done"] = done
            total = max(1, int(job.get("_total_videos") or layout.get("recursive_video_count") or 1) * max(1, int(job.get("_total_checkpoints") or 5)))
            # 50-84% while each checkpoint window is processed/recognized.
            job["progress_stage"] = "recognizing_students"
            job["progress_step"] = 2
            job["progress_percent"] = clamp_percent(50 + (min(done, total) / total) * 34)
            job["progress_text"] = line.strip()

        elif "checkpoint attendance completed" in lower:
            job["progress_stage"] = "generating_report"
            job["progress_step"] = 3
            job["progress_percent"] = max(int(job.get("progress_percent") or 0), 88)
            job["progress_text"] = "Recognition finished. Generating attendance report..."

        elif line.startswith("Saved ") or "saved outputs" in lower or "output" in lower:
            saved = int(job.get("_saved_output_count") or 0) + 1
            job["_saved_output_count"] = saved
            job["progress_stage"] = "generating_report"
            job["progress_step"] = 3
            job["progress_percent"] = clamp_percent(88 + min(saved, 6) * 2)
            job["progress_text"] = line

        elif "started" in lower or "slot:" in lower or "subject:" in lower:
            job["progress_stage"] = job.get("progress_stage") or "initializing_engine"
            job["progress_step"] = int(job.get("progress_step") or 0)
            job["progress_percent"] = max(int(job.get("progress_percent") or 0), 6)
            job["progress_text"] = line

        JOBS[job_id] = job
        _persist_jobs_locked()


def terminate_process_tree(proc: subprocess.Popen) -> None:
    if not proc or proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass


def run_slot_job(
    job_id: str,
    day: str,
    period: str | int,
    payload: dict | None = None,
    reprocess: bool = False,
    actor: dict | None = None,
) -> None:
    payload = payload or {}
    day = normalize_day(day)
    actor = public_user(actor) or public_user(load_role_users()[0])
    try:
        row, video_dir, session = resolve_processing_context(day, period, payload, actor)
        processing_options = build_effective_processing_options(video_dir, payload)
    except Exception as exc:
        set_job(job_id, status="Failed", error=str(exc), completed_at=now_text(), progress_stage="failed", progress_percent=0)
        return

    key = session["session_id"]
    slot_id = session.get("input_slot") or row.get("slot_id") or slot_id_for(day, period)
    layout = processing_options["layout"]
    diagnostic_run_id = clean_session_filename(
        f"product-2i-{session['session_id']}-{job_id.lower()}"
    )
    diagnostic_run_dir = OUTPUT_DIR / "diagnostics" / diagnostic_run_id
    start_mtime = time.time() - 2

    statuses = load_statuses()
    statuses[key] = {
        **session,
        "key": key,
        "status": "Processing",
        "job_id": job_id,
        "started_at": now_text(),
        "controlled_by": actor.get("name") or ADMIN_PROFILE["name"],
        "controller_role": actor.get("roleLabel") or actor.get("role") or ADMIN_PROFILE["role"],
        "started_by_user_id": actor.get("id"),
        "video_dir": relative_to_root(video_dir),
        "input_source_path": session.get("input_source_path"),
        "processing_profile": processing_options["processing_profile"],
        "video_layout": layout,
        "sample_fps": processing_options["sample_fps"],
        "log_mode": processing_options["log_mode"],
        "processing_contract_id": PROCESSING_CONTRACT_POLICY_VERSION,
        "official_recognition_authority": processing_options["official_recognition_authority"],
        "zone_mode": processing_options["zone_mode"],
        "tracklet_mode": processing_options["tracklet_mode"],
        "tracklets_used_for_official_attendance": processing_options["tracklets_used_for_official_attendance"],
        "diagnostic_run_id": diagnostic_run_id,
        "diagnostic_run_dir": output_ref(diagnostic_run_dir),
        "checkpoint_mode": processing_options["checkpoint_mode"],
        "present_checkpoints": processing_options["present_checkpoints"],
        "subject_track": session.get("subject_track"),
        "error": None,
    }
    save_statuses(statuses)
    layout_text = "Full 5-checkpoint folders" if layout.get("has_full_checkpoint_set") else "Demo clips mode" if layout.get("is_demo_clip_set") else "Auto video mode"
    set_job(
        job_id,
        status="Processing",
        started_at=now_text(),
        progress_stage="checking_files",
        progress_step=0,
        progress_percent=5,
        progress_text=f"Checking files: {layout_text}: {layout.get('recursive_video_count', 0)} video(s), {layout.get('checkpoint_folder_count', 0)} CP folder(s)",
        video_dir=relative_to_root(video_dir),
        session_id=session.get("session_id"),
        session_date=session.get("session_date"),
        subject_abbr=session.get("subject_abbr"),
        faculty_id=session.get("faculty_id"),
        faculty_name=session.get("faculty_name"),
        input_slot=session.get("input_slot"),
        input_source_type=session.get("input_source_type"),
        input_source_path=session.get("input_source_path"),
        video_layout=layout,
        processing_profile=processing_options["processing_profile"],
        processing_contract_id=PROCESSING_CONTRACT_POLICY_VERSION,
        official_recognition_authority=processing_options["official_recognition_authority"],
        zone_mode=processing_options["zone_mode"],
        tracklet_mode=processing_options["tracklet_mode"],
        tracklets_used_for_official_attendance=processing_options["tracklets_used_for_official_attendance"],
        diagnostic_run_id=diagnostic_run_id,
    )

    command = build_processing_command(
        python_executable=sys.executable,
        script_path=ROOT_DIR / "scripts" / "mark_attendance_checkpoints.py",
        timetable_path=find_timetable_path(),
        slot_id=slot_id,
        video_dir=video_dir,
        embeddings_path=ROOT_DIR / "models" / "student_embeddings.pkl",
        student_map_path=STUDENT_MAP_PATH,
        output_dir=OUTPUT_DIR,
        camera_zones_path=DATA_DIR / "camera_zones.json",
        session=session,
        policy=processing_options,
        diagnostic_run_id=diagnostic_run_id,
    )
    try:
        timeout_seconds = int(processing_options["timeout_seconds"])
        # The shared Product Phase 2I command builder already enables unbuffered output.
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        child_env = os.environ.copy()
        child_env["PYTHONIOENCODING"] = "utf-8"
        child_env["PYTHONUTF8"] = "1"
        proc = subprocess.Popen(
            command,
            cwd=str(ROOT_DIR),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            creationflags=creationflags,
            start_new_session=(os.name != "nt"),
            env=child_env,
        )
        with LOCK:
            RUNNING_PROCESSES[job_id] = proc
            JOBS[job_id]["pid"] = proc.pid
            JOBS[job_id]["command"] = " ".join(str(x) for x in command)
            JOBS[job_id]["timeout_seconds"] = timeout_seconds
            _persist_jobs_locked()

        stdout_lines = []
        output_queue: queue.Queue[str] = queue.Queue()

        def reader_thread() -> None:
            try:
                if not proc.stdout:
                    return
                for output_line in proc.stdout:
                    output_queue.put(output_line)
            finally:
                output_queue.put("__EOF__")

        threading.Thread(target=reader_thread, daemon=True).start()

        started = time.time()
        saw_eof = False
        while True:
            if job_id in CANCELLED_JOBS:
                terminate_process_tree(proc)
                raise RuntimeError("Cancelled by HOD")
            if timeout_seconds and (time.time() - started) > timeout_seconds:
                terminate_process_tree(proc)
                raise TimeoutError(f"Processing exceeded timeout of {timeout_seconds} seconds")
            try:
                line = output_queue.get(timeout=0.25)
            except queue.Empty:
                if proc.poll() is not None and saw_eof:
                    break
                if proc.poll() is not None and output_queue.empty():
                    break
                continue
            if line == "__EOF__":
                saw_eof = True
                if proc.poll() is not None:
                    break
                continue
            clean_line = str(line).rstrip()
            if clean_line:
                stdout_lines.append(clean_line)
                append_job_log(job_id, clean_line)

        return_code = proc.wait()
        with LOCK:
            RUNNING_PROCESSES.pop(job_id, None)
        if return_code != 0:
            tail = "\n".join(stdout_lines[-80:])
            raise RuntimeError(tail or "Attendance script failed")

        output_prefix = clean_session_filename(session["session_id"])
        attendance_csv = newest_created(f"attendance_{output_prefix}_*.csv", start_mtime)
        summary_csv = newest_created(f"slot_summary_{output_prefix}_*.csv", start_mtime)
        checkpoint_summary_csv = newest_created(f"checkpoint_summary_{output_prefix}_*.csv", start_mtime)
        student_checkpoint_csv = newest_created(f"student_checkpoint_evidence_{output_prefix}_*.csv", start_mtime)
        camera_summary_csv = newest_created(f"camera_summary_{output_prefix}_*.csv", start_mtime)
        needs_review_csv = newest_created(f"needs_review_{output_prefix}_*.csv", start_mtime)
        detection_csv = newest_created(f"detection_log_{output_prefix}_*.csv", start_mtime)
        excel_file = newest_created(f"attendance_{output_prefix}_*.xlsx", start_mtime)
        if not attendance_csv:
            raise RuntimeError(f"Attendance script completed but did not generate an attendance CSV for {session['session_id']}.")

        raw_attendance_csv = attendance_csv
        raw_attendance_data = read_attendance_csv(raw_attendance_csv)
        quality_status = read_slot_quality(summary_csv)

        missing_embedding_rolls: list[str] = []
        if raw_attendance_data:
            raw_missing = str(raw_attendance_data[0].get("Missing_Embedding_Rolls") or "")
            missing_embedding_rolls = [
                canonical_student_roll(value)
                for value in re.split(r"[;,]", raw_missing)
                if canonical_student_roll(value)
            ]

        guarded_candidates: dict[str, set[str]] = {}
        tracklet_frame: pd.DataFrame | None = None
        tracklet_diagnostics_path = diagnostic_run_dir / f"tracklet_diagnostics_{diagnostic_run_id}.csv"
        if tracklet_diagnostics_path.is_file():
            try:
                tracklet_frame = pd.read_csv(
                    tracklet_diagnostics_path,
                    dtype=str,
                    keep_default_na=False,
                    encoding="utf-8-sig",
                )
                guarded_candidates = select_guarded_review_candidates(tracklet_frame)
            except ProductPhase2IError as exc:
                raise RuntimeError(f"Phase 2I guarded-candidate safety check failed: {exc}") from exc

        attendance_data = apply_automatic_status_semantics(
            raw_attendance_data,
            quality_requires_review=bool(quality_status.get("requires_manual_review")),
            missing_embedding_rolls=missing_embedding_rolls,
            guarded_candidates=guarded_candidates,
        )

        review_carry_forward = {"applied": False, "reason": "no_exact_source_registry"}
        registry = load_review_carry_forward_registry(REVIEW_CARRY_FORWARD_REGISTRY_PATH)
        if registry and str(registry.get("session_id") or "") == str(session.get("session_id") or ""):
            if tracklet_frame is None:
                raise RuntimeError("Exact-source review registry exists but tracklet diagnostics were not generated")
            try:
                fingerprint = review_source_fingerprint(
                    video_dir=video_dir,
                    embeddings_path=ROOT_DIR / "models" / "student_embeddings.pkl",
                    session_id=session["session_id"],
                    input_source_path=str(session.get("input_source_path") or relative_to_root(video_dir)),
                )
                attendance_data, review_carry_forward = apply_exact_review_carry_forward(
                    attendance_data,
                    tracklet_frame=tracklet_frame,
                    registry=registry,
                    actual_source_fingerprint=fingerprint,
                )
            except ReviewCarryForwardError as exc:
                raise RuntimeError(f"Exact-source reviewed evidence could not be reused safely: {exc}") from exc

        roster_check = roster_coverage_for_rows(session, attendance_data)
        if not roster_check.get("roster_complete"):
            raise RuntimeError(
                "Product Phase 2I processor did not produce the authoritative roster: "
                + str(roster_check.get("reason") or "unknown roster mismatch")
            )

        authority_attendance_csv = export_attendance_rows(
            session,
            attendance_data,
            "authority_attendance",
        )
        attendance_csv = authority_attendance_csv
        summary = summarize_attendance(attendance_data)
        entry_status = "Needs Review" if (
            quality_status.get("requires_manual_review")
            or summary["needs_review_count"]
            or summary["unconfirmed_count"]
            or summary["missing_enrollment_count"]
        ) else "Completed"

        video_files = []
        if video_dir.exists():
            video_files = [p.relative_to(video_dir).as_posix() for p in sorted(video_dir.rglob("*")) if p.is_file() and p.suffix.lower() in ALLOWED_VIDEO_EXTENSIONS]

        entry = {
            **session,
            "key": key,
            "status": entry_status,
            "report_type": "Reprocessed" if reprocess else "AI",
            "job_id": job_id,
            "completed_at": now_text(),
            "controlled_by": actor.get("name") or ADMIN_PROFILE["name"],
            "controller_role": actor.get("roleLabel") or actor.get("role") or ADMIN_PROFILE["role"],
            "started_by_user_id": actor.get("id"),
            "video_dir": relative_to_root(video_dir),
            "input_source_path": session.get("input_source_path"),
            "video_files": video_files,
            "processing_profile": processing_options["processing_profile"],
            "sample_fps": processing_options["sample_fps"],
            "log_mode": processing_options["log_mode"],
            "output_layout": processing_options["output_layout"],
            "processing_contract_id": PROCESSING_CONTRACT_POLICY_VERSION,
            "official_recognition_authority": processing_options["official_recognition_authority"],
            "zone_mode": processing_options["zone_mode"],
            "tracklet_mode": processing_options["tracklet_mode"],
            "tracklets_used_for_official_attendance": processing_options["tracklets_used_for_official_attendance"],
            "diagnostic_run_id": diagnostic_run_id,
            "diagnostic_run_dir": output_ref(diagnostic_run_dir),
            "roster_contract": roster_check,
            "output_manifest": {
                "attendance_csv": output_ref(attendance_csv),
                "raw_processor_attendance_csv": output_ref(raw_attendance_csv),
                "slot_summary_csv": output_ref(summary_csv),
                "detection_log_csv": output_ref(detection_csv),
                "checkpoint_summary_csv": output_ref(checkpoint_summary_csv),
                "student_checkpoint_evidence_csv": output_ref(student_checkpoint_csv),
                "camera_summary_csv": output_ref(camera_summary_csv),
                "needs_review_csv": output_ref(needs_review_csv),
                "raw_processor_attendance_xlsx": output_ref(excel_file),
                "quality_diagnostic_run": output_ref(diagnostic_run_dir),
                "tracklet_diagnostics_csv": output_ref(tracklet_diagnostics_path) if tracklet_diagnostics_path.is_file() else None,
            },
            "attendance_csv": output_ref(attendance_csv),
            "slot_summary_csv": output_ref(summary_csv),
            "detection_log_csv": output_ref(detection_csv),
            "checkpoint_summary_csv": output_ref(checkpoint_summary_csv),
            "student_checkpoint_evidence_csv": output_ref(student_checkpoint_csv),
            "camera_summary_csv": output_ref(camera_summary_csv),
            "needs_review_csv": output_ref(needs_review_csv),
            "raw_processor_attendance_csv": output_ref(raw_attendance_csv),
            "class_report_file": output_ref(attendance_csv),
            "excel_file": None,
            "raw_processor_excel_file": output_ref(excel_file),
            "attendance_data": attendance_data,
            "guarded_recovery_automatic": False,
            "guarded_recovery_candidate_count": sum(len(value) for value in guarded_candidates.values()),
            "guarded_recovery_candidate_students": sorted(guarded_candidates),
            "automatic_status_semantics": "present_needs_review_unconfirmed_missing_enrollment_absent",
            "review_carry_forward_applied": bool(review_carry_forward.get("applied")),
            "review_carry_forward": review_carry_forward,
            "review_carry_forward_source": review_carry_forward.get("registry_id"),
            "revision_authority": "automatic_candidate",
            "error": None,
            **quality_status,
            **summary,
        }
        statuses = load_statuses()
        existing_entry = statuses.get(key) if isinstance(statuses.get(key), dict) else None
        try:
            revision_commit = preserve_official_or_commit_candidate(
                ROOT_DIR,
                key,
                existing_entry,
                entry,
            )
        except ReportRevisionError as exc:
            raise RuntimeError(f"Report revision could not be committed safely: {exc}") from exc
        statuses[key] = revision_commit.current_entry
        save_statuses(statuses)
        job_result = dict(entry)
        job_result.update({
            "revision_disposition": revision_commit.disposition,
            "official_report_preserved": revision_commit.official_preserved,
            "candidate_revision": revision_commit.candidate_revision,
            "official_session_summary": summarize_attendance(revision_commit.current_entry.get("attendance_data", [])),
        })
        if revision_commit.disposition == "equivalent_candidate_archived_official_preserved":
            completion_text = "Exact-source rerun matched the reviewed official report; review decisions were reused."
        elif revision_commit.official_preserved:
            completion_text = "Automatic candidate archived; reviewed official report preserved."
        else:
            completion_text = quality_status.get("run_quality_status") or "Completed"
        set_job(
            job_id,
            status=entry_status,
            completed_at=now_text(),
            result=job_result,
            progress_stage="completed",
            progress_step=4,
            progress_percent=100,
            progress_text=completion_text,
            stdout="\n".join(stdout_lines[-120:]),
        )
    except Exception as exc:
        with LOCK:
            proc = RUNNING_PROCESSES.pop(job_id, None)
        if proc and proc.poll() is None:
            terminate_process_tree(proc)
        final_status = "Cancelled" if job_id in CANCELLED_JOBS else "Failed"
        statuses = load_statuses()
        failed = statuses.get(key, {**session, "key": key})
        failed.update({"status": final_status, "error": str(exc), "completed_at": now_text(), "job_id": job_id})
        statuses[key] = failed
        save_statuses(statuses)
        set_job(
            job_id,
            status=final_status,
            completed_at=now_text(),
            error=str(exc),
            progress_stage="cancelled" if final_status == "Cancelled" else "failed",
            progress_step=0,
            progress_percent=0,
            progress_text=final_status,
        )
        CANCELLED_JOBS.discard(job_id)


def start_job(
    day: str,
    period: str | int,
    payload: dict | None = None,
    reprocess: bool = False,
    user: dict | None = None,
) -> tuple[str, bool]:
    ensure_jobs_hydrated()
    actor = public_user(user) or public_user(load_role_users()[0])
    row, video_dir, session = resolve_processing_context(day, period, payload or {}, actor)
    # Validate the exact five-checkpoint source and immutable policy before a job
    # is persisted, so invalid browser requests fail synchronously and cleanly.
    build_effective_processing_options(video_dir, payload or {})

    with LOCK:
        existing = active_job_for_session(JOBS, session["session_id"])
        if existing:
            return str(existing["job_id"]), False

        job_id = uuid.uuid4().hex[:8].upper()
        JOBS[job_id] = {
            "job_id": job_id,
            "key": session["session_id"],
            "session_id": session["session_id"],
            "session_date": session["session_date"],
            "day": session["day"],
            "period": session["period"],
            "section": session["section"],
            "subject": session["subject_abbr"],
            "subject_abbr": session["subject_abbr"],
            "subject_name": session.get("subject_name"),
            "faculty_id": session.get("faculty_id"),
            "faculty_name": session.get("faculty_name"),
            "input_slot": session.get("input_slot"),
            "input_source_type": session.get("input_source_type"),
            "input_source_path": session.get("input_source_path"),
            "status": "Pending",
            "progress_stage": "queued",
            "progress_step": 0,
            "progress_percent": 2,
            "progress_text": "Queued",
            "created_at": now_text(),
            "is_reprocess": bool(reprocess),
            "started_by_user_id": actor.get("id"),
            "controlled_by": actor.get("name"),
            "controller_role": actor.get("roleLabel") or actor.get("role"),
            "error": None,
        }
        _persist_jobs_locked()

    thread = threading.Thread(
        target=run_slot_job,
        args=(job_id, day, period, payload, reprocess, actor),
        daemon=True,
    )
    thread.start()
    return job_id, True


@app.get("/")
def home():
    return jsonify({
        "message": "Sreenidhi Smart Attendance backend is running",
        "frontend": "http://localhost:5173",
        "api": ["/api/health", "/api/profile", "/api/timetable", "/api/process", "/api/status", "/api/job/<id>", "/api/cancel/<id>", "/api/reset-processing", "/api/reports", "/api/videos", "/api/hod/overview", "/api/hod/config", "/api/students"],
    })


@app.get("/api/health")
def api_health():
    ensure_workflow_state_initialized()
    missing = []
    for path in [ROOT_DIR / "models" / "student_embeddings.pkl", ROOT_DIR / "models" / "face_detection_yunet_2023mar.onnx", ROOT_DIR / "models" / "face_recognition_sface_2021dec.onnx"]:
        if not path.exists():
            missing.append(path.relative_to(ROOT_DIR).as_posix())
    return jsonify({
        "ok": True,
        "backend": "connected",
        "time": now_text(),
        "missing_required_files": missing,
        "ready_for_processing": len(missing) == 0,
        "processing_policy": {
            "policy_version": PROCESSING_CONTRACT_POLICY_VERSION,
            "mode": QUALITY_AWARE_PROCESSING_MODE,
            "official_recognition_authority": "strict_tracklet_aggregate_with_guarded_review_candidates",
            "zone_mode": "zones",
            "tracklet_mode": "tracklets",
            "tracklets_used_for_official_attendance": True,
            "authoritative_roster_required": True,
            "exact_checkpoint_layout_required": True,
        },
        "active_jobs": len([j for j in JOBS.values() if j.get("status") in ("Pending", "Processing")]),
        "workflow_state": WORKFLOW_STATE_REPORT or {"status": "not_initialized_by_main"},
    })


@app.get("/api/profile")
def api_profile():
    user = public_user(get_current_user())
    if not user:
        return jsonify({"success": False, "error": "Authentication required."}), 401
    return jsonify({**ADMIN_PROFILE, **user})


@app.post("/api/auth/login")
def api_auth_login():
    payload = request.get_json(silent=True) or {}
    username = str(payload.get("username") or "").strip().lower()
    password = str(payload.get("password") or "")
    for user in load_role_users():
        if str(user.get("username", "")).lower() == username and verify_password(user, password):
            token, expires_at = issue_auth_session(user)
            return jsonify({
                "success": True,
                "user": public_user(user),
                "session_token": token,
                "session_expires_at": expires_at,
            })
    return jsonify({"success": False, "error": "Invalid username or password."}), 401


@app.post("/api/auth/logout")
def api_auth_logout():
    token = _bearer_token()
    if token:
        revoke_auth_session(token)
    return jsonify({"success": True})


@app.get("/api/auth/me")
def api_auth_me():
    user = public_user(get_current_user())
    if not user:
        return jsonify({"success": False, "error": "Stored login is no longer valid."}), 401
    return jsonify({"success": True, "user": user})


@app.get("/api/timetable")
def api_timetable():
    day = request.args.get("day") or datetime.now().strftime("%A")
    user = get_current_user()
    return jsonify({"day": normalize_day(day), "timetable": timetable_for_day(day, user), "admin": ADMIN_PROFILE, "user": public_user(user)})


@app.post("/api/process")
def api_process():
    payload = request.get_json(silent=True) or {}
    user = get_current_user()
    day = normalize_day(payload.get("day") or datetime.now().strftime("%A"))
    period = payload.get("period") or payload.get("period_number")
    subject_track = str(payload.get("subject_track") or "").upper()
    if not period:
        return jsonify({"success": False, "error": "period is required"}), 400
    if subject_track and not subject_allowed_for_period(day, period, subject_track, user):
        return jsonify({"success": False, "error": "You do not have permission to process this subject slot."}), 403
    if not subject_track and not get_period_row(day, period, user):
        return jsonify({"success": False, "error": "You do not have permission to process this period."}), 403
    try:
        _, _, session = resolve_processing_context(day, period, payload, user)
        key = session["session_id"]
        existing = repair_stale_processing_statuses(load_statuses()).get(key)
        if existing and existing.get("status") in ("Completed", "Needs Review"):
            return jsonify({"success": True, "already_done": True, "message": "Attendance already has a result. Use Reprocess to run it again.", "status": existing})
        if existing and existing.get("status") == "Processing":
            return jsonify({"success": True, "message": "Attendance is already processing.", "job_id": existing.get("job_id")})
        job_id, created = start_job(day, period, payload, reprocess=False, user=user)
        message = (
            f"Started attendance processing for {day} {period_display(period)}."
            if created
            else "This attendance session is already processing; reusing the active job."
        )
        return jsonify({"success": True, "message": message, "job_id": job_id, "job_created": created})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@app.post("/api/reprocess")
def api_reprocess():
    payload = request.get_json(silent=True) or {}
    user = get_current_user()
    day = normalize_day(payload.get("day") or datetime.now().strftime("%A"))
    period = payload.get("period") or payload.get("period_number")
    subject_track = str(payload.get("subject_track") or "").upper()
    if not period:
        return jsonify({"success": False, "error": "period is required"}), 400
    if subject_track and not subject_allowed_for_period(day, period, subject_track, user):
        return jsonify({"success": False, "error": "You do not have permission to reprocess this subject slot."}), 403
    if not subject_track and not get_period_row(day, period, user):
        return jsonify({"success": False, "error": "You do not have permission to reprocess this period."}), 403
    try:
        job_id, created = start_job(day, period, payload, reprocess=True, user=user)
        message = (
            f"Reprocessing started for {day} {period_display(period)}."
            if created
            else "This attendance session is already processing; reusing the active job."
        )
        return jsonify({"success": True, "message": message, "job_id": job_id, "job_created": created})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@app.get("/api/status")
def api_status():
    ensure_jobs_hydrated()
    user = get_current_user()
    jobs = [public_job(j) for j in JOBS.values()]
    if not is_admin_user(user):
        jobs = [job for job in jobs if user_can_access_session(user, job)]
    return jsonify(sorted(jobs, key=lambda x: x.get("created_at", ""), reverse=True))


@app.get("/api/job/<job_id>")
def api_job(job_id):
    ensure_jobs_hydrated()
    user = get_current_user()
    job = JOBS.get(str(job_id).upper()) or JOBS.get(str(job_id))
    if not job:
        return jsonify({"success": False, "error": "Job not found"}), 404
    if not user_can_access_session(user, job):
        return jsonify({"success": False, "error": "You do not have permission to view this job."}), 403
    return jsonify({"success": True, "job": public_job(job)})


@app.post("/api/cancel/<job_id>")
def api_cancel_job(job_id):
    ensure_jobs_hydrated()
    user = get_current_user()
    job_id = str(job_id).upper()
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"success": False, "error": "Job not found"}), 404
    if not user_can_access_session(user, job):
        return jsonify({"success": False, "error": "You do not have permission to cancel this job."}), 403
    if job.get("status") not in ("Pending", "Processing"):
        return jsonify({"success": True, "message": "Job is not running.", "job": public_job(job)})
    CANCELLED_JOBS.add(job_id)
    proc = RUNNING_PROCESSES.get(job_id)
    if proc:
        terminate_process_tree(proc)
    set_job(
        job_id,
        status="Cancelled",
        completed_at=now_text(),
        progress_stage="cancelled",
        progress_step=0,
        progress_percent=0,
        progress_text=f"Cancelled by {user.get('name') or 'authorized user'}",
    )
    statuses = load_statuses()
    key = job.get("key")
    if key and key in statuses:
        statuses[key].update({"status": "Cancelled", "completed_at": now_text(), "error": f"Cancelled by {user.get('name') or 'authorized user'}"})
        save_statuses(statuses)
    return jsonify({"success": True, "message": "Processing cancelled.", "job": public_job(JOBS.get(job_id, {}))})


@app.post("/api/reset-processing")
def api_reset_processing():
    user = get_current_user()
    if not is_admin_user(user):
        return jsonify({"success": False, "error": "Only HOD/Admin can reset processing state."}), 403
    statuses = load_statuses()
    changed = []
    for key, entry in list(statuses.items()):
        if isinstance(entry, dict) and entry.get("status") in ("Processing", "Cancelled"):
            entry["status"] = "Failed"
            entry["error"] = "Manually reset by HOD. Reprocess this slot."
            entry["completed_at"] = now_text()
            statuses[key] = entry
            changed.append(key)
    save_statuses(statuses)
    return jsonify({"success": True, "message": f"Reset {len(changed)} processing slot(s).", "keys": changed})


@app.get("/api/attendance-sessions")
def api_attendance_sessions():
    user = get_current_user()
    statuses = repair_stale_processing_statuses(load_statuses())
    sessions = []
    for key, entry in statuses.items():
        compact = compact_attendance_session(str(key), entry, user)
        if compact:
            sessions.append(compact)

    def sort_key(item: dict):
        return (
            str(item.get("session_date") or ""),
            str(item.get("completed_at") or item.get("finalized_at") or item.get("last_edited_at") or ""),
            str(item.get("period") or ""),
            str(item.get("session_id") or ""),
        )

    sessions.sort(key=sort_key, reverse=True)
    return jsonify({"success": True, "sessions": sessions, "count": len(sessions), "user": public_user(user)})


@app.get("/api/reports")
def api_reports():
    files = sorted([p.relative_to(OUTPUT_DIR).as_posix() for p in OUTPUT_DIR.rglob("*") if p.is_file()], reverse=True)
    names = [(f, Path(f).name) for f in files]
    return jsonify({
        "attendance_reports": [f for f, name in names if name.startswith("attendance_")],
        "slot_summaries": [f for f, name in names if name.startswith("slot_summary_")],
        "checkpoint_summaries": [f for f, name in names if name.startswith("checkpoint_summary_")],
        "student_checkpoint_evidence": [f for f, name in names if name.startswith("student_checkpoint_evidence_")],
        "camera_summaries": [f for f, name in names if name.startswith("camera_summary_")],
        "needs_review_reports": [f for f, name in names if name.startswith("needs_review_")],
        "detection_logs": [f for f, name in names if name.startswith("detection_log_")],
        "teacher_reports": [],
        "admin_reports": [f for f, name in names if name.startswith("admin_reviewed_")],
    })


@app.get("/api/videos")
def api_videos():
    videos = []
    if VIDEO_DIR.exists():
        for p in sorted(VIDEO_DIR.rglob("*")):
            if p.is_file() and p.suffix.lower() in ALLOWED_VIDEO_EXTENSIONS:
                rel = p.relative_to(ROOT_DIR).as_posix()
                stat = p.stat()
                videos.append({"name": p.name, "path": rel, "size": stat.st_size, "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()})
    return jsonify({"videos": videos})


@app.post("/api/videos/upload")
def api_videos_upload():
    files = request.files.getlist("video")
    slot_id = request.form.get("slot_id") or request.form.get("slotId")
    dest_dir = VIDEO_DIR / slot_id if slot_id else VIDEO_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for file in files:
        name = secure_filename(file.filename or "")
        if not name or Path(name).suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
            return jsonify({"success": False, "error": f"Invalid video file: {name}"}), 400
        dest = dest_dir / name
        file.save(dest)
        saved.append(dest.relative_to(ROOT_DIR).as_posix())
    return jsonify({"success": True, "message": f"Uploaded {len(saved)} video(s).", "files": saved})


def get_completed_session_entry(session_id: str, user: dict):
    entry = load_statuses().get(session_id)
    if not entry or not isinstance(entry, dict):
        return None, (jsonify({"success": False, "error": "Attendance is not completed for this session yet."}), 404)
    if entry.get("status") not in ("Completed", "Needs Review") or not entry.get("attendance_data"):
        return None, (jsonify({"success": False, "error": "Attendance is not completed for this session yet."}), 404)
    if not user_can_access_session(user, entry):
        return None, (jsonify({"success": False, "error": "Faculty cannot access another faculty's session."}), 403)
    return entry, None


@app.get("/api/attendance/<session_id>")
def api_get_attendance_session(session_id):
    user = get_current_user()
    entry, error = get_completed_session_entry(session_id, user)
    if error:
        return error
    rows = filter_attendance_rows_for_user(apply_overrides(session_id, entry.get("attendance_data", [])), user)
    review_state = review_state_for_rows(rows, entry)
    state_name = review_state_name(entry, review_state)
    response_review_state = public_review_state(review_state, user)
    return jsonify({
        "success": True,
        "entry": public_attendance_entry(entry),
        "attendance_data": rows,
        "overrides": load_overrides().get(session_id, {}),
        "review_state": {
            **response_review_state,
            "attendance_finalized": attendance_is_finalized(entry),
            "state": state_name,
        },
        "admin": ADMIN_PROFILE,
        "user": public_user(user),
    })


@app.post("/api/attendance/<session_id>/edit")
def api_edit_attendance_session(session_id):
    payload = request.get_json(silent=True) or {}
    requested_session_id = str(payload.get("session_id") or "").strip()
    if requested_session_id and requested_session_id != session_id:
        return jsonify({"success": False, "error": "Session ID mismatch."}), 400
    changes = payload.get("changes", [])
    if not isinstance(changes, list) or not changes:
        return jsonify({"success": False, "error": "At least one correction is required."}), 400

    user = get_current_user()
    with LOCK:
        statuses = load_statuses()
        entry = statuses.get(session_id)
        if not isinstance(entry, dict) or entry.get("status") not in ("Completed", "Needs Review") or not entry.get("attendance_data"):
            return jsonify({"success": False, "error": "Attendance is not completed for this session yet."}), 404
        if not user_can_access_session(user, entry):
            return jsonify({"success": False, "error": "Faculty cannot access another faculty's session."}), 403

        allowed_rolls = allowed_rolls_for_user(user)
        by_roll: dict[str, dict] = {}
        for row in entry.get("attendance_data", []):
            roll = canonical_student_roll(row.get("Roll_Number") or row.get("roll"))
            if not roll:
                continue
            if roll in by_roll:
                return jsonify({"success": False, "error": f"Duplicate attendance rows found for {roll}; HOD review is required."}), 409
            by_roll[roll] = row

        previous_overrides = load_overrides()
        session_overrides = canonical_override_map(previous_overrides.get(session_id, {}))
        applied = 0
        actor = public_user(user)["name"]

        for change in changes:
            roll = canonical_student_roll(change.get("roll"))
            status = str(change.get("status") or "").strip()
            reason = str(change.get("reason") or "").strip()
            if not roll or roll not in by_roll:
                return jsonify({"success": False, "error": f"Unknown student roll in correction: {roll or 'blank'}."}), 400
            if not is_admin_user(user) and roll not in allowed_rolls:
                return jsonify({"success": False, "error": f"You do not have permission to edit {roll}."}), 403
            if status in ("", "CLEAR"):
                if roll in session_overrides:
                    session_overrides.pop(roll, None)
                    applied += 1
                continue
            if status not in ("Present", "Absent", "Needs Review"):
                return jsonify({"success": False, "error": f"Invalid correction status for {roll}."}), 400
            if not reason:
                return jsonify({"success": False, "error": f"A correction reason is required for {roll}."}), 400
            session_overrides[roll] = {
                "original_status": by_roll[roll].get("Status") or by_roll[roll].get("Final_Status"),
                "manual_status": status,
                "reason": reason,
                "edited_by": actor,
                "edited_at": now_text(),
            }
            applied += 1

        if not applied:
            return jsonify({"success": False, "error": "The submitted corrections do not change the saved review state."}), 400

        edited_rows_all = apply_override_map(entry.get("attendance_data", []), session_overrides)
        edited_rows = filter_attendance_rows_for_user(edited_rows_all, user)
        review_state = review_state_for_rows(edited_rows_all, entry)
        summary = summarize_attendance(edited_rows_all)

        export_rows = edited_rows_all if is_admin_user(user) else edited_rows
        export_path = export_reviewed_attendance(entry, export_rows)
        export_ref = output_ref(export_path)
        updated = dict(entry)
        was_finalized = attendance_is_finalized(updated)
        state_name = "roster_mismatch" if not review_state["roster_complete"] else ("needs_attention" if review_state["unresolved_count"] else "ready_to_finalize")
        try:
            revision = int(updated.get("review_revision") or 0) + 1
        except (TypeError, ValueError):
            revision = 1

        updated.update({
            "status": "Needs Review",
            "report_type": "Edited",
            "last_edited_at": now_text(),
            "last_edited_by": actor,
            "admin_reviewed_csv": export_ref,
            "attendance_finalized": False,
            "Attendance_Finalized": "No",
            "requires_manual_review": bool(review_state["unresolved_count"] or not review_state["roster_complete"]),
            "review_state": state_name,
            "review_revision": revision,
            "roster_complete": review_state["roster_complete"],
            "roster_expected_count": review_state["expected_count"],
            "roster_row_count": review_state["row_count"],
            "missing_roster_rolls": review_state["missing_rolls"],
            "unexpected_roster_rolls": review_state["unexpected_rolls"],
            "duplicate_roster_rolls": review_state["duplicate_rolls"],
            "missing_roster_count": review_state["missing_count"],
            "unexpected_roster_count": review_state["unexpected_count"],
            "duplicate_roster_count": review_state["duplicate_count"],
            **summary,
        })
        if was_finalized:
            history = list(updated.get("finalization_history") or []) if isinstance(updated.get("finalization_history") or [], list) else []
            history.append({
                "finalized_at": updated.get("finalized_at"),
                "finalized_by": updated.get("finalized_by"),
                "final_attendance_csv": updated.get("final_attendance_csv"),
                "reopened_at": now_text(),
                "reopened_by": actor,
            })
            updated["finalization_history"] = history[-20:]
            updated["previous_finalized_at"] = updated.get("finalized_at")
            updated["previous_finalized_by"] = updated.get("finalized_by")
            updated["previous_final_attendance_csv"] = updated.get("final_attendance_csv")
        updated.pop("finalized_at", None)
        updated.pop("finalized_by", None)
        updated.pop("final_attendance_csv", None)

        next_overrides = dict(previous_overrides)
        if session_overrides:
            next_overrides[session_id] = session_overrides
        else:
            next_overrides.pop(session_id, None)
        next_statuses = dict(statuses)
        next_statuses[session_id] = updated

        try:
            save_overrides(next_overrides)
            save_statuses(next_statuses)
        except Exception as exc:
            try:
                save_overrides(previous_overrides)
                save_statuses(statuses)
            finally:
                export_path.unlink(missing_ok=True)
            return jsonify({"success": False, "error": f"Corrections were not saved safely: {exc}"}), 500

    if not review_state["roster_complete"]:
        message = f"Corrections saved, but finalization is blocked: {review_state['reason']}"
    elif review_state["unresolved_count"]:
        message = f"Corrections saved. {review_state['unresolved_count']} student(s) still need review."
    else:
        message = "Corrections saved. This report is ready for explicit finalization."
    return jsonify({
        "success": True,
        "message": message,
        "entry": public_attendance_entry(updated),
        "attendance_data": edited_rows,
        "review_state": {**public_review_state(review_state, user), "attendance_finalized": False, "state": state_name},
        "admin_reviewed_csv": export_ref,
        "download_url": f"/api/download/{export_ref}",
    })

@app.get("/api/attendance/<session_id>/export-edited")
def api_export_edited_session(session_id):
    user = get_current_user()
    entry, error = get_completed_session_entry(session_id, user)
    if error:
        return error
    rows = filter_attendance_rows_for_user(apply_overrides(session_id, entry.get("attendance_data", [])), user)
    if attendance_is_finalized(entry) and entry.get("final_attendance_csv"):
        final_path = safe_output_path(entry.get("final_attendance_csv"))
        if final_path:
            final_ref = output_ref(final_path)
            return jsonify({"success": True, "file": final_ref, "download_url": f"/api/download/{final_ref}", "finalized": True, "reused": True})
    export_path = export_reviewed_attendance(entry, rows)
    export_ref = output_ref(export_path)
    return jsonify({"success": True, "file": export_ref, "download_url": f"/api/download/{export_ref}", "finalized": False, "reused": False})


@app.post("/api/attendance/<session_id>/finalize")
def api_finalize_attendance_session(session_id):
    user = get_current_user()
    with LOCK:
        statuses = load_statuses()
        entry = statuses.get(session_id)
        if not isinstance(entry, dict) or entry.get("status") not in ("Completed", "Needs Review") or not entry.get("attendance_data"):
            return jsonify({"success": False, "error": "Attendance is not completed for this session yet."}), 404
        if not user_can_access_session(user, entry):
            return jsonify({"success": False, "error": "Faculty cannot access another faculty's session."}), 403

        rows_all = apply_overrides(session_id, entry.get("attendance_data", []))
        rows_visible = filter_attendance_rows_for_user(rows_all, user)
        review_state = review_state_for_rows(rows_all, entry)
        if not review_state["roster_complete"]:
            return jsonify({
                "success": False,
                "error": f"Cannot finalize because the attendance rows do not match the authoritative roster: {review_state['reason']}",
                "review_state": {**public_review_state(review_state, user), "attendance_finalized": False, "state": "roster_mismatch"},
            }), 409
        if review_state["unresolved_count"]:
            return jsonify({
                "success": False,
                "error": f"Cannot finalize while {review_state['unresolved_count']} student(s) still need review.",
                "review_state": {**public_review_state(review_state, user), "attendance_finalized": False, "state": "needs_attention"},
            }), 409

        updated = dict(entry)
        if attendance_is_finalized(updated) and updated.get("final_attendance_csv"):
            existing_path = safe_output_path(updated.get("final_attendance_csv"))
            if existing_path:
                existing_ref = output_ref(existing_path)
                return jsonify({
                    "success": True,
                    "already_finalized": True,
                    "message": "Attendance was already finalized; the verified final export was reused.",
                    "entry": public_attendance_entry(updated),
                    "attendance_data": rows_visible,
                    "review_state": {**public_review_state(review_state, user), "attendance_finalized": True, "state": "finalized"},
                    "file": existing_ref,
                    "download_url": f"/api/download/{existing_ref}",
                })

        final_path = export_finalized_attendance(updated, rows_all)
        final_ref = output_ref(final_path)
        summary = summarize_attendance(rows_all)
        try:
            revision = int(updated.get("review_revision") or 0) + 1
        except (TypeError, ValueError):
            revision = 1
        updated.update({
            "status": "Completed",
            "report_type": "Finalized",
            "attendance_finalized": True,
            "Attendance_Finalized": "Yes",
            "requires_manual_review": False,
            "review_state": "finalized",
            "review_revision": revision,
            "finalized_at": now_text(),
            "finalized_by": public_user(user)["name"],
            "final_attendance_csv": final_ref,
            "roster_complete": True,
            "roster_expected_count": review_state["expected_count"],
            "roster_row_count": review_state["row_count"],
            "missing_roster_rolls": [],
            "unexpected_roster_rolls": [],
            "duplicate_roster_rolls": [],
            "missing_roster_count": 0,
            "unexpected_roster_count": 0,
            "duplicate_roster_count": 0,
            **summary,
        })
        next_statuses = dict(statuses)
        next_statuses[session_id] = updated
        try:
            save_statuses(next_statuses)
        except Exception as exc:
            final_path.unlink(missing_ok=True)
            return jsonify({"success": False, "error": f"Finalization was not committed safely: {exc}"}), 500

    return jsonify({
        "success": True,
        "already_finalized": False,
        "message": "Attendance finalized successfully. The final CSV is ready to download.",
        "entry": public_attendance_entry(updated),
        "attendance_data": rows_visible,
        "review_state": {**public_review_state(review_state, user), "attendance_finalized": True, "state": "finalized"},
        "file": final_ref,
        "download_url": f"/api/download/{final_ref}",
    })

@app.get("/api/attendance/<day>/<period>")
def api_get_attendance(day, period):
    user = get_current_user()
    subject_track = request.args.get("subject_track")
    row = get_period_row(day, period, user, subject_track)
    if row and row.get("session_id"):
        return api_get_attendance_session(row["session_id"])
    key = period_key(day, period)
    entry = load_statuses().get(key)
    if not entry or entry.get("status") != "Completed":
        return jsonify({"success": False, "error": "Attendance is not completed for this slot yet."}), 404
    if not get_period_row(day, period, user):
        return jsonify({"success": False, "error": "You do not have permission to view this slot."}), 403
    rows = filter_attendance_rows_for_user(apply_overrides(key, entry.get("attendance_data", [])), user)
    return jsonify({"success": True, "entry": entry, "attendance_data": rows, "overrides": load_overrides().get(key, {}), "admin": ADMIN_PROFILE, "user": public_user(user)})


@app.post("/api/attendance/<day>/<period>/edit")
def api_edit_attendance(day, period):
    payload = request.get_json(silent=True) or {}
    changes = payload.get("changes", [])
    user = get_current_user()
    if not get_period_row(day, period, user):
        return jsonify({"success": False, "error": "You do not have permission to edit this slot."}), 403
    allowed_rolls = allowed_rolls_for_user(user)
    key = period_key(day, period)
    statuses = load_statuses()
    entry = statuses.get(key)
    if not entry or entry.get("status") != "Completed":
        return jsonify({"success": False, "error": "Attendance is not completed for this slot yet."}), 404
    by_roll = {str(r.get("Roll_Number", "")).upper(): r for r in entry.get("attendance_data", [])}
    overrides = load_overrides()
    period_overrides = overrides.get(key, {})
    for change in changes:
        roll = str(change.get("roll") or "").upper()
        status = change.get("status")
        reason = (change.get("reason") or "Attendance review correction").strip()
        if roll not in by_roll:
            continue
        if not is_admin_user(user) and roll not in allowed_rolls:
            continue
        if status in (None, "", "CLEAR"):
            period_overrides.pop(roll, None)
        elif status in ("Present", "Absent", "Needs Review"):
            period_overrides[roll] = {
                "original_status": by_roll[roll].get("Status") or by_roll[roll].get("Final_Status"),
                "manual_status": status,
                "reason": reason,
                "edited_by": public_user(user)["name"],
                "edited_at": now_text(),
            }
    if period_overrides:
        overrides[key] = period_overrides
    else:
        overrides.pop(key, None)
    save_overrides(overrides)
    edited_rows_all = apply_overrides(key, entry.get("attendance_data", []))
    edited_rows = filter_attendance_rows_for_user(edited_rows_all, user)
    summary = summarize_attendance(edited_rows_all)
    export_path = export_reviewed_attendance(entry, edited_rows_all)
    updated = dict(entry)
    updated.update({
        "report_type": "Edited",
        "last_edited_at": now_text(),
        "admin_reviewed_csv": output_ref(export_path),
        **summary,
    })
    statuses[key] = updated
    save_statuses(statuses)
    return jsonify({"success": True, "message": "Admin corrections saved and reviewed CSV exported.", "entry": entry, "admin_reviewed_csv": output_ref(export_path)})


@app.get("/api/attendance/<day>/<period>/export-edited")
def api_export_edited(day, period):
    key = period_key(day, period)
    entry = load_statuses().get(key)
    if not entry or entry.get("status") != "Completed":
        return jsonify({"success": False, "error": "Attendance is not completed for this slot yet."}), 404
    user = get_current_user()
    if not get_period_row(day, period, user):
        return jsonify({"success": False, "error": "You do not have permission to export this slot."}), 403
    rows = filter_attendance_rows_for_user(apply_overrides(key, entry.get("attendance_data", [])), user)
    export_path = export_reviewed_attendance(entry, rows)
    return jsonify({"success": True, "file": output_ref(export_path), "download_url": f"/api/download/{output_ref(export_path)}"})


@app.get("/attendance_website/<path:filename>")
def serve_report_compat(filename):
    path = safe_output_path(filename)
    if not path:
        return jsonify({"success": False, "error": "Report not found"}), 404
    rel = path.relative_to(OUTPUT_DIR).as_posix()
    return send_from_directory(OUTPUT_DIR, rel, as_attachment=False)


@app.get("/api/download/<path:filename>")
def download_output(filename):
    path = safe_output_path(filename)
    if not path:
        return jsonify({"success": False, "error": "File not found"}), 404
    rel = path.relative_to(OUTPUT_DIR).as_posix()
    return send_from_directory(OUTPUT_DIR, rel, as_attachment=True)



@app.get("/api/hod/overview")
def api_hod_overview():
    user = get_current_user()
    if not user:
        return jsonify({"success": False, "error": "Authentication required."}), 401
    if not is_admin_user(user):
        return jsonify({"success": False, "error": "HOD access is required."}), 403
    payload = hod_overview_payload()
    payload["user"] = public_user(user)
    return jsonify(payload)



@app.get("/api/hod/config")
def api_hod_config():
    user, error = _require_explicit_hod()
    if error:
        return error
    try:
        payload = configuration_governance_payload()
        payload["user"] = public_user(user)
        return jsonify(payload)
    except ConfigGovernanceError as exc:
        return jsonify({"success": False, "error": str(exc)}), 409


@app.post("/api/hod/config/faculty")
def api_hod_config_faculty():
    user, error = _require_explicit_hod()
    if error:
        return error
    payload = request.get_json(silent=True) or {}
    reason = str(payload.get("reason") or "").strip()
    with CONFIG_WRITE_LOCK:
        try:
            mapping = load_student_map()
            current_users = validate_role_users(load_role_users(), _configuration_subject_catalog(mapping))
            updated_users, change = create_or_update_faculty(
                current_users,
                _configuration_subject_catalog(mapping),
                payload,
            )
            current_validation = _configuration_timetable_validation(mapping=mapping, users=current_users)
            if not current_validation.valid:
                raise ConfigGovernanceError("Current timetable failed validation: " + "; ".join(current_validation.errors[:5]))

            updated_mapping = mapping
            updated_rows = [dict(row) for row in current_validation.rows]
            target = next(row for row in updated_users if row.get("id") == change["faculty_id"])
            for subject in target.get("subjects") or []:
                updated_users, updated_mapping, updated_rows, _ = assign_subject_to_faculty(
                    updated_users,
                    updated_mapping,
                    updated_rows,
                    subject=subject,
                    faculty_id=change["faculty_id"],
                )

            targets = {USERS_PATH: config_json_bytes(updated_users)}
            if change["action"] == "update" and target.get("subjects"):
                targets[STUDENT_MAP_PATH] = config_json_bytes(updated_mapping)
                targets[find_timetable_path()] = config_timetable_bytes(updated_rows)

            audit = _configuration_transaction(
                user=user,
                targets=targets,
                change_type=f"faculty_{change['action']}",
                reason=reason,
                metadata=change,
            )
            return jsonify({
                "success": True,
                "message": f"Faculty account {change['action']}d safely.",
                "change": audit,
                "faculty": public_user(target),
            })
        except ConfigGovernanceError as exc:
            return jsonify({"success": False, "error": str(exc)}), 409
        except Exception as exc:
            return jsonify({"success": False, "error": f"Faculty configuration failed safely: {exc}"}), 500


@app.post("/api/hod/config/subject-assignment")
def api_hod_config_subject_assignment():
    user, error = _require_explicit_hod()
    if error:
        return error
    payload = request.get_json(silent=True) or {}
    subject = str(payload.get("subject") or "").strip().upper()
    faculty_id = str(payload.get("faculty_id") or "").strip().lower()
    reason = str(payload.get("reason") or "").strip()
    with CONFIG_WRITE_LOCK:
        try:
            mapping = load_student_map()
            users = validate_role_users(load_role_users(), _configuration_subject_catalog(mapping))
            current_faculty_id = str(
                (_configuration_subject_catalog(mapping).get(subject) or {}).get("facultyId")
                or (_configuration_subject_catalog(mapping).get(subject) or {}).get("faculty_id")
                or ""
            ).strip().lower()
            if current_faculty_id == faculty_id:
                raise ConfigGovernanceError(f"{subject} is already assigned to {faculty_id}.")
            validation = _configuration_timetable_validation(mapping=mapping, users=users)
            if not validation.valid:
                raise ConfigGovernanceError("Current timetable failed validation: " + "; ".join(validation.errors[:5]))
            new_users, new_mapping, new_rows, change = assign_subject_to_faculty(
                users,
                mapping,
                validation.rows,
                subject=subject,
                faculty_id=faculty_id,
            )
            post_validation = validate_timetable_bytes(
                config_timetable_bytes(new_rows),
                _configuration_subject_catalog(new_mapping),
                new_users,
            )
            if not post_validation.valid:
                raise ConfigGovernanceError("Updated timetable failed validation: " + "; ".join(post_validation.errors[:5]))
            audit = _configuration_transaction(
                user=user,
                targets={
                    USERS_PATH: config_json_bytes(new_users),
                    STUDENT_MAP_PATH: config_json_bytes(new_mapping),
                    find_timetable_path(): post_validation.canonical_csv,
                },
                change_type="subject_assignment",
                reason=reason,
                metadata=change,
            )
            return jsonify({
                "success": True,
                "message": f"{subject} was assigned to {change['faculty_name']} safely.",
                "change": audit,
            })
        except ConfigGovernanceError as exc:
            return jsonify({"success": False, "error": str(exc)}), 409
        except Exception as exc:
            return jsonify({"success": False, "error": f"Subject assignment failed safely: {exc}"}), 500


def _uploaded_timetable_bytes() -> tuple[bytes, str]:
    upload = request.files.get("timetable")
    if upload is None:
        raise ConfigGovernanceError("A timetable CSV file is required.")
    filename = secure_filename(upload.filename or "")
    if not filename.lower().endswith(".csv"):
        raise ConfigGovernanceError("Timetable upload must be a .csv file.")
    payload = upload.read(1_000_001)
    if len(payload) > 1_000_000:
        raise ConfigGovernanceError("Timetable CSV exceeds the 1 MB safety limit.")
    return payload, filename


@app.post("/api/hod/config/timetable/preview")
def api_hod_config_timetable_preview():
    user, error = _require_explicit_hod()
    if error:
        return error
    try:
        payload, filename = _uploaded_timetable_bytes()
        mapping = load_student_map()
        users = validate_role_users(load_role_users(), _configuration_subject_catalog(mapping))
        validation = validate_timetable_bytes(payload, _configuration_subject_catalog(mapping), users)
        return jsonify({
            "success": True,
            "filename": filename,
            "preview": validation.public_dict(include_rows=True),
            "write_executed": False,
        })
    except ConfigGovernanceError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@app.post("/api/hod/config/timetable/apply")
def api_hod_config_timetable_apply():
    user, error = _require_explicit_hod()
    if error:
        return error
    with CONFIG_WRITE_LOCK:
        try:
            payload, filename = _uploaded_timetable_bytes()
            expected_sha256 = str(request.form.get("expected_sha256") or "").strip().lower()
            reason = str(request.form.get("reason") or "").strip()
            confirmation = str(request.form.get("confirmation") or "").strip()
            if confirmation != "REPLACE TIMETABLE":
                raise ConfigGovernanceError('Type "REPLACE TIMETABLE" exactly to confirm.')
            mapping = load_student_map()
            users = validate_role_users(load_role_users(), _configuration_subject_catalog(mapping))
            validation = validate_timetable_bytes(payload, _configuration_subject_catalog(mapping), users)
            if not validation.valid:
                raise ConfigGovernanceError("Timetable validation failed: " + "; ".join(validation.errors[:8]))
            if expected_sha256 != validation.file_sha256:
                raise ConfigGovernanceError("Uploaded timetable changed after preview; preview it again.")
            audit = _configuration_transaction(
                user=user,
                targets={find_timetable_path(): validation.canonical_csv},
                change_type="timetable_replace",
                reason=reason,
                metadata={
                    "filename": filename,
                    "uploaded_sha256": validation.file_sha256,
                    "normalized_sha256": validation.summary["normalized_sha256"],
                    "row_count": validation.row_count,
                    "summary": validation.summary,
                    "warnings": list(validation.warnings),
                },
            )
            return jsonify({
                "success": True,
                "message": "Timetable replaced safely with backup and audit history.",
                "change": audit,
            })
        except ConfigGovernanceError as exc:
            return jsonify({"success": False, "error": str(exc)}), 409
        except Exception as exc:
            return jsonify({"success": False, "error": f"Timetable replacement failed safely: {exc}"}), 500


@app.post("/api/hod/config/rollback")
def api_hod_config_rollback():
    user, error = _require_explicit_hod()
    if error:
        return error
    payload = request.get_json(silent=True) or {}
    with CONFIG_WRITE_LOCK:
        try:
            blocker = _configuration_write_blocker()
            if blocker:
                raise ConfigGovernanceError(blocker)
            result = rollback_latest_config_change(
                repo_root=ROOT_DIR,
                backup_root=CONFIG_BACKUP_ROOT,
                audit_root=CONFIG_AUDIT_ROOT,
                pointer_path=CONFIG_POINTER_PATH,
                actor=public_user(user),
                confirm_change_id=str(payload.get("change_id") or ""),
                reason=str(payload.get("reason") or ""),
            )
            return jsonify({
                "success": True,
                "message": "Latest configuration change rolled back safely.",
                "rollback": result,
            })
        except ConfigGovernanceError as exc:
            return jsonify({"success": False, "error": str(exc)}), 409
        except Exception as exc:
            return jsonify({"success": False, "error": f"Configuration rollback failed safely: {exc}"}), 500


@app.get("/api/students")
def api_students():
    user = get_current_user()
    if not user:
        return jsonify({"success": False, "error": "Authentication required."}), 401
    mapping = load_student_map()
    students, coverage = student_coverage_rows(mapping)
    subjects = subject_catalog(mapping)
    if is_admin_user(user):
        return jsonify({"success": True, "subjects": subjects, "students": students, "coverage": coverage, "user": public_user(user)})
    allowed = allowed_rolls_for_user(user)
    visible = [student for student in students if student.get("roll") in allowed]
    allowed_subjects = user_subjects(user)
    return jsonify({
        "success": True,
        "subjects": {key: value for key, value in subjects.items() if key in allowed_subjects},
        "students": visible,
        "coverage": {
            **coverage,
            "total_students": len(visible),
            "dataset_available": sum(student.get("dataset_available") is True for student in visible),
            "embedding_available": sum(student.get("embedding_available") is True for student in visible),
            "ready": sum(student.get("coverage_status") == "ready" for student in visible),
            "issues": sum(student.get("coverage_status") not in {"ready", "source_unavailable"} for student in visible),
        },
        "user": public_user(user),
    })



@app.post("/api/live-demo/start")
def api_live_demo_start():
    user = get_current_user()
    payload = request.get_json(silent=True) or {}
    result = LIVE_DEMO.start(payload, user)
    status_code = 200 if result.get("success") else 400
    return jsonify(result), status_code


@app.post("/api/live-demo/stop")
def api_live_demo_stop():
    return jsonify(LIVE_DEMO.stop())


@app.get("/api/live-demo/state")
def api_live_demo_state():
    return jsonify({"success": True, "state": LIVE_DEMO.state(), "user": public_user(get_current_user())})


@app.get("/api/live-demo/feed")
def api_live_demo_feed():
    def generate():
        while True:
            frame = LIVE_DEMO.get_jpeg()
            yield b"--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-cache\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.08)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/video-layout", methods=["GET"])
def api_video_layout():
    slot_id = request.args.get("slot_id") or "MON_P1"
    video_dir = resolve_video_dir(slot_id, request.args.to_dict())
    layout = inspect_video_layout(video_dir)
    layout["video_dir"] = str(video_dir.relative_to(ROOT_DIR) if video_dir.is_relative_to(ROOT_DIR) else video_dir)
    return jsonify(layout)


if __name__ == "__main__":
    state_report = ensure_workflow_state_initialized()
    print("=" * 70)
    print(" Sreenidhi Smart Attendance Backend API")
    print(" Backend: http://127.0.0.1:5000")
    print(" React:   http://localhost:5173")
    migration = state_report.get("status_migration") or {}
    print(f" State:   ready · migrated={migration.get('changed', False)} · persisted_jobs={state_report.get('persisted_jobs', 0)}")
    print("=" * 70)
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False, threaded=True)
