from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import signal
import queue
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

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
FRONTEND_POLL_SECONDS = 3

ADMIN_PROFILE = {
    "name": "Main Admin",
    "role": "Attendance Controller",
    "organization": "Sreenidhi University",
    "department": "Smart Attendance Cell",
    "email": "admin@sreenidhi.edu.in",
}


USERS_PATH = DATA_DIR / "role_users.json"
STUDENT_MAP_PATH = DATA_DIR / "student_faculty_map.json"

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
CANCELLED_JOBS: set[str] = set()

LOCK = threading.Lock()


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
    return f"{normalize_day(day)}_{period_display(period)}"


def slot_id_for(day: str, period: str | int) -> str:
    return f"{day_prefix(day)}_P{period_text(period)}"


def read_json(path: Path, default):
    if not path.exists():
        write_json(path, default)
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, data) -> None:
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


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
    return clean


def get_current_user() -> dict:
    users = load_role_users()
    user_id = (request.headers.get("X-User-Id") or request.args.get("user_id") or "admin").strip().lower()
    for user in users:
        if str(user.get("id", "")).lower() == user_id or str(user.get("username", "")).lower() == user_id:
            return user
    return users[0]


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
        return {str(s.get("roll", "")).upper() for s in mapping.get("all_students", []) if s.get("roll")}
    allowed: set[str] = set()
    subject_students = mapping.get("subject_students", {})
    for subject in user_subjects(user):
        for student in subject_students.get(subject, []):
            if student.get("roll"):
                allowed.add(str(student["roll"]).upper())
    return allowed


def filter_attendance_rows_for_user(rows: list[dict], user: dict) -> list[dict]:
    if is_admin_user(user):
        return rows
    allowed = allowed_rolls_for_user(user)
    return [row for row in rows if str(row.get("Roll_Number") or row.get("roll") or "").upper() in allowed]


def load_statuses() -> dict:
    return read_json(STATUS_PATH, {})


def save_statuses(data: dict) -> None:
    write_json(STATUS_PATH, data)


def repair_stale_processing_statuses(statuses: dict) -> dict:
    """If Flask restarted or a script crashed, old Processing states can remain in JSON.
    Mark them as Failed so the frontend does not show infinite Processing.
    """
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
        item["att_status"] = statuses.get(period_key(day, item["period"]))
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
    """Create safe processing options for frontend jobs.

    If only demo clips are available, the job should not fail for missing CP2-CP5.
    It processes available clips and uses a demo attendance rule: 1 recognized checkpoint can mark Present.
    When all 5 CP folders exist, it uses the real MVP3 rule: 3/5 checkpoints.
    """
    payload = dict(payload or {})
    layout = inspect_video_layout(video_dir)
    options = {
        "checkpoint_mode": payload.get("checkpoint_mode") or DEFAULT_CHECKPOINT_MODE,
        "checkpoint_min_detections": payload.get("checkpoint_min_detections", DEFAULT_CHECKPOINT_MIN_DETECTIONS),
        "present_checkpoints": payload.get("present_checkpoints", DEFAULT_PRESENT_CHECKPOINTS),
        "strong_checkpoints": payload.get("strong_checkpoints", DEFAULT_STRONG_CHECKPOINTS),
        "review_checkpoints": payload.get("review_checkpoints", DEFAULT_REVIEW_CHECKPOINTS),
        "processing_profile": "full_mvp3_5_checkpoint",
        "layout": layout,
    }

    # Full checkpoint folders = real attendance voting.
    if layout["has_full_checkpoint_set"]:
        options["checkpoint_mode"] = "clip-folders" if options["checkpoint_mode"] in ("auto", "clip-folders") else options["checkpoint_mode"]
        return options

    # Current testing situation: only 3 clips / one checkpoint folder.
    if layout["is_demo_clip_set"]:
        if layout["checkpoint_folder_count"] > 0:
            options["checkpoint_mode"] = "clip-folders"
        else:
            options["checkpoint_mode"] = "split-video"
        options["present_checkpoints"] = int(payload.get("present_checkpoints", 1))
        options["strong_checkpoints"] = int(payload.get("strong_checkpoints", 1))
        options["review_checkpoints"] = int(payload.get("review_checkpoints", 1))
        options["processing_profile"] = "demo_3_clip_test"
        return options

    return options

def resolve_video_dir(slot_id: str, payload: dict | None = None) -> Path:
    payload = payload or {}
    explicit = payload.get("video_dir") or payload.get("videoDir")
    if explicit:
        path = Path(explicit)
        return path if path.is_absolute() else ROOT_DIR / path
    candidates = [
        VIDEO_DIR / slot_id,
        VIDEO_DIR / slot_id.lower(),
        VIDEO_DIR / slot_id.replace("_", ""),
        VIDEO_DIR,
    ]
    for path in candidates:
        if path.exists() and any(p.is_file() and p.suffix.lower() in ALLOWED_VIDEO_EXTENSIONS for p in path.rglob("*")):
            return path
    return VIDEO_DIR


def output_ref(path: Path | None) -> str | None:
    if not path:
        return None
    try:
        return path.relative_to(OUTPUT_DIR).as_posix()
    except Exception:
        return path.name


def newest(pattern: str, after: float = 0.0) -> Path | None:
    files = [p for p in OUTPUT_DIR.rglob(pattern) if p.is_file() and p.stat().st_mtime >= after]
    if not files:
        files = [p for p in OUTPUT_DIR.rglob(pattern) if p.is_file()]
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


def summarize_attendance(rows: list[dict]) -> dict:
    total = len(rows)
    present = sum(1 for r in rows if "present" in str(r.get("Status", "")).lower())
    review = sum(1 for r in rows if "review" in str(r.get("Status", "")).lower())
    absent = max(0, total - present - review)
    pct = round((present / total) * 100, 1) if total else 0
    return {
        "total_students": total,
        "present_count": present,
        "needs_review_count": review,
        "absent_count": absent,
        "attendance_percentage": pct,
    }




def export_reviewed_attendance(day: str, period: str | int, entry: dict, rows: list[dict]) -> Path:
    slot_id = entry.get("slot_id") or slot_id_for(day, period)
    subject = entry.get("subject") or entry.get("course_abbr") or "SUBJECT"
    run_dir = OUTPUT_DIR / datetime.now().strftime("%Y-%m-%d") / f"{slot_id}_{subject}"
    run_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = run_dir / f"admin_reviewed_{slot_id}_{timestamp}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def apply_overrides(key: str, rows: list[dict]) -> list[dict]:
    overrides = load_overrides().get(key, {})
    result = []
    for row in rows:
        item = dict(row)
        roll = str(item.get("Roll_Number", ""))
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


def public_job(job: dict) -> dict:
    """Return a JSON-safe job object without internal process handles."""
    if not job:
        return {}
    hidden = {"process"}
    return {k: v for k, v in job.items() if k not in hidden}


def set_job(job_id: str, **updates) -> None:
    with LOCK:
        job = JOBS.get(job_id, {})
        job.update(updates)
        job["updated_at"] = now_text()
        JOBS[job_id] = job


def append_job_log(job_id: str, line: str, keep: int = 40) -> None:
    line = (line or "").strip()
    if not line:
        return
    with LOCK:
        job = JOBS.get(job_id, {})
        logs = list(job.get("log_tail", []))
        logs.append(line)
        job["log_tail"] = logs[-keep:]
        # Simple progress text from the script stdout.
        if "Processing camera angle:" in line:
            job["progress_text"] = line
        elif line.startswith("CP") or " CP" in line or "processed" in line.lower():
            job["progress_text"] = line.strip()
        elif "Saved outputs" in line or "Output" in line:
            job["progress_text"] = line
        JOBS[job_id] = job


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


def run_slot_job(job_id: str, day: str, period: str | int, payload: dict | None = None, reprocess: bool = False) -> None:
    payload = payload or {}
    day = normalize_day(day)
    row = get_period_row(day, period, None, payload.get('subject_track'))
    if not row:
        set_job(job_id, status="Failed", error="Slot not found", completed_at=now_text())
        return

    key = period_key(day, period)
    slot_id = row.get("slot_id") or slot_id_for(day, period)
    video_dir = resolve_video_dir(slot_id, payload)
    processing_options = build_effective_processing_options(video_dir, payload)
    layout = processing_options["layout"]
    start_mtime = time.time() - 2

    statuses = load_statuses()
    statuses[key] = {
        **row,
        "key": key,
        "status": "Processing",
        "job_id": job_id,
        "started_at": now_text(),
        "controlled_by": ADMIN_PROFILE["name"],
        "controller_role": ADMIN_PROFILE["role"],
        "video_dir": str(video_dir.relative_to(ROOT_DIR) if video_dir.is_relative_to(ROOT_DIR) else video_dir),
        "processing_profile": processing_options["processing_profile"],
        "video_layout": layout,
        "sample_fps": payload.get("sample_fps", DEFAULT_SAMPLE_FPS),
        "log_mode": payload.get("log_mode", DEFAULT_LOG_MODE),
        "checkpoint_mode": processing_options["checkpoint_mode"],
        "present_checkpoints": processing_options["present_checkpoints"],
        "subject_track": payload.get("subject_track") or row.get("subject_track") or row.get("subject"),
        "error": None,
    }
    save_statuses(statuses)
    layout_text = "Full 5-checkpoint folders" if layout.get("has_full_checkpoint_set") else "Demo clips mode" if layout.get("is_demo_clip_set") else "Auto video mode"
    set_job(
        job_id,
        status="Processing",
        started_at=now_text(),
        progress_text=f"{layout_text}: {layout.get('recursive_video_count', 0)} video(s), {layout.get('checkpoint_folder_count', 0)} CP folder(s)",
        video_dir=str(video_dir.relative_to(ROOT_DIR) if video_dir.is_relative_to(ROOT_DIR) else video_dir),
        video_layout=layout,
        processing_profile=processing_options["processing_profile"],
    )

    command = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "mark_attendance_checkpoints.py"),
        "--timetable", str(find_timetable_path()),
        "--slot-id", slot_id,
        "--video-dir", str(video_dir),
        "--output-dir", str(OUTPUT_DIR),
        "--match-threshold", str(payload.get("match_threshold", DEFAULT_MATCH_THRESHOLD)),
        "--margin-threshold", str(payload.get("margin_threshold", DEFAULT_MARGIN_THRESHOLD)),
        "--checkpoint-min-detections", str(processing_options["checkpoint_min_detections"]),
        "--present-checkpoints", str(processing_options["present_checkpoints"]),
        "--strong-checkpoints", str(processing_options["strong_checkpoints"]),
        "--review-checkpoints", str(processing_options["review_checkpoints"]),
        "--settle-minutes", str(payload.get("settle_minutes", DEFAULT_SETTLE_MINUTES)),
        "--checkpoint-every-minutes", str(payload.get("checkpoint_every_minutes", DEFAULT_CHECKPOINT_EVERY_MINUTES)),
        "--checkpoint-clip-seconds", str(payload.get("checkpoint_clip_seconds", DEFAULT_CHECKPOINT_CLIP_SECONDS)),
        "--checkpoint-mode", str(processing_options["checkpoint_mode"]),
        "--frame-skip", str(payload.get("frame_skip", DEFAULT_FRAME_SKIP)),
        "--sample-fps", str(payload.get("sample_fps", DEFAULT_SAMPLE_FPS)),
        "--log-mode", str(payload.get("log_mode", DEFAULT_LOG_MODE)),
        "--output-layout", str(payload.get("output_layout", DEFAULT_OUTPUT_LAYOUT)),
        "--aggregate", str(payload.get("aggregate", DEFAULT_AGGREGATE)),
    ]
    if bool(payload.get("save_unknown", SAVE_UNKNOWN_BY_DEFAULT)):
        command.append("--save-unknown")

    try:
        timeout_seconds = int(payload.get("timeout_seconds", DEFAULT_JOB_TIMEOUT_SECONDS))
        # Run unbuffered so the frontend can receive progress lines instead of waiting until the end.
        command = [command[0], "-u", *command[1:]]
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
                raise RuntimeError("Cancelled by Main Admin")
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

        attendance_csv = newest(f"attendance_{slot_id}_*.csv", start_mtime)
        summary_csv = newest(f"slot_summary_{slot_id}_*.csv", start_mtime)
        checkpoint_summary_csv = newest(f"checkpoint_summary_{slot_id}_*.csv", start_mtime)
        student_checkpoint_csv = newest(f"student_checkpoint_evidence_{slot_id}_*.csv", start_mtime)
        camera_summary_csv = newest(f"camera_summary_{slot_id}_*.csv", start_mtime)
        needs_review_csv = newest(f"needs_review_{slot_id}_*.csv", start_mtime)
        detection_csv = newest(f"detection_log_{slot_id}_*.csv", start_mtime)
        excel_file = newest(f"attendance_{slot_id}_*.xlsx", start_mtime)
        attendance_data = read_attendance_csv(attendance_csv)
        summary = summarize_attendance(attendance_data)

        video_files = []
        if video_dir.exists():
            video_files = [p.relative_to(video_dir).as_posix() for p in sorted(video_dir.rglob("*")) if p.is_file() and p.suffix.lower() in ALLOWED_VIDEO_EXTENSIONS]

        entry = {
            **row,
            "key": key,
            "status": "Completed",
            "report_type": "Reprocessed" if reprocess else "AI",
            "job_id": job_id,
            "completed_at": now_text(),
            "controlled_by": ADMIN_PROFILE["name"],
            "controller_role": ADMIN_PROFILE["role"],
            "video_dir": str(video_dir.relative_to(ROOT_DIR) if video_dir.is_relative_to(ROOT_DIR) else video_dir),
            "video_files": video_files,
            "processing_profile": "fast_mvp3_1",
            "sample_fps": payload.get("sample_fps", DEFAULT_SAMPLE_FPS),
            "log_mode": payload.get("log_mode", DEFAULT_LOG_MODE),
            "output_layout": payload.get("output_layout", DEFAULT_OUTPUT_LAYOUT),
            "attendance_csv": output_ref(attendance_csv),
            "slot_summary_csv": output_ref(summary_csv),
            "detection_log_csv": output_ref(detection_csv),
            "checkpoint_summary_csv": output_ref(checkpoint_summary_csv),
            "student_checkpoint_evidence_csv": output_ref(student_checkpoint_csv),
            "camera_summary_csv": output_ref(camera_summary_csv),
            "needs_review_csv": output_ref(needs_review_csv),
            "class_report_file": output_ref(excel_file) if excel_file else output_ref(attendance_csv),
            "excel_file": output_ref(excel_file),
            "attendance_data": attendance_data,
            "error": None,
            **summary,
        }
        statuses = load_statuses()
        statuses[key] = entry
        save_statuses(statuses)
        set_job(job_id, status="Completed", completed_at=now_text(), result=entry, progress_text="Completed", stdout="\n".join(stdout_lines[-120:]))
    except Exception as exc:
        with LOCK:
            proc = RUNNING_PROCESSES.pop(job_id, None)
        if proc and proc.poll() is None:
            terminate_process_tree(proc)
        final_status = "Cancelled" if job_id in CANCELLED_JOBS else "Failed"
        statuses = load_statuses()
        failed = statuses.get(key, {**row, "key": key})
        failed.update({"status": final_status, "error": str(exc), "completed_at": now_text(), "job_id": job_id})
        statuses[key] = failed
        save_statuses(statuses)
        set_job(job_id, status=final_status, completed_at=now_text(), error=str(exc), progress_text=final_status)
        CANCELLED_JOBS.discard(job_id)


def start_job(day: str, period: str | int, payload: dict | None = None, reprocess: bool = False) -> str:
    row = get_period_row(day, period, None, payload.get('subject_track'))
    if row is None:
        raise ValueError(f"Period {period} not found for {normalize_day(day)}")
    job_id = uuid.uuid4().hex[:8].upper()
    JOBS[job_id] = {
        "job_id": job_id,
        "key": period_key(day, period),
        "day": normalize_day(day),
        "period": period_display(period),
        "subject": row.get("subject"),
        "status": "Pending",
        "progress_text": "Queued",
        "created_at": now_text(),
        "is_reprocess": bool(reprocess),
        "error": None,
    }
    thread = threading.Thread(target=run_slot_job, args=(job_id, day, period, payload, reprocess), daemon=True)
    thread.start()
    return job_id


@app.get("/")
def home():
    return jsonify({
        "message": "Sreenidhi Smart Attendance backend is running",
        "frontend": "http://localhost:5173",
        "api": ["/api/health", "/api/profile", "/api/timetable", "/api/process", "/api/status", "/api/job/<id>", "/api/cancel/<id>", "/api/reset-processing", "/api/reports", "/api/videos"],
    })


@app.get("/api/health")
def api_health():
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
        "active_jobs": len([j for j in JOBS.values() if j.get("status") in ("Pending", "Processing")]),
    })


@app.get("/api/profile")
def api_profile():
    user = public_user(get_current_user())
    return jsonify({**ADMIN_PROFILE, **(user or {})})


@app.post("/api/auth/login")
def api_auth_login():
    payload = request.get_json(silent=True) or {}
    username = str(payload.get("username") or "").strip().lower()
    password = str(payload.get("password") or "")
    for user in load_role_users():
        if str(user.get("username", "")).lower() == username and str(user.get("password", "")) == password:
            return jsonify({"success": True, "user": public_user(user)})
    return jsonify({"success": False, "error": "Invalid username or password."}), 401


@app.get("/api/auth/me")
def api_auth_me():
    return jsonify({"success": True, "user": public_user(get_current_user())})


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
    key = period_key(day, period)
    existing = load_statuses().get(key)
    if existing and existing.get("status") == "Completed":
        return jsonify({"success": True, "already_done": True, "message": "Attendance already completed. Use Reprocess to run again.", "status": existing})
    if existing and existing.get("status") == "Processing":
        return jsonify({"success": True, "message": "Attendance is already processing.", "job_id": existing.get("job_id")})
    try:
        job_id = start_job(day, period, payload, reprocess=False)
        return jsonify({"success": True, "message": f"Started attendance processing for {day} {period_display(period)}.", "job_id": job_id})
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
        job_id = start_job(day, period, payload, reprocess=True)
        return jsonify({"success": True, "message": f"Reprocessing started for {day} {period_display(period)}.", "job_id": job_id})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@app.get("/api/status")
def api_status():
    user = get_current_user()
    jobs = [public_job(j) for j in JOBS.values()]
    if not is_admin_user(user):
        allowed = {(row["day"], row["period"]) for row in timetable_for_day(datetime.now().strftime("%A"), user)}
        # Include all days, not just today, by checking direct permission per job.
        jobs = [job for job in jobs if get_period_row(job.get("day", ""), job.get("period", ""), user)]
    return jsonify(sorted(jobs, key=lambda x: x.get("created_at", ""), reverse=True))


@app.get("/api/job/<job_id>")
def api_job(job_id):
    job = JOBS.get(str(job_id).upper()) or JOBS.get(str(job_id))
    if not job:
        return jsonify({"success": False, "error": "Job not found"}), 404
    return jsonify({"success": True, "job": public_job(job)})


@app.post("/api/cancel/<job_id>")
def api_cancel_job(job_id):
    job_id = str(job_id).upper()
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"success": False, "error": "Job not found"}), 404
    if job.get("status") not in ("Pending", "Processing"):
        return jsonify({"success": True, "message": "Job is not running.", "job": public_job(job)})
    CANCELLED_JOBS.add(job_id)
    proc = RUNNING_PROCESSES.get(job_id)
    if proc:
        terminate_process_tree(proc)
    set_job(job_id, status="Cancelled", completed_at=now_text(), progress_text="Cancelled by Main Admin")
    statuses = load_statuses()
    key = job.get("key")
    if key and key in statuses:
        statuses[key].update({"status": "Cancelled", "completed_at": now_text(), "error": "Cancelled by Main Admin"})
        save_statuses(statuses)
    return jsonify({"success": True, "message": "Processing cancelled.", "job": public_job(JOBS.get(job_id, {}))})


@app.post("/api/reset-processing")
def api_reset_processing():
    statuses = load_statuses()
    changed = []
    for key, entry in list(statuses.items()):
        if isinstance(entry, dict) and entry.get("status") in ("Processing", "Cancelled"):
            entry["status"] = "Failed"
            entry["error"] = "Manually reset by Main Admin. Reprocess this slot."
            entry["completed_at"] = now_text()
            statuses[key] = entry
            changed.append(key)
    save_statuses(statuses)
    return jsonify({"success": True, "message": f"Reset {len(changed)} processing slot(s).", "keys": changed})


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


@app.get("/api/attendance/<day>/<period>")
def api_get_attendance(day, period):
    key = period_key(day, period)
    entry = load_statuses().get(key)
    if not entry or entry.get("status") != "Completed":
        return jsonify({"success": False, "error": "Attendance is not completed for this slot yet."}), 404
    user = get_current_user()
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
    by_roll = {str(r.get("Roll_Number", "")): r for r in entry.get("attendance_data", [])}
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
    export_path = export_reviewed_attendance(day, period, entry, edited_rows)
    entry.update({
        "attendance_data": edited_rows,
        "report_type": "Edited",
        "last_edited_at": now_text(),
        "admin_reviewed_csv": output_ref(export_path),
        **summary,
    })
    statuses[key] = entry
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
    export_path = export_reviewed_attendance(day, period, entry, rows)
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



@app.get("/api/students")
def api_students():
    user = get_current_user()
    mapping = load_student_map()
    if is_admin_user(user):
        return jsonify({"success": True, "subjects": mapping.get("subjects", SUBJECT_INFO), "students": mapping.get("all_students", []), "user": public_user(user)})
    allowed = allowed_rolls_for_user(user)
    students = [s for s in mapping.get("all_students", []) if str(s.get("roll", "")).upper() in allowed]
    return jsonify({"success": True, "subjects": {k: v for k, v in mapping.get("subjects", SUBJECT_INFO).items() if k in user_subjects(user)}, "students": students, "user": public_user(user)})



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
    print("=" * 70)
    print(" Sreenidhi Smart Attendance Backend API")
    print(" Backend: http://127.0.0.1:5000")
    print(" React:   http://localhost:5173")
    print("=" * 70)
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False, threaded=True)
