from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

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
    DEFAULT_MIN_DETECTIONS_FOR_PRESENT,
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
from src.face_attendance.timetable import list_slots, load_timetable, select_slot
from src.face_attendance.utils import clean_filename, ensure_dirs, iter_files, resize_keep_aspect, safe_crop


class SlotAttendanceBook:
    def __init__(self, all_students: list[str], min_detections: int, review_min_detections: int) -> None:
        self.all_students = sorted(all_students)
        self.min_detections = int(min_detections)
        self.review_min_detections = int(review_min_detections)
        self.counts: Dict[str, int] = defaultdict(int)
        self.scores: Dict[str, List[float]] = defaultdict(list)
        self.videos_seen: Dict[str, Set[str]] = defaultdict(set)
        self.cameras_seen: Dict[str, Set[str]] = defaultdict(set)
        self.first_seen: Dict[str, str] = {}
        self.last_seen: Dict[str, str] = {}

    def add_detection(self, roll_no: str, score: float, video_name: str, camera_id: str, time_text: str) -> None:
        self.counts[roll_no] += 1
        self.scores[roll_no].append(float(score))
        self.videos_seen[roll_no].add(video_name)
        self.cameras_seen[roll_no].add(camera_id)
        self.first_seen.setdefault(roll_no, time_text)
        self.last_seen[roll_no] = time_text

    def status_for(self, roll_no: str) -> str:
        count = self.counts.get(roll_no, 0)
        if count >= self.min_detections:
            return "Present"
        if count >= self.review_min_detections:
            return "Needs Review"
        return "Absent"

    def present_students(self) -> set[str]:
        return {roll for roll in self.all_students if self.status_for(roll) == "Present"}

    def review_students(self) -> set[str]:
        return {roll for roll in self.all_students if self.status_for(roll) == "Needs Review"}

    def to_dataframe(self, slot_dict: dict) -> pd.DataFrame:
        rows = []
        for roll in self.all_students:
            scores = self.scores.get(roll, [])
            status = self.status_for(roll)
            rows.append({
                **slot_dict,
                "Roll_Number": roll,
                "Detection_Count": self.counts.get(roll, 0),
                "Best_Score": round(max(scores), 4) if scores else "",
                "Average_Score": round(sum(scores) / len(scores), 4) if scores else "",
                "Videos_Seen": ", ".join(sorted(self.videos_seen.get(roll, []))),
                "Cameras_Seen": ", ".join(sorted(self.cameras_seen.get(roll, []))),
                "First_Seen_In_Video": self.first_seen.get(roll, ""),
                "Last_Seen_In_Video": self.last_seen.get(roll, ""),
                "Final_Status": status,
                "Present": "Yes" if status == "Present" else "No",
            })
        return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Timetable-based CCTV attendance using YuNet + SFace.")

    parser.add_argument("--timetable", required=True, help="Path to timetable CSV/XLSX")
    parser.add_argument("--slot-id", help="Timetable slot id, for example MON_P1")
    parser.add_argument("--day", help="Day name, for example Monday")
    parser.add_argument("--period", help="Period number, for example 1")
    parser.add_argument("--class-time", help="Time inside the class slot, for example 09:10")
    parser.add_argument("--list-slots", action="store_true", help="List timetable slots and exit")

    parser.add_argument("--video-dir", default=str(VIDEO_DIR), help="Folder containing CCTV videos for this class slot")
    parser.add_argument("--embeddings", default=str(EMBEDDINGS_PATH), help="student_embeddings.pkl path")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output attendance folder")
    parser.add_argument("--unknown-dir", default=str(UNKNOWN_DIR), help="Folder to save unknown/rejected faces")

    parser.add_argument("--det-score", type=float, default=DEFAULT_DETECTION_SCORE, help="YuNet detection threshold")
    parser.add_argument("--match-threshold", type=float, default=DEFAULT_MATCH_THRESHOLD, help="Cosine match threshold. Higher = stricter")
    parser.add_argument("--margin-threshold", type=float, default=DEFAULT_MARGIN_THRESHOLD, help="Best score must beat second score by this amount")
    parser.add_argument("--min-detections", type=int, default=DEFAULT_MIN_DETECTIONS_FOR_PRESENT, help="Minimum accepted detections to mark Present")
    parser.add_argument("--review-min-detections", type=int, default=0, help="Minimum accepted detections to mark Needs Review. Default: half of min-detections")
    parser.add_argument("--frame-skip", type=int, default=DEFAULT_FRAME_SKIP, help="Process every Nth frame")
    parser.add_argument("--max-width", type=int, default=DEFAULT_MAX_WIDTH, help="Resize video frames to this max width. 0 = original size")
    parser.add_argument("--aggregate", choices=["centroid", "max", "top3"], default="top3", help="How to compare against enrolled images")
    parser.add_argument("--save-unknown", action="store_true", help="Save rejected/unknown face crops")
    parser.add_argument("--display", action="store_true", help="Show live video window. Press q to stop")
    return parser.parse_args()


def seconds_to_text(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    m = int(seconds // 60)
    s = seconds - (m * 60)
    return f"{m:02d}:{s:05.2f}"


def draw_label(frame, box, text, color):
    x, y, w, h = box
    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    y_text = max(20, y1 - 8)
    cv2.putText(frame, text, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def infer_camera_id(video_path: Path, default_camera_ids: str) -> str:
    stem = video_path.stem.lower()
    parent = video_path.parent.name.lower()
    candidates = []
    for text in (stem, parent):
        for token in text.replace("-", "_").split("_"):
            if token.startswith("cam") or token.startswith("camera"):
                candidates.append(token)
    if candidates:
        return candidates[0]

    default_ids = [x.strip() for x in str(default_camera_ids).split(",") if x.strip()]
    if len(default_ids) == 1:
        return default_ids[0]
    return video_path.stem


def build_slot_prefix(slot) -> str:
    pieces = [slot.slot_id, slot.course_abbr, slot.day, f"P{slot.period}"]
    return "_".join(clean_filename(piece.replace("/", "-")) for piece in pieces if piece)


def save_slot_outputs(attendance_df: pd.DataFrame, log_df: pd.DataFrame, summary_df: pd.DataFrame, slot, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = build_slot_prefix(slot)

    attendance_csv = output_dir / f"attendance_{prefix}_{timestamp}.csv"
    log_csv = output_dir / f"detection_log_{prefix}_{timestamp}.csv"
    summary_csv = output_dir / f"slot_summary_{prefix}_{timestamp}.csv"
    excel_file = output_dir / f"attendance_{prefix}_{timestamp}.xlsx"

    attendance_df.to_csv(attendance_csv, index=False)
    log_df.to_csv(log_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Slot Summary", index=False)
        attendance_df.to_excel(writer, sheet_name="Attendance", index=False)
        log_df.to_excel(writer, sheet_name="Detection Log", index=False)

    return {
        "attendance_csv": attendance_csv,
        "log_csv": log_csv,
        "summary_csv": summary_csv,
        "excel_file": excel_file,
    }


def main() -> None:
    args = parse_args()

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

    review_min_detections = args.review_min_detections
    if review_min_detections <= 0:
        review_min_detections = max(2, args.min_detections // 2)

    attendance = SlotAttendanceBook(
        db.roll_numbers,
        min_detections=args.min_detections,
        review_min_detections=review_min_detections,
    )

    video_files = iter_files(video_dir, VIDEO_EXTENSIONS)
    if not video_files:
        raise SystemExit(f"No video files found in: {video_dir}")

    print("Timetable-Based YuNet + SFace Attendance Started ✅")
    print(f"Slot: {slot.slot_id} | {slot.day} P{slot.period} | {slot.start_time}-{slot.end_time}")
    print(f"Subject: {slot.course_abbr} - {slot.course_name}")
    print(f"Instructor: {slot.instructor}")
    print(f"Room: {slot.room} | Section: {slot.section}")
    print(f"Students enrolled: {db.student_count}")
    print(f"Embeddings: {len(db)}")
    print(f"Videos: {len(video_files)}")
    print(f"Match threshold: {args.match_threshold}")
    print(f"Margin threshold: {args.margin_threshold}")
    print(f"Min detections for Present: {args.min_detections}")
    print(f"Min detections for Needs Review: {review_min_detections}\n")

    slot_dict = slot.to_dict()
    detection_log = []
    unknown_saved = 0
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    stop_requested = False
    total_processed_frames = 0
    videos_processed = []

    for video_path in video_files:
        video_name = video_path.name
        camera_id = infer_camera_id(video_path, slot.camera_ids)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        print(f"Processing: {video_name} | Camera: {camera_id}")
        videos_processed.append(video_name)

        frame_index = 0
        processed_frames = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_index += 1
            if args.frame_skip > 1 and frame_index % args.frame_skip != 0:
                continue

            frame, _ = resize_keep_aspect(frame, args.max_width)
            processed_frames += 1
            total_processed_frames += 1
            time_sec = frame_index / fps if fps > 0 else 0.0
            time_text = seconds_to_text(time_sec)

            detections = engine.detect_and_extract(frame)

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
                    attendance.add_detection(match.roll_no, match.best_score, video_name, camera_id, time_text)
                    color = (0, 255, 0)
                    text = f"{match.roll_no} {match.best_score:.2f}"
                    count_for_roll = attendance.counts.get(match.roll_no, 0)
                else:
                    color = (0, 0, 255)
                    text = f"Unknown {match.best_score:.2f}"
                    count_for_roll = 0
                    if args.save_unknown:
                        crop = safe_crop(frame, box)
                        if crop is not None:
                            unknown_saved += 1
                            filename = f"unknown_{run_id}_{clean_filename(video_path.stem)}_f{frame_index}_d{face_idx}_{match.reason}.jpg"
                            cv2.imwrite(str(unknown_dir / filename), crop)

                detection_log.append({
                    **slot_dict,
                    "Video": video_name,
                    "Camera_ID": camera_id,
                    "Frame": frame_index,
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
                present_now = attendance.present_students()
                cv2.putText(frame, f"{slot.slot_id} | {slot.course_abbr} | {slot.start_time}-{slot.end_time}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
                cv2.putText(frame, f"Video: {video_name} | Camera: {camera_id}", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
                cv2.putText(frame, f"Present: {len(present_now)} / {db.student_count}", (20, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
                cv2.putText(frame, "Press q to stop", (20, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
                cv2.imshow("Timetable-Based CCTV Attendance", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_requested = True
                    break

        cap.release()
        print(f"  Frames processed: {processed_frames}")
        if stop_requested:
            break

    if args.display:
        cv2.destroyAllWindows()

    attendance_df = attendance.to_dataframe(slot_dict)
    log_df = pd.DataFrame(detection_log)

    accepted_count = int((log_df["Accepted"] == "Yes").sum()) if not log_df.empty else 0
    rejected_count = int((log_df["Accepted"] == "No").sum()) if not log_df.empty else 0
    total_detections = int(len(log_df))
    present_count = int((attendance_df["Final_Status"] == "Present").sum())
    review_count = int((attendance_df["Final_Status"] == "Needs Review").sum())
    absent_count = int((attendance_df["Final_Status"] == "Absent").sum())

    summary_df = pd.DataFrame([{
        **slot_dict,
        "Run_Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Video_Dir": str(video_dir),
        "Videos_Processed": ", ".join(videos_processed),
        "Total_Students": db.student_count,
        "Frames_Processed": total_processed_frames,
        "Total_Face_Detections": total_detections,
        "Accepted_Recognitions": accepted_count,
        "Rejected_Or_Unknown": rejected_count,
        "Students_With_At_Least_One_Detection": int((attendance_df["Detection_Count"] > 0).sum()),
        "Students_Present": present_count,
        "Students_Needs_Review": review_count,
        "Students_Absent": absent_count,
        "Unknown_Faces_Saved": unknown_saved if args.save_unknown else 0,
        "Match_Threshold": args.match_threshold,
        "Margin_Threshold": args.margin_threshold,
        "Min_Detections_Present": args.min_detections,
        "Min_Detections_Review": review_min_detections,
        "Frame_Skip": args.frame_skip,
        "Aggregate_Mode": args.aggregate,
    }])

    paths = save_slot_outputs(attendance_df, log_df, summary_df, slot, output_dir)

    print("\nTimetable attendance completed ✅")
    print(f"Subject: {slot.course_abbr} - {slot.course_name}")
    print(f"Slot: {slot.day} P{slot.period} | {slot.start_time}-{slot.end_time}")
    print(f"Present: {present_count} / {db.student_count}")
    print(f"Needs Review: {review_count}")
    print(f"Absent: {absent_count}")
    print(f"Unknown/rejected faces saved: {unknown_saved}" if args.save_unknown else "Unknown saving: off")
    print(f"Saved: {paths['attendance_csv']}")
    print(f"Saved: {paths['summary_csv']}")
    print(f"Saved: {paths['log_csv']}")
    print(f"Saved: {paths['excel_file']}")


if __name__ == "__main__":
    main()
