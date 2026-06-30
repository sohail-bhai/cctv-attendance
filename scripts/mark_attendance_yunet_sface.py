from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.face_attendance.attendance import AttendanceBook, save_outputs
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
from src.face_attendance.utils import clean_filename, ensure_dirs, iter_files, resize_keep_aspect, safe_crop


def parse_args():
    parser = argparse.ArgumentParser(description="Mark attendance from CCTV videos using YuNet + SFace.")
    parser.add_argument("--video-dir", default=str(VIDEO_DIR), help="Folder containing CCTV videos")
    parser.add_argument("--embeddings", default=str(EMBEDDINGS_PATH), help="student_embeddings.pkl path")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output attendance folder")
    parser.add_argument("--unknown-dir", default=str(UNKNOWN_DIR), help="Folder to save unknown/rejected faces")
    parser.add_argument("--det-score", type=float, default=DEFAULT_DETECTION_SCORE, help="YuNet detection threshold")
    parser.add_argument("--match-threshold", type=float, default=DEFAULT_MATCH_THRESHOLD, help="Cosine match threshold. Higher = stricter")
    parser.add_argument("--margin-threshold", type=float, default=DEFAULT_MARGIN_THRESHOLD, help="Best score must beat second score by this amount")
    parser.add_argument("--min-detections", type=int, default=DEFAULT_MIN_DETECTIONS_FOR_PRESENT, help="Minimum accepted detections to mark Present")
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


def main() -> None:
    args = parse_args()
    video_dir = Path(args.video_dir)
    embeddings_path = Path(args.embeddings)
    output_dir = Path(args.output_dir)
    unknown_dir = Path(args.unknown_dir)

    if not embeddings_path.exists():
        raise SystemExit(f"Embedding database not found: {embeddings_path}\nRun: python scripts/build_embeddings.py")
    if not video_dir.exists():
        raise SystemExit(f"Video folder not found: {video_dir}")

    ensure_dirs(output_dir, unknown_dir)

    db = StudentEmbeddingDB.load(embeddings_path)
    engine = FaceEngine(
        YUNET_MODEL,
        SFACE_MODEL,
        detection_score=args.det_score,
        nms_threshold=DEFAULT_NMS_THRESHOLD,
        top_k=DEFAULT_TOP_K,
    )
    attendance = AttendanceBook(db.roll_numbers, min_detections=args.min_detections)

    video_files = iter_files(video_dir, VIDEO_EXTENSIONS)
    if not video_files:
        raise SystemExit(f"No video files found in: {video_dir}")

    print("YuNet + SFace Attendance Started ✅")
    print(f"Students enrolled: {db.student_count}")
    print(f"Embeddings: {len(db)}")
    print(f"Videos: {len(video_files)}")
    print(f"Match threshold: {args.match_threshold}")
    print(f"Margin threshold: {args.margin_threshold}")
    print(f"Min detections for Present: {args.min_detections}\n")

    detection_log = []
    unknown_saved = 0
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    stop_requested = False

    for video_path in video_files:
        video_name = video_path.name
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        print(f"Processing: {video_name}")

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
                    attendance.add_detection(match.roll_no, match.best_score, video_name, time_text)
                    color = (0, 255, 0)
                    text = f"{match.roll_no} {match.best_score:.2f}"
                else:
                    color = (0, 0, 255)
                    text = f"Unknown {match.best_score:.2f}"
                    if args.save_unknown:
                        crop = safe_crop(frame, box)
                        if crop is not None:
                            unknown_saved += 1
                            filename = f"unknown_{run_id}_{clean_filename(video_path.stem)}_f{frame_index}_d{face_idx}_{match.reason}.jpg"
                            cv2.imwrite(str(unknown_dir / filename), crop)

                detection_log.append({
                    "Video": video_name,
                    "Frame": frame_index,
                    "Time": time_text,
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
                    "Detection_Count_For_Roll": attendance.counts.get(match.roll_no, 0) if match.roll_no else 0,
                })

                if args.display:
                    draw_label(frame, box, text, color)

            if args.display:
                present_now = attendance.present_students()
                cv2.putText(frame, f"Video: {video_name}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
                cv2.putText(frame, f"Present: {len(present_now)} / {db.student_count}", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
                cv2.putText(frame, "Press q to stop", (20, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
                cv2.imshow("YuNet + SFace CCTV Attendance", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_requested = True
                    break

        cap.release()
        print(f"  Frames processed: {processed_frames}")
        if stop_requested:
            break

    if args.display:
        cv2.destroyAllWindows()

    attendance_df = attendance.to_dataframe()
    log_df = pd.DataFrame(detection_log)
    paths = save_outputs(attendance_df, log_df, output_dir)

    present = attendance.present_students()
    print("\nAttendance completed ✅")
    print(f"Present students: {len(present)} / {db.student_count}")
    print(f"Unknown/rejected faces saved: {unknown_saved}" if args.save_unknown else "Unknown saving: off")
    print(f"Saved: {paths['attendance_csv']}")
    print(f"Saved: {paths['log_csv']}")
    print(f"Saved: {paths['excel_file']}")


if __name__ == "__main__":
    main()
