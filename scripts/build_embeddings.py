from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.face_attendance.config import (
    DATASET_DIR,
    EMBEDDING_SUMMARY_PATH,
    EMBEDDINGS_PATH,
    IMAGE_EXTENSIONS,
    SFACE_MODEL,
    YUNET_MODEL,
    DEFAULT_DETECTION_SCORE,
)
from src.face_attendance.embedding_db import StudentEmbeddingDB
from src.face_attendance.enrollment_preprocessing import (
    add_padding_for_detection,
    expand_box,
    face_quality_score,
)
from src.face_attendance.face_engine import FaceEngine
from src.face_attendance.utils import clean_filename, ensure_dirs, iter_files, safe_crop


def parse_args():
    parser = argparse.ArgumentParser(description="Build student face embeddings using YuNet + SFace.")
    parser.add_argument("--dataset", default=str(DATASET_DIR), help="Dataset folder. Default: dataset")
    parser.add_argument("--output", default=str(EMBEDDINGS_PATH), help="Output .pkl embedding database path")
    parser.add_argument("--summary", default=str(EMBEDDING_SUMMARY_PATH), help="Output CSV summary path")
    parser.add_argument("--det-score", type=float, default=0.60, help="YuNet minimum detection score for enrollment photos")
    parser.add_argument("--max-images", type=int, default=0, help="Max images per student. 0 = use all")
    parser.add_argument("--det-max-width", type=int, default=1280, help="Resize large photos to this width before detecting. 0 = original size")
    parser.add_argument("--min-face-size", type=int, default=50, help="Reject detected face boxes smaller than this many pixels")
    parser.add_argument("--min-area-ratio", type=float, default=0.003, help="Reject very tiny face boxes based on image area")
    parser.add_argument("--pad-percent", type=float, default=0.25, help="Add padding around enrollment photos before detection. Helps tightly cropped selfies. Use 0 to disable")
    parser.add_argument("--min-landmarks-inside", type=int, default=4, help="Minimum YuNet landmarks that must be inside the image")
    parser.add_argument("--min-eye-distance", type=float, default=8.0, help="Minimum pixel distance between detected eyes")
    parser.add_argument("--validate-crop-score", type=float, default=0.0, help="Legacy option. Keep 0. Second-pass crop validation is disabled by default because it rejects many good tight selfies")
    parser.add_argument("--debug-crops", action="store_true", help="Save accepted and rejected face crops to debug_faces")
    return parser.parse_args()


def save_debug_crop(base_dir: Path, roll_no: str, image_stem: str, crop, suffix: str) -> None:
    if crop is None or crop.size == 0:
        return
    out_dir = base_dir / clean_filename(roll_no)
    ensure_dirs(out_dir)
    cv2.imwrite(str(out_dir / f"{clean_filename(image_stem)}_{suffix}.jpg"), crop)


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset)
    output_path = Path(args.output)
    summary_path = Path(args.summary)
    accepted_debug_dir = ROOT / "debug_faces" / "enrolled_faces"
    rejected_debug_dir = ROOT / "debug_faces" / "rejected_faces"

    if not dataset_dir.exists():
        raise SystemExit(f"Dataset folder not found: {dataset_dir}")

    engine = FaceEngine(YUNET_MODEL, SFACE_MODEL, detection_score=args.det_score)

    records = []
    summary_rows = []

    student_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not student_dirs:
        raise SystemExit(f"No student folders found inside: {dataset_dir}")

    if args.debug_crops:
        ensure_dirs(accepted_debug_dir, rejected_debug_dir)

    print(f"Dataset: {dataset_dir}")
    print(f"Students found: {len(student_dirs)}")
    print(f"Detector score: {args.det_score}")
    print(f"Padding percent: {args.pad_percent}")
    print("Building embeddings with padded-photo detection + landmark validation...\n")

    for student_dir in student_dirs:
        roll_no = student_dir.name.strip()
        image_paths = iter_files(student_dir, IMAGE_EXTENSIONS)
        if args.max_images and args.max_images > 0:
            image_paths = image_paths[: args.max_images]

        used = 0
        no_face = 0
        multi_face = 0
        read_fail = 0
        rejected = 0

        print(f"Enrolling: {roll_no} ({len(image_paths)} images)")

        for image_path in image_paths:
            img = cv2.imread(str(image_path))
            if img is None:
                read_fail += 1
                print(f"  Could not read: {image_path.name}")
                continue

            det_img, _ = add_padding_for_detection(img, args.pad_percent)
            faces = engine.detect_faces_scaled(det_img, max_width=args.det_max_width)

            if faces.size == 0:
                no_face += 1
                print(f"  No face found: {image_path.name}")
                continue
            if len(faces) > 1:
                multi_face += 1

            candidates = []
            for face in faces:
                ok, reason, rank = face_quality_score(
                    face,
                    det_img.shape,
                    args.min_face_size,
                    args.min_area_ratio,
                    args.min_landmarks_inside,
                    args.min_eye_distance,
                )

                raw_crop = safe_crop(det_img, expand_box(engine.face_box(face), det_img.shape, scale=0.20))
                aligned = engine.align_face(det_img, face)

                if not ok:
                    rejected += 1
                    if args.debug_crops:
                        save_debug_crop(rejected_debug_dir, roll_no, image_path.stem, raw_crop, reason)
                    continue

                if aligned is None or aligned.size == 0:
                    rejected += 1
                    if args.debug_crops:
                        save_debug_crop(rejected_debug_dir, roll_no, image_path.stem, raw_crop, "align_failed")
                    continue

                feature = engine.extract_feature(det_img, face)
                if feature is None:
                    rejected += 1
                    if args.debug_crops:
                        save_debug_crop(rejected_debug_dir, roll_no, image_path.stem, raw_crop, "feature_failed")
                    continue

                candidates.append({
                    "face": face,
                    "box": engine.face_box(face),
                    "score": float(engine.face_score(face)),
                    "embedding": feature,
                    "rank": rank,
                    "raw_crop": raw_crop,
                    "aligned": aligned,
                })

            if not candidates:
                no_face += 1
                print(f"  No valid face after filtering: {image_path.name}")
                continue

            best = max(candidates, key=lambda d: d["rank"])
            try:
                rel_path = str(image_path.relative_to(ROOT))
            except ValueError:
                rel_path = str(image_path)

            records.append({
                "roll_no": roll_no,
                "image_path": rel_path,
                "detection_score": float(best["score"]),
                "embedding": best["embedding"],
            })
            used += 1

            if args.debug_crops:
                # raw = original detector box with some padding; aligned = actual SFace input
                save_debug_crop(accepted_debug_dir, roll_no, image_path.stem, best["raw_crop"], "accepted_raw")
                save_debug_crop(accepted_debug_dir, roll_no, image_path.stem, best["aligned"], "accepted_aligned")

        summary_rows.append({
            "Roll_Number": roll_no,
            "Images_Read": len(image_paths),
            "Faces_Used": used,
            "No_Valid_Face": no_face,
            "Multiple_Faces": multi_face,
            "Rejected_Crops": rejected,
            "Read_Fail": read_fail,
        })
        print(f"  Used: {used}, No valid face: {no_face}, Multiple faces: {multi_face}, Rejected crops: {rejected}, Read fail: {read_fail}\n")

    if not records:
        raise SystemExit("No face embeddings were created. Check dataset images and model files.")

    db = StudentEmbeddingDB(records, metadata={
        "dataset": str(dataset_dir),
        "yunet_model": YUNET_MODEL.name,
        "sface_model": SFACE_MODEL.name,
        "detection_score": args.det_score,
        "det_max_width": args.det_max_width,
        "min_face_size": args.min_face_size,
        "min_area_ratio": args.min_area_ratio,
        "pad_percent": args.pad_percent,
        "min_landmarks_inside": args.min_landmarks_inside,
        "min_eye_distance": args.min_eye_distance,
        "validation_mode": "padded_photo_landmark_validation",
    })
    db.save(output_path)

    summary_df = pd.DataFrame(summary_rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)

    print("Embedding build completed ✅")
    print(f"Students enrolled: {db.student_count}")
    print(f"Total embeddings: {len(db)}")
    print(f"Saved database: {output_path}")
    print(f"Saved summary: {summary_path}")
    if args.debug_crops:
        print(f"Accepted crops: {accepted_debug_dir}")
        print(f"Rejected crops: {rejected_debug_dir}")


if __name__ == "__main__":
    main()
