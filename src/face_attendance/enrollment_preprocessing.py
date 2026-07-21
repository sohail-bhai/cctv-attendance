from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .face_engine import FaceEngine
from .utils import safe_crop


@dataclass(frozen=True)
class EnrollmentBuildConfig:
    detection_score: float = 0.60
    det_max_width: int = 1280
    min_face_size: int = 50
    min_area_ratio: float = 0.003
    pad_percent: float = 0.25
    min_landmarks_inside: int = 4
    min_eye_distance: float = 8.0
    validation_mode: str = "padded_photo_landmark_validation"


@dataclass(frozen=True)
class EnrollmentExtraction:
    embedding: np.ndarray | None
    reason: str
    detection_count: int
    accepted_candidate_count: int
    detector_score: float
    rank_score: float
    image_width: int
    image_height: int
    raw_crop: np.ndarray | None = None
    aligned_face: np.ndarray | None = None

    @property
    def accepted(self) -> bool:
        return self.embedding is not None


def add_padding_for_detection(
    image: np.ndarray, pad_percent: float
) -> tuple[np.ndarray, tuple[int, int]]:
    """Pad tight enrollment photos before YuNet detection."""
    if image is None or image.size == 0 or pad_percent <= 0:
        return image, (0, 0)
    height, width = image.shape[:2]
    pad_x = int(round(width * pad_percent))
    pad_y = int(round(height * pad_percent))
    padded = cv2.copyMakeBorder(
        image,
        pad_y,
        pad_y,
        pad_x,
        pad_x,
        borderType=cv2.BORDER_REPLICATE,
    )
    return padded, (pad_x, pad_y)


def landmark_points(face: np.ndarray) -> np.ndarray:
    if len(face) < 14:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(face[4:14], dtype=np.float32).reshape(5, 2)


def face_quality_score(
    face: np.ndarray,
    image_shape: tuple[int, ...],
    min_face_size: int,
    min_area_ratio: float,
    min_landmarks_inside: int,
    min_eye_distance: float,
) -> tuple[bool, str, float]:
    """Apply the production enrollment box and landmark checks."""
    image_height, image_width = image_shape[:2]
    x, y, width, height = [float(value) for value in face[:4]]
    detector_score = float(face[14]) if len(face) > 14 else 0.0

    if width <= 0 or height <= 0:
        return False, "invalid_box", -999.0

    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(image_width), x + width), min(float(image_height), y + height)
    visible_width = max(0.0, x2 - x1)
    visible_height = max(0.0, y2 - y1)
    visible_ratio = (visible_width * visible_height) / max(width * height, 1.0)
    if visible_ratio < 0.60:
        return False, "box_mostly_outside_image", -999.0
    if min(visible_width, visible_height) < min_face_size:
        return False, "face_too_small", -999.0

    area_ratio = (visible_width * visible_height) / float(
        max(image_width * image_height, 1)
    )
    if area_ratio < min_area_ratio:
        return False, "face_area_too_small", -999.0

    aspect = visible_width / max(visible_height, 1.0)
    if aspect < 0.40 or aspect > 2.20:
        return False, "bad_aspect_ratio", -999.0

    points = landmark_points(face)
    if points.shape == (5, 2):
        inside = sum(
            1
            for point_x, point_y in points
            if 0 <= point_x < image_width and 0 <= point_y < image_height
        )
        if inside < min_landmarks_inside:
            return False, "landmarks_outside_image", -999.0
        eye_distance = float(np.linalg.norm(points[0] - points[1]))
        if eye_distance < min_eye_distance:
            return False, "eye_distance_too_small", -999.0
    else:
        eye_distance = 0.0

    center_x = x + width / 2.0
    center_y = y + height / 2.0
    center_distance = np.sqrt(
        ((center_x - image_width / 2) / max(image_width, 1)) ** 2
        + ((center_y - image_height / 2) / max(image_height, 1)) ** 2
    )
    center_bonus = max(0.0, 1.0 - center_distance * 1.8)
    size_bonus = min(area_ratio * 12.0, 1.5)
    landmark_bonus = min(eye_distance / 80.0, 1.0)
    rank_score = detector_score * 2.0 + size_bonus + center_bonus + landmark_bonus
    return True, "accepted", float(rank_score)


def expand_box(
    box: tuple[float, float, float, float],
    image_shape: tuple[int, ...],
    scale: float = 0.25,
) -> tuple[float, float, float, float]:
    del image_shape
    x, y, width, height = [float(value) for value in box]
    pad_x = width * scale
    pad_y = height * scale
    return x - pad_x, y - pad_y, width + 2 * pad_x, height + 2 * pad_y


def extract_enrollment_embedding(
    image_path: Path,
    engine: FaceEngine,
    config: EnrollmentBuildConfig,
    *,
    require_single_face: bool = False,
) -> EnrollmentExtraction:
    """Extract the best enrollment feature with the production preprocessing policy."""
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None or image.size == 0:
        return EnrollmentExtraction(None, "read_failed", 0, 0, 0.0, -999.0, 0, 0)

    detection_image, _ = add_padding_for_detection(image, config.pad_percent)
    faces = engine.detect_faces_scaled(detection_image, max_width=config.det_max_width)
    detection_count = int(len(faces))
    if detection_count == 0:
        return EnrollmentExtraction(
            None,
            "no_face",
            0,
            0,
            0.0,
            -999.0,
            int(image.shape[1]),
            int(image.shape[0]),
        )
    if require_single_face and detection_count != 1:
        return EnrollmentExtraction(
            None,
            "multiple_faces_not_unambiguous",
            detection_count,
            0,
            0.0,
            -999.0,
            int(image.shape[1]),
            int(image.shape[0]),
        )

    candidates: list[dict[str, Any]] = []
    for face in faces:
        valid, _reason, rank_score = face_quality_score(
            face,
            detection_image.shape,
            config.min_face_size,
            config.min_area_ratio,
            config.min_landmarks_inside,
            config.min_eye_distance,
        )
        if not valid:
            continue
        aligned = engine.align_face(detection_image, face)
        if aligned is None or aligned.size == 0:
            continue
        feature = engine.extract_feature(detection_image, face)
        if feature is None:
            continue
        vector = np.asarray(feature, dtype=np.float32).reshape(-1)
        if not vector.size or not np.isfinite(vector).all():
            continue
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            continue
        candidates.append(
            {
                "embedding": vector / norm,
                "detector_score": float(engine.face_score(face)),
                "rank_score": float(rank_score),
                "raw_crop": safe_crop(
                    detection_image,
                    expand_box(engine.face_box(face), detection_image.shape, scale=0.20),
                ),
                "aligned": aligned,
            }
        )

    if not candidates:
        return EnrollmentExtraction(
            None,
            "no_valid_face",
            detection_count,
            0,
            0.0,
            -999.0,
            int(image.shape[1]),
            int(image.shape[0]),
        )

    best = max(candidates, key=lambda candidate: candidate["rank_score"])
    return EnrollmentExtraction(
        best["embedding"],
        "accepted",
        detection_count,
        len(candidates),
        best["detector_score"],
        best["rank_score"],
        int(image.shape[1]),
        int(image.shape[0]),
        best["raw_crop"],
        best["aligned"],
    )
