from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Iterable

import cv2
import numpy as np

from .utils import clamp_box, clean_filename


PERCENTILE_POINTS = (0, 10, 50, 90, 100)


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def percentile_fields(values: Iterable[Any], prefix: str, decimals: int = 4) -> dict[str, Any]:
    numbers = [finite_float(v) for v in values if v is not None and str(v) != ""]
    numbers = [v for v in numbers if math.isfinite(v)]
    keys = {
        f"{prefix}_Min": "",
        f"{prefix}_P10": "",
        f"{prefix}_Median": "",
        f"{prefix}_P90": "",
        f"{prefix}_Max": "",
    }
    if not numbers:
        return keys
    percentiles = np.percentile(np.asarray(numbers, dtype=np.float64), PERCENTILE_POINTS)
    labels = ["Min", "P10", "Median", "P90", "Max"]
    return {f"{prefix}_{label}": round(float(value), decimals) for label, value in zip(labels, percentiles)}


def landmark_points(face: np.ndarray) -> list[tuple[float, float]]:
    if face is None or len(face) < 14:
        return []
    pts = np.asarray(face[4:14], dtype=np.float32).reshape(5, 2)
    return [(float(x), float(y)) for x, y in pts]


def landmark_text(face: np.ndarray) -> str:
    points = landmark_points(face)
    return "; ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def face_geometry(face: np.ndarray, frame_shape: tuple[int, ...]) -> dict[str, Any]:
    ih, iw = frame_shape[:2]
    x, y, w, h = [finite_float(v) for v in face[:4]]
    x1, y1, x2, y2 = clamp_box(x, y, w, h, frame_shape)
    visible_w = max(0, x2 - x1)
    visible_h = max(0, y2 - y1)
    face_area = max(0.0, w) * max(0.0, h)
    visible_area = float(visible_w * visible_h)
    frame_area = float(max(iw * ih, 1))
    points = landmark_points(face)
    landmarks_inside = 0
    for px, py in points:
        if 0 <= px < iw and 0 <= py < ih:
            landmarks_inside += 1
    partially_outside = x < 0 or y < 0 or x + w > iw or y + h > ih
    return {
        "Bounding_Box_X": round(x, 2),
        "Bounding_Box_Y": round(y, 2),
        "Bounding_Box_W": round(w, 2),
        "Bounding_Box_H": round(h, 2),
        "Visible_Box_X1": x1,
        "Visible_Box_Y1": y1,
        "Visible_Box_X2": x2,
        "Visible_Box_Y2": y2,
        "Face_Width": round(max(0.0, w), 2),
        "Face_Height": round(max(0.0, h), 2),
        "Face_Area": round(face_area, 2),
        "Visible_Face_Area": round(visible_area, 2),
        "Face_Area_Ratio": round(face_area / frame_area, 8),
        "Visible_Face_Area_Ratio": round(visible_area / frame_area, 8),
        "Frame_Width": int(iw),
        "Frame_Height": int(ih),
        "Crop_Partially_Outside_Frame": "Yes" if partially_outside else "No",
        "Landmark_Count": len(points),
        "Landmarks_Inside_Frame": landmarks_inside,
        "Landmark_Valid": "Yes" if len(points) == 5 and landmarks_inside == 5 else "No",
        "Landmark_Coordinates": landmark_text(face),
        "Crop_Padding_Used": 0,
    }


def crop_quality_metrics(frame: np.ndarray, box: tuple[float, float, float, float]) -> dict[str, Any]:
    empty = {
        "Crop_Valid": "No",
        "Crop_Width": 0,
        "Crop_Height": 0,
        "Blur_Laplacian_Variance": "",
        "Brightness_Mean": "",
        "Contrast_StdDev": "",
        "Underexposed_Pct": "",
        "Overexposed_Pct": "",
    }
    if frame is None or frame.size == 0:
        return empty
    x1, y1, x2, y2 = clamp_box(*box, frame.shape)
    if x2 <= x1 or y2 <= y1:
        return empty
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return empty
    if crop.ndim == 2:
        gray = crop
    else:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = np.asarray(gray)
    blur = 0.0
    if gray.shape[0] >= 2 and gray.shape[1] >= 2:
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {
        "Crop_Valid": "Yes",
        "Crop_Width": int(gray.shape[1]),
        "Crop_Height": int(gray.shape[0]),
        "Blur_Laplacian_Variance": round(blur, 4),
        "Brightness_Mean": round(float(np.mean(gray)), 4),
        "Contrast_StdDev": round(float(np.std(gray)), 4),
        "Underexposed_Pct": round(float(np.mean(gray <= 20) * 100.0), 4),
        "Overexposed_Pct": round(float(np.mean(gray >= 235) * 100.0), 4),
    }


def diagnostic_quality_label(
    geometry: dict[str, Any],
    quality: dict[str, Any],
    detector_score: float,
    detector_reference: float = 0.90,
) -> tuple[str, str]:
    reasons: list[str] = []
    width = finite_float(geometry.get("Face_Width"))
    height = finite_float(geometry.get("Face_Height"))
    area_ratio = finite_float(geometry.get("Face_Area_Ratio"))
    blur = finite_float(quality.get("Blur_Laplacian_Variance"), default=-1.0)
    brightness = finite_float(quality.get("Brightness_Mean"), default=-1.0)
    contrast = finite_float(quality.get("Contrast_StdDev"), default=-1.0)

    if width < 32 or height < 32 or area_ratio < 0.0004:
        reasons.append("face_too_small_for_diagnostic_reference")
    if detector_score < detector_reference:
        reasons.append("low_detector_confidence")
    if geometry.get("Landmark_Valid") != "Yes":
        reasons.append("landmark_or_alignment_warning")
    if quality.get("Crop_Valid") != "Yes":
        reasons.append("invalid_crop")
    if blur >= 0 and blur < 25:
        reasons.append("low_blur_score")
    if brightness >= 0 and brightness < 35:
        reasons.append("underexposed_crop")
    if brightness > 220:
        reasons.append("overexposed_crop")
    if contrast >= 0 and contrast < 12:
        reasons.append("low_contrast_crop")
    if geometry.get("Crop_Partially_Outside_Frame") == "Yes":
        reasons.append("crop_partially_outside_frame")

    if not reasons:
        return "usable_reference", ""
    hard = {"face_too_small_for_diagnostic_reference", "invalid_crop", "landmark_or_alignment_warning"}
    label = "unusable_reference" if hard.intersection(reasons) else "marginal_reference"
    return label, "; ".join(reasons)


def diagnostic_rejection_reason(
    accepted: bool,
    matcher_reason: str,
    embedding_success: bool,
    best_score: float,
    margin: float,
    match_threshold: float,
    margin_threshold: float,
) -> str:
    if accepted:
        return "accepted"
    if not embedding_success:
        return "embedding_extraction_failure"
    if matcher_reason in {"empty_database", "no_candidate"}:
        return "no_candidate"
    if matcher_reason == "invalid_embedding":
        return "invalid_embedding"
    score_low = finite_float(best_score) < float(match_threshold)
    margin_low = finite_float(margin) < float(margin_threshold)
    if score_low and margin_low:
        return "score_and_margin_below_threshold"
    if score_low:
        return "score_below_threshold"
    if margin_low:
        return "margin_below_threshold"
    if matcher_reason == "margin_too_small":
        return "margin_below_threshold"
    if matcher_reason == "score_below_threshold":
        return "score_below_threshold"
    return matcher_reason or "other"


def should_log_detection(log_mode: str, accepted: bool, matcher_reason: str) -> bool:
    if log_mode == "full":
        return True
    if accepted:
        return True
    return log_mode == "compact" and matcher_reason in {"margin_too_small", "margin_below_threshold"}


def make_diagnostic_run_id(session_id: str = "", timestamp: datetime | None = None) -> str:
    stamp = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S_%f")
    prefix = clean_filename(session_id or "diagnostic")
    return f"{prefix}_{stamp}"


def checkpoint_window_warnings(
    mode: str,
    source_duration: float,
    selected_start: float,
    selected_end: float,
    requested_start: float,
    requested_end: float,
) -> tuple[bool, list[str]]:
    warnings: list[str] = []
    duration = finite_float(source_duration)
    if mode == "clip-folders":
        warnings.append("clip_folder_uses_video_relative_window")
    if duration > 0:
        if selected_start >= duration:
            warnings.append("selected_window_starts_after_source_duration")
        if selected_end > duration + 1e-6:
            warnings.append("selected_window_exceeds_source_duration")
        if selected_end < requested_end - 1e-6 and mode in {"class-time", "clip-folders"}:
            warnings.append("selected_window_clamped_to_source_duration")
        if requested_start >= duration and mode == "class-time":
            warnings.append("requested_class_window_starts_after_source_duration")
    if selected_end <= selected_start:
        warnings.append("invalid_window")
    return "invalid_window" not in warnings and "selected_window_starts_after_source_duration" not in warnings, warnings


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
