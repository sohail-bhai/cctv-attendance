from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def natural_key(text: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def iter_files(root: Path, extensions: Iterable[str]) -> List[Path]:
    extensions = {e.lower() for e in extensions}
    files: List[Path] = []
    if not root.exists():
        return files
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            files.append(path)
    return sorted(files, key=lambda p: natural_key(str(p)))


def resize_keep_aspect(frame: np.ndarray, max_width: int | None) -> tuple[np.ndarray, float]:
    if not max_width or max_width <= 0:
        return frame, 1.0
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame, 1.0
    scale = max_width / float(w)
    new_h = int(round(h * scale))
    resized = cv2.resize(frame, (max_width, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def clamp_box(x: float, y: float, w: float, h: float, image_shape) -> tuple[int, int, int, int]:
    ih, iw = image_shape[:2]
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(iw, int(round(x + w)))
    y2 = min(ih, int(round(y + h)))
    return x1, y1, x2, y2


def safe_crop(frame: np.ndarray, box) -> np.ndarray | None:
    x1, y1, x2, y2 = clamp_box(*box, frame.shape)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(emb))
    if norm <= 1e-12:
        return emb
    return emb / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = normalize_embedding(a)
    b = normalize_embedding(b)
    return float(np.dot(a, b))


def clean_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
