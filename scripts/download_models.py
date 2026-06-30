"""Download YuNet and SFace ONNX model files into the models/ folder.
Run from project root:
    python scripts/download_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODELS = {
    "face_detection_yunet_2023mar.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}


def download(name: str, url: str) -> None:
    target = MODELS_DIR / name
    if target.exists() and target.stat().st_size > 1024:
        print(f"OK already exists: {target}")
        return
    print(f"Downloading {name}...")
    try:
        urlretrieve(url, target)
    except Exception as exc:
        print(f"FAILED downloading {name}: {exc}")
        print("Download manually from OpenCV Zoo and place it inside models/.")
        raise
    print(f"Saved: {target} ({target.stat().st_size} bytes)")


def main() -> int:
    for name, url in MODELS.items():
        download(name, url)
    print("\nDone. Required model files are in models/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
