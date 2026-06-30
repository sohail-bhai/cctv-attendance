from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.face_attendance.config import EMBEDDINGS_PATH, SFACE_MODEL, YUNET_MODEL


def exists(path):
    return "✅" if path.exists() else "❌"


def main():
    print("OpenCV version:", cv2.__version__)
    print("FaceDetectorYN available:", hasattr(cv2, "FaceDetectorYN_create") or hasattr(cv2, "FaceDetectorYN"))
    print("FaceRecognizerSF available:", hasattr(cv2, "FaceRecognizerSF_create") or hasattr(cv2, "FaceRecognizerSF"))
    print(f"{exists(YUNET_MODEL)} YuNet model: {YUNET_MODEL}")
    print(f"{exists(SFACE_MODEL)} SFace model: {SFACE_MODEL}")
    print(f"{exists(EMBEDDINGS_PATH)} Embedding database: {EMBEDDINGS_PATH}")

    if not (hasattr(cv2, "FaceDetectorYN_create") or hasattr(cv2, "FaceDetectorYN")):
        print("\nInstall/upgrade OpenCV contrib package:")
        print("pip uninstall opencv-python opencv-contrib-python -y")
        print("pip install opencv-contrib-python")


if __name__ == "__main__":
    main()
