from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .utils import normalize_embedding, resize_keep_aspect


class FaceEngine:
    """
    Wrapper around OpenCV YuNet + SFace.

    YuNet detects face bounding boxes and 5 landmarks.
    SFace aligns/crops the detected face and extracts a face embedding.
    """

    def __init__(
        self,
        yunet_model: Path,
        sface_model: Path,
        detection_score: float = 0.85,
        nms_threshold: float = 0.30,
        top_k: int = 5000,
        input_size: tuple[int, int] = (320, 320),
    ) -> None:
        self.yunet_model = Path(yunet_model)
        self.sface_model = Path(sface_model)
        self.detection_score = float(detection_score)
        self.nms_threshold = float(nms_threshold)
        self.top_k = int(top_k)
        self.input_size = tuple(input_size)

        if not self.yunet_model.exists():
            raise FileNotFoundError(f"YuNet model not found: {self.yunet_model}")
        if not self.sface_model.exists():
            raise FileNotFoundError(f"SFace model not found: {self.sface_model}")
        if not hasattr(cv2, "FaceDetectorYN_create") and not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError("Your OpenCV build does not include FaceDetectorYN. Install/upgrade opencv-contrib-python.")
        if not hasattr(cv2, "FaceRecognizerSF_create") and not hasattr(cv2, "FaceRecognizerSF"):
            raise RuntimeError("Your OpenCV build does not include FaceRecognizerSF. Install/upgrade opencv-contrib-python.")

        self.detector = self._create_detector(self.input_size)
        self.recognizer = self._create_recognizer()

    def _create_detector(self, input_size: tuple[int, int]):
        model = str(self.yunet_model)
        if hasattr(cv2, "FaceDetectorYN_create"):
            return cv2.FaceDetectorYN_create(
                model,
                "",
                input_size,
                self.detection_score,
                self.nms_threshold,
                self.top_k,
            )
        return cv2.FaceDetectorYN.create(
            model,
            "",
            input_size,
            self.detection_score,
            self.nms_threshold,
            self.top_k,
        )

    def _create_recognizer(self):
        model = str(self.sface_model)
        if hasattr(cv2, "FaceRecognizerSF_create"):
            return cv2.FaceRecognizerSF_create(model, "")
        return cv2.FaceRecognizerSF.create(model, "")

    @staticmethod
    def _scale_faces_to_original(faces: np.ndarray, scale: float) -> np.ndarray:
        """
        If detection was done on a resized image, map x/y/w/h and landmarks
        back to the original image coordinate system.
        """
        if faces.size == 0 or abs(scale - 1.0) < 1e-9:
            return faces
        mapped = faces.copy()
        # columns: x,y,w,h, l0x,l0y,l1x,l1y,l2x,l2y,l3x,l3y,l4x,l4y,score
        coord_cols = list(range(0, 14))
        mapped[:, coord_cols] = mapped[:, coord_cols] / scale
        return mapped

    def detect_faces(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            return np.empty((0, 15), dtype=np.float32)

        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        if faces is None:
            return np.empty((0, 15), dtype=np.float32)

        faces = np.asarray(faces, dtype=np.float32)
        if faces.ndim == 1:
            faces = faces.reshape(1, -1)

        if faces.shape[1] >= 15:
            faces = faces[np.argsort(-faces[:, 14])]
        return faces

    def detect_faces_scaled(self, frame: np.ndarray, max_width: int = 1280) -> np.ndarray:
        """
        Detect faces on a resized copy for better stability on very large phone photos,
        then map detections back to the original image.
        """
        if not max_width or max_width <= 0:
            return self.detect_faces(frame)
        resized, scale = resize_keep_aspect(frame, max_width)
        faces = self.detect_faces(resized)
        return self._scale_faces_to_original(faces, scale)

    @staticmethod
    def face_box(face: np.ndarray) -> tuple[float, float, float, float]:
        return float(face[0]), float(face[1]), float(face[2]), float(face[3])

    @staticmethod
    def face_score(face: np.ndarray) -> float:
        return float(face[14]) if len(face) > 14 else 0.0

    def align_face(self, frame: np.ndarray, face: np.ndarray) -> Optional[np.ndarray]:
        try:
            aligned = self.recognizer.alignCrop(frame, face)
            if aligned is None or aligned.size == 0:
                return None
            return aligned
        except cv2.error:
            return None

    def extract_feature(self, frame: np.ndarray, face: np.ndarray) -> Optional[np.ndarray]:
        aligned = self.align_face(frame, face)
        if aligned is None:
            return None
        try:
            feature = self.recognizer.feature(aligned)
            return normalize_embedding(feature)
        except cv2.error:
            return None

    def detect_and_extract(self, frame: np.ndarray, max_width: int = 0) -> List[dict]:
        results: List[dict] = []
        faces = self.detect_faces_scaled(frame, max_width) if max_width else self.detect_faces(frame)
        for face in faces:
            feature = self.extract_feature(frame, face)
            if feature is None:
                continue
            results.append({
                "face": face,
                "box": self.face_box(face),
                "score": self.face_score(face),
                "embedding": feature,
            })
        return results
