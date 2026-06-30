from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "dataset"
AUGMENTED_DATASET_DIR = PROJECT_ROOT / "augmented_dataset"
VIDEO_DIR = PROJECT_ROOT / "cctv_videos"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "attendance_output"
UNKNOWN_DIR = PROJECT_ROOT / "unknown_faces"
DEBUG_DIR = PROJECT_ROOT / "debug_faces"

YUNET_MODEL = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = MODEL_DIR / "face_recognition_sface_2021dec.onnx"
EMBEDDINGS_PATH = MODEL_DIR / "student_embeddings.pkl"
EMBEDDING_SUMMARY_PATH = MODEL_DIR / "embedding_summary.csv"

# Good starting values. Tune after one test video.
DEFAULT_DETECTION_SCORE = 0.85
DEFAULT_NMS_THRESHOLD = 0.30
DEFAULT_TOP_K = 5000
DEFAULT_MATCH_THRESHOLD = 0.38
DEFAULT_MARGIN_THRESHOLD = 0.03
DEFAULT_MIN_DETECTIONS_FOR_PRESENT = 5
DEFAULT_FRAME_SKIP = 3
DEFAULT_MAX_WIDTH = 960

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
