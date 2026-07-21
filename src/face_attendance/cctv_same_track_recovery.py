from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np
import pandas as pd

from .config import DEFAULT_DETECTION_SCORE
from .embedding_version_family import (
    EXPECTED_PRODUCTION_EMBEDDINGS_SHA256,
    EXPECTED_PRODUCTION_SUMMARY_SHA256,
    EmbeddingFamilyError,
    verify_embedding_family,
)
from .face_engine import FaceEngine
from .utils import cosine_similarity, normalize_embedding
from .zones import (
    CameraZone,
    load_camera_zone_config,
    map_zone_face_to_original,
    upscale_zone_crop,
    zone_pixel_bounds,
)

RECOVERY_SCHEMA_VERSION = 1
RECOVERY_POLICY_VERSION = "phase-1.2i-c-same-track-v1"
DEFAULT_TARGET_SESSION = "2026-06-30__B51__P2__CVO"
DEFAULT_MAX_OBSERVATIONS_PER_TRACK = 5
DEFAULT_MAX_EMBEDDINGS_PER_TRACK = 2
DEFAULT_MIN_MATCH_IOU = 0.35
DEFAULT_MIN_RECOVERED_SIMILARITY = 0.25


class SameTrackRecoveryError(RuntimeError):
    pass


@dataclass(frozen=True)
class SameTrackRecoveryConfig:
    target_session: str = DEFAULT_TARGET_SESSION
    max_observations_per_track: int = DEFAULT_MAX_OBSERVATIONS_PER_TRACK
    max_embeddings_per_track: int = DEFAULT_MAX_EMBEDDINGS_PER_TRACK
    min_match_iou: float = DEFAULT_MIN_MATCH_IOU
    min_recovered_similarity: float = DEFAULT_MIN_RECOVERED_SIMILARITY
    detector_score: float = DEFAULT_DETECTION_SCORE
    audit_crop_padding: float = 0.35

    def validate(self) -> None:
        if not self.target_session.strip():
            raise SameTrackRecoveryError("Target session must not be blank")
        if self.max_observations_per_track < 2:
            raise SameTrackRecoveryError("At least two observations per track are required")
        if self.max_embeddings_per_track < 1:
            raise SameTrackRecoveryError("At least one recovered embedding per track is required")
        if self.max_embeddings_per_track > self.max_observations_per_track:
            raise SameTrackRecoveryError(
                "Recovered embedding cap cannot exceed the observation-attempt cap"
            )
        if not 0.0 < self.min_match_iou <= 1.0:
            raise SameTrackRecoveryError("min_match_iou must be in (0, 1]")
        if not -1.0 <= self.min_recovered_similarity <= 1.0:
            raise SameTrackRecoveryError(
                "min_recovered_similarity must be in [-1, 1]"
            )
        if not 0.0 < self.detector_score <= 1.0:
            raise SameTrackRecoveryError("detector_score must be in (0, 1]")
        if not 0.0 <= self.audit_crop_padding <= 2.0:
            raise SameTrackRecoveryError("audit_crop_padding must be in [0, 2]")


@dataclass(frozen=True)
class ObservationRecovery:
    item_id: str
    track_id: str
    observation_id: str
    candidate_roll: str
    source_session: str
    checkpoint: str
    camera: str
    video_path: Path
    video_sha256: str
    frame_index: int
    zone_profile: str
    zone_id: str
    zone_upscale: float
    recorded_bbox: tuple[float, float, float, float]
    matched_bbox: tuple[float, float, float, float]
    matched_iou: float
    detector_score: float
    quality_weight: float
    face_width: float
    face_height: float
    blur_variance: float
    embedding: np.ndarray
    audit_crop: np.ndarray


@dataclass(frozen=True)
class RecoveryRunResult:
    output_dir: Path
    recovery_id: str
    status: str
    manifest: dict[str, Any]
    reused: bool = False


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _yes(value: Any) -> bool:
    return _text(value).lower() in {"1", "true", "yes", "y", "on"}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _stable_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _parse_bbox(value: Any) -> tuple[float, float, float, float]:
    parts = [_float(part, math.nan) for part in _text(value).split(",")]
    if len(parts) != 4 or not all(math.isfinite(part) for part in parts):
        raise SameTrackRecoveryError(f"Invalid recorded bbox: {value}")
    x, y, width, height = parts
    if width <= 0 or height <= 0:
        raise SameTrackRecoveryError(f"Invalid recorded bbox dimensions: {value}")
    return x, y, width, height


def _bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    lx2, ly2 = lx + lw, ly + lh
    rx2, ry2 = rx + rw, ry + rh
    ix1, iy1 = max(lx, rx), max(ly, ry)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(lw * lh + rw * rh - intersection, 1e-9)
    return float(intersection / union)


def _crop_with_padding(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    padding: float,
) -> np.ndarray:
    height, width = frame.shape[:2]
    x, y, box_width, box_height = bbox
    pad_x = box_width * padding
    pad_y = box_height * padding
    x1 = max(0, int(math.floor(x - pad_x)))
    y1 = max(0, int(math.floor(y - pad_y)))
    x2 = min(width, int(math.ceil(x + box_width + pad_x)))
    y2 = min(height, int(math.ceil(y + box_height + pad_y)))
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=np.uint8)
    return frame[y1:y2, x1:x2].copy()


def _find_file(root: Path, pattern: str) -> Path:
    matches = sorted(Path(root).glob(pattern))
    if len(matches) != 1:
        raise SameTrackRecoveryError(
            f"Expected exactly one {pattern} under {root}, found {len(matches)}"
        )
    return matches[0]


def _load_family_unusable_rows(
    family_dir: Path, target_session: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    family_dir = Path(family_dir).resolve()
    manifest_path = family_dir / "family_manifest.json"
    usability_path = family_dir / "cctv_embedding_usability.csv"
    if not manifest_path.is_file() or not usability_path.is_file():
        raise SameTrackRecoveryError(
            f"Completed family is missing required recovery inputs: {family_dir}"
        )
    family_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(usability_path, dtype=str).fillna("")
    required_columns = {
        "Item_ID",
        "Source_Session",
        "Track_ID",
        "Observation_ID",
        "Candidate_Roll",
        "Human_Approval_Preserved",
        "Embedding_Usable",
        "Embedding_Status",
        "Manual_Rereview_Required",
        "Quality_Gate_Relaxed",
    }
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise SameTrackRecoveryError(
            "CCTV usability audit is missing columns: " + ", ".join(missing)
        )
    rows = []
    for row in frame.to_dict("records"):
        if _text(row["Source_Session"]) != target_session:
            continue
        if _yes(row["Embedding_Usable"]):
            continue
        if not _yes(row["Human_Approval_Preserved"]):
            raise SameTrackRecoveryError(
                f"Unusable source lost human-approval provenance: {row['Item_ID']}"
            )
        if _yes(row["Manual_Rereview_Required"]):
            raise SameTrackRecoveryError(
                f"Recovery cannot bypass a manual-rereview requirement: {row['Item_ID']}"
            )
        if _yes(row["Quality_Gate_Relaxed"]):
            raise SameTrackRecoveryError(
                f"Recovery source already used a relaxed quality gate: {row['Item_ID']}"
            )
        rows.append(row)
    if not rows:
        raise SameTrackRecoveryError(
            f"No approved-but-unusable CCTV sources found for {target_session}"
        )
    if len({row["Item_ID"] for row in rows}) != len(rows):
        raise SameTrackRecoveryError("Duplicate Item_ID values in CCTV usability audit")
    return family_manifest, sorted(rows, key=lambda row: row["Item_ID"])


def _load_diagnostic_inputs(
    diagnostic_run_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], Path, Path, Path]:
    diagnostic_run_dir = Path(diagnostic_run_dir).resolve()
    run_id = diagnostic_run_dir.name
    observations_path = diagnostic_run_dir / f"tracklet_observations_{run_id}.csv"
    diagnostics_path = diagnostic_run_dir / f"tracklet_diagnostics_{run_id}.csv"
    summary_path = diagnostic_run_dir / f"diagnostic_summary_{run_id}.json"
    for path in (observations_path, diagnostics_path, summary_path):
        if not path.is_file():
            raise SameTrackRecoveryError(f"Required diagnostic artifact missing: {path}")
    observations = pd.read_csv(observations_path, dtype=str).fillna("")
    diagnostics = pd.read_csv(diagnostics_path, dtype=str).fillna("")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return observations, diagnostics, summary, observations_path, diagnostics_path, summary_path


def _video_map(summary: dict[str, Any]) -> dict[tuple[str, str, str], Path]:
    mapping: dict[tuple[str, str, str], Path] = {}
    for row in summary.get("checkpoint_timing_map", []):
        checkpoint = _text(row.get("Checkpoint_ID"))
        camera = _text(row.get("Camera_ID"))
        filename = _text(row.get("Source_File_Name"))
        source_text = _text(row.get("Source_File"))
        if not checkpoint or not camera or not filename or not source_text:
            continue
        source = Path(source_text)
        key = checkpoint, camera, filename.lower()
        existing = mapping.get(key)
        if existing is not None and existing != source:
            raise SameTrackRecoveryError(
                f"Ambiguous video mapping for {checkpoint}/{camera}/{filename}"
            )
        mapping[key] = source
    if not mapping:
        raise SameTrackRecoveryError("Diagnostic summary contains no checkpoint video map")
    return mapping


def _eligible_observations(
    item: dict[str, Any],
    observations: pd.DataFrame,
    diagnostics: pd.DataFrame,
    config: SameTrackRecoveryConfig,
) -> list[dict[str, Any]]:
    track_id = _text(item["Track_ID"])
    track_rows = diagnostics[diagnostics["Tracklet_ID"].eq(track_id)]
    if len(track_rows) != 1:
        raise SameTrackRecoveryError(
            f"Expected exactly one diagnostic row for {track_id}, found {len(track_rows)}"
        )
    track = track_rows.iloc[0].to_dict()
    if not _yes(track.get("Tracklet_Eligible")):
        raise SameTrackRecoveryError(
            f"Human-approved recovery track is not technically eligible: {track_id}"
        )
    if _int(track.get("Inconsistent_Embedding_Count")) != 0:
        raise SameTrackRecoveryError(
            f"Recovery refuses a track with inconsistent selected embeddings: {track_id}"
        )
    if _int(track.get("Consistent_Embedding_Count")) < 2:
        raise SameTrackRecoveryError(
            f"Recovery track has fewer than two consistent embeddings: {track_id}"
        )

    rows = observations[observations["Tracklet_ID"].eq(track_id)].to_dict("records")
    if not rows:
        raise SameTrackRecoveryError(f"No observations found for {track_id}")
    source_session = _text(item["Source_Session"])
    valid: list[dict[str, Any]] = []
    for row in rows:
        if _text(row.get("Session_ID")) != source_session:
            continue
        if not _yes(row.get("Selected_For_Aggregation")):
            continue
        if not _yes(row.get("Embedding_Consistent")):
            continue
        if not _yes(row.get("Embedding_Extraction_Success")):
            continue
        if not _yes(row.get("Landmark_Valid")):
            continue
        if _text(row.get("Detection_Source")) != "zone":
            continue
        if not _text(row.get("Zone_ID")):
            continue
        if _int(row.get("Embedding_Dimension")) <= 0:
            continue
        valid.append(row)
    if len(valid) < 2:
        raise SameTrackRecoveryError(
            f"Recovery track has fewer than two eligible zone observations: {track_id}"
        )
    valid.sort(
        key=lambda row: (
            -_float(row.get("Quality_Weight")),
            -(_float(row.get("Face_Width")) * _float(row.get("Face_Height"))),
            -_float(row.get("Detector_Score")),
            -_float(row.get("Blur_Laplacian_Variance")),
            _text(row.get("Observation_ID")),
        )
    )
    return valid[: config.max_observations_per_track]


def _read_frame_1_based(video_path: Path, frame_index: int) -> np.ndarray:
    if frame_index < 1:
        raise SameTrackRecoveryError(f"Frame index must be 1-based and positive: {frame_index}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SameTrackRecoveryError(f"Unable to open source video: {video_path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index - 1)
        ok, frame = capture.read()
        if ok and frame is not None and frame.size:
            return frame
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        current = 0
        while current < frame_index:
            ok, frame = capture.read()
            current += 1
            if not ok:
                break
        if not ok or frame is None or frame.size == 0 or current != frame_index:
            raise SameTrackRecoveryError(
                f"Unable to decode frame {frame_index} from {video_path}"
            )
        return frame
    finally:
        capture.release()


def _zone_by_id(profile: Any, zone_id: str) -> CameraZone:
    matches = [zone for zone in profile.enabled_zones() if zone.zone_id == zone_id]
    if len(matches) != 1:
        raise SameTrackRecoveryError(
            f"Expected one enabled zone {profile.profile_id}.{zone_id}, found {len(matches)}"
        )
    return matches[0]


def _recover_observation(
    *,
    item: dict[str, Any],
    observation: dict[str, Any],
    video_path: Path,
    video_sha256: str,
    zone_config: Any,
    engine: FaceEngine,
    config: SameTrackRecoveryConfig,
) -> ObservationRecovery:
    frame_index = _int(observation.get("Frame"))
    frame = _read_frame_1_based(video_path, frame_index)
    camera = _text(observation.get("Camera_ID"))
    video_name = _text(observation.get("Video"))
    profile = zone_config.resolve_profile(camera, video_name)
    if profile is None:
        raise SameTrackRecoveryError(
            f"No camera-zone profile for {camera}/{video_name}"
        )
    zone_id = _text(observation.get("Zone_ID"))
    zone = _zone_by_id(profile, zone_id)
    bounds = zone_pixel_bounds(zone, frame.shape)
    x1, y1, x2, y2 = bounds
    zone_crop = frame[y1:y2, x1:x2]
    if zone_crop.size == 0:
        raise SameTrackRecoveryError(
            f"Empty zone crop for {_text(observation.get('Observation_ID'))}"
        )
    zone_input = upscale_zone_crop(zone_crop, zone.upscale)
    faces = engine.detect_faces(zone_input)
    recorded_bbox = _parse_bbox(observation.get("BBox_Original_Coordinates"))
    candidates: list[tuple[float, float, np.ndarray, tuple[float, float, float, float]]] = []
    for face in faces:
        mapped = map_zone_face_to_original(face, bounds, zone.upscale)
        mapped_bbox = tuple(float(value) for value in mapped[:4])
        iou = _bbox_iou(recorded_bbox, mapped_bbox)
        score = float(engine.face_score(face))
        candidates.append((iou, score, face, mapped_bbox))
    if not candidates:
        raise SameTrackRecoveryError(
            f"No YuNet face reproduced for {_text(observation.get('Observation_ID'))}"
        )
    candidates.sort(key=lambda item: (-item[0], -item[1]))
    matched_iou, detector_score, matched_face, matched_bbox = candidates[0]
    if matched_iou < config.min_match_iou:
        raise SameTrackRecoveryError(
            f"Recorded face could not be reproduced safely for "
            f"{_text(observation.get('Observation_ID'))}: IoU={matched_iou:.4f}"
        )
    feature = engine.extract_feature(zone_input, matched_face)
    if feature is None:
        raise SameTrackRecoveryError(
            f"SFace extraction failed for {_text(observation.get('Observation_ID'))}"
        )
    embedding = normalize_embedding(np.asarray(feature, dtype=np.float32).reshape(-1))
    if embedding.size != 128 or not np.isfinite(embedding).all():
        raise SameTrackRecoveryError(
            f"Invalid recovered embedding for {_text(observation.get('Observation_ID'))}"
        )
    audit_crop = _crop_with_padding(frame, matched_bbox, config.audit_crop_padding)
    if audit_crop.size == 0:
        raise SameTrackRecoveryError(
            f"Audit crop failed for {_text(observation.get('Observation_ID'))}"
        )
    return ObservationRecovery(
        item_id=_text(item["Item_ID"]),
        track_id=_text(item["Track_ID"]),
        observation_id=_text(observation.get("Observation_ID")),
        candidate_roll=_text(item["Candidate_Roll"]),
        source_session=_text(item["Source_Session"]),
        checkpoint=_text(observation.get("Checkpoint_ID")),
        camera=camera,
        video_path=video_path,
        video_sha256=video_sha256,
        frame_index=frame_index,
        zone_profile=profile.profile_id,
        zone_id=zone_id,
        zone_upscale=float(zone.upscale),
        recorded_bbox=recorded_bbox,
        matched_bbox=matched_bbox,
        matched_iou=float(matched_iou),
        detector_score=float(detector_score),
        quality_weight=_float(observation.get("Quality_Weight")),
        face_width=_float(observation.get("Face_Width")),
        face_height=_float(observation.get("Face_Height")),
        blur_variance=_float(observation.get("Blur_Laplacian_Variance")),
        embedding=embedding.astype(np.float32),
        audit_crop=audit_crop,
    )


def _select_consistent_recoveries(
    recoveries: list[ObservationRecovery],
    config: SameTrackRecoveryConfig,
) -> tuple[list[ObservationRecovery], dict[str, Any]]:
    if len(recoveries) < 2:
        return [], {
            "status": "insufficient_reproduced_observations",
            "reproduced": len(recoveries),
            "minimum_required": 2,
        }
    vectors = [normalize_embedding(item.embedding) for item in recoveries]
    similarities = np.asarray(
        [
            [cosine_similarity(left, right) for right in vectors]
            for left in vectors
        ],
        dtype=np.float32,
    )
    mean_similarity = similarities.mean(axis=1)
    medoid_index = sorted(
        range(len(recoveries)),
        key=lambda index: (
            -float(mean_similarity[index]),
            -recoveries[index].quality_weight,
            recoveries[index].observation_id,
        ),
    )[0]
    medoid = recoveries[medoid_index]
    consistent_indices = [
        index
        for index in range(len(recoveries))
        if index == medoid_index
        or float(similarities[medoid_index, index])
        >= config.min_recovered_similarity
    ]
    consistent = [recoveries[index] for index in consistent_indices]
    if len(consistent) < 2:
        return [], {
            "status": "insufficient_consistent_reproduced_observations",
            "reproduced": len(recoveries),
            "consistent": len(consistent),
            "medoid_observation_id": medoid.observation_id,
            "minimum_similarity": config.min_recovered_similarity,
        }
    consistent.sort(
        key=lambda item: (
            item.observation_id != medoid.observation_id,
            -item.quality_weight,
            -item.matched_iou,
            item.observation_id,
        )
    )
    selected = consistent[: config.max_embeddings_per_track]
    pairwise = [
        cosine_similarity(selected[left].embedding, selected[right].embedding)
        for left in range(len(selected))
        for right in range(left + 1, len(selected))
    ]
    return selected, {
        "status": "recovered",
        "reproduced": len(recoveries),
        "consistent": len(consistent),
        "selected": len(selected),
        "medoid_observation_id": medoid.observation_id,
        "selected_observation_ids": [item.observation_id for item in selected],
        "selected_pairwise_min": float(min(pairwise)) if pairwise else None,
        "selected_pairwise_median": float(np.median(pairwise)) if pairwise else None,
    }


def _output_manifest(output_dir: Path) -> dict[str, Any]:
    files = []
    for path in sorted(Path(output_dir).rglob("*")):
        if not path.is_file() or path.name == "output_manifest.json":
            continue
        files.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "files": files,
        "file_count": len(files),
    }


def verify_same_track_recovery(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    manifest_path = output_dir / "recovery_manifest.json"
    output_manifest_path = output_dir / "output_manifest.json"
    embeddings_path = output_dir / "recovered_embeddings.npz"
    for path in (manifest_path, output_manifest_path, embeddings_path):
        if not path.is_file():
            raise SameTrackRecoveryError(f"Recovery output missing: {path}")
    output_manifest = json.loads(output_manifest_path.read_text(encoding="utf-8"))
    for entry in output_manifest.get("files", []):
        path = output_dir / _text(entry.get("path"))
        if not path.is_file():
            raise SameTrackRecoveryError(f"Recovery manifest file missing: {path}")
        if _sha256_file(path) != _text(entry.get("sha256")):
            raise SameTrackRecoveryError(f"Recovery output hash mismatch: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if _text(manifest.get("recovered_embeddings_sha256")) != _sha256_file(
        embeddings_path
    ):
        raise SameTrackRecoveryError("Recovered embedding bundle hash mismatch")
    with np.load(embeddings_path, allow_pickle=False) as payload:
        keys = sorted(payload.files)
        expected = sorted(
            _text(row.get("embedding_key"))
            for row in manifest.get("selected_recovered_embeddings", [])
        )
        if keys != expected:
            raise SameTrackRecoveryError(
                "Recovered embedding keys do not match the recovery manifest"
            )
        for key in keys:
            vector = np.asarray(payload[key], dtype=np.float32).reshape(-1)
            if vector.size != 128 or not np.isfinite(vector).all():
                raise SameTrackRecoveryError(f"Invalid recovered vector: {key}")
    return manifest


def recover_same_track_cctv_evidence(
    *,
    repo_root: Path,
    family_dir: Path,
    diagnostic_run_dir: Path,
    camera_zones_path: Path,
    production_embeddings_path: Path,
    production_summary_path: Path,
    yunet_model: Path,
    sface_model: Path,
    output_root: Path,
    config: SameTrackRecoveryConfig = SameTrackRecoveryConfig(),
    expected_production_embeddings_sha256: str = EXPECTED_PRODUCTION_EMBEDDINGS_SHA256,
    expected_production_summary_sha256: str = EXPECTED_PRODUCTION_SUMMARY_SHA256,
    engine_factory: Callable[[Path, Path, float], FaceEngine] | None = None,
    progress: Callable[[str], None] | None = print,
) -> RecoveryRunResult:
    config.validate()
    started = time.perf_counter()
    repo_root = Path(repo_root).resolve()
    family_dir = Path(family_dir).resolve()
    diagnostic_run_dir = Path(diagnostic_run_dir).resolve()
    camera_zones_path = Path(camera_zones_path).resolve()
    production_embeddings_path = Path(production_embeddings_path).resolve()
    production_summary_path = Path(production_summary_path).resolve()
    yunet_model = Path(yunet_model).resolve()
    sface_model = Path(sface_model).resolve()
    output_root = Path(output_root).resolve()

    for path in (
        family_dir,
        diagnostic_run_dir,
        camera_zones_path,
        production_embeddings_path,
        production_summary_path,
        yunet_model,
        sface_model,
    ):
        if not path.exists():
            raise SameTrackRecoveryError(f"Required recovery input not found: {path}")
    if _sha256_file(production_embeddings_path) != expected_production_embeddings_sha256:
        raise SameTrackRecoveryError("Production embedding hash mismatch; recovery stopped")
    if _sha256_file(production_summary_path) != expected_production_summary_sha256:
        raise SameTrackRecoveryError("Production summary hash mismatch; recovery stopped")

    family_verification = verify_embedding_family(
        family_dir,
        production_embeddings_path=production_embeddings_path,
        production_summary_path=production_summary_path,
    )
    family_manifest, unusable_items = _load_family_unusable_rows(
        family_dir, config.target_session
    )
    observations, diagnostics, diagnostic_summary, observations_path, diagnostics_path, summary_path = _load_diagnostic_inputs(
        diagnostic_run_dir
    )
    source_session = _text(
        diagnostic_summary.get("run_metadata", {}).get("session_id")
    )
    if source_session != config.target_session:
        raise SameTrackRecoveryError(
            f"Diagnostic session mismatch: expected {config.target_session}, found {source_session}"
        )
    video_mapping = _video_map(diagnostic_summary)
    zone_config = load_camera_zone_config(camera_zones_path)

    input_fingerprint = {
        "policy_version": RECOVERY_POLICY_VERSION,
        "config": asdict(config),
        "family_id": _text(family_manifest.get("family_id")),
        "family_content_fingerprint": _text(
            family_manifest.get("content_fingerprint_sha256")
        ),
        "family_output_manifest_sha256": _sha256_file(
            family_dir / "output_manifest.json"
        ),
        "cctv_usability_sha256": _sha256_file(
            family_dir / "cctv_embedding_usability.csv"
        ),
        "tracklet_observations_sha256": _sha256_file(observations_path),
        "tracklet_diagnostics_sha256": _sha256_file(diagnostics_path),
        "diagnostic_summary_sha256": _sha256_file(summary_path),
        "camera_zones_sha256": _sha256_file(camera_zones_path),
        "yunet_sha256": _sha256_file(yunet_model),
        "sface_sha256": _sha256_file(sface_model),
        "production_embeddings_sha256": _sha256_file(production_embeddings_path),
        "production_summary_sha256": _sha256_file(production_summary_path),
        "target_items": [
            {
                "item_id": _text(row["Item_ID"]),
                "track_id": _text(row["Track_ID"]),
                "observation_id": _text(row["Observation_ID"]),
                "candidate_roll": _text(row["Candidate_Roll"]),
                "source_session": _text(row["Source_Session"]),
            }
            for row in unusable_items
        ],
    }
    recovery_fingerprint = _stable_digest(input_fingerprint)
    recovery_id = f"cctv-recovery-{recovery_fingerprint[:20]}"
    output_dir = output_root / recovery_id
    if output_dir.exists():
        manifest = verify_same_track_recovery(output_dir)
        if _text(manifest.get("input_fingerprint_sha256")) != recovery_fingerprint:
            raise SameTrackRecoveryError(
                f"Recovery ID already exists with different inputs: {output_dir}"
            )
        return RecoveryRunResult(
            output_dir=output_dir,
            recovery_id=recovery_id,
            status=_text(manifest.get("status")),
            manifest=manifest,
            reused=True,
        )

    output_root.mkdir(parents=True, exist_ok=True)
    temporary_dir = output_root / f".{recovery_id}.building-{os.getpid()}"
    if temporary_dir.exists():
        raise SameTrackRecoveryError(f"Stale recovery directory exists: {temporary_dir}")
    temporary_dir.mkdir(parents=True)
    try:
        if engine_factory is None:
            engine = FaceEngine(yunet_model, sface_model, detection_score=config.detector_score)
        else:
            engine = engine_factory(yunet_model, sface_model, config.detector_score)

        video_hashes: dict[str, str] = {}
        audit_rows: list[dict[str, Any]] = []
        track_results: list[dict[str, Any]] = []
        selected_recoveries: list[ObservationRecovery] = []
        selected_embeddings: dict[str, np.ndarray] = {}
        hash_owner: dict[str, str] = {}

        for item_ordinal, item in enumerate(unusable_items, start=1):
            if progress:
                progress(
                    f"Same-track recovery: item {item_ordinal}/{len(unusable_items)} "
                    f"{_text(item['Item_ID'])}"
                )
            candidate_rows = _eligible_observations(
                item, observations, diagnostics, config
            )
            reproduced: list[ObservationRecovery] = []
            failures: list[dict[str, Any]] = []
            for row in candidate_rows:
                key = (
                    _text(row.get("Checkpoint_ID")),
                    _text(row.get("Camera_ID")),
                    _text(row.get("Video")).lower(),
                )
                video_path = video_mapping.get(key)
                if video_path is None:
                    failures.append(
                        {
                            "observation_id": _text(row.get("Observation_ID")),
                            "reason": "video_mapping_missing",
                        }
                    )
                    continue
                video_path = Path(video_path).resolve()
                if not video_path.is_file():
                    failures.append(
                        {
                            "observation_id": _text(row.get("Observation_ID")),
                            "reason": "source_video_missing",
                            "video_path": str(video_path),
                        }
                    )
                    continue
                video_key = str(video_path)
                if video_key not in video_hashes:
                    video_hashes[video_key] = _sha256_file(video_path)
                try:
                    recovery = _recover_observation(
                        item=item,
                        observation=row,
                        video_path=video_path,
                        video_sha256=video_hashes[video_key],
                        zone_config=zone_config,
                        engine=engine,
                        config=config,
                    )
                    reproduced.append(recovery)
                    audit_rows.append(
                        {
                            "Item_ID": recovery.item_id,
                            "Track_ID": recovery.track_id,
                            "Observation_ID": recovery.observation_id,
                            "Candidate_Roll": recovery.candidate_roll,
                            "Source_Session": recovery.source_session,
                            "Checkpoint": recovery.checkpoint,
                            "Camera": recovery.camera,
                            "Frame": recovery.frame_index,
                            "Zone_Profile": recovery.zone_profile,
                            "Zone_ID": recovery.zone_id,
                            "Zone_Upscale": recovery.zone_upscale,
                            "Video_Path": str(recovery.video_path),
                            "Video_SHA256": recovery.video_sha256,
                            "Recorded_BBox": ",".join(f"{v:.4f}" for v in recovery.recorded_bbox),
                            "Matched_BBox": ",".join(f"{v:.4f}" for v in recovery.matched_bbox),
                            "Matched_IoU": round(recovery.matched_iou, 6),
                            "Detector_Score": round(recovery.detector_score, 6),
                            "Quality_Weight": round(recovery.quality_weight, 6),
                            "Face_Width": round(recovery.face_width, 4),
                            "Face_Height": round(recovery.face_height, 4),
                            "Blur_Variance": round(recovery.blur_variance, 4),
                            "Reproduced": True,
                            "Selected_For_Recovery": False,
                            "Embedding_Key": "",
                            "Failure_Reason": "",
                            "Identity_Source": "human_reviewed_track_actual_roll",
                            "Model_Prediction_Used_As_Identity": False,
                            "Manual_Rereview_Required": False,
                            "Quality_Gate_Relaxed": False,
                            "Super_Resolution_Used": False,
                        }
                    )
                except SameTrackRecoveryError as exc:
                    failures.append(
                        {
                            "observation_id": _text(row.get("Observation_ID")),
                            "reason": str(exc),
                        }
                    )
                    audit_rows.append(
                        {
                            "Item_ID": _text(item["Item_ID"]),
                            "Track_ID": _text(item["Track_ID"]),
                            "Observation_ID": _text(row.get("Observation_ID")),
                            "Candidate_Roll": _text(item["Candidate_Roll"]),
                            "Source_Session": _text(item["Source_Session"]),
                            "Checkpoint": _text(row.get("Checkpoint_ID")),
                            "Camera": _text(row.get("Camera_ID")),
                            "Frame": _int(row.get("Frame")),
                            "Zone_Profile": _text(row.get("Zone_Profile")),
                            "Zone_ID": _text(row.get("Zone_ID")),
                            "Zone_Upscale": "",
                            "Video_Path": str(video_path),
                            "Video_SHA256": video_hashes[video_key],
                            "Recorded_BBox": _text(row.get("BBox_Original_Coordinates")),
                            "Matched_BBox": "",
                            "Matched_IoU": "",
                            "Detector_Score": "",
                            "Quality_Weight": _float(row.get("Quality_Weight")),
                            "Face_Width": _float(row.get("Face_Width")),
                            "Face_Height": _float(row.get("Face_Height")),
                            "Blur_Variance": _float(row.get("Blur_Laplacian_Variance")),
                            "Reproduced": False,
                            "Selected_For_Recovery": False,
                            "Embedding_Key": "",
                            "Failure_Reason": str(exc),
                            "Identity_Source": "human_reviewed_track_actual_roll",
                            "Model_Prediction_Used_As_Identity": False,
                            "Manual_Rereview_Required": False,
                            "Quality_Gate_Relaxed": False,
                            "Super_Resolution_Used": False,
                        }
                    )

            selected, consistency = _select_consistent_recoveries(reproduced, config)
            selected_ids = {item.observation_id for item in selected}
            for recovery in selected:
                safe_observation = re.sub(r"[^A-Za-z0-9._-]+", "_", recovery.observation_id)
                embedding_key = f"{recovery.item_id}__{safe_observation}"
                source_hash = hashlib.sha256(
                    (
                        recovery.video_sha256
                        + "|"
                        + recovery.observation_id
                        + "|"
                        + recovery.candidate_roll
                    ).encode("utf-8")
                ).hexdigest()
                prior_owner = hash_owner.get(source_hash)
                if prior_owner is not None and prior_owner != recovery.candidate_roll:
                    raise SameTrackRecoveryError(
                        "Recovered source collision across candidate identities"
                    )
                hash_owner[source_hash] = recovery.candidate_roll
                selected_embeddings[embedding_key] = recovery.embedding
                crop_path = temporary_dir / "recovered_crops" / f"{embedding_key}.png"
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(crop_path), recovery.audit_crop):
                    raise SameTrackRecoveryError(f"Unable to write audit crop: {crop_path}")
                selected_recoveries.append(recovery)
                for audit in audit_rows:
                    if (
                        audit["Item_ID"] == recovery.item_id
                        and audit["Observation_ID"] == recovery.observation_id
                    ):
                        audit["Selected_For_Recovery"] = True
                        audit["Embedding_Key"] = embedding_key
                        audit["Audit_Crop_Path"] = str(
                            (output_dir / "recovered_crops" / crop_path.name)
                        )
                        audit["Audit_Crop_SHA256"] = _sha256_file(crop_path)
                        break
            track_results.append(
                {
                    "item_id": _text(item["Item_ID"]),
                    "track_id": _text(item["Track_ID"]),
                    "candidate_roll": _text(item["Candidate_Roll"]),
                    "source_session": _text(item["Source_Session"]),
                    "original_reviewed_observation_id": _text(item["Observation_ID"]),
                    "attempted_observation_ids": [
                        _text(row.get("Observation_ID")) for row in candidate_rows
                    ],
                    "failed_attempts": failures,
                    **consistency,
                }
            )

        npz_path = temporary_dir / "recovered_embeddings.npz"
        np.savez_compressed(npz_path, **selected_embeddings)
        _write_csv(temporary_dir / "same_track_recovery_audit.csv", audit_rows)
        recovered_tracks = [row for row in track_results if row.get("status") == "recovered"]
        recovered_identities = sorted(
            {
                _text(row["candidate_roll"])
                for row in recovered_tracks
                if _text(row.get("candidate_roll"))
            }
        )
        status = (
            "recovery_available"
            if recovered_tracks and selected_embeddings
            else "recovery_unavailable_no_technically_reproducible_observations"
        )
        manifest = {
            "schema_version": RECOVERY_SCHEMA_VERSION,
            "policy_version": RECOVERY_POLICY_VERSION,
            "recovery_id": recovery_id,
            "status": status,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "input_fingerprint_sha256": recovery_fingerprint,
            "input_fingerprint": input_fingerprint,
            "source_family_id": _text(family_manifest.get("family_id")),
            "source_family_decision": _text(family_manifest.get("final_decision")),
            "source_family_verified": True,
            "source_family_verification": {
                "verified_files": family_verification.get("verified_files"),
                "verified_variants": family_verification.get("variants_verified"),
            },
            "target_session": config.target_session,
            "target_items": len(unusable_items),
            "recovered_tracks": len(recovered_tracks),
            "recovered_identities": recovered_identities,
            "selected_embedding_count": len(selected_embeddings),
            "cross_session_source_available": bool(recovered_tracks),
            "track_results": track_results,
            "selected_recovered_embeddings": [
                {
                    "embedding_key": key,
                    "item_id": recovery.item_id,
                    "track_id": recovery.track_id,
                    "observation_id": recovery.observation_id,
                    "candidate_roll": recovery.candidate_roll,
                    "source_session": recovery.source_session,
                    "checkpoint": recovery.checkpoint,
                    "camera": recovery.camera,
                    "frame_index": recovery.frame_index,
                    "video_path": str(recovery.video_path),
                    "video_sha256": recovery.video_sha256,
                    "zone_profile": recovery.zone_profile,
                    "zone_id": recovery.zone_id,
                    "zone_upscale": recovery.zone_upscale,
                    "matched_iou": round(recovery.matched_iou, 6),
                    "detector_score": round(recovery.detector_score, 6),
                    "identity_source": "human_reviewed_track_actual_roll",
                    "model_prediction_used_as_identity": False,
                    "manual_rereview_required": False,
                    "quality_gate_relaxed": False,
                    "super_resolution_used": False,
                }
                for key, recovery in zip(selected_embeddings.keys(), selected_recoveries)
            ],
            "recovered_embeddings_sha256": _sha256_file(npz_path),
            "videos": [
                {"path": path, "sha256": digest}
                for path, digest in sorted(video_hashes.items())
            ],
            "production_embeddings_before_sha256": expected_production_embeddings_sha256,
            "production_summary_before_sha256": expected_production_summary_sha256,
            "production_embeddings_after_sha256": _sha256_file(
                production_embeddings_path
            ),
            "production_summary_after_sha256": _sha256_file(production_summary_path),
            "production_files_changed": False,
            "dataset_files_changed": False,
            "official_attendance_changed": False,
            "candidate_promoted": False,
            "mon_p3_processed": False,
            "manual_review_requested": False,
            "review_scalability_policy": {
                "existing_human_track_labels_reused": True,
                "same_track_observations_only": True,
                "model_prediction_never_controls_identity": True,
                "broad_manual_review_required": False,
                "live_exception_only_direction_preserved": True,
            },
            "exact_next_step": (
                "Validate and consume this immutable recovery bundle in a new "
                "embedding family build. Do not repeat the completed human reviews."
                if status == "recovery_available"
                else "Preserve this recovery audit. Collect future exception-only live "
                "evidence for TUE_P2 identities; do not weaken quality or repeat broad reviews."
            ),
        }
        if manifest["production_embeddings_after_sha256"] != expected_production_embeddings_sha256:
            raise SameTrackRecoveryError("Production embeddings changed during recovery")
        if manifest["production_summary_after_sha256"] != expected_production_summary_sha256:
            raise SameTrackRecoveryError("Production summary changed during recovery")
        _write_json(temporary_dir / "recovery_manifest.json", manifest)
        output_manifest = _output_manifest(temporary_dir)
        _write_json(temporary_dir / "output_manifest.json", output_manifest)
        temporary_dir.replace(output_dir)
        verified = verify_same_track_recovery(output_dir)
        return RecoveryRunResult(
            output_dir=output_dir,
            recovery_id=recovery_id,
            status=status,
            manifest=verified,
            reused=False,
        )
    except Exception as exc:
        failed = output_root / (
            f"failed-{recovery_id}-{datetime.now().strftime('%Y%m%d_%H%M%S')}-{os.getpid()}"
        )
        try:
            if temporary_dir.exists():
                temporary_dir.replace(failed)
        except OSError:
            failed = temporary_dir
        if isinstance(exc, SameTrackRecoveryError):
            raise SameTrackRecoveryError(
                f"Same-track CCTV recovery failed; partial artifacts preserved at {failed}: {exc}"
            ) from exc
        raise SameTrackRecoveryError(
            f"Same-track CCTV recovery failed; partial artifacts preserved at {failed}: {exc}"
        ) from exc
