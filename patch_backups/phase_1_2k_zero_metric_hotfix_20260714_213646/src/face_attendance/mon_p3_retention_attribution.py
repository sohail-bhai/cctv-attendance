from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd

from .config import DEFAULT_DETECTION_SCORE, DEFAULT_MAX_WIDTH, SFACE_MODEL, YUNET_MODEL
from .embedding_db import MatchResult, StudentEmbeddingDB
from .embedding_version_family import EmbeddingFamilyError, verify_embedding_family
from .face_engine import FaceEngine
from .recall_analysis import MATCH_THRESHOLD, MARGIN_THRESHOLD, canonical_roll
from .shadow_validation import ShadowValidationError, verify_output_manifest, write_output_manifest
from .utils import normalize_embedding, resize_keep_aspect
from .zones import (
    CameraZone,
    load_camera_zone_config,
    map_resized_face_to_original,
    map_zone_face_to_original,
    upscale_zone_crop,
    zone_pixel_bounds,
)

SCHEMA_VERSION = 1
POLICY_VERSION = "phase-1.2k-retention-attribution-v1"
EXPECTED_FAMILY_ID = "embfam-7bc431a3ad762398d4e9"
EXPECTED_RUN_ID = "mon-p3-shadow-8136f24575f6d319fb96"
EXPECTED_EVALUATION_ID = "mon-p3-evaluation-8f15934c693111852524"
EXPECTED_DECISION = "mon_p3_shadow_failed_retention"
EXPECTED_RETENTION_LOSSES = 6
EXPECTED_CORRECT_RECOVERIES = 4
EXPECTED_FALSE_IDENTITIES = 0
EXPECTED_PRODUCTION_EMBEDDINGS_SHA256 = (
    "c32ed31df10b7b9b43b8f19977a2adf0a82fb8e7a71f3e8bceb42c0565fedd49"
)
EXPECTED_PRODUCTION_SUMMARY_SHA256 = (
    "0df35c3bb9207e191aa49dad5536b8378e812d56463491bd7203885dca5ae9ee"
)
VARIANT_SPECS = {
    "production": None,
    "A_cleaned_enrollment": "cleaned_enrollment_only",
    "B_p2_recovered_for_p1": "cross_session_for_tue_p1",
    "C_p1_approved_for_p2": "cross_session_for_tue_p2",
    "D_full_candidate": "full_candidate",
}
REPRODUCTION_SCORE_TOLERANCE = 0.003
MIN_REPRODUCTION_IOU = 0.35


class RetentionAttributionError(RuntimeError):
    pass


@dataclass(frozen=True)
class AttributionInputs:
    repo_root: Path
    family_dir: Path
    shadow_output: Path
    evaluation_dir: Path
    production_diagnostic: Path
    candidate_diagnostic: Path
    candidate_summary: Path
    candidate_tracklets: Path
    candidate_observations: Path
    reviewed_tracks: Path
    tracklet_comparison: Path
    production_embeddings: Path
    production_summary: Path
    camera_zones: Path
    variant_embeddings: dict[str, Path]
    variant_source_records: dict[str, Path]
    fingerprint_sha256: str
    attribution_id: str
    output_dir: Path


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _yes(value: Any) -> bool:
    return _text(value).lower() in {"1", "true", "yes", "y", "on"}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _canonical(value: Any) -> str:
    return canonical_roll(_text(value)) or _text(value)


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
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


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not Path(path).is_file():
        raise RetentionAttributionError(f"{label} not found: {path}")
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RetentionAttributionError(f"Unable to read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise RetentionAttributionError(f"{label} must contain a JSON object: {path}")
    return payload


def _one_file(root: Path, pattern: str, label: str) -> Path:
    matches = sorted(Path(root).glob(pattern))
    if len(matches) != 1:
        raise RetentionAttributionError(
            f"Expected exactly one {label} under {root}; found {len(matches)}"
        )
    return matches[0]


def _parse_bbox(value: Any) -> tuple[float, float, float, float]:
    parts = [_float(part, math.nan) for part in _text(value).split(",")]
    if len(parts) != 4 or not all(math.isfinite(part) for part in parts):
        raise RetentionAttributionError(f"Invalid observation bbox: {value}")
    x, y, width, height = parts
    if width <= 0 or height <= 0:
        raise RetentionAttributionError(f"Invalid observation bbox dimensions: {value}")
    return x, y, width, height


def _bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    ix1, iy1 = max(lx, rx), max(ly, ry)
    ix2, iy2 = min(lx + lw, rx + rw), min(ly + lh, ry + rh)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(lw * lh + rw * rh - intersection, 1e-12)
    return float(intersection / union)


def _read_frame(video_path: Path, frame_index: int) -> np.ndarray:
    if frame_index < 1:
        raise RetentionAttributionError(f"Frame index must be positive: {frame_index}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RetentionAttributionError(f"Unable to open MON_P3 source video: {video_path}")
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
            raise RetentionAttributionError(
                f"Unable to decode frame {frame_index} from {video_path}"
            )
        return frame
    finally:
        capture.release()


def _zone_by_id(profile: Any, zone_id: str) -> CameraZone:
    matches = [zone for zone in profile.enabled_zones() if zone.zone_id == zone_id]
    if len(matches) != 1:
        raise RetentionAttributionError(
            f"Expected one enabled zone {profile.profile_id}.{zone_id}; found {len(matches)}"
        )
    return matches[0]


def _video_map(summary: dict[str, Any]) -> dict[tuple[str, str, str], Path]:
    mapping: dict[tuple[str, str, str], Path] = {}
    for row in summary.get("checkpoint_timing_map", []):
        checkpoint = _text(row.get("Checkpoint_ID"))
        camera = _text(row.get("Camera_ID"))
        filename = _text(row.get("Source_File_Name")).lower()
        source = Path(_text(row.get("Source_File")))
        if not checkpoint or not camera or not filename or not str(source):
            continue
        key = checkpoint, camera, filename
        previous = mapping.get(key)
        if previous is not None and previous != source:
            raise RetentionAttributionError(
                f"Ambiguous MON_P3 source mapping for {checkpoint}/{camera}/{filename}"
            )
        mapping[key] = source
    if len(mapping) != 10:
        raise RetentionAttributionError(
            f"MON_P3 diagnostic summary must map exactly 10 videos; found {len(mapping)}"
        )
    if any(not path.is_file() for path in mapping.values()):
        missing = [str(path) for path in mapping.values() if not path.is_file()]
        raise RetentionAttributionError("MON_P3 source video missing: " + ", ".join(missing[:5]))
    return mapping


def _reproduce_observation_embedding(
    *,
    row: dict[str, Any],
    video_path: Path,
    zone_config: Any,
    engine: FaceEngine,
    max_width: int,
) -> tuple[np.ndarray, float]:
    frame = _read_frame(video_path, _int(row.get("Frame")))
    recorded_bbox = _parse_bbox(row.get("BBox_Original_Coordinates"))
    source = _text(row.get("Detection_Source")).lower()
    candidates: list[tuple[float, float, np.ndarray, np.ndarray]] = []

    if source in {"zone", "zones"}:
        camera = _text(row.get("Camera_ID"))
        video_name = _text(row.get("Video"))
        profile = zone_config.resolve_profile(camera, video_name)
        if profile is None:
            raise RetentionAttributionError(
                f"No zone profile for {camera}/{video_name}"
            )
        zone = _zone_by_id(profile, _text(row.get("Zone_ID")))
        bounds = zone_pixel_bounds(zone, frame.shape)
        x1, y1, x2, y2 = bounds
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            raise RetentionAttributionError(
                f"Empty zone crop for {_text(row.get('Observation_ID'))}"
            )
        detector_input = upscale_zone_crop(crop, zone.upscale)
        for face in engine.detect_faces(detector_input):
            mapped = map_zone_face_to_original(face, bounds, zone.upscale)
            mapped_bbox = tuple(float(value) for value in mapped[:4])
            candidates.append(
                (_bbox_iou(recorded_bbox, mapped_bbox), engine.face_score(face), face, detector_input)
            )
    elif source in {"full", "full_frame"}:
        detector_input, scale = resize_keep_aspect(frame, max_width)
        for face in engine.detect_faces(detector_input):
            mapped = map_resized_face_to_original(face, scale)
            mapped_bbox = tuple(float(value) for value in mapped[:4])
            candidates.append(
                (_bbox_iou(recorded_bbox, mapped_bbox), engine.face_score(face), face, detector_input)
            )
    else:
        raise RetentionAttributionError(
            f"Unsupported detection source for exact reproduction: {source}"
        )

    if not candidates:
        raise RetentionAttributionError(
            f"No YuNet detection reproduced for {_text(row.get('Observation_ID'))}"
        )
    candidates.sort(key=lambda item: (-item[0], -item[1]))
    iou, _, face, detector_input = candidates[0]
    if iou < MIN_REPRODUCTION_IOU:
        raise RetentionAttributionError(
            f"Observation reproduction IoU below {MIN_REPRODUCTION_IOU:.2f} for "
            f"{_text(row.get('Observation_ID'))}: {iou:.4f}"
        )
    feature = engine.extract_feature(detector_input, face)
    if feature is None:
        raise RetentionAttributionError(
            f"SFace reproduction failed for {_text(row.get('Observation_ID'))}"
        )
    vector = normalize_embedding(np.asarray(feature, dtype=np.float32).reshape(-1))
    if vector.size != 128 or not np.all(np.isfinite(vector)):
        raise RetentionAttributionError(
            f"Invalid reproduced embedding for {_text(row.get('Observation_ID'))}"
        )
    return vector, float(iou)


def _aggregate_track(
    *,
    track_id: str,
    observations: pd.DataFrame,
    video_map: dict[tuple[str, str, str], Path],
    zone_config: Any,
    engine: FaceEngine,
    max_width: int,
    status: Callable[[str], None] | None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rows = observations[
        observations["Tracklet_ID"].astype(str).eq(track_id)
        & observations["Selected_For_Aggregation"].map(_yes)
        & observations["Embedding_Consistent"].map(_yes)
        & observations["Embedding_Extraction_Success"].map(_yes)
    ].copy()
    if len(rows) < 2:
        raise RetentionAttributionError(
            f"Reviewed track {track_id} has fewer than two selected consistent observations"
        )
    rows = rows.sort_values("Observation_ID")
    vectors: list[np.ndarray] = []
    weights: list[float] = []
    audit: list[dict[str, Any]] = []
    for row in rows.to_dict("records"):
        key = (
            _text(row.get("Checkpoint_ID")),
            _text(row.get("Camera_ID")),
            _text(row.get("Video")).lower(),
        )
        video_path = video_map.get(key)
        if video_path is None:
            raise RetentionAttributionError(
                f"No exact video mapping for observation {_text(row.get('Observation_ID'))}"
            )
        vector, iou = _reproduce_observation_embedding(
            row=row,
            video_path=video_path,
            zone_config=zone_config,
            engine=engine,
            max_width=max_width,
        )
        weight = max(_float(row.get("Quality_Weight")), 0.001)
        vectors.append(vector)
        weights.append(weight)
        audit.append(
            {
                "Tracklet_ID": track_id,
                "Observation_ID": _text(row.get("Observation_ID")),
                "Checkpoint_ID": key[0],
                "Camera_ID": key[1],
                "Video": key[2],
                "Frame": _int(row.get("Frame")),
                "Detection_Source": _text(row.get("Detection_Source")),
                "Zone_ID": _text(row.get("Zone_ID")),
                "Quality_Weight": weight,
                "Reproduction_IoU": iou,
                "Video_Path": str(video_path),
                "Video_SHA256": _sha256_file(video_path),
            }
        )
    aggregate = normalize_embedding(
        np.average(np.stack(vectors), axis=0, weights=np.asarray(weights, dtype=np.float32))
    )
    if aggregate.size != 128 or not np.all(np.isfinite(aggregate)):
        raise RetentionAttributionError(f"Invalid aggregate embedding for {track_id}")
    if status:
        status(f"Reproduced reviewed track {track_id}: {len(vectors)} observations")
    return aggregate, audit


def _match_dict(result: MatchResult) -> dict[str, Any]:
    return {
        "Accepted": bool(result.accepted),
        "Best_Roll": _canonical(result.roll_no),
        "Best_Score": float(result.best_score),
        "Second_Roll": _canonical(result.second_roll_no),
        "Second_Score": float(result.second_score),
        "Margin": float(result.margin),
        "Reason": _text(result.reason),
    }


def _assert_reproduced_match(
    *,
    variant: str,
    result: MatchResult,
    recorded: dict[str, Any],
    track_id: str,
) -> None:
    prefix = "Production" if variant == "production" else "Candidate"
    expected_best = _canonical(recorded.get(f"{prefix}_Tracklet_Best_Roll"))
    expected_second = _canonical(recorded.get(f"{prefix}_Tracklet_Second_Roll"))
    expected_score = _float(recorded.get(f"{prefix}_Tracklet_Best_Score"))
    expected_second_score = _float(recorded.get(f"{prefix}_Tracklet_Second_Score"))
    expected_margin = _float(recorded.get(f"{prefix}_Tracklet_Margin"))
    expected_accepted = _yes(recorded.get(f"{prefix}_Tracklet_Accepted"))
    expected_reason = _text(recorded.get(f"{prefix}_Tracklet_Matcher_Reason"))
    actual = _match_dict(result)
    failures: list[str] = []
    if actual["Best_Roll"] != expected_best:
        failures.append(f"best roll {actual['Best_Roll']} != {expected_best}")
    if actual["Second_Roll"] != expected_second:
        failures.append(f"second roll {actual['Second_Roll']} != {expected_second}")
    for label, observed, expected in (
        ("best score", actual["Best_Score"], expected_score),
        ("second score", actual["Second_Score"], expected_second_score),
        ("margin", actual["Margin"], expected_margin),
    ):
        if abs(observed - expected) > REPRODUCTION_SCORE_TOLERANCE:
            failures.append(f"{label} {observed:.6f} != {expected:.6f}")
    if actual["Accepted"] != expected_accepted:
        failures.append(f"accepted {actual['Accepted']} != {expected_accepted}")
    if actual["Reason"] != expected_reason:
        failures.append(f"reason {actual['Reason']} != {expected_reason}")
    if failures:
        raise RetentionAttributionError(
            f"Exact {variant} reproduction failed for {track_id}: " + "; ".join(failures)
        )


def _variant_additions(source_records: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base_ids = set(source_records["A_cleaned_enrollment"]["Source_ID"].astype(str))
    rows: list[dict[str, Any]] = []
    for variant in ("B_p2_recovered_for_p1", "C_p1_approved_for_p2", "D_full_candidate"):
        frame = source_records[variant]
        additions = frame[~frame["Source_ID"].astype(str).isin(base_ids)]
        for row in additions.to_dict("records"):
            rows.append(
                {
                    "Variant": variant,
                    "Source_ID": _text(row.get("Source_ID")),
                    "Canonical_Roll": _canonical(row.get("Canonical_Roll")),
                    "Source_Kind": _text(row.get("Source_Kind")),
                    "Source_Session": _text(row.get("Source_Session")),
                    "Image_Path": _text(row.get("Image_Path")),
                }
            )
    return pd.DataFrame(rows)


def _source_group_for_roll(additions: pd.DataFrame, roll: str) -> str:
    groups = set(
        additions[
            additions["Canonical_Roll"].eq(roll)
            & additions["Variant"].isin(
                ["B_p2_recovered_for_p1", "C_p1_approved_for_p2"]
            )
        ]["Source_Kind"].astype(str)
    )
    if groups == {"same_track_recovered_cctv_medoid"}:
        return "tue_p2_recovered_medoid_group"
    if groups == {"human_approved_cctv_crop"}:
        return "tue_p1_approved_cctv_group"
    if not groups:
        return "cleaned_enrollment_or_existing_sources"
    return "mixed_candidate_source_groups"


def _causal_variant_classification(row: dict[str, Any]) -> str:
    prod = bool(row["production_Accepted"])
    a = bool(row["A_cleaned_enrollment_Accepted"])
    b = bool(row["B_p2_recovered_for_p1_Accepted"])
    c = bool(row["C_p1_approved_for_p2_Accepted"])
    d = bool(row["D_full_candidate_Accepted"])
    if not prod or d:
        return "not_a_retention_loss"
    if not a:
        return "cleaned_enrollment_introduced_loss"
    if not b and c:
        return "tue_p2_recovered_medoid_group_introduced_loss"
    if not c and b:
        return "tue_p1_approved_cctv_group_introduced_loss"
    if b and c and not d:
        return "cross_session_source_interaction_introduced_loss"
    if not b and not c:
        return "both_source_groups_independently_trigger_loss"
    return "mixed_or_unresolved_variant_attribution"


def resolve_inputs(
    *,
    repo_root: Path,
    family_dir: Path,
    shadow_output: Path,
    evaluation_dir: Path,
    output_root: Path | None = None,
) -> AttributionInputs:
    repo_root = Path(repo_root).resolve()
    family_dir = Path(family_dir).resolve()
    shadow_output = Path(shadow_output).resolve()
    evaluation_dir = Path(evaluation_dir).resolve()
    try:
        verify_output_manifest(shadow_output, shadow_output / "output_manifest.json")
        verify_output_manifest(evaluation_dir, evaluation_dir / "output_manifest.json")
    except ShadowValidationError as exc:
        raise RetentionAttributionError(str(exc)) from exc
    summary = _load_json(shadow_output / "shadow_run_summary.json", "Phase 1.2J summary")
    evaluation = _load_json(
        evaluation_dir / "mon_p3_evaluation_summary.json", "Phase 1.2J evaluation"
    )
    if _text(summary.get("run_id")) != EXPECTED_RUN_ID:
        raise RetentionAttributionError("Unexpected Phase 1.2J run ID")
    if _text(evaluation.get("evaluation_id")) != EXPECTED_EVALUATION_ID:
        raise RetentionAttributionError("Unexpected Phase 1.2J evaluation ID")
    if _text(evaluation.get("decision")) != EXPECTED_DECISION:
        raise RetentionAttributionError("Phase 1.2J evaluation is not the frozen retention failure")
    if int(evaluation.get("candidate_lost_correct_production_accepts") or -1) != EXPECTED_RETENTION_LOSSES:
        raise RetentionAttributionError("Frozen evaluation no longer contains exactly six retention losses")
    if int(evaluation.get("candidate_correct_recoveries") or -1) != EXPECTED_CORRECT_RECOVERIES:
        raise RetentionAttributionError("Frozen evaluation no longer contains exactly four correct recoveries")
    if int(evaluation.get("candidate_false_identities_or_unsafe_accepts") or -1) != EXPECTED_FALSE_IDENTITIES:
        raise RetentionAttributionError("Frozen evaluation false-identity count changed")
    try:
        family_verification = verify_embedding_family(family_dir)
    except EmbeddingFamilyError as exc:
        raise RetentionAttributionError(str(exc)) from exc
    if _text(family_verification.get("family_id")) != EXPECTED_FAMILY_ID:
        raise RetentionAttributionError("Unexpected embedding family ID")

    production_embeddings = repo_root / "models" / "student_embeddings.pkl"
    production_summary = repo_root / "models" / "embedding_summary.csv"
    if _sha256_file(production_embeddings) != EXPECTED_PRODUCTION_EMBEDDINGS_SHA256:
        raise RetentionAttributionError("Production embeddings changed after Phase 1.2J")
    if _sha256_file(production_summary) != EXPECTED_PRODUCTION_SUMMARY_SHA256:
        raise RetentionAttributionError("Production summary changed after Phase 1.2J")

    production_diagnostic = Path(_text(summary.get("production_diagnostic_run"))).resolve()
    candidate_diagnostic = Path(_text(summary.get("candidate_diagnostic_run"))).resolve()
    candidate_summary = _one_file(candidate_diagnostic, "diagnostic_summary_*.json", "candidate summary")
    candidate_tracklets = _one_file(candidate_diagnostic, "tracklet_diagnostics_*.csv", "candidate tracklets")
    candidate_observations = _one_file(candidate_diagnostic, "tracklet_observations_*.csv", "candidate observations")
    reviewed_tracks = evaluation_dir / "reviewed_exception_tracks.csv"
    tracklet_comparison = shadow_output / "tracklet_comparison.csv"
    for path in (reviewed_tracks, tracklet_comparison):
        if not path.is_file():
            raise RetentionAttributionError(f"Required attribution input missing: {path}")

    variant_embeddings: dict[str, Path] = {"production": production_embeddings}
    variant_source_records: dict[str, Path] = {}
    for label, folder in VARIANT_SPECS.items():
        if label == "production":
            continue
        variant_dir = family_dir / "variants" / str(folder)
        embeddings = variant_dir / "student_embeddings.pkl"
        source_records = variant_dir / "source_records.csv"
        if not embeddings.is_file() or not source_records.is_file():
            raise RetentionAttributionError(f"Variant artifacts missing: {variant_dir}")
        variant_embeddings[label] = embeddings
        variant_source_records[label] = source_records

    camera_zones = repo_root / "data" / "camera_zones.json"
    if not camera_zones.is_file():
        raise RetentionAttributionError(f"Camera-zone configuration missing: {camera_zones}")
    fingerprint = {
        "policy_version": POLICY_VERSION,
        "family_id": EXPECTED_FAMILY_ID,
        "family_output_manifest_sha256": _sha256_file(family_dir / "output_manifest.json"),
        "run_id": EXPECTED_RUN_ID,
        "shadow_output_manifest_sha256": _sha256_file(shadow_output / "output_manifest.json"),
        "evaluation_id": EXPECTED_EVALUATION_ID,
        "evaluation_output_manifest_sha256": _sha256_file(evaluation_dir / "output_manifest.json"),
        "candidate_summary_sha256": _sha256_file(candidate_summary),
        "candidate_tracklets_sha256": _sha256_file(candidate_tracklets),
        "candidate_observations_sha256": _sha256_file(candidate_observations),
        "production_embeddings_sha256": _sha256_file(production_embeddings),
        "variant_embeddings_sha256": {
            key: _sha256_file(path) for key, path in sorted(variant_embeddings.items())
        },
        "camera_zones_sha256": _sha256_file(camera_zones),
    }
    fingerprint_sha256 = _stable_digest(fingerprint)
    attribution_id = "retention-attribution-" + fingerprint_sha256[:20]
    root = Path(output_root or repo_root / "attendance_output" / "embedding_forensics" / "phase_1_2k").resolve()
    output_dir = root / attribution_id
    return AttributionInputs(
        repo_root=repo_root,
        family_dir=family_dir,
        shadow_output=shadow_output,
        evaluation_dir=evaluation_dir,
        production_diagnostic=production_diagnostic,
        candidate_diagnostic=candidate_diagnostic,
        candidate_summary=candidate_summary,
        candidate_tracklets=candidate_tracklets,
        candidate_observations=candidate_observations,
        reviewed_tracks=reviewed_tracks,
        tracklet_comparison=tracklet_comparison,
        production_embeddings=production_embeddings,
        production_summary=production_summary,
        camera_zones=camera_zones,
        variant_embeddings=variant_embeddings,
        variant_source_records=variant_source_records,
        fingerprint_sha256=fingerprint_sha256,
        attribution_id=attribution_id,
        output_dir=output_dir,
    )


def run_retention_attribution(
    inputs: AttributionInputs,
    *,
    yunet_model: Path = YUNET_MODEL,
    sface_model: Path = SFACE_MODEL,
    status: Callable[[str], None] | None = print,
) -> tuple[dict[str, Any], Path, bool]:
    if inputs.output_dir.exists():
        try:
            verify_output_manifest(inputs.output_dir, inputs.output_dir / "output_manifest.json")
        except ShadowValidationError as exc:
            raise RetentionAttributionError(str(exc)) from exc
        existing = _load_json(inputs.output_dir / "attribution_summary.json", "existing attribution")
        return existing, inputs.output_dir, True

    reviewed = pd.read_csv(inputs.reviewed_tracks, dtype=str, keep_default_na=False)
    comparison = pd.read_csv(inputs.tracklet_comparison, dtype=str, keep_default_na=False)
    tracklets = pd.read_csv(inputs.candidate_tracklets, dtype=str, keep_default_na=False)
    observations = pd.read_csv(inputs.candidate_observations, dtype=str, keep_default_na=False)
    if len(reviewed) != 15 or reviewed["Review_ID"].nunique() != 15:
        raise RetentionAttributionError("Expected the completed 15-item Phase 1.2J review")
    if reviewed["Phase_1_2J_Result"].eq("candidate_lost_correct_production_accept").sum() != 6:
        raise RetentionAttributionError("Reviewed evidence no longer contains six retention losses")

    joined = reviewed.merge(
        comparison,
        on="Tracklet_ID",
        how="left",
        validate="one_to_one",
        suffixes=("_review", ""),
    )
    if len(joined) != 15 or joined["Session_ID"].eq("").any():
        raise RetentionAttributionError("Unable to join reviewed tracks to comparison evidence")

    source_records = {
        label: pd.read_csv(path, dtype=str, keep_default_na=False)
        for label, path in inputs.variant_source_records.items()
    }
    additions = _variant_additions(source_records)
    summary_json = _load_json(inputs.candidate_summary, "candidate diagnostic summary")
    video_map = _video_map(summary_json)
    zone_config = load_camera_zone_config(inputs.camera_zones)
    engine = FaceEngine(
        Path(yunet_model),
        Path(sface_model),
        detection_score=DEFAULT_DETECTION_SCORE,
    )
    databases = {
        label: StudentEmbeddingDB.load(path)
        for label, path in inputs.variant_embeddings.items()
    }

    score_rows: list[dict[str, Any]] = []
    observation_audit: list[dict[str, Any]] = []
    for index, review_row in enumerate(joined.to_dict("records"), start=1):
        track_id = _text(review_row["Tracklet_ID"])
        if status:
            status(f"Attribution track {index}/15: {track_id}")
        aggregate, audit = _aggregate_track(
            track_id=track_id,
            observations=observations,
            video_map=video_map,
            zone_config=zone_config,
            engine=engine,
            max_width=DEFAULT_MAX_WIDTH,
            status=None,
        )
        observation_audit.extend(audit)
        wide: dict[str, Any] = {
            "Review_ID": _text(review_row.get("Review_ID")),
            "Tracklet_ID": track_id,
            "Checkpoint_ID": _text(review_row.get("Checkpoint_ID")),
            "Camera_ID": _text(review_row.get("Camera_ID")),
            "Actual_Roll": _canonical(review_row.get("Actual_Roll_Normalized")),
            "Phase_1_2J_Result": _text(review_row.get("Phase_1_2J_Result")),
        }
        for label, database in databases.items():
            result = database.match(
                aggregate,
                match_threshold=MATCH_THRESHOLD,
                margin_threshold=MARGIN_THRESHOLD,
                aggregate="top3",
            )
            if label in {"production", "D_full_candidate"}:
                _assert_reproduced_match(
                    variant=label,
                    result=result,
                    recorded=review_row,
                    track_id=track_id,
                )
            fields = _match_dict(result)
            for key, value in fields.items():
                wide[f"{label}_{key}"] = value
        wide["Causal_Variant_Classification"] = _causal_variant_classification(wide)
        candidate_second = _canonical(wide.get("D_full_candidate_Second_Roll"))
        wide["Candidate_Second_Roll_Source_Group"] = _source_group_for_roll(
            additions, candidate_second
        )
        score_rows.append(wide)

    scores = pd.DataFrame(score_rows)
    losses = scores[
        scores["Phase_1_2J_Result"].eq("candidate_lost_correct_production_accept")
    ].copy()
    recoveries = scores[scores["Phase_1_2J_Result"].eq("candidate_correct_recovery")].copy()
    if len(losses) != 6 or len(recoveries) != 4:
        raise RetentionAttributionError("Attribution result counts changed unexpectedly")
    if not losses["production_Accepted"].all() or losses["D_full_candidate_Accepted"].any():
        raise RetentionAttributionError("Retention-loss acceptance contract failed")
    if not np.allclose(
        losses["production_Best_Score"].astype(float),
        losses["D_full_candidate_Best_Score"].astype(float),
        atol=REPRODUCTION_SCORE_TOLERANCE,
    ):
        raise RetentionAttributionError(
            "Retention losses are no longer margin-only; top-score changes require a different analysis"
        )

    group_summary = (
        losses.groupby(
            ["Causal_Variant_Classification", "Candidate_Second_Roll_Source_Group", "D_full_candidate_Second_Roll"],
            dropna=False,
        )
        .size()
        .reset_index(name="Retention_Loss_Count")
        .sort_values(["Retention_Loss_Count", "D_full_candidate_Second_Roll"], ascending=[False, True])
    )
    decision = "family_rejected_mon_p3_retention_attributed"
    rejection_record = {
        "schema_version": SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "family_id": EXPECTED_FAMILY_ID,
        "candidate_variant_id": "embfam-7bc431a3ad762398d4e9-d",
        "status": "rejected_mon_p3_retention",
        "final_decision": "reject_family_candidate",
        "production_approved": False,
        "enabled": False,
        "immutable": True,
        "rejected_at": datetime.now().isoformat(timespec="seconds"),
        "phase_1_2j_evidence": {
            "run_id": EXPECTED_RUN_ID,
            "evaluation_id": EXPECTED_EVALUATION_ID,
            "reviewed_exception_tracks": 15,
            "candidate_false_identities_or_unsafe_accepts": 0,
            "candidate_correct_recoveries": 4,
            "candidate_lost_correct_production_accepts": 6,
            "decision": EXPECTED_DECISION,
        },
        "phase_1_2k_attribution": {
            "attribution_id": inputs.attribution_id,
            "fingerprint_sha256": inputs.fingerprint_sha256,
            "all_reviewed_tracks_reproduced": True,
            "production_and_full_candidate_scores_verified": True,
            "retention_losses_are_margin_only": True,
            "source_group_summary": group_summary.to_dict("records"),
        },
        "future_policy": {
            "may_be_promoted": False,
            "may_be_reenabled": False,
            "manual_review_must_be_repeated": False,
            "mon_p3_full_recognition_must_be_repeated_for_attribution": False,
            "next_candidate_must_be_new_immutable_family": True,
        },
        "source_artifacts": {
            "family_output_manifest": {
                "path": str(inputs.family_dir / "output_manifest.json"),
                "sha256": _sha256_file(inputs.family_dir / "output_manifest.json"),
            },
            "phase_1_2j_output_manifest": {
                "path": str(inputs.shadow_output / "output_manifest.json"),
                "sha256": _sha256_file(inputs.shadow_output / "output_manifest.json"),
            },
            "phase_1_2j_evaluation_manifest": {
                "path": str(inputs.evaluation_dir / "output_manifest.json"),
                "sha256": _sha256_file(inputs.evaluation_dir / "output_manifest.json"),
            },
        },
    }

    registry_dir = inputs.repo_root / "models" / "family_registry" / "rejected"
    registry_dir.mkdir(parents=True, exist_ok=True)
    registry_path = registry_dir / f"{EXPECTED_FAMILY_ID}.json"
    registry_created = False
    if registry_path.exists():
        existing = _load_json(registry_path, "existing rejected-family record")
        stable_existing = {k: v for k, v in existing.items() if k != "rejected_at"}
        stable_new = {k: v for k, v in rejection_record.items() if k != "rejected_at"}
        if stable_existing != stable_new:
            raise RetentionAttributionError(
                f"Conflicting family-rejection record already exists: {registry_path}"
            )
        rejection_record = existing
    else:
        with registry_path.open("x", encoding="utf-8") as handle:
            json.dump(rejection_record, handle, indent=2)
        registry_created = True

    temporary = inputs.output_dir.with_name(inputs.output_dir.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=False)
    scores.to_csv(temporary / "reviewed_track_variant_scores.csv", index=False)
    losses.to_csv(temporary / "retention_loss_attribution.csv", index=False)
    recoveries.to_csv(temporary / "candidate_recovery_attribution.csv", index=False)
    additions.to_csv(temporary / "candidate_source_group_inventory.csv", index=False)
    group_summary.to_csv(temporary / "retention_loss_source_group_summary.csv", index=False)
    pd.DataFrame(observation_audit).to_csv(
        temporary / "exact_observation_reproduction_audit.csv", index=False
    )
    shutil.copy2(registry_path, temporary / "family_rejection_record.json")

    classification_counts = (
        losses["Causal_Variant_Classification"].value_counts().sort_index().to_dict()
    )
    source_roll_counts = losses["D_full_candidate_Second_Roll"].value_counts().sort_index().to_dict()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "attribution_id": inputs.attribution_id,
        "fingerprint_sha256": inputs.fingerprint_sha256,
        "family_id": EXPECTED_FAMILY_ID,
        "phase_1_2j_run_id": EXPECTED_RUN_ID,
        "phase_1_2j_evaluation_id": EXPECTED_EVALUATION_ID,
        "decision": decision,
        "reviewed_tracks_reproduced": int(len(scores)),
        "retention_losses": int(len(losses)),
        "correct_candidate_recoveries": int(len(recoveries)),
        "false_identities_or_unsafe_accepts": 0,
        "retention_losses_margin_only": True,
        "causal_variant_classification_counts": {
            str(key): int(value) for key, value in classification_counts.items()
        },
        "candidate_second_roll_loss_counts": {
            str(key): int(value) for key, value in source_roll_counts.items()
        },
        "rejection_registry": str(registry_path),
        "rejection_registry_created": registry_created,
        "production_embeddings_before_sha256": _sha256_file(inputs.production_embeddings),
        "production_summary_before_sha256": _sha256_file(inputs.production_summary),
        "production_embeddings_after_sha256": _sha256_file(inputs.production_embeddings),
        "production_summary_after_sha256": _sha256_file(inputs.production_summary),
        "production_embeddings_changed": False,
        "production_summary_changed": False,
        "dataset_files_changed": False,
        "official_attendance_changed": False,
        "candidate_promoted": False,
        "mon_p3_full_recognition_repeated": False,
        "manual_review_repeated": False,
        "exact_next_step": (
            "Use this attribution to design a new immutable source-balanced candidate; "
            "do not modify or promote embfam-7bc431a3ad762398d4e9."
        ),
    }
    (temporary / "attribution_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    report_lines = [
        "# Phase 1.2K MON_P3 Retention Attribution",
        "",
        f"- Decision: `{decision}`",
        "- Existing human review reused: 15/15 tracks",
        "- Candidate false identities / unsafe accepts: 0",
        "- Candidate correct recoveries: 4",
        "- Candidate lost correct production accepts: 6",
        "- All six losses reproduced as margin-only failures: yes",
        "- Full MON_P3 recognition repeated: no",
        "",
        "## Source-group attribution",
        "",
    ]
    for row in group_summary.to_dict("records"):
        report_lines.append(
            "- "
            + f"{row['D_full_candidate_Second_Roll']}: {row['Retention_Loss_Count']} losses; "
            + f"{row['Causal_Variant_Classification']}; "
            + f"{row['Candidate_Second_Roll_Source_Group']}"
        )
    report_lines += [
        "",
        "The rejected family remains unpromoted. Production embeddings, datasets, and official attendance were not changed.",
    ]
    (temporary / "attribution_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    write_output_manifest(
        temporary,
        {
            "policy_version": POLICY_VERSION,
            "attribution_id": inputs.attribution_id,
            "family_id": EXPECTED_FAMILY_ID,
            "decision": decision,
            "candidate_promoted": False,
            "production_changes": False,
            "official_attendance_modified": False,
            "mon_p3_full_recognition_repeated": False,
        },
    )
    try:
        verify_output_manifest(temporary, temporary / "output_manifest.json")
    except ShadowValidationError as exc:
        raise RetentionAttributionError(str(exc)) from exc
    inputs.output_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, inputs.output_dir)
    return summary, inputs.output_dir, False


def verify_attribution_output(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    try:
        verified = verify_output_manifest(output_dir, output_dir / "output_manifest.json")
    except ShadowValidationError as exc:
        raise RetentionAttributionError(str(exc)) from exc
    summary = _load_json(output_dir / "attribution_summary.json", "attribution summary")
    if _text(summary.get("decision")) != "family_rejected_mon_p3_retention_attributed":
        raise RetentionAttributionError("Attribution output does not contain the expected decision")
    if int(summary.get("retention_losses") or 0) != 6:
        raise RetentionAttributionError("Attribution output no longer contains six retention losses")
    if summary.get("production_embeddings_changed") is not False:
        raise RetentionAttributionError("Attribution claims production embeddings changed")
    return {
        "status": "PASS",
        "output_dir": str(output_dir),
        "verified_files": int(verified.get("verified_files", 0)),
        "decision": summary["decision"],
        "retention_losses": summary["retention_losses"],
        "correct_candidate_recoveries": summary["correct_candidate_recoveries"],
        "candidate_promoted": False,
        "official_attendance_changed": False,
        "mon_p3_full_recognition_repeated": False,
    }
