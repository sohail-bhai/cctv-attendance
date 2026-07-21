from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


FACE_COORD_COLUMNS = list(range(0, 14))


class ZoneConfigError(ValueError):
    pass


@dataclass(frozen=True)
class CameraZone:
    zone_id: str
    label: str
    x: float
    y: float
    width: float
    height: float
    enabled: bool = True
    upscale: float = 1.0
    notes: str = ""

    @property
    def bounds_normalized_text(self) -> str:
        return f"{self.x:.4f},{self.y:.4f},{self.width:.4f},{self.height:.4f}"


@dataclass(frozen=True)
class CameraZoneProfile:
    profile_id: str
    aliases: tuple[str, ...]
    expected_width: int
    expected_height: int
    zones: tuple[CameraZone, ...]
    notes: str = ""

    def enabled_zones(self) -> list[CameraZone]:
        return [zone for zone in self.zones if zone.enabled]


@dataclass(frozen=True)
class CameraZoneConfig:
    version: int
    profiles: dict[str, CameraZoneProfile]

    def resolve_profile(self, camera_id: str, source_name: str = "") -> CameraZoneProfile | None:
        candidates = {
            str(camera_id or "").strip().lower(),
            str(source_name or "").strip().lower(),
            Path(str(source_name or "")).stem.strip().lower(),
        }
        candidates = {candidate for candidate in candidates if candidate}
        for profile in self.profiles.values():
            alias_set = {profile.profile_id.lower(), *(alias.lower() for alias in profile.aliases)}
            if candidates & alias_set:
                return profile
        return None


@dataclass
class DetectionCandidate:
    candidate_id: str
    detection_source: str
    face_original: np.ndarray
    recognition_face: np.ndarray
    recognition_frame: np.ndarray
    recognition_frame_kind: str
    detector_score: float
    source_priority: int
    zone_profile: str = ""
    zone_id: str = ""
    zone_label: str = ""
    zone_upscale_factor: float = 1.0
    zone_bounds_normalized: str = ""
    zone_bounds_pixels: str = ""
    detector_input_width: int = 0
    detector_input_height: int = 0
    bbox_zone_coordinates: str = ""
    bbox_original_coordinates: str = ""
    merge_group_id: str = ""
    merged_detection_count: int = 1
    contributing_sources: str = ""
    selected_after_merge: bool = True
    selected_source: str = ""
    selected_zone_id: str = ""
    duplicate_iou: float = 0.0
    merge_reason: str = "single_detection"
    quality_rank: tuple[Any, ...] = field(default_factory=tuple)

    @property
    def source_label(self) -> str:
        return self.zone_id if self.detection_source == "zone" else "full_frame"


def _as_float(value: Any, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ZoneConfigError(f"{field_name} must be numeric") from exc
    if not math.isfinite(number):
        raise ZoneConfigError(f"{field_name} must be finite")
    return number


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def load_camera_zone_config(path: str | Path) -> CameraZoneConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return parse_camera_zone_config(payload)


def parse_camera_zone_config(payload: dict[str, Any]) -> CameraZoneConfig:
    if not isinstance(payload, dict):
        raise ZoneConfigError("Camera zone config must be a JSON object")
    if "version" not in payload:
        raise ZoneConfigError("Camera zone config is missing version")
    version = int(payload["version"])
    if version != 1:
        raise ZoneConfigError(f"Unsupported camera zone config version: {version}")
    profiles_raw = payload.get("profiles")
    if not isinstance(profiles_raw, dict) or not profiles_raw:
        raise ZoneConfigError("Camera zone config requires a non-empty profiles object")

    profiles: dict[str, CameraZoneProfile] = {}
    for profile_id, profile_raw in profiles_raw.items():
        if not isinstance(profile_raw, dict):
            raise ZoneConfigError(f"Profile {profile_id} must be an object")
        expected_width = int(profile_raw.get("expected_width") or 0)
        expected_height = int(profile_raw.get("expected_height") or 0)
        if expected_width <= 0 or expected_height <= 0:
            raise ZoneConfigError(f"Profile {profile_id} requires positive expected dimensions")
        aliases_raw = profile_raw.get("aliases", [])
        if aliases_raw is None:
            aliases_raw = []
        if not isinstance(aliases_raw, list):
            raise ZoneConfigError(f"Profile {profile_id} aliases must be a list")
        aliases = tuple(str(alias).strip() for alias in aliases_raw if str(alias).strip())
        zones_raw = profile_raw.get("zones")
        if not isinstance(zones_raw, list):
            raise ZoneConfigError(f"Profile {profile_id} zones must be a list")
        zones: list[CameraZone] = []
        seen_ids: set[str] = set()
        for zone_raw in zones_raw:
            if not isinstance(zone_raw, dict):
                raise ZoneConfigError(f"Profile {profile_id} contains a malformed zone")
            zone_id = str(zone_raw.get("id") or "").strip()
            if not zone_id:
                raise ZoneConfigError(f"Profile {profile_id} contains a zone without id")
            if zone_id in seen_ids:
                raise ZoneConfigError(f"Duplicate zone id in profile {profile_id}: {zone_id}")
            seen_ids.add(zone_id)
            x = _as_float(zone_raw.get("x"), f"{profile_id}.{zone_id}.x")
            y = _as_float(zone_raw.get("y"), f"{profile_id}.{zone_id}.y")
            width = _as_float(zone_raw.get("width"), f"{profile_id}.{zone_id}.width")
            height = _as_float(zone_raw.get("height"), f"{profile_id}.{zone_id}.height")
            upscale = _as_float(zone_raw.get("upscale", 1.0), f"{profile_id}.{zone_id}.upscale")
            if x < 0 or y < 0 or width <= 0 or height <= 0:
                raise ZoneConfigError(f"Zone {profile_id}.{zone_id} has negative or zero bounds")
            if x > 1 or y > 1 or width > 1 or height > 1 or x + width > 1 or y + height > 1:
                raise ZoneConfigError(f"Zone {profile_id}.{zone_id} extends outside normalized frame bounds")
            if upscale < 1.0 or upscale > 4.0:
                raise ZoneConfigError(f"Zone {profile_id}.{zone_id} has invalid upscale {upscale}")
            zones.append(CameraZone(
                zone_id=zone_id,
                label=str(zone_raw.get("label") or zone_id),
                x=x,
                y=y,
                width=width,
                height=height,
                enabled=_as_bool(zone_raw.get("enabled", True)),
                upscale=upscale,
                notes=str(zone_raw.get("notes") or ""),
            ))
        profiles[str(profile_id)] = CameraZoneProfile(
            profile_id=str(profile_id),
            aliases=aliases,
            expected_width=expected_width,
            expected_height=expected_height,
            zones=tuple(zones),
            notes=str(profile_raw.get("notes") or ""),
        )
    return CameraZoneConfig(version=version, profiles=profiles)


def zone_pixel_bounds(zone: CameraZone, frame_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    height, width = frame_shape[:2]
    x1 = int(round(zone.x * width))
    y1 = int(round(zone.y * height))
    x2 = int(round((zone.x + zone.width) * width))
    y2 = int(round((zone.y + zone.height) * height))
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def bounds_text(bounds: tuple[int, int, int, int]) -> str:
    return ",".join(str(int(v)) for v in bounds)


def face_box_text(face: np.ndarray) -> str:
    if face is None or len(face) < 4:
        return ""
    return ",".join(f"{float(v):.2f}" for v in face[:4])


def map_zone_face_to_original(face: np.ndarray, zone_bounds: tuple[int, int, int, int], upscale: float) -> np.ndarray:
    mapped = np.asarray(face, dtype=np.float32).copy()
    x1, y1, _, _ = zone_bounds
    mapped[0] = mapped[0] / upscale + x1
    mapped[1] = mapped[1] / upscale + y1
    mapped[2] = mapped[2] / upscale
    mapped[3] = mapped[3] / upscale
    if len(mapped) >= 14:
        for idx in range(4, 14, 2):
            mapped[idx] = mapped[idx] / upscale + x1
            mapped[idx + 1] = mapped[idx + 1] / upscale + y1
    return mapped


def map_resized_face_to_original(face: np.ndarray, scale: float) -> np.ndarray:
    mapped = np.asarray(face, dtype=np.float32).copy()
    if abs(scale - 1.0) < 1e-9:
        return mapped
    mapped[FACE_COORD_COLUMNS] = mapped[FACE_COORD_COLUMNS] / float(scale)
    return mapped


def upscale_zone_crop(crop: np.ndarray, upscale: float) -> np.ndarray:
    if crop is None or crop.size == 0 or upscale <= 1.0:
        return crop
    height, width = crop.shape[:2]
    out_width = max(1, int(round(width * upscale)))
    out_height = max(1, int(round(height * upscale)))
    return cv2.resize(crop, (out_width, out_height), interpolation=cv2.INTER_CUBIC)


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax, ay, aw, ah = [float(v) for v in a[:4]]
    bx, by, bw, bh = [float(v) for v in b[:4]]
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    union = max(aw * ah + bw * bh - intersection, 1.0)
    return intersection / union


def center_distance(a: np.ndarray, b: np.ndarray) -> float:
    ax, ay, aw, ah = [float(v) for v in a[:4]]
    bx, by, bw, bh = [float(v) for v in b[:4]]
    return float(math.hypot((ax + aw / 2.0) - (bx + bw / 2.0), (ay + ah / 2.0) - (by + bh / 2.0)))


def are_duplicate_faces(a: DetectionCandidate, b: DetectionCandidate, iou_threshold: float = 0.20) -> tuple[bool, float, str]:
    iou = box_iou(a.face_original, b.face_original)
    if iou >= iou_threshold:
        return True, iou, "iou_overlap"
    distance = center_distance(a.face_original, b.face_original)
    min_size = min(float(a.face_original[2]), float(a.face_original[3]), float(b.face_original[2]), float(b.face_original[3]))
    if distance <= max(8.0, min_size * 0.45):
        return True, iou, "center_proximity"
    return False, iou, ""


def merge_detection_candidates(candidates: list[DetectionCandidate], iou_threshold: float = 0.20) -> list[DetectionCandidate]:
    ordered = sorted(candidates, key=lambda c: (c.candidate_id, c.detection_source, c.zone_id))
    parent = list(range(len(ordered)))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    pair_reasons: dict[tuple[int, int], tuple[float, str]] = {}
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            duplicate, iou, reason = are_duplicate_faces(ordered[i], ordered[j], iou_threshold=iou_threshold)
            if duplicate:
                union(i, j)
                pair_reasons[(i, j)] = (iou, reason)

    groups: dict[int, list[int]] = {}
    for idx in range(len(ordered)):
        groups.setdefault(find(idx), []).append(idx)

    selected: list[DetectionCandidate] = []
    for group_number, indices in enumerate(sorted(groups.values(), key=lambda group: ordered[group[0]].candidate_id), start=1):
        members = [ordered[idx] for idx in indices]
        representative = sorted(
            members,
            key=lambda c: (c.quality_rank, c.detector_score, c.source_priority, c.candidate_id),
            reverse=True,
        )[0]
        sources = sorted({member.detection_source if not member.zone_id else f"zone:{member.zone_id}" for member in members})
        source_kinds = {member.detection_source for member in members}
        if len(source_kinds) > 1:
            selected_source = "full_frame_and_zone"
        elif len({member.zone_id for member in members if member.zone_id}) > 1:
            selected_source = "multiple_zones"
        else:
            selected_source = representative.detection_source

        duplicate_iou = 0.0
        merge_reason = "single_detection"
        if len(indices) > 1:
            reasons = []
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    pair = pair_reasons.get((min(indices[i], indices[j]), max(indices[i], indices[j])))
                    if pair:
                        duplicate_iou = max(duplicate_iou, pair[0])
                        reasons.append(pair[1])
            merge_reason = "+".join(sorted(set(reasons))) or "duplicate_group"

        representative.merge_group_id = f"MG{group_number:05d}"
        representative.merged_detection_count = len(members)
        representative.contributing_sources = "; ".join(sources)
        representative.selected_after_merge = True
        representative.selected_source = selected_source
        representative.selected_zone_id = representative.zone_id
        representative.duplicate_iou = round(float(duplicate_iou), 4)
        representative.merge_reason = merge_reason
        selected.append(representative)
    return sorted(selected, key=lambda c: c.merge_group_id)


def source_summary(candidates: list[DetectionCandidate]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for candidate in candidates:
        key = candidate.detection_source if not candidate.zone_id else f"zone:{candidate.zone_id}"
        summary[key] = summary.get(key, 0) + 1
    return dict(sorted(summary.items()))
