from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .utils import cosine_similarity, normalize_embedding


QUALITY_LABEL_RANK = {
    "unusable_reference": 0,
    "marginal_reference": 1,
    "usable_reference": 2,
}
QUALITY_LABEL_WEIGHT = {
    "unusable_reference": 0.25,
    "marginal_reference": 0.65,
    "usable_reference": 1.0,
}
AMBIGUITY_IOU_DELTA = 0.05
AMBIGUITY_CENTER_RATIO_DELTA = 0.25


@dataclass(frozen=True)
class TrackletConfig:
    min_observations: int = 2
    max_selected_observations: int = 5
    max_gap_seconds: float = 1.5
    min_iou: float = 0.10
    max_center_ratio: float = 1.25
    min_size_ratio: float = 0.50
    min_embedding_similarity: float = 0.25

    def __post_init__(self) -> None:
        if self.min_observations < 2:
            raise ValueError("min_observations must be at least 2 for multi-frame evidence")
        if self.max_selected_observations < self.min_observations:
            raise ValueError("max_selected_observations must be >= min_observations")
        if not math.isfinite(self.max_gap_seconds) or self.max_gap_seconds <= 0:
            raise ValueError("max_gap_seconds must be positive")
        if not 0 <= self.min_iou <= 1:
            raise ValueError("min_iou must be between 0 and 1")
        if not math.isfinite(self.max_center_ratio) or self.max_center_ratio <= 0:
            raise ValueError("max_center_ratio must be positive")
        if not 0 < self.min_size_ratio <= 1:
            raise ValueError("min_size_ratio must be in (0, 1]")
        if not -1 <= self.min_embedding_similarity <= 1:
            raise ValueError("min_embedding_similarity must be between -1 and 1")


@dataclass(frozen=True)
class TrackletObservation:
    observation_id: str
    frame_index: int
    timestamp_seconds: float
    bbox: tuple[float, float, float, float]
    detector_score: float
    embedding: np.ndarray | None
    quality_label: str
    quality_reasons: str
    landmark_valid: bool
    blur_score: float
    face_width: float
    face_height: float
    detection_source: str
    zone_id: str = ""
    zone_profile: str = ""
    contributing_sources: str = ""
    selected_source: str = ""
    merged_detection_count: int = 1
    frame_best_roll: str = ""
    frame_best_score: float = 0.0
    frame_second_roll: str = ""
    frame_second_score: float = 0.0
    frame_margin: float = 0.0
    frame_match_accepted: bool = False
    frame_match_reason: str = ""

    def __post_init__(self) -> None:
        if not self.observation_id:
            raise ValueError("observation_id is required")
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if not math.isfinite(self.timestamp_seconds) or self.timestamp_seconds < 0:
            raise ValueError("timestamp_seconds must be finite and non-negative")
        if len(self.bbox) != 4 or not all(math.isfinite(float(value)) for value in self.bbox):
            raise ValueError("bbox must contain four finite values")
        if self.bbox[2] <= 0 or self.bbox[3] <= 0:
            raise ValueError("bbox width and height must be positive")


@dataclass
class TrackletMember:
    observation: TrackletObservation
    association_iou: float = 0.0
    association_center_ratio: float = 0.0
    association_size_ratio: float = 1.0
    association_reason: str = "new_track"
    quality_weight: float = 0.0
    selected_for_aggregation: bool = False
    embedding_consistent: bool = False


@dataclass(frozen=True)
class FinalizedTracklet:
    tracklet_id: str
    members: tuple[TrackletMember, ...]
    eligible: bool
    rejection_reason: str
    aggregate_embedding: np.ndarray | None
    embedding_count: int
    medoid_member_id: str
    selected_member_ids: tuple[str, ...]
    consistent_member_ids: tuple[str, ...]
    inconsistent_member_ids: tuple[str, ...]
    pairwise_similarity_min: float | None
    pairwise_similarity_median: float | None

    @property
    def observation_count(self) -> int:
        return len(self.members)

    @property
    def member_ids(self) -> tuple[str, ...]:
        return tuple(member.observation.observation_id for member in self.members)

    @property
    def start_frame(self) -> int:
        return self.members[0].observation.frame_index

    @property
    def end_frame(self) -> int:
        return self.members[-1].observation.frame_index

    @property
    def start_seconds(self) -> float:
        return self.members[0].observation.timestamp_seconds

    @property
    def end_seconds(self) -> float:
        return self.members[-1].observation.timestamp_seconds


@dataclass
class _MutableTracklet:
    tracklet_id: str
    members: list[TrackletMember] = field(default_factory=list)

    @property
    def last_observation(self) -> TrackletObservation:
        return self.members[-1].observation


def _valid_embedding(embedding: np.ndarray | None) -> np.ndarray | None:
    if embedding is None:
        return None
    value = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if value.size == 0 or not np.all(np.isfinite(value)):
        return None
    if float(np.linalg.norm(value)) <= 1e-12:
        return None
    return normalize_embedding(value)


def observation_quality_weight(observation: TrackletObservation) -> float:
    label_factor = QUALITY_LABEL_WEIGHT.get(observation.quality_label, QUALITY_LABEL_WEIGHT["unusable_reference"])
    landmark_factor = 1.0 if observation.landmark_valid else 0.35
    size_factor = min(1.0, max(0.20, math.sqrt(max(observation.face_width * observation.face_height, 0.0)) / 64.0))
    blur_factor = min(1.0, max(0.20, math.log1p(max(observation.blur_score, 0.0)) / math.log1p(500.0)))
    detector_factor = min(1.0, max(0.20, (float(observation.detector_score) - 0.50) / 0.50))
    return max(0.001, label_factor * landmark_factor * size_factor * blur_factor * detector_factor)


def _quality_sort_key(member: TrackletMember) -> tuple:
    observation = member.observation
    return (
        -int(observation.landmark_valid),
        -QUALITY_LABEL_RANK.get(observation.quality_label, 0),
        -min(float(observation.face_width), float(observation.face_height)),
        -float(observation.blur_score),
        -float(observation.detector_score),
        observation.observation_id,
    )


def _box_iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    ix1, iy1 = max(lx, rx), max(ly, ry)
    ix2, iy2 = min(lx + lw, rx + rw), min(ly + lh, ry + rh)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(lw * lh + rw * rh - intersection, 1e-12)
    return float(intersection / union)


def _association_metrics(
    previous_bbox: tuple[float, float, float, float],
    current: TrackletObservation,
) -> tuple[float, float, float]:
    iou = _box_iou(previous_bbox, current.bbox)
    px, py, pw, ph = previous_bbox
    cx, cy, cw, ch = current.bbox
    distance = math.hypot((px + pw / 2.0) - (cx + cw / 2.0), (py + ph / 2.0) - (cy + ch / 2.0))
    scale = max(math.sqrt(pw * ph), math.sqrt(cw * ch), 1.0)
    center_ratio = float(distance / scale)
    previous_area = max(pw * ph, 1e-12)
    current_area = max(cw * ch, 1e-12)
    size_ratio = float(min(previous_area, current_area) / max(previous_area, current_area))
    return iou, center_ratio, size_ratio


def _predicted_bbox(tracklet: _MutableTracklet, timestamp_seconds: float) -> tuple[float, float, float, float]:
    latest = tracklet.last_observation
    if len(tracklet.members) < 2:
        return latest.bbox
    previous = tracklet.members[-2].observation
    elapsed = latest.timestamp_seconds - previous.timestamp_seconds
    horizon = timestamp_seconds - latest.timestamp_seconds
    if elapsed <= 1e-9 or horizon <= 0:
        return latest.bbox
    px, py, pw, ph = previous.bbox
    lx, ly, lw, lh = latest.bbox
    previous_center = (px + pw / 2.0, py + ph / 2.0)
    latest_center = (lx + lw / 2.0, ly + lh / 2.0)
    velocity_x = (latest_center[0] - previous_center[0]) / elapsed
    velocity_y = (latest_center[1] - previous_center[1]) / elapsed
    predicted_center_x = latest_center[0] + velocity_x * horizon
    predicted_center_y = latest_center[1] + velocity_y * horizon
    return predicted_center_x - lw / 2.0, predicted_center_y - lh / 2.0, lw, lh


def _is_ambiguous_edge_group(edges: list[tuple[float, float, float, str, str, str]]) -> bool:
    if len(edges) < 2:
        return False
    ordered = sorted(edges)
    best, second = ordered[0], ordered[1]
    best_iou, second_iou = -best[0], -second[0]
    return (
        abs(best_iou - second_iou) <= AMBIGUITY_IOU_DELTA
        and abs(best[1] - second[1]) <= AMBIGUITY_CENTER_RATIO_DELTA
    )


def _pairwise_similarity_fields(embeddings: list[np.ndarray]) -> tuple[float | None, float | None]:
    similarities = [
        cosine_similarity(embeddings[left], embeddings[right])
        for left in range(len(embeddings))
        for right in range(left + 1, len(embeddings))
    ]
    if not similarities:
        return None, None
    return float(min(similarities)), float(np.median(np.asarray(similarities, dtype=np.float32)))


def _ineligible_tracklet(
    tracklet: _MutableTracklet,
    reason: str,
    embedding_count: int,
    selected_ids: tuple[str, ...] = (),
    medoid_id: str = "",
    consistent_ids: tuple[str, ...] = (),
    inconsistent_ids: tuple[str, ...] = (),
) -> FinalizedTracklet:
    return FinalizedTracklet(
        tracklet_id=tracklet.tracklet_id,
        members=tuple(tracklet.members),
        eligible=False,
        rejection_reason=reason,
        aggregate_embedding=None,
        embedding_count=embedding_count,
        medoid_member_id=medoid_id,
        selected_member_ids=selected_ids,
        consistent_member_ids=consistent_ids,
        inconsistent_member_ids=inconsistent_ids,
        pairwise_similarity_min=None,
        pairwise_similarity_median=None,
    )


def _finalize_tracklet(tracklet: _MutableTracklet, config: TrackletConfig) -> FinalizedTracklet:
    members = sorted(
        tracklet.members,
        key=lambda member: (
            member.observation.frame_index,
            member.observation.timestamp_seconds,
            member.observation.observation_id,
        ),
    )
    tracklet.members = members
    for member in members:
        member.quality_weight = observation_quality_weight(member.observation)

    embedding_members = [member for member in members if _valid_embedding(member.observation.embedding) is not None]
    if len(members) < config.min_observations:
        return _ineligible_tracklet(tracklet, "insufficient_observations", len(embedding_members))

    if len(embedding_members) < config.min_observations:
        return _ineligible_tracklet(tracklet, "insufficient_embeddings", len(embedding_members))

    selected = sorted(embedding_members, key=_quality_sort_key)[:config.max_selected_observations]
    for member in selected:
        member.selected_for_aggregation = True
    selected_ids = tuple(member.observation.observation_id for member in selected)
    normalized = {
        member.observation.observation_id: _valid_embedding(member.observation.embedding)
        for member in selected
    }

    medoid = sorted(
        selected,
        key=lambda member: (
            -float(np.mean([
                cosine_similarity(
                    normalized[member.observation.observation_id],
                    normalized[other.observation.observation_id],
                )
                for other in selected
            ])),
            _quality_sort_key(member),
        ),
    )[0]
    medoid_id = medoid.observation.observation_id
    medoid_embedding = normalized[medoid_id]
    consistent = [
        member
        for member in selected
        if member is medoid
        or cosine_similarity(normalized[member.observation.observation_id], medoid_embedding)
        >= config.min_embedding_similarity
    ]
    consistent_ids = tuple(member.observation.observation_id for member in consistent)
    inconsistent_ids = tuple(
        member.observation.observation_id for member in selected if member not in consistent
    )
    for member in consistent:
        member.embedding_consistent = True

    if len(consistent) < config.min_observations:
        return _ineligible_tracklet(
            tracklet,
            "insufficient_consistent_embeddings",
            len(embedding_members),
            selected_ids,
            medoid_id,
            consistent_ids,
            inconsistent_ids,
        )

    weights = np.asarray([member.quality_weight for member in consistent], dtype=np.float32)
    vectors = np.stack([normalized[member.observation.observation_id] for member in consistent])
    aggregate = normalize_embedding(np.average(vectors, axis=0, weights=weights))
    if aggregate.size == 0 or not np.all(np.isfinite(aggregate)) or float(np.linalg.norm(aggregate)) <= 1e-12:
        return _ineligible_tracklet(
            tracklet,
            "invalid_aggregate_embedding",
            len(embedding_members),
            selected_ids,
            medoid_id,
            consistent_ids,
            inconsistent_ids,
        )

    similarity_min, similarity_median = _pairwise_similarity_fields([
        normalized[member.observation.observation_id] for member in consistent
    ])
    return FinalizedTracklet(
        tracklet_id=tracklet.tracklet_id,
        members=tuple(members),
        eligible=True,
        rejection_reason="ready",
        aggregate_embedding=aggregate,
        embedding_count=len(embedding_members),
        medoid_member_id=medoid_id,
        selected_member_ids=selected_ids,
        consistent_member_ids=consistent_ids,
        inconsistent_member_ids=inconsistent_ids,
        pairwise_similarity_min=similarity_min,
        pairwise_similarity_median=similarity_median,
    )


class TrackletBuilder:
    def __init__(self, config: TrackletConfig, id_prefix: str = "tracklet") -> None:
        self.config = config
        self.id_prefix = id_prefix or "tracklet"
        self._active: dict[str, _MutableTracklet] = {}
        self._completed: list[_MutableTracklet] = []
        self._next_id = 1
        self._last_frame = -1
        self._last_timestamp = -1.0
        self._seen_observation_ids: set[str] = set()
        self._finalized: tuple[FinalizedTracklet, ...] | None = None

    def _new_tracklet(self, observation: TrackletObservation, association_reason: str = "new_track") -> None:
        tracklet_id = f"{self.id_prefix}-TRK{self._next_id:05d}"
        self._next_id += 1
        self._active[tracklet_id] = _MutableTracklet(
            tracklet_id=tracklet_id,
            members=[TrackletMember(
                observation=observation,
                association_reason=association_reason,
                quality_weight=observation_quality_weight(observation),
            )],
        )

    def _expire_before(self, timestamp_seconds: float) -> None:
        expired_ids = [
            tracklet_id
            for tracklet_id, tracklet in self._active.items()
            if timestamp_seconds - tracklet.last_observation.timestamp_seconds > self.config.max_gap_seconds
        ]
        for tracklet_id in sorted(expired_ids):
            self._completed.append(self._active.pop(tracklet_id))

    def update(self, observations: list[TrackletObservation]) -> None:
        if self._finalized is not None:
            raise RuntimeError("Cannot update a finalized TrackletBuilder")
        if not observations:
            return

        ordered = sorted(observations, key=lambda item: item.observation_id)
        frame_indices = {item.frame_index for item in ordered}
        timestamps = {item.timestamp_seconds for item in ordered}
        if len(frame_indices) != 1 or len(timestamps) != 1:
            raise ValueError("Each update must contain observations from one frame and timestamp")
        frame_index = ordered[0].frame_index
        timestamp_seconds = ordered[0].timestamp_seconds
        if frame_index < self._last_frame or timestamp_seconds < self._last_timestamp:
            raise ValueError("Tracklet observations must be supplied in chronological order")
        if frame_index == self._last_frame and self._last_frame >= 0:
            raise ValueError("Each frame may be supplied only once")
        duplicate_ids = {item.observation_id for item in ordered} & self._seen_observation_ids
        if duplicate_ids:
            raise ValueError(f"Duplicate observation ids: {', '.join(sorted(duplicate_ids))}")
        self._seen_observation_ids.update(item.observation_id for item in ordered)
        self._last_frame = frame_index
        self._last_timestamp = timestamp_seconds
        self._expire_before(timestamp_seconds)

        edges: list[tuple[float, float, float, str, str, str]] = []
        observations_by_id = {item.observation_id: item for item in ordered}
        for tracklet_id, tracklet in sorted(self._active.items()):
            predicted_bbox = _predicted_bbox(tracklet, timestamp_seconds)
            for current in ordered:
                iou, center_ratio, size_ratio = _association_metrics(predicted_bbox, current)
                if size_ratio < self.config.min_size_ratio:
                    continue
                if iou < self.config.min_iou and center_ratio > self.config.max_center_ratio:
                    continue
                reason = "iou_overlap" if iou >= self.config.min_iou else "center_proximity"
                edges.append((-iou, center_ratio, -size_ratio, tracklet_id, current.observation_id, reason))

        edges_by_observation: dict[str, list[tuple[float, float, float, str, str, str]]] = {}
        edges_by_tracklet: dict[str, list[tuple[float, float, float, str, str, str]]] = {}
        for edge in edges:
            edges_by_tracklet.setdefault(edge[3], []).append(edge)
            edges_by_observation.setdefault(edge[4], []).append(edge)
        ambiguous_tracks = {
            tracklet_id for tracklet_id, candidates in edges_by_tracklet.items() if _is_ambiguous_edge_group(candidates)
        }
        ambiguous_observations = {
            observation_id for observation_id, candidates in edges_by_observation.items() if _is_ambiguous_edge_group(candidates)
        }
        ambiguous_observations.update(
            edge[4] for edge in edges if edge[3] in ambiguous_tracks
        )

        assigned_tracks: set[str] = set()
        assigned_observations: set[str] = set()
        for negative_iou, center_ratio, negative_size_ratio, tracklet_id, observation_id, reason in sorted(edges):
            if tracklet_id in ambiguous_tracks or observation_id in ambiguous_observations:
                continue
            if tracklet_id in assigned_tracks or observation_id in assigned_observations:
                continue
            observation = observations_by_id[observation_id]
            self._active[tracklet_id].members.append(TrackletMember(
                observation=observation,
                association_iou=-negative_iou,
                association_center_ratio=center_ratio,
                association_size_ratio=-negative_size_ratio,
                association_reason=reason,
                quality_weight=observation_quality_weight(observation),
            ))
            assigned_tracks.add(tracklet_id)
            assigned_observations.add(observation_id)

        for observation in ordered:
            if observation.observation_id not in assigned_observations:
                association_reason = (
                    "ambiguous_association_split"
                    if observation.observation_id in ambiguous_observations
                    else "new_track"
                )
                self._new_tracklet(observation, association_reason=association_reason)

    def finalize(self) -> tuple[FinalizedTracklet, ...]:
        if self._finalized is not None:
            return self._finalized
        self._completed.extend(self._active[tracklet_id] for tracklet_id in sorted(self._active))
        self._active.clear()
        self._finalized = tuple(
            _finalize_tracklet(tracklet, self.config)
            for tracklet in sorted(self._completed, key=lambda item: item.tracklet_id)
        )
        return self._finalized
