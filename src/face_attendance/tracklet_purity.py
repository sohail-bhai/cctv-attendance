from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path, PurePath
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .utils import cosine_similarity, normalize_embedding


POLICY_VERSION = "product-phase-2k-a-diagnostic-tracklet-purity-v1"
OUTPUT_SCHEMA_VERSION = 1

PURE_UNSPLIT = "pure_unsplit"
PURE_CHILD_CANDIDATE = "pure_child_candidate"
MIXED_QUARANTINED = "mixed_quarantined"
SPLIT_CANDIDATE = "split_candidate"
AMBIGUOUS_QUARANTINED = "ambiguous_quarantined"
INSUFFICIENT_OBSERVATIONS = "insufficient_observations"
INVALID_EMBEDDING_EVIDENCE = "invalid_embedding_evidence"
GEOMETRY_DISCONTINUITY_ONLY = "geometry_discontinuity_only"
APPEARANCE_DISCONTINUITY_ONLY = "appearance_discontinuity_only"
MULTI_SIGNAL_SPLIT = "multi_signal_split"


class TrackletPurityError(RuntimeError):
    """Raised when diagnostic purity evidence is invalid or immutable output drifted."""


@dataclass(frozen=True)
class PurityPolicy:
    """Conservative diagnostic policy; none of these values are recognition thresholds."""

    policy_version: str = POLICY_VERSION
    diagnostic_only: bool = True
    min_parent_observations: int = 3
    min_child_observations: int = 3
    max_split_boundaries: int = 1
    min_independent_split_signals: int = 2
    temporal_gap_seconds: float = 1.0
    temporal_gap_median_multiplier: float = 3.0
    geometry_max_iou: float = 0.05
    geometry_min_center_ratio: float = 1.25
    geometry_max_size_ratio_change: float = 0.50
    geometry_max_aspect_log_change: float = 0.35
    appearance_boundary_similarity_max: float = 0.25
    appearance_cluster_link_similarity: float = 0.70
    appearance_cluster_separation_max: float = 0.40
    min_significant_cluster_observations: int = 3
    min_secondary_cluster_share: float = 0.25
    pure_pairwise_median_min: float = 0.60
    identity_segment_min_share: float = 0.75
    pure_identity_min_share: float = 0.70
    max_invalid_embedding_fraction: float = 0.25
    global_match_threshold_reference: float = 0.48
    global_margin_threshold_reference: float = 0.08

    def __post_init__(self) -> None:
        if self.policy_version != POLICY_VERSION:
            raise ValueError("Phase 2K-A policy version must remain explicit")
        if not self.diagnostic_only:
            raise ValueError("Phase 2K-A must remain diagnostic-only")
        if self.min_parent_observations < 2 or self.min_child_observations < 2:
            raise ValueError("Purity observation minima must be at least two")
        if self.max_split_boundaries != 1:
            raise ValueError("Phase 2K-A permits at most one conservative split boundary")
        if self.min_independent_split_signals < 2:
            raise ValueError("A split must require multiple independent signals")
        for name in (
            "geometry_max_iou",
            "geometry_max_size_ratio_change",
            "appearance_cluster_link_similarity",
            "min_secondary_cluster_share",
            "pure_identity_min_share",
            "identity_segment_min_share",
            "max_invalid_embedding_fraction",
            "global_match_threshold_reference",
            "global_margin_threshold_reference",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")

    def as_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "recognition_thresholds_changed": False,
                "policy_basis": {
                    "temporal_and_geometry": (
                        "Derived conservatively from the existing 1.5-second, 0.10-IoU, "
                        "1.25-center-ratio and 0.50-size-ratio association contract; a split "
                        "requires multiple signals and stricter boundary evidence."
                    ),
                    "appearance": (
                        "The 0.25 boundary value matches the existing minimum association "
                        "similarity; 0.60 pairwise median and 0.70 identity share reuse reviewed "
                        "Phase 2H diagnostic evidence gates, not recognition acceptance gates."
                    ),
                    "strict_identity": (
                        "0.48/0.08 are read-only references. A child cannot pass unless an "
                        "independent strict evaluator supplies child-specific results."
                    ),
                },
            }
        )
        return payload


@dataclass(frozen=True)
class PurityObservation:
    observation_id: str
    frame_index: int
    timestamp_seconds: float
    bbox: tuple[float, float, float, float]
    predicted_roll: str = ""
    best_score: float | None = None
    margin: float | None = None
    accepted: bool = False
    selected: bool = False
    quality_label: str = ""
    embedding_extraction_success: bool | None = None
    embedding_consistent: bool | None = None
    embedding: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not self.observation_id:
            raise ValueError("observation_id is required")
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if not math.isfinite(self.timestamp_seconds) or self.timestamp_seconds < 0:
            raise ValueError("timestamp_seconds must be finite and non-negative")
        if len(self.bbox) != 4 or not all(math.isfinite(float(value)) for value in self.bbox):
            raise ValueError("bbox must contain four finite values")
        if float(self.bbox[2]) <= 0 or float(self.bbox[3]) <= 0:
            raise ValueError("bbox width and height must be positive")


@dataclass(frozen=True)
class TrackSummaryEvidence:
    pairwise_similarity_min: float | None = None
    pairwise_similarity_median: float | None = None
    selected_observation_count: int = 0
    consistent_embedding_count: int = 0
    aggregate_best_roll: str = ""
    aggregate_best_score: float | None = None
    aggregate_margin: float | None = None
    aggregate_accepted: bool = False


@dataclass(frozen=True)
class BoundaryEvidence:
    boundary_index: int
    left_observation_id: str
    right_observation_id: str
    left_timestamp_seconds: float
    right_timestamp_seconds: float
    left_count: int
    right_count: int
    time_gap_seconds: float
    bbox_iou: float
    center_ratio: float
    size_ratio: float
    aspect_log_change: float
    adjacent_embedding_similarity: float | None
    left_dominant_roll: str
    right_dominant_roll: str
    left_dominant_share: float
    right_dominant_share: float
    signals: tuple[str, ...]
    eligible_by_child_size: bool
    defensible_split: bool
    boundary_fingerprint_sha256: str


@dataclass(frozen=True)
class ChildLineage:
    parent_track_id: str
    child_track_id: str
    child_ordinal: int
    checkpoint_id: str
    camera_id: str
    source_video: str
    start_timestamp_seconds: float
    end_timestamp_seconds: float
    observation_ids: tuple[str, ...]
    split_reason: str
    policy_version: str
    source_fingerprint_sha256: str
    production_embedding_sha256: str
    evidence_fingerprint_sha256: str
    outcome: str
    dominant_roll: str
    dominant_share: float
    strict_gate_passed: bool
    strict_gate_reason: str
    strict_best_score: float | None
    strict_margin: float | None
    shadow_only: bool = True
    inherited_human_review: bool = False


@dataclass(frozen=True)
class TrackPurityResult:
    parent_track_id: str
    checkpoint_id: str
    camera_id: str
    source_video: str
    policy_version: str
    outcome: str
    split_disposition: str
    quarantined: bool
    observation_count: int
    start_timestamp_seconds: float | None
    end_timestamp_seconds: float | None
    parent_evidence_fingerprint_sha256: str
    raw_embedding_vectors_available: bool
    valid_embedding_count: int
    invalid_embedding_count: int
    pairwise_similarity_min: float | None
    pairwise_similarity_median: float | None
    significant_embedding_cluster_sizes: tuple[int, ...]
    significant_embedding_cluster_medoid_similarity: float | None
    dominant_roll: str
    dominant_share: float
    outlier_observation_ids: tuple[str, ...]
    reasons: tuple[str, ...]
    boundaries: tuple[BoundaryEvidence, ...]
    children: tuple[ChildLineage, ...]
    identity_vote_timeline: tuple[dict[str, Any], ...]
    geometry_timeline: tuple[dict[str, Any], ...]
    known_mixed_review: bool
    parent_official_checkpoint_contribution: int = 0
    review_carry_forward_allowed: bool = False


StrictEvaluator = Callable[[tuple[PurityObservation, ...]], Mapping[str, Any]]


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def canonical_json_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_relative_path(value: str | Path) -> str:
    text = str(value).replace("\\", "/")
    while "//" in text:
        text = text.replace("//", "/")
    return PurePath(text).as_posix()


def _valid_embedding(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    embedding = np.asarray(value, dtype=np.float32).reshape(-1)
    if embedding.size == 0 or not np.all(np.isfinite(embedding)):
        return None
    if float(np.linalg.norm(embedding)) <= 1e-12:
        return None
    return normalize_embedding(embedding)


def _bbox_metrics(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lx, ly, lw, lh = [float(value) for value in left]
    rx, ry, rw, rh = [float(value) for value in right]
    ix1, iy1 = max(lx, rx), max(ly, ry)
    ix2, iy2 = min(lx + lw, rx + rw), min(ly + lh, ry + rh)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(lw * lh + rw * rh - intersection, 1e-12)
    iou = float(intersection / union)
    distance = math.hypot((lx + lw / 2.0) - (rx + rw / 2.0), (ly + lh / 2.0) - (ry + rh / 2.0))
    center_ratio = float(distance / max(math.sqrt(lw * lh), math.sqrt(rw * rh), 1.0))
    left_area, right_area = max(lw * lh, 1e-12), max(rw * rh, 1e-12)
    size_ratio = float(min(left_area, right_area) / max(left_area, right_area))
    left_aspect, right_aspect = lw / max(lh, 1e-12), rw / max(rh, 1e-12)
    aspect_log_change = float(abs(math.log(max(right_aspect, 1e-12) / max(left_aspect, 1e-12))))
    return iou, center_ratio, size_ratio, aspect_log_change


def _dominant_vote(observations: Sequence[PurityObservation]) -> tuple[str, float, int]:
    counts: dict[str, int] = {}
    for item in observations:
        roll = str(item.predicted_roll or "").strip().upper()
        if roll:
            counts[roll] = counts.get(roll, 0) + 1
    if not counts:
        return "", 0.0, 0
    winner, count = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[0]
    return winner, float(count / len(observations)), count


def _embedding_clusters(
    observations: Sequence[PurityObservation],
    policy: PurityPolicy,
) -> tuple[tuple[int, ...], float | None, tuple[str, ...], int, int]:
    vectors: list[tuple[PurityObservation, np.ndarray]] = []
    invalid = 0
    for item in observations:
        vector = _valid_embedding(item.embedding)
        if vector is None:
            if item.embedding is not None or item.embedding_extraction_success is False:
                invalid += 1
            continue
        vectors.append((item, vector))
    if not vectors:
        return (), None, (), 0, invalid

    adjacency: dict[int, set[int]] = {index: set() for index in range(len(vectors))}
    for left in range(len(vectors)):
        for right in range(left + 1, len(vectors)):
            if cosine_similarity(vectors[left][1], vectors[right][1]) >= policy.appearance_cluster_link_similarity:
                adjacency[left].add(right)
                adjacency[right].add(left)
    components: list[list[int]] = []
    pending = set(adjacency)
    while pending:
        seed = min(pending)
        stack = [seed]
        component: list[int] = []
        pending.remove(seed)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor in pending:
                    pending.remove(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    components.sort(key=lambda group: (-len(group), [vectors[index][0].observation_id for index in group]))

    significant = [
        component
        for component in components
        if len(component) >= policy.min_significant_cluster_observations
        and len(component) / len(vectors) >= policy.min_secondary_cluster_share
    ]
    medoids: list[np.ndarray] = []
    for component in significant:
        medoid_index = sorted(
            component,
            key=lambda candidate: (
                -float(np.mean([cosine_similarity(vectors[candidate][1], vectors[other][1]) for other in component])),
                vectors[candidate][0].observation_id,
            ),
        )[0]
        medoids.append(vectors[medoid_index][1])
    medoid_similarity = None
    if len(medoids) >= 2:
        medoid_similarity = min(
            cosine_similarity(medoids[left], medoids[right])
            for left in range(len(medoids))
            for right in range(left + 1, len(medoids))
        )
    significant_members = {index for component in significant for index in component}
    outliers = tuple(
        vectors[index][0].observation_id
        for index in range(len(vectors))
        if index not in significant_members
    )
    return tuple(len(component) for component in significant), medoid_similarity, outliers, len(vectors), invalid


def _evidence_fingerprint(
    *,
    parent_track_id: str,
    checkpoint_id: str,
    camera_id: str,
    source_video: str,
    observations: Sequence[PurityObservation],
    source_fingerprint_sha256: str,
    production_embedding_sha256: str,
    boundary_fingerprint_sha256: str = "",
) -> str:
    payload = {
        "policy_version": POLICY_VERSION,
        "parent_track_id": parent_track_id,
        "checkpoint_id": checkpoint_id.upper(),
        "camera_id": camera_id,
        "source_video": canonical_relative_path(source_video),
        "observation_ids": [item.observation_id for item in observations],
        "frames": [item.frame_index for item in observations],
        "timestamps": [round(float(item.timestamp_seconds), 6) for item in observations],
        "source_fingerprint_sha256": source_fingerprint_sha256,
        "production_embedding_sha256": production_embedding_sha256,
        "boundary_fingerprint_sha256": boundary_fingerprint_sha256,
    }
    return canonical_json_hash(payload)


def _boundary_evidence(
    observations: Sequence[PurityObservation],
    index: int,
    policy: PurityPolicy,
    median_cadence: float,
) -> BoundaryEvidence:
    left, right = observations[index - 1], observations[index]
    gap = float(right.timestamp_seconds - left.timestamp_seconds)
    iou, center_ratio, size_ratio, aspect_change = _bbox_metrics(left.bbox, right.bbox)
    left_vector, right_vector = _valid_embedding(left.embedding), _valid_embedding(right.embedding)
    adjacent_similarity = None
    if left_vector is not None and right_vector is not None:
        adjacent_similarity = cosine_similarity(left_vector, right_vector)

    left_roll, left_share, _ = _dominant_vote(observations[:index])
    right_roll, right_share, _ = _dominant_vote(observations[index:])
    signals: list[str] = []
    dynamic_gap = max(policy.temporal_gap_seconds, median_cadence * policy.temporal_gap_median_multiplier)
    if gap >= dynamic_gap:
        signals.append("temporal_gap")
    geometry_jump = (
        (iou <= policy.geometry_max_iou and center_ratio >= policy.geometry_min_center_ratio)
        or size_ratio <= policy.geometry_max_size_ratio_change
        or aspect_change >= policy.geometry_max_aspect_log_change
    )
    if geometry_jump:
        signals.append("geometry_jump")
    if adjacent_similarity is not None and adjacent_similarity <= policy.appearance_boundary_similarity_max:
        signals.append("appearance_discontinuity")
    identity_transition = (
        left_roll
        and right_roll
        and left_roll != right_roll
        and left_share >= policy.identity_segment_min_share
        and right_share >= policy.identity_segment_min_share
    )
    if identity_transition:
        signals.append("identity_vote_transition")
    child_size_ok = index >= policy.min_child_observations and (len(observations) - index) >= policy.min_child_observations
    identity_or_appearance = bool({"appearance_discontinuity", "identity_vote_transition"}.intersection(signals))
    defensible = child_size_ok and len(signals) >= policy.min_independent_split_signals and identity_or_appearance
    fingerprint = canonical_json_hash(
        {
            "policy_version": policy.policy_version,
            "left": left.observation_id,
            "right": right.observation_id,
            "index": index,
            "signals": signals,
        }
    )
    return BoundaryEvidence(
        boundary_index=index,
        left_observation_id=left.observation_id,
        right_observation_id=right.observation_id,
        left_timestamp_seconds=left.timestamp_seconds,
        right_timestamp_seconds=right.timestamp_seconds,
        left_count=index,
        right_count=len(observations) - index,
        time_gap_seconds=gap,
        bbox_iou=iou,
        center_ratio=center_ratio,
        size_ratio=size_ratio,
        aspect_log_change=aspect_change,
        adjacent_embedding_similarity=adjacent_similarity,
        left_dominant_roll=left_roll,
        right_dominant_roll=right_roll,
        left_dominant_share=left_share,
        right_dominant_share=right_share,
        signals=tuple(signals),
        eligible_by_child_size=child_size_ok,
        defensible_split=defensible,
        boundary_fingerprint_sha256=fingerprint,
    )


def _segment_is_pure(observations: Sequence[PurityObservation], policy: PurityPolicy) -> tuple[bool, str, float]:
    if len(observations) < policy.min_child_observations:
        return False, "insufficient_child_observations", 0.0
    roll, share, _ = _dominant_vote(observations)
    if not roll or share < policy.pure_identity_min_share:
        return False, "unstable_child_identity_votes", share
    clusters, separation, _, _, invalid = _embedding_clusters(observations, policy)
    if len(clusters) >= 2 and separation is not None and separation <= policy.appearance_cluster_separation_max:
        return False, "multiple_significant_child_embedding_clusters", share
    declared = sum(1 for item in observations if item.embedding is not None or item.embedding_extraction_success is False)
    if declared and invalid / declared > policy.max_invalid_embedding_fraction:
        return False, "invalid_child_embedding_evidence", share
    return True, "independently_pure_diagnostic_segment", share


def _child_lineage(
    *,
    parent_track_id: str,
    checkpoint_id: str,
    camera_id: str,
    source_video: str,
    observations: Sequence[PurityObservation],
    ordinal: int,
    boundary: BoundaryEvidence,
    policy: PurityPolicy,
    source_fingerprint_sha256: str,
    production_embedding_sha256: str,
    strict_evaluator: StrictEvaluator | None,
) -> ChildLineage:
    fingerprint = _evidence_fingerprint(
        parent_track_id=parent_track_id,
        checkpoint_id=checkpoint_id,
        camera_id=camera_id,
        source_video=source_video,
        observations=observations,
        source_fingerprint_sha256=source_fingerprint_sha256,
        production_embedding_sha256=production_embedding_sha256,
        boundary_fingerprint_sha256=boundary.boundary_fingerprint_sha256,
    )
    child_id = f"{parent_track_id}--2KA-{boundary.boundary_fingerprint_sha256[:10]}-C{ordinal:02d}"
    pure, purity_reason, share = _segment_is_pure(observations, policy)
    dominant_roll, _, _ = _dominant_vote(observations)
    strict_passed = False
    strict_reason = "independent_strict_evaluation_unavailable"
    strict_score = None
    strict_margin = None
    if pure and strict_evaluator is not None:
        evaluation = dict(strict_evaluator(tuple(observations)))
        strict_score = _optional_float(evaluation.get("best_score"))
        strict_margin = _optional_float(evaluation.get("margin"))
        evaluated_roll = str(evaluation.get("predicted_roll") or "").strip().upper()
        evaluator_accepted = bool(evaluation.get("accepted"))
        strict_passed = bool(
            evaluator_accepted
            and evaluated_roll
            and evaluated_roll == dominant_roll
            and strict_score is not None
            and strict_score >= policy.global_match_threshold_reference
            and strict_margin is not None
            and strict_margin >= policy.global_margin_threshold_reference
        )
        strict_reason = "independent_strict_gates_passed" if strict_passed else "independent_strict_gates_failed"
    elif not pure:
        strict_reason = purity_reason
    return ChildLineage(
        parent_track_id=parent_track_id,
        child_track_id=child_id,
        child_ordinal=ordinal,
        checkpoint_id=checkpoint_id,
        camera_id=camera_id,
        source_video=canonical_relative_path(source_video),
        start_timestamp_seconds=observations[0].timestamp_seconds,
        end_timestamp_seconds=observations[-1].timestamp_seconds,
        observation_ids=tuple(item.observation_id for item in observations),
        split_reason=MULTI_SIGNAL_SPLIT,
        policy_version=policy.policy_version,
        source_fingerprint_sha256=source_fingerprint_sha256,
        production_embedding_sha256=production_embedding_sha256,
        evidence_fingerprint_sha256=fingerprint,
        outcome=PURE_CHILD_CANDIDATE if pure else AMBIGUOUS_QUARANTINED,
        dominant_roll=dominant_roll,
        dominant_share=share,
        strict_gate_passed=strict_passed,
        strict_gate_reason=strict_reason,
        strict_best_score=strict_score,
        strict_margin=strict_margin,
    )


def _optional_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def analyze_track_purity(
    *,
    parent_track_id: str,
    checkpoint_id: str,
    camera_id: str,
    source_video: str,
    observations: Sequence[PurityObservation],
    source_fingerprint_sha256: str,
    production_embedding_sha256: str,
    policy: PurityPolicy | None = None,
    summary: TrackSummaryEvidence | None = None,
    known_mixed_review: bool = False,
    strict_evaluator: StrictEvaluator | None = None,
) -> TrackPurityResult:
    policy = policy or PurityPolicy()
    summary = summary or TrackSummaryEvidence()
    members = tuple(observations)
    if len({item.observation_id for item in members}) != len(members):
        raise TrackletPurityError(f"Duplicate observation ids in {parent_track_id}")
    order_keys = [(item.timestamp_seconds, item.frame_index, item.observation_id) for item in members]
    if order_keys != sorted(order_keys):
        raise TrackletPurityError(f"Observations are not chronologically ordered for {parent_track_id}")
    if any(
        members[index].timestamp_seconds == members[index - 1].timestamp_seconds
        or members[index].frame_index == members[index - 1].frame_index
        for index in range(1, len(members))
    ):
        raise TrackletPurityError(f"Overlapping or duplicated observation times in {parent_track_id}")

    parent_fingerprint = _evidence_fingerprint(
        parent_track_id=parent_track_id,
        checkpoint_id=checkpoint_id,
        camera_id=camera_id,
        source_video=source_video,
        observations=members,
        source_fingerprint_sha256=source_fingerprint_sha256,
        production_embedding_sha256=production_embedding_sha256,
    )
    identity_timeline = tuple(
        {
            "observation_id": item.observation_id,
            "frame_index": item.frame_index,
            "timestamp_seconds": item.timestamp_seconds,
            "predicted_roll": str(item.predicted_roll or "").strip().upper(),
            "best_score": item.best_score,
            "margin": item.margin,
            "accepted": bool(item.accepted),
        }
        for item in members
    )
    geometry_timeline: list[dict[str, Any]] = []
    gaps = [members[index].timestamp_seconds - members[index - 1].timestamp_seconds for index in range(1, len(members))]
    median_cadence = float(np.median(np.asarray([gap for gap in gaps if gap > 0], dtype=np.float64))) if gaps else 0.0
    boundaries = tuple(
        _boundary_evidence(members, index, policy, median_cadence)
        for index in range(1, len(members))
    )
    for boundary in boundaries:
        geometry_timeline.append(
            {
                "left_observation_id": boundary.left_observation_id,
                "right_observation_id": boundary.right_observation_id,
                "time_gap_seconds": boundary.time_gap_seconds,
                "bbox_iou": boundary.bbox_iou,
                "center_ratio": boundary.center_ratio,
                "size_ratio": boundary.size_ratio,
                "aspect_log_change": boundary.aspect_log_change,
                "signals": list(boundary.signals),
            }
        )

    clusters, cluster_separation, outliers, valid_vectors, invalid_vectors = _embedding_clusters(members, policy)
    dominant_roll, dominant_share, _ = _dominant_vote(members)
    pairwise_min = summary.pairwise_similarity_min
    pairwise_median = summary.pairwise_similarity_median
    reasons: list[str] = []
    defensible = tuple(boundary for boundary in boundaries if boundary.defensible_split)
    signaled = tuple(boundary for boundary in boundaries if boundary.signals)
    children: tuple[ChildLineage, ...] = ()

    if len(members) < policy.min_parent_observations:
        outcome = INSUFFICIENT_OBSERVATIONS
        split_disposition = "not_evaluated"
        quarantined = True
        reasons.append("parent_has_too_few_observations")
    else:
        declared_embedding_evidence = sum(
            1 for item in members if item.embedding is not None or item.embedding_extraction_success is False
        )
        invalid_fraction = invalid_vectors / declared_embedding_evidence if declared_embedding_evidence else 0.0
        if declared_embedding_evidence and invalid_fraction > policy.max_invalid_embedding_fraction:
            outcome = INVALID_EMBEDDING_EVIDENCE
            split_disposition = "invalid_evidence_no_split"
            quarantined = True
            reasons.append("invalid_embedding_fraction_exceeded")
        elif len(defensible) == 1:
            boundary = defensible[0]
            left = members[: boundary.boundary_index]
            right = members[boundary.boundary_index :]
            proposed = (
                _child_lineage(
                    parent_track_id=parent_track_id,
                    checkpoint_id=checkpoint_id,
                    camera_id=camera_id,
                    source_video=source_video,
                    observations=left,
                    ordinal=1,
                    boundary=boundary,
                    policy=policy,
                    source_fingerprint_sha256=source_fingerprint_sha256,
                    production_embedding_sha256=production_embedding_sha256,
                    strict_evaluator=strict_evaluator,
                ),
                _child_lineage(
                    parent_track_id=parent_track_id,
                    checkpoint_id=checkpoint_id,
                    camera_id=camera_id,
                    source_video=source_video,
                    observations=right,
                    ordinal=2,
                    boundary=boundary,
                    policy=policy,
                    source_fingerprint_sha256=source_fingerprint_sha256,
                    production_embedding_sha256=production_embedding_sha256,
                    strict_evaluator=strict_evaluator,
                ),
            )
            if all(child.outcome == PURE_CHILD_CANDIDATE for child in proposed):
                children = proposed
                outcome = SPLIT_CANDIDATE
                split_disposition = MULTI_SIGNAL_SPLIT
                quarantined = True
                reasons.append("one_unique_multi_signal_boundary_with_stable_children")
            else:
                outcome = AMBIGUOUS_QUARANTINED
                split_disposition = "proposed_children_not_independently_pure"
                quarantined = True
                reasons.append("defensible_boundary_but_child_purity_failed")
        elif len(defensible) > policy.max_split_boundaries:
            outcome = AMBIGUOUS_QUARANTINED
            split_disposition = "multiple_defensible_boundaries"
            quarantined = True
            reasons.append("multiple_boundaries_fail_closed")
        elif known_mixed_review:
            outcome = MIXED_QUARANTINED
            split_disposition = "no_defensible_split"
            quarantined = True
            reasons.extend(("immutable_human_mixed_track_review", "parent_preserved_without_review_inheritance"))
        elif len(clusters) >= 2 and cluster_separation is not None and cluster_separation <= policy.appearance_cluster_separation_max:
            outcome = MIXED_QUARANTINED
            split_disposition = "multiple_embedding_clusters_without_unique_boundary"
            quarantined = True
            reasons.append("multiple_significant_separated_embedding_clusters")
        elif signaled:
            signal_union = {signal for boundary in signaled for signal in boundary.signals}
            if signal_union == {"geometry_jump"}:
                outcome = GEOMETRY_DISCONTINUITY_ONLY
                split_disposition = "single_signal_no_split"
            elif signal_union == {"appearance_discontinuity"}:
                outcome = APPEARANCE_DISCONTINUITY_ONLY
                split_disposition = "single_signal_no_split"
            else:
                outcome = AMBIGUOUS_QUARANTINED
                split_disposition = "insufficient_or_conflicting_boundary_support"
            quarantined = True
            reasons.append("no_unique_multi_signal_boundary")
        elif not dominant_roll or dominant_share < policy.pure_identity_min_share:
            outcome = AMBIGUOUS_QUARANTINED
            split_disposition = "unstable_identity_votes_no_split"
            quarantined = True
            reasons.append("dominant_identity_share_below_purity_floor")
        elif pairwise_median is not None and pairwise_median < policy.pure_pairwise_median_min:
            outcome = AMBIGUOUS_QUARANTINED
            split_disposition = "weak_aggregate_appearance_no_split"
            quarantined = True
            reasons.append("pairwise_embedding_median_below_purity_floor")
        else:
            outcome = PURE_UNSPLIT
            split_disposition = "stable_no_split"
            quarantined = False
            if outliers:
                reasons.append("isolated_embedding_outliers_recorded_without_split")
            else:
                reasons.append("stable_multi_signal_evidence")
            if valid_vectors == 0:
                reasons.append("raw_vectors_unavailable_used_derived_diagnostics_only")

    if known_mixed_review and outcome != SPLIT_CANDIDATE:
        outcome = MIXED_QUARANTINED
        split_disposition = "no_defensible_split" if not defensible else split_disposition
        quarantined = True
        if "immutable_human_mixed_track_review" not in reasons:
            reasons.append("immutable_human_mixed_track_review")
        children = ()

    return TrackPurityResult(
        parent_track_id=parent_track_id,
        checkpoint_id=checkpoint_id.upper(),
        camera_id=camera_id,
        source_video=canonical_relative_path(source_video),
        policy_version=policy.policy_version,
        outcome=outcome,
        split_disposition=split_disposition,
        quarantined=quarantined,
        observation_count=len(members),
        start_timestamp_seconds=members[0].timestamp_seconds if members else None,
        end_timestamp_seconds=members[-1].timestamp_seconds if members else None,
        parent_evidence_fingerprint_sha256=parent_fingerprint,
        raw_embedding_vectors_available=valid_vectors > 0,
        valid_embedding_count=valid_vectors,
        invalid_embedding_count=invalid_vectors,
        pairwise_similarity_min=pairwise_min,
        pairwise_similarity_median=pairwise_median,
        significant_embedding_cluster_sizes=clusters,
        significant_embedding_cluster_medoid_similarity=cluster_separation,
        dominant_roll=dominant_roll,
        dominant_share=dominant_share,
        outlier_observation_ids=outliers,
        reasons=tuple(dict.fromkeys(reasons)),
        boundaries=boundaries,
        children=children,
        identity_vote_timeline=identity_timeline,
        geometry_timeline=tuple(geometry_timeline),
        known_mixed_review=known_mixed_review,
    )


def review_carry_forward_decision(
    *,
    registry_accepted: bool,
    registry_mixed_quarantine: bool,
    existing_evidence_signature_sha256: str,
    current_evidence_signature_sha256: str,
    purity_result: TrackPurityResult,
    existing_purity_evidence_fingerprint_sha256: str = "",
    current_purity_evidence_fingerprint_sha256: str = "",
) -> dict[str, Any]:
    if registry_mixed_quarantine or purity_result.known_mixed_review:
        reason = "mixed_track_quarantine"
    elif existing_evidence_signature_sha256 != current_evidence_signature_sha256:
        reason = "evidence_signature_changed"
    elif (
        existing_purity_evidence_fingerprint_sha256
        or current_purity_evidence_fingerprint_sha256
    ) and existing_purity_evidence_fingerprint_sha256 != current_purity_evidence_fingerprint_sha256:
        reason = "purity_evidence_fingerprint_changed"
    elif purity_result.children or purity_result.outcome != PURE_UNSPLIT:
        reason = "lineage_or_purity_changed"
    elif not registry_accepted:
        reason = "review_not_accepted_for_carry_forward"
    else:
        return {
            "allowed": True,
            "reason": "exact_unchanged_evidence",
            "partial_carry_forward": False,
            "inherited_by_children": False,
        }
    return {
        "allowed": False,
        "reason": reason,
        "partial_carry_forward": False,
        "inherited_by_children": False,
    }


def csv_bytes(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for raw in rows:
        row = {}
        for field in fieldnames:
            value = raw.get(field, "")
            if isinstance(value, bool):
                value = "true" if value else "false"
            elif isinstance(value, (list, tuple, set)):
                value = ";".join(str(item) for item in value)
            elif value is None:
                value = ""
            row[field] = value
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8")


def verify_immutable_output(output_dir: Path) -> dict[str, Any]:
    root = Path(output_dir)
    manifest_path = root / "immutable_manifest.json"
    if not manifest_path.is_file():
        raise TrackletPurityError(f"Phase 2K-A immutable manifest missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise TrackletPurityError(f"Could not read Phase 2K-A immutable manifest: {exc}") from exc
    if manifest.get("schema_version") != OUTPUT_SCHEMA_VERSION:
        raise TrackletPurityError("Unsupported Phase 2K-A output schema")
    if manifest.get("policy_version") != POLICY_VERSION:
        raise TrackletPurityError("Phase 2K-A output policy changed")
    if manifest.get("run_id") != root.name:
        raise TrackletPurityError("Phase 2K-A run ID does not match its directory")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise TrackletPurityError("Phase 2K-A immutable manifest has no files")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "immutable_manifest.json"
    }
    if actual != set(files):
        raise TrackletPurityError("Phase 2K-A immutable output file set changed")
    for relative, metadata in files.items():
        path = root / Path(relative)
        if sha256_file(path) != str(metadata.get("sha256") or ""):
            raise TrackletPurityError(f"Phase 2K-A output hash changed: {relative}")
        if path.stat().st_size != int(metadata.get("size_bytes", -1)):
            raise TrackletPurityError(f"Phase 2K-A output size changed: {relative}")
    return dict(manifest)


def write_immutable_output(
    *,
    output_root: Path,
    run_id: str,
    files: Mapping[str, bytes],
) -> tuple[Path, bool]:
    root = Path(output_root)
    output = root / run_id
    if output.exists():
        verify_immutable_output(output)
        return output, True
    root.mkdir(parents=True, exist_ok=True)
    temp = root / f".{run_id}.tmp-{uuid.uuid4().hex}"
    if temp.exists():
        raise TrackletPurityError(f"Temporary Phase 2K-A path unexpectedly exists: {temp}")
    temp.mkdir(parents=False)
    try:
        normalized_files: dict[str, bytes] = {}
        for relative, payload in sorted(files.items()):
            normalized = canonical_relative_path(relative)
            if normalized.startswith("/") or ".." in PurePath(normalized).parts:
                raise TrackletPurityError(f"Unsafe Phase 2K-A output path: {relative}")
            if normalized == "immutable_manifest.json":
                raise TrackletPurityError("immutable_manifest.json is generated by the writer")
            normalized_files[normalized] = bytes(payload)
        for relative, payload in normalized_files.items():
            path = temp / Path(relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        manifest = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "policy_version": POLICY_VERSION,
            "run_id": run_id,
            "immutable": True,
            "files": {
                relative: {
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }
                for relative, payload in sorted(normalized_files.items())
            },
        }
        (temp / "immutable_manifest.json").write_bytes(canonical_json_bytes(manifest))
        temp.replace(output)
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise
    verify_immutable_output(output)
    return output, False
