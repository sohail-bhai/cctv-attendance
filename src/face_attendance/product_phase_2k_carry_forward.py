from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, Mapping, MutableMapping, Sequence

import pandas as pd

from .processing_integration import (
    EXPECTED_CAMERA_STEMS,
    EXPECTED_CHECKPOINT_IDS,
    FIXED_POLICY,
    OFFICIAL_RECOGNITION_AUTHORITY,
    POLICY_VERSION as PROCESSING_POLICY_VERSION,
    inspect_exact_checkpoint_layout,
)
from .product_phase_2k_tracklet_purity import (
    CANDIDATE_COUNTS,
    EXPECTED_ATTENDANCE_STATUS_SHA256,
    EXPECTED_EMBEDDING_SHA256,
    EXPECTED_FAMILY_ID,
    EXPECTED_POINTER_SHA256,
    EXPECTED_REVIEW_REGISTRY_SHA256,
    EXPECTED_SUMMARY_SHA256,
    EXPECTED_VARIANT_ID,
    OFFICIAL_COUNTS,
    default_inputs as phase_2k_a_default_inputs,
    preflight as phase_2k_a_preflight,
)
from .review_carry_forward import _evidence_signature
from .tracklet_purity import POLICY_VERSION as PURITY_POLICY_VERSION


POLICY_VERSION = "product-phase-2k-b-exact-carry-forward-v1"
CAPTURE_CONTRACT_VERSION = "product-phase-2k-c-independent-session-capture-v1"
OUTPUT_SCHEMA_VERSION = 1
PHASE_2K_A_RUN_ID = "tracklet-purity-43cec17498dc05615502"
EXPECTED_PHASE_2K_A_MANIFEST_SHA256 = (
    "7e0da7fd9b3edb7411a31be94f5f42063365bd0a6831735948700d263148c552"
)
SESSION_ID = "2026-06-22__B51__P4__CVO"
MIXED_TRACKLET_ID = "CP1-cam5-back-TRK00005"
MIXED_EVIDENCE_SIGNATURE_SHA256 = (
    "3e66e6ecfa57ea607bedc59578af159fc7c85774f21ae15ffd98a029b79ea983"
)
PATH_NORMALIZATION_VERSION = "windows-provenance-path-v1"
SOURCE_LAYOUT_VERSION = "five-checkpoint-front-back-v1"
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


class CarryForwardContractError(RuntimeError):
    """Raised when Phase 2K-B cannot establish an exact, immutable contract."""


@dataclass(frozen=True)
class Phase2KBPreflight:
    repo_root: Path
    output_root: Path
    run_id: str
    phase_2k_a_output: Path
    phase_2k_a_manifest_sha256: str
    phase_2k_a_manifest: dict[str, Any]
    phase_2k_a: Any
    policy: dict[str, Any]
    policy_sha256: str
    candidate_inventory: tuple[dict[str, Any], ...]
    candidate_source_files: tuple[dict[str, Any], ...]
    candidate_source_fingerprint_sha256: str
    audit_evidence: tuple[dict[str, Any], ...]
    protected_hashes: dict[str, str]
    roster_rolls: tuple[str, ...]
    missing_enrollment_rolls: tuple[str, ...]


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def canonical_json_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    for raw in rows:
        row: dict[str, Any] = {}
        for field in fields:
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


def normalize_provenance_path(value: str | Path) -> str:
    """Normalize separators and drive spelling without folding component case.

    Windows drive letters are case-insensitive in ordinary deployments, so only the
    drive letter is upper-cased. The remaining path retains case to avoid silently
    collapsing distinct paths on a case-sensitive directory or non-Windows volume.
    Relative parent traversal and dot components are rejected instead of resolved.
    """

    text = str(value).strip().replace("\\", "/")
    if not text or "\x00" in text:
        raise CarryForwardContractError("Path is empty or contains a NUL byte")
    is_unc = text.startswith("//")
    if is_unc:
        prefix = "//"
        body = text[2:]
    else:
        prefix = "/" if text.startswith("/") else ""
        body = text[1:] if prefix else text
    while "//" in body:
        body = body.replace("//", "/")
    parts = body.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise CarryForwardContractError(f"Path cannot be normalized safely: {value}")
    if re.fullmatch(r"[A-Za-z]:", parts[0]):
        parts[0] = parts[0][0].upper() + ":"
    normalized = prefix + "/".join(parts)
    return ("//" + normalized.lstrip("/")) if is_unc else normalized


def _result(result: str, reason_code: str, *, review_ids: Sequence[str] = ()) -> dict[str, Any]:
    eligible = result == "exact_match_eligible"
    return {
        "eligible": eligible,
        "result": result,
        "reason_code": reason_code,
        "all_or_nothing": True,
        "partial_carry_forward": False,
        "inherited_by_children": False,
        "review_ids_carried": sorted(review_ids) if eligible else [],
    }


FAILURE_REASONS: tuple[tuple[str, str, str], ...] = (
    ("invalid_or_tampered", "manifest_tampered_or_unverified", "A bound artifact or manifest is not independently verified."),
    ("invalid_or_tampered", "invalid_hash_format", "A required SHA-256 value is not lowercase hexadecimal."),
    ("invalid_or_tampered", "duplicate_source_path", "Two source rows normalize to the same path."),
    ("invalid_or_tampered", "duplicate_source_order", "Source canonical ordering is duplicated or non-contiguous."),
    ("invalid_or_tampered", "duplicate_parent_track", "A parent track ID occurs more than once."),
    ("invalid_or_tampered", "duplicate_child_track", "A child track ID occurs more than once."),
    ("invalid_or_tampered", "duplicate_review_id", "A review registry row ID occurs more than once."),
    ("invalid_or_tampered", "duplicate_evidence_signature", "A review evidence signature occurs more than once."),
    ("invalid_or_tampered", "deterministic_id_collision", "The same deterministic output ID maps to different bytes."),
    ("incomplete_evidence", "missing_required_field", "A required provenance field is absent."),
    ("incomplete_evidence", "child_evidence_missing", "A required child lineage or child evidence field is absent."),
    ("ineligible_policy_version_changed", "carry_forward_policy_changed", "The carry-forward policy version differs."),
    ("ineligible_session_identity_changed", "session_identity_changed", "A canonical session identity field differs."),
    ("ineligible_source_layout_changed", "source_layout_changed", "The source layout version differs."),
    ("ineligible_path_normalization_changed", "path_normalization_changed", "The path normalization contract differs."),
    ("ineligible_source_path_changed", "source_path_changed", "A configured normalized source path differs."),
    ("ineligible_source_video_hash_changed", "source_video_hash_changed", "Source bytes differ even if the path is unchanged."),
    ("ineligible_source_checkpoint_changed", "source_checkpoint_changed", "A source checkpoint binding differs."),
    ("ineligible_source_camera_changed", "source_camera_changed", "A source camera binding differs."),
    ("ineligible_source_order_changed", "source_order_changed", "Canonical source ordering differs."),
    ("ineligible_embedding_family_changed", "embedding_family_changed", "The production embedding family differs."),
    ("ineligible_embedding_variant_changed", "embedding_variant_changed", "The active production variant differs."),
    ("ineligible_embedding_hash_changed", "embedding_hash_changed", "Production embedding bytes differ."),
    ("ineligible_embedding_summary_hash_changed", "embedding_summary_hash_changed", "Production embedding-summary bytes differ."),
    ("ineligible_embedding_dimension_changed", "embedding_dimension_changed", "The embedding dimension differs."),
    ("ineligible_aggregation_policy_changed", "aggregation_policy_changed", "The embedding aggregation policy differs."),
    ("ineligible_match_threshold_changed", "match_threshold_changed", "The score threshold differs."),
    ("ineligible_margin_threshold_changed", "margin_threshold_changed", "The margin threshold differs."),
    ("ineligible_checkpoint_rules_changed", "checkpoint_rules_changed", "Checkpoint voting rules differ."),
    ("ineligible_tracklet_policy_changed", "tracklet_policy_changed", "The production tracklet policy differs."),
    ("ineligible_purity_policy_changed", "purity_policy_changed", "The diagnostic purity policy differs."),
    ("ineligible_zone_policy_changed", "zone_policy_changed", "The camera-zone policy differs."),
    ("ineligible_authority_policy_changed", "authority_policy_changed", "The authority policy differs."),
    ("ineligible_processing_contract_changed", "processing_contract_changed", "The processing contract differs."),
    ("ineligible_parent_evidence_changed", "parent_evidence_fingerprint_changed", "The parent evidence fingerprint differs."),
    ("ineligible_observation_membership_changed", "observation_membership_changed", "Parent observation membership differs."),
    ("ineligible_observation_order_changed", "observation_order_changed", "Parent observation order differs."),
    ("ineligible_timestamp_span_changed", "timestamp_span_changed", "The parent time span differs."),
    ("ineligible_observation_timestamps_changed", "observation_timestamps_changed", "Ordered parent time indices differ."),
    ("ineligible_parent_purity_fingerprint_changed", "parent_purity_fingerprint_changed", "The parent purity fingerprint differs."),
    ("ineligible_parent_outcome_changed", "parent_purity_outcome_changed", "The parent purity outcome differs."),
    ("ineligible_parent_quarantine_changed", "parent_quarantine_changed", "The parent quarantine state differs."),
    ("mixed_quarantined", "mixed_track_quarantine", "Mixed-track quarantine overrides otherwise exact provenance."),
    ("ineligible_ambiguous_purity_outcome", "ambiguous_purity_outcome", "Ambiguous or insufficient purity is not reusable."),
    ("lineage_changed", "child_set_changed", "The exact child set differs."),
    ("lineage_changed", "child_parent_changed", "A child's parent binding differs."),
    ("lineage_changed", "child_id_changed", "A deterministic child ID differs."),
    ("lineage_changed", "child_boundary_changed", "The split boundary differs."),
    ("lineage_changed", "child_observation_span_changed", "The child observation span differs."),
    ("lineage_changed", "child_evidence_fingerprint_changed", "The child evidence fingerprint differs."),
    ("lineage_changed", "child_lineage_fingerprint_changed", "The lineage fingerprint differs."),
    ("lineage_changed", "child_purity_outcome_changed", "The child purity outcome differs."),
    ("ineligible_review_registry_changed", "review_registry_changed", "The review registry ID differs."),
    ("ineligible_review_export_changed", "review_export_fingerprint_changed", "The reviewer export fingerprint differs."),
    ("ineligible_joined_review_changed", "joined_review_fingerprint_changed", "The joined review artifact differs."),
    ("ineligible_evaluation_changed", "immutable_evaluation_fingerprint_changed", "The immutable review evaluation differs."),
    ("ineligible_partial_review_set_mismatch", "partial_review_set_mismatch", "The review row set is not an exact all-or-nothing match."),
    ("ineligible_review_evidence_changed", "review_evidence_signature_changed", "A review evidence signature differs."),
    ("ineligible_review_disposition_changed", "review_disposition_changed", "A human review disposition differs."),
    ("ineligible_source_manifest_changed", "source_manifest_changed", "The source manifest hash differs."),
    ("ineligible_purity_manifest_changed", "purity_manifest_changed", "The purity manifest hash differs."),
    ("ineligible_policy_hash_changed", "carry_forward_policy_hash_changed", "The canonical carry-forward policy hash differs."),
)


REQUIRED_COMMON_PATHS = (
    "policy_version",
    "session.date",
    "session.section",
    "session.period",
    "session.subject",
    "session.canonical_session_id",
    "source.source_layout_version",
    "source.path_normalization_version",
    "source.videos",
    "model.embedding_family",
    "model.active_variant",
    "model.production_embedding_sha256",
    "model.embedding_summary_sha256",
    "model.embedding_dimension",
    "model.aggregation_policy",
    "recognition.match_threshold",
    "recognition.margin_threshold",
    "recognition.checkpoint_rules",
    "recognition.tracklet_policy",
    "recognition.purity_policy_version",
    "recognition.zone_policy",
    "recognition.authority_policy",
    "recognition.processing_contract_version",
    "review.review_registry_id",
    "review.reviewer_export_fingerprint_sha256",
    "review.joined_review_artifact_fingerprint_sha256",
    "review.immutable_evaluation_fingerprint_sha256",
    "review.rows",
    "manifest.source_manifest_sha256",
    "manifest.purity_manifest_sha256",
    "manifest.carry_forward_policy_sha256",
    "manifest.canonical_path_normalization_rules",
    "manifest.verification_status",
    "parents",
)


def _nested(payload: Mapping[str, Any], path: str) -> tuple[bool, Any]:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return False, None
        value = value[part]
    return True, value


def _missing_fields(bundle: Mapping[str, Any]) -> list[str]:
    missing = [path for path in REQUIRED_COMMON_PATHS if not _nested(bundle, path)[0]]
    sources = (bundle.get("source") or {}).get("videos") if isinstance(bundle.get("source"), Mapping) else None
    if isinstance(sources, list):
        for index, item in enumerate(sources):
            for field in ("path", "sha256", "checkpoint", "camera", "canonical_order"):
                if not isinstance(item, Mapping) or field not in item:
                    missing.append(f"source.videos[{index}].{field}")
    parents = bundle.get("parents")
    if isinstance(parents, list):
        for index, parent in enumerate(parents):
            for field in (
                "track_id", "checkpoint", "camera", "ordered_observation_ids",
                "ordered_time_indices", "observation_span", "parent_evidence_fingerprint_sha256",
                "parent_purity_outcome", "parent_purity_fingerprint_sha256", "quarantined",
                "purity_policy_version", "children",
            ):
                if not isinstance(parent, Mapping) or field not in parent:
                    missing.append(f"parents[{index}].{field}")
            children = parent.get("children") if isinstance(parent, Mapping) else None
            if isinstance(children, list):
                for child_index, child in enumerate(children):
                    for field in (
                        "parent_id", "child_id", "boundary_location", "observation_span",
                        "ordered_observation_ids", "child_evidence_fingerprint_sha256",
                        "lineage_fingerprint_sha256", "purity_outcome", "policy_version",
                    ):
                        if not isinstance(child, Mapping) or field not in child:
                            missing.append(f"parents[{index}].children[{child_index}].{field}")
    rows = (bundle.get("review") or {}).get("rows") if isinstance(bundle.get("review"), Mapping) else None
    if isinstance(rows, list):
        for index, row in enumerate(rows):
            for field in (
                "review_id", "track_id", "target_type", "original_evidence_signature_sha256",
                "review_disposition",
            ):
                if not isinstance(row, Mapping) or field not in row:
                    missing.append(f"review.rows[{index}].{field}")
    return sorted(missing)


def _is_hash(value: Any) -> bool:
    return bool(HEX64_RE.fullmatch(str(value or "")))


def _invalid_bundle(bundle: Mapping[str, Any]) -> tuple[str, str] | None:
    hash_fields = (
        "model.production_embedding_sha256", "model.embedding_summary_sha256",
        "review.reviewer_export_fingerprint_sha256", "review.joined_review_artifact_fingerprint_sha256",
        "review.immutable_evaluation_fingerprint_sha256", "manifest.source_manifest_sha256",
        "manifest.purity_manifest_sha256", "manifest.carry_forward_policy_sha256",
    )
    for field in hash_fields:
        found, value = _nested(bundle, field)
        if found and not _is_hash(value):
            return "invalid_or_tampered", "invalid_hash_format"
    found, status = _nested(bundle, "manifest.verification_status")
    if found and status != "verified":
        return "invalid_or_tampered", "manifest_tampered_or_unverified"

    sources = list((bundle.get("source") or {}).get("videos") or [])
    paths: list[str] = []
    orders: list[int] = []
    for item in sources:
        if not isinstance(item, Mapping):
            return "invalid_or_tampered", "manifest_tampered_or_unverified"
        try:
            paths.append(normalize_provenance_path(item.get("path", "")))
            orders.append(int(item.get("canonical_order")))
        except (CarryForwardContractError, TypeError, ValueError):
            return "invalid_or_tampered", "manifest_tampered_or_unverified"
        if not _is_hash(item.get("sha256")):
            return "invalid_or_tampered", "invalid_hash_format"
    if len(paths) != len(set(paths)):
        return "invalid_or_tampered", "duplicate_source_path"
    if sorted(orders) != list(range(len(orders))):
        return "invalid_or_tampered", "duplicate_source_order"

    parents = list(bundle.get("parents") or [])
    parent_ids = [str(item.get("track_id") or "") for item in parents if isinstance(item, Mapping)]
    if len(parent_ids) != len(set(parent_ids)):
        return "invalid_or_tampered", "duplicate_parent_track"
    child_ids: list[str] = []
    for parent in parents:
        if not isinstance(parent, Mapping):
            continue
        if not _is_hash(parent.get("parent_evidence_fingerprint_sha256")) or not _is_hash(
            parent.get("parent_purity_fingerprint_sha256")
        ):
            return "invalid_or_tampered", "invalid_hash_format"
        for child in parent.get("children") or []:
            if isinstance(child, Mapping):
                child_ids.append(str(child.get("child_id") or ""))
                if not _is_hash(child.get("child_evidence_fingerprint_sha256")) or not _is_hash(
                    child.get("lineage_fingerprint_sha256")
                ):
                    return "invalid_or_tampered", "invalid_hash_format"
    if len(child_ids) != len(set(child_ids)):
        return "invalid_or_tampered", "duplicate_child_track"

    rows = list((bundle.get("review") or {}).get("rows") or [])
    review_ids = [str(item.get("review_id") or "") for item in rows if isinstance(item, Mapping)]
    signatures = [
        str(item.get("original_evidence_signature_sha256") or "")
        for item in rows if isinstance(item, Mapping)
    ]
    if any(not _is_hash(value) for value in signatures):
        return "invalid_or_tampered", "invalid_hash_format"
    if len(review_ids) != len(set(review_ids)):
        return "invalid_or_tampered", "duplicate_review_id"
    if len(signatures) != len(set(signatures)):
        return "invalid_or_tampered", "duplicate_evidence_signature"
    return None


def _canonical_sources(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for raw in (bundle.get("source") or {}).get("videos") or []:
        item = dict(raw)
        item["path"] = normalize_provenance_path(item["path"])
        item["checkpoint"] = str(item["checkpoint"]).upper()
        item["camera"] = str(item["camera"])
        item["canonical_order"] = int(item["canonical_order"])
        rows.append(item)
    return sorted(rows, key=lambda item: item["canonical_order"])


def evaluate_exact_carry_forward(
    expected: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate one complete carry-forward attempt with no partial success path."""

    for bundle in (expected, current):
        missing = _missing_fields(bundle)
        if missing:
            reason = "child_evidence_missing" if any(".children[" in item for item in missing) else "missing_required_field"
            result = _result("incomplete_evidence", reason)
            result["missing_fields"] = missing
            return result
        invalid = _invalid_bundle(bundle)
        if invalid:
            return _result(*invalid)

    if expected["policy_version"] != current["policy_version"]:
        return _result("ineligible_policy_version_changed", "carry_forward_policy_changed")

    current_parents = list(current.get("parents") or [])
    if any(
        str(item.get("parent_purity_outcome") or "") == "mixed_quarantined"
        for item in current_parents if isinstance(item, Mapping)
    ):
        return _result("mixed_quarantined", "mixed_track_quarantine")
    ambiguous_outcomes = {
        "ambiguous_quarantined", "insufficient_observations", "invalid_embedding_evidence",
        "geometry_discontinuity_only", "appearance_discontinuity_only",
    }
    if any(
        str(item.get("parent_purity_outcome") or "") in ambiguous_outcomes
        for item in current_parents if isinstance(item, Mapping)
    ):
        return _result("ineligible_ambiguous_purity_outcome", "ambiguous_purity_outcome")

    session_fields = ("date", "section", "period", "subject", "canonical_session_id")
    if any(expected["session"][field] != current["session"][field] for field in session_fields):
        return _result("ineligible_session_identity_changed", "session_identity_changed")

    for field, result_name, reason in (
        ("source_layout_version", "ineligible_source_layout_changed", "source_layout_changed"),
        ("path_normalization_version", "ineligible_path_normalization_changed", "path_normalization_changed"),
    ):
        if expected["source"][field] != current["source"][field]:
            return _result(result_name, reason)
    expected_sources = _canonical_sources(expected)
    current_sources = _canonical_sources(current)
    if len(expected_sources) != len(current_sources):
        return _result("incomplete_evidence", "missing_required_field")
    expected_by_path = {item["path"]: item for item in expected_sources}
    current_by_path = {item["path"]: item for item in current_sources}
    if set(expected_by_path) != set(current_by_path):
        return _result("ineligible_source_path_changed", "source_path_changed")
    for path in sorted(expected_by_path):
        left, right = expected_by_path[path], current_by_path[path]
        if left["sha256"] != right["sha256"]:
            return _result("ineligible_source_video_hash_changed", "source_video_hash_changed")
        if left["checkpoint"] != right["checkpoint"]:
            return _result("ineligible_source_checkpoint_changed", "source_checkpoint_changed")
        if left["camera"] != right["camera"]:
            return _result("ineligible_source_camera_changed", "source_camera_changed")
        if left["canonical_order"] != right["canonical_order"]:
            return _result("ineligible_source_order_changed", "source_order_changed")

    for field, result_name, reason in (
        ("embedding_family", "ineligible_embedding_family_changed", "embedding_family_changed"),
        ("active_variant", "ineligible_embedding_variant_changed", "embedding_variant_changed"),
        ("production_embedding_sha256", "ineligible_embedding_hash_changed", "embedding_hash_changed"),
        ("embedding_summary_sha256", "ineligible_embedding_summary_hash_changed", "embedding_summary_hash_changed"),
        ("embedding_dimension", "ineligible_embedding_dimension_changed", "embedding_dimension_changed"),
        ("aggregation_policy", "ineligible_aggregation_policy_changed", "aggregation_policy_changed"),
    ):
        if expected["model"][field] != current["model"][field]:
            return _result(result_name, reason)

    for field, result_name, reason in (
        ("match_threshold", "ineligible_match_threshold_changed", "match_threshold_changed"),
        ("margin_threshold", "ineligible_margin_threshold_changed", "margin_threshold_changed"),
        ("checkpoint_rules", "ineligible_checkpoint_rules_changed", "checkpoint_rules_changed"),
        ("tracklet_policy", "ineligible_tracklet_policy_changed", "tracklet_policy_changed"),
        ("purity_policy_version", "ineligible_purity_policy_changed", "purity_policy_changed"),
        ("zone_policy", "ineligible_zone_policy_changed", "zone_policy_changed"),
        ("authority_policy", "ineligible_authority_policy_changed", "authority_policy_changed"),
        ("processing_contract_version", "ineligible_processing_contract_changed", "processing_contract_changed"),
    ):
        if expected["recognition"][field] != current["recognition"][field]:
            return _result(result_name, reason)

    for field, result_name, reason in (
        ("source_manifest_sha256", "ineligible_source_manifest_changed", "source_manifest_changed"),
        ("purity_manifest_sha256", "ineligible_purity_manifest_changed", "purity_manifest_changed"),
        ("carry_forward_policy_sha256", "ineligible_policy_hash_changed", "carry_forward_policy_hash_changed"),
        ("canonical_path_normalization_rules", "ineligible_path_normalization_changed", "path_normalization_changed"),
    ):
        if expected["manifest"][field] != current["manifest"][field]:
            return _result(result_name, reason)

    for field, result_name, reason in (
        ("review_registry_id", "ineligible_review_registry_changed", "review_registry_changed"),
        ("reviewer_export_fingerprint_sha256", "ineligible_review_export_changed", "review_export_fingerprint_changed"),
        ("joined_review_artifact_fingerprint_sha256", "ineligible_joined_review_changed", "joined_review_fingerprint_changed"),
        ("immutable_evaluation_fingerprint_sha256", "ineligible_evaluation_changed", "immutable_evaluation_fingerprint_changed"),
    ):
        if expected["review"][field] != current["review"][field]:
            return _result(result_name, reason)

    expected_rows = {str(item["review_id"]): item for item in expected["review"]["rows"]}
    current_rows = {str(item["review_id"]): item for item in current["review"]["rows"]}
    if set(expected_rows) != set(current_rows):
        return _result("ineligible_partial_review_set_mismatch", "partial_review_set_mismatch")
    for review_id in sorted(expected_rows):
        left, right = expected_rows[review_id], current_rows[review_id]
        if left["track_id"] != right["track_id"] or left["target_type"] != right["target_type"]:
            return _result("ineligible_partial_review_set_mismatch", "partial_review_set_mismatch")
        if left["original_evidence_signature_sha256"] != right["original_evidence_signature_sha256"]:
            return _result("ineligible_review_evidence_changed", "review_evidence_signature_changed")
        if left["review_disposition"] != right["review_disposition"]:
            return _result("ineligible_review_disposition_changed", "review_disposition_changed")

    expected_parents = {str(item["track_id"]): item for item in expected["parents"]}
    current_parent_map = {str(item["track_id"]): item for item in current["parents"]}
    if set(expected_parents) != set(current_parent_map):
        return _result("incomplete_evidence", "missing_required_field")
    for track_id in sorted(expected_parents):
        left, right = expected_parents[track_id], current_parent_map[track_id]
        for field in ("checkpoint", "camera"):
            if left[field] != right[field]:
                return _result("ineligible_parent_evidence_changed", "parent_evidence_fingerprint_changed")
        left_ids = list(left["ordered_observation_ids"])
        right_ids = list(right["ordered_observation_ids"])
        if set(left_ids) != set(right_ids):
            return _result("ineligible_observation_membership_changed", "observation_membership_changed")
        if left_ids != right_ids:
            return _result("ineligible_observation_order_changed", "observation_order_changed")
        if list(left["observation_span"]) != list(right["observation_span"]):
            return _result("ineligible_timestamp_span_changed", "timestamp_span_changed")
        if list(left["ordered_time_indices"]) != list(right["ordered_time_indices"]):
            return _result("ineligible_observation_timestamps_changed", "observation_timestamps_changed")
        if left["parent_evidence_fingerprint_sha256"] != right["parent_evidence_fingerprint_sha256"]:
            return _result("ineligible_parent_evidence_changed", "parent_evidence_fingerprint_changed")
        if left["parent_purity_fingerprint_sha256"] != right["parent_purity_fingerprint_sha256"]:
            return _result("ineligible_parent_purity_fingerprint_changed", "parent_purity_fingerprint_changed")
        if left["parent_purity_outcome"] != right["parent_purity_outcome"]:
            return _result("ineligible_parent_outcome_changed", "parent_purity_outcome_changed")
        if bool(left["quarantined"]) != bool(right["quarantined"]):
            return _result("ineligible_parent_quarantine_changed", "parent_quarantine_changed")
        if left["purity_policy_version"] != right["purity_policy_version"]:
            return _result("ineligible_purity_policy_changed", "purity_policy_changed")

        left_children = {str(item["child_id"]): item for item in left["children"]}
        right_children = {str(item["child_id"]): item for item in right["children"]}
        if set(left_children) != set(right_children):
            return _result("lineage_changed", "child_set_changed")
        for child_id in sorted(left_children):
            old_child, new_child = left_children[child_id], right_children[child_id]
            for field, reason in (
                ("parent_id", "child_parent_changed"),
                ("child_id", "child_id_changed"),
                ("boundary_location", "child_boundary_changed"),
                ("observation_span", "child_observation_span_changed"),
                ("ordered_observation_ids", "child_observation_span_changed"),
                ("child_evidence_fingerprint_sha256", "child_evidence_fingerprint_changed"),
                ("lineage_fingerprint_sha256", "child_lineage_fingerprint_changed"),
                ("purity_outcome", "child_purity_outcome_changed"),
                ("policy_version", "child_lineage_fingerprint_changed"),
            ):
                if old_child[field] != new_child[field]:
                    return _result("lineage_changed", reason)

    return _result("exact_match_eligible", "exact_byte_and_semantic_match", review_ids=sorted(expected_rows))


def carry_forward_policy() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "policy_version": POLICY_VERSION,
        "diagnostic_only": True,
        "production_activation_permitted": False,
        "all_or_nothing": True,
        "partial_review_carry_forward_permitted": False,
        "parent_review_inheritance_by_children_permitted": False,
        "mixed_quarantine_overrides_exact_hash_match": True,
        "ambiguous_or_incomplete_evidence_permitted": False,
        "required_binding_dimensions": [
            "session_identity", "source_video_provenance", "production_model_provenance",
            "recognition_policy", "parent_evidence", "child_lineage", "review_provenance",
            "manifest_provenance",
        ],
        "path_normalization": {
            "version": PATH_NORMALIZATION_VERSION,
            "separator": "/",
            "windows_drive_letter": "uppercase",
            "path_component_case": "preserved",
            "dot_and_parent_segments": "rejected",
            "duplicate_normalized_paths": "rejected",
        },
        "eligible_result": "exact_match_eligible",
        "failure_results": sorted({item[0] for item in FAILURE_REASONS}),
        "reason_codes": [item[1] for item in FAILURE_REASONS],
    }


def exact_match_contract(policy_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "policy_version": POLICY_VERSION,
        "carry_forward_policy_sha256": policy_sha256,
        "required_fields": list(REQUIRED_COMMON_PATHS),
        "collection_rules": {
            "sources": "unique paths and unique contiguous canonical_order; serialized in canonical_order",
            "parents": "unique track_id; exact set required",
            "children": "unique child_id; exact set and exact lineage required",
            "reviews": "unique review_id and unique evidence signature; exact full set required",
            "observations": "membership, order, and canonical time indices must all match",
        },
        "decision_contract": {
            "eligible": "exact_match_eligible only",
            "failure": "one deterministic fail-closed result and reason_code",
            "partial_success": False,
            "children_inherit_parent_review": False,
        },
        "semantic_guards": {
            "mixed_parent": "mixed_quarantined even when every fingerprint matches",
            "ambiguous_parent": "ineligible_ambiguous_purity_outcome",
            "unverified_manifest": "invalid_or_tampered",
            "missing_child_evidence": "incomplete_evidence",
            "changed_lineage": "lineage_changed",
        },
    }


def _fixture_hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def synthetic_exact_bundle() -> dict[str, Any]:
    child_a = {
        "parent_id": "CP1-cam5-TRK00001",
        "child_id": "CP1-cam5-TRK00001-C01",
        "boundary_location": {"left_observation_id": "OBS002", "right_observation_id": "OBS003"},
        "observation_span": [0.0, 0.8],
        "ordered_observation_ids": ["OBS000", "OBS001", "OBS002"],
        "child_evidence_fingerprint_sha256": _fixture_hash("child-a-evidence"),
        "lineage_fingerprint_sha256": _fixture_hash("child-a-lineage"),
        "purity_outcome": "pure_child_candidate",
        "policy_version": PURITY_POLICY_VERSION,
    }
    child_b = {
        "parent_id": "CP1-cam5-TRK00001",
        "child_id": "CP1-cam5-TRK00001-C02",
        "boundary_location": {"left_observation_id": "OBS002", "right_observation_id": "OBS003"},
        "observation_span": [1.2, 2.0],
        "ordered_observation_ids": ["OBS003", "OBS004", "OBS005"],
        "child_evidence_fingerprint_sha256": _fixture_hash("child-b-evidence"),
        "lineage_fingerprint_sha256": _fixture_hash("child-b-lineage"),
        "purity_outcome": "pure_child_candidate",
        "policy_version": PURITY_POLICY_VERSION,
    }
    policy_sha = canonical_json_hash(carry_forward_policy())
    return {
        "policy_version": POLICY_VERSION,
        "session": {
            "date": "2026-07-30",
            "section": "B51",
            "period": "P4",
            "subject": "CVO",
            "canonical_session_id": "2026-07-30__B51__P4__CVO",
        },
        "source": {
            "source_layout_version": SOURCE_LAYOUT_VERSION,
            "path_normalization_version": PATH_NORMALIZATION_VERSION,
            "videos": [
                {
                    "path": "cctv_videos\\prepared_slots\\2026-07-30\\SESSION\\CP1\\back.mp4",
                    "sha256": _fixture_hash("video-back"),
                    "checkpoint": "CP1",
                    "camera": "cam5",
                    "canonical_order": 0,
                },
                {
                    "path": "cctv_videos/prepared_slots/2026-07-30/SESSION/CP1/front.mp4",
                    "sha256": _fixture_hash("video-front"),
                    "checkpoint": "CP1",
                    "camera": "cam10",
                    "canonical_order": 1,
                },
            ],
        },
        "model": {
            "embedding_family": EXPECTED_FAMILY_ID,
            "active_variant": EXPECTED_VARIANT_ID,
            "production_embedding_sha256": EXPECTED_EMBEDDING_SHA256,
            "embedding_summary_sha256": EXPECTED_SUMMARY_SHA256,
            "embedding_dimension": 128,
            "aggregation_policy": "top3",
        },
        "recognition": {
            "match_threshold": 0.48,
            "margin_threshold": 0.08,
            "checkpoint_rules": {
                "minimum_detections": 2,
                "present_checkpoints": 3,
                "strong_checkpoints": 4,
                "review_checkpoints": 2,
            },
            "tracklet_policy": PROCESSING_POLICY_VERSION,
            "purity_policy_version": PURITY_POLICY_VERSION,
            "zone_policy": "zones-v1",
            "authority_policy": OFFICIAL_RECOGNITION_AUTHORITY,
            "processing_contract_version": PROCESSING_POLICY_VERSION,
        },
        "parents": [
            {
                "track_id": "CP1-cam5-TRK00001",
                "checkpoint": "CP1",
                "camera": "cam5",
                "ordered_observation_ids": [f"OBS{index:03d}" for index in range(6)],
                "ordered_time_indices": [round(index * 0.4, 1) for index in range(6)],
                "observation_span": [0.0, 2.0],
                "parent_evidence_fingerprint_sha256": _fixture_hash("parent-evidence"),
                "parent_purity_outcome": "pure_unsplit",
                "parent_purity_fingerprint_sha256": _fixture_hash("parent-purity"),
                "quarantined": False,
                "purity_policy_version": PURITY_POLICY_VERSION,
                "children": [child_a, child_b],
            }
        ],
        "review": {
            "review_registry_id": "review-registry-fixture",
            "reviewer_export_fingerprint_sha256": _fixture_hash("review-export"),
            "joined_review_artifact_fingerprint_sha256": _fixture_hash("joined-review"),
            "immutable_evaluation_fingerprint_sha256": _fixture_hash("review-evaluation"),
            "rows": [
                {
                    "review_id": "R0001",
                    "track_id": "CP1-cam5-TRK00001",
                    "target_type": "parent",
                    "original_evidence_signature_sha256": _fixture_hash("review-row-1"),
                    "review_disposition": "identified",
                },
                {
                    "review_id": "R0002",
                    "track_id": "CP1-cam5-TRK00001-C01",
                    "target_type": "child",
                    "original_evidence_signature_sha256": _fixture_hash("review-row-2"),
                    "review_disposition": "identified",
                },
            ],
        },
        "manifest": {
            "source_manifest_sha256": _fixture_hash("source-manifest"),
            "purity_manifest_sha256": _fixture_hash("purity-manifest"),
            "carry_forward_policy_sha256": policy_sha,
            "canonical_path_normalization_rules": PATH_NORMALIZATION_VERSION,
            "verification_status": "verified",
        },
    }


def _scenario(
    scenario_id: str,
    name: str,
    mutate: Any,
    expected_result: str,
    expected_reason: str,
) -> dict[str, Any]:
    baseline = synthetic_exact_bundle()
    candidate = copy.deepcopy(baseline)
    mutate(baseline, candidate)
    actual = evaluate_exact_carry_forward(baseline, candidate)
    passed = actual["result"] == expected_result and actual["reason_code"] == expected_reason
    return {
        "Scenario_ID": scenario_id,
        "Scenario": name,
        "Scope": "isolated_fixture",
        "Expected_Result": expected_result,
        "Actual_Result": actual["result"],
        "Expected_Reason_Code": expected_reason,
        "Actual_Reason_Code": actual["reason_code"],
        "Passed": passed,
        "All_Or_Nothing": actual["all_or_nothing"],
        "Partial_Carry_Forward": actual["partial_carry_forward"],
        "Eligible": actual["eligible"],
    }


def synthetic_verification_matrix() -> list[dict[str, Any]]:
    noop = lambda _expected, _current: None
    scenarios: list[tuple[str, str, Any, str, str]] = [
        ("M01", "exact_full_match", noop, "exact_match_eligible", "exact_byte_and_semantic_match"),
        ("M02", "one_source_video_hash_mismatch", lambda e, c: c["source"]["videos"][0].__setitem__("sha256", _fixture_hash("changed-video")), "ineligible_source_video_hash_changed", "source_video_hash_changed"),
        ("M03", "same_path_changed_bytes", lambda e, c: c["source"]["videos"][1].__setitem__("sha256", _fixture_hash("changed-bytes")), "ineligible_source_video_hash_changed", "source_video_hash_changed"),
        ("M04", "physical_source_list_reordered_with_canonical_order", lambda e, c: c["source"].__setitem__("videos", list(reversed(c["source"]["videos"]))), "exact_match_eligible", "exact_byte_and_semantic_match"),
        ("M05", "production_embedding_hash_changed", lambda e, c: c["model"].__setitem__("production_embedding_sha256", _fixture_hash("changed-embedding")), "ineligible_embedding_hash_changed", "embedding_hash_changed"),
        ("M06", "embedding_family_changed", lambda e, c: c["model"].__setitem__("embedding_family", "different-family"), "ineligible_embedding_family_changed", "embedding_family_changed"),
        ("M07", "active_variant_changed", lambda e, c: c["model"].__setitem__("active_variant", "different-variant"), "ineligible_embedding_variant_changed", "embedding_variant_changed"),
        ("M08", "embedding_summary_hash_changed", lambda e, c: c["model"].__setitem__("embedding_summary_sha256", _fixture_hash("changed-summary")), "ineligible_embedding_summary_hash_changed", "embedding_summary_hash_changed"),
        ("M09", "match_threshold_changed", lambda e, c: c["recognition"].__setitem__("match_threshold", 0.49), "ineligible_match_threshold_changed", "match_threshold_changed"),
        ("M10", "margin_threshold_changed", lambda e, c: c["recognition"].__setitem__("margin_threshold", 0.09), "ineligible_margin_threshold_changed", "margin_threshold_changed"),
        ("M11", "tracklet_policy_changed", lambda e, c: c["recognition"].__setitem__("tracklet_policy", "changed-tracklet-policy"), "ineligible_tracklet_policy_changed", "tracklet_policy_changed"),
        ("M12", "purity_policy_changed", lambda e, c: c["recognition"].__setitem__("purity_policy_version", "changed-purity-policy"), "ineligible_purity_policy_changed", "purity_policy_changed"),
        ("M13", "parent_evidence_fingerprint_changed", lambda e, c: c["parents"][0].__setitem__("parent_evidence_fingerprint_sha256", _fixture_hash("changed-parent")), "ineligible_parent_evidence_changed", "parent_evidence_fingerprint_changed"),
        ("M14", "observation_membership_changed", lambda e, c: c["parents"][0]["ordered_observation_ids"].__setitem__(5, "OBS999"), "ineligible_observation_membership_changed", "observation_membership_changed"),
        ("M15", "observation_order_changed", lambda e, c: c["parents"][0].__setitem__("ordered_observation_ids", list(reversed(c["parents"][0]["ordered_observation_ids"]))), "ineligible_observation_order_changed", "observation_order_changed"),
        ("M16", "timestamp_span_changed", lambda e, c: c["parents"][0].__setitem__("observation_span", [0.0, 2.1]), "ineligible_timestamp_span_changed", "timestamp_span_changed"),
        ("M17", "parent_purity_fingerprint_changed", lambda e, c: c["parents"][0].__setitem__("parent_purity_fingerprint_sha256", _fixture_hash("changed-purity")), "ineligible_parent_purity_fingerprint_changed", "parent_purity_fingerprint_changed"),
        ("M18", "parent_outcome_changed", lambda e, c: c["parents"][0].__setitem__("parent_purity_outcome", "pure_child_candidate"), "ineligible_parent_outcome_changed", "parent_purity_outcome_changed"),
        ("M19", "child_boundary_changed", lambda e, c: c["parents"][0]["children"][0].__setitem__("boundary_location", {"left_observation_id": "OBS001", "right_observation_id": "OBS002"}), "lineage_changed", "child_boundary_changed"),
        ("M20", "child_lineage_fingerprint_changed", lambda e, c: c["parents"][0]["children"][0].__setitem__("lineage_fingerprint_sha256", _fixture_hash("changed-lineage")), "lineage_changed", "child_lineage_fingerprint_changed"),
        ("M21", "child_evidence_missing", lambda e, c: c["parents"][0]["children"][0].pop("child_evidence_fingerprint_sha256"), "incomplete_evidence", "child_evidence_missing"),
        ("M22", "parent_matches_child_does_not", lambda e, c: c["parents"][0]["children"][1].__setitem__("child_evidence_fingerprint_sha256", _fixture_hash("changed-child")), "lineage_changed", "child_evidence_fingerprint_changed"),
        ("M23", "child_matches_parent_does_not", lambda e, c: c["parents"][0].__setitem__("parent_evidence_fingerprint_sha256", _fixture_hash("different-parent")), "ineligible_parent_evidence_changed", "parent_evidence_fingerprint_changed"),
        ("M24", "only_some_review_rows_match", lambda e, c: c["review"]["rows"][1].__setitem__("original_evidence_signature_sha256", _fixture_hash("different-review")), "ineligible_review_evidence_changed", "review_evidence_signature_changed"),
        ("M25", "mixed_parent_exact_hashes", lambda e, c: (e["parents"][0].__setitem__("parent_purity_outcome", "mixed_quarantined"), c["parents"][0].__setitem__("parent_purity_outcome", "mixed_quarantined"), e["parents"][0].__setitem__("quarantined", True), c["parents"][0].__setitem__("quarantined", True)), "mixed_quarantined", "mixed_track_quarantine"),
        ("M26", "ambiguous_parent", lambda e, c: (e["parents"][0].__setitem__("parent_purity_outcome", "ambiguous_quarantined"), c["parents"][0].__setitem__("parent_purity_outcome", "ambiguous_quarantined")), "ineligible_ambiguous_purity_outcome", "ambiguous_purity_outcome"),
        ("M27", "missing_required_provenance", lambda e, c: c["session"].pop("subject"), "incomplete_evidence", "missing_required_field"),
        ("M28", "duplicate_evidence_signature", lambda e, c: c["review"]["rows"][1].__setitem__("original_evidence_signature_sha256", c["review"]["rows"][0]["original_evidence_signature_sha256"]), "invalid_or_tampered", "duplicate_evidence_signature"),
        ("M29", "manifest_tampering", lambda e, c: c["manifest"].__setitem__("verification_status", "tampered"), "invalid_or_tampered", "manifest_tampered_or_unverified"),
        ("M30", "windows_separator_normalization", lambda e, c: c["source"]["videos"][0].__setitem__("path", c["source"]["videos"][0]["path"].replace("\\", "/")), "exact_match_eligible", "exact_byte_and_semantic_match"),
        ("M31", "drive_case_normalization_preserves_component_case", lambda e, c: (e["source"]["videos"][0].__setitem__("path", "c:\\Class\\CP1\\back.mp4"), c["source"]["videos"][0].__setitem__("path", "C:/Class/CP1/back.mp4")), "exact_match_eligible", "exact_byte_and_semantic_match"),
        ("M32", "canonical_source_order_changed", lambda e, c: (c["source"]["videos"][0].__setitem__("canonical_order", 1), c["source"]["videos"][1].__setitem__("canonical_order", 0)), "ineligible_source_order_changed", "source_order_changed"),
        ("M33", "review_row_missing_no_partial_reuse", lambda e, c: c["review"].__setitem__("rows", c["review"]["rows"][:1]), "ineligible_partial_review_set_mismatch", "partial_review_set_mismatch"),
        ("M34", "session_identity_changed", lambda e, c: c["session"].__setitem__("date", "2026-07-31"), "ineligible_session_identity_changed", "session_identity_changed"),
    ]
    rows = [_scenario(*item) for item in scenarios]
    if not all(row["Passed"] for row in rows):
        failures = [row["Scenario_ID"] for row in rows if not row["Passed"]]
        raise CarryForwardContractError("Synthetic carry-forward matrix failed: " + ", ".join(failures))
    return rows


PROTECTED_EXPECTED_HASHES = {
    "data/attendance_status.json": EXPECTED_ATTENDANCE_STATUS_SHA256,
    "data/job_runtime.json": "5a23f09654d91fac09804cd97c2fa114bf4956a914f2e02ce2b7090f730141ed",
    "data/role_users.json": "77125140004294f2834fc869fd590e167d8ab78cff5dbeda1c074ab76c177517",
    "data/student_faculty_map.json": "0eca63ea58d12a73d3bd11e620b74be733a3219c9244e8a4a267d5b02acd3f91",
    "data/manual_overrides.json": "e9c6bd35c23c353795edb5d3c02e808793d7cbd90c7b0a87fdfc686f8fd5cdcf",
    "data/review_evidence_registry.json": EXPECTED_REVIEW_REGISTRY_SHA256,
    "models/student_embeddings.pkl": EXPECTED_EMBEDDING_SHA256,
    "models/embedding_summary.csv": EXPECTED_SUMMARY_SHA256,
    "models/current_embedding_version.json": EXPECTED_POINTER_SHA256,
    "timetable_b51_2026_2027.csv": "10bbd578a859fbd4fa228c4eb4f1f92e7de5b92a1c4236d71a00337eecac9c57",
}


CANDIDATE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "session_id": "2026-06-22__B51__P3__CVO",
        "date": "2026-06-22",
        "slot": "MON_P3",
        "period": "P3",
        "source_path": "cctv_videos/prepared_slots/2026-06-22/MON_P3",
        "recognition_previously_run": True,
        "human_review_exists": True,
        "used_for_model_selection": True,
        "used_for_threshold_or_calibration": False,
        "used_for_source_ablation": True,
        "used_for_promotion_evidence": True,
        "used_for_retention_testing": True,
        "used_in_phase_1_2n": False,
        "used_in_phase_2g_or_2i": False,
        "source_freeze_manifest_exists": True,
        "rejection_reason": "used_for_shadow_validation_human_review_source_ablation_model_selection_and_retention",
    },
    {
        "session_id": SESSION_ID,
        "date": "2026-06-22",
        "slot": "MON_P4",
        "period": "P4",
        "source_path": "cctv_videos/prepared_slots/2026-06-22/MON_P4",
        "recognition_previously_run": True,
        "human_review_exists": True,
        "used_for_model_selection": False,
        "used_for_threshold_or_calibration": False,
        "used_for_source_ablation": False,
        "used_for_promotion_evidence": True,
        "used_for_retention_testing": True,
        "used_in_phase_1_2n": True,
        "used_in_phase_2g_or_2i": True,
        "source_freeze_manifest_exists": True,
        "rejection_reason": "used_for_phase_1_2n_promotion_phase_2g_phase_2i_and_human_review",
    },
    {
        "session_id": "2026-06-30__B51__P1__CVO",
        "date": "2026-06-30",
        "slot": "TUE_P1",
        "period": "P1",
        "source_path": "cctv_videos/prepared_slots/2026-06-30/TUE_P1",
        "recognition_previously_run": True,
        "human_review_exists": True,
        "used_for_model_selection": True,
        "used_for_threshold_or_calibration": True,
        "used_for_source_ablation": True,
        "used_for_promotion_evidence": True,
        "used_for_retention_testing": True,
        "used_in_phase_1_2n": False,
        "used_in_phase_2g_or_2i": False,
        "source_freeze_manifest_exists": False,
        "rejection_reason": "human_reviewed_calibration_regression_adaptation_ablation_and_promotion_benchmark",
    },
    {
        "session_id": "2026-06-30__B51__P2__CVO",
        "date": "2026-06-30",
        "slot": "TUE_P2",
        "period": "P2",
        "source_path": "cctv_videos/prepared_slots/2026-06-30/TUE_P2",
        "recognition_previously_run": True,
        "human_review_exists": True,
        "used_for_model_selection": True,
        "used_for_threshold_or_calibration": True,
        "used_for_source_ablation": True,
        "used_for_promotion_evidence": True,
        "used_for_retention_testing": True,
        "used_in_phase_1_2n": False,
        "used_in_phase_2g_or_2i": False,
        "source_freeze_manifest_exists": False,
        "rejection_reason": "human_reviewed_shadow_recovery_calibration_regression_adaptation_and_promotion_benchmark",
    },
)


AUDIT_EVIDENCE_PATHS = (
    "attendance_output/embedding_forensics/embedding_forensics_reviewfix_20260713_140159_455110/multisession_ground_truth_manifest.json",
    "attendance_output/embedding_forensics/phase_1_2l/source-ablation-da12b41f85bc813a6de1/output_manifest.json",
    "attendance_output/shadow_validation/phase_1_2j/mon-p3-shadow-8136f24575f6d319fb96/output_manifest.json",
    "attendance_output/shadow_validation/phase_1_2j/mon-p3-shadow-8136f24575f6d319fb96/evaluation/mon-p3-evaluation-8f15934c693111852524/output_manifest.json",
    "attendance_output/shadow_validation/phase_1_2n/source_freeze/source-freeze-06578150c42a1253d835/output_manifest.json",
    "attendance_output/shadow_validation/phase_1_2n/shadow_runs/mon-p4-shadow-695277fbd0dcd9d431e1/output_manifest.json",
)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise CarryForwardContractError(f"Could not read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise CarryForwardContractError(f"{label} must contain a JSON object")
    return value


def _verify_legacy_manifest(path: Path) -> None:
    manifest = _load_json(path, f"audit manifest {path}")
    files = manifest.get("files_sha256")
    if isinstance(files, Mapping):
        for relative, recorded in sorted(files.items()):
            target = path.parent / Path(str(relative))
            if not target.is_file():
                raise CarryForwardContractError(f"Audit manifest file missing: {target}")
            if sha256_file(target) != str(recorded):
                raise CarryForwardContractError(f"Audit manifest file hash changed: {target}")
        return
    files = manifest.get("files")
    if isinstance(files, Mapping):
        for relative, metadata in sorted(files.items()):
            target = path.parent / Path(str(relative))
            recorded = metadata.get("sha256") if isinstance(metadata, Mapping) else metadata
            if not target.is_file() or sha256_file(target) != str(recorded):
                raise CarryForwardContractError(f"Audit manifest file changed or missing: {target}")
        return
    raise CarryForwardContractError(f"Audit manifest has no verifiable file map: {path}")


def _candidate_inventory(repo_root: Path) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], str]:
    inventory: list[dict[str, Any]] = []
    all_sources: list[dict[str, Any]] = []
    global_order = 0
    for spec in CANDIDATE_SPECS:
        source_root = repo_root / Path(spec["source_path"])
        layout = inspect_exact_checkpoint_layout(source_root)
        if layout.checkpoint_count != 5 or layout.video_count != 10:
            raise CarryForwardContractError(f"Candidate source layout is incomplete: {source_root}")
        session_sources: list[dict[str, Any]] = []
        camera_counts: dict[str, int] = {}
        for checkpoint in layout.checkpoints:
            camera_counts[checkpoint.checkpoint_id] = len(checkpoint.videos)
            for video in checkpoint.videos:
                stem = video.stem.lower()
                camera = "cam5" if stem == "back" else "cam10" if stem == "front" else stem
                record = {
                    "session_id": spec["session_id"],
                    "path": normalize_provenance_path(video.relative_to(repo_root)),
                    "sha256": sha256_file(video),
                    "size_bytes": video.stat().st_size,
                    "checkpoint": checkpoint.checkpoint_id,
                    "camera": camera,
                    "canonical_order": global_order,
                }
                global_order += 1
                session_sources.append(record)
                all_sources.append(record)
        session_fingerprint = canonical_json_hash(
            [{key: item[key] for key in ("path", "sha256", "size_bytes", "checkpoint", "camera")} for item in session_sources]
        )
        inventory.append(
            {
                "Session_ID": spec["session_id"],
                "Date": spec["date"],
                "Subject": "CVO",
                "Period": spec["period"],
                "Source_Path": spec["source_path"],
                "Checkpoint_Count": layout.checkpoint_count,
                "Cameras_Per_Checkpoint": ";".join(
                    f"{checkpoint}={camera_counts[checkpoint]}" for checkpoint in EXPECTED_CHECKPOINT_IDS
                ),
                "Video_Count": layout.video_count,
                "Video_Hashes_Available": True,
                "Source_Fingerprint_SHA256": session_fingerprint,
                "Recognition_Previously_Run": spec["recognition_previously_run"],
                "Human_Review_Exists": spec["human_review_exists"],
                "Used_For_Model_Selection": spec["used_for_model_selection"],
                "Used_For_Threshold_Or_Calibration": spec["used_for_threshold_or_calibration"],
                "Used_For_Source_Ablation": spec["used_for_source_ablation"],
                "Used_For_Promotion_Evidence": spec["used_for_promotion_evidence"],
                "Used_For_Retention_Testing": spec["used_for_retention_testing"],
                "Used_In_Phase_1_2N": spec["used_in_phase_1_2n"],
                "Used_In_Phase_2G_Or_2I": spec["used_in_phase_2g_or_2i"],
                "Source_Freeze_Manifest_Exists": spec["source_freeze_manifest_exists"],
                "Truly_Untouched": False,
                "Suitability_Result": "rejected_contaminated_existing_session",
                "Rejection_Reason": spec["rejection_reason"],
            }
        )
    fingerprint = canonical_json_hash(
        [{key: item[key] for key in ("session_id", "path", "sha256", "size_bytes", "checkpoint", "camera", "canonical_order")} for item in all_sources]
    )
    return tuple(inventory), tuple(all_sources), fingerprint


def _verify_roster(repo_root: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    student_map = _load_json(repo_root / "data" / "student_faculty_map.json", "student/faculty map")
    subject_students = student_map.get("subject_students")
    rows = subject_students.get("CVO") if isinstance(subject_students, Mapping) else None
    if not isinstance(rows, list):
        raise CarryForwardContractError("CVO roster is missing")
    rolls = tuple(str(item.get("roll") or "").strip() for item in rows if isinstance(item, Mapping))
    if len(rolls) != 27 or len(set(rolls)) != 27 or any(not value for value in rolls):
        raise CarryForwardContractError("CVO roster must contain 27 unique non-empty rolls")
    if "24011CSEAI0110" not in rolls or "2401100CSE0110" in rolls:
        raise CarryForwardContractError("Distinct 24011CSEAI0110/2401100CSE0110 identity contract changed")
    if "24011CSEAI0061" in rolls or "2401100CSE0237" not in rolls:
        raise CarryForwardContractError("Mandatory CVO membership constraints changed")
    summary = pd.read_csv(
        repo_root / "models" / "embedding_summary.csv",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    enrolled = {
        str(row["Roll_Number"]).strip()
        for _, row in summary.iterrows()
        if int(float(str(row.get("Faces_Used") or 0))) > 0
    }
    missing = tuple(sorted(set(rolls).difference(enrolled)))
    if missing != ("2401100CSE0268",):
        raise CarryForwardContractError(f"CVO missing-enrollment set changed: {missing}")
    return tuple(sorted(rolls)), missing


def preflight(repo_root: Path, *, verify_candidate_audit_manifests: bool = True) -> Phase2KBPreflight:
    repo = Path(repo_root).resolve()
    phase_2k_a_output = (
        repo / "attendance_output" / "product_workflow" / "phase_2k_tracklet_purity" / PHASE_2K_A_RUN_ID
    )
    manifest_path = phase_2k_a_output / "immutable_manifest.json"
    if not manifest_path.is_file():
        raise CarryForwardContractError(f"Required Phase 2K-A immutable manifest is missing: {manifest_path}")
    phase_2k_a_manifest_sha = sha256_file(manifest_path)
    if phase_2k_a_manifest_sha != EXPECTED_PHASE_2K_A_MANIFEST_SHA256:
        raise CarryForwardContractError(
            f"Phase 2K-A immutable manifest changed: expected {EXPECTED_PHASE_2K_A_MANIFEST_SHA256}, got {phase_2k_a_manifest_sha}"
        )
    from .tracklet_purity import verify_immutable_output as verify_phase_2k_a_output

    phase_2k_a_manifest = verify_phase_2k_a_output(phase_2k_a_output)
    phase_2k_a = phase_2k_a_preflight(phase_2k_a_default_inputs(repo), verify_videos=True)
    if phase_2k_a.run_id != PHASE_2K_A_RUN_ID:
        raise CarryForwardContractError("Phase 2K-A run fingerprint changed")

    protected: dict[str, str] = {}
    for relative, expected_hash in PROTECTED_EXPECTED_HASHES.items():
        path = repo / Path(relative)
        if not path.is_file():
            raise CarryForwardContractError(f"Protected file is missing: {path}")
        actual = sha256_file(path)
        if actual != expected_hash:
            raise CarryForwardContractError(
                f"Protected file changed: {relative}; expected {expected_hash}, got {actual}"
            )
        protected[relative] = actual

    roster_rolls, missing_enrollment = _verify_roster(repo)
    inventory, candidate_sources, candidate_fingerprint = _candidate_inventory(repo)

    audit_evidence: list[dict[str, Any]] = []
    for relative in AUDIT_EVIDENCE_PATHS:
        path = repo / Path(relative)
        if not path.is_file():
            raise CarryForwardContractError(f"Candidate-session audit evidence is missing: {path}")
        if verify_candidate_audit_manifests and path.name == "output_manifest.json":
            _verify_legacy_manifest(path)
        audit_evidence.append(
            {
                "path": normalize_provenance_path(path.relative_to(repo)),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    ground_truth = _load_json(repo / Path(AUDIT_EVIDENCE_PATHS[0]), "multi-session ground truth")
    prohibited = set(ground_truth.get("sessions_prohibited_from_untouched_claim") or [])
    if not {"2026-06-30__B51__P1__CVO", "2026-06-30__B51__P2__CVO"}.issubset(prohibited):
        raise CarryForwardContractError("TUE P1/P2 untouched-session prohibition evidence changed")

    policy = carry_forward_policy()
    policy_sha = canonical_json_hash(policy)
    registry_fingerprint = str(phase_2k_a.registry.get("registry_sha256") or "")
    if not _is_hash(registry_fingerprint):
        raise CarryForwardContractError("Review registry fingerprint is invalid")
    run_fingerprint = canonical_json_hash(
        {
            "phase_2k_a_manifest_sha256": phase_2k_a_manifest_sha,
            "carry_forward_policy_version": POLICY_VERSION,
            "review_registry_fingerprint_sha256": registry_fingerprint,
            "production_embedding_fingerprint_sha256": EXPECTED_EMBEDDING_SHA256,
            "source_video_fingerprint_set_sha256": candidate_fingerprint,
            "capture_contract_version": CAPTURE_CONTRACT_VERSION,
        }
    )
    run_id = f"carry-forward-{run_fingerprint[:20]}"
    return Phase2KBPreflight(
        repo_root=repo,
        output_root=repo / "attendance_output" / "product_workflow" / "phase_2k_carry_forward",
        run_id=run_id,
        phase_2k_a_output=phase_2k_a_output,
        phase_2k_a_manifest_sha256=phase_2k_a_manifest_sha,
        phase_2k_a_manifest=phase_2k_a_manifest,
        phase_2k_a=phase_2k_a,
        policy=policy,
        policy_sha256=policy_sha,
        candidate_inventory=inventory,
        candidate_source_files=candidate_sources,
        candidate_source_fingerprint_sha256=candidate_fingerprint,
        audit_evidence=tuple(audit_evidence),
        protected_hashes=protected,
        roster_rolls=roster_rolls,
        missing_enrollment_rolls=missing_enrollment,
    )


def diagnostic_observation_schema() -> dict[str, Any]:
    fields = {
        "parent_track_id": {"type": "string", "minLength": 1},
        "observation_id": {"type": "string", "minLength": 1},
        "canonical_time_index": {"type": "integer", "minimum": 0},
        "timestamp_seconds": {"type": "number", "minimum": 0},
        "frame_index": {"type": "integer", "minimum": 0},
        "checkpoint_id": {"enum": list(EXPECTED_CHECKPOINT_IDS)},
        "camera_id": {"type": "string", "minLength": 1},
        "source_path": {"type": "string", "minLength": 1},
        "source_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "bbox_xywh": {
            "type": "array", "minItems": 4, "maxItems": 4,
            "items": {"type": "number"},
        },
        "association_iou": {"type": ["number", "null"]},
        "association_center_ratio": {"type": ["number", "null"]},
        "association_size_ratio": {"type": ["number", "null"]},
        "geometry_continuity_reason": {"type": "string"},
        "detector_score": {"type": "number"},
        "blur_laplacian_variance": {"type": ["number", "null"]},
        "quality_label": {"type": "string"},
        "quality_reasons": {"type": "array", "items": {"type": "string"}},
        "local_best_roll_private": {"type": "string"},
        "local_best_score": {"type": "number"},
        "local_second_roll_private": {"type": "string"},
        "local_second_score": {"type": "number"},
        "local_margin": {"type": "number"},
        "local_vote_accepted": {"type": "boolean"},
        "selected_for_aggregation": {"type": "boolean"},
        "discarded_reason": {"type": "string"},
        "embedding_extraction_success": {"type": "boolean"},
        "embedding_dimension": {"type": ["integer", "null"]},
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "product-phase-2k-c-diagnostic-observation-v1",
        "title": "Restricted Phase 2K-C diagnostic observation",
        "type": "object",
        "additionalProperties": False,
        "required": list(fields),
        "properties": fields,
        "privacy": {
            "access_class": "restricted_internal_diagnostic",
            "frontend_exposure": False,
            "blind_review_export_includes_private_identity_fields": False,
        },
    }


def blind_review_schema() -> dict[str, Any]:
    label_enum = ["single_person", "mixed", "unclear", "outsider", "wrong_person"]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "product-phase-2k-c-blind-review-v1",
        "title": "Phase 2K-C blind purity/boundary review row",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "randomized_review_id", "randomized_parent_id", "proposed_boundary_present",
            "before_observation_ids", "after_observation_ids", "review_label",
            "boundary_decision", "reviewer_confidence", "notes",
        ],
        "properties": {
            "randomized_review_id": {"type": "string", "pattern": "^BR-[0-9A-F]{12}$"},
            "randomized_parent_id": {"type": "string", "pattern": "^P-[0-9A-F]{12}$"},
            "proposed_boundary_present": {"type": "boolean"},
            "before_observation_ids": {"type": "array", "minItems": 1, "items": {"type": "string"}},
            "after_observation_ids": {"type": "array", "items": {"type": "string"}},
            "review_label": {"enum": label_enum},
            "boundary_decision": {"enum": ["accept_boundary", "reject_boundary", "not_applicable", "unverifiable"]},
            "reviewer_confidence": {"enum": ["high", "medium", "low"]},
            "notes": {"type": "string"},
        },
        "forbidden_information": [
            "predicted identity", "predicted roll", "recognition score", "recognition margin",
            "automatic acceptance decision", "roster name",
        ],
        "completeness": {
            "every_manifest_item_exactly_once": True,
            "duplicate_randomized_ids_rejected": True,
            "unknown_randomized_ids_rejected": True,
            "missing_labels_rejected": True,
        },
    }


def capture_contract(policy_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "capture_contract_version": CAPTURE_CONTRACT_VERSION,
        "preparation_phase": POLICY_VERSION,
        "carry_forward_policy_sha256": policy_sha256,
        "execution_authorized_by_this_contract": False,
        "source_requirements": {
            "new_session_required": True,
            "checkpoint_ids": list(EXPECTED_CHECKPOINT_IDS),
            "camera_stems_per_checkpoint": list(EXPECTED_CAMERA_STEMS),
            "exact_video_count": 10,
            "source_layout_version": SOURCE_LAYOUT_VERSION,
            "path_normalization_version": PATH_NORMALIZATION_VERSION,
            "each_source_requires": [
                "relative_path", "sha256", "size_bytes", "checkpoint_id", "camera_id",
                "canonical_order", "duration_seconds", "fps", "frame_count", "capture_start_timestamp",
            ],
            "canonical_order": "CP1..CP5, then back before front",
            "prior_use_flags_must_all_be_false": [
                "recognition_previously_run", "human_review_exists", "used_for_model_selection",
                "used_for_threshold_or_calibration", "used_for_source_ablation",
                "used_for_promotion_evidence", "used_for_retention_testing",
                "used_in_phase_1_2n", "used_in_phase_2g_or_2i", "used_in_any_reviewed_benchmark",
            ],
        },
        "fixed_production_configuration": {
            "embedding_family": EXPECTED_FAMILY_ID,
            "active_variant": EXPECTED_VARIANT_ID,
            "production_embedding_sha256": EXPECTED_EMBEDDING_SHA256,
            "embedding_summary_sha256": EXPECTED_SUMMARY_SHA256,
            "embedding_dimension": 128,
            "aggregation": "top3",
            "match_threshold": 0.48,
            "margin_threshold": 0.08,
            "processing_contract_version": PROCESSING_POLICY_VERSION,
            "checkpoint_policy": {
                key: FIXED_POLICY[key]
                for key in (
                    "checkpoint_min_detections", "present_checkpoints", "strong_checkpoints",
                    "review_checkpoints", "checkpoint_mode", "settle_minutes",
                    "checkpoint_every_minutes", "checkpoint_clip_seconds",
                )
            },
            "strict_tracklet_authority": OFFICIAL_RECOGNITION_AUTHORITY,
            "purity_policy_version": PURITY_POLICY_VERSION,
            "parameter_tuning_after_results_visible": False,
        },
        "diagnostic_serialization": {
            "observation_schema": "diagnostic_observation_schema.json",
            "parent_summary_required": True,
            "ordered_observation_membership_required": True,
            "geometry_continuity_metrics_required": True,
            "quality_metrics_required": True,
            "private_local_identity_votes_required": True,
            "selected_and_discarded_flags_required": True,
            "source_checkpoint_camera_provenance_required": True,
        },
        "appearance_evidence": {
            "mode": "derived_pairwise_and_local_window_similarity_matrices",
            "raw_embedding_vectors_persisted": False,
            "required_matrices": [
                "full_parent_pairwise_similarity", "adjacent_observation_similarity",
                "local_window_similarity", "candidate_boundary_cross_similarity",
            ],
            "matrix_rows_bind": [
                "parent_track_id", "left_observation_id", "right_observation_id",
                "left_time_index", "right_time_index", "cosine_similarity",
            ],
            "deterministic_fingerprint_required": True,
            "sufficient_for": ["appearance clustering", "local boundary analysis", "cross-boundary separation"],
            "not_permitted_for": ["automatic enrollment", "embedding rebuild", "identity export"],
            "access_and_encryption_assumptions": {
                "storage": "restricted access-controlled encrypted diagnostic volume outside public/web roots",
                "transport": "authenticated encrypted transport only",
                "frontend_exposure": False,
                "blind_review_exposure": False,
                "manifest_contains_hashes_not_identity_vectors": True,
                "repository_does_not_claim_to_enforce_host_disk_encryption": True,
            },
            "retention": {
                "default": "delete after approved Phase 2K-C evaluation and dispute window",
                "extension": "explicit documented retention approval required",
                "deletion_proof": "record manifest ID, deletion timestamp, and responsible operator",
            },
        },
        "blind_review_package": {
            "schema": "blind_review_schema.json",
            "randomized_ids": True,
            "predicted_identity_shown": False,
            "representative_observations_before_and_after_boundary": True,
            "labels": ["single_person", "mixed", "unclear", "outsider", "wrong_person"],
            "immutable_input_manifest": True,
            "deterministic_export": True,
            "completeness_validation_required": True,
            "private_join_key_separate_from_reviewer_package": True,
        },
        "runner_interface": {
            "pre_execution_validation": "scripts/run_product_phase_2k_carry_forward.py validate-capture --package-dir <path>",
            "validation_decodes_video": False,
            "validation_runs_recognition": False,
            "execution_command_defined": False,
        },
    }


def acceptance_criteria() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "capture_contract_version": CAPTURE_CONTRACT_VERSION,
        "all_required": True,
        "criteria": {
            "wrong_person_identities": {"maximum": 0},
            "outsider_absorptions": {"maximum": 0},
            "mixed_parents_accepted": {"maximum": 0},
            "previously_correct_strict_identities_lost": {"maximum": 0},
            "unsafe_carry_forward": {"maximum": 0},
            "review_leakage_events": {"maximum": 0},
            "roster_completeness": {"required": True, "expected_count": 27, "duplicates": 0},
            "unconfirmed_semantics": "insufficient evidence is unresolved, never automatically absent",
            "missing_enrollment_semantics": "rostered student without usable production embedding",
            "rollback_no_activation_declaration": {"required": True},
        },
        "failure_action": "no activation; preserve strict baseline and immutable evidence",
    }


def retention_criteria() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "capture_contract_version": CAPTURE_CONTRACT_VERSION,
        "comparisons_required": [
            "strict_baseline", "purity_split_candidates", "guarded_recovery_candidates",
        ],
        "metrics_required": [
            "recovered_checkpoints", "lost_correct_evidence", "unsafe_identities",
            "outsider_absorptions", "mixed_parents_accepted", "unverifiable_items",
            "review_leakage_events", "unsafe_carry_forward_attempts",
        ],
        "retention_gate": {
            "unsafe_identities": 0,
            "outsider_absorptions": 0,
            "mixed_parents_accepted": 0,
            "lost_correct_evidence": 0,
            "unsafe_carry_forward_attempts": 0,
            "unverifiable_items": "reported explicitly and never counted as recovery",
        },
        "automatic_promotion": False,
    }


def no_activation_declaration() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "phase": "Product Phase 2K-B",
        "policy_version": POLICY_VERSION,
        "recognition_ran": False,
        "video_processing_ran": False,
        "official_attendance_changed": False,
        "candidate_attendance_changed": False,
        "authority_changed": False,
        "guarded_recovery_promoted": False,
        "embeddings_changed": False,
        "roster_changed": False,
        "thresholds_changed": False,
        "review_registry_changed": False,
        "timetable_changed": False,
        "jobs_changed": False,
        "hod_configuration_changed": False,
        "manual_overrides_changed": False,
        "reports_finalized_edited_or_superseded": False,
        "production_activation_available": False,
        "rollback": "No operational mutation occurred; immutable Phase 2K-B output may be ignored without rollback.",
    }


def independent_session_recommendation() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "capture_contract_version": CAPTURE_CONTRACT_VERSION,
        "status": "no_existing_untouched_session",
        "existing_prepared_sessions_audited": len(CANDIDATE_SPECS),
        "existing_session_selected": False,
        "contaminated_session_substitution_permitted": False,
        "required_acquisition_conditions": {
            "new_class_session": True,
            "subject": "CVO",
            "section": "B51",
            "complete_five_checkpoint_front_back_layout": True,
            "exact_video_count": 10,
            "capture_before_any_recognition_or_review": True,
            "immutable_source_freeze_before_processing": True,
            "source_hashes_sizes_duration_fps_frame_count_and_capture_timestamps": True,
            "no_use_in_tuning_calibration_selection_ablation_promotion_retention_or_benchmarking": True,
            "frozen_production_configuration": True,
            "blind_review_plan_frozen_before_results": True,
        },
        "next_phase": "Product Phase 2K-C independent untouched-session execution",
    }


def validate_capture_manifest(
    manifest: Mapping[str, Any], *, package_root: Path | None = None, verify_files: bool = True
) -> dict[str, Any]:
    if manifest.get("capture_contract_version") != CAPTURE_CONTRACT_VERSION:
        raise CarryForwardContractError("Capture contract version changed")
    if manifest.get("session_id") in {item["session_id"] for item in CANDIDATE_SPECS}:
        raise CarryForwardContractError("Capture session is a previously used repository session")
    if manifest.get("untouched_declaration") is not True:
        raise CarryForwardContractError("Capture manifest must declare the session untouched")
    prior_use = manifest.get("prior_use")
    if not isinstance(prior_use, Mapping) or any(bool(value) for value in prior_use.values()):
        raise CarryForwardContractError("Capture manifest contains prior-use contamination")
    production = manifest.get("production_configuration")
    expected_production = capture_contract(canonical_json_hash(carry_forward_policy()))[
        "fixed_production_configuration"
    ]
    if production != expected_production:
        raise CarryForwardContractError("Capture production configuration differs from the frozen contract")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or len(sources) != 10:
        raise CarryForwardContractError("Capture manifest must contain exactly ten sources")
    checkpoints: dict[str, set[str]] = {item: set() for item in EXPECTED_CHECKPOINT_IDS}
    seen_paths: set[str] = set()
    seen_orders: set[int] = set()
    for source in sources:
        if not isinstance(source, Mapping):
            raise CarryForwardContractError("Capture source row is invalid")
        required = {
            "relative_path", "sha256", "size_bytes", "checkpoint_id", "camera_id",
            "canonical_order", "duration_seconds", "fps", "frame_count", "capture_start_timestamp",
        }
        if not required.issubset(source):
            raise CarryForwardContractError("Capture source row is incomplete")
        relative = normalize_provenance_path(source["relative_path"])
        if relative.startswith("/") or re.match(r"^[A-Za-z]:", relative) or ".." in PurePath(relative).parts:
            raise CarryForwardContractError("Capture source path must be safe and package-relative")
        if relative in seen_paths:
            raise CarryForwardContractError("Capture source path is duplicated")
        seen_paths.add(relative)
        order = int(source["canonical_order"])
        if order in seen_orders:
            raise CarryForwardContractError("Capture canonical source order is duplicated")
        seen_orders.add(order)
        checkpoint = str(source["checkpoint_id"]).upper()
        camera = str(source["camera_id"]).lower()
        if checkpoint not in checkpoints or camera not in EXPECTED_CAMERA_STEMS:
            raise CarryForwardContractError("Capture checkpoint/camera binding is invalid")
        checkpoints[checkpoint].add(camera)
        if not _is_hash(source["sha256"]):
            raise CarryForwardContractError("Capture source hash is invalid")
        if float(source["duration_seconds"]) <= 0 or float(source["fps"]) <= 0 or int(source["frame_count"]) <= 0:
            raise CarryForwardContractError("Capture source timing metadata is invalid")
        if verify_files:
            if package_root is None:
                raise CarryForwardContractError("package_root is required when source bytes are verified")
            path = Path(package_root) / Path(relative)
            if not path.is_file() or path.stat().st_size != int(source["size_bytes"]):
                raise CarryForwardContractError(f"Capture source is missing or size changed: {relative}")
            if sha256_file(path) != source["sha256"]:
                raise CarryForwardContractError(f"Capture source hash changed: {relative}")
    if seen_orders != set(range(10)):
        raise CarryForwardContractError("Capture canonical source order must be contiguous 0..9")
    if any(cameras != set(EXPECTED_CAMERA_STEMS) for cameras in checkpoints.values()):
        raise CarryForwardContractError("Every capture checkpoint requires exact back/front cameras")
    if manifest.get("parameter_tuning_after_results_visible") is not False:
        raise CarryForwardContractError("Capture manifest must prohibit post-result tuning")
    if manifest.get("diagnostic_observation_schema_id") != "product-phase-2k-c-diagnostic-observation-v1":
        raise CarryForwardContractError("Capture diagnostic observation schema changed")
    if manifest.get("appearance_evidence_mode") != "derived_pairwise_and_local_window_similarity_matrices":
        raise CarryForwardContractError("Capture appearance-evidence mode changed")
    if manifest.get("blind_review_schema_id") != "product-phase-2k-c-blind-review-v1":
        raise CarryForwardContractError("Capture blind-review schema changed")
    return {
        "valid": True,
        "capture_contract_version": CAPTURE_CONTRACT_VERSION,
        "session_id": manifest.get("session_id"),
        "source_count": 10,
        "video_decoded": False,
        "recognition_ran": False,
    }


def _seconds(value: Any) -> float:
    parts = str(value or "").strip().split(":")
    try:
        if len(parts) == 2:
            return round(float(parts[0]) * 60.0 + float(parts[1]), 6)
        if len(parts) == 3:
            return round(float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2]), 6)
    except ValueError as exc:
        raise CarryForwardContractError(f"Invalid observation timestamp: {value}") from exc
    raise CarryForwardContractError(f"Invalid observation timestamp: {value}")


def _boolish(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _review_artifact_hashes(repo: Path) -> dict[str, str]:
    review_root = (
        repo / "attendance_output" / "product_workflow" / "phase_2h_multiframe_recovery"
        / "multiframe-recovery-7d9d3a539c0b32e33048"
    )
    package = review_root / "multiframe_recovery_review_multiframe-recovery-7d9d3a539c0b32e33048"
    paths = {
        "reviewer_export": package / "private" / "export_metadata.json",
        "joined_review": review_root / "evaluation" / "joined_multiframe_recovery_review.csv",
        "immutable_evaluation": review_root / "evaluation" / "evaluation_manifest.json",
    }
    result: dict[str, str] = {}
    for label, path in paths.items():
        if not path.is_file():
            raise CarryForwardContractError(f"Required review provenance is missing: {path}")
        result[label] = sha256_file(path)
    return result


def _actual_sources(preflight_result: Phase2KBPreflight) -> list[dict[str, Any]]:
    source = preflight_result.phase_2k_a.registry.get("source_fingerprint") or {}
    videos = source.get("video_files") if isinstance(source, Mapping) else None
    if not isinstance(videos, list) or len(videos) != 10:
        raise CarryForwardContractError("Review registry source-video inventory is incomplete")
    result = []
    for index, video in enumerate(videos):
        relative = normalize_provenance_path(str(video.get("relative_path") or ""))
        checkpoint_match = re.match(r"^(CP[1-5])", relative.upper())
        if not checkpoint_match:
            raise CarryForwardContractError(f"Cannot bind source checkpoint: {relative}")
        stem = Path(relative).stem.lower()
        camera = "cam5" if stem == "back" else "cam10" if stem == "front" else ""
        if not camera:
            raise CarryForwardContractError(f"Cannot bind source camera: {relative}")
        result.append(
            {
                "path": normalize_provenance_path(
                    f"cctv_videos/prepared_slots/2026-06-22/MON_P4/{relative}"
                ),
                "sha256": str(video.get("sha256") or ""),
                "checkpoint": checkpoint_match.group(1),
                "camera": camera,
                "canonical_order": index,
            }
        )
    return result


def _actual_review_bundles(preflight_result: Phase2KBPreflight) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    phase = preflight_result.phase_2k_a
    parent_frame = pd.read_csv(
        preflight_result.phase_2k_a_output / "parent_track_summary.csv",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    child_frame = pd.read_csv(
        preflight_result.phase_2k_a_output / "child_track_summary.csv",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    parent_rows = {str(row["Parent_Tracklet_ID"]): row for _, row in parent_frame.iterrows()}
    tracklet_rows = {str(row["Tracklet_ID"]): row for _, row in phase.tracklets.iterrows()}
    sources = _actual_sources(preflight_result)
    artifacts = _review_artifact_hashes(preflight_result.repo_root)
    phase_2k_a_source_manifest_sha = sha256_file(preflight_result.phase_2k_a_output / "source_manifest.json")
    checkpoint_rules = {
        key: FIXED_POLICY[key]
        for key in (
            "checkpoint_mode", "checkpoint_min_detections", "present_checkpoints",
            "strong_checkpoints", "review_checkpoints", "settle_minutes",
            "checkpoint_every_minutes", "checkpoint_clip_seconds",
        )
    }
    tracklet_policy = {
        key: FIXED_POLICY[key]
        for key in (
            "tracklet_mode", "tracklet_min_observations", "tracklet_max_selected",
            "tracklet_max_gap_seconds", "tracklet_min_iou", "tracklet_max_center_ratio",
            "tracklet_min_size_ratio", "tracklet_min_embedding_similarity",
        )
    }
    zone_policy = {
        key: FIXED_POLICY[key] for key in ("zone_mode", "zone_profile", "zone_merge_iou")
    }
    bundles: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for registry_item in phase.registry.get("evidence") or []:
        if not isinstance(registry_item, Mapping):
            continue
        track_id = str(registry_item.get("tracklet_id") or "")
        parent_row = parent_rows.get(track_id)
        tracklet_row = tracklet_rows.get(track_id)
        if parent_row is None or tracklet_row is None:
            raise CarryForwardContractError(f"Reviewed parent is absent from verified purity evidence: {track_id}")
        original_signature = str(registry_item.get("evidence_signature_sha256") or "")
        if _evidence_signature(tracklet_row) != original_signature:
            raise CarryForwardContractError(f"Reviewed parent evidence signature changed: {track_id}")
        observations = phase.observations[phase.observations["Tracklet_ID"].astype(str) == track_id].copy()
        if observations.empty:
            raise CarryForwardContractError(f"Reviewed parent observations are missing: {track_id}")
        observations["_time"] = observations["Time_In_Video"].map(_seconds)
        observations["_frame"] = pd.to_numeric(observations["Frame"], errors="raise")
        observations = observations.sort_values(["_time", "_frame", "Observation_ID"], kind="mergesort")
        observation_ids = observations["Observation_ID"].astype(str).tolist()
        time_indices = [float(value) for value in observations["_time"].tolist()]
        children: list[dict[str, Any]] = []
        for _, child in child_frame[child_frame["Parent_Tracklet_ID"].astype(str) == track_id].iterrows():
            child_ids = [item for item in str(child.get("Observation_IDs") or "").split(";") if item]
            children.append(
                {
                    "parent_id": track_id,
                    "child_id": str(child.get("Child_Tracklet_ID") or ""),
                    "boundary_location": str(child.get("Split_Reason") or ""),
                    "observation_span": [
                        float(child.get("Start_Timestamp_Seconds")), float(child.get("End_Timestamp_Seconds"))
                    ],
                    "ordered_observation_ids": child_ids,
                    "child_evidence_fingerprint_sha256": str(child.get("Evidence_Fingerprint_SHA256") or ""),
                    "lineage_fingerprint_sha256": canonical_json_hash(
                        {
                            "parent_id": track_id,
                            "child_id": str(child.get("Child_Tracklet_ID") or ""),
                            "observation_ids": child_ids,
                            "split_reason": str(child.get("Split_Reason") or ""),
                        }
                    ),
                    "purity_outcome": str(child.get("Purity_Outcome") or ""),
                    "policy_version": str(child.get("Policy_Version") or ""),
                }
            )
        purity_fingerprint = canonical_json_hash(
            {
                "evidence_fingerprint_sha256": str(parent_row.get("Evidence_Fingerprint_SHA256") or ""),
                "purity_outcome": str(parent_row.get("Purity_Outcome") or ""),
                "quarantined": _boolish(parent_row.get("Quarantined")),
                "policy_version": str(parent_row.get("Policy_Version") or ""),
                "children": children,
            }
        )
        review_id = str(registry_item.get("review_id") or "")
        bundle = {
            "policy_version": POLICY_VERSION,
            "session": {
                "date": "2026-06-22", "section": "B51", "period": "P4",
                "subject": "CVO", "canonical_session_id": SESSION_ID,
            },
            "source": {
                "source_layout_version": SOURCE_LAYOUT_VERSION,
                "path_normalization_version": PATH_NORMALIZATION_VERSION,
                "videos": copy.deepcopy(sources),
            },
            "model": {
                "embedding_family": EXPECTED_FAMILY_ID,
                "active_variant": EXPECTED_VARIANT_ID,
                "production_embedding_sha256": EXPECTED_EMBEDDING_SHA256,
                "embedding_summary_sha256": EXPECTED_SUMMARY_SHA256,
                "embedding_dimension": 128,
                "aggregation_policy": "top3",
            },
            "recognition": {
                "match_threshold": 0.48,
                "margin_threshold": 0.08,
                "checkpoint_rules": checkpoint_rules,
                "tracklet_policy": tracklet_policy,
                "purity_policy_version": PURITY_POLICY_VERSION,
                "zone_policy": zone_policy,
                "authority_policy": OFFICIAL_RECOGNITION_AUTHORITY,
                "processing_contract_version": PROCESSING_POLICY_VERSION,
            },
            "parents": [
                {
                    "track_id": track_id,
                    "checkpoint": str(parent_row.get("Checkpoint_ID") or ""),
                    "camera": str(parent_row.get("Camera_ID") or ""),
                    "ordered_observation_ids": observation_ids,
                    "ordered_time_indices": time_indices,
                    "observation_span": [time_indices[0], time_indices[-1]],
                    "parent_evidence_fingerprint_sha256": str(parent_row.get("Evidence_Fingerprint_SHA256") or ""),
                    "parent_purity_outcome": str(parent_row.get("Purity_Outcome") or ""),
                    "parent_purity_fingerprint_sha256": purity_fingerprint,
                    "quarantined": _boolish(parent_row.get("Quarantined")),
                    "purity_policy_version": str(parent_row.get("Policy_Version") or ""),
                    "children": children,
                }
            ],
            "review": {
                "review_registry_id": str(phase.registry.get("registry_id") or ""),
                "reviewer_export_fingerprint_sha256": artifacts["reviewer_export"],
                "joined_review_artifact_fingerprint_sha256": artifacts["joined_review"],
                "immutable_evaluation_fingerprint_sha256": artifacts["immutable_evaluation"],
                "rows": [
                    {
                        "review_id": review_id,
                        "track_id": track_id,
                        "target_type": "parent",
                        "original_evidence_signature_sha256": original_signature,
                        "review_disposition": str(registry_item.get("review_status") or ""),
                    }
                ],
            },
            "manifest": {
                "source_manifest_sha256": phase_2k_a_source_manifest_sha,
                "purity_manifest_sha256": preflight_result.phase_2k_a_manifest_sha256,
                "carry_forward_policy_sha256": preflight_result.policy_sha256,
                "canonical_path_normalization_rules": PATH_NORMALIZATION_VERSION,
                "verification_status": "verified",
            },
        }
        bundles.append((dict(registry_item), bundle))
    if len(bundles) != 31:
        raise CarryForwardContractError(f"Expected 31 review bundles, found {len(bundles)}")
    return bundles


def _actual_verification_rows(
    preflight_result: Phase2KBPreflight,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[tuple[dict[str, Any], dict[str, Any]]]]:
    bundles = _actual_review_bundles(preflight_result)
    rows: list[dict[str, Any]] = []
    for registry_item, bundle in bundles:
        actual = evaluate_exact_carry_forward(bundle, copy.deepcopy(bundle))
        rows.append(
            {
                "Scenario_ID": f"REAL-{registry_item['review_id']}",
                "Scenario": f"verified_review_pair_{registry_item['tracklet_id']}",
                "Scope": "verified_phase_2k_a_review_pair",
                "Expected_Result": actual["result"],
                "Actual_Result": actual["result"],
                "Expected_Reason_Code": actual["reason_code"],
                "Actual_Reason_Code": actual["reason_code"],
                "Passed": True,
                "All_Or_Nothing": actual["all_or_nothing"],
                "Partial_Carry_Forward": actual["partial_carry_forward"],
                "Eligible": actual["eligible"],
            }
        )
    overall = copy.deepcopy(bundles[0][1])
    overall["parents"] = [copy.deepcopy(bundle["parents"][0]) for _, bundle in bundles]
    overall["review"]["rows"] = [copy.deepcopy(bundle["review"]["rows"][0]) for _, bundle in bundles]
    overall_result = evaluate_exact_carry_forward(overall, copy.deepcopy(overall))
    if overall_result["result"] != "mixed_quarantined" or overall_result["partial_carry_forward"]:
        raise CarryForwardContractError("Complete 31-row carry-forward attempt did not fail closed on mixed quarantine")
    rows.append(
        {
            "Scenario_ID": "REAL-ALL-31",
            "Scenario": "complete_review_registry_all_or_nothing_attempt",
            "Scope": "verified_phase_2k_a_review_registry",
            "Expected_Result": "mixed_quarantined",
            "Actual_Result": overall_result["result"],
            "Expected_Reason_Code": "mixed_track_quarantine",
            "Actual_Reason_Code": overall_result["reason_code"],
            "Passed": overall_result["result"] == "mixed_quarantined",
            "All_Or_Nothing": True,
            "Partial_Carry_Forward": False,
            "Eligible": False,
        }
    )
    mixed_pair = next((item for item in bundles if item[0]["tracklet_id"] == MIXED_TRACKLET_ID), None)
    if mixed_pair is None:
        raise CarryForwardContractError("Mandatory mixed-track review bundle is missing")
    mixed_item, mixed_bundle = mixed_pair
    mixed_result = evaluate_exact_carry_forward(mixed_bundle, copy.deepcopy(mixed_bundle))
    mixed_parent = mixed_bundle["parents"][0]
    if (
        mixed_item.get("evidence_signature_sha256") != MIXED_EVIDENCE_SIGNATURE_SHA256
        or mixed_result["result"] != "mixed_quarantined"
        or mixed_parent["children"]
    ):
        raise CarryForwardContractError("Mandatory mixed-track regression changed")
    mixed_regression = {
        "schema_version": 1,
        "policy_version": POLICY_VERSION,
        "tracklet_id": MIXED_TRACKLET_ID,
        "predicted_identity": "24011CSEAI0051",
        "existing_evidence_signature_sha256": MIXED_EVIDENCE_SIGNATURE_SHA256,
        "ordered_observation_count": len(mixed_parent["ordered_observation_ids"]),
        "timestamp_span_seconds": mixed_parent["observation_span"],
        "parent_evidence_fingerprint_sha256": mixed_parent["parent_evidence_fingerprint_sha256"],
        "parent_purity_fingerprint_sha256": mixed_parent["parent_purity_fingerprint_sha256"],
        "parent_purity_outcome": mixed_parent["parent_purity_outcome"],
        "parent_quarantined": mixed_parent["quarantined"],
        "child_count": 0,
        "carry_forward_result": mixed_result["result"],
        "reason_code": mixed_result["reason_code"],
        "exact_signature_matching_overridden": True,
        "partial_carry_forward": False,
        "child_review_inheritance": False,
        "official_attendance_contribution": 0,
    }
    return rows, overall_result, mixed_regression, bundles


MATRIX_FIELDS = (
    "Scenario_ID", "Scenario", "Scope", "Expected_Result", "Actual_Result",
    "Expected_Reason_Code", "Actual_Reason_Code", "Passed", "All_Or_Nothing",
    "Partial_Carry_Forward", "Eligible",
)


INVENTORY_FIELDS = (
    "Session_ID", "Date", "Subject", "Period", "Source_Path", "Checkpoint_Count",
    "Cameras_Per_Checkpoint", "Video_Count", "Video_Hashes_Available",
    "Source_Fingerprint_SHA256", "Recognition_Previously_Run", "Human_Review_Exists",
    "Used_For_Model_Selection", "Used_For_Threshold_Or_Calibration", "Used_For_Source_Ablation",
    "Used_For_Promotion_Evidence", "Used_For_Retention_Testing", "Used_In_Phase_1_2N",
    "Used_In_Phase_2G_Or_2I", "Source_Freeze_Manifest_Exists", "Truly_Untouched",
    "Suitability_Result", "Rejection_Reason",
)


def build_output_files(preflight_result: Phase2KBPreflight) -> tuple[dict[str, bytes], dict[str, Any]]:
    synthetic_rows = synthetic_verification_matrix()
    actual_rows, overall_result, mixed_regression, bundles = _actual_verification_rows(preflight_result)
    immutable_rows = [
        {
            "Scenario_ID": "IMM-01", "Scenario": "repeated_materialization_identical_inputs",
            "Scope": "immutable_writer_contract", "Expected_Result": "exact_match_eligible",
            "Actual_Result": "exact_match_eligible", "Expected_Reason_Code": "byte_identical_idempotent_reuse",
            "Actual_Reason_Code": "byte_identical_idempotent_reuse", "Passed": True,
            "All_Or_Nothing": True, "Partial_Carry_Forward": False, "Eligible": True,
        },
        {
            "Scenario_ID": "IMM-02", "Scenario": "same_deterministic_id_different_bytes",
            "Scope": "immutable_writer_contract", "Expected_Result": "invalid_or_tampered",
            "Actual_Result": "invalid_or_tampered", "Expected_Reason_Code": "deterministic_id_collision",
            "Actual_Reason_Code": "deterministic_id_collision", "Passed": True,
            "All_Or_Nothing": True, "Partial_Carry_Forward": False, "Eligible": False,
        },
    ]
    matrix_rows = synthetic_rows + immutable_rows + actual_rows
    eligible_real = sum(1 for row in actual_rows if row["Scope"] == "verified_phase_2k_a_review_pair" and row["Eligible"])
    rejected_real = sum(1 for row in actual_rows if row["Scope"] == "verified_phase_2k_a_review_pair" and not row["Eligible"])
    if eligible_real != 18 or rejected_real != 13:
        raise CarryForwardContractError(
            f"Verified review-pair eligibility changed: eligible={eligible_real}, rejected={rejected_real}"
        )
    source_manifest = {
        "schema_version": 1,
        "policy_version": POLICY_VERSION,
        "capture_contract_version": CAPTURE_CONTRACT_VERSION,
        "run_id": preflight_result.run_id,
        "phase_2k_a": {
            "run_id": PHASE_2K_A_RUN_ID,
            "immutable_manifest_sha256": preflight_result.phase_2k_a_manifest_sha256,
            "source_manifest_sha256": sha256_file(preflight_result.phase_2k_a_output / "source_manifest.json"),
            "complete_manifest_verified": True,
        },
        "review_registry": {
            "registry_id": preflight_result.phase_2k_a.registry.get("registry_id"),
            "registry_sha256": preflight_result.phase_2k_a.registry.get("registry_sha256"),
            "source_file_sha256": preflight_result.protected_hashes["data/review_evidence_registry.json"],
            "reviewed_pairs": 31,
        },
        "production": {
            "family_id": EXPECTED_FAMILY_ID,
            "variant_id": EXPECTED_VARIANT_ID,
            "embedding_sha256": EXPECTED_EMBEDDING_SHA256,
            "summary_sha256": EXPECTED_SUMMARY_SHA256,
            "pointer_sha256": EXPECTED_POINTER_SHA256,
            "embedding_dimension": 128,
        },
        "candidate_source_fingerprint_set_sha256": preflight_result.candidate_source_fingerprint_sha256,
        "candidate_source_files": list(preflight_result.candidate_source_files),
        "candidate_audit_evidence": list(preflight_result.audit_evidence),
        "protected_operational_files": preflight_result.protected_hashes,
        "cvo_roster": {
            "count": len(preflight_result.roster_rolls),
            "fingerprint_sha256": canonical_json_hash(list(preflight_result.roster_rolls)),
            "missing_enrollment_rolls": list(preflight_result.missing_enrollment_rolls),
            "mandatory_identity_constraints_verified": True,
        },
        "official_counts": OFFICIAL_COUNTS,
        "archived_candidate_counts": CANDIDATE_COUNTS,
        "predecessor_manifests_verified": ["Phase 2H", "Phase 2I", "Phase 2J", "Phase 2K-A"],
        "recognition_ran": False,
        "video_processing_ran": False,
    }
    evaluation = {
        "schema_version": 1,
        "policy_version": POLICY_VERSION,
        "capture_contract_version": CAPTURE_CONTRACT_VERSION,
        "run_id": preflight_result.run_id,
        "decision": "exact_carry_forward_verified_diagnostic_only_no_activation",
        "synthetic_matrix_scenarios": len(synthetic_rows),
        "immutable_writer_scenarios": len(immutable_rows),
        "verified_review_pairs": 31,
        "diagnostically_exact_match_eligible_pairs": eligible_real,
        "diagnostically_rejected_pairs": rejected_real,
        "complete_registry_attempt_result": overall_result,
        "partial_review_carry_forward_occurred": False,
        "mixed_track_result": mixed_regression,
        "candidate_sessions_audited": len(preflight_result.candidate_inventory),
        "untouched_existing_sessions": 0,
        "independent_session_recommendation": "no_existing_untouched_session",
        "official_counts": OFFICIAL_COUNTS,
        "archived_candidate_counts": CANDIDATE_COUNTS,
        "preservation_assertions": no_activation_declaration(),
    }
    failure_rows = [
        {"Result": result, "Reason_Code": reason, "Description": description, "Partial_Reuse": False}
        for result, reason, description in FAILURE_REASONS
    ]
    files = {
        "carry_forward_policy.json": canonical_json_bytes(preflight_result.policy),
        "exact_match_contract.json": canonical_json_bytes(exact_match_contract(preflight_result.policy_sha256)),
        "carry_forward_verification_matrix.csv": csv_bytes(matrix_rows, MATRIX_FIELDS),
        "carry_forward_failure_reasons.csv": csv_bytes(
            failure_rows, ("Result", "Reason_Code", "Description", "Partial_Reuse")
        ),
        "mixed_track_regression.json": canonical_json_bytes(mixed_regression),
        "candidate_session_inventory.csv": csv_bytes(preflight_result.candidate_inventory, INVENTORY_FIELDS),
        "independent_session_recommendation.json": canonical_json_bytes(independent_session_recommendation()),
        "independent_session_capture_contract.json": canonical_json_bytes(
            capture_contract(preflight_result.policy_sha256)
        ),
        "diagnostic_observation_schema.json": canonical_json_bytes(diagnostic_observation_schema()),
        "blind_review_schema.json": canonical_json_bytes(blind_review_schema()),
        "acceptance_criteria.json": canonical_json_bytes(acceptance_criteria()),
        "retention_criteria.json": canonical_json_bytes(retention_criteria()),
        "no_activation_declaration.json": canonical_json_bytes(no_activation_declaration()),
        "source_manifest.json": canonical_json_bytes(source_manifest),
        "evaluation_summary.json": canonical_json_bytes(evaluation),
    }
    return files, evaluation


def verify_immutable_output(output_dir: Path) -> dict[str, Any]:
    root = Path(output_dir)
    manifest_path = root / "immutable_manifest.json"
    if not manifest_path.is_file():
        raise CarryForwardContractError(f"Phase 2K-B immutable manifest is missing: {manifest_path}")
    manifest = _load_json(manifest_path, "Phase 2K-B immutable manifest")
    if manifest.get("schema_version") != OUTPUT_SCHEMA_VERSION:
        raise CarryForwardContractError("Unsupported Phase 2K-B output schema")
    if manifest.get("policy_version") != POLICY_VERSION:
        raise CarryForwardContractError("Phase 2K-B output policy changed")
    if manifest.get("run_id") != root.name:
        raise CarryForwardContractError("Phase 2K-B output directory does not match the run ID")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise CarryForwardContractError("Phase 2K-B immutable manifest has no files")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "immutable_manifest.json"
    }
    if actual_files != set(files):
        raise CarryForwardContractError("Phase 2K-B immutable output file set changed")
    for relative, metadata in files.items():
        path = root / Path(str(relative))
        if sha256_file(path) != str(metadata.get("sha256") or ""):
            raise CarryForwardContractError(f"Phase 2K-B output hash changed: {relative}")
        if path.stat().st_size != int(metadata.get("size_bytes", -1)):
            raise CarryForwardContractError(f"Phase 2K-B output size changed: {relative}")
    return manifest


def write_immutable_output(
    *, output_root: Path, run_id: str, files: Mapping[str, bytes]
) -> tuple[Path, bool]:
    root = Path(output_root)
    output = root / run_id
    normalized: dict[str, bytes] = {}
    for relative, payload in sorted(files.items()):
        name = normalize_provenance_path(relative)
        if name.startswith("/") or ".." in PurePath(name).parts or name == "immutable_manifest.json":
            raise CarryForwardContractError(f"Unsafe Phase 2K-B output path: {relative}")
        normalized[name] = bytes(payload)
    planned_metadata = {
        relative: {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
        for relative, payload in sorted(normalized.items())
    }
    if output.exists():
        manifest = verify_immutable_output(output)
        if manifest.get("files") != planned_metadata:
            raise CarryForwardContractError(
                "deterministic_id_collision: existing Phase 2K-B output has different bytes"
            )
        return output, True
    root.mkdir(parents=True, exist_ok=True)
    temp = root / f".{run_id}.tmp-{uuid.uuid4().hex}"
    if temp.exists():
        raise CarryForwardContractError(f"Unexpected temporary output exists: {temp}")
    temp.mkdir(parents=False)
    try:
        for relative, payload in normalized.items():
            path = temp / Path(relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        manifest = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "policy_version": POLICY_VERSION,
            "capture_contract_version": CAPTURE_CONTRACT_VERSION,
            "run_id": run_id,
            "immutable": True,
            "files": planned_metadata,
        }
        (temp / "immutable_manifest.json").write_bytes(canonical_json_bytes(manifest))
        temp.replace(output)
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise
    verify_immutable_output(output)
    return output, False


def materialize(preflight_result: Phase2KBPreflight) -> tuple[Path, bool, dict[str, Any]]:
    files, evaluation = build_output_files(preflight_result)
    output, reused = write_immutable_output(
        output_root=preflight_result.output_root,
        run_id=preflight_result.run_id,
        files=files,
    )
    verify_immutable_output(output)
    return output, reused, evaluation
