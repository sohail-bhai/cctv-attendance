from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import Any, Callable, Mapping, Sequence

from .processing_integration import (
    EXPECTED_CAMERA_STEMS,
    EXPECTED_CHECKPOINT_IDS,
    FIXED_POLICY,
    OFFICIAL_RECOGNITION_AUTHORITY,
    POLICY_VERSION as PROCESSING_POLICY_VERSION,
)
from .product_phase_2k_carry_forward import (
    CAPTURE_CONTRACT_VERSION,
    CANDIDATE_SPECS,
    POLICY_VERSION as CARRY_FORWARD_POLICY_VERSION,
    canonical_json_bytes,
    canonical_json_hash,
    capture_contract,
    carry_forward_policy,
    normalize_provenance_path,
    sha256_file,
    validate_capture_manifest,
    verify_immutable_output as verify_phase_2k_b_output,
)
from .product_phase_2k_tracklet_purity import (
    CANDIDATE_COUNTS,
    EXPECTED_EMBEDDING_SHA256,
    EXPECTED_FAMILY_ID,
    EXPECTED_POINTER_SHA256,
    EXPECTED_SUMMARY_SHA256,
    EXPECTED_VARIANT_ID,
    OFFICIAL_COUNTS,
)
from .tracklet_purity import POLICY_VERSION as PURITY_POLICY_VERSION
from .tracklet_purity import verify_immutable_output as verify_phase_2k_a_output


TOOL_POLICY_VERSION = "product-phase-2k-c0-untouched-session-intake-v1"
SOURCE_FREEZE_VERSION = "product-phase-2k-c0-immutable-source-freeze-v1"
CONTAMINATION_REGISTRY_VERSION = "product-phase-2k-c0-known-source-registry-v1"
PACKAGE_SCHEMA_VERSION = 1
TOOLING_SCHEMA_VERSION = 1
CONFIRMATION_TOKEN = "CREATE_PHASE_2K_C_SOURCE_FREEZE"
PHASE_2K_B_RUN_ID = "carry-forward-e20a746d3d34c8141ab5"
EXPECTED_PHASE_2K_B_MANIFEST_SHA256 = (
    "79a38737c6057768dc4128c3507fced268ea25c0beb84121a27cf2bb23195ac3"
)
EXPECTED_PHASE_2K_A_MANIFEST_SHA256 = (
    "7e0da7fd9b3edb7411a31be94f5f42063365bd0a6831735948700d263148c552"
)
EXPECTED_CAPTURE_ARTIFACT_HASHES = {
    "acceptance_criteria.json": "01eb1036a44a8f9a92350134c12f118ffea16a0d87d01b527cf23b8434e84493",
    "blind_review_schema.json": "0bcad27985b76cf17804990c2edfaf8d61378a9ecba21daf72c0e4927f5ae0e7",
    "candidate_session_inventory.csv": "d0efb4fefba059d513aec150fa4ee4785b573e2ef5e2e2088f800b42f3ac3d1b",
    "diagnostic_observation_schema.json": "d66c5dc8586c81da1bb0a0c6a016a579b3684afdb76f2e9a7c374a31a08f41ee",
    "independent_session_capture_contract.json": "30f361b5f63549c853f8fd9187a157230d58184a8ccd9511d7594870b0e3f092",
    "independent_session_recommendation.json": "ad6e85f80e2dcd52e2fcb9ad068db216bb59dcb975840e0f1a7d067efb688292",
    "no_activation_declaration.json": "2b95c5f67529f59f03b434eb2fc712dcd54f5cbcfaa36161b87e1cba95dda695",
    "retention_criteria.json": "91cb4e88b6cac9fafb2f3bbf7fda81d839fa39e5cfd185fc42fdd1fec57db94a",
    "source_manifest.json": "79f5ec385c222b59572680ef6cbcff8d04309617ee87a836b786c4c689ac83b9",
}
PROTECTED_EXPECTED_HASHES = {
    "data/attendance_status.json": "46fd2df55cc3f61c5fa03c893937eb2828b9cc2af064ea9e427276b79a9d0b46",
    "data/job_runtime.json": "5a23f09654d91fac09804cd97c2fa114bf4956a914f2e02ce2b7090f730141ed",
    "data/role_users.json": "77125140004294f2834fc869fd590e167d8ab78cff5dbeda1c074ab76c177517",
    "data/student_faculty_map.json": "0eca63ea58d12a73d3bd11e620b74be733a3219c9244e8a4a267d5b02acd3f91",
    "data/manual_overrides.json": "e9c6bd35c23c353795edb5d3c02e808793d7cbd90c7b0a87fdfc686f8fd5cdcf",
    "data/review_evidence_registry.json": "3b04e9aecfdfb1f64346c4a0e709fd9c36d7c56545bf816d6641b4c2a2e80841",
    "models/student_embeddings.pkl": EXPECTED_EMBEDDING_SHA256,
    "models/embedding_summary.csv": EXPECTED_SUMMARY_SHA256,
    "models/current_embedding_version.json": EXPECTED_POINTER_SHA256,
    "timetable_b51_2026_2027.csv": "10bbd578a859fbd4fa228c4eb4f1f92e7de5b92a1c4236d71a00337eecac9c57",
}
VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4"}
CONTAINER_BY_SUFFIX = {".avi": "avi", ".mkv": "matroska", ".mov": "quicktime", ".mp4": "mp4"}
EXPECTED_BINDINGS = tuple(
    (checkpoint, camera)
    for checkpoint in EXPECTED_CHECKPOINT_IDS
    for camera in EXPECTED_CAMERA_STEMS
)
PRIOR_USE_FLAGS = (
    "recognition_previously_run",
    "human_review_exists",
    "used_for_model_selection",
    "used_for_threshold_or_calibration",
    "used_for_source_ablation",
    "used_for_promotion_evidence",
    "used_for_retention_testing",
    "used_in_phase_1_2n",
    "used_in_phase_2g_or_2i",
    "used_in_any_reviewed_benchmark",
)
REASON_DESCRIPTIONS = {
    "confirmation_token_invalid": "The exact destructive-safety confirmation token was not supplied.",
    "source_count_invalid": "Exactly ten source files are required.",
    "checkpoint_binding_missing": "A CP1-CP5 front/back binding is missing.",
    "checkpoint_binding_extra": "An unexpected file or checkpoint/camera binding exists.",
    "checkpoint_binding_duplicate": "A checkpoint/camera binding occurs more than once.",
    "duplicate_inside_package": "Two proposed sources have identical SHA-256 bytes.",
    "known_contaminated_source": "A proposed source hash occurs in verified historical evidence.",
    "duplicate_session_id": "The canonical session ID already exists in verified history or another package.",
    "historical_prepared_path": "A source is located in the historical prepared-session tree.",
    "unverifiable_provenance": "Required immutable provenance could not be independently verified.",
    "session_metadata_invalid": "Required session metadata is missing or non-canonical.",
    "capture_timestamp_invalid": "An operator-supplied capture timestamp is missing or invalid.",
    "unsupported_container": "The source extension is not one of the supported video containers.",
    "container_metadata_unavailable": "Reliable width, height, FPS, and frame-count headers are unavailable.",
    "source_symlink_rejected": "Source symlinks and reparse-like indirection are forbidden.",
    "source_changed_during_copy": "Source bytes or file identity changed during staging.",
    "source_copy_mismatch": "The staged copy is not byte-identical to its source.",
    "deterministic_id_collision": "An existing deterministic package ID maps to different or invalid bytes.",
    "package_tampered": "The immutable package file set, sizes, or hashes changed.",
    "production_policy_changed": "The production model or processing policy differs from the frozen contract.",
    "canonical_capture_validator_failed": "The Phase 2K-B canonical capture validator rejected the package.",
    "phase_2k_b_contract_changed": "The verified Phase 2K-B capture contract or artifact set changed.",
    "output_path_unsafe": "The requested output or manifest path is unsafe.",
}


class CaptureIntakeError(RuntimeError):
    def __init__(self, reason_code: str, message: str | None = None, *, outcome: str = "unverifiable_provenance"):
        self.reason_code = reason_code
        self.outcome = outcome
        super().__init__(message or REASON_DESCRIPTIONS.get(reason_code, reason_code))


@dataclass(frozen=True)
class SourceSpec:
    checkpoint_id: str
    camera_id: str
    path: Path
    capture_timestamp: str


@dataclass(frozen=True)
class CaptureRequest:
    repo_root: Path
    output_root: Path
    session_date: str
    section: str
    period: str
    subject: str
    room: str
    sources: tuple[SourceSpec, ...]
    confirmation_token: str


@dataclass(frozen=True)
class CapturePlan:
    repo_root: Path
    output_root: Path
    package_id: str
    package_fingerprint_sha256: str
    session_metadata: dict[str, Any]
    sources: tuple[dict[str, Any], ...]
    dependency: dict[str, Any]
    registry: dict[str, Any]
    protected_hashes: dict[str, str]


def _fail(reason_code: str, message: str | None = None, *, outcome: str = "unverifiable_provenance") -> None:
    raise CaptureIntakeError(reason_code, message, outcome=outcome)


def _load_json(path: Path, reason_code: str = "unverifiable_provenance") -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise CaptureIntakeError(reason_code, f"Could not read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        _fail(reason_code, f"JSON payload is not an object: {path}")
    return value


def _iso_timestamp(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        _fail("capture_timestamp_invalid")
    candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise CaptureIntakeError("capture_timestamp_invalid", f"Invalid ISO-8601 capture timestamp: {text}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _fail("capture_timestamp_invalid", f"Capture timestamp requires an explicit UTC offset: {text}")
    return parsed.isoformat(timespec="seconds")


def _utc_timestamp(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat(timespec="microseconds")


def _stat_signature(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return (
        int(getattr(stat, "st_dev", 0)),
        int(getattr(stat, "st_ino", 0)),
        int(stat.st_size),
        int(stat.st_mtime_ns),
    )


def _safe_relative(relative: str) -> str:
    normalized = normalize_provenance_path(relative)
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized) or ".." in PurePath(normalized).parts:
        _fail("output_path_unsafe", f"Unsafe package-relative path: {relative}")
    return normalized


def _verify_legacy_file_map(path: Path) -> None:
    manifest = _load_json(path)
    files = manifest.get("files_sha256")
    if not isinstance(files, Mapping):
        files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        _fail("unverifiable_provenance", f"Historical manifest has no verifiable file map: {path}")
    for relative, raw in sorted(files.items()):
        expected = raw.get("sha256") if isinstance(raw, Mapping) else raw
        target = path.parent / Path(_safe_relative(str(relative)))
        if not target.is_file() or sha256_file(target) != str(expected):
            _fail("unverifiable_provenance", f"Historical artifact changed or is missing: {target}")


def _phase_2k_b_dependency(repo_root: Path) -> dict[str, Any]:
    output = (
        repo_root
        / "attendance_output"
        / "product_workflow"
        / "phase_2k_carry_forward"
        / PHASE_2K_B_RUN_ID
    )
    manifest_path = output / "immutable_manifest.json"
    if not manifest_path.is_file() or sha256_file(manifest_path) != EXPECTED_PHASE_2K_B_MANIFEST_SHA256:
        _fail("phase_2k_b_contract_changed")
    try:
        immutable = verify_phase_2k_b_output(output)
    except Exception as exc:
        raise CaptureIntakeError("phase_2k_b_contract_changed", str(exc)) from exc
    for name, expected in EXPECTED_CAPTURE_ARTIFACT_HASHES.items():
        path = output / name
        if not path.is_file() or sha256_file(path) != expected:
            _fail("phase_2k_b_contract_changed", f"Phase 2K-B artifact changed: {name}")
    capture = _load_json(output / "independent_session_capture_contract.json", "phase_2k_b_contract_changed")
    diagnostic = _load_json(output / "diagnostic_observation_schema.json", "phase_2k_b_contract_changed")
    blind = _load_json(output / "blind_review_schema.json", "phase_2k_b_contract_changed")
    acceptance = _load_json(output / "acceptance_criteria.json", "phase_2k_b_contract_changed")
    retention = _load_json(output / "retention_criteria.json", "phase_2k_b_contract_changed")
    recommendation = _load_json(output / "independent_session_recommendation.json", "phase_2k_b_contract_changed")
    no_activation = _load_json(output / "no_activation_declaration.json", "phase_2k_b_contract_changed")
    if capture.get("capture_contract_version") != CAPTURE_CONTRACT_VERSION:
        _fail("phase_2k_b_contract_changed")
    if diagnostic.get("$id") != "product-phase-2k-c-diagnostic-observation-v1":
        _fail("phase_2k_b_contract_changed")
    if blind.get("$id") != "product-phase-2k-c-blind-review-v1":
        _fail("phase_2k_b_contract_changed")
    if recommendation.get("status") != "no_existing_untouched_session":
        _fail("phase_2k_b_contract_changed")
    if no_activation.get("production_activation_available") is not False:
        _fail("phase_2k_b_contract_changed")
    source_manifest = _load_json(output / "source_manifest.json", "phase_2k_b_contract_changed")
    for evidence in source_manifest.get("candidate_audit_evidence") or []:
        if not isinstance(evidence, Mapping):
            _fail("unverifiable_provenance")
        path = repo_root / Path(_safe_relative(str(evidence.get("path") or "")))
        if not path.is_file() or sha256_file(path) != evidence.get("sha256"):
            _fail("unverifiable_provenance", f"Candidate audit evidence changed: {path}")
        if path.name == "output_manifest.json":
            _verify_legacy_file_map(path)
    phase_2k_a = (
        repo_root
        / "attendance_output"
        / "product_workflow"
        / "phase_2k_tracklet_purity"
        / "tracklet-purity-43cec17498dc05615502"
    )
    phase_2k_a_manifest = phase_2k_a / "immutable_manifest.json"
    if not phase_2k_a_manifest.is_file() or sha256_file(phase_2k_a_manifest) != EXPECTED_PHASE_2K_A_MANIFEST_SHA256:
        _fail("unverifiable_provenance", "Phase 2K-A immutable manifest changed")
    try:
        verify_phase_2k_a_output(phase_2k_a)
    except Exception as exc:
        raise CaptureIntakeError("unverifiable_provenance", str(exc)) from exc
    return {
        "output": output,
        "immutable_manifest": immutable,
        "immutable_manifest_sha256": EXPECTED_PHASE_2K_B_MANIFEST_SHA256,
        "capture_contract": capture,
        "capture_contract_sha256": EXPECTED_CAPTURE_ARTIFACT_HASHES["independent_session_capture_contract.json"],
        "diagnostic_schema": diagnostic,
        "diagnostic_schema_sha256": EXPECTED_CAPTURE_ARTIFACT_HASHES["diagnostic_observation_schema.json"],
        "blind_schema": blind,
        "blind_schema_sha256": EXPECTED_CAPTURE_ARTIFACT_HASHES["blind_review_schema.json"],
        "acceptance": acceptance,
        "acceptance_sha256": EXPECTED_CAPTURE_ARTIFACT_HASHES["acceptance_criteria.json"],
        "retention": retention,
        "retention_sha256": EXPECTED_CAPTURE_ARTIFACT_HASHES["retention_criteria.json"],
        "recommendation": recommendation,
        "source_manifest": source_manifest,
        "phase_2k_a_manifest_sha256": EXPECTED_PHASE_2K_A_MANIFEST_SHA256,
    }


def _known_source_registry(dependency: Mapping[str, Any]) -> dict[str, Any]:
    source_manifest = dependency["source_manifest"]
    rows = source_manifest.get("candidate_source_files")
    if not isinstance(rows, list) or len(rows) != 40:
        _fail("unverifiable_provenance", "Verified Phase 2K-B source registry is incomplete")
    phase_labels = {
        "2026-06-22__B51__P3__CVO": ["Phase 1.2J", "Phase 1.2L", "Phase 2K-B"],
        "2026-06-22__B51__P4__CVO": ["Phase 1.2N", "Phase 2G", "Phase 2I", "Phase 2K-A", "Phase 2K-B"],
        "2026-06-30__B51__P1__CVO": ["Phase 1 review", "Phase 1.2G", "Phase 1.2I", "Phase 1.2L", "Phase 2K-B"],
        "2026-06-30__B51__P2__CVO": ["Phase 1 review", "Phase 1.2G", "Phase 1.2I", "Phase 1.2M", "Phase 2K-B"],
    }
    entries: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            _fail("unverifiable_provenance")
        session_id = str(raw.get("session_id") or "")
        sha = str(raw.get("sha256") or "")
        if session_id not in phase_labels or not re.fullmatch(r"[0-9a-f]{64}", sha):
            _fail("unverifiable_provenance")
        entries.append(
            {
                "sha256": sha,
                "size_bytes": int(raw["size_bytes"]),
                "session_id": session_id,
                "historical_relative_path": normalize_provenance_path(str(raw["path"])),
                "checkpoint_id": str(raw["checkpoint"]),
                "camera_id": str(raw["camera"]),
                "verified_prior_use_phases": phase_labels[session_id],
                "outcome": "known_contaminated_source",
            }
        )
    entries.sort(key=lambda row: (row["session_id"], row["historical_relative_path"]))
    evidence = list(source_manifest.get("candidate_audit_evidence") or [])
    registry = {
        "schema_version": 1,
        "registry_version": CONTAMINATION_REGISTRY_VERSION,
        "source_phase_2k_b_manifest_sha256": dependency["immutable_manifest_sha256"],
        "source_phase_2k_a_manifest_sha256": dependency["phase_2k_a_manifest_sha256"],
        "known_session_ids": sorted(phase_labels),
        "entry_count": len(entries),
        "verified_evidence_artifacts": evidence,
        "entries": entries,
        "override_permitted": False,
    }
    registry["registry_fingerprint_sha256"] = canonical_json_hash(registry)
    return registry


def _protected_hashes(repo_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative, expected in PROTECTED_EXPECTED_HASHES.items():
        path = repo_root / Path(relative)
        if not path.is_file() or sha256_file(path) != expected:
            _fail("production_policy_changed", f"Protected production input changed: {relative}")
        result[relative] = expected
    pointer = _load_json(repo_root / "models" / "current_embedding_version.json", "production_policy_changed")
    required = {
        "family_id": EXPECTED_FAMILY_ID,
        "variant_id": EXPECTED_VARIANT_ID,
        "production_embeddings_sha256": EXPECTED_EMBEDDING_SHA256,
        "production_summary_sha256": EXPECTED_SUMMARY_SHA256,
        "status": "promoted",
    }
    if any(pointer.get(key) != value for key, value in required.items()):
        _fail("production_policy_changed", "Production embedding pointer fields changed")
    return result


def read_video_header_metadata(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        _fail("unsupported_container", f"Unsupported source container: {path}")
    try:
        import cv2  # Imported lazily; only VideoCapture header properties are used.
    except Exception as exc:
        raise CaptureIntakeError("container_metadata_unavailable", f"OpenCV header reader unavailable: {exc}") from exc
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            _fail("container_metadata_unavailable", f"Container headers could not be opened: {path}")
        width = int(round(float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))))
        height = int(round(float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(round(float(capture.get(cv2.CAP_PROP_FRAME_COUNT))))
        backend = capture.getBackendName() if hasattr(capture, "getBackendName") else "opencv"
    finally:
        capture.release()
    if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
        _fail("container_metadata_unavailable", f"Reliable header metadata is unavailable: {path}")
    return {
        "container": CONTAINER_BY_SUFFIX[suffix],
        "width": width,
        "height": height,
        "fps": round(fps, 6),
        "frame_count": frame_count,
        "estimated_duration_seconds": round(frame_count / fps, 6),
        "metadata_method": f"opencv_video_capture_header_properties_no_read:{backend}",
        "metadata_confidence": "header_reported_complete",
        "frames_decoded": False,
    }


def parse_binding_values(values: Sequence[str], *, value_label: str) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    pattern = re.compile(r"^(CP[1-5])[/\\](back|front)=(.+)$", re.IGNORECASE)
    for raw in values:
        match = pattern.fullmatch(str(raw).strip())
        if not match:
            code = "capture_timestamp_invalid" if value_label == "capture timestamp" else "checkpoint_binding_extra"
            _fail(code, f"Invalid {value_label} binding: {raw}")
        key = (match.group(1).upper(), match.group(2).lower())
        if key in result:
            _fail("checkpoint_binding_duplicate", f"Duplicate binding: {key[0]}/{key[1]}")
        result[key] = match.group(3).strip()
    return result


def discover_source_bindings(source_dir: Path) -> dict[tuple[str, str], Path]:
    root = Path(source_dir).resolve()
    if not root.is_dir() or root.is_symlink():
        _fail("source_count_invalid", f"Source directory is missing or unsafe: {root}")
    bindings: dict[tuple[str, str], Path] = {}
    seen_paths: set[Path] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            _fail("source_symlink_rejected", f"Source symlink rejected: {path}")
        relative = path.relative_to(root)
        if path.is_dir():
            if len(relative.parts) != 1 or relative.name.upper() not in EXPECTED_CHECKPOINT_IDS:
                _fail("checkpoint_binding_extra", f"Unexpected source directory: {relative}")
            continue
        if len(relative.parts) != 2:
            _fail("checkpoint_binding_extra", f"Unexpected source file: {relative}")
        checkpoint = relative.parts[0].upper()
        camera = path.stem.lower()
        if checkpoint not in EXPECTED_CHECKPOINT_IDS or camera not in EXPECTED_CAMERA_STEMS:
            _fail("checkpoint_binding_extra", f"Unexpected source binding: {relative}")
        if path.suffix.lower() not in VIDEO_SUFFIXES:
            _fail("unsupported_container", f"Unsupported source container: {relative}")
        key = (checkpoint, camera)
        if key in bindings:
            _fail("checkpoint_binding_duplicate", f"Duplicate binding: {checkpoint}/{camera}")
        resolved = path.resolve()
        if resolved in seen_paths:
            _fail("checkpoint_binding_duplicate", f"Source path is reused: {resolved}")
        seen_paths.add(resolved)
        bindings[key] = resolved
    missing = set(EXPECTED_BINDINGS).difference(bindings)
    extra = set(bindings).difference(EXPECTED_BINDINGS)
    if missing:
        _fail("checkpoint_binding_missing", "Missing bindings: " + ", ".join(f"{cp}/{cam}" for cp, cam in sorted(missing)))
    if extra or len(bindings) != 10:
        _fail("source_count_invalid")
    return bindings


def build_source_specs(
    *,
    source_dir: Path | None,
    source_values: Sequence[str],
    capture_timestamp_values: Sequence[str],
) -> tuple[SourceSpec, ...]:
    if (source_dir is None) == (len(source_values) == 0):
        _fail("source_count_invalid", "Supply exactly one of --source-dir or ten --source bindings")
    if source_dir is not None:
        source_bindings = discover_source_bindings(source_dir)
    else:
        parsed = parse_binding_values(source_values, value_label="source")
        source_bindings = {key: Path(value).expanduser().resolve() for key, value in parsed.items()}
    timestamp_bindings = parse_binding_values(capture_timestamp_values, value_label="capture timestamp")
    if set(source_bindings) != set(EXPECTED_BINDINGS):
        missing = set(EXPECTED_BINDINGS).difference(source_bindings)
        _fail("checkpoint_binding_missing" if missing else "source_count_invalid")
    if set(timestamp_bindings) != set(EXPECTED_BINDINGS):
        _fail("capture_timestamp_invalid", "Every source requires an operator-supplied capture timestamp")
    return tuple(
        SourceSpec(checkpoint, camera, source_bindings[(checkpoint, camera)], _iso_timestamp(timestamp_bindings[(checkpoint, camera)]))
        for checkpoint, camera in EXPECTED_BINDINGS
    )


def _session_metadata(request: CaptureRequest) -> dict[str, Any]:
    try:
        date_value = datetime.strptime(request.session_date.strip(), "%Y-%m-%d").date().isoformat()
    except ValueError as exc:
        raise CaptureIntakeError("session_metadata_invalid", "Session date must be YYYY-MM-DD") from exc
    section = request.section.strip().upper()
    subject = request.subject.strip().upper()
    room = request.room.strip()
    period_raw = request.period.strip().upper()
    period = period_raw if re.fullmatch(r"P[1-9][0-9]?", period_raw) else f"P{period_raw}"
    if section != "B51" or subject != "CVO" or not re.fullmatch(r"P[1-9][0-9]?", period) or not room:
        _fail("session_metadata_invalid", "This contract requires an explicit CVO/B51 date, period, and room")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _.-]{0,63}", room):
        _fail("session_metadata_invalid", "Room contains unsupported characters")
    session_id = f"{date_value}__{section}__{period}__{subject}"
    return {
        "schema_version": 1,
        "session_id": session_id,
        "session_date": date_value,
        "section": section,
        "period": period,
        "subject": subject,
        "room": room,
        "untouched_declaration": True,
        "declaration_scope": "no recognition, review, tuning, calibration, selection, ablation, promotion, retention, or benchmark use before source freeze",
    }


def _historical_path(repo_root: Path, path: Path) -> bool:
    normalized = normalize_provenance_path(path.resolve()).lower()
    historical = normalize_provenance_path(repo_root / "cctv_videos" / "prepared_slots").lower()
    return normalized == historical or normalized.startswith(historical.rstrip("/") + "/") or "/cctv_videos/prepared_slots/" in normalized


def _existing_session_ids(output_root: Path, *, expected_package_id: str | None = None) -> set[str]:
    result: set[str] = set()
    if not output_root.exists():
        return result
    if output_root.is_symlink() or not output_root.is_dir():
        _fail("output_path_unsafe")
    for candidate in sorted(output_root.glob("capture-package-*")):
        if candidate.name == expected_package_id:
            continue
        if candidate.is_symlink() or not candidate.is_dir():
            _fail("unverifiable_provenance", f"Unsafe existing package entry: {candidate}")
        manifest = candidate / "immutable_manifest.json"
        session = candidate / "session_metadata.json"
        if not manifest.is_file() or not session.is_file():
            _fail("unverifiable_provenance", f"Existing package is unverifiable: {candidate}")
        _verify_file_manifest(candidate, expected_policy=TOOL_POLICY_VERSION)
        result.add(str(_load_json(session).get("session_id") or ""))
    return result


def plan_capture(
    request: CaptureRequest,
    *,
    metadata_reader: Callable[[Path], Mapping[str, Any]] = read_video_header_metadata,
    dependency_loader: Callable[[Path], dict[str, Any]] = _phase_2k_b_dependency,
    registry_loader: Callable[[Mapping[str, Any]], dict[str, Any]] = _known_source_registry,
) -> CapturePlan:
    repo_root = Path(request.repo_root).resolve()
    raw_output_root = Path(os.path.abspath(request.output_root))
    if any(candidate.is_symlink() for candidate in (raw_output_root, *raw_output_root.parents)):
        _fail("output_path_unsafe", "Output roots may not use symlinks")
    output_root = raw_output_root.resolve()
    if request.confirmation_token != CONFIRMATION_TOKEN:
        _fail("confirmation_token_invalid")
    session = _session_metadata(request)
    dependency = dependency_loader(repo_root)
    registry = registry_loader(dependency)
    protected = _protected_hashes(repo_root)
    if session["session_id"] in set(registry["known_session_ids"]):
        _fail("duplicate_session_id", outcome="duplicate_session_id")
    if len(request.sources) != 10:
        _fail("source_count_invalid")
    source_keys = [(item.checkpoint_id.upper(), item.camera_id.lower()) for item in request.sources]
    if len(set(source_keys)) != len(source_keys):
        _fail("checkpoint_binding_duplicate")
    if set(source_keys) != set(EXPECTED_BINDINGS):
        _fail("checkpoint_binding_missing" if set(EXPECTED_BINDINGS).difference(source_keys) else "checkpoint_binding_extra")
    known_hashes = {entry["sha256"]: entry for entry in registry["entries"]}
    planned: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for order, (checkpoint, camera) in enumerate(EXPECTED_BINDINGS):
        spec = next(item for item in request.sources if item.checkpoint_id.upper() == checkpoint and item.camera_id.lower() == camera)
        raw_source = Path(os.path.abspath(spec.path))
        if any(candidate.is_symlink() for candidate in (raw_source, *raw_source.parents)):
            _fail("source_symlink_rejected")
        source = raw_source.resolve()
        if not source.is_file():
            _fail("unverifiable_provenance", f"Source file is missing: {source}")
        if _historical_path(repo_root, source):
            _fail("historical_prepared_path", outcome="known_contaminated_source")
        suffix = source.suffix.lower()
        if suffix not in VIDEO_SUFFIXES:
            _fail("unsupported_container")
        before = _stat_signature(source)
        source_sha = sha256_file(source)
        after_hash = _stat_signature(source)
        if before != after_hash:
            _fail("source_changed_during_copy")
        if source_sha in seen_hashes:
            _fail("duplicate_inside_package", outcome="duplicate_inside_package")
        seen_hashes.add(source_sha)
        if source_sha in known_hashes:
            entry = known_hashes[source_sha]
            _fail(
                "known_contaminated_source",
                f"Source {checkpoint}/{camera} matches {entry['session_id']} {entry['historical_relative_path']}",
                outcome="known_contaminated_source",
            )
        metadata = dict(metadata_reader(source))
        required_metadata = {
            "container", "width", "height", "fps", "frame_count", "estimated_duration_seconds",
            "metadata_method", "metadata_confidence", "frames_decoded",
        }
        if not required_metadata.issubset(metadata) or metadata.get("frames_decoded") is not False:
            _fail("container_metadata_unavailable")
        if (
            int(metadata["width"]) <= 0
            or int(metadata["height"]) <= 0
            or float(metadata["fps"]) <= 0
            or int(metadata["frame_count"]) <= 0
            or float(metadata["estimated_duration_seconds"]) <= 0
        ):
            _fail("container_metadata_unavailable")
        stat = source.stat()
        relative = f"videos/{checkpoint}/{camera}{suffix}"
        planned.append(
            {
                "canonical_order": order,
                "checkpoint_id": checkpoint,
                "camera_id": camera,
                "relative_path": relative,
                "source_path": source,
                "original_absolute_source_path": normalize_provenance_path(source),
                "sha256": source_sha,
                "size_bytes": int(stat.st_size),
                "extension": suffix,
                "container": str(metadata["container"]),
                "width": int(metadata["width"]),
                "height": int(metadata["height"]),
                "fps": round(float(metadata["fps"]), 6),
                "frame_count": int(metadata["frame_count"]),
                "estimated_duration_seconds": round(float(metadata["estimated_duration_seconds"]), 6),
                "filesystem_ctime_utc": _utc_timestamp(stat.st_ctime),
                "filesystem_mtime_utc": _utc_timestamp(stat.st_mtime),
                "operator_capture_timestamp": _iso_timestamp(spec.capture_timestamp),
                "capture_timestamp_method": "operator_supplied",
                "capture_timestamp_confidence": "operator_confirmed_exact",
                "metadata_method": str(metadata["metadata_method"]),
                "metadata_confidence": str(metadata["metadata_confidence"]),
                "contamination_outcome": "new_unique_source",
                "stat_signature": before,
            }
        )
    public_sources = [{key: value for key, value in row.items() if key not in {"source_path", "stat_signature", "original_absolute_source_path"}} for row in planned]
    fingerprint_payload = {
        "tool_policy_version": TOOL_POLICY_VERSION,
        "source_freeze_version": SOURCE_FREEZE_VERSION,
        "phase_2k_b_manifest_sha256": dependency["immutable_manifest_sha256"],
        "capture_contract_sha256": dependency["capture_contract_sha256"],
        "production_hashes": protected,
        "session": session,
        "sources": public_sources,
        "contamination_registry_fingerprint_sha256": registry["registry_fingerprint_sha256"],
    }
    package_fingerprint = canonical_json_hash(fingerprint_payload)
    package_id = f"capture-package-{package_fingerprint[:20]}"
    if session["session_id"] in _existing_session_ids(output_root, expected_package_id=package_id):
        _fail("duplicate_session_id", outcome="duplicate_session_id")
    return CapturePlan(
        repo_root=repo_root,
        output_root=output_root,
        package_id=package_id,
        package_fingerprint_sha256=package_fingerprint,
        session_metadata=session,
        sources=tuple(planned),
        dependency=dependency,
        registry=registry,
        protected_hashes=protected,
    )


INVENTORY_FIELDS = (
    "Canonical_Order", "Relative_Path", "Checkpoint_ID", "Camera_ID", "SHA256", "Size_Bytes",
    "Extension", "Container", "Width", "Height", "FPS", "Frame_Count",
    "Estimated_Duration_Seconds", "Filesystem_CTime_UTC", "Filesystem_MTime_UTC",
    "Operator_Capture_Timestamp", "Capture_Timestamp_Method", "Capture_Timestamp_Confidence",
    "Metadata_Method", "Metadata_Confidence", "Contamination_Outcome",
)


def _inventory_bytes(sources: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=INVENTORY_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in sources:
        writer.writerow(
            {
                "Canonical_Order": row["canonical_order"],
                "Relative_Path": row["relative_path"],
                "Checkpoint_ID": row["checkpoint_id"],
                "Camera_ID": row["camera_id"],
                "SHA256": row["sha256"],
                "Size_Bytes": row["size_bytes"],
                "Extension": row["extension"],
                "Container": row["container"],
                "Width": row["width"],
                "Height": row["height"],
                "FPS": row["fps"],
                "Frame_Count": row["frame_count"],
                "Estimated_Duration_Seconds": row["estimated_duration_seconds"],
                "Filesystem_CTime_UTC": row["filesystem_ctime_utc"],
                "Filesystem_MTime_UTC": row["filesystem_mtime_utc"],
                "Operator_Capture_Timestamp": row["operator_capture_timestamp"],
                "Capture_Timestamp_Method": row["capture_timestamp_method"],
                "Capture_Timestamp_Confidence": row["capture_timestamp_confidence"],
                "Metadata_Method": row["metadata_method"],
                "Metadata_Confidence": row["metadata_confidence"],
                "Contamination_Outcome": row["contamination_outcome"],
            }
        )
    return buffer.getvalue().encode("utf-8")


def _production_policy_freeze(plan: CapturePlan) -> dict[str, Any]:
    fixed = plan.dependency["capture_contract"]["fixed_production_configuration"]
    return {
        "schema_version": 1,
        "tool_policy_version": TOOL_POLICY_VERSION,
        "capture_contract_version": CAPTURE_CONTRACT_VERSION,
        "carry_forward_policy_version": CARRY_FORWARD_POLICY_VERSION,
        "carry_forward_policy_sha256": canonical_json_hash(carry_forward_policy()),
        "purity_policy_version": PURITY_POLICY_VERSION,
        "processing_contract_version": PROCESSING_POLICY_VERSION,
        "fixed_production_configuration": fixed,
        "complete_processing_policy": dict(FIXED_POLICY),
        "strict_tracklet_authority": OFFICIAL_RECOGNITION_AUTHORITY,
        "source_layout_version": "five-checkpoint-front-back-v1",
        "source_freeze_version": SOURCE_FREEZE_VERSION,
        "protected_file_hashes": plan.protected_hashes,
        "policy_must_match_again_before_recognition": True,
        "recognition_authorized": False,
        "activation_authorized": False,
    }


def _capture_manifest(plan: CapturePlan) -> dict[str, Any]:
    fixed = plan.dependency["capture_contract"]["fixed_production_configuration"]
    return {
        "schema_version": 1,
        "capture_contract_version": CAPTURE_CONTRACT_VERSION,
        "session_id": plan.session_metadata["session_id"],
        "untouched_declaration": True,
        "prior_use": {flag: False for flag in PRIOR_USE_FLAGS},
        "production_configuration": fixed,
        "sources": [
            {
                "relative_path": row["relative_path"],
                "sha256": row["sha256"],
                "size_bytes": row["size_bytes"],
                "checkpoint_id": row["checkpoint_id"],
                "camera_id": row["camera_id"],
                "canonical_order": row["canonical_order"],
                "duration_seconds": row["estimated_duration_seconds"],
                "fps": row["fps"],
                "frame_count": row["frame_count"],
                "capture_start_timestamp": row["operator_capture_timestamp"],
            }
            for row in plan.sources
        ],
        "parameter_tuning_after_results_visible": False,
        "diagnostic_observation_schema_id": "product-phase-2k-c-diagnostic-observation-v1",
        "appearance_evidence_mode": "derived_pairwise_and_local_window_similarity_matrices",
        "blind_review_schema_id": "product-phase-2k-c-blind-review-v1",
        "recognition_authorized": False,
    }


def _metadata_files(plan: CapturePlan) -> dict[str, bytes]:
    public_sources = [{key: value for key, value in row.items() if key not in {"source_path", "stat_signature", "original_absolute_source_path"}} for row in plan.sources]
    restricted_sources = [
        {
            "canonical_order": row["canonical_order"],
            "checkpoint_id": row["checkpoint_id"],
            "camera_id": row["camera_id"],
            "original_absolute_source_path": row["original_absolute_source_path"],
        }
        for row in plan.sources
    ]
    package_metadata = {
        "schema_version": 1,
        "tool_policy_version": TOOL_POLICY_VERSION,
        "package_id": plan.package_id,
        "package_fingerprint_sha256": plan.package_fingerprint_sha256,
        "immutable": True,
        "confirmation_token_verified": True,
        "restricted_operator_record": {
            "classification": "restricted_operator_provenance",
            "frontend_or_public_exposure_permitted": False,
            "original_sources": restricted_sources,
        },
        "freeze_timestamp": max(row["operator_capture_timestamp"] for row in plan.sources),
        "freeze_timestamp_basis": "latest_operator_supplied_capture_timestamp; no wall-clock value guessed",
    }
    source_layout = {
        "schema_version": 1,
        "source_layout_version": "five-checkpoint-front-back-v1",
        "canonical_order": "CP1..CP5, back before front",
        "checkpoint_ids": list(EXPECTED_CHECKPOINT_IDS),
        "camera_ids": list(EXPECTED_CAMERA_STEMS),
        "exact_video_count": 10,
        "bindings": [
            {key: row[key] for key in ("canonical_order", "checkpoint_id", "camera_id", "relative_path")}
            for row in plan.sources
        ],
    }
    inventory_fingerprint = canonical_json_hash(public_sources)
    source_freeze = {
        "schema_version": 1,
        "source_freeze_version": SOURCE_FREEZE_VERSION,
        "package_id": plan.package_id,
        "session_id": plan.session_metadata["session_id"],
        "source_count": 10,
        "source_inventory_fingerprint_sha256": inventory_fingerprint,
        "all_source_hashes_unique": True,
        "all_staged_copies_byte_identical": True,
        "source_files_modified": False,
        "frames_decoded": False,
        "recognition_ran": False,
        "freeze_completed_before_recognition": True,
        "phase_2k_b_manifest_sha256": plan.dependency["immutable_manifest_sha256"],
        "capture_contract_sha256": plan.dependency["capture_contract_sha256"],
        "contamination_registry_fingerprint_sha256": plan.registry["registry_fingerprint_sha256"],
        "sources": public_sources,
    }
    contamination = {
        "schema_version": 1,
        "registry_version": CONTAMINATION_REGISTRY_VERSION,
        "registry_fingerprint_sha256": plan.registry["registry_fingerprint_sha256"],
        "known_source_entry_count": plan.registry["entry_count"],
        "known_session_ids": plan.registry["known_session_ids"],
        "verified_evidence_artifacts": plan.registry["verified_evidence_artifacts"],
        "source_checks": [
            {
                "canonical_order": row["canonical_order"],
                "relative_path": row["relative_path"],
                "sha256": row["sha256"],
                "outcome": "new_unique_source",
            }
            for row in plan.sources
        ],
        "session_check": {"session_id": plan.session_metadata["session_id"], "outcome": "new_unique_source"},
        "overall_outcome": "new_unique_source",
        "override_permitted": False,
    }
    blind_plan = {
        "schema_version": 1,
        "capture_contract_version": CAPTURE_CONTRACT_VERSION,
        "frozen_before_recognition": True,
        "results_visible_when_frozen": False,
        "blind_review_schema_id": plan.dependency["blind_schema"]["$id"],
        "blind_review_schema_sha256": plan.dependency["blind_schema_sha256"],
        "acceptance_criteria_sha256": plan.dependency["acceptance_sha256"],
        "retention_criteria_sha256": plan.dependency["retention_sha256"],
        "randomized_ids_required": True,
        "predicted_identity_scores_and_decisions_hidden": True,
        "private_join_stored_separately": True,
        "completeness_validation_required": True,
        "recognition_authorized": False,
    }
    no_recognition = {
        "schema_version": 1,
        "phase": "Product Phase 2K-C0",
        "recognition_ran": False,
        "yunet_inference_ran": False,
        "sface_inference_ran": False,
        "frames_decoded": False,
        "video_processing_ran": False,
        "container_header_metadata_only": True,
        "attendance_changed": False,
        "authority_changed": False,
        "production_activation_available": False,
        "validation_is_recognition_authorization": False,
    }
    validation = {
        "schema_version": 1,
        "tool_policy_version": TOOL_POLICY_VERSION,
        "package_id": plan.package_id,
        "valid": True,
        "overall_outcome": "new_unique_source",
        "checks": [
            {"reason_code": code, "status": "pass"}
            for code in (
                "exact_source_count", "canonical_checkpoint_camera_layout", "unique_source_hashes",
                "new_unique_source", "new_session_id", "complete_container_header_metadata",
                "source_copy_byte_identical", "phase_2k_b_contract_compatible",
                "production_policy_frozen", "canonical_capture_validator_passed",
                "no_recognition_authorization",
            )
        ],
        "recognition_ran": False,
        "frames_decoded": False,
    }
    files = {
        "package_metadata.json": canonical_json_bytes(package_metadata),
        "session_metadata.json": canonical_json_bytes(plan.session_metadata),
        "source_layout.json": canonical_json_bytes(source_layout),
        "source_inventory.csv": _inventory_bytes(plan.sources),
        "source_freeze.json": canonical_json_bytes(source_freeze),
        "production_policy_freeze.json": canonical_json_bytes(_production_policy_freeze(plan)),
        "known_source_registry.json": canonical_json_bytes(plan.registry),
        "contamination_check.json": canonical_json_bytes(contamination),
        "capture_validation.json": canonical_json_bytes(validation),
        "capture_manifest.json": canonical_json_bytes(_capture_manifest(plan)),
        "blind_review_plan.json": canonical_json_bytes(blind_plan),
        "no_recognition_declaration.json": canonical_json_bytes(no_recognition),
    }
    return files


def _verify_file_manifest(root: Path, *, expected_policy: str) -> dict[str, Any]:
    manifest_path = root / "immutable_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        _fail("package_tampered", f"Immutable manifest is missing: {manifest_path}")
    manifest = _load_json(manifest_path, "package_tampered")
    if manifest.get("schema_version") != PACKAGE_SCHEMA_VERSION or manifest.get("tool_policy_version") != expected_policy:
        _fail("package_tampered", "Immutable manifest schema or policy changed")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        _fail("package_tampered")
    actual: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            _fail("package_tampered", f"Package symlink rejected: {path}")
        if path.is_file() and path.name != "immutable_manifest.json":
            actual.add(path.relative_to(root).as_posix())
    if actual != set(files):
        _fail("package_tampered", "Immutable package file set changed")
    for relative, raw in files.items():
        safe = _safe_relative(str(relative))
        metadata = raw if isinstance(raw, Mapping) else {}
        path = root / Path(safe)
        if path.stat().st_size != int(metadata.get("size_bytes", -1)) or sha256_file(path) != metadata.get("sha256"):
            _fail("package_tampered", f"Immutable package bytes changed: {relative}")
    return manifest


def verify_capture_package(repo_root: Path, package_dir: Path) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    package = Path(package_dir).resolve()
    if not package.is_dir() or package.is_symlink():
        _fail("package_tampered")
    dependency = _phase_2k_b_dependency(repo)
    registry = _known_source_registry(dependency)
    protected = _protected_hashes(repo)
    immutable = _verify_file_manifest(package, expected_policy=TOOL_POLICY_VERSION)
    package_metadata = _load_json(package / "package_metadata.json", "package_tampered")
    session = _load_json(package / "session_metadata.json", "package_tampered")
    source_freeze = _load_json(package / "source_freeze.json", "package_tampered")
    production = _load_json(package / "production_policy_freeze.json", "production_policy_changed")
    packaged_registry = _load_json(package / "known_source_registry.json", "package_tampered")
    contamination = _load_json(package / "contamination_check.json", "package_tampered")
    validation = _load_json(package / "capture_validation.json", "package_tampered")
    no_recognition = _load_json(package / "no_recognition_declaration.json", "package_tampered")
    capture_manifest = _load_json(package / "capture_manifest.json", "package_tampered")
    package_id = str(package_metadata.get("package_id") or "")
    if package_id != package.name or immutable.get("package_id") != package_id:
        _fail("package_tampered", "Package ID binding changed")
    if packaged_registry != registry:
        _fail("unverifiable_provenance", "Packaged known-source registry differs from verified history")
    if source_freeze.get("phase_2k_b_manifest_sha256") != dependency["immutable_manifest_sha256"]:
        _fail("phase_2k_b_contract_changed")
    if source_freeze.get("contamination_registry_fingerprint_sha256") != registry["registry_fingerprint_sha256"]:
        _fail("unverifiable_provenance", "Contamination registry binding changed")
    if contamination.get("registry_fingerprint_sha256") != registry["registry_fingerprint_sha256"]:
        _fail("unverifiable_provenance", "Contamination check registry changed")
    if contamination.get("overall_outcome") != "new_unique_source" or any(
        row.get("outcome") != "new_unique_source" for row in contamination.get("source_checks") or []
    ):
        _fail("known_contaminated_source", outcome="known_contaminated_source")
    if production != _production_policy_freeze(
        CapturePlan(repo, package.parent, package_id, str(package_metadata.get("package_fingerprint_sha256") or ""), session, tuple(source_freeze.get("sources") or []), dependency, registry, protected)
    ):
        _fail("production_policy_changed")
    if validation.get("valid") is not True or validation.get("recognition_ran") is not False:
        _fail("package_tampered")
    if no_recognition.get("recognition_ran") is not False or no_recognition.get("frames_decoded") is not False:
        _fail("package_tampered")
    frozen_sources = source_freeze.get("sources")
    if not isinstance(frozen_sources, list) or len(frozen_sources) != 10:
        _fail("package_tampered", "Source freeze does not contain exactly ten rows")
    if source_freeze.get("source_inventory_fingerprint_sha256") != canonical_json_hash(frozen_sources):
        _fail("package_tampered", "Source inventory fingerprint changed")
    expected_files = {
        "package_metadata.json", "session_metadata.json", "source_layout.json",
        "source_inventory.csv", "source_freeze.json", "production_policy_freeze.json",
        "known_source_registry.json", "contamination_check.json", "capture_validation.json",
        "capture_manifest.json", "blind_review_plan.json", "no_recognition_declaration.json",
    }
    expected_files.update(str(row.get("relative_path") or "") for row in frozen_sources)
    if set(immutable.get("files") or {}) != expected_files:
        _fail("package_tampered", "Capture package contains an unexpected or missing file")
    if (package / "source_inventory.csv").read_bytes() != _inventory_bytes(frozen_sources):
        _fail("package_tampered", "Source inventory CSV differs from the frozen source rows")
    for order, ((checkpoint, camera), row) in enumerate(zip(EXPECTED_BINDINGS, frozen_sources)):
        if (
            row.get("canonical_order") != order
            or row.get("checkpoint_id") != checkpoint
            or row.get("camera_id") != camera
            or row.get("contamination_outcome") != "new_unique_source"
            or row.get("extension") not in VIDEO_SUFFIXES
            or row.get("container") != CONTAINER_BY_SUFFIX.get(row.get("extension"))
            or int(row.get("width", 0)) <= 0
            or int(row.get("height", 0)) <= 0
            or float(row.get("fps", 0)) <= 0
            or int(row.get("frame_count", 0)) <= 0
            or float(row.get("estimated_duration_seconds", 0)) <= 0
            or row.get("capture_timestamp_method") != "operator_supplied"
            or row.get("capture_timestamp_confidence") != "operator_confirmed_exact"
        ):
            _fail("package_tampered", "Frozen source metadata or canonical ordering changed")
        _iso_timestamp(str(row.get("operator_capture_timestamp") or ""))
    capture_sources = capture_manifest.get("sources") or []
    capture_projection = [
        {
            "relative_path": row["relative_path"],
            "sha256": row["sha256"],
            "size_bytes": row["size_bytes"],
            "checkpoint_id": row["checkpoint_id"],
            "camera_id": row["camera_id"],
            "canonical_order": row["canonical_order"],
            "duration_seconds": row["estimated_duration_seconds"],
            "fps": row["fps"],
            "frame_count": row["frame_count"],
            "capture_start_timestamp": row["operator_capture_timestamp"],
        }
        for row in frozen_sources
    ]
    if capture_sources != capture_projection:
        _fail("package_tampered", "Canonical capture manifest differs from source freeze")
    fingerprint_payload = {
        "tool_policy_version": TOOL_POLICY_VERSION,
        "source_freeze_version": SOURCE_FREEZE_VERSION,
        "phase_2k_b_manifest_sha256": dependency["immutable_manifest_sha256"],
        "capture_contract_sha256": dependency["capture_contract_sha256"],
        "production_hashes": protected,
        "session": session,
        "sources": frozen_sources,
        "contamination_registry_fingerprint_sha256": registry["registry_fingerprint_sha256"],
    }
    expected_fingerprint = canonical_json_hash(fingerprint_payload)
    if (
        package_metadata.get("package_fingerprint_sha256") != expected_fingerprint
        or immutable.get("package_fingerprint_sha256") != expected_fingerprint
        or package_id != f"capture-package-{expected_fingerprint[:20]}"
    ):
        _fail("deterministic_id_collision", "Package fingerprint or deterministic ID changed")
    known_hashes = {row["sha256"] for row in registry["entries"]}
    source_hashes = [str(row.get("sha256") or "") for row in capture_manifest.get("sources") or []]
    if len(source_hashes) != 10 or len(set(source_hashes)) != 10:
        _fail("duplicate_inside_package", outcome="duplicate_inside_package")
    if set(source_hashes).intersection(known_hashes):
        _fail("known_contaminated_source", outcome="known_contaminated_source")
    if session.get("session_id") in registry["known_session_ids"]:
        _fail("duplicate_session_id", outcome="duplicate_session_id")
    if session.get("session_id") in _existing_session_ids(package.parent, expected_package_id=package_id):
        _fail("duplicate_session_id", outcome="duplicate_session_id")
    try:
        canonical = validate_capture_manifest(capture_manifest, package_root=package, verify_files=True)
    except Exception as exc:
        raise CaptureIntakeError("canonical_capture_validator_failed", str(exc)) from exc
    return {
        "valid": True,
        "package_id": package_id,
        "package_fingerprint_sha256": package_metadata.get("package_fingerprint_sha256"),
        "session_id": session.get("session_id"),
        "source_count": 10,
        "overall_outcome": "new_unique_source",
        "canonical_capture_validation": canonical,
        "frames_decoded": False,
        "recognition_ran": False,
    }


def materialize_capture_package(
    plan: CapturePlan,
    *,
    copy_function: Callable[[str | os.PathLike[str], str | os.PathLike[str]], Any] = shutil.copyfile,
) -> tuple[Path, bool]:
    output = plan.output_root / plan.package_id
    if output.exists():
        try:
            verified = verify_capture_package(plan.repo_root, output)
        except Exception as exc:
            raise CaptureIntakeError("deterministic_id_collision", str(exc)) from exc
        if (
            verified["package_id"] != plan.package_id
            or verified.get("package_fingerprint_sha256") != plan.package_fingerprint_sha256
        ):
            _fail("deterministic_id_collision")
        return output, True
    plan.output_root.mkdir(parents=True, exist_ok=True)
    staging = plan.output_root / f".{plan.package_id}.staging-{uuid.uuid4().hex}"
    if staging.exists():
        _fail("output_path_unsafe")
    staging.mkdir(parents=False)
    try:
        for row in plan.sources:
            source = Path(row["source_path"])
            if source.is_symlink() or _stat_signature(source) != tuple(row["stat_signature"]):
                _fail("source_changed_during_copy")
            target = staging / Path(_safe_relative(row["relative_path"]))
            target.parent.mkdir(parents=True, exist_ok=True)
            copy_function(source, target)
            source_sha_after = sha256_file(source)
            source_stat_after = _stat_signature(source)
            if source_sha_after != row["sha256"] or source_stat_after != tuple(row["stat_signature"]):
                _fail("source_changed_during_copy")
            if not target.is_file() or target.is_symlink() or target.stat().st_size != row["size_bytes"] or sha256_file(target) != row["sha256"]:
                _fail("source_copy_mismatch")
        for relative, payload in _metadata_files(plan).items():
            path = staging / Path(_safe_relative(relative))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        capture_manifest = _load_json(staging / "capture_manifest.json", "canonical_capture_validator_failed")
        try:
            validate_capture_manifest(capture_manifest, package_root=staging, verify_files=True)
        except Exception as exc:
            raise CaptureIntakeError("canonical_capture_validator_failed", str(exc)) from exc
        files = {
            path.relative_to(staging).as_posix(): {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        immutable = {
            "schema_version": PACKAGE_SCHEMA_VERSION,
            "tool_policy_version": TOOL_POLICY_VERSION,
            "source_freeze_version": SOURCE_FREEZE_VERSION,
            "capture_contract_version": CAPTURE_CONTRACT_VERSION,
            "package_id": plan.package_id,
            "package_fingerprint_sha256": plan.package_fingerprint_sha256,
            "immutable": True,
            "files": files,
        }
        (staging / "immutable_manifest.json").write_bytes(canonical_json_bytes(immutable))
        if output.exists():
            _fail("deterministic_id_collision", "Package target appeared during staging")
        staging.replace(output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    verify_capture_package(plan.repo_root, output)
    return output, False


def dry_run(plan: CapturePlan) -> dict[str, Any]:
    return {
        "valid": True,
        "dry_run": True,
        "proposed_package_id": plan.package_id,
        "session_id": plan.session_metadata["session_id"],
        "source_count": 10,
        "overall_outcome": "new_unique_source",
        "files_copied": 0,
        "output_created": False,
        "frames_decoded": False,
        "recognition_ran": False,
    }


def _failure_matrix_bytes() -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=("Reason_Code", "Description", "Fail_Closed", "Override_Permitted"), lineterminator="\n")
    writer.writeheader()
    for code, description in sorted(REASON_DESCRIPTIONS.items()):
        writer.writerow({"Reason_Code": code, "Description": description, "Fail_Closed": "true", "Override_Permitted": "false"})
    return buffer.getvalue().encode("utf-8")


def materialize_tooling_validation(repo_root: Path, output_root: Path | None = None) -> tuple[Path, bool]:
    repo = Path(repo_root).resolve()
    dependency = _phase_2k_b_dependency(repo)
    registry = _known_source_registry(dependency)
    protected = _protected_hashes(repo)
    policy = {
        "schema_version": 1,
        "tool_policy_version": TOOL_POLICY_VERSION,
        "source_freeze_version": SOURCE_FREEZE_VERSION,
        "capture_contract_version": CAPTURE_CONTRACT_VERSION,
        "confirmation_token": CONFIRMATION_TOKEN,
        "preparation_only": True,
        "recognition_authorized": False,
        "frame_decoding_permitted": False,
        "contamination_override_permitted": False,
        "atomic_copy_and_rename_required": True,
        "original_source_modification_permitted": False,
    }
    summary = {
        "schema_version": 1,
        "registry_version": CONTAMINATION_REGISTRY_VERSION,
        "registry_fingerprint_sha256": registry["registry_fingerprint_sha256"],
        "known_source_entry_count": registry["entry_count"],
        "known_session_ids": registry["known_session_ids"],
        "verified_evidence_artifact_count": len(registry["verified_evidence_artifacts"]),
        "phase_coverage": ["Phase 1 source freezes", "Phase 1.2N", "Phase 2G", "Phase 2I", "Phase 2K-A", "Phase 2K-B"],
        "override_permitted": False,
    }
    compatibility = {
        "schema_version": 1,
        "compatible": True,
        "phase_2k_b_manifest_sha256": dependency["immutable_manifest_sha256"],
        "capture_contract_version": CAPTURE_CONTRACT_VERSION,
        "capture_contract_sha256": dependency["capture_contract_sha256"],
        "diagnostic_schema_sha256": dependency["diagnostic_schema_sha256"],
        "blind_schema_sha256": dependency["blind_schema_sha256"],
        "acceptance_sha256": dependency["acceptance_sha256"],
        "retention_sha256": dependency["retention_sha256"],
        "canonical_validator": "scripts/run_product_phase_2k_carry_forward.py validate-capture --package-dir <path>",
    }
    no_recognition = {
        "schema_version": 1,
        "phase": "Product Phase 2K-C0 tooling validation",
        "real_capture_package_created": False,
        "classroom_video_opened": False,
        "frames_decoded": False,
        "recognition_ran": False,
        "yunet_inference_ran": False,
        "sface_inference_ran": False,
        "attendance_or_authority_changed": False,
    }
    files = {
        "tooling_policy.json": canonical_json_bytes(policy),
        "failure_reason_matrix.csv": _failure_matrix_bytes(),
        "known_contamination_registry_summary.json": canonical_json_bytes(summary),
        "phase_2k_b_contract_compatibility.json": canonical_json_bytes(compatibility),
        "protected_hashes.json": canonical_json_bytes(protected),
        "no_recognition_declaration.json": canonical_json_bytes(no_recognition),
    }
    run_fingerprint = canonical_json_hash({name: hashlib.sha256(payload).hexdigest() for name, payload in sorted(files.items())})
    run_id = f"intake-tooling-{run_fingerprint[:20]}"
    root = Path(output_root).resolve() if output_root else repo / "attendance_output" / "product_workflow" / "phase_2k_capture_intake_tooling"
    output = root / run_id
    if output.exists():
        manifest = _verify_file_manifest(output, expected_policy=TOOL_POLICY_VERSION)
        if manifest.get("tooling_run_id") != run_id:
            _fail("deterministic_id_collision")
        return output, True
    root.mkdir(parents=True, exist_ok=True)
    staging = root / f".{run_id}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False)
    try:
        for relative, payload in files.items():
            (staging / relative).write_bytes(payload)
        metadata = {
            relative: {"sha256": sha256_file(staging / relative), "size_bytes": (staging / relative).stat().st_size}
            for relative in sorted(files)
        }
        manifest = {
            "schema_version": TOOLING_SCHEMA_VERSION,
            "tool_policy_version": TOOL_POLICY_VERSION,
            "tooling_run_id": run_id,
            "immutable": True,
            "files": metadata,
        }
        (staging / "immutable_manifest.json").write_bytes(canonical_json_bytes(manifest))
        if output.exists():
            _fail("deterministic_id_collision", "Tooling target appeared during staging")
        staging.replace(output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    _verify_file_manifest(output, expected_policy=TOOL_POLICY_VERSION)
    return output, False


def verify_tooling_validation(output_dir: Path) -> dict[str, Any]:
    manifest = _verify_file_manifest(Path(output_dir).resolve(), expected_policy=TOOL_POLICY_VERSION)
    if not str(manifest.get("tooling_run_id") or "").startswith("intake-tooling-"):
        _fail("package_tampered")
    return manifest


def official_and_candidate_counts() -> dict[str, dict[str, int]]:
    return {"official": dict(OFFICIAL_COUNTS), "candidate": dict(CANDIDATE_COUNTS)}
