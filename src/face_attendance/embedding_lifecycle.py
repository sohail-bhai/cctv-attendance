from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np
import pandas as pd

from .config import SFACE_MODEL, YUNET_MODEL
from .diagnostics import json_safe
from .embedding_forensics import (
    EXPECTED_REVIEWED_TRACKS,
    OFFICIAL_ATTENDANCE_CHECKPOINTS,
    OFFICIAL_MARGIN_THRESHOLD,
    OFFICIAL_MATCH_THRESHOLD,
    EmbeddingForensicsError,
    _canonical,
    assert_candidate_not_rejected_for_approval,
    load_embedding_payload,
)
from .face_engine import FaceEngine
from .forensic_review import (
    ApprovalValidationResult,
    ForensicReviewError,
    load_forensic_review_package,
    validate_forensic_approvals,
)
from .shadow_validation import ShadowValidationError, verify_output_manifest
from .tracklet_review import _sha256_file


LIFECYCLE_SCHEMA_VERSION = 1
VERSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{5,95}$")
ENROLLMENT_KEEP_ACTION = "keep"
CCTV_APPROVE_ACTION = "approve_for_candidate_embedding_version"
ADAPTATION_SESSIONS = {
    "2026-06-30__B51__P1__CVO",
    "2026-06-30__B51__P2__CVO",
}
REQUIRED_UNTOUCHED_SESSION = "MON_P3"


class EmbeddingLifecycleError(ValueError):
    pass


@dataclass(frozen=True)
class ApprovalBundle:
    enrollment: ApprovalValidationResult
    cctv: ApprovalValidationResult
    enrollment_mapping: pd.DataFrame
    cctv_mapping: pd.DataFrame
    summary: dict[str, Any]


@dataclass(frozen=True)
class VersionBuildResult:
    version_dir: Path
    version_id: str
    manifest: dict[str, Any]
    idempotent_reuse: bool


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(json_safe(payload), indent=2, ensure_ascii=True), encoding="utf-8"
    )


def _stable_digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(json_safe(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    ).hexdigest()


def _load_private_provenance(row: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(str(row.get("Private_Provenance_JSON") or "{}"))
    except json.JSONDecodeError as exc:
        raise EmbeddingLifecycleError(
            f"Invalid private provenance JSON for {row.get('Item_ID')}"
        ) from exc
    if not isinstance(value, dict):
        raise EmbeddingLifecycleError(
            f"Private provenance must be an object for {row.get('Item_ID')}"
        )
    return value


def validate_embedding_approval_bundle(
    *,
    enrollment_review_package: Path,
    enrollment_approvals_path: Path,
    cctv_review_package: Path,
    cctv_approvals_path: Path,
) -> ApprovalBundle:
    try:
        enrollment = validate_forensic_approvals(
            enrollment_review_package,
            enrollment_approvals_path,
            expected_kind="enrollment_audit",
            verify_sources=True,
        )
        cctv = validate_forensic_approvals(
            cctv_review_package,
            cctv_approvals_path,
            expected_kind="verified_cctv_enrollment",
            verify_sources=True,
        )
        _, _, enrollment_mapping = load_forensic_review_package(enrollment_review_package)
        _, _, cctv_mapping = load_forensic_review_package(cctv_review_package)
    except ForensicReviewError as exc:
        raise EmbeddingLifecycleError(str(exc)) from exc
    if not enrollment.summary.get("complete") or not cctv.summary.get("complete"):
        raise EmbeddingLifecycleError("Both human approval files must be complete")
    enrollment_items = set(enrollment.approvals["Item_ID"].astype(str))
    cctv_items = set(cctv.approvals["Item_ID"].astype(str))
    if enrollment_items != set(enrollment_mapping["Item_ID"].astype(str)):
        raise EmbeddingLifecycleError("Enrollment approvals do not match private source provenance")
    if cctv_items != set(cctv_mapping["Item_ID"].astype(str)):
        raise EmbeddingLifecycleError("CCTV approvals do not match private source provenance")
    summary = {
        "schema_version": LIFECYCLE_SCHEMA_VERSION,
        "complete": True,
        "enrollment_package_id": enrollment.summary["package_id"],
        "cctv_package_id": cctv.summary["package_id"],
        "enrollment_reviewed_items": int(len(enrollment.approvals)),
        "cctv_reviewed_items": int(len(cctv.approvals)),
        "enrollment_action_counts": enrollment.summary["action_counts"],
        "cctv_action_counts": cctv.summary["action_counts"],
        "enrollment_approvals_sha256": _sha256_file(Path(enrollment_approvals_path)),
        "cctv_approvals_sha256": _sha256_file(Path(cctv_approvals_path)),
        "sources_verified": True,
        "production_files_changed": False,
    }
    return ApprovalBundle(
        enrollment=enrollment,
        cctv=cctv,
        enrollment_mapping=enrollment_mapping,
        cctv_mapping=cctv_mapping,
        summary=summary,
    )


def _verify_forensic_run(forensic_run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    forensic_run_dir = Path(forensic_run_dir).resolve()
    try:
        verify_output_manifest(forensic_run_dir, forensic_run_dir / "output_manifest.json")
    except ShadowValidationError as exc:
        raise EmbeddingLifecycleError(str(exc)) from exc
    summary_path = forensic_run_dir / "forensic_summary.json"
    benchmark_manifest_path = forensic_run_dir / "multisession_ground_truth_manifest.json"
    if not summary_path.is_file() or not benchmark_manifest_path.is_file():
        raise EmbeddingLifecycleError(f"Forensic run is incomplete: {forensic_run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    benchmark_manifest = json.loads(benchmark_manifest_path.read_text(encoding="utf-8"))
    if (
        int(summary.get("benchmark_reviewed_tracks") or 0) != EXPECTED_REVIEWED_TRACKS
        or summary.get("protected_state_unchanged") is not True
        or summary.get("candidate_embedding_version_built") is not False
    ):
        raise EmbeddingLifecycleError("Forensic run does not satisfy the Phase 1.2I-A safety contract")
    return summary, benchmark_manifest


def _resolved_record_path(repo_root: Path, value: Any) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    return (path if path.is_absolute() else Path(repo_root) / path).resolve()


def _normalize_embedding(value: Any) -> np.ndarray | None:
    try:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if not vector.size or not np.isfinite(vector).all():
        return None
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return None
    return vector / norm


def _extract_single_face_embedding(
    image_path: Path,
    engine: FaceEngine,
) -> tuple[np.ndarray, dict[str, Any]]:
    image = cv2.imread(str(image_path))
    if image is None or image.size == 0:
        raise EmbeddingLifecycleError(f"Approved CCTV crop cannot be decoded: {image_path}")
    faces = engine.detect_faces(image)
    if len(faces) != 1:
        raise EmbeddingLifecycleError(
            f"Approved CCTV crop must contain exactly one detectable face: {image_path} ({len(faces)} found)"
        )
    vector = _normalize_embedding(engine.extract_feature(image, faces[0]))
    if vector is None:
        raise EmbeddingLifecycleError(f"Could not extract SFace embedding from approved crop: {image_path}")
    return vector, {
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "detector_score": float(engine.face_score(faces[0])),
    }


def _version_artifact_hashes(version_dir: Path, excluded: Iterable[str] = ()) -> dict[str, str]:
    excluded_set = set(excluded)
    return {
        path.relative_to(version_dir).as_posix(): _sha256_file(path)
        for path in sorted(Path(version_dir).rglob("*"))
        if path.is_file() and path.relative_to(version_dir).as_posix() not in excluded_set
    }


def verify_embedding_version(version_dir: Path) -> dict[str, Any]:
    version_dir = Path(version_dir).resolve()
    manifest_path = version_dir / "version_manifest.json"
    if not manifest_path.is_file():
        raise EmbeddingLifecycleError(f"Version manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for relative, expected in dict(manifest.get("artifact_sha256") or {}).items():
        path = version_dir / relative
        if not path.is_file() or _sha256_file(path) != str(expected):
            failures.append(relative)
    if failures:
        raise EmbeddingLifecycleError(
            "Embedding version integrity check failed: " + ", ".join(failures[:10])
        )
    if str(manifest.get("version_id") or "") != version_dir.name:
        raise EmbeddingLifecycleError("Embedding version directory and manifest IDs do not match")
    return manifest


def build_versioned_embeddings(
    *,
    repo_root: Path,
    forensic_run_dir: Path,
    enrollment_review_package: Path,
    enrollment_approvals_path: Path,
    cctv_review_package: Path,
    cctv_approvals_path: Path,
    production_embeddings_path: Path,
    production_summary_path: Path,
    versions_root: Path,
    version_id: str = "",
    yunet_model: Path = YUNET_MODEL,
    sface_model: Path = SFACE_MODEL,
    candidate_registry_dir: Path | None = None,
    source_candidate_ids: Iterable[str] = (),
    feature_extractor: Callable[[Path], tuple[np.ndarray, dict[str, Any]]] | None = None,
) -> VersionBuildResult:
    started = time.perf_counter()
    repo_root = Path(repo_root).resolve()
    forensic_run_dir = Path(forensic_run_dir).resolve()
    production_embeddings_path = Path(production_embeddings_path).resolve()
    production_summary_path = Path(production_summary_path).resolve()
    versions_root = Path(versions_root).resolve()
    if not production_embeddings_path.is_file() or not production_summary_path.is_file():
        raise EmbeddingLifecycleError("Current production embedding database and summary are required")
    if production_embeddings_path.parent == versions_root:
        raise EmbeddingLifecycleError("Version output root must not be the production model directory itself")
    forensic_summary, benchmark_manifest = _verify_forensic_run(forensic_run_dir)
    bundle = validate_embedding_approval_bundle(
        enrollment_review_package=enrollment_review_package,
        enrollment_approvals_path=enrollment_approvals_path,
        cctv_review_package=cctv_review_package,
        cctv_approvals_path=cctv_approvals_path,
    )
    registry = Path(candidate_registry_dir or repo_root / "models" / "candidate_registry" / "rejected")
    source_candidate_ids = sorted({str(value).strip() for value in source_candidate_ids if str(value).strip()})
    for candidate_id in source_candidate_ids:
        try:
            assert_candidate_not_rejected_for_approval(candidate_id, registry)
        except EmbeddingForensicsError as exc:
            raise EmbeddingLifecycleError(str(exc)) from exc
    input_fingerprint = {
        "parent_embeddings_sha256": _sha256_file(production_embeddings_path),
        "parent_summary_sha256": _sha256_file(production_summary_path),
        "forensic_output_manifest_sha256": _sha256_file(forensic_run_dir / "output_manifest.json"),
        "benchmark_manifest_sha256": _sha256_file(
            forensic_run_dir / "multisession_ground_truth_manifest.json"
        ),
        "enrollment_package_manifest_sha256": _sha256_file(
            Path(enrollment_review_package) / "package_manifest.json"
        ),
        "cctv_package_manifest_sha256": _sha256_file(
            Path(cctv_review_package) / "package_manifest.json"
        ),
        "enrollment_approvals_sha256": bundle.summary["enrollment_approvals_sha256"],
        "cctv_approvals_sha256": bundle.summary["cctv_approvals_sha256"],
        "yunet_model_sha256": _sha256_file(Path(yunet_model)),
        "sface_model_sha256": _sha256_file(Path(sface_model)),
        "source_candidate_ids": source_candidate_ids,
    }
    build_fingerprint = _stable_digest(input_fingerprint)
    version_id = version_id.strip() or f"embv-{build_fingerprint[:16]}"
    if not VERSION_ID_PATTERN.fullmatch(version_id):
        raise EmbeddingLifecycleError(
            "Version ID must be 6-96 characters using letters, numbers, dots, dashes, or underscores"
        )
    version_dir = versions_root / version_id
    if version_dir.exists():
        existing = verify_embedding_version(version_dir)
        if str(existing.get("build_fingerprint_sha256") or "") != build_fingerprint:
            raise EmbeddingLifecycleError(
                f"Embedding version ID already exists with different inputs: {version_id}"
            )
        return VersionBuildResult(version_dir, version_id, existing, True)

    parent_payload = load_embedding_payload(production_embeddings_path)
    parent_records = list(parent_payload["records"])
    enrollment_decisions = bundle.enrollment.approvals.merge(
        bundle.enrollment_mapping,
        on=["Package_ID", "Item_ID"],
        how="inner",
        validate="one_to_one",
    )
    enrollment_by_source = {
        str(Path(row["Source_Path"]).resolve()).lower(): row
        for row in enrollment_decisions.to_dict("records")
    }
    retained_records: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for index, record in enumerate(parent_records):
        source_path = _resolved_record_path(repo_root, record.get("image_path"))
        decision = enrollment_by_source.get(str(source_path).lower()) if source_path else None
        if decision and str(decision.get("Review_Action")) != ENROLLMENT_KEEP_ACTION:
            excluded_rows.append(
                {
                    "Parent_Record_Index": index,
                    "Canonical_Roll": _canonical(record.get("roll_no")),
                    "Image_Path": record.get("image_path"),
                    "Review_Item_ID": decision.get("Item_ID"),
                    "Review_Action": decision.get("Review_Action"),
                    "Source_SHA256": decision.get("Source_SHA256"),
                }
            )
            continue
        retained_records.append(record)
        source_rows.append(
            {
                "Record_Order": len(retained_records) - 1,
                "Canonical_Roll": _canonical(record.get("roll_no")),
                "Provenance_Type": "immutable_parent_embedding_record",
                "Parent_Record_Index": index,
                "Image_Path": record.get("image_path"),
                "Source_Path": str(source_path) if source_path else "",
                "Source_SHA256": _sha256_file(source_path) if source_path and source_path.is_file() else "",
                "Approval_Item_ID": decision.get("Item_ID") if decision else "",
                "Approval_Action": decision.get("Review_Action") if decision else "not_flagged_by_forensic_audit",
            }
        )

    cctv_decisions = bundle.cctv.approvals.merge(
        bundle.cctv_mapping,
        on=["Package_ID", "Item_ID"],
        how="inner",
        validate="one_to_one",
    )
    approved_cctv = cctv_decisions[
        cctv_decisions["Review_Action"].eq(CCTV_APPROVE_ACTION)
    ].sort_values("Item_ID", kind="stable")
    engine = None
    if feature_extractor is None and len(approved_cctv):
        engine = FaceEngine(Path(yunet_model), Path(sface_model), detection_score=0.60)

    versions_root.mkdir(parents=True, exist_ok=True)
    temporary_dir = versions_root / f".{version_id}.building-{os.getpid()}"
    if temporary_dir.exists():
        raise EmbeddingLifecycleError(f"Stale version build directory exists: {temporary_dir}")
    temporary_dir.mkdir(parents=True, exist_ok=False)
    try:
        staged_cctv_root = temporary_dir / "sources" / "cctv"
        added_records: list[dict[str, Any]] = []
        for row in approved_cctv.to_dict("records"):
            provenance = _load_private_provenance(row)
            actual_roll = _canonical(provenance.get("actual_roll"))
            if not actual_roll or str(provenance.get("identity_source")) != "human_review_actual_roll":
                raise EmbeddingLifecycleError(
                    f"Approved CCTV item lacks human Actual_Roll provenance: {row.get('Item_ID')}"
                )
            source_path = Path(str(row.get("Source_Path") or "")).resolve()
            if not source_path.is_file() or _sha256_file(source_path) != str(row.get("Source_SHA256")):
                raise EmbeddingLifecycleError(
                    f"Approved CCTV source changed after review: {row.get('Item_ID')}"
                )
            if feature_extractor is None:
                vector, metrics = _extract_single_face_embedding(source_path, engine)
            else:
                vector, metrics = feature_extractor(source_path)
                normalized = _normalize_embedding(vector)
                if normalized is None:
                    raise EmbeddingLifecycleError(
                        f"Feature extractor returned an invalid embedding for {row.get('Item_ID')}"
                    )
                vector = normalized
            destination = staged_cctv_root / actual_roll / f"{row['Item_ID']}{source_path.suffix.lower()}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)
            if _sha256_file(destination) != str(row.get("Source_SHA256")):
                raise EmbeddingLifecycleError(f"Staged CCTV crop hash mismatch: {destination}")
            final_destination = version_dir / destination.relative_to(temporary_dir)
            try:
                final_image_path = final_destination.relative_to(repo_root).as_posix()
            except ValueError:
                final_image_path = str(final_destination)
            added_records.append(
                {
                    "roll_no": actual_roll,
                    "image_path": final_image_path,
                    "detection_score": float(metrics.get("detector_score", 0.0)),
                    "embedding": vector,
                }
            )
            source_rows.append(
                {
                    "Record_Order": len(retained_records) + len(added_records) - 1,
                    "Canonical_Roll": actual_roll,
                    "Provenance_Type": "human_approved_cctv_crop",
                    "Parent_Record_Index": "",
                    "Image_Path": final_image_path,
                    "Source_Path": str(source_path),
                    "Source_SHA256": _sha256_file(source_path),
                    "Approval_Item_ID": row.get("Item_ID"),
                    "Approval_Action": row.get("Review_Action"),
                    "Human_Identity_Source": "Actual_Roll",
                    "Source_Provenance_JSON": json.dumps(provenance, sort_keys=True, ensure_ascii=True),
                }
            )
        records = retained_records + added_records
        created_at = datetime.now().isoformat(timespec="seconds")
        version_metadata = {
            **dict(parent_payload.get("metadata") or {}),
            "embedding_version_id": version_id,
            "parent_embeddings_sha256": input_fingerprint["parent_embeddings_sha256"],
            "build_fingerprint_sha256": build_fingerprint,
            "status": "built_unapproved",
            "human_approval_packages": [
                bundle.summary["enrollment_package_id"],
                bundle.summary["cctv_package_id"],
            ],
            "production_match_threshold_unchanged": OFFICIAL_MATCH_THRESHOLD,
            "production_margin_threshold_unchanged": OFFICIAL_MARGIN_THRESHOLD,
            "attendance_checkpoint_rule_unchanged": OFFICIAL_ATTENDANCE_CHECKPOINTS,
        }
        with (temporary_dir / "student_embeddings.pkl").open("wb") as file_obj:
            pickle.dump(
                {
                    "version": 1,
                    "created_at": created_at,
                    "metadata": version_metadata,
                    "records": records,
                },
                file_obj,
            )
        per_roll = pd.DataFrame(
            [{"Canonical_Roll": _canonical(record.get("roll_no"))} for record in records]
        )
        coverage = (
            per_roll.value_counts("Canonical_Roll")
            .rename("Embedding_Count")
            .reset_index()
            .sort_values("Canonical_Roll", kind="stable")
        ) if len(per_roll) else pd.DataFrame(columns=["Canonical_Roll", "Embedding_Count"])
        parent_counts = pd.Series(
            [_canonical(record.get("roll_no")) for record in parent_records]
        ).value_counts()
        added_counts = pd.Series(
            [_canonical(record.get("roll_no")) for record in added_records], dtype=str
        ).value_counts()
        coverage["Parent_Embedding_Count"] = coverage["Canonical_Roll"].map(parent_counts).fillna(0).astype(int)
        coverage["Approved_CCTV_Added"] = coverage["Canonical_Roll"].map(added_counts).fillna(0).astype(int)
        coverage.to_csv(temporary_dir / "student_coverage.csv", index=False, encoding="utf-8-sig")
        coverage.rename(columns={"Canonical_Roll": "Roll_Number", "Embedding_Count": "Faces_Used"})[
            ["Roll_Number", "Faces_Used"]
        ].to_csv(temporary_dir / "embedding_summary.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(excluded_rows).to_csv(
            temporary_dir / "excluded_image_decisions.csv", index=False, encoding="utf-8-sig"
        )
        source_manifest = {
            "schema_version": LIFECYCLE_SCHEMA_VERSION,
            "version_id": version_id,
            "build_fingerprint_sha256": build_fingerprint,
            "input_fingerprint": input_fingerprint,
            "source_records": source_rows,
            "source_record_count": len(source_rows),
            "source_hashes_verified_at_build": True,
            "live_dataset_modified": False,
        }
        _write_json(temporary_dir / "source_manifest.json", source_manifest)
        regression_status = {
            "schema_version": LIFECYCLE_SCHEMA_VERSION,
            "version_id": version_id,
            "status": "not_evaluated",
            "regression_passed": False,
            "untouched_session_validation_passed": False,
            "required_benchmark_id": benchmark_manifest.get("benchmark_id"),
            "required_untouched_session": REQUIRED_UNTOUCHED_SESSION,
            "promotion_allowed": False,
        }
        _write_json(temporary_dir / "regression_status.json", regression_status)
        rollback_metadata = {
            "schema_version": LIFECYCLE_SCHEMA_VERSION,
            "version_id": version_id,
            "prepared_before_promotion": True,
            "production_embeddings_path": str(production_embeddings_path),
            "production_embeddings_sha256": _sha256_file(production_embeddings_path),
            "production_summary_path": str(production_summary_path),
            "production_summary_sha256": _sha256_file(production_summary_path),
            "backup_directory": str((version_dir / "rollback" / "production_before_promotion").resolve()),
            "rollback_command": (
                f"python scripts/validate_tracklet_ground_truth.py rollback-embedding-version "
                f"--version-dir \"{version_dir}\""
            ),
            "promotion_executed": False,
        }
        _write_json(temporary_dir / "rollback_metadata.json", rollback_metadata)
        current_rolls = {_canonical(record.get("roll_no")) for record in parent_records}
        candidate_rolls = {_canonical(record.get("roll_no")) for record in records}
        diff = {
            "parent_embedding_records": len(parent_records),
            "candidate_embedding_records": len(records),
            "excluded_parent_records": len(excluded_rows),
            "approved_cctv_records_added": len(added_records),
            "parent_students": len(current_rolls),
            "candidate_students": len(candidate_rolls),
            "students_added": sorted(candidate_rolls - current_rolls),
            "students_removed": sorted(current_rolls - candidate_rolls),
        }
        _write_json(temporary_dir / "current_vs_candidate_diff.json", diff)
        (temporary_dir / "build_report.md").write_text(
            "\n".join(
                [
                    f"# Embedding Version {version_id}",
                    "",
                    "Status: `built_unapproved`",
                    "",
                    f"- Parent records: {len(parent_records)}",
                    f"- Excluded parent records: {len(excluded_rows)}",
                    f"- Human-approved CCTV records added: {len(added_records)}",
                    f"- Candidate records: {len(records)}",
                    "- Production embeddings changed: no",
                    "- Live datasets changed: no",
                    "- Promotion performed: no",
                    "- Regression and untouched MON_P3 validation are required before promotion.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        artifact_hashes = _version_artifact_hashes(
            temporary_dir,
            excluded={"version_manifest.json", "regression_status.json", "promotion_status.json", "rollback_status.json"},
        )
        manifest = {
            "schema_version": LIFECYCLE_SCHEMA_VERSION,
            "version_id": version_id,
            "status": "built_unapproved",
            "promotion_status": "not_promoted",
            "build_timestamp": created_at,
            "build_fingerprint_sha256": build_fingerprint,
            "parent_version": str(dict(parent_payload.get("metadata") or {}).get("embedding_version_id") or "legacy_production"),
            "parent_embeddings_path": str(production_embeddings_path),
            "parent_embeddings_sha256": _sha256_file(production_embeddings_path),
            "parent_summary_sha256": _sha256_file(production_summary_path),
            "model_hashes": {
                "yunet_sha256": _sha256_file(Path(yunet_model)),
                "sface_sha256": _sha256_file(Path(sface_model)),
            },
            "dataset_manifest_hash": input_fingerprint["forensic_output_manifest_sha256"],
            "approval_csv_hashes": {
                "enrollment": bundle.summary["enrollment_approvals_sha256"],
                "cctv": bundle.summary["cctv_approvals_sha256"],
            },
            "benchmark": {
                "id": benchmark_manifest.get("benchmark_id"),
                "manifest_sha256": input_fingerprint["benchmark_manifest_sha256"],
                "reviewed_tracks": forensic_summary.get("benchmark_reviewed_tracks"),
            },
            "source_candidate_ids": source_candidate_ids,
            "record_counts": diff,
            "artifact_sha256": artifact_hashes,
            "mutable_lifecycle_status_files_excluded_from_artifact_hash": [
                "regression_status.json",
                "promotion_status.json",
                "rollback_status.json",
            ],
            "production_embeddings_overwritten_by_build": False,
            "live_dataset_modified": False,
            "promotion_performed": False,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        }
        _write_json(temporary_dir / "version_manifest.json", manifest)
        temporary_dir.replace(version_dir)
        verified = verify_embedding_version(version_dir)
        return VersionBuildResult(version_dir, version_id, verified, False)
    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise


def evaluate_embedding_version(
    *,
    version_dir: Path,
    forensic_run_dir: Path,
    regression_results_path: Path,
    untouched_validation_path: Path | None = None,
) -> dict[str, Any]:
    version_dir = Path(version_dir).resolve()
    manifest = verify_embedding_version(version_dir)
    _, benchmark_manifest = _verify_forensic_run(forensic_run_dir)
    regression_results_path = Path(regression_results_path).resolve()
    if not regression_results_path.is_file():
        raise EmbeddingLifecycleError(f"Regression results not found: {regression_results_path}")
    regression = json.loads(regression_results_path.read_text(encoding="utf-8"))
    expected_benchmark_hash = _sha256_file(
        Path(forensic_run_dir) / "multisession_ground_truth_manifest.json"
    )
    if (
        str(regression.get("version_id") or "") != manifest["version_id"]
        or str(regression.get("benchmark_id") or "") != str(benchmark_manifest.get("benchmark_id") or "")
        or str(regression.get("benchmark_manifest_sha256") or "") != expected_benchmark_hash
        or int(regression.get("reviewed_tracks") or 0) != EXPECTED_REVIEWED_TRACKS
    ):
        raise EmbeddingLifecycleError("Regression evidence does not match this version and frozen benchmark")
    if (
        float(regression.get("match_threshold", OFFICIAL_MATCH_THRESHOLD)) != OFFICIAL_MATCH_THRESHOLD
        or float(regression.get("margin_threshold", OFFICIAL_MARGIN_THRESHOLD)) != OFFICIAL_MARGIN_THRESHOLD
        or int(regression.get("attendance_checkpoints", OFFICIAL_ATTENDANCE_CHECKPOINTS))
        != OFFICIAL_ATTENDANCE_CHECKPOINTS
    ):
        raise EmbeddingLifecycleError("Regression evidence changed official recognition or attendance thresholds")
    regression_passed = regression.get("regression_passed") is True
    untouched: dict[str, Any] = {
        "provided": False,
        "session_id": "",
        "validation_passed": False,
        "untouched_before_validation": False,
    }
    if untouched_validation_path:
        untouched_validation_path = Path(untouched_validation_path).resolve()
        evidence = json.loads(untouched_validation_path.read_text(encoding="utf-8"))
        session_id = str(evidence.get("session_id") or "")
        if (
            str(evidence.get("version_id") or "") != manifest["version_id"]
            or session_id in ADAPTATION_SESSIONS
            or session_id != REQUIRED_UNTOUCHED_SESSION
            or evidence.get("untouched_before_validation") is not True
        ):
            raise EmbeddingLifecycleError("Untouched-session evidence is not valid MON_P3 evidence for this version")
        untouched = {
            "provided": True,
            "session_id": session_id,
            "validation_passed": evidence.get("validation_passed") is True,
            "untouched_before_validation": True,
            "evidence_path": str(untouched_validation_path),
            "evidence_sha256": _sha256_file(untouched_validation_path),
        }
    status = {
        "schema_version": LIFECYCLE_SCHEMA_VERSION,
        "version_id": manifest["version_id"],
        "status": "passed" if regression_passed else "failed",
        "regression_passed": regression_passed,
        "regression_evidence": {
            "path": str(regression_results_path),
            "sha256": _sha256_file(regression_results_path),
            "benchmark_id": benchmark_manifest.get("benchmark_id"),
            "benchmark_manifest_sha256": expected_benchmark_hash,
            "reviewed_tracks": EXPECTED_REVIEWED_TRACKS,
        },
        "untouched_session_validation": untouched,
        "untouched_session_validation_passed": bool(untouched["validation_passed"]),
        "promotion_allowed": bool(regression_passed and untouched["validation_passed"]),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json(version_dir / "regression_status.json", status)
    return status


def _atomic_replace_from(source: Path, target: Path) -> None:
    temporary = target.with_name(f".{target.name}.promoting-{os.getpid()}")
    if temporary.exists():
        raise EmbeddingLifecycleError(f"Stale atomic promotion file exists: {temporary}")
    shutil.copy2(source, temporary)
    if _sha256_file(temporary) != _sha256_file(source):
        temporary.unlink(missing_ok=True)
        raise EmbeddingLifecycleError(f"Atomic promotion copy hash mismatch: {target}")
    temporary.replace(target)


def promote_embedding_version(
    *,
    version_dir: Path,
    production_embeddings_path: Path,
    production_summary_path: Path,
    candidate_registry_dir: Path,
) -> dict[str, Any]:
    version_dir = Path(version_dir).resolve()
    manifest = verify_embedding_version(version_dir)
    if manifest.get("status") != "built_unapproved":
        raise EmbeddingLifecycleError("Only a built_unapproved version may enter promotion checks")
    for candidate_id in manifest.get("source_candidate_ids", []):
        try:
            assert_candidate_not_rejected_for_approval(str(candidate_id), candidate_registry_dir)
        except EmbeddingForensicsError as exc:
            raise EmbeddingLifecycleError(str(exc)) from exc
    status_path = version_dir / "regression_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    untouched = dict(status.get("untouched_session_validation") or {})
    if status.get("regression_passed") is not True:
        raise EmbeddingLifecycleError("Promotion refused: frozen multisession regression has not passed")
    if (
        status.get("untouched_session_validation_passed") is not True
        or untouched.get("session_id") != REQUIRED_UNTOUCHED_SESSION
        or untouched.get("untouched_before_validation") is not True
    ):
        raise EmbeddingLifecycleError("Promotion refused: passed untouched MON_P3 validation is required")
    production_embeddings_path = Path(production_embeddings_path).resolve()
    production_summary_path = Path(production_summary_path).resolve()
    if _sha256_file(production_embeddings_path) != manifest["parent_embeddings_sha256"]:
        raise EmbeddingLifecycleError("Production embeddings changed since the version build; promotion refused")
    if _sha256_file(production_summary_path) != manifest["parent_summary_sha256"]:
        raise EmbeddingLifecycleError("Production embedding summary changed since the version build; promotion refused")
    rollback_dir = version_dir / "rollback" / "production_before_promotion"
    if rollback_dir.exists():
        raise EmbeddingLifecycleError("Rollback backup already exists; refusing ambiguous repeat promotion")
    rollback_dir.mkdir(parents=True, exist_ok=False)
    backup_embeddings = rollback_dir / "student_embeddings.pkl"
    backup_summary = rollback_dir / "embedding_summary.csv"
    shutil.copy2(production_embeddings_path, backup_embeddings)
    shutil.copy2(production_summary_path, backup_summary)
    rollback_manifest = {
        "schema_version": LIFECYCLE_SCHEMA_VERSION,
        "version_id": manifest["version_id"],
        "production_embeddings_path": str(production_embeddings_path),
        "production_summary_path": str(production_summary_path),
        "backup_embeddings_sha256": _sha256_file(backup_embeddings),
        "backup_summary_sha256": _sha256_file(backup_summary),
        "candidate_embeddings_sha256": _sha256_file(version_dir / "student_embeddings.pkl"),
        "candidate_summary_sha256": _sha256_file(version_dir / "embedding_summary.csv"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rollback_completed": False,
    }
    _write_json(rollback_dir / "rollback_manifest.json", rollback_manifest)
    try:
        _atomic_replace_from(version_dir / "student_embeddings.pkl", production_embeddings_path)
        _atomic_replace_from(version_dir / "embedding_summary.csv", production_summary_path)
    except Exception as promotion_error:
        restoration_errors = []
        for backup, production in (
            (backup_embeddings, production_embeddings_path),
            (backup_summary, production_summary_path),
        ):
            try:
                _atomic_replace_from(backup, production)
            except Exception as restore_error:
                restoration_errors.append(f"{production}: {restore_error}")
        if restoration_errors:
            raise EmbeddingLifecycleError(
                "Promotion failed and automatic restoration was incomplete: "
                + "; ".join(restoration_errors)
            ) from promotion_error
        raise EmbeddingLifecycleError(
            "Promotion failed; the pre-promotion embedding files were restored"
        ) from promotion_error
    promotion = {
        "schema_version": LIFECYCLE_SCHEMA_VERSION,
        "version_id": manifest["version_id"],
        "status": "promoted",
        "promoted_at": datetime.now().isoformat(timespec="seconds"),
        "regression_status_sha256": _sha256_file(status_path),
        "rollback_manifest": str((rollback_dir / "rollback_manifest.json").resolve()),
        "production_embeddings_sha256": _sha256_file(production_embeddings_path),
        "production_summary_sha256": _sha256_file(production_summary_path),
    }
    _write_json(version_dir / "promotion_status.json", promotion)
    return promotion


def rollback_embedding_version(*, version_dir: Path) -> dict[str, Any]:
    version_dir = Path(version_dir).resolve()
    manifest = verify_embedding_version(version_dir)
    promotion_path = version_dir / "promotion_status.json"
    rollback_dir = version_dir / "rollback" / "production_before_promotion"
    rollback_manifest_path = rollback_dir / "rollback_manifest.json"
    if not promotion_path.is_file() or not rollback_manifest_path.is_file():
        raise EmbeddingLifecycleError("Rollback refused: no completed promotion backup exists")
    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    rollback_manifest = json.loads(rollback_manifest_path.read_text(encoding="utf-8"))
    if promotion.get("status") != "promoted" or promotion.get("version_id") != manifest["version_id"]:
        raise EmbeddingLifecycleError("Rollback refused: promotion state does not match this version")
    backup_embeddings = rollback_dir / "student_embeddings.pkl"
    backup_summary = rollback_dir / "embedding_summary.csv"
    if (
        _sha256_file(backup_embeddings) != rollback_manifest.get("backup_embeddings_sha256")
        or _sha256_file(backup_summary) != rollback_manifest.get("backup_summary_sha256")
    ):
        raise EmbeddingLifecycleError("Rollback backup integrity check failed")
    production_embeddings = Path(rollback_manifest["production_embeddings_path"])
    production_summary = Path(rollback_manifest["production_summary_path"])
    _atomic_replace_from(backup_embeddings, production_embeddings)
    _atomic_replace_from(backup_summary, production_summary)
    status = {
        "schema_version": LIFECYCLE_SCHEMA_VERSION,
        "version_id": manifest["version_id"],
        "status": "rolled_back",
        "rolled_back_at": datetime.now().isoformat(timespec="seconds"),
        "restored_embeddings_sha256": _sha256_file(production_embeddings),
        "restored_summary_sha256": _sha256_file(production_summary),
    }
    _write_json(version_dir / "rollback_status.json", status)
    return status
