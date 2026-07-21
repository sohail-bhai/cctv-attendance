from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import cv2
import numpy as np
import pandas as pd

from .config import IMAGE_EXTENSIONS, SFACE_MODEL, YUNET_MODEL
from .diagnostics import json_safe
from .embedding_db import StudentEmbeddingDB
from .embedding_forensics import (
    OFFICIAL_ATTENDANCE_CHECKPOINTS,
    OFFICIAL_MARGIN_THRESHOLD,
    OFFICIAL_MATCH_THRESHOLD,
    _canonical,
    load_embedding_payload,
)
from .enrollment_preprocessing import (
    EnrollmentBuildConfig,
    EnrollmentExtraction,
    extract_enrollment_embedding,
    face_quality_score,
)
from .face_engine import FaceEngine
from .forensic_review import (
    APPROVAL_COLUMNS,
    ForensicReviewError,
    load_forensic_review_package,
)
from .shadow_validation import ShadowValidationError, verify_output_manifest
from .tracklet_review import _sha256_file, load_student_mapping
from .utils import cosine_similarity, normalize_embedding


FAMILY_SCHEMA_VERSION = 1
POLICY_VERSION = "phase-1.2i-b-v1"
OFFICIAL_AGGREGATE = "top3"
EXPECTED_PRODUCTION_EMBEDDINGS_SHA256 = (
    "c32ed31df10b7b9b43b8f19977a2adf0a82fb8e7a71f3e8bceb42c0565fedd49"
)
EXPECTED_PRODUCTION_SUMMARY_SHA256 = (
    "0df35c3bb9207e191aa49dad5536b8378e812d56463491bd7203885dca5ae9ee"
)
SOURCE_IDENTITY = "2401100CSE0110"
TARGET_IDENTITY = "24011CSEAI0110"
MISSING_CVO_IDENTITY = "2401100CSE0268"
TUE_P1_SESSION = "2026-06-30__B51__P1__CVO"
TUE_P2_SESSION = "2026-06-30__B51__P2__CVO"
MON_P3_TOKEN = "MON_P3"
REJECTED_CANDIDATE_ID = "cal-5cd35b60dd83"
IDENTITY_CORRECTION_NOTE = (
    "This image belongs to 24011CSEAI0110, not 2401100CSE0110."
)
CORRECTED_ENROLLMENT_ITEM_IDS = (
    "CEA-21cc8966217d8c51b8",
    "CEA-6365fd0aa90b36b357",
)

ENROLLMENT_KEEP = "keep"
ENROLLMENT_DUPLICATE = "duplicate_of_another_image"
ENROLLMENT_MULTIPLE = "multiple_people"
ENROLLMENT_WRONG = "wrong_person_or_mislabeled"
CCTV_APPROVE = "approve_for_candidate_embedding_version"
CCTV_REJECT = "reject_wrong_or_unclear_identity"


class EmbeddingFamilyError(ValueError):
    pass


@dataclass(frozen=True)
class CompactApprovalContract:
    enrollment_package_id: str = "compact-enrollment-audit-df0b986b716a2cd5"
    cctv_package_id: str = "compact-verified-cctv-8785d6b28789ae6f"
    enrollment_kind: str = "compact_enrollment_audit"
    cctv_kind: str = "compact_verified_cctv"
    package_revision: str = "compact-priority-review-v1"
    enrollment_rows: int = 50
    cctv_rows: int = 36
    enrollment_action_counts: tuple[tuple[str, int], ...] = (
        (ENROLLMENT_KEEP, 22),
        (ENROLLMENT_DUPLICATE, 21),
        (ENROLLMENT_MULTIPLE, 5),
        (ENROLLMENT_WRONG, 2),
    )
    cctv_action_counts: tuple[tuple[str, int], ...] = (
        (CCTV_APPROVE, 35),
        (CCTV_REJECT, 1),
    )
    corrected_enrollment_item_ids: tuple[str, ...] = CORRECTED_ENROLLMENT_ITEM_IDS
    correction_note: str = IDENTITY_CORRECTION_NOTE
    source_roll: str = SOURCE_IDENTITY
    target_roll: str = TARGET_IDENTITY
    enrollment_broad_package_id: str = "enrollment-audit-5430747113d0daeb"
    cctv_broad_package_id: str = "verified-cctv-77507b2f5d348aad"


@dataclass(frozen=True)
class CompactApprovalBundle:
    enrollment: pd.DataFrame
    cctv: pd.DataFrame
    summary: dict[str, Any]
    input_manifest: dict[str, Any]


@dataclass(frozen=True)
class IdentityCorrectionResult:
    manifest: dict[str, Any]
    items: pd.DataFrame
    affected_hashes: frozenset[str]
    affected_enrollment_item_ids: frozenset[str]
    affected_cctv_item_ids: frozenset[str]


@dataclass(frozen=True)
class SourcePolicyResult:
    sources: pd.DataFrame
    groups: pd.DataFrame
    decisions: pd.DataFrame
    summary: dict[str, Any]
    selected_source_ids: tuple[str, ...]


@dataclass(frozen=True)
class FamilyBuildResult:
    family_dir: Path
    family_id: str
    family_manifest: dict[str, Any]
    evaluation_decision: dict[str, Any]
    idempotent_reuse: bool


@dataclass(frozen=True)
class VariantDefinition:
    key: str
    directory: str
    role: str
    included_cctv_sessions: tuple[str, ...]
    evaluation_sessions: tuple[str, ...]
    contaminated_sessions: tuple[str, ...]
    descriptive_only: bool


VARIANT_DEFINITIONS = (
    VariantDefinition(
        "A",
        "cleaned_enrollment_only",
        "cleaned_enrollment_only",
        (),
        (TUE_P1_SESSION, TUE_P2_SESSION),
        (),
        False,
    ),
    VariantDefinition(
        "B",
        "cross_session_for_tue_p1",
        "tue_p1_cross_session_evaluation",
        (TUE_P2_SESSION,),
        (TUE_P1_SESSION,),
        (TUE_P2_SESSION,),
        False,
    ),
    VariantDefinition(
        "C",
        "cross_session_for_tue_p2",
        "tue_p2_cross_session_evaluation",
        (TUE_P1_SESSION,),
        (TUE_P2_SESSION,),
        (TUE_P1_SESSION,),
        False,
    ),
    VariantDefinition(
        "D",
        "full_candidate",
        "full_candidate_for_future_mon_p3",
        (TUE_P1_SESSION, TUE_P2_SESSION),
        (),
        (TUE_P1_SESSION, TUE_P2_SESSION),
        True,
    ),
)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _text(value: Any) -> str:
    return str(value or "").strip()


def _yes(value: Any) -> bool:
    return _text(value).lower() in {"1", "true", "yes", "y"}


def _stable_digest(payload: Any) -> str:
    encoded = json.dumps(
        json_safe(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(json_safe(payload), indent=2, ensure_ascii=True), encoding="utf-8"
    )


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _load_private(value: Any, item_id: str) -> dict[str, Any]:
    try:
        payload = json.loads(_text(value) or "{}")
    except json.JSONDecodeError as exc:
        raise EmbeddingFamilyError(
            f"Invalid private provenance JSON for {item_id}"
        ) from exc
    if not isinstance(payload, dict):
        raise EmbeddingFamilyError(
            f"Private provenance must be an object for {item_id}"
        )
    return payload


def _action_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        _text(key): int(value)
        for key, value in frame["Review_Action"].value_counts().sort_index().items()
    }


def _verify_parent_output_manifest(package_dir: Path, *, required: bool) -> dict[str, Any]:
    parent = Path(package_dir).resolve().parent
    manifest_path = parent / "output_manifest.json"
    if not manifest_path.is_file():
        if required:
            raise EmbeddingFamilyError(
                f"Compact parent output manifest not found: {manifest_path}"
            )
        return {"required": False, "present": False}
    try:
        verified = verify_output_manifest(parent, manifest_path)
    except ShadowValidationError as exc:
        raise EmbeddingFamilyError(str(exc)) from exc
    relative = (Path(package_dir).resolve() / "package_manifest.json").relative_to(parent).as_posix()
    recorded = _text(
        dict(dict(verified.get("manifest") or {}).get("files_sha256") or {}).get(relative)
    )
    actual = _sha256_file(Path(package_dir) / "package_manifest.json")
    if recorded != actual:
        raise EmbeddingFamilyError(
            f"Compact parent manifest does not own the package manifest: {relative}"
        )
    return {
        "required": required,
        "present": True,
        "path": str(manifest_path),
        "sha256": _sha256_file(manifest_path),
        "package_manifest_entry": relative,
        "package_manifest_sha256": actual,
    }


def _review_rank_map(public: dict[str, Any]) -> dict[str, str]:
    rank_map: dict[str, str] = {}
    for item in list(public.get("review_items") or []):
        item_id = _text(item.get("item_id"))
        values = dict(item.get("public") or {})
        rank = _text(values.get("Compact_Rank") or values.get("compact_rank"))
        if not rank:
            continue
        if rank in rank_map:
            raise EmbeddingFamilyError(f"Compact reviewer rank {rank} is not unique")
        rank_map[rank] = item_id
    return rank_map


def _normalize_duplicate_references(
    approvals: pd.DataFrame,
    public: dict[str, Any],
) -> pd.DataFrame:
    normalized = approvals.copy()
    normalized["Duplicate_Of_Item_ID_Raw"] = normalized["Duplicate_Of_Item_ID"].map(_text)
    rank_map = _review_rank_map(public)
    expected_ids = {_text(item.get("item_id")) for item in public.get("review_items", [])}
    resolved_values: list[str] = []
    for row in normalized.to_dict("records"):
        item_id = _text(row.get("Item_ID"))
        action = _text(row.get("Review_Action"))
        raw_target = _text(row.get("Duplicate_Of_Item_ID_Raw"))
        if action != ENROLLMENT_DUPLICATE:
            if raw_target:
                raise EmbeddingFamilyError(
                    f"{item_id} supplies Duplicate_Of_Item_ID without the duplicate action"
                )
            resolved_values.append("")
            continue
        if not raw_target:
            raise EmbeddingFamilyError(
                f"{item_id} requires Duplicate_Of_Item_ID for the duplicate action"
            )
        target = raw_target if raw_target in expected_ids else rank_map.get(raw_target, "")
        if not target or target == item_id:
            raise EmbeddingFamilyError(
                f"{item_id} must reference a different valid duplicate item ID"
            )
        resolved_values.append(target)
    normalized["Duplicate_Of_Item_ID"] = resolved_values

    actions = dict(
        zip(normalized["Item_ID"].map(_text), normalized["Review_Action"].map(_text))
    )
    duplicate_edges: dict[str, str] = {}
    for row in normalized[normalized["Review_Action"].eq(ENROLLMENT_DUPLICATE)].to_dict(
        "records"
    ):
        item_id = _text(row["Item_ID"])
        target = _text(row["Duplicate_Of_Item_ID"])
        if actions.get(target) != ENROLLMENT_KEEP:
            raise EmbeddingFamilyError(
                f"{item_id} duplicate target {target} must be marked keep"
            )
        duplicate_edges[item_id] = target
    for start in duplicate_edges:
        seen: set[str] = set()
        current = start
        while current in duplicate_edges:
            if current in seen:
                raise EmbeddingFamilyError(f"Duplicate reference cycle detected at {start}")
            seen.add(current)
            current = duplicate_edges[current]
        if len(seen) > 1:
            raise EmbeddingFamilyError(f"Duplicate reference chain detected at {start}")
    return normalized


def _resolve_review_source(
    *,
    source_path: Path,
    source_sha256: str,
    provenance: dict[str, Any],
    dataset_root: Path | None,
    source_roll: str,
    target_roll: str,
) -> tuple[Path, str]:
    source_path = Path(source_path)
    if source_path.is_file() and _sha256_file(source_path) == source_sha256:
        return source_path.resolve(), "reviewed_path_verified"
    expected_roll = _canonical(provenance.get("expected_roll"))
    if not dataset_root or expected_roll != source_roll:
        raise EmbeddingFamilyError(f"Approval source integrity check failed: {source_path}")
    target_dir = Path(dataset_root).resolve() / target_roll
    if not target_dir.is_dir():
        raise EmbeddingFamilyError(
            f"Corrected target dataset folder not found while resolving {source_path.name}: {target_dir}"
        )
    candidates = [
        candidate.resolve()
        for candidate in sorted(target_dir.rglob(source_path.name))
        if candidate.is_file() and _sha256_file(candidate) == source_sha256
    ]
    if len(candidates) != 1:
        raise EmbeddingFamilyError(
            f"Could not resolve reviewed source by exact corrected-folder hash: "
            f"{source_path} (matches={len(candidates)})"
        )
    return candidates[0], "resolved_human_confirmed_folder_rename_by_hash"


def _validate_one_compact_package(
    *,
    package_dir: Path,
    approvals_path: Path,
    expected_package_id: str,
    expected_kind: str,
    expected_revision: str,
    expected_rows: int,
    expected_counts: dict[str, int],
    dataset_root: Path | None,
    source_roll: str,
    target_roll: str,
    require_parent_manifest: bool,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    package_dir = Path(package_dir).resolve()
    approvals_path = Path(approvals_path).resolve()
    try:
        public, metadata, mapping = load_forensic_review_package(package_dir)
    except ForensicReviewError as exc:
        raise EmbeddingFamilyError(str(exc)) from exc
    if _text(public.get("package_id")) != expected_package_id:
        raise EmbeddingFamilyError(
            f"Expected package {expected_package_id}, found {public.get('package_id')}"
        )
    if _text(public.get("package_kind")) != expected_kind:
        raise EmbeddingFamilyError(
            f"Expected package kind {expected_kind}, found {public.get('package_kind')}"
        )
    if _text(metadata.get("package_revision")) != expected_revision:
        raise EmbeddingFamilyError(
            f"Compact package revision mismatch for {expected_package_id}"
        )
    if not approvals_path.is_file():
        raise EmbeddingFamilyError(f"Approval CSV not found: {approvals_path}")
    approvals = pd.read_csv(approvals_path, dtype=str, keep_default_na=False)
    missing_columns = sorted(set(APPROVAL_COLUMNS).difference(approvals.columns))
    if missing_columns:
        raise EmbeddingFamilyError(
            "Approval CSV missing columns: " + ", ".join(missing_columns)
        )
    approvals = approvals[APPROVAL_COLUMNS].copy()
    if len(approvals) != expected_rows:
        raise EmbeddingFamilyError(
            f"Approval row count mismatch for {expected_package_id}: "
            f"expected {expected_rows}, found {len(approvals)}"
        )
    if approvals["Item_ID"].duplicated().any():
        duplicates = sorted(
            approvals.loc[approvals["Item_ID"].duplicated(False), "Item_ID"].unique()
        )
        raise EmbeddingFamilyError(
            "Approval CSV contains duplicate item IDs: " + ", ".join(duplicates[:10])
        )
    if set(approvals["Package_ID"].map(_text)) != {expected_package_id}:
        raise EmbeddingFamilyError("Approval CSV package ID does not match the review package")
    expected_ids = {_text(item.get("item_id")) for item in public.get("review_items", [])}
    actual_ids = set(approvals["Item_ID"].map(_text))
    if actual_ids != expected_ids:
        raise EmbeddingFamilyError(
            f"Approval item set mismatch; missing={sorted(expected_ids - actual_ids)[:10]}, "
            f"extra={sorted(actual_ids - expected_ids)[:10]}"
        )
    allowed_actions = set(dict(public.get("actions") or {}))
    actual_actions = set(approvals["Review_Action"].map(_text))
    invalid_actions = sorted(actual_actions - allowed_actions)
    if invalid_actions or approvals["Review_Action"].map(_text).eq("").any():
        raise EmbeddingFamilyError(
            "Approval CSV contains missing or invalid actions: "
            + ", ".join(invalid_actions or ["<blank>"])
        )
    counts = _action_counts(approvals)
    if counts != expected_counts:
        raise EmbeddingFamilyError(
            f"Approval action counts mismatch for {expected_package_id}: "
            f"expected {expected_counts}, found {counts}"
        )
    if ENROLLMENT_DUPLICATE in allowed_actions:
        approvals = _normalize_duplicate_references(approvals, public)
    elif approvals["Duplicate_Of_Item_ID"].map(_text).ne("").any():
        raise EmbeddingFamilyError("CCTV approvals cannot contain duplicate references")
    else:
        approvals["Duplicate_Of_Item_ID_Raw"] = ""

    joined = approvals.merge(
        mapping,
        on=["Package_ID", "Item_ID"],
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(approvals):
        raise EmbeddingFamilyError("Approvals do not match private source provenance")
    resolved_paths: list[str] = []
    integrity_statuses: list[str] = []
    provenances: list[dict[str, Any]] = []
    for row in joined.to_dict("records"):
        item_id = _text(row.get("Item_ID"))
        provenance = _load_private(row.get("Private_Provenance_JSON"), item_id)
        expected_hash_key = (
            "source_file_sha256"
            if expected_kind == "compact_enrollment_audit"
            else "source_crop_sha256"
        )
        if _text(provenance.get(expected_hash_key)) != _text(row.get("Source_SHA256")):
            raise EmbeddingFamilyError(
                f"Private provenance source hash mismatch for {item_id}"
            )
        resolved, status = _resolve_review_source(
            source_path=Path(_text(row.get("Source_Path"))),
            source_sha256=_text(row.get("Source_SHA256")),
            provenance=provenance,
            dataset_root=dataset_root,
            source_roll=source_roll,
            target_roll=target_roll,
        )
        resolved_paths.append(str(resolved))
        integrity_statuses.append(status)
        provenances.append(provenance)
    joined["Resolved_Source_Path"] = resolved_paths
    joined["Source_Integrity_Status"] = integrity_statuses
    joined["Private_Provenance"] = provenances
    parent_manifest = _verify_parent_output_manifest(
        package_dir, required=require_parent_manifest
    )
    package_manifest_path = package_dir / "package_manifest.json"
    private_mapping_path = package_dir / "private" / "source_mapping.csv"
    package_record = {
        "package_id": expected_package_id,
        "package_kind": expected_kind,
        "package_revision": expected_revision,
        "package_dir": str(package_dir),
        "package_manifest": {
            "path": str(package_manifest_path),
            "sha256": _sha256_file(package_manifest_path),
        },
        "private_source_mapping": {
            "path": str(private_mapping_path),
            "sha256": _sha256_file(private_mapping_path),
        },
        "review_manifest": {
            "path": str(package_dir / "reviewer" / "review_manifest.json"),
            "sha256": _sha256_file(package_dir / "reviewer" / "review_manifest.json"),
        },
        "approval_csv": {
            "path": str(approvals_path),
            "sha256": _sha256_file(approvals_path),
        },
        "parent_output_manifest": parent_manifest,
        "item_count": len(joined),
        "item_ids_sha256": _stable_digest(sorted(actual_ids)),
        "action_counts": counts,
        "source_hashes_verified": True,
    }
    return joined, package_record, metadata


def validate_compact_approval_bundle(
    *,
    enrollment_review_package: Path,
    enrollment_approvals_path: Path,
    cctv_review_package: Path,
    cctv_approvals_path: Path,
    dataset_root: Path | None = None,
    contract: CompactApprovalContract = CompactApprovalContract(),
    require_parent_manifest: bool = True,
) -> CompactApprovalBundle:
    enrollment, enrollment_record, _ = _validate_one_compact_package(
        package_dir=enrollment_review_package,
        approvals_path=enrollment_approvals_path,
        expected_package_id=contract.enrollment_package_id,
        expected_kind=contract.enrollment_kind,
        expected_revision=contract.package_revision,
        expected_rows=contract.enrollment_rows,
        expected_counts=dict(contract.enrollment_action_counts),
        dataset_root=dataset_root,
        source_roll=contract.source_roll,
        target_roll=contract.target_roll,
        require_parent_manifest=require_parent_manifest,
    )
    cctv, cctv_record, _ = _validate_one_compact_package(
        package_dir=cctv_review_package,
        approvals_path=cctv_approvals_path,
        expected_package_id=contract.cctv_package_id,
        expected_kind=contract.cctv_kind,
        expected_revision=contract.package_revision,
        expected_rows=contract.cctv_rows,
        expected_counts=dict(contract.cctv_action_counts),
        dataset_root=dataset_root,
        source_roll=contract.source_roll,
        target_roll=contract.target_roll,
        require_parent_manifest=require_parent_manifest,
    )

    enrollment_provenance = {
        _text(row["Item_ID"]): row["Private_Provenance"]
        for row in enrollment.to_dict("records")
    }
    wrong_rows = enrollment[enrollment["Review_Action"].eq(ENROLLMENT_WRONG)]
    wrong_ids = set(wrong_rows["Item_ID"].map(_text))
    if wrong_ids != set(contract.corrected_enrollment_item_ids):
        raise EmbeddingFamilyError(
            f"AI0110 correction item set mismatch: expected "
            f"{sorted(contract.corrected_enrollment_item_ids)}, found {sorted(wrong_ids)}"
        )
    for row in wrong_rows.to_dict("records"):
        item_id = _text(row["Item_ID"])
        if _text(row.get("Reviewer_Notes")) != contract.correction_note:
            raise EmbeddingFamilyError(
                f"AI0110 correction note mismatch for {item_id}"
            )
        if _canonical(enrollment_provenance[item_id].get("expected_roll")) != contract.source_roll:
            raise EmbeddingFamilyError(
                f"AI0110 correction source identity mismatch for {item_id}"
            )
    enrollment["Expected_Roll"] = [
        _canonical(provenance.get("expected_roll"))
        for provenance in enrollment["Private_Provenance"]
    ]
    enrollment["Candidate_Roll"] = [
        contract.target_roll
        if _text(item_id) in set(contract.corrected_enrollment_item_ids)
        else _canonical(provenance.get("expected_roll"))
        for item_id, provenance in zip(
            enrollment["Item_ID"], enrollment["Private_Provenance"]
        )
    ]
    enrollment_hash_by_item = dict(
        zip(enrollment["Item_ID"].map(_text), enrollment["Source_SHA256"].map(_text))
    )
    enrollment["Duplicate_Target_Source_SHA256"] = [
        enrollment_hash_by_item.get(_text(target), "")
        for target in enrollment["Duplicate_Of_Item_ID"]
    ]
    for row in enrollment.to_dict("records"):
        provenance = row["Private_Provenance"]
        broad_id = _text(provenance.get("source_broad_package_id"))
        if broad_id and broad_id != contract.enrollment_broad_package_id:
            raise EmbeddingFamilyError(
                f"Enrollment item {row['Item_ID']} is owned by an unexpected broad package"
            )

    cctv["Actual_Roll"] = [
        _canonical(provenance.get("actual_roll"))
        for provenance in cctv["Private_Provenance"]
    ]
    cctv["Candidate_Roll"] = cctv["Actual_Roll"]
    cctv["Source_Session"] = [
        _text(provenance.get("source_session"))
        for provenance in cctv["Private_Provenance"]
    ]
    for row in cctv.to_dict("records"):
        provenance = row["Private_Provenance"]
        broad_id = _text(provenance.get("source_broad_package_id"))
        if broad_id and broad_id != contract.cctv_broad_package_id:
            raise EmbeddingFamilyError(
                f"CCTV item {row['Item_ID']} is owned by an unexpected broad package"
            )
        if row["Review_Action"] == CCTV_APPROVE:
            if (
                _text(provenance.get("identity_source")) != "human_review_actual_roll"
                or not _canonical(provenance.get("actual_roll"))
                or provenance.get("model_prediction_used_as_identity") is not False
            ):
                raise EmbeddingFamilyError(
                    f"Approved CCTV item lacks human Actual_Roll provenance: {row['Item_ID']}"
                )
            if _text(provenance.get("source_session")) not in {
                TUE_P1_SESSION,
                TUE_P2_SESSION,
            }:
                raise EmbeddingFamilyError(
                    f"Approved CCTV item has an unexpected source session: {row['Item_ID']}"
                )

    input_manifest = {
        "schema_version": FAMILY_SCHEMA_VERSION,
        "validation_policy": POLICY_VERSION,
        "validated_at": _now(),
        "packages": {
            "enrollment": enrollment_record,
            "cctv": cctv_record,
        },
        "filenames_trusted_as_identity": False,
        "embedded_package_ids_and_exact_item_sets_validated": True,
        "source_evidence_hashes_validated": True,
    }
    summary = {
        "schema_version": FAMILY_SCHEMA_VERSION,
        "status": "passed",
        "complete": True,
        "enrollment_package_id": contract.enrollment_package_id,
        "cctv_package_id": contract.cctv_package_id,
        "package_revision": contract.package_revision,
        "enrollment_rows": len(enrollment),
        "cctv_rows": len(cctv),
        "enrollment_action_counts": _action_counts(enrollment),
        "cctv_action_counts": _action_counts(cctv),
        "duplicate_reference_mode": "opaque_item_id_or_unique_compact_rank_normalized_to_item_id",
        "duplicate_references_valid": True,
        "identity_correction_rows_valid": True,
        "identity_correction_note": contract.correction_note,
        "identity_correction_source_roll": contract.source_roll,
        "identity_correction_target_roll": contract.target_roll,
        "corrected_enrollment_item_ids": sorted(
            contract.corrected_enrollment_item_ids
        ),
        "approved_cctv_rows": int(cctv["Review_Action"].eq(CCTV_APPROVE).sum()),
        "rejected_cctv_rows": int(cctv["Review_Action"].ne(CCTV_APPROVE).sum()),
        "candidate_build_started": False,
        "production_files_changed": False,
    }
    return CompactApprovalBundle(enrollment, cctv, summary, input_manifest)


def _image_files(root: Path) -> list[Path]:
    if not Path(root).is_dir():
        return []
    return [
        path.resolve()
        for path in sorted(Path(root).rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def reconcile_identity_correction(
    *,
    repo_root: Path,
    dataset_root: Path,
    historical_inventory_path: Path,
    production_embeddings_path: Path,
    student_map_path: Path,
    approval_bundle: CompactApprovalBundle,
    subject_abbr: str = "CVO",
    source_roll: str = SOURCE_IDENTITY,
    target_roll: str = TARGET_IDENTITY,
    expected_historical_images: int | None = 16,
    proven_cctv_item_ids: Iterable[str] = (),
) -> IdentityCorrectionResult:
    repo_root = Path(repo_root).resolve()
    dataset_root = Path(dataset_root).resolve()
    source_dir = dataset_root / source_roll
    target_dir = dataset_root / target_roll
    source_files = _image_files(source_dir)
    target_files = _image_files(target_dir)
    if source_files and target_files:
        raise EmbeddingFamilyError(
            "Identity correction is ambiguous because source and target dataset folders both contain images"
        )
    if source_files:
        raise EmbeddingFamilyError(
            "The source identity folder still contains images; the human folder correction is incomplete"
        )
    if not target_files:
        raise EmbeddingFamilyError(
            f"Corrected target identity folder has no enrollment images: {target_dir}"
        )

    inventory = pd.read_csv(
        historical_inventory_path, dtype=str, keep_default_na=False
    )
    required_columns = {"Owning_Folder", "File_SHA256", "Relative_Path", "Dataset_Source_Role"}
    if not required_columns.issubset(inventory.columns):
        raise EmbeddingFamilyError("Historical dataset inventory schema is incomplete")
    historical_source = inventory[
        inventory["Owning_Folder"].map(_canonical).eq(source_roll)
        & inventory["Dataset_Source_Role"].eq("production_recorded_dataset")
    ].copy()
    historical_target = inventory[
        inventory["Owning_Folder"].map(_canonical).eq(target_roll)
        & inventory["Dataset_Source_Role"].eq("production_recorded_dataset")
    ].copy()
    if expected_historical_images is not None and len(historical_source) != expected_historical_images:
        raise EmbeddingFamilyError(
            f"Historical source-folder inventory count mismatch: expected "
            f"{expected_historical_images}, found {len(historical_source)}"
        )
    if len(historical_target):
        raise EmbeddingFamilyError(
            "Historical inventory contains an independent target folder; ownership collision is ambiguous"
        )

    current_by_hash: dict[str, list[Path]] = defaultdict(list)
    for path in target_files:
        current_by_hash[_sha256_file(path)].append(path)
    historical_hashes = list(historical_source["File_SHA256"].map(_text))
    if len(historical_hashes) != len(set(historical_hashes)):
        raise EmbeddingFamilyError(
            "Historical source-folder inventory contains duplicate image hashes"
        )
    if set(current_by_hash) != set(historical_hashes):
        missing = sorted(set(historical_hashes) - set(current_by_hash))
        extra = sorted(set(current_by_hash) - set(historical_hashes))
        raise EmbeddingFamilyError(
            f"Corrected target folder does not exactly match historical source hashes; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    if any(len(paths) != 1 for paths in current_by_hash.values()):
        raise EmbeddingFamilyError("Corrected target folder contains duplicate historical hashes")

    roster, _, subject_rolls_raw = load_student_mapping(student_map_path, subject_abbr)
    subject_rolls = {_canonical(value) for value in subject_rolls_raw}
    if target_roll not in subject_rolls or source_roll in subject_rolls:
        raise EmbeddingFamilyError(
            "Authoritative roster does not support the narrow source-to-target correction"
        )
    if _canonical(source_roll) == _canonical(target_roll):
        raise EmbeddingFamilyError("Source and target identities canonicalize to the same key")

    reviewed_source_rows = approval_bundle.enrollment[
        approval_bundle.enrollment["Expected_Roll"].eq(source_roll)
    ].copy()
    wrong_rows = reviewed_source_rows[
        reviewed_source_rows["Review_Action"].eq(ENROLLMENT_WRONG)
    ]
    affected_enrollment_ids = frozenset(wrong_rows["Item_ID"].map(_text))
    expected_wrong_ids = set(
        approval_bundle.summary.get("corrected_enrollment_item_ids") or []
    )
    if affected_enrollment_ids != expected_wrong_ids:
        raise EmbeddingFamilyError("Human-confirmed correction enrollment item set changed")

    proven_cctv = {_text(value) for value in proven_cctv_item_ids if _text(value)}
    source_cctv_rows = approval_bundle.cctv[
        approval_bundle.cctv["Actual_Roll"].eq(source_roll)
        & approval_bundle.cctv["Review_Action"].eq(CCTV_APPROVE)
    ]
    source_cctv_ids = set(source_cctv_rows["Item_ID"].map(_text))
    if source_cctv_ids - proven_cctv:
        raise EmbeddingFamilyError(
            "CCTV correction provenance is ambiguous for items: "
            + ", ".join(sorted(source_cctv_ids - proven_cctv))
        )
    affected_cctv_ids = frozenset(source_cctv_ids & proven_cctv)

    parent = load_embedding_payload(production_embeddings_path)
    parent_records = list(parent["records"])
    parent_source_records = [
        (index, record)
        for index, record in enumerate(parent_records)
        if _canonical(record.get("roll_no")) == source_roll
    ]
    parent_target_records = [
        (index, record)
        for index, record in enumerate(parent_records)
        if _canonical(record.get("roll_no")) == target_roll
    ]
    if parent_target_records:
        raise EmbeddingFamilyError(
            "Production already contains independent target embeddings; correction collision requires review"
        )
    if expected_historical_images is not None and len(parent_source_records) != expected_historical_images:
        raise EmbeddingFamilyError(
            f"Production source-record count mismatch: expected {expected_historical_images}, "
            f"found {len(parent_source_records)}"
        )

    review_by_hash = {
        _text(row["Source_SHA256"]): {
            "item_id": _text(row["Item_ID"]),
            "action": _text(row["Review_Action"]),
        }
        for row in reviewed_source_rows.to_dict("records")
    }
    parent_index_by_basename = {
        Path(_text(record.get("image_path"))).name.lower(): index
        for index, record in parent_source_records
    }
    item_rows: list[dict[str, Any]] = []
    for historical in historical_source.sort_values("Relative_Path", kind="stable").to_dict(
        "records"
    ):
        source_hash = _text(historical["File_SHA256"])
        current_path = current_by_hash[source_hash][0]
        review = review_by_hash.get(source_hash, {})
        review_action = _text(review.get("action"))
        candidate_status = (
            "excluded_reviewer_confirmed_multiple_people"
            if review_action == ENROLLMENT_MULTIPLE
            else "candidate_owner_reconciled_to_target"
        )
        item_rows.append(
            {
                "Correction_ID": "",
                "Source_Roll": source_roll,
                "Target_Roll": target_roll,
                "Historical_Path": _text(historical["Relative_Path"]),
                "Historical_SHA256": source_hash,
                "Current_Path": str(current_path),
                "Current_SHA256": _sha256_file(current_path),
                "Enrollment_Item_ID": _text(review.get("item_id")),
                "Enrollment_Review_Action": review_action,
                "Production_Record_Index": parent_index_by_basename.get(
                    current_path.name.lower(), ""
                ),
                "Candidate_Status": candidate_status,
            }
        )
    correction_basis = {
        "source_roll": source_roll,
        "target_roll": target_roll,
        "affected_hashes": sorted(current_by_hash),
        "affected_enrollment_item_ids": sorted(affected_enrollment_ids),
        "affected_cctv_item_ids": sorted(affected_cctv_ids),
        "historical_inventory_sha256": _sha256_file(historical_inventory_path),
        "current_target_folder": str(target_dir),
    }
    correction_id = f"idcorr-{_stable_digest(correction_basis)[:16]}"
    for row in item_rows:
        row["Correction_ID"] = correction_id
    manifest = {
        "schema_version": FAMILY_SCHEMA_VERSION,
        "correction_id": correction_id,
        "created_at": _now(),
        "status": "validated_narrow_candidate_only_correction",
        "source_roll": source_roll,
        "target_roll": target_roll,
        "human_confirmation_basis": {
            "folder_manually_renamed_by_user_before_this_phase": True,
            "explicit_wrong_person_note": _text(
                approval_bundle.summary.get("identity_correction_note")
            ),
            "explicit_enrollment_item_ids": sorted(affected_enrollment_ids),
        },
        "affected_source_folders": {
            "historical": str(dataset_root / source_roll),
            "current": str(target_dir),
        },
        "affected_image_count": len(item_rows),
        "affected_image_hashes": sorted(current_by_hash),
        "affected_enrollment_review_item_ids": sorted(affected_enrollment_ids),
        "review_rows_with_historical_source_folder": sorted(
            reviewed_source_rows["Item_ID"].map(_text)
        ),
        "affected_cctv_review_item_ids": sorted(affected_cctv_ids),
        "old_provenance": {
            "historical_inventory": str(Path(historical_inventory_path).resolve()),
            "historical_source_rows": len(historical_source),
            "historical_target_rows": len(historical_target),
            "production_source_embedding_records": len(parent_source_records),
            "production_target_embedding_records": len(parent_target_records),
        },
        "new_candidate_provenance": {
            "candidate_owner": target_roll,
            "physical_source_folder": str(target_dir),
            "source_hash_set_exact_historical_match": True,
            "production_database_rewritten": False,
        },
        "registry_checks": {
            "subject_abbr": subject_abbr,
            "authoritative_roster_count": len(subject_rolls),
            "source_in_authoritative_roster": source_roll in subject_rolls,
            "target_in_authoritative_roster": target_roll in subject_rolls,
            "registry_rows_loaded": len(roster),
            "student_map_sha256": _sha256_file(student_map_path),
        },
        "collision_checks": {
            "source_folder_images": len(source_files),
            "target_folder_images": len(target_files),
            "historical_independent_target_images": len(historical_target),
            "production_independent_target_records": len(parent_target_records),
            "duplicate_current_hashes": 0,
            "passed": True,
        },
        "scope_limitations": [
            "Only the hash-proven dataset folder contents are corrected in candidate artifacts.",
            "Unrelated records for the source roll are not reassigned.",
            "No model prediction was used as identity evidence.",
            "The production embedding database and global registry remain unchanged.",
        ],
        "global_alias_created": False,
        "production_embeddings_changed": False,
    }
    return IdentityCorrectionResult(
        manifest=manifest,
        items=pd.DataFrame(item_rows),
        affected_hashes=frozenset(current_by_hash),
        affected_enrollment_item_ids=affected_enrollment_ids,
        affected_cctv_item_ids=affected_cctv_ids,
    )


def reconcile_shruthi_source(
    *,
    dataset_roots: Sequence[Path],
    embedding_outliers_path: Path,
    historical_inventory_path: Path,
) -> dict[str, Any]:
    outliers = pd.read_csv(embedding_outliers_path, dtype=str, keep_default_na=False)
    matches = outliers[
        outliers["Canonical_Roll"].map(_canonical).eq("2401100CSE0140")
        & outliers["Image_Path"].str.contains("18.58.21", regex=False)
    ]
    if len(matches) != 1:
        raise EmbeddingFamilyError(
            f"Expected one Shruthi forensic outlier, found {len(matches)}"
        )
    outlier = matches.iloc[0]
    basename = Path(_text(outlier["Image_Path"])).name
    inventory = pd.read_csv(
        historical_inventory_path, dtype=str, keep_default_na=False
    )
    inventory_matches = inventory[
        inventory["Canonical_Roll"].map(_canonical).eq("2401100CSE0140")
        & inventory["Source_Path"].map(lambda value: Path(_text(value)).name.lower()).eq(
            basename.lower()
        )
    ]
    recorded_hashes = set(inventory_matches["File_SHA256"].map(_text))
    candidates: list[Path] = []
    for root in dataset_roots:
        if not Path(root).is_dir():
            continue
        candidates.extend(
            path.resolve()
            for path in sorted(Path(root).rglob("*"))
            if path.is_file() and path.name.lower() == basename.lower()
        )
    exact = [
        path
        for path in candidates
        if not recorded_hashes or _sha256_file(path) in recorded_hashes
    ]
    if len(exact) == 1:
        return {
            "status": "found_unique_renamed_folder_match",
            "raw_embedding_key": _text(outlier.get("Raw_Embedding_Key")),
            "historical_path": _text(outlier.get("Image_Path")),
            "resolved_path": str(exact[0]),
            "sha256": _sha256_file(exact[0]),
            "historical_inventory_matches": len(inventory_matches),
            "candidate_action": "preserve_baseline_inclusion_unless_build_quality_rejects",
            "human_exclusion_fabricated": False,
        }
    if not exact:
        return {
            "status": "source_missing",
            "raw_embedding_key": _text(outlier.get("Raw_Embedding_Key")),
            "historical_path": _text(outlier.get("Image_Path")),
            "resolved_path": "",
            "sha256": "",
            "historical_inventory_matches": len(inventory_matches),
            "candidate_action": "exclude_unavailable_source",
            "human_exclusion_fabricated": False,
        }
    raise EmbeddingFamilyError(
        f"Shruthi forensic source resolves ambiguously to {len(exact)} files"
    )


_AUGMENTATION_SUFFIX = re.compile(
    r"(?i)(?:[_\-\s]+)(original|orig|rot_left|rot_right|rotate_left|rotate_right|"
    r"flip|flipped|bright|brightened|dark|darkened|crop|cropped|zoom|zoomed|aug|augmented)$"
)


def _augmentation_lineage(path: Path) -> tuple[str, str]:
    stem = path.stem.strip()
    marker = ""
    match = _AUGMENTATION_SUFFIX.search(stem)
    if match:
        marker = match.group(1).lower()
        stem = stem[: match.start()].rstrip("_- ")
    return stem.lower(), marker


def discover_enrollment_sources(
    *,
    dataset_root: Path,
    approval_bundle: CompactApprovalBundle,
    correction: IdentityCorrectionResult,
) -> pd.DataFrame:
    dataset_root = Path(dataset_root).resolve()
    if not dataset_root.is_dir():
        raise EmbeddingFamilyError(f"Configured enrollment dataset not found: {dataset_root}")
    folder_owners: dict[str, str] = {}
    for folder in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        canonical = _canonical(folder.name)
        if not canonical:
            raise EmbeddingFamilyError(f"Enrollment folder has no canonical identity: {folder}")
        if canonical in folder_owners and folder_owners[canonical] != folder.name:
            raise EmbeddingFamilyError(
                f"Canonical enrollment folder collision: {folder_owners[canonical]} and {folder.name}"
            )
        folder_owners[canonical] = folder.name

    decisions_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for decision in approval_bundle.enrollment.to_dict("records"):
        expected = _canonical(decision.get("Expected_Roll"))
        candidate = _canonical(decision.get("Candidate_Roll"))
        source_hash = _text(decision.get("Source_SHA256"))
        owner = candidate if candidate else expected
        key = (source_hash, owner)
        if key in decisions_by_key:
            other = decisions_by_key[key]
            if (
                _text(other.get("Review_Action")) != _text(decision.get("Review_Action"))
                or _text(other.get("Duplicate_Of_Item_ID"))
                != _text(decision.get("Duplicate_Of_Item_ID"))
            ):
                raise EmbeddingFamilyError(
                    f"Conflicting compact decisions for source hash {source_hash}"
                )
        decisions_by_key[key] = decision

    rows: list[dict[str, Any]] = []
    hashes_by_owner: dict[str, set[str]] = defaultdict(set)
    owners_by_hash: dict[str, set[str]] = defaultdict(set)
    for path in _image_files(dataset_root):
        relative = path.relative_to(dataset_root)
        owner = _canonical(relative.parts[0])
        source_hash = _sha256_file(path)
        owners_by_hash[source_hash].add(owner)
        hashes_by_owner[owner].add(source_hash)
        decision = decisions_by_key.get((source_hash, owner))
        action = _text(decision.get("Review_Action")) if decision else ""
        item_id = _text(decision.get("Item_ID")) if decision else ""
        policy_eligible = True
        reason = "deferred_unreviewed_preserved_by_baseline_policy"
        if action == ENROLLMENT_DUPLICATE:
            policy_eligible = False
            reason = "reviewer_confirmed_duplicate"
        elif action == ENROLLMENT_MULTIPLE:
            policy_eligible = False
            reason = "reviewer_confirmed_multiple_people"
        elif action == ENROLLMENT_WRONG:
            if item_id in correction.affected_enrollment_item_ids and owner == TARGET_IDENTITY:
                reason = "human_confirmed_identity_correction_candidate_only"
            else:
                policy_eligible = False
                reason = "wrong_person_without_validated_target"
        elif action == ENROLLMENT_KEEP:
            reason = "reviewer_confirmed_keep"
        elif action:
            policy_eligible = False
            reason = f"reviewer_excluded_{action}"
        lineage, marker = _augmentation_lineage(path)
        source_id = f"ENS-{_stable_digest([relative.as_posix(), source_hash, owner])[:18]}"
        rows.append(
            {
                "Source_ID": source_id,
                "Source_Role": "production_recorded_dataset",
                "Source_Path": str(path),
                "Relative_Path": relative.as_posix(),
                "Source_SHA256": source_hash,
                "File_Size_Bytes": int(path.stat().st_size),
                "Original_Folder": relative.parts[0],
                "Canonical_Owner": owner,
                "Candidate_Owner": owner,
                "Review_Item_ID": item_id,
                "Review_Action": action or "not_selected_for_compact_review",
                "Duplicate_Target_Item_ID": _text(
                    decision.get("Duplicate_Of_Item_ID") if decision else ""
                ),
                "Duplicate_Target_Source_SHA256": _text(
                    decision.get("Duplicate_Target_Source_SHA256") if decision else ""
                ),
                "Identity_Correction_ID": (
                    _text(correction.manifest.get("correction_id"))
                    if source_hash in correction.affected_hashes
                    else ""
                ),
                "Policy_Eligible": policy_eligible,
                "Policy_Reason": reason,
                "Lineage_Stem": lineage,
                "Augmentation_Marker": marker,
                "Is_Augmentation": bool(marker and marker not in {"original", "orig"}),
                "Is_Explicit_Original": marker in {"original", "orig"},
            }
        )
    cross_owner = {
        source_hash: sorted(owners)
        for source_hash, owners in owners_by_hash.items()
        if len(owners) > 1
    }
    if cross_owner:
        source_hash, owners = next(iter(sorted(cross_owner.items())))
        raise EmbeddingFamilyError(
            f"Exact source hash is owned by multiple canonical identities: {source_hash} -> {owners}"
        )
    if not rows:
        raise EmbeddingFamilyError("Configured enrollment dataset contains no supported images")
    return pd.DataFrame(rows).sort_values("Relative_Path", kind="stable").reset_index(drop=True)


def apply_exact_duplicate_policy(sources: pd.DataFrame) -> pd.DataFrame:
    frame = sources.copy()
    frame["Exact_Duplicate_Group_ID"] = ""
    eligible = frame[frame["Policy_Eligible"].map(bool)]
    for source_hash, group in eligible.groupby("Source_SHA256", sort=True):
        if len(group) < 2:
            continue
        owners = set(group["Candidate_Owner"].map(_canonical))
        if len(owners) != 1:
            raise EmbeddingFamilyError(
                f"Exact duplicate group crosses identities: {source_hash}"
            )
        group_id = f"EXACT-{source_hash[:16]}"
        ordered = group.assign(
            _review_priority=group["Review_Action"].map(
                lambda value: 0 if value == ENROLLMENT_KEEP else 1
            ),
            _path_priority=group["Relative_Path"].map(str.lower),
        ).sort_values(["_review_priority", "_path_priority"], kind="stable")
        keep_index = ordered.index[0]
        frame.loc[group.index, "Exact_Duplicate_Group_ID"] = group_id
        for index in group.index:
            if index == keep_index:
                continue
            frame.at[index, "Policy_Eligible"] = False
            frame.at[index, "Policy_Reason"] = "exact_duplicate_included_once"
    return frame


def select_balanced_sources(
    sources: pd.DataFrame,
    extractions: dict[str, EnrollmentExtraction],
) -> SourcePolicyResult:
    frame = apply_exact_duplicate_policy(sources)
    frame["Decode_Success"] = False
    frame["Embedding_Valid"] = False
    frame["Extraction_Reason"] = "not_processed_policy_excluded"
    frame["Detection_Count"] = 0
    frame["Detector_Score"] = 0.0
    frame["Rank_Score"] = -999.0
    for index, row in frame.iterrows():
        source_id = _text(row["Source_ID"])
        extraction = extractions.get(source_id)
        if extraction is None:
            continue
        frame.at[index, "Decode_Success"] = extraction.reason != "read_failed"
        frame.at[index, "Embedding_Valid"] = extraction.accepted
        frame.at[index, "Extraction_Reason"] = extraction.reason
        frame.at[index, "Detection_Count"] = extraction.detection_count
        frame.at[index, "Detector_Score"] = extraction.detector_score
        frame.at[index, "Rank_Score"] = extraction.rank_score
        if not extraction.accepted and bool(frame.at[index, "Policy_Eligible"]):
            frame.at[index, "Policy_Eligible"] = False
            frame.at[index, "Policy_Reason"] = f"production_build_rejected_{extraction.reason}"

    frame["Source_Group_ID"] = ""
    frame["Selected_For_Candidate"] = False
    frame["Selection_Reason"] = frame["Policy_Reason"]
    groups_rows: list[dict[str, Any]] = []
    candidates = frame[frame["Policy_Eligible"].map(bool)].copy()
    group_keys: dict[int, tuple[str, str, str]] = {}
    for owner, owner_group in candidates.groupby("Candidate_Owner", sort=True):
        lineage_counts = owner_group.groupby("Lineage_Stem")["Augmentation_Marker"].apply(
            lambda values: any(_text(value) for value in values)
        )
        for index, row in owner_group.iterrows():
            lineage = _text(row["Lineage_Stem"])
            if bool(lineage_counts.get(lineage, False)):
                key = (owner, Path(_text(row["Relative_Path"])).parent.as_posix(), lineage)
            else:
                key = (owner, "independent", _text(row["Source_ID"]))
            group_keys[index] = key
    grouped_indices: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, key in group_keys.items():
        grouped_indices[key].append(index)

    selected_ids: list[str] = []
    for key in sorted(grouped_indices):
        indices = grouped_indices[key]
        group = frame.loc[indices].copy()
        group_id = f"AUG-{_stable_digest(key)[:16]}"
        frame.loc[indices, "Source_Group_ID"] = group_id
        ordered = group.assign(
            _original_priority=group.apply(
                lambda row: 0
                if bool(row["Is_Explicit_Original"]) or not bool(row["Is_Augmentation"])
                else 1,
                axis=1,
            ),
            _review_priority=group["Review_Action"].map(
                lambda value: 0 if value == ENROLLMENT_KEEP else 1
            ),
            _quality_priority=-group["Rank_Score"].astype(float),
            _path_priority=group["Relative_Path"].map(str.lower),
        ).sort_values(
            [
                "_original_priority",
                "_review_priority",
                "_quality_priority",
                "_path_priority",
            ],
            kind="stable",
        )
        primary_index = ordered.index[0]
        chosen = [primary_index]
        primary_id = _text(frame.at[primary_index, "Source_ID"])
        primary_vector = extractions[primary_id].embedding
        if len(ordered) > 1:
            for candidate_index in ordered.index[1:]:
                candidate_id = _text(frame.at[candidate_index, "Source_ID"])
                candidate_vector = extractions[candidate_id].embedding
                similarity = cosine_similarity(primary_vector, candidate_vector)
                if similarity > 0.995:
                    frame.at[candidate_index, "Selection_Reason"] = (
                        "augmentation_not_meaningfully_different"
                    )
                    continue
                if similarity < 0.25:
                    frame.at[candidate_index, "Selection_Reason"] = (
                        "augmentation_embedding_inconsistent"
                    )
                    continue
                chosen.append(candidate_index)
                break
        for order, index in enumerate(chosen):
            frame.at[index, "Selected_For_Candidate"] = True
            frame.at[index, "Selection_Reason"] = (
                "selected_original_or_primary" if order == 0 else "selected_one_diverse_augmentation"
            )
            selected_ids.append(_text(frame.at[index, "Source_ID"]))
        for index in ordered.index:
            if index not in chosen and frame.at[index, "Selection_Reason"] == frame.at[index, "Policy_Reason"]:
                frame.at[index, "Selection_Reason"] = "excess_augmentation_group_limit"
        groups_rows.append(
            {
                "Source_Group_ID": group_id,
                "Candidate_Owner": key[0],
                "Lineage_Key": "|".join(key),
                "Group_Members": len(group),
                "Active_Members": len(chosen),
                "Active_Originals": int(
                    sum(not bool(frame.at[index, "Is_Augmentation"]) for index in chosen)
                ),
                "Active_Augmentations": int(
                    sum(bool(frame.at[index, "Is_Augmentation"]) for index in chosen)
                ),
                "Selected_Source_IDs": ";".join(
                    _text(frame.at[index, "Source_ID"]) for index in chosen
                ),
                "Maximum_Active": 2,
            }
        )

    for owner, group in frame.groupby("Candidate_Owner", sort=True):
        valid_candidates = group[
            group["Embedding_Valid"].map(bool) & group["Policy_Eligible"].map(bool)
        ]
        selected = group[group["Selected_For_Candidate"].map(bool)]
        if len(valid_candidates) and not len(selected):
            raise EmbeddingFamilyError(
                f"Source balancing removed all valid evidence for {owner}"
            )
    selected_frame = frame[frame["Selected_For_Candidate"].map(bool)]
    if selected_frame["Source_Group_ID"].value_counts().gt(2).any():
        raise EmbeddingFamilyError("Augmentation group active-image limit was violated")
    summary = {
        "schema_version": FAMILY_SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "configured_source_role": "production_recorded_dataset",
        "configured_source_images": len(frame),
        "source_groups": len(groups_rows),
        "active_sources": len(selected_frame),
        "active_originals": int((~selected_frame["Is_Augmentation"].map(bool)).sum()),
        "active_augmentations": int(selected_frame["Is_Augmentation"].map(bool).sum()),
        "excluded_exact_duplicates": int(
            frame["Policy_Reason"].eq("exact_duplicate_included_once").sum()
        ),
        "excluded_reviewer_confirmed_duplicates": int(
            frame["Policy_Reason"].eq("reviewer_confirmed_duplicate").sum()
        ),
        "excluded_excess_augmentations": int(
            frame["Selection_Reason"].isin(
                ["excess_augmentation_group_limit", "augmentation_not_meaningfully_different"]
            ).sum()
        ),
        "reviewer_multiple_people_excluded": int(
            frame["Policy_Reason"].eq("reviewer_confirmed_multiple_people").sum()
        ),
        "deferred_images_active": int(
            selected_frame["Review_Action"].eq("not_selected_for_compact_review").sum()
        ),
        "dataset_files_modified": False,
        "maximum_active_images_per_source_group": 2,
        "maximum_additional_augmentation_per_group": 1,
        "deterministic": True,
    }
    groups = pd.DataFrame(groups_rows).sort_values(
        ["Candidate_Owner", "Source_Group_ID"], kind="stable"
    ) if groups_rows else pd.DataFrame()
    decision_columns = [
        "Source_ID",
        "Candidate_Owner",
        "Relative_Path",
        "Source_SHA256",
        "Review_Item_ID",
        "Review_Action",
        "Exact_Duplicate_Group_ID",
        "Source_Group_ID",
        "Is_Augmentation",
        "Embedding_Valid",
        "Extraction_Reason",
        "Selected_For_Candidate",
        "Selection_Reason",
    ]
    decisions = frame[decision_columns].copy()
    return SourcePolicyResult(
        sources=frame,
        groups=groups,
        decisions=decisions,
        summary=summary,
        selected_source_ids=tuple(sorted(selected_ids)),
    )


def _tree_state(
    root: Path,
    *,
    label: str,
    excluded_directory_names: Iterable[str] = (),
    prohibit_mon_p3: bool = True,
) -> dict[str, Any]:
    root = Path(root).resolve()
    excluded = {_text(value).lower() for value in excluded_directory_names}
    rows: list[dict[str, Any]] = []
    if root.is_dir():
        for current, directories, files in os.walk(root):
            current_path = Path(current)
            directories[:] = [
                name
                for name in sorted(directories)
                if name.lower() not in excluded
                and (not prohibit_mon_p3 or MON_P3_TOKEN.lower() not in name.lower())
            ]
            if prohibit_mon_p3 and MON_P3_TOKEN.lower() in str(current_path).lower():
                directories[:] = []
                continue
            for name in sorted(files):
                path = current_path / name
                if prohibit_mon_p3 and MON_P3_TOKEN.lower() in str(path).lower():
                    continue
                rows.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "size_bytes": int(path.stat().st_size),
                        "sha256": _sha256_file(path),
                    }
                )
    return {
        "label": label,
        "root": str(root),
        "exists": root.is_dir(),
        "file_count": len(rows),
        "total_bytes": sum(row["size_bytes"] for row in rows),
        "aggregate_sha256": _stable_digest(rows),
        "files": rows,
        "mon_p3_paths_accessed": False,
    }


def snapshot_phase_i_b_protected_state(
    *,
    repo_root: Path,
    production_embeddings_path: Path,
    production_summary_path: Path,
    student_map_path: Path,
    enrollment_review_package: Path,
    enrollment_approvals_path: Path,
    cctv_review_package: Path,
    cctv_approvals_path: Path,
    forensic_run_dir: Path,
    dataset_root: Path,
    augmented_dataset_root: Path | None = None,
    yunet_model: Path = YUNET_MODEL,
    sface_model: Path = SFACE_MODEL,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    forensic_run_dir = Path(forensic_run_dir).resolve()
    fixed_paths = [
        Path(production_embeddings_path).resolve(),
        Path(production_summary_path).resolve(),
        Path(student_map_path).resolve(),
        Path(yunet_model).resolve(),
        Path(sface_model).resolve(),
        repo_root / "src" / "face_attendance" / "config.py",
        repo_root / "scripts" / "mark_attendance_checkpoints.py",
        repo_root
        / "models"
        / "candidate_registry"
        / "rejected"
        / f"{REJECTED_CANDIDATE_ID}.json",
        Path(enrollment_approvals_path).resolve(),
        Path(cctv_approvals_path).resolve(),
        Path(enrollment_review_package).resolve() / "package_manifest.json",
        Path(enrollment_review_package).resolve() / "private" / "source_mapping.csv",
        Path(cctv_review_package).resolve() / "package_manifest.json",
        Path(cctv_review_package).resolve() / "private" / "source_mapping.csv",
        forensic_run_dir / "multisession_ground_truth.csv",
        forensic_run_dir / "multisession_ground_truth_manifest.json",
        forensic_run_dir / "output_manifest.json",
    ]
    benchmark_manifest_path = forensic_run_dir / "multisession_ground_truth_manifest.json"
    if benchmark_manifest_path.is_file():
        benchmark_manifest = json.loads(
            benchmark_manifest_path.read_text(encoding="utf-8")
        )
        for source in benchmark_manifest.get("sources", []):
            for artifact in dict(source.get("files") or {}).values():
                path = Path(_text(dict(artifact or {}).get("path")))
                if path.is_file() and MON_P3_TOKEN.lower() not in str(path).lower():
                    fixed_paths.append(path.resolve())
    fixed_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in fixed_paths:
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        if MON_P3_TOKEN.lower() in key:
            raise EmbeddingFamilyError("Protected snapshot configuration attempted to access MON_P3")
        fixed_rows.append(
            {
                "path": str(path),
                "exists": path.is_file(),
                "size_bytes": int(path.stat().st_size) if path.is_file() else 0,
                "sha256": _sha256_file(path) if path.is_file() else "",
            }
        )
    trees = {
        "dataset": _tree_state(dataset_root, label="configured_enrollment_dataset"),
        "actual_present": _tree_state(
            repo_root / "actual_present", label="actual_present_inputs"
        ),
        "official_attendance": _tree_state(
            repo_root / "attendance_output",
            label="official_attendance_outputs",
            excluded_directory_names={"diagnostics", "embedding_forensics"},
        ),
        "enrollment_review_package": _tree_state(
            enrollment_review_package, label="compact_enrollment_review_package"
        ),
        "cctv_review_package": _tree_state(
            cctv_review_package, label="compact_cctv_review_package"
        ),
    }
    if augmented_dataset_root is not None:
        trees["augmented_dataset"] = _tree_state(
            augmented_dataset_root,
            label="configured_augmented_dataset_nonproduction",
        )
    payload = {
        "schema_version": FAMILY_SCHEMA_VERSION,
        "captured_at": _now(),
        "fixed_files": fixed_rows,
        "trees": trees,
        "versions_root_excluded": True,
        "mon_p3_paths_accessed": False,
    }
    payload["state_fingerprint_sha256"] = _stable_digest(
        {"fixed_files": fixed_rows, "trees": trees}
    )
    return payload


def compare_protected_states(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    before_fixed = {
        _text(row["path"]): (_text(row["sha256"]), int(row["size_bytes"]))
        for row in before.get("fixed_files", [])
    }
    after_fixed = {
        _text(row["path"]): (_text(row["sha256"]), int(row["size_bytes"]))
        for row in after.get("fixed_files", [])
    }
    changed_fixed = sorted(
        path
        for path in set(before_fixed) | set(after_fixed)
        if before_fixed.get(path) != after_fixed.get(path)
    )
    before_trees = {
        key: _text(value.get("aggregate_sha256"))
        for key, value in dict(before.get("trees") or {}).items()
    }
    after_trees = {
        key: _text(value.get("aggregate_sha256"))
        for key, value in dict(after.get("trees") or {}).items()
    }
    changed_trees = sorted(
        key
        for key in set(before_trees) | set(after_trees)
        if before_trees.get(key) != after_trees.get(key)
    )
    return {
        "unchanged": not changed_fixed and not changed_trees,
        "changed_fixed_files": changed_fixed,
        "changed_trees": changed_trees,
        "production_embeddings_unchanged": before_fixed.get(
            next(
                (path for path in before_fixed if path.lower().endswith("student_embeddings.pkl")),
                "",
            )
        )
        == after_fixed.get(
            next(
                (path for path in after_fixed if path.lower().endswith("student_embeddings.pkl")),
                "",
            )
        ),
        "datasets_unchanged": not any(
            key in changed_trees for key in {"dataset", "augmented_dataset"}
        ),
        "attendance_unchanged": "official_attendance" not in changed_trees,
        "mon_p3_processed": False,
    }


def _extract_single_cctv_embedding(
    image_path: Path, engine: FaceEngine
) -> tuple[np.ndarray, dict[str, Any]]:
    image = cv2.imread(str(image_path))
    if image is None or image.size == 0:
        raise EmbeddingFamilyError(f"Approved CCTV crop cannot be decoded: {image_path}")
    faces = engine.detect_faces(image)
    if len(faces) != 1:
        raise EmbeddingFamilyError(
            f"Approved CCTV crop must contain exactly one detectable face: "
            f"{image_path} ({len(faces)} found)"
        )
    feature = engine.extract_feature(image, faces[0])
    if feature is None:
        raise EmbeddingFamilyError(
            f"Could not extract SFace embedding from approved crop: {image_path}"
        )
    vector = normalize_embedding(feature)
    if not vector.size or not np.isfinite(vector).all():
        raise EmbeddingFamilyError(f"Invalid CCTV embedding: {image_path}")
    return vector.astype(np.float32), {
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "detector_score": float(engine.face_score(faces[0])),
    }


def extract_cleaned_enrollment_records(
    *,
    repo_root: Path,
    sources: pd.DataFrame,
    engine: FaceEngine,
    config: EnrollmentBuildConfig,
    progress: Callable[[str], None] | None = None,
) -> tuple[SourcePolicyResult, list[dict[str, Any]], dict[str, EnrollmentExtraction]]:
    repo_root = Path(repo_root).resolve()
    prefiltered = apply_exact_duplicate_policy(sources)
    eligible = prefiltered[prefiltered["Policy_Eligible"].map(bool)]
    extractions: dict[str, EnrollmentExtraction] = {}
    total = len(eligible)
    for ordinal, row in enumerate(eligible.to_dict("records"), start=1):
        if progress and (ordinal == 1 or ordinal % 25 == 0 or ordinal == total):
            progress(f"Enrollment embedding extraction: {ordinal}/{total}")
        source_id = _text(row["Source_ID"])
        require_single = _text(row.get("Review_Action")) == ENROLLMENT_WRONG
        extractions[source_id] = extract_enrollment_embedding(
            Path(_text(row["Source_Path"])),
            engine,
            config,
            require_single_face=require_single,
        )
    policy = select_balanced_sources(prefiltered, extractions)
    source_by_id = {
        _text(row["Source_ID"]): row for row in policy.sources.to_dict("records")
    }
    records: list[dict[str, Any]] = []
    for source_id in policy.selected_source_ids:
        row = source_by_id[source_id]
        extraction = extractions[source_id]
        if extraction.embedding is None:
            raise EmbeddingFamilyError(f"Selected enrollment source has no embedding: {source_id}")
        source_path = Path(_text(row["Source_Path"])).resolve()
        try:
            image_path = source_path.relative_to(repo_root).as_posix()
        except ValueError:
            image_path = str(source_path)
        records.append(
            {
                "roll_no": _canonical(row["Candidate_Owner"]),
                "image_path": image_path,
                "detection_score": float(extraction.detector_score),
                "embedding": np.asarray(extraction.embedding, dtype=np.float32),
                "_source_id": source_id,
                "_source_kind": "cleaned_enrollment",
                "_source_session": "",
            }
        )
    records.sort(key=lambda record: (record["roll_no"], record["image_path"]))
    return policy, records, extractions


def stage_approved_cctv_sources(
    *,
    temporary_family_dir: Path,
    final_family_dir: Path,
    repo_root: Path,
    approval_bundle: CompactApprovalBundle,
    correction: IdentityCorrectionResult,
    engine: FaceEngine,
    roster_rolls: set[str],
    expected_approved: int = 35,
    progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    approved = approval_bundle.cctv[
        approval_bundle.cctv["Review_Action"].eq(CCTV_APPROVE)
    ].sort_values("Item_ID", kind="stable")
    if len(approved) != expected_approved:
        raise EmbeddingFamilyError(
            f"Expected {expected_approved} approved CCTV items, found {len(approved)}"
        )
    if approved["Source_SHA256"].duplicated().any():
        duplicates = sorted(
            approved.loc[
                approved["Source_SHA256"].duplicated(False), "Source_SHA256"
            ].unique()
        )
        raise EmbeddingFamilyError(
            "Approved CCTV source hashes are duplicated: " + ", ".join(duplicates[:5])
        )
    staging_root = Path(temporary_family_dir) / "candidate_sources" / "verified_cctv"
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for ordinal, row in enumerate(approved.to_dict("records"), start=1):
        item_id = _text(row["Item_ID"])
        provenance = row["Private_Provenance"]
        source_roll = _canonical(provenance.get("actual_roll"))
        candidate_roll = (
            TARGET_IDENTITY
            if item_id in correction.affected_cctv_item_ids
            else source_roll
        )
        if candidate_roll not in roster_rolls:
            raise EmbeddingFamilyError(
                f"Approved CCTV owner is not in the authoritative subject roster: "
                f"{item_id} -> {candidate_roll}"
            )
        if _text(provenance.get("identity_source")) != "human_review_actual_roll":
            raise EmbeddingFamilyError(
                f"Model prediction attempted to control CCTV ownership: {item_id}"
            )
        source_path = Path(_text(row["Resolved_Source_Path"])).resolve()
        source_hash = _text(row["Source_SHA256"])
        if not source_path.is_file() or _sha256_file(source_path) != source_hash:
            raise EmbeddingFamilyError(f"Approved CCTV source changed: {item_id}")
        session = _text(provenance.get("source_session"))
        session_token = "tue_p1" if session == TUE_P1_SESSION else "tue_p2"
        destination = staging_root / session_token / f"{item_id}{source_path.suffix.lower()}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        if _sha256_file(destination) != source_hash:
            raise EmbeddingFamilyError(f"Staged CCTV crop hash mismatch: {item_id}")
        vector, metrics = _extract_single_cctv_embedding(destination, engine)
        final_destination = Path(final_family_dir) / destination.relative_to(
            temporary_family_dir
        )
        try:
            image_path = final_destination.relative_to(repo_root).as_posix()
        except ValueError:
            image_path = str(final_destination)
        source_id = f"CCTV-{item_id}"
        records.append(
            {
                "roll_no": candidate_roll,
                "image_path": image_path,
                "detection_score": float(metrics["detector_score"]),
                "embedding": vector,
                "_source_id": source_id,
                "_source_kind": "human_approved_cctv_crop",
                "_source_session": session,
            }
        )
        rows.append(
            {
                "Source_ID": source_id,
                "Item_ID": item_id,
                "Source_Package_ID": _text(row["Package_ID"]),
                "Source_Broad_Item_ID": _text(provenance.get("source_broad_item_id")),
                "Source_Path": str(source_path),
                "Source_SHA256": source_hash,
                "Staged_Path": str(final_destination),
                "Staged_SHA256": source_hash,
                "Source_Session": session,
                "Track_ID": _text(provenance.get("tracklet_id")),
                "Observation_ID": _text(provenance.get("observation_id")),
                "Checkpoint": _text(provenance.get("checkpoint")),
                "Camera": _text(provenance.get("camera")),
                "Human_Actual_Roll": source_roll,
                "Candidate_Roll": candidate_roll,
                "Identity_Source": "human_review_actual_roll",
                "Identity_Correction_ID": (
                    _text(correction.manifest.get("correction_id"))
                    if item_id in correction.affected_cctv_item_ids
                    else ""
                ),
                "Detector_Score": float(metrics["detector_score"]),
                "Embedding_Dimension": int(vector.size),
            }
        )
        if progress and (ordinal == 1 or ordinal % 10 == 0 or ordinal == len(approved)):
            progress(f"Approved CCTV staging and extraction: {ordinal}/{len(approved)}")
    staging = pd.DataFrame(rows)
    session_counts = {
        _text(key): int(value)
        for key, value in staging["Source_Session"].value_counts().sort_index().items()
    }
    manifest = {
        "schema_version": FAMILY_SCHEMA_VERSION,
        "approved_rows": len(approved),
        "staged_unique_crops": len(staging),
        "rejected_rows_excluded": int(
            approval_bundle.cctv["Review_Action"].ne(CCTV_APPROVE).sum()
        ),
        "source_session_counts": session_counts,
        "source_hashes_verified": True,
        "staged_hashes_verified": True,
        "human_actual_roll_controls_ownership": True,
        "model_prediction_controls_ownership": False,
        "live_dataset_writes": False,
        "super_resolution_used": False,
        "records": rows,
    }
    records.sort(
        key=lambda record: (
            record["_source_session"],
            record["roll_no"],
            record["_source_id"],
        )
    )
    return staging, records, manifest


def _public_embedding_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "roll_no": _canonical(record.get("roll_no")),
        "image_path": _text(record.get("image_path")),
        "detection_score": float(record.get("detection_score") or 0.0),
        "embedding": np.asarray(record.get("embedding"), dtype=np.float32).reshape(-1),
    }


def _validate_candidate_records(
    records: Sequence[dict[str, Any]], expected_dimension: int
) -> None:
    if not records:
        raise EmbeddingFamilyError("Candidate embedding variant contains no records")
    seen_sources: set[str] = set()
    for record in records:
        roll = _canonical(record.get("roll_no"))
        if not roll:
            raise EmbeddingFamilyError("Candidate embedding record has an empty canonical roll")
        source_id = _text(record.get("_source_id"))
        if source_id in seen_sources:
            raise EmbeddingFamilyError(f"Candidate source reused twice: {source_id}")
        seen_sources.add(source_id)
        vector = np.asarray(record.get("embedding"), dtype=np.float32).reshape(-1)
        if (
            vector.size != expected_dimension
            or not np.isfinite(vector).all()
            or float(np.linalg.norm(vector)) <= 1e-12
        ):
            raise EmbeddingFamilyError(
                f"Candidate record has invalid embedding dimension/data: {source_id}"
            )
        path_parts = {
            _canonical(part)
            for part in Path(_text(record.get("image_path"))).parts
            if _canonical(part)
        }
        if TARGET_IDENTITY in path_parts and roll != TARGET_IDENTITY:
            raise EmbeddingFamilyError(
                "Corrected AI0110 source has a non-AI0110 owner in candidate records"
            )


def _variant_artifact_hashes(variant_dir: Path) -> dict[str, str]:
    return {
        path.relative_to(variant_dir).as_posix(): _sha256_file(path)
        for path in sorted(Path(variant_dir).rglob("*"))
        if path.is_file() and path.name != "version_manifest.json"
    }


def _write_variant(
    *,
    temporary_variant_dir: Path,
    final_variant_dir: Path,
    family_id: str,
    definition: VariantDefinition,
    family_fingerprint: str,
    records: Sequence[dict[str, Any]],
    parent_payload: dict[str, Any],
    parent_embeddings_sha256: str,
    parent_summary_sha256: str,
    model_hashes: dict[str, str],
    source_manifest_sha256: str,
    enrollment_approval_sha256: str,
    cctv_approval_sha256: str,
    identity_correction_sha256: str,
    augmentation_policy_sha256: str,
    roster_rolls: set[str],
    build_timestamp: str,
    expected_dimension: int,
) -> dict[str, Any]:
    variant_id = f"{family_id}-{definition.key.lower()}"
    temporary_variant_dir.mkdir(parents=True, exist_ok=False)
    _validate_candidate_records(records, expected_dimension)
    public_records = [_public_embedding_record(record) for record in records]
    source_ids = [_text(record.get("_source_id")) for record in records]
    content_fingerprint = _stable_digest(
        {
            "family_fingerprint": family_fingerprint,
            "variant": asdict(definition),
            "source_ids": source_ids,
            "source_manifest_sha256": source_manifest_sha256,
            "model_hashes": model_hashes,
        }
    )
    metadata = {
        **dict(parent_payload.get("metadata") or {}),
        "embedding_version_id": variant_id,
        "embedding_version_family_id": family_id,
        "variant_role": definition.role,
        "status": "built_unapproved",
        "production_promoted": False,
        "dataset": "review_controlled_candidate_sources",
        "detection_score": 0.60,
        "det_max_width": 1280,
        "min_face_size": 50,
        "min_area_ratio": 0.003,
        "pad_percent": 0.25,
        "min_landmarks_inside": 4,
        "min_eye_distance": 8.0,
        "validation_mode": "padded_photo_landmark_validation",
        "match_threshold_unchanged": OFFICIAL_MATCH_THRESHOLD,
        "margin_threshold_unchanged": OFFICIAL_MARGIN_THRESHOLD,
        "aggregate_mode_unchanged": OFFICIAL_AGGREGATE,
        "attendance_checkpoints_unchanged": OFFICIAL_ATTENDANCE_CHECKPOINTS,
    }
    payload = {
        "version": 1,
        "created_at": build_timestamp,
        "metadata": metadata,
        "records": public_records,
    }
    with (temporary_variant_dir / "student_embeddings.pkl").open("wb") as file_obj:
        pickle.dump(payload, file_obj)

    per_roll: list[dict[str, Any]] = []
    for roll in sorted(roster_rolls | {_canonical(record["roll_no"]) for record in records}):
        roll_records = [record for record in records if _canonical(record["roll_no"]) == roll]
        enrollment_count = sum(
            _text(record.get("_source_kind")) == "cleaned_enrollment"
            for record in roll_records
        )
        cctv_count = len(roll_records) - enrollment_count
        per_roll.append(
            {
                "Roll_Number": roll,
                "Images_Read": len(roll_records),
                "Faces_Used": len(roll_records),
                "No_Valid_Face": 0,
                "Multiple_Faces": 0,
                "Rejected_Crops": 0,
                "Read_Fail": 0,
                "Enrollment_Faces": enrollment_count,
                "Approved_CCTV_Faces": cctv_count,
                "In_Authoritative_CVO_Roster": roll in roster_rolls,
                "Dataset_Status": (
                    "Dataset Missing"
                    if roll == MISSING_CVO_IDENTITY and not roll_records
                    else ("Covered" if roll_records else "No Candidate Embedding")
                ),
            }
        )
    coverage = pd.DataFrame(per_roll)
    summary_columns = [
        "Roll_Number",
        "Images_Read",
        "Faces_Used",
        "No_Valid_Face",
        "Multiple_Faces",
        "Rejected_Crops",
        "Read_Fail",
    ]
    _write_csv(temporary_variant_dir / "embedding_summary.csv", coverage[summary_columns])
    _write_csv(temporary_variant_dir / "student_coverage.csv", coverage)
    source_manifest = pd.DataFrame(
        [
            {
                "Record_Order": index,
                "Source_ID": _text(record.get("_source_id")),
                "Canonical_Roll": _canonical(record.get("roll_no")),
                "Source_Kind": _text(record.get("_source_kind")),
                "Source_Session": _text(record.get("_source_session")),
                "Image_Path": _text(record.get("image_path")),
            }
            for index, record in enumerate(records)
        ]
    )
    _write_csv(temporary_variant_dir / "source_records.csv", source_manifest)
    covered_roster = sorted(
        set(coverage.loc[coverage["Faces_Used"].gt(0), "Roll_Number"]) & roster_rolls
    )
    missing = sorted(roster_rolls - set(covered_roster))
    (temporary_variant_dir / "build_report.md").write_text(
        "\n".join(
            [
                f"# {variant_id}",
                "",
                "Status: `built_unapproved`",
                "",
                f"- Role: `{definition.role}`",
                f"- Embedding records: {len(records)}",
                f"- CVO coverage: {len(covered_roster)}/{len(roster_rolls)}",
                f"- CCTV source sessions included: {', '.join(definition.included_cctv_sessions) or 'none'}",
                f"- Evaluation sessions allowed: {', '.join(definition.evaluation_sessions) or 'none'}",
                f"- Training-contaminated sessions: {', '.join(definition.contaminated_sessions) or 'none'}",
                f"- Descriptive only: {'yes' if definition.descriptive_only else 'no'}",
                "- Production promoted: no",
                "- Production files changed: no",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": FAMILY_SCHEMA_VERSION,
        "version_id": variant_id,
        "family_id": family_id,
        "variant_key": definition.key,
        "variant_role": definition.role,
        "status": "built_unapproved",
        "production_promoted": False,
        "build_timestamp": build_timestamp,
        "deterministic_content_fingerprint": content_fingerprint,
        "parent_production_embedding_sha256": parent_embeddings_sha256,
        "production_summary_sha256": parent_summary_sha256,
        "model_hashes": model_hashes,
        "build_configuration": asdict(EnrollmentBuildConfig()),
        "official_recognition_configuration": {
            "match_threshold": OFFICIAL_MATCH_THRESHOLD,
            "margin_threshold": OFFICIAL_MARGIN_THRESHOLD,
            "aggregate": OFFICIAL_AGGREGATE,
            "attendance_checkpoints": OFFICIAL_ATTENDANCE_CHECKPOINTS,
        },
        "source_manifest_sha256": source_manifest_sha256,
        "approval_csv_hashes": {
            "enrollment": enrollment_approval_sha256,
            "cctv": cctv_approval_sha256,
        },
        "identity_correction_manifest_sha256": identity_correction_sha256,
        "augmentation_policy_sha256": augmentation_policy_sha256,
        "per_student_image_counts": {
            row["Roll_Number"]: int(row["Images_Read"]) for row in per_roll
        },
        "per_student_embedding_counts": {
            row["Roll_Number"]: int(row["Faces_Used"]) for row in per_roll
        },
        "embedding_records": len(records),
        "embedding_dimension": expected_dimension,
        "source_sessions_included": list(definition.included_cctv_sessions),
        "source_sessions_excluded": sorted(
            {TUE_P1_SESSION, TUE_P2_SESSION} - set(definition.included_cctv_sessions)
        ),
        "evaluation_sessions_allowed": list(definition.evaluation_sessions),
        "evaluation_sessions_contaminated": list(definition.contaminated_sessions),
        "training_contaminated_descriptive_only": definition.descriptive_only,
        "missing_students": missing,
        "canonical_identity_coverage": {
            "covered": len(covered_roster),
            "roster": len(roster_rolls),
            "rolls": covered_roster,
        },
        "artifact_sha256": _variant_artifact_hashes(temporary_variant_dir),
        "promotion_allowed_in_phase_1_2i_b": False,
    }
    _write_json(temporary_variant_dir / "version_manifest.json", manifest)
    return manifest


def load_frozen_benchmark_evidence(
    *,
    forensic_run_dir: Path,
    student_map_path: Path,
    expected_rows: int = 120,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    forensic_run_dir = Path(forensic_run_dir).resolve()
    output_manifest_path = forensic_run_dir / "output_manifest.json"
    try:
        verify_output_manifest(forensic_run_dir, output_manifest_path)
    except ShadowValidationError as exc:
        raise EmbeddingFamilyError(str(exc)) from exc
    benchmark_path = forensic_run_dir / "multisession_ground_truth.csv"
    manifest_path = forensic_run_dir / "multisession_ground_truth_manifest.json"
    if not benchmark_path.is_file() or not manifest_path.is_file():
        raise EmbeddingFamilyError(f"Frozen benchmark is incomplete: {forensic_run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    benchmark = pd.read_csv(benchmark_path, dtype=str, keep_default_na=False)
    if len(benchmark) != expected_rows:
        raise EmbeddingFamilyError(
            f"Frozen benchmark row count mismatch: expected {expected_rows}, found {len(benchmark)}"
        )
    required = {
        "Benchmark_Row_ID",
        "Package_ID",
        "Review_ID",
        "Tracklet_ID",
        "Session_ID",
        "Review_Status",
        "Actual_Roll",
        "Checkpoint_ID",
        "Camera_ID",
        "Evidence_SHA256",
        "Source_Kind",
        "Baseline_Status",
    }
    missing = sorted(required.difference(benchmark.columns))
    if missing:
        raise EmbeddingFamilyError(
            "Frozen benchmark schema is incomplete: " + ", ".join(missing)
        )
    if benchmark["Benchmark_Row_ID"].duplicated().any():
        raise EmbeddingFamilyError("Frozen benchmark contains duplicate benchmark row IDs")
    if benchmark.duplicated(["Session_ID", "Tracklet_ID"]).any():
        raise EmbeddingFamilyError("Frozen benchmark contains duplicate session track rows")
    sessions = set(benchmark["Session_ID"].map(_text))
    if sessions != {TUE_P1_SESSION, TUE_P2_SESSION}:
        raise EmbeddingFamilyError(f"Frozen benchmark sessions changed: {sorted(sessions)}")
    source_counts = {
        _text(key): int(value)
        for key, value in benchmark["Source_Kind"].value_counts().sort_index().items()
    }
    expected_source_counts = {
        "tue_p1_accepted_review": 37,
        "tue_p1_corrected_unresolved_review": 63,
        "tue_p2_shadow_recovery_review": 20,
    }
    if source_counts != expected_source_counts:
        raise EmbeddingFamilyError(
            f"Frozen benchmark source counts changed: {source_counts}"
        )
    if _text(manifest.get("benchmark_id")) == "":
        raise EmbeddingFamilyError("Frozen benchmark ID is missing")
    recorded_artifact = dict(manifest.get("benchmark_artifact") or {})
    if _text(recorded_artifact.get("sha256")) != _sha256_file(benchmark_path):
        raise EmbeddingFamilyError("Frozen benchmark artifact hash does not match its manifest")
    if _text(dict(manifest.get("student_map") or {}).get("sha256")) != _sha256_file(
        student_map_path
    ):
        raise EmbeddingFamilyError("Frozen benchmark roster hash does not match the current roster")

    package_evidence: dict[tuple[str, str], tuple[Path, int]] = {}
    verified_source_files = 0
    for source in manifest.get("sources", []):
        package_id = _text(source.get("package_id"))
        files = dict(source.get("files") or {})
        for label, artifact in files.items():
            artifact = dict(artifact or {})
            path_text = _text(artifact.get("path"))
            if MON_P3_TOKEN.lower() in path_text.lower():
                raise EmbeddingFamilyError("Frozen benchmark manifest references MON_P3")
            path = Path(path_text)
            expected_hash = _text(artifact.get("sha256"))
            if not path.is_file() or _sha256_file(path) != expected_hash:
                raise EmbeddingFamilyError(
                    f"Frozen benchmark source hash mismatch: {label} -> {path}"
                )
            verified_source_files += 1
        review_manifest_path = Path(
            _text(dict(files.get("review_manifest") or {}).get("path"))
        )
        review_manifest = json.loads(
            review_manifest_path.read_text(encoding="utf-8")
        )
        if _text(review_manifest.get("package_id")) != package_id:
            raise EmbeddingFamilyError(
                f"Frozen reviewer package ID mismatch: {review_manifest_path}"
            )
        reviewer_dir = review_manifest_path.parent
        for item in review_manifest.get("review_items", []):
            review_id = _text(item.get("review_id") or item.get("item_id"))
            evidence_path = reviewer_dir / _text(item.get("evidence_file"))
            if not evidence_path.is_file():
                raise EmbeddingFamilyError(
                    f"Frozen reviewer evidence not found: {evidence_path}"
                )
            key = (package_id, review_id)
            if key in package_evidence:
                raise EmbeddingFamilyError(
                    f"Frozen reviewer evidence key is duplicated: {key}"
                )
            package_evidence[key] = (
                evidence_path.resolve(),
                int(item.get("evidence_views") or 0),
            )

    evidence_paths: list[str] = []
    evidence_views: list[int] = []
    for row in benchmark.to_dict("records"):
        key = (_text(row["Package_ID"]), _text(row["Review_ID"]))
        if key not in package_evidence:
            raise EmbeddingFamilyError(f"Frozen benchmark evidence key not found: {key}")
        evidence_path, views = package_evidence[key]
        if _sha256_file(evidence_path) != _text(row["Evidence_SHA256"]):
            raise EmbeddingFamilyError(
                f"Frozen benchmark evidence hash mismatch: {row['Benchmark_Row_ID']}"
            )
        evidence_paths.append(str(evidence_path))
        evidence_views.append(views)
    benchmark = benchmark.copy()
    benchmark["Evidence_Path"] = evidence_paths
    benchmark["Evidence_Views"] = evidence_views
    benchmark["Actual_Roll"] = benchmark["Actual_Roll"].map(_canonical)
    evidence_manifest = {
        "schema_version": FAMILY_SCHEMA_VERSION,
        "benchmark_id": _text(manifest.get("benchmark_id")),
        "benchmark_path": str(benchmark_path),
        "benchmark_sha256": _sha256_file(benchmark_path),
        "benchmark_manifest_path": str(manifest_path),
        "benchmark_manifest_sha256": _sha256_file(manifest_path),
        "reviewed_rows": len(benchmark),
        "source_counts": source_counts,
        "source_sessions": sorted(sessions),
        "verified_source_files": verified_source_files,
        "verified_evidence_files": len(evidence_paths),
        "evidence_mode": "reviewer_contact_sheet_reextraction",
        "video_decoding_rerun": False,
        "recognition_rerun": False,
        "official_attendance_written": False,
        "mon_p3_accessed": False,
    }
    return benchmark.sort_values("Benchmark_Row_ID", kind="stable").reset_index(
        drop=True
    ), evidence_manifest


def _aggregate_contact_sheet_embeddings(
    candidates: list[dict[str, Any]], max_selected: int = 5
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if not candidates:
        return None, {
            "selected": 0,
            "consistent": 0,
            "medoid_index": -1,
            "pairwise_similarity_min": None,
            "pairwise_similarity_median": None,
        }
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            -float(candidate["rank_score"]),
            float(candidate["face"][0]),
            float(candidate["face"][1]),
            float(candidate["face"][2]),
            float(candidate["face"][3]),
        ),
    )[:max_selected]
    vectors = [normalize_embedding(candidate["embedding"]) for candidate in ordered]
    mean_similarities = [
        float(np.mean([cosine_similarity(vector, other) for other in vectors]))
        for vector in vectors
    ]
    medoid_index = sorted(
        range(len(vectors)),
        key=lambda index: (-mean_similarities[index], index),
    )[0]
    medoid = vectors[medoid_index]
    consistent_indices = [
        index
        for index, vector in enumerate(vectors)
        if index == medoid_index or cosine_similarity(vector, medoid) >= 0.25
    ]
    consistent_vectors = [vectors[index] for index in consistent_indices]
    aggregate = normalize_embedding(np.mean(np.stack(consistent_vectors), axis=0))
    if not aggregate.size or not np.isfinite(aggregate).all():
        return None, {
            "selected": len(vectors),
            "consistent": len(consistent_vectors),
            "medoid_index": medoid_index,
            "pairwise_similarity_min": None,
            "pairwise_similarity_median": None,
        }
    pairwise = [
        cosine_similarity(consistent_vectors[left], consistent_vectors[right])
        for left in range(len(consistent_vectors))
        for right in range(left + 1, len(consistent_vectors))
    ]
    return aggregate.astype(np.float32), {
        "selected": len(vectors),
        "consistent": len(consistent_vectors),
        "medoid_index": medoid_index,
        "pairwise_similarity_min": float(min(pairwise)) if pairwise else None,
        "pairwise_similarity_median": (
            float(np.median(np.asarray(pairwise, dtype=np.float32))) if pairwise else None
        ),
    }


def extract_benchmark_contact_sheet_features(
    benchmark: pd.DataFrame,
    *,
    engine: FaceEngine,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    features: dict[str, np.ndarray] = {}
    audit_rows: list[dict[str, Any]] = []
    total = len(benchmark)
    for ordinal, row in enumerate(benchmark.to_dict("records"), start=1):
        row_id = _text(row["Benchmark_Row_ID"])
        evidence_path = Path(_text(row["Evidence_Path"]))
        image = cv2.imread(str(evidence_path))
        candidates: list[dict[str, Any]] = []
        detected = 0
        reason = "accepted"
        if image is None or image.size == 0:
            reason = "evidence_decode_failed"
        else:
            faces = engine.detect_faces_scaled(image, max_width=1280)
            detected = int(len(faces))
            for face in faces:
                valid, _quality_reason, rank_score = face_quality_score(
                    face,
                    image.shape,
                    30,
                    0.0005,
                    4,
                    5.0,
                )
                if not valid:
                    continue
                feature = engine.extract_feature(image, face)
                if feature is None:
                    continue
                vector = normalize_embedding(feature)
                if not vector.size or not np.isfinite(vector).all():
                    continue
                candidates.append(
                    {
                        "face": face,
                        "embedding": vector,
                        "rank_score": rank_score,
                    }
                )
        aggregate, aggregate_metrics = _aggregate_contact_sheet_embeddings(candidates)
        if aggregate is None:
            reason = reason if reason != "accepted" else "no_compatible_face_feature"
        else:
            features[row_id] = aggregate
        audit_rows.append(
            {
                "Benchmark_Row_ID": row_id,
                "Session_ID": _text(row["Session_ID"]),
                "Source_Kind": _text(row["Source_Kind"]),
                "Evidence_Path": str(evidence_path),
                "Evidence_SHA256": _text(row["Evidence_SHA256"]),
                "Evidence_Views": int(row.get("Evidence_Views") or 0),
                "Detected_Faces": detected,
                "Valid_Features": len(candidates),
                "Selected_Features": aggregate_metrics["selected"],
                "Consistent_Features": aggregate_metrics["consistent"],
                "Pairwise_Similarity_Min": aggregate_metrics[
                    "pairwise_similarity_min"
                ],
                "Pairwise_Similarity_Median": aggregate_metrics[
                    "pairwise_similarity_median"
                ],
                "Feature_Available": aggregate is not None,
                "Feature_Dimension": int(aggregate.size) if aggregate is not None else 0,
                "Reason": reason,
                "Evidence_Mode": "reviewer_contact_sheet_reextraction",
            }
        )
        if progress and (ordinal == 1 or ordinal % 20 == 0 or ordinal == total):
            progress(f"Frozen reviewer evidence feature extraction: {ordinal}/{total}")
    return features, pd.DataFrame(audit_rows)


def _actual_class(row: dict[str, Any], roster_rolls: set[str]) -> str:
    status = _text(row.get("Review_Status")).lower()
    actual = _canonical(row.get("Actual_Roll"))
    if status == "identified" and actual in roster_rolls:
        return "known_student"
    if status == "not_in_mapping":
        return "not_in_mapping"
    if status in {"mixed", "mixed_track"}:
        return "mixed"
    if status in {"unidentifiable", "uncertain"}:
        return "unverifiable"
    return "unverifiable"


def score_benchmark(
    benchmark: pd.DataFrame,
    *,
    features: dict[str, np.ndarray],
    database: StudentEmbeddingDB,
    roster_rolls: set[str],
    model_id: str,
    allowed_sessions: Iterable[str] | None = None,
    training_contaminated_descriptive: bool = False,
) -> pd.DataFrame:
    allowed = set(allowed_sessions or [])
    rows: list[dict[str, Any]] = []
    selected = benchmark[
        benchmark["Session_ID"].isin(allowed)
    ] if allowed else benchmark
    selected = selected.drop_duplicates("Benchmark_Row_ID", keep="first")
    for source in selected.to_dict("records"):
        row_id = _text(source["Benchmark_Row_ID"])
        feature = features.get(row_id)
        actual_class = _actual_class(source, roster_rolls)
        actual_roll = _canonical(source.get("Actual_Roll"))
        if feature is None:
            accepted = False
            top1_roll = ""
            top2_roll = ""
            top1_score = 0.0
            top2_score = 0.0
            margin = 0.0
            matcher_reason = "evidence_feature_unavailable"
        else:
            match = database.match(
                feature,
                match_threshold=OFFICIAL_MATCH_THRESHOLD,
                margin_threshold=OFFICIAL_MARGIN_THRESHOLD,
                aggregate=OFFICIAL_AGGREGATE,
            )
            accepted = bool(match.accepted)
            top1_roll = _canonical(match.roll_no)
            top2_roll = _canonical(match.second_roll_no)
            top1_score = float(match.best_score)
            top2_score = float(match.second_score)
            margin = float(match.margin)
            matcher_reason = _text(match.reason)
        accepted_roll = top1_roll if accepted else ""
        correct_accepted = bool(
            accepted
            and actual_class == "known_student"
            and accepted_roll == actual_roll
        )
        wrong_accepted = bool(
            accepted
            and actual_class == "known_student"
            and accepted_roll != actual_roll
        )
        outsider = bool(accepted and actual_class == "not_in_mapping")
        mixed = bool(accepted and actual_class == "mixed")
        unverifiable = bool(accepted and actual_class == "unverifiable")
        rows.append(
            {
                "Model_ID": model_id,
                "Benchmark_Row_ID": row_id,
                "Package_ID": _text(source["Package_ID"]),
                "Review_ID": _text(source["Review_ID"]),
                "Tracklet_ID": _text(source["Tracklet_ID"]),
                "Session_ID": _text(source["Session_ID"]),
                "Source_Kind": _text(source["Source_Kind"]),
                "Baseline_Status": _text(source["Baseline_Status"]),
                "Checkpoint_ID": _text(source["Checkpoint_ID"]),
                "Camera_ID": _text(source["Camera_ID"]),
                "Review_Status": _text(source["Review_Status"]),
                "Actual_Class": actual_class,
                "Actual_Roll": actual_roll if actual_class == "known_student" else "",
                "Feature_Available": feature is not None,
                "Top1_Roll": top1_roll,
                "Top2_Roll": top2_roll,
                "Top1_Score": top1_score,
                "Top2_Score": top2_score,
                "Margin": margin,
                "Match_Threshold": OFFICIAL_MATCH_THRESHOLD,
                "Margin_Threshold": OFFICIAL_MARGIN_THRESHOLD,
                "Aggregate_Mode": OFFICIAL_AGGREGATE,
                "Accepted": accepted,
                "Accepted_Roll": accepted_roll,
                "Predicted_Identity_In_Authoritative_Roster": (
                    top1_roll in roster_rolls if top1_roll else False
                ),
                "Correct_Accepted": correct_accepted,
                "Wrong_Accepted": wrong_accepted,
                "Outsider_Absorption": outsider,
                "Mixed_Acceptance": mixed,
                "Unverifiable_Acceptance": unverifiable,
                "Correct_Unresolved": bool(not accepted and actual_class != "known_student"),
                "Incorrect_Unresolved": bool(
                    not accepted and actual_class == "known_student"
                ),
                "Raw_Top1_Correct": bool(
                    actual_class == "known_student" and top1_roll == actual_roll
                ),
                "Matcher_Reason": matcher_reason,
                "Evidence_Mode": "reviewer_contact_sheet_reextraction",
                "Training_Contaminated_Descriptive": training_contaminated_descriptive,
                "Promotion_Evidence_Eligible": not training_contaminated_descriptive,
            }
        )
    return pd.DataFrame(rows).sort_values("Benchmark_Row_ID", kind="stable").reset_index(
        drop=True
    )


def summarize_predictions(frame: pd.DataFrame) -> dict[str, Any]:
    if not len(frame):
        return {
            "rows": 0,
            "evaluable_rows": 0,
            "accepted": 0,
            "correct_accepted": 0,
            "wrong_accepted": 0,
            "outsider_absorption": 0,
            "mixed_acceptance": 0,
            "unverifiable_acceptance": 0,
        }
    known = frame[frame["Actual_Class"].eq("known_student")]
    unsafe_accepts = (
        frame["Wrong_Accepted"].map(bool)
        | frame["Outsider_Absorption"].map(bool)
        | frame["Mixed_Acceptance"].map(bool)
    )
    accepted = frame[frame["Accepted"].map(bool)]
    scores = frame.loc[frame["Feature_Available"].map(bool), "Top1_Score"].astype(float)
    margins = frame.loc[frame["Feature_Available"].map(bool), "Margin"].astype(float)
    return {
        "rows": len(frame),
        "evaluable_rows": int(frame["Feature_Available"].map(bool).sum()),
        "identified_rows": len(known),
        "accepted": int(frame["Accepted"].map(bool).sum()),
        "correct_accepted": int(frame["Correct_Accepted"].map(bool).sum()),
        "wrong_accepted": int(frame["Wrong_Accepted"].map(bool).sum()),
        "outsider_absorption": int(frame["Outsider_Absorption"].map(bool).sum()),
        "mixed_acceptance": int(frame["Mixed_Acceptance"].map(bool).sum()),
        "unverifiable_acceptance": int(
            frame["Unverifiable_Acceptance"].map(bool).sum()
        ),
        "unsafe_confirmed_acceptances": int(unsafe_accepts.sum()),
        "correct_unresolved": int(frame["Correct_Unresolved"].map(bool).sum()),
        "incorrect_unresolved": int(frame["Incorrect_Unresolved"].map(bool).sum()),
        "raw_top1_correct": int(frame["Raw_Top1_Correct"].map(bool).sum()),
        "unique_accepted_identities": int(accepted["Accepted_Roll"].nunique()),
        "identified_correct_accept_rate": (
            round(float(frame["Correct_Accepted"].map(bool).sum()) / len(known), 6)
            if len(known)
            else None
        ),
        "confirmed_accepted_precision": (
            round(
                float(frame["Correct_Accepted"].map(bool).sum())
                / max(
                    1,
                    int(frame["Correct_Accepted"].map(bool).sum())
                    + int(unsafe_accepts.sum()),
                ),
                6,
            )
            if len(accepted)
            else None
        ),
        "score_distribution": {
            "median": float(scores.median()) if len(scores) else None,
            "p90": float(scores.quantile(0.90)) if len(scores) else None,
            "min": float(scores.min()) if len(scores) else None,
            "max": float(scores.max()) if len(scores) else None,
        },
        "margin_distribution": {
            "median": float(margins.median()) if len(margins) else None,
            "p90": float(margins.quantile(0.90)) if len(margins) else None,
            "min": float(margins.min()) if len(margins) else None,
            "max": float(margins.max()) if len(margins) else None,
        },
    }


def compare_prediction_frames(
    baseline: pd.DataFrame, candidate: pd.DataFrame, comparison_id: str
) -> pd.DataFrame:
    columns = [
        "Benchmark_Row_ID",
        "Session_ID",
        "Source_Kind",
        "Checkpoint_ID",
        "Camera_ID",
        "Actual_Class",
        "Actual_Roll",
        "Accepted",
        "Accepted_Roll",
        "Correct_Accepted",
        "Wrong_Accepted",
        "Outsider_Absorption",
        "Mixed_Acceptance",
        "Unverifiable_Acceptance",
        "Incorrect_Unresolved",
        "Top1_Score",
        "Margin",
    ]
    joined = baseline[columns].merge(
        candidate[columns],
        on="Benchmark_Row_ID",
        how="inner",
        validate="one_to_one",
        suffixes=("_Baseline", "_Candidate"),
    )
    joined.insert(0, "Comparison_ID", comparison_id)
    joined["Correct_To_Wrong_Regression"] = (
        joined["Correct_Accepted_Baseline"].map(bool)
        & joined["Wrong_Accepted_Candidate"].map(bool)
    )
    joined["Lost_Correct_Acceptance"] = (
        joined["Correct_Accepted_Baseline"].map(bool)
        & ~joined["Correct_Accepted_Candidate"].map(bool)
    )
    joined["Recovered_Correct_Acceptance"] = (
        ~joined["Correct_Accepted_Baseline"].map(bool)
        & joined["Correct_Accepted_Candidate"].map(bool)
    )
    joined["New_Wrong_Accepted"] = joined.apply(
        lambda row: bool(row["Wrong_Accepted_Candidate"])
        and not (
            bool(row["Wrong_Accepted_Baseline"])
            and _text(row["Accepted_Roll_Baseline"])
            == _text(row["Accepted_Roll_Candidate"])
        ),
        axis=1,
    )
    joined["New_Outsider_Absorption"] = (
        joined["Outsider_Absorption_Candidate"].map(bool)
        & ~joined["Outsider_Absorption_Baseline"].map(bool)
    )
    joined["New_Mixed_Acceptance"] = (
        joined["Mixed_Acceptance_Candidate"].map(bool)
        & ~joined["Mixed_Acceptance_Baseline"].map(bool)
    )
    return joined.sort_values("Benchmark_Row_ID", kind="stable").reset_index(drop=True)


def _confusion_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    source = frame.copy()
    source["Actual_Label"] = source.apply(
        lambda row: _text(row["Actual_Roll"])
        if row["Actual_Class"] == "known_student"
        else f"<{_text(row['Actual_Class']).upper()}>",
        axis=1,
    )
    source["Predicted_Label"] = source.apply(
        lambda row: _text(row["Accepted_Roll"])
        if bool(row["Accepted"])
        else "<UNRESOLVED>",
        axis=1,
    )
    return (
        source.groupby(["Actual_Label", "Predicted_Label"], sort=True)
        .size()
        .rename("Track_Count")
        .reset_index()
    )


def _grouped_metric_rows(
    predictions: dict[str, pd.DataFrame], group_column: str, group_type: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_id, frame in predictions.items():
        for key, group in frame.groupby(group_column, sort=True):
            metrics = summarize_predictions(group)
            rows.append(
                {
                    "Model_ID": model_id,
                    "Group_Type": group_type,
                    "Group_Value": _text(key),
                    **{
                        name: value
                        for name, value in metrics.items()
                        if not isinstance(value, dict)
                    },
                }
            )
    return rows


def apply_family_safety_gates(
    *,
    baseline: pd.DataFrame,
    variant_predictions: dict[str, pd.DataFrame],
    variant_manifests: dict[str, dict[str, Any]],
    protected_comparison: dict[str, Any],
    production_roster_coverage: int,
    benchmark_expected_rows: int = 120,
) -> tuple[dict[str, Any], pd.DataFrame]:
    safe_specs = {
        "A": {TUE_P1_SESSION, TUE_P2_SESSION},
        "B": {TUE_P1_SESSION},
        "C": {TUE_P2_SESSION},
    }
    comparison_frames: list[pd.DataFrame] = []
    comparison_metrics: dict[str, Any] = {}
    false_identity_events = 0
    retention_comparisons: dict[str, Any] = {}
    leakage_failures: list[str] = []
    for key, sessions in safe_specs.items():
        candidate = variant_predictions[key]
        baseline_subset = baseline[baseline["Session_ID"].isin(sessions)]
        comparison = compare_prediction_frames(
            baseline_subset, candidate, f"production_vs_variant_{key.lower()}"
        )
        comparison_frames.append(comparison)
        metrics = {
            "rows": len(comparison),
            "new_wrong_accepted": int(comparison["New_Wrong_Accepted"].map(bool).sum()),
            "new_outsider_absorption": int(
                comparison["New_Outsider_Absorption"].map(bool).sum()
            ),
            "new_mixed_acceptance": int(
                comparison["New_Mixed_Acceptance"].map(bool).sum()
            ),
            "correct_to_wrong": int(
                comparison["Correct_To_Wrong_Regression"].map(bool).sum()
            ),
            "candidate_outsider_absorption": int(
                comparison["Outsider_Absorption_Candidate"].map(bool).sum()
            ),
            "candidate_mixed_acceptance": int(
                comparison["Mixed_Acceptance_Candidate"].map(bool).sum()
            ),
            "lost_correct_acceptance": int(
                comparison["Lost_Correct_Acceptance"].map(bool).sum()
            ),
            "recovered_correct_acceptance": int(
                comparison["Recovered_Correct_Acceptance"].map(bool).sum()
            ),
        }
        unsafe_event_mask = (
            comparison["New_Wrong_Accepted"].map(bool)
            | comparison["New_Outsider_Absorption"].map(bool)
            | comparison["New_Mixed_Acceptance"].map(bool)
        )
        metrics["unique_false_identity_events"] = int(unsafe_event_mask.sum())
        comparison_metrics[key] = metrics
        false_identity_events += metrics["unique_false_identity_events"]

        if key in {"A", "B"}:
            accepted_review = comparison[
                comparison["Source_Kind_Baseline"].eq(
                    "tue_p1_accepted_review"
                )
            ].copy()
            baseline_correct = accepted_review[
                accepted_review["Correct_Accepted_Baseline"].map(bool)
            ]
            lost = baseline_correct[
                ~baseline_correct["Correct_Accepted_Candidate"].map(bool)
            ]
            retention_comparisons[key] = {
                "reviewed_rows": len(accepted_review),
                "baseline_correct_acceptances": len(baseline_correct),
                "retained_correct_acceptances": len(baseline_correct) - len(lost),
                "lost_correct_acceptances": len(lost),
                "lost_benchmark_row_ids": sorted(
                    lost["Benchmark_Row_ID"].map(_text).unique()
                ),
                "retention_pct": (
                    round(100.0 * (len(baseline_correct) - len(lost)) / len(baseline_correct), 3)
                    if len(baseline_correct)
                    else None
                ),
                "decision_role": (
                    "diagnostic_cleaned_enrollment" if key == "A" else "decisive_cross_session"
                ),
            }

    for key, manifest in variant_manifests.items():
        included = set(manifest.get("source_sessions_included") or [])
        allowed = set(manifest.get("evaluation_sessions_allowed") or [])
        if key == "B" and TUE_P1_SESSION in included:
            leakage_failures.append("Variant B includes TUE_P1 adaptation crops")
        if key == "C" and TUE_P2_SESSION in included:
            leakage_failures.append("Variant C includes TUE_P2 adaptation crops")
        if included & allowed:
            leakage_failures.append(
                f"Variant {key} evaluates a session used for its CCTV adaptation"
            )
        if key == "D" and manifest.get("training_contaminated_descriptive_only") is not True:
            leakage_failures.append("Variant D is not marked descriptive and contaminated")

    feature_rows = int(baseline["Feature_Available"].map(bool).sum())
    candidate_coverage = int(
        dict(variant_manifests["A"].get("canonical_identity_coverage") or {}).get(
            "covered", 0
        )
    )
    integrity_passed = bool(protected_comparison.get("unchanged"))
    approval_integrity = True
    correction_integrity = True
    leakage_passed = not leakage_failures
    false_identity_passed = false_identity_events == 0
    decisive_retention = retention_comparisons.get("B", {})
    lost_correct_acceptances = int(
        decisive_retention.get("lost_correct_acceptances", 0)
    )
    retention_passed = lost_correct_acceptances == 0
    coverage_passed = candidate_coverage >= production_roster_coverage
    evidence_sufficient = feature_rows == benchmark_expected_rows

    gates = {
        "production_integrity": {
            "passed": integrity_passed,
            "details": protected_comparison,
        },
        "approval_integrity": {"passed": approval_integrity},
        "identity_correction_integrity": {
            "passed": correction_integrity,
            "global_alias_created": False,
        },
        "reviewed_accepted_track_retention": {
            "passed": retention_passed,
            "target_retention_pct": 100.0,
            "population": "37 human-reviewed TUE_P1 baseline accepted tracks",
            "decisive_variant": "B",
            "lost_correct_acceptances": lost_correct_acceptances,
            "comparisons": retention_comparisons,
        },
        "zero_new_false_accepts": {
            "passed": false_identity_passed,
            "false_identity_events": false_identity_events,
            "comparisons": comparison_metrics,
        },
        "candidate_coverage": {
            "passed": coverage_passed,
            "production_cvo_coverage": production_roster_coverage,
            "candidate_cvo_coverage": candidate_coverage,
            "missing_0268_allowed_but_explicit": True,
        },
        "leakage": {
            "passed": leakage_passed,
            "failures": leakage_failures,
            "variant_d_promotion_evidence": False,
            "mon_p3_accessed": False,
        },
        "promotion": {
            "passed": True,
            "production_approved": False,
            "candidate_promoted": False,
            "promotion_possible_in_phase": False,
        },
        "evidence_sufficiency": {
            "passed": evidence_sufficient,
            "expected_rows": benchmark_expected_rows,
            "evaluable_rows": feature_rows,
            "evidence_mode": "reviewer_contact_sheet_reextraction",
        },
    }
    if not integrity_passed or not approval_integrity or not correction_integrity or not leakage_passed:
        decision = "regression_failed_integrity"
        next_step = "Correct the integrity or leakage blocker and rebuild a new immutable family."
    elif not false_identity_passed:
        decision = "regression_failed_false_identity"
        next_step = (
            "Archive this family and investigate the listed false-identity rows and their approved "
            "sources before any MON_P3 run; do not tune thresholds against this benchmark."
        )
    elif not evidence_sufficient:
        decision = "built_unapproved_insufficient_evidence"
        next_step = (
            "Repair only the stored-evidence feature gap, then rescore the same frozen benchmark "
            "without rebuilding or processing MON_P3."
        )
    elif not retention_passed or not coverage_passed:
        decision = "hold_coverage_or_retention_regression"
        next_step = (
            "Review the lost correct acceptances and source coverage before deciding whether a "
            "new candidate family is justified."
        )
    else:
        decision = "built_unapproved_pending_mon_p3"
        next_step = "PHASE 1.2J - MON_P3 untouched shadow validation."
    result = {
        "schema_version": FAMILY_SCHEMA_VERSION,
        "decision": decision,
        "status": decision,
        "production_approved": False,
        "candidate_promoted": False,
        "all_variants_built_unapproved": True,
        "match_threshold": OFFICIAL_MATCH_THRESHOLD,
        "margin_threshold": OFFICIAL_MARGIN_THRESHOLD,
        "aggregate_mode": OFFICIAL_AGGREGATE,
        "attendance_checkpoints": OFFICIAL_ATTENDANCE_CHECKPOINTS,
        "gates": gates,
        "exact_next_step": next_step,
        "mon_p3_processed": False,
        "rejected_candidate_remains_rejected": REJECTED_CANDIDATE_ID,
    }
    comparisons = pd.concat(comparison_frames, ignore_index=True) if comparison_frames else pd.DataFrame()
    return result, comparisons


def _write_output_manifest(
    root: Path,
    *,
    filename: str,
    declared_output_dir: Path,
    extra: dict[str, Any],
) -> Path:
    root = Path(root)
    manifest_path = root / filename
    files = {
        path.relative_to(root).as_posix(): _sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != manifest_path
    }
    payload = {
        "schema_version": FAMILY_SCHEMA_VERSION,
        "created_at": _now(),
        "output_dir": str(Path(declared_output_dir).resolve()),
        "files_sha256": files,
        **json_safe(extra),
    }
    _write_json(manifest_path, payload)
    return manifest_path


def _coverage_from_records(
    records: Sequence[dict[str, Any]], roster_rolls: set[str]
) -> tuple[int, list[str], dict[str, int]]:
    counts = Counter(_canonical(record.get("roll_no")) for record in records)
    covered = sorted(roll for roll in roster_rolls if counts.get(roll, 0) > 0)
    return len(covered), sorted(roster_rolls - set(covered)), dict(sorted(counts.items()))


def _per_student_comparison(comparisons: pd.DataFrame) -> pd.DataFrame:
    known = comparisons[
        comparisons["Actual_Class_Baseline"].eq("known_student")
    ].copy()
    rows: list[dict[str, Any]] = []
    for (comparison_id, actual_roll), group in known.groupby(
        ["Comparison_ID", "Actual_Roll_Baseline"], sort=True
    ):
        rows.append(
            {
                "Comparison_ID": comparison_id,
                "Actual_Roll": actual_roll,
                "Tracks": len(group),
                "Baseline_Correct_Accepted": int(
                    group["Correct_Accepted_Baseline"].map(bool).sum()
                ),
                "Candidate_Correct_Accepted": int(
                    group["Correct_Accepted_Candidate"].map(bool).sum()
                ),
                "Correct_Acceptance_Delta": int(
                    group["Correct_Accepted_Candidate"].map(bool).sum()
                    - group["Correct_Accepted_Baseline"].map(bool).sum()
                ),
                "Baseline_Wrong_Accepted": int(
                    group["Wrong_Accepted_Baseline"].map(bool).sum()
                ),
                "Candidate_Wrong_Accepted": int(
                    group["Wrong_Accepted_Candidate"].map(bool).sum()
                ),
                "New_Wrong_Accepted": int(
                    group["New_Wrong_Accepted"].map(bool).sum()
                ),
                "Lost_Correct_Acceptance": int(
                    group["Lost_Correct_Acceptance"].map(bool).sum()
                ),
                "Recovered_Correct_Acceptance": int(
                    group["Recovered_Correct_Acceptance"].map(bool).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _attractor_comparison(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    all_rolls = sorted(
        {
            _text(value)
            for frame in predictions.values()
            for value in frame.loc[frame["Accepted"].map(bool), "Accepted_Roll"]
            if _text(value)
        }
    )
    for model_id, frame in predictions.items():
        for roll in all_rolls:
            attracted = frame[
                frame["Accepted"].map(bool) & frame["Accepted_Roll"].eq(roll)
            ]
            rows.append(
                {
                    "Model_ID": model_id,
                    "Attractor_Roll": roll,
                    "Accepted_Tracks": len(attracted),
                    "Correct_Accepted": int(
                        attracted["Correct_Accepted"].map(bool).sum()
                    ),
                    "Wrong_Accepted": int(
                        attracted["Wrong_Accepted"].map(bool).sum()
                    ),
                    "Outsider_Absorption": int(
                        attracted["Outsider_Absorption"].map(bool).sum()
                    ),
                    "Mixed_Acceptance": int(
                        attracted["Mixed_Acceptance"].map(bool).sum()
                    ),
                    "Dominant_Share_Pct": round(
                        100.0 * len(attracted) / max(1, int(frame["Accepted"].map(bool).sum())),
                        3,
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_family_evaluation(
    *,
    temporary_family_dir: Path,
    final_family_dir: Path,
    family_id: str,
    forensic_run_dir: Path,
    student_map_path: Path,
    production_embeddings_path: Path,
    variant_manifests: dict[str, dict[str, Any]],
    variant_directories: dict[str, Path],
    engine: FaceEngine,
    roster_rolls: set[str],
    production_roster_coverage: int,
    protected_comparison: dict[str, Any],
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    evaluation_dir = Path(temporary_family_dir) / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=False)
    benchmark, benchmark_manifest = load_frozen_benchmark_evidence(
        forensic_run_dir=forensic_run_dir,
        student_map_path=student_map_path,
    )
    features, feature_audit = extract_benchmark_contact_sheet_features(
        benchmark, engine=engine, progress=progress
    )
    _write_csv(evaluation_dir / "benchmark_feature_audit.csv", feature_audit)
    feature_cache_payload = {
        "schema_version": FAMILY_SCHEMA_VERSION,
        "benchmark_id": benchmark_manifest["benchmark_id"],
        "benchmark_manifest_sha256": benchmark_manifest[
            "benchmark_manifest_sha256"
        ],
        "evidence_mode": "reviewer_contact_sheet_reextraction",
        "model_hashes": {
            "yunet": _sha256_file(YUNET_MODEL),
            "sface": _sha256_file(SFACE_MODEL),
        },
        "features": [
            {
                "benchmark_row_id": row_id,
                "embedding": features[row_id],
            }
            for row_id in sorted(features)
        ],
    }
    with (evaluation_dir / "benchmark_feature_cache.pkl").open("wb") as file_obj:
        pickle.dump(feature_cache_payload, file_obj)
    _write_json(
        evaluation_dir / "benchmark_feature_manifest.json",
        {
            **benchmark_manifest,
            "feature_rows": len(features),
            "unavailable_rows": len(benchmark) - len(features),
            "feature_dimension": (
                int(next(iter(features.values())).size) if features else 0
            ),
            "cache_sha256": _sha256_file(
                evaluation_dir / "benchmark_feature_cache.pkl"
            ),
            "stored_observation_embeddings_available": False,
            "fallback_reason": (
                "Frozen diagnostics contain scores and contact-sheet evidence but no compatible "
                "observation embedding vectors."
            ),
            "video_decode_rerun": False,
        },
    )

    production_db = StudentEmbeddingDB.load(production_embeddings_path)
    baseline = score_benchmark(
        benchmark,
        features=features,
        database=production_db,
        roster_rolls=roster_rolls,
        model_id="production_baseline",
    )
    variant_predictions: dict[str, pd.DataFrame] = {}
    definitions = {definition.key: definition for definition in VARIANT_DEFINITIONS}
    for key in ("A", "B", "C", "D"):
        definition = definitions[key]
        database = StudentEmbeddingDB.load(
            variant_directories[key] / "student_embeddings.pkl"
        )
        variant_predictions[key] = score_benchmark(
            benchmark,
            features=features,
            database=database,
            roster_rolls=roster_rolls,
            model_id=variant_manifests[key]["version_id"],
            allowed_sessions=definition.evaluation_sessions,
            training_contaminated_descriptive=definition.descriptive_only,
        )

    prediction_files = {
        "production_baseline_predictions.csv": baseline,
        "cleaned_enrollment_predictions.csv": variant_predictions["A"],
        "tue_p1_cross_session_predictions.csv": variant_predictions["B"],
        "tue_p2_cross_session_predictions.csv": variant_predictions["C"],
        "full_candidate_descriptive_predictions.csv": variant_predictions["D"],
    }
    for filename, frame in prediction_files.items():
        _write_csv(evaluation_dir / filename, frame)

    decision, comparisons = apply_family_safety_gates(
        baseline=baseline,
        variant_predictions=variant_predictions,
        variant_manifests=variant_manifests,
        protected_comparison=protected_comparison,
        production_roster_coverage=production_roster_coverage,
    )
    decision["family_id"] = family_id
    decision["evaluated_at"] = _now()
    _write_csv(evaluation_dir / "baseline_vs_candidate_tracks.csv", comparisons)
    student_comparison = _per_student_comparison(comparisons)
    _write_csv(
        evaluation_dir / "baseline_vs_candidate_students.csv", student_comparison
    )

    safe_combined = pd.concat(
        [variant_predictions["B"], variant_predictions["C"]], ignore_index=True
    ).sort_values("Benchmark_Row_ID", kind="stable")
    if safe_combined["Benchmark_Row_ID"].duplicated().any() or len(safe_combined) != len(
        benchmark
    ):
        raise EmbeddingFamilyError(
            "Cross-session predictions do not form one leakage-safe result per benchmark row"
        )
    _write_csv(
        evaluation_dir / "confusion_matrix_production.csv", _confusion_matrix(baseline)
    )
    _write_csv(
        evaluation_dir / "confusion_matrix_candidate.csv",
        _confusion_matrix(safe_combined),
    )
    _write_csv(
        evaluation_dir / "confusion_matrix_cleaned_enrollment.csv",
        _confusion_matrix(variant_predictions["A"]),
    )

    predictions_for_summary = {
        "production_baseline": baseline,
        "variant_a_cleaned_enrollment": variant_predictions["A"],
        "variant_b_tue_p1_cross_session": variant_predictions["B"],
        "variant_c_tue_p2_cross_session": variant_predictions["C"],
        "variant_d_full_contaminated_descriptive": variant_predictions["D"],
        "leakage_safe_cross_session_combined": safe_combined,
    }
    _write_csv(
        evaluation_dir / "identity_attractor_comparison.csv",
        _attractor_comparison(predictions_for_summary),
    )
    session_rows = _grouped_metric_rows(
        predictions_for_summary, "Session_ID", "session"
    )
    checkpoint_rows = _grouped_metric_rows(
        predictions_for_summary, "Checkpoint_ID", "checkpoint"
    )
    camera_rows = _grouped_metric_rows(
        predictions_for_summary, "Camera_ID", "camera"
    )
    identity_frames = {
        name: frame[frame["Actual_Class"].eq("known_student")]
        for name, frame in predictions_for_summary.items()
    }
    identity_rows = _grouped_metric_rows(identity_frames, "Actual_Roll", "identity")
    _write_csv(
        evaluation_dir / "session_evaluation_summary.csv", pd.DataFrame(session_rows)
    )
    _write_csv(
        evaluation_dir / "checkpoint_evaluation_summary.csv",
        pd.DataFrame(checkpoint_rows),
    )
    _write_csv(
        evaluation_dir / "camera_evaluation_summary.csv", pd.DataFrame(camera_rows)
    )
    _write_csv(
        evaluation_dir / "identity_evaluation_summary.csv", pd.DataFrame(identity_rows)
    )

    production_canonical_rolls = {
        _canonical(roll) for roll in production_db.roll_numbers if _canonical(roll)
    }
    coverage_rows = [
        {
            "Model_ID": "production_baseline",
            "CVO_Covered": production_roster_coverage,
            "CVO_Roster": len(roster_rolls),
            "Missing_Rolls": ";".join(
                sorted(roster_rolls - production_canonical_rolls)
            ),
            "Embedding_Records": len(production_db.records),
            "Production_Approved": True,
        }
    ]
    for key in ("A", "B", "C", "D"):
        manifest = variant_manifests[key]
        coverage = dict(manifest["canonical_identity_coverage"])
        coverage_rows.append(
            {
                "Model_ID": manifest["version_id"],
                "CVO_Covered": int(coverage["covered"]),
                "CVO_Roster": int(coverage["roster"]),
                "Missing_Rolls": ";".join(manifest["missing_students"]),
                "Embedding_Records": int(manifest["embedding_records"]),
                "Production_Approved": False,
            }
        )
    coverage_frame = pd.DataFrame(coverage_rows)
    _write_csv(evaluation_dir / "coverage_comparison.csv", coverage_frame)

    metrics = {
        name: summarize_predictions(frame)
        for name, frame in predictions_for_summary.items()
    }
    score_rows: list[dict[str, Any]] = []
    for name, values in metrics.items():
        score_rows.append(
            {
                "Model_ID": name,
                **{
                    f"Top1_Score_{key}": value
                    for key, value in values.get("score_distribution", {}).items()
                },
                **{
                    f"Margin_{key}": value
                    for key, value in values.get("margin_distribution", {}).items()
                },
            }
        )
    _write_csv(
        evaluation_dir / "score_margin_distributions.csv", pd.DataFrame(score_rows)
    )
    regression_summary = {
        "schema_version": FAMILY_SCHEMA_VERSION,
        "family_id": family_id,
        "benchmark": benchmark_manifest,
        "thresholds": {
            "match": OFFICIAL_MATCH_THRESHOLD,
            "margin": OFFICIAL_MARGIN_THRESHOLD,
            "aggregate": OFFICIAL_AGGREGATE,
            "attendance_checkpoints": OFFICIAL_ATTENDANCE_CHECKPOINTS,
        },
        "metrics": metrics,
        "safe_comparison_metrics": decision["gates"]["zero_new_false_accepts"],
        "decision": decision["decision"],
        "production_approved": False,
        "candidate_promoted": False,
        "variant_d_label": "training-contaminated descriptive result",
        "video_recognition_rerun": False,
        "mon_p3_processed": False,
    }
    _write_json(evaluation_dir / "regression_summary.json", regression_summary)
    _write_json(evaluation_dir / "evaluation_decision.json", decision)

    table_rows = []
    report_order = [
        "production_baseline",
        "variant_a_cleaned_enrollment",
        "variant_b_tue_p1_cross_session",
        "variant_c_tue_p2_cross_session",
        "variant_d_full_contaminated_descriptive",
    ]
    for name in report_order:
        value = metrics[name]
        table_rows.append(
            f"| {name} | {value['rows']} | {value['accepted']} | "
            f"{value['correct_accepted']} | {value['wrong_accepted']} | "
            f"{value['outsider_absorption']} | {value['mixed_acceptance']} |"
        )
    (evaluation_dir / "regression_report.md").write_text(
        "\n".join(
            [
                f"# Phase 1.2I-B Regression - {family_id}",
                "",
                f"Decision: `{decision['decision']}`",
                "",
                "The benchmark was rescored from the 120 frozen, hash-verified reviewer contact "
                "sheets because compatible observation embedding vectors were not stored. No "
                "video was decoded and no attendance output was written.",
                "",
                "| Model | Rows | Accepted | Correct | Wrong | Outsider | Mixed |",
                "|---|---:|---:|---:|---:|---:|---:|",
                *table_rows,
                "",
                "Variant D is a **training-contaminated descriptive result**. It is not promotion "
                "evidence. Variants B and C are evaluated only on the opposite CCTV session.",
                "",
                "## Safety status",
                "",
                f"- Production integrity: {'PASS' if decision['gates']['production_integrity']['passed'] else 'FAIL'}",
                f"- False-identity gate: {'PASS' if decision['gates']['zero_new_false_accepts']['passed'] else 'FAIL'}",
                f"- Leakage gate: {'PASS' if decision['gates']['leakage']['passed'] else 'FAIL'}",
                "- Candidate approved: no",
                "- Candidate promoted: no",
                "- MON_P3 processed: no",
                "",
                f"Next step: {decision['exact_next_step']}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    evaluation_manifest_path = _write_output_manifest(
        evaluation_dir,
        filename="evaluation_output_manifest.json",
        declared_output_dir=Path(final_family_dir) / "evaluation",
        extra={
            "family_id": family_id,
            "decision": decision["decision"],
            "production_approved": False,
            "candidate_promoted": False,
            "variant_d_promotion_evidence": False,
            "mon_p3_processed": False,
        },
    )
    try:
        verify_output_manifest(evaluation_dir, evaluation_manifest_path)
    except ShadowValidationError as exc:
        raise EmbeddingFamilyError(str(exc)) from exc
    return decision, regression_summary


def _approval_output_frame(frame: pd.DataFrame, *, enrollment: bool) -> pd.DataFrame:
    output = frame.copy()
    output["Private_Provenance_JSON"] = output["Private_Provenance"].map(
        lambda value: json.dumps(json_safe(value), sort_keys=True, ensure_ascii=True)
    )
    output = output.drop(columns=["Private_Provenance"], errors="ignore")
    preferred = (
        [
            "Package_ID",
            "Item_ID",
            "Review_Action",
            "Duplicate_Of_Item_ID_Raw",
            "Duplicate_Of_Item_ID",
            "Duplicate_Target_Source_SHA256",
            "Reviewer_Notes",
            "Expected_Roll",
            "Candidate_Roll",
            "Source_Path",
            "Resolved_Source_Path",
            "Source_SHA256",
            "Source_Integrity_Status",
            "Private_Provenance_JSON",
        ]
        if enrollment
        else [
            "Package_ID",
            "Item_ID",
            "Review_Action",
            "Reviewer_Notes",
            "Actual_Roll",
            "Candidate_Roll",
            "Source_Session",
            "Source_Path",
            "Resolved_Source_Path",
            "Source_SHA256",
            "Source_Integrity_Status",
            "Private_Provenance_JSON",
        ]
    )
    columns = [column for column in preferred if column in output.columns]
    return output[columns].sort_values("Item_ID", kind="stable").reset_index(drop=True)


def _source_reconciliation_frame(
    *,
    policy: SourcePolicyResult,
    approval_bundle: CompactApprovalBundle,
    dataset_root: Path,
    shruthi: dict[str, Any],
) -> pd.DataFrame:
    active = policy.sources.copy()
    active["Reconciliation_Type"] = "configured_dataset_source"
    active["Active_Configured_Root"] = True
    active["Resolved_Status"] = active["Source_SHA256"].map(lambda _value: "verified")
    active["Special_Note"] = active.apply(
        lambda row: "resolved_shruthi_renamed_folder"
        if _text(row["Source_SHA256"]) == _text(shruthi.get("sha256"))
        else "",
        axis=1,
    )
    inactive_rows: list[dict[str, Any]] = []
    dataset_root = Path(dataset_root).resolve()
    active_hashes = set(active["Source_SHA256"].map(_text))
    for row in approval_bundle.enrollment.to_dict("records"):
        resolved = Path(_text(row["Resolved_Source_Path"])).resolve()
        try:
            resolved.relative_to(dataset_root)
            is_active_root = True
        except ValueError:
            is_active_root = False
        if is_active_root or _text(row["Source_SHA256"]) in active_hashes:
            continue
        inactive_rows.append(
            {
                "Source_ID": f"INACTIVE-{_text(row['Item_ID'])}",
                "Source_Role": "configured_augmented_dataset_nonproduction",
                "Source_Path": str(resolved),
                "Relative_Path": resolved.name,
                "Source_SHA256": _text(row["Source_SHA256"]),
                "File_Size_Bytes": int(resolved.stat().st_size),
                "Original_Folder": resolved.parent.name,
                "Canonical_Owner": _canonical(row.get("Candidate_Roll")),
                "Candidate_Owner": _canonical(row.get("Candidate_Roll")),
                "Review_Item_ID": _text(row["Item_ID"]),
                "Review_Action": _text(row["Review_Action"]),
                "Policy_Eligible": False,
                "Policy_Reason": "nonproduction_augmented_root_not_activated",
                "Selected_For_Candidate": False,
                "Selection_Reason": "nonproduction_augmented_root_not_activated",
                "Reconciliation_Type": "reviewed_nonproduction_source",
                "Active_Configured_Root": False,
                "Resolved_Status": _text(row["Source_Integrity_Status"]),
                "Special_Note": "",
            }
        )
    if inactive_rows:
        return pd.concat([active, pd.DataFrame(inactive_rows)], ignore_index=True).sort_values(
            ["Active_Configured_Root", "Candidate_Owner", "Source_Path"],
            ascending=[False, True, True],
            kind="stable",
        )
    return active.sort_values(
        ["Candidate_Owner", "Source_Path"], kind="stable"
    ).reset_index(drop=True)


def _student_source_coverage(
    *,
    roster_rolls: set[str],
    policy: SourcePolicyResult,
    cctv_staging: pd.DataFrame,
    variant_manifests: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    selected = policy.sources[policy.sources["Selected_For_Candidate"].map(bool)]
    rows: list[dict[str, Any]] = []
    for roll in sorted(roster_rolls):
        source_group = policy.sources[policy.sources["Candidate_Owner"].eq(roll)]
        selected_group = selected[selected["Candidate_Owner"].eq(roll)]
        cctv_group = cctv_staging[cctv_staging["Candidate_Roll"].eq(roll)]
        row = {
            "Canonical_Roll": roll,
            "Configured_Dataset_Images": len(source_group),
            "Selected_Enrollment_Sources": len(selected_group),
            "Enrollment_Source_Groups": int(selected_group["Source_Group_ID"].nunique()),
            "Approved_CCTV_TUE_P1": int(
                cctv_group["Source_Session"].eq(TUE_P1_SESSION).sum()
            ),
            "Approved_CCTV_TUE_P2": int(
                cctv_group["Source_Session"].eq(TUE_P2_SESSION).sum()
            ),
            "AI0110_Corrected_Coverage": roll == TARGET_IDENTITY and len(selected_group) > 0,
            "Dataset_Missing": len(selected_group) == 0,
            "Missing_Status": (
                "Dataset Missing"
                if roll == MISSING_CVO_IDENTITY and len(selected_group) == 0
                else ("Covered" if len(selected_group) else "No Candidate Embedding")
            ),
        }
        if variant_manifests:
            for key, manifest in variant_manifests.items():
                row[f"Variant_{key}_Embeddings"] = int(
                    dict(manifest.get("per_student_embedding_counts") or {}).get(roll, 0)
                )
        rows.append(row)
    return pd.DataFrame(rows)


def _ensure_production_preflight(
    *,
    production_embeddings_path: Path,
    production_summary_path: Path,
    rejected_registry_path: Path,
    expected_embeddings_sha256: str,
    expected_summary_sha256: str,
) -> dict[str, Any]:
    production_embeddings_path = Path(production_embeddings_path).resolve()
    production_summary_path = Path(production_summary_path).resolve()
    embeddings_hash = _sha256_file(production_embeddings_path)
    summary_hash = _sha256_file(production_summary_path)
    if expected_embeddings_sha256 and embeddings_hash != expected_embeddings_sha256:
        raise EmbeddingFamilyError(
            "Production embedding hash changed without a validated later lifecycle artifact: "
            f"expected {expected_embeddings_sha256}, found {embeddings_hash}"
        )
    if expected_summary_sha256 and summary_hash != expected_summary_sha256:
        raise EmbeddingFamilyError(
            "Production embedding summary hash changed without a validated later lifecycle artifact: "
            f"expected {expected_summary_sha256}, found {summary_hash}"
        )
    rejected_registry_path = Path(rejected_registry_path).resolve()
    if not rejected_registry_path.is_file():
        raise EmbeddingFamilyError(
            f"Rejected candidate registry record not found: {rejected_registry_path}"
        )
    rejected = json.loads(rejected_registry_path.read_text(encoding="utf-8"))
    if (
        _text(rejected.get("candidate_id")) != REJECTED_CANDIDATE_ID
        or rejected.get("production_approved") is not False
        or rejected.get("enabled") is not False
        or rejected.get("immutable") is not True
        or _text(rejected.get("final_decision")) != "reject_candidate"
    ):
        raise EmbeddingFamilyError(
            f"Rejected candidate registry state is unsafe for {REJECTED_CANDIDATE_ID}"
        )
    return {
        "production_embeddings": {
            "path": str(production_embeddings_path),
            "sha256": embeddings_hash,
        },
        "production_summary": {
            "path": str(production_summary_path),
            "sha256": summary_hash,
        },
        "rejected_candidate_registry": {
            "path": str(rejected_registry_path),
            "sha256": _sha256_file(rejected_registry_path),
            "candidate_id": REJECTED_CANDIDATE_ID,
            "status": _text(rejected.get("status")),
            "production_approved": False,
            "enabled": False,
        },
    }


def build_and_evaluate_embedding_family(
    *,
    repo_root: Path,
    forensic_run_dir: Path,
    enrollment_review_package: Path,
    enrollment_approvals_path: Path,
    cctv_review_package: Path,
    cctv_approvals_path: Path,
    production_embeddings_path: Path,
    production_summary_path: Path,
    student_map_path: Path,
    dataset_root: Path,
    augmented_dataset_root: Path,
    versions_root: Path,
    historical_inventory_path: Path | None = None,
    embedding_outliers_path: Path | None = None,
    yunet_model: Path = YUNET_MODEL,
    sface_model: Path = SFACE_MODEL,
    family_id: str = "",
    contract: CompactApprovalContract = CompactApprovalContract(),
    expected_production_embeddings_sha256: str = EXPECTED_PRODUCTION_EMBEDDINGS_SHA256,
    expected_production_summary_sha256: str = EXPECTED_PRODUCTION_SUMMARY_SHA256,
    expected_historical_images: int | None = 16,
    expected_approved_cctv: int = 35,
    require_parent_manifest: bool = True,
    launcher_source: Path | None = None,
    progress: Callable[[str], None] | None = print,
    engine_factory: Callable[[Path, Path, float], FaceEngine] | None = None,
) -> FamilyBuildResult:
    started = time.perf_counter()
    repo_root = Path(repo_root).resolve()
    forensic_run_dir = Path(forensic_run_dir).resolve()
    production_embeddings_path = Path(production_embeddings_path).resolve()
    production_summary_path = Path(production_summary_path).resolve()
    student_map_path = Path(student_map_path).resolve()
    dataset_root = Path(dataset_root).resolve()
    augmented_dataset_root = Path(augmented_dataset_root).resolve()
    versions_root = Path(versions_root).resolve()
    historical_inventory_path = Path(
        historical_inventory_path
        or forensic_run_dir / "dataset_image_inventory.csv"
    ).resolve()
    embedding_outliers_path = Path(
        embedding_outliers_path or forensic_run_dir / "embedding_outliers.csv"
    ).resolve()
    yunet_model = Path(yunet_model).resolve()
    sface_model = Path(sface_model).resolve()
    rejected_registry_path = (
        repo_root
        / "models"
        / "candidate_registry"
        / "rejected"
        / f"{REJECTED_CANDIDATE_ID}.json"
    )
    production_preflight = _ensure_production_preflight(
        production_embeddings_path=production_embeddings_path,
        production_summary_path=production_summary_path,
        rejected_registry_path=rejected_registry_path,
        expected_embeddings_sha256=expected_production_embeddings_sha256,
        expected_summary_sha256=expected_production_summary_sha256,
    )
    if progress:
        progress("Approval and package integrity validation")
    approvals = validate_compact_approval_bundle(
        enrollment_review_package=enrollment_review_package,
        enrollment_approvals_path=enrollment_approvals_path,
        cctv_review_package=cctv_review_package,
        cctv_approvals_path=cctv_approvals_path,
        dataset_root=dataset_root,
        contract=contract,
        require_parent_manifest=require_parent_manifest,
    )
    if progress:
        progress("Narrow CSE0110 to AI0110 identity reconciliation")
    correction = reconcile_identity_correction(
        repo_root=repo_root,
        dataset_root=dataset_root,
        historical_inventory_path=historical_inventory_path,
        production_embeddings_path=production_embeddings_path,
        student_map_path=student_map_path,
        approval_bundle=approvals,
        source_roll=contract.source_roll,
        target_roll=contract.target_roll,
        expected_historical_images=expected_historical_images,
    )
    shruthi = reconcile_shruthi_source(
        dataset_roots=[dataset_root, augmented_dataset_root],
        embedding_outliers_path=embedding_outliers_path,
        historical_inventory_path=historical_inventory_path,
    )
    roster, _, subject_rolls_raw = load_student_mapping(student_map_path, "CVO")
    roster_rolls = {_canonical(value) for value in subject_rolls_raw if _canonical(value)}
    if len(roster_rolls) != 27:
        raise EmbeddingFamilyError(
            f"Authoritative CVO roster must contain 27 students, found {len(roster_rolls)}"
        )
    if progress:
        progress("Protected-state snapshot and candidate source discovery")
    protected_before = snapshot_phase_i_b_protected_state(
        repo_root=repo_root,
        production_embeddings_path=production_embeddings_path,
        production_summary_path=production_summary_path,
        student_map_path=student_map_path,
        enrollment_review_package=enrollment_review_package,
        enrollment_approvals_path=enrollment_approvals_path,
        cctv_review_package=cctv_review_package,
        cctv_approvals_path=cctv_approvals_path,
        forensic_run_dir=forensic_run_dir,
        dataset_root=dataset_root,
        augmented_dataset_root=augmented_dataset_root,
        yunet_model=yunet_model,
        sface_model=sface_model,
    )
    discovered_sources = discover_enrollment_sources(
        dataset_root=dataset_root,
        approval_bundle=approvals,
        correction=correction,
    )
    input_fingerprint = {
        "policy_version": POLICY_VERSION,
        "production_embeddings_sha256": production_preflight["production_embeddings"][
            "sha256"
        ],
        "production_summary_sha256": production_preflight["production_summary"][
            "sha256"
        ],
        "student_map_sha256": _sha256_file(student_map_path),
        "yunet_model_sha256": _sha256_file(yunet_model),
        "sface_model_sha256": _sha256_file(sface_model),
        "enrollment_approval_sha256": _sha256_file(enrollment_approvals_path),
        "cctv_approval_sha256": _sha256_file(cctv_approvals_path),
        "enrollment_package_manifest_sha256": _sha256_file(
            Path(enrollment_review_package) / "package_manifest.json"
        ),
        "cctv_package_manifest_sha256": _sha256_file(
            Path(cctv_review_package) / "package_manifest.json"
        ),
        "benchmark_manifest_sha256": _sha256_file(
            forensic_run_dir / "multisession_ground_truth_manifest.json"
        ),
        "historical_inventory_sha256": _sha256_file(historical_inventory_path),
        "embedding_outliers_sha256": _sha256_file(embedding_outliers_path),
        "dataset_aggregate_sha256": protected_before["trees"]["dataset"][
            "aggregate_sha256"
        ],
        "active_source_inputs": [
            {
                "source_id": _text(row["Source_ID"]),
                "sha256": _text(row["Source_SHA256"]),
                "owner": _canonical(row["Candidate_Owner"]),
                "review_action": _text(row["Review_Action"]),
                "policy_eligible": bool(row["Policy_Eligible"]),
            }
            for row in discovered_sources.to_dict("records")
        ],
        "identity_correction": {
            key: value
            for key, value in correction.manifest.items()
            if key != "created_at"
        },
        "build_configuration": asdict(EnrollmentBuildConfig()),
        "variant_definitions": [asdict(value) for value in VARIANT_DEFINITIONS],
    }
    content_fingerprint = _stable_digest(input_fingerprint)
    resolved_family_id = family_id.strip() or f"embfam-{content_fingerprint[:20]}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{5,95}", resolved_family_id):
        raise EmbeddingFamilyError(
            "Family ID must be 6-96 characters using letters, numbers, dots, dashes, or underscores"
        )
    final_family_dir = versions_root / resolved_family_id
    if final_family_dir.exists():
        verified = verify_embedding_family(
            final_family_dir,
            production_embeddings_path=production_embeddings_path,
            production_summary_path=production_summary_path,
        )
        if _text(verified.get("content_fingerprint_sha256")) != content_fingerprint:
            raise EmbeddingFamilyError(
                f"Embedding family ID already exists with different inputs: {resolved_family_id}"
            )
        decision = json.loads(
            (final_family_dir / "evaluation" / "evaluation_decision.json").read_text(
                encoding="utf-8"
            )
        )
        family_manifest = json.loads(
            (final_family_dir / "family_manifest.json").read_text(encoding="utf-8")
        )
        return FamilyBuildResult(
            final_family_dir,
            resolved_family_id,
            family_manifest,
            decision,
            True,
        )

    versions_root.mkdir(parents=True, exist_ok=True)
    temporary_family_dir = versions_root / f".{resolved_family_id}.building-{os.getpid()}"
    if temporary_family_dir.exists():
        raise EmbeddingFamilyError(
            f"Stale family build directory exists: {temporary_family_dir}"
        )
    temporary_family_dir.mkdir(parents=True, exist_ok=False)
    build_timestamp = _now()
    try:
        _write_json(
            temporary_family_dir / "approval_validation_summary.json",
            approvals.summary,
        )
        _write_json(
            temporary_family_dir / "approval_input_manifest.json",
            approvals.input_manifest,
        )
        _write_csv(
            temporary_family_dir / "validated_enrollment_decisions.csv",
            _approval_output_frame(approvals.enrollment, enrollment=True),
        )
        _write_csv(
            temporary_family_dir / "validated_cctv_decisions.csv",
            _approval_output_frame(approvals.cctv, enrollment=False),
        )
        _write_json(
            temporary_family_dir / "identity_correction_manifest.json",
            correction.manifest,
        )
        _write_csv(
            temporary_family_dir / "identity_correction_items.csv", correction.items
        )
        _write_json(temporary_family_dir / "protected_state_before.json", protected_before)

        if engine_factory is None:
            engine = FaceEngine(yunet_model, sface_model, detection_score=0.60)
        else:
            engine = engine_factory(yunet_model, sface_model, 0.60)
        enrollment_config = EnrollmentBuildConfig()
        policy, enrollment_records, _ = extract_cleaned_enrollment_records(
            repo_root=repo_root,
            sources=discovered_sources,
            engine=engine,
            config=enrollment_config,
            progress=progress,
        )
        _write_csv(
            temporary_family_dir / "augmentation_source_groups.csv", policy.groups
        )
        _write_csv(
            temporary_family_dir / "augmentation_selection_decisions.csv",
            policy.decisions,
        )
        _write_json(
            temporary_family_dir / "augmentation_policy_summary.json", policy.summary
        )
        source_reconciliation = _source_reconciliation_frame(
            policy=policy,
            approval_bundle=approvals,
            dataset_root=dataset_root,
            shruthi=shruthi,
        )
        _write_csv(
            temporary_family_dir / "source_reconciliation.csv", source_reconciliation
        )

        cctv_staging, cctv_records, staging_manifest = stage_approved_cctv_sources(
            temporary_family_dir=temporary_family_dir,
            final_family_dir=final_family_dir,
            repo_root=repo_root,
            approval_bundle=approvals,
            correction=correction,
            engine=engine,
            roster_rolls=roster_rolls,
            expected_approved=expected_approved_cctv,
            progress=progress,
        )
        _write_json(
            temporary_family_dir / "candidate_staging_manifest.json", staging_manifest
        )
        selected_enrollment = policy.sources[
            policy.sources["Selected_For_Candidate"].map(bool)
        ]
        candidate_source_manifest = {
            "schema_version": FAMILY_SCHEMA_VERSION,
            "family_id": resolved_family_id,
            "content_fingerprint_sha256": content_fingerprint,
            "created_at": build_timestamp,
            "configured_enrollment_root": str(dataset_root),
            "configured_augmented_root": {
                "path": str(augmented_dataset_root),
                "production_recorded_input": False,
                "activated_for_candidate": False,
            },
            "enrollment_source_count": len(selected_enrollment),
            "approved_cctv_source_count": len(cctv_staging),
            "enrollment_sources": [
                {
                    "source_id": _text(row["Source_ID"]),
                    "path": _text(row["Source_Path"]),
                    "sha256": _text(row["Source_SHA256"]),
                    "candidate_owner": _canonical(row["Candidate_Owner"]),
                    "source_group_id": _text(row["Source_Group_ID"]),
                    "review_item_id": _text(row["Review_Item_ID"]),
                    "review_action": _text(row["Review_Action"]),
                    "selection_reason": _text(row["Selection_Reason"]),
                }
                for row in selected_enrollment.to_dict("records")
            ],
            "approved_cctv_sources": staging_manifest["records"],
            "source_hashes_verified": True,
            "live_dataset_modified": False,
            "source_files_deleted": 0,
            "source_files_moved": 0,
            "model_prediction_used_as_ground_truth": False,
        }
        _write_json(
            temporary_family_dir / "candidate_source_manifest.json",
            candidate_source_manifest,
        )
        source_manifest_sha256 = _sha256_file(
            temporary_family_dir / "candidate_source_manifest.json"
        )
        identity_correction_sha256 = _sha256_file(
            temporary_family_dir / "identity_correction_manifest.json"
        )
        augmentation_policy_sha256 = _sha256_file(
            temporary_family_dir / "augmentation_policy_summary.json"
        )
        parent_payload = load_embedding_payload(production_embeddings_path)
        parent_records = list(parent_payload["records"])
        dimensions = {
            int(np.asarray(record.get("embedding")).reshape(-1).size)
            for record in parent_records
        }
        if len(dimensions) != 1:
            raise EmbeddingFamilyError(
                f"Production embedding dimensions are inconsistent: {sorted(dimensions)}"
            )
        expected_dimension = next(iter(dimensions))
        model_hashes = {
            "yunet_sha256": _sha256_file(yunet_model),
            "sface_sha256": _sha256_file(sface_model),
        }
        cctv_by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in cctv_records:
            cctv_by_session[_text(record["_source_session"])].append(record)
        variant_records = {
            "A": list(enrollment_records),
            "B": list(enrollment_records) + list(cctv_by_session[TUE_P2_SESSION]),
            "C": list(enrollment_records) + list(cctv_by_session[TUE_P1_SESSION]),
            "D": list(enrollment_records) + list(cctv_records),
        }
        variant_manifests: dict[str, dict[str, Any]] = {}
        variant_directories: dict[str, Path] = {}
        if progress:
            progress("Writing four immutable built_unapproved embedding variants")
        for definition in VARIANT_DEFINITIONS:
            temporary_variant_dir = (
                temporary_family_dir / "variants" / definition.directory
            )
            final_variant_dir = final_family_dir / "variants" / definition.directory
            manifest = _write_variant(
                temporary_variant_dir=temporary_variant_dir,
                final_variant_dir=final_variant_dir,
                family_id=resolved_family_id,
                definition=definition,
                family_fingerprint=content_fingerprint,
                records=variant_records[definition.key],
                parent_payload=parent_payload,
                parent_embeddings_sha256=production_preflight[
                    "production_embeddings"
                ]["sha256"],
                parent_summary_sha256=production_preflight["production_summary"][
                    "sha256"
                ],
                model_hashes=model_hashes,
                source_manifest_sha256=source_manifest_sha256,
                enrollment_approval_sha256=_sha256_file(enrollment_approvals_path),
                cctv_approval_sha256=_sha256_file(cctv_approvals_path),
                identity_correction_sha256=identity_correction_sha256,
                augmentation_policy_sha256=augmentation_policy_sha256,
                roster_rolls=roster_rolls,
                build_timestamp=build_timestamp,
                expected_dimension=expected_dimension,
            )
            variant_manifests[definition.key] = manifest
            variant_directories[definition.key] = temporary_variant_dir

        coverage = _student_source_coverage(
            roster_rolls=roster_rolls,
            policy=policy,
            cctv_staging=cctv_staging,
            variant_manifests=variant_manifests,
        )
        _write_csv(temporary_family_dir / "student_source_coverage.csv", coverage)
        rollback_plan = {
            "schema_version": FAMILY_SCHEMA_VERSION,
            "family_id": resolved_family_id,
            "prepared_before_any_future_promotion": True,
            "phase_1_2i_b_promotion_allowed": False,
            "production_embeddings": production_preflight["production_embeddings"],
            "production_summary": production_preflight["production_summary"],
            "candidate_promoted": False,
            "backup_created": False,
            "reason_no_backup_yet": (
                "Production was not modified. A byte-for-byte backup is required immediately "
                "before any separately approved future promotion."
            ),
            "future_rollback_requirements": [
                "Verify production hashes still match this family parent hashes.",
                "Create and hash a production pair backup before promotion.",
                "Atomically replace both production files or restore both on failure.",
                "Require passed MON_P3 untouched shadow validation and explicit user approval.",
            ],
        }
        _write_json(
            temporary_family_dir / "rollback" / "rollback_plan.json", rollback_plan
        )

        protected_after = snapshot_phase_i_b_protected_state(
            repo_root=repo_root,
            production_embeddings_path=production_embeddings_path,
            production_summary_path=production_summary_path,
            student_map_path=student_map_path,
            enrollment_review_package=enrollment_review_package,
            enrollment_approvals_path=enrollment_approvals_path,
            cctv_review_package=cctv_review_package,
            cctv_approvals_path=cctv_approvals_path,
            forensic_run_dir=forensic_run_dir,
            dataset_root=dataset_root,
            augmented_dataset_root=augmented_dataset_root,
            yunet_model=yunet_model,
            sface_model=sface_model,
        )
        protected_comparison = compare_protected_states(
            protected_before, protected_after
        )
        if not protected_comparison["unchanged"]:
            raise EmbeddingFamilyError(
                "Protected production state changed during candidate build: "
                f"files={protected_comparison['changed_fixed_files'][:5]}, "
                f"trees={protected_comparison['changed_trees']}"
            )
        _write_json(temporary_family_dir / "protected_state_after.json", protected_after)
        _write_json(
            temporary_family_dir / "protected_state_comparison.json",
            protected_comparison,
        )
        parent_coverage, parent_missing, parent_counts = _coverage_from_records(
            parent_records, roster_rolls
        )
        if progress:
            progress("Leakage-safe frozen benchmark rescore")
        decision, regression_summary = run_family_evaluation(
            temporary_family_dir=temporary_family_dir,
            final_family_dir=final_family_dir,
            family_id=resolved_family_id,
            forensic_run_dir=forensic_run_dir,
            student_map_path=student_map_path,
            production_embeddings_path=production_embeddings_path,
            variant_manifests=variant_manifests,
            variant_directories=variant_directories,
            engine=engine,
            roster_rolls=roster_rolls,
            production_roster_coverage=parent_coverage,
            protected_comparison=protected_comparison,
            progress=progress,
        )
        variant_summary = {
            key: {
                "version_id": manifest["version_id"],
                "role": manifest["variant_role"],
                "status": manifest["status"],
                "embedding_records": manifest["embedding_records"],
                "embedding_dimension": manifest["embedding_dimension"],
                "embedding_sha256": _sha256_file(
                    variant_directories[key] / "student_embeddings.pkl"
                ),
                "summary_sha256": _sha256_file(
                    variant_directories[key] / "embedding_summary.csv"
                ),
                "source_sessions_included": manifest[
                    "source_sessions_included"
                ],
                "evaluation_sessions_allowed": manifest[
                    "evaluation_sessions_allowed"
                ],
                "training_contaminated_descriptive_only": manifest[
                    "training_contaminated_descriptive_only"
                ],
                "production_promoted": False,
            }
            for key, manifest in variant_manifests.items()
        }
        family_manifest = {
            "schema_version": FAMILY_SCHEMA_VERSION,
            "family_id": resolved_family_id,
            "status": "built_unapproved",
            "final_decision": decision["decision"],
            "build_timestamp": build_timestamp,
            "completed_at": _now(),
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "content_fingerprint_sha256": content_fingerprint,
            "input_fingerprint": input_fingerprint,
            "parent_production": production_preflight,
            "model_hashes": model_hashes,
            "variants": variant_summary,
            "approval_validation_status": "passed",
            "identity_correction_status": correction.manifest["status"],
            "identity_correction_id": correction.manifest["correction_id"],
            "global_alias_created": False,
            "shruthi_source_reconciliation": shruthi,
            "enrollment_source_policy": policy.summary,
            "approved_cctv_staged_count": len(cctv_staging),
            "candidate_cvo_coverage": variant_manifests["A"][
                "canonical_identity_coverage"
            ],
            "production_cvo_coverage": {
                "covered": parent_coverage,
                "missing": parent_missing,
                "embedding_counts": parent_counts,
            },
            "ai0110_candidate_coverage": int(
                dict(variant_manifests["A"]["per_student_embedding_counts"]).get(
                    TARGET_IDENTITY, 0
                )
            )
            > 0,
            "missing_0268_status": (
                "Dataset Missing"
                if MISSING_CVO_IDENTITY
                in variant_manifests["A"]["missing_students"]
                else "Covered"
            ),
            "benchmark_metrics": regression_summary["metrics"],
            "false_identity_gate": decision["gates"]["zero_new_false_accepts"],
            "leakage_control": decision["gates"]["leakage"],
            "exact_next_step": decision["exact_next_step"],
            "production_embeddings_changed": False,
            "production_summary_changed": False,
            "datasets_changed_by_this_phase": False,
            "official_thresholds_changed": False,
            "attendance_rule_changed": False,
            "official_attendance_changed": False,
            "mon_p3_processed": False,
            "candidate_built": True,
            "candidate_promoted": False,
            "production_approved": False,
            "current_version_pointer_updated": False,
            "rejected_candidate_remains_rejected": REJECTED_CANDIDATE_ID,
            "output_dir": str(final_family_dir),
        }
        _write_json(temporary_family_dir / "family_manifest.json", family_manifest)
        if launcher_source and Path(launcher_source).is_file():
            shutil.copy2(
                launcher_source, temporary_family_dir / "run_phase_1_2i_b.ps1"
            )
        output_manifest_path = _write_output_manifest(
            temporary_family_dir,
            filename="output_manifest.json",
            declared_output_dir=final_family_dir,
            extra={
                "family_id": resolved_family_id,
                "content_fingerprint_sha256": content_fingerprint,
                "status": "built_unapproved",
                "decision": decision["decision"],
                "production_embeddings_changed": False,
                "production_summary_changed": False,
                "datasets_changed": False,
                "official_attendance_changed": False,
                "candidate_promoted": False,
                "production_approved": False,
                "mon_p3_processed": False,
            },
        )
        try:
            verify_output_manifest(temporary_family_dir, output_manifest_path)
        except ShadowValidationError as exc:
            raise EmbeddingFamilyError(str(exc)) from exc
        temporary_family_dir.replace(final_family_dir)
        verified = verify_embedding_family(
            final_family_dir,
            production_embeddings_path=production_embeddings_path,
            production_summary_path=production_summary_path,
        )
        if progress:
            progress(f"Family build decision: {decision['decision']}")
        return FamilyBuildResult(
            final_family_dir,
            resolved_family_id,
            family_manifest,
            decision,
            False,
        )
    except Exception as exc:
        preserved_path = temporary_family_dir
        try:
            _write_json(
                temporary_family_dir / "failure_report.json",
                {
                    "schema_version": FAMILY_SCHEMA_VERSION,
                    "family_id": resolved_family_id,
                    "status": "build_failed_preserved",
                    "failed_at": _now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "production_promoted": False,
                    "production_approved": False,
                    "mon_p3_processed": False,
                },
            )
            failed_name = (
                f"failed-{resolved_family_id}-"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}-{os.getpid()}"
            )
            preserved_path = versions_root / failed_name
            temporary_family_dir.replace(preserved_path)
        except OSError:
            pass
        raise EmbeddingFamilyError(
            f"Phase 1.2I-B family build failed; partial artifacts preserved at "
            f"{preserved_path}: {exc}"
        ) from exc


def verify_embedding_family(
    family_dir: Path,
    *,
    production_embeddings_path: Path | None = None,
    production_summary_path: Path | None = None,
) -> dict[str, Any]:
    family_dir = Path(family_dir).resolve()
    manifest_path = family_dir / "output_manifest.json"
    if not family_dir.is_dir() or not manifest_path.is_file():
        raise EmbeddingFamilyError(f"Embedding family output is incomplete: {family_dir}")
    try:
        verified = verify_output_manifest(family_dir, manifest_path)
    except ShadowValidationError as exc:
        raise EmbeddingFamilyError(str(exc)) from exc
    output_manifest = dict(verified["manifest"])
    recorded_files = set(dict(output_manifest.get("files_sha256") or {}))
    actual_files = {
        path.relative_to(family_dir).as_posix()
        for path in family_dir.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if recorded_files != actual_files:
        raise EmbeddingFamilyError(
            f"Embedding family manifest file set mismatch; "
            f"missing={sorted(recorded_files - actual_files)[:5]}, "
            f"extra={sorted(actual_files - recorded_files)[:5]}"
        )
    family_manifest = json.loads(
        (family_dir / "family_manifest.json").read_text(encoding="utf-8")
    )
    family_id = _text(family_manifest.get("family_id"))
    if family_id != family_dir.name or family_id != _text(output_manifest.get("family_id")):
        raise EmbeddingFamilyError("Embedding family directory and manifest IDs do not match")
    if (
        family_manifest.get("production_approved") is not False
        or family_manifest.get("candidate_promoted") is not False
        or family_manifest.get("mon_p3_processed") is not False
        or family_manifest.get("global_alias_created") is not False
    ):
        raise EmbeddingFamilyError("Embedding family violates the unapproved safety contract")
    variants = dict(family_manifest.get("variants") or {})
    definitions = {definition.key: definition for definition in VARIANT_DEFINITIONS}
    for key, definition in definitions.items():
        if key not in variants:
            raise EmbeddingFamilyError(f"Embedding family variant {key} is missing")
        variant_dir = family_dir / "variants" / definition.directory
        variant_manifest_path = variant_dir / "version_manifest.json"
        variant_manifest = json.loads(
            variant_manifest_path.read_text(encoding="utf-8")
        )
        if (
            variant_manifest.get("status") != "built_unapproved"
            or variant_manifest.get("production_promoted") is not False
            or _text(variant_manifest.get("family_id")) != family_id
        ):
            raise EmbeddingFamilyError(f"Variant {key} lifecycle state is unsafe")
        failures = [
            relative
            for relative, expected_hash in dict(
                variant_manifest.get("artifact_sha256") or {}
            ).items()
            if not (variant_dir / relative).is_file()
            or _sha256_file(variant_dir / relative) != _text(expected_hash)
        ]
        if failures:
            raise EmbeddingFamilyError(
                f"Variant {key} artifact integrity failed: {failures[:5]}"
            )
        payload = load_embedding_payload(variant_dir / "student_embeddings.pkl")
        records = list(payload["records"])
        _validate_candidate_records(
            [
                {
                    **record,
                    "_source_id": f"verify-{index}",
                    "_source_kind": "verified",
                    "_source_session": "",
                }
                for index, record in enumerate(records)
            ],
            int(variant_manifest["embedding_dimension"]),
        )
        if len(records) != int(variant_manifest["embedding_records"]):
            raise EmbeddingFamilyError(f"Variant {key} record count changed")
        included = set(variant_manifest.get("source_sessions_included") or [])
        allowed = set(variant_manifest.get("evaluation_sessions_allowed") or [])
        if included & allowed:
            raise EmbeddingFamilyError(f"Variant {key} has an evaluation leakage overlap")
        if key == "D" and variant_manifest.get(
            "training_contaminated_descriptive_only"
        ) is not True:
            raise EmbeddingFamilyError("Variant D is not marked contaminated/descriptive")
    evaluation_manifest = family_dir / "evaluation" / "evaluation_output_manifest.json"
    try:
        verify_output_manifest(family_dir / "evaluation", evaluation_manifest)
    except ShadowValidationError as exc:
        raise EmbeddingFamilyError(str(exc)) from exc
    decision = json.loads(
        (family_dir / "evaluation" / "evaluation_decision.json").read_text(
            encoding="utf-8"
        )
    )
    if decision.get("production_approved") is not False or decision.get(
        "candidate_promoted"
    ) is not False:
        raise EmbeddingFamilyError("Evaluation decision attempts to approve or promote")
    protected = json.loads(
        (family_dir / "protected_state_comparison.json").read_text(encoding="utf-8")
    )
    if protected.get("unchanged") is not True:
        raise EmbeddingFamilyError("Family records a protected production-state change")
    if production_embeddings_path and _sha256_file(
        Path(production_embeddings_path)
    ) != _text(
        dict(family_manifest.get("parent_production") or {})
        .get("production_embeddings", {})
        .get("sha256")
    ):
        raise EmbeddingFamilyError("Current production embeddings differ from the family parent")
    if production_summary_path and _sha256_file(Path(production_summary_path)) != _text(
        dict(family_manifest.get("parent_production") or {})
        .get("production_summary", {})
        .get("sha256")
    ):
        raise EmbeddingFamilyError("Current production summary differs from the family parent")
    return {
        "family_id": family_id,
        "status": family_manifest["status"],
        "decision": decision["decision"],
        "content_fingerprint_sha256": family_manifest[
            "content_fingerprint_sha256"
        ],
        "verified_files": verified["verified_files"],
        "variants_verified": len(definitions),
        "production_approved": False,
        "candidate_promoted": False,
        "mon_p3_processed": False,
    }


def show_embedding_family(family_dir: Path) -> dict[str, Any]:
    family_dir = Path(family_dir).resolve()
    family_manifest = json.loads(
        (family_dir / "family_manifest.json").read_text(encoding="utf-8")
    )
    decision = json.loads(
        (family_dir / "evaluation" / "evaluation_decision.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        "family_id": family_manifest["family_id"],
        "status": family_manifest["status"],
        "decision": decision["decision"],
        "variants": family_manifest["variants"],
        "candidate_cvo_coverage": family_manifest["candidate_cvo_coverage"],
        "ai0110_candidate_coverage": family_manifest[
            "ai0110_candidate_coverage"
        ],
        "missing_0268_status": family_manifest["missing_0268_status"],
        "exact_next_step": decision["exact_next_step"],
        "output_dir": str(family_dir),
        "production_approved": False,
        "candidate_promoted": False,
        "mon_p3_processed": False,
    }
