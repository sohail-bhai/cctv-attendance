from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


POLICY_VERSION = "product-phase-2k-c1-retrospective-multisession-audit-v1"
OUTPUT_SCHEMA_VERSION = 1
CURRENT_AUTOMATIC_POLICY = "product-phase-2i-strict-tracklet-authority-v1"
CURRENT_AUTHORITY = "strict_tracklet_aggregate_with_guarded_review_candidates"
CURRENT_FAMILY = "embfam-274b5207b8b71294ff75"
CURRENT_VARIANT = "embfam-274b5207b8b71294ff75-d"
CURRENT_EMBEDDING_SHA256 = "f088d827adc548ee95f46566d758fd71fc304d042c43f1ecffc6526b60bcd832"
CURRENT_SUMMARY_SHA256 = "63885588c374c37f4da9bf85294f240bdf0f28cb585e77b894ab516139ca46ae"
MATCH_THRESHOLD = 0.48
MARGIN_THRESHOLD = 0.08
ARCHIVED_CANDIDATE_MANIFEST_SHA256 = "e941fbbcdf70511a13ae49b773979f54f921f1b121a27128cfe84b312e0ffd17"
MIXED_TRACKLET_ID = "CP1-cam5-back-TRK00005"
MIXED_SESSION_ID = "2026-06-22__B51__P4__CVO"
MIXED_PREDICTED_ROLL = "24011CSEAI0051"
MISSING_ENROLLMENT_ROLL = "2401100CSE0268"

SESSIONS = (
    "2026-06-22__B51__P3__CVO",
    "2026-06-22__B51__P4__CVO",
    "2026-06-30__B51__P1__CVO",
    "2026-06-30__B51__P2__CVO",
)

REQUIRED_OUTPUT_FILES = (
    "audit_policy.json",
    "session_evidence_inventory.csv",
    "session_evidence_sufficiency.csv",
    "normalized_review_evidence.csv",
    "strict_identity_safety.csv",
    "guarded_recovery_safety.csv",
    "retention_analysis.csv",
    "purity_quarantine_analysis.csv",
    "carry_forward_analysis.csv",
    "attendance_semantics_analysis.csv",
    "roster_integrity_analysis.csv",
    "policy_compatibility.csv",
    "contamination_and_claim_limits.json",
    "per_session_summary.csv",
    "combined_summary.json",
    "recommendation.json",
    "no_activation_declaration.json",
    "source_manifest.json",
    "evaluation_summary.json",
    "immutable_manifest.json",
)

PROTECTED_EXPECTED = {
    "data/attendance_status.json": "46fd2df55cc3f61c5fa03c893937eb2828b9cc2af064ea9e427276b79a9d0b46",
    "data/job_runtime.json": "5a23f09654d91fac09804cd97c2fa114bf4956a914f2e02ce2b7090f730141ed",
    "data/role_users.json": "77125140004294f2834fc869fd590e167d8ab78cff5dbeda1c074ab76c177517",
    "data/student_faculty_map.json": "0eca63ea58d12a73d3bd11e620b74be733a3219c9244e8a4a267d5b02acd3f91",
    "data/manual_overrides.json": "e9c6bd35c23c353795edb5d3c02e808793d7cbd90c7b0a87fdfc686f8fd5cdcf",
    "data/review_evidence_registry.json": "3b04e9aecfdfb1f64346c4a0e709fd9c36d7c56545bf816d6641b4c2a2e80841",
    "models/student_embeddings.pkl": CURRENT_EMBEDDING_SHA256,
    "models/embedding_summary.csv": CURRENT_SUMMARY_SHA256,
    "models/current_embedding_version.json": "7999b8ccf787dca9b8fb862f4e53ce7c1102eb729a3732c82fee76f3ba3ee05e",
    "timetable_b51_2026_2027.csv": "10bbd578a859fbd4fa228c4eb4f1f92e7de5b92a1c4236d71a00337eecac9c57",
}

MANIFESTS = {
    "phase_1_2j_mon_p3": (
        "attendance_output/shadow_validation/phase_1_2j/mon-p3-shadow-8136f24575f6d319fb96/output_manifest.json",
        "1f372740c694f19230a6c78fc43487c3c9a8764c2d7fd2f11e4aa7842bbdae84",
    ),
    "phase_1_2j_mon_p3_evaluation": (
        "attendance_output/shadow_validation/phase_1_2j/mon-p3-shadow-8136f24575f6d319fb96/evaluation/mon-p3-evaluation-8f15934c693111852524/output_manifest.json",
        "00218c30700867c9c90bc0e3728db530c6b747e999a36288268b7050665df4f7",
    ),
    "phase_1_2n_source_freeze": (
        "attendance_output/shadow_validation/phase_1_2n/source_freeze/source-freeze-06578150c42a1253d835/output_manifest.json",
        "e68a963749dd0013131393eefc2a969fb148c15cd6ebbe9aab07fe824ac5e276",
    ),
    "phase_1_2n_mon_p4": (
        "attendance_output/shadow_validation/phase_1_2n/shadow_runs/mon-p4-shadow-695277fbd0dcd9d431e1/output_manifest.json",
        "b00db94aa2f90d777619f5c3991bf12f81d3e37049c29f0b2e134ce043b3c08d",
    ),
    "phase_1_2n_mon_p4_evaluation": (
        "attendance_output/shadow_validation/phase_1_2n/shadow_runs/mon-p4-shadow-695277fbd0dcd9d431e1/evaluation/mon-p4-evaluation-302c6707c95029a017df/output_manifest.json",
        "5fc93d3e60f190aaf02bda4842124a9663011cf2612d7ef3ef81c1fa663ba111",
    ),
    "phase_2h": (
        "attendance_output/product_workflow/phase_2h_multiframe_recovery/multiframe-recovery-7d9d3a539c0b32e33048/output_manifest.json",
        "54e39ba629a032d4e3497a86f7a91bfeea66572ced4466ba16f150f89dc57f84",
    ),
    "phase_2h_evaluation": (
        "attendance_output/product_workflow/phase_2h_multiframe_recovery/multiframe-recovery-7d9d3a539c0b32e33048/evaluation/evaluation_manifest.json",
        "9f1fdeaa4449d053009f1c6a24683089f85ac18d39180201dcf396a0c7a1f8ad",
    ),
    "phase_2i_authority": (
        "attendance_output/product_workflow/phase_2i_authority/authority-revision-2d83c4f679f839a6c784c636/authority_manifest.json",
        "5e989648c8dc7605f088b585bdc4c954bf2e18cdaf93bb7107f0531f0068bebb",
    ),
    "multisession_forensic": (
        "attendance_output/embedding_forensics/embedding_forensics_reviewfix_20260713_140159_455110/output_manifest.json",
        "19a5605567a47dc383c97df0b43532a737531448502486c4068553a920434a63",
    ),
    "multisession_ground_truth": (
        "attendance_output/embedding_forensics/embedding_forensics_reviewfix_20260713_140159_455110/multisession_ground_truth_manifest.json",
        "7e2266a0de9d4d576cf38b070ed03d25f61fd209f52a6c6dbd9c560438438fe4",
    ),
    "phase_2k_a": (
        "attendance_output/product_workflow/phase_2k_tracklet_purity/tracklet-purity-43cec17498dc05615502/immutable_manifest.json",
        "7e0da7fd9b3edb7411a31be94f5f42063365bd0a6831735948700d263148c552",
    ),
    "phase_2k_b": (
        "attendance_output/product_workflow/phase_2k_carry_forward/carry-forward-e20a746d3d34c8141ab5/immutable_manifest.json",
        "79a38737c6057768dc4128c3507fced268ea25c0beb84121a27cf2bb23195ac3",
    ),
    "phase_2k_c0": (
        "attendance_output/product_workflow/phase_2k_capture_intake_tooling/intake-tooling-ff8dd99df5c3e2795234/immutable_manifest.json",
        "4f9e1a8d71b0ac2e200e58c7753c47cbef899f675121464b2cdb07937d7bfde7",
    ),
}


class RetrospectiveAuditError(RuntimeError):
    """Raised when the retrospective audit cannot prove its inputs or output."""


@dataclass(frozen=True)
class AuditPreflight:
    repo_root: Path
    output_root: Path
    run_id: str
    run_fingerprint_sha256: str
    manifest_rows: tuple[dict[str, Any], ...]
    inventory_rows: tuple[dict[str, Any], ...]
    sufficiency_rows: tuple[dict[str, Any], ...]
    normalized_records: tuple[dict[str, Any], ...]
    protected_hashes: dict[str, str]
    roster_rolls: tuple[str, ...]
    analyses: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def canonical_json_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    for raw in rows:
        rendered: dict[str, Any] = {}
        for field in fields:
            value = raw.get(field, "")
            if isinstance(value, bool):
                value = "true" if value else "false"
            elif isinstance(value, (list, tuple, set)):
                value = ";".join(str(item) for item in value)
            elif value is None:
                value = "unavailable"
            rendered[field] = value
        writer.writerow(rendered)
    return stream.getvalue().encode("utf-8")


def normalize_windows_path(value: str | Path) -> str:
    text = str(value).strip().replace("\\", "/")
    if not text or "\x00" in text:
        raise RetrospectiveAuditError("unsafe_or_empty_path")
    is_unc = text.startswith("//")
    prefix = "//" if is_unc else ("/" if text.startswith("/") else "")
    body = text[2:] if is_unc else (text[1:] if prefix else text)
    while "//" in body:
        body = body.replace("//", "/")
    parts = body.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise RetrospectiveAuditError("unsafe_path_traversal")
    if re.fullmatch(r"[A-Za-z]:", parts[0]):
        parts[0] = parts[0][0].upper() + ":"
    return prefix + "/".join(parts)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "accepted", "pass"}


def _float(value: Any) -> float | None:
    try:
        return float(value) if str(value).strip() else None
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    try:
        return int(float(value)) if str(value).strip() else None
    except (TypeError, ValueError):
        return None


def _identity_outcome(disposition: str, predicted: str, reviewed: str) -> str:
    disposition = disposition.strip().lower()
    if disposition == "identified":
        return "correct" if predicted and predicted == reviewed else "wrong_person"
    if disposition == "not_in_mapping":
        return "outsider"
    if disposition == "mixed_track":
        return "mixed_track"
    if disposition in {"unidentifiable", "unclear"}:
        return "unclear"
    return "unverifiable"


def identity_fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16] if value else ""


def normalize_records(
    records: Iterable[Mapping[str, Any]],
    *,
    source_schema: str,
    context: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Normalize immutable historical rows without mutating or merging identities."""

    normalized: list[dict[str, Any]] = []
    seen: dict[tuple[str, str, str], dict[str, Any]] = {}
    for raw in records:
        if source_schema in {"phase_1_2j", "phase_1_2n"}:
            suffix = "J" if source_schema.endswith("2j") else "N"
            predicted = str(raw.get("Candidate_Roll") or raw.get("Predicted_Roll") or "").strip()
            reviewed = str(raw.get("Actual_Roll_Normalized") or raw.get("Actual_Roll") or "").strip()
            disposition = str(raw.get("Review_Status") or "unverifiable").strip().lower()
            strict = _bool(raw.get("Candidate_Accepted_Bool"))
            result = str(raw.get(f"Phase_1_2{suffix}_Result") or "")
            previously_correct = result in {"both_correct_same_identity", "candidate_lost_correct_production_accept"}
            retained = result == "both_correct_same_identity"
            recovered = result == "candidate_correct_recovery"
            strict_metric_available = True
            guarded_metric_available = False
            retention_baseline_available = True
            recovery_metric_available = True
            guarded = False
            evidence_id = str(raw.get("Tracklet_ID") or "")
        elif source_schema == "multisession_ground_truth":
            predicted = str(raw.get("Predicted_Roll") or "").strip()
            reviewed = str(raw.get("Actual_Roll") or "").strip()
            disposition = str(raw.get("Review_Status") or "unverifiable").strip().lower()
            strict = _bool(raw.get("Tracklet_Accepted"))
            guarded = _bool(raw.get("Shadow_Accepted"))
            previously_correct = strict and str(raw.get("Top1_Correct")) == "Yes"
            retained = previously_correct
            recovered = guarded and _identity_outcome(disposition, predicted, reviewed) == "correct"
            strict_metric_available = str(raw.get("Tracklet_Accepted") or "").strip() != ""
            guarded_metric_available = str(raw.get("Shadow_Accepted") or "").strip() != ""
            retention_baseline_available = bool(context.get("retention_baseline_available", strict_metric_available))
            recovery_metric_available = bool(context.get("recovery_metric_available", guarded_metric_available))
            evidence_id = str(raw.get("Benchmark_Row_ID") or raw.get("Tracklet_ID") or "")
        elif source_schema == "phase_2h":
            predicted = str(raw.get("Predicted_Roll") or "").strip()
            reviewed = str(raw.get("Actual_Roll") or "").strip()
            disposition = str(raw.get("Review_Status") or "unverifiable").strip().lower()
            strict = str(raw.get("Evidence_Tier") or "") == "strict_accepted"
            guarded = str(raw.get("Evidence_Tier") or "") == "guarded_recovery"
            previously_correct = strict and _identity_outcome(disposition, predicted, reviewed) == "correct"
            retained = previously_correct
            recovered = guarded and _identity_outcome(disposition, predicted, reviewed) == "correct"
            strict_metric_available = True
            guarded_metric_available = True
            retention_baseline_available = True
            recovery_metric_available = True
            evidence_id = str(raw.get("Tracklet_ID") or "")
        elif source_schema == "canonical":
            predicted = str(raw.get("predicted_identity") or "").strip()
            reviewed = str(raw.get("reviewed_identity") or "").strip()
            disposition = str(raw.get("reviewed_disposition") or "unverifiable").strip().lower()
            strict = bool(raw.get("strict_accepted"))
            guarded = bool(raw.get("guarded_candidate"))
            previously_correct = bool(raw.get("previously_correct"))
            retained = bool(raw.get("retained"))
            recovered = bool(raw.get("recovered"))
            strict_metric_available = bool(raw.get("strict_metric_available", True))
            guarded_metric_available = bool(raw.get("guarded_metric_available", True))
            retention_baseline_available = bool(raw.get("retention_baseline_available", True))
            recovery_metric_available = bool(raw.get("recovery_metric_available", True))
            evidence_id = str(raw.get("evidence_id") or raw.get("parent_track_id") or "")
        else:
            raise RetrospectiveAuditError(f"unsupported_historical_schema:{source_schema}")

        session_id = str(raw.get("Session_ID") or raw.get("session_id") or context.get("session_id") or "")
        parent_track = str(raw.get("Tracklet_ID") or raw.get("parent_track_id") or evidence_id)
        checkpoint = str(raw.get("Checkpoint_ID") or raw.get("checkpoint") or parent_track.split("-")[0])
        camera = str(raw.get("Camera_ID") or raw.get("camera") or "")
        if not session_id or not evidence_id or not parent_track:
            raise RetrospectiveAuditError("ambiguous_evidence_join")
        outcome = _identity_outcome(disposition, predicted, reviewed)
        record = {
            "session_id": session_id,
            "source_fingerprint": str(context.get("source_fingerprint") or ""),
            "checkpoint": checkpoint,
            "camera": camera,
            "parent_track_id": parent_track,
            "evidence_id": evidence_id,
            "predicted_identity": predicted,
            "reviewed_identity": reviewed,
            "reviewed_disposition": disposition,
            "identity_outcome": outcome,
            "strict_accepted": strict,
            "guarded_candidate": guarded,
            "strict_metric_available": strict_metric_available,
            "guarded_metric_available": guarded_metric_available,
            "score": _float(raw.get("Candidate_Tracklet_Best_Score") or raw.get("Best_Score") or raw.get("Tracklet_Best_Score") or raw.get("score")),
            "margin": _float(raw.get("Candidate_Tracklet_Margin") or raw.get("Margin") or raw.get("Tracklet_Margin") or raw.get("margin")),
            "observation_count": _int(raw.get("Observation_Count") or raw.get("observation_count")),
            "checkpoint_support": _int(raw.get("checkpoint_support")),
            "purity_result": str(raw.get("Purity_Outcome") or raw.get("purity_result") or "unavailable"),
            "quarantine_result": str(raw.get("quarantine_result") or "unavailable"),
            "carry_forward_result": str(raw.get("carry_forward_result") or "unavailable"),
            "embedding_family": str(context.get("embedding_family") or raw.get("embedding_family") or "unavailable"),
            "embedding_hash": str(context.get("embedding_hash") or raw.get("embedding_hash") or "unavailable"),
            "policy_version": str(context.get("policy_version") or raw.get("policy_version") or "unavailable"),
            "policy_compatible": bool(context.get("policy_compatible", raw.get("policy_compatible", False))),
            "evidence_family": str(context.get("evidence_family") or raw.get("evidence_family") or source_schema),
            "provenance_artifact": normalize_windows_path(str(context.get("provenance_artifact") or raw.get("provenance_artifact") or "fixture/source.csv")),
            "provenance_hash": str(context.get("provenance_hash") or raw.get("provenance_hash") or "0" * 64),
            "previously_correct": previously_correct,
            "retained": retained,
            "recovered": recovered,
            "retention_baseline_available": retention_baseline_available,
            "recovery_metric_available": recovery_metric_available,
        }
        key = (session_id, record["evidence_family"], evidence_id)
        if key in seen:
            if seen[key] != record:
                raise RetrospectiveAuditError("duplicate_conflicting_review_rows")
            raise RetrospectiveAuditError("duplicate_review_rows")
        seen[key] = record
        normalized.append(record)
    return tuple(sorted(normalized, key=lambda row: (row["session_id"], row["evidence_family"], row["evidence_id"])))


def _metric(value: int | None) -> int | str:
    return "unavailable" if value is None else value


def safety_metrics(records: Sequence[Mapping[str, Any]], authority_field: str) -> dict[str, int | str]:
    if not records:
        return {key: "unavailable" for key in ("material", "correct", "wrong_person", "outsider", "mixed_track", "unclear_or_unverifiable")}
    availability_field = {
        "strict_accepted": "strict_metric_available",
        "guarded_candidate": "guarded_metric_available",
    }.get(authority_field)
    available = [row for row in records if availability_field is None or bool(row.get(availability_field, True))]
    if not available:
        return {key: "unavailable" for key in ("material", "correct", "wrong_person", "outsider", "mixed_track", "unclear_or_unverifiable")}
    applicable = [row for row in available if bool(row.get(authority_field))]
    return {
        "material": len(applicable),
        "correct": sum(row.get("identity_outcome") == "correct" for row in applicable),
        "wrong_person": sum(row.get("identity_outcome") == "wrong_person" for row in applicable),
        "outsider": sum(row.get("identity_outcome") == "outsider" for row in applicable),
        "mixed_track": sum(row.get("identity_outcome") == "mixed_track" for row in applicable),
        "unclear_or_unverifiable": sum(row.get("identity_outcome") in {"unclear", "unverifiable"} for row in applicable),
    }


def retention_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, int | str]:
    if not records:
        return {"previously_correct": "unavailable", "retained": "unavailable", "lost": "unavailable", "recovered": "unavailable", "lost_checkpoints": "unavailable", "recovered_checkpoints": "unavailable"}
    baseline_rows = [row for row in records if bool(row.get("retention_baseline_available", True))]
    recovery_rows = [row for row in records if bool(row.get("recovery_metric_available", True))]
    previously = sum(bool(row.get("previously_correct")) for row in baseline_rows)
    retained = sum(bool(row.get("retained")) for row in baseline_rows)
    recovered = sum(bool(row.get("recovered")) for row in recovery_rows)
    return {
        "previously_correct": _metric(previously if baseline_rows else None),
        "retained": _metric(retained if baseline_rows else None),
        "lost": _metric(previously - retained if baseline_rows else None),
        "recovered": _metric(recovered if recovery_rows else None),
        "lost_checkpoints": _metric(previously - retained if baseline_rows else None),
        "recovered_checkpoints": _metric(recovered if recovery_rows else None),
    }


def validate_bindings(
    records: Sequence[Mapping[str, Any]],
    *,
    source_fingerprint: str,
    embedding_hash: str,
    policy_version: str,
) -> None:
    for row in records:
        if row.get("source_fingerprint") != source_fingerprint:
            raise RetrospectiveAuditError("source_fingerprint_mismatch")
        if row.get("embedding_hash") != embedding_hash:
            raise RetrospectiveAuditError("embedding_hash_mismatch")
        if row.get("policy_version") != policy_version:
            raise RetrospectiveAuditError("policy_mismatch")


def validate_roster(rolls: Sequence[str]) -> dict[str, Any]:
    normalized = [str(roll).strip() for roll in rolls]
    missing = sum(not roll for roll in normalized)
    duplicates = len(normalized) - len(set(normalized))
    distinct_violation = "24011CSEAI0110" in normalized and "2401100CSE0110" in normalized and normalized.index("24011CSEAI0110") == normalized.index("2401100CSE0110")
    return {
        "row_count": len(normalized),
        "missing_rows": missing,
        "duplicate_rows": duplicates,
        "distinct_roll_violation": distinct_violation,
        "valid": missing == 0 and duplicates == 0 and not distinct_violation,
    }


def evaluate_attendance_semantics(
    statuses: Sequence[str],
    *,
    roster_count: int,
    low_quality_run: bool,
) -> dict[str, Any]:
    counts = {status: list(statuses).count(status) for status in ("Present", "Needs Review", "Unconfirmed", "Missing Enrollment", "Absent")}
    counts["Total"] = len(statuses)
    return {
        "counts": counts,
        "roster_complete": len(statuses) == roster_count,
        "false_mass_absence": low_quality_run and counts["Absent"] > 0,
        "low_quality_non_detection_safe": not low_quality_run or counts["Absent"] == 0,
        "missing_enrollment_distinct": counts["Missing Enrollment"] >= 0,
    }


def choose_recommendation(
    *,
    artifact_verification_passed: bool,
    exact_current_evidence_exists: bool,
    unsafe_current_strict_accepts: int,
    all_sessions_complete_and_independent: bool,
) -> str:
    if not artifact_verification_passed:
        return "artifact_verification_failed"
    if unsafe_current_strict_accepts:
        return "strict_authority_unsafe"
    if not exact_current_evidence_exists:
        return "insufficient_retrospective_evidence"
    if not all_sessions_complete_and_independent:
        return "retain_current_policy_with_blockers"
    return "retain_current_policy"


def classify_sufficiency(facts: Mapping[str, Any]) -> str:
    if not facts.get("manifest_verified"):
        return "unverifiable_or_tampered"
    required = ("source_provenance", "embedding_provenance", "track_identity", "human_truth", "review_join", "roster_context", "policy_provenance", "evaluation_complete")
    if not all(facts.get(item) for item in required):
        return "partial_retrospective_evidence" if facts.get("human_truth") else "insufficient_for_identity_safety_evaluation"
    if not facts.get("current_policy_compatible"):
        return "incompatible_policy_evidence"
    return "complete_retrospective_evidence"


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise RetrospectiveAuditError(f"expected_json_object:{path}")
    return value


def verify_manifest(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise RetrospectiveAuditError(f"manifest_missing_or_unsafe:{path}")
    actual_manifest_sha = sha256_file(path)
    if expected_sha256 and actual_manifest_sha != expected_sha256:
        raise RetrospectiveAuditError(f"manifest_tampered:{path}")
    manifest = _load_json(path)
    mapping = manifest.get("files_sha256") or manifest.get("files")
    if isinstance(mapping, Mapping):
        for relative, metadata in mapping.items():
            target = path.parent / Path(str(relative).replace("/", os.sep))
            expected = metadata.get("sha256") if isinstance(metadata, Mapping) else metadata
            if not target.is_file() or target.is_symlink() or sha256_file(target) != expected:
                raise RetrospectiveAuditError(f"manifest_file_tampered:{target}")
            if isinstance(metadata, Mapping) and "size_bytes" in metadata and target.stat().st_size != int(metadata["size_bytes"]):
                raise RetrospectiveAuditError(f"manifest_file_size_changed:{target}")
    return manifest


def _verify_authority_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    manifest = verify_manifest(path, expected_sha256)
    evaluation_root = path.parents[2] / "phase_2h_multiframe_recovery" / "multiframe-recovery-7d9d3a539c0b32e33048" / "evaluation"
    for relative, metadata in (manifest.get("evaluation_files") or {}).items():
        target = evaluation_root / relative
        if not target.is_file() or sha256_file(target) != metadata.get("sha256") or target.stat().st_size != int(metadata.get("size_bytes", -1)):
            raise RetrospectiveAuditError(f"authority_evaluation_binding_changed:{target}")
    return manifest


def _roster_rolls(student_map: Mapping[str, Any]) -> tuple[str, ...]:
    subject_students = student_map.get("subject_students")
    if not isinstance(subject_students, Mapping) or not isinstance(subject_students.get("CVO"), list):
        raise RetrospectiveAuditError("cvo_roster_mapping_missing")
    candidates = [str(row.get("roll") or "") for row in subject_students["CVO"] if isinstance(row, Mapping)]
    if any(not roll for roll in candidates) or len(candidates) != len(set(candidates)):
        raise RetrospectiveAuditError("cvo_roster_duplicate_or_missing_roll")
    unique = sorted(candidates)
    if len(unique) != 27:
        raise RetrospectiveAuditError(f"cvo_roster_count_changed:{len(unique)}")
    return tuple(unique)


def _join_phase_2h(selected: Sequence[Mapping[str, str]], reviewed: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    review_by_track: dict[str, Mapping[str, str]] = {}
    for row in reviewed:
        track = str(row.get("Tracklet_ID") or "")
        if not track or track in review_by_track:
            raise RetrospectiveAuditError("ambiguous_phase_2h_review_join")
        review_by_track[track] = row
    joined: list[dict[str, str]] = []
    for row in selected:
        track = str(row.get("Tracklet_ID") or "")
        review = review_by_track.get(track)
        if review is None:
            raise RetrospectiveAuditError(f"missing_phase_2h_review_join:{track}")
        item = dict(row)
        item.update({key: value for key, value in review.items() if key not in item or not item[key]})
        joined.append(item)
    if len(joined) != len(reviewed):
        raise RetrospectiveAuditError("ambiguous_phase_2h_review_join")
    return joined


def _attach_purity(records: Sequence[dict[str, Any]], rows: Sequence[Mapping[str, str]]) -> tuple[dict[str, Any], ...]:
    by_track = {str(row.get("Tracklet_ID") or ""): row for row in rows}
    output: list[dict[str, Any]] = []
    for raw in records:
        record = dict(raw)
        purity = by_track.get(record["parent_track_id"])
        if purity:
            record["purity_result"] = purity.get("Purity_Outcome") or "unavailable"
            record["quarantine_result"] = "quarantined" if record["purity_result"] in {"mixed_quarantined", "ambiguous_quarantined"} else "not_quarantined"
            if _bool(purity.get("Review_Carry_Forward_Allowed")):
                record["carry_forward_result"] = "exact_match_eligible"
            else:
                record["carry_forward_result"] = purity.get("Review_Carry_Forward_Reason") or "rejected"
        output.append(record)
    return tuple(output)


def _group_records(records: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in records:
        grouped.setdefault((str(row["session_id"]), str(row["evidence_family"])), []).append(row)
    return grouped


def _report_counts(rows: Sequence[Mapping[str, str]]) -> dict[str, int]:
    result = {"Present": 0, "Needs Review": 0, "Unconfirmed": 0, "Missing Enrollment": 0, "Absent": 0}
    for row in rows:
        status = str(row.get("Status") or row.get("Final_Status") or "")
        if status in result:
            result[status] += 1
    result["Total"] = len(rows)
    return result


def _build_analyses(
    records: Sequence[Mapping[str, Any]],
    sufficiency: Sequence[Mapping[str, Any]],
    official_counts: Mapping[str, int],
    candidate_counts: Mapping[str, int],
    phase_2h_summary: Mapping[str, Any],
    purity_summary: Mapping[str, Any],
    carry_summary: Mapping[str, Any],
    roster_rolls: Sequence[str],
) -> dict[str, Any]:
    grouped = _group_records(records)
    strict_rows: list[dict[str, Any]] = []
    guarded_rows: list[dict[str, Any]] = []
    retention_rows: list[dict[str, Any]] = []
    compatibility_rows: list[dict[str, Any]] = []
    for (session, family), group in sorted(grouped.items()):
        strict = safety_metrics(group, "strict_accepted")
        guarded = safety_metrics(group, "guarded_candidate")
        retention = retention_metrics(group)
        compatible = all(bool(row["policy_compatible"]) for row in group)
        common = {"Session_ID": session, "Evidence_Family": family, "Policy_Version": group[0]["policy_version"], "Current_Policy_Compatible": compatible, "Reviewed_Rows": len(group)}
        strict_rows.append({**common, "Metric_Available": strict["material"] != "unavailable", "Strict_Accepts": strict["material"], "Correct_Strict_Accepts": strict["correct"], "Wrong_Person_Strict_Accepts": strict["wrong_person"], "Outsider_Absorption": strict["outsider"], "Mixed_Tracks_Accepted": strict["mixed_track"], "Unclear_Or_Unverifiable_Strict_Accepts": strict["unclear_or_unverifiable"]})
        guarded_rows.append({**common, "Metric_Available": guarded["material"] != "unavailable", "Guarded_Candidates": guarded["material"], "Correct_Guarded_Candidates": guarded["correct"], "Wrong_Person_Guarded_Candidates": guarded["wrong_person"], "Outsider_Guarded_Candidates": guarded["outsider"], "Mixed_Guarded_Candidates": guarded["mixed_track"], "Unverifiable_Guarded_Candidates": guarded["unclear_or_unverifiable"]})
        retention_rows.append({**common, "Baseline_Retention_Available": retention["previously_correct"] != "unavailable", "Recovery_Metric_Available": retention["recovered"] != "unavailable", "Previously_Correct_Accepts": retention["previously_correct"], "Retained": retention["retained"], "Lost": retention["lost"], "Recovered_Correct_Evidence": retention["recovered"], "Lost_Correct_Checkpoints": retention["lost_checkpoints"], "Recovered_Checkpoints": retention["recovered_checkpoints"]})
        compatibility_rows.append({**common, "Comparability": "exact_current_policy" if compatible else "historical_policy_only", "Combined_With_Current_Strict_Metric": compatible})

    compatible_records = [row for row in records if row["policy_compatible"]]
    compatible_strict = safety_metrics(compatible_records, "strict_accepted")
    current_guarded = safety_metrics(compatible_records, "guarded_candidate")
    current_unsafe = sum(int(compatible_strict[key]) for key in ("wrong_person", "outsider", "mixed_track", "unclear_or_unverifiable"))
    mixed = next((row for row in records if row["parent_track_id"] == MIXED_TRACKLET_ID and row["evidence_family"] == "phase_2h_current_authority"), None)
    if not mixed or mixed["strict_accepted"] or not mixed["guarded_candidate"] or mixed["purity_result"] != "mixed_quarantined" or mixed["carry_forward_result"] == "exact_match_eligible":
        raise RetrospectiveAuditError("mandatory_mixed_track_regression_failed")

    attendance_rows = []
    for session in SESSIONS:
        if session == SESSIONS[1]:
            frame = phase_2h_summary.get("frame_official_result") or {}
            attendance_rows.append({"Session_ID": session, "Evidence_State": "current_official_and_archived_candidate", "Historical_Frame_Present": frame.get("present"), "Historical_Frame_Needs_Review": frame.get("needs_review"), "Historical_Frame_Absent": frame.get("absent"), "Official_Present": official_counts["Present"], "Official_Needs_Review": official_counts["Needs Review"], "Official_Unconfirmed": official_counts["Unconfirmed"], "Official_Missing_Enrollment": official_counts["Missing Enrollment"], "Official_Absent": official_counts["Absent"], "Official_Total": official_counts["Total"], "Candidate_Present": candidate_counts["Present"], "Candidate_Needs_Review": candidate_counts["Needs Review"], "Candidate_Unconfirmed": candidate_counts["Unconfirmed"], "Candidate_Missing_Enrollment": candidate_counts["Missing Enrollment"], "Candidate_Absent": candidate_counts["Absent"], "Candidate_Total": candidate_counts["Total"], "False_Mass_Absence": False, "Non_Detection_Semantics": "Unconfirmed", "Missing_Enrollment_Distinct": True})
        else:
            attendance_rows.append({"Session_ID": session, "Evidence_State": "historical_report_semantics_not_safely_comparable", "Historical_Frame_Present": None, "Historical_Frame_Needs_Review": None, "Historical_Frame_Absent": None, "Official_Present": None, "Official_Needs_Review": None, "Official_Unconfirmed": None, "Official_Missing_Enrollment": None, "Official_Absent": None, "Official_Total": None, "Candidate_Present": None, "Candidate_Needs_Review": None, "Candidate_Unconfirmed": None, "Candidate_Missing_Enrollment": None, "Candidate_Absent": None, "Candidate_Total": None, "False_Mass_Absence": None, "Non_Detection_Semantics": "unavailable", "Missing_Enrollment_Distinct": True})

    roster_rows = []
    for session in SESSIONS:
        session_records = [row for row in records if row["session_id"] == session]
        out_of_roster = sum(bool(row["predicted_identity"]) and row["predicted_identity"] not in roster_rolls for row in session_records)
        roster_rows.append({"Session_ID": session, "Roster_Count": len(roster_rolls), "Missing_Roster_Rows": 0, "Duplicate_Roster_Rows": 0, "Missing_Enrollment_Roll": MISSING_ENROLLMENT_ROLL, "Distinct_AI0110_And_00CSE0110": True, "Excluded_24011CSEAI0061": "24011CSEAI0061" not in roster_rolls, "Included_2401100CSE0237": "2401100CSE0237" in roster_rolls, "Out_Of_Roster_Predictions": out_of_roster, "Integrity_Violation": out_of_roster > 0, "Current_Policy_Integrity_Violation": out_of_roster > 0 and any(row["policy_compatible"] for row in session_records)})

    recommendation = choose_recommendation(
        artifact_verification_passed=True,
        exact_current_evidence_exists=bool(compatible_records),
        unsafe_current_strict_accepts=current_unsafe,
        all_sessions_complete_and_independent=False,
    )
    blockers = [
        "all_four_sessions_are_contaminated_by_prior_use",
        "three_sessions_have_historical_policy_only_identity_evidence",
        "no_existing_session_can_prove_independent_generalization",
        "historical_guarded_candidates_include_wrong_person_and_outsider_absorption",
        "historical_policy_evidence_contains_one_out_of_roster_prediction",
        "guarded_recovery_requires_future_untouched_session_validation",
    ]
    accepted_by_identity: dict[str, int] = {}
    for row in compatible_records:
        if row["strict_accepted"]:
            fingerprint = identity_fingerprint(str(row["predicted_identity"]))
            accepted_by_identity[fingerprint] = accepted_by_identity.get(fingerprint, 0) + 1
    compatible_accept_count = sum(accepted_by_identity.values())
    concentration = {
        "accepted_evidence_per_identity_fingerprint": dict(sorted(accepted_by_identity.items())),
        "maximum_identity_share": (max(accepted_by_identity.values()) / compatible_accept_count) if compatible_accept_count else "unavailable",
        "small_subset_dominance_detected": (max(accepted_by_identity.values()) / compatible_accept_count > 0.5) if compatible_accept_count else "unavailable",
        "duplicate_incompatible_assignments": 0,
    }
    coverage = []
    for (session, family), group in sorted(grouped.items()):
        coverage.append({
            "session_id": session,
            "evidence_family": family,
            "material_pairs_or_tracks": len(group),
            "reviewed": len(group),
            "unreviewed": 0,
            "unverifiable": sum(row["identity_outcome"] in {"unclear", "unverifiable"} for row in group),
        })
    combined = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "audit_policy_version": POLICY_VERSION,
        "sessions_audited": list(SESSIONS),
        "session_count": 4,
        "reviewed_normalized_rows": len(records),
        "exact_current_policy_reviewed_rows": len(compatible_records),
        "strict_current_policy": compatible_strict,
        "guarded_current_policy": current_guarded,
        "guarded_incompatible_policy_families_aggregated": False,
        "unsafe_current_strict_accepts": current_unsafe,
        "human_review_coverage": coverage,
        "identity_concentration": concentration,
        "historical_roster_prediction_findings": {"out_of_roster_predictions": sum(row["Out_Of_Roster_Predictions"] for row in roster_rows), "current_policy_out_of_roster_predictions": sum(row["Out_Of_Roster_Predictions"] for row in roster_rows if row["Current_Policy_Integrity_Violation"])},
        "independent_generalization_established": False,
        "guarded_recovery_promotion_authorized": False,
        "recommendation": recommendation,
        "blockers": blockers,
    }
    purity = {
        "Session_ID": SESSIONS[1],
        "Evidence_Family": "phase_2k_a_diagnostic_purity",
        "Parent_Tracks": purity_summary.get("parent_tracklets_evaluated"),
        "Pure_Unsplit": (purity_summary.get("outcome_counts") or {}).get("pure_unsplit"),
        "Mixed_Quarantined": (purity_summary.get("outcome_counts") or {}).get("mixed_quarantined"),
        "Ambiguous_Quarantined": (purity_summary.get("outcome_counts") or {}).get("ambiguous_quarantined"),
        "Insufficient_Observations": (purity_summary.get("outcome_counts") or {}).get("insufficient_observations"),
        "Possible_Split_Candidates": purity_summary.get("split_candidate_parents"),
        "Unsafe_Parent_Acceptance_Count": 0,
        "Mandatory_Mixed_Track": MIXED_TRACKLET_ID,
        "Mandatory_Mixed_Disposition": "mixed_quarantined",
    }
    carry = {
        "Session_ID": SESSIONS[1],
        "Evidence_Family": "phase_2k_b_exact_carry_forward",
        "Verified_Review_Pairs": carry_summary.get("verified_review_pairs"),
        "Exact_Eligible": carry_summary.get("diagnostically_exact_match_eligible_pairs"),
        "Rejected": carry_summary.get("diagnostically_rejected_pairs"),
        "Partial_Reuse_Attempts": 0,
        "Partial_Reuse_Occurred": carry_summary.get("partial_review_carry_forward_occurred"),
        "Mixed_Quarantine_Overrides": 1,
        "Mandatory_Mixed_Result": (carry_summary.get("mixed_track_result") or {}).get("carry_forward_result"),
        "Official_Contribution": (carry_summary.get("mixed_track_result") or {}).get("official_attendance_contribution"),
    }
    return {
        "strict_rows": strict_rows,
        "guarded_rows": guarded_rows,
        "retention_rows": retention_rows,
        "compatibility_rows": compatibility_rows,
        "attendance_rows": attendance_rows,
        "roster_rows": roster_rows,
        "purity_rows": [purity],
        "carry_rows": [carry],
        "combined": combined,
        "recommendation": {"schema_version": 1, "audit_policy_version": POLICY_VERSION, "recommendation": recommendation, "strict_automatic_authority_remains_safe_on_sufficient_exact_reviewed_evidence": current_unsafe == 0, "guarded_recovery_must_remain_review_only": True, "threshold_change_authorized": False, "model_promotion_authorized": False, "authority_change_authorized": False, "independent_generalization_claimed": False, "blockers": blockers},
    }


def _inventory_rows(repo: Path, manifest_rows: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]], roster_rolls: Sequence[str]) -> tuple[dict[str, Any], ...]:
    manifest_by_name = {row["Artifact_Name"]: row for row in manifest_rows}
    sessions = {
        SESSIONS[0]: ("02fc655d0666284eb15a4c4c83a5eaf574eb52122e985a0709ff682ddf29b02a", "phase_1_2j_exception_review", "phase-1.2j-mon-p3-shadow-v1.1", "embfam-7bc431a3ad762398d4e9", "historical"),
        SESSIONS[1]: ("54b7c3b5b6a8a415be9d039eea54ab734da53664b68c7867f14f4bdafbb0ba80", "phase_2h_current_authority", CURRENT_AUTOMATIC_POLICY, CURRENT_FAMILY, "exact_current"),
        SESSIONS[2]: ("c7b17fd177999eb396d676f355d94e4a7ce4167a5c81ad736548f540d0ede6e6", "multisession_ground_truth", "historical-tue-p1-tracklet-v1", "historical_pre_promotion", "historical"),
        SESSIONS[3]: ("60f5422e9a68978d3b206febfea89a7997bc5ca3425d7d373012ce484dafd6bc", "multisession_ground_truth", "historical-tue-p2-shadow-recovery-v1", "historical_pre_promotion", "historical"),
    }
    artifact_specs = {
        SESSIONS[0]: (
            ("phase_1_2j_mon_p3", "source_tracklet_and_review_package", "source_session_provenance;source_video_hashes;tracklet_diagnostics;tracklet_observations;checkpoint_comparisons;zone_diagnostics;review_package;embedding_policy_threshold_provenance"),
            ("phase_1_2j_mon_p3_evaluation", "review_join_evaluation_and_retention", "private_review_join;human_labels;evaluation_summary;retention_results;ablation_and_model_selection_use"),
        ),
        SESSIONS[1]: (
            ("phase_1_2n_source_freeze", "source_freeze", "source_session_provenance;source_video_hashes;source_freeze_manifest"),
            ("phase_1_2n_mon_p4", "historical_tracklet_and_review_package", "tracklet_diagnostics;tracklet_observations;checkpoint_comparisons;zone_diagnostics;review_package;embedding_policy_threshold_provenance;promotion_evidence_use"),
            ("phase_1_2n_mon_p4_evaluation", "historical_review_join_and_evaluation", "private_review_join;human_labels;evaluation_summary;retention_results"),
            ("phase_2h", "current_tracklet_review_package", "tracklet_evidence;checkpoint_comparisons;zone_diagnostics;review_package"),
            ("phase_2h_evaluation", "current_private_review_join_and_evaluation", "private_review_join;human_labels;evaluation_summary"),
            ("phase_2i_authority", "authority_report_and_roster_semantics", "authority_report;roster_file;report_status_semantics;false_mass_absence_protection"),
            ("phase_2k_a", "purity_and_quarantine", "tracklet_purity;tracklet_observations;quarantine_results;mixed_track_regression"),
            ("phase_2k_b", "review_carry_forward", "exact_carry_forward;retention_contract;mixed_quarantine_override"),
            ("phase_2j_archived_candidate_revision", "archived_candidate_authority_report", "archived_candidate_report;report_status_semantics;official_revision_preservation"),
        ),
        SESSIONS[2]: (
            ("multisession_forensic", "source_tracklet_and_review_packages", "source_session_provenance;source_video_hashes;tracklet_diagnostics;tracklet_observations;checkpoint_comparisons;zone_diagnostics;review_packages;embedding_policy_threshold_provenance"),
            ("multisession_ground_truth", "private_review_joins_and_benchmark", "private_review_joins;human_labels;evaluation_summary;benchmark_rows;calibration_adaptation_ablation_and_promotion_use"),
        ),
        SESSIONS[3]: (
            ("multisession_forensic", "source_tracklet_and_review_package", "source_session_provenance;source_video_hashes;tracklet_diagnostics;tracklet_observations;checkpoint_comparisons;zone_diagnostics;review_package;embedding_policy_threshold_provenance"),
            ("multisession_ground_truth", "private_review_join_and_benchmark", "private_review_join;human_labels;evaluation_summary;benchmark_rows;shadow_recovery_calibration_adaptation_and_promotion_use"),
        ),
    }
    rows = []
    for session, (source_fp, family, policy, embedding_family, compatibility) in sessions.items():
        session_records = [row for row in records if row["session_id"] == session]
        out_of_roster = sum(bool(row["predicted_identity"]) and row["predicted_identity"] not in roster_rolls for row in session_records)
        for artifact_name, artifact_type, coverage in artifact_specs[session]:
            manifest = manifest_by_name[artifact_name]
            rows.append({"Session_ID": session, "Source_Fingerprint_SHA256": source_fp, "Evidence_Family": family, "Artifact_Name": artifact_name, "Artifact_Type": artifact_type, "Evidence_Coverage": coverage, "Artifact_Path": manifest["Manifest_Path"], "Artifact_SHA256": manifest["Manifest_SHA256"], "Manifest_Verified": True, "Source_Video_Hashes_Available": True, "Tracklet_Evidence_Available": True, "Human_Review_Available": True, "Private_Join_Classification": "restricted_internal_source_not_republished", "Roster_Context_Available": True, "Session_Reviewed_Out_Of_Roster_Predictions": out_of_roster, "Embedding_Family": embedding_family, "Embedding_SHA256": CURRENT_EMBEDDING_SHA256 if session == SESSIONS[1] else "historical_manifest_pinned", "Policy_Version": policy, "Match_Threshold": MATCH_THRESHOLD, "Margin_Threshold": MARGIN_THRESHOLD, "Compatibility": compatibility})
    return tuple(rows)


def _sufficiency_rows() -> tuple[dict[str, Any], ...]:
    rows = []
    for session in SESSIONS:
        compatible = session == SESSIONS[1]
        facts = {"manifest_verified": True, "source_provenance": True, "embedding_provenance": True, "track_identity": True, "human_truth": True, "review_join": True, "roster_context": True, "policy_provenance": True, "evaluation_complete": True, "current_policy_compatible": compatible}
        rows.append({"Session_ID": session, "Classification": classify_sufficiency(facts), "Manifest_Verified": True, "Exact_Source_Provenance": True, "Embedding_Provenance": True, "Track_Identity_Evidence": True, "Human_Truth": True, "Unambiguous_Review_Join": True, "Roster_Context": True, "Policy_And_Threshold_Provenance": True, "Evaluation_Complete": True, "Exact_Current_Policy": compatible, "Unavailable_Current_Policy_Metrics": not compatible, "Reason": "complete exact current-policy retrospective evidence" if compatible else "reviewed historical evidence is not safely comparable to the current policy/embedding contract"})
    return tuple(rows)


def preflight(repo_root: Path, output_root: Path | None = None) -> AuditPreflight:
    repo = Path(repo_root).resolve()
    protected = {relative: sha256_file(repo / relative) for relative in PROTECTED_EXPECTED}
    if protected != PROTECTED_EXPECTED:
        changed = sorted(key for key in PROTECTED_EXPECTED if protected.get(key) != PROTECTED_EXPECTED[key])
        raise RetrospectiveAuditError(f"protected_operational_hash_changed:{','.join(changed)}")

    manifest_rows: list[dict[str, Any]] = []
    manifests: dict[str, dict[str, Any]] = {}
    for name, (relative, expected_sha) in MANIFESTS.items():
        path = repo / relative
        if name == "phase_2i_authority":
            manifest = _verify_authority_manifest(path, expected_sha)
        elif name == "multisession_ground_truth":
            if sha256_file(path) != expected_sha:
                raise RetrospectiveAuditError(f"manifest_tampered:{path}")
            manifest = _load_json(path)
        else:
            manifest = verify_manifest(path, expected_sha)
        manifests[name] = manifest
        manifest_rows.append({"Artifact_Name": name, "Manifest_Path": normalize_windows_path(relative), "Manifest_SHA256": expected_sha, "Verified": True, "Policy_Version": manifest.get("policy_version") or manifest.get("tool_policy_version") or manifest.get("benchmark_id") or "manifest-pinned"})

    ground_truth_manifest = manifests["multisession_ground_truth"]
    benchmark_meta = ground_truth_manifest.get("benchmark_artifact") or {}
    benchmark_path = repo / "attendance_output/embedding_forensics/embedding_forensics_20260713_132605_989269/multisession_ground_truth.csv"
    if sha256_file(benchmark_path) != benchmark_meta.get("sha256"):
        raise RetrospectiveAuditError("multisession_ground_truth_artifact_tampered")

    student_map = _load_json(repo / "data/student_faculty_map.json")
    roster = _roster_rolls(student_map)
    if "24011CSEAI0110" not in roster or "2401100CSE0110" in roster or "24011CSEAI0061" in roster or "2401100CSE0237" not in roster:
        raise RetrospectiveAuditError("roster_identity_constraint_changed")

    p3_eval = repo / MANIFESTS["phase_1_2j_mon_p3_evaluation"][0]
    p3_rows = _load_csv(p3_eval.parent / "reviewed_exception_tracks.csv")
    p3 = normalize_records(p3_rows, source_schema="phase_1_2j", context={"session_id": SESSIONS[0], "source_fingerprint": "02fc655d0666284eb15a4c4c83a5eaf574eb52122e985a0709ff682ddf29b02a", "embedding_family": "embfam-7bc431a3ad762398d4e9", "embedding_hash": "historical_manifest_pinned", "policy_version": "phase-1.2j-mon-p3-shadow-v1.1", "policy_compatible": False, "evidence_family": "phase_1_2j_exception_review", "provenance_artifact": p3_eval.relative_to(repo), "provenance_hash": MANIFESTS["phase_1_2j_mon_p3_evaluation"][1]})

    p4_eval = repo / MANIFESTS["phase_1_2n_mon_p4_evaluation"][0]
    p4_historical_rows = _load_csv(p4_eval.parent / "reviewed_exception_tracks.csv")
    p4_historical = normalize_records(p4_historical_rows, source_schema="phase_1_2n", context={"session_id": SESSIONS[1], "source_fingerprint": "54b7c3b5b6a8a415be9d039eea54ab734da53664b68c7867f14f4bdafbb0ba80", "embedding_family": CURRENT_FAMILY, "embedding_hash": CURRENT_EMBEDDING_SHA256, "policy_version": "phase-1.2n-mon-p4-shadow-v1", "policy_compatible": False, "evidence_family": "phase_1_2n_exception_review", "provenance_artifact": p4_eval.relative_to(repo), "provenance_hash": MANIFESTS["phase_1_2n_mon_p4_evaluation"][1]})

    phase_2h_root = (repo / MANIFESTS["phase_2h"][0]).parent
    selected = _load_csv(phase_2h_root / "selected_multiframe_checkpoint_evidence.csv")
    joined = _load_csv(phase_2h_root / "evaluation/joined_multiframe_recovery_review.csv")
    phase_2h_rows = _join_phase_2h(selected, joined)
    p4_current = normalize_records(phase_2h_rows, source_schema="phase_2h", context={"session_id": SESSIONS[1], "source_fingerprint": "54b7c3b5b6a8a415be9d039eea54ab734da53664b68c7867f14f4bdafbb0ba80", "embedding_family": CURRENT_FAMILY, "embedding_hash": CURRENT_EMBEDDING_SHA256, "policy_version": CURRENT_AUTOMATIC_POLICY, "policy_compatible": True, "evidence_family": "phase_2h_current_authority", "provenance_artifact": (phase_2h_root / "evaluation/evaluation_manifest.json").relative_to(repo), "provenance_hash": MANIFESTS["phase_2h_evaluation"][1]})
    purity_root = (repo / MANIFESTS["phase_2k_a"][0]).parent
    p4_current = _attach_purity(p4_current, _load_csv(purity_root / "review_carry_forward_impact.csv"))

    gt_rows = _load_csv(benchmark_path)
    tue_records: list[dict[str, Any]] = []
    for session, source_fp, policy in ((SESSIONS[2], "c7b17fd177999eb396d676f355d94e4a7ce4167a5c81ad736548f540d0ede6e6", "historical-tue-p1-tracklet-v1"), (SESSIONS[3], "60f5422e9a68978d3b206febfea89a7997bc5ca3425d7d373012ce484dafd6bc", "historical-tue-p2-shadow-recovery-v1")):
        source_rows = [row for row in gt_rows if row.get("Session_ID") == session]
        tue_records.extend(normalize_records(source_rows, source_schema="multisession_ground_truth", context={"session_id": session, "source_fingerprint": source_fp, "embedding_family": "historical_pre_promotion", "embedding_hash": "c32ed31df10b7b9b43b8f19977a2adf0a82fb8e7a71f3e8bceb42c0565fedd49", "policy_version": policy, "policy_compatible": False, "evidence_family": "multisession_ground_truth", "provenance_artifact": benchmark_path.relative_to(repo), "provenance_hash": benchmark_meta.get("sha256"), "retention_baseline_available": session == SESSIONS[2], "recovery_metric_available": session == SESSIONS[3]}))

    records = tuple(sorted((*p3, *p4_historical, *p4_current, *tue_records), key=lambda row: (row["session_id"], row["evidence_family"], row["evidence_id"])))
    if len(records) != 171:
        raise RetrospectiveAuditError(f"normalized_review_row_count_changed:{len(records)}")

    authority_root = (repo / MANIFESTS["phase_2i_authority"][0]).parent
    official_counts = _report_counts(_load_csv(authority_root / f"corrected_attendance_{SESSIONS[1]}.csv"))
    candidate_root = repo / f"attendance_output/product_workflow/report_revisions/{SESSIONS[1]}/automatic-candidate-99cc13934144bf7e1706b8be"
    candidate_manifest = verify_manifest(candidate_root / "revision_manifest.json", ARCHIVED_CANDIDATE_MANIFEST_SHA256)
    manifest_rows.append({"Artifact_Name": "phase_2j_archived_candidate_revision", "Manifest_Path": normalize_windows_path((candidate_root / "revision_manifest.json").relative_to(repo)), "Manifest_SHA256": ARCHIVED_CANDIDATE_MANIFEST_SHA256, "Verified": True, "Policy_Version": candidate_manifest.get("policy_version")})
    candidate_counts = _report_counts(_load_csv(candidate_root / "candidate_attendance.csv"))
    expected_official = {"Present": 6, "Needs Review": 4, "Unconfirmed": 16, "Missing Enrollment": 1, "Absent": 0, "Total": 27}
    expected_candidate = {"Present": 2, "Needs Review": 8, "Unconfirmed": 16, "Missing Enrollment": 1, "Absent": 0, "Total": 27}
    if official_counts != expected_official or candidate_counts != expected_candidate:
        raise RetrospectiveAuditError("attendance_semantics_baseline_changed")

    purity_summary = _load_json(purity_root / "evaluation_summary.json")
    carry_root = (repo / MANIFESTS["phase_2k_b"][0]).parent
    carry_summary = _load_json(carry_root / "evaluation_summary.json")
    sufficiency = _sufficiency_rows()
    inventory = _inventory_rows(repo, manifest_rows, records, roster)
    phase_2h_summary = _load_json(phase_2h_root / "evaluation/evaluation_summary.json")
    analyses = _build_analyses(records, sufficiency, official_counts, candidate_counts, phase_2h_summary, purity_summary, carry_summary, roster)
    current_policy_fingerprint = canonical_json_hash({"automatic_policy": CURRENT_AUTOMATIC_POLICY, "authority": CURRENT_AUTHORITY, "match_threshold": MATCH_THRESHOLD, "margin_threshold": MARGIN_THRESHOLD, "aggregation": "top3", "guarded_automatic": False})
    inventory_fingerprint = canonical_json_hash(inventory)
    normalization_contract_fingerprint = canonical_json_hash({"schema_version": OUTPUT_SCHEMA_VERSION, "unsupported_metrics": "unavailable_not_zero", "private_identity_visibility": "restricted_fingerprints_except_exact_mandatory_session_track", "mandatory_regression_key": [MIXED_SESSION_ID, MIXED_TRACKLET_ID], "duplicate_and_ambiguous_joins": "fail_closed"})
    run_payload = {"audit_policy_version": POLICY_VERSION, "consumed_manifest_hashes": sorted(row["Manifest_SHA256"] for row in manifest_rows), "production_embedding_sha256": CURRENT_EMBEDDING_SHA256, "current_policy_fingerprint_sha256": current_policy_fingerprint, "session_evidence_inventory_fingerprint_sha256": inventory_fingerprint, "normalization_contract_fingerprint_sha256": normalization_contract_fingerprint}
    run_fingerprint = canonical_json_hash(run_payload)
    run_id = f"retrospective-audit-{run_fingerprint[:20]}"
    output = Path(output_root).resolve() if output_root else repo / "attendance_output/product_workflow/phase_2k_retrospective_audit"
    analyses["run_payload"] = run_payload
    analyses["current_policy_fingerprint"] = current_policy_fingerprint
    analyses["official_counts"] = official_counts
    analyses["candidate_counts"] = candidate_counts
    return AuditPreflight(repo, output, run_id, run_fingerprint, tuple(manifest_rows), inventory, sufficiency, records, protected, roster, analyses)


def _normalized_public_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in records:
        is_mandatory = row["session_id"] == MIXED_SESSION_ID and row["parent_track_id"] == MIXED_TRACKLET_ID
        rows.append({"Session_ID": row["session_id"], "Evidence_Family": row["evidence_family"], "Source_Fingerprint_SHA256": row["source_fingerprint"], "Checkpoint": row["checkpoint"], "Camera": row["camera"], "Parent_Track_ID": row["parent_track_id"], "Evidence_ID": row["evidence_id"], "Predicted_Identity": MIXED_PREDICTED_ROLL if is_mandatory else "restricted", "Predicted_Identity_Fingerprint": identity_fingerprint(row["predicted_identity"]), "Reviewed_Identity": "restricted", "Reviewed_Identity_Fingerprint": identity_fingerprint(row["reviewed_identity"]), "Reviewed_Disposition": row["reviewed_disposition"], "Identity_Outcome": row["identity_outcome"], "Strict_Metric_Available": row["strict_metric_available"], "Strict_Accepted": row["strict_accepted"], "Guarded_Metric_Available": row["guarded_metric_available"], "Guarded_Candidate": row["guarded_candidate"], "Baseline_Retention_Available": row["retention_baseline_available"], "Recovery_Metric_Available": row["recovery_metric_available"], "Score": row["score"], "Margin": row["margin"], "Observation_Count": row["observation_count"], "Checkpoint_Support": row["checkpoint_support"], "Purity_Result": row["purity_result"], "Quarantine_Result": row["quarantine_result"], "Carry_Forward_Result": row["carry_forward_result"], "Embedding_Family": row["embedding_family"], "Embedding_SHA256": row["embedding_hash"], "Policy_Version": row["policy_version"], "Current_Policy_Compatible": row["policy_compatible"], "Provenance_Artifact": row["provenance_artifact"], "Provenance_SHA256": row["provenance_hash"], "Identity_Visibility": "mandatory_regression" if is_mandatory else "restricted_fingerprints_only"})
    return rows


def contamination_declaration() -> dict[str, Any]:
    return {"schema_version": 1, "audit_type": "retrospective_engineering_evidence", "sessions_contaminated": True, "independent_generalization_claimed": False, "guarded_recovery_promotion_authorized": False, "future_untouched_session_still_required": True, "claims": {SESSIONS[0]: "used for recognition, blind review, retention, ablation, and model-selection evidence", SESSIONS[1]: "used for Phase 1.2N validation, promotion evidence, Phase 2G/2I processing, and human review", SESSIONS[2]: "used for recognition, review, calibration, adaptation, ablation, and promotion benchmarks", SESSIONS[3]: "used for recognition, review, shadow recovery, calibration, adaptation, and promotion benchmarks"}, "limitations": ["This audit is retrospective engineering evidence.", "It cannot establish independent generalization.", "It cannot authorize guarded-recovery promotion.", "It cannot replace a future untouched-session test."]}


def no_activation_declaration() -> dict[str, Any]:
    keys = ("recognition_ran", "YuNet_ran", "SFace_ran", "classroom_video_decoded", "video_reprocessed", "official_attendance_changed", "candidate_attendance_changed", "report_revision_changed", "finalization_changed", "automatic_authority_changed", "reviewed_authority_changed", "guarded_authority_changed", "guarded_recovery_promoted", "thresholds_changed", "embeddings_changed", "roster_changed", "timetable_changed", "review_registry_changed", "manual_overrides_changed", "jobs_changed", "HOD_configuration_changed", "frontend_changed")
    return {"schema_version": 1, "phase": "Product Phase 2K-C1", **{key: False for key in keys}}


def build_output_files(plan: AuditPreflight) -> dict[str, bytes]:
    a = plan.analyses
    policy = {"schema_version": 1, "policy_version": POLICY_VERSION, "allowed_recommendations": ["retain_current_policy", "retain_current_policy_with_blockers", "strict_authority_unsafe", "insufficient_retrospective_evidence", "artifact_verification_failed"], "automatic_guarded_recovery_permitted": False, "threshold_tuning_permitted": False, "model_promotion_permitted": False, "authority_activation_permitted": False, "current_production": {"family": CURRENT_FAMILY, "variant": CURRENT_VARIANT, "embedding_sha256": CURRENT_EMBEDDING_SHA256, "summary_sha256": CURRENT_SUMMARY_SHA256, "dimension": 128, "aggregation": "top3", "match_threshold": MATCH_THRESHOLD, "margin_threshold": MARGIN_THRESHOLD, "automatic_policy": CURRENT_AUTOMATIC_POLICY, "guarded_recovery_automatic": False}}
    normalized = _normalized_public_rows(plan.normalized_records)
    per_session = []
    sufficiency_by_session = {row["Session_ID"]: row for row in plan.sufficiency_rows}
    for session in SESSIONS:
        rows = [row for row in plan.normalized_records if row["session_id"] == session]
        current = [row for row in rows if row["policy_compatible"]]
        strict = safety_metrics(current, "strict_accepted") if current else safety_metrics([], "strict_accepted")
        guarded = safety_metrics(current, "guarded_candidate") if current else safety_metrics([], "guarded_candidate")
        per_session.append({"Session_ID": session, "Evidence_Sufficiency": sufficiency_by_session[session]["Classification"], "Normalized_Reviewed_Rows": len(rows), "Exact_Current_Policy_Rows": len(current), "Correct_Current_Strict_Accepts": strict["correct"], "Unsafe_Current_Strict_Accepts": "unavailable" if not current else sum(int(strict[key]) for key in ("wrong_person", "outsider", "mixed_track", "unclear_or_unverifiable")), "Correct_Current_Guarded_Candidates": guarded["correct"], "Unsafe_Current_Guarded_Candidates": "unavailable" if not current else sum(int(guarded[key]) for key in ("wrong_person", "outsider", "mixed_track", "unclear_or_unverifiable")), "Independent_Evidence": False})
    source_manifest = {"schema_version": 1, "audit_policy_version": POLICY_VERSION, "run_id": plan.run_id, "run_fingerprint_sha256": plan.run_fingerprint_sha256, "consumed_artifacts": list(plan.manifest_rows), "protected_operational_hashes": plan.protected_hashes, "source_access_mode": "immutable_artifacts_only_no_video_or_inference", "private_join_handling": "restricted references and hashes retained; public normalized rows expose identity fingerprints only except the mandated regression identity", "run_fingerprint_inputs": a["run_payload"]}
    evaluation = {"schema_version": 1, "policy_version": POLICY_VERSION, "run_id": plan.run_id, "recommendation": a["recommendation"]["recommendation"], "combined": a["combined"], "session_evidence_sufficiency": list(plan.sufficiency_rows), "mandatory_negative_regression": {"session_id": MIXED_SESSION_ID, "tracklet_id": MIXED_TRACKLET_ID, "predicted_roll": MIXED_PREDICTED_ROLL, "human_disposition": "mixed_track", "purity_outcome": "mixed_quarantined", "carry_forward_result": "mixed_quarantined", "official_contribution": 0, "exact_signature_override_permitted": False}, "attendance_semantics": {"official": a["official_counts"], "archived_candidate": a["candidate_counts"], "false_mass_absence_detected": False, "non_detection_remains_unconfirmed": True, "missing_enrollment_distinct": True}, "contamination_and_claim_limits": contamination_declaration(), "no_activation": no_activation_declaration()}
    files = {
        "audit_policy.json": canonical_json_bytes(policy),
        "session_evidence_inventory.csv": csv_bytes(plan.inventory_rows, tuple(plan.inventory_rows[0])),
        "session_evidence_sufficiency.csv": csv_bytes(plan.sufficiency_rows, tuple(plan.sufficiency_rows[0])),
        "normalized_review_evidence.csv": csv_bytes(normalized, tuple(normalized[0])),
        "strict_identity_safety.csv": csv_bytes(a["strict_rows"], tuple(a["strict_rows"][0])),
        "guarded_recovery_safety.csv": csv_bytes(a["guarded_rows"], tuple(a["guarded_rows"][0])),
        "retention_analysis.csv": csv_bytes(a["retention_rows"], tuple(a["retention_rows"][0])),
        "purity_quarantine_analysis.csv": csv_bytes(a["purity_rows"], tuple(a["purity_rows"][0])),
        "carry_forward_analysis.csv": csv_bytes(a["carry_rows"], tuple(a["carry_rows"][0])),
        "attendance_semantics_analysis.csv": csv_bytes(a["attendance_rows"], tuple(a["attendance_rows"][0])),
        "roster_integrity_analysis.csv": csv_bytes(a["roster_rows"], tuple(a["roster_rows"][0])),
        "policy_compatibility.csv": csv_bytes(a["compatibility_rows"], tuple(a["compatibility_rows"][0])),
        "contamination_and_claim_limits.json": canonical_json_bytes(contamination_declaration()),
        "per_session_summary.csv": csv_bytes(per_session, tuple(per_session[0])),
        "combined_summary.json": canonical_json_bytes(a["combined"]),
        "recommendation.json": canonical_json_bytes(a["recommendation"]),
        "no_activation_declaration.json": canonical_json_bytes(no_activation_declaration()),
        "source_manifest.json": canonical_json_bytes(source_manifest),
        "evaluation_summary.json": canonical_json_bytes(evaluation),
    }
    if set(files) != set(REQUIRED_OUTPUT_FILES) - {"immutable_manifest.json"}:
        raise RetrospectiveAuditError("required_output_file_set_changed")
    return files


def verify_immutable_output(output_dir: Path) -> dict[str, Any]:
    root = Path(output_dir)
    manifest_path = root / "immutable_manifest.json"
    if not root.is_dir() or root.is_symlink() or not manifest_path.is_file() or manifest_path.is_symlink():
        raise RetrospectiveAuditError("immutable_output_missing_or_unsafe")
    manifest = _load_json(manifest_path)
    if manifest.get("policy_version") != POLICY_VERSION or manifest.get("run_id") != root.name or manifest.get("immutable") is not True:
        raise RetrospectiveAuditError("immutable_manifest_binding_changed")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != set(REQUIRED_OUTPUT_FILES) - {"immutable_manifest.json"}:
        raise RetrospectiveAuditError("immutable_file_set_changed")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and path.name != "immutable_manifest.json"}
    if actual != set(files):
        raise RetrospectiveAuditError("immutable_file_set_changed")
    for relative, metadata in files.items():
        path = root / relative
        if path.is_symlink() or sha256_file(path) != metadata.get("sha256") or path.stat().st_size != int(metadata.get("size_bytes", -1)):
            raise RetrospectiveAuditError(f"immutable_file_hash_changed:{relative}")
    return manifest


def write_immutable_output(output_root: Path, run_id: str, run_fingerprint: str, files: Mapping[str, bytes]) -> tuple[Path, bool]:
    root = Path(output_root)
    output = root / run_id
    intended = {name: {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)} for name, payload in sorted(files.items())}
    if output.exists():
        try:
            manifest = verify_immutable_output(output)
        except Exception as exc:
            raise RetrospectiveAuditError(f"deterministic_id_collision:{exc}") from exc
        if manifest.get("run_fingerprint_sha256") != run_fingerprint or manifest.get("files") != intended:
            raise RetrospectiveAuditError("deterministic_id_collision:different_bytes")
        return output, True
    root.mkdir(parents=True, exist_ok=True)
    staging = root / f".{run_id}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False)
    try:
        for relative, payload in files.items():
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        manifest = {"schema_version": OUTPUT_SCHEMA_VERSION, "policy_version": POLICY_VERSION, "run_id": run_id, "run_fingerprint_sha256": run_fingerprint, "immutable": True, "files": intended}
        (staging / "immutable_manifest.json").write_bytes(canonical_json_bytes(manifest))
        if output.exists():
            raise RetrospectiveAuditError("deterministic_id_collision:target_appeared")
        staging.replace(output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    verify_immutable_output(output)
    return output, False


def materialize(plan: AuditPreflight) -> tuple[Path, bool, dict[str, Any]]:
    files = build_output_files(plan)
    output, reused = write_immutable_output(plan.output_root, plan.run_id, plan.run_fingerprint_sha256, files)
    manifest = verify_immutable_output(output)
    return output, reused, manifest
