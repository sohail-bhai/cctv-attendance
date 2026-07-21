from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .recall_analysis import canonical_roll
from .workflow_state import atomic_write_json, read_json_object

POLICY_VERSION = "product-phase-2i-reviewed-multiframe-authority-v1"
AUTOMATIC_POLICY_VERSION = "product-phase-2i-strict-tracklet-authority-v1"
AUTHORITY_REVISION_ID = "phase-2i-mon-p4-reviewed-multiframe-7d9d3a539c0b32e33048"
SESSION_ID = "2026-06-22__B51__P4__CVO"
PHASE_2H_OUTPUT_ID = "multiframe-recovery-7d9d3a539c0b32e33048"
MISSING_ENROLLMENT_ROLL = "2401100CSE0268"
OFFICIAL_RECOGNITION_AUTHORITY = "reviewed_multiframe_tracklet_evidence"
AUTOMATIC_RECOGNITION_AUTHORITY = "strict_tracklet_aggregate_with_guarded_review_candidates"
EXPECTED_PRESENT = (
    "2401100CSE0016",
    "2401100CSE0019",
    "2401100CSE0050",
    "2401100CSE0060",
    "2401100CSE0140",
    "24011CSEAI0007",
)
EXPECTED_NEEDS_REVIEW = (
    "2401100CSE0028",
    "2401100CSE0044",
    "2401100CSE0124",
    "24011CSEAI0051",
)
EXPECTED_COUNTS = {
    "present": 6,
    "needs_review": 4,
    "unconfirmed": 16,
    "missing_enrollment": 1,
    "absent": 0,
    "total": 27,
}
EXPECTED_EVALUATION_HASHES = {
    "evaluation_summary.json": "bf5bc857b8c35f0e6e0acb0da354d8a64be48b7d6dcb05bf37836e1860b587cb",
    "identity_checkpoint_validation.csv": "838a2d2bfac7d51798e9f6e303c017b647cc5a1c26cf97418c029676ced011fa",
    "joined_multiframe_recovery_review.csv": "2b2a2119d3eb44ea4aa9a3c9256f6a3173303944143684d59abd7097a32f2f46",
    "multiframe_shadow_roster_report.csv": "a2e25dc646c691d7c8627974c823d5f7dfef51510c1524e26bd61a732a8c881a",
}
EXPECTED_SOURCE_ENTRY_SHA256 = "30de92eabfb3eb6e49748147d06d6caa3e8b2db788ac524f5de2a87b4bbce8d9"
MIXED_TRACKLET_ID = "CP1-cam5-back-TRK00005"
MIXED_TRACKLET_ROLL = "24011CSEAI0051"

RECOVERY_RULE_ID = "mfrr-v1-score040-margin005-vote70-obs5-pair060-repeat2"
RECOVERY_MIN_SCORE = 0.40
RECOVERY_MIN_MARGIN = 0.05
RECOVERY_MIN_DOMINANT_SHARE_PCT = 70.0
RECOVERY_MIN_OBSERVATIONS = 5
RECOVERY_MIN_SELECTED_OBSERVATIONS = 3
RECOVERY_MIN_CONSISTENT_EMBEDDINGS = 3
RECOVERY_MIN_PAIRWISE_MEDIAN = 0.60
RECOVERY_MIN_CHECKPOINT_REPETITION = 2


class ProductPhase2IError(RuntimeError):
    """Raised when the authority revision cannot be proven safe."""


@dataclass(frozen=True)
class Inputs:
    repo_root: Path
    status_path: Path
    overrides_path: Path
    student_map_path: Path
    evaluation_dir: Path
    output_root: Path
    state_backup_dir: Path


@dataclass(frozen=True)
class PreflightResult:
    status_store: dict[str, Any]
    source_entry: dict[str, Any]
    source_rows: list[dict[str, Any]]
    corrected_rows: list[dict[str, Any]]
    summary: dict[str, int | float]
    source_entry_sha256: str
    evaluation_files: dict[str, dict[str, Any]]


def default_inputs(repo_root: Path) -> Inputs:
    root = Path(repo_root).resolve()
    return Inputs(
        repo_root=root,
        status_path=root / "data" / "attendance_status.json",
        overrides_path=root / "data" / "manual_overrides.json",
        student_map_path=root / "data" / "student_faculty_map.json",
        evaluation_dir=(
            root
            / "attendance_output"
            / "product_workflow"
            / "phase_2h_multiframe_recovery"
            / PHASE_2H_OUTPUT_ID
            / "evaluation"
        ),
        output_root=root / "attendance_output" / "product_workflow" / "phase_2i_authority",
        state_backup_dir=root / "data" / "state_backups",
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _entry_hash(entry: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json_bytes(entry))


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not Path(path).is_file():
        raise ProductPhase2IError(f"{label} not found: {path}")
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise ProductPhase2IError(f"Could not read {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProductPhase2IError(f"{label} must contain a JSON object")
    return payload


def _load_csv(path: Path, label: str) -> pd.DataFrame:
    if not Path(path).is_file():
        raise ProductPhase2IError(f"{label} not found: {path}")
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except Exception as exc:
        raise ProductPhase2IError(f"Could not read {label}: {exc}") from exc


def _boolish(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _intish(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value or "").strip()))
    except (TypeError, ValueError):
        return default


def _floatish(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value or "").strip())
    except (TypeError, ValueError):
        return default


def _verify_evaluation(inputs: Inputs) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    evaluation_dir = Path(inputs.evaluation_dir)
    manifest = _load_json(evaluation_dir / "evaluation_manifest.json", "Phase 2H evaluation manifest")
    summary = _load_json(evaluation_dir / "evaluation_summary.json", "Phase 2H evaluation summary")
    files_meta: dict[str, dict[str, Any]] = {}

    recorded = manifest.get("files")
    if not isinstance(recorded, dict):
        raise ProductPhase2IError("Phase 2H evaluation manifest has no file map")
    for name, expected_hash in EXPECTED_EVALUATION_HASHES.items():
        path = evaluation_dir / name
        if not path.is_file():
            raise ProductPhase2IError(f"Required Phase 2H evaluation file is missing: {path}")
        actual_hash = _sha256_file(path)
        if actual_hash != expected_hash:
            raise ProductPhase2IError(
                f"Phase 2H evaluation file changed: {name}; expected {expected_hash}, got {actual_hash}"
            )
        recorded_hash = str((recorded.get(name) or {}).get("sha256") or "")
        if recorded_hash != expected_hash:
            raise ProductPhase2IError(f"Phase 2H manifest hash mismatch for {name}")
        files_meta[name] = {"sha256": actual_hash, "size_bytes": path.stat().st_size}

    if summary.get("decision") != "multiframe_recovery_passed_pending_explicit_authority_patch":
        raise ProductPhase2IError(
            "Phase 2H evaluation is not approved for an explicit authority revision: "
            + str(summary.get("decision"))
        )
    if tuple(summary.get("present_rolls") or ()) != EXPECTED_PRESENT:
        raise ProductPhase2IError("Phase 2H Present rolls changed")
    if tuple(summary.get("needs_review_rolls") or ()) != EXPECTED_NEEDS_REVIEW:
        raise ProductPhase2IError("Phase 2H Needs Review rolls changed")
    if int(summary.get("unverifiable_pairs") or 0) != 0:
        raise ProductPhase2IError("Phase 2H has unverifiable evidence")
    if int(summary.get("strict_accepted_unsafe_errors") or 0) != 0:
        raise ProductPhase2IError("Strict tracklet evidence has an unsafe identity error")

    joined = _load_csv(evaluation_dir / "joined_multiframe_recovery_review.csv", "Phase 2H joined review")
    validation = _load_csv(evaluation_dir / "identity_checkpoint_validation.csv", "Phase 2H identity validation")
    mixed = joined[joined["Tracklet_ID"].astype(str).eq(MIXED_TRACKLET_ID)]
    if len(mixed) != 1:
        raise ProductPhase2IError("The frozen mixed-track regression row is missing or duplicated")
    mixed_row = mixed.iloc[0]
    if str(mixed_row.get("Predicted_Roll") or "").strip() != MIXED_TRACKLET_ROLL:
        raise ProductPhase2IError("The frozen mixed-track regression identity changed")
    if str(mixed_row.get("Review_Status") or "").strip() != "mixed_track":
        raise ProductPhase2IError("The frozen mixed-track regression status changed")
    if _boolish(mixed_row.get("Identity_Correct")):
        raise ProductPhase2IError("Mixed-track evidence cannot be counted as identity-correct")

    # Phase 2H's aggregate unsafe count includes the mixed track. It is accepted
    # here only because that exact checkpoint is excluded from the confirmed
    # validation CSV and preserved as a mandatory negative regression.
    if int(summary.get("unsafe_identity_errors") or 0) != 1:
        raise ProductPhase2IError("Expected exactly one frozen mixed-track safety exception")
    ai0051 = validation[validation["Roll"].astype(str).eq(MIXED_TRACKLET_ROLL)]
    if len(ai0051) != 1 or str(ai0051.iloc[0].get("Confirmed_Checkpoints") or "") != "CP3; CP5":
        raise ProductPhase2IError("Mixed CP1 tracklet leaked into confirmed attendance evidence")

    return summary, joined, validation, files_meta


def _load_roster(inputs: Inputs) -> tuple[list[str], dict[str, str]]:
    mapping = _load_json(inputs.student_map_path, "authoritative student mapping")
    rows = (mapping.get("subject_students") or {}).get("CVO")
    if not isinstance(rows, list):
        raise ProductPhase2IError("Authoritative CVO roster is missing")
    rolls: list[str] = []
    names: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ProductPhase2IError("Invalid CVO roster row")
        roll = canonical_roll(row.get("roll"))
        if not roll:
            raise ProductPhase2IError("Blank roll in CVO roster")
        if roll in names:
            raise ProductPhase2IError(f"Duplicate roll in CVO roster: {roll}")
        rolls.append(roll)
        names[roll] = str(row.get("name") or "")
    if len(rolls) != 27:
        raise ProductPhase2IError(f"Expected 27 CVO students; found {len(rolls)}")
    if MISSING_ENROLLMENT_ROLL not in names:
        raise ProductPhase2IError("Missing-enrollment student is not in the CVO roster")
    return rolls, names


def _load_source_entry(inputs: Inputs) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    store = read_json_object(inputs.status_path, missing_default={})
    entry = store.get(SESSION_ID)
    if not isinstance(entry, dict):
        raise ProductPhase2IError(f"Source attendance session not found: {SESSION_ID}")
    if str(entry.get("official_recognition_authority") or "") != "frame_detection":
        if str(entry.get("authority_revision_id") or "") == AUTHORITY_REVISION_ID:
            raise ProductPhase2IError("Phase 2I authority revision is already applied")
        raise ProductPhase2IError("Source report no longer uses the expected frame-detection authority")
    if _boolish(entry.get("attendance_finalized") or entry.get("Attendance_Finalized")):
        raise ProductPhase2IError("Refusing to supersede a finalized attendance report")
    rows = entry.get("attendance_data")
    if not isinstance(rows, list) or len(rows) != 27:
        raise ProductPhase2IError("Source attendance report must contain exactly 27 roster rows")
    row_rolls = [canonical_roll(row.get("Roll_Number") or row.get("roll")) for row in rows if isinstance(row, dict)]
    if len(row_rolls) != 27 or len(set(row_rolls)) != 27:
        raise ProductPhase2IError("Source attendance rows are blank or duplicated")
    if entry.get("status") not in {"Completed", "Needs Review"}:
        raise ProductPhase2IError("Source attendance report is not in a completed reviewable state")
    return store, dict(entry), [dict(row) for row in rows]


def _verify_no_existing_overrides(inputs: Inputs) -> None:
    path = Path(inputs.overrides_path)
    if not path.exists():
        return
    payload = read_json_object(path, missing_default={})
    session_overrides = payload.get(SESSION_ID)
    if isinstance(session_overrides, dict) and session_overrides:
        raise ProductPhase2IError(
            "The source report has saved manual overrides. Clear or export them before applying Phase 2I."
        )


def _checkpoint_set(value: Any) -> set[str]:
    return {
        token.strip().upper()
        for token in str(value or "").replace(",", ";").split(";")
        if token.strip().upper() in {"CP1", "CP2", "CP3", "CP4", "CP5"}
    }


def _status_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int | float]:
    counts = {"present": 0, "needs_review": 0, "unconfirmed": 0, "missing_enrollment": 0, "absent": 0}
    for row in rows:
        status = str(row.get("Status") or row.get("Final_Status") or "").strip().lower()
        if status.startswith("present"):
            counts["present"] += 1
        elif "missing enrollment" in status:
            counts["missing_enrollment"] += 1
        elif "unconfirmed" in status:
            counts["unconfirmed"] += 1
        elif "review" in status:
            counts["needs_review"] += 1
        elif status == "absent":
            counts["absent"] += 1
        else:
            raise ProductPhase2IError(f"Unsupported corrected attendance status: {status!r}")
    counts["total"] = len(rows)
    counts["attendance_percentage"] = round((counts["present"] / len(rows)) * 100, 1) if rows else 0.0
    return counts


def build_corrected_rows(
    source_rows: Sequence[Mapping[str, Any]],
    validation: pd.DataFrame,
    roster_rolls: Sequence[str],
    roster_names: Mapping[str, str],
) -> list[dict[str, Any]]:
    by_roll = {
        canonical_roll(row.get("Roll_Number") or row.get("roll")): dict(row)
        for row in source_rows
        if canonical_roll(row.get("Roll_Number") or row.get("roll"))
    }
    validation_by_roll = {
        canonical_roll(row.get("Roll")): row
        for row in validation.to_dict("records")
        if canonical_roll(row.get("Roll"))
    }
    corrected: list[dict[str, Any]] = []
    for roll in roster_rolls:
        if roll not in by_roll:
            raise ProductPhase2IError(f"Source report is missing roster roll {roll}")
        source = dict(by_roll[roll])
        evidence = validation_by_roll.get(roll, {})
        checkpoints = _checkpoint_set(evidence.get("Confirmed_Checkpoints"))
        count = len(checkpoints)
        shadow_status = str(evidence.get("Shadow_Status") or "Unconfirmed").strip()
        if roll == MISSING_ENROLLMENT_ROLL:
            status = "Missing Enrollment"
            interpretation = "No production embedding exists; attendance cannot be inferred from recognition."
            flags = "Missing Enrollment"
            review_reason = "Enrollment data is missing. Resolve enrollment or make an authorized manual decision."
        elif shadow_status == "Present":
            status = "Present"
            interpretation = "Confirmed by blind-reviewed multi-frame tracklet evidence."
            flags = ""
            review_reason = ""
        elif shadow_status == "Needs Review":
            status = "Needs Review"
            interpretation = "Identity was confirmed, but fewer than three checkpoints were verified."
            flags = "Insufficient Confirmed Checkpoints"
            review_reason = "Human-validated identity evidence exists, but the 3-of-5 attendance rule was not met."
        else:
            status = "Unconfirmed"
            interpretation = "Camera evidence was insufficient; this student is not proven absent."
            flags = "Insufficient Camera Evidence"
            review_reason = "The low-quality session cannot safely distinguish absence from failed recognition."

        source.update(
            {
                "Name": source.get("Name") or roster_names.get(roll, ""),
                "Roll_Number": roll,
                "Status": status,
                "Final_Status": status,
                "Present": "Yes" if status == "Present" else "No",
                "Original_AI_Status": source.get("Original_AI_Status") or source.get("Status") or source.get("Final_Status") or "Absent",
                "Superseded_Frame_Status": source.get("Status") or source.get("Final_Status") or "Absent",
                "Recognized_Checkpoints": count,
                "Reviewed_Tracklet_Checkpoints": "; ".join(sorted(checkpoints)),
                "Reviewed_Tracklet_Checkpoint_Count": count,
                "Official_Tracklet_Evidence_Count": count,
                "Evidence_Interpretation": interpretation,
                "Flags": flags,
                "Review_Queue_Reason": review_reason,
                "Review_Queue_Note": interpretation,
                "Processing_Contract_ID": AUTOMATIC_POLICY_VERSION,
                "Processing_Mode": "quality_aware_tracklet_authority_v1",
                "Official_Recognition_Authority": OFFICIAL_RECOGNITION_AUTHORITY,
                "Zone_Mode": "zones",
                "Tracklet_Mode": "tracklets",
                "Tracklets_Used_For_Official_Attendance": "Yes",
                "Guarded_Recovery_Authority": "human_validated_for_this_revision",
                "Automatic_Guarded_Recovery_Enabled": "No",
                "Authority_Revision_ID": AUTHORITY_REVISION_ID,
                "Authority_Policy_Version": POLICY_VERSION,
                "Attendance_Finalized": "No",
                "Requires_Manual_Review": "Yes" if status != "Present" else "No",
            }
        )
        for cp_id in ("CP1", "CP2", "CP3", "CP4", "CP5"):
            source[f"{cp_id}_Tracklet_Confirmed"] = "Yes" if cp_id in checkpoints else "No"
        corrected.append(source)

    summary = _status_counts(corrected)
    for key, expected in EXPECTED_COUNTS.items():
        if int(summary[key]) != expected:
            raise ProductPhase2IError(f"Corrected summary {key} changed: expected {expected}, got {summary[key]}")
    return corrected


def preflight(inputs: Inputs) -> PreflightResult:
    summary, _joined, validation, files_meta = _verify_evaluation(inputs)
    roster_rolls, roster_names = _load_roster(inputs)
    status_store, source_entry, source_rows = _load_source_entry(inputs)
    _verify_no_existing_overrides(inputs)
    source_rolls = {canonical_roll(row.get("Roll_Number") or row.get("roll")) for row in source_rows}
    if source_rolls != set(roster_rolls):
        raise ProductPhase2IError("Source attendance rows do not exactly match the authoritative CVO roster")
    source_entry_sha256 = _entry_hash(source_entry)
    if source_entry_sha256 != EXPECTED_SOURCE_ENTRY_SHA256:
        raise ProductPhase2IError(
            "The MON_P4 source report changed after the Phase 2I snapshot; "
            f"expected {EXPECTED_SOURCE_ENTRY_SHA256}, got {source_entry_sha256}"
        )
    corrected_rows = build_corrected_rows(source_rows, validation, roster_rolls, roster_names)
    corrected_summary = _status_counts(corrected_rows)
    if tuple(summary.get("present_rolls") or ()) != EXPECTED_PRESENT:
        raise ProductPhase2IError("Frozen Present rolls changed during preflight")
    return PreflightResult(
        status_store=status_store,
        source_entry=source_entry,
        source_rows=source_rows,
        corrected_rows=corrected_rows,
        summary=corrected_summary,
        source_entry_sha256=source_entry_sha256,
        evaluation_files=files_meta,
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    frame = pd.DataFrame(list(rows))
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _manifest_for(root: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file() or path.name == "authority_manifest.json":
            continue
        rel = path.relative_to(root).as_posix()
        files[rel] = {"sha256": _sha256_file(path), "size_bytes": path.stat().st_size}
    return files


def verify_authority_output(output_dir: Path) -> dict[str, Any]:
    root = Path(output_dir)
    manifest = _load_json(root / "authority_manifest.json", "Phase 2I authority manifest")
    if manifest.get("policy_version") != POLICY_VERSION:
        raise ProductPhase2IError("Phase 2I authority output policy changed")
    if manifest.get("authority_revision_id") != AUTHORITY_REVISION_ID:
        raise ProductPhase2IError("Phase 2I authority revision ID changed")
    recorded = manifest.get("files")
    if not isinstance(recorded, dict) or not recorded:
        raise ProductPhase2IError("Phase 2I authority manifest has no files")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "authority_manifest.json"
    }
    if actual != set(recorded):
        raise ProductPhase2IError("Phase 2I authority output file set changed")
    for rel, metadata in recorded.items():
        path = root / rel
        if _sha256_file(path) != str(metadata.get("sha256") or ""):
            raise ProductPhase2IError(f"Phase 2I authority output hash changed: {rel}")
        if path.stat().st_size != int(metadata.get("size_bytes") or -1):
            raise ProductPhase2IError(f"Phase 2I authority output size changed: {rel}")
    return manifest


def _output_id(result: PreflightResult) -> str:
    digest = hashlib.sha256()
    digest.update(POLICY_VERSION.encode("utf-8"))
    digest.update(AUTHORITY_REVISION_ID.encode("utf-8"))
    digest.update(result.source_entry_sha256.encode("ascii"))
    for name in sorted(result.evaluation_files):
        digest.update(name.encode("utf-8"))
        digest.update(result.evaluation_files[name]["sha256"].encode("ascii"))
    return "authority-revision-" + digest.hexdigest()[:24]


def _relative_to_repo(path: Path, repo_root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(repo_root).resolve()).as_posix()
    except Exception:
        return str(path)


def create_authority_output(inputs: Inputs, result: PreflightResult) -> tuple[Path, bool]:
    output_dir = Path(inputs.output_root) / _output_id(result)
    if output_dir.exists():
        verify_authority_output(output_dir)
        return output_dir, True

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / (output_dir.name + ".staging")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        corrected_csv = staging / f"corrected_attendance_{SESSION_ID}.csv"
        _write_csv(corrected_csv, result.corrected_rows)
        source_csv = staging / f"superseded_frame_attendance_{SESSION_ID}.csv"
        _write_csv(source_csv, result.source_rows)
        decision = {
            "schema_version": 1,
            "policy_version": POLICY_VERSION,
            "automatic_policy_version": AUTOMATIC_POLICY_VERSION,
            "authority_revision_id": AUTHORITY_REVISION_ID,
            "session_id": SESSION_ID,
            "official_recognition_authority": OFFICIAL_RECOGNITION_AUTHORITY,
            "future_automatic_authority": AUTOMATIC_RECOGNITION_AUTHORITY,
            "guarded_recovery_automatic": False,
            "guarded_recovery_reason": (
                "The exact mixed tracklet CP1-cam5-back-TRK00005 remains a mandatory negative regression. "
                "Guarded recovery is official only when human-validated for this revision; future automatic "
                "runs expose guarded candidates for review until track purity/splitting is validated."
            ),
            "source_entry_sha256": result.source_entry_sha256,
            "counts": result.summary,
            "present_rolls": list(EXPECTED_PRESENT),
            "needs_review_rolls": list(EXPECTED_NEEDS_REVIEW),
            "missing_enrollment_rolls": [MISSING_ENROLLMENT_ROLL],
            "absent_rolls": [],
            "recognition_repeated": False,
            "video_reprocessed": False,
            "source_report_overwritten": False,
        }
        (staging / "authority_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
        (staging / "README.md").write_text(
            "# Product Phase 2I Authority Revision\n\n"
            "This immutable output supersedes the low-quality frame-only MON_P4 result with the frozen, "
            "blind-reviewed Phase 2H multi-frame evidence. It produces 6 Present, 4 Needs Review, "
            "16 Unconfirmed, 1 Missing Enrollment, and 0 Absent. Recognition was not rerun.\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "policy_version": POLICY_VERSION,
            "authority_revision_id": AUTHORITY_REVISION_ID,
            "source_entry_sha256": result.source_entry_sha256,
            "evaluation_files": result.evaluation_files,
            "files": _manifest_for(staging),
        }
        (staging / "authority_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        staging.replace(output_dir)
        verify_authority_output(output_dir)
        return output_dir, False
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _backup_status_entry(inputs: Inputs, result: PreflightResult, output_dir: Path) -> Path:
    Path(inputs.state_backup_dir).mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = Path(inputs.state_backup_dir) / f"phase_2i_before_{SESSION_ID}_{stamp}.json"
    if backup_path.exists():
        raise ProductPhase2IError(f"Refusing to overwrite authority backup: {backup_path}")
    payload = {
        "schema_version": 1,
        "policy_version": POLICY_VERSION,
        "authority_revision_id": AUTHORITY_REVISION_ID,
        "session_id": SESSION_ID,
        "source_entry_sha256": result.source_entry_sha256,
        "source_entry": result.source_entry,
        "authority_output": _relative_to_repo(output_dir, inputs.repo_root),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(backup_path, payload)
    return backup_path


def build_updated_entry(inputs: Inputs, result: PreflightResult, output_dir: Path, backup_path: Path) -> dict[str, Any]:
    corrected_csv = output_dir / f"corrected_attendance_{SESSION_ID}.csv"
    superseded_csv = output_dir / f"superseded_frame_attendance_{SESSION_ID}.csv"
    decision_json = output_dir / "authority_decision.json"
    source_entry = dict(result.source_entry)
    history = list(source_entry.get("report_revision_history") or [])
    history.append(
        {
            "revision_type": "superseded_frame_only_quality_failed",
            "superseded_at": datetime.now(timezone.utc).isoformat(),
            "status": source_entry.get("status"),
            "report_type": source_entry.get("report_type"),
            "official_recognition_authority": source_entry.get("official_recognition_authority"),
            "attendance_csv": source_entry.get("attendance_csv"),
            "excel_file": source_entry.get("excel_file"),
            "output_manifest": source_entry.get("output_manifest"),
            "source_entry_sha256": result.source_entry_sha256,
        }
    )
    unresolved = EXPECTED_COUNTS["needs_review"] + EXPECTED_COUNTS["unconfirmed"] + EXPECTED_COUNTS["missing_enrollment"]
    corrected_ref = _relative_to_repo(corrected_csv, inputs.repo_root).replace("attendance_output/", "", 1)
    superseded_ref = _relative_to_repo(superseded_csv, inputs.repo_root).replace("attendance_output/", "", 1)
    decision_ref = _relative_to_repo(decision_json, inputs.repo_root).replace("attendance_output/", "", 1)
    output_manifest = dict(source_entry.get("output_manifest") or {})
    output_manifest.update(
        {
            "attendance_csv": corrected_ref,
            "authority_decision_json": decision_ref,
            "superseded_frame_attendance_csv": superseded_ref,
            "source_processor_attendance_csv": source_entry.get("attendance_csv"),
        }
    )
    updated = dict(source_entry)
    updated.update(
        {
            "status": "Needs Review",
            "report_type": "Corrected Multi-frame Authority",
            "attendance_data": result.corrected_rows,
            "attendance_csv": corrected_ref,
            "class_report_file": corrected_ref,
            "output_manifest": output_manifest,
            "excel_file": None,
            "official_recognition_authority": OFFICIAL_RECOGNITION_AUTHORITY,
            "automatic_recognition_authority": AUTOMATIC_RECOGNITION_AUTHORITY,
            "processing_contract_id": AUTOMATIC_POLICY_VERSION,
            "processing_profile": "quality_aware_tracklet_authority_v1",
            "zone_mode": "zones",
            "tracklet_mode": "tracklets",
            "tracklets_used_for_official_attendance": True,
            "guarded_recovery_automatic": False,
            "guarded_recovery_authority": "human_validated_for_this_revision",
            "authority_revision_id": AUTHORITY_REVISION_ID,
            "authority_policy_version": POLICY_VERSION,
            "authority_output_dir": _relative_to_repo(output_dir, inputs.repo_root),
            "authority_backup": _relative_to_repo(backup_path, inputs.repo_root),
            "authority_applied_at": datetime.now(timezone.utc).isoformat(),
            "recognition_repeated_for_authority_revision": False,
            "source_report_superseded": True,
            "source_report_quality_failed": True,
            "report_revision_history": history[-20:],
            "requires_manual_review": True,
            "attendance_finalized": False,
            "Attendance_Finalized": "No",
            "review_state": "needs_attention",
            "unresolved_count": unresolved,
            "present_count": EXPECTED_COUNTS["present"],
            "needs_review_count": EXPECTED_COUNTS["needs_review"],
            "unconfirmed_count": EXPECTED_COUNTS["unconfirmed"],
            "missing_enrollment_count": EXPECTED_COUNTS["missing_enrollment"],
            "absent_count": EXPECTED_COUNTS["absent"],
            "total_students": EXPECTED_COUNTS["total"],
            "attendance_percentage": round(EXPECTED_COUNTS["present"] / EXPECTED_COUNTS["total"] * 100, 1),
            "corrected_status_summary": dict(EXPECTED_COUNTS),
            "final_attendance_csv": None,
        }
    )
    updated.pop("finalized_at", None)
    updated.pop("finalized_by", None)
    return updated


def apply_authority_revision(inputs: Inputs) -> dict[str, Any]:
    result = preflight(inputs)
    output_dir, reused = create_authority_output(inputs, result)
    current_store = read_json_object(inputs.status_path, missing_default={})
    current_entry = current_store.get(SESSION_ID)
    if not isinstance(current_entry, dict) or _entry_hash(current_entry) != result.source_entry_sha256:
        raise ProductPhase2IError("attendance_status.json drifted after preflight; no authority change was made")
    backup_path = _backup_status_entry(inputs, result, output_dir)
    updated_entry = build_updated_entry(inputs, result, output_dir, backup_path)
    next_store = dict(current_store)
    next_store[SESSION_ID] = updated_entry
    try:
        atomic_write_json(inputs.status_path, next_store)
    except Exception:
        backup_path.unlink(missing_ok=True)
        raise
    committed = read_json_object(inputs.status_path)
    committed_entry = committed.get(SESSION_ID)
    if not isinstance(committed_entry, dict) or committed_entry.get("authority_revision_id") != AUTHORITY_REVISION_ID:
        atomic_write_json(inputs.status_path, current_store)
        raise ProductPhase2IError("Authority revision commit verification failed; source state was restored")
    committed_summary = _status_counts(committed_entry.get("attendance_data") or [])
    if any(int(committed_summary[key]) != EXPECTED_COUNTS[key] for key in EXPECTED_COUNTS):
        atomic_write_json(inputs.status_path, current_store)
        raise ProductPhase2IError("Committed attendance totals changed; source state was restored")
    return {
        "policy_version": POLICY_VERSION,
        "authority_revision_id": AUTHORITY_REVISION_ID,
        "session_id": SESSION_ID,
        "output_dir": str(output_dir),
        "output_reused": reused,
        "backup_path": str(backup_path),
        "summary": committed_summary,
        "official_recognition_authority": OFFICIAL_RECOGNITION_AUTHORITY,
        "automatic_recognition_authority": AUTOMATIC_RECOGNITION_AUTHORITY,
        "guarded_recovery_automatic": False,
        "recognition_repeated": False,
    }


def rollback_authority_revision(inputs: Inputs, backup_path: Path) -> dict[str, Any]:
    backup = _load_json(Path(backup_path), "Phase 2I authority backup")
    if backup.get("policy_version") != POLICY_VERSION or backup.get("authority_revision_id") != AUTHORITY_REVISION_ID:
        raise ProductPhase2IError("Backup does not belong to the current Phase 2I authority policy")
    source_entry = backup.get("source_entry")
    if not isinstance(source_entry, dict) or _entry_hash(source_entry) != backup.get("source_entry_sha256"):
        raise ProductPhase2IError("Authority backup source entry failed integrity verification")
    store = read_json_object(inputs.status_path, missing_default={})
    current = store.get(SESSION_ID)
    if not isinstance(current, dict) or current.get("authority_revision_id") != AUTHORITY_REVISION_ID:
        raise ProductPhase2IError("Current session is not the Phase 2I authority revision; rollback refused")
    next_store = dict(store)
    next_store[SESSION_ID] = source_entry
    atomic_write_json(inputs.status_path, next_store)
    restored = read_json_object(inputs.status_path).get(SESSION_ID)
    if not isinstance(restored, dict) or _entry_hash(restored) != backup.get("source_entry_sha256"):
        atomic_write_json(inputs.status_path, store)
        raise ProductPhase2IError("Rollback verification failed; Phase 2I state was restored")
    return {
        "session_id": SESSION_ID,
        "restored_source_entry_sha256": backup.get("source_entry_sha256"),
        "backup_path": str(backup_path),
        "rollback_status": "PASS",
    }


def select_guarded_review_candidates(tracklet_frame: pd.DataFrame) -> dict[str, set[str]]:
    """Return conservative near-threshold identity/checkpoint candidates.

    This function intentionally does not promote candidates to Present. It is
    used after strict tracklet processing to expose material recovery evidence
    for review while the mixed-track purity/splitting work remains pending.
    """

    required = {
        "Tracklet_ID",
        "Checkpoint_ID",
        "Tracklet_Eligible",
        "Tracklet_Accepted",
        "Tracklet_Best_Roll",
        "Tracklet_Best_Score",
        "Tracklet_Margin",
        "Dominant_Frame_Best_Roll",
        "Dominant_Frame_Best_Share_Pct",
        "Observation_Count",
        "Selected_Observation_Count",
        "Consistent_Embedding_Count",
        "Pairwise_Similarity_Median",
    }
    missing = sorted(required.difference(tracklet_frame.columns))
    if missing:
        raise ProductPhase2IError("Tracklet diagnostics missing guarded-recovery fields: " + ", ".join(missing))
    frame = tracklet_frame.copy()
    yes = lambda series: series.astype(str).str.strip().str.lower().isin({"yes", "true", "1"})
    best_roll = frame["Tracklet_Best_Roll"].map(canonical_roll)
    dominant_roll = frame["Dominant_Frame_Best_Roll"].map(canonical_roll)
    eligible = yes(frame["Tracklet_Eligible"])
    rejected = ~yes(frame["Tracklet_Accepted"])
    numeric = {
        name: pd.to_numeric(frame[name], errors="coerce").fillna(0)
        for name in (
            "Tracklet_Best_Score",
            "Tracklet_Margin",
            "Dominant_Frame_Best_Share_Pct",
            "Observation_Count",
            "Selected_Observation_Count",
            "Consistent_Embedding_Count",
            "Pairwise_Similarity_Median",
        )
    }
    candidate_mask = (
        eligible
        & rejected
        & best_roll.ne("")
        & best_roll.eq(dominant_roll)
        & numeric["Tracklet_Best_Score"].ge(RECOVERY_MIN_SCORE)
        & numeric["Tracklet_Margin"].ge(RECOVERY_MIN_MARGIN)
        & numeric["Dominant_Frame_Best_Share_Pct"].ge(RECOVERY_MIN_DOMINANT_SHARE_PCT)
        & numeric["Observation_Count"].ge(RECOVERY_MIN_OBSERVATIONS)
        & numeric["Selected_Observation_Count"].ge(RECOVERY_MIN_SELECTED_OBSERVATIONS)
        & numeric["Consistent_Embedding_Count"].ge(RECOVERY_MIN_CONSISTENT_EMBEDDINGS)
        & numeric["Pairwise_Similarity_Median"].ge(RECOVERY_MIN_PAIRWISE_MEDIAN)
    )
    candidates = frame.loc[candidate_mask, ["Tracklet_ID", "Checkpoint_ID"]].copy()
    candidates["Roll"] = best_roll.loc[candidate_mask].values
    checkpoint_sets = {
        roll: set(group["Checkpoint_ID"].astype(str).str.upper())
        for roll, group in candidates.groupby("Roll")
    }
    return {
        roll: checkpoints
        for roll, checkpoints in checkpoint_sets.items()
        if len(checkpoints) >= RECOVERY_MIN_CHECKPOINT_REPETITION
    }


def apply_automatic_status_semantics(
    rows: Sequence[Mapping[str, Any]],
    *,
    quality_requires_review: bool,
    missing_embedding_rolls: Iterable[str],
    guarded_candidates: Mapping[str, set[str]] | None = None,
) -> list[dict[str, Any]]:
    """Apply safe status semantics to a newly processed strict-tracklet report."""

    missing = {canonical_roll(value) for value in missing_embedding_rolls if canonical_roll(value)}
    guarded_candidates = guarded_candidates or {}
    output: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        roll = canonical_roll(row.get("Roll_Number") or row.get("roll"))
        status = str(row.get("Status") or row.get("Final_Status") or "Absent").strip()
        if roll in missing:
            new_status = "Missing Enrollment"
            reason = "No production embedding exists; recognition cannot determine attendance."
        elif status.lower().startswith("present"):
            new_status = "Present"
            reason = "Confirmed by strict accepted tracklet evidence."
        elif "review" in status.lower():
            new_status = "Needs Review"
            reason = "Strict tracklet evidence exists, but the 3-of-5 rule was not met."
        elif roll in guarded_candidates:
            new_status = "Needs Review"
            reason = "Guarded near-threshold evidence repeats across checkpoints and requires review."
        elif quality_requires_review:
            new_status = "Unconfirmed"
            reason = "Session quality was insufficient; failed recognition is not proof of absence."
        else:
            new_status = "Absent"
            reason = "Session quality passed and no qualifying strict tracklet evidence was found."
        row.update(
            {
                "Status": new_status,
                "Final_Status": new_status,
                "Present": "Yes" if new_status == "Present" else "No",
                "Official_Recognition_Authority": AUTOMATIC_RECOGNITION_AUTHORITY,
                "Tracklets_Used_For_Official_Attendance": "Yes",
                "Automatic_Guarded_Recovery_Enabled": "No",
                "Guarded_Recovery_Candidate_Checkpoints": "; ".join(sorted(guarded_candidates.get(roll, set()))),
                "Evidence_Interpretation": reason,
                "Review_Queue_Reason": reason if new_status in {"Needs Review", "Unconfirmed", "Missing Enrollment"} else "",
                "Requires_Manual_Review": "Yes" if new_status in {"Needs Review", "Unconfirmed", "Missing Enrollment"} else "No",
            }
        )
        if new_status == "Unconfirmed":
            row["Flags"] = "Insufficient Camera Evidence"
        elif new_status == "Missing Enrollment":
            row["Flags"] = "Missing Enrollment"
        elif roll in guarded_candidates and new_status == "Needs Review":
            row["Flags"] = "Guarded Recovery Candidate"
        output.append(row)
    return output
