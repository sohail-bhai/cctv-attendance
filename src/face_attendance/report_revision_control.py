from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .product_phase_2i_authority import (
    AUTHORITY_REVISION_ID,
    AUTOMATIC_RECOGNITION_AUTHORITY,
    OFFICIAL_RECOGNITION_AUTHORITY,
    POLICY_VERSION as PHASE_2I_POLICY_VERSION,
    SESSION_ID as PHASE_2I_SESSION_ID,
    verify_authority_output,
)
from .recall_analysis import canonical_roll
from .workflow_state import atomic_write_json, read_json_object

POLICY_VERSION = "product-phase-2j-report-revision-control-v2"
REVISION_SCHEMA_VERSION = 2
EXPECTED_CURRENT_CANDIDATE_SHA256 = "e28dc0278abfad034bd78bb576620a50f90100461f23b5e0b920e1e65e39c57f"
EXPECTED_CURRENT_CANDIDATE_COUNTS = {
    "present": 2,
    "needs_review": 8,
    "unconfirmed": 16,
    "missing_enrollment": 1,
    "absent": 0,
    "total": 27,
}
EXPECTED_REVIEWED_COUNTS = {
    "present": 6,
    "needs_review": 4,
    "unconfirmed": 16,
    "missing_enrollment": 1,
    "absent": 0,
    "total": 27,
}


class ReportRevisionError(RuntimeError):
    """Raised when a report revision cannot be preserved or restored safely."""


@dataclass(frozen=True)
class CommitResult:
    current_entry: dict[str, Any]
    candidate_entry: dict[str, Any]
    candidate_revision: dict[str, Any] | None
    official_preserved: bool
    disposition: str


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _entry_hash(entry: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(entry)).hexdigest()


def _normal_status(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", " ")


def _normal_checkpoint_list(value: Any) -> list[str]:
    items = {
        item.strip().upper()
        for item in re.split(r"[;,]", str(value or ""))
        if item.strip()
    }
    return sorted(items)


def _attendance_result_payload(entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = entry.get("attendance_data")
    if not isinstance(rows, list):
        return []
    result: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        roll = canonical_roll(raw.get("Roll_Number") or raw.get("roll"))
        result.append(
            {
                "roll": roll,
                "status": _normal_status(raw.get("Final_Status") or raw.get("Status")),
                "recognized_checkpoints": str(raw.get("Recognized_Checkpoints") or "0").strip(),
                "reviewed_tracklet_checkpoints": _normal_checkpoint_list(raw.get("Reviewed_Tracklet_Checkpoints")),
                "mixed_track_checkpoints_rejected": _normal_checkpoint_list(raw.get("Mixed_Track_Checkpoints_Rejected")),
                "guarded_recovery_candidate_checkpoints": _normal_checkpoint_list(raw.get("Guarded_Recovery_Candidate_Checkpoints")),
            }
        )
    return sorted(result, key=lambda item: item["roll"])


def attendance_result_hash(entry: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(_attendance_result_payload(entry))).hexdigest()


def attendance_decision_hash(entry: Mapping[str, Any]) -> str:
    # Equivalence is based on the official attendance decision, not incidental
    # evidence counts on rows that remain Unconfirmed. A rerun may find one
    # strict checkpoint for an Unconfirmed student while the reviewed revision
    # records zero; neither result proves attendance, so that must not create a
    # new review task. Present/Needs Review decisions remain bound to their
    # checkpoint count and exact reviewed checkpoint labels.
    payload: list[dict[str, Any]] = []
    for item in _attendance_result_payload(entry):
        status = item["status"]
        decision = {
            "roll": item["roll"],
            "status": status,
            "recognized_checkpoints": (
                item["recognized_checkpoints"]
                if status in {"present", "needs review"}
                else None
            ),
            "reviewed_tracklet_checkpoints": (
                item["reviewed_tracklet_checkpoints"]
                if status in {"present", "needs review"}
                else []
            ),
        }
        payload.append(decision)
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _candidate_semantic_hash(session_id: str, entry: Mapping[str, Any]) -> str:
    payload = {
        "session_id": str(session_id),
        "input_source_path": str(entry.get("input_source_path") or entry.get("video_dir") or "").replace("\\", "/"),
        "official_recognition_authority": str(entry.get("official_recognition_authority") or ""),
        "processing_contract_id": str(entry.get("processing_contract_id") or ""),
        "review_registry_id": str(entry.get("review_carry_forward_source") or ""),
        "attendance_result_sha256": attendance_result_hash(entry),
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _safe_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "").strip()).strip("-.")
    return text or "session"


def _status_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {
        "present": 0,
        "needs_review": 0,
        "unconfirmed": 0,
        "missing_enrollment": 0,
        "absent": 0,
        "total": 0,
    }
    for row in rows:
        status = _normal_status(row.get("Final_Status") or row.get("Status"))
        counts["total"] += 1
        if status.startswith("present"):
            counts["present"] += 1
        elif "review" in status:
            counts["needs_review"] += 1
        elif status == "unconfirmed":
            counts["unconfirmed"] += 1
        elif status in {"missing enrollment", "missing embedding"}:
            counts["missing_enrollment"] += 1
        elif status == "absent":
            counts["absent"] += 1
    return counts


def _relative_output(path: Path, repo_root: Path) -> str:
    output_root = Path(repo_root) / "attendance_output"
    return Path(path).resolve().relative_to(output_root.resolve()).as_posix()


def _revision_root(repo_root: Path, session_id: str) -> Path:
    return (
        Path(repo_root)
        / "attendance_output"
        / "product_workflow"
        / "report_revisions"
        / _safe_name(session_id)
    )


def _revision_manifest_files(root: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file() or path.name == "revision_manifest.json":
            continue
        rel = path.relative_to(root).as_posix()
        files[rel] = {"sha256": _sha256_file(path), "size_bytes": path.stat().st_size}
    return files


def verify_revision_output(output_dir: Path) -> dict[str, Any]:
    root = Path(output_dir)
    manifest_path = root / "revision_manifest.json"
    if not manifest_path.is_file():
        raise ReportRevisionError(f"Revision manifest not found: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise ReportRevisionError(f"Could not read revision manifest: {exc}") from exc
    if payload.get("schema_version") != REVISION_SCHEMA_VERSION:
        raise ReportRevisionError("Unsupported report-revision schema")
    if payload.get("policy_version") != POLICY_VERSION:
        raise ReportRevisionError("Report-revision policy changed")
    expected = payload.get("files")
    if not isinstance(expected, dict) or not expected:
        raise ReportRevisionError("Revision manifest has no immutable files")
    actual = _revision_manifest_files(root)
    if set(actual) != set(expected):
        raise ReportRevisionError("Revision output file set changed")
    for rel, metadata in expected.items():
        if actual[rel]["sha256"] != str(metadata.get("sha256") or ""):
            raise ReportRevisionError(f"Revision output hash changed: {rel}")
        if actual[rel]["size_bytes"] != int(metadata.get("size_bytes") or -1):
            raise ReportRevisionError(f"Revision output size changed: {rel}")
    return payload


def is_protected_official(entry: Mapping[str, Any] | None) -> bool:
    if not isinstance(entry, Mapping):
        return False
    if bool(entry.get("attendance_finalized")) or str(entry.get("Attendance_Finalized") or "").strip().lower() == "yes":
        return True
    if str(entry.get("revision_authority") or "").strip().lower() in {"official", "official_reviewed", "finalized"}:
        return True
    if str(entry.get("authority_revision_id") or "").strip():
        return True
    try:
        if int(entry.get("review_revision") or 0) > 0:
            return True
    except (TypeError, ValueError):
        pass
    report_type = str(entry.get("report_type") or "").strip().lower()
    return any(token in report_type for token in ("corrected", "edited", "finalized", "reviewed authority"))


def revision_summary(entry: Mapping[str, Any], *, revision_id: str, role: str, output_dir: Path) -> dict[str, Any]:
    rows = list(entry.get("attendance_data") or []) if isinstance(entry.get("attendance_data"), list) else []
    counts = _status_counts(rows)
    return {
        "revision_id": revision_id,
        "revision_role": role,
        "report_type": entry.get("report_type"),
        "status": entry.get("status"),
        "official_recognition_authority": entry.get("official_recognition_authority"),
        "processing_contract_id": entry.get("processing_contract_id"),
        "completed_at": entry.get("completed_at"),
        "created_at_utc": _now_utc(),
        "present_count": counts["present"],
        "needs_review_count": counts["needs_review"],
        "unconfirmed_count": counts["unconfirmed"],
        "missing_enrollment_count": counts["missing_enrollment"],
        "absent_count": counts["absent"],
        "total_students": counts["total"],
        "attendance_result_sha256": attendance_result_hash(entry),
        "output_dir": str(output_dir),
    }


def archive_candidate_revision(repo_root: Path, session_id: str, candidate_entry: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(candidate_entry, Mapping) or not isinstance(candidate_entry.get("attendance_data"), list):
        raise ReportRevisionError("Candidate report has no attendance rows")
    entry_digest = _entry_hash(candidate_entry)
    semantic_digest = _candidate_semantic_hash(session_id, candidate_entry)
    revision_id = f"automatic-candidate-{semantic_digest[:24]}"
    output_dir = _revision_root(repo_root, session_id) / revision_id
    if output_dir.exists():
        manifest = verify_revision_output(output_dir)
        return dict(manifest.get("summary") or {})

    staging = output_dir.with_name(output_dir.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=False)
    try:
        entry_path = staging / "candidate_entry.json"
        atomic_write_json(entry_path, dict(candidate_entry))
        rows_path = staging / "candidate_attendance.csv"
        pd.DataFrame(list(candidate_entry.get("attendance_data") or [])).to_csv(rows_path, index=False)
        summary = revision_summary(candidate_entry, revision_id=revision_id, role="automatic_candidate", output_dir=output_dir)
        manifest = {
            "schema_version": REVISION_SCHEMA_VERSION,
            "policy_version": POLICY_VERSION,
            "session_id": session_id,
            "revision_id": revision_id,
            "entry_sha256": entry_digest,
            "semantic_sha256": semantic_digest,
            "attendance_result_sha256": attendance_result_hash(candidate_entry),
            "summary": summary,
            "files": _revision_manifest_files(staging),
        }
        atomic_write_json(staging / "revision_manifest.json", manifest)
        if output_dir.exists():
            raise ReportRevisionError(f"Refusing to overwrite revision output: {output_dir}")
        staging.replace(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    manifest = verify_revision_output(output_dir)
    return dict(manifest.get("summary") or {})


def _append_revision_history(entry: Mapping[str, Any], summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    history = list(entry.get("report_revision_history") or []) if isinstance(entry.get("report_revision_history"), list) else []
    revision_id = str(summary.get("revision_id") or "")
    if revision_id and not any(str(item.get("revision_id") or "") == revision_id for item in history if isinstance(item, Mapping)):
        history.append(dict(summary))
    return history[-50:]


def preserve_official_or_commit_candidate(
    repo_root: Path,
    session_id: str,
    existing_entry: Mapping[str, Any] | None,
    candidate_entry: Mapping[str, Any],
) -> CommitResult:
    candidate = dict(candidate_entry)
    existing = dict(existing_entry or {})
    if not is_protected_official(existing):
        candidate.setdefault("revision_authority", "automatic_current")
        candidate.setdefault("candidate_revision_count", 0)
        return CommitResult(
            current_entry=candidate,
            candidate_entry=candidate,
            candidate_revision=None,
            official_preserved=False,
            disposition="automatic_candidate_became_current",
        )

    candidate_summary = archive_candidate_revision(repo_root, session_id, candidate)
    equivalent = attendance_decision_hash(existing) == attendance_decision_hash(candidate)
    runtime_summary = dict(candidate_summary)
    runtime_summary["equivalent_to_official"] = bool(equivalent)
    current = dict(existing)
    current["revision_authority"] = "official_reviewed" if not bool(current.get("attendance_finalized")) else "finalized"
    current["latest_automatic_candidate"] = dict(runtime_summary)
    if equivalent:
        current.pop("pending_candidate_revision", None)
        current["latest_equivalent_candidate_revision"] = dict(runtime_summary)
        current["latest_candidate_matches_official"] = True
        disposition = "equivalent_candidate_archived_official_preserved"
    else:
        current["pending_candidate_revision"] = dict(runtime_summary)
        current["latest_candidate_matches_official"] = False
        disposition = "automatic_candidate_archived_official_preserved"
    history = _append_revision_history(current, runtime_summary)
    current["report_revision_history"] = history
    current["candidate_revision_count"] = len({
        str(item.get("revision_id") or "")
        for item in history
        if isinstance(item, Mapping) and str(item.get("revision_role") or "") == "automatic_candidate"
    })
    current["official_report_preserved_after_reprocess"] = True
    current["last_candidate_completed_at"] = candidate.get("completed_at") or _now_utc()
    return CommitResult(
        current_entry=current,
        candidate_entry=candidate,
        candidate_revision=dict(runtime_summary),
        official_preserved=True,
        disposition=disposition,
    )


def _find_phase_2i_authority_output(repo_root: Path) -> Path:
    root = Path(repo_root) / "attendance_output" / "product_workflow" / "phase_2i_authority"
    candidates = [path for path in root.glob("authority-revision-*") if path.is_dir()]
    verified: list[Path] = []
    for path in candidates:
        try:
            payload = verify_authority_output(path)
        except Exception:
            continue
        if payload.get("authority_revision_id") == AUTHORITY_REVISION_ID:
            verified.append(path)
    if len(verified) != 1:
        raise ReportRevisionError(f"Expected one verified Phase 2I authority output, found {len(verified)}")
    return verified[0]


def _find_entry_attendance_csv(repo_root: Path, entry: Mapping[str, Any]) -> Path:
    raw = str(entry.get("attendance_csv") or entry.get("class_report_file") or "").strip()
    if not raw:
        raise ReportRevisionError("Current automatic report has no attendance CSV reference")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = Path(repo_root) / "attendance_output" / candidate
    if not candidate.is_file():
        raise ReportRevisionError(f"Current automatic attendance CSV not found: {candidate}")
    return candidate


def _build_reviewed_official_entry(
    repo_root: Path,
    current_candidate: Mapping[str, Any],
    corrected_rows: list[dict[str, Any]],
    authority_output: Path,
    candidate_summary: Mapping[str, Any],
) -> dict[str, Any]:
    counts = _status_counts(corrected_rows)
    if counts != EXPECTED_REVIEWED_COUNTS:
        raise ReportRevisionError(f"Reviewed authority totals changed: {counts}")
    corrected_csv = authority_output / f"corrected_attendance_{PHASE_2I_SESSION_ID}.csv"
    superseded_csv = authority_output / f"superseded_frame_attendance_{PHASE_2I_SESSION_ID}.csv"
    decision_json = authority_output / "authority_decision.json"
    if not corrected_csv.is_file() or not superseded_csv.is_file() or not decision_json.is_file():
        raise ReportRevisionError("Phase 2I immutable authority output is incomplete")

    current = dict(current_candidate)
    output_manifest = dict(current.get("output_manifest") or {})
    output_manifest.update(
        {
            "attendance_csv": _relative_output(corrected_csv, repo_root),
            "authority_decision_json": _relative_output(decision_json, repo_root),
            "superseded_frame_attendance_csv": _relative_output(superseded_csv, repo_root),
            "latest_automatic_candidate_output": candidate_summary.get("output_dir"),
            "latest_automatic_candidate_attendance_csv": current.get("attendance_csv"),
        }
    )
    history = _append_revision_history(current, candidate_summary)
    restored = dict(current)
    restored.update(
        {
            "status": "Needs Review",
            "report_type": "Corrected Multi-frame Authority",
            "attendance_data": corrected_rows,
            "attendance_csv": _relative_output(corrected_csv, repo_root),
            "class_report_file": _relative_output(corrected_csv, repo_root),
            "output_manifest": output_manifest,
            "excel_file": None,
            "official_recognition_authority": OFFICIAL_RECOGNITION_AUTHORITY,
            "automatic_recognition_authority": AUTOMATIC_RECOGNITION_AUTHORITY,
            "processing_contract_id": "product-phase-2i-strict-tracklet-authority-v1",
            "processing_profile": "quality_aware_tracklet_authority_v1",
            "zone_mode": "zones",
            "tracklet_mode": "tracklets",
            "tracklets_used_for_official_attendance": True,
            "guarded_recovery_automatic": False,
            "guarded_recovery_authority": "human_validated_exact_source_carry_forward",
            "authority_revision_id": AUTHORITY_REVISION_ID,
            "authority_policy_version": PHASE_2I_POLICY_VERSION,
            "authority_output_dir": str(authority_output),
            "source_report_superseded": True,
            "source_report_quality_failed": True,
            "revision_authority": "official_reviewed",
            "official_report_preserved_after_reprocess": True,
            "pending_candidate_revision": dict(candidate_summary),
            "latest_automatic_candidate": dict(candidate_summary),
            "latest_candidate_matches_official": False,
            "candidate_revision_count": len({
                str(item.get("revision_id") or "")
                for item in history
                if isinstance(item, Mapping) and str(item.get("revision_role") or "") == "automatic_candidate"
            }),
            "report_revision_history": history,
            "requires_manual_review": True,
            "attendance_finalized": False,
            "Attendance_Finalized": "No",
            "review_state": "needs_attention",
            "unresolved_count": counts["needs_review"] + counts["unconfirmed"] + counts["missing_enrollment"],
            "present_count": counts["present"],
            "needs_review_count": counts["needs_review"],
            "unconfirmed_count": counts["unconfirmed"],
            "missing_enrollment_count": counts["missing_enrollment"],
            "absent_count": counts["absent"],
            "total_students": counts["total"],
            "attendance_percentage": round(counts["present"] / max(counts["total"], 1) * 100, 1),
            "corrected_status_summary": dict(counts),
            "review_carry_forward_applied": True,
            "review_carry_forward_source": "phase_2h_multiframe_recovery-7d9d3a539c0b32e33048",
            "mixed_track_quarantine": ["CP1-cam5-back-TRK00005"],
            "final_attendance_csv": None,
        }
    )
    restored.pop("finalized_at", None)
    restored.pop("finalized_by", None)
    return restored


def repair_mon_p4_after_reprocess(
    repo_root: Path,
    *,
    apply: bool,
    previous_registry_bytes: bytes | None = None,
    installed_registry_sha256: str | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    status_path = root / "data" / "attendance_status.json"
    store = read_json_object(status_path)
    current = store.get(PHASE_2I_SESSION_ID)
    if not isinstance(current, dict):
        raise ReportRevisionError("MON_P4 attendance entry is missing")
    if str(current.get("official_recognition_authority") or "") != AUTOMATIC_RECOGNITION_AUTHORITY:
        if str(current.get("authority_revision_id") or "") == AUTHORITY_REVISION_ID:
            return {
                "status": "already_repaired",
                "session_id": PHASE_2I_SESSION_ID,
                "summary": _status_counts(current.get("attendance_data") or []),
                "reviewed_summary": _status_counts(current.get("attendance_data") or []),
                "official_authority": current.get("official_recognition_authority"),
                "automatic_authority": current.get("automatic_recognition_authority"),
                "candidate_revision": current.get("pending_candidate_revision") or current.get("latest_automatic_candidate"),
                "recognition_repeated": False,
                "video_reprocessed": False,
                "official_entry": current,
            }
        raise ReportRevisionError("Current MON_P4 entry is not the expected automatic reprocess candidate")
    current_counts = _status_counts(current.get("attendance_data") or [])
    if current_counts != EXPECTED_CURRENT_CANDIDATE_COUNTS:
        raise ReportRevisionError(f"Current automatic candidate totals changed: {current_counts}")
    current_csv = _find_entry_attendance_csv(root, current)
    current_csv_hash = _sha256_file(current_csv)
    if current_csv_hash != EXPECTED_CURRENT_CANDIDATE_SHA256:
        raise ReportRevisionError(
            "Current automatic MON_P4 CSV changed; expected "
            f"{EXPECTED_CURRENT_CANDIDATE_SHA256}, got {current_csv_hash}"
        )
    authority_output = _find_phase_2i_authority_output(root)
    corrected_csv = authority_output / f"corrected_attendance_{PHASE_2I_SESSION_ID}.csv"
    corrected_rows = pd.read_csv(corrected_csv, dtype=str, keep_default_na=False, encoding="utf-8-sig").to_dict("records")
    candidate_summary = archive_candidate_revision(root, PHASE_2I_SESSION_ID, current) if apply else revision_summary(
        current,
        revision_id=f"automatic-candidate-{_candidate_semantic_hash(PHASE_2I_SESSION_ID, current)[:24]}",
        role="automatic_candidate",
        output_dir=_revision_root(root, PHASE_2I_SESSION_ID) / f"automatic-candidate-{_candidate_semantic_hash(PHASE_2I_SESSION_ID, current)[:24]}",
    )
    restored = _build_reviewed_official_entry(root, current, corrected_rows, authority_output, candidate_summary)
    result = {
        "status": "ready_to_apply" if not apply else "applied",
        "session_id": PHASE_2I_SESSION_ID,
        "current_candidate_sha256": current_csv_hash,
        "candidate_revision": candidate_summary,
        "reviewed_summary": _status_counts(restored.get("attendance_data") or []),
        "official_authority": restored.get("official_recognition_authority"),
        "automatic_authority": restored.get("automatic_recognition_authority"),
        "recognition_repeated": False,
        "video_reprocessed": False,
    }
    if not apply:
        return result

    backups = root / "data" / "state_backups"
    backups.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = backups / f"phase_2j_before_revision_repair_{stamp}.json"
    if backup_path.exists():
        raise ReportRevisionError(f"Refusing to overwrite state backup: {backup_path}")
    previous_registry_sha256 = (
        hashlib.sha256(previous_registry_bytes).hexdigest()
        if previous_registry_bytes is not None
        else None
    )
    atomic_write_json(
        backup_path,
        {
            "schema_version": 2,
            "policy_version": POLICY_VERSION,
            "session_id": PHASE_2I_SESSION_ID,
            "source_entry_sha256": _entry_hash(current),
            "source_entry": current,
            "registry_existed_before_apply": previous_registry_bytes is not None,
            "registry_before_sha256": previous_registry_sha256,
            "registry_before_base64": (
                base64.b64encode(previous_registry_bytes).decode("ascii")
                if previous_registry_bytes is not None
                else None
            ),
            "installed_registry_sha256": installed_registry_sha256,
            "created_at_utc": _now_utc(),
        },
    )
    next_store = dict(store)
    next_store[PHASE_2I_SESSION_ID] = restored
    try:
        atomic_write_json(status_path, next_store)
        committed = read_json_object(status_path).get(PHASE_2I_SESSION_ID)
        if not isinstance(committed, dict) or str(committed.get("authority_revision_id") or "") != AUTHORITY_REVISION_ID:
            raise ReportRevisionError("Phase 2J repair commit verification failed")
        if _status_counts(committed.get("attendance_data") or []) != EXPECTED_REVIEWED_COUNTS:
            raise ReportRevisionError("Phase 2J repair totals changed during commit")
    except Exception:
        atomic_write_json(status_path, store)
        backup_path.unlink(missing_ok=True)
        raise
    result["backup_path"] = str(backup_path)
    result["official_entry"] = restored
    return result

def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        delete=False,
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def rollback_mon_p4_repair(repo_root: Path, backup_path: Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    backups_root = (root / "data" / "state_backups").resolve()
    backup = Path(backup_path).resolve()
    try:
        backup.relative_to(backups_root)
    except ValueError as exc:
        raise ReportRevisionError("Rollback backup must be inside data/state_backups") from exc
    if not backup.is_file():
        raise ReportRevisionError(f"Phase 2J state backup not found: {backup}")
    try:
        payload = json.loads(backup.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise ReportRevisionError(f"Could not read Phase 2J state backup: {exc}") from exc
    if payload.get("schema_version") != 2 or payload.get("policy_version") != POLICY_VERSION:
        raise ReportRevisionError("Rollback backup is not a Product Phase 2J v2 state backup")
    if str(payload.get("session_id") or "") != PHASE_2I_SESSION_ID:
        raise ReportRevisionError("Rollback backup session changed")
    source_entry = payload.get("source_entry")
    if not isinstance(source_entry, dict):
        raise ReportRevisionError("Rollback backup has no source attendance entry")
    if _entry_hash(source_entry) != str(payload.get("source_entry_sha256") or ""):
        raise ReportRevisionError("Rollback backup source entry hash changed")

    registry_existed_before = bool(payload.get("registry_existed_before_apply"))
    registry_before_bytes: bytes | None = None
    if registry_existed_before:
        encoded = payload.get("registry_before_base64")
        if not isinstance(encoded, str) or not encoded:
            raise ReportRevisionError("Rollback backup is missing the previous review registry")
        try:
            registry_before_bytes = base64.b64decode(encoded, validate=True)
        except Exception as exc:
            raise ReportRevisionError("Rollback backup previous registry is not valid base64") from exc
        expected_before_hash = str(payload.get("registry_before_sha256") or "")
        if hashlib.sha256(registry_before_bytes).hexdigest() != expected_before_hash:
            raise ReportRevisionError("Rollback backup previous registry hash changed")
    elif payload.get("registry_before_base64") not in {None, ""}:
        raise ReportRevisionError("Rollback backup registry-presence contract is inconsistent")

    status_path = root / "data" / "attendance_status.json"
    store = read_json_object(status_path)
    current = store.get(PHASE_2I_SESSION_ID)
    if not isinstance(current, dict) or str(current.get("authority_revision_id") or "") != AUTHORITY_REVISION_ID:
        raise ReportRevisionError("Current MON_P4 entry is not the Phase 2I reviewed authority revision")
    if _status_counts(current.get("attendance_data") or []) != EXPECTED_REVIEWED_COUNTS:
        raise ReportRevisionError("Current reviewed authority totals changed; refusing rollback")

    registry_path = root / "data" / "review_evidence_registry.json"
    installed_registry_sha256 = str(payload.get("installed_registry_sha256") or "")
    current_registry_bytes = registry_path.read_bytes() if registry_path.is_file() else None
    if not installed_registry_sha256:
        raise ReportRevisionError("Rollback backup is missing the installed review-registry hash")
    if current_registry_bytes is None:
        raise ReportRevisionError("Installed Product Phase 2J review registry is missing")
    if hashlib.sha256(current_registry_bytes).hexdigest() != installed_registry_sha256:
        raise ReportRevisionError("Installed review registry changed; refusing rollback")
    try:
        current_registry_payload = json.loads(current_registry_bytes.decode("utf-8-sig"))
    except Exception as exc:
        raise ReportRevisionError(f"Installed review registry is unreadable: {exc}") from exc
    if str(current_registry_payload.get("policy_version") or "") != "product-phase-2j-exact-review-carry-forward-v2":
        raise ReportRevisionError("Installed review registry is not owned by Product Phase 2J")

    next_store = dict(store)
    next_store[PHASE_2I_SESSION_ID] = source_entry
    try:
        atomic_write_json(status_path, next_store)
        committed = read_json_object(status_path).get(PHASE_2I_SESSION_ID)
        if not isinstance(committed, dict) or _entry_hash(committed) != _entry_hash(source_entry):
            raise ReportRevisionError("Phase 2J rollback attendance commit verification failed")

        if registry_before_bytes is None:
            registry_path.unlink(missing_ok=False)
            if registry_path.exists():
                raise ReportRevisionError("Phase 2J rollback could not remove the installed review registry")
            registry_action = "removed_phase_2j_registry"
        else:
            _atomic_write_bytes(registry_path, registry_before_bytes)
            if hashlib.sha256(registry_path.read_bytes()).hexdigest() != str(payload.get("registry_before_sha256") or ""):
                raise ReportRevisionError("Phase 2J rollback previous registry verification failed")
            registry_action = "restored_previous_registry"
    except Exception:
        atomic_write_json(status_path, store)
        _atomic_write_bytes(registry_path, current_registry_bytes)
        raise

    return {
        "status": "rolled_back",
        "session_id": PHASE_2I_SESSION_ID,
        "restored_summary": _status_counts(source_entry.get("attendance_data") or []),
        "restored_authority": source_entry.get("official_recognition_authority"),
        "registry_action": registry_action,
        "candidate_revision_outputs_preserved_for_audit": True,
        "recognition_repeated": False,
        "video_reprocessed": False,
    }
