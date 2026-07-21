from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


STATUS_TRANSIENT_KEYS = {
    "att_status",
    "rt_status",
    "controller",
    "legacy_status",
    "temporary_state",
    "frontend_state",
}
RUNNING_JOB_STATUSES = {"Pending", "Processing"}
JOB_REGISTRY_SCHEMA_VERSION = 1


class WorkflowStateError(RuntimeError):
    """Raised when persistent workflow state cannot be trusted safely."""


@dataclass(frozen=True)
class StatusSanitizationReport:
    changed: bool
    entries_changed: int
    removed_fields: int
    invalid_entries: int

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "changed": self.changed,
            "entries_changed": self.entries_changed,
            "removed_fields": self.removed_fields,
            "invalid_entries": self.invalid_entries,
        }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _sanitize_status_value(value: Any, counters: dict[str, int]) -> Any:
    if isinstance(value, Mapping):
        clean: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            if key in STATUS_TRANSIENT_KEYS:
                counters["removed_fields"] += 1
                continue
            clean[key] = _sanitize_status_value(raw_value, counters)
        return clean
    if isinstance(value, list):
        return [_sanitize_status_value(item, counters) for item in value]
    return _json_safe(value)


def sanitize_status_store(payload: Any) -> tuple[dict[str, Any], StatusSanitizationReport]:
    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        raise WorkflowStateError("attendance_status.json must contain a JSON object at the top level.")

    clean: dict[str, Any] = {}
    entries_changed = 0
    removed_fields = 0
    invalid_entries = 0

    for raw_key, raw_entry in payload.items():
        key = str(raw_key)
        if not isinstance(raw_entry, Mapping):
            invalid_entries += 1
            clean[key] = _json_safe(raw_entry)
            continue
        counters = {"removed_fields": 0}
        sanitized = _sanitize_status_value(raw_entry, counters)
        clean[key] = sanitized
        if counters["removed_fields"] or sanitized != raw_entry:
            entries_changed += 1
        removed_fields += counters["removed_fields"]

    changed = clean != dict(payload)
    return clean, StatusSanitizationReport(
        changed=changed,
        entries_changed=entries_changed,
        removed_fields=removed_fields,
        invalid_entries=invalid_entries,
    )


def read_json_object(path: Path, *, missing_default: dict[str, Any] | None = None) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return dict(missing_default or {})
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise WorkflowStateError(f"Could not parse trusted JSON state: {path}") from exc
    if not isinstance(payload, dict):
        raise WorkflowStateError(f"Trusted JSON state must be an object: {path}")
    return payload


def atomic_write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = _json_safe(payload)
    serialized = json.dumps(safe_payload, indent=2, ensure_ascii=False) + "\n"

    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        finally:
            raise


def migrate_status_file(
    status_path: Path,
    backup_dir: Path,
    *,
    timestamp: str | None = None,
) -> dict[str, Any]:
    status_path = Path(status_path)
    backup_dir = Path(backup_dir)
    payload = read_json_object(status_path, missing_default={})
    clean, report = sanitize_status_store(payload)

    backup_path: Path | None = None
    if report.changed and status_path.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"{status_path.stem}_before_sanitize_{stamp}{status_path.suffix or '.json'}"
        if backup_path.exists():
            raise WorkflowStateError(f"Refusing to overwrite existing workflow-state backup: {backup_path}")
        backup_path.write_bytes(status_path.read_bytes())
        atomic_write_json(status_path, clean)
    elif not status_path.exists():
        atomic_write_json(status_path, clean)

    return {
        **report.as_dict(),
        "status_path": str(status_path),
        "backup_path": str(backup_path) if backup_path else None,
    }


def compact_job_for_storage(job: Mapping[str, Any]) -> dict[str, Any]:
    hidden = {"process", "command", "stdout"}
    compact: dict[str, Any] = {}
    for raw_key, value in job.items():
        key = str(raw_key)
        if key in hidden or key.startswith("_"):
            continue
        if key == "result" and isinstance(value, Mapping):
            result = {
                str(result_key): result_value
                for result_key, result_value in value.items()
                if str(result_key) not in {"attendance_data", "stdout", "command"}
            }
            compact[key] = _json_safe(result)
        else:
            compact[key] = _json_safe(value)
    return compact


def serialize_job_registry(jobs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rows = [compact_job_for_storage(job) for job in jobs.values() if isinstance(job, Mapping)]
    rows.sort(key=lambda item: (str(item.get("created_at") or ""), str(item.get("job_id") or "")), reverse=True)
    return {
        "schema_version": JOB_REGISTRY_SCHEMA_VERSION,
        "jobs": rows,
    }


def load_job_registry(
    path: Path,
    *,
    now_text: str,
    interrupted_message: str,
) -> tuple[dict[str, dict[str, Any]], bool]:
    path = Path(path)
    if not path.exists():
        return {}, False
    payload = read_json_object(path)
    if payload.get("schema_version") != JOB_REGISTRY_SCHEMA_VERSION:
        raise WorkflowStateError(
            f"Unsupported job registry schema_version in {path}: {payload.get('schema_version')!r}"
        )
    rows = payload.get("jobs")
    if not isinstance(rows, list):
        raise WorkflowStateError(f"Job registry jobs must be a list: {path}")

    jobs: dict[str, dict[str, Any]] = {}
    changed = False
    for row in rows:
        if not isinstance(row, Mapping):
            raise WorkflowStateError(f"Job registry contains a non-object row: {path}")
        job = compact_job_for_storage(row)
        job_id = str(job.get("job_id") or "").strip().upper()
        if not job_id:
            raise WorkflowStateError(f"Job registry row is missing job_id: {path}")
        if job_id in jobs:
            raise WorkflowStateError(f"Job registry contains duplicate job_id {job_id}: {path}")
        job["job_id"] = job_id
        if str(job.get("status") or "") in RUNNING_JOB_STATUSES:
            job.update(
                {
                    "status": "Failed",
                    "completed_at": now_text,
                    "error": interrupted_message,
                    "progress_stage": "failed",
                    "progress_step": 0,
                    "progress_percent": 0,
                    "progress_text": "Interrupted by backend restart",
                }
            )
            changed = True
        jobs[job_id] = job
    return jobs, changed


def active_job_for_session(
    jobs: Mapping[str, Mapping[str, Any]],
    session_id: str,
) -> dict[str, Any] | None:
    target = str(session_id or "").strip()
    if not target:
        return None
    candidates = [
        compact_job_for_storage(job)
        for job in jobs.values()
        if isinstance(job, Mapping)
        and str(job.get("session_id") or job.get("key") or "") == target
        and str(job.get("status") or "") in RUNNING_JOB_STATUSES
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: (str(item.get("created_at") or ""), str(item.get("job_id") or "")), reverse=True)
    return candidates[0]
