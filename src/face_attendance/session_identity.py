from __future__ import annotations

from datetime import datetime
import re
from typing import Any

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
UNSAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")
RUNTIME_ROW_FIELDS = {
    "att_status",
    "rt_status",
    "controller",
    "legacy_status",
    "temporary_state",
    "frontend_state",
}


def normalize_period_label(period: str | int) -> str:
    raw = str(period or "").strip().upper()
    if not raw:
        raise ValueError("Period is required for attendance session identity.")
    return raw if raw.startswith("P") else f"P{raw}"


def normalize_identity_part(value: Any, field_name: str) -> str:
    text = str(value or "").strip().upper()
    if not text:
        raise ValueError(f"{field_name} is required for attendance session identity.")
    text = text.replace("/", "_").replace("\\", "_")
    text = UNSAFE_RE.sub("_", text).strip("_")
    if not text:
        raise ValueError(f"{field_name} is required for attendance session identity.")
    return text


def normalize_session_date(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Attendance date is required for this footage source.")
    if not DATE_RE.match(text):
        raise ValueError("Attendance date must use YYYY-MM-DD format.")
    try:
        datetime.strptime(text, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("Attendance date must be a valid calendar date.") from exc
    return text


def build_attendance_session_id(
    session_date: Any,
    section: Any,
    period: Any,
    subject_abbr: Any,
) -> str:
    date_text = normalize_session_date(session_date)
    section_text = normalize_identity_part(section, "Section")
    period_text = normalize_period_label(period)
    subject_text = normalize_identity_part(subject_abbr, "Subject/course")
    return f"{date_text}__{section_text}__{period_text}__{subject_text}"


def parse_attendance_session_id(session_id: Any) -> dict[str, str]:
    text = str(session_id or "").strip()
    parts = text.split("__")
    if len(parts) != 4:
        raise ValueError("Invalid attendance session_id format.")
    session_date, section, period, subject_abbr = parts
    return {
        "session_id": text,
        "session_date": normalize_session_date(session_date),
        "section": normalize_identity_part(section, "Section"),
        "period": normalize_period_label(period),
        "subject_abbr": normalize_identity_part(subject_abbr, "Subject/course"),
    }


def clean_filename(value: Any) -> str:
    text = str(value or "").strip()
    return UNSAFE_RE.sub("_", text).strip("_") or "session"


def strip_runtime_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Remove nested/transient timetable status fields before persistence."""
    clean: dict[str, Any] = {}
    for key, value in dict(row or {}).items():
        if key in RUNTIME_ROW_FIELDS:
            continue
        clean[key] = value
    return clean
