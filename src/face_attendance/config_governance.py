from __future__ import annotations

import base64
import copy
import csv
import hashlib
import hmac
import io
import json
import os
import re
import secrets
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CONFIG_GOVERNANCE_SCHEMA_VERSION = 1
PASSWORD_HASH_PREFIX = "pbkdf2_sha256"
PASSWORD_HASH_ITERATIONS = 260_000
MAX_TIMETABLE_BYTES = 1_000_000
FACULTY_ID_PATTERN = re.compile(r"^[a-z][a-z0-9._-]{2,63}$")
USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9._-]{3,64}$")
REQUIRED_TIMETABLE_COLUMNS = (
    "slot_id",
    "academic_year",
    "year",
    "semester",
    "wef",
    "day",
    "period",
    "start_time",
    "end_time",
    "course_code",
    "course_abbr",
    "course_name",
    "instructor",
    "session_type",
    "room",
    "section",
    "camera_ids",
    "attendance_required",
    "notes",
)
DAY_ORDER = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
DAY_ABBR = {
    "Monday": "MON",
    "Tuesday": "TUE",
    "Wednesday": "WED",
    "Thursday": "THU",
    "Friday": "FRI",
}
PERIOD_WINDOWS = {
    "1": ("09:00", "09:50"),
    "2": ("09:50", "10:40"),
    "3": ("10:50", "11:40"),
    "4": ("11:40", "12:30"),
    "6": ("13:20", "14:10"),
    "7": ("14:20", "15:10"),
    "8": ("15:10", "16:00"),
}
EXPECTED_PERIODS = tuple(PERIOD_WINDOWS)


class ConfigGovernanceError(RuntimeError):
    """Raised when guarded configuration state cannot be trusted or changed safely."""


@dataclass(frozen=True)
class TimetableValidationResult:
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    file_sha256: str
    row_count: int
    rows: tuple[dict[str, str], ...]
    canonical_csv: bytes
    summary: dict[str, Any]

    def public_dict(self, *, include_rows: bool = False) -> dict[str, Any]:
        payload = {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "file_sha256": self.file_sha256,
            "row_count": self.row_count,
            "summary": copy.deepcopy(self.summary),
        }
        if include_rows:
            payload["rows"] = [dict(row) for row in self.rows]
        return payload


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def stable_digest(payload: Any) -> str:
    encoded = json.dumps(
        _json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, payload: Any) -> None:
    data = (json.dumps(_json_safe(payload), indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    _atomic_write_bytes(path, data)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise ConfigGovernanceError(f"Could not parse trusted JSON: {path}") from exc


def _subject_info(subject_catalog: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for raw_abbr, raw_info in subject_catalog.items():
        abbr = str(raw_abbr or "").strip().upper()
        if not abbr:
            continue
        info = dict(raw_info or {})
        result[abbr] = {
            "abbr": abbr,
            "course_code": str(info.get("course_code") or info.get("courseCode") or "").strip(),
            "course_name": str(info.get("course_name") or info.get("courseName") or abbr).strip(),
            "faculty_id": str(info.get("faculty_id") or info.get("facultyId") or "").strip(),
            "faculty_name": str(info.get("faculty_name") or info.get("facultyName") or "").strip(),
        }
    if not result:
        raise ConfigGovernanceError("No authoritative subject catalog is available.")
    return result


def hash_password(password: str, *, salt: bytes | None = None, iterations: int = PASSWORD_HASH_ITERATIONS) -> str:
    password = str(password or "")
    if len(password) < 8:
        raise ConfigGovernanceError("New passwords must contain at least 8 characters.")
    if iterations < 100_000:
        raise ConfigGovernanceError("Password hash iteration count is too weak.")
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"{PASSWORD_HASH_PREFIX}${iterations}${base64.urlsafe_b64encode(salt).decode('ascii')}${base64.urlsafe_b64encode(digest).decode('ascii')}"


def verify_password(user: Mapping[str, Any], password: str) -> bool:
    password = str(password or "")
    stored_hash = str(user.get("password_hash") or "")
    if stored_hash:
        parts = stored_hash.split("$")
        if len(parts) != 4 or parts[0] != PASSWORD_HASH_PREFIX:
            return False
        try:
            iterations = int(parts[1])
            salt = base64.urlsafe_b64decode(parts[2].encode("ascii"))
            expected = base64.urlsafe_b64decode(parts[3].encode("ascii"))
        except Exception:
            return False
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(actual, expected)
    return hmac.compare_digest(str(user.get("password") or ""), password)


def validate_role_users(
    users: Any,
    subject_catalog: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(users, list) or not users:
        raise ConfigGovernanceError("role_users.json must contain a non-empty list.")
    subjects = _subject_info(subject_catalog)
    subject_order = list(subjects)
    clean: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_usernames: set[str] = set()
    admin_count = 0

    for index, raw in enumerate(users, start=1):
        if not isinstance(raw, Mapping):
            raise ConfigGovernanceError(f"Role row {index} is not an object.")
        row = copy.deepcopy(dict(raw))
        user_id = str(row.get("id") or "").strip().lower()
        username = str(row.get("username") or "").strip()
        name = str(row.get("name") or "").strip()
        role = str(row.get("role") or "").strip().lower()
        if not FACULTY_ID_PATTERN.fullmatch(user_id):
            raise ConfigGovernanceError(f"Invalid user id: {user_id!r}")
        if not USERNAME_PATTERN.fullmatch(username):
            raise ConfigGovernanceError(f"Invalid username for {user_id}.")
        if not name or len(name) > 120:
            raise ConfigGovernanceError(f"Invalid display name for {user_id}.")
        if role not in {"admin", "faculty"}:
            raise ConfigGovernanceError(f"Unsupported role for {user_id}: {role!r}")
        if user_id in seen_ids:
            raise ConfigGovernanceError(f"Duplicate user id: {user_id}")
        username_key = username.casefold()
        if username_key in seen_usernames:
            raise ConfigGovernanceError(f"Duplicate username: {username}")
        seen_ids.add(user_id)
        seen_usernames.add(username_key)

        has_legacy_password = isinstance(row.get("password"), str) and bool(row.get("password"))
        has_password_hash = isinstance(row.get("password_hash"), str) and bool(row.get("password_hash"))
        if not (has_legacy_password or has_password_hash):
            raise ConfigGovernanceError(f"User {user_id} has no credential.")
        if has_password_hash and not str(row["password_hash"]).startswith(f"{PASSWORD_HASH_PREFIX}$"):
            raise ConfigGovernanceError(f"User {user_id} has an unsupported password hash.")

        assigned = []
        for raw_subject in row.get("subjects") or []:
            subject = str(raw_subject or "").strip().upper()
            if subject not in subjects:
                raise ConfigGovernanceError(f"User {user_id} references unknown subject {subject!r}.")
            if subject not in assigned:
                assigned.append(subject)
        assigned.sort(key=subject_order.index)

        if role == "admin":
            admin_count += 1
            row["canManageSystem"] = True
            row["canSeeAll"] = True
            assigned = subject_order[:]
        else:
            row["canManageSystem"] = False
            row["canSeeAll"] = False

        row.update(
            {
                "id": user_id,
                "username": username,
                "name": name,
                "facultyName": name,
                "role": role,
                "subjects": assigned,
            }
        )
        clean.append(row)

    if admin_count != 1:
        raise ConfigGovernanceError("Exactly one HOD/Admin account is required.")
    return clean


def _faculty_role_label(subjects: Sequence[str], subject_catalog: Mapping[str, Mapping[str, Any]]) -> str:
    catalog = _subject_info(subject_catalog)
    if not subjects:
        return "Faculty - Unassigned"
    if len(subjects) == 1:
        return f"Faculty - {catalog[subjects[0]]['course_name']}"
    return "Faculty - " + ", ".join(subjects)


def create_or_update_faculty(
    users: Sequence[Mapping[str, Any]],
    subject_catalog: Mapping[str, Mapping[str, Any]],
    payload: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    clean = validate_role_users(list(users), subject_catalog)
    action = str(payload.get("action") or "").strip().lower()
    if action not in {"create", "update"}:
        raise ConfigGovernanceError("Faculty action must be create or update.")

    faculty_id = str(payload.get("faculty_id") or payload.get("id") or "").strip().lower()
    username = str(payload.get("username") or "").strip()
    name = str(payload.get("name") or "").strip()
    password = str(payload.get("password") or "")
    if not FACULTY_ID_PATTERN.fullmatch(faculty_id):
        raise ConfigGovernanceError("Faculty id must start with a letter and contain only letters, numbers, dot, underscore, or hyphen.")
    if faculty_id in {"admin", "hod"}:
        raise ConfigGovernanceError("The HOD account cannot be created or edited through faculty management.")
    if not USERNAME_PATTERN.fullmatch(username):
        raise ConfigGovernanceError("Username must contain 3-64 letters, numbers, dot, underscore, or hyphen.")
    if not name or len(name) > 120:
        raise ConfigGovernanceError("Faculty name is required and must be at most 120 characters.")

    by_id = {str(row["id"]): row for row in clean}
    existing = by_id.get(faculty_id)
    if action == "create" and existing:
        raise ConfigGovernanceError(f"Faculty id {faculty_id} already exists.")
    if action == "update" and not existing:
        raise ConfigGovernanceError(f"Faculty id {faculty_id} does not exist.")
    if existing and existing.get("role") != "faculty":
        raise ConfigGovernanceError("Only faculty accounts can be edited here.")

    for row in clean:
        if str(row["id"]) != faculty_id and str(row["username"]).casefold() == username.casefold():
            raise ConfigGovernanceError(f"Username {username!r} is already in use.")

    if action == "create":
        if len(password) < 8:
            raise ConfigGovernanceError("A new faculty account requires a password of at least 8 characters.")
        row = {
            "id": faculty_id,
            "username": username,
            "password_hash": hash_password(password),
            "name": name,
            "role": "faculty",
            "roleLabel": "Faculty - Unassigned",
            "subjects": [],
            "facultyName": name,
            "canManageSystem": False,
            "canSeeAll": False,
        }
        clean.append(row)
        changed_fields = ["account_created"]
    else:
        row = existing
        changed_fields = []
        if row.get("username") != username:
            row["username"] = username
            changed_fields.append("username")
        if row.get("name") != name:
            row["name"] = name
            row["facultyName"] = name
            changed_fields.append("name")
        if password:
            row["password_hash"] = hash_password(password)
            row.pop("password", None)
            changed_fields.append("password")
        row["roleLabel"] = _faculty_role_label(row.get("subjects") or [], subject_catalog)

    result = validate_role_users(clean, subject_catalog)
    return result, {
        "action": action,
        "faculty_id": faculty_id,
        "username": username,
        "name": name,
        "changed_fields": changed_fields,
        "password_changed": "password" in changed_fields or action == "create",
    }


def _split_components(value: Any) -> list[str]:
    return [part.strip() for part in str(value or "").split("/")]


def _normalize_period(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text.startswith("P"):
        text = text[1:]
    if not text.isdigit():
        return ""
    return str(int(text))


def _normalize_day(value: Any) -> str:
    text = str(value or "").strip().casefold()
    for day in DAY_ORDER:
        if text in {day.casefold(), day[:3].casefold()}:
            return day
    return ""


def _canonical_timetable_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(REQUIRED_TIMETABLE_COLUMNS), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: str(row.get(column) or "") for column in REQUIRED_TIMETABLE_COLUMNS})
    return stream.getvalue().encode("utf-8")


def validate_timetable_bytes(
    payload: bytes,
    subject_catalog: Mapping[str, Mapping[str, Any]],
    role_users: Sequence[Mapping[str, Any]],
) -> TimetableValidationResult:
    if not isinstance(payload, (bytes, bytearray)):
        raise ConfigGovernanceError("Timetable upload must be bytes.")
    payload = bytes(payload)
    if not payload:
        raise ConfigGovernanceError("Timetable CSV is empty.")
    if len(payload) > MAX_TIMETABLE_BYTES:
        raise ConfigGovernanceError("Timetable CSV exceeds the 1 MB safety limit.")

    file_hash = sha256_bytes(payload)
    catalog = _subject_info(subject_catalog)
    users = validate_role_users(list(role_users), subject_catalog)
    faculty_by_subject: dict[str, str] = {}
    for user in users:
        if user.get("role") != "faculty":
            continue
        for subject in user.get("subjects") or []:
            if subject in faculty_by_subject:
                raise ConfigGovernanceError(f"Subject {subject} is assigned to more than one faculty account.")
            faculty_by_subject[subject] = str(user.get("name") or "")

    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ConfigGovernanceError("Timetable CSV must be UTF-8 encoded.") from exc

    reader = csv.DictReader(io.StringIO(text))
    headers = tuple(reader.fieldnames or ())
    errors: list[str] = []
    warnings: list[str] = []
    missing_headers = [column for column in REQUIRED_TIMETABLE_COLUMNS if column not in headers]
    extra_headers = [column for column in headers if column not in REQUIRED_TIMETABLE_COLUMNS]
    if missing_headers:
        errors.append("Missing required columns: " + ", ".join(missing_headers))
    if extra_headers:
        warnings.append("Extra columns will be dropped: " + ", ".join(extra_headers))

    normalized_rows: list[dict[str, str]] = []
    seen_slots: set[str] = set()
    seen_day_period: set[tuple[str, str]] = set()
    day_counts = {day: 0 for day in DAY_ORDER}
    subject_counts = {subject: 0 for subject in catalog}
    sections: set[str] = set()
    wef_values: set[str] = set()

    if not missing_headers:
        for row_number, raw_row in enumerate(reader, start=2):
            row = {column: str(raw_row.get(column) or "").strip() for column in REQUIRED_TIMETABLE_COLUMNS}
            day = _normalize_day(row["day"])
            period = _normalize_period(row["period"])
            if not day:
                errors.append(f"Row {row_number}: invalid day {row['day']!r}.")
            if period not in PERIOD_WINDOWS:
                errors.append(f"Row {row_number}: unsupported period {row['period']!r}.")
            if not day or period not in PERIOD_WINDOWS:
                continue

            expected_slot = f"{DAY_ABBR[day]}_P{period}"
            slot_id = row["slot_id"].upper()
            if slot_id != expected_slot:
                errors.append(f"Row {row_number}: slot_id must be {expected_slot}.")
            if slot_id in seen_slots:
                errors.append(f"Row {row_number}: duplicate slot_id {slot_id}.")
            if (day, period) in seen_day_period:
                errors.append(f"Row {row_number}: duplicate {day} P{period}.")
            seen_slots.add(slot_id)
            seen_day_period.add((day, period))

            expected_start, expected_end = PERIOD_WINDOWS[period]
            if row["start_time"] != expected_start or row["end_time"] != expected_end:
                errors.append(
                    f"Row {row_number}: P{period} must use {expected_start}-{expected_end}, not {row['start_time']}-{row['end_time']}."
                )

            subjects = [part.upper() for part in _split_components(row["course_abbr"]) if part]
            codes = [part for part in _split_components(row["course_code"]) if part]
            names = [part for part in _split_components(row["course_name"]) if part]
            instructors = [part for part in _split_components(row["instructor"]) if part]
            if not subjects:
                errors.append(f"Row {row_number}: course_abbr is required.")
                continue
            unknown = [subject for subject in subjects if subject not in catalog]
            if unknown:
                errors.append(f"Row {row_number}: unknown subject(s): {', '.join(unknown)}.")
                continue
            if len(codes) != len(subjects) or len(names) != len(subjects) or len(instructors) != len(subjects):
                errors.append(f"Row {row_number}: course, name, and instructor components must match course_abbr order.")
                continue

            for index, subject in enumerate(subjects):
                info = catalog[subject]
                if codes[index].casefold() != info["course_code"].casefold():
                    errors.append(f"Row {row_number}: {subject} course code must be {info['course_code']}.")
                if names[index].casefold() != info["course_name"].casefold():
                    errors.append(f"Row {row_number}: {subject} course name must be {info['course_name']}.")
                expected_faculty = faculty_by_subject.get(subject)
                if not expected_faculty:
                    errors.append(f"Row {row_number}: {subject} has no assigned faculty account.")
                elif instructors[index].casefold() != expected_faculty.casefold():
                    warnings.append(
                        f"Row {row_number}: {subject} instructor {instructors[index]!r} differs from assigned faculty {expected_faculty!r}; the guarded write will normalize it."
                    )
                    instructors[index] = expected_faculty
                subject_counts[subject] += 1
            row["instructor"] = " / ".join(instructors)

            if row["section"].upper() != "B51":
                errors.append(f"Row {row_number}: section must remain B51.")
            if not row["academic_year"] or not row["year"] or not row["semester"] or not row["wef"]:
                errors.append(f"Row {row_number}: academic year, year, semester, and WEF are required.")
            if row["attendance_required"].casefold() not in {"yes", "no"}:
                errors.append(f"Row {row_number}: attendance_required must be Yes or No.")
            if not row["camera_ids"]:
                errors.append(f"Row {row_number}: camera_ids is required.")
            if not row["room"] or not row["session_type"]:
                errors.append(f"Row {row_number}: room and session_type are required.")

            row.update(
                {
                    "slot_id": expected_slot,
                    "day": day,
                    "period": period,
                    "course_abbr": "/".join(subjects),
                    "attendance_required": "Yes" if row["attendance_required"].casefold() == "yes" else "No",
                    "section": row["section"].upper(),
                }
            )
            day_counts[day] += 1
            sections.add(row["section"])
            wef_values.add(row["wef"])
            normalized_rows.append(row)

    expected_slots = {(day, period) for day in DAY_ORDER for period in EXPECTED_PERIODS}
    actual_slots = set(seen_day_period)
    missing_slots = sorted(expected_slots - actual_slots, key=lambda item: (DAY_ORDER.index(item[0]), EXPECTED_PERIODS.index(item[1])))
    extra_slots = sorted(actual_slots - expected_slots)
    if missing_slots:
        errors.append("Missing timetable slots: " + ", ".join(f"{day[:3].upper()}_P{period}" for day, period in missing_slots))
    if extra_slots:
        errors.append("Unexpected timetable slots: " + ", ".join(f"{day[:3].upper()}_P{period}" for day, period in extra_slots))
    if len(normalized_rows) != 35:
        errors.append(f"Timetable must contain exactly 35 class rows; found {len(normalized_rows)}.")
    if len(wef_values) > 1:
        errors.append("All timetable rows must use one WEF date.")
    if sections and sections != {"B51"}:
        errors.append("All timetable rows must remain in section B51.")

    # Preserve order only after the complete slot set has been validated.
    normalized_rows.sort(key=lambda row: (DAY_ORDER.index(row["day"]), EXPECTED_PERIODS.index(row["period"])))
    canonical = _canonical_timetable_bytes(normalized_rows)
    warnings = list(dict.fromkeys(warnings))
    errors = list(dict.fromkeys(errors))
    summary = {
        "days": day_counts,
        "subjects": subject_counts,
        "sections": sorted(sections),
        "wef": sorted(wef_values),
        "expected_rows": 35,
        "normalized_sha256": sha256_bytes(canonical),
    }
    return TimetableValidationResult(
        valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        file_sha256=file_hash,
        row_count=len(normalized_rows),
        rows=tuple(normalized_rows),
        canonical_csv=canonical,
        summary=summary,
    )


def update_timetable_subject_faculty(
    rows: Sequence[Mapping[str, Any]],
    subject: str,
    faculty_name: str,
) -> list[dict[str, str]]:
    subject = str(subject or "").strip().upper()
    faculty_name = str(faculty_name or "").strip()
    if not subject or not faculty_name:
        raise ConfigGovernanceError("Subject and faculty name are required.")
    updated: list[dict[str, str]] = []
    changed = 0
    for raw in rows:
        row = {column: str(raw.get(column) or "") for column in REQUIRED_TIMETABLE_COLUMNS}
        subjects = [part.upper() for part in _split_components(row["course_abbr"]) if part]
        instructors = [part for part in _split_components(row["instructor"]) if part]
        if len(subjects) != len(instructors):
            raise ConfigGovernanceError(f"Timetable row {row.get('slot_id')} has an ambiguous instructor mapping.")
        if subject in subjects:
            instructors[subjects.index(subject)] = faculty_name
            row["instructor"] = " / ".join(instructors)
            changed += 1
        updated.append(row)
    if not changed:
        raise ConfigGovernanceError(f"Subject {subject} does not appear in the timetable.")
    return updated


def assign_subject_to_faculty(
    users: Sequence[Mapping[str, Any]],
    mapping: Mapping[str, Any],
    timetable_rows: Sequence[Mapping[str, Any]],
    *,
    subject: str,
    faculty_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, str]], dict[str, Any]]:
    subject_catalog = mapping.get("subjects") or {}
    catalog = _subject_info(subject_catalog)
    subject = str(subject or "").strip().upper()
    faculty_id = str(faculty_id or "").strip().lower()
    if subject not in catalog:
        raise ConfigGovernanceError(f"Unknown subject {subject!r}.")

    clean_users = validate_role_users(list(users), subject_catalog)
    target = next((row for row in clean_users if row["id"] == faculty_id and row["role"] == "faculty"), None)
    if not target:
        raise ConfigGovernanceError(f"Faculty account {faculty_id!r} was not found.")
    previous = catalog[subject].get("faculty_id") or None

    subject_order = list(catalog)
    for row in clean_users:
        if row["role"] != "faculty":
            continue
        assigned = [item for item in row.get("subjects") or [] if item != subject]
        if row["id"] == faculty_id:
            assigned.append(subject)
        assigned = sorted(set(assigned), key=subject_order.index)
        row["subjects"] = assigned
        row["roleLabel"] = _faculty_role_label(assigned, subject_catalog)
        row["facultyName"] = row["name"]

    new_mapping = copy.deepcopy(dict(mapping))
    if not isinstance(new_mapping.get("subjects"), dict):
        raise ConfigGovernanceError("student_faculty_map.json subjects must be an object.")
    info = dict(new_mapping["subjects"].get(subject) or {})
    info["facultyId"] = faculty_id
    info["facultyName"] = target["name"]
    if "faculty_id" in info:
        info["faculty_id"] = faculty_id
    if "faculty_name" in info:
        info["faculty_name"] = target["name"]
    new_mapping["subjects"][subject] = info

    new_rows = update_timetable_subject_faculty(timetable_rows, subject, target["name"])
    validate_role_users(clean_users, new_mapping["subjects"])
    return clean_users, new_mapping, new_rows, {
        "subject": subject,
        "previous_faculty_id": previous,
        "faculty_id": faculty_id,
        "faculty_name": target["name"],
    }


def json_bytes(payload: Any) -> bytes:
    return (json.dumps(_json_safe(payload), indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def timetable_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return _canonical_timetable_bytes(rows)


def _relative_target(repo_root: Path, path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(repo_root).resolve()).as_posix()
    except ValueError as exc:
        raise ConfigGovernanceError(f"Configuration target is outside the repository: {path}") from exc


def _public_actor(actor: Mapping[str, Any]) -> dict[str, str]:
    actor_id = str(actor.get("id") or "").strip()
    actor_name = str(actor.get("name") or actor_id).strip()
    if not actor_id:
        raise ConfigGovernanceError("Configuration actor is missing an id.")
    return {"id": actor_id, "name": actor_name}


def apply_config_transaction(
    *,
    repo_root: Path,
    targets: Mapping[Path, bytes],
    backup_root: Path,
    audit_root: Path,
    pointer_path: Path,
    actor: Mapping[str, Any],
    change_type: str,
    reason: str,
    metadata: Mapping[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    backup_root = Path(backup_root)
    audit_root = Path(audit_root)
    pointer_path = Path(pointer_path)
    reason = str(reason or "").strip()
    change_type = str(change_type or "").strip()
    if len(reason) < 8:
        raise ConfigGovernanceError("A configuration-change reason of at least 8 characters is required.")
    if not change_type:
        raise ConfigGovernanceError("Configuration change type is required.")
    if not targets:
        raise ConfigGovernanceError("No configuration targets were supplied.")

    actor_public = _public_actor(actor)
    before_bytes: dict[str, bytes] = {}
    after_bytes: dict[str, bytes] = {}
    path_by_relative: dict[str, Path] = {}
    for raw_path, raw_payload in targets.items():
        path = Path(raw_path)
        if not path.is_file():
            raise ConfigGovernanceError(f"Configuration target is missing: {path}")
        relative = _relative_target(repo_root, path)
        if relative in path_by_relative:
            raise ConfigGovernanceError(f"Duplicate configuration target: {relative}")
        path_by_relative[relative] = path
        before_bytes[relative] = path.read_bytes()
        after_bytes[relative] = bytes(raw_payload)

    before_hashes = {relative: sha256_bytes(payload) for relative, payload in before_bytes.items()}
    after_hashes = {relative: sha256_bytes(payload) for relative, payload in after_bytes.items()}
    changed_files = [relative for relative in sorted(path_by_relative) if before_hashes[relative] != after_hashes[relative]]
    if not changed_files:
        raise ConfigGovernanceError("The requested configuration is identical to the current state.")

    current_pointer = None
    if pointer_path.is_file():
        current_pointer = _read_json(pointer_path)
        if not isinstance(current_pointer, dict):
            raise ConfigGovernanceError("Configuration pointer must be a JSON object.")
        if current_pointer.get("status") == "applying":
            raise ConfigGovernanceError("A previous configuration transaction is incomplete.")

    now = now or datetime.now(timezone.utc)
    created_at = now.isoformat()
    digest_payload = {
        "created_at": created_at,
        "actor": actor_public,
        "change_type": change_type,
        "reason": reason,
        "before": before_hashes,
        "after": after_hashes,
    }
    change_id = f"config-{now.strftime('%Y%m%dT%H%M%SZ')}-{stable_digest(digest_payload)[:12]}"
    backup_dir = backup_root / change_id
    audit_dir = audit_root / change_id
    if backup_dir.exists() or audit_dir.exists():
        raise ConfigGovernanceError(f"Configuration change id already exists: {change_id}")

    pending_backup = backup_root / f".pending-{change_id}"
    pending_audit = audit_root / f".pending-{change_id}"
    pending_backup.mkdir(parents=True, exist_ok=False)
    pending_audit.mkdir(parents=True, exist_ok=False)

    try:
        for relative, payload in before_bytes.items():
            backup_path = pending_backup / "before" / Path(*relative.split("/"))
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            backup_path.write_bytes(payload)

        backup_manifest = {
            "schema_version": CONFIG_GOVERNANCE_SCHEMA_VERSION,
            "change_id": change_id,
            "created_at": created_at,
            "target_files": sorted(path_by_relative),
            "before_sha256": before_hashes,
            "after_sha256": after_hashes,
        }
        _atomic_write_json(pending_backup / "backup_manifest.json", backup_manifest)

        written: list[str] = []
        try:
            for relative in sorted(path_by_relative):
                _atomic_write_bytes(path_by_relative[relative], after_bytes[relative])
                written.append(relative)
            for relative, expected in after_hashes.items():
                if sha256_file(path_by_relative[relative]) != expected:
                    raise ConfigGovernanceError(f"Post-write hash mismatch for {relative}.")
        except Exception:
            for relative in written:
                _atomic_write_bytes(path_by_relative[relative], before_bytes[relative])
            raise

        audit_record = {
            "schema_version": CONFIG_GOVERNANCE_SCHEMA_VERSION,
            "change_id": change_id,
            "status": "applied",
            "change_type": change_type,
            "created_at": created_at,
            "actor": actor_public,
            "reason": reason,
            "changed_files": changed_files,
            "before_sha256": before_hashes,
            "after_sha256": after_hashes,
            "metadata": _json_safe(metadata or {}),
            "recognition_executed": False,
            "attendance_changed": False,
            "embeddings_changed": False,
            "roster_changed": False,
        }
        _atomic_write_json(pending_audit / "audit_record.json", audit_record)

        backup_dir.parent.mkdir(parents=True, exist_ok=True)
        audit_dir.parent.mkdir(parents=True, exist_ok=True)
        pending_backup.rename(backup_dir)
        pending_audit.rename(audit_dir)

        pointer = {
            "schema_version": CONFIG_GOVERNANCE_SCHEMA_VERSION,
            "status": "applied",
            "change_id": change_id,
            "change_type": change_type,
            "created_at": created_at,
            "actor": actor_public,
            "reason": reason,
            "target_files": sorted(path_by_relative),
            "before_sha256": before_hashes,
            "after_sha256": after_hashes,
            "backup_dir": _relative_target(repo_root, backup_dir),
            "audit_dir": _relative_target(repo_root, audit_dir),
        }
        try:
            _atomic_write_json(pointer_path, pointer)
        except Exception:
            for relative in sorted(path_by_relative):
                _atomic_write_bytes(path_by_relative[relative], before_bytes[relative])
            shutil.rmtree(backup_dir, ignore_errors=True)
            shutil.rmtree(audit_dir, ignore_errors=True)
            raise
        return {**audit_record, "backup_dir": str(backup_dir), "audit_dir": str(audit_dir)}
    except Exception:
        shutil.rmtree(pending_backup, ignore_errors=True)
        shutil.rmtree(pending_audit, ignore_errors=True)
        raise


def load_config_history(audit_root: Path, *, limit: int = 20) -> list[dict[str, Any]]:
    audit_root = Path(audit_root)
    rows: list[dict[str, Any]] = []
    if not audit_root.is_dir():
        return rows
    for path in audit_root.glob("config-*/audit_record.json"):
        try:
            payload = _read_json(path)
        except ConfigGovernanceError:
            continue
        if isinstance(payload, dict):
            rollback_path = path.parent / "rollback_record.json"
            if rollback_path.is_file():
                rollback = _read_json(rollback_path)
                payload = {**payload, "rollback": rollback}
            rows.append(payload)
    rows.sort(key=lambda row: (str(row.get("created_at") or ""), str(row.get("change_id") or "")), reverse=True)
    return rows[: max(0, int(limit))]


def rollback_latest_config_change(
    *,
    repo_root: Path,
    backup_root: Path,
    audit_root: Path,
    pointer_path: Path,
    actor: Mapping[str, Any],
    confirm_change_id: str,
    reason: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    pointer_path = Path(pointer_path)
    if not pointer_path.is_file():
        raise ConfigGovernanceError("There is no applied configuration change to roll back.")
    pointer = _read_json(pointer_path)
    if not isinstance(pointer, dict):
        raise ConfigGovernanceError("Configuration pointer is invalid.")
    change_id = str(pointer.get("change_id") or "")
    if str(confirm_change_id or "") != change_id:
        raise ConfigGovernanceError("Rollback confirmation must exactly match the latest change id.")
    if pointer.get("status") == "rolled_back":
        return {
            "success": True,
            "already_rolled_back": True,
            "change_id": change_id,
            "status": "rolled_back",
        }
    if pointer.get("status") != "applied":
        raise ConfigGovernanceError(f"Latest configuration state cannot be rolled back: {pointer.get('status')!r}.")
    reason = str(reason or "").strip()
    if len(reason) < 8:
        raise ConfigGovernanceError("A rollback reason of at least 8 characters is required.")

    backup_dir = repo_root / Path(*str(pointer.get("backup_dir") or "").split("/"))
    audit_dir = repo_root / Path(*str(pointer.get("audit_dir") or "").split("/"))
    manifest_path = backup_dir / "backup_manifest.json"
    if not manifest_path.is_file():
        raise ConfigGovernanceError("Rollback backup manifest is missing.")
    manifest = _read_json(manifest_path)
    target_files = [str(item) for item in manifest.get("target_files") or []]
    before_hashes = dict(manifest.get("before_sha256") or {})
    after_hashes = dict(manifest.get("after_sha256") or {})
    if target_files != list(pointer.get("target_files") or []):
        raise ConfigGovernanceError("Rollback target list does not match the current pointer.")

    current_bytes: dict[str, bytes] = {}
    before_bytes: dict[str, bytes] = {}
    paths: dict[str, Path] = {}
    for relative in target_files:
        path = repo_root / Path(*relative.split("/"))
        backup_path = backup_dir / "before" / Path(*relative.split("/"))
        if not path.is_file() or not backup_path.is_file():
            raise ConfigGovernanceError(f"Rollback file is missing: {relative}")
        if sha256_file(path) != str(after_hashes.get(relative) or ""):
            raise ConfigGovernanceError(f"Current configuration drift blocks rollback: {relative}")
        if sha256_file(backup_path) != str(before_hashes.get(relative) or ""):
            raise ConfigGovernanceError(f"Rollback backup hash mismatch: {relative}")
        paths[relative] = path
        current_bytes[relative] = path.read_bytes()
        before_bytes[relative] = backup_path.read_bytes()

    restored: list[str] = []
    try:
        for relative in target_files:
            _atomic_write_bytes(paths[relative], before_bytes[relative])
            restored.append(relative)
        for relative in target_files:
            if sha256_file(paths[relative]) != before_hashes[relative]:
                raise ConfigGovernanceError(f"Rollback verification failed: {relative}")

        now = now or datetime.now(timezone.utc)
        rollback_record = {
            "schema_version": CONFIG_GOVERNANCE_SCHEMA_VERSION,
            "change_id": change_id,
            "status": "rolled_back",
            "rolled_back_at": now.isoformat(),
            "actor": _public_actor(actor),
            "reason": reason,
            "restored_sha256": before_hashes,
            "recognition_executed": False,
            "attendance_changed": False,
            "embeddings_changed": False,
            "roster_changed": False,
        }
        _atomic_write_json(audit_dir / "rollback_record.json", rollback_record)
        updated_pointer = {
            **pointer,
            "status": "rolled_back",
            "rolled_back_at": rollback_record["rolled_back_at"],
            "rolled_back_by": rollback_record["actor"],
            "rollback_reason": reason,
        }
        try:
            _atomic_write_json(pointer_path, updated_pointer)
        except Exception:
            for relative in target_files:
                _atomic_write_bytes(paths[relative], current_bytes[relative])
            (audit_dir / "rollback_record.json").unlink(missing_ok=True)
            raise
        return rollback_record
    except Exception:
        for relative in restored:
            _atomic_write_bytes(paths[relative], current_bytes[relative])
        raise
