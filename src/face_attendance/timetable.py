from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


DAY_ALIASES = {
    "mon": "Monday",
    "monday": "Monday",
    "tue": "Tuesday",
    "tues": "Tuesday",
    "tuesday": "Tuesday",
    "wed": "Wednesday",
    "wednesday": "Wednesday",
    "thu": "Thursday",
    "thur": "Thursday",
    "thurs": "Thursday",
    "thursday": "Thursday",
    "fri": "Friday",
    "friday": "Friday",
    "sat": "Saturday",
    "saturday": "Saturday",
    "sun": "Sunday",
    "sunday": "Sunday",
}

REQUIRED_COLUMNS = [
    "slot_id",
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
]


@dataclass(frozen=True)
class TimetableSlot:
    slot_id: str
    academic_year: str
    year: str
    semester: str
    wef: str
    day: str
    period: str
    start_time: str
    end_time: str
    course_code: str
    course_abbr: str
    course_name: str
    instructor: str
    session_type: str
    room: str
    section: str
    camera_ids: str
    attendance_required: str
    notes: str = ""

    @classmethod
    def from_series(cls, row: pd.Series) -> "TimetableSlot":
        def get(name: str, default: str = "") -> str:
            value = row.get(name, default)
            if pd.isna(value):
                return default
            return str(value).strip()

        return cls(
            slot_id=get("slot_id"),
            academic_year=get("academic_year"),
            year=get("year"),
            semester=get("semester"),
            wef=get("wef"),
            day=normalize_day(get("day")),
            period=get("period"),
            start_time=normalize_time_text(get("start_time")),
            end_time=normalize_time_text(get("end_time")),
            course_code=get("course_code"),
            course_abbr=get("course_abbr"),
            course_name=get("course_name"),
            instructor=get("instructor"),
            session_type=get("session_type"),
            room=get("room"),
            section=get("section"),
            camera_ids=get("camera_ids"),
            attendance_required=get("attendance_required", "Yes"),
            notes=get("notes"),
        )

    def to_dict(self) -> dict:
        return {
            "Slot_ID": self.slot_id,
            "Academic_Year": self.academic_year,
            "Year": self.year,
            "Semester": self.semester,
            "WEF": self.wef,
            "Day": self.day,
            "Period": self.period,
            "Start_Time": self.start_time,
            "End_Time": self.end_time,
            "Course_Code": self.course_code,
            "Course_Abbr": self.course_abbr,
            "Course_Name": self.course_name,
            "Instructor": self.instructor,
            "Session_Type": self.session_type,
            "Room": self.room,
            "Section": self.section,
            "Camera_IDs": self.camera_ids,
            "Attendance_Required": self.attendance_required,
            "Notes": self.notes,
        }

    @property
    def safe_name(self) -> str:
        parts = [self.slot_id, self.course_abbr, self.day, f"P{self.period}"]
        return "_".join(str(part).strip().replace("/", "-").replace(" ", "_") for part in parts if str(part).strip())


def normalize_day(value: str) -> str:
    key = str(value).strip().lower()
    if not key:
        return ""
    return DAY_ALIASES.get(key, value.strip().capitalize())


def parse_time(value: str) -> time:
    text = str(value).strip().upper().replace(".", "")
    for fmt in ("%H:%M", "%I:%M %p", "%I:%M%p", "%H%M"):
        try:
            return datetime.strptime(text, fmt).time()
        except ValueError:
            pass
    raise ValueError(f"Invalid time value: {value!r}. Use HH:MM, for example 09:00.")


def normalize_time_text(value: str) -> str:
    if str(value).strip() == "":
        return ""
    return parse_time(value).strftime("%H:%M")


def load_timetable(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Timetable file not found: {path}")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    df.columns = [str(col).strip() for col in df.columns]
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Timetable is missing required columns: {missing}")

    df = df.copy()
    df["day"] = df["day"].map(normalize_day)
    df["start_time"] = df["start_time"].map(normalize_time_text)
    df["end_time"] = df["end_time"].map(normalize_time_text)
    df["slot_id"] = df["slot_id"].astype(str).str.strip()
    df["period"] = df["period"].astype(str).str.strip()
    return df


def list_slots(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "slot_id",
        "day",
        "period",
        "start_time",
        "end_time",
        "course_abbr",
        "course_name",
        "instructor",
        "room",
    ]
    existing = [col for col in columns if col in df.columns]
    return df[existing].copy()


def select_slot(
    df: pd.DataFrame,
    slot_id: Optional[str] = None,
    day: Optional[str] = None,
    period: Optional[str | int] = None,
    class_time: Optional[str] = None,
) -> TimetableSlot:
    if slot_id:
        matches = df[df["slot_id"].str.lower() == str(slot_id).strip().lower()]
        if matches.empty:
            available = ", ".join(df["slot_id"].astype(str).head(10).tolist())
            raise ValueError(f"Slot ID not found: {slot_id}. Example available slots: {available}")
        return TimetableSlot.from_series(matches.iloc[0])

    if not day:
        raise ValueError("Provide either --slot-id OR --day with --period/--class-time.")

    normalized_day = normalize_day(day)
    day_df = df[df["day"].str.lower() == normalized_day.lower()]
    if day_df.empty:
        raise ValueError(f"No timetable slots found for day: {day}")

    if period is not None:
        matches = day_df[day_df["period"].astype(str).str.strip() == str(period).strip()]
        if matches.empty:
            raise ValueError(f"No slot found for {normalized_day} period {period}.")
        return TimetableSlot.from_series(matches.iloc[0])

    if class_time:
        query_time = parse_time(class_time)
        matches = []
        for _, row in day_df.iterrows():
            start = parse_time(row["start_time"])
            end = parse_time(row["end_time"])
            if start <= query_time < end:
                matches.append(row)
        if not matches:
            raise ValueError(f"No slot found for {normalized_day} at {class_time}.")
        return TimetableSlot.from_series(matches[0])

    raise ValueError("When using --day, provide either --period or --class-time.")


def split_camera_ids(camera_ids: str) -> list[str]:
    return [part.strip() for part in str(camera_ids).split(",") if part.strip()]
