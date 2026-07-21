from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


LOW_QUALITY_STATUS = "Needs Review - Low Recognition Quality"
INSUFFICIENT_FOOTAGE_STATUS = "Needs Review - Insufficient Footage"
VALID_STATUS = "Valid"
LOW_QUALITY_REVIEW_NOTE = (
    "Attendance requires review because recognition quality was too low. "
    "The system detected faces but could not confidently identify enough students."
)


@dataclass(frozen=True)
class RunQuality:
    status: str
    reason: str
    recognition_rate: float
    usable_checkpoints: int
    poor_checkpoints: list[str]
    detected_faces: int
    accepted_recognitions: int
    unique_students_recognized: int
    requires_manual_review: bool
    attendance_finalized: bool
    dominant_identity_roll: str = ""
    dominant_identity_share: float = 0.0
    review_queue_note: str = ""


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except Exception:
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "manual_review", "needs_review"}


def _split_poor_checkpoints(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    return [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]


def evaluate_run_quality(
    *,
    total_students: int,
    total_checkpoints: int,
    detected_faces: int,
    accepted_recognitions: int,
    unique_students_recognized: int,
    poor_checkpoints: list[str] | tuple[str, ...] | set[str],
    present_count: int,
    review_count: int,
    absent_count: int,
    dominant_identity_roll: str = "",
    dominant_identity_count: int = 0,
) -> RunQuality:
    """Classify run-level recognition safety without changing student-level evidence."""
    total_students = max(0, _as_int(total_students))
    total_checkpoints = max(0, _as_int(total_checkpoints))
    detected_faces = max(0, _as_int(detected_faces))
    accepted_recognitions = max(0, _as_int(accepted_recognitions))
    unique_students_recognized = max(0, _as_int(unique_students_recognized))
    present_count = max(0, _as_int(present_count))
    review_count = max(0, _as_int(review_count))
    absent_count = max(0, _as_int(absent_count))
    poor = sorted(set(_split_poor_checkpoints(poor_checkpoints)))

    recognition_rate = round((accepted_recognitions / detected_faces) * 100, 1) if detected_faces else 0.0
    usable_checkpoints = max(0, total_checkpoints - len(poor))
    poor_limit = max(2, math.ceil(total_checkpoints * 0.60)) if total_checkpoints else 0
    student_coverage_rate = round((unique_students_recognized / total_students) * 100, 1) if total_students else 0.0
    dominant_share = round((max(0, _as_int(dominant_identity_count)) / accepted_recognitions) * 100, 1) if accepted_recognitions else 0.0

    if total_checkpoints == 0 or (detected_faces < 10 and total_students > 0):
        reason = "Insufficient usable face detections for an attendance decision."
        return RunQuality(
            status=INSUFFICIENT_FOOTAGE_STATUS,
            reason=reason,
            recognition_rate=recognition_rate,
            usable_checkpoints=usable_checkpoints,
            poor_checkpoints=poor,
            detected_faces=detected_faces,
            accepted_recognitions=accepted_recognitions,
            unique_students_recognized=unique_students_recognized,
            requires_manual_review=True,
            attendance_finalized=False,
            dominant_identity_roll=str(dominant_identity_roll or ""),
            dominant_identity_share=dominant_share,
            review_queue_note=LOW_QUALITY_REVIEW_NOTE,
        )

    reasons: list[str] = []
    if detected_faces >= 50 and recognition_rate < 10.0:
        reasons.append(f"Recognition rate {recognition_rate}% is below the 10% safety floor.")
    if poor_limit and len(poor) >= poor_limit and detected_faces >= 20:
        reasons.append(f"{len(poor)} of {total_checkpoints} checkpoints were Poor.")
    if total_students >= 10 and detected_faces >= 50 and student_coverage_rate < 5.0:
        reasons.append(f"Student coverage {student_coverage_rate}% is below the 5% safety floor.")
    if total_students >= 10 and present_count == 0 and review_count == 0 and absent_count >= max(10, math.ceil(total_students * 0.80)) and detected_faces >= 50:
        reasons.append("The run would become mass absence despite substantial face detections.")
    if accepted_recognitions >= 10 and dominant_share >= 75.0 and unique_students_recognized <= 2:
        reasons.append(f"Accepted matches are dominated by {dominant_identity_roll or 'one identity'} ({dominant_share}%).")

    if reasons:
        return RunQuality(
            status=LOW_QUALITY_STATUS,
            reason=" ".join(reasons),
            recognition_rate=recognition_rate,
            usable_checkpoints=usable_checkpoints,
            poor_checkpoints=poor,
            detected_faces=detected_faces,
            accepted_recognitions=accepted_recognitions,
            unique_students_recognized=unique_students_recognized,
            requires_manual_review=True,
            attendance_finalized=False,
            dominant_identity_roll=str(dominant_identity_roll or ""),
            dominant_identity_share=dominant_share,
            review_queue_note=LOW_QUALITY_REVIEW_NOTE,
        )

    return RunQuality(
        status=VALID_STATUS,
        reason="Recognition quality is within run-level safety limits.",
        recognition_rate=recognition_rate,
        usable_checkpoints=usable_checkpoints,
        poor_checkpoints=poor,
        detected_faces=detected_faces,
        accepted_recognitions=accepted_recognitions,
        unique_students_recognized=unique_students_recognized,
        requires_manual_review=False,
        attendance_finalized=True,
        dominant_identity_roll=str(dominant_identity_roll or ""),
        dominant_identity_share=dominant_share,
        review_queue_note="",
    )


def quality_to_csv_fields(quality: RunQuality) -> dict[str, Any]:
    return {
        "Run_Quality_Status": quality.status,
        "Run_Quality_Reason": quality.reason,
        "Recognition_Rate": quality.recognition_rate,
        "Usable_Checkpoints": quality.usable_checkpoints,
        "Poor_Checkpoints": ", ".join(quality.poor_checkpoints),
        "Detected_Faces": quality.detected_faces,
        "Accepted_Recognitions": quality.accepted_recognitions,
        "Unique_Students_Recognized": quality.unique_students_recognized,
        "Requires_Manual_Review": "Yes" if quality.requires_manual_review else "No",
        "Attendance_Finalized": "Yes" if quality.attendance_finalized else "No",
        "Dominant_Identity_Roll": quality.dominant_identity_roll,
        "Dominant_Identity_Share": quality.dominant_identity_share,
        "Review_Queue_Note": quality.review_queue_note,
    }


def quality_to_status_fields(quality: RunQuality) -> dict[str, Any]:
    return {
        "run_quality_status": quality.status,
        "run_quality_reason": quality.reason,
        "recognition_rate": quality.recognition_rate,
        "usable_checkpoints": quality.usable_checkpoints,
        "poor_checkpoints": quality.poor_checkpoints,
        "detected_faces": quality.detected_faces,
        "accepted_recognitions": quality.accepted_recognitions,
        "unique_students_recognized": quality.unique_students_recognized,
        "requires_manual_review": quality.requires_manual_review,
        "attendance_finalized": quality.attendance_finalized,
        "dominant_identity_roll": quality.dominant_identity_roll,
        "dominant_identity_share": quality.dominant_identity_share,
        "review_queue_note": quality.review_queue_note,
    }


def quality_from_slot_summary(row: dict[str, Any] | None) -> RunQuality:
    row = dict(row or {})
    status = str(row.get("Run_Quality_Status") or "").strip()
    if status:
        poor = _split_poor_checkpoints(row.get("Poor_Checkpoints"))
        requires = _as_bool(row.get("Requires_Manual_Review"))
        finalized = _as_bool(row.get("Attendance_Finalized"))
        return RunQuality(
            status=status,
            reason=str(row.get("Run_Quality_Reason") or ""),
            recognition_rate=_as_float(row.get("Recognition_Rate")),
            usable_checkpoints=_as_int(row.get("Usable_Checkpoints")),
            poor_checkpoints=poor,
            detected_faces=_as_int(row.get("Detected_Faces") or row.get("Total_Face_Detections")),
            accepted_recognitions=_as_int(row.get("Accepted_Recognitions")),
            unique_students_recognized=_as_int(row.get("Unique_Students_Recognized") or row.get("Students_With_At_Least_One_Detection")),
            requires_manual_review=requires,
            attendance_finalized=finalized,
            dominant_identity_roll=str(row.get("Dominant_Identity_Roll") or ""),
            dominant_identity_share=_as_float(row.get("Dominant_Identity_Share")),
            review_queue_note=str(row.get("Review_Queue_Note") or (LOW_QUALITY_REVIEW_NOTE if requires else "")),
        )

    return evaluate_run_quality(
        total_students=_as_int(row.get("Total_Students")),
        total_checkpoints=_as_int(row.get("Total_Checkpoints")),
        detected_faces=_as_int(row.get("Total_Face_Detections")),
        accepted_recognitions=_as_int(row.get("Accepted_Recognitions")),
        unique_students_recognized=_as_int(row.get("Students_With_At_Least_One_Detection")),
        poor_checkpoints=_split_poor_checkpoints(row.get("Poor_Checkpoints")),
        present_count=_as_int(row.get("Students_Present")),
        review_count=_as_int(row.get("Students_Needs_Review")),
        absent_count=_as_int(row.get("Students_Absent")),
    )


def review_queue_reason(final_status: Any, flags: Any, run_requires_manual_review: bool) -> str:
    flag_text = str(flags or "").strip()
    status_text = str(final_status or "").strip()
    if status_text == "Needs Review":
        return "Raw outcome is Needs Review."
    if flag_text:
        return f"Flagged for review despite raw outcome {status_text or 'Unknown'}: {flag_text}."
    if run_requires_manual_review:
        return "Run-level recognition quality requires manual review before official finalization."
    return ""
