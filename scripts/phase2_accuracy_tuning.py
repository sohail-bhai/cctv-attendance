"""
Phase 2 Accuracy Tuning for CCTV-Based Smart Attendance System

This script evaluates MVP 3 checkpoint attendance output against a ground-truth
actual-present list and can simulate different threshold settings using the
MVP 3 detection log.

Usage examples:

1) Compare generated attendance/evidence with actual present list:
python scripts/phase2_accuracy_tuning.py ^
  --attendance attendance_output\attendance_MON_P1_xxx.csv ^
  --student-evidence attendance_output\student_checkpoint_evidence_MON_P1_xxx.csv ^
  --actual-present actual_present\MON_P1_actual_present.txt

2) Run threshold sweep from detection log:
python scripts/phase2_accuracy_tuning.py ^
  --detection-log attendance_output\detection_log_MON_P1_xxx.csv ^
  --attendance attendance_output\attendance_MON_P1_xxx.csv ^
  --actual-present actual_present\MON_P1_actual_present.txt ^
  --sweep
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd


PRESENT_STATUSES = {"present", "present strong", "present - strong", "present strong"}
REVIEW_STATUSES = {"needs review", "review", "manual review"}
ABSENT_STATUSES = {"absent"}

DEFAULT_THRESHOLDS = [0.44, 0.46, 0.48, 0.50, 0.52, 0.54]
DEFAULT_MARGINS = [0.06, 0.08, 0.10, 0.12]
DEFAULT_CHECKPOINT_MIN_DETECTIONS = [1, 2, 3]


@dataclass
class Metrics:
    threshold: Optional[float]
    margin: Optional[float]
    checkpoint_min_detections: Optional[int]
    total_students: int
    actual_present: int
    actual_absent: int
    predicted_present: int
    predicted_review: int
    predicted_absent: int
    true_present: int
    false_present: int
    false_absent: int
    actual_present_in_review: int
    actual_absent_in_review: int
    true_absent: int
    precision: float
    recall: float
    f1: float
    auto_accuracy: float
    safety_score: float


def clean_roll(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", ""}:
        return ""
    return text


def normalize_status(value: object) -> str:
    return str(value or "").strip().lower().replace("_", " ")


def is_present_status(status: object) -> bool:
    s = normalize_status(status)
    return s.startswith("present")


def is_review_status(status: object) -> bool:
    s = normalize_status(status)
    return "review" in s


def is_absent_status(status: object) -> bool:
    return normalize_status(status) == "absent"


def read_actual_present(path: str | Path) -> Set[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Actual-present file not found: {p}")
    actual: Set[str] = set()
    with p.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # allow comma-separated accidental input too
            for part in line.split(","):
                roll = clean_roll(part)
                if roll:
                    actual.add(roll)
    return actual


def find_col(df: pd.DataFrame, options: Sequence[str], required: bool = True) -> Optional[str]:
    normalized = {c.lower().strip().replace(" ", "_"): c for c in df.columns}
    for opt in options:
        key = opt.lower().strip().replace(" ", "_")
        if key in normalized:
            return normalized[key]
    if required:
        raise ValueError(f"Missing required column. Tried: {', '.join(options)}. Available: {list(df.columns)}")
    return None


def read_status_table(attendance_path: Optional[str], evidence_path: Optional[str]) -> pd.DataFrame:
    # Prefer the final attendance file when both are supplied.
    # student_checkpoint_evidence is usually one row per student per checkpoint,
    # so it does not always contain the final attendance status.
    if attendance_path:
        df = pd.read_csv(attendance_path)
    elif evidence_path:
        df = pd.read_csv(evidence_path)
    else:
        raise ValueError("Provide --attendance or --student-evidence for direct comparison.")

    roll_col = find_col(df, ["Roll_Number", "Roll Number", "roll_number", "roll"])
    status_col = find_col(df, ["Final_Status", "Final Status", "Status", "Present", "Attendance_Status"], required=False)

    if status_col is None:
        # Backward compatibility: Present column might contain Yes/No.
        present_col = find_col(df, ["Present"], required=False)
        if present_col:
            status_col = present_col
        else:
            raise ValueError("Could not find status column in attendance/evidence file.")

    out = pd.DataFrame()
    out["Roll_Number"] = df[roll_col].map(clean_roll)

    def map_status(raw: object) -> str:
        s = normalize_status(raw)
        if s in {"yes", "y", "true", "1"}:
            return "Present"
        if s in {"no", "n", "false", "0"}:
            return "Absent"
        if s.startswith("present"):
            return str(raw).strip()
        if "review" in s:
            return "Needs Review"
        if s == "absent":
            return "Absent"
        return str(raw).strip() if str(raw).strip() else "Absent"

    out["Final_Status"] = df[status_col].map(map_status)

    # Optional useful columns if present.
    optional_cols = {
        "Recognized_Checkpoints": ["Recognized_Checkpoints", "Recognized Checkpoints"],
        "Total_Checkpoints": ["Total_Checkpoints", "Total Checkpoints"],
        "Total_Accepted_Detections": ["Total_Accepted_Detections", "Total Accepted Detections", "Detection_Count"],
        "Average_Score": ["Average_Score", "Average Score", "Avg_Score"],
        "Best_Score": ["Best_Score", "Best Score"],
        "Flags": ["Flags", "Flag"],
    }
    for out_col, opts in optional_cols.items():
        col = find_col(df, opts, required=False)
        if col:
            out[out_col] = df[col]

    out = out[out["Roll_Number"] != ""].drop_duplicates(subset=["Roll_Number"], keep="first")
    return out


def compute_metrics(status_df: pd.DataFrame, actual_present: Set[str], threshold=None, margin=None, checkpoint_min=None) -> Metrics:
    students = set(status_df["Roll_Number"].astype(str)) | actual_present
    status_map = dict(zip(status_df["Roll_Number"], status_df["Final_Status"]))

    pred_present = {r for r in students if is_present_status(status_map.get(r, "Absent"))}
    pred_review = {r for r in students if is_review_status(status_map.get(r, "Absent"))}
    pred_absent = students - pred_present - pred_review

    actual_absent = students - actual_present

    tp = len(pred_present & actual_present)
    fp = len(pred_present & actual_absent)
    fn = len(pred_absent & actual_present)
    review_present = len(pred_review & actual_present)
    review_absent = len(pred_review & actual_absent)
    tn = len(pred_absent & actual_absent)

    precision = tp / len(pred_present) if pred_present else 0.0
    # Recall here means automatic present recall, not counting review as success.
    recall = tp / len(actual_present) if actual_present else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    auto_accuracy = (tp + tn) / len(students) if students else 0.0

    # Lower is better internally; convert to 0-100 score.
    # False present is most dangerous, false absent is next, review is acceptable but not ideal.
    penalty = (fp * 8.0) + (fn * 3.0) + (review_present * 1.2) + (review_absent * 0.8)
    safety_score = max(0.0, 100.0 - (penalty * 100.0 / max(len(students), 1)))

    return Metrics(
        threshold=threshold,
        margin=margin,
        checkpoint_min_detections=checkpoint_min,
        total_students=len(students),
        actual_present=len(actual_present),
        actual_absent=len(actual_absent),
        predicted_present=len(pred_present),
        predicted_review=len(pred_review),
        predicted_absent=len(pred_absent),
        true_present=tp,
        false_present=fp,
        false_absent=fn,
        actual_present_in_review=review_present,
        actual_absent_in_review=review_absent,
        true_absent=tn,
        precision=precision,
        recall=recall,
        f1=f1,
        auto_accuracy=auto_accuracy,
        safety_score=safety_score,
    )


def list_groups(status_df: pd.DataFrame, actual_present: Set[str]) -> Dict[str, List[str]]:
    students = set(status_df["Roll_Number"].astype(str)) | actual_present
    status_map = dict(zip(status_df["Roll_Number"], status_df["Final_Status"]))
    pred_present = {r for r in students if is_present_status(status_map.get(r, "Absent"))}
    pred_review = {r for r in students if is_review_status(status_map.get(r, "Absent"))}
    pred_absent = students - pred_present - pred_review
    actual_absent = students - actual_present
    return {
        "true_present": sorted(pred_present & actual_present),
        "false_present": sorted(pred_present & actual_absent),
        "false_absent": sorted(pred_absent & actual_present),
        "actual_present_in_review": sorted(pred_review & actual_present),
        "actual_absent_in_review": sorted(pred_review & actual_absent),
        "true_absent": sorted(pred_absent & actual_absent),
    }


def get_checkpoint_col(df: pd.DataFrame) -> str:
    col = find_col(df, ["Checkpoint", "Checkpoint_ID", "Checkpoint_Label", "CP", "Checkpoint_Name"], required=False)
    if col:
        return col
    # If older log has no checkpoint column, treat entire video as CP1.
    df["Checkpoint"] = "CP1"
    return "Checkpoint"


def simulate_from_detection_log(
    detection_log_path: str | Path,
    enrolled_students: Iterable[str],
    match_threshold: float,
    margin_threshold: float,
    checkpoint_min_detections: int,
) -> pd.DataFrame:
    log = pd.read_csv(detection_log_path)
    if log.empty:
        raise ValueError("Detection log is empty.")

    best_roll_col = find_col(log, ["Best_Roll", "Predicted_Roll", "Roll_Number", "Roll Number"])
    best_score_col = find_col(log, ["Best_Score", "Score", "Similarity", "Confidence"])
    margin_col = find_col(log, ["Margin", "Score_Margin"], required=False)
    checkpoint_col = get_checkpoint_col(log)
    camera_col = find_col(log, ["Camera", "Camera_ID", "Camera_Id", "Video"], required=False)

    work = log.copy()
    work["_roll"] = work[best_roll_col].map(clean_roll)
    work["_score"] = pd.to_numeric(work[best_score_col], errors="coerce").fillna(-999)
    if margin_col:
        work["_margin"] = pd.to_numeric(work[margin_col], errors="coerce").fillna(-999)
    else:
        work["_margin"] = 999.0
    work["_checkpoint"] = work[checkpoint_col].astype(str).fillna("CP1")
    if camera_col:
        work["_camera"] = work[camera_col].astype(str).fillna("unknown")
    else:
        work["_camera"] = "unknown"

    valid = (
        (work["_roll"] != "")
        & (work["_score"] >= match_threshold)
        & (work["_margin"] >= margin_threshold)
    )
    accepted = work[valid].copy()

    # Roll x checkpoint evidence.
    if accepted.empty:
        checkpoint_counts = pd.DataFrame(columns=["_roll", "_checkpoint", "count", "best", "avg"])
    else:
        checkpoint_counts = (
            accepted.groupby(["_roll", "_checkpoint"])
            .agg(count=("_score", "size"), best=("_score", "max"), avg=("_score", "mean"))
            .reset_index()
        )

    recognized_checkpoint_rows = checkpoint_counts[checkpoint_counts["count"] >= checkpoint_min_detections]
    recognized_by_roll: Dict[str, Set[str]] = {}
    for _, row in recognized_checkpoint_rows.iterrows():
        recognized_by_roll.setdefault(row["_roll"], set()).add(row["_checkpoint"])

    total_by_roll = accepted.groupby("_roll").size().to_dict() if not accepted.empty else {}
    best_by_roll = accepted.groupby("_roll")["_score"].max().to_dict() if not accepted.empty else {}
    avg_by_roll = accepted.groupby("_roll")["_score"].mean().to_dict() if not accepted.empty else {}
    camera_by_roll = accepted.groupby("_roll")["_camera"].apply(lambda s: sorted(set(map(str, s)))).to_dict() if not accepted.empty else {}

    # Determine total checkpoints from log.
    all_checkpoints = sorted(set(work["_checkpoint"].dropna().astype(str)))
    total_checkpoints = max(len(all_checkpoints), 1)

    rows = []
    for roll in sorted(set(map(clean_roll, enrolled_students)) | set(work["_roll"]) - {""}):
        cps = recognized_by_roll.get(roll, set())
        cp_count = len(cps)
        if cp_count >= 4:
            status = "Present Strong"
        elif cp_count == 3:
            status = "Present"
        elif cp_count == 2:
            status = "Needs Review"
        else:
            status = "Absent"

        cp_map = {cp: ("Yes" if cp in cps else "No") for cp in all_checkpoints}
        flags = []
        # Simple pattern flags for 5 checkpoints only; safe for other counts too.
        ordered = all_checkpoints
        if len(ordered) >= 5:
            first_two = [cp_map.get(ordered[0]), cp_map.get(ordered[1])]
            last_two = [cp_map.get(ordered[-2]), cp_map.get(ordered[-1])]
            if first_two == ["No", "No"] and cp_count >= 3:
                flags.append("Late Entry")
            if last_two == ["No", "No"] and cp_count >= 3:
                flags.append("Left Early Flag")
        if total_by_roll.get(roll, 0) > 0 and cp_count <= 1:
            flags.append("Weak Evidence Only")
        if avg_by_roll.get(roll, float("nan")) == avg_by_roll.get(roll, float("nan")) and avg_by_roll.get(roll, 0) < (match_threshold + 0.03):
            flags.append("Low Confidence")
        if len(camera_by_roll.get(roll, [])) == 1 and total_by_roll.get(roll, 0) > 0:
            flags.append("Single Camera Evidence")

        row = {
            "Roll_Number": roll,
            "Final_Status": status,
            "Recognized_Checkpoints": cp_count,
            "Total_Checkpoints": total_checkpoints,
            "Total_Accepted_Detections": int(total_by_roll.get(roll, 0)),
            "Best_Score": round(float(best_by_roll.get(roll, 0.0)), 4) if roll in best_by_roll else "",
            "Average_Score": round(float(avg_by_roll.get(roll, 0.0)), 4) if roll in avg_by_roll else "",
            "Cameras_Seen": ",".join(camera_by_roll.get(roll, [])),
            "Flags": "; ".join(flags),
        }
        for cp in all_checkpoints:
            row[str(cp)] = cp_map[cp]
        rows.append(row)

    return pd.DataFrame(rows)


def metrics_to_dict(m: Metrics) -> Dict[str, object]:
    return {
        "match_threshold": m.threshold,
        "margin_threshold": m.margin,
        "checkpoint_min_detections": m.checkpoint_min_detections,
        "total_students": m.total_students,
        "actual_present": m.actual_present,
        "actual_absent": m.actual_absent,
        "predicted_present": m.predicted_present,
        "predicted_review": m.predicted_review,
        "predicted_absent": m.predicted_absent,
        "true_present": m.true_present,
        "false_present": m.false_present,
        "false_absent": m.false_absent,
        "actual_present_in_review": m.actual_present_in_review,
        "actual_absent_in_review": m.actual_absent_in_review,
        "true_absent": m.true_absent,
        "precision": round(m.precision, 4),
        "recall": round(m.recall, 4),
        "f1": round(m.f1, 4),
        "auto_accuracy": round(m.auto_accuracy, 4),
        "safety_score": round(m.safety_score, 2),
    }


def write_text_report(
    out_path: Path,
    metrics: Metrics,
    groups: Dict[str, List[str]],
    best: Optional[pd.Series] = None,
) -> None:
    lines: List[str] = []
    lines.append("PHASE 2 ACCURACY TUNING REPORT")
    lines.append("=" * 40)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("CURRENT RESULT")
    lines.append("-" * 20)
    for key, value in metrics_to_dict(metrics).items():
        lines.append(f"{key}: {value}")
    lines.append("")

    labels = {
        "true_present": "Correctly marked present",
        "false_present": "FALSE PRESENT: absent students wrongly marked present",
        "false_absent": "FALSE ABSENT: actual present students marked absent",
        "actual_present_in_review": "Actual present students sent to Needs Review",
        "actual_absent_in_review": "Actual absent students sent to Needs Review",
        "true_absent": "Correctly kept absent",
    }
    for key, title in labels.items():
        values = groups.get(key, [])
        lines.append(title)
        lines.append("-" * len(title))
        if values:
            for roll in values:
                lines.append(f"- {roll}")
        else:
            lines.append("None")
        lines.append("")

    if best is not None:
        lines.append("BEST THRESHOLD CANDIDATE FROM SWEEP")
        lines.append("-" * 35)
        for key in [
            "match_threshold",
            "margin_threshold",
            "checkpoint_min_detections",
            "predicted_present",
            "predicted_review",
            "predicted_absent",
            "true_present",
            "false_present",
            "false_absent",
            "actual_present_in_review",
            "actual_absent_in_review",
            "precision",
            "recall",
            "f1",
            "auto_accuracy",
            "safety_score",
        ]:
            if key in best:
                lines.append(f"{key}: {best[key]}")
        lines.append("")

    lines.append("INTERPRETATION")
    lines.append("-" * 14)
    if metrics.false_present > 0:
        lines.append("False present exists, so the system is still too loose. Increase match threshold/margin or checkpoint-min-detections.")
    elif metrics.false_absent > 0 and metrics.actual_present_in_review == 0:
        lines.append("No false present, but actual present students are being missed. Try slightly relaxing threshold or checkpoint-min-detections.")
    elif metrics.actual_present_in_review > 0:
        lines.append("Some actual present students are going to Needs Review. This is safer than false present; tune only if manual workload is too high.")
    else:
        lines.append("This is a strong setting for the current test set. Validate on more slots before freezing thresholds.")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_float_list(value: Optional[str], default: Sequence[float]) -> List[float]:
    if not value:
        return list(default)
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_int_list(value: Optional[str], default: Sequence[int]) -> List[int]:
    if not value:
        return list(default)
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 accuracy evaluation and threshold tuning for MVP 3 attendance.")
    parser.add_argument("--attendance", help="Final attendance CSV from MVP 3.")
    parser.add_argument("--student-evidence", help="student_checkpoint_evidence CSV from MVP 3.")
    parser.add_argument("--detection-log", help="detection_log CSV from MVP 3. Required for --sweep.")
    parser.add_argument("--actual-present", required=True, help="Text file containing actual present roll numbers, one per line.")
    parser.add_argument("--output-dir", default="attendance_output/phase2_accuracy", help="Folder to write reports.")
    parser.add_argument("--sweep", action="store_true", help="Run threshold sweep using the detection log.")
    parser.add_argument("--thresholds", help="Comma-separated match thresholds, e.g. 0.46,0.48,0.50")
    parser.add_argument("--margins", help="Comma-separated margin thresholds, e.g. 0.06,0.08,0.10")
    parser.add_argument("--checkpoint-mins", help="Comma-separated checkpoint min detections, e.g. 1,2,3")
    args = parser.parse_args()

    actual_present = read_actual_present(args.actual_present)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    current_df = read_status_table(args.attendance, args.student_evidence)
    current_metrics = compute_metrics(current_df, actual_present)
    current_groups = list_groups(current_df, actual_present)

    best_row = None
    if args.sweep:
        if not args.detection_log:
            raise ValueError("--detection-log is required when using --sweep.")
        thresholds = parse_float_list(args.thresholds, DEFAULT_THRESHOLDS)
        margins = parse_float_list(args.margins, DEFAULT_MARGINS)
        checkpoint_mins = parse_int_list(args.checkpoint_mins, DEFAULT_CHECKPOINT_MIN_DETECTIONS)
        enrolled = set(current_df["Roll_Number"].astype(str)) | actual_present

        sweep_rows = []
        simulated_dir = output_dir / f"simulated_student_evidence_{stamp}"
        simulated_dir.mkdir(parents=True, exist_ok=True)

        for th in thresholds:
            for mg in margins:
                for cp_min in checkpoint_mins:
                    sim_df = simulate_from_detection_log(args.detection_log, enrolled, th, mg, cp_min)
                    m = compute_metrics(sim_df, actual_present, threshold=th, margin=mg, checkpoint_min=cp_min)
                    row = metrics_to_dict(m)
                    sweep_rows.append(row)

        sweep_df = pd.DataFrame(sweep_rows)
        # Sort safely: false present first priority, then false absent, then review count, then safety desc/F1 desc.
        sweep_df = sweep_df.sort_values(
            by=["false_present", "false_absent", "actual_present_in_review", "actual_absent_in_review", "safety_score", "f1"],
            ascending=[True, True, True, True, False, False],
        )
        sweep_path = output_dir / f"phase2_threshold_sweep_{stamp}.csv"
        sweep_df.to_csv(sweep_path, index=False)
        best_row = sweep_df.iloc[0]

        # Save best simulated student evidence for inspection.
        best_sim = simulate_from_detection_log(
            args.detection_log,
            enrolled,
            float(best_row["match_threshold"]),
            float(best_row["margin_threshold"]),
            int(best_row["checkpoint_min_detections"]),
        )
        best_sim_path = output_dir / f"best_simulated_student_evidence_{stamp}.csv"
        best_sim.to_csv(best_sim_path, index=False)

        print(f"Threshold sweep saved: {sweep_path}")
        print(f"Best simulated evidence saved: {best_sim_path}")

    report_path = output_dir / f"phase2_accuracy_report_{stamp}.txt"
    write_text_report(report_path, current_metrics, current_groups, best=best_row)

    print("\nPHASE 2 ACCURACY SUMMARY")
    print("=" * 32)
    for key, value in metrics_to_dict(current_metrics).items():
        print(f"{key}: {value}")
    print(f"\nReport saved: {report_path}")

    if current_groups["false_present"]:
        print("\nFALSE PRESENT roll numbers:")
        for roll in current_groups["false_present"]:
            print(f"- {roll}")
    if current_groups["false_absent"]:
        print("\nFALSE ABSENT roll numbers:")
        for roll in current_groups["false_absent"]:
            print(f"- {roll}")
    if current_groups["actual_present_in_review"]:
        print("\nActual present students in Needs Review:")
        for roll in current_groups["actual_present_in_review"]:
            print(f"- {roll}")


if __name__ == "__main__":
    main()
