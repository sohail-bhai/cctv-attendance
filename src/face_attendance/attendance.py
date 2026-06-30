from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd


class AttendanceBook:
    def __init__(self, all_students: list[str], min_detections: int) -> None:
        self.all_students = sorted(all_students)
        self.min_detections = int(min_detections)
        self.counts: Dict[str, int] = defaultdict(int)
        self.best_scores: Dict[str, List[float]] = defaultdict(list)
        self.videos_seen: Dict[str, Set[str]] = defaultdict(set)
        self.first_seen: Dict[str, str] = {}
        self.last_seen: Dict[str, str] = {}

    def add_detection(self, roll_no: str, score: float, video_name: str, time_text: str) -> None:
        self.counts[roll_no] += 1
        self.best_scores[roll_no].append(float(score))
        self.videos_seen[roll_no].add(video_name)
        self.first_seen.setdefault(roll_no, time_text)
        self.last_seen[roll_no] = time_text

    def present_students(self) -> set[str]:
        return {roll for roll, count in self.counts.items() if count >= self.min_detections}

    def to_dataframe(self) -> pd.DataFrame:
        present = self.present_students()
        rows = []
        for roll in self.all_students:
            scores = self.best_scores.get(roll, [])
            rows.append({
                "Roll_Number": roll,
                "Detection_Count": self.counts.get(roll, 0),
                "Best_Score": round(max(scores), 4) if scores else "",
                "Average_Score": round(sum(scores) / len(scores), 4) if scores else "",
                "Videos_Seen": ", ".join(sorted(self.videos_seen.get(roll, []))),
                "First_Seen": self.first_seen.get(roll, ""),
                "Last_Seen": self.last_seen.get(roll, ""),
                "Present": "Yes" if roll in present else "No",
            })
        return pd.DataFrame(rows)


def save_outputs(attendance_df: pd.DataFrame, log_df: pd.DataFrame, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    attendance_csv = output_dir / f"attendance_{timestamp}.csv"
    log_csv = output_dir / f"detection_log_{timestamp}.csv"
    excel_file = output_dir / f"attendance_{timestamp}.xlsx"

    attendance_df.to_csv(attendance_csv, index=False)
    log_df.to_csv(log_csv, index=False)

    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        attendance_df.to_excel(writer, sheet_name="Attendance", index=False)
        log_df.to_excel(writer, sheet_name="Detection Log", index=False)

    return {
        "attendance_csv": attendance_csv,
        "log_csv": log_csv,
        "excel_file": excel_file,
    }
