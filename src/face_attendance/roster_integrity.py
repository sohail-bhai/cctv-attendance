from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .diagnostics import json_safe
from .recall_analysis import build_embedding_roster_coverage, load_embedding_counts
from .tracklet_review import load_actual_present, normalize_roll


class RosterIntegrityError(ValueError):
    pass


@dataclass(frozen=True)
class RosterAuditResult:
    roster: pd.DataFrame
    embedding_coverage: pd.DataFrame
    summary: dict[str, Any]


def _clean_name(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _normalized_subjects(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted({str(subject or "").strip().upper() for subject in value if str(subject or "").strip()})


def _duplicate_rolls(rows: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for row in rows:
        roll = normalize_roll(row.get("roll"))
        if not roll:
            continue
        if roll in seen:
            duplicates.add(roll)
        seen.add(roll)
    return sorted(duplicates)


def audit_subject_roster(
    student_map_path: Path,
    subject_abbr: str,
    *,
    actual_present_path: Path | None = None,
    embedding_summary_path: Path | None = None,
    embeddings_path: Path | None = None,
) -> RosterAuditResult:
    student_map_path = Path(student_map_path)
    try:
        payload = json.loads(student_map_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RosterIntegrityError(f"Could not load student mapping: {student_map_path}") from exc

    subject = str(subject_abbr or "").strip().upper()
    if not subject:
        raise RosterIntegrityError("subject_abbr is required")
    subject_students = payload.get("subject_students", {})
    if not isinstance(subject_students, dict) or subject not in subject_students:
        raise RosterIntegrityError(f"Student mapping has no subject roster for {subject}")
    subject_rows = subject_students.get(subject, [])
    all_rows = payload.get("all_students", [])
    if not isinstance(subject_rows, list) or not isinstance(all_rows, list):
        raise RosterIntegrityError("Student mapping roster collections must be lists")

    errors: list[str] = []
    warnings: list[str] = []
    subject_duplicates = _duplicate_rolls(subject_rows)
    all_duplicates = _duplicate_rolls(all_rows)
    if subject_duplicates:
        errors.append(f"Duplicate {subject} roster rolls: {', '.join(subject_duplicates)}")
    if all_duplicates:
        errors.append(f"Duplicate all_students rolls: {', '.join(all_duplicates)}")

    all_by_roll = {
        normalize_roll(row.get("roll")): row
        for row in all_rows
        if normalize_roll(row.get("roll"))
    }
    subject_by_roll = {
        normalize_roll(row.get("roll")): row
        for row in subject_rows
        if normalize_roll(row.get("roll"))
    }
    if any(not normalize_roll(row.get("roll")) for row in subject_rows):
        errors.append(f"{subject} roster contains a blank roll number")

    missing_from_all: list[str] = []
    membership_missing: list[str] = []
    name_mismatches: list[str] = []
    for roll, subject_row in subject_by_roll.items():
        master_row = all_by_roll.get(roll)
        if master_row is None:
            missing_from_all.append(roll)
            continue
        if subject not in _normalized_subjects(master_row.get("subjects")):
            membership_missing.append(roll)
        subject_name = _clean_name(subject_row.get("name"))
        master_name = _clean_name(master_row.get("name"))
        if subject_name.casefold() != master_name.casefold():
            name_mismatches.append(roll)

    stale_membership = sorted(
        roll
        for roll, row in all_by_roll.items()
        if subject in _normalized_subjects(row.get("subjects")) and roll not in subject_by_roll
    )
    if missing_from_all:
        errors.append(f"{subject} students missing from all_students: {', '.join(sorted(missing_from_all))}")
    if membership_missing:
        errors.append(
            f"all_students subjects missing {subject} membership: {', '.join(sorted(membership_missing))}"
        )
    if stale_membership:
        errors.append(
            f"all_students has stale {subject} membership: {', '.join(stale_membership)}"
        )
    if name_mismatches:
        errors.append(f"Student-name mismatch between rosters: {', '.join(sorted(name_mismatches))}")

    actual_present: set[str] | None = None
    missing_from_actual: list[str] = []
    actual_not_in_roster: list[str] = []
    if actual_present_path is not None:
        actual_present = load_actual_present(Path(actual_present_path))
        roster_rolls = set(subject_by_roll)
        missing_from_actual = sorted(roster_rolls.difference(actual_present))
        actual_not_in_roster = sorted(actual_present.difference(roster_rolls))
        if missing_from_actual:
            errors.append(
                f"Actual-present file is missing {subject} roster students: {', '.join(missing_from_actual)}"
            )
        if actual_not_in_roster:
            errors.append(
                f"Actual-present file contains non-{subject} rolls: {', '.join(actual_not_in_roster)}"
            )

    roster_records: list[dict[str, Any]] = []
    for row in subject_rows:
        roll = normalize_roll(row.get("roll"))
        master = all_by_roll.get(roll, {})
        roster_records.append({
            "Canonical_Roll": roll,
            "Student_Name": _clean_name(row.get("name")),
            "In_All_Students": "Yes" if roll in all_by_roll else "No",
            "All_Students_Subjects": "; ".join(_normalized_subjects(master.get("subjects"))),
            "Subject_Membership_Consistent": (
                "Yes" if roll in all_by_roll and subject in _normalized_subjects(master.get("subjects")) else "No"
            ),
            "Actual_Present": (
                "Yes" if actual_present is not None and roll in actual_present
                else "No" if actual_present is not None
                else "Not_Checked"
            ),
        })
    roster_frame = pd.DataFrame(roster_records)

    coverage = pd.DataFrame()
    if embedding_summary_path is not None or embeddings_path is not None:
        if embedding_summary_path is None or embeddings_path is None:
            raise RosterIntegrityError(
                "Both embedding_summary_path and embeddings_path are required for embedding coverage"
            )
        summary_path = Path(embedding_summary_path)
        database_path = Path(embeddings_path)
        if not summary_path.is_file() or not database_path.is_file():
            raise RosterIntegrityError("Embedding summary or embedding database was not found")
        embedding_summary = pd.read_csv(summary_path, dtype=str, keep_default_na=False)
        coverage = build_embedding_roster_coverage(
            subject_rows,
            embedding_summary,
            embedding_counts=load_embedding_counts(database_path),
        )
        if not coverage.empty:
            coverage_by_roll = coverage.set_index("Canonical_Roster_Roll")
            roster_frame["Embedding_Available"] = roster_frame["Canonical_Roll"].map(
                coverage_by_roll["Embedding_Available"]
            ).fillna("No")
            roster_frame["Embedding_Record_Count"] = roster_frame["Canonical_Roll"].map(
                coverage_by_roll["Embedding_Record_Count"]
            ).fillna(0).astype(int)
            roster_frame["Embedding_Mapping_Status"] = roster_frame["Canonical_Roll"].map(
                coverage_by_roll["Canonical_Mapping_Status"]
            ).fillna("missing")
    else:
        warnings.append("Embedding coverage was not checked")

    missing_embeddings = (
        sorted(coverage.loc[coverage["Embedding_Available"].eq("No"), "Canonical_Roster_Roll"].astype(str))
        if not coverage.empty
        else []
    )
    summary = {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "subject_abbr": subject,
        "student_map": str(student_map_path.resolve()),
        "actual_present": str(Path(actual_present_path).resolve()) if actual_present_path is not None else None,
        "roster_students": len(subject_rows),
        "unique_roster_rolls": len(subject_by_roll),
        "all_students": len(all_rows),
        "actual_present_students": len(actual_present) if actual_present is not None else None,
        "embedding_available": (
            int(coverage["Embedding_Available"].eq("Yes").sum()) if not coverage.empty else None
        ),
        "embedding_missing": len(missing_embeddings) if not coverage.empty else None,
        "missing_embedding_rolls": missing_embeddings,
        "subject_duplicate_rolls": subject_duplicates,
        "all_students_duplicate_rolls": all_duplicates,
        "missing_from_all_students": sorted(missing_from_all),
        "missing_subject_membership": sorted(membership_missing),
        "stale_subject_membership": stale_membership,
        "name_mismatches": sorted(name_mismatches),
        "missing_from_actual_present": missing_from_actual,
        "actual_present_not_in_roster": actual_not_in_roster,
        "errors": errors,
        "warnings": warnings,
        "integrity_passed": not errors,
    }
    return RosterAuditResult(roster=roster_frame, embedding_coverage=coverage, summary=summary)


def write_roster_audit(result: RosterAuditResult, output_dir: Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    subject = str(result.summary.get("subject_abbr") or "subject").lower()
    roster_path = output_dir / f"{subject}_roster_audit.csv"
    coverage_path = output_dir / f"{subject}_embedding_coverage.csv"
    summary_path = output_dir / f"{subject}_roster_audit_summary.json"
    report_path = output_dir / f"{subject}_roster_audit_report.md"
    result.roster.to_csv(roster_path, index=False, encoding="utf-8-sig")
    result.embedding_coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(
        json.dumps(json_safe(result.summary), indent=2, ensure_ascii=True), encoding="utf-8"
    )
    status = "PASS" if result.summary["integrity_passed"] else "FAIL"
    errors = result.summary.get("errors", [])
    warnings = result.summary.get("warnings", [])
    report_path.write_text(
        "# Subject Roster and Embedding Audit\n\n"
        f"- Subject: {result.summary['subject_abbr']}\n"
        f"- Integrity: {status}\n"
        f"- Roster students: {result.summary['roster_students']}\n"
        f"- Actual-present students: {result.summary['actual_present_students']}\n"
        f"- Embedding available: {result.summary['embedding_available']}\n"
        f"- Missing embeddings: {result.summary['embedding_missing']}\n"
        f"- Missing embedding rolls: {', '.join(result.summary['missing_embedding_rolls']) or 'none'}\n\n"
        "## Errors\n\n"
        + ("\n".join(f"- {value}" for value in errors) if errors else "- None")
        + "\n\n## Warnings\n\n"
        + ("\n".join(f"- {value}" for value in warnings) if warnings else "- None")
        + "\n",
        encoding="utf-8",
    )
    return {
        "roster": roster_path,
        "embedding_coverage": coverage_path,
        "summary": summary_path,
        "report": report_path,
    }
