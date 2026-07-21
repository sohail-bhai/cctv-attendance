from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.face_attendance.product_phase_2i_authority import (  # noqa: E402
    AUTHORITY_REVISION_ID,
    AUTOMATIC_RECOGNITION_AUTHORITY,
    MISSING_ENROLLMENT_ROLL,
    OFFICIAL_RECOGNITION_AUTHORITY,
    POLICY_VERSION,
    ProductPhase2IError,
    SESSION_ID,
    apply_authority_revision,
    default_inputs,
    preflight,
    rollback_authority_revision,
)

APPLY_CONFIRMATION = "APPLY_PHASE_2I_AUTHORITY"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate, apply, or roll back the Product Phase 2I attendance-authority revision."
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("preflight", help="Verify frozen evidence and calculate the revision without writing state.")

    apply_parser = sub.add_parser("apply", help="Apply the reviewed MON_P4 authority revision explicitly.")
    apply_parser.add_argument("--confirm", required=True)

    rollback_parser = sub.add_parser("rollback", help="Restore the exact pre-Phase-2I session entry.")
    rollback_parser.add_argument("--backup", required=True, type=Path)
    return parser


def _print_counts(summary: dict) -> None:
    print(f"Present: {int(summary.get('present', 0))}")
    print(f"Needs review: {int(summary.get('needs_review', 0))}")
    print(f"Unconfirmed: {int(summary.get('unconfirmed', 0))}")
    print(f"Missing enrollment: {int(summary.get('missing_enrollment', 0))}")
    print(f"Absent: {int(summary.get('absent', 0))}")
    print(f"Total roster rows: {int(summary.get('total', 0))}")


def _run_preflight(repo_root: Path) -> int:
    inputs = default_inputs(repo_root)
    result = preflight(inputs)
    print("Product Phase 2I authority preflight passed.")
    print(f"Session: {SESSION_ID}")
    print(f"Policy version: {POLICY_VERSION}")
    print(f"Authority revision: {AUTHORITY_REVISION_ID}")
    print(f"Current authority: {result.source_entry.get('official_recognition_authority')}")
    print(f"Reviewed revision authority: {OFFICIAL_RECOGNITION_AUTHORITY}")
    print(f"Future automatic authority: {AUTOMATIC_RECOGNITION_AUTHORITY}")
    _print_counts(result.summary)
    print(f"Missing-enrollment roll: {MISSING_ENROLLMENT_ROLL}")
    print("Mixed-track negative regression: PASS (CP1-cam5-back-TRK00005 excluded)")
    print("Guarded recovery automatic for future runs: no")
    print("Recognition executed: no")
    print("Official attendance changed: no")
    print("PREFLIGHT_STATUS=PASS")
    return 0


def _run_apply(repo_root: Path, confirmation: str) -> int:
    if confirmation != APPLY_CONFIRMATION:
        raise ProductPhase2IError(
            f"Explicit confirmation mismatch. Required: {APPLY_CONFIRMATION}"
        )
    result = apply_authority_revision(default_inputs(repo_root))
    print("Product Phase 2I authority revision applied safely.")
    print(f"Session: {result['session_id']}")
    print(f"Authority revision: {result['authority_revision_id']}")
    print(f"Official authority: {result['official_recognition_authority']}")
    print(f"Future automatic authority: {result['automatic_recognition_authority']}")
    _print_counts(result["summary"])
    print(f"Immutable output: {result['output_dir']}")
    print(f"State backup: {result['backup_path']}")
    print(f"Output reused: {'yes' if result['output_reused'] else 'no'}")
    print("Guarded recovery automatic for future runs: no")
    print("Recognition repeated: no")
    print("Video reprocessed: no")
    print("Old frame-only report overwritten: no")
    print("AUTHORITY_APPLY_STATUS=PASS")
    print(f"PRODUCT_PHASE_2I_BACKUP={result['backup_path']}")
    return 0


def _run_rollback(repo_root: Path, backup: Path) -> int:
    result = rollback_authority_revision(default_inputs(repo_root), backup)
    print("Product Phase 2I authority revision rolled back safely.")
    print(f"Session: {result['session_id']}")
    print(f"Backup: {result['backup_path']}")
    print(f"Restored source entry SHA-256: {result['restored_source_entry_sha256']}")
    print("Recognition executed: no")
    print("ROLLBACK_STATUS=PASS")
    return 0


def main() -> int:
    args = _parser().parse_args()
    command = args.command or "preflight"
    try:
        if command == "preflight":
            return _run_preflight(args.repo_root)
        if command == "apply":
            return _run_apply(args.repo_root, args.confirm)
        if command == "rollback":
            return _run_rollback(args.repo_root, args.backup)
        raise ProductPhase2IError(f"Unknown command: {command}")
    except ProductPhase2IError as exc:
        print(f"PRODUCT_PHASE_2I_ERROR={exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
