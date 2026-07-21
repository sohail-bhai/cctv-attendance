from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.face_attendance.report_revision_control import (  # noqa: E402
    POLICY_VERSION as REVISION_POLICY_VERSION,
    ReportRevisionError,
    repair_mon_p4_after_reprocess,
    rollback_mon_p4_repair,
)
from src.face_attendance.review_carry_forward import (  # noqa: E402
    POLICY_VERSION as CARRY_FORWARD_POLICY_VERSION,
    ReviewCarryForwardError,
    build_exact_mon_p4_registry,
    verify_registry,
)
from src.face_attendance.workflow_state import atomic_write_json  # noqa: E402

SESSION_ID = "2026-06-22__B51__P4__CVO"
SOURCE_RELATIVE = "cctv_videos/prepared_slots/2026-06-22/MON_P4"


def _build_registry(repo_root: Path) -> dict:
    return build_exact_mon_p4_registry(
        repo_root=repo_root,
        video_dir=repo_root / SOURCE_RELATIVE,
        embeddings_path=repo_root / "models" / "student_embeddings.pkl",
        input_source_path=SOURCE_RELATIVE,
    )


def run(repo_root: Path, *, apply: bool) -> dict:
    root = Path(repo_root).resolve()
    registry_path = root / "data" / "review_evidence_registry.json"

    registry = _build_registry(root)
    verify_registry(registry)
    repair = repair_mon_p4_after_reprocess(root, apply=False)
    result = {
        "status": "preflight_passed",
        "session_id": SESSION_ID,
        "revision_policy_version": REVISION_POLICY_VERSION,
        "carry_forward_policy_version": CARRY_FORWARD_POLICY_VERSION,
        "registry_id": registry.get("registry_id"),
        "reviewed_pairs": registry.get("reviewed_pairs"),
        "accepted_pairs": registry.get("accepted_pairs"),
        "mixed_track_pairs": registry.get("mixed_track_pairs"),
        "reviewed_summary": repair.get("reviewed_summary"),
        "recognition_repeated": False,
        "video_reprocessed": False,
        "official_attendance_changed": False,
    }
    if not apply:
        return result

    previous_registry_bytes = registry_path.read_bytes() if registry_path.exists() else None
    try:
        atomic_write_json(registry_path, registry)
        installed_registry_bytes = registry_path.read_bytes()
        installed_registry_hash = hashlib.sha256(installed_registry_bytes).hexdigest()
        verify_registry(json.loads(installed_registry_bytes.decode("utf-8-sig")))
        committed = repair_mon_p4_after_reprocess(
            root,
            apply=True,
            previous_registry_bytes=previous_registry_bytes,
            installed_registry_sha256=installed_registry_hash,
        )
    except Exception:
        if previous_registry_bytes is None:
            registry_path.unlink(missing_ok=True)
        else:
            registry_path.write_bytes(previous_registry_bytes)
        raise

    changed = committed.get("status") == "applied"
    result.update(
        {
            "status": "applied" if changed else "already_applied",
            "official_attendance_changed": bool(changed),
            "registry_path": str(registry_path),
            "registry_sha256": installed_registry_hash,
            "backup_path": committed.get("backup_path"),
            "candidate_revision": committed.get("candidate_revision"),
            "reviewed_summary": committed.get("reviewed_summary") or committed.get("summary"),
            "official_authority": committed.get("official_authority"),
            "automatic_authority": committed.get("automatic_authority"),
        }
    )
    return result


def run_rollback(repo_root: Path, backup_path: Path) -> dict:
    result = rollback_mon_p4_repair(repo_root, backup_path)
    result.update(
        {
            "revision_policy_version": REVISION_POLICY_VERSION,
            "carry_forward_policy_version": CARRY_FORWARD_POLICY_VERSION,
            "backup_path": str(Path(backup_path).resolve()),
            "official_attendance_changed": True,
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Product Phase 2J report-revision and exact-review carry-forward stabilization")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--rollback", action="store_true")
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()

    if args.apply and args.confirm != "APPLY_PHASE_2J_STABILIZATION":
        print("Refusing to apply without --confirm APPLY_PHASE_2J_STABILIZATION", file=sys.stderr)
        return 2
    if args.rollback:
        if args.confirm != "ROLLBACK_PHASE_2J_STABILIZATION":
            print("Refusing to roll back without --confirm ROLLBACK_PHASE_2J_STABILIZATION", file=sys.stderr)
            return 2
        if args.backup is None:
            print("Rollback requires --backup <Phase 2J state backup path>", file=sys.stderr)
            return 2
    elif args.backup is not None:
        print("--backup is valid only with --rollback", file=sys.stderr)
        return 2

    try:
        result = (
            run_rollback(args.repo_root, args.backup)
            if args.rollback
            else run(args.repo_root, apply=args.apply)
        )
    except (ReportRevisionError, ReviewCarryForwardError, RuntimeError, ValueError) as exc:
        print(f"PRODUCT_PHASE_2J_STATUS=FAIL\n{exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("")
    if args.rollback:
        summary = result.get("restored_summary") or {}
        print("Product Phase 2J report stabilization rolled back safely.")
        print("Mode: ROLLBACK")
        print(f"Session: {result.get('session_id')}")
        print(
            "Restored automatic candidate: "
            f"{summary.get('present', 0)} Present / "
            f"{summary.get('needs_review', 0)} Needs Review / "
            f"{summary.get('unconfirmed', 0)} Unconfirmed / "
            f"{summary.get('missing_enrollment', 0)} Missing Enrollment / "
            f"{summary.get('absent', 0)} Absent"
        )
        print(f"Review registry action: {result.get('registry_action')}")
        print("Candidate revision archive preserved: true")
    else:
        summary = result.get("reviewed_summary") or {}
        candidate = result.get("candidate_revision") or {}
        print("Product Phase 2J report stabilization verified safely.")
        print(f"Mode: {'APPLY' if args.apply else 'PREFLIGHT'}")
        print(f"Session: {result.get('session_id')}")
        print(
            "Official reviewed result: "
            f"{summary.get('present', 0)} Present / "
            f"{summary.get('needs_review', 0)} Needs Review / "
            f"{summary.get('unconfirmed', 0)} Unconfirmed / "
            f"{summary.get('missing_enrollment', 0)} Missing Enrollment / "
            f"{summary.get('absent', 0)} Absent"
        )
        if candidate:
            print(
                "Archived automatic candidate: "
                f"{candidate.get('present_count', 0)} Present / "
                f"{candidate.get('needs_review_count', 0)} Needs Review / "
                f"{candidate.get('unconfirmed_count', 0)} Unconfirmed / "
                f"{candidate.get('missing_enrollment_count', 0)} Missing Enrollment / "
                f"{candidate.get('absent_count', 0)} Absent"
            )
        print(f"Reviewed evidence pairs registered: {result.get('reviewed_pairs', 0)}")
        print(f"Known mixed tracks quarantined: {result.get('mixed_track_pairs', 0)}")
    print(f"Recognition repeated: {str(bool(result.get('recognition_repeated'))).lower()}")
    print(f"Video reprocessed: {str(bool(result.get('video_reprocessed'))).lower()}")
    print(f"Official attendance changed in this command: {str(bool(result.get('official_attendance_changed'))).lower()}")
    if result.get("backup_path"):
        print(f"State backup: {result.get('backup_path')}")
        if args.apply:
            print("Rollback command (use only if this applied revision must be reverted):")
            print(
                f'& "{Path(args.repo_root).resolve()}\\scripts\\run_product_phase_2j_stabilization.ps1" '
                f'-Rollback -Backup "{result.get("backup_path")}" '
                '-Confirm ROLLBACK_PHASE_2J_STABILIZATION'
            )
    print("PRODUCT_PHASE_2J_STATUS=PASS")
    print(f"PRODUCT_PHASE_2J_MODE={'ROLLBACK' if args.rollback else 'APPLY' if args.apply else 'PREFLIGHT'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
