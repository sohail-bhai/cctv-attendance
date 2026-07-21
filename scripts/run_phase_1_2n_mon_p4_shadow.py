from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.face_attendance.mon_p4_shadow_validation import (  # noqa: E402
    EXPECTED_SESSION_ID,
    EXPECTED_SLOT_ID,
    EXPECTED_SUBJECT,
    MonP4ShadowError,
    create_mon_p4_source_freeze,
    evaluate_mon_p4_review,
    preflight_mon_p4_shadow,
    run_mon_p4_shadow,
    verify_mon_p4_output,
    write_preflight,
)

DEFAULT_FAMILY_ID = "embfam-274b5207b8b71294ff75"
DEFAULT_REFERENCE_DIAGNOSTIC = (
    ROOT
    / "attendance_output"
    / "diagnostics"
    / "2026-06-30__B51__P2__CVO_20260712_221105_286649"
)
DEFAULT_FAMILY_DIR = ROOT / "models" / "versions" / DEFAULT_FAMILY_ID
DEFAULT_CANONICAL_VIDEO_ROOT = (
    ROOT / "cctv_videos" / "prepared_slots" / "2026-06-22" / "MON_P4"
)
DEFAULT_MIRROR_VIDEO_ROOT = ROOT / "cctv_videos" / "MON_P4"
DEFAULT_EXPECTED_INVENTORY = ROOT / "data" / "phase_1_2n_mon_p4_expected_inventory.csv"
DEFAULT_PHASE_ROOT = ROOT / "attendance_output" / "shadow_validation" / "phase_1_2n"
DEFAULT_FREEZE_ROOT = DEFAULT_PHASE_ROOT / "source_freeze"
DEFAULT_OUTPUT_ROOT = DEFAULT_PHASE_ROOT / "shadow_runs"
DEFAULT_PREFLIGHT_JSON = ROOT / "models" / "versions" / "phase_1_2n_preflight.json"


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--family-dir", default=str(DEFAULT_FAMILY_DIR))
    parser.add_argument("--expected-family-id", default=DEFAULT_FAMILY_ID)
    parser.add_argument("--freeze-json", required=True)
    parser.add_argument("--freeze-csv", required=True)
    parser.add_argument("--canonical-video-root", default=str(DEFAULT_CANONICAL_VIDEO_ROOT))
    parser.add_argument("--mirror-video-root", default=str(DEFAULT_MIRROR_VIDEO_ROOT))
    parser.add_argument("--reference-diagnostic-run", default=str(DEFAULT_REFERENCE_DIAGNOSTIC))
    parser.add_argument("--student-map", default=str(ROOT / "data" / "student_faculty_map.json"))
    parser.add_argument("--timetable", default=str(ROOT / "timetable_b51_2026_2027.csv"))
    parser.add_argument("--production-embeddings", default=str(ROOT / "models" / "student_embeddings.pkl"))
    parser.add_argument("--production-summary", default=str(ROOT / "models" / "embedding_summary.csv"))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--session-id", default=EXPECTED_SESSION_ID)
    parser.add_argument("--slot-id", default=EXPECTED_SLOT_ID)
    parser.add_argument("--subject-abbr", default=EXPECTED_SUBJECT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run guarded Phase 1.2N production-vs-candidate shadow validation on "
            "the newly frozen untouched MON_P4 CVO source."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze_parser = subparsers.add_parser(
        "freeze", help="Verify untouched status and create an immutable MON_P4 source freeze"
    )
    freeze_parser.add_argument("--repo-root", default=str(ROOT))
    freeze_parser.add_argument("--family-dir", default=str(DEFAULT_FAMILY_DIR))
    freeze_parser.add_argument("--canonical-video-root", default=str(DEFAULT_CANONICAL_VIDEO_ROOT))
    freeze_parser.add_argument("--mirror-video-root", default=str(DEFAULT_MIRROR_VIDEO_ROOT))
    freeze_parser.add_argument("--expected-inventory", default=str(DEFAULT_EXPECTED_INVENTORY))
    freeze_parser.add_argument("--output-root", default=str(DEFAULT_FREEZE_ROOT))

    preflight_parser = subparsers.add_parser(
        "preflight", help="Validate all immutable inputs without running recognition"
    )
    _add_common(preflight_parser)
    preflight_parser.add_argument("--output-json", default=str(DEFAULT_PREFLIGHT_JSON))

    run_parser = subparsers.add_parser(
        "run", help="Run two diagnostic-only passes and export blind material-delta review evidence"
    )
    _add_common(run_parser)

    verify_parser = subparsers.add_parser(
        "verify", help="Verify a completed immutable Phase 1.2N output"
    )
    verify_parser.add_argument("--output-dir", required=True)
    verify_parser.add_argument("--family-dir", default=str(DEFAULT_FAMILY_DIR))
    verify_parser.add_argument(
        "--production-embeddings", default=str(ROOT / "models" / "student_embeddings.pkl")
    )
    verify_parser.add_argument(
        "--production-summary", default=str(ROOT / "models" / "embedding_summary.csv")
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate completed blind exception labels without promoting the candidate"
    )
    evaluate_parser.add_argument("--output-dir", required=True)
    evaluate_parser.add_argument("--labels", required=True)
    evaluate_parser.add_argument(
        "--student-map", default=str(ROOT / "data" / "student_faculty_map.json")
    )
    evaluate_parser.add_argument("--subject-abbr", default=EXPECTED_SUBJECT)
    return parser


def _preflight_from_args(args: argparse.Namespace):
    return preflight_mon_p4_shadow(
        repo_root=Path(args.repo_root),
        family_dir=Path(args.family_dir),
        freeze_json=Path(args.freeze_json),
        freeze_csv=Path(args.freeze_csv),
        canonical_video_root=Path(args.canonical_video_root),
        mirror_video_root=Path(args.mirror_video_root),
        reference_diagnostic_run=Path(args.reference_diagnostic_run),
        student_map=Path(args.student_map),
        timetable=Path(args.timetable),
        production_embeddings=Path(args.production_embeddings),
        production_summary=Path(args.production_summary),
        output_root=Path(args.output_root),
        session_id=args.session_id,
        slot_id=args.slot_id,
        subject_abbr=args.subject_abbr,
        expected_family_id=args.expected_family_id,
    )


def _print_preflight(preflight) -> None:
    print("Phase 1.2N MON_P4 immutable shadow preflight")
    print(f"Repository: {preflight.repo_root}")
    print(f"Session: {preflight.session_id}")
    print(f"Canonical videos: {preflight.video_root} ({len(preflight.videos)} videos)")
    print(f"Mirror videos: {preflight.mirror_video_root} (byte-identical)")
    print(f"Family ID: {preflight.family.family_id}")
    print(f"Candidate variant: {preflight.family.candidate_variant_id}")
    print(f"Candidate embeddings: {preflight.family.candidate_embedding_records}")
    print(f"CVO roster: {len(preflight.subject_rolls)}")
    print(f"Run ID: {preflight.run_id}")
    print(f"Planned output: {preflight.output_dir}")
    print(f"Production diagnostic: {preflight.production_diagnostic_run}")
    print(f"Candidate diagnostic: {preflight.candidate_diagnostic_run}")
    print("Recognition executed: no")
    print("Official attendance written: no")
    print("Candidate promoted: no")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "freeze":
            output, freeze_json, freeze_csv = create_mon_p4_source_freeze(
                repo_root=Path(args.repo_root),
                family_dir=Path(args.family_dir),
                canonical_root=Path(args.canonical_video_root),
                mirror_root=Path(args.mirror_video_root),
                expected_inventory_csv=Path(args.expected_inventory),
                output_root=Path(args.output_root),
            )
            print("Phase 1.2N MON_P4 source freeze: PASS")
            print(f"Freeze output: {output}")
            print(f"Freeze JSON: {freeze_json}")
            print(f"Freeze CSV: {freeze_csv}")
            print("Recognition executed: no")
            print("Official attendance written: no")
            print("Candidate promoted: no")
            print(f"PHASE_1_2N_FREEZE_OUTPUT={output}")
            print(f"PHASE_1_2N_FREEZE_JSON={freeze_json}")
            print(f"PHASE_1_2N_FREEZE_CSV={freeze_csv}")
            print("PHASE_1_2N_FREEZE_STATUS=PASS")
            return

        if args.command == "preflight":
            preflight = _preflight_from_args(args)
            output = write_preflight(preflight, Path(args.output_json))
            _print_preflight(preflight)
            print(f"Preflight JSON: {output}")
            print("PREFLIGHT_STATUS=PASS")
            return

        if args.command == "run":
            preflight = _preflight_from_args(args)
            _print_preflight(preflight)
            result = run_mon_p4_shadow(
                preflight,
                python_executable=Path(sys.executable),
                status_callback=print,
            )
            print(f"Phase 1.2N output: {result.output_dir}")
            print(f"Production diagnostic run: {result.production_diagnostic_run}")
            print(f"Candidate diagnostic run: {result.candidate_diagnostic_run}")
            print(f"Blind exception review package: {result.review_package}")
            print(f"Material exception tracklets: {result.summary['review_tracklets']}")
            print(f"Decision: {result.summary['decision']}")
            print(f"Idempotent reuse: {'yes' if result.idempotent_reuse else 'no'}")
            print("Production embeddings changed: no")
            print("Official attendance changed: no")
            print("Candidate promoted: no")
            print("MON_P4 processed: yes")
            print(f"PHASE_1_2N_OUTPUT={result.output_dir}")
            return

        if args.command == "verify":
            verified = verify_mon_p4_output(
                output_dir=Path(args.output_dir),
                production_embeddings=Path(args.production_embeddings),
                production_summary=Path(args.production_summary),
                family_dir=Path(args.family_dir),
            )
            print("Phase 1.2N output verification: PASS")
            print(json.dumps(verified, indent=2))
            print("PHASE_1_2N_VERIFY_STATUS=PASS")
            return

        if args.command == "evaluate":
            summary, output = evaluate_mon_p4_review(
                output_dir=Path(args.output_dir),
                labels_path=Path(args.labels),
                student_map=Path(args.student_map),
                subject_abbr=args.subject_abbr,
            )
            print(f"Phase 1.2N evaluation output: {output}")
            print(f"Reviewed exception tracklets: {summary['reviewed_exception_tracks']}")
            print(
                "Candidate false identities / unsafe accepts: "
                f"{summary['candidate_false_identities_or_unsafe_accepts']}"
            )
            print(
                "Candidate lost correct production accepts: "
                f"{summary['candidate_lost_correct_production_accepts']}"
            )
            print(f"Unverifiable exceptions: {summary['unverifiable_exceptions']}")
            print(f"Decision: {summary['decision']}")
            print("Production embeddings changed: no")
            print("Official attendance changed: no")
            print("Candidate promoted: no")
            print(f"PHASE_1_2N_EVALUATION={output}")
            return
    except MonP4ShadowError as exc:
        raise SystemExit(f"Phase 1.2N failed safely: {exc}") from exc


if __name__ == "__main__":
    main()
