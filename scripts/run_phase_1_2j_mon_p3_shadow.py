from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.face_attendance.mon_p3_shadow_validation import (  # noqa: E402
    EXPECTED_SESSION_ID,
    EXPECTED_SLOT_ID,
    EXPECTED_SUBJECT,
    MonP3ShadowError,
    evaluate_mon_p3_review,
    preflight_mon_p3_shadow,
    run_mon_p3_shadow,
    verify_mon_p3_output,
    write_preflight,
)


DEFAULT_FAMILY_ID = "embfam-7bc431a3ad762398d4e9"
DEFAULT_REFERENCE_DIAGNOSTIC = (
    ROOT
    / "attendance_output"
    / "diagnostics"
    / "2026-06-30__B51__P2__CVO_20260712_221105_286649"
)
DEFAULT_FAMILY_DIR = ROOT / "models" / "versions" / DEFAULT_FAMILY_ID
DEFAULT_CANONICAL_VIDEO_ROOT = (
    ROOT / "cctv_videos" / "prepared_slots" / "2026-06-22" / "MON_P3"
)
DEFAULT_MIRROR_VIDEO_ROOT = ROOT / "cctv_videos" / "MON_P3"
DEFAULT_FREEZE_JSON = Path(r"D:\Downloads\phase_1_2j_mon_p3_source_freeze.json")
DEFAULT_FREEZE_CSV = Path(r"D:\Downloads\phase_1_2j_mon_p3_source_freeze.csv")
DEFAULT_OUTPUT_ROOT = ROOT / "attendance_output" / "shadow_validation" / "phase_1_2j"
DEFAULT_PREFLIGHT_JSON = ROOT / "models" / "versions" / "phase_1_2j_preflight.json"


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--family-dir", default=str(DEFAULT_FAMILY_DIR))
    parser.add_argument("--expected-family-id", default=DEFAULT_FAMILY_ID)
    parser.add_argument("--freeze-json", default=str(DEFAULT_FREEZE_JSON))
    parser.add_argument("--freeze-csv", default=str(DEFAULT_FREEZE_CSV))
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
            "Run guarded Phase 1.2J production-vs-candidate shadow validation on "
            "the frozen untouched MON_P3 source."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

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
        "verify", help="Verify a completed immutable Phase 1.2J output"
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
    return preflight_mon_p3_shadow(
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
    print("Phase 1.2J MON_P3 immutable shadow preflight")
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
            result = run_mon_p3_shadow(
                preflight,
                python_executable=Path(sys.executable),
                status_callback=print,
            )
            print(f"Phase 1.2J output: {result.output_dir}")
            print(f"Production diagnostic run: {result.production_diagnostic_run}")
            print(f"Candidate diagnostic run: {result.candidate_diagnostic_run}")
            print(f"Blind exception review package: {result.review_package}")
            print(f"Material exception tracklets: {result.summary['review_tracklets']}")
            print(f"Decision: {result.summary['decision']}")
            print(f"Idempotent reuse: {'yes' if result.idempotent_reuse else 'no'}")
            print("Production embeddings changed: no")
            print("Official attendance changed: no")
            print("Candidate promoted: no")
            print("MON_P3 processed: yes")
            print(f"PHASE_1_2J_OUTPUT={result.output_dir}")
            return

        if args.command == "verify":
            verified = verify_mon_p3_output(
                output_dir=Path(args.output_dir),
                production_embeddings=Path(args.production_embeddings),
                production_summary=Path(args.production_summary),
                family_dir=Path(args.family_dir),
            )
            print("Phase 1.2J output verification: PASS")
            print(json.dumps(verified, indent=2))
            print("PHASE_1_2J_VERIFY_STATUS=PASS")
            return

        if args.command == "evaluate":
            summary, output = evaluate_mon_p3_review(
                output_dir=Path(args.output_dir),
                labels_path=Path(args.labels),
                student_map=Path(args.student_map),
                subject_abbr=args.subject_abbr,
            )
            print(f"Phase 1.2J evaluation output: {output}")
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
            print(f"PHASE_1_2J_EVALUATION={output}")
            return
    except MonP3ShadowError as exc:
        raise SystemExit(f"Phase 1.2J failed safely: {exc}") from exc


if __name__ == "__main__":
    main()
