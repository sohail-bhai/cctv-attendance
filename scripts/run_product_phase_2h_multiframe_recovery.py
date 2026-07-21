from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.face_attendance.product_phase_2h_multiframe_recovery import (  # noqa: E402
    EXPECTED_PENDING_REVIEW,
    EXPECTED_REUSED_LABELS,
    EXPECTED_SELECTED_PAIRS,
    ProductPhase2HError,
    default_inputs,
    evaluate_recovery,
    export_recovery_review,
    preflight,
    verify_output,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reuse the completed Product Phase 2G MON_P4 diagnostics, validate a conservative "
            "multi-frame recovery rule against frozen human-reviewed evidence, export only "
            "attendance-changing tracklets for blind review, and create a roster-first shadow "
            "report without rerunning recognition or changing official attendance."
        )
    )
    parser.add_argument("--repo-root", default=str(ROOT))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight", help="Verify diagnostics, frozen benchmark, prior labels, roster, and videos.")
    sub.add_parser("export-review", help=f"Create the immutable {EXPECTED_PENDING_REVIEW}-item blind review package.")
    evaluate = sub.add_parser("evaluate", help="Evaluate completed blind labels without changing official attendance.")
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--labels", required=True)
    verify = sub.add_parser("verify", help="Verify an existing immutable Phase 2H output.")
    verify.add_argument("--output-dir", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    inputs = default_inputs(Path(args.repo_root))
    try:
        if args.command == "preflight":
            result = preflight(inputs, verify_videos=True)
            tiers = result.selected["Evidence_Tier"].value_counts().to_dict()
            print("Product Phase 2H - guarded multi-frame recall recovery preflight")
            print(f"Output ID: {result.output_id}")
            print(f"Selected identity/checkpoint pairs: {len(result.selected)}")
            print(f"  - strict accepted representatives: {tiers.get('strict_accepted', 0)}")
            print(f"  - guarded near-threshold recoveries: {tiers.get('guarded_recovery', 0)}")
            print(f"Frozen leakage-safe rule validation: {len(result.historical_validation)} correct, 0 unsafe")
            print(f"Reused frozen Phase 1.2N labels: {len(result.reused)}")
            print(f"New blind review items: {len(result.pending)}")
            print("Low-evidence roster status: Unconfirmed, not Absent")
            print("Recognition repeated: no")
            print("Official attendance changed: no")
            print("Tracklet authority activated: no")
            print("PREFLIGHT_STATUS=PASS")
            return 0
        if args.command == "export-review":
            output, reviewer, reused = export_recovery_review(inputs)
            print("Product Phase 2H multi-frame recovery review exported safely.")
            print(f"Output: {output}")
            print(f"Blind reviewer: {reviewer}")
            print(f"Selected identity/checkpoint pairs: {EXPECTED_SELECTED_PAIRS}")
            print(f"New review items: {EXPECTED_PENDING_REVIEW}")
            print(f"Reused Phase 1.2N labels: {EXPECTED_REUSED_LABELS}")
            print(f"Idempotent reuse: {'yes' if reused else 'no'}")
            print("Recognition repeated: no")
            print("Official attendance changed: no")
            print("Tracklet authority activated: no")
            print(f"PRODUCT_PHASE_2H_OUTPUT={output}")
            print(f"PRODUCT_PHASE_2H_REVIEWER={reviewer}")
            return 0
        if args.command == "evaluate":
            evaluation = evaluate_recovery(inputs, Path(args.output_dir), Path(args.labels))
            summary = json.loads((evaluation / "evaluation_summary.json").read_text(encoding="utf-8"))
            shadow = summary["multiframe_shadow_result"]
            print("Product Phase 2H multi-frame recovery evaluated safely.")
            print(f"Evaluation: {evaluation}")
            print(f"Decision: {summary['decision']}")
            print(f"Unsafe identity errors: {summary['unsafe_identity_errors']}")
            print(f"Unverifiable pairs: {summary['unverifiable_pairs']}")
            print(f"Present: {shadow['present']}")
            print(f"Needs review: {shadow['needs_review']}")
            print(f"Unconfirmed: {shadow['unconfirmed']}")
            print(f"Present rolls: {', '.join(summary['present_rolls']) or 'none'}")
            print(f"Needs-review rolls: {', '.join(summary['needs_review_rolls']) or 'none'}")
            print("Recognition repeated: no")
            print("Official attendance changed: no")
            print("Tracklet authority activated: no")
            print(f"PRODUCT_PHASE_2H_EVALUATION={evaluation}")
            return 0
        if args.command == "verify":
            manifest = verify_output(Path(args.output_dir))
            print(f"Verified Phase 2H output: {args.output_dir}")
            print(f"Output ID: {manifest['output_id']}")
            print("VERIFY_STATUS=PASS")
            return 0
    except ProductPhase2HError as exc:
        print(f"Product Phase 2H stopped safely: {exc}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
