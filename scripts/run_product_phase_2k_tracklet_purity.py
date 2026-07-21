from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.face_attendance.product_phase_2k_tracklet_purity import (  # noqa: E402
    MIXED_TRACKLET_ID,
    TrackletPurityError,
    default_inputs,
    materialize_shadow,
    preflight,
)
from src.face_attendance.tracklet_purity import verify_immutable_output  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Product Phase 2K-A conservative mixed-track purity diagnostics from immutable "
            "MON P4 artifacts without decoding video or changing attendance authority."
        )
    )
    parser.add_argument("--repo-root", default=str(ROOT))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight", help="Verify pinned diagnostics, manifests, reports, model, and source hashes.")
    sub.add_parser("materialize", help="Create or verify the deterministic immutable shadow output.")
    verify = sub.add_parser("verify", help="Verify an existing immutable Phase 2K-A output.")
    verify.add_argument("--output-dir", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "verify":
            manifest = verify_immutable_output(Path(args.output_dir))
            print(f"Verified Phase 2K-A output: {args.output_dir}")
            print(f"Run ID: {manifest['run_id']}")
            print("VERIFY_STATUS=PASS")
            return 0
        result = preflight(default_inputs(Path(args.repo_root)), verify_videos=True)
        if args.command == "preflight":
            print("Product Phase 2K-A purity preflight")
            print(f"Run ID: {result.run_id}")
            print(f"Tracklets: {len(result.tracklets)}")
            print(f"Known mixed track observations: {len(result.observations[result.observations['Tracklet_ID'] == MIXED_TRACKLET_ID])}")
            print("Recognition repeated: no")
            print("Video decoded/reprocessed: no")
            print("Official attendance changed: no")
            print("Authority changed: no")
            print("PREFLIGHT_STATUS=PASS")
            return 0
        output, reused, evaluation = materialize_shadow(result)
        known = evaluation["known_mixed_track_analysis"]
        print("Product Phase 2K-A shadow materialized safely.")
        print(f"Output: {output}")
        print(f"Idempotent reuse: {'yes' if reused else 'no'}")
        print(f"Known mixed track: {known['final_shadow_disposition']}")
        print(f"Known mixed track split: {'yes' if known['defensible_split_exists'] else 'no'}")
        print("Recognition repeated: no")
        print("Video decoded/reprocessed: no")
        print("Official attendance changed: no")
        print("Authority changed: no")
        print(f"PRODUCT_PHASE_2K_A_OUTPUT={output}")
        return 0
    except (TrackletPurityError, ValueError) as exc:
        print(f"Product Phase 2K-A stopped safely: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
