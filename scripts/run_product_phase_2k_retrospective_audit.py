from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.face_attendance.product_phase_2k_retrospective_audit import (  # noqa: E402
    RetrospectiveAuditError,
    materialize,
    preflight,
    sha256_file,
    verify_immutable_output,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and materialize the read-only Product Phase 2K-C1 retrospective "
            "multi-session audit without opening video or running recognition."
        )
    )
    parser.add_argument("--repo-root", default=str(ROOT))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight", help="Verify all immutable inputs and compute the audit in memory.")
    sub.add_parser("materialize", help="Write or byte-verify the deterministic immutable audit output.")
    verify = sub.add_parser("verify", help="Independently verify an existing audit output.")
    verify.add_argument("--output-dir", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "verify":
            manifest = verify_immutable_output(Path(args.output_dir))
            print(f"Verified Product Phase 2K-C1 output: {args.output_dir}")
            print(f"Run ID: {manifest['run_id']}")
            print("VERIFY_STATUS=PASS")
            return 0
        plan = preflight(Path(args.repo_root))
        if args.command == "preflight":
            print("Product Phase 2K-C1 retrospective audit preflight")
            print(f"Run ID: {plan.run_id}")
            print(f"Normalized reviewed rows: {len(plan.normalized_records)}")
            print("Sessions audited: 4")
            print("Recognition ran: no")
            print("Video decoded/processed: no")
            print("Operational state changed: no")
            print("PREFLIGHT_STATUS=PASS")
            return 0
        output, reused, manifest = materialize(plan)
        summary = json.loads((output / "combined_summary.json").read_text(encoding="utf-8"))
        print("Product Phase 2K-C1 materialized safely.")
        print(f"Output: {output}")
        print(f"Immutable manifest SHA-256: {sha256_file(output / 'immutable_manifest.json')}")
        print(f"Idempotent reuse: {'yes' if reused else 'no'}")
        print(f"Recommendation: {summary['recommendation']}")
        print("Recognition ran: no")
        print("Video decoded/processed: no")
        print("Operational state changed: no")
        print(f"PRODUCT_PHASE_2K_C1_OUTPUT={output}")
        return 0
    except (RetrospectiveAuditError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Product Phase 2K-C1 stopped safely: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
