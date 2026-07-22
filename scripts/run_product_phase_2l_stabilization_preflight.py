from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.face_attendance.product_phase_2l_stabilization import (  # noqa: E402
    REQUIRED_EXTERNAL_VALIDATIONS,
    StabilizationError,
    materialize,
    preflight,
    sha256_file,
    verify_immutable_output,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify and freeze the Product Phase 2L controlled-demo release candidate.")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("preflight", help="Run the read-only state, frontend-contract, and backend-contract checks.")
    materialize_parser = sub.add_parser("materialize", help="Create or reuse the deterministic immutable Phase 2L package.")
    materialize_parser.add_argument("--validated", action="append", default=[], choices=REQUIRED_EXTERNAL_VALIDATIONS)
    verify_parser = sub.add_parser("verify", help="Independently verify an existing immutable Phase 2L package.")
    verify_parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    command = args.command or "preflight"
    try:
        if command == "verify":
            manifest = verify_immutable_output(args.output_dir)
            print(f"Verified Product Phase 2L output: {Path(args.output_dir)}")
            print(f"Run ID: {manifest['run_id']}")
            print(f"Immutable manifest SHA-256: {sha256_file(Path(args.output_dir) / 'immutable_manifest.json')}")
            print("VERIFY_STATUS=PASS")
            return 0

        plan = preflight(args.repo_root, args.output_root, run_tests=True)
        print("Product Phase 2L read-only preflight passed.")
        print(f"Run ID: {plan.run_id}")
        print(f"Official totals: {plan.official_counts}")
        print(f"Candidate totals: {plan.candidate_counts}")
        print("Recognition executed: false")
        print("Operational hash drift: false")
        if command == "preflight":
            print("PREFLIGHT_STATUS=PASS")
            return 0

        output_dir, reused, manifest = materialize(plan, args.validated)
        print(f"Immutable output: {output_dir}")
        print(f"Output reused: {str(reused).lower()}")
        print(f"Immutable manifest SHA-256: {sha256_file(output_dir / 'immutable_manifest.json')}")
        print(f"Run fingerprint: {manifest['run_fingerprint_sha256']}")
        print("MATERIALIZE_STATUS=PASS")
        return 0
    except StabilizationError as exc:
        print(f"PRODUCT_PHASE_2L_ERROR={exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
