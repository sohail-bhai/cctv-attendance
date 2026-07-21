from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.face_attendance.product_phase_2k_carry_forward import (  # noqa: E402
    CarryForwardContractError,
    materialize,
    preflight,
    validate_capture_manifest,
    verify_immutable_output,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify Product Phase 2K-B exact review carry-forward provenance and prepare "
            "the Phase 2K-C untouched-session contract without decoding video or running recognition."
        )
    )
    parser.add_argument("--repo-root", default=str(ROOT))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight", help="Verify immutable inputs, protected state, and candidate-session provenance.")
    sub.add_parser("materialize", help="Create or byte-verify the deterministic immutable Phase 2K-B output.")
    verify = sub.add_parser("verify", help="Verify an existing Phase 2K-B immutable output.")
    verify.add_argument("--output-dir", required=True)
    capture = sub.add_parser("validate-capture", help="Hash-validate a future Phase 2K-C capture package without decoding video.")
    capture.add_argument("--package-dir", required=True)
    capture.add_argument("--manifest", default="capture_manifest.json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "verify":
            manifest = verify_immutable_output(Path(args.output_dir))
            print(f"Verified Phase 2K-B output: {args.output_dir}")
            print(f"Run ID: {manifest['run_id']}")
            print("VERIFY_STATUS=PASS")
            return 0
        if args.command == "validate-capture":
            package = Path(args.package_dir).resolve()
            manifest_path = package / args.manifest
            payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            result = validate_capture_manifest(payload, package_root=package, verify_files=True)
            print(f"Validated untouched capture package: {result['session_id']}")
            print("Video decoded: no")
            print("Recognition ran: no")
            print("CAPTURE_VALIDATION_STATUS=PASS")
            return 0
        result = preflight(Path(args.repo_root))
        if args.command == "preflight":
            print("Product Phase 2K-B exact carry-forward preflight")
            print(f"Run ID: {result.run_id}")
            print(f"Candidate sessions audited: {len(result.candidate_inventory)}")
            print("Existing untouched sessions: 0")
            print("Recognition ran: no")
            print("Video decoded/processed: no")
            print("Operational state changed: no")
            print("PREFLIGHT_STATUS=PASS")
            return 0
        output, reused, evaluation = materialize(result)
        print("Product Phase 2K-B materialized safely.")
        print(f"Output: {output}")
        print(f"Idempotent reuse: {'yes' if reused else 'no'}")
        print(f"Exact safe review pairs: {evaluation['diagnostically_exact_match_eligible_pairs']}")
        print(f"Rejected review pairs: {evaluation['diagnostically_rejected_pairs']}")
        print("Existing untouched sessions: 0")
        print("Recognition ran: no")
        print("Video decoded/processed: no")
        print("Operational state changed: no")
        print(f"PRODUCT_PHASE_2K_B_OUTPUT={output}")
        return 0
    except (CarryForwardContractError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Product Phase 2K-B stopped safely: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
