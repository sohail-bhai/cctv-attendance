from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.face_attendance.product_phase_2l_authentication import (  # noqa: E402
    AuthenticationHardeningError,
    REQUIRED_EXTERNAL_VALIDATIONS,
    materialize,
    preflight,
    protected_state_hashes,
    sha256_file,
    verify_immutable_output,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Verify Product Phase 2L-C explicit authentication and create immutable evidence.")
    result.add_argument("--repo-root", type=Path, default=ROOT)
    result.add_argument("--output-root", type=Path)
    commands = result.add_subparsers(dest="command")
    commands.add_parser("preflight")
    build = commands.add_parser("materialize")
    build.add_argument("--frontend-build-dir", type=Path, required=True)
    build.add_argument("--validated", action="append", default=[], choices=REQUIRED_EXTERNAL_VALIDATIONS)
    verify = commands.add_parser("verify")
    verify.add_argument("--output-dir", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    command = args.command or "preflight"
    try:
        if command == "verify":
            manifest = verify_immutable_output(args.output_dir)
            print(f"Verified Product Phase 2L-C output: {Path(args.output_dir)}")
            print(f"Run ID: {manifest['run_id']}")
            print(f"Immutable manifest SHA-256: {sha256_file(Path(args.output_dir) / 'immutable_manifest.json')}")
            print("VERIFY_STATUS=PASS")
            return 0
        plan = preflight(args.repo_root, args.output_root)
        print(f"Protected routes: {len(plan.protected_rows)}")
        print(f"No-token checks: {len(plan.no_token_rows)}")
        print(f"Synthetic role checks: {len(plan.role_rows)}")
        print("Recognition executed: false")
        print("Operational hash drift: false")
        if command == "preflight":
            print("PREFLIGHT_STATUS=PASS")
            return 0
        before = protected_state_hashes(plan.repo_root)
        output_dir, reused, manifest = materialize(plan, args.validated, args.frontend_build_dir)
        after = protected_state_hashes(plan.repo_root)
        if before != after:
            raise AuthenticationHardeningError("operational_hash_drift_during_materialization")
        print(f"Immutable output: {output_dir}")
        print(f"Output reused: {str(reused).lower()}")
        print(f"Run ID: {manifest['run_id']}")
        print(f"Immutable manifest SHA-256: {sha256_file(output_dir / 'immutable_manifest.json')}")
        print("MATERIALIZE_STATUS=PASS")
        return 0
    except AuthenticationHardeningError as exc:
        print(f"PRODUCT_PHASE_2L_C_ERROR={exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
