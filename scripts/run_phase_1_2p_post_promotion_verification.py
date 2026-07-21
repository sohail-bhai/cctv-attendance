from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.face_attendance.phase_1_2p_post_promotion_verification import (
    Phase12PError,
    Phase12PInputs,
    preflight_phase_1_2p,
    run_phase_1_2p,
    write_preflight,
)

DEFAULT_REPO = Path(r"F:\sohail\Class_Attendance_YuNet_SFace")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 1.2P read-only post-promotion verification of production, "
            "frozen recognition fingerprints, and rollback readiness."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    preflight = sub.add_parser("preflight", help="Verify inputs without scoring or writing a final output.")
    preflight.add_argument("--repo-root", type=Path, default=DEFAULT_REPO)
    preflight.add_argument("--output", type=Path, required=True)

    run = sub.add_parser("run", help="Execute the read-only post-promotion verification.")
    run.add_argument("--repo-root", type=Path, default=DEFAULT_REPO)
    run.add_argument("--output-root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        inputs = Phase12PInputs.for_repo(args.repo_root)
        if args.command == "preflight":
            result = preflight_phase_1_2p(inputs)
            output = write_preflight(result, args.output)
            print("Phase 1.2P post-promotion verification preflight")
            print(f"Repository: {inputs.repo_root}")
            print(f"Promotion: {result.promotion_id}")
            print(f"Production family: {result.family_id}")
            print(f"Candidate variant: {result.candidate_variant_id}")
            print(f"Production embedding records: {result.production_embedding_records}")
            print(f"Production embedding dimension: {result.production_embedding_dimension}")
            print(f"Planned frozen benchmark rows: {result.benchmark_rows_planned}")
            print(f"Planned MON_P3 rows: {result.mon_p3_rows_planned}")
            print("Rollback backup: verified")
            print("Production embeddings changed: no")
            print("Production summary changed: no")
            print("Official attendance changed: no")
            print("Recognition executed: no")
            print(f"PREFLIGHT_OUTPUT={output}")
            print("PREFLIGHT_STATUS=PASS")
            return 0

        result = run_phase_1_2p(
            inputs=inputs,
            output_root=args.output_root,
        )
        print("Phase 1.2P post-promotion operational verification passed.")
        print(f"Verification ID: {result.verification_id}")
        print(f"Output: {result.output_dir}")
        print(f"Production embedding records: {result.production_embedding_records}")
        print(f"Frozen benchmark rows reproduced: {result.benchmark_rows_reproduced}")
        print(f"MON_P3 rows reproduced: {result.mon_p3_rows_reproduced}")
        print(f"Idempotent reuse: {'yes' if result.idempotent_reuse else 'no'}")
        print("Production embeddings changed: no")
        print("Production summary changed: no")
        print("Official attendance changed: no")
        print("Recognition executed: no")
        print("Datasets changed: no")
        print("Rollback backup: verified")
        print(f"Rollback command: {result.rollback_command}")
        print(f"PHASE_1_2P_OUTPUT={result.output_dir}")
        print("PHASE_1_2P_STATUS=PASS")
        return 0
    except Phase12PError as exc:
        print(f"PHASE_1_2P_STATUS=FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
