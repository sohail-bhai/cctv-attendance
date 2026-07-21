from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from src.face_attendance.phase_1_2o_explicit_promotion import (
    DEFAULT_CONTRACT,
    POLICY_VERSION,
    Phase12OPromotionError,
    PromotionInputs,
    promote_candidate_family,
    rollback_promotion,
    verify_promoted_state,
    verify_promotion_evidence,
    write_preflight,
)

DEFAULT_REPO = Path(r"F:\sohail\Class_Attendance_YuNet_SFace")
DEFAULT_LABELS = Path(
    r"D:\Downloads\tracklet_review_labels_mon-p4-shadow-695277fbd0dcd9d431e1-20260717152714-915b9c.csv"
)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 1.2O explicit candidate-family promotion with immutable evidence and rollback."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    preflight = sub.add_parser("preflight", help="Verify all promotion evidence without changing production.")
    _common(preflight)
    preflight.add_argument("--output", type=Path, required=True)

    promote = sub.add_parser("promote", help="Explicitly promote the verified full-candidate variant.")
    _common(promote)
    promote.add_argument("--confirm-family-id", required=True)

    verify = sub.add_parser("verify", help="Verify an already completed promotion and its rollback backup.")
    verify.add_argument("--repo-root", type=Path, default=DEFAULT_REPO)
    verify.add_argument("--promotion-dir", type=Path, required=True)

    rollback = sub.add_parser("rollback", help="Restore the pre-promotion production files.")
    rollback.add_argument("--repo-root", type=Path, default=DEFAULT_REPO)
    rollback.add_argument("--promotion-dir", type=Path, required=True)
    rollback.add_argument("--confirm-promotion-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "preflight":
            inputs = PromotionInputs.for_repo(args.repo_root, args.labels)
            result = verify_promotion_evidence(inputs)
            output = write_preflight(result, args.output)
            print("Phase 1.2O explicit-promotion preflight")
            print(f"Repository: {inputs.repo_root}")
            print(f"Family: {result.family_id}")
            print(f"Candidate variant: {result.candidate_variant_id}")
            print(f"MON_P4 evaluation: {result.evaluation_decision}")
            print(f"Reviewed exception tracklets: {result.reviewed_exception_tracks}")
            print(f"Correct candidate recoveries: {result.candidate_correct_recoveries}")
            print(
                "Candidate false identities / unsafe accepts: "
                f"{result.candidate_false_identities_or_unsafe_accepts}"
            )
            print(
                "Candidate lost correct production accepts: "
                f"{result.candidate_lost_correct_production_accepts}"
            )
            print(f"Unverifiable exceptions: {result.unverifiable_exceptions}")
            print(f"Promotion ID: {result.promotion_id}")
            print("Production embeddings changed: no")
            print("Production summary changed: no")
            print("Official attendance changed: no")
            print("Candidate promoted: no")
            print(f"PREFLIGHT_OUTPUT={output}")
            print("PREFLIGHT_STATUS=PASS")
            return 0

        if args.command == "promote":
            inputs = PromotionInputs.for_repo(args.repo_root, args.labels)
            result = promote_candidate_family(
                inputs,
                confirm_family_id=args.confirm_family_id,
            )
            print("Phase 1.2O explicit promotion completed safely.")
            print(f"Promotion ID: {result.promotion_id}")
            print(f"Promotion output: {result.promotion_dir}")
            print(f"Family ID: {result.family_id}")
            print(f"Candidate variant: {result.candidate_variant_id}")
            print(f"Production embeddings SHA-256: {result.production_embeddings_sha256}")
            print(f"Production summary SHA-256: {result.production_summary_sha256}")
            print(f"Idempotent reuse: {'yes' if result.idempotent_reuse else 'no'}")
            print("Official attendance changed: no")
            print("Recognition executed: no")
            print("Datasets changed: no")
            print("Candidate promoted: yes")
            print(f"Rollback command: {result.rollback_command}")
            print(f"PHASE_1_2O_PROMOTION={result.promotion_dir}")
            return 0

        if args.command == "verify":
            result = verify_promoted_state(
                repo_root=args.repo_root,
                promotion_dir=args.promotion_dir,
            )
            print("Phase 1.2O promoted-state verification passed.")
            print(f"Promotion ID: {result.promotion_id}")
            print(f"Promotion output: {result.promotion_dir}")
            print(f"Family ID: {result.family_id}")
            print(f"Candidate variant: {result.candidate_variant_id}")
            print("Production candidate hashes: verified")
            print("Rollback backup hashes: verified")
            print("Promotion manifest: verified")
            print("PHASE_1_2O_VERIFY_STATUS=PASS")
            return 0

        if args.command == "rollback":
            result = rollback_promotion(
                repo_root=args.repo_root,
                promotion_dir=args.promotion_dir,
                confirm_promotion_id=args.confirm_promotion_id,
            )
            print("Phase 1.2O rollback completed safely.")
            print(f"Promotion ID: {result.promotion_id}")
            print(f"Promotion output: {result.promotion_dir}")
            print(f"Restored embeddings SHA-256: {result.restored_embeddings_sha256}")
            print(f"Restored summary SHA-256: {result.restored_summary_sha256}")
            print(f"Idempotent reuse: {'yes' if result.idempotent_reuse else 'no'}")
            print("Official attendance changed: no")
            print("Recognition executed: no")
            print("Datasets changed: no")
            print("Candidate promoted: no (rolled back)")
            print("PHASE_1_2O_ROLLBACK_STATUS=PASS")
            return 0

        raise AssertionError(f"Unhandled command: {args.command}")
    except Phase12OPromotionError as exc:
        print(f"PHASE_1_2O_STATUS=FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
