from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.face_attendance.phase_1_2m_candidate_family import (
    Phase12MError,
    Phase12MInputs,
    build_phase_1_2m,
    preflight_phase_1_2m,
    validate_built_family_against_abl16,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Phase 1.2M immutable candidate-family build from the corrected "
            "Phase 1.2L.1 ABL-16 source composition"
        )
    )
    value.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    value.add_argument("--forensic-run", type=Path, required=True)
    value.add_argument("--enrollment-review-package", type=Path, required=True)
    value.add_argument("--enrollment-approvals", type=Path, required=True)
    value.add_argument("--cctv-review-package", type=Path, required=True)
    value.add_argument("--cctv-approvals", type=Path, required=True)
    value.add_argument("--production-embeddings", type=Path, required=True)
    value.add_argument("--production-summary", type=Path, required=True)
    value.add_argument("--student-map", type=Path, required=True)
    value.add_argument("--dataset", type=Path, required=True)
    value.add_argument("--augmented-dataset", type=Path, required=True)
    value.add_argument("--versions-root", type=Path, required=True)
    value.add_argument("--recovery-dir", type=Path, required=True)
    value.add_argument("--source-ablation-dir", type=Path, required=True)
    value.add_argument("--selection-audit-dir", type=Path, required=True)
    value.add_argument("--parent-family-dir", type=Path, required=True)
    value.add_argument("--family-id", default="")
    value.add_argument("--preflight-only", action="store_true")
    value.add_argument("--verify-family", type=Path)
    return value


def _inputs(args: argparse.Namespace) -> Phase12MInputs:
    return Phase12MInputs(
        repo_root=args.repo_root.resolve(),
        forensic_run_dir=args.forensic_run.resolve(),
        enrollment_review_package=args.enrollment_review_package.resolve(),
        enrollment_approvals_path=args.enrollment_approvals.resolve(),
        cctv_review_package=args.cctv_review_package.resolve(),
        cctv_approvals_path=args.cctv_approvals.resolve(),
        production_embeddings_path=args.production_embeddings.resolve(),
        production_summary_path=args.production_summary.resolve(),
        student_map_path=args.student_map.resolve(),
        dataset_root=args.dataset.resolve(),
        augmented_dataset_root=args.augmented_dataset.resolve(),
        versions_root=args.versions_root.resolve(),
        recovery_dir=args.recovery_dir.resolve(),
        source_ablation_dir=args.source_ablation_dir.resolve(),
        selection_audit_dir=args.selection_audit_dir.resolve(),
        parent_family_dir=args.parent_family_dir.resolve(),
    )


def main() -> int:
    args = parser().parse_args()
    try:
        inputs = _inputs(args)
        if args.verify_family:
            summary, output_dir, reused = validate_built_family_against_abl16(
                inputs=inputs,
                family_dir=args.verify_family,
            )
            print(json.dumps(summary, indent=2))
            print(f"Idempotent validation reuse: {'yes' if reused else 'no'}")
            print(f"PHASE_1_2M_VALIDATION_OUTPUT={output_dir}")
            print("PHASE_1_2M_VERIFY_STATUS=PASS")
            return 0

        preflight = preflight_phase_1_2m(inputs=inputs)
        print("Phase 1.2M - corrected immutable candidate-family workflow")
        print(f"Repository: {inputs.repo_root}")
        print(f"Parent family: {inputs.parent_family_dir}")
        print(f"Source ablation: {inputs.source_ablation_dir}")
        print(f"Selection audit: {inputs.selection_audit_dir}")
        print("Corrected composition: ABL-16-ee3a4ff6")
        print("Kept implicated sources: 1")
        print("Removed implicated sources: 4")
        print("Full MON_P3 recognition repeated: no")
        print("Manual review repeated: no")
        print("Production promotion allowed: no")
        print("New untouched session required: yes")
        if args.preflight_only:
            print(json.dumps(preflight, indent=2))
            print("Candidate family built: no")
            print("PREFLIGHT_STATUS=PASS")
            return 0

        result, summary, validation_dir, validation_reused = build_phase_1_2m(
            inputs=inputs,
            family_id=args.family_id,
            launcher_source=Path(__file__).with_suffix(".ps1"),
            progress=print,
        )
        print(f"Decision: {result.evaluation_decision['decision']}")
        print(f"Family ID: {result.family_id}")
        print(f"Family output: {result.family_dir}")
        print(f"Family idempotent reuse: {'yes' if result.idempotent_reuse else 'no'}")
        print(f"Validation output: {validation_dir}")
        print(f"Validation idempotent reuse: {'yes' if validation_reused else 'no'}")
        print(f"Variant records: {summary['variant_record_counts']}")
        print("Production embeddings changed: no")
        print("Official attendance changed: no")
        print("Candidate promoted: no")
        print("Full MON_P3 recognition repeated: no")
        print("Manual review repeated: no")
        print("Next action: reserve a different untouched CVO validation session.")
        print(f"FAMILY_OUTPUT={result.family_dir}")
        print(f"PHASE_1_2M_VALIDATION_OUTPUT={validation_dir}")
        return 0
    except (Phase12MError, OSError, ValueError, RuntimeError) as exc:
        print(f"Phase 1.2M failed safely: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
