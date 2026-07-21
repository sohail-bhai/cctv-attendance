from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.face_attendance.mon_p3_source_ablation import (
    SourceAblationError,
    resolve_inputs,
    run_source_ablation,
    verify_source_ablation_output,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Phase 1.2L exhaustive source-subset ablation for the five "
            "Phase 1.2K-attributed embedding sources"
        )
    )
    value.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    value.add_argument(
        "--family-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "models"
            / "versions"
            / "embfam-7bc431a3ad762398d4e9"
        ),
    )
    value.add_argument(
        "--shadow-output",
        type=Path,
        default=(
            REPO_ROOT
            / "attendance_output"
            / "shadow_validation"
            / "phase_1_2j"
            / "mon-p3-shadow-8136f24575f6d319fb96"
        ),
    )
    value.add_argument(
        "--evaluation-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "attendance_output"
            / "shadow_validation"
            / "phase_1_2j"
            / "mon-p3-shadow-8136f24575f6d319fb96"
            / "evaluation"
            / "mon-p3-evaluation-8f15934c693111852524"
        ),
    )
    value.add_argument(
        "--attribution-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "attendance_output"
            / "embedding_forensics"
            / "phase_1_2k"
            / "retention-attribution-3f10a2ef843d58e92ced"
        ),
    )
    value.add_argument(
        "--output-root",
        type=Path,
        help="Optional Phase 1.2L output parent. The deterministic ablation ID is appended.",
    )
    value.add_argument("--preflight-only", action="store_true")
    value.add_argument("--verify-only", type=Path)
    return value


def main() -> int:
    args = parser().parse_args()
    try:
        if args.verify_only:
            verified = verify_source_ablation_output(args.verify_only)
            print(json.dumps(verified, indent=2))
            print("PHASE_1_2L_VERIFY_STATUS=PASS")
            return 0

        inputs = resolve_inputs(
            repo_root=args.repo_root,
            family_dir=args.family_dir,
            shadow_output=args.shadow_output,
            evaluation_dir=args.evaluation_dir,
            attribution_dir=args.attribution_dir,
            output_root=args.output_root,
        )
        print("Phase 1.2L - exhaustive attributed-source subset ablation")
        print(f"Repository: {inputs.repo_root}")
        print(f"Rejected family: {inputs.family_dir}")
        print(f"Phase 1.2J output: {inputs.shadow_output}")
        print(f"Phase 1.2J evaluation: {inputs.evaluation_dir}")
        print(f"Phase 1.2K attribution: {inputs.attribution_dir}")
        print(f"Ablation ID: {inputs.ablation_id}")
        print(f"Planned output: {inputs.output_dir}")
        print("Attributed sources tested: 5")
        print("Source subsets tested: 32")
        print("Frozen benchmark rows reused: 120")
        print("Reviewed MON_P3 exception tracks reused: 15")
        print("Full MON_P3 recognition repeated: no")
        print("Manual review repeated: no")
        print("New embedding family built: no")
        print("Production promotion allowed: no")
        if args.preflight_only:
            print("PREFLIGHT_STATUS=PASS")
            print("Source ablation executed: no")
            print("MON_P3 video frames decoded: no")
            return 0

        verified, output_dir, reused = run_source_ablation(inputs)
        print(f"Decision: {verified['decision']}")
        print(
            "Hard-gate candidate compositions: "
            f"{verified['hard_gate_candidate_count']}"
        )
        print(
            "Recommended configuration: "
            f"{verified['recommended_config_id'] or 'none'}"
        )
        print(f"Idempotent reuse: {'yes' if reused else 'no'}")
        print(f"Output: {output_dir}")
        print(f"Verified files: {verified['verified_files']}")
        print("Production embeddings changed: no")
        print("Official attendance changed: no")
        print("New embedding family built: no")
        print("Candidate promoted: no")
        print("Full MON_P3 recognition repeated: no")
        print("Manual review repeated: no")
        print("MON_P3 remains design evidence, not a future untouched promotion session: yes")
        print(f"PHASE_1_2L_OUTPUT={output_dir}")
        return 0
    except (SourceAblationError, OSError, ValueError, RuntimeError) as exc:
        print(f"Phase 1.2L failed safely: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
