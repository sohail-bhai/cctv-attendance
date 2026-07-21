from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.face_attendance.mon_p3_retention_attribution import (
    RetentionAttributionError,
    resolve_inputs,
    run_retention_attribution,
    verify_attribution_output,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Phase 1.2K bounded MON_P3 retention attribution")
    value.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    value.add_argument(
        "--family-dir",
        type=Path,
        default=REPO_ROOT / "models" / "versions" / "embfam-7bc431a3ad762398d4e9",
    )
    value.add_argument(
        "--shadow-output",
        type=Path,
        default=REPO_ROOT / "attendance_output" / "shadow_validation" / "phase_1_2j" / "mon-p3-shadow-8136f24575f6d319fb96",
    )
    value.add_argument(
        "--evaluation-dir",
        type=Path,
        default=REPO_ROOT / "attendance_output" / "shadow_validation" / "phase_1_2j" / "mon-p3-shadow-8136f24575f6d319fb96" / "evaluation" / "mon-p3-evaluation-8f15934c693111852524",
    )
    value.add_argument("--preflight-only", action="store_true")
    value.add_argument("--verify-only", type=Path)
    return value


def main() -> int:
    args = parser().parse_args()
    try:
        if args.verify_only:
            print(json.dumps(verify_attribution_output(args.verify_only), indent=2))
            print("PHASE_1_2K_VERIFY_STATUS=PASS")
            return 0
        inputs = resolve_inputs(
            repo_root=args.repo_root,
            family_dir=args.family_dir,
            shadow_output=args.shadow_output,
            evaluation_dir=args.evaluation_dir,
        )
        print("Phase 1.2K - bounded MON_P3 retention attribution")
        print(f"Repository: {inputs.repo_root}")
        print(f"Family: {inputs.family_dir}")
        print(f"Phase 1.2J output: {inputs.shadow_output}")
        print(f"Evaluation: {inputs.evaluation_dir}")
        print(f"Attribution ID: {inputs.attribution_id}")
        print(f"Planned output: {inputs.output_dir}")
        print("Full MON_P3 recognition repeated: no")
        print("Manual review repeated: no")
        print("Production promotion allowed: no")
        if args.preflight_only:
            print("PREFLIGHT_STATUS=PASS")
            print("Attribution executed: no")
            return 0
        summary, output_dir, reused = run_retention_attribution(inputs)
        verified = verify_attribution_output(output_dir)
        print(f"Decision: {summary['decision']}")
        print(f"Reviewed tracks reproduced: {summary['reviewed_tracks_reproduced']}")
        print(f"Retention losses: {summary['retention_losses']}")
        print(f"Correct candidate recoveries: {summary['correct_candidate_recoveries']}")
        print(f"Idempotent reuse: {'yes' if reused else 'no'}")
        print(f"Output: {output_dir}")
        print(f"Verified files: {verified['verified_files']}")
        print("Production embeddings changed: no")
        print("Official attendance changed: no")
        print("Candidate promoted: no")
        print("Full MON_P3 recognition repeated: no")
        print("Manual review repeated: no")
        print(f"PHASE_1_2K_OUTPUT={output_dir}")
        return 0
    except (RetentionAttributionError, OSError, ValueError, RuntimeError) as exc:
        print(f"Phase 1.2K failed safely: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
