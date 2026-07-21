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
    create_source_ablation_selection_audit,
    inspect_source_ablation_selection,
    verify_source_ablation_selection_audit,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Phase 1.2L.1 immutable promotion-evidence/parsimony selection audit "
            "for the completed Phase 1.2L source-subset experiment"
        )
    )
    value.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    value.add_argument(
        "--source-output",
        type=Path,
        help="Optional exact Phase 1.2L output directory.",
    )
    value.add_argument("--preflight-only", action="store_true")
    value.add_argument("--verify-only", type=Path)
    return value


def main() -> int:
    args = parser().parse_args()
    try:
        if args.verify_only:
            verified = verify_source_ablation_selection_audit(args.verify_only)
            print(json.dumps(verified, indent=2))
            print("PHASE_1_2L_1_VERIFY_STATUS=PASS")
            return 0

        inspected = inspect_source_ablation_selection(
            repo_root=args.repo_root,
            source_output=args.source_output,
        )
        legacy = inspected["legacy"]
        corrected = inspected["corrected"]
        print("Phase 1.2L.1 - promotion-evidence/parsimony selection audit")
        print(f"Repository: {inspected['repo_root']}")
        print(f"Source Phase 1.2L output: {inspected['source_output']}")
        print(f"Selection audit ID: {inspected['audit_id']}")
        print(f"Planned output: {inspected['output_dir']}")
        print(f"Legacy recommendation: {legacy['Config_ID']}")
        print(f"Corrected recommendation: {corrected['Config_ID']}")
        print(
            "Added implicated sources: "
            f"{legacy['Kept_Source_Count']} -> {corrected['Kept_Source_Count']}"
        )
        print("Promotion-eligible metrics changed: no")
        print("Unsafe accepts changed: no")
        print("Recognition executed: no")
        print("Source ablation rescored: no")
        print("New embedding family built: no")
        print("Production promotion allowed: no")
        if args.preflight_only:
            print("PREFLIGHT_STATUS=PASS")
            print("Selection audit written: no")
            return 0

        verified, output_dir, reused = create_source_ablation_selection_audit(
            repo_root=args.repo_root,
            source_output=args.source_output,
            status=print,
        )
        print("Decision: legacy_recommendation_superseded_by_promotion_evidence_parsimony")
        print(
            "Corrected recommended configuration: "
            f"{verified['corrected_recommended_config_id']}"
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
        print(f"PHASE_1_2L_1_OUTPUT={output_dir}")
        return 0
    except (SourceAblationError, OSError, ValueError, RuntimeError) as exc:
        print(f"Phase 1.2L.1 failed safely: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
