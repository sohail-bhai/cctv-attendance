from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.face_attendance.config import SFACE_MODEL, YUNET_MODEL
from src.face_attendance.cctv_same_track_recovery import (
    SameTrackRecoveryConfig,
    SameTrackRecoveryError,
    recover_same_track_cctv_evidence,
    verify_same_track_recovery,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recover technically usable embeddings only from observations belonging "
            "to already human-reviewed CCTV tracks. This is a diagnostic/adaptation "
            "artifact workflow and never changes production embeddings or attendance."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--family-dir", type=Path, required=True)
    parser.add_argument("--diagnostic-run-dir", type=Path, required=True)
    parser.add_argument("--camera-zones", type=Path, default=REPO_ROOT / "data" / "camera_zones.json")
    parser.add_argument("--production-embeddings", type=Path, default=REPO_ROOT / "models" / "student_embeddings.pkl")
    parser.add_argument("--production-summary", type=Path, default=REPO_ROOT / "models" / "embedding_summary.csv")
    parser.add_argument("--yunet-model", type=Path, default=YUNET_MODEL)
    parser.add_argument("--sface-model", type=Path, default=SFACE_MODEL)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "models" / "versions" / "recovery_runs")
    parser.add_argument("--target-session", default="2026-06-30__B51__P2__CVO")
    parser.add_argument("--max-observations-per-track", type=int, default=5)
    parser.add_argument("--max-embeddings-per-track", type=int, default=2)
    parser.add_argument("--min-match-iou", type=float, default=0.35)
    parser.add_argument("--min-recovered-similarity", type=float, default=0.25)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.verify_only:
            manifest = verify_same_track_recovery(args.family_dir)
            print(json.dumps(manifest, indent=2))
            return 0
        result = recover_same_track_cctv_evidence(
            repo_root=args.repo_root,
            family_dir=args.family_dir,
            diagnostic_run_dir=args.diagnostic_run_dir,
            camera_zones_path=args.camera_zones,
            production_embeddings_path=args.production_embeddings,
            production_summary_path=args.production_summary,
            yunet_model=args.yunet_model,
            sface_model=args.sface_model,
            output_root=args.output_root,
            config=SameTrackRecoveryConfig(
                target_session=args.target_session,
                max_observations_per_track=args.max_observations_per_track,
                max_embeddings_per_track=args.max_embeddings_per_track,
                min_match_iou=args.min_match_iou,
                min_recovered_similarity=args.min_recovered_similarity,
            ),
        )
    except SameTrackRecoveryError as exc:
        print(f"Same-track CCTV recovery failed: {exc}", file=sys.stderr)
        return 1
    print(f"Recovery ID: {result.recovery_id}")
    print(f"Recovery status: {result.status}")
    print(f"Recovery output: {result.output_dir}")
    print(f"Recovered tracks: {result.manifest.get('recovered_tracks', 0)}")
    print(f"Recovered identities: {len(result.manifest.get('recovered_identities', []))}")
    print(f"Selected embeddings: {result.manifest.get('selected_embedding_count', 0)}")
    print(f"Idempotent reuse: {'yes' if result.reused else 'no'}")
    print("Production embeddings changed: no")
    print("Candidate promoted: no")
    print("MON_P3 processed: no")
    print(f"RECOVERY_OUTPUT={result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
