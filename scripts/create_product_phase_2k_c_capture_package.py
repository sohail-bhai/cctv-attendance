from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.face_attendance.product_phase_2k_capture_intake import (  # noqa: E402
    CONFIRMATION_TOKEN,
    CaptureIntakeError,
    CaptureRequest,
    build_source_specs,
    dry_run,
    materialize_capture_package,
    materialize_tooling_validation,
    plan_capture,
    verify_capture_package,
    verify_tooling_validation,
)


def _add_create_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source-dir")
    group.add_argument(
        "--source",
        action="append",
        default=[],
        help="Repeat exactly ten times as CP1/back=<absolute-path> through CP5/front=<absolute-path>.",
    )
    parser.add_argument(
        "--capture-timestamp",
        action="append",
        default=[],
        help="Repeat exactly ten times as CP1/back=<ISO-8601-with-offset> through CP5/front=<value>.",
    )
    parser.add_argument("--session-date", required=True)
    parser.add_argument("--section", required=True)
    parser.add_argument("--period", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--room", required=True)
    parser.add_argument("--confirm", required=True, help=f"Required exact token: {CONFIRMATION_TOKEN}")
    parser.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create or verify an immutable Product Phase 2K-C untouched-session source freeze. "
            "This tool reads bytes and container headers only; it does not decode frames or run recognition."
        )
    )
    parser.add_argument("--repo-root", default=str(ROOT))
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create", help="Dry-run or atomically materialize a source-freeze package.")
    _add_create_arguments(create)
    verify = sub.add_parser("verify", help="Verify an immutable capture package and canonical Phase 2K-B contract.")
    verify.add_argument("--package-dir", required=True)
    tooling = sub.add_parser("tooling-validate", help="Materialize the deterministic no-video tooling validation artifact.")
    tooling.add_argument("--output-root")
    tooling_verify = sub.add_parser("tooling-verify", help="Verify a tooling validation artifact.")
    tooling_verify.add_argument("--output-dir", required=True)
    return parser


def _print(payload: dict) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path(args.repo_root).resolve()
    try:
        if args.command == "verify":
            _print(verify_capture_package(repo_root, Path(args.package_dir)))
            return 0
        if args.command == "tooling-validate":
            output, reused = materialize_tooling_validation(
                repo_root,
                Path(args.output_root) if args.output_root else None,
            )
            _print(
                {
                    "valid": True,
                    "output": str(output),
                    "idempotent_reuse": reused,
                    "real_capture_package_created": False,
                    "frames_decoded": False,
                    "recognition_ran": False,
                }
            )
            return 0
        if args.command == "tooling-verify":
            manifest = verify_tooling_validation(Path(args.output_dir))
            _print({"valid": True, "tooling_run_id": manifest["tooling_run_id"]})
            return 0
        sources = build_source_specs(
            source_dir=Path(args.source_dir) if args.source_dir else None,
            source_values=args.source,
            capture_timestamp_values=args.capture_timestamp,
        )
        request = CaptureRequest(
            repo_root=repo_root,
            output_root=Path(args.output_root),
            session_date=args.session_date,
            section=args.section,
            period=args.period,
            subject=args.subject,
            room=args.room,
            sources=sources,
            confirmation_token=args.confirm,
        )
        plan = plan_capture(request)
        if args.dry_run:
            _print(dry_run(plan))
            return 0
        output, reused = materialize_capture_package(plan)
        result = verify_capture_package(repo_root, output)
        result.update({"output": str(output), "idempotent_reuse": reused})
        _print(result)
        return 0
    except CaptureIntakeError as exc:
        _print(
            {
                "valid": False,
                "reason_code": exc.reason_code,
                "outcome": exc.outcome,
                "message": str(exc),
                "frames_decoded": False,
                "recognition_ran": False,
            }
        )
        return 2
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        _print(
            {
                "valid": False,
                "reason_code": "unverifiable_provenance",
                "outcome": "unverifiable_provenance",
                "message": str(exc),
                "frames_decoded": False,
                "recognition_ran": False,
            }
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
