from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.face_attendance.processing_integration import (  # noqa: E402
    POLICY_VERSION,
    PROCESSING_MODE,
    ProcessingIntegrationError,
    filter_embedding_records_to_roster,
    fixed_processing_policy,
    inspect_exact_checkpoint_layout,
    load_subject_roster,
)
from src.face_attendance.zones import load_camera_zone_config  # noqa: E402

EXPECTED_FAMILY_ID = "embfam-274b5207b8b71294ff75"
EXPECTED_CVO_ROSTER_COUNT = 27
EXPECTED_CVO_EMBEDDING_COVERAGE = 26
EXPECTED_PRODUCTION_RECORDS = 354


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProcessingIntegrationError(f"Invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ProcessingIntegrationError(f"JSON root must be an object: {path}")
    return payload


def _load_embedding_payload(path: Path) -> dict:
    if not path.is_file():
        raise ProcessingIntegrationError(f"Production embeddings not found: {path}")
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        raise ProcessingIntegrationError(f"Could not load production embeddings: {path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ProcessingIntegrationError("Production embedding payload is invalid")
    return payload


def _discover_valid_sources(repo_root: Path) -> list[dict]:
    video_root = repo_root / "cctv_videos"
    candidates: list[Path] = []
    prepared_root = video_root / "prepared_slots"
    if prepared_root.is_dir():
        for date_dir in sorted(path for path in prepared_root.iterdir() if path.is_dir()):
            candidates.extend(sorted(path for path in date_dir.iterdir() if path.is_dir()))
    if video_root.is_dir():
        candidates.extend(
            sorted(
                path
                for path in video_root.iterdir()
                if path.is_dir() and path.name.lower() != "prepared_slots"
            )
        )

    output: list[dict] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = str(candidate.resolve()).lower()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            layout = inspect_exact_checkpoint_layout(candidate)
        except ProcessingIntegrationError:
            continue
        try:
            relative = candidate.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            relative = str(candidate.resolve())
        output.append({"path": relative, **layout.to_public_dict()})
    return output


def run_preflight(repo_root: Path) -> dict:
    repo_root = Path(repo_root).resolve()
    required = {
        "app": repo_root / "app.py",
        "processor": repo_root / "scripts" / "mark_attendance_checkpoints.py",
        "frontend": repo_root / "frontend" / "src" / "pages" / "Timetable.jsx",
        "student_map": repo_root / "data" / "student_faculty_map.json",
        "camera_zones": repo_root / "data" / "camera_zones.json",
        "embeddings": repo_root / "models" / "student_embeddings.pkl",
        "summary": repo_root / "models" / "embedding_summary.csv",
        "version_pointer": repo_root / "models" / "current_embedding_version.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise ProcessingIntegrationError("Missing required files: " + ", ".join(missing))

    pointer = _read_json(required["version_pointer"])
    family_id = str(pointer.get("family_id") or "")
    if family_id != EXPECTED_FAMILY_ID or str(pointer.get("status") or "") != "promoted":
        raise ProcessingIntegrationError(
            f"Production pointer is not the verified promoted family: status={pointer.get('status')!r}, family={family_id!r}"
        )

    roster = load_subject_roster(required["student_map"], "CVO")
    if roster.count != EXPECTED_CVO_ROSTER_COUNT:
        raise ProcessingIntegrationError(
            f"CVO roster count changed: {roster.count}; expected {EXPECTED_CVO_ROSTER_COUNT}"
        )
    if "2401100CSE0110" in roster.rolls and "24011CSEAI0110" in roster.rolls:
        # Both may exist in other contexts, but the known CVO roster must contain only AI0110.
        raise ProcessingIntegrationError("CVO roster incorrectly contains both CSE0110 and AI0110")
    if "24011CSEAI0110" not in roster.rolls:
        raise ProcessingIntegrationError("Corrected AI0110 identity is missing from CVO roster")

    embedding_payload = _load_embedding_payload(required["embeddings"])
    records = embedding_payload["records"]
    if len(records) != EXPECTED_PRODUCTION_RECORDS:
        raise ProcessingIntegrationError(
            f"Production embedding record count changed: {len(records)}; expected {EXPECTED_PRODUCTION_RECORDS}"
        )
    filtered, missing_rolls = filter_embedding_records_to_roster(records, roster.rolls)
    covered = sorted({str(row.get("roll_no")) for row in filtered})
    if len(covered) != EXPECTED_CVO_EMBEDDING_COVERAGE:
        raise ProcessingIntegrationError(
            f"CVO embedding coverage changed: {len(covered)}; expected {EXPECTED_CVO_EMBEDDING_COVERAGE}"
        )
    if missing_rolls != ["2401100CSE0268"]:
        raise ProcessingIntegrationError(
            f"Unexpected CVO missing-embedding set: {missing_rolls}"
        )

    zone_config = load_camera_zone_config(required["camera_zones"])
    if not {"front", "back"}.issubset(set(zone_config.profiles)):
        raise ProcessingIntegrationError("Camera zone configuration lacks front/back profiles")

    policy = fixed_processing_policy({"processing_mode": PROCESSING_MODE})
    valid_sources = _discover_valid_sources(repo_root)
    if not valid_sources:
        raise ProcessingIntegrationError(
            "No exact five-checkpoint front/back source is available for the controlled browser test"
        )

    app_text = required["app"].read_text(encoding="utf-8")
    processor_text = required["processor"].read_text(encoding="utf-8")
    frontend_text = required["frontend"].read_text(encoding="utf-8")
    source_contracts = {
        "app_uses_shared_command_builder": "build_processing_command(" in app_text,
        "app_requires_roster_complete_output": "Product Phase 2G processor did not produce the authoritative roster" in app_text,
        "processor_requires_authoritative_roster": "--require-authoritative-roster" in processor_text,
        "frontend_uses_quality_aware_mode": "quality_aware_production_v1" in frontend_text,
        "frontend_does_not_send_match_threshold": "match_threshold" not in frontend_text.split("const CONTROLLED_PROCESSING_PAYLOAD", 1)[1].split("};", 1)[0],
    }
    failed_contracts = [key for key, passed in source_contracts.items() if not passed]
    if failed_contracts:
        raise ProcessingIntegrationError(
            "Phase 2G source wiring is incomplete: " + ", ".join(failed_contracts)
        )

    return {
        "status": "PASS",
        "policy_version": POLICY_VERSION,
        "processing_mode": PROCESSING_MODE,
        "production_family": family_id,
        "production_embedding_records": len(records),
        "cvo_roster_count": roster.count,
        "cvo_embedding_coverage": len(covered),
        "cvo_missing_embeddings": missing_rolls,
        "official_recognition_authority": policy["official_recognition_authority"],
        "zone_mode": policy["zone_mode"],
        "tracklet_mode": policy["tracklet_mode"],
        "tracklets_used_for_official_attendance": policy["tracklets_used_for_official_attendance"],
        "valid_controlled_sources": valid_sources,
        "source_contracts": source_contracts,
        "recognition_executed": False,
        "attendance_written": False,
        "production_embeddings_changed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only Product Phase 2G processing-integration preflight."
    )
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run_preflight(Path(args.repo_root))
    except ProcessingIntegrationError as exc:
        print(f"PREFLIGHT_STATUS=FAIL\n{exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("Product Phase 2G - quality-aware browser processing preflight")
        print(f"Repository: {Path(args.repo_root).resolve()}")
        print(f"Production family: {result['production_family']}")
        print(f"CVO roster / embedding coverage: {result['cvo_roster_count']} / {result['cvo_embedding_coverage']}")
        print(f"Missing embedding: {', '.join(result['cvo_missing_embeddings'])}")
        print(f"Official authority: {result['official_recognition_authority']}")
        print(f"Zone / tracklet modes: {result['zone_mode']} / {result['tracklet_mode']}")
        print("Tracklets mark official attendance: no")
        print(f"Valid controlled sources: {len(result['valid_controlled_sources'])}")
        for source in result["valid_controlled_sources"]:
            print(f"  - {source['path']} ({source['checkpoint_folder_count']} checkpoints, {source['recursive_video_count']} videos)")
        print("Recognition executed: no")
        print("Attendance written: no")
        print("Production embeddings changed: no")
        print("PREFLIGHT_STATUS=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
