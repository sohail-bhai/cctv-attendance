from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .product_phase_2h_multiframe_recovery import (
    DIAGNOSTIC_HASHES as PHASE_2G_DIAGNOSTIC_HASHES,
    RUN_ID as PHASE_2G_RUN_ID,
    default_inputs as phase_2h_default_inputs,
    preflight as phase_2h_preflight,
    verify_output as verify_phase_2h_output,
)
from .product_phase_2i_authority import (
    AUTHORITY_REVISION_ID,
    AUTOMATIC_POLICY_VERSION,
    MIXED_TRACKLET_ID,
    PHASE_2H_OUTPUT_ID,
    SESSION_ID,
    verify_authority_output,
)
from .report_revision_control import verify_revision_output
from .review_carry_forward import (
    _evidence_signature,
    load_registry,
    source_fingerprint,
)
from .tracklet_purity import (
    AMBIGUOUS_QUARANTINED,
    INVALID_EMBEDDING_EVIDENCE,
    MIXED_QUARANTINED,
    POLICY_VERSION,
    PURE_UNSPLIT,
    SPLIT_CANDIDATE,
    PurityObservation,
    PurityPolicy,
    TrackPurityResult,
    TrackSummaryEvidence,
    TrackletPurityError,
    analyze_track_purity,
    canonical_json_bytes,
    canonical_json_hash,
    csv_bytes,
    review_carry_forward_decision,
    sha256_file,
    verify_immutable_output,
    write_immutable_output,
)


PHASE_2I_DIAGNOSTIC_RUN_ID = "product-2i-2026-06-22__B51__P4__CVO-59aa3b84"
PHASE_2J_CANDIDATE_ID = "automatic-candidate-99cc13934144bf7e1706b8be"
EXPECTED_FAMILY_ID = "embfam-274b5207b8b71294ff75"
EXPECTED_VARIANT_ID = "embfam-274b5207b8b71294ff75-d"
EXPECTED_EMBEDDING_SHA256 = "f088d827adc548ee95f46566d758fd71fc304d042c43f1ecffc6526b60bcd832"
EXPECTED_SUMMARY_SHA256 = "63885588c374c37f4da9bf85294f240bdf0f28cb585e77b894ab516139ca46ae"
EXPECTED_POINTER_SHA256 = "7999b8ccf787dca9b8fb862f4e53ce7c1102eb729a3732c82fee76f3ba3ee05e"
EXPECTED_ATTENDANCE_STATUS_SHA256 = "46fd2df55cc3f61c5fa03c893937eb2828b9cc2af064ea9e427276b79a9d0b46"
EXPECTED_REVIEW_REGISTRY_SHA256 = "3b04e9aecfdfb1f64346c4a0e709fd9c36d7c56545bf816d6641b4c2a2e80841"
EXPECTED_PHASE_2H_OUTPUT_MANIFEST_SHA256 = "54e39ba629a032d4e3497a86f7a91bfeea66572ced4466ba16f150f89dc57f84"
EXPECTED_PHASE_2H_EVALUATION_MANIFEST_SHA256 = "9f1fdeaa4449d053009f1c6a24683089f85ac18d39180201dcf396a0c7a1f8ad"
EXPECTED_PHASE_2I_AUTHORITY_MANIFEST_SHA256 = "5e989648c8dc7605f088b585bdc4c954bf2e18cdaf93bb7107f0531f0068bebb"
EXPECTED_PHASE_2J_REVISION_MANIFEST_SHA256 = "e941fbbcdf70511a13ae49b773979f54f921f1b121a27128cfe84b312e0ffd17"
EXPECTED_OFFICIAL_CSV_SHA256 = "0211ac1433831aceb4176ee199f64bf962dbc784765a99a00bed74c7b38d51e3"
EXPECTED_CANDIDATE_CSV_SHA256 = "e28dc0278abfad034bd78bb576620a50f90100461f23b5e0b920e1e65e39c57f"
EXPECTED_KNOWN_MIXED_REVIEW_IMAGE_SHA256 = "3dc550614950d3d00bbcfe931f3a0ae9029d5708333bdc05321dac2cfa2ccabd"

PHASE_2I_DIAGNOSTIC_HASHES: dict[str, str] = {
    f"checkpoint_diagnostics_{PHASE_2I_DIAGNOSTIC_RUN_ID}.csv": "bb9ea9c839ccf554ad8aca21da89a82be5f03930e5480146a9d161a26bbaf78b",
    f"diagnostic_summary_{PHASE_2I_DIAGNOSTIC_RUN_ID}.json": "0b55caece89fe8e3e77394a798d66fb66b36c9cb3f834fce7f5c4b01e9011d1c",
    f"face_diagnostics_{PHASE_2I_DIAGNOSTIC_RUN_ID}.csv": "7eb70257593da06f682a3ddda54a9d260d4fc514f645237f5ca74b08f8da61a6",
    f"rejection_summary_{PHASE_2I_DIAGNOSTIC_RUN_ID}.csv": "7eb70257593da06f682a3ddda54a9d260d4fc514f645237f5ca74b08f8da61a6",
    f"tracklet_checkpoint_comparison_{PHASE_2I_DIAGNOSTIC_RUN_ID}.csv": "937d54838a2470274b875c8d3d00e2381237be3ce7ac818b1cf6a5acf8648c19",
    f"tracklet_comparison_{PHASE_2I_DIAGNOSTIC_RUN_ID}.json": "3d20149aef9aad9048c9d09896209e7248cd4849965e587c0872c796269bbe70",
    f"tracklet_diagnostics_{PHASE_2I_DIAGNOSTIC_RUN_ID}.csv": "b20dcccca83a1408c1123508c891e4fe7cfab2f142c49da295405c486065ba78",
    f"tracklet_observations_{PHASE_2I_DIAGNOSTIC_RUN_ID}.csv": "76fe7acb77488f17bc2aa025b59e3041c442c173ee5100b9f8565989fd2527a5",
    f"tracklet_window_comparison_{PHASE_2I_DIAGNOSTIC_RUN_ID}.csv": "599ad7671171247443f711971ae8c6b1b7b60499eb065dbf7948779e487789a8",
    f"zone_candidate_comparison_{PHASE_2I_DIAGNOSTIC_RUN_ID}.csv": "48d484e775e0b9c63123b0bb7813673d6dfda11e28fbd9495dbfada4a225b019",
    f"zone_checkpoint_comparison_{PHASE_2I_DIAGNOSTIC_RUN_ID}.csv": "11aba552d93c3ee8e3766547ae629d1dae89943e579bc59d94823be91dfaa4f6",
    f"zone_comparison_{PHASE_2I_DIAGNOSTIC_RUN_ID}.json": "7c9ca177515dc7f9df269eeb8d5d9afc2c2874a0827cb5ad663e9c7cf49dc08c",
    f"zone_frame_comparison_{PHASE_2I_DIAGNOSTIC_RUN_ID}.csv": "9083b92890319181b3115a92ff103ff3925e74e90a6053d19904fae44d455ece",
}

OFFICIAL_COUNTS = {
    "Present": 6,
    "Needs Review": 4,
    "Unconfirmed": 16,
    "Missing Enrollment": 1,
    "Absent": 0,
    "Total": 27,
}
CANDIDATE_COUNTS = {
    "Present": 2,
    "Needs Review": 8,
    "Unconfirmed": 16,
    "Missing Enrollment": 1,
    "Absent": 0,
    "Total": 27,
}


@dataclass(frozen=True)
class Inputs:
    repo_root: Path
    phase_2g_diagnostic_root: Path
    phase_2i_diagnostic_root: Path
    phase_2i_tracklets_csv: Path
    phase_2i_observations_csv: Path
    phase_2i_summary_json: Path
    phase_2h_output: Path
    phase_2h_output_manifest: Path
    phase_2h_evaluation_manifest: Path
    phase_2i_authority_output: Path
    phase_2i_authority_manifest: Path
    official_csv: Path
    phase_2j_candidate_output: Path
    phase_2j_revision_manifest: Path
    candidate_csv: Path
    attendance_status: Path
    review_registry: Path
    student_map: Path
    embeddings: Path
    embedding_summary: Path
    embedding_pointer: Path
    video_root: Path
    known_mixed_review_image: Path
    output_root: Path


@dataclass(frozen=True)
class PreflightResult:
    run_id: str
    policy: PurityPolicy
    tracklets: pd.DataFrame
    observations: pd.DataFrame
    phase_2h_selected: pd.DataFrame
    registry: dict[str, Any]
    source_fingerprint: dict[str, Any]
    source_manifest: dict[str, Any]
    official_counts: dict[str, int]
    candidate_counts: dict[str, int]
    inputs: Inputs


def default_inputs(repo_root: Path) -> Inputs:
    repo = Path(repo_root).resolve()
    phase_2g = repo / "attendance_output" / "diagnostics" / PHASE_2G_RUN_ID
    phase_2i = repo / "attendance_output" / "diagnostics" / PHASE_2I_DIAGNOSTIC_RUN_ID
    phase_2h = (
        repo
        / "attendance_output"
        / "product_workflow"
        / "phase_2h_multiframe_recovery"
        / PHASE_2H_OUTPUT_ID
    )
    authority = (
        repo
        / "attendance_output"
        / "product_workflow"
        / "phase_2i_authority"
        / "authority-revision-2d83c4f679f839a6c784c636"
    )
    candidate = (
        repo
        / "attendance_output"
        / "product_workflow"
        / "report_revisions"
        / SESSION_ID
        / PHASE_2J_CANDIDATE_ID
    )
    review = phase_2h / f"multiframe_recovery_review_{PHASE_2H_OUTPUT_ID}"
    return Inputs(
        repo_root=repo,
        phase_2g_diagnostic_root=phase_2g,
        phase_2i_diagnostic_root=phase_2i,
        phase_2i_tracklets_csv=phase_2i / f"tracklet_diagnostics_{PHASE_2I_DIAGNOSTIC_RUN_ID}.csv",
        phase_2i_observations_csv=phase_2i / f"tracklet_observations_{PHASE_2I_DIAGNOSTIC_RUN_ID}.csv",
        phase_2i_summary_json=phase_2i / f"diagnostic_summary_{PHASE_2I_DIAGNOSTIC_RUN_ID}.json",
        phase_2h_output=phase_2h,
        phase_2h_output_manifest=phase_2h / "output_manifest.json",
        phase_2h_evaluation_manifest=phase_2h / "evaluation" / "evaluation_manifest.json",
        phase_2i_authority_output=authority,
        phase_2i_authority_manifest=authority / "authority_manifest.json",
        official_csv=authority / f"corrected_attendance_{SESSION_ID}.csv",
        phase_2j_candidate_output=candidate,
        phase_2j_revision_manifest=candidate / "revision_manifest.json",
        candidate_csv=candidate / "candidate_attendance.csv",
        attendance_status=repo / "data" / "attendance_status.json",
        review_registry=repo / "data" / "review_evidence_registry.json",
        student_map=repo / "data" / "student_faculty_map.json",
        embeddings=repo / "models" / "student_embeddings.pkl",
        embedding_summary=repo / "models" / "embedding_summary.csv",
        embedding_pointer=repo / "models" / "current_embedding_version.json",
        video_root=repo / "cctv_videos" / "prepared_slots" / "2026-06-22" / "MON_P4",
        known_mixed_review_image=review / "reviewer" / "evidence" / "R0008.jpg",
        output_root=repo / "attendance_output" / "product_workflow" / "phase_2k_tracklet_purity",
    )


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise TrackletPurityError(f"Could not read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise TrackletPurityError(f"{label} must contain a JSON object")
    return value


def _load_csv(path: Path, label: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except Exception as exc:
        raise TrackletPurityError(f"Could not read {label}: {exc}") from exc


def _verify_hash(path: Path, expected: str, label: str) -> dict[str, Any]:
    if not Path(path).is_file():
        raise TrackletPurityError(f"Required {label} is missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise TrackletPurityError(f"{label} hash changed. Expected {expected}, got {actual}")
    return {"path": str(Path(path).resolve()), "sha256": actual, "size_bytes": Path(path).stat().st_size}


def _verify_recorded_manifest(manifest_path: Path, *, expected_hash: str, label: str) -> dict[str, Any]:
    metadata = _verify_hash(manifest_path, expected_hash, label)
    manifest = _load_json(manifest_path, label)
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise TrackletPurityError(f"{label} has no recorded files")
    for relative, recorded in files.items():
        path = manifest_path.parent / Path(str(relative))
        if not path.is_file():
            raise TrackletPurityError(f"{label} file missing: {relative}")
        if sha256_file(path) != str(recorded.get("sha256") or ""):
            raise TrackletPurityError(f"{label} file hash changed: {relative}")
        if path.stat().st_size != int(recorded.get("size_bytes", -1)):
            raise TrackletPurityError(f"{label} file size changed: {relative}")
    return metadata


def _status_counts(frame: pd.DataFrame) -> dict[str, int]:
    column = "Final_Status" if "Final_Status" in frame.columns else "Status"
    counts = frame[column].astype(str).str.strip().value_counts().to_dict()
    result = {name: int(counts.get(name, 0)) for name in OFFICIAL_COUNTS if name != "Total"}
    result["Total"] = int(len(frame))
    return result


def _assert_counts(actual: Mapping[str, Any], expected: Mapping[str, int], label: str) -> None:
    normalized = {key: int(actual.get(key, 0)) for key in expected}
    if normalized != dict(expected):
        raise TrackletPurityError(f"{label} totals changed: expected {dict(expected)}, got {normalized}")


def preflight(inputs: Inputs, *, verify_videos: bool = True) -> PreflightResult:
    policy = PurityPolicy()
    source_files: dict[str, dict[str, Any]] = {}

    # Phase 2G is resolved by the established hash-pinned Phase 2H preflight,
    # never by newest-directory discovery.
    phase_2h_inputs = phase_2h_default_inputs(inputs.repo_root)
    phase_2h_result = phase_2h_preflight(phase_2h_inputs, verify_videos=False)
    if phase_2h_result.output_id != PHASE_2H_OUTPUT_ID:
        raise TrackletPurityError("Phase 2H deterministic input ID changed")
    for filename, expected in sorted(PHASE_2G_DIAGNOSTIC_HASHES.items()):
        source_files[f"phase_2g/{filename}"] = _verify_hash(
            inputs.phase_2g_diagnostic_root / filename,
            expected,
            f"Phase 2G diagnostic {filename}",
        )

    for filename, expected in sorted(PHASE_2I_DIAGNOSTIC_HASHES.items()):
        source_files[f"phase_2i/{filename}"] = _verify_hash(
            inputs.phase_2i_diagnostic_root / filename,
            expected,
            f"Phase 2I diagnostic {filename}",
        )

    verify_phase_2h_output(inputs.phase_2h_output)
    source_files["phase_2h/output_manifest.json"] = _verify_hash(
        inputs.phase_2h_output_manifest,
        EXPECTED_PHASE_2H_OUTPUT_MANIFEST_SHA256,
        "Phase 2H output manifest",
    )
    source_files["phase_2h/evaluation/evaluation_manifest.json"] = _verify_recorded_manifest(
        inputs.phase_2h_evaluation_manifest,
        expected_hash=EXPECTED_PHASE_2H_EVALUATION_MANIFEST_SHA256,
        label="Phase 2H evaluation manifest",
    )
    verify_authority_output(inputs.phase_2i_authority_output)
    source_files["phase_2i_authority/authority_manifest.json"] = _verify_hash(
        inputs.phase_2i_authority_manifest,
        EXPECTED_PHASE_2I_AUTHORITY_MANIFEST_SHA256,
        "Phase 2I authority manifest",
    )
    verify_revision_output(inputs.phase_2j_candidate_output)
    source_files["phase_2j/revision_manifest.json"] = _verify_hash(
        inputs.phase_2j_revision_manifest,
        EXPECTED_PHASE_2J_REVISION_MANIFEST_SHA256,
        "Phase 2J revision manifest",
    )
    source_files["phase_2h/reviewer/evidence/R0008.jpg"] = _verify_hash(
        inputs.known_mixed_review_image,
        EXPECTED_KNOWN_MIXED_REVIEW_IMAGE_SHA256,
        "known mixed-track review image",
    )

    source_files["operational/attendance_status.json"] = _verify_hash(
        inputs.attendance_status,
        EXPECTED_ATTENDANCE_STATUS_SHA256,
        "attendance status",
    )
    source_files["operational/review_evidence_registry.json"] = _verify_hash(
        inputs.review_registry,
        EXPECTED_REVIEW_REGISTRY_SHA256,
        "review evidence registry",
    )
    source_files["production/student_embeddings.pkl"] = _verify_hash(
        inputs.embeddings,
        EXPECTED_EMBEDDING_SHA256,
        "production embeddings",
    )
    source_files["production/embedding_summary.csv"] = _verify_hash(
        inputs.embedding_summary,
        EXPECTED_SUMMARY_SHA256,
        "production embedding summary",
    )
    source_files["production/current_embedding_version.json"] = _verify_hash(
        inputs.embedding_pointer,
        EXPECTED_POINTER_SHA256,
        "production embedding pointer",
    )
    source_files["reports/official_reviewed.csv"] = _verify_hash(
        inputs.official_csv,
        EXPECTED_OFFICIAL_CSV_SHA256,
        "official reviewed attendance CSV",
    )
    source_files["reports/archived_candidate.csv"] = _verify_hash(
        inputs.candidate_csv,
        EXPECTED_CANDIDATE_CSV_SHA256,
        "archived automatic candidate CSV",
    )

    pointer = _load_json(inputs.embedding_pointer, "production embedding pointer")
    if pointer.get("family_id") != EXPECTED_FAMILY_ID or pointer.get("variant_id") != EXPECTED_VARIANT_ID:
        raise TrackletPurityError("Production embedding family or variant changed")
    if pointer.get("production_embeddings_sha256") != EXPECTED_EMBEDDING_SHA256:
        raise TrackletPurityError("Production embedding pointer hash does not match production")
    if pointer.get("production_summary_sha256") != EXPECTED_SUMMARY_SHA256:
        raise TrackletPurityError("Production summary pointer hash does not match production")

    status = _load_json(inputs.attendance_status, "attendance status")
    entry = status.get(SESSION_ID)
    if not isinstance(entry, Mapping):
        raise TrackletPurityError("Official MON P4 attendance entry is missing")
    persisted_official = {
        "Present": int(entry.get("present_count", -1)),
        "Needs Review": int(entry.get("needs_review_count", -1)),
        "Unconfirmed": int(entry.get("unconfirmed_count", -1)),
        "Missing Enrollment": int(entry.get("missing_enrollment_count", -1)),
        "Absent": int(entry.get("absent_count", -1)),
        "Total": int(entry.get("total_students", -1)),
    }
    _assert_counts(persisted_official, OFFICIAL_COUNTS, "Persisted official MON P4")
    candidate_entry = entry.get("latest_automatic_candidate")
    if not isinstance(candidate_entry, Mapping):
        raise TrackletPurityError("Persisted archived MON P4 candidate is missing")
    persisted_candidate = {
        "Present": int(candidate_entry.get("present_count", -1)),
        "Needs Review": int(candidate_entry.get("needs_review_count", -1)),
        "Unconfirmed": int(candidate_entry.get("unconfirmed_count", -1)),
        "Missing Enrollment": int(candidate_entry.get("missing_enrollment_count", -1)),
        "Absent": int(candidate_entry.get("absent_count", -1)),
        "Total": int(candidate_entry.get("total_students", -1)),
    }
    _assert_counts(persisted_candidate, CANDIDATE_COUNTS, "Persisted archived MON P4 candidate")
    if entry.get("official_recognition_authority") != "reviewed_multiframe_tracklet_evidence":
        raise TrackletPurityError("Official MON P4 authority changed")
    if entry.get("automatic_recognition_authority") != "strict_tracklet_aggregate_with_guarded_review_candidates":
        raise TrackletPurityError("Future automatic authority changed")
    if bool(entry.get("guarded_recovery_automatic")):
        raise TrackletPurityError("Guarded recovery unexpectedly became automatic")
    if bool(entry.get("attendance_finalized")) or str(entry.get("Attendance_Finalized") or "").lower() == "yes":
        raise TrackletPurityError("MON P4 attendance unexpectedly became finalized")

    official = _load_csv(inputs.official_csv, "official reviewed attendance CSV")
    candidate = _load_csv(inputs.candidate_csv, "archived automatic candidate CSV")
    official_counts = _status_counts(official)
    candidate_counts = _status_counts(candidate)
    _assert_counts(official_counts, OFFICIAL_COUNTS, "Official reviewed CSV")
    _assert_counts(candidate_counts, CANDIDATE_COUNTS, "Archived candidate CSV")

    diagnostic_summary = _load_json(inputs.phase_2i_summary_json, "Phase 2I diagnostic summary")
    metadata = diagnostic_summary.get("run_metadata") or {}
    if metadata.get("diagnostic_run_id") != PHASE_2I_DIAGNOSTIC_RUN_ID:
        raise TrackletPurityError("Phase 2I diagnostic run ID changed")
    if metadata.get("session_id") != SESSION_ID:
        raise TrackletPurityError("Phase 2I diagnostic session changed")
    if float(metadata.get("match_threshold", -1)) != 0.48 or float(metadata.get("margin_threshold", -1)) != 0.08:
        raise TrackletPurityError("Strict 0.48/0.08 thresholds changed")
    if metadata.get("official_recognition_unit") != "tracklet_aggregate":
        raise TrackletPurityError("Phase 2I diagnostic authority is not strict tracklet aggregation")

    registry = load_registry(inputs.review_registry)
    if registry is None:
        raise TrackletPurityError("Review evidence registry is missing")
    expected_source = dict(registry.get("source_fingerprint") or {})
    if verify_videos:
        actual_source = source_fingerprint(
            video_dir=inputs.video_root,
            embeddings_path=inputs.embeddings,
            session_id=SESSION_ID,
            input_source_path="cctv_videos/prepared_slots/2026-06-22/MON_P4",
        )
        if actual_source.get("fingerprint_sha256") != expected_source.get("fingerprint_sha256"):
            raise TrackletPurityError("Source videos or production embedding fingerprint changed")
        source_verified = True
    else:
        actual_source = expected_source
        source_verified = False

    tracklets = _load_csv(inputs.phase_2i_tracklets_csv, "Phase 2I tracklet diagnostics")
    observations = _load_csv(inputs.phase_2i_observations_csv, "Phase 2I tracklet observations")
    if len(tracklets) != 308:
        raise TrackletPurityError(f"Expected 308 Phase 2I tracklets, found {len(tracklets)}")
    known_observations = observations[observations["Tracklet_ID"] == MIXED_TRACKLET_ID]
    if len(known_observations) != 39:
        raise TrackletPurityError(
            f"Exact mixed-track observations changed; expected 39, found {len(known_observations)}"
        )
    mixed_entries = [
        item
        for item in registry.get("evidence") or []
        if isinstance(item, Mapping) and bool(item.get("mixed_track_quarantine"))
    ]
    if len(mixed_entries) != 1 or mixed_entries[0].get("tracklet_id") != MIXED_TRACKLET_ID:
        raise TrackletPurityError("Known mixed-track registry quarantine changed")

    input_hashes = {key: value["sha256"] for key, value in sorted(source_files.items())}
    run_fingerprint = canonical_json_hash(
        {
            "policy": policy.as_payload(),
            "input_hashes": input_hashes,
            "source_fingerprint_sha256": expected_source.get("fingerprint_sha256"),
        }
    )
    run_id = f"tracklet-purity-{run_fingerprint[:20]}"
    source_manifest = {
        "schema_version": 1,
        "policy_version": POLICY_VERSION,
        "run_id": run_id,
        "session_id": SESSION_ID,
        "phase_2g_diagnostic_run_id": PHASE_2G_RUN_ID,
        "phase_2i_diagnostic_run_id": PHASE_2I_DIAGNOSTIC_RUN_ID,
        "phase_2h_output_id": PHASE_2H_OUTPUT_ID,
        "phase_2i_authority_revision_id": AUTHORITY_REVISION_ID,
        "phase_2j_candidate_revision_id": PHASE_2J_CANDIDATE_ID,
        "production_family_id": EXPECTED_FAMILY_ID,
        "production_variant_id": EXPECTED_VARIANT_ID,
        "production_embedding_sha256": EXPECTED_EMBEDDING_SHA256,
        "source_fingerprint": expected_source,
        "source_files_verified_against_current_bytes": source_verified,
        "raw_observation_embedding_vectors_persisted": False,
        "input_files": source_files,
        "run_fingerprint_sha256": run_fingerprint,
    }
    return PreflightResult(
        run_id=run_id,
        policy=policy,
        tracklets=tracklets,
        observations=observations,
        phase_2h_selected=phase_2h_result.selected,
        registry=registry,
        source_fingerprint=actual_source,
        source_manifest=source_manifest,
        official_counts=official_counts,
        candidate_counts=candidate_counts,
        inputs=inputs,
    )


def _boolish(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _float(value: Any) -> float | None:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _time_seconds(value: Any) -> float:
    parts = str(value or "").strip().split(":")
    if len(parts) not in {2, 3}:
        raise TrackletPurityError(f"Invalid diagnostic timestamp: {value}")
    try:
        if len(parts) == 2:
            return float(parts[0]) * 60.0 + float(parts[1])
        return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
    except ValueError as exc:
        raise TrackletPurityError(f"Invalid diagnostic timestamp: {value}") from exc


def _bbox(value: Any) -> tuple[float, float, float, float]:
    try:
        parts = tuple(float(part.strip()) for part in str(value or "").split(","))
    except ValueError as exc:
        raise TrackletPurityError(f"Invalid diagnostic bounding box: {value}") from exc
    if len(parts) != 4:
        raise TrackletPurityError(f"Invalid diagnostic bounding box: {value}")
    return parts


def _observations_from_frame(frame: pd.DataFrame) -> tuple[PurityObservation, ...]:
    rows: list[PurityObservation] = []
    for _, row in frame.iterrows():
        rows.append(
            PurityObservation(
                observation_id=str(row.get("Observation_ID") or "").strip(),
                frame_index=_int(row.get("Frame"), -1),
                timestamp_seconds=_time_seconds(row.get("Time_In_Video")),
                bbox=_bbox(row.get("BBox_Original_Coordinates")),
                predicted_roll=str(row.get("Frame_Best_Roll") or "").strip().upper(),
                best_score=_float(row.get("Frame_Best_Score")),
                margin=_float(row.get("Frame_Margin")),
                accepted=_boolish(row.get("Frame_Accepted")),
                selected=_boolish(row.get("Selected_For_Aggregation")),
                quality_label=str(row.get("Diagnostic_Quality_Label") or "").strip(),
                embedding_extraction_success=_boolish(row.get("Embedding_Extraction_Success")),
                embedding_consistent=_boolish(row.get("Embedding_Consistent")),
                embedding=None,
            )
        )
    return tuple(sorted(rows, key=lambda item: (item.timestamp_seconds, item.frame_index, item.observation_id)))


def _summary_from_row(row: Mapping[str, Any]) -> TrackSummaryEvidence:
    return TrackSummaryEvidence(
        pairwise_similarity_min=_float(row.get("Pairwise_Similarity_Min")),
        pairwise_similarity_median=_float(row.get("Pairwise_Similarity_Median")),
        selected_observation_count=_int(row.get("Selected_Observation_Count")),
        consistent_embedding_count=_int(row.get("Consistent_Embedding_Count")),
        aggregate_best_roll=str(row.get("Tracklet_Best_Roll") or "").strip().upper(),
        aggregate_best_score=_float(row.get("Tracklet_Best_Score")),
        aggregate_margin=_float(row.get("Tracklet_Margin")),
        aggregate_accepted=_boolish(row.get("Tracklet_Accepted")),
    )


def analyze_preflight(result: PreflightResult) -> tuple[list[TrackPurityResult], list[dict[str, Any]]]:
    mixed_ids = {
        str(item.get("tracklet_id") or "")
        for item in result.registry.get("evidence") or []
        if isinstance(item, Mapping) and bool(item.get("mixed_track_quarantine"))
    }
    grouped = {track_id: frame for track_id, frame in result.observations.groupby("Tracklet_ID", sort=True)}
    analyses: list[TrackPurityResult] = []
    tracklet_rows: dict[str, Mapping[str, Any]] = {}
    for _, row in result.tracklets.sort_values("Tracklet_ID").iterrows():
        track_id = str(row.get("Tracklet_ID") or "").strip()
        tracklet_rows[track_id] = row
        observation_frame = grouped.get(track_id)
        if observation_frame is None or observation_frame.empty:
            raise TrackletPurityError(f"Tracklet observations missing for {track_id}")
        checkpoint = str(row.get("Checkpoint_ID") or "").strip().upper()
        camera = str(row.get("Camera_ID") or "").strip()
        video = f"{checkpoint}/{camera}/{str(row.get('Video') or '').strip()}"
        analyses.append(
            analyze_track_purity(
                parent_track_id=track_id,
                checkpoint_id=checkpoint,
                camera_id=camera,
                source_video=video,
                observations=_observations_from_frame(observation_frame),
                source_fingerprint_sha256=str(result.source_fingerprint.get("fingerprint_sha256") or ""),
                production_embedding_sha256=EXPECTED_EMBEDDING_SHA256,
                policy=result.policy,
                summary=_summary_from_row(row),
                known_mixed_review=track_id in mixed_ids,
                strict_evaluator=None,
            )
        )

    analysis_by_id = {item.parent_track_id: item for item in analyses}
    review_impact: list[dict[str, Any]] = []
    for item in result.registry.get("evidence") or []:
        if not isinstance(item, Mapping):
            continue
        track_id = str(item.get("tracklet_id") or "").strip()
        analysis = analysis_by_id.get(track_id)
        row = tracklet_rows.get(track_id)
        if analysis is None or row is None:
            raise TrackletPurityError(f"Reviewed tracklet missing from Phase 2I diagnostics: {track_id}")
        expected_signature = str(item.get("evidence_signature_sha256") or "")
        current_signature = _evidence_signature(row)
        decision = review_carry_forward_decision(
            registry_accepted=bool(item.get("accepted_for_carry_forward")),
            registry_mixed_quarantine=bool(item.get("mixed_track_quarantine")),
            existing_evidence_signature_sha256=expected_signature,
            current_evidence_signature_sha256=current_signature,
            purity_result=analysis,
        )
        review_impact.append(
            {
                "Tracklet_ID": track_id,
                "Checkpoint_ID": str(item.get("checkpoint_id") or "").upper(),
                "Predicted_Roll": str(item.get("predicted_roll") or ""),
                "Registry_Accepted": bool(item.get("accepted_for_carry_forward")),
                "Registry_Mixed_Quarantine": bool(item.get("mixed_track_quarantine")),
                "Existing_Evidence_Signature_SHA256": expected_signature,
                "Current_Evidence_Signature_SHA256": current_signature,
                "Evidence_Signature_Match": expected_signature == current_signature,
                "Purity_Outcome": analysis.outcome,
                "Child_Count": len(analysis.children),
                "Review_Carry_Forward_Allowed": bool(decision["allowed"]),
                "Review_Carry_Forward_Reason": decision["reason"],
                "Partial_Carry_Forward": False,
                "Child_Review_Inheritance": False,
            }
        )
    return analyses, review_impact


def _parent_rows(analyses: Sequence[TrackPurityResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in analyses:
        signals = sorted({signal for boundary in item.boundaries for signal in boundary.signals})
        rows.append(
            {
                "Parent_Tracklet_ID": item.parent_track_id,
                "Checkpoint_ID": item.checkpoint_id,
                "Camera_ID": item.camera_id,
                "Source_Video": item.source_video,
                "Observation_Count": item.observation_count,
                "Start_Timestamp_Seconds": item.start_timestamp_seconds,
                "End_Timestamp_Seconds": item.end_timestamp_seconds,
                "Purity_Outcome": item.outcome,
                "Split_Disposition": item.split_disposition,
                "Quarantined": item.quarantined,
                "Defensible_Split_Count": sum(1 for boundary in item.boundaries if boundary.defensible_split),
                "Child_Count": len(item.children),
                "Detected_Signals": signals,
                "Raw_Embedding_Vectors_Available": item.raw_embedding_vectors_available,
                "Pairwise_Similarity_Min": item.pairwise_similarity_min,
                "Pairwise_Similarity_Median": item.pairwise_similarity_median,
                "Significant_Embedding_Cluster_Sizes": item.significant_embedding_cluster_sizes,
                "Dominant_Roll": item.dominant_roll,
                "Dominant_Share": round(item.dominant_share, 6),
                "Known_Mixed_Review": item.known_mixed_review,
                "Parent_Official_Checkpoint_Contribution": 0,
                "Review_Carry_Forward_Allowed": False,
                "Evidence_Fingerprint_SHA256": item.parent_evidence_fingerprint_sha256,
                "Policy_Version": item.policy_version,
                "Reasons": item.reasons,
            }
        )
    return rows


def _boundary_rows(analyses: Sequence[TrackPurityResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in analyses:
        for boundary in item.boundaries:
            if not boundary.signals:
                continue
            rows.append(
                {
                    "Parent_Tracklet_ID": item.parent_track_id,
                    "Checkpoint_ID": item.checkpoint_id,
                    "Camera_ID": item.camera_id,
                    "Boundary_Index": boundary.boundary_index,
                    "Left_Observation_ID": boundary.left_observation_id,
                    "Right_Observation_ID": boundary.right_observation_id,
                    "Left_Count": boundary.left_count,
                    "Right_Count": boundary.right_count,
                    "Time_Gap_Seconds": boundary.time_gap_seconds,
                    "BBox_IoU": boundary.bbox_iou,
                    "Center_Ratio": boundary.center_ratio,
                    "Size_Ratio": boundary.size_ratio,
                    "Aspect_Log_Change": boundary.aspect_log_change,
                    "Adjacent_Embedding_Similarity": boundary.adjacent_embedding_similarity,
                    "Left_Dominant_Roll": boundary.left_dominant_roll,
                    "Right_Dominant_Roll": boundary.right_dominant_roll,
                    "Left_Dominant_Share": boundary.left_dominant_share,
                    "Right_Dominant_Share": boundary.right_dominant_share,
                    "Signals": boundary.signals,
                    "Eligible_By_Child_Size": boundary.eligible_by_child_size,
                    "Defensible_Split": boundary.defensible_split,
                    "Boundary_Fingerprint_SHA256": boundary.boundary_fingerprint_sha256,
                    "Policy_Version": item.policy_version,
                }
            )
    return rows


def _child_rows(analyses: Sequence[TrackPurityResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in analyses:
        for child in item.children:
            rows.append(
                {
                    "Parent_Tracklet_ID": child.parent_track_id,
                    "Child_Tracklet_ID": child.child_track_id,
                    "Child_Ordinal": child.child_ordinal,
                    "Checkpoint_ID": child.checkpoint_id,
                    "Camera_ID": child.camera_id,
                    "Source_Video": child.source_video,
                    "Start_Timestamp_Seconds": child.start_timestamp_seconds,
                    "End_Timestamp_Seconds": child.end_timestamp_seconds,
                    "Observation_Count": len(child.observation_ids),
                    "Observation_IDs": child.observation_ids,
                    "Split_Reason": child.split_reason,
                    "Purity_Outcome": child.outcome,
                    "Dominant_Roll": child.dominant_roll,
                    "Dominant_Share": child.dominant_share,
                    "Strict_Gate_Passed": child.strict_gate_passed,
                    "Strict_Gate_Reason": child.strict_gate_reason,
                    "Strict_Best_Score": child.strict_best_score,
                    "Strict_Margin": child.strict_margin,
                    "Shadow_Only": True,
                    "Inherited_Human_Review": False,
                    "Source_Fingerprint_SHA256": child.source_fingerprint_sha256,
                    "Production_Embedding_SHA256": child.production_embedding_sha256,
                    "Evidence_Fingerprint_SHA256": child.evidence_fingerprint_sha256,
                    "Policy_Version": child.policy_version,
                }
            )
    return rows


def _known_mixed_analysis(analysis: TrackPurityResult, review_impact: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    carry = next(item for item in review_impact if item.get("Tracklet_ID") == MIXED_TRACKLET_ID)
    signals = sorted({signal for boundary in analysis.boundaries for signal in boundary.signals})
    defensible = [boundary for boundary in analysis.boundaries if boundary.defensible_split]
    return {
        "tracklet_id": MIXED_TRACKLET_ID,
        "observation_count": analysis.observation_count,
        "timestamp_range_seconds": [analysis.start_timestamp_seconds, analysis.end_timestamp_seconds],
        "detected_discontinuity_signals": signals,
        "embedding_cluster_evidence": {
            "raw_embedding_vectors_available": analysis.raw_embedding_vectors_available,
            "significant_clusters_evaluable": analysis.raw_embedding_vectors_available,
            "significant_cluster_sizes": list(analysis.significant_embedding_cluster_sizes),
            "cluster_medoid_similarity": analysis.significant_embedding_cluster_medoid_similarity,
            "derived_selected_pairwise_similarity_min": analysis.pairwise_similarity_min,
            "derived_selected_pairwise_similarity_median": analysis.pairwise_similarity_median,
            "interpretation": (
                "Raw per-observation vectors were not persisted, so Phase 2K-A does not infer "
                "clusters for this real track. The immutable mixed-track human review remains decisive."
            ),
        },
        "identity_vote_timeline": list(analysis.identity_vote_timeline),
        "geometry_timeline": list(analysis.geometry_timeline),
        "defensible_split_exists": bool(defensible),
        "possible_child_segment_count": len(analysis.children),
        "child_minimum_observation_results": [
            len(child.observation_ids) >= PurityPolicy().min_child_observations for child in analysis.children
        ],
        "child_strict_gate_results": [child.strict_gate_passed for child in analysis.children],
        "review_carry_forward_rejected": not bool(carry.get("Review_Carry_Forward_Allowed")),
        "review_carry_forward_reason": carry.get("Review_Carry_Forward_Reason"),
        "parent_official_checkpoint_contribution": 0,
        "parent_quarantined": analysis.quarantined,
        "final_shadow_disposition": analysis.outcome,
    }


def build_shadow_files(result: PreflightResult) -> tuple[dict[str, bytes], dict[str, Any]]:
    analyses, review_impact = analyze_preflight(result)
    by_id = {item.parent_track_id: item for item in analyses}
    known = by_id.get(MIXED_TRACKLET_ID)
    if known is None or known.outcome != MIXED_QUARANTINED or not known.quarantined or known.children:
        raise TrackletPurityError("Known mixed-track negative regression did not fail closed")

    strict_reviewed_ids = set(
        result.phase_2h_selected[
            result.phase_2h_selected["Evidence_Tier"].astype(str).str.strip() == "strict_accepted"
        ]["Tracklet_ID"].astype(str)
    )
    unnecessarily_split = sorted(
        track_id for track_id in strict_reviewed_ids if by_id.get(track_id) and by_id[track_id].outcome == SPLIT_CANDIDATE
    )
    if unnecessarily_split:
        raise TrackletPurityError(
            "Reviewed strict historical tracks were unnecessarily split: " + ", ".join(unnecessarily_split)
        )

    parent_rows = _parent_rows(analyses)
    boundary_rows = _boundary_rows(analyses)
    child_rows = _child_rows(analyses)
    quarantine_rows = [
        row for row in parent_rows if str(row.get("Quarantined")).lower() in {"true", "1"} or bool(row.get("Quarantined"))
    ]
    shadow_counts = dict(result.candidate_counts)
    attendance_rows = []
    for scenario, counts, role, applied in (
        ("current_reviewed_official", result.official_counts, "official_reviewed", True),
        ("current_archived_automatic_candidate", result.candidate_counts, "archived_candidate", False),
        ("strict_tracklet_baseline", result.candidate_counts, "diagnostic_baseline", False),
        ("purity_split_shadow", shadow_counts, "diagnostic_shadow", False),
    ):
        attendance_rows.append(
            {
                "Scenario": scenario,
                "Role": role,
                "Present": counts["Present"],
                "Needs_Review": counts["Needs Review"],
                "Unconfirmed": counts["Unconfirmed"],
                "Missing_Enrollment": counts["Missing Enrollment"],
                "Absent": counts["Absent"],
                "Total": counts["Total"],
                "Applied_To_Official": applied,
                "Diagnostic_Child_Candidate_Count": len(child_rows) if scenario == "purity_split_shadow" else 0,
                "Official_Attendance_Changed_By_Phase_2K": False,
                "Recognition_Repeated": False,
                "Video_Reprocessed": False,
                "Authority_Changed": False,
                "Guarded_Recovery_Promoted": False,
            }
        )

    outcome_counts: dict[str, int] = {}
    for analysis in analyses:
        outcome_counts[analysis.outcome] = outcome_counts.get(analysis.outcome, 0) + 1
    evaluation = {
        "schema_version": 1,
        "policy_version": POLICY_VERSION,
        "run_id": result.run_id,
        "session_id": SESSION_ID,
        "decision": "diagnostic_shadow_completed_no_authority_change",
        "parent_tracklets_evaluated": len(analyses),
        "outcome_counts": outcome_counts,
        "signaled_boundaries": len(boundary_rows),
        "split_candidate_parents": sum(1 for item in analyses if item.outcome == SPLIT_CANDIDATE),
        "diagnostic_child_candidates": len(child_rows),
        "quarantined_parents": len(quarantine_rows),
        "strict_reviewed_historical_tracks_checked": len(strict_reviewed_ids),
        "strict_reviewed_historical_tracks_unnecessarily_split": unnecessarily_split,
        "review_registry_pairs_checked": len(review_impact),
        "review_carry_forward_allowed_unchanged_pairs": sum(
            1 for item in review_impact if bool(item.get("Review_Carry_Forward_Allowed"))
        ),
        "review_carry_forward_rejected_pairs": sum(
            1 for item in review_impact if not bool(item.get("Review_Carry_Forward_Allowed"))
        ),
        "known_mixed_track_analysis": _known_mixed_analysis(known, review_impact),
        "limitations": [
            "Phase 2I CSV diagnostics persist derived embedding evidence but not raw per-observation vectors.",
            "No embedding clusters or child strict identity scores are inferred when raw vectors are absent.",
            "All generated child lineages, if any, are shadow-only and inherit no human review.",
        ],
        "preservation_assertions": {
            "official_attendance_changed": False,
            "recognition_repeated": False,
            "video_reprocessed": False,
            "authority_changed": False,
            "guarded_recovery_promoted": False,
            "production_embeddings_changed": False,
            "roster_changed": False,
            "thresholds_changed": False,
            "timetable_changed": False,
            "hod_configuration_changed": False,
            "official_counts": result.official_counts,
            "archived_candidate_counts": result.candidate_counts,
            "production_family_id": EXPECTED_FAMILY_ID,
            "production_variant_id": EXPECTED_VARIANT_ID,
            "production_embedding_sha256": EXPECTED_EMBEDDING_SHA256,
            "production_summary_sha256": EXPECTED_SUMMARY_SHA256,
            "match_threshold": 0.48,
            "margin_threshold": 0.08,
            "automatic_policy_version": AUTOMATIC_POLICY_VERSION,
        },
    }

    parent_fields = [
        "Parent_Tracklet_ID", "Checkpoint_ID", "Camera_ID", "Source_Video", "Observation_Count",
        "Start_Timestamp_Seconds", "End_Timestamp_Seconds", "Purity_Outcome", "Split_Disposition",
        "Quarantined", "Defensible_Split_Count", "Child_Count", "Detected_Signals",
        "Raw_Embedding_Vectors_Available", "Pairwise_Similarity_Min", "Pairwise_Similarity_Median",
        "Significant_Embedding_Cluster_Sizes", "Dominant_Roll", "Dominant_Share", "Known_Mixed_Review",
        "Parent_Official_Checkpoint_Contribution", "Review_Carry_Forward_Allowed",
        "Evidence_Fingerprint_SHA256", "Policy_Version", "Reasons",
    ]
    boundary_fields = [
        "Parent_Tracklet_ID", "Checkpoint_ID", "Camera_ID", "Boundary_Index", "Left_Observation_ID",
        "Right_Observation_ID", "Left_Count", "Right_Count", "Time_Gap_Seconds", "BBox_IoU",
        "Center_Ratio", "Size_Ratio", "Aspect_Log_Change", "Adjacent_Embedding_Similarity",
        "Left_Dominant_Roll", "Right_Dominant_Roll", "Left_Dominant_Share", "Right_Dominant_Share",
        "Signals", "Eligible_By_Child_Size", "Defensible_Split", "Boundary_Fingerprint_SHA256",
        "Policy_Version",
    ]
    child_fields = [
        "Parent_Tracklet_ID", "Child_Tracklet_ID", "Child_Ordinal", "Checkpoint_ID", "Camera_ID",
        "Source_Video", "Start_Timestamp_Seconds", "End_Timestamp_Seconds", "Observation_Count",
        "Observation_IDs", "Split_Reason", "Purity_Outcome", "Dominant_Roll", "Dominant_Share",
        "Strict_Gate_Passed", "Strict_Gate_Reason", "Strict_Best_Score", "Strict_Margin", "Shadow_Only",
        "Inherited_Human_Review", "Source_Fingerprint_SHA256", "Production_Embedding_SHA256",
        "Evidence_Fingerprint_SHA256", "Policy_Version",
    ]
    review_fields = [
        "Tracklet_ID", "Checkpoint_ID", "Predicted_Roll", "Registry_Accepted",
        "Registry_Mixed_Quarantine", "Existing_Evidence_Signature_SHA256",
        "Current_Evidence_Signature_SHA256", "Evidence_Signature_Match", "Purity_Outcome", "Child_Count",
        "Review_Carry_Forward_Allowed", "Review_Carry_Forward_Reason", "Partial_Carry_Forward",
        "Child_Review_Inheritance",
    ]
    attendance_fields = [
        "Scenario", "Role", "Present", "Needs_Review", "Unconfirmed", "Missing_Enrollment", "Absent",
        "Total", "Applied_To_Official", "Diagnostic_Child_Candidate_Count",
        "Official_Attendance_Changed_By_Phase_2K", "Recognition_Repeated", "Video_Reprocessed",
        "Authority_Changed", "Guarded_Recovery_Promoted",
    ]
    files = {
        "purity_policy.json": canonical_json_bytes(result.policy.as_payload()),
        "source_manifest.json": canonical_json_bytes(result.source_manifest),
        "parent_track_summary.csv": csv_bytes(parent_rows, parent_fields),
        "split_boundary_candidates.csv": csv_bytes(boundary_rows, boundary_fields),
        "child_track_summary.csv": csv_bytes(child_rows, child_fields),
        "quarantined_tracks.csv": csv_bytes(quarantine_rows, parent_fields),
        "review_carry_forward_impact.csv": csv_bytes(review_impact, review_fields),
        "attendance_impact_shadow.csv": csv_bytes(attendance_rows, attendance_fields),
        "evaluation_summary.json": canonical_json_bytes(evaluation),
    }
    return files, evaluation


def materialize_shadow(result: PreflightResult) -> tuple[Path, bool, dict[str, Any]]:
    files, evaluation = build_shadow_files(result)
    output, reused = write_immutable_output(
        output_root=result.inputs.output_root,
        run_id=result.run_id,
        files=files,
    )
    verify_immutable_output(output)
    return output, reused, evaluation
