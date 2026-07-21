from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .product_phase_2i_authority import (
    MIXED_TRACKLET_ID,
    PHASE_2H_OUTPUT_ID,
    RECOVERY_RULE_ID,
    SESSION_ID,
)
from .recall_analysis import canonical_roll

POLICY_VERSION = "product-phase-2j-exact-review-carry-forward-v2"
REGISTRY_SCHEMA_VERSION = 2
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
EVIDENCE_SIGNATURE_FIELDS = (
    "Tracklet_ID",
    "Checkpoint_ID",
    "Camera_ID",
    "Video",
    "Start_Frame",
    "End_Frame",
    "Observation_Count",
    "Embedding_Count",
    "Selected_Observation_Count",
    "Consistent_Embedding_Count",
    "Medoid_Observation_ID",
    "Member_Observation_IDs",
    "Selected_Observation_IDs",
    "Consistent_Observation_IDs",
    "Dominant_Frame_Best_Roll",
    "Dominant_Frame_Best_Count",
    "Dominant_Frame_Best_Share_Pct",
    "Tracklet_Best_Roll",
    "Tracklet_Best_Score",
    "Tracklet_Second_Roll",
    "Tracklet_Second_Score",
    "Tracklet_Margin",
    "Pairwise_Similarity_Median",
    "Aggregate_Mode",
)


class ReviewCarryForwardError(RuntimeError):
    """Raised when reviewed evidence cannot be reused safely."""


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _canonical_json_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _evidence_signature(row: Mapping[str, Any]) -> str:
    payload: dict[str, str] = {}
    for field in EVIDENCE_SIGNATURE_FIELDS:
        value = str(row.get(field) or "").strip()
        if field in {"Tracklet_Best_Roll", "Tracklet_Second_Roll", "Dominant_Frame_Best_Roll"}:
            value = canonical_roll(value)
        elif field == "Checkpoint_ID":
            value = value.upper()
        elif field == "Video":
            value = value.replace("\\", "/")
        payload[field] = value
    return _canonical_json_hash(payload)


def _tracklet_evidence_map(tracklet_frame: pd.DataFrame) -> dict[tuple[str, str, str], str]:
    missing = sorted(set(EVIDENCE_SIGNATURE_FIELDS).difference(tracklet_frame.columns))
    if missing:
        raise ReviewCarryForwardError(
            "Tracklet diagnostics missing evidence-signature fields: " + ", ".join(missing)
        )
    result: dict[tuple[str, str, str], str] = {}
    for _, row in tracklet_frame.iterrows():
        key = (
            str(row.get("Tracklet_ID") or "").strip(),
            str(row.get("Checkpoint_ID") or "").strip().upper(),
            canonical_roll(row.get("Tracklet_Best_Roll")),
        )
        if key in result:
            raise ReviewCarryForwardError("Tracklet diagnostics contain duplicate evidence keys")
        result[key] = _evidence_signature(row)
    return result


def source_fingerprint(
    *,
    video_dir: Path,
    embeddings_path: Path,
    session_id: str,
    input_source_path: str,
) -> dict[str, Any]:
    root = Path(video_dir)
    if not root.is_dir():
        raise ReviewCarryForwardError(f"Video source directory not found: {root}")
    videos = [path for path in sorted(root.rglob("*")) if path.is_file() and path.suffix.lower() in ALLOWED_VIDEO_SUFFIXES]
    if len(videos) != 10:
        raise ReviewCarryForwardError(f"Exact reviewed source requires 10 videos; found {len(videos)}")
    files = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in videos
    ]
    embeddings = Path(embeddings_path)
    if not embeddings.is_file():
        raise ReviewCarryForwardError(f"Production embeddings not found: {embeddings}")
    payload = {
        "session_id": str(session_id),
        "input_source_path": str(input_source_path).replace("\\", "/"),
        "video_files": files,
        "embeddings_sha256": _sha256_file(embeddings),
    }
    return {**payload, "fingerprint_sha256": _canonical_json_hash(payload)}


def _evaluation_dir(repo_root: Path) -> Path:
    return (
        Path(repo_root)
        / "attendance_output"
        / "product_workflow"
        / "phase_2h_multiframe_recovery"
        / PHASE_2H_OUTPUT_ID
        / "evaluation"
    )


def build_exact_mon_p4_registry(
    *,
    repo_root: Path,
    video_dir: Path,
    embeddings_path: Path,
    input_source_path: str,
) -> dict[str, Any]:
    # Re-run the established Phase 2H read-only preflight so the registry is
    # bound to the exact ten reviewed videos, immutable frozen source copies,
    # diagnostics, and leakage-safe historical benchmark. This prevents review
    # decisions from being attached to changed footage merely because a folder
    # path stayed the same.
    from .product_phase_2h_multiframe_recovery import (
        default_inputs as phase_2h_default_inputs,
        preflight as phase_2h_preflight,
    )

    phase_2h_result = phase_2h_preflight(phase_2h_default_inputs(repo_root), verify_videos=True)
    if phase_2h_result.output_id != PHASE_2H_OUTPUT_ID:
        raise ReviewCarryForwardError("Phase 2H reviewed source fingerprint changed")
    reviewed_tracklet_signatures = _tracklet_evidence_map(phase_2h_result.tracklets)

    evaluation_dir = _evaluation_dir(repo_root)
    review_path = evaluation_dir / "joined_multiframe_recovery_review.csv"
    summary_path = evaluation_dir / "evaluation_summary.json"
    if not review_path.is_file() or not summary_path.is_file():
        raise ReviewCarryForwardError("Phase 2H reviewed evidence is missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    if summary.get("decision") != "multiframe_recovery_passed_pending_explicit_authority_patch":
        raise ReviewCarryForwardError("Phase 2H evaluation is not approved for carry-forward")
    review = pd.read_csv(review_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if len(review) != 31:
        raise ReviewCarryForwardError(f"Expected 31 reviewed identity/checkpoint pairs, found {len(review)}")
    evidence: list[dict[str, Any]] = []
    for _, row in review.iterrows():
        predicted = canonical_roll(row.get("Predicted_Roll"))
        actual = canonical_roll(row.get("Actual_Roll"))
        status = str(row.get("Review_Status") or "").strip().lower()
        tracklet_id = str(row.get("Tracklet_ID") or "").strip()
        checkpoint = str(row.get("Checkpoint_ID") or "").strip().upper()
        accepted = status == "identified" and predicted and actual == predicted
        mixed = status == "mixed_track"
        evidence_key = (tracklet_id, checkpoint, predicted)
        evidence_signature = reviewed_tracklet_signatures.get(evidence_key)
        if not evidence_signature:
            raise ReviewCarryForwardError(
                "Reviewed evidence no longer maps to the frozen Phase 2G tracklet: " + "|".join(evidence_key)
            )
        evidence.append(
            {
                "tracklet_id": tracklet_id,
                "checkpoint_id": checkpoint,
                "predicted_roll": predicted,
                "review_status": status,
                "actual_roll": actual,
                "accepted_for_carry_forward": bool(accepted),
                "mixed_track_quarantine": bool(mixed),
                "ground_truth_source": str(row.get("Ground_Truth_Source") or ""),
                "review_id": str(row.get("Review_ID") or ""),
                "evidence_signature_sha256": evidence_signature,
            }
        )
    mixed = [item for item in evidence if item["mixed_track_quarantine"]]
    if len(mixed) != 1 or mixed[0]["tracklet_id"] != MIXED_TRACKLET_ID:
        raise ReviewCarryForwardError("Known mixed-track negative control changed")
    fingerprint = source_fingerprint(
        video_dir=video_dir,
        embeddings_path=embeddings_path,
        session_id=SESSION_ID,
        input_source_path=input_source_path,
    )
    registry_payload = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "registry_id": f"review-registry-{fingerprint['fingerprint_sha256'][:24]}",
        "session_id": SESSION_ID,
        "source_fingerprint": fingerprint,
        "review_source": f"phase_2h_multiframe_recovery/{PHASE_2H_OUTPUT_ID}",
        "recovery_rule_id": RECOVERY_RULE_ID,
        "reviewed_pairs": len(evidence),
        "accepted_pairs": sum(1 for item in evidence if item["accepted_for_carry_forward"]),
        "mixed_track_pairs": sum(1 for item in evidence if item["mixed_track_quarantine"]),
        "evidence": evidence,
    }
    registry_payload["registry_sha256"] = _canonical_json_hash({k: v for k, v in registry_payload.items() if k != "registry_sha256"})
    return registry_payload


def verify_registry(payload: Mapping[str, Any]) -> dict[str, Any]:
    registry = dict(payload)
    if registry.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise ReviewCarryForwardError("Unsupported review registry schema")
    if registry.get("policy_version") != POLICY_VERSION:
        raise ReviewCarryForwardError("Review carry-forward policy changed")
    recorded = str(registry.get("registry_sha256") or "")
    actual = _canonical_json_hash({k: v for k, v in registry.items() if k != "registry_sha256"})
    if recorded != actual:
        raise ReviewCarryForwardError("Review registry hash changed")
    evidence = registry.get("evidence")
    if not isinstance(evidence, list) or len(evidence) != 31:
        raise ReviewCarryForwardError("Review registry evidence count changed")
    signatures = [
        str(item.get("evidence_signature_sha256") or "")
        for item in evidence
        if isinstance(item, Mapping)
    ]
    if len(signatures) != 31 or any(len(value) != 64 for value in signatures):
        raise ReviewCarryForwardError("Review registry evidence signatures are incomplete")
    mixed = [item for item in evidence if isinstance(item, Mapping) and bool(item.get("mixed_track_quarantine"))]
    if len(mixed) != 1 or str(mixed[0].get("tracklet_id") or "") != MIXED_TRACKLET_ID:
        raise ReviewCarryForwardError("Mixed-track quarantine record changed")
    return registry


def load_registry(path: Path) -> dict[str, Any] | None:
    registry_path = Path(path)
    if not registry_path.is_file():
        return None
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise ReviewCarryForwardError(f"Could not read review registry: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReviewCarryForwardError("Review registry must contain a JSON object")
    return verify_registry(payload)


def apply_exact_review_carry_forward(
    rows: Sequence[Mapping[str, Any]],
    *,
    tracklet_frame: pd.DataFrame,
    registry: Mapping[str, Any],
    actual_source_fingerprint: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    verified = verify_registry(registry)
    if str(verified.get("session_id") or "") != str(actual_source_fingerprint.get("session_id") or ""):
        return [dict(row) for row in rows], {"applied": False, "reason": "session_not_registered"}
    expected_fingerprint = str((verified.get("source_fingerprint") or {}).get("fingerprint_sha256") or "")
    actual_fingerprint = str(actual_source_fingerprint.get("fingerprint_sha256") or "")
    if not expected_fingerprint or expected_fingerprint != actual_fingerprint:
        return [dict(row) for row in rows], {"applied": False, "reason": "source_or_embedding_fingerprint_changed"}

    available = _tracklet_evidence_map(tracklet_frame)
    reviewed_checkpoints: dict[str, set[str]] = {}
    mixed_rejections: dict[str, set[str]] = {}
    missing_pairs: list[tuple[str, str, str]] = []
    changed_pairs: list[tuple[str, str, str]] = []
    for item in verified.get("evidence") or []:
        if not isinstance(item, Mapping):
            continue
        key = (
            str(item.get("tracklet_id") or "").strip(),
            str(item.get("checkpoint_id") or "").strip().upper(),
            canonical_roll(item.get("predicted_roll")),
        )
        if key not in available:
            missing_pairs.append(key)
            continue
        expected_signature = str(item.get("evidence_signature_sha256") or "")
        if available[key] != expected_signature:
            changed_pairs.append(key)
            continue
        roll = key[2]
        checkpoint = key[1]
        if bool(item.get("mixed_track_quarantine")):
            mixed_rejections.setdefault(roll, set()).add(checkpoint)
        elif bool(item.get("accepted_for_carry_forward")):
            reviewed_checkpoints.setdefault(roll, set()).add(checkpoint)
    if missing_pairs:
        return [dict(row) for row in rows], {
            "applied": False,
            "reason": "reviewed_evidence_drift",
            "missing_pair_count": len(missing_pairs),
            "missing_pairs": ["|".join(pair) for pair in missing_pairs[:10]],
        }
    if changed_pairs:
        return [dict(row) for row in rows], {
            "applied": False,
            "reason": "reviewed_evidence_signature_drift",
            "changed_pair_count": len(changed_pairs),
            "changed_pairs": ["|".join(pair) for pair in changed_pairs[:10]],
        }

    output: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        roll = canonical_roll(row.get("Roll_Number") or row.get("roll"))
        current_status = str(row.get("Final_Status") or row.get("Status") or "Unconfirmed").strip()
        strict_count = int(float(str(row.get("Recognized_Checkpoints") or 0)))
        strict_checkpoints = [
            str(row.get(f"CP{index}_{'11_50' if index == 1 else '12_00' if index == 2 else '12_10' if index == 3 else '12_20' if index == 4 else '12_30'}") or "").strip()
            for index in range(1, 6)
        ]
        reviewed = sorted(reviewed_checkpoints.get(roll, set()))
        mixed = sorted(mixed_rejections.get(roll, set()))
        if current_status == "Missing Enrollment":
            new_status = current_status
        elif len(reviewed) >= 3:
            new_status = "Present"
        elif len(reviewed) == 2:
            new_status = "Needs Review"
        else:
            new_status = current_status
        if len(reviewed) >= 2:
            row["Recognized_Checkpoints"] = str(len(reviewed))
            row["Checkpoint_Attendance_Score"] = f"{len(reviewed) / 5 * 100:.1f}"
        row.update(
            {
                "Status": new_status,
                "Final_Status": new_status,
                "Present": "Yes" if new_status == "Present" else "No",
                "Strict_Recognized_Checkpoints": str(strict_count),
                "Strict_Checkpoint_Labels": "; ".join(value for value in strict_checkpoints if value),
                "Reviewed_Tracklet_Checkpoints": "; ".join(reviewed),
                "Reviewed_Tracklet_Checkpoint_Count": str(len(reviewed)),
                "Mixed_Track_Checkpoints_Rejected": "; ".join(mixed),
                "Review_Carry_Forward_Applied": "Yes",
                "Review_Registry_ID": str(verified.get("registry_id") or ""),
                "Review_Carry_Forward_Policy": POLICY_VERSION,
            }
        )
        if new_status == "Present" and len(reviewed) >= 3:
            row["Evidence_Interpretation"] = "Confirmed by exact-source reviewed multi-frame evidence carried forward safely."
            row["Flags"] = "Reviewed Evidence Carried Forward"
        elif new_status == "Needs Review" and len(reviewed) == 2:
            row["Evidence_Interpretation"] = "Two exact-source reviewed checkpoints are confirmed; explicit attendance review is still required."
            row["Flags"] = "Reviewed Evidence Carried Forward"
        if mixed:
            note = f"Mixed track rejected automatically at {', '.join(mixed)}."
            row["Evidence_Interpretation"] = (str(row.get("Evidence_Interpretation") or "").strip() + " " + note).strip()
        output.append(row)

    counts: dict[str, int] = {}
    for row in output:
        status = str(row.get("Final_Status") or row.get("Status") or "").strip()
        counts[status] = counts.get(status, 0) + 1
    return output, {
        "applied": True,
        "registry_id": verified.get("registry_id"),
        "reviewed_pairs": len(verified.get("evidence") or []),
        "mixed_track_pairs_rejected": sum(1 for item in verified.get("evidence") or [] if bool(item.get("mixed_track_quarantine"))),
        "status_counts": counts,
    }
