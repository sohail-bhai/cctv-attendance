from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.face_attendance.tracklet_review import (  # noqa: E402
    LabelValidationError,
    ReviewExportError,
    evaluate_review_package,
    export_label_correction_package,
    export_review_package_from_diagnostics,
    merge_label_corrections,
)
from src.face_attendance.recall_analysis import (  # noqa: E402
    RecallAnalysisError,
    UnresolvedSelectionConfig,
    evaluate_unresolved_review_package,
    run_recall_gap_analysis,
)
from src.face_attendance.roster_integrity import (  # noqa: E402
    RosterIntegrityError,
    audit_subject_roster,
    write_roster_audit,
)
from src.face_attendance.tracklet_calibration import (  # noqa: E402
    TrackletCalibrationError,
    run_tracklet_calibration,
)
from src.face_attendance.shadow_validation import (  # noqa: E402
    EXPECTED_ROSTER_COUNT,
    ShadowValidationError,
    build_baseline_command,
    evaluate_shadow_review_package,
    preflight_shadow_session,
    run_shadow_session,
    serve_review_package,
)
from src.face_attendance.embedding_forensics import (  # noqa: E402
    BenchmarkSource,
    EmbeddingForensicsError,
    regenerate_forensic_review_packages,
    run_embedding_forensics,
    verify_reusable_forensic_run,
)
from src.face_attendance.embedding_lifecycle import (  # noqa: E402
    EmbeddingLifecycleError,
    build_versioned_embeddings,
    evaluate_embedding_version,
    promote_embedding_version,
    rollback_embedding_version,
    validate_embedding_approval_bundle,
)
from src.face_attendance.embedding_version_family import (  # noqa: E402
    EmbeddingFamilyError,
    build_and_evaluate_embedding_family,
    preflight_embedding_family_inputs,
    reconcile_identity_correction,
    reconcile_shruthi_source,
    show_embedding_family,
    validate_compact_approval_bundle,
    verify_embedding_family,
)
from src.face_attendance.compact_forensic_review import (  # noqa: E402
    CompactReviewError,
    generate_compact_forensic_reviews,
    verify_reusable_compact_review,
)
from src.face_attendance.forensic_review import (  # noqa: E402
    ForensicReviewError,
    serve_forensic_review_package,
)


REFERENCE_DIAGNOSTIC = (
    ROOT
    / "attendance_output"
    / "diagnostics"
    / "2026-06-30__B51__P1__CVO_20260712_000753_991890"
)
FROZEN_CANDIDATE_DIR = (
    REFERENCE_DIAGNOSTIC
    / "calibration"
    / "tracklet_calibration_20260712_175448"
)
TUE_P2_VIDEO_DIR = ROOT / "cctv_videos" / "prepared_slots" / "2026-06-30" / "TUE_P2"
P1_ACCEPTED_REVIEW_PACKAGE = (
    REFERENCE_DIAGNOSTIC
    / "tracklet_review_2026-06-30__B51__P1__CVO_20260712_000753_991890-20260712074431-763c4d"
)
P1_UNRESOLVED_REVIEW_PACKAGE = (
    REFERENCE_DIAGNOSTIC
    / "recall_analysis_20260712_124500_128763"
    / "unresolved_tracklet_review_2026-06-30__B51__P1__CVO_20260712_000753_991890-20260712124500-1f3271"
)
P2_DIAGNOSTIC = (
    ROOT
    / "attendance_output"
    / "diagnostics"
    / "2026-06-30__B51__P2__CVO_20260712_221105_286649"
)
P2_SHADOW_REVIEW_PACKAGE = (
    P2_DIAGNOSTIC
    / "shadow_validation"
    / "shadow-validation_20260713_084319_043368"
    / "shadow_recovery_review_2026-06-30__B51__P2__CVO_20260712_221105_286649-20260713084319-55466e"
)
P2_COMPLETED_LABELS = Path(
    r"D:\Downloads\tracklet_review_labels_2026-06-30__B51__P2__CVO_20260712_221105_286649-20260713084319-55466e.csv"
)
P1_GROUND_TRUTH_MANIFEST = FROZEN_CANDIDATE_DIR / "ground_truth_manifest.json"
PRIMARY_EMBEDDING_FORENSIC_RUN = (
    ROOT
    / "attendance_output"
    / "embedding_forensics"
    / "embedding_forensics_20260713_132605_989269"
)
REVIEWFIX_EMBEDDING_FORENSIC_RUN = (
    ROOT
    / "attendance_output"
    / "embedding_forensics"
    / "embedding_forensics_reviewfix_20260713_140159_455110"
)
COMPACT_EMBEDDING_FORENSIC_RUN = (
    ROOT
    / "attendance_output"
    / "embedding_forensics"
    / "embedding_forensics_compact_review_20260713_161633_550718"
)
COMPACT_ENROLLMENT_REVIEW_PACKAGE = (
    COMPACT_EMBEDDING_FORENSIC_RUN / "compact_enrollment_review"
)
COMPACT_CCTV_REVIEW_PACKAGE = COMPACT_EMBEDDING_FORENSIC_RUN / "compact_cctv_review"
FINAL_COMPACT_ENROLLMENT_APPROVALS = Path(
    r"D:\Downloads\forensic_review_approvals_compact-enrollment-audit-df0b986b716a2cd5.csv"
)
FINAL_COMPACT_CCTV_APPROVALS = Path(
    r"D:\Downloads\forensic_review_approvals_compact-verified-cctv-8785d6b28789ae6f.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export blind accepted-tracklet evidence or evaluate completed ground-truth labels."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Build a blind package from an existing diagnostic run")
    export_parser.add_argument("--diagnostic-run", required=True, help="Phase 1.2D diagnostic run directory")
    export_parser.add_argument("--video-dir", default="", help="Source video root; inferred from diagnostics when omitted")
    export_parser.add_argument("--output-dir", default="", help="Package parent directory; defaults to the diagnostic run")
    export_parser.add_argument("--student-map", default=str(ROOT / "data" / "student_faculty_map.json"))
    export_parser.add_argument("--subject-abbr", default="", help="Subject roster key; inferred from diagnostics when omitted")
    export_parser.add_argument("--evidence-count", type=int, default=5, help="Maximum selected views per accepted tracklet")
    export_parser.add_argument("--crop-padding", type=float, default=0.45, help="Face-box padding ratio for review crops")

    evaluate_parser = subparsers.add_parser("evaluate", help="Join final labels with hidden predictions and calculate metrics")
    evaluate_parser.add_argument("--review-package", required=True, help="Review package root containing reviewer/ and private/")
    evaluate_parser.add_argument("--labels", required=True, help="Completed labels CSV exported by the blind reviewer")
    evaluate_parser.add_argument("--actual-present", default="", help="Optional actual-present TXT or CSV roster")
    evaluate_parser.add_argument("--output-dir", default="", help="Validation output directory; defaults inside the package")
    evaluate_parser.add_argument("--present-checkpoints", type=int, default=3, help="Checkpoint count used for candidate attendance metrics")

    recall_parser = subparsers.add_parser(
        "recall-gap",
        help="Analyze actual-present recall gaps and export a balanced blind unresolved-track package",
    )
    recall_parser.add_argument("--diagnostic-run", required=True)
    recall_parser.add_argument("--accepted-review-package", required=True)
    recall_parser.add_argument("--accepted-labels", required=True)
    recall_parser.add_argument("--actual-present", required=True)
    recall_parser.add_argument("--student-map", default=str(ROOT / "data" / "student_faculty_map.json"))
    recall_parser.add_argument("--embedding-summary", default=str(ROOT / "models" / "embedding_summary.csv"))
    recall_parser.add_argument("--embeddings", default=str(ROOT / "models" / "student_embeddings.pkl"))
    recall_parser.add_argument("--video-dir", default="", help="Source video root; inferred from diagnostics when omitted")
    recall_parser.add_argument("--output-dir", default="", help="New recall-analysis directory; defaults inside the diagnostic run")
    recall_parser.add_argument("--subject-abbr", default="CVO")
    recall_parser.add_argument("--target-unresolved", type=int, default=70)
    recall_parser.add_argument("--candidate-cap", type=int, default=6)
    recall_parser.add_argument("--checkpoint-cap", type=int, default=0)
    recall_parser.add_argument("--camera-cap", type=int, default=0)
    recall_parser.add_argument("--zone-cap", type=int, default=0)

    unresolved_parser = subparsers.add_parser(
        "evaluate-unresolved",
        help="Join unresolved blind-review labels and calculate recall-recovery evidence",
    )
    unresolved_parser.add_argument("--review-package", required=True)
    unresolved_parser.add_argument("--labels", required=True)
    unresolved_parser.add_argument("--recall-analysis", required=True)
    unresolved_parser.add_argument("--accepted-review-package", required=True)
    unresolved_parser.add_argument("--accepted-labels", required=True)
    unresolved_parser.add_argument("--output-dir", default="")
    unresolved_parser.add_argument(
        "--student-map",
        default=str(ROOT / "data" / "student_faculty_map.json"),
        help="Current authoritative student mapping used after roster repair",
    )
    unresolved_parser.add_argument("--subject-abbr", default="CVO")
    unresolved_parser.add_argument(
        "--actual-present",
        default="",
        help=(
            "Current actual-present roster. When supplied, roster-sensitive context is recomputed "
            "from existing diagnostics without rerunning recognition."
        ),
    )
    unresolved_parser.add_argument(
        "--embedding-summary",
        default=str(ROOT / "models" / "embedding_summary.csv"),
    )
    unresolved_parser.add_argument(
        "--embeddings",
        default=str(ROOT / "models" / "student_embeddings.pkl"),
    )

    calibration_parser = subparsers.add_parser(
        "calibrate-tracklets",
        help=(
            "Freeze reviewed accepted/unresolved tracklets and run conservative identity-grouped "
            "shadow calibration without changing production recognition"
        ),
    )
    calibration_parser.add_argument("--diagnostic-run", required=True)
    calibration_parser.add_argument("--accepted-review-package", required=True)
    calibration_parser.add_argument("--accepted-labels", required=True)
    calibration_parser.add_argument("--unresolved-review-package", required=True)
    calibration_parser.add_argument("--unresolved-labels", required=True)
    calibration_parser.add_argument("--student-map", default=str(ROOT / "data" / "student_faculty_map.json"))
    calibration_parser.add_argument("--subject-abbr", default="CVO")
    calibration_parser.add_argument("--output-dir", default="")
    calibration_parser.add_argument("--folds", type=int, default=5)
    calibration_parser.add_argument("--min-oof-recoveries", type=int, default=3)
    calibration_parser.add_argument("--min-recovery-folds", type=int, default=3)

    correction_export_parser = subparsers.add_parser(
        "export-corrections",
        help="Build a blind correction-only package from previously unmapped review cards",
    )
    correction_export_parser.add_argument("--review-package", required=True)
    correction_export_parser.add_argument("--labels", required=True)
    correction_export_parser.add_argument("--student-map", default=str(ROOT / "data" / "student_faculty_map.json"))
    correction_export_parser.add_argument("--subject-abbr", default="CVO")
    correction_export_parser.add_argument("--target-status", default="not_in_mapping")
    correction_export_parser.add_argument("--output-dir", default="")

    correction_merge_parser = subparsers.add_parser(
        "merge-corrections",
        help="Safely merge a completed correction package into a new labels CSV",
    )
    correction_merge_parser.add_argument("--correction-package", required=True)
    correction_merge_parser.add_argument("--corrections", required=True)
    correction_merge_parser.add_argument("--source-labels", required=True)
    correction_merge_parser.add_argument("--output", required=True)

    roster_audit_parser = subparsers.add_parser(
        "audit-roster",
        help="Validate subject-roster integrity and report embedding coverage without changing embeddings",
    )
    roster_audit_parser.add_argument("--student-map", default=str(ROOT / "data" / "student_faculty_map.json"))
    roster_audit_parser.add_argument("--subject-abbr", default="CVO")
    roster_audit_parser.add_argument("--actual-present", required=True)
    roster_audit_parser.add_argument("--embedding-summary", default=str(ROOT / "models" / "embedding_summary.csv"))
    roster_audit_parser.add_argument("--embeddings", default=str(ROOT / "models" / "student_embeddings.pkl"))
    roster_audit_parser.add_argument("--output-dir", default="")

    shadow_parser = subparsers.add_parser(
        "run-shadow-session",
        help="Run the frozen Phase 1.2H candidate against untouched footage in diagnostic-only mode",
    )
    shadow_parser.add_argument("--video-dir", default=str(TUE_P2_VIDEO_DIR))
    shadow_parser.add_argument("--session-id", default="2026-06-30__B51__P2__CVO")
    shadow_parser.add_argument("--subject-abbr", default="CVO")
    shadow_parser.add_argument("--student-map", default=str(ROOT / "data" / "student_faculty_map.json"))
    shadow_parser.add_argument(
        "--candidate-config",
        default=str(FROZEN_CANDIDATE_DIR / "candidate_config.json"),
    )
    shadow_parser.add_argument(
        "--candidate-manifest",
        default=str(FROZEN_CANDIDATE_DIR / "output_manifest.json"),
    )
    shadow_parser.add_argument(
        "--reference-diagnostic-run",
        default=str(REFERENCE_DIAGNOSTIC),
    )
    shadow_parser.add_argument(
        "--timetable",
        default=str(ROOT / "timetable_b51_2026_2027.csv"),
    )
    shadow_parser.add_argument(
        "--embeddings",
        default=str(ROOT / "models" / "student_embeddings.pkl"),
    )
    shadow_parser.add_argument(
        "--diagnostic-run",
        default="",
        help="Reuse an existing compatible diagnostic-only TUE_P2 run instead of running recognition",
    )
    shadow_parser.add_argument("--output-dir", default="")
    shadow_parser.add_argument("--expected-roster-count", type=int, default=EXPECTED_ROSTER_COUNT)
    shadow_parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Verify all frozen inputs and print the exact diagnostic command without running it",
    )

    open_shadow_parser = subparsers.add_parser(
        "open-shadow-review",
        help="Serve only the blind reviewer subtree on loopback using a long-path-safe launcher",
    )
    open_shadow_parser.add_argument("--review-package", required=True)
    open_shadow_parser.add_argument("--port", type=int, default=0)
    open_shadow_parser.add_argument("--no-browser", action="store_true")

    evaluate_shadow_parser = subparsers.add_parser(
        "evaluate-shadow-session",
        help="Join completed blind labels with private shadow predictions and apply the reject policy",
    )
    evaluate_shadow_parser.add_argument("--review-package", required=True)
    evaluate_shadow_parser.add_argument("--labels", required=True)
    evaluate_shadow_parser.add_argument("--output-dir", default="")

    forensic_parser = subparsers.add_parser(
        "embedding-forensics",
        help="Run Phase 1.2I-A multisession identity, embedding, and dataset forensics",
    )
    forensic_parser.add_argument("--candidate-config", default=str(FROZEN_CANDIDATE_DIR / "candidate_config.json"))
    forensic_parser.add_argument("--calibration-dir", default=str(FROZEN_CANDIDATE_DIR))
    forensic_parser.add_argument("--p1-diagnostic-run", default=str(REFERENCE_DIAGNOSTIC))
    forensic_parser.add_argument("--p1-accepted-review-package", default=str(P1_ACCEPTED_REVIEW_PACKAGE))
    forensic_parser.add_argument("--p1-accepted-labels", default=str(ROOT / "tracklet_labels_TUE_P1.csv"))
    forensic_parser.add_argument("--p1-unresolved-review-package", default=str(P1_UNRESOLVED_REVIEW_PACKAGE))
    forensic_parser.add_argument("--p1-unresolved-labels", default=str(ROOT / "unresolved_labels_TUE_P1_corrected.csv"))
    forensic_parser.add_argument("--p2-diagnostic-run", default=str(P2_DIAGNOSTIC))
    forensic_parser.add_argument("--p2-shadow-review-package", default=str(P2_SHADOW_REVIEW_PACKAGE))
    forensic_parser.add_argument("--p2-shadow-labels", default=str(P2_COMPLETED_LABELS))
    forensic_parser.add_argument("--student-map", default=str(ROOT / "data" / "student_faculty_map.json"))
    forensic_parser.add_argument("--subject-abbr", default="CVO")
    forensic_parser.add_argument("--embeddings", default=str(ROOT / "models" / "student_embeddings.pkl"))
    forensic_parser.add_argument("--embedding-summary", default=str(ROOT / "models" / "embedding_summary.csv"))
    forensic_parser.add_argument("--output-dir", default="")
    forensic_parser.add_argument(
        "--reuse-output",
        default="",
        help="Verify and reuse a completed forensic run only when all protected source hashes still match",
    )
    forensic_parser.add_argument(
        "--no-augmented-dataset",
        action="store_true",
        help="Skip the optional nonproduction augmented-dataset audit",
    )

    regenerate_parser = subparsers.add_parser(
        "regenerate-forensic-review-packages",
        help="Create fresh reviewer packages from an integrity-checked forensic run without rerunning models",
    )
    regenerate_parser.add_argument("--source-forensic-run", required=True)
    regenerate_parser.add_argument("--candidate-config", default=str(FROZEN_CANDIDATE_DIR / "candidate_config.json"))
    regenerate_parser.add_argument("--embeddings", default=str(ROOT / "models" / "student_embeddings.pkl"))
    regenerate_parser.add_argument("--embedding-summary", default=str(ROOT / "models" / "embedding_summary.csv"))
    regenerate_parser.add_argument("--student-map", default=str(ROOT / "data" / "student_faculty_map.json"))
    regenerate_parser.add_argument("--subject-abbr", default="CVO")
    regenerate_parser.add_argument("--output-dir", default="")
    regenerate_parser.add_argument("--package-revision", default="reviewer-v5-final-runtime-fix")

    compact_parser = subparsers.add_parser(
        "generate-compact-forensic-reviews",
        help="Derive small deterministic priority reviewers from integrity-checked Phase 1.2I-A outputs",
    )
    compact_parser.add_argument(
        "--source-forensic-run", default=str(PRIMARY_EMBEDDING_FORENSIC_RUN)
    )
    compact_parser.add_argument(
        "--source-reviewfix-run", default=str(REVIEWFIX_EMBEDDING_FORENSIC_RUN)
    )
    compact_parser.add_argument(
        "--candidate-config", default=str(FROZEN_CANDIDATE_DIR / "candidate_config.json")
    )
    compact_parser.add_argument(
        "--embeddings", default=str(ROOT / "models" / "student_embeddings.pkl")
    )
    compact_parser.add_argument(
        "--embedding-summary", default=str(ROOT / "models" / "embedding_summary.csv")
    )
    compact_parser.add_argument(
        "--student-map", default=str(ROOT / "data" / "student_faculty_map.json")
    )
    compact_parser.add_argument(
        "--candidate-registry",
        default=str(
            ROOT
            / "models"
            / "candidate_registry"
            / "rejected"
            / "cal-5cd35b60dd83.json"
        ),
    )
    compact_parser.add_argument("--subject-abbr", default="CVO")
    compact_parser.add_argument("--enrollment-maximum", type=int, default=60)
    compact_parser.add_argument("--cctv-maximum", type=int, default=40)
    compact_parser.add_argument("--output-dir", default="")
    compact_parser.add_argument("--package-revision", default="compact-priority-review-v1")
    compact_parser.add_argument(
        "--reuse-output",
        default="",
        help="Explicitly reuse only after validating output, source, broad-package, and protected hashes",
    )

    open_compact_enrollment_parser = subparsers.add_parser(
        "open-compact-enrollment-review",
        help="Serve a compact enrollment reviewer on 127.0.0.1 only",
    )
    open_compact_enrollment_parser.add_argument("--review-package", required=True)
    open_compact_enrollment_parser.add_argument("--port", type=int, default=0)
    open_compact_enrollment_parser.add_argument("--no-browser", action="store_true")

    open_compact_cctv_parser = subparsers.add_parser(
        "open-compact-cctv-review",
        help="Serve a compact human-confirmed CCTV reviewer on 127.0.0.1 only",
    )
    open_compact_cctv_parser.add_argument("--review-package", required=True)
    open_compact_cctv_parser.add_argument("--port", type=int, default=0)
    open_compact_cctv_parser.add_argument("--no-browser", action="store_true")

    open_enrollment_parser = subparsers.add_parser(
        "open-enrollment-audit-review",
        help="Serve an enrollment-audit reviewer on loopback",
    )
    open_enrollment_parser.add_argument("--review-package", required=True)
    open_enrollment_parser.add_argument("--port", type=int, default=0)
    open_enrollment_parser.add_argument("--no-browser", action="store_true")

    open_cctv_parser = subparsers.add_parser(
        "open-cctv-enrollment-review",
        help="Serve a verified-CCTV-crop reviewer on loopback",
    )
    open_cctv_parser.add_argument("--review-package", required=True)
    open_cctv_parser.add_argument("--port", type=int, default=0)
    open_cctv_parser.add_argument("--no-browser", action="store_true")

    approval_parser = subparsers.add_parser(
        "validate-enrollment-approvals",
        help="Validate both completed Phase 1.2I-A human approval CSVs without building embeddings",
    )
    approval_parser.add_argument("--enrollment-review-package", required=True)
    approval_parser.add_argument("--enrollment-approvals", required=True)
    approval_parser.add_argument("--cctv-review-package", required=True)
    approval_parser.add_argument("--cctv-approvals", required=True)

    build_version_parser = subparsers.add_parser(
        "build-versioned-embeddings",
        help="Build an immutable built_unapproved embedding version after both approval files validate",
    )
    build_version_parser.add_argument("--forensic-run", required=True)
    build_version_parser.add_argument("--enrollment-review-package", required=True)
    build_version_parser.add_argument("--enrollment-approvals", required=True)
    build_version_parser.add_argument("--cctv-review-package", required=True)
    build_version_parser.add_argument("--cctv-approvals", required=True)
    build_version_parser.add_argument("--production-embeddings", default=str(ROOT / "models" / "student_embeddings.pkl"))
    build_version_parser.add_argument("--production-summary", default=str(ROOT / "models" / "embedding_summary.csv"))
    build_version_parser.add_argument("--versions-root", default=str(ROOT / "models" / "versions"))
    build_version_parser.add_argument("--version-id", default="")
    build_version_parser.add_argument("--candidate-registry", default=str(ROOT / "models" / "candidate_registry" / "rejected"))
    build_version_parser.add_argument("--source-candidate-id", action="append", default=[])

    evaluate_version_parser = subparsers.add_parser(
        "evaluate-embedding-version",
        help="Attach integrity-checked frozen-benchmark and optional untouched-session evidence to a version",
    )
    evaluate_version_parser.add_argument("--version-dir", required=True)
    evaluate_version_parser.add_argument("--forensic-run", required=True)
    evaluate_version_parser.add_argument("--regression-results", required=True)
    evaluate_version_parser.add_argument("--untouched-validation", default="")

    promote_version_parser = subparsers.add_parser(
        "promote-embedding-version",
        help="Explicitly promote only after passed regression and untouched MON_P3 validation",
    )
    promote_version_parser.add_argument("--version-dir", required=True)
    promote_version_parser.add_argument("--production-embeddings", default=str(ROOT / "models" / "student_embeddings.pkl"))
    promote_version_parser.add_argument("--production-summary", default=str(ROOT / "models" / "embedding_summary.csv"))
    promote_version_parser.add_argument("--candidate-registry", default=str(ROOT / "models" / "candidate_registry" / "rejected"))
    promote_version_parser.add_argument(
        "--confirm-version-id",
        required=True,
        help="Must exactly match the version directory name",
    )

    rollback_version_parser = subparsers.add_parser(
        "rollback-embedding-version",
        help="Restore the integrity-checked pre-promotion production backup for a version",
    )
    rollback_version_parser.add_argument("--version-dir", required=True)
    rollback_version_parser.add_argument(
        "--confirm-version-id",
        required=True,
        help="Must exactly match the version directory name",
    )

    compact_approval_parser = subparsers.add_parser(
        "validate-compact-approvals",
        help="Validate the final Phase 1.2I-A.1 compact approvals and package provenance",
    )
    compact_approval_parser.add_argument(
        "--enrollment-review-package", default=str(COMPACT_ENROLLMENT_REVIEW_PACKAGE)
    )
    compact_approval_parser.add_argument(
        "--enrollment-approvals", default=str(FINAL_COMPACT_ENROLLMENT_APPROVALS)
    )
    compact_approval_parser.add_argument(
        "--cctv-review-package", default=str(COMPACT_CCTV_REVIEW_PACKAGE)
    )
    compact_approval_parser.add_argument(
        "--cctv-approvals", default=str(FINAL_COMPACT_CCTV_APPROVALS)
    )
    compact_approval_parser.add_argument(
        "--dataset", default=str(ROOT / "dataset")
    )

    correction_parser = subparsers.add_parser(
        "reconcile-identity-correction",
        help="Validate the narrow CSE0110-to-AI0110 candidate-only ownership correction",
    )
    correction_parser.add_argument(
        "--enrollment-review-package", default=str(COMPACT_ENROLLMENT_REVIEW_PACKAGE)
    )
    correction_parser.add_argument(
        "--enrollment-approvals", default=str(FINAL_COMPACT_ENROLLMENT_APPROVALS)
    )
    correction_parser.add_argument(
        "--cctv-review-package", default=str(COMPACT_CCTV_REVIEW_PACKAGE)
    )
    correction_parser.add_argument(
        "--cctv-approvals", default=str(FINAL_COMPACT_CCTV_APPROVALS)
    )
    correction_parser.add_argument("--dataset", default=str(ROOT / "dataset"))
    correction_parser.add_argument(
        "--historical-inventory",
        default=str(REVIEWFIX_EMBEDDING_FORENSIC_RUN / "dataset_image_inventory.csv"),
    )
    correction_parser.add_argument(
        "--embedding-outliers",
        default=str(REVIEWFIX_EMBEDDING_FORENSIC_RUN / "embedding_outliers.csv"),
    )
    correction_parser.add_argument(
        "--production-embeddings", default=str(ROOT / "models" / "student_embeddings.pkl")
    )
    correction_parser.add_argument(
        "--student-map", default=str(ROOT / "data" / "student_faculty_map.json")
    )

    family_parser = subparsers.add_parser(
        "build-evaluate-embedding-family",
        help="Build all Phase 1.2I-B variants and run leakage-safe frozen-benchmark evaluation",
    )
    family_parser.add_argument(
        "--forensic-run", default=str(REVIEWFIX_EMBEDDING_FORENSIC_RUN)
    )
    family_parser.add_argument(
        "--enrollment-review-package", default=str(COMPACT_ENROLLMENT_REVIEW_PACKAGE)
    )
    family_parser.add_argument(
        "--enrollment-approvals", default=str(FINAL_COMPACT_ENROLLMENT_APPROVALS)
    )
    family_parser.add_argument(
        "--cctv-review-package", default=str(COMPACT_CCTV_REVIEW_PACKAGE)
    )
    family_parser.add_argument(
        "--cctv-approvals", default=str(FINAL_COMPACT_CCTV_APPROVALS)
    )
    family_parser.add_argument(
        "--production-embeddings", default=str(ROOT / "models" / "student_embeddings.pkl")
    )
    family_parser.add_argument(
        "--production-summary", default=str(ROOT / "models" / "embedding_summary.csv")
    )
    family_parser.add_argument(
        "--student-map", default=str(ROOT / "data" / "student_faculty_map.json")
    )
    family_parser.add_argument("--dataset", default=str(ROOT / "dataset"))
    family_parser.add_argument(
        "--augmented-dataset", default=str(ROOT / "augmented_dataset")
    )
    family_parser.add_argument(
        "--versions-root", default=str(ROOT / "models" / "versions")
    )
    family_parser.add_argument("--family-id", default="")

    family_preflight_parser = subparsers.add_parser(
        "preflight-embedding-family",
        help=(
            "Run all read-only Phase 1.2I-B approval, identity, roster, benchmark, "
            "and protected-state checks without extracting faces or creating a version"
        ),
    )
    family_preflight_parser.add_argument(
        "--forensic-run", default=str(REVIEWFIX_EMBEDDING_FORENSIC_RUN)
    )
    family_preflight_parser.add_argument(
        "--enrollment-review-package", default=str(COMPACT_ENROLLMENT_REVIEW_PACKAGE)
    )
    family_preflight_parser.add_argument(
        "--enrollment-approvals", default=str(FINAL_COMPACT_ENROLLMENT_APPROVALS)
    )
    family_preflight_parser.add_argument(
        "--cctv-review-package", default=str(COMPACT_CCTV_REVIEW_PACKAGE)
    )
    family_preflight_parser.add_argument(
        "--cctv-approvals", default=str(FINAL_COMPACT_CCTV_APPROVALS)
    )
    family_preflight_parser.add_argument(
        "--production-embeddings", default=str(ROOT / "models" / "student_embeddings.pkl")
    )
    family_preflight_parser.add_argument(
        "--production-summary", default=str(ROOT / "models" / "embedding_summary.csv")
    )
    family_preflight_parser.add_argument(
        "--student-map", default=str(ROOT / "data" / "student_faculty_map.json")
    )
    family_preflight_parser.add_argument("--dataset", default=str(ROOT / "dataset"))
    family_preflight_parser.add_argument(
        "--augmented-dataset", default=str(ROOT / "augmented_dataset")
    )
    family_preflight_parser.add_argument(
        "--output-json",
        default="",
        help="Optional path for a machine-readable preflight result",
    )

    verify_family_parser = subparsers.add_parser(
        "verify-embedding-family",
        help="Verify every recorded family, variant, pickle, evaluation, and protected-state hash",
    )
    verify_family_parser.add_argument("--family-dir", required=True)
    verify_family_parser.add_argument(
        "--production-embeddings", default=str(ROOT / "models" / "student_embeddings.pkl")
    )
    verify_family_parser.add_argument(
        "--production-summary", default=str(ROOT / "models" / "embedding_summary.csv")
    )

    show_family_parser = subparsers.add_parser(
        "show-embedding-family",
        help="Show the immutable family status, variants, coverage, and decision",
    )
    show_family_parser.add_argument("--family-dir", required=True)
    return parser.parse_args()


def print_shadow_preflight(preflight, command: list[str]) -> None:
    print("Phase 1.2H preflight: PASS")
    print(f"Session: {preflight.session_id}")
    print(f"Subject roster: {preflight.student_map_path}")
    print(f"Subject students: {len(preflight.subject_rolls)}")
    print(f"Roster SHA-256: {preflight.roster_sha256}")
    print(f"Frozen candidate config: {preflight.frozen_candidate.config_path}")
    print(f"Frozen candidate manifest: {preflight.frozen_candidate.manifest_path}")
    print(f"Candidate ID: {preflight.frozen_candidate.candidate.candidate_id}")
    print(f"Candidate SHA-256: {preflight.frozen_candidate.config_sha256}")
    print(f"Reference diagnostic: {preflight.reference_diagnostic_run}")
    print(f"Prepared footage: {preflight.video_root}")
    for video in preflight.videos:
        print(
            f"  {video['checkpoint_id']} {video['source_id']}: {video['source_path']} "
            f"({video['frame_count']} frames, {video['sha256']})"
        )
    print("Exact diagnostic command:")
    print("  " + subprocess.list2cmdline(command))


def forensic_benchmark_sources(args: argparse.Namespace) -> list[BenchmarkSource]:
    return [
        BenchmarkSource(
            source_kind="tue_p1_accepted_review",
            package_dir=Path(args.p1_accepted_review_package),
            labels_path=Path(args.p1_accepted_labels),
            diagnostic_run=Path(args.p1_diagnostic_run),
            expected_rows=37,
            baseline_status="baseline_accepted_human_reviewed",
            expected_labels_sha256="185c9d4e09ff97e5cc1cd99f04e08f090b08dd58efe6dd391ab4e5aa1a1eee4e",
            trusted_manifest_path=P1_GROUND_TRUTH_MANIFEST,
        ),
        BenchmarkSource(
            source_kind="tue_p1_corrected_unresolved_review",
            package_dir=Path(args.p1_unresolved_review_package),
            labels_path=Path(args.p1_unresolved_labels),
            diagnostic_run=Path(args.p1_diagnostic_run),
            expected_rows=63,
            baseline_status="baseline_unresolved_human_reviewed",
            expected_labels_sha256="ce850c74550e9f8cc82a9468ce553b35c2a5f401e940b6073e03f412779fbb03",
            trusted_manifest_path=P1_GROUND_TRUTH_MANIFEST,
        ),
        BenchmarkSource(
            source_kind="tue_p2_shadow_recovery_review",
            package_dir=Path(args.p2_shadow_review_package),
            labels_path=Path(args.p2_shadow_labels),
            diagnostic_run=Path(args.p2_diagnostic_run),
            expected_rows=20,
            baseline_status="baseline_rejected_shadow_proposal_human_reviewed",
            expected_labels_sha256="cfd489d092256bbbff9822954c22cdafcce0aaa4823c7422957a44a3c7802ee8",
        ),
    ]


def print_embedding_forensic_inputs(args: argparse.Namespace, sources: list[BenchmarkSource]) -> None:
    print("Phase 1.2I-A resolved inputs:")
    print(f"  Candidate config: {Path(args.candidate_config).resolve()}")
    print(f"  Calibration directory: {Path(args.calibration_dir).resolve()}")
    for source in sources:
        print(f"  {source.source_kind} package: {Path(source.package_dir).resolve()}")
        print(f"  {source.source_kind} labels: {Path(source.labels_path).resolve()}")
        print(f"  {source.source_kind} diagnostics: {Path(source.diagnostic_run).resolve()}")
    print(f"  Student map: {Path(args.student_map).resolve()}")
    print(f"  Production embeddings (read-only): {Path(args.embeddings).resolve()}")
    print(f"  Production embedding summary (read-only): {Path(args.embedding_summary).resolve()}")
    if args.output_dir:
        print(f"  Requested output: {Path(args.output_dir).resolve()}")
    print(f"  Optional augmented dataset audit: {'no' if args.no_augmented_dataset else 'yes'}")


def main() -> None:
    args = parse_args()
    try:
        if args.command == "generate-compact-forensic-reviews":
            print("Phase 1.2I-A.1 compact review resolved inputs:")
            print(f"  Primary forensic run: {Path(args.source_forensic_run).resolve()}")
            print(f"  Reviewfix forensic run: {Path(args.source_reviewfix_run).resolve()}")
            print(f"  Candidate config (read-only): {Path(args.candidate_config).resolve()}")
            print(f"  Production embeddings (read-only): {Path(args.embeddings).resolve()}")
            print(f"  Production summary (read-only): {Path(args.embedding_summary).resolve()}")
            print(f"  Student map (read-only): {Path(args.student_map).resolve()}")
            print(f"  Rejected candidate registry: {Path(args.candidate_registry).resolve()}")
            print(f"  Enrollment hard maximum: {args.enrollment_maximum}")
            print(f"  CCTV hard maximum: {args.cctv_maximum}")
            print(f"  Package revision: {args.package_revision}")
            if args.output_dir:
                print(f"  Requested output: {Path(args.output_dir).resolve()}")
            if args.reuse_output:
                print(f"  Explicit reuse candidate: {Path(args.reuse_output).resolve()}")
                summary = verify_reusable_compact_review(
                    output_dir=Path(args.reuse_output),
                    repo_root=ROOT,
                    source_forensic_run=Path(args.source_forensic_run),
                    source_reviewfix_run=Path(args.source_reviewfix_run),
                    candidate_config_path=Path(args.candidate_config),
                    embeddings_path=Path(args.embeddings),
                    embedding_summary_path=Path(args.embedding_summary),
                    student_map_path=Path(args.student_map),
                    candidate_registry_path=Path(args.candidate_registry),
                )
                print(f"Verified reusable compact review: {Path(args.reuse_output).resolve()}")
                print(
                    "Source broad enrollment/CCTV: "
                    f"{summary['source_broad_enrollment_count']}/"
                    f"{summary['source_broad_cctv_count']}"
                )
                print(
                    "Compact enrollment selected/deferred: "
                    f"{summary['compact_enrollment_selected_count']}/"
                    f"{summary['compact_enrollment_deferred_count']}"
                )
                print(
                    "Compact CCTV selected/deferred: "
                    f"{summary['compact_cctv_selected_count']}/"
                    f"{summary['compact_cctv_deferred_count']}"
                )
                print("Stale outputs reused: no")
                print("Recognition/model rerun: no")
                print("Production files modified: no")
                return
            result = generate_compact_forensic_reviews(
                repo_root=ROOT,
                source_forensic_run=Path(args.source_forensic_run),
                source_reviewfix_run=Path(args.source_reviewfix_run),
                candidate_config_path=Path(args.candidate_config),
                embeddings_path=Path(args.embeddings),
                embedding_summary_path=Path(args.embedding_summary),
                student_map_path=Path(args.student_map),
                candidate_registry_path=Path(args.candidate_registry),
                enrollment_maximum=args.enrollment_maximum,
                cctv_maximum=args.cctv_maximum,
                output_dir=Path(args.output_dir) if args.output_dir else None,
                subject_abbr=args.subject_abbr,
                package_revision=args.package_revision,
            )
            print(f"Compact forensic output: {result.output_dir}")
            print(f"Output manifest: {result.output_manifest}")
            print(
                "Source broad enrollment/CCTV: "
                f"{result.summary['source_broad_enrollment_count']}/"
                f"{result.summary['source_broad_cctv_count']}"
            )
            print(
                "Compact enrollment selected/deferred: "
                f"{result.summary['compact_enrollment_selected_count']}/"
                f"{result.summary['compact_enrollment_deferred_count']}"
            )
            for roll, count in result.summary["compact_enrollment_per_student_counts"].items():
                print(f"  Enrollment {roll}: {count}")
            print(
                "Compact CCTV selected/deferred: "
                f"{result.summary['compact_cctv_selected_count']}/"
                f"{result.summary['compact_cctv_deferred_count']}"
            )
            for roll, count in result.summary["compact_cctv_per_student_counts"].items():
                print(f"  CCTV {roll}: {count}")
            print(f"Compact enrollment package: {result.enrollment_review.root}")
            print(f"Compact CCTV package: {result.cctv_review.root}")
            print(f"Elapsed seconds: {result.summary['elapsed_seconds']}")
            print("Recognition/model rerun: no")
            print("Production embeddings/datasets/broad packages modified: no")
            print("Candidate embedding built/promoted: no")
            return

        if args.command in {
            "open-compact-enrollment-review",
            "open-compact-cctv-review",
        }:
            print(f"Compact review package: {Path(args.review_package).resolve()}")
            serve_forensic_review_package(
                Path(args.review_package),
                port=args.port,
                open_browser=not args.no_browser,
            )
            return

        if args.command == "regenerate-forensic-review-packages":
            print(f"Source forensic run: {Path(args.source_forensic_run).resolve()}")
            print(f"Candidate config (read-only): {Path(args.candidate_config).resolve()}")
            print(f"Production embeddings (read-only): {Path(args.embeddings).resolve()}")
            print(f"Student map (read-only): {Path(args.student_map).resolve()}")
            print(f"Reviewer bundle revision: {args.package_revision}")
            result = regenerate_forensic_review_packages(
                source_forensic_run=Path(args.source_forensic_run),
                repo_root=ROOT,
                candidate_config_path=Path(args.candidate_config),
                embeddings_path=Path(args.embeddings),
                embedding_summary_path=Path(args.embedding_summary),
                student_map_path=Path(args.student_map),
                output_dir=Path(args.output_dir) if args.output_dir else None,
                subject_abbr=args.subject_abbr,
                package_revision=args.package_revision,
            )
            print(f"Regenerated forensic output: {result.output_dir}")
            print(f"Enrollment review package: {result.enrollment_review.root}")
            print(f"CCTV review package: {result.cctv_review.root}")
            print(f"Output manifest: {result.output_manifest}")
            print("Recognition/model rerun: no")
            print("Production files modified: no")
            return

        if args.command == "embedding-forensics":
            sources = forensic_benchmark_sources(args)
            print_embedding_forensic_inputs(args, sources)
            if args.reuse_output:
                print(f"  Reuse candidate: {Path(args.reuse_output).resolve()}")
                summary = verify_reusable_forensic_run(
                    output_dir=Path(args.reuse_output),
                    repo_root=ROOT,
                    candidate_config_path=Path(args.candidate_config),
                    embeddings_path=Path(args.embeddings),
                    embedding_summary_path=Path(args.embedding_summary),
                    student_map_path=Path(args.student_map),
                )
                print(f"Verified reusable forensic run: {Path(args.reuse_output).resolve()}")
                print(f"Benchmark reviewed tracks: {summary['benchmark_reviewed_tracks']}")
                print(f"Enrollment review package: {summary['review_packages']['enrollment_audit']}")
                print(f"CCTV review package: {summary['review_packages']['verified_cctv']}")
                print("Stale outputs reused: no")
                print("Production files modified: no")
                return
            result = run_embedding_forensics(
                repo_root=ROOT,
                candidate_config_path=Path(args.candidate_config),
                calibration_dir=Path(args.calibration_dir),
                shadow_review_package=Path(args.p2_shadow_review_package),
                benchmark_sources=sources,
                student_map_path=Path(args.student_map),
                embeddings_path=Path(args.embeddings),
                embedding_summary_path=Path(args.embedding_summary),
                output_dir=Path(args.output_dir) if args.output_dir else None,
                subject_abbr=args.subject_abbr,
                include_augmented_dataset=not args.no_augmented_dataset,
            )
            print(f"Forensic output: {result.output_dir}")
            print(f"Output manifest: {result.output_manifest}")
            print(f"Frozen benchmark: {result.summary['benchmark_reviewed_tracks']} reviewed tracks")
            print(f"Suspect enrollment images: {result.summary['suspect_enrollment_images']}")
            print(f"Candidate CCTV crops: {result.summary['candidate_cctv_crops']}")
            print(f"Enrollment review package: {result.enrollment_review.root}")
            print(f"CCTV review package: {result.cctv_review.root}")
            print("Candidate embedding version built: no")
            print("Production embeddings, datasets, thresholds, attendance rule, and attendance changed: no")
            return

        if args.command in {"open-enrollment-audit-review", "open-cctv-enrollment-review"}:
            print(f"Review package: {Path(args.review_package).resolve()}")
            serve_forensic_review_package(
                Path(args.review_package),
                port=args.port,
                open_browser=not args.no_browser,
            )
            return

        if args.command == "validate-compact-approvals":
            print(f"Enrollment review package: {Path(args.enrollment_review_package).resolve()}")
            print(f"Enrollment approvals: {Path(args.enrollment_approvals).resolve()}")
            print(f"CCTV review package: {Path(args.cctv_review_package).resolve()}")
            print(f"CCTV approvals: {Path(args.cctv_approvals).resolve()}")
            bundle = validate_compact_approval_bundle(
                enrollment_review_package=Path(args.enrollment_review_package),
                enrollment_approvals_path=Path(args.enrollment_approvals),
                cctv_review_package=Path(args.cctv_review_package),
                cctv_approvals_path=Path(args.cctv_approvals),
                dataset_root=Path(args.dataset),
            )
            print("Compact approval validation: PASS")
            print(f"Enrollment rows: {bundle.summary['enrollment_rows']}")
            print(f"Enrollment actions: {bundle.summary['enrollment_action_counts']}")
            print(f"CCTV rows: {bundle.summary['cctv_rows']}")
            print(f"CCTV actions: {bundle.summary['cctv_action_counts']}")
            print("Candidate build executed: no")
            print("Production files modified: no")
            return

        if args.command == "reconcile-identity-correction":
            bundle = validate_compact_approval_bundle(
                enrollment_review_package=Path(args.enrollment_review_package),
                enrollment_approvals_path=Path(args.enrollment_approvals),
                cctv_review_package=Path(args.cctv_review_package),
                cctv_approvals_path=Path(args.cctv_approvals),
                dataset_root=Path(args.dataset),
            )
            correction = reconcile_identity_correction(
                repo_root=ROOT,
                dataset_root=Path(args.dataset),
                historical_inventory_path=Path(args.historical_inventory),
                production_embeddings_path=Path(args.production_embeddings),
                student_map_path=Path(args.student_map),
                approval_bundle=bundle,
            )
            shruthi = reconcile_shruthi_source(
                dataset_roots=[Path(args.dataset), ROOT / "augmented_dataset"],
                embedding_outliers_path=Path(args.embedding_outliers),
                historical_inventory_path=Path(args.historical_inventory),
            )
            print("Identity correction reconciliation: PASS")
            print(f"Correction ID: {correction.manifest['correction_id']}")
            print(f"Affected folder images: {correction.manifest['affected_image_count']}")
            print(
                "Affected enrollment review items: "
                + ", ".join(correction.manifest["affected_enrollment_review_item_ids"])
            )
            print(
                "Affected CCTV review items: "
                + (", ".join(correction.manifest["affected_cctv_review_item_ids"]) or "none")
            )
            print(f"Shruthi source: {shruthi['status']} ({shruthi['resolved_path']})")
            print("Global alias created: no")
            print("Production files modified: no")
            return

        if args.command == "preflight-embedding-family":
            print("Phase 1.2I-B read-only preflight")
            print(f"Repository: {ROOT}")
            print(f"Frozen benchmark: {Path(args.forensic_run).resolve()}")
            print(f"Enrollment approvals: {Path(args.enrollment_approvals).resolve()}")
            print(f"CCTV approvals: {Path(args.cctv_approvals).resolve()}")
            result = preflight_embedding_family_inputs(
                repo_root=ROOT,
                forensic_run_dir=Path(args.forensic_run),
                enrollment_review_package=Path(args.enrollment_review_package),
                enrollment_approvals_path=Path(args.enrollment_approvals),
                cctv_review_package=Path(args.cctv_review_package),
                cctv_approvals_path=Path(args.cctv_approvals),
                production_embeddings_path=Path(args.production_embeddings),
                production_summary_path=Path(args.production_summary),
                student_map_path=Path(args.student_map),
                dataset_root=Path(args.dataset),
                augmented_dataset_root=Path(args.augmented_dataset),
            )
            if args.output_json:
                output_json = Path(args.output_json).resolve()
                output_json.parent.mkdir(parents=True, exist_ok=True)
                output_json.write_text(
                    json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8"
                )
                print(f"Preflight JSON: {output_json}")
            print("Preflight status: PASS")
            print(f"CVO roster: {result['cvo_roster_count']}")
            print(
                "Enrollment approvals: "
                f"{result['approval_validation']['enrollment_rows']} rows"
            )
            print(
                "CCTV approvals: "
                f"{result['approval_validation']['cctv_rows']} rows"
            )
            print(
                "Identity reconciliation: "
                f"{result['identity_correction']['reconciled_review_rows']} reviewed rows, "
                f"{result['identity_correction']['affected_image_hashes']} folder hashes"
            )
            print(f"Frozen benchmark rows: {result['frozen_benchmark_rows']}")
            print("Candidate build started: no")
            print("Production files changed: no")
            print("MON_P3 processed: no")
            print("PREFLIGHT_STATUS=PASS")
            return

        if args.command == "build-evaluate-embedding-family":
            print("Phase 1.2I-B version-family build")
            print(f"Repository: {ROOT}")
            print(f"Frozen benchmark: {Path(args.forensic_run).resolve()}")
            print(f"Enrollment approvals: {Path(args.enrollment_approvals).resolve()}")
            print(f"CCTV approvals: {Path(args.cctv_approvals).resolve()}")
            print(f"Production embeddings (read-only): {Path(args.production_embeddings).resolve()}")
            print(f"Version output root: {Path(args.versions_root).resolve()}")
            started = datetime.now()
            result = build_and_evaluate_embedding_family(
                repo_root=ROOT,
                forensic_run_dir=Path(args.forensic_run),
                enrollment_review_package=Path(args.enrollment_review_package),
                enrollment_approvals_path=Path(args.enrollment_approvals),
                cctv_review_package=Path(args.cctv_review_package),
                cctv_approvals_path=Path(args.cctv_approvals),
                production_embeddings_path=Path(args.production_embeddings),
                production_summary_path=Path(args.production_summary),
                student_map_path=Path(args.student_map),
                dataset_root=Path(args.dataset),
                augmented_dataset_root=Path(args.augmented_dataset),
                versions_root=Path(args.versions_root),
                family_id=args.family_id,
                launcher_source=ROOT / "scripts" / "run_phase_1_2i_b.ps1",
            )
            print(f"Family ID: {result.family_id}")
            print(f"Family output: {result.family_dir}")
            for key, variant in result.family_manifest["variants"].items():
                print(
                    f"Variant {key}: {variant['version_id']} - {variant['status']} - "
                    f"{variant['embedding_records']} embeddings"
                )
            print(f"Decision: {result.evaluation_decision['decision']}")
            print(f"Idempotent reuse: {'yes' if result.idempotent_reuse else 'no'}")
            print(f"Elapsed: {(datetime.now() - started).total_seconds():.2f} seconds")
            print("Production embeddings changed: no")
            print("Candidate approved/promoted: no")
            print("MON_P3 processed: no")
            print(f"FAMILY_OUTPUT={result.family_dir}")
            return

        if args.command == "verify-embedding-family":
            result = verify_embedding_family(
                Path(args.family_dir),
                production_embeddings_path=Path(args.production_embeddings),
                production_summary_path=Path(args.production_summary),
            )
            print("Embedding family verification: PASS")
            print(f"Family ID: {result['family_id']}")
            print(f"Verified files: {result['verified_files']}")
            print(f"Verified variants: {result['variants_verified']}")
            print(f"Decision: {result['decision']}")
            print("Production approved/promoted: no")
            print("MON_P3 processed: no")
            return

        if args.command == "show-embedding-family":
            result = show_embedding_family(Path(args.family_dir))
            print(f"Family ID: {result['family_id']}")
            print(f"Status: {result['status']}")
            print(f"Decision: {result['decision']}")
            for key, variant in result["variants"].items():
                print(
                    f"Variant {key}: {variant['version_id']} - "
                    f"{variant['embedding_records']} embeddings"
                )
            print(f"CVO coverage: {result['candidate_cvo_coverage']}")
            print(f"AI0110 covered: {result['ai0110_candidate_coverage']}")
            print(f"0268 status: {result['missing_0268_status']}")
            print(f"Next step: {result['exact_next_step']}")
            print(f"Output: {result['output_dir']}")
            return

        if args.command == "validate-enrollment-approvals":
            print(f"Enrollment review package: {Path(args.enrollment_review_package).resolve()}")
            print(f"Enrollment approvals: {Path(args.enrollment_approvals).resolve()}")
            print(f"CCTV review package: {Path(args.cctv_review_package).resolve()}")
            print(f"CCTV approvals: {Path(args.cctv_approvals).resolve()}")
            bundle = validate_embedding_approval_bundle(
                enrollment_review_package=Path(args.enrollment_review_package),
                enrollment_approvals_path=Path(args.enrollment_approvals),
                cctv_review_package=Path(args.cctv_review_package),
                cctv_approvals_path=Path(args.cctv_approvals),
            )
            print("Approval bundle: PASS")
            print(f"Enrollment items: {bundle.summary['enrollment_reviewed_items']}")
            print(f"CCTV items: {bundle.summary['cctv_reviewed_items']}")
            print("Embedding build executed: no")
            print("Production files modified: no")
            return

        if args.command == "build-versioned-embeddings":
            print(f"Forensic run: {Path(args.forensic_run).resolve()}")
            print(f"Enrollment review package: {Path(args.enrollment_review_package).resolve()}")
            print(f"Enrollment approvals: {Path(args.enrollment_approvals).resolve()}")
            print(f"CCTV review package: {Path(args.cctv_review_package).resolve()}")
            print(f"CCTV approvals: {Path(args.cctv_approvals).resolve()}")
            print(f"Production embeddings (immutable parent): {Path(args.production_embeddings).resolve()}")
            print(f"Version output root: {Path(args.versions_root).resolve()}")
            result = build_versioned_embeddings(
                repo_root=ROOT,
                forensic_run_dir=Path(args.forensic_run),
                enrollment_review_package=Path(args.enrollment_review_package),
                enrollment_approvals_path=Path(args.enrollment_approvals),
                cctv_review_package=Path(args.cctv_review_package),
                cctv_approvals_path=Path(args.cctv_approvals),
                production_embeddings_path=Path(args.production_embeddings),
                production_summary_path=Path(args.production_summary),
                versions_root=Path(args.versions_root),
                version_id=args.version_id,
                candidate_registry_dir=Path(args.candidate_registry),
                source_candidate_ids=args.source_candidate_id,
            )
            print(f"Embedding version: {result.version_dir}")
            print(f"Status: {result.manifest['status']}")
            print(f"Idempotent reuse: {'yes' if result.idempotent_reuse else 'no'}")
            print("Production embeddings overwritten: no")
            print("Promotion performed: no")
            return

        if args.command == "evaluate-embedding-version":
            print(f"Embedding version: {Path(args.version_dir).resolve()}")
            print(f"Frozen forensic run: {Path(args.forensic_run).resolve()}")
            print(f"Regression evidence: {Path(args.regression_results).resolve()}")
            if args.untouched_validation:
                print(f"Untouched validation evidence: {Path(args.untouched_validation).resolve()}")
            status = evaluate_embedding_version(
                version_dir=Path(args.version_dir),
                forensic_run_dir=Path(args.forensic_run),
                regression_results_path=Path(args.regression_results),
                untouched_validation_path=Path(args.untouched_validation) if args.untouched_validation else None,
            )
            print(f"Regression passed: {'yes' if status['regression_passed'] else 'no'}")
            print(
                "Untouched MON_P3 validation passed: "
                f"{'yes' if status['untouched_session_validation_passed'] else 'no'}"
            )
            print(f"Promotion allowed by evidence: {'yes' if status['promotion_allowed'] else 'no'}")
            print("Promotion performed: no")
            return

        if args.command == "promote-embedding-version":
            version_dir = Path(args.version_dir).resolve()
            if args.confirm_version_id != version_dir.name:
                raise EmbeddingLifecycleError("--confirm-version-id must exactly match the version directory name")
            print(f"Embedding version selected for explicit promotion: {version_dir}")
            print(f"Production embeddings target: {Path(args.production_embeddings).resolve()}")
            result = promote_embedding_version(
                version_dir=version_dir,
                production_embeddings_path=Path(args.production_embeddings),
                production_summary_path=Path(args.production_summary),
                candidate_registry_dir=Path(args.candidate_registry),
            )
            print(f"Promotion status: {result['status']}")
            print(f"Rollback manifest: {result['rollback_manifest']}")
            return

        if args.command == "rollback-embedding-version":
            version_dir = Path(args.version_dir).resolve()
            if args.confirm_version_id != version_dir.name:
                raise EmbeddingLifecycleError("--confirm-version-id must exactly match the version directory name")
            print(f"Embedding version selected for explicit rollback: {version_dir}")
            result = rollback_embedding_version(version_dir=version_dir)
            print(f"Rollback status: {result['status']}")
            print(f"Restored embeddings SHA-256: {result['restored_embeddings_sha256']}")
            return

        if args.command == "run-shadow-session":
            if args.preflight_only:
                preflight = preflight_shadow_session(
                    video_root=Path(args.video_dir),
                    session_id=args.session_id,
                    subject_abbr=args.subject_abbr,
                    student_map_path=Path(args.student_map),
                    candidate_config_path=Path(args.candidate_config),
                    candidate_manifest_path=Path(args.candidate_manifest),
                    reference_diagnostic_run=Path(args.reference_diagnostic_run),
                    expected_roster_count=args.expected_roster_count,
                )
                command = build_baseline_command(
                    preflight,
                    repo_root=ROOT,
                    diagnostic_dir=ROOT / "attendance_output" / "diagnostics",
                    diagnostic_run_id="2026-06-30__B51__P2__CVO_PHASE_12H_PREFLIGHT",
                    timetable_path=Path(args.timetable),
                    embeddings_path=Path(args.embeddings),
                    python_executable=Path(sys.executable),
                )
                print_shadow_preflight(preflight, command)
                print("Recognition executed: no")
                print("Official attendance modified: no")
                return
            result = run_shadow_session(
                repo_root=ROOT,
                video_root=Path(args.video_dir),
                session_id=args.session_id,
                subject_abbr=args.subject_abbr,
                student_map_path=Path(args.student_map),
                candidate_config_path=Path(args.candidate_config),
                candidate_manifest_path=Path(args.candidate_manifest),
                reference_diagnostic_run=Path(args.reference_diagnostic_run),
                timetable_path=Path(args.timetable),
                embeddings_path=Path(args.embeddings),
                diagnostic_run=Path(args.diagnostic_run) if args.diagnostic_run else None,
                output_dir=Path(args.output_dir) if args.output_dir else None,
                expected_roster_count=args.expected_roster_count,
            )
            print(f"Shadow diagnostic run: {result.diagnostic_run}")
            print(f"Shadow validation output: {result.output_dir}")
            print(f"Blind review package: {result.review_package.root}")
            print(f"Reviewer launcher: {result.summary['reviewer_launcher']}")
            print(
                "Tracklets: "
                f"{result.summary['baseline_tracks']} baseline, "
                f"{result.summary['baseline_rejected']} rejected, "
                f"{result.summary['shadow_accepted']} shadow recoveries"
            )
            print(f"Human review status: {result.summary['review_status']}")
            print(f"Recognition rerun: {'yes' if result.recognition_rerun else 'no'}")
            print("Official attendance modified: no")
            print("Production candidate enabled: no")
            return

        if args.command == "open-shadow-review":
            serve_review_package(
                Path(args.review_package),
                port=args.port,
                open_browser=not args.no_browser,
            )
            return

        if args.command == "evaluate-shadow-session":
            result, output_dir = evaluate_shadow_review_package(
                Path(args.review_package),
                Path(args.labels),
                output_dir=Path(args.output_dir) if args.output_dir else None,
            )
            print(f"Shadow evaluation output: {output_dir}")
            print(f"Reviewed shadow tracks: {result.summary['reviewed_shadow_tracks']}")
            print(f"Correct recoveries: {result.summary['correct_recoveries']}")
            print(f"Confirmed false identities: {result.summary['confirmed_false_identities']}")
            print(f"Decision: {result.summary['decision']}")
            print("Production candidate enabled: no")
            return

        if args.command == "calibrate-tracklets":
            result = run_tracklet_calibration(
                diagnostic_run=Path(args.diagnostic_run),
                accepted_review_package=Path(args.accepted_review_package),
                accepted_labels_path=Path(args.accepted_labels),
                unresolved_review_package=Path(args.unresolved_review_package),
                unresolved_labels_path=Path(args.unresolved_labels),
                student_map_path=Path(args.student_map),
                subject_abbr=args.subject_abbr,
                output_dir=Path(args.output_dir) if args.output_dir else None,
                folds=args.folds,
                min_oof_recoveries=args.min_oof_recoveries,
                min_recovery_folds=args.min_recovery_folds,
            )
            summary = result.summary
            print(f"Tracklet calibration output: {result.output_dir}")
            print(f"Frozen reviewed benchmark: {summary['benchmark']['total_tracks']} tracks")
            print(
                "Out-of-fold recovery: "
                f"{summary['out_of_fold']['true_recoveries']} correct / "
                f"{summary['out_of_fold']['false_accepts']} false accepts"
            )
            print(
                "Final shadow candidate: "
                f"{summary['final_candidate']['true_recoveries']} correct / "
                f"{summary['final_candidate']['false_accepts']} false accepts"
            )
            print(f"Recommendation: {summary['recommendation']['decision']}")
            print("Candidate enabled: no")
            print("Production recognition changed: no")
            return

        if args.command == "export-corrections":
            result = export_label_correction_package(
                source_review_package=Path(args.review_package),
                source_labels_path=Path(args.labels),
                student_map_path=Path(args.student_map),
                subject_abbr=args.subject_abbr,
                output_root=Path(args.output_dir) if args.output_dir else None,
                target_status=args.target_status,
            )
            print(f"Correction review package: {result.root}")
            print(f"Blind reviewer directory: {result.reviewer_dir}")
            print(f"Correction cards exported: {result.evidence_images}")
            print("Original labels changed: no")
            return

        if args.command == "merge-corrections":
            result = merge_label_corrections(
                correction_package=Path(args.correction_package),
                corrections_path=Path(args.corrections),
                source_labels_path=Path(args.source_labels),
                output_path=Path(args.output),
            )
            print(f"Merged labels: {result.output_path}")
            print(f"Merge audit: {result.audit_path}")
            print(f"Rows: {result.summary['source_rows']}")
            print(f"Corrected target rows: {result.summary['target_rows']}")
            print("Original labels changed: no")
            return

        if args.command == "audit-roster":
            result = audit_subject_roster(
                student_map_path=Path(args.student_map),
                subject_abbr=args.subject_abbr,
                actual_present_path=Path(args.actual_present),
                embedding_summary_path=Path(args.embedding_summary),
                embeddings_path=Path(args.embeddings),
            )
            output_dir = (
                Path(args.output_dir)
                if args.output_dir
                else ROOT / "attendance_output" / "diagnostics" / datetime.now().strftime(
                    f"roster_audit_{str(args.subject_abbr).upper()}_%Y%m%d_%H%M%S_%f"
                )
            )
            paths = write_roster_audit(result, output_dir)
            print(f"Roster audit: {output_dir}")
            print(f"Integrity: {'PASS' if result.summary['integrity_passed'] else 'FAIL'}")
            print(f"Roster students: {result.summary['roster_students']}")
            print(
                f"Embedding coverage: {result.summary['embedding_available']}/"
                f"{result.summary['roster_students']}"
            )
            print(f"Missing embedding rolls: {', '.join(result.summary['missing_embedding_rolls']) or 'none'}")
            print(f"Report: {paths['report']}")
            if not result.summary["integrity_passed"]:
                raise RosterIntegrityError("Roster audit failed; inspect the generated report")
            return

        if args.command == "export":
            result = export_review_package_from_diagnostics(
                Path(args.diagnostic_run),
                video_root=Path(args.video_dir) if args.video_dir else None,
                output_root=Path(args.output_dir) if args.output_dir else None,
                student_map_path=Path(args.student_map),
                subject_abbr=args.subject_abbr,
                evidence_count=args.evidence_count,
                crop_padding=args.crop_padding,
            )
            print(f"Review package: {result.root}")
            print(f"Blind reviewer directory: {result.reviewer_dir}")
            print(f"Accepted tracklets exported: {result.accepted_tracklets}")
            print("Keep private/ withheld until the reviewer exports final labels.")
            return

        if args.command == "recall-gap":
            selection_config = UnresolvedSelectionConfig(
                target_tracks=args.target_unresolved,
                candidate_cap=args.candidate_cap,
                checkpoint_cap=args.checkpoint_cap or None,
                camera_cap=args.camera_cap or None,
                zone_cap=args.zone_cap or None,
            )
            result = run_recall_gap_analysis(
                diagnostic_run=Path(args.diagnostic_run),
                accepted_review_package=Path(args.accepted_review_package),
                accepted_labels=Path(args.accepted_labels),
                actual_present_path=Path(args.actual_present),
                student_map_path=Path(args.student_map),
                embedding_summary_path=Path(args.embedding_summary),
                embeddings_path=Path(args.embeddings),
                output_dir=Path(args.output_dir) if args.output_dir else None,
                video_root=Path(args.video_dir) if args.video_dir else None,
                subject_abbr=args.subject_abbr,
                selection_config=selection_config,
            )
            coverage = result.summary["embedding_roster_coverage"]
            print(f"Recall analysis: {result.output_dir}")
            print(f"Unresolved review package: {result.review_package.root}")
            print(f"Selected unresolved tracks: {len(result.unresolved_selection)}")
            print(f"Embedding coverage: {coverage['embedding_available']}/{coverage['roster_students']}")
            print(f"Missing embedding rolls: {', '.join(coverage['missing_rolls']) or 'none'}")
            print("Recognition rerun: no")
            return

        if args.command == "evaluate-unresolved":
            result, output_dir = evaluate_unresolved_review_package(
                review_package=Path(args.review_package),
                labels_path=Path(args.labels),
                recall_analysis_dir=Path(args.recall_analysis),
                accepted_review_package=Path(args.accepted_review_package),
                accepted_labels=Path(args.accepted_labels),
                output_dir=Path(args.output_dir) if args.output_dir else None,
                student_map_path=Path(args.student_map) if args.actual_present else None,
                subject_abbr=args.subject_abbr,
                actual_present_path=Path(args.actual_present) if args.actual_present else None,
                embedding_summary_path=Path(args.embedding_summary) if args.actual_present else None,
                embeddings_path=Path(args.embeddings) if args.actual_present else None,
            )
            print(f"Unresolved validation output: {output_dir}")
            print(f"Review completion: {result.summary['review']['reviewed_tracks']}/{result.summary['review']['tracks']}")
            print(f"Recommendation: {result.summary['recommendation']}")
            print(f"Roster context: {result.summary['evaluation_context']['mode']}")
            return

        result, output_dir = evaluate_review_package(
            Path(args.review_package),
            Path(args.labels),
            actual_present_path=Path(args.actual_present) if args.actual_present else None,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            present_checkpoints=args.present_checkpoints,
        )
        identity = result.summary["identity_metrics"]
        recommendation = result.summary["recommendation"]
        print(f"Validation output: {output_dir}")
        print(f"Review completion: {identity['reviewed_tracklets']}/{identity['accepted_tracklets']}")
        print(f"Identified top-1 accuracy: {identity['identified_top1_accuracy']}")
        print(f"Reviewable accepted precision: {identity['reviewable_accepted_precision']}")
        print(f"Recommendation: {recommendation['next_step']} ({recommendation['production_decision']})")
    except (
        OSError,
        ValueError,
        ReviewExportError,
        LabelValidationError,
        RecallAnalysisError,
        RosterIntegrityError,
        TrackletCalibrationError,
        ShadowValidationError,
        EmbeddingForensicsError,
        EmbeddingLifecycleError,
        ForensicReviewError,
        CompactReviewError,
        EmbeddingFamilyError,
    ) as exc:
        raise SystemExit(f"Tracklet ground-truth validation failed: {exc}") from exc


if __name__ == "__main__":
    main()
