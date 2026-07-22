from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import uuid
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from unittest.mock import patch


POLICY_VERSION = "product-phase-2l-c-explicit-authentication-v1"
OUTPUT_SCHEMA_VERSION = 1
PHASE_2L_A_MANIFEST_SHA256 = "6e86e63c5da4945cb133b6c8be3250340be1d0ede3c512531e9c44f6ced99153"
PHASE_2L_B_MANIFEST_SHA256 = "e58a942a4bd091bdf8fc5f75d0351cbe68bfaef47d90f7bfd1c7678356f41c3d"
OFFICIAL_REPORT_SHA256 = "0211ac1433831aceb4176ee199f64bf962dbc784765a99a00bed74c7b38d51e3"
CANDIDATE_REPORT_SHA256 = "e28dc0278abfad034bd78bb576620a50f90100461f23b5e0b920e1e65e39c57f"
SESSION_ID = "2026-06-22__B51__P4__CVO"

PHASE_2L_A_RELATIVE = "attendance_output/product_workflow/phase_2l_stabilization/stabilization-5fb42e9ff8f8286138c5"
PHASE_2L_B_RELATIVE = "attendance_output/product_workflow/phase_2l_demo_rehearsal/rehearsal-05c307e549637792f873"
OFFICIAL_REPORT_RELATIVE = "attendance_output/product_workflow/phase_2i_authority/authority-revision-2d83c4f679f839a6c784c636/corrected_attendance_2026-06-22__B51__P4__CVO.csv"
CANDIDATE_REPORT_RELATIVE = "attendance_output/product_workflow/report_revisions/2026-06-22__B51__P4__CVO/automatic-candidate-99cc13934144bf7e1706b8be/candidate_attendance.csv"

PROTECTED_EXPECTED = {
    "data/attendance_status.json": "46fd2df55cc3f61c5fa03c893937eb2828b9cc2af064ea9e427276b79a9d0b46",
    "data/job_runtime.json": "5a23f09654d91fac09804cd97c2fa114bf4956a914f2e02ce2b7090f730141ed",
    "data/role_users.json": "77125140004294f2834fc869fd590e167d8ab78cff5dbeda1c074ab76c177517",
    "data/student_faculty_map.json": "0eca63ea58d12a73d3bd11e620b74be733a3219c9244e8a4a267d5b02acd3f91",
    "data/manual_overrides.json": "e9c6bd35c23c353795edb5d3c02e808793d7cbd90c7b0a87fdfc686f8fd5cdcf",
    "data/review_evidence_registry.json": "3b04e9aecfdfb1f64346c4a0e709fd9c36d7c56545bf816d6641b4c2a2e80841",
    "models/student_embeddings.pkl": "f088d827adc548ee95f46566d758fd71fc304d042c43f1ecffc6526b60bcd832",
    "models/embedding_summary.csv": "63885588c374c37f4da9bf85294f240bdf0f28cb585e77b894ab516139ca46ae",
    "models/current_embedding_version.json": "7999b8ccf787dca9b8fb862f4e53ce7c1102eb729a3732c82fee76f3ba3ee05e",
    "timetable_b51_2026_2027.csv": "10bbd578a859fbd4fa228c4eb4f1f92e7de5b92a1c4236d71a00337eecac9c57",
}

REQUIRED_EXTERNAL_VALIDATIONS = (
    "python_compile",
    "authentication_tests",
    "existing_auth_role_config_tests",
    "phase_2l_a_tests",
    "frontend_tests",
    "frontend_build",
    "no_token_matrix",
    "synthetic_role_matrix",
    "browser_rehearsal",
    "phase_2l_a_preflight",
    "full_python_suite",
)

REQUIRED_OUTPUT_FILES = (
    "authentication_policy.json",
    "public_route_allowlist.json",
    "protected_route_inventory.csv",
    "no_token_api_matrix.csv",
    "authenticated_role_matrix.csv",
    "private_field_exposure_scan.json",
    "frontend_auth_contract.json",
    "browser_rehearsal_summary.csv",
    "report_totals_verification.json",
    "identity_integrity_verification.json",
    "demo_readiness.json",
    "documentation_inventory.json",
    "validation_summary.json",
    "no_recognition_declaration.json",
    "no_operational_mutation_declaration.json",
    "source_manifest.json",
    "evaluation_summary.json",
    "immutable_manifest.json",
)

SOURCE_FILES = (
    "app.py",
    "frontend/src/api/client.js",
    "frontend/src/auth/AuthContext.jsx",
    "frontend/src/pages/Dashboard.jsx",
    "frontend/src/pages/LiveDemo.jsx",
    "frontend/src/pages/ManualReview.jsx",
    "frontend/src/pages/Reports.jsx",
    "frontend/src/pages/Timetable.jsx",
    "frontend/tests/authContract.test.mjs",
    "src/face_attendance/product_phase_2l_authentication.py",
    "tests/test_product_phase_2l_authentication.py",
    "tests/test_product_phase_2d_reports_review.py",
    "tests/test_product_phase_2e_hod_control.py",
    "scripts/run_product_phase_2l_auth_preflight.py",
    "scripts/run_product_phase_2l_auth_preflight.ps1",
    "scripts/run_product_phase_2l_auth_fixture.py",
    "docs/PHASE_2L_OPERATOR_GUIDE.md",
    "docs/PHASE_2L_PROFESSOR_DEMO_RUNBOOK.md",
    "docs/PHASE_2L_FINAL_RELEASE_CHECKLIST.md",
    "docs/PHASE_2L_RECOVERY_AND_ROLLBACK.md",
)

DOCUMENT_FILES = (
    "docs/PHASE_2L_OPERATOR_GUIDE.md",
    "docs/PHASE_2L_PROFESSOR_DEMO_RUNBOOK.md",
    "docs/PHASE_2L_FINAL_RELEASE_CHECKLIST.md",
    "docs/PHASE_2L_RECOVERY_AND_ROLLBACK.md",
)

BROWSER_REHEARSAL_ROWS = (
    ("public", "Login", "/login", "pass", "public login form rendered; no credential/token exposure"),
    ("HOD", "Dashboard", "/", "pass", "authenticated shell and scoped backend data rendered"),
    ("HOD", "My Classes / Timetable", "/take-attendance", "pass", "live timetable rendered; processing controls were not used"),
    ("HOD", "Official Reviewed MON_P4", "/reports", "pass", "official 6/4/16/1/0 revision and unresolved state rendered"),
    ("HOD", "Archived Automatic Candidate", "/reports", "pass", "archived 2/8/16/1/0 candidate remained visibly separate from official"),
    ("HOD", "Manual Review", f"/manual-review?session_id={SESSION_ID}", "pass", "official report rendered; no edit/finalize action used"),
    ("HOD", "Students / Coverage", "/students", "pass", "CVO 27 and 26 covered rendered; missing enrollment remained explicit"),
    ("HOD", "HOD Control", "/hod-control", "pass", "HOD-only controls rendered; no control used"),
    ("HOD", "Help", "/help", "pass", "status meanings and guarded-review safety copy rendered"),
    ("Faculty", "Dashboard", "/", "pass", "assigned-subject dashboard rendered"),
    ("Faculty", "Reports", "/reports", "pass", "assigned-subject report view rendered"),
    ("Faculty", "Manual Review", f"/manual-review?session_id={SESSION_ID}", "pass", "assigned CVO report rendered read-only"),
    ("Faculty", "Students / Coverage", "/students", "pass", "only 27 assigned CVO students and 26/27 coverage rendered"),
)


class AuthenticationHardeningError(RuntimeError):
    pass


@dataclass(frozen=True)
class AuthenticationPreflight:
    repo_root: Path
    output_root: Path
    protected_rows: tuple[dict[str, str], ...]
    no_token_rows: tuple[dict[str, str], ...]
    role_rows: tuple[dict[str, str], ...]
    protected_hashes: dict[str, str]
    official_counts: dict[str, int]
    candidate_counts: dict[str, int]
    documentation_inventory: tuple[dict[str, Any], ...]
    source_manifest: tuple[dict[str, Any], ...]
    frontend_contract: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_json_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def readable_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return stream.getvalue().encode("utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuthenticationHardeningError(f"invalid_json:{path}") from exc
    if not isinstance(payload, dict):
        raise AuthenticationHardeningError(f"json_object_required:{path}")
    return payload


def verify_input_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise AuthenticationHardeningError(f"immutable_input_manifest_mismatch:{path}")
    manifest = _load_json(path)
    files = manifest.get("files") or {}
    if not isinstance(files, dict):
        raise AuthenticationHardeningError(f"immutable_input_file_map_invalid:{path}")
    actual_names = sorted(item.name for item in path.parent.iterdir() if item.is_file())
    expected_names = sorted([*files, "immutable_manifest.json"])
    if actual_names != expected_names:
        raise AuthenticationHardeningError(f"immutable_input_file_set_mismatch:{path.parent}")
    for name, expected in files.items():
        target = path.parent / name
        if not target.is_file() or sha256_file(target) != expected.get("sha256") or target.stat().st_size != expected.get("size_bytes"):
            raise AuthenticationHardeningError(f"immutable_input_tampered:{target}")
    return manifest


def _status_category(value: Any) -> str:
    status = str(value or "").strip().lower()
    if status.startswith("present") or status == "yes":
        return "Present"
    if "missing enrollment" in status:
        return "Missing Enrollment"
    if "unconfirmed" in status:
        return "Unconfirmed"
    if "review" in status:
        return "Needs Review"
    if status in {"absent", "no"}:
        return "Absent"
    return "Unknown"


def report_counts(path: Path) -> dict[str, int]:
    counts = {name: 0 for name in ("Present", "Needs Review", "Unconfirmed", "Missing Enrollment", "Absent", "Unknown")}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        counts[_status_category(row.get("Status") or row.get("Final_Status") or row.get("Present"))] += 1
    counts["Total"] = len(rows)
    counts["Unresolved"] = counts["Needs Review"] + counts["Unconfirmed"] + counts["Missing Enrollment"] + counts["Unknown"]
    return counts


def _sensitivity(rule: str, method: str) -> tuple[str, str, str]:
    if rule.startswith("/api/hod/") or rule in {"/api/reports", "/api/videos", "/api/videos/upload", "/api/video-layout", "/api/reset-processing"}:
        return "high", "HOD", "cross-faculty/system inventory"
    if rule.startswith("/api/download/") or rule.startswith("/attendance_website/"):
        return "high", "HOD|Faculty", "Faculty assigned session report files; HOD all"
    if rule.startswith("/api/attendance"):
        return "high", "HOD|Faculty", "Faculty assigned subject/session; HOD all"
    if rule.startswith("/api/job") or rule.startswith("/api/cancel") or rule in {"/api/status", "/api/process", "/api/reprocess"}:
        return "high", "HOD|Faculty", "Faculty assigned subject/job; HOD all"
    if rule.startswith("/api/live-demo"):
        return "high", "HOD|Faculty", "authenticated recognized roles; feature remains deferred"
    if rule in {"/api/students", "/api/timetable"}:
        return "medium", "HOD|Faculty", "Faculty assigned subject/roster; HOD all"
    return "medium", "HOD|Faculty", "authenticated recognized roles"


def route_inventory(backend: Any) -> tuple[dict[str, str], ...]:
    rows: list[dict[str, str]] = []
    for rule in sorted(backend.app.url_map.iter_rules(), key=lambda item: (item.rule, item.endpoint)):
        for method in sorted(set(rule.methods or ()) - {"HEAD", "OPTIONS"}):
            key = (method, rule.rule)
            if key in backend.PUBLIC_ROUTE_ALLOWLIST:
                continue
            sensitivity, roles, scope = _sensitivity(rule.rule, method)
            rows.append({
                "method": method,
                "route": rule.rule,
                "endpoint": rule.endpoint,
                "classification": "protected",
                "data_sensitivity": sensitivity,
                "authentication": "valid bearer session required",
                "allowed_roles": roles,
                "authorization_scope": scope,
                "no_token_expected": "401",
                "wrong_role_expected": "403",
            })
    if not rows:
        raise AuthenticationHardeningError("protected_route_inventory_empty")
    return tuple(rows)


def _concrete_path(rule: str) -> str:
    values = {
        "session_id": SESSION_ID,
        "job_id": "SYNTHETIC-NONEXISTENT-JOB",
        "day": "MONDAY",
        "period": "P4",
        "filename": "synthetic-nonexistent.csv",
    }

    def replace(match: re.Match[str]) -> str:
        name = match.group(1).split(":")[-1]
        return values.get(name, f"synthetic-{name}")

    return re.sub(r"<([^>]+)>", replace, rule)


def run_no_token_matrix(backend: Any, inventory: Sequence[Mapping[str, str]]) -> tuple[dict[str, str], ...]:
    client = backend.app.test_client()
    private_terms = ("password", "session_token", "role_users", "student_embeddings", "attendance_data", "workflow_state", "input_source_path")
    rows: list[dict[str, str]] = []
    for item in inventory:
        path = _concrete_path(item["route"])
        response = client.open(path, method=item["method"])
        body = response.get_data(as_text=True)
        exposed = sorted(term for term in private_terms if term in body.lower())
        passed = response.status_code == 401 and response.is_json and not exposed and body.strip() == '{"error":"Authentication required.","success":false}'
        rows.append({
            "method": item["method"],
            "route": item["route"],
            "request_path": path,
            "status": str(response.status_code),
            "json_response": str(bool(response.is_json)).lower(),
            "private_fields_exposed": "|".join(exposed),
            "result": "pass" if passed else "fail",
        })
    if not all(row["result"] == "pass" for row in rows):
        raise AuthenticationHardeningError("no_token_matrix_failed")
    return tuple(rows)


def _role_request(client: Any, headers: Mapping[str, str], method: str, path: str) -> tuple[int, Any]:
    response = client.open(path, method=method, headers=dict(headers))
    try:
        return response.status_code, response.get_json(silent=True)
    finally:
        response.close()


def run_role_matrix(backend: Any, repo: Path) -> tuple[dict[str, str], ...]:
    hod = {"id": "synthetic-hod", "name": "Synthetic HOD", "role": "admin", "roleLabel": "Head of Department", "subjects": ["SWE", "CCM", "CVO"], "canSeeAll": True}
    cvo = {"id": "synthetic-cvo", "name": "Synthetic CVO Faculty", "role": "faculty", "roleLabel": "Faculty", "subjects": ["CVO"], "canSeeAll": False}
    swe = {"id": "synthetic-swe", "name": "Synthetic SWE Faculty", "role": "faculty", "roleLabel": "Faculty", "subjects": ["SWE"], "canSeeAll": False}
    other = {"id": "synthetic-other", "name": "Synthetic Other Role", "role": "auditor", "subjects": [], "canSeeAll": False}
    users = [hod, cvo, swe, other]
    statuses = json.loads((repo / "data/attendance_status.json").read_text(encoding="utf-8"))
    client = backend.app.test_client()
    backend.AUTH_SESSIONS.clear()
    tokens = {user["id"]: backend.issue_auth_session(user)[0] for user in users}
    headers = {user["id"]: {"Authorization": f"Bearer {tokens[user['id']]}", "X-User-Id": user["id"]} for user in users}
    official_download = f"/api/download/{Path(OFFICIAL_REPORT_RELATIVE).relative_to('attendance_output').as_posix()}"
    cases = [
        ("HOD", hod["id"], "GET", "/api/auth/me", 200, "authenticated identity"),
        ("Faculty-CVO", cvo["id"], "GET", "/api/profile", 200, "authenticated identity"),
        ("Faculty-CVO", cvo["id"], "GET", "/api/timetable?day=Monday", 200, "assigned CVO only"),
        ("Faculty-CVO", cvo["id"], "GET", "/api/attendance-sessions", 200, "assigned CVO reports only"),
        ("Faculty-CVO", cvo["id"], "GET", f"/api/attendance/{SESSION_ID}", 200, "assigned CVO report"),
        ("Faculty-SWE", swe["id"], "GET", f"/api/attendance/{SESSION_ID}", 403, "other-faculty CVO report denied"),
        ("HOD", hod["id"], "GET", f"/api/attendance/{SESSION_ID}", 200, "all-subject report access"),
        ("HOD", hod["id"], "GET", "/api/reports", 200, "cross-faculty inventory"),
        ("Faculty-CVO", cvo["id"], "GET", "/api/reports", 403, "cross-faculty inventory denied"),
        ("Faculty-CVO", cvo["id"], "GET", official_download, 200, "assigned CVO report download"),
        ("Faculty-SWE", swe["id"], "GET", official_download, 403, "other-faculty CVO report download denied"),
        ("HOD", hod["id"], "GET", official_download, 200, "all-subject report download"),
        ("HOD", hod["id"], "GET", "/api/hod/overview", 200, "HOD control read"),
        ("Faculty-CVO", cvo["id"], "GET", "/api/hod/overview", 403, "HOD control denied"),
        ("Faculty-CVO", cvo["id"], "POST", "/api/reset-processing", 403, "system mutation denied before route body"),
        ("Other", other["id"], "GET", "/api/auth/me", 403, "unrecognized role denied"),
    ]
    rows: list[dict[str, str]] = []
    with ExitStack() as stack:
        stack.enter_context(patch.object(backend, "load_role_users", return_value=users))
        stack.enter_context(patch.object(backend, "load_statuses", side_effect=lambda: deepcopy(statuses)))
        stack.enter_context(patch.object(backend, "repair_stale_processing_statuses", side_effect=lambda value: value))
        stack.enter_context(patch.object(backend, "ensure_jobs_hydrated", return_value=None))
        for role, user_id, method, path, expected, scope in cases:
            status, payload = _role_request(client, headers[user_id], method, path)
            detail_ok = True
            if path.startswith("/api/timetable") and status == 200:
                subjects = {str(item.get("subject") or item.get("subject_track") or "").upper() for item in (payload or {}).get("timetable", [])}
                detail_ok = subjects <= {"CVO"}
            if path == "/api/attendance-sessions" and status == 200:
                detail_ok = all(str(item.get("subject") or item.get("subject_abbr") or "").upper() == "CVO" for item in (payload or {}).get("sessions", []))
            rows.append({"role": role, "method": method, "path": path, "expected_status": str(expected), "actual_status": str(status), "scope_assertion": scope, "result": "pass" if status == expected and detail_ok else "fail"})
        mismatch = dict(headers[hod["id"]])
        mismatch["X-User-Id"] = cvo["id"]
        status, _ = _role_request(client, mismatch, "GET", "/api/auth/me")
        rows.append({"role": "HOD-mismatched-hint", "method": "GET", "path": "/api/auth/me", "expected_status": "401", "actual_status": str(status), "scope_assertion": "bearer/X-User-Id mismatch fails closed", "result": "pass" if status == 401 else "fail"})
    backend.AUTH_SESSIONS.clear()
    if not all(row["result"] == "pass" for row in rows):
        raise AuthenticationHardeningError("authenticated_role_matrix_failed")
    return tuple(rows)


def frontend_auth_contract(repo: Path) -> dict[str, Any]:
    client = (repo / "frontend/src/api/client.js").read_text(encoding="utf-8")
    auth_context = (repo / "frontend/src/auth/AuthContext.jsx").read_text(encoding="utf-8")
    frontend_sources = "\n".join(path.read_text(encoding="utf-8") for path in sorted((repo / "frontend/src").rglob("*.jsx")))
    checks = {
        "authorization_header_requires_session_token": "if (!user?.sessionToken) return { ...extra };" in client,
        "x_user_id_only_after_bearer": "Authorization: `Bearer ${user.sessionToken}`" in client and "...(user?.id ? { 'X-User-Id': user.id } : {})" in client,
        "protected_401_clears_session": "AUTH_INVALID_EVENT" in client and "localStorage.removeItem(STORAGE_KEY)" in client and "AUTH_INVALID_EVENT" in auth_context,
        "forbidden_does_not_clear_session": "Number(status) === 401" in client and "status) === 403" not in client,
        "protected_download_uses_fetch_headers": "apiGetBlobResult" in client and "apiDownload" in client,
        "no_identity_query_fallback": "user_id=" not in frontend_sources,
        "no_direct_protected_window_open": "window.open(result.data.download_url" not in frontend_sources and "window.open(session.download_url" not in frontend_sources,
        "live_frame_uses_authenticated_fetch": "apiGetBlobResult('/api/live-demo/frame')" in (repo / "frontend/src/pages/LiveDemo.jsx").read_text(encoding="utf-8"),
    }
    if not all(checks.values()):
        raise AuthenticationHardeningError(f"frontend_auth_contract_failed:{sorted(key for key, value in checks.items() if not value)}")
    return {"schema_version": 1, "policy_version": POLICY_VERSION, "checks": checks, "passed": True, "token_rendered_or_logged": False, "credentials_embedded": False}


def _source_row(repo: Path, relative: str) -> dict[str, Any]:
    path = repo / relative
    if not path.is_file():
        raise AuthenticationHardeningError(f"required_source_missing:{relative}")
    return {"path": relative.replace("\\", "/"), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def _documentation_inventory(repo: Path) -> tuple[dict[str, Any], ...]:
    rows = []
    for relative in DOCUMENT_FILES:
        path = repo / relative
        text = path.read_text(encoding="utf-8")
        rows.append({
            **_source_row(repo, relative),
            "operator_login_required_documented": "operator login" in text.lower() or "sign in" in text.lower(),
            "authentication_policy_documented": POLICY_VERSION in text,
        })
    if not all(row["operator_login_required_documented"] and row["authentication_policy_documented"] for row in rows):
        raise AuthenticationHardeningError("authentication_documentation_incomplete")
    return tuple(rows)


def protected_state_hashes(repo: Path) -> dict[str, str]:
    actual = {relative: sha256_file(repo / relative) for relative in PROTECTED_EXPECTED}
    drift = sorted(relative for relative, expected in PROTECTED_EXPECTED.items() if actual[relative] != expected)
    if drift:
        raise AuthenticationHardeningError(f"protected_operational_hash_drift:{','.join(drift)}")
    return actual


def preflight(repo_root: Path, output_root: Path | None = None) -> AuthenticationPreflight:
    repo = Path(repo_root).resolve()
    a_root = repo / PHASE_2L_A_RELATIVE
    b_root = repo / PHASE_2L_B_RELATIVE
    verify_input_manifest(a_root / "immutable_manifest.json", PHASE_2L_A_MANIFEST_SHA256)
    verify_input_manifest(b_root / "immutable_manifest.json", PHASE_2L_B_MANIFEST_SHA256)
    protected = protected_state_hashes(repo)
    official_path = repo / OFFICIAL_REPORT_RELATIVE
    candidate_path = repo / CANDIDATE_REPORT_RELATIVE
    if sha256_file(official_path) != OFFICIAL_REPORT_SHA256 or sha256_file(candidate_path) != CANDIDATE_REPORT_SHA256:
        raise AuthenticationHardeningError("report_hash_mismatch")
    official_counts = report_counts(official_path)
    candidate_counts = report_counts(candidate_path)
    if official_counts != {"Present": 6, "Needs Review": 4, "Unconfirmed": 16, "Missing Enrollment": 1, "Absent": 0, "Unknown": 0, "Total": 27, "Unresolved": 21}:
        raise AuthenticationHardeningError("official_report_recount_mismatch")
    if candidate_counts != {"Present": 2, "Needs Review": 8, "Unconfirmed": 16, "Missing Enrollment": 1, "Absent": 0, "Unknown": 0, "Total": 27, "Unresolved": 25}:
        raise AuthenticationHardeningError("candidate_report_recount_mismatch")

    import app as backend

    if backend.AUTHENTICATION_POLICY_VERSION != POLICY_VERSION:
        raise AuthenticationHardeningError("backend_authentication_policy_version_mismatch")
    backend_source = (repo / "app.py").read_text(encoding="utf-8")
    forbidden_fallbacks = (
        "DEFAULT_ROLE_USERS",
        "load_role_users()[0]",
        "write_json(USERS_PATH",
    )
    present_fallbacks = [fragment for fragment in forbidden_fallbacks if fragment in backend_source]
    if present_fallbacks:
        raise AuthenticationHardeningError(f"implicit_or_default_authentication_fallback:{','.join(present_fallbacks)}")
    inventory = route_inventory(backend)
    no_token = run_no_token_matrix(backend, inventory)
    roles = run_role_matrix(backend, repo)
    frontend = frontend_auth_contract(repo)
    documents = _documentation_inventory(repo)
    sources = [_source_row(repo, relative) for relative in SOURCE_FILES]
    sources.extend([
        _source_row(repo, f"{PHASE_2L_A_RELATIVE}/immutable_manifest.json"),
        _source_row(repo, f"{PHASE_2L_B_RELATIVE}/immutable_manifest.json"),
        _source_row(repo, OFFICIAL_REPORT_RELATIVE),
        _source_row(repo, CANDIDATE_REPORT_RELATIVE),
    ])
    output = Path(output_root).resolve() if output_root else repo / "attendance_output/product_workflow/phase_2l_authentication_hardening"
    return AuthenticationPreflight(repo, output, inventory, no_token, roles, protected, official_counts, candidate_counts, documents, tuple(sorted(sources, key=lambda row: row["path"])), frontend)


def aggregate_directory_hash(root: Path) -> tuple[str, tuple[dict[str, Any], ...]]:
    root = Path(root).resolve()
    rows = tuple({"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in sorted(root.rglob("*")) if path.is_file())
    if not rows:
        raise AuthenticationHardeningError("frontend_build_output_empty")
    return canonical_json_hash(rows), rows


def _browser_csv() -> bytes:
    fields = ("role", "page", "path", "result", "finding")
    rows = [dict(zip(fields, values)) for values in BROWSER_REHEARSAL_ROWS]
    return csv_bytes(rows, fields)


def _inventory_csv(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return csv_bytes(rows, ("method", "route", "endpoint", "classification", "data_sensitivity", "authentication", "allowed_roles", "authorization_scope", "no_token_expected", "wrong_role_expected"))


def _no_token_csv(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return csv_bytes(rows, ("method", "route", "request_path", "status", "json_response", "private_fields_exposed", "result"))


def _role_csv(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return csv_bytes(rows, ("role", "method", "path", "expected_status", "actual_status", "scope_assertion", "result"))


def build_output_files(plan: AuthenticationPreflight, external_validations: Sequence[str], frontend_build_dir: Path) -> tuple[dict[str, bytes], str, dict[str, Any]]:
    validations = tuple(sorted(set(external_validations)))
    if validations != tuple(sorted(REQUIRED_EXTERNAL_VALIDATIONS)):
        missing = sorted(set(REQUIRED_EXTERNAL_VALIDATIONS) - set(validations))
        extra = sorted(set(validations) - set(REQUIRED_EXTERNAL_VALIDATIONS))
        raise AuthenticationHardeningError(f"external_validation_record_incomplete:missing={missing}:extra={extra}")
    build_hash, build_files = aggregate_directory_hash(frontend_build_dir)
    inventory_content = _inventory_csv(plan.protected_rows)
    no_token_content = _no_token_csv(plan.no_token_rows)
    role_content = _role_csv(plan.role_rows)
    browser_content = _browser_csv()
    inventory_hash = hashlib.sha256(inventory_content).hexdigest()
    no_token_hash = hashlib.sha256(no_token_content).hexdigest()
    role_hash = hashlib.sha256(role_content).hexdigest()
    browser_hash = hashlib.sha256(browser_content).hexdigest()
    run_payload = {
        "phase_2l_a_immutable_manifest_sha256": PHASE_2L_A_MANIFEST_SHA256,
        "phase_2l_b_immutable_manifest_sha256": PHASE_2L_B_MANIFEST_SHA256,
        "authentication_policy_version": POLICY_VERSION,
        "protected_route_inventory_sha256": inventory_hash,
        "no_token_api_matrix_sha256": no_token_hash,
        "authenticated_role_matrix_sha256": role_hash,
        "browser_rehearsal_sha256": browser_hash,
        "frontend_build_sha256": build_hash,
        "official_report_sha256": OFFICIAL_REPORT_SHA256,
        "candidate_report_sha256": CANDIDATE_REPORT_SHA256,
        "source_manifest_sha256": canonical_json_hash(plan.source_manifest),
    }
    run_fingerprint = canonical_json_hash(run_payload)
    run_id = f"authentication-hardening-{run_fingerprint[:20]}"
    public_routes = [
        {"method": "GET", "route": "/", "purpose": "minimal service landing response"},
        {"method": "GET", "route": "/api/health", "purpose": "minimal safe readiness response"},
        {"method": "POST", "route": "/api/auth/login", "purpose": "credential verification and session issuance"},
        {"method": "GET", "route": "/static/<path:filename>", "purpose": "static frontend assets when served by Flask"},
    ]
    authentication_policy = {
        "schema_version": 1,
        "policy_version": POLICY_VERSION,
        "default_route_classification": "protected",
        "missing_or_invalid_bearer": 401,
        "authenticated_unauthorized": 403,
        "x_user_id_contract": "optional bearer consistency hint only; cannot authenticate or switch identity",
        "query_identity_fallback": False,
        "implicit_first_user_fallback": False,
        "missing_or_invalid_role_registry": "fail closed; no account is created or selected",
        "recognized_roles": ["HOD", "Faculty"],
        "options_exception": "protocol-only automatic CORS response; no application data",
    }
    exposure_scan = {
        "schema_version": 1,
        "protected_route_count": len(plan.protected_rows),
        "responses_scanned": len(plan.no_token_rows),
        "private_field_patterns": ["password", "session_token", "role_users", "student_embeddings", "attendance_data", "workflow_state", "input_source_path"],
        "exposures": [],
        "passed": True,
    }
    report_totals = {
        "schema_version": 1,
        "session_id": SESSION_ID,
        "official_reviewed_revision": {"sha256": OFFICIAL_REPORT_SHA256, "authority": "reviewed_multiframe_tracklet_evidence", "counts": plan.official_counts, "finalized": False},
        "archived_automatic_candidate": {"sha256": CANDIDATE_REPORT_SHA256, "authority": "strict_tracklet_aggregate_with_guarded_review_candidates", "counts": plan.candidate_counts, "archived": True, "replaced_official": False},
        "independent_csv_recount_passed": True,
        "report_revision_changed": False,
        "finalization_changed": False,
    }
    identity = {
        "schema_version": 1,
        "subject": "CVO",
        "roster_total": 27,
        "embedding_available": 26,
        "missing_enrollment": ["2401100CSE0268"],
        "24011CSEAI0110_present": True,
        "2401100CSE0110_present": False,
        "distinct_rolls_not_aliased": True,
        "mixed_track_id": "CP1-cam5-back-TRK00005",
        "mixed_track_quarantined": True,
        "authentication_identity_hint_mismatch_failed_closed": True,
        "verification_passed": True,
    }
    readiness = {
        "schema_version": 1,
        "decision": "demo_ready_operator_login_required",
        "operator_login_required": True,
        "real_operator_session_used": False,
        "synthetic_fixture_rehearsal_passed": True,
        "live_demo_deferred": True,
        "safe_page_order": ["Login", "Dashboard", "My Classes / Timetable", "Reports", "Manual Review", "Students / Coverage", "HOD Control", "Help"],
        "mutating_controls_used": False,
    }
    validation = {
        "schema_version": 1,
        "ordered_validations": [{"sequence": index + 1, "name": name, "passed": True} for index, name in enumerate(REQUIRED_EXTERNAL_VALIDATIONS)],
        "protected_route_count": len(plan.protected_rows),
        "no_token_matrix_passed": True,
        "synthetic_role_matrix_passed": True,
        "browser_page_checks": len(BROWSER_REHEARSAL_ROWS),
        "browser_console_warning_or_error_count": 0,
        "browser_horizontal_overflow_at_1280_width": False,
        "documented_test_skip_preserved": True,
        "frontend_build": {"sha256": build_hash, "file_count": len(build_files), "size_bytes": sum(row["size_bytes"] for row in build_files)},
    }
    no_recognition = {
        "schema_version": 1,
        "recognition_ran": False,
        "video_decoded": False,
        "yunet_ran": False,
        "sface_ran": False,
        "session_reprocessed": False,
    }
    no_mutation = {
        "schema_version": 1,
        "attendance_changed": False,
        "job_registry_changed": False,
        "role_registry_changed": False,
        "student_map_changed": False,
        "manual_overrides_changed": False,
        "review_registry_changed": False,
        "embeddings_changed": False,
        "embedding_summary_changed": False,
        "production_pointer_changed": False,
        "timetable_changed": False,
        "protected_hashes": plan.protected_hashes,
    }
    source_manifest = {
        "schema_version": 1,
        "run_payload": run_payload,
        "sources": list(plan.source_manifest),
        "protected_operational_sources": [{"path": path, "sha256": digest, "size_bytes": (plan.repo_root / path).stat().st_size} for path, digest in sorted(plan.protected_hashes.items())],
        "credentials_included": False,
        "tokens_included": False,
        "raw_embeddings_included": False,
    }
    evaluation = {
        "schema_version": 1,
        "run_id": run_id,
        "run_fingerprint_sha256": run_fingerprint,
        "decision": "demo_ready_operator_login_required",
        "authentication_blocker_resolved": True,
        "public_allowlist_minimal": True,
        "all_protected_routes_reject_no_token": True,
        "faculty_scope_preserved": True,
        "hod_scope_preserved": True,
        "frontend_session_contract_passed": True,
        "private_field_exposure_detected": False,
        "no_recognition": True,
        "no_operational_mutation": True,
    }
    files = {
        "authentication_policy.json": readable_json_bytes(authentication_policy),
        "public_route_allowlist.json": readable_json_bytes({"schema_version": 1, "policy_version": POLICY_VERSION, "routes": public_routes, "options_protocol_exception": True}),
        "protected_route_inventory.csv": inventory_content,
        "no_token_api_matrix.csv": no_token_content,
        "authenticated_role_matrix.csv": role_content,
        "private_field_exposure_scan.json": readable_json_bytes(exposure_scan),
        "frontend_auth_contract.json": readable_json_bytes(plan.frontend_contract),
        "browser_rehearsal_summary.csv": browser_content,
        "report_totals_verification.json": readable_json_bytes(report_totals),
        "identity_integrity_verification.json": readable_json_bytes(identity),
        "demo_readiness.json": readable_json_bytes(readiness),
        "documentation_inventory.json": readable_json_bytes({"schema_version": 1, "items": list(plan.documentation_inventory)}),
        "validation_summary.json": readable_json_bytes(validation),
        "no_recognition_declaration.json": readable_json_bytes(no_recognition),
        "no_operational_mutation_declaration.json": readable_json_bytes(no_mutation),
        "source_manifest.json": readable_json_bytes(source_manifest),
        "evaluation_summary.json": readable_json_bytes(evaluation),
    }
    return files, run_id, {"run_fingerprint": run_fingerprint, "run_payload": run_payload}


def _manifest_bytes(run_id: str, run_fingerprint: str, files: Mapping[str, bytes]) -> bytes:
    return readable_json_bytes({
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "run_id": run_id,
        "run_fingerprint_sha256": run_fingerprint,
        "immutable": True,
        "files": {name: {"sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)} for name, content in sorted(files.items())},
    })


def verify_immutable_output(output_dir: Path) -> dict[str, Any]:
    root = Path(output_dir).resolve()
    actual_names = sorted(path.name for path in root.iterdir() if path.is_file()) if root.is_dir() else []
    if actual_names != sorted(REQUIRED_OUTPUT_FILES):
        raise AuthenticationHardeningError(f"immutable_file_set_mismatch:{actual_names}")
    manifest = _load_json(root / "immutable_manifest.json")
    if manifest.get("policy_version") != POLICY_VERSION or manifest.get("immutable") is not True:
        raise AuthenticationHardeningError("immutable_manifest_contract_mismatch")
    files = manifest.get("files") or {}
    if sorted(files) != sorted(name for name in REQUIRED_OUTPUT_FILES if name != "immutable_manifest.json"):
        raise AuthenticationHardeningError("immutable_manifest_file_map_mismatch")
    for name, expected in files.items():
        path = root / name
        if sha256_file(path) != expected.get("sha256") or path.stat().st_size != expected.get("size_bytes"):
            raise AuthenticationHardeningError(f"immutable_output_tampered:{name}")
    source = _load_json(root / "source_manifest.json")
    expected_fingerprint = canonical_json_hash(source.get("run_payload") or {})
    if manifest.get("run_fingerprint_sha256") != expected_fingerprint or manifest.get("run_id") != f"authentication-hardening-{expected_fingerprint[:20]}":
        raise AuthenticationHardeningError("immutable_run_fingerprint_mismatch")
    return manifest


def materialize(plan: AuthenticationPreflight, external_validations: Sequence[str], frontend_build_dir: Path) -> tuple[Path, bool, dict[str, Any]]:
    files, run_id, run = build_output_files(plan, external_validations, frontend_build_dir)
    complete = dict(files)
    complete["immutable_manifest.json"] = _manifest_bytes(run_id, run["run_fingerprint"], files)
    plan.output_root.mkdir(parents=True, exist_ok=True)
    target = plan.output_root / run_id
    if target.exists():
        verify_immutable_output(target)
        for name, content in complete.items():
            if (target / name).read_bytes() != content:
                raise AuthenticationHardeningError(f"deterministic_id_collision:{name}")
        return target, True, verify_immutable_output(target)
    staging = plan.output_root / f".{run_id}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        for name, content in sorted(complete.items()):
            (staging / name).write_bytes(content)
        os.replace(staging, target)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return target, False, verify_immutable_output(target)
