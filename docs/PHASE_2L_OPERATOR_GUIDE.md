# Product Phase 2L operator guide

## Phase 2L-C authentication gate

Policy `product-phase-2l-c-explicit-authentication-v1` is mandatory for the controlled demo. The only public application endpoints are `GET /`, `GET /api/health`, and `POST /api/auth/login`; static frontend assets and automatic empty `OPTIONS` responses are protocol exceptions. Every report, review, attendance, student, timetable, job, HOD, configuration, diagnostics, upload, processing, download, and Live Demo endpoint requires a valid bearer session.

Start the rehearsal at the Login page and use a real operator login supplied outside this repository. `X-User-Id` is an optional consistency hint after bearer authentication; it is not a login mechanism. A missing, malformed, expired, or mismatched session returns `401`. An authenticated Faculty request for HOD-only or another-faculty data returns `403`. Never put credentials or tokens in commands, URLs, screenshots, notes, logs, or evidence artifacts.

If a real operator login is unavailable, the release decision is `demo_ready_operator_login_required`. A temporary isolated synthetic fixture may be used only for visual rehearsal; it does not establish production login readiness. Do not visit Live Demo or use any mutating control during the rehearsal.

This guide starts the current safe application for a controlled, read-only professor demonstration. It does not authorize recognition, video decoding, reprocessing, attendance edits, finalization, configuration changes, or Live Demo use.

## Required preflight

Open PowerShell and run:

```powershell
Set-Location -LiteralPath "F:\sohail\Class_Attendance_YuNet_SFace"
& ".\scripts\run_product_phase_2l_auth_preflight.ps1" -Command preflight
& ".\scripts\run_product_phase_2l_stabilization_preflight.ps1" -Command preflight
```

Continue only when both commands print `PREFLIGHT_STATUS=PASS`. The authentication preflight verifies the minimal public allowlist, all protected routes, the complete no-token matrix, synthetic Faculty/HOD scope, frontend bearer behavior, both predecessor manifests, report totals, and protected hashes. The stabilization preflight verifies the frozen model/policy, roster, mixed quarantine, rollback references, frontend/backend release contracts, and operational hashes. Neither command decodes video or runs recognition.

## Start the backend

In the first PowerShell window:

```powershell
Set-Location -LiteralPath "F:\sohail\Class_Attendance_YuNet_SFace"
& ".\venv\Scripts\python.exe" ".\app.py"
```

Expected address: `http://127.0.0.1:5000`.

Read-only health check from another PowerShell window:

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:5000/api/health"
```

Do not use `/api/process`, `/api/reprocess`, attendance edit/finalize endpoints, configuration write endpoints, or Live Demo endpoints during the controlled demo.

## Start the frontend

In the second PowerShell window:

```powershell
Set-Location -LiteralPath "F:\sohail\Class_Attendance_YuNet_SFace\frontend"
& ".\node_modules\.bin\vite.cmd" --host 127.0.0.1
```

Expected address: `http://127.0.0.1:5173`.

Read-only frontend check:

```powershell
Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:5173"
```

## Login roles

- Faculty: sees only assigned subjects and roster rows; use an existing authorized faculty account.
- HOD: sees cross-subject reports, governance, and model coverage; use the existing authorized HOD account.

Do not put usernames, passwords, tokens, or session credentials in files, shell history, screenshots, or this guide.

## Safe report and review workflow

1. Open Dashboard or My Classes without selecting Process Attendance.
2. Open Reports and load session `2026-06-22__B51__P4__CVO`.
3. Confirm the Official Reviewed Revision remains 6 Present, 4 Needs Review, 16 Unconfirmed, 1 Missing Enrollment, 0 Absent, total 27.
4. Confirm the Automatic Candidate Revision remains 2 Present, 8 Needs Review, 16 Unconfirmed, 1 Missing Enrollment, 0 Absent, total 27 and is archived rather than official.
5. Open Review Students and load the same session. Inspect only; do not select a correction, type a reason, save, export, or finalize.
6. For HOD view, open HOD Control and inspect production recognition state only. Do not create/edit faculty, assign subjects, apply timetable changes, or roll back configuration.

Status meanings:

- Present: sufficient authoritative identity evidence or a traceable accepted review.
- Needs Review: meaningful but unresolved evidence.
- Unconfirmed: insufficient camera evidence; not an absence.
- Missing Enrollment: rostered student has no usable production embedding.
- Absent: explicit resolved outcome; never inferred only from non-detection.
- Unknown: unrecognized legacy status that fails closed.
- Unresolved total: Needs Review + Unconfirmed + Missing Enrollment + Unknown.

## Prohibited demo operations

- Do not click Process Attendance, Reprocess, Retry, Cancel, Reset Processing, Save Changes, Export Reviewed CSV, or Finalize Attendance.
- Do not open or use Live Demo.
- Do not edit HOD configuration, users, subject assignments, timetable, roster, credentials, thresholds, models, or embeddings.
- Do not run recognition scripts or commands that open classroom video.

## Shutdown

Press `Ctrl+C` once in the frontend PowerShell window and once in the backend PowerShell window. After both stop, rerun the Phase 2L preflight command. Stop and investigate if protected hashes or report totals differ.

## Troubleshooting

- Backend unavailable: confirm the backend window is still running, then repeat the `/api/health` request. Do not substitute a recognition command.
- Frontend unavailable: confirm the Vite window is running and port 5173 is listening.
- Port check: `Get-NetTCPConnection -State Listen -LocalPort 5000,5173`.
- Login rejected: use the existing authorized login flow; do not reset or expose credentials for the demo.
- Session expired or protected request returned `401`: the frontend clears the stale local session and returns to Login; authenticate again through the normal operator flow.
- Authenticated request returned `403`: keep the valid session, confirm the intended role and assigned subject, and do not broaden access or switch identity through headers or URLs.
- Report missing or totals differ: stop the demo. Do not process, repair, edit, or finalize the session. Rerun the preflight and use the recovery guide for diagnosis only.
- Preflight failure: treat the release candidate as unavailable until the exact mismatch is resolved without changing protected operational state.
