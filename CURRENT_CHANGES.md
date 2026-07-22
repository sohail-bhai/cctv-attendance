# Product Phase 2L-C: explicit authentication enforcement and controlled rehearsal

## 1. Run title and timestamp

- Run: Product Phase 2L-C explicit authentication enforcement for protected reads and final controlled demo rehearsal.
- Date: 2026-07-22, Asia/Calcutta.
- Authentication policy: `product-phase-2l-c-explicit-authentication-v1`.
- Blocker closed: `phase2l-b-unauthenticated-default-hod-read-access`.

## 2. Goal and scope

The sole implementation target was to remove implicit/default user resolution and require a valid bearer session on every protected Flask route while retaining HOD and Faculty authorization scope. The phase also completed isolated negative/security tests, frontend session compatibility, the controlled synthetic authenticated browser rehearsal, immutable evidence, and stopped-state preservation checks. It did not authorize recognition, video decoding, processing, reprocessing, attendance edits, finalization, configuration changes, credential changes, rollback, commit, push, deployment, or promotion.

## 3. Git baseline

- Branch: `main`.
- HEAD: `03b06a451bd411f5efc3523e800ef0a6f3901386`.
- The worktree was already dirty with prior Product Phase 2K/2L and user changes. The tracked and untracked baseline was captured before editing; no reset, clean, checkout, discard, or normalization was used.
- Existing dirty entries were preserved, including the prior context/change documents, backend/frontend work, Phase 2K/2L scripts/modules/tests/docs, and frontend contract work.
- No commit, push, branch change, deployment, or repository cleanup was performed.

## 4. Exact Phase 2L-C files changed

Implementation and frontend compatibility:

- `app.py`
- `frontend/src/api/client.js`
- `frontend/src/auth/AuthContext.jsx`
- `frontend/src/pages/Dashboard.jsx`
- `frontend/src/pages/LiveDemo.jsx`
- `frontend/src/pages/ManualReview.jsx`
- `frontend/src/pages/Reports.jsx`
- `frontend/src/pages/Timetable.jsx`

Tests and controlled tooling:

- `frontend/tests/authContract.test.mjs`
- `tests/test_product_phase_2d_reports_review.py`
- `tests/test_product_phase_2e_hod_control.py`
- `tests/test_product_phase_2l_authentication.py`
- `src/face_attendance/product_phase_2l_authentication.py`
- `scripts/run_product_phase_2l_auth_preflight.py`
- `scripts/run_product_phase_2l_auth_preflight.ps1`
- `scripts/run_product_phase_2l_auth_fixture.py`

Documentation and handoff:

- `docs/PHASE_2L_OPERATOR_GUIDE.md`
- `docs/PHASE_2L_PROFESSOR_DEMO_RUNBOOK.md`
- `docs/PHASE_2L_FINAL_RELEASE_CHECKLIST.md`
- `docs/PHASE_2L_RECOVERY_AND_ROLLBACK.md`
- `PROJECT_CONTEXT.md`
- `CURRENT_CHANGES.md`

Immutable outputs were added under `attendance_output/product_workflow/phase_2l_authentication_hardening/`. The final directory and one earlier immutable attempt each contain exactly these 18 names: `authentication_policy.json`, `public_route_allowlist.json`, `protected_route_inventory.csv`, `no_token_api_matrix.csv`, `authenticated_role_matrix.csv`, `private_field_exposure_scan.json`, `frontend_auth_contract.json`, `browser_rehearsal_summary.csv`, `report_totals_verification.json`, `identity_integrity_verification.json`, `demo_readiness.json`, `documentation_inventory.json`, `validation_summary.json`, `no_recognition_declaration.json`, `no_operational_mutation_declaration.json`, `source_manifest.json`, `evaluation_summary.json`, and `immutable_manifest.json`.

## 5. Root cause

`get_current_user()` accepted identity hints without requiring a bearer and fell back to the first role-registry user when no request identity was supplied. Because the first record was HOD, protected reads could return private HOD-scoped data to an unauthenticated request. Several internal timetable/job helpers also selected `load_role_users()[0]`, and the backend retained fallback account literals that could recreate accounts if the role registry was missing. These paths violated fail-closed authentication even though `/api/hod/config` had a stronger explicit HOD guard.

## 6. Authentication design

- `get_current_user()` now resolves identity only from a valid, unexpired, non-revoked in-memory bearer session.
- `X-User-Id` is only an optional post-bearer consistency hint. It cannot authenticate or switch identity; mismatch returns `401`.
- Missing, empty, malformed, unknown, expired, revoked, or inconsistent sessions are unauthenticated.
- A central `before_request` guard makes protected the default for every registered route.
- `require_authenticated_user`, `require_role`, and `require_hod` provide one consistent `401`/`403` contract.
- Only recognized HOD/admin and Faculty roles are accepted. Other roles fail closed with `403`.
- Missing/invalid role-registry state returns no users. The backend no longer creates fallback accounts or selects a first/default user.
- HOD-only inventory/configuration/system routes remain centrally classified and existing explicit HOD write guards remain intact.
- Faculty attendance, timetable, student, job, review, and report reads remain subject/session scoped.
- Faculty report downloads are limited to exact assigned-session report references and the two generated reviewed/final CSV families; HOD retains authorized global download scope.
- Unauthorized report/download existence is not disclosed through a different response.
- Automatic empty `OPTIONS` responses remain a protocol-only CORS exception and execute no route body.

## 7. Public routes

The explicit public application allowlist is exactly:

- `GET /`
- `GET /api/health`
- `POST /api/auth/login`
- `GET /static/<path:filename>` for Flask-served static assets

`HEAD` is normalized to `GET`; automatic empty `OPTIONS` is documented separately as a protocol exception. Health returns only minimal readiness fields and the authentication-policy version. It does not return users, roles, rosters, attendance, private report details, workflow state, operational paths, credentials, or tokens.

## 8. Protected-route inventory

- Registered protected method/route pairs: 37.
- Inventory columns include method, Flask rule, endpoint, classification, sensitivity, authentication, allowed roles, Faculty/HOD scope, expected no-token result, and expected wrong-role result.
- Coverage includes profile/current user, logout, timetable/classes, processing/reprocessing/reset, jobs/status/cancel, attendance sessions/reports/review/edit/export/finalize, students, report/download endpoints, videos/upload, HOD overview/configuration, diagnostics, and Live Demo.
- Inventory SHA-256: `a1f9691c5e2cc48133d5c73d6cf86263fa97c1dbd0fb36f61248795426edbc7a`.

## 9. No-token matrix

- All 37 protected method/route pairs returned HTTP `401`.
- Every response was generic JSON exactly equivalent to `{"error":"Authentication required.","success":false}`.
- No response exposed password/session, role-registry, embedding, attendance-row, workflow-state, or operational-path fields.
- `X-User-Id` alone, query identity, malformed authorization, empty bearer, invalid bearer, unknown session, and bearer/hint mismatch all fail closed.
- Repeated controlled preflights were deterministic.
- Matrix SHA-256: `82249181e337c527a493293c5958d888290db623937a6db31eb14eb12d23556d`.

## 10. Faculty/HOD role matrix

- 17 isolated synthetic role checks passed.
- Authorized HOD identity, reports, attendance, downloads, and overview reads returned `200`.
- Assigned CVO Faculty profile, timetable, session list, attendance report, and report download returned `200` with CVO-only scope.
- SWE Faculty access to the CVO report and download returned `403`.
- Faculty access to cross-faculty reports, HOD overview, and reset processing returned `403` before route bodies could expose or mutate state.
- An unrecognized authenticated role returned `403`.
- Bearer plus mismatched identity hint returned `401`.
- Role-matrix SHA-256: `be06008921b9389133e7aebecce58827b809c4c26d1042ebeeba4c24d61fd74d`.

## 11. Frontend behavior

- Protected JSON, upload, and blob/download requests include `Authorization: Bearer ...` only when the existing session token is present.
- `X-User-Id` is sent only alongside bearer authentication.
- No identity query fallback remains, including the Live Demo frame request.
- A protected `401` removes stale local session state and dispatches the authentication-invalid event so routing returns cleanly to Login.
- A `403` preserves the valid session and remains an authorization error.
- Login `401` stays a generic invalid-login response and does not trigger the protected-session invalidation path.
- Direct protected `window.open`/plain-link downloads were replaced by authenticated fetch/blob downloads.
- No credential fields are displayed or prefilled, no quick-fill/demo account exists, and no token is rendered or logged.
- Manual Review now constrains its page/card width; the horizontal-overflow issue found during visual rehearsal was rechecked as resolved.

## 12. Browser rehearsal

- Real operator credentials were not available and were not read. The rehearsal therefore used an isolated temporary fixture with randomized synthetic authentication and is not evidence of production credential usability.
- Thirteen recorded browser checks passed: Login; HOD Dashboard; My Classes/Timetable; Official Reviewed MON_P4; Archived Automatic Candidate; Manual Review; Students/Coverage; HOD Control; Help; and Faculty Dashboard, Reports, Manual Review, and Students/Coverage.
- Official reporting visibly showed 6 Present, 4 Needs Review, 16 Unconfirmed, 1 Missing Enrollment, 0 Absent, unresolved 21, and not finalized.
- The archived automatic candidate visibly remained separate at 2 Present, 8 Needs Review, 16 Unconfirmed, 1 Missing Enrollment, 0 Absent, unresolved 25.
- Faculty navigation and data remained CVO-only; Dashboard showed 4 CVO slots and 27 students. Faculty Students showed 27 CVO rows, 26 covered, and one coverage issue.
- Missing Enrollment `2401100CSE0268`, long roll values, status labels, official/candidate distinction, and read-only HOD controls were visible.
- The active 1280-wide laptop viewport had no document-level horizontal overflow. Browser warning/error count was zero.
- No Process, Reprocess, Retry, Cancel, Reset, upload, correction, export, finalization, configuration, or rollback control was used. Live Demo was not visited.
- Browser matrix SHA-256: `83e24b0b4c664409f7c376069b7a12ada6a52c6838f65f6efb00210a54e64a91`.

## 13. Credential and private-data exposure

- No operational password file was read and no real credential, bearer value, cookie, or browser session store was inspected or printed.
- Synthetic credentials existed only in exact temporary fixture files under `C:\tmp` while the controlled servers ran. Both fixture files were removed after their services stopped; the deletions were direct temporary-file cleanup and are not recoverable through the application.
- Temporary full-test logs and frontend build directories created by this phase were also removed after evidence materialization.
- Immutable evidence contains no credentials, bearer values, cookies, private joins, hidden predictions, or raw embeddings.

## 14. Tests, build, and full regression

- Python compilation: pass.
- New targeted authentication/evidence suite: 14/14 pass.
- Existing report/HOD/configuration/workflow plus Phase 2L-A regression group: 61/61 pass.
- Frontend Node suite: 42/42 pass, including the prior 38 Phase 2L tests and 4 authentication-contract tests.
- Vite production build: pass; 50 modules, 13 output files, 527531 bytes.
- Frontend build aggregate SHA-256: `7e2c1ec68381ae1f7c9079d3413756edc0f9efe7f47cbc1d7d64167728ec70e3`.
- Complete Python discovery suite against final source: 640 tests, zero failures/errors, one documented real Phase 1.2E/H artifact test skipped. The skip was not hidden or converted.
- Final authentication and Phase 2L-A stabilization preflights passed after service shutdown.

## 15. Immutable output and hashes

Final authoritative output:

- Directory: `attendance_output/product_workflow/phase_2l_authentication_hardening/authentication-hardening-dc1e4eb0bdbef36515af`
- Run fingerprint: `dc1e4eb0bdbef36515af63ec00ab8e57db0ea5993a39ac78ac8c1bc9c9ac26db`
- Immutable manifest SHA-256: `c8d5a38d1e89cf0928e1cc18ff91e87b41ce0b1b237c32dddcdf9ac8e50dfac7`
- Source-manifest SHA-256: `75069a37b67737d54d1ec2b2da16b215098d383ed840d79e0f43ea8f11a7563e`
- Exact file count: 18; all 17 payload hashes/sizes independently verified.
- Phase 2L-A manifest input: `6e86e63c5da4945cb133b6c8be3250340be1d0ede3c512531e9c44f6ced99153`.
- Phase 2L-B manifest input: `e58a942a4bd091bdf8fc5f75d0351cbe68bfaef47d90f7bfd1c7678356f41c3d`.
- Official report SHA-256: `0211ac1433831aceb4176ee199f64bf962dbc784765a99a00bed74c7b38d51e3`.
- Candidate report SHA-256: `e28dc0278abfad034bd78bb576620a50f90100461f23b5e0b920e1e65e39c57f`.
- First materialization created the final directory; repeated materialization reused it byte-for-byte. The independent verifier and independent PowerShell file-set/hash verifier passed. Isolated deterministic collision and byte-tamper tests failed closed as intended.

An earlier immutable attempt, `authentication-hardening-f9028874ba95ab1f39bc` with manifest SHA-256 `85deeb5dbe4d34c3eb4a3e66b37300c14dcd35b205436249c7f167430ea72f79`, remains preserved and superseded. It preceded binding the complete Timetable/Students/Help browser matrix and source-manifest fingerprint into the deterministic run payload. It was not rewritten or deleted.

## 16. Protected operational hashes

- `data/attendance_status.json`: `46fd2df55cc3f61c5fa03c893937eb2828b9cc2af064ea9e427276b79a9d0b46`
- `data/job_runtime.json`: `5a23f09654d91fac09804cd97c2fa114bf4956a914f2e02ce2b7090f730141ed`
- `data/role_users.json`: `77125140004294f2834fc869fd590e167d8ab78cff5dbeda1c074ab76c177517`
- `data/student_faculty_map.json`: `0eca63ea58d12a73d3bd11e620b74be733a3219c9244e8a4a267d5b02acd3f91`
- `data/manual_overrides.json`: `e9c6bd35c23c353795edb5d3c02e808793d7cbd90c7b0a87fdfc686f8fd5cdcf`
- `data/review_evidence_registry.json`: `3b04e9aecfdfb1f64346c4a0e709fd9c36d7c56545bf816d6641b4c2a2e80841`
- `models/student_embeddings.pkl`: `f088d827adc548ee95f46566d758fd71fc304d042c43f1ecffc6526b60bcd832`
- `models/embedding_summary.csv`: `63885588c374c37f4da9bf85294f240bdf0f28cb585e77b894ab516139ca46ae`
- `models/current_embedding_version.json`: `7999b8ccf787dca9b8fb862f4e53ce7c1102eb729a3732c82fee76f3ba3ee05e`
- `timetable_b51_2026_2027.csv`: `10bbd578a859fbd4fa228c4eb4f1f92e7de5b92a1c4236d71a00337eecac9c57`

All match the pre-edit Phase 2L freeze.

## 17. Recognition, video, and state-change declaration

Recognition, YuNet, SFace, classroom-video decoding, processing, reprocessing, Live Demo, attendance edits, report export/finalization, candidate/authority revision, guarded promotion, threshold/checkpoint/tracklet/zone changes, embedding rebuild/promotion, roster/timetable/manual-override/review-registry/job/HOD-configuration/role changes, and rollback all remained false. The official and candidate files, job runtime, finalization state, authority, review registry, production family/pointer, thresholds 0.48/0.08, and mandatory mixed-track quarantine remain unchanged. The two controlled services were stopped; ports 5000 and 5174 were released. The unrelated pre-existing Vite listener on port 5173 was left untouched.

## 18. Demo readiness decision

`demo_ready_operator_login_required`

The unauthenticated exposure blocker is closed, all synthetic route/browser checks pass, and state is unchanged. `demo_ready` is intentionally not claimed because no real authorized operator completed the production login flow during this run.

## 19. Exact real-demo operator steps

1. Run `scripts/run_product_phase_2l_auth_preflight.ps1` and `scripts/run_product_phase_2l_stabilization_preflight.ps1`; require `PREFLIGHT_STATUS=PASS` from both.
2. Start the documented Flask and Vite services once and confirm the minimal health response plus zero active jobs.
3. Have the authorized operator enter the intended Faculty or HOD credential privately through Login.
4. Confirm the role-scoped landing page before showing protected data.
5. Follow the documented order: Dashboard, My Classes/Timetable, Reports with official and archived revisions, Manual Review, Students/Coverage, HOD Control for HOD, and Help.
6. Use no mutating control and do not visit Live Demo.
7. Log out normally, stop only the controlled services, rerun both preflights, rehash protected state, and recount both reports.

## 20. Remaining limitations

- Production credential usability still requires a real authorized operator login rehearsal.
- The isolated synthetic fixture proves authentication, authorization, frontend session behavior, and visual layout only.
- All existing recognition sessions remain retrospective/prior-use contaminated; no independent generalization claim is permitted.
- A genuinely new frozen CVO/B51 session remains required before any future guarded-recovery/generalization claim.
- The earlier superseded immutable authentication attempt remains preserved for audit and is not the authoritative Phase 2L-C package.

## 21. Recommended next task

No further coding is recommended. Perform only the authorized operator-login production rehearsal using the existing Phase 2L-C build and documented read-only page order. If that succeeds with stopped-state verification, record `demo_ready`; otherwise fix only the newly evidenced blocker. Independent-session acquisition remains a separate later phase and still requires explicit authorization.

## 22. Planner handoff

Use `authentication-hardening-dc1e4eb0bdbef36515af` as the sole authoritative Phase 2L-C output. Begin by verifying its immutable manifest and running both preflights. Do not use the superseded `authentication-hardening-f9028874ba95ab1f39bc` as release evidence. Do not modify authentication, recognition, attendance, authority, configuration, or operational data unless a new narrowly scoped task explicitly authorizes it. The only open demo action is a real operator login plus the documented read-only rehearsal and final stopped-state verification.
