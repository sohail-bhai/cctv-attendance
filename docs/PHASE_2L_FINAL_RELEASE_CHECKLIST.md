# Product Phase 2L final release checklist

## Phase 2L-C explicit-authentication checks

Authentication policy: `product-phase-2l-c-explicit-authentication-v1`.

- [x] Only `GET /`, `GET /api/health`, and `POST /api/auth/login` are public application endpoints; static assets and automatic empty `OPTIONS` responses are documented protocol exceptions.
- [x] Every protected route returns generic JSON `401` without a valid bearer and exposes no private fields.
- [x] `X-User-Id` alone and identity query parameters cannot authenticate; a bearer/hint mismatch fails closed.
- [x] Authenticated Faculty can read only assigned-subject data and receives `403` for HOD-only and other-faculty resources.
- [x] Authenticated HOD can read the HOD-controlled inventory without changing operational state.
- [x] Frontend protected requests and downloads send bearer authentication; protected `401` clears stale state and `403` preserves the unrelated valid session.
- [x] The decision is exactly `demo_ready_operator_login_required` because no real operator login was available; isolated synthetic HOD/Faculty browser checks completed.
- [x] Credentials and tokens are absent from source literals, URLs, logs, screenshots, browser evidence, and immutable artifacts.
- [x] Live Demo was not visited, and no Process, Reprocess, Edit, Finalize, upload, reset, or configuration action was used.

Phase 2L-C release decision: **`demo_ready_operator_login_required`**. The Phase 2L-B implicit-default-HOD blocker is closed: every registered protected route returns generic `401` without a valid bearer, authenticated scope violations return `403`, and the isolated synthetic browser rehearsal passed. A real authorized operator must still complete the production login rehearsal before the professor demonstration.

This checklist never contains credentials and does not authorize recognition, reprocessing, review edits, export, finalization, configuration writes, rollback, or Live Demo.

## 1. Pre-demo machine checklist

- [x] Product Phase 2L-C authentication fix is complete and separately validated.
- [ ] `git status --short` is captured and all pre-existing work is identified.
- [ ] Branch and HEAD are recorded.
- [ ] The exact Phase 2L-A immutable manifest SHA-256 is `6e86e63c5da4945cb133b6c8be3250340be1d0ede3c512531e9c44f6ced99153`.
- [ ] The Phase 2L stabilization preflight prints `PREFLIGHT_STATUS=PASS`.
- [ ] Ports 5000 and 5173 are either free or already owned by the explicitly identified rehearsal services.
- [ ] Protected operational hashes are recorded before startup.
- [ ] No recognition, video-decoding, rollback, configuration, or attendance command is prepared in the demo terminal history.

## 2. Service startup checklist

- [ ] Start Flask with the exact command in `docs/PHASE_2L_OPERATOR_GUIDE.md`.
- [ ] Start Vite with the exact command in `docs/PHASE_2L_OPERATOR_GUIDE.md`.
- [ ] Keep backend and frontend in separate controlled processes.
- [ ] Capture startup logs without credentials, tokens, cookies, or private evidence.
- [ ] Confirm there is no startup traceback and no recognition job starts automatically.
- [ ] Confirm startup does not change attendance or job-registry hashes.

## 3. Health checks

- [ ] `GET http://127.0.0.1:5000/api/health` returns HTTP 200, `ok=true`, `active_jobs=0`, and authentication policy `product-phase-2l-c-explicit-authentication-v1`, with no private paths or state.
- [ ] `GET http://127.0.0.1:5173` returns HTTP 200.
- [ ] No-token checks for `/api/profile`, `/api/auth/me`, `/api/hod/overview`, `/api/students`, and the MON_P4 attendance endpoint return generic HTTP 401 after Phase 2L-C. Any other status is a stop condition.
- [ ] No warning, error, or credential appears in the browser console.

## 4. Login readiness

- [ ] Username and password fields are empty.
- [ ] No demo credentials or quick-fill control is displayed.
- [ ] Guidance requests an existing authorized faculty or HOD login without revealing secrets.
- [ ] One synthetic invalid-login attempt fails with a generic error and creates no authenticated session.
- [ ] The operator has a valid credential through the existing secure mechanism; do not retrieve it from repository files or show it during the demo.
- [ ] Faculty and HOD access are checked only after the unauthenticated API blocker is closed.

## 5. Required page order

Use this order and do not open Live Demo:

1. Login.
2. Dashboard.
3. My Classes / Timetable.
4. My Reports.
5. Official Reviewed MON_P4 report.
6. Archived Automatic Candidate Revision.
7. Review Students / Manual Review.
8. Students / enrollment coverage.
9. HOD Control read-only overview.
10. Help.

Do not use a page control if its effect is not clearly read-only.

## 6. Expected exact totals

- [ ] Official Reviewed Revision: Present 6; Needs Review 4; Unconfirmed 16; Missing Enrollment 1; Absent 0; Unknown 0; total 27; unresolved 21; finalized false.
- [ ] Archived Automatic Candidate Revision: Present 2; Needs Review 8; Unconfirmed 16; Missing Enrollment 1; Absent 0; Unknown 0; total 27; unresolved 25.
- [ ] The official reviewed result remains primary and the automatic candidate remains archived.
- [ ] The four official Needs Review rows are shown without saving a decision.
- [ ] `CP1-cam5-back-TRK00005` remains mixed/quarantined with zero official contribution.
- [ ] CVO has 27 roster rows and 26/27 embedding coverage; `2401100CSE0268` is Missing Enrollment.

## 7. Prohibited actions

- [ ] Do not click Process Attendance, Reprocess, Retry, Cancel, Reset Processing, Save Changes, Export Reviewed CSV, or Finalize Attendance.
- [ ] Do not start or stop Live Demo and do not access its feed.
- [ ] Do not edit HOD configuration, faculty, roles, credentials, subject assignments, timetable, roster, thresholds, checkpoints, tracklets, zones, models, or embeddings.
- [ ] Do not run YuNet, SFace, recognition, video processing, reprocessing, promotion, or rollback commands.
- [ ] Do not claim independent generalization or automatic/production-safe guarded recovery.

## 8. Demo talking points

- [ ] Early frame-only evidence was poor and is preserved as historical evidence.
- [ ] Multi-frame tracklets improved usable identity evidence.
- [ ] Strict automatic evidence is enabled.
- [ ] Guarded recovery remains human-reviewed and automatic promotion is false.
- [ ] Unconfirmed prevents false absence.
- [ ] Missing Enrollment is separate from absence.
- [ ] Official MON_P4 is 6/4/16/1/0; automatic candidate is 2/8/16/1/0.
- [ ] The difference demonstrates reviewed carry-forward and revision protection.
- [ ] The mixed track remains quarantined.
- [ ] Existing evidence is retrospective and contaminated.
- [ ] Independent generalization is not claimed.
- [ ] A genuinely new frozen session remains required for future guarded promotion.

## 9. Known limitations

- The prior unauthenticated default-HOD read-access blocker is resolved under the versioned Phase 2L-C policy.
- All four existing sessions are contaminated by prior recognition/review/calibration use.
- Three retrospective sessions are incompatible with the exact current authority policy.
- Historical guarded evidence includes wrong-person and outsider candidates.
- The exact-current mixed guarded candidate remains quarantined.
- Authenticated page visuals were rehearsed with isolated synthetic HOD and Faculty sessions only. This proves route/layout behavior, not production credential usability.
- Login, Dashboard, My Classes/Timetable, Reports with both revisions, Manual Review, Students/Coverage, HOD Control, and Help were visually checked at common laptop width; Faculty scoping was also checked on Dashboard, Reports, Manual Review, and Students. The browser recorded no warnings/errors or horizontal overflow. The Manual Review horizontal-overflow defect found during rehearsal was corrected and rechecked.

## 10. Fallback procedure

- Backend unavailable or API timeout: stop the interactive demo, repeat only the read-only `/api/health` check, and do not substitute a processing or repair command.
- Frontend unavailable: verify the identified Vite process and read-only root request; do not start another instance blindly if port 5173 is occupied.
- Login unavailable or rejected: do not reset, retrieve, print, or fabricate credentials. Defer authenticated pages to the authorized operator.
- Report page unavailable or any total differs: stop the demo and use the verified Phase 2L-A `release_candidate_freeze.json`, `frontend_contract_summary.json`, `demo_readiness.json`, and this checklist to explain the frozen state.
- Application unavailable: use only verified immutable data references and the professor-demo runbook. Do not generate screenshots or attendance results.
- Security check failure: end the demo. Do not display private roster/report data and do not work around authentication.

## 11. Shutdown checklist

- [ ] Close/finalize the rehearsal browser tab without leaving credentials or tokens.
- [ ] Stop the controlled Vite process.
- [ ] Stop the controlled Flask process.
- [ ] Verify ports used by the controlled processes are no longer listening.
- [ ] Leave unrelated pre-existing processes untouched.

## 12. Post-demo state verification

- [ ] Rehash attendance, jobs, roles, roster map, manual overrides, review registry, embeddings, embedding summary, production pointer, and timetable.
- [ ] Recount both MON_P4 CSVs independently.
- [ ] Verify job state, report revision count, finalization, review registry, mixed quarantine, authority, and guarded-recovery state are unchanged.
- [ ] Verify recognition, YuNet, SFace, classroom-video decoding, and reprocessing did not run.
- [ ] Verify the rehearsal immutable manifest and deterministic rehearsal ID independently.

## 13. Stop conditions

Stop immediately for any of the following:

- Phase 2L preflight failure or immutable-manifest mismatch.
- Any protected hash drift, report-total drift, new report revision, job change, finalization change, or authority change.
- Any no-token request returning HOD-scoped profile, roster, attendance, report, student, or governance data.
- Missing or changed production pointer, threshold, roster, timetable, review registry, or mixed-track quarantine.
- Recognition/video activity, a mutating UI action, credential exposure, private-evidence exposure, startup traceback, or unexpected background job.
- A route, total, status label, or authority label that cannot be verified read-only.

## 14. Sign-off

Phase 2L-C technical sign-off is complete. Final professor-demo sign-off remains operator-login-required.

| Field | Status | Operator / timestamp |
| --- | --- | --- |
| backend ready | VERIFIED; operator production login still required | Phase 2L-C / 2026-07-22 |
| frontend ready | VERIFIED with isolated synthetic authentication | Phase 2L-C / 2026-07-22 |
| data verified | VERIFIED READ-ONLY | Phase 2L-C / 2026-07-22 |
| authority verified | VERIFIED READ-ONLY | Phase 2L-C / 2026-07-22 |
| rollback references verified | VERIFIED READ-ONLY | Phase 2L-A preflight |
| credentials not exposed | VERIFIED | Phase 2L-C / 2026-07-22 |
| demo complete | OPERATOR LOGIN REQUIRED | |
| state unchanged | VERIFIED READ-ONLY | Phase 2L-C / 2026-07-22 |

## Exact operator actions for the real professor demo

1. Run both authentication and stabilization preflights and record protected hashes.
2. Start the documented backend and frontend once, verify health, and ensure no job starts.
3. Have the authorized operator enter the intended faculty or HOD credential privately.
4. Confirm the role-scoped landing page before showing any protected data.
5. Follow the ten-page order above and use only read-only navigation.
6. Stop immediately for any unexpected access, total, label, control, error, or state change.
7. Log out normally, stop both controlled services, and complete the post-demo state verification.
