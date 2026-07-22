# Product Phase 2L professor demo runbook

## Authentication prerequisite

This runbook uses policy `product-phase-2l-c-explicit-authentication-v1`. Begin on Login and complete an operator login before opening any protected page. Do not use `X-User-Id`, a query parameter, a default account, or a development fallback as authentication. Keep the bearer token inside the application session; never display, copy, log, or place it in a URL.

The safe authenticated order is Dashboard, Reports, Manual Review, then HOD Control for an authorized HOD. Faculty must see only assigned-subject data and must receive `403` for HOD-only or other-faculty resources. A `401` means the session is missing or invalid and requires a return to Login. Do not visit Live Demo and do not click Process, Reprocess, Edit, Finalize, upload, reset, or configuration controls.

When no real operator login is available, stop the production rehearsal at Login and record `demo_ready_operator_login_required`. A temporary synthetic fixture can verify layout only; it must remain isolated and must not be presented as a production operator login.

This is a controlled demonstration of preserved MON P4 evidence. It is not a live recognition demo, an untouched-session validation, or permission to change attendance.

## Before the professor arrives

1. Run the Phase 2L preflight from `docs/PHASE_2L_OPERATOR_GUIDE.md` and require `PREFLIGHT_STATUS=PASS`.
2. Start backend and frontend with the exact commands in the operator guide.
3. Keep the terminal windows visible only to the operator. Do not display credentials or tokens.
4. Login normally with the role intended for the demonstration.
5. Do not click Process Attendance, Reprocess, edit, export, finalize, change HOD configuration, or open Live Demo.

## Exact page order and talking points

### 1. Login

Say: “The product uses existing role-backed login. Faculty access is subject- and roster-scoped; HOD access is broader. Credentials are not part of this demonstration.”

### 2. Dashboard / My Classes

Show the scheduled class context only.

Say: “The timetable selects the subject, faculty, room, and authoritative roster. We are not starting processing today; the demo uses an already preserved report.”

Do not click Process Attendance or any retry/reprocess control.

### 3. Reports: Official Reviewed Revision

Open Reports and load `2026-06-22__B51__P4__CVO`.

Show and say:

- “This Official Reviewed Revision contains 27 rostered students.”
- “The result is 6 Present, 4 Needs Review, 16 Unconfirmed, 1 Missing Enrollment, and 0 Absent.”
- “The unresolved total is 21: 4 + 16 + 1 + 0 Unknown.”
- “Official authority is reviewed multi-frame tracklet evidence. The report is not finalized.”

### 4. Reports: Automatic Candidate Revision and historical result

Show the archived candidate and the superseded historical indicator.

Say:

- “A later automatic rerun produced 2 Present, 8 Needs Review, 16 Unconfirmed, 1 Missing Enrollment, and 0 Absent.”
- “It was archived as the Automatic Candidate Revision and did not overwrite the reviewed official result.”
- “The older frame-only approach could overreact to isolated detections and low-quality non-detection. Its result is retained for audit but is superseded or quality-failed where shown.”

### 5. Explain the status contract

Say:

- “Present means sufficient authoritative identity evidence or a traceable accepted review.”
- “Needs Review means meaningful evidence exists but is unresolved.”
- “Unconfirmed means camera evidence is insufficient. It is safer than a false Absent because failure to recognize a face does not prove the student was absent.”
- “Missing Enrollment means the rostered student has no usable production embedding.”
- “Absent is an explicit resolved outcome, not a synonym for non-detection.”
- “Unknown is defensive: an unrecognized status fails closed and remains unresolved.”

### 6. Review Students: evidence explanation

Open Review Students and load the same session. Do not click any correction or finalization control.

Point out:

- strict accepted checkpoints;
- guarded recovery candidate checkpoints, explicitly review-only;
- human-reviewed checkpoints;
- rejected mixed checkpoints;
- observation count;
- evidence authority;
- review carry-forward state.

Say: “Tracklets combine observations across frames, improving on the old frame-only limitation. They do not make every candidate safe. The mixed track `CP1-cam5-back-TRK00005` remains quarantined and contributes zero official attendance.”

Say: “Guarded recovery remains review-only. Historical TUE P2 evidence included wrong-person and outsider candidates, so it cannot be promoted automatically.”

### 7. Missing Enrollment

Filter Missing Enrollment and show the one row.

Say: “This student is on the authoritative roster but has no usable production embedding. The system refuses to invent identity evidence or mark absence automatically.”

### 8. Finalization protection

Show the unresolved and finalization state without clicking Finalize Attendance.

Say: “Finalization is explicit and blocked while unresolved rows remain or roster coverage is incomplete. This demo does not edit or finalize attendance.”

### 9. HOD Control read-only overview

With an authorized HOD login, show production family, match/margin thresholds, five checkpoints, front/back cameras, strict automatic authority, guarded recovery review-only, and coverage. Do not use any configuration control.

Say: “The current production family and strict policy are frozen for this release candidate. This screen is read-only for the demo.”

## Claims to avoid

- Do not claim perfect recognition.
- Do not claim all students were independently validated.
- Do not call the retrospective audit untouched validation.
- Do not claim guarded recovery is production-safe or automatic.
- Do not claim the 19/19 exact-current strict reviewed slice proves generalization.

Say instead: “The retrospective audit supports retaining the current strict policy with blockers. All existing sessions have prior-use contamination, so a genuinely new frozen session remains necessary for independent generalization evidence.”

## Short fallback if the application is unavailable

1. Do not start recognition, repair state, or change configuration.
2. Show this runbook and the verified immutable Phase 2L `release_candidate_freeze.json`, `frontend_contract_summary.json`, and `demo_readiness.json` from the exact output directory printed by the verifier.
3. Explain the two fixed report totals, status meanings, quarantine, review-only guarded recovery, and retrospective limitations.
4. End with: “The interactive demo is deferred until the read-only health check passes; no attendance data was changed.”
