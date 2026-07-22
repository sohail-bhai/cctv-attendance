# Product Phase 2L recovery and rollback

## Authentication recovery

Policy `product-phase-2l-c-explicit-authentication-v1` fails closed. A protected `401` means the bearer session is missing, invalid, expired, or inconsistent with the optional `X-User-Id` hint; clear the stale browser session and return to operator login. A `403` means the authenticated role or Faculty scope is insufficient; do not clear the valid session or broaden access as a workaround.

Do not restore implicit first-user, header-only, query-identity, environment, default-account, or development authentication fallbacks. If operator login cannot be completed, record `demo_ready_operator_login_required`, leave services stopped, and preserve all operational state. Never place credentials or tokens in recovery commands or evidence.

Rollback is not a demo action. Stop the application, preserve evidence, diagnose the exact mismatch, and obtain explicit authorization before running any rollback command. Never use rollback to conceal a failed preflight or to make report totals look expected.

## Stop conditions

Stop immediately if any of these occurs:

- Phase 2L preflight fails.
- Protected operational hashes drift.
- Official or candidate totals differ from the frozen counts.
- The official report is finalized unexpectedly.
- The production pointer, thresholds, authority, roster, missing-enrollment identity, review registry, or mixed quarantine differs.
- A user accidentally starts recognition, edits attendance, finalizes a report, or changes HOD configuration.

Do not reset, clean, checkout, overwrite state, process video, or improvise a rollback command.

## Phase 2I attendance-authority rollback reference

Verified backup:

`data/state_backups/phase_2i_before_2026-06-22__B51__P4__CVO_20260720_143351.json`

Recorded wrapper command:

```powershell
Set-Location -LiteralPath "F:\sohail\Class_Attendance_YuNet_SFace"
& ".\scripts\run_product_phase_2i_authority.ps1" -Rollback -Backup ".\data\state_backups\phase_2i_before_2026-06-22__B51__P4__CVO_20260720_143351.json"
```

What it changes: restores the exact pre-Phase-2I MON P4 session entry from the named backup.

What it does not change: production embeddings, thresholds, roster, timetable, users, credentials, video, or recognition outputs outside that session entry. It does not run recognition.

## Phase 2J report-revision rollback reference

Verified backup:

`data/state_backups/phase_2j_before_revision_repair_20260721_043229.json`

Exact recorded command form from the Phase 2J runner:

```powershell
Set-Location -LiteralPath "F:\sohail\Class_Attendance_YuNet_SFace"
& ".\scripts\run_product_phase_2j_stabilization.ps1" -Rollback -Backup ".\data\state_backups\phase_2j_before_revision_repair_20260721_043229.json" -Confirm "ROLLBACK_PHASE_2J_STABILIZATION"
```

What it changes: restores the exact pre-Phase-2J MON P4 session entry and review-registry state encoded by the backup transaction.

What it does not change: source videos, recognition diagnostics, production embeddings, thresholds, roster, timetable, users, credentials, or the preserved candidate archive. It does not run recognition.

## Production embedding rollback reference

Verified promotion directory:

`models/promotion_history/promotion-7cc01070e9bc644380c5`

Exact command stored in `models/current_embedding_version.json` and the promotion record:

```powershell
& "F:\sohail\Class_Attendance_YuNet_SFace\scripts\run_phase_1_2o_explicit_promotion.ps1" -Rollback -PromotionDir "F:\sohail\Class_Attendance_YuNet_SFace\models\promotion_history\promotion-7cc01070e9bc644380c5" -ConfirmRollback "promotion-7cc01070e9bc644380c5"
```

Verified backup payloads:

- `models/promotion_history/promotion-7cc01070e9bc644380c5/production_before_promotion/student_embeddings.pkl`
- `models/promotion_history/promotion-7cc01070e9bc644380c5/production_before_promotion/embedding_summary.csv`

What it changes: restores the parent production embedding and summary bytes and records the rollback transaction/pointer state.

What it does not change: official attendance, candidate reports, roster, timetable, review registry, manual overrides, jobs, HOD configuration, users, credentials, or source videos. It does not run recognition.

## Recovery sequence

1. Stop frontend and backend with `Ctrl+C`.
2. Capture current Git status, protected hashes, the exact preflight error, and the affected file path without exposing credentials or private review joins.
3. Compare with the immutable Phase 2L source manifest and the known protected hashes.
4. Determine whether the issue is application availability, accidental mutation, corruption, or an intentional authorized change.
5. If rollback is genuinely required, obtain explicit authorization for the specific rollback target and command above.
6. Run only one rollback at a time.
7. Rerun the Phase 2L preflight and independently recount both MON P4 reports.
8. Do not resume the demo unless the preflight passes and the operational impact is documented.

Rollback to an older state can intentionally change attendance/report authority or the production embedding pointer. It therefore ends the Phase 2L release-candidate freeze until a new authorized stabilization audit is completed.
