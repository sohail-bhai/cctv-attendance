# Project Context

This is the canonical implementation and handoff record for the Sreenidhi University CCTV-Based Smart Attendance System. Read this file before planning or changing the repository. Treat historical phase artifacts as evidence, not automatically as current production truth.

## 1. Identity, purpose, path, and users

- Repository: `F:\sohail\Class_Attendance_YuNet_SFace`
- Product: a CCTV-based classroom attendance workflow for Sreenidhi University.
- Purpose: turn scheduled classroom video into conservative, reviewable, roster-complete attendance while preserving evidence, human accountability, and rollback paths.
- Primary users: CV engineers, faculty reviewers, and the HOD/administrator.
- Product character: forensic and restrained. A recognition score is evidence, not permission to silently mark a student present.

## 2. System architecture

### Backend and frontend

- `app.py` is the Flask backend and API surface. It enables CORS and serves authentication, timetable/classes, processing and reprocessing, persistent job status/cancel/reset, sessions, reports, video/upload, attendance review/edit/export/finalize, student data, HOD overview/configuration, and diagnostic Live Demo APIs.
- `frontend/` is a React/Vite application. Current routes include Dashboard, Timetable, Reports, Manual Review, Students, Help, Live Demo, Faculty Activity, Jobs, and HOD Control. The old `/settings` route redirects to HOD Control.
- Frontend API requests use bearer authentication plus `X-User-Id`; health/status access has a deliberately safe fallback.

### Recognition pipeline

- OpenCV YuNet supplies face boxes and five-point landmarks.
- OpenCV SFace aligns each face with `alignCrop`, extracts a 128-dimensional feature, and compares normalized embeddings with cosine similarity.
- The embedding database supports top-3, centroid, and maximum aggregation. The current production processing contract fixes top-3 aggregation with separate score and margin gates.
- Scheduled classes are expanded into five checkpoints. The fixed production layout uses front and back clip folders, camera zones, cross-frame tracklets, per-track evidence selection, and roster-first reporting.
- Recognition must resolve only against the authoritative subject/class roster. Output still contains every rostered student, including students with no detection or no enrollment embedding.

### Workflow and persistence

- Processing jobs are persisted atomically in `data/job_runtime.json`. Transient process objects and large runtime-only fields are sanitized or compacted before persistence. Jobs left Pending or Processing across a restart become Failed rather than pretending to continue.
- Attendance/session state is persisted in `data/attendance_status.json`.
- Reports expose automatic evidence, review categories, manual corrections, and finalization state. Corrections require a reason and a roll number already present in the roster-complete report.
- Finalization is blocked until the full authoritative roster is represented and no unresolved rows remain. Finalization produces a distinct finalized CSV; it does not silently rewrite the source recognition result.
- HOD configuration writes require an explicit valid HOD bearer session. They are hash-audited, backed up, transactional, rollback-capable, and blocked while relevant jobs are active.

## 3. Recognition invariants and safety rules

These constraints are part of the product contract, not optional tuning advice.

1. Never relax recognition thresholds merely to increase attendance yield.
2. Preserve the independent score gate and best-versus-second-best margin gate.
3. Use roster-first reporting: all authoritative roster members must appear exactly once in the report.
4. A missing face, weak evidence, poor run quality, or missing embedding is not automatically an absence.
5. Keep `Unconfirmed`, `Missing Enrollment`, and `Needs Review` distinct from an explicit `Absent` decision.
6. Mixed-identity tracklets are unsafe. Quarantine them; do not carry their labels into official attendance.
7. Manual review or carry-forward is allowed only with traceable evidence and exact provenance rules.
8. Never let a later automatic run overwrite an already reviewed official revision. Archive the new automatic result as a candidate.
9. Do not automatically promote an embedding family, a calibration policy, or a new authority policy. Promotion requires an explicit guarded decision plus a verified rollback path.
10. Do not run recognition or video reprocessing during documentation, audit, UI-only, or state-reconciliation work unless the task explicitly authorizes it.
11. Do not claim recognition ran unless the command actually decoded and processed classroom video. Unit fixtures, report parsing, builds, and artifact inspection are not recognition runs.
12. Preserve source reports, diagnostics, model versions, hashes, backups, audit logs, and rollback commands.

## 4. Exact production embedding family and promotion

The production pointer is `models/current_embedding_version.json`.

- Family: `embfam-274b5207b8b71294ff75`
- Active variant: `embfam-274b5207b8b71294ff75-d`
- Status: `promoted`
- Promotion ID: `promotion-7cc01070e9bc644380c5`
- Promoted at: `2026-07-17T22:37:15+05:30`
- Production embedding SHA-256: `f088d827adc548ee95f46566d758fd71fc304d042c43f1ecffc6526b60bcd832`
- Production summary SHA-256: `63885588c374c37f4da9bf85294f240bdf0f28cb585e77b894ab516139ca46ae`
- Parent embedding SHA-256: `c32ed31df10b7b9b43b8f19977a2adf0a82fb8e7a71f3e8bceb42c0565fedd49`
- Parent summary SHA-256: `0df35c3bb9207e191aa49dad5536b8378e812d56463491bd7203885dca5ae9ee`
- Production content: 354 embedding records, 128 dimensions.
- CVO coverage: 26 of 27 rostered students have usable enrollment records.
- Missing enrollment: `2401100CSE0268`.
- The promotion pointer contains the verified backup and rollback command. Use that exact metadata; do not invent a new rollback procedure from memory.

The family directory is `models/versions/embfam-274b5207b8b71294ff75`; the promotion record is `models/promotion_history/promotion-7cc01070e9bc644380c5`.

## 5. Exact processing authority and status semantics

The authoritative browser-processing contract is fixed in `src/face_attendance/processing_integration.py`:

- Policy version: `product-phase-2i-strict-tracklet-authority-v1`
- Processing mode: `quality_aware_tracklet_authority_v1`
- Official recognition authority: `strict_tracklet_aggregate_with_guarded_review_candidates`
- Checkpoints: `CP1`, `CP2`, `CP3`, `CP4`, `CP5`
- Checkpoint mode: `clip-folders`
- Cameras: exact configured front and back classroom videos for every checkpoint
- Minimum detections: 2
- Present votes: 3
- Strong votes: 4
- Review votes: 2
- Match threshold: 0.48
- Margin threshold: 0.08
- Sample FPS: 2
- Frame skip: 3
- Settle delay: 10 minutes
- Checkpoint interval: 10 minutes
- Clip duration: 20 seconds
- Aggregation: top-3
- Save unknown crops: false
- Processing timeout: 1,800 seconds
- Diagnostics: full/organized, camera zones enabled
- Zone merge IoU: 0.20
- Tracklet minimum observations: 2
- Tracklet maximum selected observations: 5
- Tracklet maximum gap: 1.5 seconds
- Tracklet minimum IoU: 0.10
- Tracklet center-distance factor: 1.25
- Tracklet size-change limit: 0.50
- Tracklet embedding-similarity floor: 0.25
- Tracklets are official: true

Browser users cannot change these thresholds. Historical defaults in `src/config.py` are not the production product contract.

### Status meanings

- `Present`: sufficient authoritative identity evidence under the current policy or an accepted, traceable reviewed correction.
- `Needs Review`: evidence exists but is ambiguous, guarded, manually flagged, or otherwise requires a human decision.
- `Unconfirmed`: the run cannot support either presence or absence. This includes inadequate/low-quality evidence after Phase 2I semantics.
- `Missing Enrollment`: the student is on the authoritative roster but lacks a usable production embedding.
- `Absent`: an explicit resolved attendance outcome. Never infer it merely from non-detection, low run quality, or missing enrollment.
- `Unknown`: a defensive UI/API category for unrecognized legacy values; it is unresolved and must not be finalized silently.

The frontend compact summary's unresolved count is the sum of Needs Review, Unconfirmed, Missing Enrollment, and Unknown. Current backend finalization independently enforces full-roster and zero-unresolved gates.

### Authority precedence

1. A finalized report is immutable attendance output unless a separately audited correction workflow is introduced.
2. A reviewed official revision remains official.
3. A later automatic reprocess becomes an archived candidate when an official reviewed revision already exists.
4. Exact reviewed-evidence carry-forward may be guarded in only when source-video fingerprints, production-embedding fingerprint, and evidence signature match the registry.
5. Ordinary automatic tracklet output remains the automatic authority; guarded review candidates never become automatic identity evidence.

## 6. CVO roster and corrections

The authoritative CVO roster comes from `data/student_faculty_map.json` and has 27 students:

`2401100CSE0016`, `2401100CSE0019`, `2401100CSE0028`, `2401100CSE0044`, `2401100CSE0050`, `2401100CSE0060`, `2401100CSE0124`, `2401100CSE0140`, `2401100CSE0268`, `24011CSEAI0007`, `24011CSEAI0051`, `24011CSEAI0110`, `2401100CSE0007`, `2401100CSE0010`, `2401100CSE0021`, `2401100CSE0024`, `2401100CSE0046`, `2401100CSE0057`, `2401100CSE0068`, `2401100CSE0071`, `2401100CSE0075`, `2401100CSE0080`, `2401100CSE0096`, `2401100CSE0121`, `2401100CSE0131`, `2401100CSE0237`, `2401100CSE0270`.

Mandatory normalization facts:

- `24011CSEAI0110` and `2401100CSE0110` are distinct students. Never alias or merge them.
- CVO contains `24011CSEAI0110`; it does not contain `2401100CSE0110`.
- `24011CSEAI0061` belongs to SWE only and must not appear in the CVO roster.
- `2401100CSE0237` is a valid CVO member and has production embeddings.
- `2401100CSE0268` is a valid CVO member but currently has zero usable production embeddings.
- Historical reports can contain `2401100CSE0110`; historical presence does not change the current CVO roster.

## 7. Phase history

### Phase 1.x: recognition and embedding evidence

- Early Phase 1 established timetable-driven YuNet/SFace recognition, independent score/margin gates, five-checkpoint voting, TUE P1/P2 diagnostics, camera zones, tracklets, blind review, and roster repair. The frozen review benchmark contains 120 tracks: 37 accepted TUE P1 tracks, 63 corrected/unresolved tracks, and 20 TUE P2 tracks. Thresholds were not relaxed.
- Phase 1.2G produced tracklet calibration `cal-5cd...`: 100 tracks, out-of-fold 7 correct/0 false and final 8/0. The candidate was disabled and later rejected after multi-session evidence.
- Phase 1.2I built review-approved embedding families. Partial family `embfam-189...` lacked sufficient TUE P2 evidence. A same-track recovery bundle was produced, and complete but unapproved family `embfam-7bc...` was built.
- Phase 1.2J tested MON P3 in shadow mode. It was rejected on retention: 4 recoveries and 0 unsafe false identities, but it lost 6 correct production accepts and exposed 15 reviewed exception tracks. Nothing was promoted.
- Phase 1.2K traced the six margin-only losses to two source groups: roll `2401100CSE0121` in TUE P1 and roll `2401100CSE0140` in TUE P2.
- Phase 1.2L ran exhaustive 32-subset source ablation over five sources using 120 leakage-safe benchmark rows plus 15 exact MON P3 descriptive rows. `ABL-19` was initially recommended; production was unchanged.
- Phase 1.2L.1 corrected the selection rule because same-session descriptive evidence had improperly broken a tie. The parsimonious `ABL-16-ee3a4ff6`, using only the `2401100CSE0140` TUE P2 medoid, retained all 4 recoveries with 0 losses and 0 unsafe identities. No family was built in that phase.
- Phase 1.2M built `embfam-274b5207b8b71294ff75` with variants A=347, B=351, C=350, and D=354 records. It reproduced 15/15 MON P3 and 120/120 leakage-safe/descriptive checks but remained unapproved pending a new untouched session.
- Phase 1.2N ran the untouched MON P4 recognition validation: 308 tracklets, 3,956 observations, and 5 exception tracks. The candidate delivered 3 correct recoveries, 0 unsafe identities, 0 lost production accepts, and 0 unverifiable outcomes. It passed but still required explicit promotion. This phase did run classroom-video recognition.
- Phase 1.2O explicitly and transactionally promoted variant D. It backed up the parent, recorded exact hashes and rollback metadata, and did not run recognition or modify attendance.
- Phase 1.2P verified the promotion byte-for-byte: 354 records at 128 dimensions, 120/120 benchmark reproduction, 15/15 MON P3 reproduction, and a valid rollback backup. It did not run recognition or modify attendance.

### Product Phase 2A through 2K-A

- 2A established workflow-state foundations: atomic persistence, sanitized attendance state, persistent jobs, and interrupted-job recovery. The initial sanitation changed four entries, removed twelve transient fields, and backed up `attendance_status_before_sanitize_20260718_154407`. Attendance meaning and embeddings were preserved.
- 2B connected Dashboard, class cards, and Jobs to persistent state with compact polling.
- 2C enforced roster/report consistency, authoritative roster fallbacks, and distinct roll-number normalization.
- 2D added compact session APIs, report review, reasoned corrections, full-roster finalization gates, and distinct reviewed/final CSVs.
- 2E added role-backed authentication, HOD overview, faculty/subject scoping, and roster/dataset/embedding coverage. Public admin identity is normalized to the HOD role and secret fields are stripped.
- 2F added guarded HOD configuration: explicit bearer authorization, hashing, transactional backups, audit/rollback, timetable preview/apply, active-job blocking, and hashed newly created credentials.
- 2G fixed quality-aware browser processing around the exact five-checkpoint front/back layout, authoritative roster filtering, zones, tracklet diagnostics, and run-quality summaries. The current implementation has since evolved to the stricter Phase 2I authority, and browser callers still cannot tune thresholds. MON P4 diagnostic job `7A239045` belongs to this line.
- 2H reused existing diagnostics without rerunning video. It materialized 31 evidence pairs (3 reused and 28 new); blind review identified 30 and quarantined one unsafe mixed track. Its proposed report contained 6 Present, 4 Needs Review, and 17 Unconfirmed before the missing-enrollment split. It passed evidence review but did not activate tracklets or alter official attendance.
- 2I explicitly applied reviewed multi-frame authority to MON P4. It produced 6 Present, 4 Needs Review, 16 Unconfirmed, 1 Missing Enrollment, and 0 Absent. Future automatic authority became strict tracklet aggregation with guarded review candidates. The source report was preserved, a backup was created, and video recognition was not rerun.
- 2J stabilized report revisions. A newer automatic run is archived instead of overwriting the reviewed official revision. Exact review carry-forward requires matching source-video hashes, embedding hash, and evidence signature. The registry holds 31 reviews: 30 accepted and 1 mixed/quarantined. A state backup was created before revision repair.
- 2K-A added a diagnostic-only, conservative tracklet-purity and split-candidate subsystem under policy `product-phase-2k-a-diagnostic-tracklet-purity-v1`. It evaluates temporal, geometric, appearance/embedding, identity-vote, quality, and invalid-evidence signals; permits at most one deterministic split only when at least two signals include an identity/appearance signal; creates immutable parent/child lineage; and fail-closes every ambiguous, undersized, invalid, or independently unverified child. The immutable shadow run is `tracklet-purity-43cec17498dc05615502`. It produced no split candidates and no child candidates, changed no attendance or authority, and kept `CP1-cam5-back-TRK00005` mixed, quarantined, ineligible for carry-forward, and at zero official contribution.

## 8. Current MON P4 official and archived state

Session key: `2026-06-22__B51__P4__CVO`.

### Official reviewed revision

- Session status: `Needs Review`
- Report type: `Corrected Multi-frame Authority`
- Official authority: `reviewed_multiframe_tracklet_evidence`
- Processing profile: `quality_aware_tracklet_authority_v1`
- Processing contract: `product-phase-2i-strict-tracklet-authority-v1`
- Authority revision: `phase-2i-mon-p4-reviewed-multiframe-7d9d3a539c0b32e33048`
- Authority policy: `product-phase-2i-reviewed-multiframe-authority-v1`
- Automatic authority: `strict_tracklet_aggregate_with_guarded_review_candidates`
- Guarded authority: `human_validated_exact_source_carry_forward`
- Guarded authority is automatic: false
- Phase 2J report revision policy: `product-phase-2j-report-revision-control-v2`
- Carry-forward policy: `product-phase-2j-exact-review-carry-forward-v2`
- Review registry: `review-registry-c1e03cbe948d98800dae93e7`
- Review registry counts: 31 reviewed, 30 accepted, 1 mixed
- Totals: 27 students; 6 Present, 4 Needs Review, 16 Unconfirmed, 1 Missing Enrollment, 0 Absent
- Attendance percentage: 22.2%
- Unresolved: 21
- Finalized: false (`attendance_finalized=false`, legacy `Attendance_Finalized=No`)
- Review state: `needs_attention`

Present rolls: `2401100CSE0016`, `2401100CSE0019`, `2401100CSE0050`, `2401100CSE0060`, `2401100CSE0140`, `24011CSEAI0007`.

Needs Review rolls: `2401100CSE0028`, `2401100CSE0044`, `2401100CSE0124`, `24011CSEAI0051`.

Missing Enrollment roll: `2401100CSE0268`.

The direct CSV import and persisted status totals agree.

### Archived automatic candidate

- Revision: `automatic-candidate-99cc13934144bf7e1706b8be`
- Totals: 27 students; 2 Present, 8 Needs Review, 16 Unconfirmed, 1 Missing Enrollment, 0 Absent
- Authority: strict tracklet aggregate with guarded review candidates
- The direct candidate CSV import and stored totals agree.
- `official_preserved_after_reprocess` is true. The candidate is intentionally different and did not replace the official reviewed revision.

## 9. Mixed track that must remain quarantined

- Track ID: `CP1-cam5-back-TRK00005`
- Predicted identity: `24011CSEAI0051`
- Review disposition: `mixed_track`
- Carry-forward applied: false
- Quarantined: true

This track contains mixed identity evidence and is the central negative regression for the next implementation phase. Do not promote, carry forward, or convert it into automatic presence. Any purity/splitting change must prove that this track remains unsafe as a whole while allowing only independently pure child tracklets to be considered.

Phase 2K-A evaluated all 39 ordered observations from 0.48 through 19.20 seconds. The serialized diagnostics provide identity votes and geometry but intentionally do not expose raw per-observation embedding vectors. Available pairwise similarity is 0.5452 minimum and 0.7497 median; all 39 recorded identity votes name `24011CSEAI0051`. No temporal, geometric, appearance, or identity boundary met the conservative multi-signal split rule, so there is no defensible child boundary and no independently strict-gated child. The parent remains `mixed_quarantined`, produces no child lineage, rejects review carry-forward, and contributes nothing to official attendance. The immutable human mixed-track disposition remains authoritative evidence; absent raw vectors are a documented diagnostic limitation, not permission to infer a split.

## 10. Important repository and artifact paths

### Current source and state

- `app.py`
- `src/face_attendance/processing_integration.py`
- `src/face_attendance/tracklets.py`
- `scripts/mark_attendance_checkpoints.py`
- `frontend/src/`
- `data/attendance_status.json`
- `data/job_runtime.json`
- `data/role_users.json`
- `data/student_faculty_map.json`
- `data/manual_overrides.json`
- `data/review_evidence_registry.json`
- `data/camera_zones.json`
- `timetable_b51_2026_2027.csv`

### Production model

- `models/student_embeddings.pkl`
- `models/embedding_summary.csv`
- `models/current_embedding_version.json`
- `models/versions/embfam-274b5207b8b71294ff75/`
- `models/promotion_history/promotion-7cc01070e9bc644380c5/`

### Phase and authority artifacts

- `attendance_output/embedding_forensics/phase_1_2*/`
- `attendance_output/shadow_validation/phase_1_2n/`
- `attendance_output/product_workflow/phase_2h_multiframe_recovery/multiframe-recovery-7d9d3a539c0b32e33048/`
- `attendance_output/product_workflow/phase_2h_multiframe_recovery/multiframe-recovery-7d9d3a539c0b32e33048/evaluation/evaluation_summary.json`
- `attendance_output/product_workflow/phase_2i_authority/authority-revision-2d83c4f679f839a6c784c636/`
- `attendance_output/product_workflow/report_revisions/`
- `attendance_output/product_workflow/phase_2k_tracklet_purity/tracklet-purity-43cec17498dc05615502/`
- `attendance_output/diagnostics/product-2i-*/`
- `data/state_backups/phase_2j_before_revision_repair_20260721_043229.json`
- `patch_backups/`

Use manifests and hashes inside the artifact directories to select exact files. Do not pick an artifact solely because its directory name looks newest.

## 11. Tests and validation expectations

Normal validation should be proportional to the patch and must avoid operational recognition unless explicitly authorized.

- Python: `venv\Scripts\python.exe -m unittest discover -s tests -v`
- Frontend logic: from `frontend`, `node --test tests/*.test.mjs`
- Frontend production build: from `frontend`, `node_modules\.bin\vite.cmd build`
- State checks: compare hashes for attendance, jobs, roster, overrides, review registry, production embeddings, summary, current pointer, and timetable before and after tests that might touch filesystem state.
- Report checks: independently count the authoritative CSV and the persisted JSON summary.
- Promotion checks: compare production files with the recorded promotion payload byte-for-byte and verify the recorded rollback target exists.

Bootstrap audit on 2026-07-21:

- Full Python suite: 477 tests passed; 1 test was skipped because its expected real Phase 1.2E/H review artifacts were not present in the fixture path.
- Frontend logic: 25 tests ran; 22 passed and 3 failed. The failures are expectation drift in `frontend/tests/consistency.test.mjs`: one test expects flagged Absent evidence to become Needs Review even though current code only force-reviews an Unknown category, and two tests expect the older compact-summary shape without Unconfirmed, Missing Enrollment, Unknown, and Unresolved fields.
- Frontend production build: passed when directed to a temporary output directory outside the repository.
- No recognition/video processing ran during the bootstrap audit.

Phase 2K-A validation on 2026-07-21:

- Tracklet-purity targeted suite: 15 tests passed, including stable/noisy tracks, clear and ambiguous boundaries, minimum child size, embedding clusters, identity-only and invalid evidence, duplicate timestamps, exact carry-forward, changed evidence/lineage rejection, Windows paths, immutable idempotency/tamper detection, the real mixed track, strict historical non-splitting, and required shadow outputs.
- Existing tracklet suite: 24 tests passed.
- Product Phase 2H suite: 13 tests passed.
- Product Phase 2I suite: 8 tests passed.
- Product Phase 2J review/revision suites: 12 tests passed.
- Full Python suite: 492 tests passed in 1,761.771 seconds; the same one real Phase 1.2E/H artifact test was skipped.
- The immutable Phase 2K-A manifest verified successfully. Protected operational hashes and direct official/candidate report totals were identical before and after validation.
- Frontend tests/build were not rerun because Phase 2K-A changed no UI, API, or shared frontend schema. The three documented frontend expectation failures remain untouched.
- No classroom video was decoded, no recognition/reprocessing ran, and no attendance, authority, model, roster, threshold, timetable, HOD, or manual-override state changed.

Do not hide or normalize away the three frontend failures. Reconcile the intended contract and update code or tests only in a separately scoped implementation patch.

## 12. Lifecycle and tooling lessons

- Tests must be authority-independent: do not make their expected result depend on mutable live JSON authority. Prefer isolated fixtures and temporary roots, then verify operational hashes afterward.
- On PowerShell, Python `unittest` progress is commonly written to stderr; an apparently empty captured stdout is not evidence that no tests ran.
- PowerShell uses `elseif`, not `elif`.
- Windows paths and manifest-relative paths need explicit separator normalization. Never assume POSIX separators.
- Installers and promotions must fail closed, restore the exact previous bytes on error, and record a verifiable rollback command.
- Never claim recognition ran unless actual classroom video decoding and inference occurred.
- PowerShell's default `ConvertFrom-Json` can reject documents containing property names that differ only by case. Current attendance state includes both `attendance_finalized` and legacy `Attendance_Finalized`; use the application parser or a case-preserving JSON tool until the schema is explicitly migrated.
- Existing phase artifacts, diagnostic outputs, and dirty Git changes may belong to prior work. Preserve them unless a task explicitly authorizes their replacement.
- Credentials in `data/role_users.json` are operational secrets, even if legacy records are plaintext. Never copy them into documentation, logs, tests, or chat.

## 13. Patch-only workflow

For a bounded implementation phase:

1. Read `PROJECT_CONTEXT.md`, then `CURRENT_CHANGES.md`.
2. Inspect Git status and identify pre-existing user changes.
3. Name the exact policy/invariant being changed and the negative regression that must remain safe.
4. Hash or back up every operational file that the patch may mutate.
5. Change the smallest source and test surface necessary. Do not mix unrelated cleanup, UI redesign, threshold changes, model work, and authority changes.
6. Use fixtures or copied state for tests. Never point experimental tests at live attendance/model data unless the phase explicitly requires it and has a transaction/rollback plan.
7. Run targeted tests, then the broader relevant suite. Record skipped or failing validations exactly.
8. Compare operational hashes and semantic totals after validation.
9. Record the patch, affected paths, evidence, rollback, and recommended next task in `CURRENT_CHANGES.md`.
10. Do not commit, push, deploy, promote, finalize, or run recognition unless the user explicitly requested that action.

## 14. Codex workflow

- Always read this file first.
- Overwrite `CURRENT_CHANGES.md` on every run; it is the latest-run ledger, not an append-only diary.
- Keep chat minimal. Put durable details in these two repository files.
- Stay within the requested phase. Do not fix unrelated dirty-worktree changes.
- Before risky work, create a narrowly named backup or capture exact hashes and record the rollback.
- Validate both behavior and preservation of operational state.
- Distinguish source edits, generated artifacts, operational mutations, and temporary validation output.
- Record whether recognition ran, whether attendance changed, and whether embeddings, roster, thresholds, authority, or HOD configuration changed.
- If evidence conflicts, preserve the contradiction and mark it unresolved; never guess which artifact is authoritative.

## 15. Mandatory near-term roadmap

The order below is mandatory unless a later explicit decision records why it changed.

1. Phase 2K-A completed: diagnostic/shadow mixed-track purity detection, deterministic split-candidate logic, immutable lineage, and the `CP1-cam5-back-TRK00005` negative regression. No split was defensible for the mixed parent, and no authority or embedding promotion occurred.
2. Phase 2K-B completed: carry-forward verification and independent-session contract preparation. Exact source, embedding, purity, lineage, and evidence bindings fail closed, and no existing prepared session is eligible as untouched.
3. Phase 2K-C0 completed: new-session intake, historical contamination detection, byte-identical staging, immutable source-freeze packaging, operator commands, and no-video tooling validation. No real capture package or recognition run was produced.
4. Phase 2K-C1 completed: frozen-policy retrospective multi-session safety and retention audit. It found the exact current strict slice safe, preserved the mixed quarantine, and documented that all four existing sessions are contaminated and three are policy-incompatible with current authority.
5. Phase 2L stabilization and professor-demo preparation. Freeze the validated read-only workflow, resolve validation drift, document recovery/operator procedures, and prepare a controlled demonstration using non-destructive sample data.
6. Future independent session acquisition and freeze remains required for any generalization or guarded-promotion claim. An operator must capture a genuinely new CVO/B51 session, run the dry-run, create the exact ten-video immutable package, and pass both validators. Recognition still requires separate explicit authorization.
7. Guarded promotion gate. Consider activation only if a future independent-session retention and safety evaluation passes, source/embedding fingerprints are complete, rollback is verified, and approval is explicit.

## 16. Later backlog, separate from the mandatory roadmap

Do not pull these into the mixed-track patch unless separately authorized:

- Tiled/high-resolution detection for small faces
- Denser temporal sampling windows
- Camera-specific calibration and geometry
- Global cross-camera/cross-checkpoint assignment
- Approved-crop enrollment workflow
- Alternate detector/recognizer benchmarking
- Real-time processing
- Student/faculty portal expansion
- Live Demo productization

The Live Demo code exists, but it is diagnostic/deferred and comes after authority stabilization, not before it.

## 17. Product Phase 2K-B: exact carry-forward verification and untouched-session preparation

Product Phase 2K-B completed on 2026-07-22 as a diagnostic-only safety, provenance, fixture, and preparation phase. It introduced no production recognition or attendance behavior. The policy is `product-phase-2k-b-exact-carry-forward-v1`; the future capture contract is `product-phase-2k-c-independent-session-capture-v1`.

The exact carry-forward evaluator binds the complete attempt to all of the following dimensions:

- canonical session date, section, period, subject, and session ID;
- every source path, source SHA-256, checkpoint, camera, canonical order, source-layout version, and path-normalization version;
- production embedding family, active variant, embedding bytes, summary bytes, dimension, and aggregation policy;
- match and margin thresholds, checkpoint rules, tracklet policy, purity policy, zone policy, authority policy, and processing contract;
- every parent track ID, checkpoint, camera, ordered observation membership, order, time indices/span, evidence fingerprint, purity outcome/fingerprint, and quarantine state;
- every child parent/child ID, boundary, observation span/membership, evidence fingerprint, lineage fingerprint, purity outcome, and policy version when children exist;
- review registry ID, original evidence signature, disposition, reviewer-export fingerprint, joined-review fingerprint, and immutable evaluation fingerprint;
- source-manifest hash, Phase 2K-A purity-manifest hash, canonical carry-forward-policy hash, verification status, and path-normalization rules.

Reuse is all-or-nothing. A missing, changed, duplicate, reordered, unverified, mixed, ambiguous, incomplete, or partially matching component rejects the entire attempt with a deterministic result and reason code. No matching subset is applied. Physical source-list order may vary only when unique contiguous canonical-order bindings prove equivalence. Windows separators and drive-letter spelling normalize deterministically, while component case remains preserved so distinct case-sensitive paths are not collapsed.

The verified Phase 2K-A registry still has 31 reviewed pairs. Eighteen unchanged `pure_unsplit` pairs are diagnostically exact-match eligible in isolation, while thirteen are rejected by purity/quarantine rules. The complete 31-row attempt is rejected as `mixed_quarantined`; it does not reuse the eligible subset. This is diagnostic verification only and does not change the already reviewed official revision.

The mandatory negative regression `CP1-cam5-back-TRK00005` remains `mixed_quarantined` with 39 ordered observations, no child lineage, no review inheritance, no partial carry-forward, and zero official contribution. Its original evidence signature still matches `3e66e6ecfa57ea607bedc59578af159fc7c85774f21ae15ffd98a029b79ea983`, proving that an exact signature cannot override mixed quarantine.

The candidate-session audit hashed and classified all four complete prepared-slot sessions: MON P3, MON P4, TUE P1, and TUE P2. Each has five checkpoints and exact front/back clips, but every session has already been used in recognition plus at least one disqualifying human-review, calibration, selection, ablation, retention, promotion, Phase 1.2N, Phase 2G, or Phase 2I workflow. Therefore no existing complete session is genuinely untouched. The recommendation is `no_existing_untouched_session`; Product Phase 2K-C requires newly acquired CVO/B51 footage frozen before recognition or review.

The Phase 2K-C contract fixes ten new clips across CP1-CP5 and back/front cameras, complete source hashes and timing metadata, the current production family and hashes, `0.48`/`0.08`, the strict Phase 2I processing/tracklet authority, and a no-post-result-tuning rule. Diagnostic observations must retain exact geometry, quality, private local-vote, selection, and source provenance. Privacy-safe appearance evidence uses restricted derived pairwise/local-window similarity matrices rather than persisted raw observation vectors. These artifacts are excluded from the frontend, blind-review exports, enrollment, and model rebuilding; they require access-controlled encrypted storage and lifecycle deletion evidence.

Blind review uses randomized IDs, hides predicted identity and scores, presents representative observations before/after proposed boundaries, supports single-person/mixed/unclear/outsider/wrong-person outcomes, and requires immutable manifest and completeness validation. Acceptance requires zero wrong-person identities, outsider absorption, accepted mixed parents, lost previously correct strict identities, unsafe carry-forward, and review leakage, plus exact roster completeness and preserved Unconfirmed/Missing Enrollment semantics. No automatic promotion is permitted.

The immutable output is `attendance_output/product_workflow/phase_2k_carry_forward/carry-forward-e20a746d3d34c8141ab5`. Its immutable manifest SHA-256 is `79a38737c6057768dc4128c3507fced268ea25c0beb84121a27cf2bb23195ac3`. Its source-video fingerprint set is `96fd16e6d0189c43dc8aa9dc997b37274c129d8512c4f1d2d340d2802eaaa941`. Repeated materialization was byte-identical, independent verification passed, deterministic-ID collisions fail closed, and tampering is detected.

Validation completed with 506 tests run: 505 passed, zero failures/errors, and the one documented real Phase 1.2E/H artifact test skipped. No frontend files or shared API schemas changed, so frontend tests were not rerun. Recognition and video processing did not run. Official and candidate attendance, finalization, all authority layers, embeddings, pointer, roster, thresholds, checkpoint policy, timetable, review registry, manual overrides, jobs, roles/credentials, and HOD state remained byte-for-byte unchanged.

The next task is Product Phase 2K-C independent untouched-session execution. It requires newly acquired complete five-checkpoint front/back footage, an immutable source freeze and provenance audit before any processing, the exact frozen production configuration, restricted diagnostic storage, the predeclared blind-review package, and explicit authorization to run recognition. Product Phase 2K-B itself grants no such execution or activation authority.

## 18. Product Phase 2K-C0: untouched-session intake and immutable source freeze

Product Phase 2K-C0 completed on 2026-07-22 as a preparation/source-integrity implementation only. Its policy is `product-phase-2k-c0-untouched-session-intake-v1`; its source-freeze contract is `product-phase-2k-c0-immutable-source-freeze-v1`; its known-source registry is `product-phase-2k-c0-known-source-registry-v1`. It consumes the verified Phase 2K-B capture contract `product-phase-2k-c-independent-session-capture-v1` without changing that contract.

The intake workflow accepts exactly CP1 through CP5 with one back and one front source each. It requires explicit session date, section, period, subject, room, ten timezone-qualified operator capture timestamps, an output root, and confirmation token `CREATE_PHASE_2K_C_SOURCE_FREEZE`. Sources may be supplied through an exact `CP1`-`CP5` directory layout or ten explicit checkpoint/camera path bindings. Missing metadata is never guessed.

Only source bytes, filesystem metadata, and container/header properties are read. The header reader obtains container, width, height, FPS, and frame count without calling frame-read operations. It records estimated duration, filesystem creation/change and modification timestamps, operator capture timestamp/method/confidence, and metadata method/confidence. Unsupported containers, unreliable headers, symlinks, extra/missing/duplicate bindings, duplicate bytes, changed sources, or copy mismatches fail closed.

The verified contamination registry contains 40 source hashes across the four disqualified sessions: MON P3, MON P4, TUE P1, and TUE P2. Its fingerprint is `611988c53cdd1fe751f1e2f0b18cf5ea8ad89f9fab8bee69bda97ec7548a5814`. Evidence coverage binds Phase 1 source freezes, Phase 1.2N, Phase 2G, Phase 2I, Phase 2K-A, and the Phase 2K-B immutable candidate inventory. The exact contamination outcomes are `new_unique_source`, `duplicate_inside_package`, `known_contaminated_source`, `duplicate_session_id`, and `unverifiable_provenance`; no override path exists. A source located in the historical prepared-session tree is rejected even if its current bytes are new.

A successful create stages copies in a narrowly named private directory, hashes source and copy independently, detects source drift during the copy window, writes deterministic manifests, invokes the canonical Phase 2K-B `validate-capture` function, writes the immutable manifest last, and atomically renames the complete directory. Completed packages are never overwritten. Identical reruns verify and reuse the existing bytes; invalid existing targets or deterministic collisions fail closed. Verification rechecks the exact file set, all hashes/sizes, source-freeze and CSV equivalence, complete known-source registry, production policy, package fingerprint/ID, sibling session IDs, no-recognition declaration, and the canonical Phase 2K-B validator.

Every real package is expected under an operator-selected restricted output root as `capture-package-<20-hex>`. It contains package/session metadata, canonical layout/inventory/source freeze, complete contamination registry and results, the exact production-policy freeze, blind-review plan, capture/canonical validation manifests, no-recognition declaration, immutable manifest, and the ten byte-identical videos under `videos/CP1` through `videos/CP5`. Original absolute source paths are classified as a restricted operator record. No raw observation embeddings or derived appearance matrices are produced during intake.

The current production freeze remains family `embfam-274b5207b8b71294ff75`, variant `embfam-274b5207b8b71294ff75-d`, embedding SHA-256 `f088d827adc548ee95f46566d758fd71fc304d042c43f1ecffc6526b60bcd832`, summary SHA-256 `63885588c374c37f4da9bf85294f240bdf0f28cb585e77b894ab516139ca46ae`, 128 dimensions, top-3 aggregation, thresholds `0.48`/`0.08`, and the complete current Phase 2I checkpoint/zone/tracklet/authority policy. A package becomes invalid if these production values change before recognition.

The operator guide is `docs/PHASE_2K_C_NEW_SESSION_CAPTURE_GUIDE.md`. The Python runner is `scripts/create_product_phase_2k_c_capture_package.py`; the PowerShell wrapper is `scripts/create_product_phase_2k_c_capture_package.ps1`. The dry-run/create interface requires the exact source/session/timestamp inputs above. Package verification uses `verify --package-dir <path>`, followed independently by `scripts/run_product_phase_2k_carry_forward.py validate-capture --package-dir <path>`. Neither validation command is recognition authorization.

The no-video tooling artifact is `attendance_output/product_workflow/phase_2k_capture_intake_tooling/intake-tooling-ff8dd99df5c3e2795234`. Its immutable manifest SHA-256 is `4f9e1a8d71b0ac2e200e58c7753c47cbef899f675121464b2cdb07937d7bfde7`. It records policy, fail-closed reasons, contamination-registry summary, Phase 2K-B contract compatibility, protected hashes, and the no-recognition declaration. Verification passed, repeated materialization was byte-identical, and temporary-fixture tampering was detected.

Validation completed with 544 Python tests run: 543 passed, zero failures/errors, and the one pre-existing real Phase 1.2E/H artifact test skipped. The new suite contributed 38 passing tests. Phase 2K-B, Phase 2K-A, Phase 2J, and Phase 2I targeted regressions all passed. Frontend tests/build were not run because no frontend, API, or shared frontend schema changed.

No real capture package was created. No classroom video was opened, decoded, or processed. YuNet and SFace inference did not run. Recognition did not run. Official and candidate attendance, reports, finalization, every authority layer, embeddings, pointer, roster, thresholds, checkpoint/zone/tracklet policy, timetable, review registry, manual overrides, jobs, roles/credentials, and HOD configuration remained unchanged. Independent CSV counts remain official 6 Present, 4 Needs Review, 16 Unconfirmed, 1 Missing Enrollment, 0 Absent, total 27; candidate 2 Present, 8 Needs Review, 16 Unconfirmed, 1 Missing Enrollment, 0 Absent, total 27.

The next operator action is acquisition of a genuinely new CVO/B51 session. The operator must preserve originals, record all exact timestamps and session metadata, run the documented dry-run, create the immutable package, upload the entire package through restricted encrypted transport/storage, and pass both validators. Do not use browser **Process Attendance**. After the freeze passes, stop and obtain separate explicit authorization before any frame decoding, recognition, evaluation, authority change, or activation.

## 19. Product Phase 2K-C1: frozen-policy retrospective multi-session audit

Product Phase 2K-C1 completed on 2026-07-22 as a read-only retrospective engineering-validation phase. Its policy is `product-phase-2k-c1-retrospective-multisession-audit-v1`. It did not open or decode classroom video, run YuNet or SFace, run recognition, regenerate tracklets, change reports, or alter any operational state. It normalized only already-generated, hash-verified diagnostics, review joins, evaluations, reports, and immutable phase outputs.

The exact sessions audited were:

- `2026-06-22__B51__P3__CVO`
- `2026-06-22__B51__P4__CVO`
- `2026-06-30__B51__P1__CVO`
- `2026-06-30__B51__P2__CVO`

Evidence sufficiency is session-specific. MON P4 is `complete_retrospective_evidence` because its exact current production embedding, strict automatic policy, reviewed 31-pair identity evidence, roster/report semantics, Phase 2K-A purity, and Phase 2K-B carry-forward artifacts all verify. MON P3, TUE P1, and TUE P2 are `incompatible_policy_evidence`: their reviewed historical evidence is useful for regression, retention, and risk analysis but cannot be represented as exact current-policy accuracy. No metric from an incompatible family is combined into the current strict safety figure. Missing or incomparable metrics are recorded as `unavailable`, never zero.

The audit normalized 171 reviewed rows across five evidence families: 15 MON P3 Phase 1.2J exception rows, 5 MON P4 Phase 1.2N exception rows, 31 MON P4 exact-current Phase 2H/2I rows, 100 TUE P1 multisession-ground-truth rows, and 20 TUE P2 multisession-ground-truth rows. Its session inventory explicitly maps 15 verified components: 2 for MON P3, 9 for MON P4, 2 for TUE P1, and 2 for TUE P2, covering source freezes/hashes, diagnostics/observations, review packages and restricted joins, labels/evaluations, authority reports, purity, and carry-forward. TUE P1 contains two human-reviewed unclear/unidentifiable rows. Restricted joins remain referenced by path and SHA-256; normalized public rows expose identity fingerprints rather than raw private identities, except the mandated already-known negative-regression identity. That exception is scoped to the exact MON P4 session-plus-track key, so the same textual track ID in TUE P1 remains restricted.

Strict identity safety remains separated by policy family:

- Exact current policy, MON P4: 19 strict accepts, all 19 correct; zero wrong-person, outsider, mixed, or unclear/unverifiable strict accepts.
- Historical MON P3 Phase 1.2J candidate: 7 strict accepts, all 7 correct; policy-incompatible with current authority.
- Historical MON P4 Phase 1.2N candidate: 4 strict accepts, all 4 correct; policy-incompatible with the current automatic authority contract.
- Historical TUE P1: 37 strict accepts, all 37 correct; policy/embedding-incompatible with current authority.
- Historical TUE P2: zero strict accepts; its reviewed recovery evidence belongs to a historical guarded family.

Guarded evidence is not treated as automatic authority. The exact-current MON P4 slice has 12 guarded candidates: 11 human-correct and one mixed-track candidate. The historical TUE P2 slice has 20 guarded candidates: 13 correct, 5 wrong-person, and 2 outsider absorptions. MON P3, historical MON P4 Phase 1.2N, and TUE P1 have no comparable guarded-candidate contract, so every guarded field for those families is explicitly `unavailable`, not zero. These results are not aggregated into one accuracy value because the policies are incompatible. They reinforce that guarded recovery must remain review-only and require a genuinely independent future session before any promotion decision.

Retention is likewise policy-scoped. MON P3 historical evidence had 9 previously correct production accepts, retained 3, lost 6, and recovered 4 correct rows. Historical MON P4 Phase 1.2N retained its one previously correct accept and recovered 3. The exact-current MON P4 reviewed authority retained all 19 strict-correct rows and supplied 11 human-confirmed guarded recovery rows. Historical TUE P1 retained 37 reviewed strict-correct accepts; its recovery fields are unavailable because that family contains no guarded-candidate contract. Historical TUE P2 supplied 13 correct guarded rows; its baseline retention/loss fields are unavailable because that slice contains only baseline-rejected shadow candidates. These are separate evidence-family findings, not one cross-policy retention score.

Phase 2K-A purity remains diagnostic and unchanged: 308 parent tracks produced 36 `pure_unsplit`, 179 `ambiguous_quarantined`, 91 `insufficient_observations`, 1 `mixed_quarantined`, and no defensible split candidates. Unsafe parent acceptance count is zero. The mandatory `CP1-cam5-back-TRK00005`, predicted `24011CSEAI0051`, remains human-disposition `mixed_track`, purity result `mixed_quarantined`, strict-accepted false, carry-forward `mixed_quarantined`, and official contribution zero. Exact evidence/signature matching never overrides its quarantine.

Phase 2K-B carry-forward findings remain 31 verified reviewed pairs: 18 exact-match eligible in isolation, 13 rejected, zero partial reuse, and one decisive mixed-quarantine override. The complete registry attempt remains rejected; no matching subset was applied and no child inherited a parent review.

Report semantics remain safe on the exact current MON P4 artifacts. Official totals remain 6 Present, 4 Needs Review, 16 Unconfirmed, 1 Missing Enrollment, 0 Absent, total 27. The archived automatic candidate remains 2 Present, 8 Needs Review, 16 Unconfirmed, 1 Missing Enrollment, 0 Absent, total 27. Historical frame-only output had represented 26 rows as Absent, but current low-quality/non-detection semantics use Unconfirmed and preserve Missing Enrollment distinctly, preventing false mass absence.

The authoritative CVO roster remains 27 unique rows with missing enrollment `2401100CSE0268`. `24011CSEAI0110` remains distinct from and is not aliased to `2401100CSE0110`; the CVO roster includes the former and excludes the latter. `24011CSEAI0061` remains excluded and `2401100CSE0237` remains included. Exact-current reviewed evidence has no out-of-roster prediction. Historical TUE P1 has one out-of-roster prediction, recorded as a historical roster-integrity finding and blocker, not silently erased and not attributed to the current policy.

All four sessions are contaminated. MON P3 was used for recognition, blind review, retention, ablation, and model-selection evidence. MON P4 was used for Phase 1.2N validation, promotion evidence, Phase 2G/2I processing, and human review. TUE P1 was used for recognition, review, calibration, adaptation, ablation, and promotion benchmarks. TUE P2 was used for recognition, review, shadow recovery, calibration, adaptation, and promotion benchmarks. This audit cannot establish independent generalization, authorize guarded-recovery promotion, or replace a future untouched-session test.

The final recommendation is `retain_current_policy_with_blockers`. The sufficiently reviewed exact-current strict evidence has zero unsafe accepts, no current roster-integrity violation, no false mass-absence behavior, and no unsafe carry-forward. The blockers are contamination of every existing session, historical-policy-only evidence for three sessions, one historical out-of-roster prediction, wrong-person/outsider historical guarded candidates, the exact-current mixed guarded candidate, and the absence of independent generalization evidence. No threshold change, model promotion, authority activation, or guarded promotion is authorized.

The final immutable output is `attendance_output/product_workflow/phase_2k_retrospective_audit/retrospective-audit-efe4cd29aa22738a079e`. It contains the required 20 artifacts. Its immutable manifest SHA-256 is `d9498d2268b99ea0acc1e6f412bd84c660004e3247032e1da87ca1ee6eca113e`. Independent verification passed; repeated materialization reused byte-identical output; deterministic-ID collisions and byte tampering fail closed. Three earlier immutable attempts remain preserved and superseded: `retrospective-audit-d7eab311ce1a9f81e756` preceded the explicit TUE P1 out-of-roster finding, `retrospective-audit-a57a75aa9fa45670bb14` preceded fail-closed availability fields and the exact 15-component inventory, and `retrospective-audit-cb4cf835e50e2feb1fc7` preceded session-scoping the mandatory public identity exception and binding normalization/privacy rules into the run fingerprint. None was rewritten or deleted.

Validation completed with 601 Python tests run: 600 passed, zero failures/errors, and the one pre-existing real Phase 1.2E/H artifact test skipped. The Phase 2K-C1 suite contributes 57 passing tests. Ordered Phase 2K-B, Phase 2K-A, Phase 2J, Phase 2I, and Phase 2H regressions passed. No frontend tests were run because no frontend, API, or shared frontend schema changed.

The final no-activation declaration records false for recognition, YuNet, SFace, classroom video decoding, reprocessing, official/candidate attendance changes, report revision/finalization changes, all authority changes, guarded promotion, thresholds, embeddings, roster, timetable, review registry, manual overrides, jobs, HOD configuration, and frontend changes. Protected operational hashes and independent official/candidate recounts are unchanged.

The recommended next task is Product Phase 2L stabilization and professor-demo preparation. A future genuinely untouched session remains mandatory before any independent-generalization or guarded-promotion claim.

## 20. Product Phase 2L-A: current-policy stabilization and professor-demo preparation

Product Phase 2L-A completed on 2026-07-22 as a frontend/backend contract-hardening, release-preflight, documentation, and immutable release-candidate phase. Its stabilization policy is `product-phase-2l-a-current-policy-stabilization-v1`; its frontend contract is `product-phase-2l-frontend-status-contract-v1`. It did not decode classroom video, run YuNet or SFace, run recognition, reprocess a session, change attendance, finalize a report, activate guarded recovery, change thresholds, promote a model, or alter protected operational state.

The production freeze remains family `embfam-274b5207b8b71294ff75`, variant `embfam-274b5207b8b71294ff75-d`, embedding SHA-256 `f088d827adc548ee95f46566d758fd71fc304d042c43f1ecffc6526b60bcd832`, summary SHA-256 `63885588c374c37f4da9bf85294f240bdf0f28cb585e77b894ab516139ca46ae`, pointer SHA-256 `7999b8ccf787dca9b8fb862f4e53ce7c1102eb729a3732c82fee76f3ba3ee05e`, 128 dimensions, 354 embedding records, top-3 aggregation, thresholds `0.48`/`0.08`, five checkpoints `CP1`-`CP5`, front/back cameras, and strict tracklet automatic authority. Guarded recovery remains review-only.

The frontend and backend now share one six-status contract: Present, Needs Review, Unconfirmed, Missing Enrollment, Absent, and Unknown. `Unresolved = Needs Review + Unconfirmed + Missing Enrollment + Unknown`. Unknown values fail closed. An evidence-flagged explicit Absent remains unresolved unless it is an explicit current manual choice or a persisted manual resolution. Present and Absent remain resolved outcomes; Unconfirmed means insufficient camera evidence rather than absence; Missing Enrollment means no usable production embedding.

Reports and Review Students now expose the same totals, filters, and meanings and always separate the Official Reviewed Revision, Automatic Candidate Revision, superseded/quality-failed historical result, review carry-forward state, and finalization state. Evidence presentation distinguishes strict accepted checkpoints, guarded recovery candidates, human-reviewed multi-frame checkpoints, mixed-track rejection, observation counts, evidence authority, and carry-forward. HOD Control, Help, Settings, and fallback copy reflect the current five-checkpoint front/back policy and the review-only guarded boundary. Existing Phase 2I/2J compatibility labels remain present.

The login page no longer embeds, displays, or prefills frontend credentials and no longer offers quick-fill demo accounts. Authentication remains backend-verified through the existing role-login flow. Frontend role metadata retains no username or password literals. The browser smoke check verified empty login fields, the authorized-login guidance, and zero browser warning/error messages; no credentials were entered or retrieved.

The read-only stabilization preflight verifies all protected hashes, the immutable Phase 2K-C1 package and recommendation, the production pointer/family/variant and embedding metadata, exact thresholds/authority/checkpoint layout, guarded recovery disabled for automatic authority, the unfinalized official session, exact official/candidate reports and totals, the 27-row roster and one unusable enrollment, the 31/30/1 review registry, mandatory mixed-track quarantine, all five rollback references, all three operator documents, final source hashes, and relevant frontend/backend contract tests. Every mismatch fails closed. The isolated test suite covers embedding/summary/threshold/authority drift, automatic guarded recovery, mixed quarantine, official/candidate totals, roster/enrollment, registry, finalization, recommendation, Phase 2K-C1 tampering, rollback references, protected-state preservation, no recognition/video invocation, deterministic output, idempotency, collision detection, and manifest tampering.

The controlled-demo documentation is:

- `docs/PHASE_2L_OPERATOR_GUIDE.md`
- `docs/PHASE_2L_PROFESSOR_DEMO_RUNBOOK.md`
- `docs/PHASE_2L_RECOVERY_AND_ROLLBACK.md`

The guides contain exact startup/health commands, safe page order, fixed report totals, status and evidence talking points, prohibited actions, shutdown/troubleshooting, short application-unavailable fallback, and verified Phase 2I, Phase 2J, and production-embedding rollback references. Rollback is explicitly outside the demo and requires separate authorization.

The final immutable output is `attendance_output/product_workflow/phase_2l_stabilization/stabilization-5fb42e9ff8f8286138c5`. Its run fingerprint is `5fb42e9ff8f8286138c505ff648d95ce4877f5fadeb68172ca133ee9d9ba1969`; its source-manifest fingerprint is `b31626b9b0adaf76ec4f27304218015db0f4dbb6569310e169943bd4b1974217`; and its immutable manifest SHA-256 is `6e86e63c5da4945cb133b6c8be3250340be1d0ede3c512531e9c44f6ced99153`. The source-manifest fingerprint is part of the deterministic run payload, so any implementation/documentation/source change produces a different run ID. The package contains exactly 12 required files. Independent verification passed, all 38 recorded source hashes match current bytes, and repeated materialization reused byte-identical output. Temporary-fixture tests prove deterministic collision and tamper detection.

An earlier immutable attempt, `stabilization-be157e81442916f08586` with manifest SHA-256 `c5f0980fbd9d24a8cefddcdded0cf5e201051dbdfe3e37fa2fb8b10f6910b215`, remains preserved and superseded. It preceded binding the source-manifest fingerprint into the deterministic run payload and was not rewritten or deleted.

Final exact report counts remain unchanged. Official Reviewed Revision: 6 Present, 4 Needs Review, 16 Unconfirmed, 1 Missing Enrollment, 0 Absent, 0 Unknown, total 27, unresolved 21. Archived Automatic Candidate Revision: 2 Present, 8 Needs Review, 16 Unconfirmed, 1 Missing Enrollment, 0 Absent, 0 Unknown, total 27, unresolved 25. The mandatory `CP1-cam5-back-TRK00005` remains quarantined and contributes zero official attendance. The Phase 2K-C1 recommendation remains `retain_current_policy_with_blockers`.

Validation completed with Python compilation passing; 23/23 Phase 2L stabilization tests; 38/38 frontend logic tests; a production Vite build with 50 modules; 57/57 Phase 2K-C1 tests; 14/14 Phase 2K-B tests; 15/15 Phase 2K-A purity tests; 12/12 Phase 2J tests; 8/8 Phase 2I tests; and the complete 626-test Python suite with 625 passed, zero failures/errors, and the one documented real Phase 1.2E/H artifact test skipped. The final read-only preflight, browser smoke check, independent artifact verifier, deterministic reuse check, source-manifest rehash, and protected-state rehash all passed.

The release candidate is ready only for the documented controlled professor demonstration of preserved evidence. It does not authorize live recognition, reprocessing, edits, finalization, configuration changes, or Live Demo. A genuinely untouched frozen session remains mandatory before any independent-generalization or guarded-recovery promotion claim.

## 21. Product Phase 2L-B: controlled professor-demo rehearsal and final release checklist

Product Phase 2L-B ran on 2026-07-22 under policy `product-phase-2l-b-controlled-demo-rehearsal-v1` as a read-only service, browser, release-checklist, and state-preservation rehearsal. It did not modify functional code. It verified the exact Phase 2L-A immutable package `stabilization-5fb42e9ff8f8286138c5` and manifest SHA-256 `6e86e63c5da4945cb133b6c8be3250340be1d0ede3c512531e9c44f6ced99153`; the documented preflight passed both before service startup and after service shutdown.

The backend started with no migration, no traceback, three persisted historical jobs, zero active jobs, and a healthy frozen strict-policy response. The controlled Vite instance started on port 5174 because port 5173 already had a pre-existing listener; the pre-existing listener was left untouched. The controlled backend and frontend processes were stopped. Startup and shutdown did not change attendance or job hashes.

The unauthenticated browser rehearsal verified empty username and password fields, no displayed/demo/quick-fill credentials, safe authorized-login guidance, a generic failure for one synthetic invalid login, no stored credential retrieval, and zero browser warnings/errors. At 1366×768 and 1024×768 the login page had no horizontal overflow, and the login fields/button were visible at the common laptop viewport. The login hero's general four-step product flow still mentions processing and export; this is an `important_post_demo` wording item because those actions are prohibited in the controlled demonstration.

Authenticated page rehearsal was not attempted. No secure operator credential was supplied, and the mandatory no-token API precheck exposed a release blocker: `get_current_user()` falls back to the first role-registry user when neither a bearer token nor `X-User-Id` is supplied. A request with no authentication therefore returned HTTP 200 with default HOD-scoped read access on `/api/profile`, `/api/auth/me`, `/api/attendance-sessions`, `/api/attendance/2026-06-22__B51__P4__CVO`, `/api/hod/overview`, and `/api/students`. `/api/hod/config` correctly returned 401 because it uses the explicit-HOD guard, but the other protected read surfaces expose roster/report data. Response bodies were not persisted in the rehearsal artifact. Credentials themselves were not exposed.

This is a professor-demo release blocker. The browser route order was exercised read-only without authentication: Login passed, while Dashboard, My Classes/Timetable, Reports, official/candidate views, Manual Review, Students, HOD Control, and Help correctly redirected the browser to Login. Static contracts and the passing frontend suite verify the expected copy and data mapping, but authenticated visuals, controls, totals, status colors, evidence presentation, and responsive states remain operator-required after the API authentication blocker is fixed.

Independent CSV recounts and the compact API summary still agree. Official Reviewed Revision is 6 Present, 4 Needs Review, 16 Unconfirmed, 1 Missing Enrollment, 0 Absent, 0 Unknown, total 27, unresolved 21, finalized false. Archived Automatic Candidate Revision is 2 Present, 8 Needs Review, 16 Unconfirmed, 1 Missing Enrollment, 0 Absent, 0 Unknown, total 27, unresolved 25. The official revision remains primary, guarded recovery remains review-only, and `CP1-cam5-back-TRK00005` remains mixed/quarantined with zero official contribution.

Identity integrity remains unchanged: CVO has 27 distinct roster rows and 26/27 production-embedding coverage; `2401100CSE0268` remains Missing Enrollment. `24011CSEAI0110` remains present and distinct from excluded `2401100CSE0110`; `24011CSEAI0061` remains excluded and `2401100CSE0237` remains included.

The final release checklist is `docs/PHASE_2L_FINAL_RELEASE_CHECKLIST.md`, SHA-256 `4e16d8fb989c1cb944b9985909c5963234c706fa2e51c9acd3651066c7a40fab`. It records all machine, startup, health, login, page-order, exact-total, prohibition, talking-point, limitation, fallback, shutdown, post-demo, stop-condition, and sign-off requirements. Sign-off is withheld. The fallback references the verified Phase 2L-A freeze, frontend contract, demo-readiness data, professor runbook, and checklist; no screenshots or fake attendance were generated.

The immutable rehearsal package is `attendance_output/product_workflow/phase_2l_demo_rehearsal/rehearsal-05c307e549637792f873`. Its run fingerprint is `05c307e549637792f87378d8391b6d28518ae91183624ef3c2bab120f26e5253`; route-check SHA-256 is `058b3ea3fd9bf2abd2fa8d43eb461e5189e3ebd0d74d4b8de1b17301fc278830`; frontend-build aggregate SHA-256 is `00740d6f72aa6f6295de7ee35af28a53e527e3d04abb9e93ec4fb2f5fcae15f3`; immutable-manifest SHA-256 is `e58a942a4bd091bdf8fc5f75d0351cbe68bfaef47d90f7bfd1c7678356f41c3d`. The package contains exactly the 15 required files, independently verifies, and repeated deterministic run-ID computation resolves to the same immutable directory without changing it.

Validation completed with 38/38 frontend Node tests, a 50-module temporary Vite production build, 23/23 Phase 2L-A Python tests, two passing Phase 2L stabilization preflights, browser login/redirect/viewport checks, independent Phase 2L-A and Phase 2L-B manifest verification, independent report recounts, and final protected-state rehash. A first temporary Vite build target under the elevated service-log directory failed with an EPERM mkdir error; a clean retry in the writable visualization workspace passed. The full Python suite was not rerun because no Python source or tests changed.

Every protected hash remains byte-for-byte identical to the Phase 2L-A freeze. Recognition, YuNet, SFace, classroom-video decoding, session reprocessing, Live Demo, attendance edits, candidate changes, new report revisions, finalization, authority changes, guarded promotion, embedding changes, threshold changes, roster/timetable changes, review-registry/manual-override/job changes, HOD configuration, roles, credentials, rollback, commit, push, deploy, and promotion all remained false.

Demo readiness is `not_demo_ready_release_blocker`. The next task must be a narrowly scoped Product Phase 2L-C authentication patch: remove implicit first-user fallback, require valid bearer authentication for protected read endpoints, preserve only deliberately public health data, add no-token negative regressions, rerun the full authenticated read-only page rehearsal with an authorized operator, and repeat protected-state verification. Do not perform the professor demo before that patch passes. Independent-session acquisition remains separately required for any generalization or guarded-promotion claim.

## 22. Product Phase 2L-C: explicit authentication enforcement and final controlled rehearsal

Product Phase 2L-C completed on 2026-07-22 under policy `product-phase-2l-c-explicit-authentication-v1`. It closes blocker `phase2l-b-unauthenticated-default-hod-read-access`. The root cause was implicit identity resolution: `get_current_user()` could accept a non-bearer identity hint and select the first role-registry user when no identity was supplied, while internal timetable/job helpers and missing-registry behavior also retained first/default-user paths. Because the first registry record was HOD, unauthenticated protected reads could disclose HOD-scoped data.

Authentication now fails closed. A request identity is established only by a valid, unexpired, non-revoked bearer session. `X-User-Id` is optional only as a post-bearer consistency hint and cannot authenticate or switch identity; mismatch is `401`. Missing, empty, malformed, unknown, expired, revoked, or inconsistent sessions are `401`. Authenticated but unauthorized roles/scopes are `403`. Only recognized HOD/admin and Faculty roles are accepted. Missing/invalid role-registry state yields no user, creates no fallback account, and selects no default/first user. A central `before_request` guard makes protected the default for every registered Flask route, with shared `require_authenticated_user`, `require_role`, and `require_hod` helpers. Existing explicit HOD write guards remain intact.

The deliberate public application allowlist is exactly `GET /`, `GET /api/health`, `POST /api/auth/login`, and `GET /static/<path:filename>`. `HEAD` follows `GET`; automatic empty `OPTIONS` is a protocol-only CORS exception and does not execute application-data route bodies. Health is minimal and does not expose users, roles, rosters, attendance rows, private report details, workflow state, operational paths, credentials, or session values. Every other registered route is protected.

Authorization remains role and ownership scoped. HOD has the existing authorized global read scope and guarded system writes. Faculty attendance, timetable, roster, report, review, job, and student reads remain limited to assigned subjects/sessions. Faculty downloads are permitted only for exact report references belonging to an assigned session and the two controlled reviewed/final CSV families; other-faculty report and download requests fail `403` without a distinct existence leak. Cross-faculty reports, raw video/upload inventory, reset, video-layout configuration, HOD overview, and HOD configuration remain HOD-only. Unrecognized roles fail closed.

Frontend API, upload, and blob/download calls now attach the bearer consistently. `X-User-Id` is never sent alone. Identity query fallback and direct unauthenticated report navigation were removed. A protected `401` clears stale local session state and returns the UI to Login; a `403` preserves the otherwise valid session. Login failures remain generic and are excluded from the protected-session invalidation event. No credential/token is displayed, prefilled, logged, placed in a URL, or stored in evidence. Manual Review received a narrow width/min-width correction after the controlled visual check found horizontal overflow; the corrected page was rechecked without overflow.

The generated inventory covers 37 protected method/route pairs. The controlled no-token matrix returned generic JSON `401` for all 37 with zero private-field exposures; its SHA-256 is `82249181e337c527a493293c5958d888290db623937a6db31eb14eb12d23556d`. The isolated authenticated role matrix contains 17 passing HOD, assigned-CVO Faculty, other-Faculty, unrecognized-role, download-scope, and bearer/hint-mismatch cases; its SHA-256 is `be06008921b9389133e7aebecce58827b809c4c26d1042ebeeba4c24d61fd74d`. The protected-route inventory SHA-256 is `a1f9691c5e2cc48133d5c73d6cf86263fa97c1dbd0fb36f61248795426edbc7a`.

No real operator credential was available or read. A temporary isolated fixture with randomized synthetic authentication was used only for automated route and visual checks, so production credential usability is not claimed. The 13 recorded browser checks cover Login; HOD Dashboard; My Classes/Timetable; Official Reviewed MON_P4; Archived Automatic Candidate; Manual Review; Students/Coverage; HOD Control; Help; and Faculty Dashboard, Reports, Manual Review, and Students/Coverage. Faculty navigation/data remained CVO-only. The browser showed 27 assigned CVO students, 26 covered, one Missing Enrollment, the exact official/candidate totals, the not-finalized state, and separate official versus archived labels. No browser warning/error or document-level overflow occurred at the active 1280-wide laptop viewport. No mutating control was used and Live Demo was not visited. The browser-matrix SHA-256 is `83e24b0b4c664409f7c376069b7a12ada6a52c6838f65f6efb00210a54e64a91`.

Validation passed in the required order: Python compilation; 14 targeted authentication/evidence tests; 61 existing report/HOD/configuration/workflow plus Phase 2L-A tests; 42 frontend Node tests; a 50-module Vite build containing 13 files and 527531 bytes with aggregate SHA-256 `7e2c1ec68381ae1f7c9079d3413756edc0f9efe7f47cbc1d7d64167728ec70e3`; the 37-case no-token matrix; the 17-case role matrix; controlled browser rehearsal; Phase 2L-A preflight; and the complete final 640-test Python suite with zero failures/errors and the one documented real Phase 1.2E/H artifact test skipped. Final stopped-state authentication and stabilization preflights passed.

The authoritative immutable output is `attendance_output/product_workflow/phase_2l_authentication_hardening/authentication-hardening-dc1e4eb0bdbef36515af`. Its run fingerprint is `dc1e4eb0bdbef36515af63ec00ab8e57db0ea5993a39ac78ac8c1bc9c9ac26db`; source-manifest SHA-256 is `75069a37b67737d54d1ec2b2da16b215098d383ed840d79e0f43ea8f11a7563e`; immutable-manifest SHA-256 is `c8d5a38d1e89cf0928e1cc18ff91e87b41ce0b1b237c32dddcdf9ac8e50dfac7`. It contains exactly the required 18 files. All 17 payload hashes/sizes independently verified, repeated materialization reused byte-identical output, and isolated deterministic-ID collision and byte-tamper tests fail closed. The run payload binds the verified Phase 2L-A and Phase 2L-B manifests, authentication policy, route inventory, no-token matrix, role matrix, complete browser matrix, frontend build, official/candidate reports, and source manifest.

An earlier immutable attempt, `authentication-hardening-f9028874ba95ab1f39bc` with manifest SHA-256 `85deeb5dbe4d34c3eb4a3e66b37300c14dcd35b205436249c7f167430ea72f79`, remains preserved and superseded. It preceded binding the complete Timetable/Students/Help browser evidence and source-manifest fingerprint into the run payload. It was not rewritten or deleted and is not release evidence.

All protected operational hashes remain the exact Phase 2L freeze values. Official MON_P4 remains 6 Present, 4 Needs Review, 16 Unconfirmed, 1 Missing Enrollment, 0 Absent, 0 Unknown, total 27, unresolved 21, finalized false. The archived automatic candidate remains 2 Present, 8 Needs Review, 16 Unconfirmed, 1 Missing Enrollment, 0 Absent, 0 Unknown, total 27, unresolved 25. Production remains family `embfam-274b5207b8b71294ff75`, variant `embfam-274b5207b8b71294ff75-d`, thresholds 0.48/0.08, strict automatic authority with guarded recovery automatic false. `CP1-cam5-back-TRK00005` remains mixed/quarantined with no carry-forward and zero official contribution. Attendance, jobs, roles, roster, timetable, overrides, review registry, reports, finalization, authority, HOD configuration, production pointer, embeddings, and summaries did not change.

Recognition, YuNet, SFace, classroom-video decoding, processing/reprocessing, report edit/export/finalization, guarded promotion, embedding rebuild, rollback, commit, push, deploy, and promotion did not occur. Both controlled services were stopped and their temporary credential/build/log files were removed after materialization; the pre-existing frontend listener on port 5173 was left untouched.

Demo readiness is `demo_ready_operator_login_required`. The authentication blocker is closed and synthetic authenticated route/visual verification is complete, but `demo_ready` requires a real authorized operator to complete the production login rehearsal privately. No further coding is recommended. The next task is only that operator-login read-only rehearsal followed by logout, controlled-service shutdown, both preflights, protected-state rehash, and independent report recount. A genuinely new frozen session remains separately required before any independent-generalization or guarded-recovery promotion claim.
