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
2. Phase 2K-B next: carry-forward verification and independent-session preparation. Revalidate exact-source, exact-embedding, exact-purity, exact-lineage, and exact-evidence signature matching; mixed or changed evidence must fail closed. Freeze the diagnostic policy and prepare reproducible inputs/review material for the next untouched session without activating authority.
3. Second independent session. Run the revised pipeline on a new untouched class session with frozen policy and blind review.
4. Guarded promotion gate. Consider activation only if independent-session retention and safety criteria pass, source/embedding fingerprints are complete, rollback is verified, and approval is explicit.
5. Stabilization and professor demo. Freeze the validated workflow, resolve validation drift, document recovery/operator procedures, and prepare a controlled demonstration using non-destructive sample data.

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
