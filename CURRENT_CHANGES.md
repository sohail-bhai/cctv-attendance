# Current Changes

## 1. Run title and timestamp

- Title: Product Phase 2K-B — exact review carry-forward verification and untouched-session preparation
- Completed: `2026-07-22 00:27:59 +05:30`
- Repository: `F:\sohail\Class_Attendance_YuNet_SFace`
- Result: complete; implementation, immutable evidence, candidate-session audit, capture contract, full regression, and preservation checks passed.

## 2. Goal and scope

This phase implemented a versioned, deterministic, independently testable exact-review carry-forward contract and prepared—but did not execute—the next independent untouched-session experiment. The implementation is diagnostic-only and has no production integration path into recognition, attendance, guarded recovery, or report authority.

No classroom video was decoded. YuNet/SFace did not run. MON P4 was not reprocessed. No new classroom session was processed. No official or candidate attendance, finalization, authority, model, threshold, roster, timetable, review registry, job, credential, role, HOD, manual-override, or frontend state was changed.

## 3. Git baseline and dirty-state notes

- Branch baseline: `main`
- HEAD baseline: `cc585a3aaceaa9b1987579c68e862da9dbf95ede`
- Branch and HEAD after completion: unchanged.
- The worktree was already substantially dirty before this phase. Existing tracked changes included backend, operational JSON, frontend, and recognition-script paths; `debug_faces/.gitkeep` was already deleted; many source, test, model, backup, artifact, and documentation paths were already untracked.
- All pre-existing user changes were preserved. No reset, clean, checkout, discard, normalization, commit, push, deploy, or branch change was performed.

## 4. Exact files created or changed

Functional implementation:

- `src/face_attendance/product_phase_2k_carry_forward.py`
- `scripts/run_product_phase_2k_carry_forward.py`
- `scripts/run_product_phase_2k_carry_forward.ps1`

Tests:

- `tests/test_product_phase_2k_carry_forward.py`

Durable documentation:

- `PROJECT_CONTEXT.md`
- `CURRENT_CHANGES.md`

Immutable output directory:

- `attendance_output/product_workflow/phase_2k_carry_forward/carry-forward-e20a746d3d34c8141ab5/`

No existing production source, operational state, frontend file, Phase 2K-A artifact, review registry, model, report, or historical artifact was edited.

## 5. Carry-forward policy

- Policy version: `product-phase-2k-b-exact-carry-forward-v1`
- Policy file SHA-256: `09b88a62f69fabd1b85594889d1dab0b959d9bff515ebb3919b5904592a097c0`
- Diagnostic-only: true
- Production activation permitted: false
- All-or-nothing: true
- Partial review reuse: forbidden
- Parent-review inheritance by children: forbidden
- Mixed quarantine overrides exact hashes/signatures: true
- Ambiguous, incomplete, missing, duplicated, unverified, or tampered evidence: rejected

The pure evaluator returns exactly one deterministic result plus one reason code. `exact_match_eligible` is the only eligible result. Every failure returns an empty carried-review set, `partial_carry_forward=false`, and `inherited_by_children=false`.

## 6. Eligibility dimensions

Every attempt binds exactly to:

1. Session date, section, period, subject, and canonical session ID.
2. Every source path, SHA-256, checkpoint, camera, canonical order, source-layout version, and path-normalization version.
3. Production family, active variant, embedding SHA-256, summary SHA-256, dimension, and aggregation policy.
4. Match threshold, margin threshold, checkpoint rules, tracklet policy, purity policy, zone policy, authority policy, and processing contract.
5. Parent track ID, checkpoint, camera, ordered observation membership/order/time indices/span, evidence fingerprint, purity outcome/fingerprint, and quarantine state.
6. Child parent/child IDs, boundary, observation span/membership, child evidence fingerprint, lineage fingerprint, purity outcome, and policy version when children exist.
7. Registry ID, original evidence signature, review disposition, reviewer-export fingerprint, joined-review fingerprint, and immutable-evaluation fingerprint.
8. Source-manifest hash, Phase 2K-A purity-manifest hash, carry-forward-policy hash, artifact verification status, and canonical path-normalization rules.

Physical list reorder is equivalent only when unique contiguous canonical-order bindings remain exact. Separators and Windows drive-letter case normalize; path-component case is preserved so distinct case-sensitive paths are not collapsed.

## 7. Fail-closed reasons

The immutable failure-reason registry contains explicit codes for invalid hashes, unverified/tampered manifests, deterministic-ID collisions, duplicate sources/orders/parents/children/reviews/signatures, missing required fields, missing child evidence, every session/source/model/threshold/policy/manifest/review mismatch, observation membership/order/time drift, parent purity/quarantine drift, ambiguous purity, mixed quarantine, changed child set/boundary/span/evidence/lineage/outcome, and partial review-set mismatch.

The mandatory fixture matrix covers 34 provenance/semantic scenarios. The immutable verification matrix contains 68 rows total: 34 isolated contract scenarios, two immutable-writer scenarios, 31 verified real review-pair scenarios, and the complete 31-row registry attempt. All rows passed their expected result/reason contract.

## 8. Mandatory mixed-track result

- Track: `CP1-cam5-back-TRK00005`
- Predicted identity: `24011CSEAI0051`
- Existing evidence signature: `3e66e6ecfa57ea607bedc59578af159fc7c85774f21ae15ffd98a029b79ea983`
- Ordered observations: 39
- Time span: 0.48 to 19.20 seconds
- Phase 2K-A outcome: `mixed_quarantined`
- Child count: 0
- Carry-forward result: `mixed_quarantined`
- Reason code: `mixed_track_quarantine`
- Partial carry-forward: false
- Child review inheritance: false
- Official contribution: 0

All other bound hashes and signatures match. This proves exact signature/provenance cannot override the human mixed-track quarantine.

## 9. Exact Phase 2K-A inputs and hashes

- Phase 2K-A policy: `product-phase-2k-a-diagnostic-tracklet-purity-v1`
- Phase 2K-A run ID: `tracklet-purity-43cec17498dc05615502`
- Phase 2K-A immutable manifest SHA-256: `7e0da7fd9b3edb7411a31be94f5f42063365bd0a6831735948700d263148c552`
- Phase 2K-A source manifest SHA-256: `a1c13eabffaa05396669b41cec6af0bfe655533a4312083073e07a6d5cba225b`
- Phase 2K-A complete manifest verification: passed before any rows were consumed.
- Phase 2K-A read-only preflight reverified Phase 2G diagnostics, Phase 2H output/evaluation manifests, Phase 2I diagnostics/authority manifest, Phase 2J revision manifest, source-video hashes, official/candidate CSVs, review registry, attendance state, production files, and the known mixed review image.
- Review registry internal fingerprint: `b2afa12c10104025f31c76f02ea91b9c33720cccb09773ca5639da39c1577a69`
- Review registry source file SHA-256: `3b04e9aecfdfb1f64346c4a0e709fd9c36d7c56545bf816d6641b4c2a2e80841`
- Verified review pairs: 31; exact pure pairs diagnostically eligible in isolation: 18; rejected pairs: 13.
- Complete registry attempt: rejected `mixed_quarantined`; the 18-row subset was not reused.

## 10. Candidate-session inventory

The audit hashed all 40 videos in four complete prepared sessions. Candidate source fingerprint set: `96fd16e6d0189c43dc8aa9dc997b37274c129d8512c4f1d2d340d2802eaaa941`.

| Session | Layout | Prior uses | Result |
| --- | --- | --- | --- |
| `2026-06-22__B51__P3__CVO` / MON P3 | 5 checkpoints, back/front, 10 videos | recognition, blind review, retention, source ablation, model selection/promotion evidence | rejected contaminated |
| `2026-06-22__B51__P4__CVO` / MON P4 | 5 checkpoints, back/front, 10 videos | Phase 1.2N, promotion evidence, Phase 2G/2I, human review, retention | rejected contaminated |
| `2026-06-30__B51__P1__CVO` / TUE P1 | 5 checkpoints, back/front, 10 videos | recognition, human review, calibration/regression/adaptation, ablation, promotion benchmark | rejected contaminated |
| `2026-06-30__B51__P2__CVO` / TUE P2 | 5 checkpoints, back/front, 10 videos | recognition, human review, shadow recovery, calibration/regression/adaptation, promotion benchmark | rejected contaminated |

The audit verified relevant historical source-freeze, shadow-validation, evaluation, multi-session ground-truth, and source-ablation manifests. A session was not treated as untouched merely because it lacked a Phase 2K artifact.

## 11. Untouched-session recommendation

- Recommendation status: `no_existing_untouched_session`
- Existing session selected: false
- Contaminated substitute allowed: false

Required acquisition: a new CVO/B51 class session, exact CP1-CP5 back/front clips, ten unique source hashes, complete size/duration/FPS/frame-count/capture-timestamp metadata, immutable source freeze before processing, no prior recognition/review/tuning/calibration/selection/ablation/promotion/retention/benchmark use, frozen production configuration, and a blind-review plan fixed before results are visible.

## 12. Capture contract

- Version: `product-phase-2k-c-independent-session-capture-v1`
- Execution authorized by this phase: false
- Runner validation interface: `scripts/run_product_phase_2k_carry_forward.py validate-capture --package-dir <path>`
- Validation hashes files and checks provenance/configuration only; it does not decode video or run recognition.

The contract fixes current production family/variant and hashes, 128 dimensions, top-3 aggregation, `0.48`/`0.08`, exact Phase 2I checkpoint/tracklet/zone settings, current strict automatic authority, Phase 2K-A purity policy, and no parameter tuning after results.

Diagnostic rows require stable parent/observation IDs, time indices/timestamps, boxes, geometry continuity, quality metrics, private local vote/score/margin evidence, selected/discarded flags, and exact source/checkpoint/camera provenance.

Appearance evidence uses restricted derived full-parent pairwise, adjacent, local-window, and cross-boundary similarity matrices. Raw per-observation vectors are not persisted. Matrices are prohibited from public/frontend/blind-review exposure, enrollment, and model rebuilding. The contract assumes authenticated encrypted transport and access-controlled encrypted diagnostic storage outside web/public roots, explicitly does not claim the repository can enforce host disk encryption, and requires deletion evidence after the approved diagnostic lifecycle unless retention is explicitly extended.

Blind review uses randomized IDs, hides identity/scores/automatic decisions, shows before/after observations, supports single-person/mixed/unclear/outsider/wrong-person labels, stores its private join separately, and fails closed on missing, duplicate, or unknown review items.

## 13. Tests and results

Validation order and results:

1. Python compilation for the new module, runner, and tests: passed.
2. New Phase 2K-B targeted suite: 14 passed.
3. Existing Phase 2K-A purity suite: 15 passed.
4. Existing Phase 2J carry-forward/revision suites: 12 passed.
5. Existing Phase 2I authority suite: 8 passed.
6. Phase 2H was not separately targeted because no shared Phase 2H/evidence implementation changed; its tests ran in full discovery.
7. Full `venv\Scripts\python.exe -m unittest discover -s tests -v`: completed successfully. Exact programmatic capture: 506 run, 505 passed, zero failures, zero errors, one skipped.
8. The single skip remains the documented real Phase 1.2E/H benchmark fixture whose external artifacts are unavailable. It was not hidden or converted.
9. Phase 2K-B immutable verification after tests: passed.
10. Repeated materialization: byte-identical idempotent reuse passed.
11. Tamper detection and deterministic-ID collision tests: passed.

No frontend or shared API schema changed. Frontend tests/build were not run, and the unrelated documented frontend expectation drift remains untouched.

## 14. Immutable output path and hashes

Output: `attendance_output/product_workflow/phase_2k_carry_forward/carry-forward-e20a746d3d34c8141ab5/`

| Artifact | SHA-256 |
| --- | --- |
| `acceptance_criteria.json` | `01eb1036a44a8f9a92350134c12f118ffea16a0d87d01b527cf23b8434e84493` |
| `blind_review_schema.json` | `0bcad27985b76cf17804990c2edfaf8d61378a9ecba21daf72c0e4927f5ae0e7` |
| `candidate_session_inventory.csv` | `d0efb4fefba059d513aec150fa4ee4785b573e2ef5e2e2088f800b42f3ac3d1b` |
| `carry_forward_failure_reasons.csv` | `6717a442be5d85a5202d38780223b740cae035b4d7a0169e18f6dff734373c3f` |
| `carry_forward_policy.json` | `09b88a62f69fabd1b85594889d1dab0b959d9bff515ebb3919b5904592a097c0` |
| `carry_forward_verification_matrix.csv` | `04cc8c66a53f5c0e45a759d853ac0af26a2dcfd546552038d837eb0fae3ffab1` |
| `diagnostic_observation_schema.json` | `d66c5dc8586c81da1bb0a0c6a016a579b3684afdb76f2e9a7c374a31a08f41ee` |
| `evaluation_summary.json` | `e4165d3333dd64b2ce7a06d84327efa9e0b033991e23e3585e29edacc049777d` |
| `exact_match_contract.json` | `62ebe0640324e5235dc85fa94db45f6de6934451f0b773bd62dd113ab704c883` |
| `immutable_manifest.json` | `79a38737c6057768dc4128c3507fced268ea25c0beb84121a27cf2bb23195ac3` |
| `independent_session_capture_contract.json` | `30f361b5f63549c853f8fd9187a157230d58184a8ccd9511d7594870b0e3f092` |
| `independent_session_recommendation.json` | `ad6e85f80e2dcd52e2fcb9ad068db216bb59dcb975840e0f1a7d067efb688292` |
| `mixed_track_regression.json` | `a35ffe258fa2ef8b03492bda46d7728ba6107d85f4b6909d3dfba2b48e675f89` |
| `no_activation_declaration.json` | `2b95c5f67529f59f03b434eb2fc712dcd54f5cbcfaa36161b87e1cba95dda695` |
| `retention_criteria.json` | `91cb4e88b6cac9fafb2f3bbf7fda81d839fa39e5cfd185fc42fdd1fec57db94a` |
| `source_manifest.json` | `79f5ec385c222b59572680ef6cbcff8d04309617ee87a836b786c4c689ac83b9` |

The run ID is derived from the verified Phase 2K-A manifest, carry-forward policy version, review-registry fingerprint, production embedding fingerprint, complete prepared-source fingerprint set, and capture-contract version.

## 15. Protected operational before/after hashes

Every protected file had the same SHA-256 before implementation and after all tests/materialization:

| File | Before and after SHA-256 |
| --- | --- |
| `data/attendance_status.json` | `46fd2df55cc3f61c5fa03c893937eb2828b9cc2af064ea9e427276b79a9d0b46` |
| `data/job_runtime.json` | `5a23f09654d91fac09804cd97c2fa114bf4956a914f2e02ce2b7090f730141ed` |
| `data/role_users.json` | `77125140004294f2834fc869fd590e167d8ab78cff5dbeda1c074ab76c177517` |
| `data/student_faculty_map.json` | `0eca63ea58d12a73d3bd11e620b74be733a3219c9244e8a4a267d5b02acd3f91` |
| `data/manual_overrides.json` | `e9c6bd35c23c353795edb5d3c02e808793d7cbd90c7b0a87fdfc686f8fd5cdcf` |
| `data/review_evidence_registry.json` | `3b04e9aecfdfb1f64346c4a0e709fd9c36d7c56545bf816d6641b4c2a2e80841` |
| `models/student_embeddings.pkl` | `f088d827adc548ee95f46566d758fd71fc304d042c43f1ecffc6526b60bcd832` |
| `models/embedding_summary.csv` | `63885588c374c37f4da9bf85294f240bdf0f28cb585e77b894ab516139ca46ae` |
| `models/current_embedding_version.json` | `7999b8ccf787dca9b8fb862f4e53ce7c1102eb729a3732c82fee76f3ba3ee05e` |
| `timetable_b51_2026_2027.csv` | `10bbd578a859fbd4fa228c4eb4f1f92e7de5b92a1c4236d71a00337eecac9c57` |

Direct report verification after all tests:

- Official MON P4: 6 Present, 4 Needs Review, 16 Unconfirmed, 1 Missing Enrollment, 0 Absent, total 27.
- Archived candidate: 2 Present, 8 Needs Review, 16 Unconfirmed, 1 Missing Enrollment, 0 Absent, total 27.
- CVO roster: 27 unique members; `2401100CSE0268` is the only missing enrollment; mandatory distinct/membership constraints passed.
- Production: `embfam-274b5207b8b71294ff75` / `embfam-274b5207b8b71294ff75-d`, exact embedding and summary hashes above.

## 16. Whether recognition ran

- Recognition ran: false
- YuNet inference ran: false
- SFace inference ran: false
- Test fixtures and file hashing are not recognition.

## 17. Whether video was processed

- Classroom video decoded: false
- Video processing/reprocessing ran: false
- MON P4 reprocessed: false
- New session processed: false

## 18. Attendance, finalization, and reports

- Official attendance changed: false
- Archived candidate attendance changed: false
- Report finalized: false and unchanged
- Reports finalized, edited, or superseded: false
- Manual overrides changed: false

## 19. Authority

- Official reviewed authority changed: false
- Future automatic authority changed: false
- Guarded authority changed: false
- Final authority changed: false
- Guarded recovery promoted: false
- Phase 2K purity activated in production: false
- Production activation path added: false

## 20. Other protected state

- Embeddings/family/variant/summary/pointer changed: false
- Embeddings rebuilt: false
- Roster membership changed: false
- Thresholds `0.48`/`0.08` changed: false
- Checkpoint/tracklet/zone production policy changed: false
- Review registry changed: false
- Timetable changed: false
- Jobs created/retried/cancelled/reset/changed: false
- HOD configuration changed: false
- Roles, credentials, or permissions changed: false
- Frontend/UI changed: false

## 21. Blockers and uncertainties

- Phase 2K-B blocker: none.
- No genuinely untouched complete session exists among current prepared sources; this is an acquisition prerequisite for Phase 2K-C, not a Phase 2K-B implementation failure.
- Historical TUE P1/P2 sources have no dedicated source-freeze manifest comparable to MON P3/P4, but current video hashes are recorded and their human-reviewed/calibration contamination is independently established, so they remain definitively ineligible as untouched sessions.
- The repository cannot itself guarantee host-disk encryption; the capture contract records encryption/access assumptions and requires the Phase 2K-C operator to satisfy them.
- The pre-existing real Phase 1.2E/H test skip and unrelated frontend expectation drift remain unchanged.
- No useful missing Codex capability blocked the phase; no dependency or skill installation was needed.

## 22. Recommended next task

Product Phase 2K-C independent untouched-session execution.

Do not execute Phase 2K-C until a genuinely new session and every prerequisite below are available and explicit recognition authorization is given.

## 23. Exact prerequisites for Phase 2K-C

1. Newly acquired CVO/B51 session never used for recognition, review, tuning, calibration, selection, ablation, retention, promotion, or benchmarking.
2. Exact CP1-CP5 checkpoint folders with one back and one front clip each; exactly ten source videos.
3. Immutable source freeze before any recognition, including canonical paths/order, SHA-256, sizes, duration, FPS, frame counts, and capture timestamps.
4. Successful `validate-capture` runner verification with no video decoding.
5. Frozen production family `embfam-274b5207b8b71294ff75`, variant D, exact production hashes, 128 dimensions, top-3 aggregation, thresholds `0.48`/`0.08`, and current Phase 2I checkpoint/zone/tracklet authority.
6. No parameter tuning after results become visible.
7. Restricted encrypted diagnostic storage and authenticated encrypted transport; no web/frontend access.
8. Derived pairwise/local-window appearance matrix serialization plus complete observation/geometry/quality/local-vote provenance.
9. Blind-review export contract frozen in advance, randomized IDs, no predicted identity/scores, private join separated, immutable input manifest, and completeness validator.
10. Acceptance/retention criteria frozen in advance: zero wrong-person, outsider, mixed-parent, lost-strict-identity, unsafe-carry-forward, or leakage events.
11. Protected operational hashes captured again before execution.
12. Separate explicit authorization to decode video and run recognition. Phase 2K-B does not grant it.

## 24. Planner handoff

Start the next prompt by reading `PROJECT_CONTEXT.md`, this file, and the immutable Phase 2K-B output. Verify the Phase 2K-B immutable manifest SHA-256 `79a38737c6057768dc4128c3507fced268ea25c0beb84121a27cf2bb23195ac3` and every listed file before consuming the capture contract. Do not reuse MON P3, MON P4, TUE P1, or TUE P2 as the independent session.

Require the new capture package to pass the hash-only `validate-capture` interface before recognition. Freeze the blind-review material and acceptance plan before results. If any prior-use flag, source hash, model hash, policy value, path/order binding, or manifest verification differs, stop safely. During Phase 2K-C, recognition may run only under explicit authorization; official attendance and authority must still remain unchanged until a separate guarded promotion decision is explicitly approved.

After the independent run, compare the strict baseline, purity/split candidates, and guarded recovery candidates; report recovered checkpoints, lost correct evidence, unsafe identities, outsider absorption, mixed parents, unverifiable items, leakage, and carry-forward safety. A passing evaluation still must not auto-promote. The next decision after Phase 2K-C is a separately authorized guarded promotion gate, followed by stabilization and the professor demo.
