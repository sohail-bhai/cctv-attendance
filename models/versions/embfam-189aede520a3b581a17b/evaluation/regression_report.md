# Phase 1.2I-B Regression - embfam-189aede520a3b581a17b

Decision: `built_unapproved_insufficient_evidence`

The benchmark was rescored from the 120 frozen, hash-verified reviewer contact sheets because compatible observation embedding vectors were not stored. No video was decoded and no attendance output was written.

| Model | Availability | Rows | Accepted | Correct | Wrong | Outsider | Mixed |
|---|---|---:|---:|---:|---:|---:|---:|
| production_baseline | available | 120 | 27 | 27 | 0 | 0 | 0 |
| variant_a_cleaned_enrollment | available | 120 | 27 | 27 | 0 | 0 | 0 |
| variant_b_tue_p1_cross_session | unavailable_no_usable_source_evidence | 0 | 0 | 0 | 0 | 0 | 0 |
| variant_c_tue_p2_cross_session | available | 20 | 2 | 2 | 0 | 0 | 0 |
| variant_d_full_contaminated_descriptive | available | 120 | 31 | 31 | 0 | 0 | 0 |

Variant D is a **training-contaminated descriptive result over the CCTV sessions that produced technically usable embeddings**. It is not promotion evidence. Unavailable cross-session variants are recorded explicitly and are never fabricated from enrollment-only records.

## Safety status

- Production integrity: PASS
- False-identity gate: PASS
- Leakage gate: PASS
- Candidate approved: no
- Candidate promoted: no
- MON_P3 processed: no

## Review scalability contract

- Existing manual reviews are one-time bootstrap ground truth and are reused.
- Future embedding updates are incremental and automatically regression-tested.
- Live CCTV review is exception-only for unknown, conflicting, or weak evidence.
- Human review must not grow linearly with frames, cameras, or classrooms.

Next step: Preserve this partial family and do not repeat the completed manual reviews. Recover technically usable TUE_P2 adaptation evidence through an automated, same-track best-frame workflow or future live exception evidence, then build a new immutable family. MON_P3 remains untouched and blocked.
