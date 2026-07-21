# Phase 1.2I-B Regression - embfam-274b5207b8b71294ff75

Decision: `built_unapproved_pending_new_untouched_session`

The benchmark was rescored from the 120 frozen, hash-verified reviewer contact sheets because compatible observation embedding vectors were not stored. No video was decoded and no attendance output was written.

| Model | Availability | Rows | Accepted | Correct | Wrong | Outsider | Mixed |
|---|---|---:|---:|---:|---:|---:|---:|
| production_baseline | available | 120 | 27 | 27 | 0 | 0 | 0 |
| variant_a_cleaned_enrollment | available | 120 | 27 | 27 | 0 | 0 | 0 |
| variant_b_tue_p1_cross_session | available | 100 | 26 | 26 | 0 | 0 | 0 |
| variant_c_tue_p2_cross_session | available | 20 | 2 | 2 | 0 | 0 | 0 |
| variant_d_full_contaminated_descriptive | available | 120 | 32 | 32 | 0 | 0 | 0 |

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

Next step: Reserve and run a different untouched validation session. MON_P3 is design evidence and cannot be reused as final promotion evidence.
