# Phase 1.2G Validation Results

## Static and regression validation

- Python compilation: PASS
- New calibration tests: 4/4 PASS
- Focused regression suite: 19/19 PASS
- Broader repository suite: 79 tests PASS; one unrelated import failure because Flask is not installed in the sandbox (`tests.test_session_identity` imports `app.py`)

## Real TUE_P1 benchmark validation

- Frozen benchmark: 100 reviewed tracklets
  - accepted reviewed: 37
  - corrected unresolved reviewed: 63
- Baseline accepted false identities: 0
- Recoverable correct top-1 unresolved tracks: 33
- Safety-negative unresolved tracks: 30
- Identity-grouped folds: 5
- Out-of-fold recoveries: 7
- Out-of-fold false accepts: 0
- Out-of-fold recoveries appeared in all 5 folds
- Full-benchmark shadow recoveries: 8
- Full-benchmark false accepts: 0
- Candidate: `cal-5cd35b60dd83`
- Decision: `promote_to_multisession_shadow_validation`
- Production decision: `keep_disabled`

## Candidate safety gates

- candidate must belong to authoritative CVO roster
- aggregate top candidate must match dominant frame candidate
- best score >= 0.38
- margin >= 0.03
- dominant vote >= 50%
- observations >= 3
- selected observations >= 3
- consistent embeddings >= 3
- consistent/embedding ratio >= 0.50
- pairwise embedding similarity median >= 0.45

## Unchanged production behavior

- official match threshold remains 0.48
- official margin threshold remains 0.08
- attendance requires three checkpoints
- embeddings unchanged
- recognition not rerun
- candidate disabled
