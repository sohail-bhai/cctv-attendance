# Phase 1.2G Conservative Tracklet Acceptance Calibration

## Safety boundary

- Offline shadow analysis only
- Candidate enabled: no
- Production recognition changed: no
- Match threshold changed: no
- Margin threshold changed: no
- Three-checkpoint attendance rule changed: no

## Frozen reviewed benchmark

- Total reviewed tracklets: 100
- Baseline accepted reviewed tracks: 37
- Baseline accepted false identities: 0
- Reviewed unresolved tracks: 63
- Recoverable correct top-1 unresolved tracks: 33
- Safety-negative unresolved tracks: 30

## Identity-grouped holdout result

- Folds: 5
- Out-of-fold recovered tracks: 7
- Out-of-fold false accepts: 0
- Out-of-fold shadow precision: 1.0
- Folds with at least one recovery: 5
- Folds with a false accept: 0

## Full-benchmark shadow candidate

- Candidate ID: cal-5cd35b60dd83
- Recovered unresolved tracks: 8
- False accepts: 0
- Shadow precision: 1.0
- Recovery rate among correct top-1 unresolved tracks: 0.2424

## Decision

- Recommendation: promote_to_multisession_shadow_validation
- Production decision: keep_disabled
- Next validation: Run the disabled candidate on an untouched session such as TUE_P2 or MON_P3 and blind-review every newly accepted recovery before considering any production integration.
- Rationale: Zero false accepts were observed across identity-grouped holdouts and the full benchmark, with recoveries distributed across multiple folds.

This is a single-session, reviewer-selected benchmark. A safe result here may only advance the candidate to an untouched-session shadow test; it is not production approval.
