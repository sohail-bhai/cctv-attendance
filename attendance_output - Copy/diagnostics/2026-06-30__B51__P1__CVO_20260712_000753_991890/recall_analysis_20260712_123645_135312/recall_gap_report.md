# Phase 1.2F Recall-Gap Analysis

## Scope

This report analyzes existing Phase 1.2D/1.2E artifacts. Recognition thresholds, the three-checkpoint attendance rule, and official frame-mode attendance were not changed. Initial model-derived categories remain provisional until the unresolved blind review is complete.

## Embedding Coverage

- CVO roster students: 26
- Embedding available: 23
- Missing embeddings: 3 (2401100CSE0268, 24011CSEAI0061, 24011CSEAI0110)
- Annotated keys canonicalized: 4
- Canonical collisions: 0

## Current Student Outcomes

- Actual present: 26
- Present under three-checkpoint evidence: 3
- False negatives: 23

- correct_candidate_score_below_threshold: 10
- embedding_missing: 3
- insufficient_checkpoint_coverage: 9
- no_usable_crop: 1
- present_confirmed: 3

## Unresolved Review Selection

- Selected tracks: 63
- Camera distribution: {"cam10": 21, "cam5": 42}
- Checkpoint distribution: {"CP1": 15, "CP2": 10, "CP3": 13, "CP4": 14, "CP5": 11}
- Rejection distribution: {"margin_below_threshold": 1, "score_and_margin_below_threshold": 51, "score_below_threshold": 11}

## Decision

No final A-E engineering recommendation is made yet. Complete the unresolved blind review, then run the unresolved evaluator.
