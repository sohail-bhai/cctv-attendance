# Compact Enrollment Ranking

This package is a deterministic subset of the immutable 855-item broad review.
It does not classify an image as wrong and does not modify an enrollment source.

## Priority order

1. Identity-integrity signals: ownership/folder mismatch, cross-student duplication, meaningful other-vs-own medoid disagreement, multiple faces, and reviewed false-accept involvement.
2. Embedding-integrity signals: the 13 stored embedding outliers, own-class outliers, and severe disagreement with the owning identity's enrollment set.
3. Material usability blockers: unreadable/no-face/tiny-face conditions and severe quality issues only when the image contributes or belongs to an already high-risk identity.

## Deterministic controls

- Risk points are additive by confirmed issue type; model similarity contributes only when the forensic audit already marked an own-vs-other disagreement.
- Model similarity is never identity ground truth.
- Exact source hashes are deduplicated.
- Ordinary identities are capped at 3; reviewed attractors or identities with multiple critical issue types are capped at 5.
- Blur/brightness-only rows are deferred rather than used as package padding.

Selected: 50
Deferred: 805
Embedding outliers considered: 13
Embedding outliers selected: 12
