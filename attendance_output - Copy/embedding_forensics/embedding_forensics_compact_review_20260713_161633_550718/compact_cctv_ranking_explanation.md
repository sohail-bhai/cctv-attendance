# Compact CCTV Crop Ranking

This package is a deterministic subset of the immutable 98-item human-identified CCTV review.
Actual_Roll from completed human review is the only identity authority.

## Priority order

1. Reviewed false-accept attractors and identities with repeated top-1 misses.
2. Identities with weak current embedding coverage or robust enrollment outliers.
3. Strong CCTV-domain crop quality with checkpoint, camera, pose, and distance diversity.

## Deterministic controls

- Only human-eligible, identified rows with identity_source=human_review_actual_roll are eligible.
- Mixed, not-in-mapping, uncertain, unidentifiable, invalid-roll, and broken-evidence rows are excluded.
- At most one crop per track and source hash is selected.
- Ordinary identities are capped at 3; confusion/miss priorities are capped at 4.
- Repeated checkpoints/cameras and high nearest-crop similarity receive deterministic penalties.
- No crop is copied into a dataset during compact review generation.

Selected: 36
Deferred: 62
Selected students: 16
