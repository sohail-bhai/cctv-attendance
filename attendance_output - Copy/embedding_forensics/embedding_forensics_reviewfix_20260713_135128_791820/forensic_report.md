# Phase 1.2I-A Embedding Forensics

## Safety outcome

- Rejected candidate: `cal-5cd35b60dd83` remains disabled and unapproved.
- Production embeddings changed: no.
- Enrollment datasets changed: no.
- Official thresholds, attendance rule, and official attendance changed: no.
- Candidate embedding version built or promoted: no.

## Reviewed benchmark

- Tracks: 120
- Sessions: 2026-06-30__B51__P1__CVO, 2026-06-30__B51__P2__CVO
- Status distribution: {"identified": 105, "mixed_track": 1, "not_in_mapping": 12, "unidentifiable": 2}
- Top-1 correct: 83
- Top-1 wrong for identified people: 22
- Outsider/not-in-mapping absorptions: 12
- TUE_P1 and TUE_P2 are reviewed regression/possible-adaptation data, not untouched validation.
- MON_P3 remains reserved and was not processed.

## Attractors for investigation

- 2401100CSE0050: 6 reviewed false accepts, 3 distinct known actual identities, 3 outsider/not-in-mapping accepts
- 24011CSEAI0007: 5 reviewed false accepts, 3 distinct known actual identities, 0 outsider/not-in-mapping accepts
- 24011CSEAI0103: 4 reviewed false accepts, 0 distinct known actual identities, 4 outsider/not-in-mapping accepts
- 2401100CSE0074: 3 reviewed false accepts, 3 distinct known actual identities, 0 outsider/not-in-mapping accepts
- 2401100CSE0121: 3 reviewed false accepts, 3 distinct known actual identities, 0 outsider/not-in-mapping accepts
- 2401100CSE0044: 3 reviewed false accepts, 3 distinct known actual identities, 0 outsider/not-in-mapping accepts
- 2401100CSE0124: 2 reviewed false accepts, 1 distinct known actual identities, 1 outsider/not-in-mapping accepts
- 2401100CSE0019: 1 reviewed false accepts, 1 distinct known actual identities, 0 outsider/not-in-mapping accepts
- 2401100CSE0060: 1 reviewed false accepts, 1 distinct known actual identities, 0 outsider/not-in-mapping accepts
- 2401100CSE0110: 1 reviewed false accepts, 1 distinct known actual identities, 0 outsider/not-in-mapping accepts

These are reviewed-evidence investigation flags, not automatic blacklists or proof of bad enrollment.

## Enrollment and embedding audit

- Embedding records: 355
- Canonical identities in embedding database: 26
- Dataset images audited: 2170
- Suspect enrollment images requiring review: 855
- Candidate CCTV crops requiring review: 98

## Missing embeddings

- 2401100CSE0268: dataset_missing
- 24011CSEAI0110: dataset_missing

## Human actions required

1. Complete every item in the enrollment-image reviewer and return its exported approval CSV.
2. Complete every item in the CCTV-crop reviewer and return its exported approval CSV.
3. Do not run a versioned embedding build until both CSVs pass validation.
4. Preserve MON_P3 for untouched validation before any future promotion.

## Review packages

- Enrollment audit: `F:\sohail\Class_Attendance_YuNet_SFace\attendance_output\embedding_forensics\embedding_forensics_reviewfix_20260713_135128_791820\suspicious_enrollment_image_review`
- Verified CCTV proposals: `F:\sohail\Class_Attendance_YuNet_SFace\attendance_output\embedding_forensics\embedding_forensics_reviewfix_20260713_135128_791820\verified_cctv_enrollment_review`
