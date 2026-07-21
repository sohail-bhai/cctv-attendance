# Product

## Register

product

## Users

Computer-vision engineers and authorized human reviewers investigating CCTV attendance identity quality. They need to inspect evidence, make explicit decisions, and preserve a defensible chain of provenance without changing live attendance or enrollment data.

## Product Purpose

The project provides a conservative CCTV attendance pipeline and offline reliability tooling around YuNet, SFace, camera zones, tracklets, blind review, and regression evidence. Success means identity decisions remain reviewable, reproducible, and fail closed whenever evidence is incomplete or unsafe.

## Brand Personality

Forensic, restrained, accountable.

## Anti-references

Do not present model similarity as ground truth, hide provenance, imply certainty from weak evidence, auto-modify enrollment data, or use decorative UI that competes with review work. Do not expose private model predictions in blind-review surfaces.

## Design Principles

1. Evidence before action: show the source, integrity state, and reason for review.
2. Human decisions stay explicit: never convert a prediction into a label or approval.
3. Safety state is visible: incomplete, uncertain, and rejected states must be unmistakable.
4. Familiar controls win: use keyboard-accessible native form controls and offline-safe behavior.
5. Preserve provenance: exported approvals must map back to immutable source hashes.

## Accessibility & Inclusion

Target WCAG 2.1 AA for generated reviewer interfaces. Support keyboard-only review, visible focus, readable contrast, narrow viewports, reduced motion, and status cues that do not rely on color alone.
