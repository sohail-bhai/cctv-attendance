import assert from 'node:assert/strict';
import test from 'node:test';
import {
  correctionReasonErrors,
  entryToSlotSummaryRow,
  evidencePresentation,
  reportRevisionPresentation,
  reviewActionState,
  sessionStateLabel,
  sortAttendanceSessions,
} from '../src/utils/reportWorkflow.js';

test('session labels distinguish review readiness and finalization', () => {
  assert.equal(sessionStateLabel({ review_state: 'needs_attention', unresolved_count: 2 }), 'Needs Review');
  assert.equal(sessionStateLabel({ review_state: 'roster_mismatch', roster_complete: false }), 'Roster Mismatch');
  assert.equal(sessionStateLabel({ review_state: 'ready_to_finalize', unresolved_count: 0 }), 'Ready to Finalize');
  assert.equal(sessionStateLabel({ attendance_finalized: true }), 'Finalized');
});

test('saved sessions sort newest first deterministically', () => {
  const sessions = sortAttendanceSessions([
    { session_id: 'A', session_date: '2026-06-22', period: 'P4' },
    { session_id: 'B', session_date: '2026-06-30', period: 'P1' },
  ]);
  assert.deepEqual(sessions.map((item) => item.session_id), ['B', 'A']);
});

test('backend entry maps to report summary fields', () => {
  const row = entryToSlotSummaryRow({
    subject_abbr: 'CVO',
    course_name: 'Computer Vision through OpenCV',
    day: 'Tuesday',
    period: 'P1',
    attendance_finalized: false,
    requires_manual_review: true,
  });
  assert.equal(row.Course_Abbr, 'CVO');
  assert.equal(row.Period, '1');
  assert.equal(row.Attendance_Finalized, false);
  assert.equal(row.Requires_Manual_Review, true);
});

test('correction reasons are required except for clear actions', () => {
  assert.deepEqual(correctionReasonErrors({ A: { status: 'Present', reason: '' }, B: { status: 'CLEAR', reason: '' } }), ['A']);
});

test('review action state blocks unsafe export and finalization', () => {
  assert.deepEqual(reviewActionState({ pendingChanges: 1, unresolvedCount: 0, finalized: false, busy: false }), {
    canSave: true,
    canExport: false,
    canFinalize: false,
    finalized: false,
  });
  assert.equal(reviewActionState({ pendingChanges: 0, unresolvedCount: 0, finalized: false }).canFinalize, true);
  assert.equal(reviewActionState({ pendingChanges: 0, unresolvedCount: 0, rosterComplete: false, finalized: false }).canFinalize, false);
  assert.equal(reviewActionState({ pendingChanges: 0, unresolvedCount: 0, finalized: true }).canFinalize, false);
});

test('evidence display distinguishes strict, guarded, reviewed, mixed, observations, and carry-forward', () => {
  const evidence = evidencePresentation({
    Strict_Recognized_Checkpoints: 'CP1; CP2',
    Guarded_Recovery_Candidate_Checkpoints: 'CP3',
    Reviewed_Tracklet_Checkpoints: 'CP4',
    Mixed_Track_Checkpoints_Rejected: 'CP5',
    Total_Accepted_Detections: 9,
    Automatic_Guarded_Recovery_Enabled: 'No',
  }, {
    review_carry_forward_applied: true,
    review_carry_forward_source: 'review-registry-demo',
  });
  assert.equal(evidence.strict, 'CP1; CP2');
  assert.equal(evidence.guarded, 'CP3');
  assert.equal(evidence.reviewed, 'CP4');
  assert.equal(evidence.mixed, 'CP5');
  assert.equal(evidence.observations, 9);
  assert.equal(evidence.authority, 'Human-reviewed tracklet evidence');
  assert.equal(evidence.guardedRecoveryAutomatic, false);
  assert.match(evidence.carryForward, /Applied.*review-registry-demo/);
});

test('revision display keeps official, automatic, historical, carry-forward, and finalization states separate', () => {
  const revision = reportRevisionPresentation({
    authority_revision_id: 'official-1',
    official_recognition_authority: 'reviewed_multiframe_tracklet_evidence',
    automatic_recognition_authority: 'strict_tracklet_aggregate_with_guarded_review_candidates',
    pending_candidate_revision: { revision_id: 'candidate-1' },
    source_report_superseded: true,
    review_carry_forward_applied: true,
    unresolved_count: 21,
    attendance_finalized: false,
  });
  assert.equal(revision.officialLabel, 'Official Reviewed Revision');
  assert.equal(revision.automaticLabel, 'Automatic Candidate Revision');
  assert.equal(revision.automaticRevision, 'candidate-1');
  assert.equal(revision.historicalVisible, true);
  assert.match(revision.historicalLabel, /Superseded/);
  assert.match(revision.carryForward, /Applied/);
  assert.equal(revision.finalization, 'Not finalized; 21 unresolved');
});
