import assert from 'node:assert/strict';
import test from 'node:test';
import {
  correctionReasonErrors,
  entryToSlotSummaryRow,
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
