import test from 'node:test';
import assert from 'node:assert/strict';
import {
  actionForSlot,
  deriveSlotState,
  elapsedFromText,
  isRunningJob,
  jobMatchesRow,
  latestJobForRow,
  normalizePeriod,
} from '../src/utils/workflow.js';

test('normalizePeriod is deterministic', () => {
  assert.equal(normalizePeriod('1'), 'P1');
  assert.equal(normalizePeriod('p2'), 'P2');
  assert.equal(normalizePeriod(''), '');
});

test('job matching prefers canonical session identity', () => {
  const row = { session_id: '2026-06-22__B51__P4__CVO', day: 'Monday', period: 'P4', subject: 'CVO' };
  assert.equal(jobMatchesRow({ session_id: row.session_id }, row), true);
  assert.equal(jobMatchesRow({ session_id: 'different' }, row), false);
});

test('fallback job matching remains course scoped', () => {
  const row = { day: 'Monday', period: '4', subject_track: 'CVO' };
  assert.equal(jobMatchesRow({ day: 'Monday', period: 'P4', subject_abbr: 'CVO' }, row), true);
  assert.equal(jobMatchesRow({ day: 'Monday', period: 'P4', subject_abbr: 'CCM' }, row), false);
});

test('latestJobForRow selects the newest matching job', () => {
  const row = { session_id: 'S1' };
  const result = latestJobForRow([
    { job_id: 'A', session_id: 'S1', updated_at: '2026-07-18 10:00:00' },
    { job_id: 'B', session_id: 'S1', updated_at: '2026-07-18 11:00:00' },
  ], row);
  assert.equal(result.job_id, 'B');
});

test('slot state prioritizes active job and review safety', () => {
  assert.equal(deriveSlotState({ subject: 'CVO', att_status: { status: 'Completed' } }, { status: 'Processing' }), 'Processing');
  assert.equal(deriveSlotState({ subject: 'CVO', att_status: { status: 'Needs Review' } }, null), 'Needs Review');
  assert.equal(deriveSlotState({ subject: 'Lunch' }, null), 'Lunch');
});

test('slot actions never process a needs-review session implicitly', () => {
  const row = { subject: 'CVO', att_status: { status: 'Needs Review' } };
  assert.deepEqual(actionForSlot({ row, job: null, backendReady: true }), {
    primary: 'review', secondary: 'reprocess', disabled: false, state: 'Needs Review',
  });
});

test('mutating actions are disabled when backend is not ready', () => {
  const row = { subject: 'CVO', att_status: { status: 'Pending' } };
  assert.equal(actionForSlot({ row, job: null, backendReady: false }).disabled, true);
  assert.equal(actionForSlot({ row, job: { status: 'Processing' }, backendReady: false }).primary, 'cancel');
});

test('running job and elapsed helpers are stable', () => {
  assert.equal(isRunningJob({ status: 'Pending' }), true);
  assert.equal(isRunningJob({ status: 'Completed' }), false);
  assert.equal(elapsedFromText('2026-07-18T10:00:00Z', Date.parse('2026-07-18T10:02:05Z')), '2:05');
});
