import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import {
  ATTENDANCE_STATUS_LABELS,
  classifyReviewRows,
  rowMatchesReviewMode,
  validateReportScope,
} from '../src/utils/consistency.js';
import { getStatus, parseCsv } from '../src/utils/csv.js';

const frontendRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const repoRoot = path.resolve(frontendRoot, '..');

async function source(relative) {
  return readFile(path.join(frontendRoot, relative), 'utf8');
}

test('the six status labels and unresolved formula are explicit', () => {
  assert.deepEqual(Object.values(ATTENDANCE_STATUS_LABELS), [
    'Present',
    'Needs Review',
    'Unconfirmed',
    'Missing Enrollment',
    'Absent',
    'Unknown',
  ]);
  const rows = Object.values(ATTENDANCE_STATUS_LABELS).map((Status) => ({ Roll_Number: Status, Status }));
  const summary = classifyReviewRows(rows);
  assert.deepEqual(summary, {
    present: 1,
    review: 1,
    unconfirmed: 1,
    missingEnrollment: 1,
    absent: 1,
    unknown: 1,
    unresolved: 4,
    total: 6,
    pct: 16.7,
  });
});

test('review filters are coherent and unknown fails closed', () => {
  const row = { Roll_Number: 'U1', Status: 'future status' };
  assert.equal(rowMatchesReviewMode(row, 'review'), true);
  assert.equal(rowMatchesReviewMode(row, 'unknown'), true);
  assert.equal(rowMatchesReviewMode(row, 'present'), false);
  assert.equal(rowMatchesReviewMode(row, 'absent'), false);
});

test('evidence-flagged explicit absence remains unresolved until manual resolution', () => {
  const row = { Roll_Number: 'A1', Status: 'Absent', Flags: 'guarded_candidate' };
  const helpers = { isEvidenceReview: (item) => Boolean(item.Flags) };
  assert.equal(classifyReviewRows([row], {}, helpers).review, 1);
  assert.equal(classifyReviewRows([row], { A1: { status: 'Absent' } }, helpers).absent, 1);
});

test('preserved official and automatic reports show exact current totals', async () => {
  const officialPath = path.join(
    repoRoot,
    'attendance_output/product_workflow/phase_2i_authority/authority-revision-2d83c4f679f839a6c784c636/corrected_attendance_2026-06-22__B51__P4__CVO.csv',
  );
  const candidatePath = path.join(
    repoRoot,
    'attendance_output/product_workflow/report_revisions/2026-06-22__B51__P4__CVO/automatic-candidate-99cc13934144bf7e1706b8be/candidate_attendance.csv',
  );
  const [official, candidate] = await Promise.all([
    readFile(officialPath, 'utf8').then(parseCsv),
    readFile(candidatePath, 'utf8').then(parseCsv),
  ]);
  const officialCounts = classifyReviewRows(official, {}, { statusOf: getStatus });
  const candidateCounts = classifyReviewRows(candidate, {}, { statusOf: getStatus });
  assert.deepEqual(
    { present: officialCounts.present, review: officialCounts.review, unconfirmed: officialCounts.unconfirmed, missing: officialCounts.missingEnrollment, absent: officialCounts.absent, unknown: officialCounts.unknown, unresolved: officialCounts.unresolved, total: officialCounts.total },
    { present: 6, review: 4, unconfirmed: 16, missing: 1, absent: 0, unknown: 0, unresolved: 21, total: 27 },
  );
  assert.deepEqual(
    { present: candidateCounts.present, review: candidateCounts.review, unconfirmed: candidateCounts.unconfirmed, missing: candidateCounts.missingEnrollment, absent: candidateCounts.absent, unknown: candidateCounts.unknown, unresolved: candidateCounts.unresolved, total: candidateCounts.total },
    { present: 2, review: 8, unconfirmed: 16, missing: 1, absent: 0, unknown: 0, unresolved: 25, total: 27 },
  );
});

test('Reports and Review Students render revision, evidence, and finalization distinctions', async () => {
  const reports = await source('src/pages/Reports.jsx');
  const review = await source('src/pages/ManualReview.jsx');
  const workflow = await source('src/utils/reportWorkflow.js');
  for (const text of [reports, review]) {
    assert.match(text, /reportRevisionPresentation/);
    assert.match(text, /Carry-forward/);
    assert.match(text, /Unresolved total/);
    assert.match(text, /Unknown/);
  }
  assert.match(workflow, /Official Reviewed Revision/);
  assert.match(workflow, /Automatic Candidate Revision/);
  assert.match(workflow, /historical/i);
  assert.match(workflow, /Not finalized/);
  assert.match(reports, /Strict: accepted checkpoints/);
  assert.match(reports, /Guarded recovery candidate checkpoints/);
  assert.match(reports, /Human-reviewed checkpoints/);
  assert.match(reports, /Mixed track rejected checkpoints/);
  assert.match(review, /guarded recovery candidates/i);
  assert.match(review, /review-only/i);
  assert.match(review, /Finalize Attendance/);
});

test('help, settings, and fallback copy preserve the current safety boundary', async () => {
  const copy = (await Promise.all([
    source('src/pages/Help.jsx'),
    source('src/pages/Settings.jsx'),
    source('src/data/fallback.js'),
  ])).join('\n');
  assert.match(copy, /CP1/i);
  assert.match(copy, /CP5/i);
  assert.match(copy, /front/i);
  assert.match(copy, /back/i);
  assert.match(copy, /review-only/i);
  assert.match(copy, /insufficient camera evidence/i);
  assert.doesNotMatch(copy, /three checkpoints|four checkpoints|3\s*\/\s*4/i);
  assert.doesNotMatch(copy, /guarded recovery.{0,40}(official|automatic attendance)/i);
});

test('faculty report scope remains subject-limited while HOD scope remains global', () => {
  const faculty = { role: 'faculty', subjects: ['CVO'], canSeeAll: false };
  const hod = { role: 'admin', subjects: ['SWE', 'CCM', 'CVO'], canSeeAll: true };
  assert.equal(validateReportScope(faculty, [{ Course_Abbr: 'CVO' }]).allowed, true);
  assert.equal(validateReportScope(faculty, [{ Course_Abbr: 'SWE' }]).allowed, false);
  assert.equal(validateReportScope(hod, [{ Course_Abbr: 'SWE/CCM' }]).allowed, true);
});

test('login does not embed, display, or prefill credentials', async () => {
  const login = await source('src/pages/Login.jsx');
  const users = await source('src/data/users.js');
  assert.match(login, /existing authorized faculty or HOD account/);
  assert.match(login, /useState\(''\)/);
  assert.doesNotMatch(login, /Demo accounts|quickFill|user\.password|user\.username/);
  assert.doesNotMatch(users, /"password"\s*:|"username"\s*:/);
});
