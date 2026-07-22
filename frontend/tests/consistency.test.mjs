import assert from 'node:assert/strict';
import test from 'node:test';
import { SUBJECT_STUDENTS, ALL_STUDENTS, getRollFromRow, normalizeRoll } from '../src/data/students.js';
import {
  ATTENDANCE_STATUS_CONTRACT_VERSION,
  ATTENDANCE_STATUS_LABELS,
  attendanceStatusCategory,
  classifyReviewRows,
  reportSubjects,
  rowMatchesReviewMode,
  validateReportScope,
} from '../src/utils/consistency.js';

const cvoFaculty = { role: 'faculty', subjects: ['CVO'] };
const admin = { role: 'admin', subjects: ['SWE', 'CCM', 'CVO'], canSeeAll: true };

test('bundled CVO fallback matches the authoritative 27-student roster', () => {
  const rolls = SUBJECT_STUDENTS.CVO.map((student) => student.roll);
  assert.equal(rolls.length, 27);
  assert.ok(rolls.includes('2401100CSE0237'));
  assert.ok(rolls.includes('24011CSEAI0110'));
  assert.ok(!rolls.includes('24011CSEAI0061'));

  const siri = ALL_STUDENTS.find((student) => student.roll === '2401100CSE0237');
  const vivek = ALL_STUDENTS.find((student) => student.roll === '24011CSEAI0061');
  assert.deepEqual(siri.subjects, ['CVO']);
  assert.deepEqual(vivek.subjects, ['SWE']);
});

test('annotated attendance rolls are canonicalized without merging distinct roll numbers', () => {
  assert.equal(getRollFromRow({ Roll_Number: '2401100CSE0016 (A) EESHA' }), '2401100CSE0016');
  assert.equal(normalizeRoll('2401100CSE0110'), '2401100CSE0110');
  assert.equal(normalizeRoll('24011CSEAI0110'), '24011CSEAI0110');
});

test('report subject discovery is deterministic', () => {
  assert.deepEqual(
    reportSubjects([{ Course_Abbr: 'CVO' }], [{ Subject_Track: 'CCM/CVO' }]),
    ['CCM', 'CVO'],
  );
});

test('faculty cannot load an out-of-scope demo or report', () => {
  const result = validateReportScope(cvoFaculty, [], [{ Course_Abbr: 'SWE' }]);
  assert.equal(result.allowed, false);
  assert.match(result.reason, /SWE/);
  assert.match(result.reason, /CVO/);
});

test('faculty can load assigned-subject reports and HOD can load all', () => {
  assert.equal(validateReportScope(cvoFaculty, [{ Course_Abbr: 'CVO' }], []).allowed, true);
  assert.equal(validateReportScope(admin, [], [{ Course_Abbr: 'SWE' }]).allowed, true);
});

test('mixed logical subject reports are blocked for faculty', () => {
  const result = validateReportScope(cvoFaculty, [], [{ Course_Abbr: 'CCM/CVO' }]);
  assert.equal(result.allowed, false);
  assert.match(result.reason, /Mixed or out-of-scope/);
});

test('review, present, and absent counts are mutually exclusive', () => {
  const rows = [
    { Roll_Number: '1', Status: 'Absent', Flags: 'Single Camera Evidence', Detection_Count: 5 },
    { Roll_Number: '2', Status: 'Absent', Flags: '', Detection_Count: 0 },
    { Roll_Number: '3', Status: 'Present', Flags: '', Detection_Count: 3 },
  ];
  const helpers = {
    statusOf: (row, changes) => changes[row.Roll_Number]?.status || row.Status,
    isPresent: (status) => /present/i.test(status),
    isEvidenceReview: (row) => Boolean(row.Flags) || (/absent/i.test(row.Status) && row.Detection_Count > 0),
  };

  const summary = classifyReviewRows(rows, {}, helpers);
  assert.deepEqual(summary, {
    present: 1,
    review: 1,
    unconfirmed: 0,
    missingEnrollment: 0,
    absent: 1,
    unknown: 0,
    unresolved: 1,
    total: 3,
    pct: 33.3,
  });
  assert.equal(summary.present + summary.review + summary.unconfirmed + summary.missingEnrollment + summary.absent + summary.unknown, summary.total);

  assert.deepEqual(
    ['review', 'present', 'absent'].map((mode) => rows.filter((row) => rowMatchesReviewMode(row, mode, {}, helpers)).length),
    [1, 1, 1],
  );
});

test('an explicit manual decision resolves a flagged review row', () => {
  const row = { Roll_Number: '1', Status: 'Absent', Flags: 'Single Camera Evidence', Detection_Count: 5 };
  const changes = { 1: { status: 'Present' } };
  const helpers = {
    statusOf: (item, current) => current[item.Roll_Number]?.status || item.Status,
    isPresent: (status) => /present/i.test(status),
    isEvidenceReview: (item) => Boolean(item.Flags),
  };
  assert.deepEqual(classifyReviewRows([row], changes, helpers), {
    present: 1,
    review: 0,
    unconfirmed: 0,
    missingEnrollment: 0,
    absent: 0,
    unknown: 0,
    unresolved: 0,
    total: 1,
    pct: 100,
  });
});

test('a saved manual override remains resolved after reload', () => {
  const row = {
    Roll_Number: '1',
    Status: 'Absent',
    Flags: 'Single Camera Evidence',
    Detection_Count: 5,
    Manual_Override: true,
  };
  const helpers = {
    statusOf: (item) => item.Status,
    isPresent: (status) => /present/i.test(status),
    isEvidenceReview: (item) => Boolean(item.Flags),
  };
  assert.deepEqual(classifyReviewRows([row], {}, helpers), {
    present: 0,
    review: 0,
    unconfirmed: 0,
    missingEnrollment: 0,
    absent: 1,
    unknown: 0,
    unresolved: 0,
    total: 1,
    pct: 0,
  });
});

test('compact summary covers every Phase 2L status and unresolved is additive', () => {
  assert.equal(ATTENDANCE_STATUS_CONTRACT_VERSION, 'product-phase-2l-frontend-status-contract-v1');
  assert.deepEqual(Object.values(ATTENDANCE_STATUS_LABELS), [
    'Present', 'Needs Review', 'Unconfirmed', 'Missing Enrollment', 'Absent', 'Unknown',
  ]);
  const rows = [
    { Roll_Number: '1', Status: 'Present' },
    { Roll_Number: '2', Status: 'Needs Review' },
    { Roll_Number: '3', Status: 'Unconfirmed' },
    { Roll_Number: '4', Status: 'Missing Enrollment' },
    { Roll_Number: '5', Status: 'Absent' },
    { Roll_Number: '6', Status: 'Legacy Maybe' },
  ];
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

test('Unconfirmed, Missing Enrollment, and Unknown remain distinct unresolved categories', () => {
  assert.equal(attendanceStatusCategory('Unconfirmed'), 'unconfirmed');
  assert.equal(attendanceStatusCategory('Missing Enrollment'), 'missingEnrollment');
  assert.equal(attendanceStatusCategory('legacy-value'), 'unknown');
  const rows = [
    { Roll_Number: '1', Status: 'Unconfirmed' },
    { Roll_Number: '2', Status: 'Missing Enrollment' },
    { Roll_Number: '3', Status: 'legacy-value' },
  ];
  assert.deepEqual(
    ['unconfirmed', 'missing', 'unknown'].map((mode) => rows.filter((row) => rowMatchesReviewMode(row, mode)).length),
    [1, 1, 1],
  );
  assert.equal(rows.filter((row) => rowMatchesReviewMode(row, 'review')).length, 3);
});

test('evidence-flagged explicit Absent fails closed until manually resolved', () => {
  const row = { Roll_Number: '1', Status: 'Absent', Flags: 'Single Camera Evidence', Detection_Count: 5 };
  const helpers = {
    statusOf: (item, current) => current[item.Roll_Number]?.status || item.Status,
    isEvidenceReview: (item) => Boolean(item.Flags) || Number(item.Detection_Count || 0) > 0,
  };
  assert.equal(classifyReviewRows([row], {}, helpers).review, 1);
  assert.equal(rowMatchesReviewMode(row, 'absent', {}, helpers), false);
  assert.equal(rowMatchesReviewMode(row, 'review', {}, helpers), true);
  assert.equal(classifyReviewRows([row], { 1: { status: 'Absent' } }, helpers).absent, 1);
});
