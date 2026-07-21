import assert from 'node:assert/strict';
import test from 'node:test';
import { SUBJECT_STUDENTS, ALL_STUDENTS, getRollFromRow, normalizeRoll } from '../src/data/students.js';
import {
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
  assert.deepEqual(summary, { present: 1, review: 1, absent: 1, total: 3, pct: 33.3 });
  assert.equal(summary.present + summary.review + summary.absent, summary.total);

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
    absent: 0,
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
    absent: 1,
    total: 1,
    pct: 0,
  });
});
