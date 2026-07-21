import { canAccessSubject, isAdmin } from '../data/users.js';

function clean(value) {
  return String(value ?? '').trim();
}

function splitSubjectValue(value) {
  return clean(value)
    .toUpperCase()
    .replace(/-/g, '/')
    .split(/[\/,;+&|]+/)
    .map((item) => item.replace(/\b(LAB|THEORY)\b/g, '').trim())
    .filter(Boolean);
}

export function reportSubjects(attendanceRows = [], summaryRows = []) {
  const subjects = new Set();
  const fields = [
    'Course_Abbr', 'course_abbr', 'Subject', 'subject', 'Course', 'course',
    'Subject_Track', 'subject_track',
  ];

  [...summaryRows, ...attendanceRows].forEach((row) => {
    fields.forEach((field) => splitSubjectValue(row?.[field]).forEach((subject) => subjects.add(subject)));
  });

  return Array.from(subjects).sort();
}

export function validateReportScope(user, attendanceRows = [], summaryRows = []) {
  const subjects = reportSubjects(attendanceRows, summaryRows);
  if (isAdmin(user)) return { allowed: true, subjects, reason: '' };
  if (!subjects.length) {
    return {
      allowed: true,
      subjects,
      reason: 'This report does not declare a subject. Student-roster filtering will still be applied.',
    };
  }

  const deniedSubjects = subjects.filter((subject) => !canAccessSubject(user, subject));
  if (!deniedSubjects.length) return { allowed: true, subjects, reason: '' };

  const assigned = (user?.subjects || []).map((subject) => String(subject).toUpperCase());
  return {
    allowed: false,
    subjects,
    reason: `This report declares ${subjects.join(', ')}, but this login is assigned to ${assigned.join(', ') || 'no subjects'}. Mixed or out-of-scope subject reports cannot be opened.`,
  };
}

export function classifyReviewRows(rows = [], changes = {}, helpers = {}) {
  const statusOf = helpers.statusOf || ((row) => row?.Status || row?.Final_Status || 'Absent');
  const isPresent = helpers.isPresent || ((status) => /present/i.test(String(status || '')));
  const isEvidenceReview = helpers.isEvidenceReview || (() => false);

  let present = 0;
  let review = 0;
  let absent = 0;

  rows.forEach((row) => {
    const roll = row?.Roll_Number;
    const explicit = changes?.[roll]?.status;
    const status = statusOf(row, changes);
    const manuallyResolved = explicit === 'Present' || explicit === 'Absent';
    const savedResolved = Boolean(row?.Manual_Override) && (/present|absent/i.test(String(status || '')));
    const needsReview = !manuallyResolved && !savedResolved && (/review/i.test(String(status || '')) || isEvidenceReview(row));

    if (needsReview) review += 1;
    else if (isPresent(status)) present += 1;
    else absent += 1;
  });

  return {
    present,
    review,
    absent,
    total: rows.length,
    pct: rows.length ? Math.round((present / rows.length) * 1000) / 10 : 0,
  };
}

export function rowMatchesReviewMode(row, mode, changes = {}, helpers = {}) {
  const statusOf = helpers.statusOf || ((item) => item?.Status || item?.Final_Status || 'Absent');
  const isPresent = helpers.isPresent || ((status) => /present/i.test(String(status || '')));
  const isEvidenceReview = helpers.isEvidenceReview || (() => false);
  const roll = row?.Roll_Number;
  const explicit = changes?.[roll]?.status;
  const status = statusOf(row, changes);
  const manuallyResolved = explicit === 'Present' || explicit === 'Absent';
  const savedResolved = Boolean(row?.Manual_Override) && (/present|absent/i.test(String(status || '')));
  const needsReview = !manuallyResolved && !savedResolved && (/review/i.test(String(status || '')) || isEvidenceReview(row));

  if (mode === 'review') return needsReview;
  if (mode === 'present') return !needsReview && isPresent(status);
  if (mode === 'absent') return !needsReview && !isPresent(status);
  return true;
}
