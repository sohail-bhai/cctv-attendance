import { canAccessSubject, isAdmin } from '../data/users.js';

export const ATTENDANCE_STATUS_CONTRACT_VERSION = 'product-phase-2l-frontend-status-contract-v1';

export const ATTENDANCE_STATUS_LABELS = Object.freeze({
  present: 'Present',
  review: 'Needs Review',
  unconfirmed: 'Unconfirmed',
  missingEnrollment: 'Missing Enrollment',
  absent: 'Absent',
  unknown: 'Unknown',
});

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

export function attendanceStatusCategory(status) {
  const value = clean(status).toLowerCase();
  if (value.startsWith('present') || value === 'yes') return 'present';
  if (value.includes('missing enrollment')) return 'missingEnrollment';
  if (value.includes('unconfirmed')) return 'unconfirmed';
  if (value.includes('review')) return 'review';
  if (value === 'absent' || value === 'no') return 'absent';
  return 'unknown';
}

function resolvedCategory(row, changes, statusOf) {
  const roll = row?.Roll_Number;
  const explicit = changes?.[roll]?.status;
  const status = statusOf(row, changes);
  const category = attendanceStatusCategory(status);
  const manuallyResolved = explicit === 'Present' || explicit === 'Absent';
  const savedResolved = Boolean(row?.Manual_Override) && (category === 'present' || category === 'absent');
  return { explicit, status, category, manuallyResolved, savedResolved };
}

function reviewCategory(row, changes, statusOf, isEvidenceReview) {
  const state = resolvedCategory(row, changes, statusOf);
  const unsafeEvidence = isEvidenceReview(row);
  const otherwiseResolved = ['present', 'absent', 'unknown'].includes(state.category);
  const forcedReview = !state.manuallyResolved
    && !state.savedResolved
    && otherwiseResolved
    && unsafeEvidence;
  return forcedReview ? 'review' : state.category;
}

export function classifyReviewRows(rows = [], changes = {}, helpers = {}) {
  const statusOf = helpers.statusOf || ((row) => row?.Status || row?.Final_Status || 'Unknown');
  const isEvidenceReview = helpers.isEvidenceReview || (() => false);

  const counts = {
    present: 0,
    review: 0,
    unconfirmed: 0,
    missingEnrollment: 0,
    absent: 0,
    unknown: 0,
  };

  rows.forEach((row) => {
    const category = reviewCategory(row, changes, statusOf, isEvidenceReview);
    if (Object.hasOwn(counts, category)) counts[category] += 1;
    else counts.unknown += 1;
  });

  const unresolved = counts.review + counts.unconfirmed + counts.missingEnrollment + counts.unknown;
  return {
    ...counts,
    unresolved,
    total: rows.length,
    pct: rows.length ? Math.round((counts.present / rows.length) * 1000) / 10 : 0,
  };
}

export function rowMatchesReviewMode(row, mode, changes = {}, helpers = {}) {
  const statusOf = helpers.statusOf || ((item) => item?.Status || item?.Final_Status || 'Unknown');
  const isEvidenceReview = helpers.isEvidenceReview || (() => false);
  const category = reviewCategory(row, changes, statusOf, isEvidenceReview);
  const unresolved = ['review', 'unconfirmed', 'missingEnrollment', 'unknown'].includes(category);

  if (mode === 'review') return unresolved;
  if (mode === 'present') return category === 'present';
  if (mode === 'needs_review') return category === 'review';
  if (mode === 'unconfirmed') return category === 'unconfirmed';
  if (mode === 'missing') return category === 'missingEnrollment';
  if (mode === 'absent') return category === 'absent';
  if (mode === 'unknown') return category === 'unknown';
  return true;
}
