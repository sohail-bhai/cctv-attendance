export const MIN_REASON_LENGTH = 8;
export const TIMETABLE_CONFIRMATION = 'REPLACE TIMETABLE';

export function normalizeFacultyForm(form = {}) {
  return {
    action: form.action === 'update' ? 'update' : 'create',
    faculty_id: String(form.faculty_id || '').trim().toLowerCase(),
    username: String(form.username || '').trim(),
    name: String(form.name || '').trim(),
    password: String(form.password || ''),
    reason: String(form.reason || '').trim(),
  };
}

export function facultyFormError(form = {}) {
  const value = normalizeFacultyForm(form);
  if (!/^[a-z][a-z0-9._-]{2,63}$/.test(value.faculty_id)) {
    return 'Faculty ID must start with a letter and use only letters, numbers, dot, underscore, or hyphen.';
  }
  if (!/^[a-zA-Z0-9._-]{3,64}$/.test(value.username)) {
    return 'Username must contain 3–64 letters, numbers, dot, underscore, or hyphen.';
  }
  if (!value.name || value.name.length > 120) return 'Faculty name is required.';
  if (value.action === 'create' && value.password.length < 8) {
    return 'New faculty passwords must contain at least 8 characters.';
  }
  if (value.password && value.password.length < 8) {
    return 'A replacement password must contain at least 8 characters.';
  }
  if (value.reason.length < MIN_REASON_LENGTH) {
    return 'Enter a reason of at least 8 characters.';
  }
  return '';
}

export function subjectAssignmentError({ subject, faculty_id, reason } = {}) {
  if (!String(subject || '').trim()) return 'Choose a subject.';
  if (!String(faculty_id || '').trim()) return 'Choose a faculty account.';
  if (String(reason || '').trim().length < MIN_REASON_LENGTH) {
    return 'Enter a reason of at least 8 characters.';
  }
  return '';
}

export function timetableApplyError({ preview, reason, confirmation, file } = {}) {
  if (!file) return 'Choose the same timetable CSV that was previewed.';
  if (!preview?.valid) return 'Only a valid preview can be applied.';
  if (!preview?.file_sha256) return 'Preview hash is missing. Preview the file again.';
  if (String(reason || '').trim().length < MIN_REASON_LENGTH) {
    return 'Enter a reason of at least 8 characters.';
  }
  if (String(confirmation || '') !== TIMETABLE_CONFIRMATION) {
    return `Type "${TIMETABLE_CONFIRMATION}" exactly.`;
  }
  return '';
}

export function changeTypeLabel(value) {
  const labels = {
    faculty_create: 'Faculty created',
    faculty_update: 'Faculty updated',
    subject_assignment: 'Subject reassigned',
    timetable_replace: 'Timetable replaced',
  };
  return labels[String(value || '')] || String(value || 'Configuration change').replaceAll('_', ' ');
}

export function canRollbackLatest(config) {
  return Boolean(
    config?.safety?.latest_change_rollback_available
    && config?.current_change?.status === 'applied'
    && config?.current_change?.change_id,
  );
}

export function summarizeTimetablePreview(preview) {
  if (!preview) return null;
  const days = preview.summary?.days || {};
  const subjects = preview.summary?.subjects || {};
  return {
    valid: Boolean(preview.valid),
    rows: Number(preview.row_count || 0),
    days: Object.entries(days).map(([label, count]) => `${label.slice(0, 3)} ${count}`).join(' · '),
    subjects: Object.entries(subjects).map(([label, count]) => `${label} ${count}`).join(' · '),
    warningCount: Array.isArray(preview.warnings) ? preview.warnings.length : 0,
    errorCount: Array.isArray(preview.errors) ? preview.errors.length : 0,
  };
}
