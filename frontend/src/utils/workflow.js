export const RUNNING_JOB_STATUSES = new Set(['Pending', 'Processing']);

export function normalizePeriod(period) {
  const raw = String(period || '').trim().toUpperCase();
  if (!raw) return '';
  return raw.startsWith('P') ? raw : `P${raw}`;
}

export function isRunningJob(job) {
  return RUNNING_JOB_STATUSES.has(String(job?.status || ''));
}

export function jobMatchesRow(job, row) {
  if (!job || !row) return false;
  if (row.session_id && job.session_id) return String(job.session_id) === String(row.session_id);
  const rowSubject = String(row.subject_track || row.subject || '').toUpperCase();
  const jobSubject = String(job.subject_abbr || job.subject || '').toUpperCase();
  return String(job.day || '') === String(row.day || '')
    && normalizePeriod(job.period) === normalizePeriod(row.period)
    && (!jobSubject || !rowSubject || jobSubject === rowSubject);
}

export function latestJobForRow(jobs, row) {
  const matches = (Array.isArray(jobs) ? jobs : []).filter((job) => jobMatchesRow(job, row));
  matches.sort((a, b) => {
    const timeCompare = String(b.updated_at || b.created_at || '').localeCompare(String(a.updated_at || a.created_at || ''));
    return timeCompare || String(b.job_id || '').localeCompare(String(a.job_id || ''));
  });
  return matches[0] || null;
}

export function needsQualityReview(entry) {
  if (!entry || typeof entry !== 'object') return false;
  const review = String(entry.requires_manual_review ?? entry.Requires_Manual_Review ?? '').toLowerCase();
  const finalized = String(entry.attendance_finalized ?? entry.Attendance_Finalized ?? '').toLowerCase();
  return entry.status === 'Needs Review'
    || review === 'true'
    || review === 'yes'
    || finalized === 'false'
    || finalized === 'no';
}

export function deriveSlotState(row, job) {
  if (/lunch/i.test(String(row?.subject || ''))) return 'Lunch';
  if (isRunningJob(job)) return job.status;
  if (needsQualityReview(row?.att_status)) return 'Needs Review';
  return row?.att_status?.status || job?.status || 'Pending';
}

export function actionForSlot({ row, job, backendReady }) {
  const state = deriveSlotState(row, job);
  if (state === 'Lunch') return { primary: 'none', disabled: true, state };
  if (isRunningJob(job)) return { primary: 'cancel', disabled: !backendReady, state };
  if (state === 'Needs Review') return { primary: 'review', secondary: 'reprocess', disabled: !backendReady, state };
  if (state === 'Completed') return { primary: 'reprocess', disabled: !backendReady, state };
  if (state === 'Failed' || state === 'Cancelled') return { primary: 'retry', disabled: !backendReady, state };
  return { primary: 'process', disabled: !backendReady, state };
}

export function elapsedFromText(value, now = Date.now()) {
  if (!value) return '—';
  const normalized = String(value).replace(/(\d{2})-(\d{2})-(\d{4})/, '$3-$2-$1');
  const parsed = Date.parse(normalized);
  if (Number.isNaN(parsed)) return 'Running';
  const seconds = Math.max(0, Math.floor((now - parsed) / 1000));
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}:${String(secs).padStart(2, '0')}`;
}
