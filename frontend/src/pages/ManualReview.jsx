import { useEffect, useMemo, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import PageHeader from '../components/PageHeader.jsx';
import StatusBadge from '../components/StatusBadge.jsx';
import { apiGetResult, apiPost } from '../api/client.js';
import { DAYS } from '../data/fallback.js';
import { useAuth } from '../auth/AuthContext.jsx';
import { canAccessRow, displaySubjectForUser, isAdmin, subjectInfo } from '../data/users.js';
import { filterStudentRowsForUser } from '../data/students.js';
import { classifyReviewRows, rowMatchesReviewMode } from '../utils/consistency.js';
import { correctionReasonErrors, reviewActionState } from '../utils/reportWorkflow.js';

const S = {
  page: {
    display: 'grid',
    gap: 18,
    maxWidth: 1280,
    margin: '0 auto',
    paddingBottom: 40,
  },
  card: {
    background: '#ffffff',
    border: '1px solid #dbe6f1',
    borderRadius: 24,
    boxShadow: '0 18px 45px rgba(15, 23, 42, 0.07)',
    padding: 24,
  },
  pickerCard: {
    background: 'linear-gradient(135deg, #ffffff 0%, #f8fbff 62%, #ecfeff 100%)',
  },
  topGrid: {
    display: 'grid',
    gridTemplateColumns: 'minmax(0, 1fr) auto',
    gap: 20,
    alignItems: 'start',
  },
  eyebrow: {
    margin: '0 0 7px',
    color: '#04756f',
    textTransform: 'uppercase',
    fontSize: 12,
    fontWeight: 900,
    letterSpacing: '0.14em',
  },
  h2: {
    margin: 0,
    fontSize: 25,
    color: '#0f172a',
    lineHeight: 1.18,
    letterSpacing: '-0.035em',
  },
  h3: {
    margin: 0,
    fontSize: 21,
    color: '#0f172a',
    lineHeight: 1.2,
    letterSpacing: '-0.025em',
  },
  muted: {
    color: '#64748b',
    margin: '7px 0 0',
    lineHeight: 1.55,
    fontSize: 14,
  },
  primaryBtn: {
    border: 0,
    borderRadius: 14,
    background: 'linear-gradient(135deg, #0f8b8d, #0f7aa8)',
    color: '#ffffff',
    fontWeight: 900,
    padding: '12px 18px',
    minHeight: 46,
    boxShadow: '0 13px 26px rgba(15, 122, 168, 0.22)',
    cursor: 'pointer',
    whiteSpace: 'nowrap',
  },
  secondaryBtn: {
    border: '1px solid #d6e0ea',
    borderRadius: 13,
    background: '#ffffff',
    color: '#0f172a',
    fontWeight: 850,
    padding: '10px 14px',
    cursor: 'pointer',
    whiteSpace: 'nowrap',
  },
  dangerBtn: {
    border: '1px solid #fecaca',
    borderRadius: 12,
    background: '#fff1f2',
    color: '#be123c',
    fontWeight: 900,
    padding: '8px 11px',
    cursor: 'pointer',
  },
  warningBtn: {
    border: '1px solid #fde68a',
    borderRadius: 12,
    background: '#fffbeb',
    color: '#92400e',
    fontWeight: 900,
    padding: '8px 11px',
    cursor: 'pointer',
  },
  smallBtn: {
    border: '1px solid #d6e0ea',
    borderRadius: 12,
    background: '#ffffff',
    color: '#0f172a',
    fontWeight: 850,
    padding: '8px 11px',
    cursor: 'pointer',
  },
  fieldLabel: {
    display: 'block',
    marginBottom: 9,
    color: '#475569',
    fontWeight: 900,
    fontSize: 13,
  },
  dayRow: {
    display: 'flex',
    gap: 8,
    flexWrap: 'wrap',
  },
  dayBtn: {
    border: '1px solid #d6e0ea',
    borderRadius: 999,
    background: '#ffffff',
    color: '#334155',
    fontWeight: 900,
    padding: '10px 15px',
    minWidth: 62,
    cursor: 'pointer',
  },
  dayBtnActive: {
    border: '1px solid #0f8b8d',
    background: '#0f8b8d',
    color: '#ffffff',
    boxShadow: '0 12px 22px rgba(15, 139, 141, 0.18)',
  },
  pickerGrid: {
    display: 'grid',
    gap: 18,
    marginTop: 20,
  },
  slotGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
    gap: 12,
  },
  slotCard: {
    textAlign: 'left',
    border: '1px solid #dbe6f1',
    borderRadius: 18,
    background: '#ffffff',
    padding: '15px 16px',
    cursor: 'pointer',
    minHeight: 104,
    display: 'grid',
    gap: 6,
    boxShadow: '0 10px 26px rgba(15, 23, 42, 0.055)',
  },
  slotCardActive: {
    border: '2px solid #0f8b8d',
    background: 'linear-gradient(135deg, #ecfeff, #ffffff)',
    boxShadow: '0 16px 32px rgba(15, 139, 141, 0.16)',
  },
  slotTitle: {
    color: '#0f172a',
    fontSize: 15,
    fontWeight: 950,
  },
  slotDesc: {
    color: '#475569',
    fontSize: 13,
    lineHeight: 1.35,
  },
  slotMeta: {
    color: '#64748b',
    fontSize: 12,
    fontStyle: 'normal',
  },
  notice: {
    borderRadius: 18,
    padding: '13px 16px',
    fontWeight: 850,
    fontSize: 14,
    border: '1px solid #bfdbfe',
    background: '#eff6ff',
    color: '#1d4ed8',
  },
  noticeError: { background: '#fff1f2', borderColor: '#fecaca', color: '#be123c' },
  noticeSuccess: { background: '#ecfdf5', borderColor: '#bbf7d0', color: '#047857' },
  noticeInfo: { background: '#eff6ff', borderColor: '#bfdbfe', color: '#1d4ed8' },
  noticeWarning: { background: '#fffbeb', borderColor: '#fde68a', color: '#92400e' },
  facultyBadge: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 8,
    border: '1px solid #bae6fd',
    background: '#f0f9ff',
    color: '#0369a1',
    borderRadius: 999,
    padding: '8px 12px',
    fontSize: 13,
    fontWeight: 900,
  },
  empty: {
    display: 'flex',
    alignItems: 'center',
    gap: 15,
  },
  emptyIcon: {
    width: 54,
    height: 54,
    borderRadius: 18,
    background: '#ecfeff',
    display: 'grid',
    placeItems: 'center',
    fontSize: 24,
  },
  metricGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))',
    gap: 12,
    marginTop: 18,
  },
  metric: {
    borderRadius: 18,
    border: '1px solid #e2e8f0',
    padding: 15,
    background: '#f8fafc',
  },
  metricLabel: {
    display: 'block',
    color: '#64748b',
    fontSize: 12,
    fontWeight: 900,
    textTransform: 'uppercase',
    letterSpacing: '0.07em',
  },
  metricValue: {
    display: 'block',
    marginTop: 7,
    color: '#0f172a',
    fontSize: 27,
    fontWeight: 950,
    letterSpacing: '-0.04em',
  },
  toolbar: {
    display: 'grid',
    gridTemplateColumns: 'minmax(220px, 1fr) auto auto auto',
    gap: 10,
    alignItems: 'center',
    marginTop: 18,
  },
  input: {
    width: '100%',
    border: '1px solid #d6e0ea',
    background: '#ffffff',
    borderRadius: 14,
    padding: '11px 13px',
    fontWeight: 750,
    color: '#0f172a',
    outline: 'none',
  },
  pillSelectRow: {
    display: 'flex',
    gap: 8,
    flexWrap: 'wrap',
  },
  filterPill: {
    border: '1px solid #d6e0ea',
    background: '#ffffff',
    color: '#334155',
    borderRadius: 999,
    padding: '9px 12px',
    fontSize: 13,
    fontWeight: 900,
    cursor: 'pointer',
  },
  filterPillActive: {
    background: '#0f172a',
    borderColor: '#0f172a',
    color: '#ffffff',
  },
  tableWrap: {
    overflowX: 'auto',
    marginTop: 16,
    border: '1px solid #e2e8f0',
    borderRadius: 18,
  },
  table: {
    width: '100%',
    borderCollapse: 'separate',
    borderSpacing: 0,
    minWidth: 940,
    background: '#ffffff',
  },
  th: {
    textAlign: 'left',
    background: '#f8fafc',
    color: '#475569',
    fontSize: 12,
    textTransform: 'uppercase',
    letterSpacing: '0.07em',
    padding: '13px 14px',
    borderBottom: '1px solid #e2e8f0',
  },
  td: {
    padding: '13px 14px',
    borderBottom: '1px solid #eef2f7',
    color: '#334155',
    verticalAlign: 'top',
  },
  actionRow: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: 7,
  },
  reasonInput: {
    minWidth: 210,
    border: '1px solid #d6e0ea',
    background: '#ffffff',
    borderRadius: 12,
    padding: '9px 10px',
    color: '#0f172a',
    fontWeight: 700,
    outline: 'none',
  },
};

function merge(...items) {
  return Object.assign({}, ...items.filter(Boolean));
}

function normalizePeriod(period) {
  const raw = String(period || '').toUpperCase();
  return raw.startsWith('P') ? raw : `P${raw}`;
}

function statusOf(row, changes) {
  return changes[row.Roll_Number]?.status || row.Status || row.Final_Status || 'Absent';
}

function isPresentStatus(status) {
  return /present/i.test(String(status || ''));
}

function isReviewCase(row) {
  const status = row.Status || row.Final_Status || '';
  const flags = row.Flags || '';
  const detections = Number(row.Total_Accepted_Detections || row.Detection_Count || 0) || 0;
  return /review|weak|low confidence|late|early/i.test(`${status} ${flags}`) || (flags && flags !== '—') || (/absent/i.test(status) && detections > 0);
}

function rollCompare(a, b) {
  return String(a.Roll_Number || '').localeCompare(String(b.Roll_Number || ''), undefined, { numeric: true, sensitivity: 'base' });
}

function scoreText(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) return '—';
  return parsed.toFixed(3);
}

function StatTile({ label, value, tone = 'default', hint }) {
  const toneStyles = {
    success: { background: '#ecfdf5', borderColor: '#bbf7d0' },
    warning: { background: '#fffbeb', borderColor: '#fde68a' },
    danger: { background: '#fff1f2', borderColor: '#fecaca' },
    info: { background: '#eff6ff', borderColor: '#bfdbfe' },
  };
  return (
    <div style={merge(S.metric, toneStyles[tone])}>
      <small style={S.metricLabel}>{label}</small>
      <strong style={S.metricValue}>{value}</strong>
      {hint ? <span style={{ color: '#64748b', fontSize: 12, fontWeight: 800 }}>{hint}</span> : null}
    </div>
  );
}

function ActiveButton({ active, children, onClick }) {
  return (
    <button type="button" style={merge(S.filterPill, active && S.filterPillActive)} onClick={onClick}>
      {children}
    </button>
  );
}

function ChangePill({ change }) {
  if (!change) return <small style={{ color: '#94a3b8', fontWeight: 800 }}>No manual change</small>;
  if (change.status === 'CLEAR') return <small style={{ color: '#475569', fontWeight: 900 }}>Manual correction cleared</small>;
  return <small style={{ color: '#047857', fontWeight: 900 }}>Selected: {change.status}</small>;
}

export default function ManualReview() {
  const { currentUser } = useAuth();
  const [searchParams] = useSearchParams();
  const requestedSessionId = searchParams.get('session_id') || '';
  const requestedDay = searchParams.get('day') || '';
  const admin = isAdmin(currentUser);
  const today = new Date().toLocaleDateString('en-US', { weekday: 'long' });
  const [day, setDay] = useState(DAYS.includes(requestedDay) ? requestedDay : (DAYS.includes(today) ? today : 'Monday'));
  const [period, setPeriod] = useState('P1');
  const [periodOptions, setPeriodOptions] = useState([]);
  const [entry, setEntry] = useState(null);
  const [rows, setRows] = useState([]);
  const [changes, setChanges] = useState({});
  const [message, setMessage] = useState(null);
  const [viewMode, setViewMode] = useState('review');
  const [sortMode, setSortMode] = useState('roll_asc');
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [finalizing, setFinalizing] = useState(false);
  const [reviewState, setReviewState] = useState(null);
  const [lastDownloadUrl, setLastDownloadUrl] = useState('');

  useEffect(() => {
    let cancelled = false;
    async function loadPeriods() {
      const result = await apiGetResult(`/api/timetable?day=${day}`);
      if (cancelled) return;
      if (!result.ok) {
        setPeriodOptions([]);
        setEntry(null);
        setRows([]);
        setReviewState(null);
        setMessage({ type: 'error', text: result.networkError ? 'Backend offline. Review Students requires live session data.' : (result.error || 'Could not load timetable.') });
        return;
      }
      const rawRows = result.data?.timetable || [];
      const visible = rawRows
        .filter((row) => !/lunch/i.test(row.subject || ''))
        .filter((row) => canAccessRow(currentUser, row));
      const opts = visible.map((row, index) => {
        const subject = displaySubjectForUser(row, currentUser);
        const info = subjectInfo(subject);
        return {
          value: row.session_id || `${normalizePeriod(row.period)}-${subject}-${index}`,
          key: row.session_id || `${normalizePeriod(row.period)}-${subject}-${index}`,
          sessionId: row.session_id,
          sessionDate: row.session_date,
          period: normalizePeriod(row.period),
          shortLabel: normalizePeriod(row.period),
          subject,
          courseName: info?.courseName || row.course_name || '',
          time: `${row.start_time}–${row.end_time}`,
          room: row.room || '—',
        };
      });
      setPeriodOptions(opts);
      const requested = requestedSessionId ? opts.find((opt) => opt.sessionId === requestedSessionId) : null;
      if (requested) setPeriod(requested.value);
      else if (opts.length && !opts.some((opt) => opt.value === period)) setPeriod(opts[0].value);
      if (!opts.length) setMessage({ type: 'error', text: 'No reviewable class slots are available for this day and login.' });
    }
    loadPeriods();
    return () => { cancelled = true; };
  }, [day, currentUser, requestedSessionId]);

  useEffect(() => {
    if (!Object.keys(changes).length) return undefined;
    const warn = (event) => {
      event.preventDefault();
      event.returnValue = '';
    };
    window.addEventListener('beforeunload', warn);
    return () => window.removeEventListener('beforeunload', warn);
  }, [changes]);

  const confirmDiscardPending = () => (
    !Object.keys(changes).length || window.confirm('Discard the unsaved attendance corrections?')
  );

  const resetLoadedReport = () => {
    setEntry(null);
    setRows([]);
    setChanges({});
    setReviewState(null);
    setLastDownloadUrl('');
    setMessage(null);
  };

  const load = async () => {
    const selected = periodOptions.find((option) => option.value === period);
    if (!selected?.sessionId) {
      setMessage({ type: 'error', text: 'This class needs a safe session date before review can be loaded.' });
      return;
    }
    if (!confirmDiscardPending()) return;
    setLoading(true);
    setMessage({ type: 'info', text: `Loading ${selected.sessionDate || day} ${selected.period} ${selected.subject} attendance...` });
    const result = await apiGetResult(`/api/attendance/${encodeURIComponent(selected.sessionId)}`);
    if (!result.ok || !result.data?.success) {
      resetLoadedReport();
      setMessage({ type: 'error', text: result.error || result.data?.error || 'No completed attendance found for this slot yet.' });
      setLoading(false);
      return;
    }
    const data = result.data;
    const visibleRows = filterStudentRowsForUser(data.attendance_data || [], currentUser);
    setEntry(data.entry);
    setRows(visibleRows);
    setReviewState(data.review_state || null);
    setChanges({});
    setSearch('');
    setViewMode((data.review_state?.unresolved_count || 0) > 0 ? 'review' : 'all');
    setLastDownloadUrl(data.entry?.final_attendance_csv ? `/api/download/${data.entry.final_attendance_csv}` : '');
    const stateMessage = data.review_state?.attendance_finalized
      ? 'Finalized attendance loaded.'
      : data.review_state?.roster_complete === false
        ? `${data.review_state?.reason || 'The attendance rows do not match the authoritative roster.'} Finalization is blocked.`
      : (data.review_state?.unresolved_count || 0) > 0
        ? `${data.review_state.unresolved_count} student(s) remain unresolved. Needs Review rows have model evidence; Unconfirmed and Missing Enrollment rows require an explicit faculty attendance decision.`
        : 'All unresolved cases are resolved. This report is ready to finalize.';
    setMessage({ type: 'success', text: `Attendance loaded: ${visibleRows.length} visible student record(s). ${stateMessage}` });
    setLoading(false);
  };

  const summary = useMemo(() => {
    const result = classifyReviewRows(rows, changes, {
      statusOf,
      isPresent: isPresentStatus,
      isEvidenceReview: isReviewCase,
    });
    return { ...result, reviewCases: result.unresolved };
  }, [rows, changes]);

  const filteredRows = useMemo(() => {
    let output = rows.filter((row) => String(row.Roll_Number || '').toLowerCase().includes(search.toLowerCase()));
    output = output.filter((row) => rowMatchesReviewMode(row, viewMode, changes, {
      statusOf,
      isPresent: isPresentStatus,
      isEvidenceReview: isReviewCase,
    }));

    output.sort((a, b) => {
      if (sortMode === 'roll_desc') return -rollCompare(a, b);
      if (sortMode === 'status') return String(statusOf(a, changes)).localeCompare(String(statusOf(b, changes))) || rollCompare(a, b);
      return rollCompare(a, b);
    });
    return output;
  }, [rows, search, viewMode, sortMode, changes]);

  const selectedChangeCount = Object.keys(changes).length;
  const hasRows = rows.length > 0;
  const hasOptions = periodOptions.length > 0;
  const selectedPeriod = periodOptions.find((option) => option.value === period);
  const finalized = Boolean(reviewState?.attendance_finalized || entry?.attendance_finalized);
  const actionState = reviewActionState({
    pendingChanges: selectedChangeCount,
    unresolvedCount: summary.reviewCases,
    rosterComplete: reviewState?.roster_complete !== false,
    finalized,
    busy: saving || exporting || finalizing,
  });

  const setStatus = (roll, status) => {
    setChanges((prev) => ({ ...prev, [roll]: { status, reason: prev[roll]?.reason || '' } }));
  };

  const clearChange = (roll) => {
    const row = rows.find((item) => item.Roll_Number === roll);
    setChanges((prev) => {
      const next = { ...prev };
      if (row?.Manual_Override) next[roll] = { status: 'CLEAR', reason: 'Clear saved manual correction' };
      else delete next[roll];
      return next;
    });
  };

  const setReason = (roll, reason) => {
    setChanges((prev) => {
      if (!prev[roll] || prev[roll].status === 'CLEAR') return prev;
      return { ...prev, [roll]: { ...prev[roll], reason } };
    });
  };

  const save = async () => {
    const payload = Object.entries(changes).map(([roll, change]) => ({ roll, status: change.status, reason: change.reason }));
    if (!payload.length) {
      setMessage({ type: 'error', text: 'No changes to save.' });
      return;
    }
    const missingReasons = correctionReasonErrors(changes);
    if (missingReasons.length) {
      setMessage({ type: 'error', text: `Add a correction reason for: ${missingReasons.join(', ')}.` });
      return;
    }
    setSaving(true);
    try {
      const selected = periodOptions.find((option) => option.value === period);
      if (!selected?.sessionId) throw new Error('This class needs a safe session id before corrections can be saved.');
      const data = await apiPost(`/api/attendance/${encodeURIComponent(selected.sessionId)}/edit`, { session_id: selected.sessionId, changes: payload });
      setEntry(data.entry || entry);
      setRows(filterStudentRowsForUser(data.attendance_data || rows, currentUser));
      setReviewState(data.review_state || null);
      setChanges({});
      setLastDownloadUrl(data.download_url || '');
      setViewMode((data.review_state?.unresolved_count || 0) > 0 ? 'review' : 'all');
      setMessage({ type: 'success', text: data.message || 'Corrections saved.' });
    } catch (error) {
      setMessage({ type: 'error', text: error.message || 'Could not save corrections.' });
    } finally {
      setSaving(false);
    }
  };

  const exportReviewed = async () => {
    if (selectedChangeCount) {
      setMessage({ type: 'error', text: 'Save or discard pending corrections before exporting.' });
      return;
    }
    const selected = periodOptions.find((option) => option.value === period);
    if (!selected?.sessionId) {
      setMessage({ type: 'error', text: 'This class needs a safe session id before export.' });
      return;
    }
    setExporting(true);
    const result = await apiGetResult(`/api/attendance/${encodeURIComponent(selected.sessionId)}/export-edited`);
    if (result.ok && result.data?.success && result.data.download_url) {
      window.open(result.data.download_url, '_blank', 'noopener,noreferrer');
      setLastDownloadUrl(result.data.download_url);
      setMessage({ type: 'success', text: result.data.finalized ? 'Final attendance CSV downloaded.' : `Reviewed export created: ${result.data.file}` });
    } else {
      setMessage({ type: 'error', text: result.error || result.data?.error || 'Could not export reviewed attendance.' });
    }
    setExporting(false);
  };

  const finalizeAttendance = async () => {
    if (selectedChangeCount) {
      setMessage({ type: 'error', text: 'Save or discard pending corrections before finalizing.' });
      return;
    }
    if (reviewState?.roster_complete === false) {
      setMessage({ type: 'error', text: `${reviewState?.reason || 'The attendance rows do not match the authoritative roster.'} Reprocess the session or ask HOD to correct the source report.` });
      return;
    }
    if (summary.reviewCases > 0) {
      setMessage({ type: 'error', text: `${summary.reviewCases} student(s) still need an explicit Present or Absent decision.` });
      setViewMode('review');
      return;
    }
    const selected = periodOptions.find((option) => option.value === period);
    if (!selected?.sessionId) {
      setMessage({ type: 'error', text: 'This class needs a safe session id before finalization.' });
      return;
    }
    if (!window.confirm('Finalize this attendance report? This creates the official reviewed CSV. Later corrections will reopen the report.')) return;
    setFinalizing(true);
    try {
      const data = await apiPost(`/api/attendance/${encodeURIComponent(selected.sessionId)}/finalize`, {});
      setEntry(data.entry || entry);
      setRows(filterStudentRowsForUser(data.attendance_data || rows, currentUser));
      setReviewState(data.review_state || { attendance_finalized: true, unresolved_count: 0, state: 'finalized' });
      setChanges({});
      setLastDownloadUrl(data.download_url || '');
      setViewMode('all');
      setMessage({ type: 'success', text: data.message || 'Attendance finalized successfully.' });
    } catch (error) {
      setMessage({ type: 'error', text: error.message || 'Could not finalize attendance.' });
    } finally {
      setFinalizing(false);
    }
  };

  return (
    <div style={S.page}>
      <PageHeader
        eyebrow={admin ? 'Admin Corrections' : 'Faculty Corrections'}
        title={admin ? 'Review & Corrections' : 'Review Students'}
        subtitle={admin ? 'Open a completed slot, review doubtful cases, and save clean corrections.' : 'Only your subject and assigned student list are visible here.'}
      />

      <section style={merge(S.card, S.pickerCard)}>
        <div style={S.topGrid}>
          <div>
            <p style={S.eyebrow}>Load completed class</p>
            <h2 style={S.h2}>Select class report</h2>
            <p style={S.muted}>Choose the day and slot. Corrections stay hidden until a completed attendance report is loaded.</p>
          </div>
          <button type="button" style={merge(S.primaryBtn, (!hasOptions || loading) && { opacity: 0.55, cursor: 'not-allowed' })} onClick={load} disabled={!hasOptions || loading}>
            {loading ? 'Loading…' : 'Load Report'}
          </button>
        </div>

        <div style={S.pickerGrid}>
          <div>
            <span style={S.fieldLabel}>Day</span>
            <div style={S.dayRow}>
              {DAYS.map((item) => (
                <button
                  key={item}
                  type="button"
                  style={merge(S.dayBtn, day === item && S.dayBtnActive)}
                  onClick={() => { if (!confirmDiscardPending()) return; setDay(item); resetLoadedReport(); }}
                >
                  {item.slice(0, 3)}
                </button>
              ))}
            </div>
          </div>

          <div>
            <span style={S.fieldLabel}>Class slot</span>
            {hasOptions ? (
              <div style={S.slotGrid}>
                {periodOptions.map((option) => (
                  <button
                    key={option.key}
                    type="button"
                    style={merge(S.slotCard, period === option.value && S.slotCardActive)}
                    onClick={() => { if (!confirmDiscardPending()) return; setPeriod(option.value); resetLoadedReport(); }}
                  >
                    <span style={S.slotTitle}>{option.shortLabel} · {option.subject}</span>
                    <span style={S.slotDesc}>{option.courseName}</span>
                    <span style={S.slotMeta}>{option.time} · Room {option.room}</span>
                  </button>
                ))}
              </div>
            ) : (
              <div style={merge(S.notice, S.noticeError)}>No class slots are available for this day and login.</div>
            )}
          </div>
        </div>
      </section>

      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
        {!admin && <span style={S.facultyBadge}>● Faculty mode: {currentUser?.subjects?.join(', ') || 'assigned'} only</span>}
        {selectedPeriod && <span style={S.facultyBadge}>Selected: {selectedPeriod.shortLabel} · {selectedPeriod.subject} · {selectedPeriod.time}</span>}
      </div>

      {message && (
        <div style={merge(S.notice, message.type === 'error' && S.noticeError, message.type === 'success' && S.noticeSuccess, message.type === 'info' && S.noticeInfo)}>
          {message.text}
        </div>
      )}

      {!hasRows && (
        <section style={S.card}>
          <div style={S.empty}>
            <span style={S.emptyIcon}>☝️</span>
            <div>
              <h3 style={S.h3}>No class loaded yet</h3>
              <p style={S.muted}>Select a class card above and click <b>Load Report</b>. The correction table will appear only after data is loaded.</p>
            </div>
          </div>
        </section>
      )}

      {entry && hasRows && (
        <section style={S.card}>
          <div style={S.topGrid}>
            <div>
              <p style={S.eyebrow}>Loaded report</p>
              <h2 style={S.h2}>{entry.subject || selectedPeriod?.subject || 'Selected class'} · {entry.period || period}</h2>
              <p style={S.muted}>{entry.course_name || selectedPeriod?.courseName || 'Course'} · {entry.start_time || selectedPeriod?.time?.split('–')[0] || ''}–{entry.end_time || selectedPeriod?.time?.split('–')[1] || ''} · Room {entry.room || selectedPeriod?.room || '—'}</p>
            </div>
            <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
              <button type="button" style={merge(S.secondaryBtn, !actionState.canExport && { opacity: 0.5, cursor: 'not-allowed' })} onClick={exportReviewed} disabled={!actionState.canExport}>
                {exporting ? 'Exporting…' : finalized ? 'Download Final CSV' : 'Export Reviewed CSV'}
              </button>
              <button type="button" style={merge(S.primaryBtn, !actionState.canSave && { opacity: 0.5, cursor: 'not-allowed' })} onClick={save} disabled={!actionState.canSave}>
                {saving ? 'Saving…' : selectedChangeCount ? `Save ${selectedChangeCount} Change${selectedChangeCount === 1 ? '' : 's'}` : 'No Changes to Save'}
              </button>
              {!finalized && (
                <button type="button" style={merge(S.primaryBtn, { background: 'linear-gradient(135deg, #047857, #059669)' }, !actionState.canFinalize && { opacity: 0.5, cursor: 'not-allowed' })} onClick={finalizeAttendance} disabled={!actionState.canFinalize}>
                  {finalizing ? 'Finalizing…' : 'Finalize Attendance'}
                </button>
              )}
            </div>
          </div>

          {entry.source_report_superseded && (
            <div style={merge(S.notice, S.noticeInfo, { marginTop: 16 })}>
              <strong>Official reviewed multi-frame revision</strong>
              <div style={{ marginTop: 5, fontWeight: 700 }}>The original frame-only report is preserved as superseded. Exact-source reviewed evidence remains authoritative and insufficient camera evidence is not treated as absence.</div>
            </div>
          )}

          {entry.pending_candidate_revision && (
            <div style={merge(S.notice, S.noticeWarning, { marginTop: 16 })}>
              <strong>Automatic rerun archived without replacing this report</strong>
              <div style={{ marginTop: 5, fontWeight: 700 }}>Candidate result: {entry.pending_candidate_revision.present_count || 0} present, {entry.pending_candidate_revision.needs_review_count || 0} review, {entry.pending_candidate_revision.unconfirmed_count || 0} unconfirmed. Review decisions are not discarded when the same footage is processed again.</div>
            </div>
          )}

          {entry.latest_candidate_matches_official && entry.latest_equivalent_candidate_revision && (
            <div style={merge(S.notice, S.noticeSuccess, { marginTop: 16 })}>
              <strong>Latest identical-source rerun matched this reviewed report</strong>
              <div style={{ marginTop: 5, fontWeight: 700 }}>Exact video, embedding, and tracklet-evidence signatures matched. Existing decisions were reused safely and no repeated review was created.</div>
            </div>
          )}

          <div style={merge(
            S.notice,
            finalized && S.noticeSuccess,
            !finalized && reviewState?.roster_complete === false && S.noticeError,
            !finalized && reviewState?.roster_complete !== false && summary.reviewCases > 0 && S.noticeWarning,
            !finalized && reviewState?.roster_complete !== false && summary.reviewCases === 0 && S.noticeInfo,
            { marginTop: 16 },
          )}>
            <strong>
              {finalized
                ? 'Attendance finalized'
                : reviewState?.roster_complete === false
                  ? 'Roster mismatch blocks finalization'
                : summary.reviewCases > 0
                  ? `${summary.review} evidence review · ${summary.unconfirmed} unconfirmed · ${summary.missingEnrollment} missing enrollment`
                  : 'Ready for explicit finalization'}
            </strong>
            <div style={{ marginTop: 5, fontWeight: 700 }}>
              {finalized
                ? `Finalized${entry.finalized_by ? ` by ${entry.finalized_by}` : ''}${entry.finalized_at ? ` on ${entry.finalized_at}` : ''}. Later saved corrections will reopen this report.`
                : reviewState?.roster_complete === false
                  ? `${reviewState?.reason || 'The saved attendance rows do not match the authoritative roster.'} Missing roster students: ${reviewState?.missing_count ?? (reviewState?.missing_rolls || []).length}. Out-of-roster rows: ${reviewState?.unexpected_count ?? (reviewState?.unexpected_rolls || []).length}. Reprocess or ask HOD to resolve the source report.`
                : summary.reviewCases > 0
                  ? 'Inspect model evidence for Needs Review rows. For Unconfirmed or Missing Enrollment rows, use verified classroom information rather than treating recognition failure as absence. Every manual decision requires a reason.'
                  : 'All unresolved cases are resolved. Finalize to create the official reviewed CSV.'}
            </div>
          </div>

          <div style={S.metricGrid}>
            <StatTile label="Present" value={summary.present} tone="success" />
            <StatTile label="Need review" value={summary.review} tone="warning" />
            <StatTile label="Unconfirmed" value={summary.unconfirmed} tone="info" />
            <StatTile label="Missing enrollment" value={summary.missingEnrollment} tone="warning" />
            <StatTile label="Absent" value={summary.absent} tone="danger" />
            <StatTile label="Present rate" value={`${summary.pct}%`} tone="info" />
          </div>
        </section>
      )}

      {hasRows && (
        <section style={S.card}>
          <div style={S.topGrid}>
            <div>
              <h3 style={S.h3}>Attendance corrections</h3>
              <p style={S.muted}>Review doubtful cases first, or switch to all students when needed.</p>
            </div>
            <div style={{ textAlign: 'right' }}>
              <span style={S.facultyBadge}>{selectedChangeCount} pending change{selectedChangeCount === 1 ? '' : 's'}</span>
            </div>
          </div>

          <div style={S.toolbar}>
            <input style={S.input} value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search roll number" />
            <div style={S.pillSelectRow}>
              <ActiveButton active={viewMode === 'review'} onClick={() => setViewMode('review')}>Unresolved</ActiveButton>
              <ActiveButton active={viewMode === 'all'} onClick={() => setViewMode('all')}>All</ActiveButton>
              <ActiveButton active={viewMode === 'present'} onClick={() => setViewMode('present')}>Present</ActiveButton>
              <ActiveButton active={viewMode === 'needs_review'} onClick={() => setViewMode('needs_review')}>Needs review</ActiveButton>
              <ActiveButton active={viewMode === 'unconfirmed'} onClick={() => setViewMode('unconfirmed')}>Unconfirmed</ActiveButton>
              <ActiveButton active={viewMode === 'missing'} onClick={() => setViewMode('missing')}>Missing enrollment</ActiveButton>
              <ActiveButton active={viewMode === 'absent'} onClick={() => setViewMode('absent')}>Absent</ActiveButton>
            </div>
            <div style={S.pillSelectRow}>
              <ActiveButton active={sortMode === 'roll_asc'} onClick={() => setSortMode('roll_asc')}>Roll ↑</ActiveButton>
              <ActiveButton active={sortMode === 'roll_desc'} onClick={() => setSortMode('roll_desc')}>Roll ↓</ActiveButton>
              <ActiveButton active={sortMode === 'status'} onClick={() => setSortMode('status')}>Status</ActiveButton>
            </div>
          </div>

          <div style={S.tableWrap}>
            <table style={S.table}>
              <thead>
                <tr>
                  <th style={S.th}>Roll Number</th>
                  <th style={S.th}>System Status</th>
                  <th style={S.th}>Evidence</th>
                  <th style={S.th}>Correction</th>
                  <th style={S.th}>Flags</th>
                  <th style={S.th}>Reason</th>
                </tr>
              </thead>
              <tbody>
                {filteredRows.map((row) => {
                  const change = changes[row.Roll_Number];
                  const evidence = `${row.Recognized_Checkpoints || 0}/${row.Total_Checkpoints || 5} official checkpoints · ${row.Total_Accepted_Detections || row.Detection_Count || 0} observations`;
                  const reviewedTracklets = row.Reviewed_Tracklet_Checkpoints || '';
                  const recoveryCandidates = row.Guarded_Recovery_Candidate_Checkpoints || '';
                  const mixedRejected = row.Mixed_Track_Checkpoints_Rejected || '';
                  const needsReview = rowMatchesReviewMode(row, 'review', changes, {
                    statusOf,
                    isPresent: isPresentStatus,
                    isEvidenceReview: isReviewCase,
                  });
                  return (
                    <tr key={row.Roll_Number} style={needsReview ? { background: '#fffbeb' } : null}>
                      <td style={S.td}>
                        <strong style={{ color: '#0f172a' }}>{row.Roll_Number}</strong>
                        <div style={{ color: '#94a3b8', fontSize: 12, marginTop: 3 }}>{row.Cameras_Seen || 'No camera evidence'}</div>
                        {row.Manual_Override && (
                          <div style={{ color: '#047857', fontSize: 12, marginTop: 4, fontWeight: 850 }}>Saved manual decision · {row.Edited_By || 'authorized reviewer'}</div>
                        )}
                      </td>
                      <td style={S.td}><StatusBadge status={statusOf(row, changes)} /></td>
                      <td style={S.td}>
                        <strong style={{ color: '#334155' }}>{evidence}</strong>
                        <div style={{ color: '#64748b', fontSize: 12, marginTop: 3 }}>avg {scoreText(row.Average_Score)} {row.Best_Score ? `· best ${scoreText(row.Best_Score)}` : ''}</div>
                        {row.Strict_Recognized_Checkpoints !== undefined && row.Strict_Recognized_Checkpoints !== '' ? <div style={{ color: '#64748b', fontSize: 12, marginTop: 4 }}>Strict checkpoints: {row.Strict_Recognized_Checkpoints}/{row.Total_Checkpoints || 5}</div> : null}
                        {reviewedTracklets ? <div style={{ color: '#0f766e', fontSize: 12, marginTop: 4, fontWeight: 850 }}>Multi-frame checkpoints (reviewed): {reviewedTracklets}</div> : null}
                        {recoveryCandidates && !reviewedTracklets ? <div style={{ color: '#b45309', fontSize: 12, marginTop: 4, fontWeight: 850 }}>Recovery candidate: {recoveryCandidates}</div> : null}
                        {mixedRejected ? <div style={{ color: '#b45309', fontSize: 12, marginTop: 4, fontWeight: 850 }}>Mixed track rejected: {mixedRejected}</div> : null}
                        {row.Evidence_Interpretation ? <div style={{ color: '#64748b', fontSize: 12, marginTop: 4 }}>{row.Evidence_Interpretation}</div> : null}
                      </td>
                      <td style={S.td}>
                        <div style={S.actionRow}>
                          <button type="button" style={change?.status === 'Present' ? S.primaryBtn : S.smallBtn} onClick={() => setStatus(row.Roll_Number, 'Present')}>Present</button>
                          <button type="button" style={change?.status === 'Needs Review' ? S.warningBtn : S.smallBtn} onClick={() => setStatus(row.Roll_Number, 'Needs Review')}>Review</button>
                          <button type="button" style={change?.status === 'Absent' ? S.dangerBtn : S.smallBtn} onClick={() => setStatus(row.Roll_Number, 'Absent')}>Absent</button>
                          <button type="button" style={S.smallBtn} onClick={() => clearChange(row.Roll_Number)}>{row.Manual_Override ? 'Clear Saved' : 'Undo'}</button>
                        </div>
                        <div style={{ marginTop: 7 }}><ChangePill change={change} /></div>
                      </td>
                      <td style={S.td}>{row.Flags || '—'}</td>
                      <td style={S.td}>
                        <input
                          style={merge(S.reasonInput, (!change || change.status === 'CLEAR') && { background: '#f8fafc', color: '#94a3b8' })}
                          value={change?.status === 'CLEAR' ? '' : (change?.reason || '')}
                          placeholder={change && change.status !== 'CLEAR' ? 'Reason required' : 'Select a correction first'}
                          disabled={!change || change.status === 'CLEAR'}
                          onChange={(event) => setReason(row.Roll_Number, event.target.value)}
                        />
                      </td>
                    </tr>
                  );
                })}
                {filteredRows.length === 0 && (
                  <tr>
                    <td style={{ ...S.td, textAlign: 'center', padding: 28 }} colSpan="6">
                      <strong>No students match this filter.</strong>
                      <div style={{ marginTop: 8 }}>
                        <button type="button" style={S.secondaryBtn} onClick={() => setViewMode('all')}>Show all students</button>
                      </div>
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}
