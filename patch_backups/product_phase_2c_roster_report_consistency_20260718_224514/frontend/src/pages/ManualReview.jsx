import { useEffect, useMemo, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import PageHeader from '../components/PageHeader.jsx';
import StatusBadge from '../components/StatusBadge.jsx';
import { apiGet, apiPost } from '../api/client.js';
import { DAYS, FALLBACK_TIMETABLE } from '../data/fallback.js';
import { useAuth } from '../auth/AuthContext.jsx';
import { canAccessRow, displaySubjectForUser, isAdmin, subjectInfo } from '../data/users.js';
import { filterStudentRowsForUser } from '../data/students.js';

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
  const admin = isAdmin(currentUser);
  const today = new Date().toLocaleDateString('en-US', { weekday: 'long' });
  const [day, setDay] = useState(DAYS.includes(today) ? today : 'Monday');
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

  useEffect(() => {
    let cancelled = false;
    async function loadPeriods() {
      const data = await apiGet(`/api/timetable?day=${day}`, null);
      if (cancelled) return;
      const rawRows = data?.timetable || FALLBACK_TIMETABLE.filter((row) => row.day === day);
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
    }
    loadPeriods();
    return () => { cancelled = true; };
  }, [day, currentUser, period, requestedSessionId]);

  const load = async () => {
    const selected = periodOptions.find((option) => option.value === period);
    if (!selected?.sessionId) {
      setMessage({ type: 'error', text: 'This class needs a safe session date before review can be loaded.' });
      return;
    }
    setLoading(true);
    setMessage({ type: 'info', text: `Loading ${selected.sessionDate || day} ${selected.period} ${selected.subject} attendance...` });
    const data = await apiGet(`/api/attendance/${encodeURIComponent(selected.sessionId)}`, null);
    if (!data?.success) {
      setEntry(null);
      setRows([]);
      setChanges({});
      setMessage({ type: 'error', text: 'No completed attendance found for this slot yet.' });
      setLoading(false);
      return;
    }
    const visibleRows = filterStudentRowsForUser(data.attendance_data || [], currentUser);
    setEntry(data.entry);
    setRows(visibleRows);
    setChanges({});
    setSearch('');
    setViewMode('review');
    setMessage({ type: 'success', text: `Attendance loaded: ${visibleRows.length} visible student record(s).` });
    setLoading(false);
  };

  const summary = useMemo(() => {
    const present = rows.filter((row) => isPresentStatus(statusOf(row, changes))).length;
    const review = rows.filter((row) => /review/i.test(statusOf(row, changes))).length;
    const absent = Math.max(0, rows.length - present - review);
    const reviewCases = rows.filter(isReviewCase).length;
    return { present, review, absent, reviewCases, total: rows.length, pct: rows.length ? Math.round((present / rows.length) * 1000) / 10 : 0 };
  }, [rows, changes]);

  const filteredRows = useMemo(() => {
    let output = rows.filter((row) => String(row.Roll_Number || '').toLowerCase().includes(search.toLowerCase()));
    if (viewMode === 'review') output = output.filter(isReviewCase);
    if (viewMode === 'present') output = output.filter((row) => isPresentStatus(statusOf(row, changes)));
    if (viewMode === 'absent') output = output.filter((row) => /absent/i.test(statusOf(row, changes)));

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

  const setStatus = (roll, status) => {
    setChanges((prev) => ({ ...prev, [roll]: { status, reason: prev[roll]?.reason || 'Attendance review correction' } }));
  };

  const clearChange = (roll) => {
    setChanges((prev) => ({ ...prev, [roll]: { status: 'CLEAR', reason: 'Clear manual correction' } }));
  };

  const setReason = (roll, reason) => {
    setChanges((prev) => ({ ...prev, [roll]: { status: prev[roll]?.status || rows.find((r) => r.Roll_Number === roll)?.Status, reason } }));
  };

  const save = async () => {
    const payload = Object.entries(changes).map(([roll, change]) => ({ roll, status: change.status, reason: change.reason }));
    if (!payload.length) {
      setMessage({ type: 'error', text: 'No changes to save.' });
      return;
    }
    try {
      const selected = periodOptions.find((option) => option.value === period);
      if (!selected?.sessionId) throw new Error('This class needs a safe session id before corrections can be saved.');
      const data = await apiPost(`/api/attendance/${encodeURIComponent(selected.sessionId)}/edit`, { changes: payload });
      setMessage({ type: 'success', text: data.admin_reviewed_csv ? `Corrections saved. Reviewed CSV: ${data.admin_reviewed_csv}` : (data.message || 'Corrections saved.') });
      await load();
    } catch (error) {
      setMessage({ type: 'error', text: error.message || 'Could not save corrections.' });
    }
  };

  const exportReviewed = async () => {
    const selected = periodOptions.find((option) => option.value === period);
    if (!selected?.sessionId) {
      setMessage({ type: 'error', text: 'This class needs a safe session id before export.' });
      return;
    }
    const data = await apiGet(`/api/attendance/${encodeURIComponent(selected.sessionId)}/export-edited`, null);
    if (data?.success && data.download_url) {
      window.open(data.download_url, '_blank');
      setMessage({ type: 'success', text: `Export created: ${data.file}` });
    } else {
      setMessage({ type: 'error', text: data?.error || 'Could not export reviewed attendance.' });
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
                  onClick={() => { setDay(item); setEntry(null); setRows([]); setChanges({}); setMessage(null); }}
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
                    onClick={() => { setPeriod(option.value); setEntry(null); setRows([]); setChanges({}); setMessage(null); }}
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
              <button type="button" style={S.secondaryBtn} onClick={exportReviewed}>Export Reviewed CSV</button>
              <button type="button" style={S.primaryBtn} onClick={save}>Save Corrections</button>
            </div>
          </div>

          <div style={S.metricGrid}>
            <StatTile label="Present" value={summary.present} tone="success" />
            <StatTile label="Need review" value={summary.reviewCases} tone="warning" />
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
              <ActiveButton active={viewMode === 'review'} onClick={() => setViewMode('review')}>Need review</ActiveButton>
              <ActiveButton active={viewMode === 'all'} onClick={() => setViewMode('all')}>All</ActiveButton>
              <ActiveButton active={viewMode === 'present'} onClick={() => setViewMode('present')}>Present</ActiveButton>
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
                  const evidence = `${row.Recognized_Checkpoints || 0}/${row.Total_Checkpoints || 5} checkpoints · ${row.Total_Accepted_Detections || row.Detection_Count || 0} detections`;
                  return (
                    <tr key={row.Roll_Number} style={isReviewCase(row) ? { background: '#fffbeb' } : null}>
                      <td style={S.td}>
                        <strong style={{ color: '#0f172a' }}>{row.Roll_Number}</strong>
                        <div style={{ color: '#94a3b8', fontSize: 12, marginTop: 3 }}>{row.Cameras_Seen || 'No camera evidence'}</div>
                      </td>
                      <td style={S.td}><StatusBadge status={row.Status || row.Final_Status} /></td>
                      <td style={S.td}>
                        <strong style={{ color: '#334155' }}>{evidence}</strong>
                        <div style={{ color: '#64748b', fontSize: 12, marginTop: 3 }}>avg {scoreText(row.Average_Score)} {row.Best_Score ? `· best ${scoreText(row.Best_Score)}` : ''}</div>
                      </td>
                      <td style={S.td}>
                        <div style={S.actionRow}>
                          <button type="button" style={change?.status === 'Present' ? S.primaryBtn : S.smallBtn} onClick={() => setStatus(row.Roll_Number, 'Present')}>Present</button>
                          <button type="button" style={change?.status === 'Needs Review' ? S.warningBtn : S.smallBtn} onClick={() => setStatus(row.Roll_Number, 'Needs Review')}>Review</button>
                          <button type="button" style={change?.status === 'Absent' ? S.dangerBtn : S.smallBtn} onClick={() => setStatus(row.Roll_Number, 'Absent')}>Absent</button>
                          <button type="button" style={S.smallBtn} onClick={() => clearChange(row.Roll_Number)}>Clear</button>
                        </div>
                        <div style={{ marginTop: 7 }}><ChangePill change={change} /></div>
                      </td>
                      <td style={S.td}>{row.Flags || '—'}</td>
                      <td style={S.td}>
                        <input style={S.reasonInput} value={change?.reason || ''} placeholder="Reason" onChange={(event) => setReason(row.Roll_Number, event.target.value)} />
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
