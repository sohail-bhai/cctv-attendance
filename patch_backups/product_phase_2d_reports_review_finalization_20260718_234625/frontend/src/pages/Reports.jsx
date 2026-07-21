import { useEffect, useMemo, useState } from 'react';
import PageHeader from '../components/PageHeader.jsx';
import StatusBadge from '../components/StatusBadge.jsx';
import { useAuth } from '../auth/AuthContext.jsx';
import { isAdmin } from '../data/users.js';
import { filterStudentRowsForUser } from '../data/students.js';
import { classifyReviewRows, rowMatchesReviewMode, validateReportScope } from '../utils/consistency.js';
import {
  fileToRows,
  parseCsv,
  getAvgScore,
  getBestScore,
  getDetectionCount,
  getRoll,
  getStatus,
  summarizeDetectionLog,
} from '../utils/csv.js';

const DEMO_MANIFEST_URL = '/demo_reports/manifest.json';

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
  sourceCard: {
    display: 'grid',
    gridTemplateColumns: 'minmax(0, 1fr) auto',
    gap: 20,
    alignItems: 'center',
    background: 'linear-gradient(135deg, #ffffff 0%, #f8fbff 62%, #ecfeff 100%)',
  },
  icon: {
    width: 56,
    height: 56,
    borderRadius: 18,
    display: 'grid',
    placeItems: 'center',
    background: 'linear-gradient(135deg, #0f8b8d, #0f7aa8)',
    color: '#ffffff',
    fontSize: 24,
    boxShadow: '0 14px 30px rgba(15, 122, 168, 0.2)',
    flex: '0 0 56px',
  },
  sourceCopy: {
    display: 'flex',
    gap: 16,
    alignItems: 'center',
    minWidth: 0,
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
  actions: {
    display: 'flex',
    gap: 10,
    flexWrap: 'wrap',
    justifyContent: 'flex-end',
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
    padding: '11px 15px',
    cursor: 'pointer',
    whiteSpace: 'nowrap',
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
  statusLine: {
    display: 'inline-flex',
    marginTop: 10,
    padding: '7px 11px',
    borderRadius: 999,
    fontSize: 12,
    fontWeight: 900,
    background: '#f1f5f9',
    color: '#475569',
    border: '1px solid #e2e8f0',
  },
  statusReady: { background: '#ecfdf5', color: '#047857', borderColor: '#bbf7d0' },
  statusMissing: { background: '#fffbeb', color: '#b45309', borderColor: '#fde68a' },
  qualityWarning: {
    marginTop: 16,
    border: '1px solid #fde68a',
    background: '#fffbeb',
    color: '#92400e',
    borderRadius: 16,
    padding: 14,
    lineHeight: 1.45,
  },
  uploadGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(230px, 1fr))',
    gap: 12,
    marginTop: 16,
  },
  fileCard: {
    minHeight: 82,
    border: '1px dashed rgba(15, 139, 141, 0.38)',
    borderRadius: 18,
    background: '#ffffff',
    display: 'flex',
    alignItems: 'center',
    gap: 13,
    padding: 14,
    cursor: 'pointer',
    boxShadow: '0 10px 25px rgba(15, 23, 42, 0.045)',
  },
  fileIcon: {
    width: 40,
    height: 40,
    borderRadius: 14,
    background: '#ccfbf1',
    color: '#0f766e',
    display: 'grid',
    placeItems: 'center',
    fontSize: 18,
    flex: '0 0 40px',
  },
  empty: {
    display: 'flex',
    alignItems: 'center',
    gap: 15,
  },
  emptyIcon: {
    width: 56,
    height: 56,
    borderRadius: 18,
    background: '#ecfeff',
    display: 'grid',
    placeItems: 'center',
    fontSize: 24,
  },
  topGrid: {
    display: 'grid',
    gridTemplateColumns: 'minmax(0, 1fr) auto',
    gap: 20,
    alignItems: 'start',
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
  techGrid: {
    marginTop: 16,
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))',
    gap: 10,
    paddingTop: 16,
    borderTop: '1px solid #e2e8f0',
  },
  techItem: {
    borderRadius: 14,
    background: '#f8fafc',
    border: '1px solid #e2e8f0',
    padding: 12,
    color: '#64748b',
    fontSize: 13,
  },
  toolbar: {
    display: 'grid',
    gridTemplateColumns: 'minmax(220px, 1fr) auto auto',
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
  pillRow: {
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
    minWidth: 980,
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
  drawer: {
    position: 'fixed',
    inset: 0,
    background: 'rgba(15, 23, 42, 0.42)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
    zIndex: 50,
  },
  drawerCard: {
    width: 'min(720px, 100%)',
    background: '#ffffff',
    borderRadius: 24,
    padding: 24,
    boxShadow: '0 30px 70px rgba(15, 23, 42, 0.28)',
  },
};

function merge(...items) {
  return Object.assign({}, ...items.filter(Boolean));
}

function getSlotValue(row, keys, fallback = '—') {
  if (!row) return fallback;
  for (const key of keys) {
    const value = row[key];
    if (value !== undefined && value !== null && String(value).trim() !== '') return value;
  }
  return fallback;
}

function minutesText(seconds) {
  const n = Number(seconds);
  if (!Number.isFinite(n)) return seconds || '—';
  if (n < 60) return `${Math.round(n * 10) / 10}s`;
  return `${Math.floor(n / 60)}m ${Math.round(n % 60)}s`;
}

function rollCompare(a, b) {
  return String(getRoll(a)).localeCompare(String(getRoll(b)), undefined, { numeric: true, sensitivity: 'base' });
}

function truthyReportValue(value) {
  const text = String(value ?? '').trim().toLowerCase();
  return ['1', 'true', 'yes', 'needs_review'].includes(text);
}

function falseyReportValue(value) {
  const text = String(value ?? '').trim().toLowerCase();
  return ['0', 'false', 'no'].includes(text);
}

function numberValue(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function buildReportQuality(row) {
  if (!row) return { requiresManualReview: false, finalized: true, status: 'Valid', reason: '' };
  const explicitStatus = String(row.Run_Quality_Status || '').trim();
  const explicitReview = truthyReportValue(row.Requires_Manual_Review);
  const explicitUnfinalized = falseyReportValue(row.Attendance_Finalized);
  const detected = numberValue(row.Total_Face_Detections || row.Detected_Faces);
  const accepted = numberValue(row.Accepted_Recognitions);
  const total = numberValue(row.Total_Students);
  const present = numberValue(row.Students_Present);
  const review = numberValue(row.Students_Needs_Review);
  const absent = numberValue(row.Students_Absent);
  const recognitionRate = detected ? Math.round((accepted / detected) * 1000) / 10 : 0;
  const poor = String(row.Poor_Checkpoints || '').split(',').map((item) => item.trim()).filter(Boolean);
  const inferredLowQuality = detected >= 50 && recognitionRate < 10 && present === 0 && review === 0 && absent >= Math.max(10, Math.ceil(total * 0.8));
  const requiresManualReview = explicitReview || explicitUnfinalized || inferredLowQuality;
  return {
    requiresManualReview,
    finalized: !requiresManualReview,
    status: explicitStatus || (requiresManualReview ? 'Needs Review - Low Recognition Quality' : 'Valid'),
    reason: row.Run_Quality_Reason || (requiresManualReview ? 'Attendance requires review because recognition quality was too low. The system detected faces but could not confidently identify enough students.' : ''),
    recognitionRate,
    poorCheckpoints: poor.join(', '),
  };
}

function buildCleanSlotSummary(row) {
  if (!row) return null;
  const start = getSlotValue(row, ['Start_Time', 'Start Time']);
  const end = getSlotValue(row, ['End_Time', 'End Time']);
  const checkpoints = getSlotValue(row, ['Total_Checkpoints']);
  const mode = getSlotValue(row, ['Checkpoint_Mode']);
  const isDemo = Number(checkpoints) <= 1 || /demo/i.test(String(mode));

  return {
    modeLabel: isDemo ? 'Demo Clip Mode' : 'Full Checkpoint Mode',
    modeText: isDemo
      ? 'Short sample clips are being used. This report is for recognition testing and presentation.'
      : 'CP1–CP5 checkpoint voting is active for full attendance decisions.',
    title: `${getSlotValue(row, ['Course_Abbr'])} · ${getSlotValue(row, ['Course_Name'])}`,
    subtitle: `${getSlotValue(row, ['Day'])} · P${getSlotValue(row, ['Period'])} · ${start}-${end}`,
    faculty: getSlotValue(row, ['Instructor']),
    room: `${getSlotValue(row, ['Room'])} · Section ${getSlotValue(row, ['Section'])}`,
    runtime: minutesText(getSlotValue(row, ['Processing_Runtime_Seconds'])),
    checkpoints: `${checkpoints} ${getSlotValue(row, ['Checkpoint_Times']) !== '—' ? `(${getSlotValue(row, ['Checkpoint_Times'])})` : ''}`,
    cameras: getSlotValue(row, ['Camera_Angles_Processed']),
    videos: getSlotValue(row, ['Videos_Processed']),
    totalFaces: getSlotValue(row, ['Total_Face_Detections']),
    accepted: getSlotValue(row, ['Accepted_Recognitions']),
    rejected: getSlotValue(row, ['Rejected_Or_Unknown']),
  };
}

async function publicCsvToRows(url) {
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) throw new Error(`Could not load ${url} (${response.status})`);
  const text = await response.text();
  return parseCsv(text);
}

function isReviewCase(row) {
  const status = getStatus(row);
  const flags = String(row.Flags || row.Flag || '').trim();
  const count = getDetectionCount(row);
  return /review|weak|low confidence|late|early/i.test(`${status} ${flags}`) || (flags && flags !== '—') || (/absent/i.test(status) && count > 0);
}

function MetricCard({ label, value, hint, tone = 'neutral' }) {
  const tones = {
    success: { background: '#ecfdf5', borderColor: '#bbf7d0' },
    warning: { background: '#fffbeb', borderColor: '#fde68a' },
    danger: { background: '#fff1f2', borderColor: '#fecaca' },
    info: { background: '#eff6ff', borderColor: '#bfdbfe' },
  };
  return (
    <div style={merge(S.metric, tones[tone])}>
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

function CsvPicker({ title, hint, onFile }) {
  return (
    <label style={S.fileCard}>
      <input
        type="file"
        accept=".csv"
        style={{ display: 'none' }}
        onChange={(event) => onFile(event.target.files?.[0] || null)}
      />
      <span style={S.fileIcon}>⇪</span>
      <span style={{ minWidth: 0 }}>
        <strong style={{ display: 'block', color: '#0f172a', fontSize: 14, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{title}</strong>
        <small style={{ display: 'block', color: '#64748b', fontSize: 12, marginTop: 3, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{hint}</small>
      </span>
    </label>
  );
}

export default function Reports() {
  const { currentUser } = useAuth();
  const admin = isAdmin(currentUser);
  const [attendanceRows, setAttendanceRows] = useState([]);
  const [slotSummaryRows, setSlotSummaryRows] = useState([]);
  const [logRows, setLogRows] = useState([]);
  const [fileNames, setFileNames] = useState({});
  const [search, setSearch] = useState('');
  const [viewMode, setViewMode] = useState('all');
  const [sortMode, setSortMode] = useState('roll_asc');
  const [selectedStudent, setSelectedStudent] = useState(null);
  const [demoManifest, setDemoManifest] = useState(null);
  const [demoStatus, setDemoStatus] = useState('checking');
  const [demoMessage, setDemoMessage] = useState('Checking demo report files...');
  const [loadingDemo, setLoadingDemo] = useState(false);
  const [showTech, setShowTech] = useState(false);
  const [showUpload, setShowUpload] = useState(false);
  const [reportScopeMessage, setReportScopeMessage] = useState('');

  useEffect(() => {
    let cancelled = false;
    async function loadManifest() {
      try {
        const response = await fetch(DEMO_MANIFEST_URL, { cache: 'no-store' });
        if (!response.ok) {
          if (!cancelled) {
            setDemoStatus('missing');
            setDemoMessage('Demo report files are not prepared yet. Upload CSVs manually or run the demo prepare script.');
          }
          return;
        }
        const manifest = await response.json();
        let summary = [];
        if (manifest?.files?.slot_summary) {
          summary = await publicCsvToRows(manifest.files.slot_summary);
        }
        const scope = validateReportScope(currentUser, [], summary);
        if (!cancelled) {
          setDemoManifest(manifest);
          if (scope.allowed) {
            setDemoStatus('ready');
            setDemoMessage(scope.subjects.length
              ? `Demo report ready for ${scope.subjects.join(', ')}.`
              : 'Demo report files are ready.');
          } else {
            setDemoStatus('incompatible');
            setDemoMessage(`${scope.reason} Ask HOD to prepare a demo for your assigned subject.`);
          }
        }
      } catch (error) {
        if (!cancelled) {
          setDemoStatus('missing');
          setDemoMessage(error.message || 'Demo files are not ready yet.');
        }
      }
    }
    loadManifest();
    return () => { cancelled = true; };
  }, [currentUser]);

  const loadFile = async (kind, file) => {
    if (!file) return;
    const rows = await fileToRows(file);
    const nextAttendance = kind === 'attendance' ? rows : attendanceRows;
    const nextSummary = kind === 'summary' ? rows : slotSummaryRows;
    const scope = validateReportScope(currentUser, nextAttendance, nextSummary);
    if (!scope.allowed) {
      setReportScopeMessage(scope.reason);
      return;
    }

    setReportScopeMessage(scope.reason || '');
    setFileNames((prev) => ({ ...prev, [kind]: file.name }));
    if (kind === 'attendance') setAttendanceRows(rows);
    if (kind === 'summary') setSlotSummaryRows(rows);
    if (kind === 'log') setLogRows(rows);
    setSelectedStudent(null);
  };

  const loadDemoFiles = async () => {
    if (!demoManifest?.files?.attendance) {
      setDemoMessage('Demo files are missing. Run scripts\\prepare_demo_report_files.py first.');
      return;
    }
    setLoadingDemo(true);
    setDemoMessage('Loading demo report files...');
    try {
      const files = demoManifest.files;
      const attendance = await publicCsvToRows(files.attendance);
      const summary = files.slot_summary ? await publicCsvToRows(files.slot_summary) : [];
      const log = files.detection_log ? await publicCsvToRows(files.detection_log) : [];
      const scope = validateReportScope(currentUser, attendance, summary);
      if (!scope.allowed) {
        setReportScopeMessage(scope.reason);
        setDemoMessage(scope.reason);
        setDemoStatus('incompatible');
        return;
      }
      setReportScopeMessage('');
      setAttendanceRows(attendance);
      setSlotSummaryRows(summary);
      setLogRows(log);
      setFileNames({
        attendance: files.attendance?.split('/').pop() || 'demo attendance',
        summary: files.slot_summary?.split('/').pop() || 'demo summary',
        log: files.detection_log?.split('/').pop() || 'demo detection log',
      });
      setSearch('');
      setViewMode('all');
      setSelectedStudent(null);
      setShowUpload(false);
      setDemoMessage('Demo report loaded.');
      setDemoStatus('ready');
    } catch (error) {
      setDemoMessage(error.message || 'Failed to load demo report files.');
      setDemoStatus('missing');
    } finally {
      setLoadingDemo(false);
    }
  };

  const clearLoadedFiles = () => {
    setAttendanceRows([]);
    setSlotSummaryRows([]);
    setLogRows([]);
    setFileNames({});
    setSearch('');
    setSelectedStudent(null);
    setShowTech(false);
    setReportScopeMessage('');
  };

  const visibleAttendanceRows = useMemo(() => filterStudentRowsForUser(attendanceRows, currentUser), [attendanceRows, currentUser]);
  const attendanceSummary = useMemo(() => {
    const result = classifyReviewRows(visibleAttendanceRows, {}, {
      statusOf: (row) => getStatus(row),
      isPresent: (status) => /present|yes/i.test(String(status || '')),
      isEvidenceReview: isReviewCase,
    });
    return { ...result, percentage: result.pct };
  }, [visibleAttendanceRows]);
  const logSummary = useMemo(() => summarizeDetectionLog(logRows), [logRows]);
  const cleanSlot = buildCleanSlotSummary(slotSummaryRows[0]);
  const reportQuality = buildReportQuality(slotSummaryRows[0]);
  const hasReport = visibleAttendanceRows.length > 0 || slotSummaryRows.length > 0;
  const reportLoadedButHidden = attendanceRows.length > 0 && visibleAttendanceRows.length === 0;

  const filteredRows = useMemo(() => {
    let output = visibleAttendanceRows.filter((row) => getRoll(row).toLowerCase().includes(search.toLowerCase()));
    output = output.filter((row) => rowMatchesReviewMode(row, viewMode, {}, {
      statusOf: (item) => getStatus(item),
      isPresent: (status) => /present|yes/i.test(String(status || '')),
      isEvidenceReview: isReviewCase,
    }));
    output.sort((a, b) => {
      if (sortMode === 'roll_desc') return -rollCompare(a, b);
      if (sortMode === 'status') return getStatus(a).localeCompare(getStatus(b)) || rollCompare(a, b);
      if (sortMode === 'detections') return getDetectionCount(b) - getDetectionCount(a);
      return rollCompare(a, b);
    });
    return output;
  }, [visibleAttendanceRows, search, viewMode, sortMode]);

  return (
    <div style={S.page}>
      <PageHeader
        eyebrow="Attendance Output"
        title={admin ? 'Attendance Reports' : 'My Reports'}
        subtitle={admin ? 'View attendance summaries, student evidence, and optional technical processing details.' : 'Reports are filtered to your assigned subject and students.'}
        actions={hasReport ? <button type="button" style={S.secondaryBtn} onClick={clearLoadedFiles}>Clear report</button> : null}
      />

      <section style={merge(S.card, S.sourceCard)}>
        <div style={S.sourceCopy}>
          <span style={S.icon}>📊</span>
          <div>
            <p style={S.eyebrow}>Report source</p>
            <h2 style={S.h2}>{admin ? 'Start with demo report or CSV upload' : 'Load an assigned-subject report'}</h2>
            <p style={S.muted}>{admin ? 'Use the demo for presentation. Use CSV upload when checking a generated attendance run.' : 'Only reports matching your assigned subject can be opened. Use a compatible demo or upload your generated CSV.'}</p>
            <span style={merge(S.statusLine, demoStatus === 'ready' && S.statusReady, (demoStatus === 'missing' || demoStatus === 'incompatible') && S.statusMissing)}>{demoMessage}</span>
          </div>
        </div>
        <div style={S.actions}>
          <button type="button" style={merge(S.primaryBtn, (demoStatus !== 'ready' || loadingDemo) && { opacity: 0.55, cursor: 'not-allowed' })} onClick={loadDemoFiles} disabled={demoStatus !== 'ready' || loadingDemo}>
            {loadingDemo ? 'Loading…' : demoStatus === 'incompatible' ? 'Demo not assigned to you' : 'Load Demo Report'}
          </button>
          <button type="button" style={S.secondaryBtn} onClick={() => setShowUpload((value) => !value)}>{showUpload ? 'Hide CSV Upload' : 'Upload CSV Manually'}</button>
        </div>
      </section>

      {reportScopeMessage && (
        <section style={merge(S.card, { borderColor: '#fecaca', background: '#fff7f7' })}>
          <div style={S.empty}>
            <span style={{ ...S.emptyIcon, background: '#fff1f2' }}>⚠️</span>
            <div>
              <h3 style={S.h3}>Report blocked by faculty scope</h3>
              <p style={S.muted}>{reportScopeMessage}</p>
            </div>
          </div>
        </section>
      )}

      {showUpload && (
        <section style={S.card}>
          <p style={S.eyebrow}>Manual upload</p>
          <h3 style={S.h3}>Upload generated report files</h3>
          <p style={S.muted}>Final attendance CSV is enough for the table. Slot summary and detection log improve the overview and evidence numbers.</p>
          <div style={S.uploadGrid}>
            <CsvPicker title="Final attendance CSV" hint={fileNames.attendance || 'attendance_*.csv'} onFile={(file) => loadFile('attendance', file)} />
            <CsvPicker title="Slot summary CSV" hint={fileNames.summary || 'slot_summary_*.csv'} onFile={(file) => loadFile('summary', file)} />
            <CsvPicker title="Detection log CSV" hint={fileNames.log || 'detection_log_*.csv'} onFile={(file) => loadFile('log', file)} />
          </div>
        </section>
      )}

      {!hasReport && !reportLoadedButHidden && (
        <section style={S.card}>
          <div style={S.empty}>
            <span style={S.emptyIcon}>📈</span>
            <div>
              <h3 style={S.h3}>No report loaded yet</h3>
              <p style={S.muted}>{demoStatus === 'ready' ? 'Load the compatible demo report, or upload generated CSV files for your assigned subject.' : 'Upload generated CSV files for your assigned subject. The prepared demo is not available for this login.'}</p>
            </div>
          </div>
        </section>
      )}

      {reportLoadedButHidden && (
        <section style={S.card}>
          <div style={S.empty}>
            <span style={{ ...S.emptyIcon, background: '#fff7ed' }}>⚠️</span>
            <div>
              <h3 style={S.h3}>This report has no visible records for your login</h3>
              <p style={S.muted}>You loaded a report, but it does not match your assigned subject/student list. Admin can see all records.</p>
            </div>
          </div>
        </section>
      )}

      {hasReport && cleanSlot && (
        <section style={S.card}>
          <div style={S.topGrid}>
            <div>
              <p style={S.eyebrow}>{cleanSlot.modeLabel}</p>
              <h2 style={S.h2}>{cleanSlot.title}</h2>
              <p style={S.muted}>{cleanSlot.subtitle} · Faculty: {cleanSlot.faculty} · {cleanSlot.room}</p>
            </div>
            <button type="button" style={S.secondaryBtn} onClick={() => setShowTech((x) => !x)}>{showTech ? 'Hide technical details' : 'Show technical details'}</button>
          </div>

          {reportQuality.requiresManualReview && (
            <div style={S.qualityWarning}>
              <strong>{reportQuality.status}</strong>
              <p style={{ margin: '6px 0 0' }}>{reportQuality.reason}</p>
              <small>Raw calculated counts are shown below, but this attendance is unfinalized until Review Students is completed.</small>
            </div>
          )}

          <div style={S.metricGrid}>
            <MetricCard label="Students" value={attendanceSummary.total} hint={admin ? 'All loaded students' : 'Filtered by faculty'} tone="info" />
            <MetricCard label="Present" value={attendanceSummary.present} hint={reportQuality.finalized ? `${attendanceSummary.percentage}% attendance` : 'Raw calculated result'} tone="success" />
            <MetricCard label="Need review" value={visibleAttendanceRows.filter(isReviewCase).length} hint={reportQuality.requiresManualReview ? 'Manual review required' : 'Doubtful / flagged'} tone="warning" />
            <MetricCard label="Absent" value={attendanceSummary.absent} hint={reportQuality.finalized ? 'System result' : 'Raw unfinalized result'} tone="danger" />
          </div>

          {showTech && (
            <div style={S.techGrid}>
              <span style={S.techItem}>Runtime<br /><b>{cleanSlot.runtime}</b></span>
              <span style={S.techItem}>Checkpoints<br /><b>{cleanSlot.checkpoints}</b></span>
              <span style={S.techItem}>Cameras<br /><b>{cleanSlot.cameras}</b></span>
              <span style={S.techItem}>Videos<br /><b>{cleanSlot.videos}</b></span>
              <span style={S.techItem}>Total faces<br /><b>{cleanSlot.totalFaces}</b></span>
              <span style={S.techItem}>Accepted<br /><b>{cleanSlot.accepted || logSummary.accepted}</b></span>
              <span style={S.techItem}>Rejected/unknown<br /><b>{cleanSlot.rejected || logSummary.rejected}</b></span>
              <span style={S.techItem}>Run quality<br /><b>{reportQuality.status}</b></span>
              <span style={S.techItem}>Mode<br /><b>{cleanSlot.modeText}</b></span>
            </div>
          )}
        </section>
      )}

      {hasReport && (
        <section style={S.card}>
          <div style={S.topGrid}>
            <div>
              <h3 style={S.h3}>{reportQuality.requiresManualReview ? 'Raw calculated attendance' : 'Final attendance'}</h3>
              <p style={S.muted}>{reportQuality.requiresManualReview ? 'Search, filter, and review raw evidence before treating this as official.' : 'Search, filter, and open evidence for the selected student.'}</p>
            </div>
          </div>

          <div style={S.toolbar}>
            <input style={S.input} value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search roll number" />
            <div style={S.pillRow}>
              <ActiveButton active={viewMode === 'all'} onClick={() => setViewMode('all')}>All</ActiveButton>
              <ActiveButton active={viewMode === 'review'} onClick={() => setViewMode('review')}>Need review</ActiveButton>
              <ActiveButton active={viewMode === 'present'} onClick={() => setViewMode('present')}>Present</ActiveButton>
              <ActiveButton active={viewMode === 'absent'} onClick={() => setViewMode('absent')}>Absent</ActiveButton>
            </div>
            <div style={S.pillRow}>
              <ActiveButton active={sortMode === 'roll_asc'} onClick={() => setSortMode('roll_asc')}>Roll ↑</ActiveButton>
              <ActiveButton active={sortMode === 'roll_desc'} onClick={() => setSortMode('roll_desc')}>Roll ↓</ActiveButton>
              <ActiveButton active={sortMode === 'status'} onClick={() => setSortMode('status')}>Status</ActiveButton>
              <ActiveButton active={sortMode === 'detections'} onClick={() => setSortMode('detections')}>Evidence</ActiveButton>
            </div>
          </div>

          <div style={S.tableWrap}>
            <table style={S.table}>
              <thead>
                <tr>
                  <th style={S.th}>Roll Number</th>
                  <th style={S.th}>Status</th>
                  <th style={S.th}>Checkpoints</th>
                  <th style={S.th}>Evidence</th>
                  <th style={S.th}>Cameras</th>
                  <th style={S.th}>Action</th>
                </tr>
              </thead>
              <tbody>
                {filteredRows.map((row) => (
                  <tr key={getRoll(row)} style={isReviewCase(row) ? { background: '#fffbeb' } : null}>
                    <td style={S.td}><strong style={{ color: '#0f172a' }}>{getRoll(row)}</strong></td>
                    <td style={S.td}><StatusBadge status={getStatus(row)} /></td>
                    <td style={S.td}>{row.Recognized_Checkpoints || 0}/{row.Total_Checkpoints || cleanSlot?.checkpoints || '—'}</td>
                    <td style={S.td}>{getDetectionCount(row)} detections · avg {getAvgScore(row) || '—'} · best {getBestScore(row) || '—'}</td>
                    <td style={S.td}>{row.Cameras_Seen || row.Videos_Seen || '—'}</td>
                    <td style={S.td}><button type="button" style={S.smallBtn} onClick={() => setSelectedStudent(row)}>View Evidence</button></td>
                  </tr>
                ))}
                {filteredRows.length === 0 && <tr><td style={{ ...S.td, textAlign: 'center', padding: 28 }} colSpan="6">No students match this filter.</td></tr>}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {selectedStudent && (
        <div style={S.drawer} role="dialog" aria-modal="true">
          <div style={S.drawerCard}>
            <div style={S.topGrid}>
              <div>
                <p style={S.eyebrow}>Student evidence</p>
                <h2 style={S.h2}>{getRoll(selectedStudent)}</h2>
                <div style={{ marginTop: 10 }}><StatusBadge status={getStatus(selectedStudent)} /></div>
              </div>
              <button type="button" style={S.secondaryBtn} onClick={() => setSelectedStudent(null)}>Close</button>
            </div>
            <div style={S.metricGrid}>
              <MetricCard label="Checkpoints" value={`${selectedStudent.Recognized_Checkpoints || 0}/${selectedStudent.Total_Checkpoints || '—'}`} />
              <MetricCard label="Accepted detections" value={getDetectionCount(selectedStudent)} />
              <MetricCard label="Average score" value={getAvgScore(selectedStudent) || '—'} />
              <MetricCard label="Best score" value={getBestScore(selectedStudent) || '—'} />
            </div>
            <p style={S.muted}>Cameras: {selectedStudent.Cameras_Seen || '—'} · Flags: {selectedStudent.Flags || '—'}</p>
          </div>
        </div>
      )}
    </div>
  );
}
