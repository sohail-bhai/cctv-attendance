import { useEffect, useMemo, useState } from 'react';
import PageHeader from '../components/PageHeader.jsx';
import StatusBadge from '../components/StatusBadge.jsx';
import NiceSelect from '../components/NiceSelect.jsx';
import { apiCancelJob, apiGet, apiPost } from '../api/client.js';
import { DAYS, FALLBACK_TIMETABLE } from '../data/fallback.js';
import { fileToRows } from '../utils/csv.js';
import { useAuth } from '../auth/AuthContext.jsx';
import { canAccessRow, displaySubjectForUser, isAdmin, subjectInfo } from '../data/users.js';

const FAST_PROCESSING_PAYLOAD = {
  checkpoint_mode: 'auto',
  checkpoint_min_detections: 2,
  match_threshold: 0.48,
  margin_threshold: 0.08,
  sample_fps: 2,
  log_mode: 'accepted',
  output_layout: 'organized',
  save_unknown: false,
  timeout_seconds: 20 * 60,
};

function normalizePeriod(period) {
  const raw = String(period || '').toUpperCase();
  return raw.startsWith('P') ? raw : `P${raw}`;
}

function elapsedFromText(value) {
  if (!value) return '—';
  const parsed = Date.parse(String(value).replace(/(\d{2})-(\d{2})-(\d{4})/, '$3-$2-$1'));
  if (Number.isNaN(parsed)) return 'Running';
  const seconds = Math.max(0, Math.floor((Date.now() - parsed) / 1000));
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}:${String(secs).padStart(2, '0')}`;
}

const PROGRESS_STEPS = [
  { key: 'checking_files', label: 'Checking files' },
  { key: 'processing_clips', label: 'Processing camera clips' },
  { key: 'recognizing_students', label: 'Recognizing students' },
  { key: 'generating_report', label: 'Generating report' },
];

function clampNumber(value, min, max) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return min;
  return Math.max(min, Math.min(max, parsed));
}

function fallbackProgressStep(job) {
  const text = String(job?.progress_text || '').toLowerCase();
  const stage = String(job?.progress_stage || '').toLowerCase();
  if (job?.status === 'Completed' || stage === 'completed') return PROGRESS_STEPS.length;
  if (job?.status === 'Failed' || job?.status === 'Cancelled') return 0;
  if (stage.includes('generating') || text.includes('saved ') || text.includes('generating')) return 3;
  if (stage.includes('recognizing') || /^cp\d+/.test(text) || text.includes('processed') || text.includes('skipped')) return 2;
  if (stage.includes('processing') || text.includes('camera angle') || text.includes('camera ')) return 1;
  return 0;
}

function progressModel(job) {
  const activeStep = clampNumber(
    Number.isFinite(Number(job?.progress_step)) ? Number(job.progress_step) : fallbackProgressStep(job),
    0,
    PROGRESS_STEPS.length
  );
  const percent = clampNumber(
    Number.isFinite(Number(job?.progress_percent)) ? Number(job.progress_percent) : activeStep * 25,
    0,
    100
  );
  const status = String(job?.status || '');
  return {
    activeStep,
    percent,
    isComplete: status === 'Completed' || activeStep >= PROGRESS_STEPS.length,
    isError: status === 'Failed' || status === 'Cancelled',
  };
}

function stepStyle(state) {
  const base = {
    fontStyle: 'normal',
    borderRadius: 999,
    padding: '9px 12px',
    fontWeight: 800,
    fontSize: '0.82rem',
    border: '1px solid transparent',
    transition: 'all 160ms ease',
  };
  if (state === 'done') {
    return { ...base, background: '#dcfce7', color: '#166534', borderColor: '#bbf7d0' };
  }
  if (state === 'active') {
    return { ...base, background: '#e0f2fe', color: '#075985', borderColor: '#7dd3fc', boxShadow: '0 0 0 3px rgba(14, 165, 233, 0.12)' };
  }
  if (state === 'error') {
    return { ...base, background: '#fee2e2', color: '#991b1b', borderColor: '#fecaca' };
  }
  return { ...base, background: '#f1f5f9', color: '#64748b', borderColor: '#e2e8f0' };
}

function needsQualityReview(entry) {
  if (!entry) return false;
  const review = String(entry.requires_manual_review ?? entry.Requires_Manual_Review ?? '').toLowerCase();
  const finalized = String(entry.attendance_finalized ?? entry.Attendance_Finalized ?? '').toLowerCase();
  return entry.status === 'Needs Review' || review === 'true' || review === 'yes' || finalized === 'false' || finalized === 'no';
}

function qualityReason(entry) {
  return entry?.review_queue_note || entry?.run_quality_reason || 'Attendance requires review because recognition quality was too low. The system detected faces but could not confidently identify enough students.';
}

function LiveJobProgress({ job, compact = false }) {
  const model = progressModel(job);
  const latestLogs = Array.isArray(job?.log_tail) ? job.log_tail.slice(-3) : [];
  const headline = job?.progress_text || job?.status || 'Preparing attendance job...';

  return (
    <div style={{ display: 'grid', gap: compact ? 6 : 10 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 10 }}>
        <span style={{ color: '#475569', lineHeight: 1.35 }}>{headline}</span>
        <strong style={{ color: model.isError ? '#dc2626' : '#0f766e', whiteSpace: 'nowrap' }}>{Math.round(model.percent)}%</strong>
      </div>
      <div style={{ height: compact ? 7 : 10, background: '#e2e8f0', borderRadius: 999, overflow: 'hidden' }}>
        <div
          style={{
            width: `${model.percent}%`,
            height: '100%',
            background: model.isError ? '#ef4444' : 'linear-gradient(90deg, #0f766e, #14b8a6)',
            borderRadius: 999,
            transition: 'width 300ms ease',
          }}
        />
      </div>
      {!compact && (
        <>
          <div className="processing-steps">
            {PROGRESS_STEPS.map((step, index) => {
              let state = 'pending';
              if (model.isError) state = index === model.activeStep ? 'error' : 'pending';
              else if (model.isComplete || index < model.activeStep) state = 'done';
              else if (index === model.activeStep) state = 'active';
              return <i key={step.key} style={stepStyle(state)}>{state === 'done' ? '✓ ' : state === 'active' ? '● ' : ''}{step.label}</i>;
            })}
          </div>
          {latestLogs.length > 0 && (
            <div style={{ border: '1px solid #e2e8f0', borderRadius: 14, padding: 10, background: '#f8fafc' }}>
              <strong style={{ display: 'block', color: '#334155', fontSize: '0.78rem', marginBottom: 6 }}>Live log</strong>
              {latestLogs.map((line, index) => (
                <small key={`${line}-${index}`} style={{ display: 'block', color: '#64748b', lineHeight: 1.45, overflowWrap: 'anywhere' }}>› {line}</small>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default function Timetable() {
  const { currentUser } = useAuth();
  const admin = isAdmin(currentUser);
  const today = new Date().toLocaleDateString('en-US', { weekday: 'long' });
  const [day, setDay] = useState(DAYS.includes(today) ? today : 'Monday');
  const [rows, setRows] = useState([]);
  const [jobs, setJobs] = useState([]);
  const [message, setMessage] = useState(null);
  const [localPreview, setLocalPreview] = useState([]);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [, forceTick] = useState(0);

  const loadRows = async () => {
    setIsRefreshing(true);
    const data = await apiGet(`/api/timetable?day=${day}`, null);
    if (data?.timetable) setRows(data.timetable);
    else setRows(FALLBACK_TIMETABLE.filter((row) => row.day === day && canAccessRow(currentUser, row)));
    setIsRefreshing(false);
  };

  const loadJobs = async () => {
    const data = await apiGet('/api/status', []);
    setJobs(Array.isArray(data) ? data : []);
    return Array.isArray(data) ? data : [];
  };

  useEffect(() => {
    loadRows();
    loadJobs();
  }, [day, currentUser]);

  const hasProcessing = useMemo(() => jobs.some((job) => job.status === 'Processing' || job.status === 'Pending'), [jobs]);

  useEffect(() => {
    if (!hasProcessing) return undefined;
    const id = setInterval(async () => {
      const latest = await loadJobs();
      const stillRunning = latest.some((job) => job.status === 'Processing' || job.status === 'Pending');
      if (!stillRunning) loadRows();
    }, 3000);
    return () => clearInterval(id);
  }, [hasProcessing, day]);

  useEffect(() => {
    if (!hasProcessing) return undefined;
    const id = setInterval(() => forceTick((x) => x + 1), 1000);
    return () => clearInterval(id);
  }, [hasProcessing]);

  const start = async (row, action) => {
    const p = normalizePeriod(row.period);
    const subject = displaySubjectForUser(row, currentUser);
    const info = subjectInfo(subject);
    setMessage({ type: 'info', text: `${action === 'reprocess' ? 'Reprocessing' : 'Starting'} ${day} ${p} ${subject}...` });
    try {
      const data = await apiPost(action === 'reprocess' ? '/api/reprocess' : '/api/process', {
        day,
        period: p,
        subject_track: subject,
        session_id: row.session_id,
        session_date: row.session_date,
        section: row.section,
        slot_id: row.slot_id,
        course_code: info?.courseCode || row.course_code,
        subject_name: info?.courseName || row.course_name,
        ...FAST_PROCESSING_PAYLOAD,
      });
      setMessage({ type: 'success', text: data.message || `Background job started: ${data.job_id}` });
      await loadJobs();
      await loadRows();
    } catch (error) {
      setMessage({ type: 'error', text: error.message });
    }
  };

  const cancelJob = async (jobId) => {
    if (!jobId) return;
    setMessage({ type: 'info', text: `Cancelling job ${jobId}...` });
    try {
      const data = await apiCancelJob(jobId);
      setMessage({ type: 'success', text: data.message || `Cancelled job ${jobId}.` });
      await loadJobs();
      await loadRows();
    } catch (error) {
      setMessage({ type: 'error', text: error.message });
    }
  };

  const handleTimetableCsv = async (file) => {
    if (!file) return;
    const parsed = await fileToRows(file);
    setLocalPreview(parsed);
    setMessage({ type: 'success', text: `Loaded ${parsed.length} timetable rows locally for preview.` });
  };

  const latestJobForRow = (row) => jobs.find((job) => {
    if (row.session_id && job.session_id) return job.session_id === row.session_id;
    return job.day === row.day && normalizePeriod(job.period) === normalizePeriod(row.period) && (!job.subject_abbr || job.subject_abbr === (row.subject_track || row.subject));
  });

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow={admin ? 'Admin Attendance Processing' : 'Faculty Attendance Processing'}
        title={admin ? 'Take Attendance' : 'My Classes'}
        subtitle={admin ? 'Select any slot and process attendance.' : 'Only your subject slots are shown here. Simultaneous CCM/CVO periods are separated by faculty.'}
        actions={<NiceSelect compact value={day} onChange={setDay} options={DAYS} />}
      />

      {message && <div className={`notice ${message.type}`}>{message.text}</div>}

      <section className="panel split-cards">
        <div>
          <h3>Simple workflow</h3>
          <p className="muted">Choose a class period and click Process. The system automatically uses demo clips if only sample clips exist, or full checkpoint mode when CP1–CP5 folders are available.</p>
        </div>
        <div>
          <h3>{admin ? 'Admin access' : 'Faculty access'}</h3>
          <p className="muted">{admin ? 'You can see all classes and all students.' : `You can see only ${currentUser.subjects.join(', ')} classes and assigned students.`}</p>
        </div>
      </section>


      {hasProcessing && (
        <section className="panel processing-panel">
          <div className="panel-title-row">
            <div>
              <h3>Attendance processing in progress</h3>
              <p className="muted">You can keep this page open. The system is checking job status without refreshing the whole timetable repeatedly.</p>
            </div>
            <span className="soft-pill live">Live</span>
          </div>
          <div className="processing-grid">
            {jobs.filter((job) => job.status === 'Processing' || job.status === 'Pending').slice(0, 3).map((job) => (
              <article className="processing-card" key={job.job_id}>
                <strong>{job.session_date || job.day} {job.period} {job.subject_abbr || job.subject || ''}</strong>
                <LiveJobProgress job={job} />
                <div className="row-actions spread-actions">
                  <small>Elapsed: {elapsedFromText(job.started_at || job.created_at)}</small>
                  <button className="button tiny danger" onClick={() => cancelJob(job.job_id)}>Cancel</button>
                </div>
              </article>
            ))}
          </div>
        </section>
      )}

      {admin && (
        <section className="panel">
          <div className="panel-title-row">
            <div>
              <h3>Upload timetable CSV preview</h3>
              <p className="muted">Admin-only preview. To change backend timetable, replace timetable_b51_2026_2027.csv in the project root.</p>
            </div>
            <label className="button secondary">Upload CSV<input hidden type="file" accept=".csv" onChange={(event) => handleTimetableCsv(event.target.files?.[0])} /></label>
          </div>
          {localPreview.length > 0 && (
            <div className="table-wrap mini-table">
              <table>
                <thead><tr>{Object.keys(localPreview[0]).slice(0, 9).map((h) => <th key={h}>{h}</th>)}</tr></thead>
                <tbody>{localPreview.slice(0, 8).map((row, index) => <tr key={index}>{Object.keys(localPreview[0]).slice(0, 9).map((h) => <td key={h}>{row[h]}</td>)}</tr>)}</tbody>
              </table>
            </div>
          )}
        </section>
      )}

      <section className="panel">
        <div className="panel-title-row">
          <h3>{day} {admin ? 'Slots' : 'My Slots'}</h3>
          <span className="muted">{isRefreshing ? 'Refreshing...' : hasProcessing ? 'Checking job status' : 'Ready'}</span>
        </div>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Period</th><th>Subject</th><th>Faculty</th><th>Time</th><th>Room</th><th>State</th><th>Job Progress</th><th>Action</th></tr></thead>
            <tbody>
              {rows.map((row) => {
                const isLunch = /lunch/i.test(row.subject || '');
                const subject = displaySubjectForUser(row, currentUser);
                const info = subjectInfo(subject);
                const job = latestJobForRow(row);
                const liveState = job?.status === 'Processing' || job?.status === 'Pending' ? job.status : null;
                const qualityReview = needsQualityReview(row.att_status);
                const state = isLunch ? 'Lunch' : liveState || (qualityReview ? 'Needs Review' : row.att_status?.status) || 'Pending';
                const report = row.att_status?.class_report_file;
                const running = job && (job.status === 'Processing' || job.status === 'Pending');
                const reviewSession = row.session_id || row.att_status?.session_id;
                const reviewHref = reviewSession ? `/manual-review?session_id=${encodeURIComponent(reviewSession)}` : '/manual-review';
                return (
                  <tr key={row.session_id || `${row.day}-${row.period}-${subject}`}>
                    <td><strong>{row.period}</strong></td>
                    <td>{subject}<small>{info?.courseCode || row.course_code} · {info?.courseName || row.course_name}</small></td>
                    <td>{info?.facultyName || row.teacher || row.instructor || '-'}</td>
                    <td>{row.start_time}–{row.end_time}</td>
                    <td>{row.room}</td>
                    <td><StatusBadge status={state} /></td>
                    <td>{job ? <div className="job-progress-cell"><strong>{job.job_id}</strong><LiveJobProgress job={job} compact />{running && <small>Timer: {elapsedFromText(job.started_at || job.created_at)}</small>}{job.error && <small className="error-text">{job.error}</small>}</div> : '—'}</td>
                    <td>{isLunch ? '—' : <div className="row-actions">{report && <a className="button tiny secondary" href={`/attendance_website/${report}`} target="_blank" rel="noreferrer">Report</a>}{running ? <button className="button tiny danger" onClick={() => cancelJob(job.job_id)}>Cancel</button> : state === 'Completed' ? <button className="button tiny warning" onClick={() => start(row, 'reprocess')}>Reprocess</button> : <button className="button tiny" onClick={() => start(row, 'process')}>Process</button>}</div>}</td>
                  </tr>
                );
              })}
              {rows.length === 0 && <tr><td colSpan="8" className="empty-cell">No visible classes for this login.</td></tr>}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
