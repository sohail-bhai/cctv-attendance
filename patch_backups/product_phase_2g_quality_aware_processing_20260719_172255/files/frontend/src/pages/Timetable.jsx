import { useCallback, useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import PageHeader from '../components/PageHeader.jsx';
import StatusBadge from '../components/StatusBadge.jsx';
import NiceSelect from '../components/NiceSelect.jsx';
import JobProgress from '../components/JobProgress.jsx';
import { apiCancelJob, apiGetResult, apiPost } from '../api/client.js';
import { DAYS, FALLBACK_TIMETABLE } from '../data/fallback.js';
import { fileToRows } from '../utils/csv.js';
import { useAuth } from '../auth/AuthContext.jsx';
import { canAccessRow, displaySubjectForUser, isAdmin, subjectInfo } from '../data/users.js';
import { useBackendHealth } from '../health/BackendHealthContext.jsx';
import {
  actionForSlot,
  deriveSlotState,
  elapsedFromText,
  isRunningJob,
  latestJobForRow,
  normalizePeriod,
} from '../utils/workflow.js';

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

function qualityReason(entry) {
  return entry?.review_queue_note
    || entry?.run_quality_reason
    || 'Attendance requires review because recognition quality was too low.';
}

export default function Timetable() {
  const { currentUser } = useAuth();
  const admin = isAdmin(currentUser);
  const health = useBackendHealth();
  const today = new Date().toLocaleDateString('en-US', { weekday: 'long' });
  const [day, setDay] = useState(DAYS.includes(today) ? today : 'Monday');
  const [rows, setRows] = useState([]);
  const [jobs, setJobs] = useState([]);
  const [message, setMessage] = useState(null);
  const [localPreview, setLocalPreview] = useState([]);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [rowsSource, setRowsSource] = useState('loading');
  const [jobsAvailable, setJobsAvailable] = useState(false);
  const [pendingAction, setPendingAction] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [, forceTick] = useState(0);

  const loadRows = useCallback(async () => {
    setIsRefreshing(true);
    const result = await apiGetResult(`/api/timetable?day=${encodeURIComponent(day)}`, { cache: 'no-store' });
    if (result.ok && Array.isArray(result.data?.timetable)) {
      setRows(result.data.timetable);
      setRowsSource('live');
    } else {
      setRows(FALLBACK_TIMETABLE.filter((row) => row.day === day && canAccessRow(currentUser, row)));
      setRowsSource('fallback');
    }
    setLastUpdated(new Date());
    setIsRefreshing(false);
    return result;
  }, [currentUser, day]);

  const loadJobs = useCallback(async () => {
    const result = await apiGetResult('/api/status', { cache: 'no-store' });
    if (result.ok && Array.isArray(result.data)) {
      setJobs(result.data);
      setJobsAvailable(true);
      return result.data;
    }
    setJobsAvailable(false);
    return [];
  }, []);

  const refreshAll = useCallback(async () => {
    await Promise.all([loadRows(), loadJobs(), health.refreshHealth()]);
  }, [health, loadJobs, loadRows]);

  useEffect(() => {
    loadRows();
    loadJobs();
  }, [loadJobs, loadRows]);

  const hasProcessing = useMemo(() => jobs.some(isRunningJob), [jobs]);

  useEffect(() => {
    if (!hasProcessing || !health.connected) return undefined;
    const id = window.setInterval(async () => {
      const latest = await loadJobs();
      if (!latest.some(isRunningJob)) await loadRows();
    }, 3000);
    return () => window.clearInterval(id);
  }, [hasProcessing, health.connected, loadJobs, loadRows]);

  useEffect(() => {
    if (!hasProcessing) return undefined;
    const id = window.setInterval(() => forceTick((value) => value + 1), 1000);
    return () => window.clearInterval(id);
  }, [hasProcessing]);

  const canStartProcessing = health.connected && health.ready && rowsSource === 'live' && jobsAvailable;
  const canControlJobs = health.connected && jobsAvailable;

  const start = async (row, action) => {
    if (!canStartProcessing) {
      setMessage({ type: 'error', text: 'Attendance processing is disabled until the live backend is connected and ready.' });
      return;
    }
    if (action === 'reprocess' && !window.confirm('Reprocess this session? Existing results remain auditable, but a new processing job will be started.')) return;

    const period = normalizePeriod(row.period);
    const subject = displaySubjectForUser(row, currentUser);
    const info = subjectInfo(subject);
    const actionKey = `${row.session_id || `${day}-${period}-${subject}`}:${action}`;
    setPendingAction(actionKey);
    setMessage({ type: 'info', text: `${action === 'reprocess' ? 'Reprocessing' : 'Starting'} ${day} ${period} ${subject}…` });
    try {
      const data = await apiPost(action === 'reprocess' ? '/api/reprocess' : '/api/process', {
        day,
        period,
        subject_track: subject,
        session_id: row.session_id,
        session_date: row.session_date,
        section: row.section,
        slot_id: row.slot_id,
        course_code: info?.courseCode || row.course_code,
        subject_name: info?.courseName || row.course_name,
        ...FAST_PROCESSING_PAYLOAD,
      });
      const reused = data.job_created === false;
      setMessage({
        type: reused || data.already_done ? 'info' : 'success',
        text: data.message || (reused ? `Using existing job ${data.job_id}.` : `Background job started: ${data.job_id}`),
      });
      await refreshAll();
    } catch (error) {
      setMessage({ type: 'error', text: error.message });
    } finally {
      setPendingAction(null);
    }
  };

  const cancelJob = async (jobId) => {
    if (!canControlJobs || !jobId) {
      setMessage({ type: 'error', text: 'Cancellation is unavailable while the backend is offline or not ready.' });
      return;
    }
    if (!window.confirm(`Cancel job ${jobId}?`)) return;
    setPendingAction(`cancel:${jobId}`);
    setMessage({ type: 'info', text: `Cancelling job ${jobId}…` });
    try {
      const data = await apiCancelJob(jobId);
      setMessage({ type: 'success', text: data.message || `Cancelled job ${jobId}.` });
      await refreshAll();
    } catch (error) {
      setMessage({ type: 'error', text: error.message });
    } finally {
      setPendingAction(null);
    }
  };

  const handleTimetableCsv = async (file) => {
    if (!file) return;
    try {
      const parsed = await fileToRows(file);
      setLocalPreview(parsed);
      setMessage({ type: 'success', text: `Loaded ${parsed.length} timetable rows locally for preview.` });
    } catch (error) {
      setMessage({ type: 'error', text: `Could not read CSV: ${error.message}` });
    }
  };

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow={admin ? 'HOD Attendance Processing' : 'Faculty Attendance Processing'}
        title={admin ? 'Take Attendance' : 'My Classes'}
        subtitle={admin
          ? 'Start, monitor, cancel, review, or reprocess a session from one controlled workflow.'
          : 'Only your assigned subject slots are shown. Every job remains tied to its exact session.'}
        actions={
          <div className="page-actions">
            <NiceSelect compact value={day} onChange={setDay} options={DAYS} />
            <button className="button secondary" type="button" onClick={refreshAll} disabled={isRefreshing}>
              {isRefreshing ? 'Refreshing…' : 'Refresh'}
            </button>
          </div>
        }
      />

      {message && <div className={`notice ${message.type}`}>{message.text}</div>}

      {rowsSource === 'fallback' && (
        <section className="connection-banner offline">
          <div>
            <strong>Local timetable preview only</strong>
            <p>The backend timetable or job API is unavailable. Class rows are shown for reference, but Process, Reprocess, and Cancel are disabled.</p>
          </div>
          <button className="button tiny secondary" type="button" onClick={refreshAll}>Retry connection</button>
        </section>
      )}

      {health.connected && !health.ready && (
        <section className="connection-banner warning">
          <div>
            <strong>Backend connected, processing unavailable</strong>
            <p>Required files are missing: {health.missing.join(', ') || 'unknown readiness dependency'}.</p>
          </div>
        </section>
      )}

      <section className="workflow-readiness-grid">
        <article className={`readiness-card ${rowsSource === 'live' ? 'ready' : 'blocked'}`}>
          <span>1</span><div><strong>Session data</strong><small>{rowsSource === 'live' ? 'Live timetable loaded' : 'Local preview only'}</small></div>
        </article>
        <article className={`readiness-card ${health.ready ? 'ready' : 'blocked'}`}>
          <span>2</span><div><strong>Recognition backend</strong><small>{health.ready ? 'Models and workflow ready' : 'Unavailable or incomplete'}</small></div>
        </article>
        <article className={`readiness-card ${jobsAvailable ? 'ready' : 'blocked'}`}>
          <span>3</span><div><strong>Persistent jobs</strong><small>{jobsAvailable ? 'Job registry available' : 'Job history unavailable'}</small></div>
        </article>
      </section>

      {hasProcessing && jobsAvailable && (
        <section className="panel processing-panel">
          <div className="panel-title-row">
            <div>
              <h3>Attendance processing</h3>
              <p className="muted">The page polls only job status while processing is active.</p>
            </div>
            <span className="soft-pill live">{jobs.filter(isRunningJob).length} active</span>
          </div>
          <div className="processing-grid">
            {jobs.filter(isRunningJob).slice(0, 4).map((job) => (
              <article className="processing-card" key={job.job_id}>
                <div className="job-card-heading">
                  <div>
                    <strong>{job.session_date || job.day} · {job.period} · {job.subject_abbr || job.subject || ''}</strong>
                    <small>{job.controlled_by || job.faculty_name || 'Authorized user'} · {job.job_id}</small>
                  </div>
                  <StatusBadge status={job.status} />
                </div>
                <JobProgress job={job} />
                <div className="row-actions spread-actions">
                  <small>Elapsed: {elapsedFromText(job.started_at || job.created_at)}</small>
                  <button
                    className="button tiny danger"
                    type="button"
                    disabled={!canControlJobs || pendingAction === `cancel:${job.job_id}`}
                    onClick={() => cancelJob(job.job_id)}
                  >
                    {pendingAction === `cancel:${job.job_id}` ? 'Cancelling…' : 'Cancel'}
                  </button>
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
              <h3>Timetable CSV preview</h3>
              <p className="muted">Preview only. It does not replace the backend timetable or alter historical sessions.</p>
            </div>
            <label className="button secondary">Choose CSV<input hidden type="file" accept=".csv" onChange={(event) => handleTimetableCsv(event.target.files?.[0])} /></label>
          </div>
          {localPreview.length > 0 && (
            <div className="table-wrap mini-table">
              <table>
                <thead><tr>{Object.keys(localPreview[0]).slice(0, 9).map((heading) => <th key={heading}>{heading}</th>)}</tr></thead>
                <tbody>{localPreview.slice(0, 8).map((row, index) => <tr key={index}>{Object.keys(localPreview[0]).slice(0, 9).map((heading) => <td key={heading}>{row[heading]}</td>)}</tr>)}</tbody>
              </table>
            </div>
          )}
        </section>
      )}

      <section className="panel">
        <div className="panel-title-row">
          <div>
            <h3>{day} {admin ? 'sessions' : 'my sessions'}</h3>
            <p className="muted">{rowsSource === 'live' ? 'Live timetable, attendance state, and latest matching job' : 'Static local preview with all actions disabled'}</p>
          </div>
          <div className="source-meta">
            <span className={`data-source-pill ${rowsSource === 'live' ? 'live' : 'preview'}`}>{rowsSource === 'live' ? 'Live' : 'Preview'}</span>
            <small>{lastUpdated ? lastUpdated.toLocaleTimeString('en-IN') : 'Loading…'}</small>
          </div>
        </div>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Period</th><th>Subject</th><th>Faculty</th><th>Time</th><th>Room</th><th>State</th><th>Job Progress</th><th>Action</th></tr></thead>
            <tbody>
              {rows.map((row) => {
                const isLunch = /lunch/i.test(row.subject || '');
                const subject = displaySubjectForUser(row, currentUser);
                const info = subjectInfo(subject);
                const job = latestJobForRow(jobs, row);
                const state = rowsSource === 'live' ? deriveSlotState(row, job) : isLunch ? 'Lunch' : 'Preview';
                const action = rowsSource === 'live'
                  ? actionForSlot({ row, job, backendReady: isRunningJob(job) ? canControlJobs : canStartProcessing })
                  : { primary: 'none', disabled: true };
                const report = row.att_status?.class_report_file;
                const running = isRunningJob(job);
                const reviewSession = row.session_id || row.att_status?.session_id;
                const reviewHref = reviewSession ? `/manual-review?session_id=${encodeURIComponent(reviewSession)}` : '/manual-review';
                const actionKey = `${row.session_id || `${day}-${normalizePeriod(row.period)}-${subject}`}:${action.primary}`;
                return (
                  <tr key={row.session_id || `${row.day}-${row.period}-${subject}`} className={state === 'Needs Review' ? 'risk-row' : ''}>
                    <td><strong>{row.period}</strong></td>
                    <td>{subject}<small>{info?.courseCode || row.course_code} · {info?.courseName || row.course_name}</small></td>
                    <td>{info?.facultyName || row.teacher || row.instructor || '-'}</td>
                    <td>{row.start_time}–{row.end_time}</td>
                    <td>{row.room}</td>
                    <td><StatusBadge status={state} />{state === 'Needs Review' && <small>{qualityReason(row.att_status)}</small>}</td>
                    <td>{job
                      ? <div className="job-progress-cell"><strong>{job.job_id}</strong><JobProgress job={job} compact />{running && <small>Elapsed: {elapsedFromText(job.started_at || job.created_at)}</small>}{job.error && <small className="error-text">{job.error}</small>}</div>
                      : '—'}</td>
                    <td>
                      {isLunch || rowsSource !== 'live' ? '—' : (
                        <div className="row-actions">
                          {report && <a className="button tiny secondary" href={`/attendance_website/${report}`} target="_blank" rel="noreferrer">Report</a>}
                          {action.primary === 'review' && <Link className="button tiny warning" to={reviewHref}>Review</Link>}
                          {action.primary === 'cancel' && <button className="button tiny danger" type="button" disabled={action.disabled || pendingAction === `cancel:${job?.job_id}`} onClick={() => cancelJob(job?.job_id)}>Cancel</button>}
                          {action.primary === 'process' && <button className="button tiny" type="button" disabled={action.disabled || pendingAction === actionKey} onClick={() => start(row, 'process')}>Process</button>}
                          {action.primary === 'retry' && <button className="button tiny" type="button" disabled={action.disabled || pendingAction === actionKey} onClick={() => start(row, 'process')}>Retry</button>}
                          {(action.primary === 'reprocess' || action.secondary === 'reprocess') && <button className="button tiny secondary" type="button" disabled={action.disabled || pendingAction === actionKey} onClick={() => start(row, 'reprocess')}>Reprocess</button>}
                        </div>
                      )}
                    </td>
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
