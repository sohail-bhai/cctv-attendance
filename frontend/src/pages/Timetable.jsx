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
    setMessage({ type: 'info', text: `${action === 'reprocess' ? 'Reprocessing' : 'Starting'} ${day} ${p} ${subject}...` });
    try {
      const data = await apiPost(action === 'reprocess' ? '/api/reprocess' : '/api/process', {
        day,
        period: p,
        subject_track: subject,
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

  const latestJobForRow = (row) => jobs.find((job) => job.day === row.day && normalizePeriod(job.period) === normalizePeriod(row.period));

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
                <strong>{job.day} {job.period}</strong>
                <span>{job.progress_text || 'Preparing attendance job...'}</span>
                <div className="processing-steps">
                  {['Checking files', 'Processing clips', 'Recognizing students', 'Generating report'].map((step, index) => (
                    <i key={step} className={index === 0 || /processing|recognizing|generating|completed/i.test(job.progress_text || '') ? 'done' : ''}>{step}</i>
                  ))}
                </div>
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
                const state = isLunch ? 'Lunch' : liveState || row.att_status?.status || 'Pending';
                const report = row.att_status?.class_report_file;
                const running = job && (job.status === 'Processing' || job.status === 'Pending');
                return (
                  <tr key={`${row.day}-${row.period}-${subject}`}>
                    <td><strong>{row.period}</strong></td>
                    <td>{subject}<small>{info?.courseCode || row.course_code} · {info?.courseName || row.course_name}</small></td>
                    <td>{info?.facultyName || row.teacher || row.instructor || '-'}</td>
                    <td>{row.start_time}–{row.end_time}</td>
                    <td>{row.room}</td>
                    <td><StatusBadge status={state} /></td>
                    <td>{job ? <div className="job-progress-cell"><strong>{job.job_id}</strong><small>{job.progress_text || job.status}</small>{running && <small>Timer: {elapsedFromText(job.started_at || job.created_at)}</small>}{job.error && <small className="error-text">{job.error}</small>}</div> : '—'}</td>
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
