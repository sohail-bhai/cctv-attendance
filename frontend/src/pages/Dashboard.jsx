import { useCallback, useEffect, useMemo, useState } from 'react';
import PageHeader from '../components/PageHeader.jsx';
import StatCard from '../components/StatCard.jsx';
import StatusBadge from '../components/StatusBadge.jsx';
import NiceSelect from '../components/NiceSelect.jsx';
import { apiGetResult } from '../api/client.js';
import { DAYS, FALLBACK_TIMETABLE } from '../data/fallback.js';
import { useAuth } from '../auth/AuthContext.jsx';
import { canAccessRow, displaySubjectForUser, isAdmin, subjectInfo } from '../data/users.js';
import { studentsForUser } from '../data/students.js';
import { useBackendHealth } from '../health/BackendHealthContext.jsx';
import { isRunningJob } from '../utils/workflow.js';

function slotKey(row) {
  return row.session_id || `${row.day}_${row.period}_${row.subject}`;
}

function fallbackRows(day, user) {
  return FALLBACK_TIMETABLE.filter((row) => row.day === day && canAccessRow(user, row));
}

export default function Dashboard() {
  const { currentUser } = useAuth();
  const admin = isAdmin(currentUser);
  const health = useBackendHealth();
  const today = new Date().toLocaleDateString('en-US', { weekday: 'long' });
  const [selectedDay, setSelectedDay] = useState(DAYS.includes(today) ? today : 'Monday');
  const [timetable, setTimetable] = useState([]);
  const [jobs, setJobs] = useState([]);
  const [reports, setReports] = useState({ attendance_reports: [] });
  const [source, setSource] = useState('loading');
  const [jobsAvailable, setJobsAvailable] = useState(false);
  const [reportsAvailable, setReportsAvailable] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);

  const loadDashboard = useCallback(async () => {
    setLoading(true);
    setError(null);

    const timetableRequest = apiGetResult(`/api/timetable?day=${encodeURIComponent(selectedDay)}`, { cache: 'no-store' });
    const jobsRequest = apiGetResult('/api/status', { cache: 'no-store' });
    const reportsRequest = admin
      ? apiGetResult('/api/reports', { cache: 'no-store' })
      : Promise.resolve({ ok: true, data: { attendance_reports: [] } });

    const [timetableResult, jobsResult, reportsResult] = await Promise.all([
      timetableRequest,
      jobsRequest,
      reportsRequest,
    ]);

    if (timetableResult.ok && Array.isArray(timetableResult.data?.timetable)) {
      setTimetable(timetableResult.data.timetable);
      setSource('live');
    } else {
      setTimetable(fallbackRows(selectedDay, currentUser));
      setSource('fallback');
      setError(timetableResult.error || 'Live timetable is unavailable.');
    }

    if (jobsResult.ok && Array.isArray(jobsResult.data)) {
      setJobs(jobsResult.data);
      setJobsAvailable(true);
    } else {
      setJobs([]);
      setJobsAvailable(false);
    }

    if (reportsResult.ok) {
      setReports(reportsResult.data || { attendance_reports: [] });
      setReportsAvailable(true);
    } else {
      setReports({ attendance_reports: [] });
      setReportsAvailable(false);
    }

    setLastUpdated(new Date());
    setLoading(false);
  }, [admin, currentUser, selectedDay]);

  useEffect(() => {
    loadDashboard();
  }, [loadDashboard]);

  const visibleStudents = studentsForUser(currentUser);
  const liveData = source === 'live';

  const stats = useMemo(() => {
    const classSlots = timetable.filter((row) => !/lunch/i.test(row.subject || '')).length;
    const completed = liveData ? timetable.filter((row) => row.att_status?.status === 'Completed').length : '—';
    const processing = jobsAvailable ? jobs.filter(isRunningJob).length : '—';
    return { classSlots, completed, processing, students: visibleStudents.length };
  }, [jobs, jobsAvailable, liveData, timetable, visibleStudents.length]);

  const current = timetable.find((row) => row.rt_status === 'active')
    || timetable.find((row) => !/lunch/i.test(row.subject || ''));
  const currentSubject = current ? displaySubjectForUser(current, currentUser) : '-';
  const currentInfo = subjectInfo(currentSubject);

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow={admin ? 'HOD Portal' : 'Faculty Portal'}
        title={admin ? 'Attendance Operations' : `Welcome, ${currentUser.name}`}
        subtitle={admin
          ? 'Monitor classes, processing jobs, reports, and review readiness from one trusted view.'
          : 'Your dashboard is limited to your assigned classes, students, and processing activity.'}
        actions={
          <div className="page-actions">
            <NiceSelect compact value={selectedDay} onChange={setSelectedDay} options={DAYS} />
            <button className="button secondary" type="button" onClick={loadDashboard} disabled={loading}>
              {loading ? 'Refreshing…' : 'Refresh'}
            </button>
          </div>
        }
      />

      {source === 'fallback' && (
        <section className="connection-banner offline">
          <div>
            <strong>Live backend data is unavailable</strong>
            <p>{error || health.error || 'The backend could not be reached.'} The timetable below is a local reference preview only; processing and live status are disabled.</p>
          </div>
          <button className="button tiny secondary" type="button" onClick={loadDashboard}>Retry</button>
        </section>
      )}

      {health.connected && !health.ready && (
        <section className="connection-banner warning">
          <div>
            <strong>Backend is online but not ready to process</strong>
            <p>Missing: {health.missing.join(', ') || 'required model files'}. Live data may load, but attendance processing must remain disabled.</p>
          </div>
        </section>
      )}

      <section className="dashboard-source-row">
        <span className={`data-source-pill ${liveData ? 'live' : 'preview'}`}>
          {liveData ? 'Live backend data' : 'Local timetable preview'}
        </span>
        <small>{lastUpdated ? `Updated ${lastUpdated.toLocaleTimeString('en-IN')}` : 'Loading dashboard…'}</small>
      </section>

      <section className="stats-grid">
        <StatCard label={admin ? 'Class slots' : 'My class slots'} value={stats.classSlots} hint={`${selectedDay} · lunch excluded`} />
        <StatCard label="Completed" value={stats.completed} hint={liveData ? 'Live processed slots' : 'Unavailable in preview'} tone="success" />
        <StatCard label="Active jobs" value={stats.processing} hint={jobsAvailable ? 'Visible processing jobs' : 'Backend status unavailable'} tone="warning" />
        <StatCard label={admin ? 'Roster students' : 'My students'} value={stats.students} hint={(currentUser.subjects || []).join(', ')} tone="info" />
      </section>

      <section className="two-column">
        <article className="panel hero-panel">
          <div className="hero-copy">
            <p className="eyebrow">{liveData ? 'Next visible class' : 'Preview class'}</p>
            <h2>{current ? `${currentSubject} · ${current.period}` : 'No class selected'}</h2>
            <p>{current
              ? `${currentInfo?.courseName || current.course_name || '-'} · ${current.start_time}–${current.end_time} · Room ${current.room || '-'}`
              : 'Select a day to view available slots.'}</p>
            {!admin && <p className="muted">Access is limited to {(currentUser.subjects || []).join(', ')} and the assigned roster.</p>}
          </div>
          <div className="logic-card">
            <strong>{admin ? 'HOD scope' : 'Faculty scope'}</strong>
            <p>{admin
              ? 'You can monitor all classes and processing jobs, while every action remains session-scoped and auditable.'
              : 'You can process and review only your own subject sessions.'}</p>
          </div>
        </article>

        <article className="panel">
          <div className="panel-title-row">
            <div>
              <h3>Recent jobs</h3>
              <p className="muted">Persistent backend job history</p>
            </div>
            <span className="muted">{jobsAvailable ? `${jobs.length} visible` : 'Unavailable'}</span>
          </div>
          <div className="compact-list">
            {jobsAvailable && jobs.slice(0, 5).map((job) => (
              <div className="compact-item" key={job.job_id || `${job.day}-${job.period}`}>
                <div>
                  <strong>{job.session_date || job.day} {job.period} {job.subject_abbr || job.subject || ''}</strong>
                  <span>{job.progress_text || job.created_at || job.started_at || 'No timestamp'}</span>
                </div>
                <StatusBadge status={job.status} />
              </div>
            ))}
            {jobsAvailable && jobs.length === 0 && <p className="empty-text">No jobs have been started yet.</p>}
            {!jobsAvailable && <p className="empty-text">Job history is hidden until the backend reconnects.</p>}
          </div>
        </article>
      </section>

      <section className="panel">
        <div className="panel-title-row">
          <div>
            <h3>{admin ? `${selectedDay} timetable` : 'My classes'}</h3>
            <p className="muted">{liveData ? 'Live session and attendance state' : 'Static timetable reference; not current attendance state'}</p>
          </div>
          <span className={`data-source-pill ${liveData ? 'live' : 'preview'}`}>{liveData ? 'Live' : 'Preview'}</span>
        </div>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Period</th><th>Subject</th><th>Faculty</th><th>Time</th><th>Room</th><th>Status</th></tr></thead>
            <tbody>
              {timetable.map((row) => {
                const subject = displaySubjectForUser(row, currentUser);
                const info = subjectInfo(subject);
                const status = /lunch/i.test(row.subject || '') ? 'Lunch' : liveData ? row.att_status?.status || 'Pending' : 'Preview';
                return (
                  <tr key={slotKey(row)}>
                    <td><strong>{row.period}</strong></td>
                    <td>{subject}<small>{info?.courseName || row.course_name}</small></td>
                    <td>{row.teacher || row.instructor}</td>
                    <td>{row.start_time}–{row.end_time}</td>
                    <td>{row.room}</td>
                    <td><StatusBadge status={status} /></td>
                  </tr>
                );
              })}
              {timetable.length === 0 && <tr><td colSpan="6" className="empty-cell">No class slots are visible for this login.</td></tr>}
            </tbody>
          </table>
        </div>
      </section>

      {admin && (
        <section className="panel">
          <div className="panel-title-row"><h3>Generated reports</h3><span className="muted">Latest files</span></div>
          <div className="report-link-grid">
            {reportsAvailable && (reports.attendance_reports || []).slice(0, 6).map((file) => (
              <a key={file} className="report-link" href={`/attendance_website/${file}`} target="_blank" rel="noreferrer">{file}</a>
            ))}
            {reportsAvailable && (!reports.attendance_reports || reports.attendance_reports.length === 0) && <p className="empty-text">No reports are available yet.</p>}
            {!reportsAvailable && <p className="empty-text">Reports are unavailable while the backend is offline.</p>}
          </div>
        </section>
      )}
    </div>
  );
}
