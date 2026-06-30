import { useEffect, useMemo, useState } from 'react';
import PageHeader from '../components/PageHeader.jsx';
import StatCard from '../components/StatCard.jsx';
import StatusBadge from '../components/StatusBadge.jsx';
import NiceSelect from '../components/NiceSelect.jsx';
import { apiGet } from '../api/client.js';
import { DAYS, FALLBACK_TIMETABLE } from '../data/fallback.js';
import { useAuth } from '../auth/AuthContext.jsx';
import { canAccessRow, displaySubjectForUser, isAdmin, subjectInfo } from '../data/users.js';
import { studentsForUser } from '../data/students.js';

function slotKey(row) {
  return `${row.day}_${row.period}_${row.subject}`;
}

export default function Dashboard() {
  const { currentUser } = useAuth();
  const admin = isAdmin(currentUser);
  const today = new Date().toLocaleDateString('en-US', { weekday: 'long' });
  const [selectedDay, setSelectedDay] = useState(DAYS.includes(today) ? today : 'Monday');
  const [timetable, setTimetable] = useState([]);
  const [jobs, setJobs] = useState([]);
  const [reports, setReports] = useState({ attendance_reports: [] });

  useEffect(() => {
    apiGet(`/api/timetable?day=${selectedDay}`, null).then((data) => {
      if (data?.timetable) setTimetable(data.timetable);
      else setTimetable(FALLBACK_TIMETABLE.filter((row) => row.day === selectedDay && canAccessRow(currentUser, row)));
    });
  }, [selectedDay, currentUser]);

  useEffect(() => {
    apiGet('/api/status', []).then((data) => setJobs(Array.isArray(data) ? data : []));
    apiGet('/api/reports', { attendance_reports: [] }).then((data) => setReports(data || {}));
  }, []);

  const visibleStudents = studentsForUser(currentUser);

  const stats = useMemo(() => {
    const classSlots = timetable.filter((row) => !/lunch/i.test(row.subject || '')).length;
    const completed = timetable.filter((row) => row.att_status?.status === 'Completed').length;
    const processing = timetable.filter((row) => row.att_status?.status === 'Processing').length;
    return { classSlots, completed, processing, students: visibleStudents.length };
  }, [timetable, visibleStudents.length]);

  const current = timetable.find((row) => row.rt_status === 'active') || timetable.find((row) => !/lunch/i.test(row.subject || ''));
  const currentSubject = current ? displaySubjectForUser(current, currentUser) : '-';
  const currentInfo = subjectInfo(currentSubject);

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow={admin ? 'Main Admin Portal' : 'Faculty Portal'}
        title={admin ? 'Smart Attendance Home' : `Welcome, ${currentUser.name}`}
        subtitle={admin ? 'Monitor all classes, reports, reviews, students, and system readiness.' : 'Your dashboard only shows the classes and students assigned to you.'}
        actions={
          <NiceSelect compact value={selectedDay} onChange={setSelectedDay} options={DAYS} />
        }
      />

      <section className="stats-grid">
        <StatCard label={admin ? 'Class slots today' : 'My slots today'} value={stats.classSlots} hint="Lunch excluded" />
        <StatCard label="Completed" value={stats.completed} hint="Processed slots" tone="success" />
        <StatCard label="Processing" value={stats.processing} hint="Running jobs" tone="warning" />
        <StatCard label={admin ? 'Total students' : 'My students'} value={stats.students} hint={(currentUser.subjects || []).join(', ')} tone="info" />
      </section>

      <section className="two-column">
        <article className="panel hero-panel">
          <div className="hero-copy">
            <p className="eyebrow">Next visible class</p>
            <h2>{current ? `${currentSubject} · ${current.period}` : 'No class selected'}</h2>
            <p>{current ? `${currentInfo?.courseName || current.course_name || '-'} · ${current.start_time}–${current.end_time} · Room ${current.room || '-'}` : 'Select a day to view available slots.'}</p>
            {!admin && <p className="muted">Access is limited to {currentUser.subjects.join(', ')} and the assigned student list.</p>}
          </div>
          <div className="logic-card">
            <strong>{admin ? 'Admin access' : 'Faculty access'}</strong>
            <p>{admin ? 'You can process and review every slot.' : 'You can process and review only your own subject slots.'}</p>
          </div>
        </article>

        <article className="panel">
          <div className="panel-title-row">
            <h3>Recent jobs</h3>
            <span className="muted">{jobs.length} visible job(s)</span>
          </div>
          <div className="compact-list">
            {jobs.slice(0, 5).map((job) => (
              <div className="compact-item" key={job.job_id || `${job.day}-${job.period}`}>
                <div>
                  <strong>{job.day} {job.period}</strong>
                  <span>{job.progress_text || job.created_at || job.started_at || 'No timestamp'}</span>
                </div>
                <StatusBadge status={job.status} />
              </div>
            ))}
            {jobs.length === 0 && <p className="empty-text">No jobs found yet.</p>}
          </div>
        </article>
      </section>

      <section className="panel">
        <div className="panel-title-row">
          <h3>{admin ? 'Today’s timetable' : 'My classes'}</h3>
          <span className="muted">{selectedDay}</span>
        </div>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Period</th><th>Subject</th><th>Faculty</th><th>Time</th><th>Room</th><th>Status</th></tr></thead>
            <tbody>
              {timetable.map((row) => {
                const subject = displaySubjectForUser(row, currentUser);
                const info = subjectInfo(subject);
                return (
                  <tr key={slotKey(row)}>
                    <td><strong>{row.period}</strong></td>
                    <td>{subject}<small>{info?.courseName || row.course_name}</small></td>
                    <td>{row.teacher || row.instructor}</td>
                    <td>{row.start_time}–{row.end_time}</td>
                    <td>{row.room}</td>
                    <td><StatusBadge status={/lunch/i.test(row.subject) ? 'Lunch' : row.att_status?.status || 'Pending'} /></td>
                  </tr>
                );
              })}
              {timetable.length === 0 && <tr><td colSpan="6" className="empty-cell">No class slots visible for this login.</td></tr>}
            </tbody>
          </table>
        </div>
      </section>

      {admin && (
        <section className="panel">
          <div className="panel-title-row"><h3>Generated reports</h3><span className="muted">Latest files</span></div>
          <div className="report-link-grid">
            {(reports.attendance_reports || []).slice(0, 6).map((file) => (
              <a key={file} className="report-link" href={`/attendance_website/${file}`} target="_blank" rel="noreferrer">{file}</a>
            ))}
            {(!reports.attendance_reports || reports.attendance_reports.length === 0) && <p className="empty-text">No reports available yet.</p>}
          </div>
        </section>
      )}
    </div>
  );
}
