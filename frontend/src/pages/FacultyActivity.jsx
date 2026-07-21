import { useEffect, useMemo, useState } from 'react';
import PageHeader from '../components/PageHeader.jsx';
import StatCard from '../components/StatCard.jsx';
import StatusBadge from '../components/StatusBadge.jsx';
import NiceSelect from '../components/NiceSelect.jsx';
import { apiGet } from '../api/client.js';
import { DAYS, FALLBACK_TIMETABLE } from '../data/fallback.js';
import { USERS, canAccessRow, displaySubjectForUser, isAdmin, rowSubjects, subjectInfo } from '../data/users.js';
import { useAuth } from '../auth/AuthContext.jsx';

function normalizePeriod(period) {
  const raw = String(period || '').toUpperCase();
  return raw.startsWith('P') ? raw : `P${raw}`;
}

function safeRows(data, day) {
  if (data?.timetable) return data.timetable;
  return FALLBACK_TIMETABLE.filter((row) => row.day === day);
}

function latestJobForSlot(jobs, row) {
  return jobs.find((job) => {
    if (row.session_id && job.session_id) return job.session_id === row.session_id;
    return job.day === row.day && normalizePeriod(job.period) === normalizePeriod(row.period) && (!job.subject_abbr || job.subject_abbr === (row.subject_track || row.subject));
  });
}

export default function FacultyActivity() {
  const { currentUser } = useAuth();
  const today = new Date().toLocaleDateString('en-US', { weekday: 'long' });
  const [day, setDay] = useState(DAYS.includes(today) ? today : 'Monday');
  const [rows, setRows] = useState([]);
  const [jobs, setJobs] = useState([]);

  useEffect(() => {
    apiGet(`/api/timetable?day=${day}`, null).then((data) => setRows(safeRows(data, day)));
    apiGet('/api/status', []).then((data) => setJobs(Array.isArray(data) ? data : []));
  }, [day]);

  const facultyUsers = USERS.filter((user) => user.role === 'faculty');

  const activity = useMemo(() => facultyUsers.map((faculty) => {
    const visibleSlots = rows
      .filter((row) => !/lunch/i.test(row.subject || ''))
      .filter((row) => canAccessRow(faculty, row))
      .map((row) => {
        const subject = displaySubjectForUser(row, faculty);
        const job = latestJobForSlot(jobs, row);
        const state = job?.status || row.att_status?.status || 'Pending';
        return { ...row, visibleSubject: subject, job, state };
      });

    const completed = visibleSlots.filter((slot) => slot.state === 'Completed').length;
    const processing = visibleSlots.filter((slot) => ['Processing', 'Pending'].includes(slot.state) && slot.job).length;
    const failed = visibleSlots.filter((slot) => slot.state === 'Failed').length;
    const pending = Math.max(visibleSlots.length - completed - processing - failed, 0);
    const reviewSlots = visibleSlots.filter((slot) => slot.att_status?.review_count || slot.att_status?.needs_review_count);
    const latest = visibleSlots.find((slot) => slot.job?.started_at || slot.job?.created_at || slot.att_status?.updated_at);

    return {
      faculty,
      visibleSlots,
      completed,
      processing,
      failed,
      pending,
      reviewCount: reviewSlots.length,
      latestActivity: latest?.job?.started_at || latest?.job?.created_at || latest?.att_status?.updated_at || 'Not started',
    };
  }), [rows, jobs]);

  const totals = useMemo(() => ({
    faculty: activity.length,
    slots: activity.reduce((sum, item) => sum + item.visibleSlots.length, 0),
    completed: activity.reduce((sum, item) => sum + item.completed, 0),
    pending: activity.reduce((sum, item) => sum + item.pending, 0),
  }), [activity]);

  if (!isAdmin(currentUser)) {
    return (
      <div className="page-stack">
        <PageHeader eyebrow="Restricted" title="Faculty Activity" subtitle="Only the HOD can view all faculty activity." />
        <div className="notice error">You do not have access to this page.</div>
      </div>
    );
  }

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="HOD Monitoring"
        title="Faculty Activity"
        subtitle="Track which faculty slots are pending, processing, completed, or failed. This page is for HOD visibility only."
        actions={<NiceSelect compact value={day} onChange={setDay} options={DAYS} />}
      />

      <section className="stats-grid">
        <StatCard label="Faculty members" value={totals.faculty} hint="Role-based users" />
        <StatCard label="Visible slots" value={totals.slots} hint={day} tone="info" />
        <StatCard label="Completed" value={totals.completed} hint="Processed slots" tone="success" />
        <StatCard label="Pending" value={totals.pending} hint="Not processed" tone="warning" />
      </section>

      <section className="panel">
        <div className="panel-title-row">
          <div>
            <h3>{day} faculty overview</h3>
            <p className="muted">CCM and CVO are shown separately during simultaneous slots, based on the faculty login subject.</p>
          </div>
        </div>
        <div className="faculty-activity-grid">
          {activity.map((item) => (
            <article className="faculty-card" key={item.faculty.id}>
              <div className="faculty-card-head">
                <div className="faculty-avatar">{item.faculty.name.split(' ').filter(Boolean).slice(-2).map((x) => x[0]).join('')}</div>
                <div>
                  <h3>{item.faculty.name}</h3>
                  <p>{item.faculty.roleLabel}</p>
                  <strong>{item.faculty.subjects.join(', ')}</strong>
                </div>
              </div>
              <div className="faculty-metrics">
                <span><b>{item.visibleSlots.length}</b> classes</span>
                <span><b>{item.completed}</b> completed</span>
                <span><b>{item.pending}</b> pending</span>
                <span><b>{item.reviewCount}</b> review slots</span>
              </div>
              <div className="compact-list compact-list-tight">
                {item.visibleSlots.slice(0, 6).map((slot) => {
                  const info = subjectInfo(slot.visibleSubject);
                  return (
                    <div className="compact-item" key={`${item.faculty.id}-${slot.session_id || `${slot.day}-${slot.period}-${slot.visibleSubject}`}`}>
                      <div>
                        <strong>{slot.period} · {slot.visibleSubject}</strong>
                        <span>{info?.courseName || slot.course_name} · {slot.start_time}–{slot.end_time}</span>
                      </div>
                      <StatusBadge status={slot.state} />
                    </div>
                  );
                })}
                {item.visibleSlots.length === 0 && <p className="empty-text">No visible slots for this faculty on {day}.</p>}
              </div>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}
