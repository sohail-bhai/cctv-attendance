import { useCallback, useEffect, useMemo, useState } from 'react';
import PageHeader from '../components/PageHeader.jsx';
import { apiGetResult } from '../api/client.js';
import { useAuth } from '../auth/AuthContext.jsx';
import { SUBJECT_STUDENTS, studentsForUser } from '../data/students.js';
import { SUBJECT_INFO, isAdmin } from '../data/users.js';

function normalizeStudent(student) {
  return {
    ...student,
    roll: String(student?.roll || '').trim().toUpperCase(),
    name: String(student?.name || '').trim(),
    subjects: Array.isArray(student?.subjects)
      ? student.subjects.map((subject) => String(subject).toUpperCase())
      : [],
  };
}

export default function Students() {
  const { currentUser } = useAuth();
  const admin = isAdmin(currentUser);
  const fallbackStudents = useMemo(() => studentsForUser(currentUser).map(normalizeStudent), [currentUser]);
  const [visibleStudents, setVisibleStudents] = useState(fallbackStudents);
  const [source, setSource] = useState('loading');
  const [message, setMessage] = useState('Loading the authoritative roster…');

  const loadRoster = useCallback(async () => {
    setSource('loading');
    setMessage('Loading the authoritative roster…');
    const result = await apiGetResult('/api/students', { cache: 'no-store' });
    if (result.ok && Array.isArray(result.data?.students)) {
      setVisibleStudents(result.data.students.map(normalizeStudent));
      setSource('live');
      setMessage('Authoritative backend roster loaded.');
      return;
    }

    setVisibleStudents(fallbackStudents);
    setSource('fallback');
    setMessage(`${result.error || 'The backend roster is unavailable.'} Showing the bundled roster preview only.`);
  }, [fallbackStudents]);

  useEffect(() => {
    loadRoster();
  }, [loadRoster]);

  const subjects = admin ? Object.keys(SUBJECT_STUDENTS) : currentUser.subjects;
  const rowsBySubject = useMemo(() => Object.fromEntries(subjects.map((subject) => [
    subject,
    visibleStudents.filter((student) => (student.subjects || []).includes(subject)),
  ])), [subjects, visibleStudents]);

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow={admin ? 'All Students' : 'Faculty Student List'}
        title={admin ? 'Students' : 'My Students'}
        subtitle={admin ? 'HOD can see every student in the authoritative subject rosters.' : 'Faculty view is limited to students assigned to your subject.'}
        actions={<button className="button secondary" type="button" onClick={loadRoster} disabled={source === 'loading'}>{source === 'loading' ? 'Refreshing…' : 'Refresh roster'}</button>}
      />

      <section className={`connection-banner ${source === 'live' ? 'online' : source === 'fallback' ? 'offline' : 'warning'}`}>
        <div>
          <strong>{source === 'live' ? 'Live authoritative roster' : source === 'fallback' ? 'Local roster preview' : 'Checking roster source'}</strong>
          <p>{message}</p>
        </div>
      </section>

      <section className="stats-grid">
        <div className="stat-card"><span>Visible students</span><strong>{visibleStudents.length}</strong><small>{admin ? 'All accessible subjects' : currentUser.subjects.join(', ')}</small></div>
        {subjects.map((subject) => (
          <div className="stat-card" key={subject}>
            <span>{subject}</span>
            <strong>{rowsBySubject[subject]?.length || 0}</strong>
            <small>{SUBJECT_INFO[subject]?.courseName}</small>
          </div>
        ))}
      </section>

      {subjects.map((subject) => {
        const rows = rowsBySubject[subject] || [];
        return (
          <section className="panel" key={subject}>
            <div className="panel-title-row">
              <div>
                <h3>{subject} — {SUBJECT_INFO[subject]?.courseName}</h3>
                <p className="muted">Faculty: {SUBJECT_INFO[subject]?.facultyName}</p>
              </div>
              <span className="mode-pill">{rows.length} students</span>
            </div>
            <div className="table-wrap">
              <table>
                <thead><tr><th>Roll Number</th><th>Name</th><th>Subject</th></tr></thead>
                <tbody>
                  {rows.map((student) => (
                    <tr key={`${subject}-${student.roll}`}>
                      <td><strong>{student.roll}</strong></td>
                      <td>{student.name}</td>
                      <td>{subject}</td>
                    </tr>
                  ))}
                  {rows.length === 0 && <tr><td className="empty-cell" colSpan="3">No students are available for this subject.</td></tr>}
                </tbody>
              </table>
            </div>
          </section>
        );
      })}
    </div>
  );
}
