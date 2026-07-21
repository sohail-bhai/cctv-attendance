import PageHeader from '../components/PageHeader.jsx';
import { useAuth } from '../auth/AuthContext.jsx';
import { SUBJECT_STUDENTS, studentsForUser } from '../data/students.js';
import { SUBJECT_INFO, isAdmin } from '../data/users.js';

export default function Students() {
  const { currentUser } = useAuth();
  const visibleStudents = studentsForUser(currentUser);
  const admin = isAdmin(currentUser);
  const subjects = admin ? Object.keys(SUBJECT_STUDENTS) : currentUser.subjects;

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow={admin ? 'All Students' : 'Faculty Student List'}
        title={admin ? 'Students' : 'My Students'}
        subtitle={admin ? 'Admin can see every student enrolled in the current class group.' : 'Faculty view is limited to students assigned to your subject.'}
      />

      <section className="stats-grid">
        <div className="stat-card"><span>Visible students</span><strong>{visibleStudents.length}</strong><small>{admin ? 'All subjects' : currentUser.subjects.join(', ')}</small></div>
        {subjects.map((subject) => (
          <div className="stat-card" key={subject}>
            <span>{subject}</span>
            <strong>{SUBJECT_STUDENTS[subject]?.length || 0}</strong>
            <small>{SUBJECT_INFO[subject]?.courseName}</small>
          </div>
        ))}
      </section>

      {subjects.map((subject) => {
        const rows = admin ? SUBJECT_STUDENTS[subject] : visibleStudents.filter((student) => (student.subjects || []).includes(subject));
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
                </tbody>
              </table>
            </div>
          </section>
        );
      })}
    </div>
  );
}
