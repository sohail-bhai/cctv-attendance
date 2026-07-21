import { useCallback, useEffect, useMemo, useState } from 'react';
import PageHeader from '../components/PageHeader.jsx';
import { apiGetResult } from '../api/client.js';
import { useAuth } from '../auth/AuthContext.jsx';
import { SUBJECT_STUDENTS, studentsForUser } from '../data/students.js';
import { SUBJECT_INFO, isAdmin } from '../data/users.js';
import { coverageLabel, coverageTone, filterCoverageRows, sourceValue } from '../utils/hodControl.js';

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
  const [coverage, setCoverage] = useState(null);
  const [source, setSource] = useState('loading');
  const [message, setMessage] = useState('Loading the authoritative roster…');
  const [search, setSearch] = useState('');
  const [issuesOnly, setIssuesOnly] = useState(false);

  const loadRoster = useCallback(async () => {
    setSource('loading');
    setMessage('Loading the authoritative roster…');
    const result = await apiGetResult('/api/students', { cache: 'no-store' });
    if (result.ok && Array.isArray(result.data?.students)) {
      setVisibleStudents(result.data.students.map(normalizeStudent));
      setCoverage(result.data.coverage || null);
      setSource('live');
      setMessage('Authoritative backend roster and coverage loaded.');
      return;
    }

    setVisibleStudents(fallbackStudents);
    setCoverage(null);
    setSource('fallback');
    setMessage(`${result.error || 'The backend roster is unavailable.'} Showing the bundled roster preview only.`);
  }, [fallbackStudents]);

  useEffect(() => { loadRoster(); }, [loadRoster]);

  const subjects = admin ? Object.keys(SUBJECT_STUDENTS) : currentUser.subjects;
  const filtered = useMemo(() => filterCoverageRows(visibleStudents, { search, issuesOnly }), [issuesOnly, search, visibleStudents]);
  const rowsBySubject = useMemo(() => Object.fromEntries(subjects.map((subject) => [
    subject,
    filtered.filter((student) => (student.subjects || []).includes(subject)),
  ])), [filtered, subjects]);

  const coverageAvailable = source === 'live' && coverage?.embedding_source_available;

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow={admin ? 'All Students' : 'Faculty Student List'}
        title={admin ? 'Students & Coverage' : 'My Students'}
        subtitle={admin ? 'HOD sees authoritative subject rosters and model-readiness diagnostics without changing membership.' : 'Faculty view is limited to students assigned to your subject.'}
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
        {coverage && <div className="stat-card"><span>Embedding covered</span><strong>{coverage.embedding_source_available ? coverage.embedding_available : '—'}</strong><small>{coverage.embedding_source_available ? 'Current production summary' : 'Coverage unavailable'}</small></div>}
        {coverage && <div className="stat-card"><span>Coverage issues</span><strong>{coverage.issues ?? '—'}</strong><small>Roster students remain visible</small></div>}
        {subjects.map((subject) => (
          <div className="stat-card" key={subject}>
            <span>{subject}</span>
            <strong>{visibleStudents.filter((student) => (student.subjects || []).includes(subject)).length}</strong>
            <small>{SUBJECT_INFO[subject]?.courseName}</small>
          </div>
        ))}
      </section>

      <section className="panel roster-filter-panel">
        <div className="coverage-filters">
          <input className="input" placeholder="Search roll or name" value={search} onChange={(event) => setSearch(event.target.value)} />
          {source === 'live' && <label className="toggle-label"><input type="checkbox" checked={issuesOnly} onChange={(event) => setIssuesOnly(event.target.checked)} /> Coverage issues only</label>}
          <span className="mode-pill">{filtered.length} matching students</span>
        </div>
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
                <thead><tr><th>Roll Number</th><th>Name</th><th>Subject</th>{source === 'live' && <><th>Dataset</th><th>Embedding</th><th>Coverage</th></>}</tr></thead>
                <tbody>
                  {rows.map((student) => (
                    <tr key={`${subject}-${student.roll}`}>
                      <td><strong>{student.roll}</strong></td>
                      <td>{student.name}</td>
                      <td>{subject}</td>
                      {source === 'live' && <>
                        <td>{sourceValue(student.dataset_available)}</td>
                        <td>{sourceValue(student.embedding_available)}{student.faces_used != null ? <small>{student.faces_used} faces</small> : null}</td>
                        <td><span className={`coverage-badge ${coverageTone(student.coverage_status)}`}>{coverageLabel(student.coverage_status)}</span></td>
                      </>}
                    </tr>
                  ))}
                  {rows.length === 0 && <tr><td className="empty-cell" colSpan={source === 'live' ? 6 : 3}>No students match this subject and filter.</td></tr>}
                </tbody>
              </table>
            </div>
            {!coverageAvailable && source === 'live' && <p className="muted coverage-source-note">Embedding coverage is unavailable because the production summary could not be read. Roster membership is still authoritative.</p>}
          </section>
        );
      })}
    </div>
  );
}
