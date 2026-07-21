import { useCallback, useEffect, useMemo, useState } from 'react';
import PageHeader from '../components/PageHeader.jsx';
import StatCard from '../components/StatCard.jsx';
import { apiGetResult } from '../api/client.js';
import { useAuth } from '../auth/AuthContext.jsx';
import { coverageLabel, coverageTone, filterCoverageRows, sourceValue } from '../utils/hodControl.js';

function CoverageBadge({ status }) {
  return <span className={`coverage-badge ${coverageTone(status)}`}>{coverageLabel(status)}</span>;
}

export default function HodControl() {
  const { isAdmin } = useAuth();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [search, setSearch] = useState('');
  const [subject, setSubject] = useState('ALL');
  const [issuesOnly, setIssuesOnly] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    const result = await apiGetResult('/api/hod/overview', { cache: 'no-store' });
    if (result.ok) setData(result.data);
    else setError(result.error || 'HOD overview is unavailable.');
    setLoading(false);
  }, []);

  useEffect(() => { load(); }, [load]);

  const filteredStudents = useMemo(() => filterCoverageRows(data?.students, { search, subject, issuesOnly }), [data?.students, issuesOnly, search, subject]);

  if (!isAdmin) {
    return (
      <div className="page-stack">
        <PageHeader eyebrow="Restricted" title="HOD Control" subtitle="Only the HOD can view system governance and coverage." />
        <div className="notice error">You do not have access to this page.</div>
      </div>
    );
  }

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="HOD Governance"
        title="HOD Control Center"
        subtitle="One trusted view of faculty access, subject rosters, model coverage, production state, and operating policy."
        actions={<button className="button secondary" type="button" onClick={load} disabled={loading}>{loading ? 'Refreshing…' : 'Refresh'}</button>}
      />

      {error && <section className="connection-banner offline"><div><strong>HOD overview unavailable</strong><p>{error}</p></div></section>}
      {!error && loading && <section className="connection-banner warning"><div><strong>Checking governance data</strong><p>Reading role, roster, dataset, embedding, timetable, and production-version sources.</p></div></section>}

      {data && (
        <>
          <section className="stats-grid">
            <StatCard label="Faculty" value={data.faculty?.length || 0} hint="Backend role registry" />
            <StatCard label="Subjects" value={data.subjects?.length || 0} hint="Mapped course tracks" tone="info" />
            <StatCard label="Unique students" value={data.coverage?.total_students ?? '—'} hint="Authoritative roster union" />
            <StatCard label="Embedding covered" value={data.coverage?.embedding_source_available ? data.coverage.embedding_available : '—'} hint={data.coverage?.embedding_source_available ? 'Current production summary' : 'Summary unavailable'} tone="success" />
          </section>

          <section className="two-column hod-control-grid">
            <article className="panel">
              <div className="panel-title-row"><div><h3>Production recognition state</h3><p className="muted">Read-only lifecycle pointer and approved operating policy.</p></div><span className="mode-pill">{data.production?.status || 'unknown'}</span></div>
              <dl className="detail-list">
                <div><dt>Production family</dt><dd>{data.production?.family_id || 'Legacy / unversioned'}</dd></div>
                <div><dt>Promotion</dt><dd>{data.production?.promotion_id || '—'}</dd></div>
                <div><dt>Recognition stack</dt><dd>{data.policy?.recognition_stack}</dd></div>
                <div><dt>Match / margin</dt><dd>{data.policy?.match_threshold} / {data.policy?.margin_threshold}</dd></div>
                <div><dt>Attendance rule</dt><dd>{data.policy?.present_checkpoints} of {data.policy?.checkpoint_count} checkpoints</dd></div>
                <div><dt>Checkpoint evidence</dt><dd>Minimum {data.policy?.checkpoint_min_detections} detections</dd></div>
              </dl>
            </article>

            <article className="panel">
              <div className="panel-title-row"><div><h3>Faculty access</h3><p className="muted">Passwords are never exposed by this page.</p></div><span className="mode-pill">{data.faculty?.length || 0} faculty</span></div>
              <div className="compact-list">
                {(data.faculty || []).map((faculty) => (
                  <div className="compact-item" key={faculty.id}>
                    <div><strong>{faculty.name}</strong><span>@{faculty.username} · {(faculty.subjects || []).join(', ') || 'No subject'}</span></div>
                    <span className="coverage-badge neutral">Faculty</span>
                  </div>
                ))}
              </div>
              <div className="governance-note">Faculty creation, reassignment, and timetable import remain write-locked until the next guarded configuration phase.</div>
            </article>
          </section>

          <section className="panel">
            <div className="panel-title-row"><div><h3>Subject readiness</h3><p className="muted">Roster-first coverage. Missing embeddings remain visible instead of removing students.</p></div></div>
            <div className="table-wrap">
              <table>
                <thead><tr><th>Subject</th><th>Faculty</th><th>Roster</th><th>Dataset</th><th>Embeddings</th><th>Timetable slots</th><th>Missing embeddings</th></tr></thead>
                <tbody>
                  {(data.subjects || []).map((item) => (
                    <tr key={item.abbr}>
                      <td><strong>{item.abbr}</strong><small>{item.course_code} · {item.course_name}</small></td>
                      <td>{item.faculty_name || 'Unassigned'}</td>
                      <td>{item.roster_count}</td>
                      <td>{data.coverage?.dataset_source_available ? `${item.dataset_available}/${item.roster_count}` : 'Unknown'}</td>
                      <td>{data.coverage?.embedding_source_available ? `${item.embedding_available}/${item.roster_count}` : 'Unknown'}</td>
                      <td>{item.timetable_slots}</td>
                      <td>{item.missing_embeddings?.length ? item.missing_embeddings.length : '0'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="panel">
            <div className="panel-title-row">
              <div><h3>Student coverage exceptions</h3><p className="muted">Dataset and production-embedding readiness are diagnostics; roster membership remains authoritative.</p></div>
              <span className="mode-pill">{filteredStudents.length} visible</span>
            </div>
            <div className="coverage-filters">
              <input className="input" placeholder="Search roll or name" value={search} onChange={(event) => setSearch(event.target.value)} />
              <select className="input" value={subject} onChange={(event) => setSubject(event.target.value)}>
                <option value="ALL">All subjects</option>
                {(data.subjects || []).map((item) => <option value={item.abbr} key={item.abbr}>{item.abbr}</option>)}
              </select>
              <label className="toggle-label"><input type="checkbox" checked={issuesOnly} onChange={(event) => setIssuesOnly(event.target.checked)} /> Issues only</label>
            </div>
            <div className="table-wrap">
              <table>
                <thead><tr><th>Roll Number</th><th>Name</th><th>Subjects</th><th>Dataset</th><th>Embedding</th><th>Faces used</th><th>Coverage</th></tr></thead>
                <tbody>
                  {filteredStudents.map((student) => (
                    <tr key={student.roll}>
                      <td><strong>{student.roll}</strong></td>
                      <td>{student.name}</td>
                      <td>{(student.subjects || []).join(', ')}</td>
                      <td>{sourceValue(student.dataset_available)}</td>
                      <td>{sourceValue(student.embedding_available)}</td>
                      <td>{student.faces_used ?? '—'}</td>
                      <td><CoverageBadge status={student.coverage_status} /></td>
                    </tr>
                  ))}
                  {filteredStudents.length === 0 && <tr><td colSpan="7" className="empty-cell">No students match this coverage filter.</td></tr>}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}
    </div>
  );
}
