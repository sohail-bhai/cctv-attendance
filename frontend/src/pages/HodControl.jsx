import { useCallback, useEffect, useMemo, useState } from 'react';
import PageHeader from '../components/PageHeader.jsx';
import StatCard from '../components/StatCard.jsx';
import { apiGetResult, apiPost, apiPostForm } from '../api/client.js';
import { useAuth } from '../auth/AuthContext.jsx';
import { coverageLabel, coverageTone, filterCoverageRows, sourceValue } from '../utils/hodControl.js';
import {
  TIMETABLE_CONFIRMATION,
  canRollbackLatest,
  changeTypeLabel,
  facultyFormError,
  normalizeFacultyForm,
  subjectAssignmentError,
  summarizeTimetablePreview,
  timetableApplyError,
} from '../utils/configGovernance.js';

const EMPTY_FACULTY = {
  action: 'create',
  faculty_id: '',
  username: '',
  name: '',
  password: '',
  reason: '',
};

function CoverageBadge({ status }) {
  return <span className={`coverage-badge ${coverageTone(status)}`}>{coverageLabel(status)}</span>;
}

function ConfigMessage({ message }) {
  if (!message?.text) return null;
  return <div className={`notice ${message.type === 'error' ? 'error' : 'success'}`}>{message.text}</div>;
}

export default function HodControl() {
  const { isAdmin } = useAuth();
  const [data, setData] = useState(null);
  const [config, setConfig] = useState(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const [message, setMessage] = useState(null);
  const [search, setSearch] = useState('');
  const [subject, setSubject] = useState('ALL');
  const [issuesOnly, setIssuesOnly] = useState(true);
  const [facultyForm, setFacultyForm] = useState(EMPTY_FACULTY);
  const [assignment, setAssignment] = useState({ subject: '', faculty_id: '', reason: '' });
  const [timetableFile, setTimetableFile] = useState(null);
  const [timetablePreview, setTimetablePreview] = useState(null);
  const [timetableReason, setTimetableReason] = useState('');
  const [timetableConfirmation, setTimetableConfirmation] = useState('');
  const [rollbackReason, setRollbackReason] = useState('');
  const [rollbackConfirmation, setRollbackConfirmation] = useState('');

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    const [overviewResult, configResult] = await Promise.all([
      apiGetResult('/api/hod/overview', { cache: 'no-store' }),
      apiGetResult('/api/hod/config', { cache: 'no-store' }),
    ]);
    if (overviewResult.ok) setData(overviewResult.data);
    else setError(overviewResult.error || 'HOD overview is unavailable.');
    if (configResult.ok) {
      setConfig(configResult.data);
      const firstSubject = configResult.data?.subjects?.[0];
      if (firstSubject) {
        setAssignment((current) => current.subject ? current : {
          subject: firstSubject.abbr,
          faculty_id: firstSubject.faculty_id || '',
          reason: '',
        });
      }
    } else {
      setError((current) => current || configResult.error || 'HOD configuration is unavailable.');
    }
    setLoading(false);
  }, []);

  useEffect(() => { load(); }, [load]);

  const filteredStudents = useMemo(
    () => filterCoverageRows(data?.students, { search, subject, issuesOnly }),
    [data?.students, issuesOnly, search, subject],
  );
  const previewSummary = useMemo(() => summarizeTimetablePreview(timetablePreview), [timetablePreview]);

  const runAction = async (label, action, successText) => {
    setBusy(label);
    setMessage(null);
    try {
      await action();
      setMessage({ type: 'success', text: successText });
      await load();
      return true;
    } catch (actionError) {
      setMessage({ type: 'error', text: actionError.message || 'Configuration action failed.' });
      return false;
    } finally {
      setBusy('');
    }
  };

  const startCreateFaculty = () => {
    setFacultyForm(EMPTY_FACULTY);
    setMessage(null);
  };

  const startEditFaculty = (faculty) => {
    setFacultyForm({
      action: 'update',
      faculty_id: faculty.id,
      username: faculty.username,
      name: faculty.name,
      password: '',
      reason: '',
    });
    setMessage(null);
  };

  const submitFaculty = async (event) => {
    event.preventDefault();
    const payload = normalizeFacultyForm(facultyForm);
    const formError = facultyFormError(payload);
    if (formError) {
      setMessage({ type: 'error', text: formError });
      return;
    }
    const verb = payload.action === 'create' ? 'create' : 'update';
    if (!window.confirm(`Confirm ${verb} faculty account ${payload.faculty_id}?`)) return;
    const ok = await runAction(
      'faculty',
      () => apiPost('/api/hod/config/faculty', payload),
      `Faculty account ${payload.faculty_id} ${payload.action === 'create' ? 'created' : 'updated'} safely.`,
    );
    if (ok) setFacultyForm(EMPTY_FACULTY);
  };

  const chooseAssignment = (item) => {
    setAssignment({
      subject: item.abbr,
      faculty_id: item.faculty_id || '',
      reason: '',
    });
    setMessage(null);
  };

  const submitAssignment = async (event) => {
    event.preventDefault();
    const formError = subjectAssignmentError(assignment);
    if (formError) {
      setMessage({ type: 'error', text: formError });
      return;
    }
    const faculty = config?.faculty?.find((item) => item.id === assignment.faculty_id);
    if (!window.confirm(`Assign ${assignment.subject} to ${faculty?.name || assignment.faculty_id}? This also updates the timetable instructor field.`)) return;
    const ok = await runAction(
      'assignment',
      () => apiPost('/api/hod/config/subject-assignment', assignment),
      `${assignment.subject} assignment updated safely.`,
    );
    if (ok) setAssignment((current) => ({ ...current, reason: '' }));
  };

  const previewTimetable = async () => {
    if (!timetableFile) {
      setMessage({ type: 'error', text: 'Choose a timetable CSV first.' });
      return;
    }
    setBusy('timetable-preview');
    setMessage(null);
    try {
      const formData = new FormData();
      formData.append('timetable', timetableFile);
      const result = await apiPostForm('/api/hod/config/timetable/preview', formData);
      setTimetablePreview(result.preview);
      setMessage({
        type: result.preview?.valid ? 'success' : 'error',
        text: result.preview?.valid
          ? `Preview passed for ${result.preview.row_count} timetable rows.`
          : `Preview found ${result.preview?.errors?.length || 0} blocking issue(s).`,
      });
    } catch (previewError) {
      setTimetablePreview(null);
      setMessage({ type: 'error', text: previewError.message || 'Timetable preview failed.' });
    } finally {
      setBusy('');
    }
  };

  const applyTimetable = async () => {
    const formError = timetableApplyError({
      preview: timetablePreview,
      reason: timetableReason,
      confirmation: timetableConfirmation,
      file: timetableFile,
    });
    if (formError) {
      setMessage({ type: 'error', text: formError });
      return;
    }
    if (!window.confirm('Replace the authoritative timetable now? Historical attendance sessions will not be changed.')) return;
    const ok = await runAction(
      'timetable-apply',
      async () => {
        const formData = new FormData();
        formData.append('timetable', timetableFile);
        formData.append('expected_sha256', timetablePreview.file_sha256);
        formData.append('reason', timetableReason.trim());
        formData.append('confirmation', timetableConfirmation);
        return apiPostForm('/api/hod/config/timetable/apply', formData);
      },
      'Timetable replaced safely with backup and audit history.',
    );
    if (ok) {
      setTimetableFile(null);
      setTimetablePreview(null);
      setTimetableReason('');
      setTimetableConfirmation('');
    }
  };

  const rollbackLatest = async () => {
    const changeId = config?.current_change?.change_id || '';
    if (!canRollbackLatest(config)) {
      setMessage({ type: 'error', text: 'There is no applied configuration change available for rollback.' });
      return;
    }
    if (rollbackConfirmation !== changeId) {
      setMessage({ type: 'error', text: 'Type the exact latest change ID to confirm rollback.' });
      return;
    }
    if (rollbackReason.trim().length < 8) {
      setMessage({ type: 'error', text: 'Enter a rollback reason of at least 8 characters.' });
      return;
    }
    if (!window.confirm(`Roll back ${changeId}? Current configuration hashes must still match exactly.`)) return;
    const ok = await runAction(
      'rollback',
      () => apiPost('/api/hod/config/rollback', {
        change_id: changeId,
        reason: rollbackReason.trim(),
      }),
      `Configuration change ${changeId} rolled back safely.`,
    );
    if (ok) {
      setRollbackReason('');
      setRollbackConfirmation('');
    }
  };

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
        subtitle="Govern faculty access, subject ownership, timetable configuration, model coverage, and immutable change history."
        actions={<button className="button secondary" type="button" onClick={load} disabled={loading || Boolean(busy)}>{loading ? 'Refreshing…' : 'Refresh'}</button>}
      />

      <ConfigMessage message={message} />
      {error && <section className="connection-banner offline"><div><strong>HOD control unavailable</strong><p>{error}</p></div></section>}
      {!error && loading && <section className="connection-banner warning"><div><strong>Checking governance data</strong><p>Reading role, roster, timetable, coverage, and lifecycle sources.</p></div></section>}

      {data && config && (
        <>
          <section className="stats-grid">
            <StatCard label="Faculty" value={data.faculty?.length || 0} hint="Backend role registry" />
            <StatCard label="Subjects" value={data.subjects?.length || 0} hint="Mapped course tracks" tone="info" />
            <StatCard label="Unique students" value={data.coverage?.total_students ?? '—'} hint="Authoritative roster union" />
            <StatCard label="Embedding covered" value={data.coverage?.embedding_source_available ? data.coverage.embedding_available : '—'} hint={data.coverage?.embedding_source_available ? 'Current production summary' : 'Summary unavailable'} tone="success" />
          </section>

          <section className="two-column hod-control-grid">
            <article className="panel">
              <div className="panel-title-row"><div><h3>Production recognition state</h3><p className="muted">Configuration writes cannot alter this lifecycle state.</p></div><span className="mode-pill">{data.production?.status || 'unknown'}</span></div>
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
              <div className="panel-title-row"><div><h3>Configuration safety</h3><p className="muted">All successful writes are backed up, hashed, audited, and rollback-aware.</p></div><span className="mode-pill success">Guarded writes</span></div>
              <dl className="detail-list">
                <div><dt>Active jobs</dt><dd>Block every configuration write</dd></div>
                <div><dt>Roster writes</dt><dd>Disabled</dd></div>
                <div><dt>Embedding/model writes</dt><dd>Disabled</dd></div>
                <div><dt>Historical attendance</dt><dd>Never rewritten</dd></div>
                <div><dt>Latest change</dt><dd>{config.current_change?.change_id || 'No changes yet'}</dd></div>
              </dl>
            </article>
          </section>

          <section className="panel configuration-panel">
            <div className="panel-title-row">
              <div><h3>Faculty accounts</h3><p className="muted">Create or edit faculty accounts. Passwords are never returned by the API; new or replaced passwords are stored as PBKDF2 hashes.</p></div>
              <button className="button secondary" type="button" onClick={startCreateFaculty} disabled={Boolean(busy)}>New faculty</button>
            </div>
            <div className="configuration-split">
              <div className="compact-list">
                {(config.faculty || []).map((faculty) => (
                  <div className="compact-item config-faculty-item" key={faculty.id}>
                    <div>
                      <strong>{faculty.name}</strong>
                      <span>@{faculty.username} · {(faculty.subjects || []).join(', ') || 'Unassigned'} · {faculty.credential_format || 'credential'}</span>
                    </div>
                    <button className="button ghost small" type="button" onClick={() => startEditFaculty(faculty)} disabled={Boolean(busy)}>Edit</button>
                  </div>
                ))}
              </div>
              <form className="config-form" onSubmit={submitFaculty}>
                <h4>{facultyForm.action === 'create' ? 'Create faculty account' : `Edit ${facultyForm.faculty_id}`}</h4>
                <div className="form-grid two">
                  <label>Faculty ID<input className="input" value={facultyForm.faculty_id} disabled={facultyForm.action === 'update'} onChange={(event) => setFacultyForm({ ...facultyForm, faculty_id: event.target.value })} placeholder="faculty-id" /></label>
                  <label>Username<input className="input" value={facultyForm.username} onChange={(event) => setFacultyForm({ ...facultyForm, username: event.target.value })} placeholder="login username" /></label>
                  <label className="span-two">Faculty name<input className="input" value={facultyForm.name} onChange={(event) => setFacultyForm({ ...facultyForm, name: event.target.value })} placeholder="Dr. Faculty Name" /></label>
                  <label className="span-two">{facultyForm.action === 'create' ? 'Password' : 'New password (optional)'}<input className="input" type="password" value={facultyForm.password} onChange={(event) => setFacultyForm({ ...facultyForm, password: event.target.value })} autoComplete="new-password" /></label>
                  <label className="span-two">Reason<input className="input" value={facultyForm.reason} onChange={(event) => setFacultyForm({ ...facultyForm, reason: event.target.value })} placeholder="Why this account change is needed" /></label>
                </div>
                <button className="button primary" type="submit" disabled={Boolean(busy)}>{busy === 'faculty' ? 'Saving…' : facultyForm.action === 'create' ? 'Create faculty' : 'Save account'}</button>
              </form>
            </div>
          </section>

          <section className="panel configuration-panel">
            <div className="panel-title-row"><div><h3>Subject assignment</h3><p className="muted">One faculty owner per subject. Applying a change updates role access, subject metadata, and every matching timetable instructor component atomically.</p></div></div>
            <div className="subject-assignment-grid">
              {(config.subjects || []).map((item) => (
                <button key={item.abbr} className={`subject-assignment-card ${assignment.subject === item.abbr ? 'selected' : ''}`} type="button" onClick={() => chooseAssignment(item)}>
                  <strong>{item.abbr}</strong>
                  <span>{item.course_name}</span>
                  <small>{item.faculty_name || 'Unassigned'}</small>
                </button>
              ))}
            </div>
            <form className="config-inline-form" onSubmit={submitAssignment}>
              <label>Subject<select className="input" value={assignment.subject} onChange={(event) => setAssignment({ ...assignment, subject: event.target.value })}>{(config.subjects || []).map((item) => <option value={item.abbr} key={item.abbr}>{item.abbr}</option>)}</select></label>
              <label>Faculty<select className="input" value={assignment.faculty_id} onChange={(event) => setAssignment({ ...assignment, faculty_id: event.target.value })}>{(config.faculty || []).map((item) => <option value={item.id} key={item.id}>{item.name}</option>)}</select></label>
              <label className="grow">Reason<input className="input" value={assignment.reason} onChange={(event) => setAssignment({ ...assignment, reason: event.target.value })} placeholder="Why this subject is being reassigned" /></label>
              <button className="button primary" type="submit" disabled={Boolean(busy)}>{busy === 'assignment' ? 'Applying…' : 'Apply assignment'}</button>
            </form>
          </section>

          <section className="panel configuration-panel">
            <div className="panel-title-row"><div><h3>Timetable replacement</h3><p className="muted">Preview is mandatory. The upload must contain the exact B51 35-slot structure and known subject codes. Historical sessions remain unchanged.</p></div><span className={`coverage-badge ${config.timetable?.valid ? 'ready' : 'missing'}`}>{config.timetable?.valid ? 'Current timetable valid' : 'Current timetable invalid'}</span></div>
            <div className="timetable-config-grid">
              <div className="config-upload-card">
                <label className="button secondary file-button">Choose timetable CSV<input hidden type="file" accept=".csv,text/csv" onChange={(event) => { setTimetableFile(event.target.files?.[0] || null); setTimetablePreview(null); setMessage(null); }} /></label>
                <strong>{timetableFile?.name || 'No CSV selected'}</strong>
                <button className="button primary" type="button" onClick={previewTimetable} disabled={!timetableFile || Boolean(busy)}>{busy === 'timetable-preview' ? 'Checking…' : 'Preview & validate'}</button>
              </div>
              <div className="config-preview-card">
                {!previewSummary && <p className="muted">Preview a CSV to see row, day, subject, warning, and error counts.</p>}
                {previewSummary && (
                  <>
                    <div className="panel-title-row"><strong>{previewSummary.valid ? 'Preview passed' : 'Preview blocked'}</strong><span className={`coverage-badge ${previewSummary.valid ? 'ready' : 'missing'}`}>{previewSummary.rows} rows</span></div>
                    <p>{previewSummary.days || 'No day summary'}</p>
                    <p>{previewSummary.subjects || 'No subject summary'}</p>
                    <small>{previewSummary.warningCount} warning(s) · {previewSummary.errorCount} error(s)</small>
                    {(timetablePreview?.errors || []).slice(0, 6).map((item) => <div className="config-error-line" key={item}>{item}</div>)}
                    {(timetablePreview?.warnings || []).slice(0, 4).map((item) => <div className="config-warning-line" key={item}>{item}</div>)}
                  </>
                )}
              </div>
            </div>
            <div className="config-inline-form timetable-apply-form">
              <label className="grow">Reason<input className="input" value={timetableReason} onChange={(event) => setTimetableReason(event.target.value)} placeholder="Why the timetable is being replaced" /></label>
              <label>Confirmation<input className="input" value={timetableConfirmation} onChange={(event) => setTimetableConfirmation(event.target.value)} placeholder={TIMETABLE_CONFIRMATION} /></label>
              <button className="button danger" type="button" onClick={applyTimetable} disabled={Boolean(busy) || !timetablePreview?.valid}>{busy === 'timetable-apply' ? 'Replacing…' : 'Replace timetable'}</button>
            </div>
          </section>

          <section className="panel configuration-panel">
            <div className="panel-title-row"><div><h3>Configuration history & rollback</h3><p className="muted">Only the latest successfully applied configuration transaction can be rolled back, and only when current hashes still match.</p></div><span className="mode-pill">{config.history?.length || 0} audited</span></div>
            <div className="audit-list">
              {(config.history || []).map((item) => (
                <div className="audit-item" key={item.change_id}>
                  <div><strong>{changeTypeLabel(item.change_type)}</strong><span>{item.change_id}</span><small>{item.created_at} · {item.actor?.name || item.actor?.id} · {item.reason}</small></div>
                  <span className={`coverage-badge ${item.rollback ? 'neutral' : 'ready'}`}>{item.rollback ? 'Rolled back' : 'Applied'}</span>
                </div>
              ))}
              {!config.history?.length && <div className="empty-state compact">No HOD configuration writes have been made yet.</div>}
            </div>
            <div className="rollback-box">
              <strong>Rollback latest change</strong>
              <p className="muted">Type the exact change ID shown below. Rollback never touches roster membership, attendance, embeddings, or models.</p>
              <code>{config.current_change?.change_id || 'No applied change'}</code>
              <div className="config-inline-form">
                <label className="grow">Rollback reason<input className="input" value={rollbackReason} onChange={(event) => setRollbackReason(event.target.value)} placeholder="Why this configuration must be restored" /></label>
                <label>Confirm change ID<input className="input" value={rollbackConfirmation} onChange={(event) => setRollbackConfirmation(event.target.value)} placeholder={config.current_change?.change_id || 'change id'} /></label>
                <button className="button danger" type="button" onClick={rollbackLatest} disabled={!canRollbackLatest(config) || Boolean(busy)}>{busy === 'rollback' ? 'Rolling back…' : 'Rollback latest'}</button>
              </div>
            </div>
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
