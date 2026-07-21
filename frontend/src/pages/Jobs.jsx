import { useCallback, useEffect, useMemo, useState } from 'react';
import PageHeader from '../components/PageHeader.jsx';
import StatusBadge from '../components/StatusBadge.jsx';
import JobProgress from '../components/JobProgress.jsx';
import { apiCancelJob, apiGetResult } from '../api/client.js';
import { useBackendHealth } from '../health/BackendHealthContext.jsx';
import { elapsedFromText, isRunningJob } from '../utils/workflow.js';

export default function Jobs() {
  const health = useBackendHealth();
  const [jobs, setJobs] = useState([]);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [message, setMessage] = useState(null);
  const [loading, setLoading] = useState(true);
  const [stale, setStale] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [cancelling, setCancelling] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    const result = await apiGetResult('/api/status', { cache: 'no-store' });
    if (result.ok && Array.isArray(result.data)) {
      setJobs(result.data);
      setStale(false);
      setLastUpdated(new Date());
    } else {
      setStale(true);
      setMessage({ type: 'error', text: result.error || 'Could not load job history.' });
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    if (!autoRefresh || !health.connected) return undefined;
    const id = window.setInterval(load, 3000);
    return () => window.clearInterval(id);
  }, [autoRefresh, health.connected, load]);

  const cancel = async (jobId) => {
    if (!health.connected) {
      setMessage({ type: 'error', text: 'Cancellation is unavailable until the backend reconnects.' });
      return;
    }
    if (!window.confirm(`Cancel job ${jobId}?`)) return;
    setCancelling(jobId);
    try {
      const data = await apiCancelJob(jobId);
      setMessage({ type: 'success', text: data.message || `Cancelled ${jobId}` });
      await load();
      await health.refreshHealth();
    } catch (error) {
      setMessage({ type: 'error', text: error.message });
    } finally {
      setCancelling(null);
    }
  };

  const summary = useMemo(() => ({
    active: jobs.filter(isRunningJob).length,
    completed: jobs.filter((job) => job.status === 'Completed').length,
    review: jobs.filter((job) => job.status === 'Needs Review').length,
    failed: jobs.filter((job) => job.status === 'Failed' || job.status === 'Cancelled').length,
  }), [jobs]);

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="HOD Processing Control"
        title="Processing Jobs"
        subtitle="Persistent job history, checkpoint progress, failures, cancellations, and responsible users."
        actions={
          <div className="page-actions">
            <button className="button secondary" type="button" onClick={() => setAutoRefresh((value) => !value)}>
              Auto-refresh {autoRefresh ? 'ON' : 'OFF'}
            </button>
            <button className="button secondary" type="button" onClick={load} disabled={loading}>{loading ? 'Refreshing…' : 'Refresh now'}</button>
          </div>
        }
      />

      {message && <div className={`notice ${message.type}`}>{message.text}</div>}

      {(!health.connected || stale) && (
        <section className="connection-banner offline">
          <div>
            <strong>{jobs.length ? 'Showing last loaded job data' : 'Job registry unavailable'}</strong>
            <p>The backend could not be reached. No cancellation or live progress action is allowed until it reconnects.</p>
          </div>
          <button className="button tiny secondary" type="button" onClick={load}>Retry</button>
        </section>
      )}

      <section className="stats-grid job-stats-grid">
        <Stat label="Active" value={summary.active} tone="processing" />
        <Stat label="Completed" value={summary.completed} tone="completed" />
        <Stat label="Needs review" value={summary.review} tone="review" />
        <Stat label="Failed / cancelled" value={summary.failed} tone="failed" />
      </section>

      <section className="dashboard-source-row">
        <span className={`data-source-pill ${stale ? 'preview' : 'live'}`}>{stale ? 'Stale cached view' : 'Persistent backend registry'}</span>
        <small>{lastUpdated ? `Updated ${lastUpdated.toLocaleTimeString('en-IN')}` : 'Not loaded yet'}</small>
      </section>

      <section className="job-board">
        {jobs.map((job) => (
          <article className="panel job-board-card" key={job.job_id}>
            <div className="job-card-heading">
              <div>
                <p className="eyebrow">{job.session_date || job.day} · {job.period}</p>
                <h3>{job.subject_abbr || job.subject || 'Attendance job'}</h3>
                <small>{job.session_id || job.key || job.job_id}</small>
              </div>
              <StatusBadge status={job.status} />
            </div>

            <JobProgress job={job} />

            <div className="job-detail-grid">
              <span><small>Job ID</small><strong>{job.job_id}</strong></span>
              <span><small>Started by</small><strong>{job.controlled_by || job.faculty_name || 'Authorized user'}</strong></span>
              <span><small>Created</small><strong>{job.created_at || job.started_at || '—'}</strong></span>
              <span><small>Elapsed</small><strong>{isRunningJob(job) ? elapsedFromText(job.started_at || job.created_at) : '—'}</strong></span>
            </div>

            {job.result && (
              <div className="job-result-strip">
                <strong>{job.result.present_count ?? '—'}/{job.result.total_students ?? '—'} present</strong>
                <span>{job.result.status || 'Result saved'}</span>
              </div>
            )}
            {job.error && <div className="notice error compact-notice">{job.error}</div>}

            <div className="row-actions spread-actions">
              <small>{job.completed_at ? `Completed ${job.completed_at}` : job.is_reprocess ? 'Reprocess job' : 'Initial processing job'}</small>
              {isRunningJob(job) && (
                <button
                  className="button tiny danger"
                  type="button"
                  disabled={!health.connected || cancelling === job.job_id}
                  onClick={() => cancel(job.job_id)}
                >
                  {cancelling === job.job_id ? 'Cancelling…' : 'Cancel'}
                </button>
              )}
            </div>
          </article>
        ))}
        {!loading && jobs.length === 0 && (
          <section className="panel empty-job-board">
            <strong>No processing jobs yet</strong>
            <p>Start attendance from Take Attendance. New jobs will persist here across page refreshes and backend restarts.</p>
          </section>
        )}
      </section>
    </div>
  );
}

function Stat({ label, value, tone }) {
  return (
    <article className={`job-stat ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  );
}
