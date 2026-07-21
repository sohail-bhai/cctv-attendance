import { useEffect, useState } from 'react';
import PageHeader from '../components/PageHeader.jsx';
import StatusBadge from '../components/StatusBadge.jsx';
import { apiCancelJob, apiGet } from '../api/client.js';

function isRunning(job) {
  return job?.status === 'Processing' || job?.status === 'Pending';
}

export default function Jobs() {
  const [jobs, setJobs] = useState([]);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [message, setMessage] = useState(null);

  const load = () => apiGet('/api/status', []).then((data) => setJobs(Array.isArray(data) ? data : []));

  useEffect(() => {
    load();
    if (!autoRefresh) return undefined;
    const id = setInterval(load, 3000);
    return () => clearInterval(id);
  }, [autoRefresh]);

  const cancel = async (jobId) => {
    try {
      const data = await apiCancelJob(jobId);
      setMessage({ type: 'success', text: data.message || `Cancelled ${jobId}` });
      await load();
    } catch (error) {
      setMessage({ type: 'error', text: error.message });
    }
  };

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Backend Processing"
        title="Jobs"
        subtitle="Track Flask jobs, checkpoint progress, failures, and cancellations."
        actions={<button className="button secondary" onClick={() => setAutoRefresh((v) => !v)}>{autoRefresh ? 'Auto-refresh ON' : 'Auto-refresh OFF'}</button>}
      />
      {message && <div className={`notice ${message.type}`}>{message.text}</div>}
      <section className="panel">
        <div className="table-wrap">
          <table>
            <thead><tr><th>Job ID</th><th>Slot</th><th>Status</th><th>Progress</th><th>Created</th><th>Completed</th><th>Result</th><th>Action</th></tr></thead>
            <tbody>
              {jobs.map((job) => (
                <tr key={job.job_id}>
                  <td><strong>{job.job_id}</strong><small>PID: {job.pid || '—'}</small></td>
                  <td>{job.session_date || job.day} {job.period}<small>{job.subject_abbr || job.subject || ''}</small></td>
                  <td><StatusBadge status={job.status} /></td>
                  <td><span className="progress-text">{job.progress_text || '—'}</span>{job.log_tail?.length ? <small>{job.log_tail.at(-1)}</small> : null}</td>
                  <td>{job.created_at || job.started_at || '—'}</td>
                  <td>{job.completed_at || '—'}</td>
                  <td>{job.result ? `${job.result.present_count}/${job.result.total_students} present` : job.error || '—'}</td>
                  <td>{isRunning(job) ? <button className="button tiny danger" onClick={() => cancel(job.job_id)}>Cancel</button> : '—'}</td>
                </tr>
              ))}
              {jobs.length === 0 && <tr><td colSpan="8" className="empty-cell">No jobs found. Start processing from Timetable Control.</td></tr>}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
