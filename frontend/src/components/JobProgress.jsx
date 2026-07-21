const PROGRESS_STEPS = [
  { key: 'checking_files', label: 'Checking files' },
  { key: 'processing_clips', label: 'Processing clips' },
  { key: 'recognizing_students', label: 'Recognizing students' },
  { key: 'generating_report', label: 'Generating report' },
];

function clampNumber(value, min, max) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return min;
  return Math.max(min, Math.min(max, parsed));
}

function fallbackProgressStep(job) {
  const text = String(job?.progress_text || '').toLowerCase();
  const stage = String(job?.progress_stage || '').toLowerCase();
  if (job?.status === 'Completed' || stage === 'completed') return PROGRESS_STEPS.length;
  if (job?.status === 'Failed' || job?.status === 'Cancelled') return 0;
  if (stage.includes('generating') || text.includes('saved ') || text.includes('generating')) return 3;
  if (stage.includes('recognizing') || /^cp\d+/.test(text) || text.includes('processed') || text.includes('skipped')) return 2;
  if (stage.includes('processing') || text.includes('camera angle') || text.includes('camera ')) return 1;
  return 0;
}

export function progressModel(job) {
  const activeStep = clampNumber(
    Number.isFinite(Number(job?.progress_step)) ? Number(job.progress_step) : fallbackProgressStep(job),
    0,
    PROGRESS_STEPS.length
  );
  const percent = clampNumber(
    Number.isFinite(Number(job?.progress_percent)) ? Number(job.progress_percent) : activeStep * 25,
    0,
    100
  );
  const status = String(job?.status || '');
  return {
    activeStep,
    percent,
    isComplete: status === 'Completed' || activeStep >= PROGRESS_STEPS.length,
    isError: status === 'Failed' || status === 'Cancelled',
  };
}

export default function JobProgress({ job, compact = false }) {
  const model = progressModel(job);
  const latestLogs = Array.isArray(job?.log_tail) ? job.log_tail.slice(-3) : [];
  const headline = job?.progress_text || job?.status || 'Preparing attendance job...';

  return (
    <div className={`job-progress ${compact ? 'compact' : ''}`}>
      <div className="job-progress-head">
        <span>{headline}</span>
        <strong className={model.isError ? 'error-text' : ''}>{Math.round(model.percent)}%</strong>
      </div>
      <div className="job-progress-track" aria-label={`Progress ${Math.round(model.percent)} percent`}>
        <div className={`job-progress-fill ${model.isError ? 'error' : ''}`} style={{ width: `${model.percent}%` }} />
      </div>
      {!compact && (
        <>
          <div className="job-step-row">
            {PROGRESS_STEPS.map((step, index) => {
              let state = 'pending';
              if (model.isError) state = index === model.activeStep ? 'error' : 'pending';
              else if (model.isComplete || index < model.activeStep) state = 'done';
              else if (index === model.activeStep) state = 'active';
              return <span key={step.key} className={`job-step ${state}`}>{state === 'done' ? '✓ ' : state === 'active' ? '● ' : ''}{step.label}</span>;
            })}
          </div>
          {latestLogs.length > 0 && (
            <div className="job-live-log">
              <strong>Latest activity</strong>
              {latestLogs.map((line, index) => <small key={`${line}-${index}`}>› {line}</small>)}
            </div>
          )}
        </>
      )}
    </div>
  );
}
