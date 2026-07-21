export default function StatusBadge({ status = 'Unknown' }) {
  const clean = String(status || 'Unknown');
  const key = clean.toLowerCase();
  let cls = 'badge neutral';
  if (key.includes('present') || key === 'yes' || key.includes('complete')) cls = 'badge present';
  if (key.includes('absent') || key === 'no') cls = 'badge absent';
  if (key.includes('review') || key.includes('warning')) cls = 'badge review';
  if (key.includes('processing') || key.includes('pending') || key.includes('queued')) cls = 'badge processing';
  if (key.includes('failed')) cls = 'badge failed';
  if (key.includes('cancel')) cls = 'badge cancelled';
  if (key.includes('lunch')) cls = 'badge lunch';
  return <span className={cls}>{clean}</span>;
}
