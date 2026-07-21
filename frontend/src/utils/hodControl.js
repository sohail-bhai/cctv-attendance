export function coverageLabel(status) {
  const value = String(status || '').toLowerCase();
  if (value === 'ready') return 'Ready';
  if (value === 'dataset_only') return 'Dataset only';
  if (value === 'embedding_only') return 'Embedding only';
  if (value === 'missing_both') return 'Missing dataset & embedding';
  return 'Coverage unavailable';
}

export function coverageTone(status) {
  const value = String(status || '').toLowerCase();
  if (value === 'ready') return 'success';
  if (value === 'source_unavailable') return 'neutral';
  return 'warning';
}

export function filterCoverageRows(rows, { search = '', subject = 'ALL', issuesOnly = false } = {}) {
  const query = String(search || '').trim().toUpperCase();
  const wantedSubject = String(subject || 'ALL').toUpperCase();
  return (Array.isArray(rows) ? rows : []).filter((row) => {
    const subjects = Array.isArray(row.subjects) ? row.subjects.map((value) => String(value).toUpperCase()) : [];
    const matchesSubject = wantedSubject === 'ALL' || subjects.includes(wantedSubject);
    const matchesSearch = !query || String(row.roll || '').toUpperCase().includes(query) || String(row.name || '').toUpperCase().includes(query);
    const matchesIssue = !issuesOnly || !['ready', 'source_unavailable'].includes(String(row.coverage_status || '').toLowerCase());
    return matchesSubject && matchesSearch && matchesIssue;
  });
}

export function sourceValue(value) {
  if (value === true) return 'Available';
  if (value === false) return 'Missing';
  return 'Unknown';
}
