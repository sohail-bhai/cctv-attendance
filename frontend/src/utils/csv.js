export function parseCsv(text) {
  const rows = [];
  let row = [];
  let value = '';
  let inQuotes = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];

    if (char === '"' && inQuotes && next === '"') {
      value += '"';
      i += 1;
      continue;
    }
    if (char === '"') {
      inQuotes = !inQuotes;
      continue;
    }
    if (char === ',' && !inQuotes) {
      row.push(value.trim());
      value = '';
      continue;
    }
    if ((char === '\n' || char === '\r') && !inQuotes) {
      if (char === '\r' && next === '\n') i += 1;
      row.push(value.trim());
      if (row.some((cell) => cell !== '')) rows.push(row);
      row = [];
      value = '';
      continue;
    }
    value += char;
  }
  row.push(value.trim());
  if (row.some((cell) => cell !== '')) rows.push(row);

  if (rows.length === 0) return [];
  const headers = rows[0].map((h) => h.trim());
  return rows.slice(1).map((cells) => {
    const obj = {};
    headers.forEach((header, index) => {
      obj[header] = cells[index] ?? '';
    });
    return obj;
  });
}

export async function fileToRows(file) {
  const text = await file.text();
  return parseCsv(text);
}

export function num(value, fallback = 0) {
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export function getRoll(row) {
  return row.Roll_Number || row.roll_number || row.roll || row.Roll || row.Student || row.student || '';
}

export function getStatus(row) {
  return row.Final_Status || row.Status || row.Present || row.status || 'Unknown';
}

export function getDetectionCount(row) {
  return num(row.Total_Accepted_Detections || row.Detection_Count || row.Total_Detections || row.Accepted_Detections || row.Accepted_Recognitions || row.detection_count || 0);
}

export function getAvgScore(row) {
  return num(row.Average_Score || row.Average_Similarity || row.Avg_Score || row.avg_score || 0);
}

export function getBestScore(row) {
  return num(row.Best_Score || row.Best_Similarity || row.best_score || 0);
}

export function groupBy(rows, keyGetter) {
  return rows.reduce((acc, row) => {
    const key = keyGetter(row) || 'Unknown';
    acc[key] = acc[key] || [];
    acc[key].push(row);
    return acc;
  }, {});
}

export function summarizeAttendance(rows = []) {
  const total = rows.length;
  const present = rows.filter((row) => /yes|present/i.test(getStatus(row))).length;
  const review = rows.filter((row) => /review/i.test(getStatus(row))).length;
  const absent = Math.max(total - present - review, 0);
  const avgScoreValues = rows.map(getAvgScore).filter((score) => score > 0);
  const avgScore = avgScoreValues.length
    ? avgScoreValues.reduce((a, b) => a + b, 0) / avgScoreValues.length
    : 0;
  return {
    total,
    present,
    review,
    absent,
    percentage: total ? Math.round((present / total) * 1000) / 10 : 0,
    avgScore: Math.round(avgScore * 1000) / 1000,
  };
}

export function summarizeDetectionLog(rows = []) {
  const total = rows.length;
  const accepted = rows.filter((row) => String(row.Accepted).toLowerCase() === 'true' || String(row.Reject_Reason).toLowerCase() === 'accepted').length;
  const rejected = total - accepted;
  const byReason = groupBy(rows, (row) => row.Reject_Reason || (String(row.Accepted).toLowerCase() === 'true' ? 'accepted' : 'rejected'));
  const byVideo = groupBy(rows, (row) => row.Video || row.Camera || row.video || 'Unknown camera');
  const cameraRows = Object.entries(byVideo).map(([camera, items]) => ({
    camera,
    detections: items.length,
    accepted: items.filter((row) => String(row.Accepted).toLowerCase() === 'true' || String(row.Reject_Reason).toLowerCase() === 'accepted').length,
    uniqueStudents: new Set(items.map((row) => row.Predicted_Roll || row.Best_Roll || row.Roll_Number).filter(Boolean).filter((r) => !/unknown/i.test(r))).size,
  }));
  return { total, accepted, rejected, byReason, cameraRows };
}
