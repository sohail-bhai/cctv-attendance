function clean(value) {
  return String(value ?? '').trim();
}

export function normalizePeriodLabel(value) {
  const raw = clean(value).toUpperCase();
  if (!raw) return '—';
  return raw.startsWith('P') ? raw : `P${raw}`;
}

export function sessionStateLabel(session = {}) {
  if (session.attendance_finalized || session.review_state === 'finalized') return 'Finalized';
  if (session.roster_complete === false || session.review_state === 'roster_mismatch') return 'Roster Mismatch';
  if (session.unresolved_count > 0 || session.review_state === 'needs_attention' || /needs review/i.test(clean(session.status))) return 'Needs Review';
  if (session.review_state === 'ready_to_finalize') return 'Ready to Finalize';
  return clean(session.status) || 'Completed';
}

export function sessionStateTone(session = {}) {
  const label = sessionStateLabel(session);
  if (label === 'Finalized') return 'success';
  if (label === 'Roster Mismatch') return 'danger';
  if (label === 'Needs Review') return 'warning';
  if (label === 'Ready to Finalize') return 'info';
  return 'neutral';
}

export function sortAttendanceSessions(sessions = []) {
  return [...sessions].sort((a, b) => {
    const aKey = `${clean(a.session_date)}|${clean(a.completed_at || a.finalized_at || a.last_edited_at)}|${normalizePeriodLabel(a.period)}|${clean(a.session_id)}`;
    const bKey = `${clean(b.session_date)}|${clean(b.completed_at || b.finalized_at || b.last_edited_at)}|${normalizePeriodLabel(b.period)}|${clean(b.session_id)}`;
    return bKey.localeCompare(aKey, undefined, { numeric: true, sensitivity: 'base' });
  });
}

export function entryToSlotSummaryRow(entry = {}) {
  return {
    Course_Abbr: entry.subject_abbr || entry.subject_track || entry.subject || entry.course_abbr || '',
    Course_Name: entry.subject_name || entry.course_name || '',
    Day: entry.day || '',
    Period: clean(entry.period || entry.period_number).replace(/^P/i, ''),
    Start_Time: entry.start_time || '',
    End_Time: entry.end_time || '',
    Instructor: entry.faculty_name || entry.instructor || entry.teacher || '',
    Room: entry.room || '',
    Section: entry.section || '',
    Total_Checkpoints: entry.total_checkpoints || entry.Total_Checkpoints || 5,
    Checkpoint_Mode: entry.checkpoint_mode || entry.Checkpoint_Mode || 'full',
    Processing_Runtime_Seconds: entry.processing_runtime_seconds || entry.Processing_Runtime_Seconds || '',
    Camera_Angles_Processed: entry.camera_ids || entry.Camera_Angles_Processed || '',
    Videos_Processed: entry.video_files?.length || entry.Videos_Processed || '',
    Total_Face_Detections: entry.detected_faces || entry.Total_Face_Detections || '',
    Accepted_Recognitions: entry.accepted_recognitions || entry.Accepted_Recognitions || '',
    Rejected_Or_Unknown: entry.rejected_or_unknown || entry.Rejected_Or_Unknown || '',
    Run_Quality_Status: entry.run_quality_status || entry.Run_Quality_Status || '',
    Run_Quality_Reason: entry.run_quality_reason || entry.Run_Quality_Reason || '',
    Requires_Manual_Review: entry.requires_manual_review,
    Attendance_Finalized: entry.attendance_finalized,
    Roster_Complete: entry.roster_complete,
    Roster_Expected_Count: entry.roster_expected_count,
    Roster_Row_Count: entry.roster_row_count,
  };
}

export function correctionReasonErrors(changes = {}) {
  return Object.entries(changes)
    .filter(([, change]) => change?.status && change.status !== 'CLEAR' && !clean(change.reason))
    .map(([roll]) => roll);
}

export function reviewActionState({ pendingChanges = 0, unresolvedCount = 0, rosterComplete = true, finalized = false, busy = false } = {}) {
  return {
    canSave: !busy && pendingChanges > 0,
    canExport: !busy && pendingChanges === 0,
    canFinalize: !busy && !finalized && pendingChanges === 0 && unresolvedCount === 0 && rosterComplete,
    finalized,
  };
}

function checkpointText(value) {
  const text = clean(value);
  return text || 'None recorded';
}

export function evidencePresentation(row = {}, session = {}) {
  const strict = checkpointText(row.Strict_Recognized_Checkpoints ?? row.Strict_Accepted_Checkpoints);
  const guarded = checkpointText(row.Guarded_Recovery_Candidate_Checkpoints);
  const reviewed = checkpointText(row.Reviewed_Tracklet_Checkpoints);
  const mixed = checkpointText(row.Mixed_Track_Checkpoints_Rejected);
  const observations = Number(row.Total_Accepted_Detections ?? row.Detection_Count ?? 0) || 0;
  const guardedAutomatic = String(row.Automatic_Guarded_Recovery_Enabled ?? session.guarded_recovery_automatic ?? '').toLowerCase();
  let authority = 'No accepted identity evidence';
  if (mixed !== 'None recorded') authority = 'Rejected mixed tracklet evidence';
  if (guarded !== 'None recorded') authority = 'Guarded recovery candidate (review only)';
  if (strict !== 'None recorded') authority = 'Strict automatic tracklet evidence';
  if (reviewed !== 'None recorded') authority = 'Human-reviewed tracklet evidence';
  const carryForwardApplied = Boolean(session.review_carry_forward_applied || row.Review_Carry_Forward_Applied === true || String(row.Review_Carry_Forward_Applied || '').toLowerCase() === 'yes');
  return {
    strict,
    guarded,
    reviewed,
    mixed,
    observations,
    authority,
    guardedRecoveryAutomatic: guardedAutomatic === 'true' || guardedAutomatic === 'yes',
    carryForward: carryForwardApplied
      ? `Applied${session.review_carry_forward_source ? ` from ${session.review_carry_forward_source}` : ''}`
      : 'Not applied',
  };
}

export function reportRevisionPresentation(session = {}) {
  const candidate = session.pending_candidate_revision || session.latest_automatic_candidate || null;
  const unresolved = Number(session.unresolved_count || 0);
  return {
    officialLabel: 'Official Reviewed Revision',
    officialRevision: clean(session.authority_revision_id) || 'Current official revision',
    officialAuthority: clean(session.official_recognition_authority) || 'Not declared',
    automaticLabel: 'Automatic Candidate Revision',
    automaticRevision: clean(candidate?.revision_id) || (candidate ? 'Archived automatic candidate' : 'No archived candidate'),
    automaticAuthority: clean(session.automatic_recognition_authority || candidate?.official_recognition_authority) || 'Not declared',
    automaticTotals: candidate
      ? `${Number(candidate.present_count || 0)} Present; ${Number(candidate.needs_review_count || 0)} Needs Review; ${Number(candidate.unconfirmed_count || 0)} Unconfirmed; ${Number(candidate.missing_enrollment_count || 0)} Missing Enrollment; ${Number(candidate.absent_count || 0)} Absent`
      : 'No archived candidate totals',
    historicalLabel: 'Superseded / quality-failed historical result',
    historicalVisible: Boolean(session.source_report_superseded || session.source_report_quality_failed),
    carryForward: session.latest_candidate_matches_official
      ? 'Exact-source evidence matched; reviewed decisions were reused.'
      : session.review_carry_forward_applied
        ? `Applied${session.review_carry_forward_source ? ` from ${session.review_carry_forward_source}` : ''}.`
        : 'Not applied.',
    finalization: session.attendance_finalized
      ? 'Finalized'
      : `Not finalized${unresolved ? `; ${unresolved} unresolved` : '; ready only after explicit review'}`,
  };
}
