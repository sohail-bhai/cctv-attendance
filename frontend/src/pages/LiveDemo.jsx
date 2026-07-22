import { useCallback, useEffect, useState } from 'react';
import PageHeader from '../components/PageHeader.jsx';
import { apiGet, apiGetBlobResult, apiPost } from '../api/client.js';
import { useAuth } from '../auth/AuthContext.jsx';

const profileOptions = {
  presentation: {
    label: 'Presentation mode',
    hint: 'Balanced detection for smooth demo flow',
    match_threshold: 0.48,
    margin_threshold: 0.08,
    detection_score: 0.78,
    process_every_n: 2,
  },
  strict: {
    label: 'Strict unknown protection',
    hint: 'Safer when unknown faces should not get names',
    match_threshold: 0.54,
    margin_threshold: 0.10,
    detection_score: 0.82,
    process_every_n: 2,
  },
  fast: {
    label: 'Fast preview',
    hint: 'Lower CPU load for weaker laptops',
    match_threshold: 0.46,
    margin_threshold: 0.07,
    detection_score: 0.75,
    process_every_n: 3,
  },
};

const styles = {
  page: {
    display: 'grid',
    gap: 22,
  },
  topGrid: {
    display: 'grid',
    gridTemplateColumns: 'minmax(0, 1.55fr) minmax(340px, 0.85fr)',
    gap: 22,
    alignItems: 'stretch',
  },
  commandPanel: {
    borderRadius: 28,
    border: '1px solid rgba(15, 118, 110, 0.18)',
    background: 'linear-gradient(145deg, #071827 0%, #08243a 48%, #0d3b4b 100%)',
    color: '#f8fbff',
    boxShadow: '0 24px 70px rgba(7, 24, 39, 0.28)',
    overflow: 'hidden',
  },
  commandInner: {
    padding: 24,
    display: 'grid',
    gridTemplateColumns: '1fr auto',
    gap: 18,
    alignItems: 'center',
  },
  statusPill: (running) => ({
    display: 'inline-flex',
    alignItems: 'center',
    gap: 9,
    width: 'fit-content',
    padding: '9px 13px',
    borderRadius: 999,
    background: running ? 'rgba(16, 185, 129, 0.16)' : 'rgba(148, 163, 184, 0.15)',
    color: running ? '#7fffd4' : '#dbeafe',
    border: running ? '1px solid rgba(16, 185, 129, 0.35)' : '1px solid rgba(148, 163, 184, 0.22)',
    fontWeight: 900,
    fontSize: 13,
  }),
  dot: (running) => ({
    width: 10,
    height: 10,
    borderRadius: '50%',
    background: running ? '#10b981' : '#94a3b8',
    boxShadow: running ? '0 0 0 7px rgba(16, 185, 129, 0.16)' : 'none',
  }),
  title: {
    margin: '14px 0 8px',
    fontSize: 34,
    lineHeight: 1.05,
    letterSpacing: '-0.04em',
  },
  sub: {
    margin: 0,
    maxWidth: 760,
    color: '#bdd7e8',
    fontSize: 15,
    lineHeight: 1.7,
  },
  controlGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(4, minmax(140px, 1fr))',
    gap: 12,
    padding: '0 24px 24px',
  },
  controlCard: {
    borderRadius: 18,
    padding: 14,
    background: 'rgba(255,255,255,0.07)',
    border: '1px solid rgba(255,255,255,0.12)',
  },
  label: {
    display: 'block',
    fontSize: 11,
    textTransform: 'uppercase',
    letterSpacing: '0.12em',
    color: '#89b7cf',
    fontWeight: 900,
    marginBottom: 8,
  },
  input: {
    width: '100%',
    border: '1px solid rgba(255,255,255,0.18)',
    borderRadius: 13,
    background: 'rgba(3, 12, 22, 0.55)',
    color: '#ffffff',
    padding: '11px 12px',
    outline: 'none',
    fontWeight: 850,
  },
  buttonRow: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: 10,
    justifyContent: 'flex-end',
  },
  button: (kind = 'primary', disabled = false) => {
    const map = {
      primary: ['linear-gradient(135deg, #06b6d4, #10b981)', '#04202c'],
      danger: ['linear-gradient(135deg, #ef4444, #f97316)', '#fff'],
      ghost: ['rgba(255,255,255,0.08)', '#eaf6ff'],
    };
    return {
      border: kind === 'ghost' ? '1px solid rgba(255,255,255,0.16)' : '0',
      borderRadius: 15,
      padding: '12px 17px',
      background: disabled ? 'rgba(148, 163, 184, 0.28)' : map[kind][0],
      color: disabled ? '#cbd5e1' : map[kind][1],
      fontWeight: 950,
      cursor: disabled ? 'not-allowed' : 'pointer',
      boxShadow: disabled || kind === 'ghost' ? 'none' : '0 14px 30px rgba(16, 185, 129, 0.24)',
      minWidth: 128,
    };
  },
  videoShell: {
    borderRadius: 28,
    border: '1px solid rgba(8, 145, 178, 0.20)',
    background: '#071827',
    boxShadow: '0 24px 70px rgba(15, 23, 42, 0.18)',
    overflow: 'hidden',
  },
  videoTop: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: 12,
    padding: '15px 18px',
    borderBottom: '1px solid rgba(255,255,255,0.08)',
    color: '#e6f7ff',
  },
  stream: {
    display: 'block',
    width: '100%',
    aspectRatio: '16 / 9',
    objectFit: 'cover',
    background: '#071827',
  },
  telemetry: {
    borderRadius: 28,
    border: '1px solid rgba(226, 232, 240, 0.9)',
    background: 'rgba(255,255,255,0.92)',
    boxShadow: '0 20px 55px rgba(15, 23, 42, 0.10)',
    overflow: 'hidden',
  },
  telemetryHead: {
    padding: 20,
    borderBottom: '1px solid #e2e8f0',
    display: 'flex',
    justifyContent: 'space-between',
    gap: 12,
    alignItems: 'center',
  },
  statGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
    gap: 10,
    padding: 16,
  },
  stat: {
    padding: 14,
    borderRadius: 18,
    background: '#f8fafc',
    border: '1px solid #e2e8f0',
  },
  statValue: {
    display: 'block',
    color: '#082f49',
    fontSize: 24,
    fontWeight: 950,
    letterSpacing: '-0.04em',
  },
  statLabel: {
    color: '#64748b',
    fontSize: 12,
    fontWeight: 800,
  },
  tablePanel: {
    borderRadius: 28,
    border: '1px solid #e2e8f0',
    background: '#ffffff',
    boxShadow: '0 20px 55px rgba(15, 23, 42, 0.08)',
    overflow: 'hidden',
  },
  sectionHead: {
    padding: '18px 20px',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: 12,
    borderBottom: '1px solid #e2e8f0',
  },
  tableWrap: {
    overflowX: 'auto',
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse',
  },
  th: {
    textAlign: 'left',
    padding: '13px 16px',
    background: '#f8fafc',
    color: '#64748b',
    fontSize: 12,
    textTransform: 'uppercase',
    letterSpacing: '0.08em',
  },
  td: {
    padding: '13px 16px',
    borderTop: '1px solid #e2e8f0',
    color: '#0f172a',
    fontWeight: 700,
  },
  badge: (status) => ({
    display: 'inline-flex',
    alignItems: 'center',
    borderRadius: 999,
    padding: '7px 10px',
    fontSize: 12,
    fontWeight: 950,
    background: status === 'Recognized' ? '#dcfce7' : '#ffedd5',
    color: status === 'Recognized' ? '#047857' : '#c2410c',
  }),
  empty: {
    padding: 28,
    color: '#64748b',
    textAlign: 'center',
    fontWeight: 800,
  },
  pipeline: {
    display: 'grid',
    gridTemplateColumns: 'repeat(4, minmax(0, 1fr))',
    gap: 12,
  },
  pipeCard: {
    borderRadius: 22,
    border: '1px solid #dbeafe',
    background: 'linear-gradient(180deg, #ffffff, #f8fbff)',
    padding: 16,
  },
};

function valueOrDash(value) {
  return value === null || value === undefined || value === '' ? '-' : value;
}

function formatScore(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(2) : '-';
}

export default function LiveDemo() {
  const { currentUser } = useAuth();
  const [state, setState] = useState(null);
  const [cameraIndex, setCameraIndex] = useState('0');
  const [profile, setProfile] = useState('presentation');
  const [mirror, setMirror] = useState(true);
  const [streamKey, setStreamKey] = useState(Date.now());
  const [feedFrameUrl, setFeedFrameUrl] = useState('');
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');

  const running = Boolean(state?.running);
  const stats = state?.stats || {};
  const events = state?.events || [];
  const recognized = state?.recognized || [];
  const selectedProfile = profileOptions[profile] || profileOptions.presentation;

  const refreshState = useCallback(async () => {
    const data = await apiGet('/api/live-demo/state', null);
    if (data?.state) setState(data.state);
  }, []);

  useEffect(() => {
    refreshState();
    const timer = setInterval(refreshState, 900);
    return () => clearInterval(timer);
  }, [refreshState]);

  useEffect(() => {
    let cancelled = false;
    let activeUrl = '';
    let timer;
    const refreshFrame = async () => {
      const result = await apiGetBlobResult('/api/live-demo/frame');
      if (!cancelled && result.ok && result.blob) {
        const nextUrl = URL.createObjectURL(result.blob);
        if (activeUrl) URL.revokeObjectURL(activeUrl);
        activeUrl = nextUrl;
        setFeedFrameUrl(nextUrl);
      }
      if (!cancelled) timer = setTimeout(refreshFrame, running ? 180 : 900);
    };
    refreshFrame();
    return () => {
      cancelled = true;
      clearTimeout(timer);
      if (activeUrl) URL.revokeObjectURL(activeUrl);
    };
  }, [currentUser?.sessionToken, running, streamKey]);

  async function startDemo() {
    setBusy(true);
    setMessage('Starting camera and loading recognition models...');
    try {
      const payload = {
        camera_index: Number(cameraIndex) || 0,
        mirror,
        aggregate: 'top3',
        display_width: 1120,
        process_width: 960,
        ...selectedProfile,
      };
      const data = await apiPost('/api/live-demo/start', payload);
      setState(data.state);
      setStreamKey(Date.now());
      setMessage(data.message || 'Live demo started.');
    } catch (error) {
      setMessage(error.message || 'Could not start live demo.');
      await refreshState();
    } finally {
      setBusy(false);
    }
  }

  async function stopDemo() {
    setBusy(true);
    setMessage('Stopping camera safely...');
    try {
      const data = await apiPost('/api/live-demo/stop', {});
      setState(data.state);
      setStreamKey(Date.now());
      setMessage(data.message || 'Live demo stopped.');
    } catch (error) {
      setMessage(error.message || 'Could not stop live demo.');
    } finally {
      setBusy(false);
    }
  }

  return (
    <div style={styles.page}>
      <PageHeader
        eyebrow="Demo Mode"
        title="Live Face Recognition Demo"
        subtitle="Open the camera, detect faces in real time, recognize enrolled students, and show live evidence beside the stream."
      />

      <section style={styles.commandPanel}>
        <div style={styles.commandInner}>
          <div>
            <span style={styles.statusPill(running)}><span style={styles.dot(running)} />{running ? 'Live camera running' : 'Camera ready'}</span>
            <h2 style={styles.title}>CCTV attendance engine, shown live.</h2>
            <p style={styles.sub}>
              This demo uses the same YuNet face detector, SFace embedding model, and cosine matching database from the attendance system. The right side updates as faces are detected.
            </p>
          </div>
          <div style={styles.buttonRow}>
            <button type="button" style={styles.button('primary', busy || running)} disabled={busy || running} onClick={startDemo}>▶ Start Demo</button>
            <button type="button" style={styles.button('danger', busy || !running)} disabled={busy || !running} onClick={stopDemo}>■ Stop Camera</button>
          </div>
        </div>

        <div style={styles.controlGrid}>
          <label style={styles.controlCard}>
            <span style={styles.label}>Camera ID</span>
            <input style={styles.input} type="number" min="0" max="8" value={cameraIndex} onChange={(event) => setCameraIndex(event.target.value)} disabled={running || busy} />
          </label>
          <label style={styles.controlCard}>
            <span style={styles.label}>Demo profile</span>
            <select style={styles.input} value={profile} onChange={(event) => setProfile(event.target.value)} disabled={running || busy}>
              {Object.entries(profileOptions).map(([key, option]) => <option key={key} value={key}>{option.label}</option>)}
            </select>
          </label>
          <label style={styles.controlCard}>
            <span style={styles.label}>Mirror preview</span>
            <select style={styles.input} value={mirror ? 'yes' : 'no'} onChange={(event) => setMirror(event.target.value === 'yes')} disabled={running || busy}>
              <option value="yes">Yes - natural webcam view</option>
              <option value="no">No - CCTV direction</option>
            </select>
          </label>
          <div style={styles.controlCard}>
            <span style={styles.label}>Current strictness</span>
            <strong>{selectedProfile.label}</strong>
            <p style={{ margin: '6px 0 0', color: '#bdd7e8', fontSize: 12 }}>{selectedProfile.hint}</p>
          </div>
        </div>
      </section>

      {message && <div style={{ padding: '12px 16px', borderRadius: 18, background: state?.error ? '#fee2e2' : '#e0f2fe', color: state?.error ? '#991b1b' : '#075985', fontWeight: 850 }}>{state?.error || message}</div>}

      <section style={styles.topGrid}>
        <article style={styles.videoShell}>
          <div style={styles.videoTop}>
            <strong>Live annotated camera feed</strong>
            <span style={{ color: '#a9d8e8', fontWeight: 850 }}>Actual camera: {valueOrDash(stats.actual_camera_index)}</span>
          </div>
          {feedFrameUrl
            ? <img style={styles.stream} src={feedFrameUrl} alt="Live face recognition stream" />
            : <div style={{ ...styles.stream, display: 'grid', placeItems: 'center', color: '#bdd7e8' }}>Authenticated preview loadingâ€¦</div>}
        </article>

        <aside style={styles.telemetry}>
          <div style={styles.telemetryHead}>
            <div>
              <p style={{ ...styles.label, margin: 0, color: '#0f766e' }}>Live telemetry</p>
              <h3 style={{ margin: '6px 0 0', color: '#0f172a' }}>Recognition console</h3>
            </div>
            <span style={styles.badge(running ? 'Recognized' : 'Unknown')}>{running ? 'ONLINE' : 'STANDBY'}</span>
          </div>
          <div style={styles.statGrid}>
            <div style={styles.stat}><span style={styles.statValue}>{valueOrDash(stats.faces_detected)}</span><span style={styles.statLabel}>Faces scanned</span></div>
            <div style={styles.stat}><span style={styles.statValue}>{valueOrDash(stats.recognized_total)}</span><span style={styles.statLabel}>Recognized hits</span></div>
            <div style={styles.stat}><span style={styles.statValue}>{valueOrDash(stats.unknown_total)}</span><span style={styles.statLabel}>Unknown hits</span></div>
            <div style={styles.stat}><span style={styles.statValue}>{formatScore(stats.fps)}</span><span style={styles.statLabel}>Preview FPS</span></div>
          </div>
          <div style={{ padding: '0 16px 16px' }}>
            <div style={{ borderRadius: 20, background: '#ecfeff', border: '1px solid #bae6fd', padding: 14 }}>
              <strong style={{ color: '#075985' }}>Model pipeline</strong>
              <p style={{ margin: '6px 0 0', color: '#64748b', lineHeight: 1.55 }}>Camera frame → YuNet face box → SFace embedding → cosine match → live attendance evidence</p>
            </div>
          </div>
        </aside>
      </section>

      <section style={styles.pipeline}>
        {[
          ['01', 'Camera capture', running ? 'Streaming frames from webcam' : 'Waiting for start'],
          ['02', 'Face detection', 'YuNet scans every selected frame'],
          ['03', 'Recognition match', `Threshold ${selectedProfile.match_threshold} · Margin ${selectedProfile.margin_threshold}`],
          ['04', 'Live evidence', `${recognized.length} unique recognized student(s)`],
        ].map(([step, title, text]) => (
          <article key={step} style={styles.pipeCard}>
            <span style={{ color: '#0891b2', fontWeight: 950 }}>{step}</span>
            <h4 style={{ margin: '8px 0 5px', color: '#0f172a' }}>{title}</h4>
            <p style={{ margin: 0, color: '#64748b', fontWeight: 700 }}>{text}</p>
          </article>
        ))}
      </section>

      <section style={{ display: 'grid', gridTemplateColumns: '1.2fr 0.8fr', gap: 22 }}>
        <article style={styles.tablePanel}>
          <div style={styles.sectionHead}>
            <div>
              <p style={{ ...styles.label, margin: 0, color: '#0f766e' }}>Live updating table</p>
              <h3 style={{ margin: '6px 0 0' }}>Latest detections</h3>
            </div>
            <span style={{ color: '#64748b', fontWeight: 850 }}>{events.length} recent event(s)</span>
          </div>
          <div style={styles.tableWrap}>
            <table style={styles.table}>
              <thead><tr><th style={styles.th}>Time</th><th style={styles.th}>Result</th><th style={styles.th}>Roll / Candidate</th><th style={styles.th}>Score</th><th style={styles.th}>Count</th></tr></thead>
              <tbody>
                {events.map((event) => (
                  <tr key={event.id}>
                    <td style={styles.td}>{event.time}</td>
                    <td style={styles.td}><span style={styles.badge(event.status)}>{event.status}</span></td>
                    <td style={styles.td}>{event.status === 'Recognized' ? event.roll : `Unknown · best: ${event.best_candidate || '-'}`}</td>
                    <td style={styles.td}>{formatScore(event.score)}</td>
                    <td style={styles.td}>{event.count}</td>
                  </tr>
                ))}
                {events.length === 0 && <tr><td colSpan="5" style={styles.empty}>No detections yet. Start the demo and face the camera.</td></tr>}
              </tbody>
            </table>
          </div>
        </article>

        <article style={styles.tablePanel}>
          <div style={styles.sectionHead}>
            <div>
              <p style={{ ...styles.label, margin: 0, color: '#0f766e' }}>Recognized list</p>
              <h3 style={{ margin: '6px 0 0' }}>Students seen live</h3>
            </div>
            <span style={{ color: '#64748b', fontWeight: 850 }}>{recognized.length} unique</span>
          </div>
          <div style={{ padding: 16, display: 'grid', gap: 10 }}>
            {recognized.map((student) => (
              <div key={student.roll} style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'center', padding: 14, borderRadius: 18, background: '#f8fafc', border: '1px solid #e2e8f0' }}>
                <div>
                  <strong style={{ color: '#0f172a' }}>{student.roll}</strong>
                  <p style={{ margin: '4px 0 0', color: '#64748b', fontWeight: 700 }}>Last seen {student.last_seen}</p>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <strong style={{ color: '#047857' }}>{student.count} hits</strong>
                  <p style={{ margin: '4px 0 0', color: '#64748b', fontWeight: 700 }}>best {formatScore(student.best_score)}</p>
                </div>
              </div>
            ))}
            {recognized.length === 0 && <div style={styles.empty}>Recognized students will appear here during the demo.</div>}
          </div>
        </article>
      </section>
    </div>
  );
}
