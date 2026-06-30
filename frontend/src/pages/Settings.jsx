import PageHeader from '../components/PageHeader.jsx';

export default function Settings() {
  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="System Design"
        title="Attendance Logic Settings"
        subtitle="This page documents the logic we should use for MVP 2 and MVP 3."
      />

      <section className="panel split-cards">
        <div>
          <h3>MVP 2 current logic</h3>
          <ul className="check-list">
            <li>Subject comes from timetable slot.</li>
            <li>All videos in the folder are treated as camera angles.</li>
            <li>YuNet detects faces.</li>
            <li>SFace compares with student_embeddings.pkl.</li>
            <li>CSV/Excel/log files are generated for explainability.</li>
          </ul>
        </div>
        <div>
          <h3>Recommended strict thresholds</h3>
          <div className="summary-chip-grid compact">
            <div className="summary-chip"><span>match-threshold</span><strong>0.48</strong></div>
            <div className="summary-chip"><span>margin-threshold</span><strong>0.08</strong></div>
            <div className="summary-chip"><span>min-detections</span><strong>15</strong></div>
            <div className="summary-chip"><span>review min</span><strong>7</strong></div>
          </div>
        </div>
      </section>

      <section className="panel">
        <h3>MVP 3 checkpoint voting plan</h3>
        <div className="timeline">
          <div><strong>09:05</strong><span>Checkpoint 1 after class settles</span></div>
          <div><strong>09:17</strong><span>Checkpoint 2 early-middle</span></div>
          <div><strong>09:30</strong><span>Checkpoint 3 middle</span></div>
          <div><strong>09:43</strong><span>Checkpoint 4 late class</span></div>
        </div>
        <p className="muted">Final rule: recognized in 3/4 checkpoints = Present, 2/4 = Needs Review, 0–1/4 = Absent. Each checkpoint can use any camera angle.</p>
      </section>

      <section className="panel warning-panel">
        <h3>Problems this logic solves</h3>
        <div className="tip-grid">
          <div><strong>Head down while writing</strong><p>Longer clips and four checkpoints reduce one-frame unfair misses.</p></div>
          <div><strong>Detected faces ≠ recognized students</strong><p>Reports should show detection count, accepted count, unknowns, and rejected reasons.</p></div>
          <div><strong>Absent student getting attendance</strong><p>Needs Review prevents weak evidence from directly becoming Present.</p></div>
        </div>
      </section>
    </div>
  );
}
