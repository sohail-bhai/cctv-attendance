import PageHeader from '../components/PageHeader.jsx';

export default function Settings() {
  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="System Design"
        title="Attendance Logic Settings"
        subtitle="Read-only guidance for the current five-checkpoint production contract."
      />

      <section className="panel split-cards">
        <div>
          <h3>Current production logic</h3>
          <ul className="check-list">
            <li>Subject and roster come from the timetable slot.</li>
            <li>Each of CP1 through CP5 requires exact front and back clip bindings.</li>
            <li>YuNet detects faces and SFace compares production embeddings.</li>
            <li>Strict tracklet evidence is automatic; guarded recovery remains review-only.</li>
            <li>Reports preserve official, automatic-candidate, and superseded revisions separately.</li>
          </ul>
        </div>
        <div>
          <h3>Frozen strict thresholds</h3>
          <div className="summary-chip-grid compact">
            <div className="summary-chip"><span>match threshold</span><strong>0.48</strong></div>
            <div className="summary-chip"><span>margin threshold</span><strong>0.08</strong></div>
            <div className="summary-chip"><span>checkpoint minimum</span><strong>2 observations</strong></div>
            <div className="summary-chip"><span>present rule</span><strong>3 of 5</strong></div>
          </div>
        </div>
      </section>

      <section className="panel">
        <h3>Five-checkpoint attendance contract</h3>
        <div className="timeline">
          <div><strong>CP1</strong><span>First settled classroom sample</span></div>
          <div><strong>CP2</strong><span>Early-middle evidence</span></div>
          <div><strong>CP3</strong><span>Middle evidence</span></div>
          <div><strong>CP4</strong><span>Late evidence</span></div>
          <div><strong>CP5</strong><span>End-of-period evidence</span></div>
        </div>
        <p className="muted">Present requires sufficient strict or human-reviewed authority. Needs Review means meaningful unresolved evidence. Unconfirmed means insufficient camera evidence, Missing Enrollment means no usable production embedding, and Absent is only an explicit resolved outcome.</p>
      </section>

      <section className="panel warning-panel">
        <h3>Safety boundaries</h3>
        <div className="tip-grid">
          <div><strong>One-frame limitation</strong><p>Multi-frame tracklets across five checkpoints reduce unfair one-frame misses.</p></div>
          <div><strong>Detection is not identity</strong><p>Reports separate observations, strict acceptance, guarded candidates, reviewed evidence, and rejected mixed tracks.</p></div>
          <div><strong>Non-detection is not absence</strong><p>Unconfirmed remains unresolved until a reviewer makes an explicit, reasoned decision.</p></div>
        </div>
      </section>
    </div>
  );
}
