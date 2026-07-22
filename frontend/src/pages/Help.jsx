import PageHeader from '../components/PageHeader.jsx';
import { useAuth } from '../auth/AuthContext.jsx';
import { isAdmin } from '../data/users.js';

export default function Help() {
  const { currentUser } = useAuth();
  const admin = isAdmin(currentUser);
  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="User Guide"
        title="How to use Smart Attendance"
        subtitle="Simple workflow for faculty and admin users. Technical settings are kept away from normal usage."
      />

      <section className="panel workflow-help">
        <h3>{admin ? 'Admin workflow' : 'Faculty workflow'}</h3>
        <div className="workflow-steps">
          <div><span>1</span><strong>Login</strong><p>{admin ? 'Admin can access all classes, reports, settings, and users.' : 'Faculty users only see their subject slots and students.'}</p></div>
          <div><span>2</span><strong>Take Attendance</strong><p>For normal operation, process only an authorized new session. During the controlled professor demo, do not click Process Attendance.</p></div>
          <div><span>3</span><strong>View Report</strong><p>Compare the official reviewed revision with any archived automatic candidate and inspect every status.</p></div>
          <div><span>4</span><strong>Review & Export</strong><p>Resolve doubtful cases with verified classroom information and a reason. Finalization is always explicit.</p></div>
        </div>
      </section>

      <section className="panel">
        <h3>Important terms</h3>
        <div className="help-grid">
          <div><strong>Present</strong><p>Sufficient authoritative identity evidence exists, or an authorized reviewer made a traceable decision.</p></div>
          <div><strong>Needs Review</strong><p>Meaningful evidence exists but remains ambiguous, guarded, flagged, or otherwise unresolved.</p></div>
          <div><strong>Unconfirmed</strong><p>Camera evidence is insufficient to support either presence or absence. It is not an absence.</p></div>
          <div><strong>Missing Enrollment</strong><p>The rostered student has no usable production embedding, so recognition cannot determine attendance.</p></div>
          <div><strong>Absent</strong><p>An explicit resolved outcome. It is never inferred only from non-detection or low-quality footage.</p></div>
          <div><strong>Unknown</strong><p>An unrecognized legacy value that fails closed and remains unresolved.</p></div>
          <div><strong>Full Checkpoint Mode</strong><p>Uses five checkpoints, CP1 through CP5, with front and back classroom clips and conservative tracklet evidence.</p></div>
          <div><strong>Guarded recovery</strong><p>A review-only candidate. It is not automatic attendance authority.</p></div>
          <div><strong>Simultaneous CCM/CVO</strong><p>CCM and CVO happen in the same period but are separate subject groups with separate faculty/student access.</p></div>
        </div>
      </section>
    </div>
  );
}
