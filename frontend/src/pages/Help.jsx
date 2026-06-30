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
          <div><span>2</span><strong>Take Attendance</strong><p>Select the period and click Process. The system checks the available CCTV clips automatically.</p></div>
          <div><span>3</span><strong>View Report</strong><p>Open Attendance Reports to see present, absent, review cases, and evidence.</p></div>
          <div><span>4</span><strong>Review & Export</strong><p>Correct doubtful cases and export the final attendance file.</p></div>
        </div>
      </section>

      <section className="panel">
        <h3>Important terms</h3>
        <div className="help-grid">
          <div><strong>Present</strong><p>The student has enough recognition evidence.</p></div>
          <div><strong>Needs Review</strong><p>The system found weak or borderline evidence. A faculty/admin should check it.</p></div>
          <div><strong>Absent</strong><p>The student had no useful recognition evidence.</p></div>
          <div><strong>Demo Clip Mode</strong><p>Only short sample clips are available. Useful for testing, not final full-period attendance.</p></div>
          <div><strong>Full Checkpoint Mode</strong><p>Uses CP1 to CP5 clips for a class period and applies checkpoint voting.</p></div>
          <div><strong>Simultaneous CCM/CVO</strong><p>CCM and CVO happen in the same period but are separate subject groups with separate faculty/student access.</p></div>
        </div>
      </section>
    </div>
  );
}
