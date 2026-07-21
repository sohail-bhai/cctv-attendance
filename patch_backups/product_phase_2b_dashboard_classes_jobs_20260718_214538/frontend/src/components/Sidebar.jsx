import { NavLink } from 'react-router-dom';
import { useAuth } from '../auth/AuthContext.jsx';

const adminLinks = [
  { to: '/', label: 'Home', icon: '🏠' },
  { to: '/live-demo', label: 'Live Demo', icon: '🎥' },
  { to: '/take-attendance', label: 'Take Attendance', icon: '✅' },
  { to: '/reports', label: 'Attendance Reports', icon: '📊' },
  { to: '/manual-review', label: 'Review & Corrections', icon: '✍️' },
  { to: '/faculty-activity', label: 'Faculty Activity', icon: '📌' },
  { to: '/students', label: 'Students', icon: '👥' },
  { to: '/settings', label: 'Settings', icon: '⚙️' },
  { to: '/jobs', label: 'Jobs', icon: '↻' },
  { to: '/help', label: 'Help', icon: '❓' },
];

const facultyLinks = [
  { to: '/', label: 'My Dashboard', icon: '🏠' },
  { to: '/live-demo', label: 'Live Demo', icon: '🎥' },
  { to: '/take-attendance', label: 'My Classes', icon: '✅' },
  { to: '/reports', label: 'My Reports', icon: '📊' },
  { to: '/manual-review', label: 'Review Students', icon: '✍️' },
  { to: '/students', label: 'My Students', icon: '👥' },
  { to: '/help', label: 'Help', icon: '❓' },
];

export default function Sidebar() {
  const { currentUser, isAdmin } = useAuth();
  const links = isAdmin ? adminLinks : facultyLinks;
  const initials = currentUser?.role === 'admin' ? 'A' : currentUser?.name?.split(' ').filter(Boolean).slice(-2).map((x) => x[0]).join('') || 'F';

  return (
    <aside className="sidebar">
      <div className="brand-block">
        <div className="brand-logo">SU</div>
        <div>
          <h2>Sreenidhi</h2>
          <p>Smart Attendance</p>
        </div>
      </div>

      <div className="admin-card user-card">
        <div className="admin-avatar">{initials}</div>
        <div>
          <strong>{currentUser?.name}</strong>
          <span>{currentUser?.roleLabel}</span>
          {!isAdmin && <small>{(currentUser?.subjects || []).join(', ')}</small>}
        </div>
      </div>

      <nav className="nav-list">
        {links.map((link) => (
          <NavLink key={link.to} to={link.to} className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
            <span>{link.icon}</span>
            {link.label}
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-footer">
        <span className="pulse-dot" /> Role: {isAdmin ? 'Main Admin' : 'Faculty'}
      </div>
    </aside>
  );
}
