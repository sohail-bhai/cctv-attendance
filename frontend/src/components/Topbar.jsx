import { useEffect, useState } from 'react';
import { useAuth } from '../auth/AuthContext.jsx';
import { useBackendHealth } from '../health/BackendHealthContext.jsx';

export default function Topbar() {
  const [time, setTime] = useState(new Date());
  const { currentUser, logout, isAdmin } = useAuth();
  const health = useBackendHealth();

  useEffect(() => {
    const clockId = window.setInterval(() => setTime(new Date()), 1000);
    return () => window.clearInterval(clockId);
  }, []);

  let healthLabel = 'Checking backend';
  let healthTone = 'checking';
  if (!health.checking && !health.connected) {
    healthLabel = 'Backend Offline';
    healthTone = 'offline';
  } else if (health.connected && !health.ready) {
    healthLabel = 'Backend Online · Not Ready';
    healthTone = 'warning';
  } else if (health.connected && health.ready) {
    healthLabel = health.activeJobs > 0 ? `Backend Connected · ${health.activeJobs} Active` : 'Backend Connected';
    healthTone = 'online';
  }

  const healthTitle = health.connected && !health.ready
    ? `Missing required files:\n${health.missing.join('\n') || 'Unknown readiness issue'}`
    : health.error || 'Backend health is checked every five seconds.';

  return (
    <header className="topbar">
      <div>
        <strong>Sreenidhi University</strong>
        <span>{isAdmin ? 'HOD Portal' : 'Faculty Portal'} · {currentUser?.name}</span>
      </div>
      <div className="topbar-right">
        <div className="time-pill">
          {time.toLocaleDateString('en-IN', { weekday: 'short', day: '2-digit', month: 'short' })}
          <b>{time.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</b>
        </div>
        <button className={`health-pill health-button ${healthTone}`} title={healthTitle} onClick={health.refreshHealth} type="button">
          <span className="pulse-dot" /> {healthLabel}
        </button>
        <button className="button tiny secondary" onClick={logout}>Logout</button>
      </div>
    </header>
  );
}
