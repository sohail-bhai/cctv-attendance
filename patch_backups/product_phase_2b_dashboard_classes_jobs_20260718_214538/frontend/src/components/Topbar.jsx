import { useEffect, useState } from 'react';
import { apiGet } from '../api/client';
import { useAuth } from '../auth/AuthContext.jsx';

export default function Topbar() {
  const [time, setTime] = useState(new Date());
  const [health, setHealth] = useState({ connected: false, ready: false });
  const { currentUser, logout, isAdmin } = useAuth();

  const loadHealth = async () => {
    const data = await apiGet('/api/health', null);
    setHealth({
      connected: Boolean(data?.ok),
      ready: Boolean(data?.ready_for_processing),
      missing: data?.missing_required_files || [],
    });
  };

  useEffect(() => {
    const clockId = setInterval(() => setTime(new Date()), 1000);
    const healthId = setInterval(loadHealth, 5000);
    loadHealth();
    return () => {
      clearInterval(clockId);
      clearInterval(healthId);
    };
  }, []);

  const healthLabel = health.connected
    ? health.ready
      ? 'Backend Connected'
      : 'Backend Connected · Files Missing'
    : 'Backend Offline';

  return (
    <header className="topbar">
      <div>
        <strong>Sreenidhi University</strong>
        <span>{isAdmin ? 'Admin Portal' : 'Faculty Portal'} · {currentUser?.name}</span>
      </div>
      <div className="topbar-right">
        <div className="time-pill">
          {time.toLocaleDateString('en-IN', { weekday: 'short', day: '2-digit', month: 'short' })}
          <b>{time.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</b>
        </div>
        <div className={`health-pill ${health.connected ? 'online' : 'offline'}`} title={(health.missing || []).join('\n')}>
          <span className="pulse-dot" /> {healthLabel}
        </div>
        <button className="button tiny secondary" onClick={logout}>Logout</button>
      </div>
    </header>
  );
}
