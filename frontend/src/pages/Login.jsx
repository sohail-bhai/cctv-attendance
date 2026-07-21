import { Navigate } from 'react-router-dom';
import { useState } from 'react';
import { useAuth } from '../auth/AuthContext.jsx';
import { USERS } from '../data/users.js';

const roleCards = [
  { title: 'HOD', body: 'Monitor every class, roster, faculty assignment, report, and system-readiness warning.' },
  { title: 'Faculty', body: 'View only assigned subjects, process own class slots, review own students, and export reports.' },
];

export default function Login() {
  const { currentUser, login } = useAuth();
  const [username, setUsername] = useState('admin');
  const [password, setPassword] = useState('admin123');
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  if (currentUser) return <Navigate to="/" replace />;

  const submit = async (event) => {
    event.preventDefault();
    setSubmitting(true);
    setError('');
    const result = await login(username, password);
    if (!result.ok) setError(result.error);
    setSubmitting(false);
  };

  const quickFill = (user) => {
    setUsername(user.username);
    setPassword(user.password);
    setError('');
  };

  return (
    <main className="login-page final-login-page">
      <section className="login-hero final-login-hero">
        <div className="login-logo">SU</div>
        <p className="eyebrow login-eyebrow">Sreenidhi University</p>
        <h1>Smart Attendance Portal</h1>
        <p className="login-intro">
          Backend-verified role access for faculty and HOD. Users see only the classes, reports, and students assigned to them.
        </p>

        <div className="login-role-grid">
          {roleCards.map((card) => (
            <article key={card.title} className="login-role-card">
              <strong>{card.title}</strong>
              <span>{card.body}</span>
            </article>
          ))}
        </div>

        <div className="login-flow-card final-login-flow">
          <span>1. Login</span>
          <span>2. Select class</span>
          <span>3. Process attendance</span>
          <span>4. Review & export</span>
        </div>
      </section>

      <section className="login-card final-login-card">
        <div className="login-card-header">
          <div>
            <p className="eyebrow">Secure role access</p>
            <h2>Login</h2>
            <p className="muted">Credentials are verified by the backend role registry.</p>
          </div>
        </div>

        <form onSubmit={submit} className="login-form">
          <label>
            Username
            <input className="input" value={username} onChange={(event) => setUsername(event.target.value)} autoComplete="username" disabled={submitting} />
          </label>
          <label>
            Password
            <input className="input" type="password" value={password} onChange={(event) => setPassword(event.target.value)} autoComplete="current-password" disabled={submitting} />
          </label>
          {error && <div className="notice error">{error}</div>}
          <button className="button login-submit-button" type="submit" disabled={submitting}>
            {submitting ? 'Verifying…' : 'Login'}
          </button>
        </form>

        <div className="demo-login-list final-demo-login-list">
          <p className="muted">Demo accounts</p>
          {USERS.map((user) => (
            <button key={user.id} className="demo-login-button final-demo-login-button" type="button" onClick={() => quickFill(user)} disabled={submitting}>
              <strong>{user.name}</strong>
              <span>{user.role === 'admin' ? 'Head of Department' : user.roleLabel}</span>
              <small>{user.username} / {user.password}</small>
            </button>
          ))}
        </div>
      </section>
    </main>
  );
}
