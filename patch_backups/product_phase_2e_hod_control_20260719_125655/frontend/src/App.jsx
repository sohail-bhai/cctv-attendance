import { Routes, Route, Navigate } from 'react-router-dom';
import Sidebar from './components/Sidebar.jsx';
import Topbar from './components/Topbar.jsx';
import Dashboard from './pages/Dashboard.jsx';
import Timetable from './pages/Timetable.jsx';
import Reports from './pages/Reports.jsx';
import ManualReview from './pages/ManualReview.jsx';
import Jobs from './pages/Jobs.jsx';
import Settings from './pages/Settings.jsx';
import FacultyActivity from './pages/FacultyActivity.jsx';
import Students from './pages/Students.jsx';
import Help from './pages/Help.jsx';
import LiveDemo from './pages/LiveDemo.jsx';
import Login from './pages/Login.jsx';
import { AuthProvider, useAuth } from './auth/AuthContext.jsx';
import { BackendHealthProvider } from './health/BackendHealthContext.jsx';

function ProtectedShell() {
  const { currentUser, isAdmin } = useAuth();

  if (!currentUser) {
    return (
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="*" element={<Navigate to="/login" replace />} />
      </Routes>
    );
  }

  return (
    <div className="app-shell">
      <Sidebar />
      <div className="content-shell">
        <Topbar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/take-attendance" element={<Timetable />} />
            <Route path="/timetable" element={<Timetable />} />
            <Route path="/reports" element={<Reports />} />
            <Route path="/manual-review" element={<ManualReview />} />
            <Route path="/students" element={<Students />} />
            <Route path="/help" element={<Help />} />
            <Route path="/live-demo" element={<LiveDemo />} />
            {isAdmin && <Route path="/faculty-activity" element={<FacultyActivity />} />}
            {isAdmin && <Route path="/jobs" element={<Jobs />} />}
            {isAdmin && <Route path="/settings" element={<Settings />} />}
            <Route path="/login" element={<Navigate to="/" replace />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <BackendHealthProvider>
        <ProtectedShell />
      </BackendHealthProvider>
    </AuthProvider>
  );
}
