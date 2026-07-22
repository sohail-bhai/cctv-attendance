import { createContext, useContext, useEffect, useMemo, useState } from 'react';
import { apiGetResult, apiPost, AUTH_INVALID_EVENT } from '../api/client.js';

const STORAGE_KEY = 'sreenidhi_attendance_user';
const AuthContext = createContext(null);

function readStoredUser() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null');
  } catch {
    return null;
  }
}

function persistUser(user) {
  if (user) localStorage.setItem(STORAGE_KEY, JSON.stringify(user));
  else localStorage.removeItem(STORAGE_KEY);
}

export function AuthProvider({ children }) {
  const [currentUser, setCurrentUser] = useState(readStoredUser);
  const [authReady, setAuthReady] = useState(false);

  useEffect(() => {
    const clearInvalidSession = () => {
      persistUser(null);
      setCurrentUser(null);
      setAuthReady(true);
    };
    window.addEventListener(AUTH_INVALID_EVENT, clearInvalidSession);
    return () => window.removeEventListener(AUTH_INVALID_EVENT, clearInvalidSession);
  }, []);

  useEffect(() => {
    let cancelled = false;
    const stored = readStoredUser();
    if (!stored?.id || !stored?.sessionToken) {
      persistUser(null);
      setCurrentUser(null);
      setAuthReady(true);
      return () => { cancelled = true; };
    }

    apiGetResult('/api/auth/me', { cache: 'no-store' }).then((result) => {
      if (cancelled) return;
      if (result.ok && result.data?.user) {
        const verified = {
          ...result.data.user,
          sessionToken: stored.sessionToken,
          sessionExpiresAt: stored.sessionExpiresAt,
        };
        persistUser(verified);
        setCurrentUser(verified);
      } else if (!result.networkError) {
        // A definite 401/invalid identity clears stale access. Offline mode may
        // retain the last verified role, while all mutations remain disabled.
        persistUser(null);
        setCurrentUser(null);
      }
      setAuthReady(true);
    });

    return () => { cancelled = true; };
  }, []);

  const login = async (username, password) => {
    try {
      const result = await apiPost('/api/auth/login', { username, password });
      const user = result?.user;
      const sessionToken = result?.session_token;
      if (!user || !sessionToken) return { ok: false, error: 'The backend returned an invalid login session.' };
      const verified = {
        ...user,
        sessionToken,
        sessionExpiresAt: result?.session_expires_at || null,
      };
      persistUser(verified);
      setCurrentUser(verified);
      return { ok: true, user: verified };
    } catch (error) {
      return { ok: false, error: error.message || 'Login failed.' };
    }
  };

  const logout = () => {
    apiPost('/api/auth/logout', {}).catch(() => {});
    persistUser(null);
    setCurrentUser(null);
  };

  const isAdmin = currentUser?.role === 'admin' || Boolean(currentUser?.canSeeAll);
  const value = useMemo(() => ({
    currentUser,
    user: currentUser,
    isLoggedIn: Boolean(currentUser),
    isAdmin,
    authReady,
    login,
    logout,
  }), [authReady, currentUser, isAdmin]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const value = useContext(AuthContext);
  if (!value) throw new Error('useAuth must be used inside AuthProvider');
  return value;
}

export function getStoredAuthUser() {
  return readStoredUser();
}
