import { createContext, useContext, useMemo, useState } from 'react';
import { findLogin, isAdmin } from '../data/users.js';

const STORAGE_KEY = 'sreenidhi_attendance_user';
const AuthContext = createContext(null);

function readStoredUser() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null');
  } catch {
    return null;
  }
}

export function AuthProvider({ children }) {
  const [currentUser, setCurrentUser] = useState(readStoredUser);

  const login = (username, password) => {
    const user = findLogin(username, password);
    if (!user) return { ok: false, error: 'Invalid username or password.' };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(user));
    setCurrentUser(user);
    return { ok: true, user };
  };

  const logout = () => {
    localStorage.removeItem(STORAGE_KEY);
    setCurrentUser(null);
  };

  const value = useMemo(() => ({
    currentUser,
    user: currentUser,
    isLoggedIn: Boolean(currentUser),
    isAdmin: isAdmin(currentUser),
    login,
    logout,
  }), [currentUser]);

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
