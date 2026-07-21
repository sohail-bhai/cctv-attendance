import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { apiGetResult } from '../api/client.js';

const BackendHealthContext = createContext(null);

const INITIAL_STATE = {
  checking: true,
  connected: false,
  ready: false,
  missing: [],
  activeJobs: 0,
  workflowState: null,
  error: null,
  lastCheckedAt: null,
};

export function BackendHealthProvider({ children }) {
  const [health, setHealth] = useState(INITIAL_STATE);

  const refreshHealth = useCallback(async () => {
    setHealth((current) => ({ ...current, checking: current.lastCheckedAt === null }));
    const result = await apiGetResult('/api/health', { cache: 'no-store' });
    if (!result.ok) {
      setHealth((current) => ({
        ...current,
        checking: false,
        connected: false,
        ready: false,
        activeJobs: 0,
        error: result.error || 'Backend unavailable.',
        lastCheckedAt: new Date(),
      }));
      return result;
    }

    const data = result.data || {};
    setHealth({
      checking: false,
      connected: Boolean(data.ok),
      ready: Boolean(data.ready_for_processing),
      missing: Array.isArray(data.missing_required_files) ? data.missing_required_files : [],
      activeJobs: Number(data.active_jobs || 0),
      workflowState: data.workflow_state || null,
      error: null,
      lastCheckedAt: new Date(),
    });
    return result;
  }, []);

  useEffect(() => {
    refreshHealth();
    const id = window.setInterval(refreshHealth, 5000);
    return () => window.clearInterval(id);
  }, [refreshHealth]);

  const value = useMemo(() => ({ ...health, refreshHealth }), [health, refreshHealth]);
  return <BackendHealthContext.Provider value={value}>{children}</BackendHealthContext.Provider>;
}

export function useBackendHealth() {
  const value = useContext(BackendHealthContext);
  if (!value) throw new Error('useBackendHealth must be used inside BackendHealthProvider');
  return value;
}
