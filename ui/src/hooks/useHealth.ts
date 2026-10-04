import { useCallback, useEffect, useState } from 'react';
import { fetchHealth } from '../api/health';
import { HealthResponse } from '../api/types';

export function useHealth(pollIntervalMs = 30000) {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isOnline, setIsOnline] = useState<boolean>(true);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const checkHealth = useCallback(async () => {
    setIsLoading(true);
    try {
      const data = await fetchHealth();
      setHealth(data);
      setIsOnline(data.status === 'ok');
    } catch {
      setIsOnline(false);
      setHealth(null);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    checkHealth();
    if (pollIntervalMs > 0) {
      const timer = setInterval(checkHealth, pollIntervalMs);
      return () => clearInterval(timer);
    }
  }, [checkHealth, pollIntervalMs]);

  return {
    health,
    isOnline,
    isLoading,
    refreshHealth: checkHealth,
  };
}
