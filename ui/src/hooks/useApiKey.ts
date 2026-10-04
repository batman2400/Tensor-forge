import { useEffect, useState } from 'react';
import { apiClient } from '../api/client';

const STORAGE_KEY = 'tf_api_key';

export function useApiKey() {
  const [apiKey, setApiKeyState] = useState<string | null>(() => {
    try {
      const stored = sessionStorage.getItem(STORAGE_KEY);
      if (stored) {
        apiClient.setApiKey(stored);
        return stored;
      }
    } catch {
      // sessionStorage may fail in private mode
    }
    return null;
  });

  const [authError, setAuthError] = useState<string | null>(null);

  useEffect(() => {
    // When apiClient encounters 401, clear key and surface auth error
    apiClient.onUnauthorized(() => {
      try {
        sessionStorage.removeItem(STORAGE_KEY);
      } catch {
        // ignore
      }
      apiClient.setApiKey(null);
      setApiKeyState(null);
      setAuthError('Your session expired or the API key was revoked.');
    });
  }, []);

  const setKey = (key: string) => {
    const trimmed = key.trim();
    if (!trimmed) return;
    try {
      sessionStorage.setItem(STORAGE_KEY, trimmed);
    } catch {
      // ignore
    }
    apiClient.setApiKey(trimmed);
    setApiKeyState(trimmed);
    setAuthError(null);
  };

  const lock = () => {
    try {
      sessionStorage.removeItem(STORAGE_KEY);
    } catch {
      // ignore
    }
    apiClient.setApiKey(null);
    setApiKeyState(null);
    setAuthError(null);
  };

  return {
    apiKey,
    isUnlocked: Boolean(apiKey),
    setKey,
    lock,
    authError,
    setAuthError,
  };
}
