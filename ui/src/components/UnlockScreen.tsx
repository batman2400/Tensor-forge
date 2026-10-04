import React, { useState } from 'react';
import { apiClient } from '../api/client';
import { predictSingle } from '../api/predict';
import { HealthResponse } from '../api/types';
import { ErrorBanner } from './ErrorBanner';
import { LoadingSpinner } from './LoadingSpinner';
import './UnlockScreen.css';

interface UnlockScreenProps {
  health: HealthResponse | null;
  isOnline: boolean;
  onUnlock: (key: string) => void;
  initialError?: string | null;
}

export const UnlockScreen: React.FC<UnlockScreenProps> = ({
  health,
  isOnline,
  onUnlock,
  initialError,
}) => {
  const [keyInput, setKeyInput] = useState('');
  const [isValidating, setIsValidating] = useState(false);
  const [error, setError] = useState<string | null>(initialError || null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = keyInput.trim();
    if (!trimmed) {
      setError('Please enter your API key to continue.');
      return;
    }

    setIsValidating(true);
    setError(null);

    // Test the API key with a fast verification request
    const prevKey = apiClient.getApiKey();
    try {
      apiClient.setApiKey(trimmed);
      await predictSingle({
        ticket_id: 'probe-000',
        channel: 'chat',
        text: 'API key authorization verification check',
      });
      // Valid!
      onUnlock(trimmed);
    } catch (err: unknown) {
      apiClient.setApiKey(prevKey);
      if (err && typeof err === 'object' && 'status' in err && (err as { status: number }).status === 401) {
        setError('Invalid API key. Please check your credentials.');
      } else if (err && typeof err === 'object' && 'code' in err && (err as { code: string }).code === 'NETWORK_ERROR') {
        setError('Cannot reach the backend server. Make sure the server is running.');
      } else {
        // Even if the probe failed on validation or model loading, if it wasn't 401, key was accepted
        if (err && typeof err === 'object' && 'status' in err && (err as { status: number }).status !== 401 && (err as { status: number }).status !== 0) {
          onUnlock(trimmed);
        } else {
          setError(err instanceof Error ? err.message : 'Authentication check failed.');
        }
      }
    } finally {
      setIsValidating(false);
    }
  };

  const version = health?.model_version ? health.model_version.split('-')[0] : 'v2.0';

  return (
    <div className="unlock-container">
      <div className="unlock-card">
        <h1>Dispatch Desk</h1>
        <div className="unlock-subtitle">Enter your API key to access the desk.</div>

        {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}

        <form onSubmit={handleSubmit} className="unlock-form">
          <div className="unlock-input-group">
            <label htmlFor="api-key" className="lbl">
              API Key
            </label>
            <input
              id="api-key"
              type="password"
              value={keyInput}
              onChange={(e) => setKeyInput(e.target.value)}
              placeholder="e.g. tf2_... or test-key-dev"
              className="unlock-input"
              autoFocus
              autoComplete="current-password"
            />
            <div className="unlock-help">
              Stored only in browser session storage for this tab. Accepts competition keys (<code>tf2_...</code>) or <code>test-key-dev</code>.
            </div>
          </div>

          <button
            type="submit"
            className="primary"
            disabled={isValidating || !keyInput.trim()}
          >
            {isValidating ? <LoadingSpinner label="Verifying…" /> : 'Unlock Desk'}
          </button>
        </form>

        <div className="unlock-footer">
          <span>Backend status:</span>
          <span>
            <span
              style={{
                display: 'inline-block',
                width: 7,
                height: 7,
                borderRadius: '50%',
                background: isOnline ? 'var(--success)' : 'var(--stamp)',
                marginRight: 6,
              }}
            />
            {isOnline ? `Service ready · ${version}` : 'Service offline'}
          </span>
        </div>
      </div>
    </div>
  );
};
