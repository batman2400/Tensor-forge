import React from 'react';
import { HealthResponse } from '../api/types';
import './Header.css';

interface HeaderProps {
  health: HealthResponse | null;
  isOnline: boolean;
  theme: 'light' | 'dark';
  onToggleTheme: () => void;
  isUnlocked: boolean;
  onLock: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  health,
  isOnline,
  theme,
  onToggleTheme,
  isUnlocked,
  onLock,
}) => {
  const version = health?.model_version ? health.model_version.split('-')[0] : 'v2.0';
  const device = health?.device || 'cpu';

  return (
    <header className="desk-header">
      <div className="brand">
        TensorForge<small>Dispatch Desk</small>
      </div>

      <div className="hl">
        <span className="health-status">
          <span className={`health-dot ${isOnline ? 'online' : 'offline'}`} />
          {isOnline
            ? `model loaded · ${version} · ${device}`
            : 'service offline'}
        </span>

        <div className="actions-group">
          <button
            id="theme"
            onClick={onToggleTheme}
            aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
          >
            {theme === 'dark' ? 'Light' : 'Dark'}
          </button>

          {isUnlocked && (
            <button
              onClick={onLock}
              className="lock-btn"
              title="Lock session and clear API key"
              aria-label="Lock session"
            >
              🔒 Lock
            </button>
          )}
        </div>
      </div>
    </header>
  );
};
