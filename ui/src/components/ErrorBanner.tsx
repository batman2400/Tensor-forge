import React from 'react';
import { ErrorDetail } from '../api/types';

interface ErrorBannerProps {
  message: string;
  details?: ErrorDetail[];
  onDismiss?: () => void;
}

export const ErrorBanner: React.FC<ErrorBannerProps> = ({ message, details, onDismiss }) => {
  return (
    <div
      role="alert"
      style={{
        border: '1px solid var(--stamp)',
        background: 'var(--error-bg)',
        padding: '12px 14px',
        margin: '12px 0',
        borderRadius: '2px',
        color: 'var(--stamp)',
        fontFamily: 'var(--sans)',
        fontSize: '14px',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <strong style={{ fontFamily: 'var(--mono)', textTransform: 'uppercase', fontSize: '11px', letterSpacing: '0.08em', display: 'block', marginBottom: '4px' }}>
            Error
          </strong>
          <span>{message}</span>
        </div>
        {onDismiss && (
          <button
            onClick={onDismiss}
            aria-label="Dismiss error"
            style={{
              border: 'none',
              background: 'transparent',
              color: 'var(--stamp)',
              cursor: 'pointer',
              padding: '0 4px',
              fontSize: '16px',
            }}
          >
            ×
          </button>
        )}
      </div>

      {details && details.length > 0 && (
        <ul style={{ margin: '8px 0 0', paddingLeft: '18px', fontSize: '13px', fontFamily: 'var(--mono)' }}>
          {details.map((d, i) => (
            <li key={i}>
              <strong>{d.field}</strong>: {d.issue}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};
