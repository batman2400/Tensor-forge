import React from 'react';

interface LoadingSpinnerProps {
  label?: string;
  size?: number;
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  label = 'Loading…',
  size = 18,
}) => {
  return (
    <div
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '8px',
        color: 'var(--muted)',
        fontFamily: 'var(--mono)',
        fontSize: '12px',
      }}
      role="status"
    >
      <svg
        width={size}
        height={size}
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        style={{
          animation: 'spin 1s linear infinite',
        }}
      >
        <style>
          {`
            @keyframes spin {
              from { transform: rotate(0deg); }
              to { transform: rotate(360deg); }
            }
          `}
        </style>
        <path d="M21 12a9 9 0 1 1-6.219-8.56" />
      </svg>
      {label && <span>{label}</span>}
    </div>
  );
};
