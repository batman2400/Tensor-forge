import React, { useEffect, useState } from 'react';
import { PredictRequest, PredictResponse } from '../../api/types';
import { formatCategory } from '../../lib/categories';
import { formatConfidence, formatLatency } from '../../lib/format';
import './ResultCard.css';

interface ResultCardProps {
  result: PredictResponse | null;
  request: PredictRequest | null;
}

export const ResultCard: React.FC<ResultCardProps> = ({ result, request }) => {
  const [showJson, setShowJson] = useState(false);
  const [barWidth, setBarWidth] = useState('0%');

  useEffect(() => {
    if (result) {
      // Trigger animation frame for ink-fill confidence bar
      const pct = `${Math.min(100, Math.max(0, Math.round(result.confidence * 100)))}%`;
      const raf = requestAnimationFrame(() => {
        setBarWidth(pct);
      });
      return () => cancelAnimationFrame(raf);
    } else {
      setBarWidth('0%');
    }
  }, [result]);

  if (!result) {
    return (
      <div className="card" aria-live="polite">
        <div className="empty">
          Nothing on the desk yet.
          <br />
          Pick a sample or paste a ticket.
        </div>
      </div>
    );
  }

  const categoryLabel = formatCategory(result.category);
  const secondaryLabel = formatCategory(result.secondary_category);

  // Clean JSON response for display
  const jsonOutput = {
    ticket_id: result.ticket_id || 'demo-001',
    category: result.category,
    secondary_category: result.secondary_category,
    team: result.team,
    is_urgent: result.is_urgent,
    confidence: result.confidence,
    model_version: result.model_version,
  };

  const curlReqBody = {
    ticket_id: request?.ticket_id || 'demo-001',
    channel: request?.channel || 'chat',
    ...(request?.subject ? { subject: request.subject } : {}),
    text: request?.text ? (request.text.length > 50 ? `${request.text.slice(0, 48)}…` : request.text) : '',
  };

  const curlCommand = `curl -X POST $API/predict \\\n  -H 'Content-Type: application/json' \\\n  -H 'X-API-Key: $API_KEY' \\\n  -d '${JSON.stringify(curlReqBody)}'`;

  return (
    <div className="card" aria-live="polite">
      <div className="lbl">Stamped as</div>
      <div>
        <span className="stamp">{categoryLabel}</span>
        {result.is_urgent && <span className="stamp u">Urgent</span>}
      </div>

      <div className="route">
        <i>→</i>
        {result.team}
      </div>

      <div className="lbl">Secondary</div>
      <span className="tag">{secondaryLabel}</span>

      <div className="conf">
        <div className="lbl" style={{ margin: 0 }}>
          Confidence
        </div>
        <div className="track" role="progressbar" aria-valuenow={Math.round(result.confidence * 100)} aria-valuemin={0} aria-valuemax={100}>
          <div className="fill" style={{ width: barWidth }} />
        </div>
        <b>{formatConfidence(result.confidence)}</b>
      </div>

      <div className="meta">
        <span>
          {formatLatency(result.latency_ms)} · calibrated probability
        </span>
        <button
          className="meta-toggle"
          onClick={() => setShowJson(!showJson)}
          aria-expanded={showJson}
        >
          {showJson ? 'hide' : 'view'} JSON / cURL
        </button>
      </div>

      {showJson && (
        <pre className="code-panel">
          {JSON.stringify(jsonOutput, null, 2)}
          {'\n\n'}
          {curlCommand}
        </pre>
      )}
    </div>
  );
};
