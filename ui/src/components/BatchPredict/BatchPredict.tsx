import React, { useMemo, useState } from 'react';
import { predictBatch } from '../../api/batch';
import { BatchRequest, ErrorDetail } from '../../api/types';
import { downloadCSV } from '../../lib/download';
import { SAMPLE_TICKETS } from '../../lib/samples';
import { ErrorBanner } from '../ErrorBanner';
import { LoadingSpinner } from '../LoadingSpinner';
import './BatchPredict.css';
import { BatchTable, BatchTableRow } from './BatchTable';

const INITIAL_BATCH_TEXT = [
  ...SAMPLE_TICKETS.map((s) => s.text),
  'The app freezes every time I open the map screen.',
  'Wrong item delivered and the fries were cold.',
  'Please explain how your loyalty points work.',
  'Driver threatened me at pickup, I am stranded and scared.',
].join('\n');

export const BatchPredict: React.FC = () => {
  const [text, setText] = useState<string>(INITIAL_BATCH_TEXT);
  const [urgentOnly, setUrgentOnly] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [errorDetails, setErrorDetails] = useState<ErrorDetail[] | undefined>(undefined);
  const [rows, setRows] = useState<BatchTableRow[]>([]);

  const lineCount = useMemo(() => {
    return text
      .split('\n')
      .map((s) => s.trim())
      .filter(Boolean).length;
  }, [text]);

  const isOverLimit = lineCount > 100;

  const handleClassifyBatch = async () => {
    const lines = text
      .split('\n')
      .map((s) => s.trim())
      .filter(Boolean);

    if (lines.length === 0) {
      setError('Please provide at least 1 ticket.');
      return;
    }

    if (lines.length > 100) {
      setError(`Batch size is ${lines.length} tickets, but maximum allowed is 100.`);
      return;
    }

    setIsLoading(true);
    setError(null);
    setErrorDetails(undefined);

    const batchReq: BatchRequest = {
      tickets: lines.map((line, idx) => ({
        ticket_id: `TF-UI-${String(idx + 1).padStart(3, '0')}`,
        channel: 'chat',
        text: line,
      })),
    };

    try {
      const res = await predictBatch(batchReq);
      const tableRows: BatchTableRow[] = res.predictions.map((p, idx) => ({
        ...p,
        i: idx + 1,
        text: lines[idx] || '',
      }));
      setRows(tableRows);
    } catch (err: unknown) {
      if (err && typeof err === 'object' && 'details' in err) {
        const apiErr = err as { message: string; details?: ErrorDetail[] };
        setError(apiErr.message);
        setErrorDetails(apiErr.details);
      } else {
        setError(err instanceof Error ? err.message : 'Batch prediction failed');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleExportCSV = () => {
    if (rows.length === 0) return;
    const exportData = rows.map((r) => ({
      index: r.i,
      ticket_id: r.ticket_id,
      text: r.text,
      category: r.category,
      team: r.team,
      secondary_category: r.secondary_category || '',
      is_urgent: r.is_urgent ? 'YES' : 'NO',
      confidence: (r.confidence * 100).toFixed(0) + '%',
      model_version: r.model_version,
    }));
    downloadCSV(exportData, `batch_predictions_${Date.now()}.csv`);
  };

  return (
    <section>
      <h2>
        <span>02</span>Batch
      </h2>
      <p className="ep">POST /predict/batch · 1–100 tickets, one per line</p>

      {error && (
        <ErrorBanner
          message={error}
          details={errorDetails}
          onDismiss={() => {
            setError(null);
            setErrorDetails(undefined);
          }}
        />
      )}

      <textarea
        className="batch-textarea"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Enter 1–100 tickets, one per line…"
        aria-label="Batch tickets input"
      />

      <div className="bar">
        <div className="batch-actions-left">
          <button
            className="primary"
            onClick={handleClassifyBatch}
            disabled={isLoading || lineCount === 0 || isOverLimit}
          >
            {isLoading ? <LoadingSpinner label="Classifying…" /> : 'Classify batch'}
          </button>

          <span className={`ticket-count ${isOverLimit ? 'over-limit' : ''}`}>
            {isOverLimit
              ? `⚠ ${lineCount} tickets (max 100)`
              : `${lineCount} ticket${lineCount === 1 ? '' : 's'}`}
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          {rows.length > 0 && (
            <button type="button" onClick={handleExportCSV} className="ghost">
              Export CSV
            </button>
          )}

          <label>
            <input
              type="checkbox"
              checked={urgentOnly}
              onChange={(e) => setUrgentOnly(e.target.checked)}
            />
            Urgent only
          </label>
        </div>
      </div>

      <BatchTable rows={rows} urgentOnly={urgentOnly} />
    </section>
  );
};
