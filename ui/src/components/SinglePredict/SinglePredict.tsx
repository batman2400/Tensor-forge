import React, { useState } from 'react';
import { predictSingle } from '../../api/predict';
import { Channel, ErrorDetail, PredictRequest, PredictResponse } from '../../api/types';
import { SAMPLE_TICKETS, SampleTicket } from '../../lib/samples';
import { ErrorBanner } from '../ErrorBanner';
import { LoadingSpinner } from '../LoadingSpinner';
import { ResultCard } from './ResultCard';
import './SinglePredict.css';

export const SinglePredict: React.FC = () => {
  const [channel, setChannel] = useState<Channel>('chat');
  const [subject, setSubject] = useState<string>('');
  const [text, setText] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [errorDetails, setErrorDetails] = useState<ErrorDetail[] | undefined>(undefined);

  const [lastRequest, setLastRequest] = useState<PredictRequest | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);

  const handleSelectSample = (sample: SampleTicket) => {
    setChannel(sample.channel);
    setSubject(sample.subject || '');
    setText(sample.text);
    setError(null);
    setErrorDetails(undefined);
  };

  const handleClassify = async () => {
    const trimmed = text.trim();
    if (!trimmed) {
      setError('Please enter ticket text before classifying.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setErrorDetails(undefined);

    const req: PredictRequest = {
      ticket_id: `TF-SINGLE-${Date.now().toString(36).slice(-4).toUpperCase()}`,
      channel,
      ...(channel === 'email' && subject.trim() ? { subject: subject.trim() } : {}),
      text: trimmed,
    };

    setLastRequest(req);

    try {
      const res = await predictSingle(req);
      setResult(res);
    } catch (err: unknown) {
      if (err && typeof err === 'object' && 'details' in err) {
        const apiErr = err as { message: string; details?: ErrorDetail[] };
        setError(apiErr.message);
        setErrorDetails(apiErr.details);
      } else {
        setError(err instanceof Error ? err.message : 'Failed to classify ticket');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      handleClassify();
    }
  };

  return (
    <section>
      <h2>
        <span>01</span>Single ticket
      </h2>
      <p className="ep">POST /predict</p>

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

      <div className="grid">
        <div>
          <div className="lbl">Channel</div>
          <div className="channel-selector" role="radiogroup" aria-label="Ticket channel">
            {(['chat', 'email', 'call_transcript'] as Channel[]).map((ch) => (
              <button
                key={ch}
                type="button"
                role="radio"
                aria-checked={channel === ch}
                className={`channel-btn ${channel === ch ? 'active' : ''}`}
                onClick={() => setChannel(ch)}
              >
                {ch === 'call_transcript' ? 'Call' : ch}
              </button>
            ))}
          </div>

          {channel === 'email' && (
            <div className="subject-input-wrapper">
              <label htmlFor="ticket-subject" className="lbl">
                Subject
              </label>
              <input
                id="ticket-subject"
                type="text"
                value={subject}
                onChange={(e) => setSubject(e.target.value)}
                placeholder="e.g. Question regarding my charge"
                className="subject-input"
              />
            </div>
          )}

          <label htmlFor="ticket-text" className="lbl">
            Incoming ticket
          </label>
          <textarea
            id="ticket-text"
            className="ticket-input"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Paste a customer support ticket… (Ctrl+Enter to classify)"
          />

          <div className="chips">
            {SAMPLE_TICKETS.map((s, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => handleSelectSample(s)}
              >
                {s.label}
              </button>
            ))}
          </div>

          <button
            className="primary"
            onClick={handleClassify}
            disabled={isLoading || !text.trim()}
          >
            {isLoading ? <LoadingSpinner label="Classifying…" /> : 'Classify ticket'}
          </button>
        </div>

        <ResultCard result={result} request={lastRequest} />
      </div>
    </section>
  );
};
