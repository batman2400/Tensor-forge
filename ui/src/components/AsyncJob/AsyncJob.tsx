import React, { useRef, useState } from 'react';
import { BatchTicketItem } from '../../api/types';
import { useJobPoller } from '../../hooks/useJobPoller';
import { downloadCSV, downloadJSON } from '../../lib/download';
import { SAMPLE_TICKETS } from '../../lib/samples';
import { ErrorBanner } from '../ErrorBanner';
import { LoadingSpinner } from '../LoadingSpinner';
import './AsyncJob.css';
import { JobProgress } from './JobProgress';

export const AsyncJob: React.FC = () => {
  const {
    jobStatus,
    results,
    isSubmitting,
    isPolling,
    error,
    startJob,
    cancelJob,
    resetJob,
  } = useJobPoller();

  const [dragOver, setDragOver] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<{ name: string; tickets: BatchTicketItem[] } | null>(null);
  const [parseError, setParseError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Generate 1,200 sample tickets for the default demo
  const generateDemoTickets = (): BatchTicketItem[] => {
    const list: BatchTicketItem[] = [];
    const baseCount = SAMPLE_TICKETS.length;
    for (let i = 0; i < 1200; i++) {
      const sample = SAMPLE_TICKETS[i % baseCount];
      list.push({
        ticket_id: `TF-JOB-${String(i + 1).padStart(4, '0')}`,
        channel: sample.channel,
        subject: sample.subject,
        text: `${sample.text} (ref #${i + 100})`,
      });
    }
    return list;
  };

  const handleProcessFile = async (file: File) => {
    setParseError(null);
    try {
      const text = await file.text();
      let tickets: BatchTicketItem[] = [];

      if (file.name.endsWith('.jsonl')) {
        const lines = text.split('\n').filter((l) => l.trim().length > 0);
        tickets = lines.map((line, idx) => {
          try {
            const parsed = JSON.parse(line);
            return {
              ticket_id: parsed.ticket_id || `TF-FILE-${String(idx + 1).padStart(4, '0')}`,
              channel: parsed.channel || 'chat',
              subject: parsed.subject,
              text: parsed.text || '',
            };
          } catch {
            return {
              ticket_id: `TF-FILE-${String(idx + 1).padStart(4, '0')}`,
              channel: 'chat',
              text: line,
            };
          }
        });
      } else if (file.name.endsWith('.json')) {
        const parsed = JSON.parse(text);
        const arr = Array.isArray(parsed) ? parsed : parsed.tickets || [];
        tickets = arr.map((item: Record<string, unknown>, idx: number) => ({
          ticket_id: String(item.ticket_id || `TF-FILE-${String(idx + 1).padStart(4, '0')}`),
          channel: (item.channel as 'email' | 'chat' | 'call_transcript') || 'chat',
          subject: item.subject ? String(item.subject) : undefined,
          text: String(item.text || ''),
        }));
      } else {
        // Plain CSV or text lines
        const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
        tickets = lines.slice(0, 5000).map((line, idx) => ({
          ticket_id: `TF-FILE-${String(idx + 1).padStart(4, '0')}`,
          channel: 'chat',
          text: line.replace(/^"|"$/g, '').trim(),
        }));
      }

      if (tickets.length === 0) {
        setParseError('The uploaded file did not contain any valid tickets.');
        return;
      }

      if (tickets.length > 5000) {
        tickets = tickets.slice(0, 5000);
      }

      setUploadedFile({
        name: `${file.name} (${tickets.length} tickets)`,
        tickets,
      });
    } catch {
      setParseError('Failed to parse file. Please upload a valid .csv or .jsonl file.');
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleProcessFile(file);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      handleProcessFile(file);
    }
  };

  const handleSubmitJob = () => {
    const tickets = uploadedFile ? uploadedFile.tickets : generateDemoTickets();
    startJob(tickets);
  };

  const handleDownloadCSV = () => {
    if (!results || results.predictions.length === 0) return;
    const rows = results.predictions.map((p, idx) => ({
      index: idx + 1,
      ticket_id: p.ticket_id,
      category: p.category,
      team: p.team,
      secondary_category: p.secondary_category || '',
      is_urgent: p.is_urgent ? 'YES' : 'NO',
      confidence: (p.confidence * 100).toFixed(0) + '%',
      model_version: p.model_version,
    }));
    downloadCSV(rows, `job_${jobStatus?.job_id || 'results'}.csv`);
  };

  const handleDownloadJSON = () => {
    if (!results) return;
    downloadJSON(results, `job_${jobStatus?.job_id || 'results'}.json`);
  };

  const status = jobStatus?.status || 'idle';
  const isBusy = status === 'queued' || status === 'running';
  const isComplete = status === 'succeeded';

  return (
    <section>
      <h2>
        <span>03</span>Async job
      </h2>
      <p className="ep">POST /batch/jobs · up to 5,000 tickets · polled until complete</p>

      {(error || parseError) && (
        <ErrorBanner
          message={error || parseError || ''}
          onDismiss={() => setParseError(null)}
        />
      )}

      <div className="job">
        <div className="lbl">Upload tickets (.csv / .jsonl)</div>

        <div
          className={`dropzone ${dragOver ? 'active' : ''}`}
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          role="button"
          tabIndex={0}
          aria-label="Upload tickets file"
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,.jsonl,.json,.txt"
            onChange={handleFileChange}
          />
          {uploadedFile
            ? `Ready: ${uploadedFile.name}`
            : 'Drop a file here — demo uses 1,200 sample tickets'}
        </div>

        <JobProgress jobStatus={jobStatus} isPolling={isPolling} />

        <div className="jfoot">
          <div className="job-buttons-group">
            {isBusy ? (
              <button onClick={cancelJob}>Cancel job</button>
            ) : isComplete ? (
              <button onClick={resetJob}>New job</button>
            ) : null}
          </div>

          <div className="job-buttons-group">
            <button
              className="primary"
              onClick={handleSubmitJob}
              disabled={isSubmitting || isBusy}
            >
              {isSubmitting ? (
                <LoadingSpinner label="Submitting…" />
              ) : isBusy ? (
                'Running…'
              ) : (
                'Submit job'
              )}
            </button>

            <button
              onClick={handleDownloadCSV}
              disabled={!isComplete}
              title={isComplete ? 'Download results as CSV' : 'Complete job to download'}
            >
              Download CSV
            </button>

            <button
              onClick={handleDownloadJSON}
              disabled={!isComplete}
              title={isComplete ? 'Download results as JSON' : 'Complete job to download'}
            >
              Download JSON
            </button>
          </div>
        </div>
      </div>
    </section>
  );
};
