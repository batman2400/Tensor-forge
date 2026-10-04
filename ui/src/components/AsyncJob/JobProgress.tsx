import React from 'react';
import { BatchJobStatus } from '../../api/types';
import { formatNumber } from '../../lib/format';
import './JobProgress.css';

interface JobProgressProps {
  jobStatus: BatchJobStatus | null;
  isPolling: boolean;
}

export const JobProgress: React.FC<JobProgressProps> = ({ jobStatus, isPolling }) => {
  const jobId = jobStatus?.job_id ? `job: ${jobStatus.job_id}` : 'job: —';
  const status = jobStatus?.status || 'idle';
  const total = jobStatus?.total || 1200;
  const processed = jobStatus?.processed || 0;
  const pct = total > 0 ? Math.min(100, (processed / total) * 100) : 0;

  let note = 'Submit a job to begin.';
  if (status === 'queued') note = 'Job queued, waiting for worker…';
  else if (status === 'running') note = isPolling ? 'Processing… polled until complete' : 'Processing…';
  else if (status === 'succeeded') note = 'Stamped and filed. Results are ready.';
  else if (status === 'failed') note = 'Job failed on server.';
  else if (status === 'cancelled') note = 'Job was cancelled.';

  return (
    <>
      <div className="jrow">
        <span>{jobId}</span>
        <span className={`jrow-status ${status}`}>{status}</span>
        <span>
          {formatNumber(processed)} / {formatNumber(total)}
        </span>
      </div>

      <div
        className="jt"
        role="progressbar"
        aria-valuenow={Math.round(pct)}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <div className="jf" style={{ width: `${pct}%` }} />
      </div>

      <div className="note" style={{ marginTop: '8px' }}>
        {note}
      </div>
    </>
  );
};
