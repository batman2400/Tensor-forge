import { useCallback, useEffect, useRef, useState } from 'react';
import { cancelBatchJob, createBatchJob, fetchBatchJobResults, fetchBatchJobStatus } from '../api/jobs';
import { BatchJobResults, BatchJobStatus, BatchTicketItem } from '../api/types';

const STORAGE_ACTIVE_JOB = 'tf_active_job_id';

export function useJobPoller() {
  const [jobStatus, setJobStatus] = useState<BatchJobStatus | null>(null);
  const [results, setResults] = useState<BatchJobResults | null>(null);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [isPolling, setIsPolling] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const pollTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const isMountedRef = useRef<boolean>(true);

  const clearTimer = () => {
    if (pollTimeoutRef.current) {
      clearTimeout(pollTimeoutRef.current);
      pollTimeoutRef.current = null;
    }
  };

  const pollJob = useCallback(async (jobId: string) => {
    if (!isMountedRef.current) return;
    setIsPolling(true);

    try {
      const status = await fetchBatchJobStatus(jobId);
      if (!isMountedRef.current) return;
      setJobStatus(status);

      if (status.status === 'succeeded') {
        setIsPolling(false);
        try {
          const res = await fetchBatchJobResults(jobId, 0, 5000);
          if (isMountedRef.current) setResults(res);
        } catch (fetchErr) {
          if (isMountedRef.current) {
            setError(fetchErr instanceof Error ? fetchErr.message : 'Failed to fetch job results');
          }
        }
        return;
      }

      if (status.status === 'failed') {
        setIsPolling(false);
        setError(status.error?.message || 'Job execution failed on server');
        return;
      }

      if (status.status === 'cancelled') {
        setIsPolling(false);
        return;
      }

      // If still queued or running, schedule next poll
      pollTimeoutRef.current = setTimeout(() => {
        pollJob(jobId);
      }, 1500);
    } catch (err) {
      if (!isMountedRef.current) return;
      setIsPolling(false);
      setError(err instanceof Error ? err.message : 'Error polling job status');
    }
  }, []);

  // Resume on mount if there's an active job in localStorage
  useEffect(() => {
    isMountedRef.current = true;
    try {
      const savedJobId = localStorage.getItem(STORAGE_ACTIVE_JOB);
      if (savedJobId) {
        pollJob(savedJobId);
      }
    } catch {
      // ignore
    }

    return () => {
      isMountedRef.current = false;
      clearTimer();
    };
  }, [pollJob]);

  const startJob = async (tickets: BatchTicketItem[]) => {
    clearTimer();
    setError(null);
    setResults(null);
    setIsSubmitting(true);

    try {
      const initialStatus = await createBatchJob({ tickets });
      setJobStatus(initialStatus);
      try {
        localStorage.setItem(STORAGE_ACTIVE_JOB, initialStatus.job_id);
      } catch {
        // ignore
      }
      pollJob(initialStatus.job_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create batch job');
    } finally {
      setIsSubmitting(false);
    }
  };

  const cancelJob = async () => {
    if (!jobStatus?.job_id) return;
    clearTimer();
    setIsPolling(false);

    try {
      const res = await cancelBatchJob(jobStatus.job_id);
      setJobStatus(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to cancel job');
    }
  };

  const resetJob = () => {
    clearTimer();
    setIsPolling(false);
    setJobStatus(null);
    setResults(null);
    setError(null);
    try {
      localStorage.removeItem(STORAGE_ACTIVE_JOB);
    } catch {
      // ignore
    }
  };

  return {
    jobStatus,
    results,
    isSubmitting,
    isPolling,
    error,
    startJob,
    cancelJob,
    resetJob,
  };
}
