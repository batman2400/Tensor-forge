import { act, renderHook } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import * as jobsApi from '../api/jobs';
import { BatchJobStatus } from '../api/types';
import { useJobPoller } from '../hooks/useJobPoller';

describe('useJobPoller', () => {
  beforeEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
  });

  it('starts a job and updates status to queued', async () => {
    const mockCreatedStatus: BatchJobStatus = {
      job_id: 'job_test_123',
      status: 'queued',
      total: 10,
      processed: 0,
      created_at: new Date().toISOString(),
    };

    vi.spyOn(jobsApi, 'createBatchJob').mockResolvedValue(mockCreatedStatus);
    vi.spyOn(jobsApi, 'fetchBatchJobStatus').mockResolvedValue(mockCreatedStatus);

    const { result } = renderHook(() => useJobPoller());

    await act(async () => {
      await result.current.startJob([
        { ticket_id: 'T1', channel: 'chat', text: 'help with order' },
      ]);
    });

    expect(result.current.jobStatus?.job_id).toBe('job_test_123');
    expect(result.current.jobStatus?.status).toBe('queued');
    expect(localStorage.getItem('tf_active_job_id')).toBe('job_test_123');
  });

  it('cancels active job and updates status', async () => {
    const mockCreatedStatus: BatchJobStatus = {
      job_id: 'job_cancel_me',
      status: 'running',
      total: 10,
      processed: 3,
      created_at: new Date().toISOString(),
    };

    const mockCancelledStatus: BatchJobStatus = {
      ...mockCreatedStatus,
      status: 'cancelled',
    };

    vi.spyOn(jobsApi, 'createBatchJob').mockResolvedValue(mockCreatedStatus);
    vi.spyOn(jobsApi, 'fetchBatchJobStatus').mockResolvedValue(mockCreatedStatus);
    vi.spyOn(jobsApi, 'cancelBatchJob').mockResolvedValue(mockCancelledStatus);

    const { result } = renderHook(() => useJobPoller());

    await act(async () => {
      await result.current.startJob([
        { ticket_id: 'T1', channel: 'chat', text: 'test cancel' },
      ]);
    });

    await act(async () => {
      await result.current.cancelJob();
    });

    expect(result.current.jobStatus?.status).toBe('cancelled');
    expect(result.current.isPolling).toBe(false);
  });
});
