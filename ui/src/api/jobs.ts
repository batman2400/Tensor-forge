import { apiClient } from './client';
import { BatchJobResults, BatchJobStatus, BatchRequest } from './types';

export async function createBatchJob(request: BatchRequest): Promise<BatchJobStatus> {
  if (request.tickets.length === 0) {
    throw new Error('Job must contain at least 1 ticket');
  }
  if (request.tickets.length > 5000) {
    throw new Error('Job exceeds maximum limit of 5,000 tickets');
  }

  return apiClient.request<BatchJobStatus>('/batch/jobs', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

export async function fetchBatchJobStatus(jobId: string): Promise<BatchJobStatus> {
  return apiClient.request<BatchJobStatus>(`/batch/jobs/${encodeURIComponent(jobId)}`, {
    method: 'GET',
  });
}

export async function fetchBatchJobResults(
  jobId: string,
  offset = 0,
  limit = 500
): Promise<BatchJobResults> {
  const query = new URLSearchParams({
    offset: String(offset),
    limit: String(limit),
  }).toString();

  return apiClient.request<BatchJobResults>(
    `/batch/jobs/${encodeURIComponent(jobId)}/results?${query}`,
    {
      method: 'GET',
    },
    60000
  );
}

export async function cancelBatchJob(jobId: string): Promise<BatchJobStatus> {
  return apiClient.request<BatchJobStatus>(`/batch/jobs/${encodeURIComponent(jobId)}`, {
    method: 'DELETE',
  });
}
