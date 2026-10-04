import { apiClient } from './client';
import { BatchRequest, BatchResponse } from './types';

export async function predictBatch(request: BatchRequest): Promise<BatchResponse> {
  if (request.tickets.length === 0) {
    throw new Error('Batch must contain at least 1 ticket');
  }
  if (request.tickets.length > 100) {
    throw new Error('Batch exceeds maximum limit of 100 tickets');
  }

  return apiClient.request<BatchResponse>(
    '/predict/batch',
    {
      method: 'POST',
      body: JSON.stringify(request),
    },
    60000
  );
}
