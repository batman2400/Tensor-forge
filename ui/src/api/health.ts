import { apiClient } from './client';
import { HealthResponse } from './types';

export async function fetchHealth(): Promise<HealthResponse> {
  return apiClient.request<HealthResponse>('/health', { method: 'GET' }, 5000);
}
