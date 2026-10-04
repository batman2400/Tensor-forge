import { apiClient } from './client';
import { PredictRequest, PredictResponse } from './types';

export async function predictSingle(request: PredictRequest): Promise<PredictResponse> {
  const start = performance.now();
  const response = await apiClient.request<PredictResponse>('/predict', {
    method: 'POST',
    body: JSON.stringify(request),
  });
  const latency = Math.round(performance.now() - start);

  return {
    ...response,
    latency_ms: response.latency_ms ?? latency,
  };
}
