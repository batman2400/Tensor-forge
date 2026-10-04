import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';
import { ApiError, apiClient } from '../api/client';

describe('ApiClient', () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    apiClient.setApiKey(null);
  });

  afterEach(() => {
    global.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it('injects X-API-Key header when key is set', async () => {
    let capturedHeaders: Headers | undefined;
    global.fetch = vi.fn().mockImplementation(async (_url: string, init?: RequestInit) => {
      capturedHeaders = new Headers(init?.headers);
      return new Response(JSON.stringify({ status: 'ok' }), { status: 200 });
    });

    apiClient.setApiKey('test-secret-key-123');
    await apiClient.request('/health');

    expect(capturedHeaders?.get('X-API-Key')).toBe('test-secret-key-123');
    expect(capturedHeaders?.get('Content-Type')).toBe('application/json');
  });

  it('triggers onUnauthorized callback on 401 response', async () => {
    const onUnauth = vi.fn();
    apiClient.onUnauthorized(onUnauth);

    global.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ error: { code: 'UNAUTHORIZED', message: 'Invalid key' } }), {
        status: 401,
      })
    );

    apiClient.setApiKey('wrong-key');
    await expect(apiClient.request('/predict')).rejects.toThrow(ApiError);
    expect(onUnauth).toHaveBeenCalledTimes(1);
  });

  it('parses structured ErrorResponse details on 422', async () => {
    global.fetch = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          error: {
            code: 'VALIDATION_ERROR',
            message: 'Channel invalid',
            details: [{ field: 'channel', issue: 'Invalid value' }],
          },
        }),
        { status: 422 }
      )
    );

    try {
      await apiClient.request('/predict');
      expect.unreachable();
    } catch (err: unknown) {
      expect(err).toBeInstanceOf(ApiError);
      const apiErr = err as ApiError;
      expect(apiErr.status).toBe(422);
      expect(apiErr.code).toBe('VALIDATION_ERROR');
      expect(apiErr.details?.[0].field).toBe('channel');
    }
  });

  it('handles network error cleanly', async () => {
    global.fetch = vi.fn().mockRejectedValue(new TypeError('Failed to fetch'));

    try {
      await apiClient.request('/predict');
      expect.unreachable();
    } catch (err: unknown) {
      expect(err).toBeInstanceOf(ApiError);
      const apiErr = err as ApiError;
      expect(apiErr.code).toBe('NETWORK_ERROR');
    }
  });
});
