import { ErrorResponse } from './types';

export class ApiError extends Error {
  public status: number;
  public code: string;
  public details?: Array<{ index?: number; field: string; issue: string }>;

  constructor(status: number, code: string, message: string, details?: Array<{ index?: number; field: string; issue: string }>) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

type UnauthorizedHandler = () => void;

class ApiClient {
  private apiKey: string | null = null;
  private onUnauthorizedCallback: UnauthorizedHandler | null = null;

  public setApiKey(key: string | null): void {
    this.apiKey = key;
  }

  public getApiKey(): string | null {
    return this.apiKey;
  }

  public onUnauthorized(cb: UnauthorizedHandler): void {
    this.onUnauthorizedCallback = cb;
  }

  public async request<T>(
    endpoint: string,
    options: RequestInit = {},
    timeoutMs = 30000
  ): Promise<T> {
    const headers = new Headers(options.headers || {});
    headers.set('Content-Type', 'application/json');

    if (this.apiKey) {
      headers.set('X-API-Key', this.apiKey);
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const response = await fetch(endpoint, {
        ...options,
        headers,
        signal: controller.signal,
      });

      if (response.status === 401) {
        if (this.onUnauthorizedCallback) {
          this.onUnauthorizedCallback();
        }
        let msg = 'Unauthorized: invalid or missing API key';
        try {
          const body = (await response.json()) as ErrorResponse;
          if (body?.error?.message) msg = body.error.message;
        } catch {
          // ignore json parse error on 401
        }
        throw new ApiError(401, 'UNAUTHORIZED', msg);
      }

      if (!response.ok) {
        let code = `HTTP_${response.status}`;
        let message = `Request failed with status ${response.status}`;
        let details: Array<{ index?: number; field: string; issue: string }> | undefined;

        try {
          const errBody = (await response.json()) as ErrorResponse;
          if (errBody?.error) {
            code = errBody.error.code || code;
            message = errBody.error.message || message;
            details = errBody.error.details;
          }
        } catch {
          // non-JSON error body fallback
        }

        throw new ApiError(response.status, code, message, details);
      }

      // Handle 204 or empty responses
      if (response.status === 204) {
        return {} as T;
      }

      return (await response.json()) as T;
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        throw err;
      }
      if (err instanceof DOMException && err.name === 'AbortError') {
        throw new ApiError(408, 'TIMEOUT', `Request timed out after ${timeoutMs / 1000}s`);
      }
      const message = err instanceof Error ? err.message : 'Network error or server unreachable';
      throw new ApiError(0, 'NETWORK_ERROR', message);
    } finally {
      clearTimeout(timeoutId);
    }
  }
}

export const apiClient = new ApiClient();
