import { act, renderHook } from '@testing-library/react';
import { beforeEach, describe, expect, it } from 'vitest';
import { apiClient } from '../api/client';
import { useApiKey } from '../hooks/useApiKey';

describe('useApiKey', () => {
  beforeEach(() => {
    sessionStorage.clear();
    apiClient.setApiKey(null);
  });

  it('starts unlocked as false when no key in sessionStorage', () => {
    const { result } = renderHook(() => useApiKey());
    expect(result.current.isUnlocked).toBe(false);
    expect(result.current.apiKey).toBeNull();
  });

  it('sets API key, updates sessionStorage, and sets isUnlocked', () => {
    const { result } = renderHook(() => useApiKey());

    act(() => {
      result.current.setKey('valid-token-xyz');
    });

    expect(result.current.isUnlocked).toBe(true);
    expect(result.current.apiKey).toBe('valid-token-xyz');
    expect(sessionStorage.getItem('tf_api_key')).toBe('valid-token-xyz');
    expect(apiClient.getApiKey()).toBe('valid-token-xyz');
  });

  it('clears API key and locks on lock()', () => {
    const { result } = renderHook(() => useApiKey());

    act(() => {
      result.current.setKey('key-to-lock');
    });
    expect(result.current.isUnlocked).toBe(true);

    act(() => {
      result.current.lock();
    });

    expect(result.current.isUnlocked).toBe(false);
    expect(result.current.apiKey).toBeNull();
    expect(sessionStorage.getItem('tf_api_key')).toBeNull();
    expect(apiClient.getApiKey()).toBeNull();
  });
});
