import { describe, it, expect, vi, beforeEach } from 'vitest';

describe('api-client', () => {
  beforeEach(() => {
    localStorage.clear();
    // authToken is cached in module state on first read; reset the module so
    // state from a prior test never leaks into this one.
    vi.resetModules();
  });

  it('defaults to the localhost base', async () => {
    const { getApiBase } = await import('./api-client');
    expect(getApiBase()).toBe('http://127.0.0.1:8000');
  });

  it('setApiBase strips a trailing slash', async () => {
    const { getApiBase, setApiBase } = await import('./api-client');
    setApiBase('http://127.0.0.1:9000/');
    expect(getApiBase()).toBe('http://127.0.0.1:9000');
  });

  it('setApiBase keeps a base without a trailing slash', async () => {
    const { getApiBase, setApiBase } = await import('./api-client');
    setApiBase('http://127.0.0.1:9000');
    expect(getApiBase()).toBe('http://127.0.0.1:9000');
  });

  it('reads the auth token from localStorage on first access', async () => {
    const { getAuthToken } = await import('./api-client');
    localStorage.setItem('huginn:auth_token', 'jwt-abc');
    expect(getAuthToken()).toBe('jwt-abc');
  });

  it('returns null when no token is stored', async () => {
    // fresh module so any cached token from a prior test is cleared
    const { getAuthToken } = await import('./api-client');
    expect(getAuthToken()).toBeNull();
  });
});