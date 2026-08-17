import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { api, authHeaders } from './api';
import { setApiBase } from './api-client';

type FetchCall = { url: string; init: RequestInit };
const calls: FetchCall[] = [];

/** Stub fetch deterministically. */
function mockFetch(responseProducer: () => Response | Promise<Response>) {
  (globalThis as any).fetch = vi.fn(async (url: string, init?: RequestInit) => {
    calls.push({ url, init: init ?? {} });
    return responseProducer();
  });
}

/** Minimal Response-lookalike for request(). */
function makeResponse(o: { status?: number; json?: unknown; text?: string; contentType?: string } = {}): Response {
  const status = o.status ?? 200;
  const contentType = o.contentType ?? 'application/json';
  return {
    ok: status < 300,
    status,
    headers: { get: (name: string) => (name === 'content-type' ? contentType : null), has: () => false },
    json: async () => o.json,
    text: async () => o.text ?? '',
    blob: async () => new Blob([o.text ?? '']),
  } as unknown as Response;
}

beforeEach(() => {
  calls.length = 0;
  localStorage.clear();
  setApiBase('http://127.0.0.1:8000');
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe('api HTTP methods', () => {
  it('GET sends nothing but the auth header + content-type', async () => {
    mockFetch(() => makeResponse({ json: { ok: 1 } }));
    const res = await api.get<{ ok: number }>('/health');
    expect(calls[0].init.method).toBe('GET');
    expect(calls[0].url).toBe('http://127.0.0.1:8000/health');
    expect(res.ok).toBe(1);
    expect((calls[0].init.headers as any)['Content-Type']).toBe('application/json');
  });

  it('POST stringifies the body', async () => {
    mockFetch(() => makeResponse({ json: { id: 7 } }));
    await api.post('/tools/run', { name: 'bash' });
    expect(calls[0].init.method).toBe('POST');
    expect(calls[0].init.body).toBe(JSON.stringify({ name: 'bash' }));
  });

  it('does not set a body when POST payload is undefined', async () => {
    mockFetch(() => makeResponse({ json: {} }));
    await api.post('/run/once');
    expect(calls[0].init.body).toBeUndefined();
  });

  it('appends params to the URL as a query string', async () => {
    mockFetch(() => makeResponse({ json: [] }));
    const params = new URLSearchParams({ query: 'carbon', limit: '5' });
    await api.get('/search/global', { params });
    expect(calls[0].url).toBe('http://127.0.0.1:8000/search/global?query=carbon&limit=5');
  });

  it('merges caller-provided headers on top of auth headers', async () => {
    localStorage.setItem('huginn:api_key', 'RAWKEY');
    mockFetch(() => makeResponse({ json: {} }));
    await api.get('/x', { headers: { 'X-Custom': 'v' } as any });
    const h = calls[0].init.headers as Record<string, string>;
    expect(h['X-HUGINN-API-KEY']).toBe('RAWKEY'); // auth fallback
    expect(h['X-Custom']).toBe('v');
  });
});

// authHeaders reads the module-cached getAuthToken(); isolate each case with a
// fresh module so the cached token doesn't leak across tests.
describe('authHeaders', () => {
  async function withFreshApi(cb: (m: typeof import('./api')) => void) {
    vi.resetModules();
    const mod = (await import('./api')) as typeof import('./api');
    (await import('./api-client')).setApiBase('http://127.0.0.1:8000');
    cb(mod);
  }

  it('prefers the JWT token as Bearer', async () => {
    await withFreshApi(async (m) => {
      localStorage.setItem('huginn:auth_token', 'jwt-x');
      expect(m.authHeaders()).toEqual({ Authorization: 'Bearer jwt-x' });
    });
  });

  it('falls back to the raw api key header when no token', async () => {
    await withFreshApi((m) => {
      localStorage.setItem('huginn:api_key', 'key-y');
      expect(m.authHeaders()).toEqual({ 'X-HUGINN-API-KEY': 'key-y' });
    });
  });

  it('returns no auth headers when neither is available', async () => {
    await withFreshApi((m) => {
      expect(m.authHeaders()).toEqual({});
    });
  });
});

describe('api error handling', () => {
  it('throws a parsed error message on a non-2xx JSON response', async () => {
    mockFetch(() => makeResponse({ status: 500, json: { detail: 'boom' } }));
    await expect(api.get('/x')).rejects.toThrow('boom');
  });

  it('falls back to the status code when the body has no message', async () => {
    mockFetch(() => makeResponse({ status: 401, json: { foo: 1 } }));
    await expect(api.get('/x')).rejects.toThrow('API error: 401');
  });

  it('handles a text error body without throwing on .json()', async () => {
    mockFetch(() => makeResponse({ status: 400, text: 'bad request', contentType: 'text/plain' }));
    await expect(api.get('/x')).rejects.toThrow('API error: 400');
  });

  it('returns text content when content-type is text/*', async () => {
    mockFetch(() => makeResponse({ text: 'hello-raw', contentType: 'text/plain' }));
    const res = await api.get<string>('/raw');
    expect(res).toBe('hello-raw');
  });

  it('retries once on a 5xx with a 1s backoff', async () => {
    vi.useFakeTimers();
    let n = 0;
    mockFetch(() => {
      n++;
      if (n === 1) return makeResponse({ status: 503, json: {} });
      return makeResponse({ json: { recovered: true } });
    });
    const p = api.get('/fragile');
    await vi.advanceTimersByTimeAsync(2000); // jitter up to 1.5s
    await expect(p).resolves.toEqual({ recovered: true });
    expect(n).toBe(2);
  });

  it('resolves to empty string for an empty/204 body', async () => {
    // Contract: request() returns the raw text() result when there's no JSON
    // or text content-type (204 typically yields ""). Assert the real behavior,
    // not a guessed undefined.
    mockFetch(() => makeResponse({ status: 204, contentType: '' }));
    const res = await api.post('/no-content');
    expect(res).toBe('');
  });
});