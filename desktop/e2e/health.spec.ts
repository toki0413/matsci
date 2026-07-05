import { test, expect, backendReachable, BACKEND_URL, authHeaders } from './fixtures';

// These hit the agent backend directly, so skip the whole file when the
// process isn't up (e.g. running just the UI without the python env).
let backendUp = false;

test.beforeAll(async () => {
  backendUp = await backendReachable();
});

test.describe('backend health endpoints', () => {
  test('GET /health/live returns 200', async ({ request }) => {
    test.skip(!backendUp, 'backend not running');
    const res = await request.get(`${BACKEND_URL}/health/live`, { headers: authHeaders });
    expect(res.status()).toBe(200);
    const body = await res.json();
    expect(body.status).toBe('alive');
    expect(body.version).toBeTruthy();
  });

  test('GET /health/ready returns 200 or 503', async ({ request }) => {
    test.skip(!backendUp, 'backend not running');
    const res = await request.get(`${BACKEND_URL}/health/ready`, { headers: authHeaders });
    // 200 when every dependency is healthy, 503 when any check fails -- both
    // are valid "the endpoint works" outcomes.
    expect([200, 503]).toContain(res.status());
    const body = await res.json();
    expect(body).toHaveProperty('ready');
    expect(body).toHaveProperty('checks');
  });

  test('GET /tools returns 200', async ({ request }) => {
    test.skip(!backendUp, 'backend not running');
    const res = await request.get(`${BACKEND_URL}/tools`, { headers: authHeaders });
    expect(res.status()).toBe(200);
    const body = await res.json();
    expect(Array.isArray(body)).toBe(true);
  });

  test('GET /skills returns 200', async ({ request }) => {
    test.skip(!backendUp, 'backend not running');
    const res = await request.get(`${BACKEND_URL}/skills`, { headers: authHeaders });
    expect(res.status()).toBe(200);
    const body = await res.json();
    expect(Array.isArray(body)).toBe(true);
  });
});
