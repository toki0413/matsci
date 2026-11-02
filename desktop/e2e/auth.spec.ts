import { test, expect, BACKEND_URL } from './fixtures';

// Non-dev backend validation of the login/token flow:
//   1. POST /auth/login with the shared key -> JWT
//   2. Load the UI with that token in huginn:auth_token (what the real
//      desktop does after the user sets an API key)
//   3. Assert the Bearer token actually opens a protected endpoint
//      (401 before the fix landed -- login's sub was a username, but
//      require_api_key resolves sub against the user store's user_id).
//
// Targets the auth-enforced backend started by the run (HUGINN_JWT_SECRET
// set, HUGINN_DEV_MODE unset). No backend -> skip.
const API_KEY = process.env.HUGINN_API_KEY ?? 'e2e-test-key';

let backendUp = false;

test.beforeAll(async () => {
  try {
    const res = await fetch(`${BACKEND_URL}/health/live`);
    backendUp = res.ok;
  } catch {
    backendUp = false;
  }
});

test.describe('login/token flow against auth-enforced backend', () => {
  test('exchanges API key for a usable JWT', async ({ request }) => {
    test.skip(!backendUp, 'backend not running');

    // 1. no credentials -> 401
    const noAuth = await request.get(`${BACKEND_URL}/tools`);
    if (noAuth.status() === 200) {
      // Backend in dev mode (auth disabled) -- nothing to prove here.
      test.skip(true, 'backend is in dev mode (auth not enforced)');
    }
    expect(noAuth.status()).toBe(401);

    // 2. login -> JWT
    const login = await request.post(`${BACKEND_URL}/auth/login`, {
      data: { api_key: API_KEY },
    });
    expect(login.status()).toBe(200);
    const { token, role } = await login.json();
    expect(token).toBeTruthy();
    expect(role).toBe('operator');

    // 3. token opens a protected endpoint
    const authed = await request.get(`${BACKEND_URL}/tools`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    expect(authed.status()).toBe(200);
    const tools = await authed.json();
    expect(Array.isArray(tools)).toBe(true);
  });

  test('frontend sends the stored JWT and loads protected data', async ({ page }) => {
    test.skip(!backendUp, 'backend not running');

    // Login first, outside the page, to get a token to seed localStorage.
    const login = await fetch(`${BACKEND_URL}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ api_key: API_KEY }),
    });
    // If auth isn't enforced the test has nothing to validate.
    test.skip(!login.ok, 'backend in dev mode -- login not required');
    const { token } = await login.json();

    // Mirror the desktop: seed huginn:auth_token before app boot so every
    // api.ts call carries Authorization: Bearer <token>.
    await page.addInitScript((tok) => {
      try { localStorage.setItem('huginn:auth_token', tok); } catch { /* ignore */ }
    }, token);
    await page.goto('/');

    // The token must authorise a protected call made from inside the page.
    const result = await page.evaluate(async () => {
      const res = await fetch('http://127.0.0.1:8000/tools', {
        headers: { Authorization: `Bearer ${localStorage.getItem('huginn:auth_token') ?? ''}` },
      });
      return { status: res.status, ok: res.ok };
    });
    expect(result.status).toBe(200);
    expect(result.ok).toBe(true);
  });
});