import { test as base, expect } from '@playwright/test';

// Backend base URL. The desktop app talks to this directly (no Vite proxy),
// so tests that hit REST endpoints use the same host.
export const BACKEND_URL = process.env.HUGINN_BACKEND_URL ?? 'http://localhost:8000';

// API key for endpoints behind require_api_key. The ones we exercise
// (/health/*, /tools, /skills, /viewer3d/elements) are open, so the
// default placeholder is fine; set HUGINN_API_KEY for protected routes.
export const API_KEY = process.env.HUGINN_API_KEY ?? 'test-key';

export const authHeaders: Record<string, string> = {
  'X-HUGINN-API-KEY': API_KEY,
};

// A representative user message + thread, shared across chat specs so the
// shape stays in one place if the contract changes.
export const TEST_MESSAGE = {
  content: 'Hello from Playwright E2E',
  thread_id: 'e2e-default',
};

/**
 * Probe the backend once. Specs call this in beforeAll and skip the tests
 * that genuinely need a live backend, instead of hard-failing the run.
 */
export async function backendReachable(timeoutMs = 3000): Promise<boolean> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${BACKEND_URL}/health/live`, { signal: ctrl.signal });
    return res.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

// Override the page fixture so the "Welcome to Huginn" onboarding guide is
// pre-dismissed for every test. Its z-50 backdrop intercepts pointer events
// and would otherwise block the chat send button / sidebar clicks. Setting
// the localStorage flag via addInitScript runs before the app's first paint,
// so the modal never mounts.
const GUIDE_KEY = 'huginn:guide:v1';

export const test = base.extend({
  page: async ({ page }, use) => {
    await page.addInitScript((key) => {
      try { localStorage.setItem(key, '1'); } catch { /* ignore */ }
    }, GUIDE_KEY);
    await use(page);
  },
});

export { expect };
