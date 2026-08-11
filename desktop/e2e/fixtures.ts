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
 *
 * A single 3s probe is too fragile: the backend takes ~50s to boot (MCP init)
 * even though /health/live responds early, and under parallel load the event
 * loop can be saturated. So retry a few times with backoff before giving up --
 * "genuinely down" still skips fast, but a merely-slow live backend is not
 * mistaken for a dead one (which would silently skip the contract tests).
 */
export async function backendReachable(timeoutMs = 10_000, attempts = 6): Promise<boolean> {
  for (let i = 0; i < attempts; i++) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeoutMs);
    try {
      const res = await fetch(`${BACKEND_URL}/health/live`, { signal: ctrl.signal });
      if (res.ok) return true;
      console.error(`[backendReachable] attempt ${i + 1}: status ${res.status}`);
    } catch (e) {
      console.error(`[backendReachable] attempt ${i + 1}: ${(e as Error).name}: ${(e as Error).message}`);
    } finally {
      clearTimeout(timer);
    }
    await new Promise((r) => setTimeout(r, 1000));
  }
  return false;
}

// Override the page fixture so the "Welcome to Huginn" onboarding guide is
// pre-dismissed for every test. Its z-50 backdrop intercepts pointer events
// and would otherwise block the chat send button / sidebar clicks. Setting
// the localStorage flag via addInitScript runs before the app's first paint,
// so the modal never mounts.
const GUIDE_KEY = 'muninn:guide:v2';

export const test = base.extend({
  page: async ({ page }, use) => {
    await page.addInitScript((key) => {
      try { localStorage.setItem(key, '1'); } catch { /* ignore */ }
    }, GUIDE_KEY);
    await use(page);
  },
});

export { expect };
