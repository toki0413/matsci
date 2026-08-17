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

// In CI the backend-aware specs must not silently skip when the backend is
// down -- that would let an acceptance run pass-with-hole. So when CI is set
// and the probe gives up, we throw and the whole file reports a beforeAll
// failure instead of a quiet skip. Local dev keeps the graceful skip.
const CI = !!process.env.CI;

/**
 * Probe the backend. Specs call this in beforeAll and skip the tests that
 * genuinely need a live backend, instead of hard-failing the run.
 *
 * The backend's very first /health/live hit is slow (~2s+, and far worse on a
 * cold MCP/agent init), so a single probe with a tight timeout flaps into
 * "not running" and silently skips the most important integration specs. We
 * therefore retry with backoff: the first slow response doubles as warmup,
 * and a retry on the now-warm backend answers fast. In CI the Playwright
 * webServer has already waited for /health/live before tests run, so this
 * almost never waits; it's a local-dev/robustness guard.
 */
export async function backendReachable(attempts = 4): Promise<boolean> {
  for (let i = 0; i < attempts; i++) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 8000);
    try {
      const res = await fetch(`${BACKEND_URL}/health/live`, { signal: ctrl.signal });
      if (res.ok) return true;
    } catch {
      // connection refused / aborted -> try again below
    } finally {
      clearTimeout(timer);
    }
    // Gap lets the backend finish whatever cold-init was blocking /health/live.
    await new Promise((r) => setTimeout(r, 1500));
  }
  // Down. In CI a backend-dependent spec must not silently skip -- an
  // acceptance run passing-with-hole is worse than a loud failure -- so throw
  // and let the file's beforeAll report red. Locally, return false so the body
  // skips and a frontend-only run stays green.
  if (CI) {
    throw new Error(
      `backend ${BACKEND_URL} unreachable after ${attempts} probes: ` +
        'backend-aware specs would silently skip. Start it (HUGINN_DEV_MODE=1) or fix the probe.',
    );
  }
  return false;
}

// Override the page fixture so the "Welcome to Huginn" onboarding guide is
// pre-dismissed for every test. Its z-50 backdrop intercepts pointer events
// and would otherwise block the chat send button / sidebar clicks. Setting
// the localStorage flag via addInitScript runs before the app's first paint,
// so the modal never mounts.
// Must match the app's key in useChatAndConnection.ts: GUIDE_KEY = "muninn:guide:v2".
// A mismatch lets the onboarding modal mount and block pointer events.
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
