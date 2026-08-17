// Runs once, before any spec file, in a dedicated Playwright setup worker.
//
// Why: the per-spec `backendReachable` probe lives inside each file's
// beforeAll, so whichever spec files run first race the backend's cold
// start (the very first /health/live hit can take seconds while MCP/agent
// init is lazy). Those early probes can flap into "not running" and the
// most important integration specs silently skip. `reuseExistingServer`
// sidesteps Playwright's own webServer readiness wait, so nothing warms the
// backend before specs start. Warming it here makes the later probes
// near-instant and skips deterministic.
const BACKEND_URL = process.env.HUGINN_BACKEND_URL ?? 'http://localhost:8000';

async function warmBackend(): Promise<void> {
  // Give the backend a generous window: a cold MCP/agent init can take ~1min.
  for (let i = 0; i < 30; i++) {
    try {
      const r = await fetch(`${BACKEND_URL}/health/live`, { signal: AbortSignal.timeout(8000) });
      if (r.ok) return;
    } catch {
      // not up yet (or aborted) -> keep polling
    }
    await new Promise((res) => setTimeout(res, 2000));
  }
  // If it never warmed, don't hard-fail: the per-spec probes will still
  // gracefully skip the backend-dependent cases. Frontend-only runs stay green.
  console.warn(`[global-setup] backend ${BACKEND_URL} not reachable after warmup window`);
}

async function globalSetup(): Promise<void> {
  await warmBackend();
}

export default globalSetup;