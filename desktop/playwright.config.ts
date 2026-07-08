import { defineConfig, devices } from '@playwright/test';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// "type": "module" in package.json means there's no __dirname here.
const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Vite dev server port. vite.config.ts pins this with strictPort: true,
// so 5173 won't work here -- we have to match the real port.
const VITE_PORT = 1420;
const VITE_URL = `http://localhost:${VITE_PORT}`;

const BACKEND_URL = process.env.HUGINN_BACKEND_URL ?? 'http://localhost:8000';
const BACKEND_CWD = path.resolve(__dirname, '..', 'agent');
// Overridable so a venv python can be supplied without touching the config:
//   HUGINN_BACKEND_CMD=".venv/Scripts/python -m uvicorn huginn.server:app ..."
const BACKEND_CMD =
  process.env.HUGINN_BACKEND_CMD ??
  'python -m uvicorn huginn.server:app --host 127.0.0.1 --port 8000';

/**
 * Playwright config for the Huginn desktop frontend.
 *
 * Two web servers are started: the Vite dev server (frontend) and the
 * uvicorn backend. Both reuse an already-running instance when present,
 * which keeps local iteration fast and lets CI bring its own servers.
 */
export default defineConfig({
  testDir: './e2e',
  outputDir: './test-results',
  fullyParallel: true,
  // WS-backed chat tests share the default thread, so serialize to avoid
  // interleaving. Bump up if a spec ever needs isolation via thread_id.
  workers: 1,
  retries: 1,
  // Vite's first-load dep optimization can eat ~20s on a cold run, so give
  // individual tests headroom beyond the 30s default.
  timeout: 60_000,
  reporter: process.env.CI ? 'list' : [['list']],
  use: {
    baseURL: VITE_URL,
    trace: 'on-first-retry',
    actionTimeout: 15_000,
    navigationTimeout: 60_000,
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] },
    },
  ],
  webServer: [
    {
      command: 'npm run dev',
      url: VITE_URL,
      cwd: __dirname,
      reuseExistingServer: !process.env.CI,
      timeout: 90_000,
    },
    {
      command: BACKEND_CMD,
      url: `${BACKEND_URL}/health/live`,
      cwd: BACKEND_CWD,
      reuseExistingServer: !process.env.CI,
      // MCP server initialization takes ~50s on a cold start, so give the
      // backend ample time to come up before Playwright starts probing.
      timeout: 120_000,
      // Dev mode so the spawned backend accepts the unauthenticated WS the
      // desktop app opens (no JWT login in e2e). Ignored when reusing an
      // already-running backend. Don't fail the whole run if the agent env
      // isn't set up -- per-spec beforeAll probes /health/live and skips.
      env: { ...process.env, HUGINN_DEV_MODE: '1' },
      stderr: 'pipe',
      stdout: 'pipe',
    },
  ],
});
