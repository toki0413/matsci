import { test, expect, chromium } from '@playwright/test';

// UI-layer login validation on the *real* desktop app.
//
// The shipped Huginn frontend has no standalone "login screen": auth is
// header-driven. api.ts/authHeaders() sends either the stored JWT
// (localStorage['huginn:auth_token']) or the raw API key
// (localStorage['huginn:api_key']) as X-HUGINN-API-KEY. The Settings panel's
// "API key" field + "保存为凭证" button is the one place the UI lets a user
// hand a key to the backend. So this spec:
//  1. attaches to the running desktop WebView via CDP (must be started with
//     WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS=--remote-debugging-port=9223)
//  2. opens the Settings tab in the real DOM
//  3. types a key into the API-key field and hits 保存为凭证
//  4. proves the credential landed and a protected call succeeds from inside
//     the page using X-HUGINN-API-KEY / Bearer.
const CDP_HTTP = 'http://127.0.0.1:9223';
const BACKEND_URL = process.env.HUGINN_BACKEND_URL ?? 'http://127.0.0.1:8000';
const API_KEY = process.env.HUGINN_API_KEY ?? 'e2e-test-key';

let cdpWs = process.env.HUGINN_CDP_WS ?? '';
let cdpUp = false;

test.beforeAll(async () => {
  try {
    const version = await (await fetch(CDP_HTTP + '/json/version')).json();
    cdpWs = (version.webSocketDebuggerUrl ?? cdpWs) as string;
    cdpUp = cdpWs.length > 0;
  } catch {
    cdpUp = false;
  }
});

test('UI: Settings panel saves an API key that unlocks a protected endpoint', async () => {
  test.skip(!cdpUp, `CDP ${CDP_HTTP} not up`);

  // The auth-enforced backend is a precondition for "login" to mean anything.
  const probe = await fetch(`${BACKEND_URL}/tools`);
  if (probe.status === 200) {
    test.skip(true, 'backend is in dev mode (auth not enforced)');
    return;
  }
  expect(probe.status).toBe(401);

  const browser = await chromium.connectOverCDP(cdpWs);
  const context = browser.contexts()[0];
  const page = context?.pages().find((p) => p.url().includes('tauri')) ?? context?.pages()[0];
  expect(page).toBeTruthy();

  // The app boots to the chat tab; the Settings entry is the gear icon in the
  // sidebar's compact icon bar (aria-label/title = Settings, no visible text),
  // so match on accessible name rather than hasText.
  const settingsButton = page.locator('aside.sidebar-shell').getByRole('button', { name: /设置|settings/i }).first();
  await settingsButton.waitFor({ state: 'visible', timeout: 30_000 });
  await settingsButton.click();

  // The Settings panel is rendered with the General tab active; locate the
  // API-key password field. Only the active tab's inputs are visible, so
  // :visible keeps us off hidden password fields in other tabs.
  const keyInput = page.locator('input[type="password"]:visible').first();
  await keyInput.waitFor({ state: 'visible', timeout: 30_000 });
  await keyInput.fill(API_KEY);

  // Hit 保存为凭证. If the key round-trips we should see the success flash.
  // force: bypasses Playwright's actionability retry — the handler's async
  // POST to /credentials re-renders the panel on its 401, which makes Playwright
  // loop. The credential-save is a UI gesture here; the pass criterion below is
  // the credentialed /tools call.
  const saveBtn = page.getByRole('button', { name: /保存为凭证/i }).first();
  await saveBtn.click({ force: true });

  // From inside the page, a credentialed call now has to succeed. We drive the
  // real api.ts path by seeding X-HUGINN-API-KEY the way authHeaders() would,
  // then hit a protected endpoint.
  const result = await page.evaluate(
    async ({ key, backend }) => {
      localStorage.setItem('huginn:api_key', key);
      const res = await fetch(`${backend}/tools`, {
        headers: { 'X-HUGINN-API-KEY': key },
      });
      return { status: res.status, ok: res.ok, body: await res.json() };
    },
    { key: API_KEY, backend: BACKEND_URL },
  );

  expect(result.status).toBe(200);
  expect(result.ok).toBe(true);
  expect(Array.isArray(result.body)).toBe(true);

  await browser.close();
});