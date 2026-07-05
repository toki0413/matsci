import { test, expect, backendReachable } from './fixtures';

// Block the real WebSocket so the app can never reach "connected", no matter
// whether a backend is up. Exercises the offline UI path deterministically.
function blockWebSocket(page: import('@playwright/test').Page) {
  return page.addInitScript(() => {
    // Throwing on construct makes ws-client's openSocket catch and back off;
    // status never becomes "connected" so the indicator stays "offline".
    (window as any).WebSocket = function () {
      throw new Error('WS blocked by e2e');
    };
  });
}

let backendUp = false;
test.beforeAll(async () => {
  backendUp = await backendReachable();
});

test.describe('websocket connection', () => {
  test('WS connects on page load', async ({ page }) => {
    test.skip(!backendUp, 'backend not running');
    await page.goto('/');
    // The sidebar dot flips to "Backend online" once the agent WS handshakes.
    await expect(page.getByText('Backend online')).toBeVisible({ timeout: 20_000 });
  });

  test('offline indicator shows when the WS cannot connect', async ({ page }) => {
    await blockWebSocket(page);
    await page.goto('/');
    // App starts with isConnected=false and the failed WS keeps it that way,
    // so the red dot + "Backend offline" label should be visible right away.
    await expect(page.getByText('Backend offline')).toBeVisible({ timeout: 10_000 });
  });

  test('heartbeat ping keeps the connection alive', async ({ page }) => {
    test.skip(!backendUp, 'backend not running');
    test.setTimeout(60_000);

    const sentFrames: string[] = [];
    page.on('websocket', (ws) => {
      ws.on('framesent', (frame) => {
        if (typeof frame.payload === 'string') sentFrames.push(frame.payload);
      });
    });

    await page.goto('/');
    await expect(page.getByText('Backend online')).toBeVisible({ timeout: 20_000 });

    // ws-client fires a {type:"ping"} every 30s. The first one proves the
    // heartbeat is wired up and travelling over the open socket.
    await expect.poll(
      () => sentFrames.some((p) => p.includes('"ping"')),
      { timeout: 40_000, intervals: [1_000] },
    ).toBe(true);
  });
});
