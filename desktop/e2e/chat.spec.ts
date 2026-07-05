import { test, expect, backendReachable, TEST_MESSAGE } from './fixtures';

// Chat input is the only textarea carrying the `flex-1` class, so this stays
// stable without having to tag the component.
const chatInput = (page: import('@playwright/test').Page) => page.locator('textarea.flex-1');

// Inject a fake WebSocket that answers user_input with a streaming sequence.
// Lets us exercise the render path for text_delta / tool_call / tool_result
// without a configured LLM on the backend.
function installMockWS(page: import('@playwright/test').Page) {
  return page.addInitScript(() => {
    window.WebSocket = class MockWS {
      onopen: ((ev: any) => void) | null = null;
      onmessage: ((ev: any) => void) | null = null;
      onclose: ((ev: any) => void) | null = null;
      onerror: ((ev: any) => void) | null = null;
      url: string;
      readyState = 0;
      private closed = false;
      constructor(url: string) {
        this.url = url;
        setTimeout(() => {
          if (this.closed) return;
          this.readyState = 1; // OPEN
          this.onopen?.({ type: 'open' });
        }, 0);
      }
      send(data: string) {
        if (this.readyState !== 1) return;
        let msg: any;
        try { msg = JSON.parse(data); } catch { return; }
        if (msg.type !== 'user_input') return;
        // Emit on a microtask delay so the app's send() returns first.
        const emit = (m: any) => setTimeout(() => {
          if (this.closed) return;
          this.onmessage?.({ data: JSON.stringify(m) });
        }, 15);
        emit({ type: 'text_delta', text: 'Done ' });
        emit({ type: 'text_delta', text: 'streaming.' });
        emit({ type: 'tool_call', id: 'tc-1', name: 'shell', args: { cmd: 'echo hi' } });
        emit({ type: 'tool_result', id: 'tc-1', content: 'hi' });
        emit({ type: 'done' });
      }
      close() { this.closed = true; this.readyState = 3; }
    };
  });
}

let backendUp = false;
test.beforeAll(async () => {
  backendUp = await backendReachable();
});

test.describe('chat flow', () => {
  test('page loads with chat input visible', async ({ page }) => {
    await page.goto('/');
    // Runs regardless of backend state -- the input still renders (disabled
    // when offline), so this is a pure structural smoke test.
    await expect(chatInput(page)).toBeVisible();
  });

  test('can type in the message input', async ({ page }) => {
    test.skip(!backendUp, 'backend not running');
    await page.goto('/');
    await expect(page.getByText('Backend online')).toBeVisible({ timeout: 20_000 });
    const input = chatInput(page);
    await expect(input).toBeEnabled();
    await input.fill('ping from e2e');
    await expect(input).toHaveValue('ping from e2e');
  });

  test('send message and see it in the conversation', async ({ page }) => {
    test.skip(!backendUp, 'backend not running');
    await page.goto('/');
    await expect(page.getByText('Backend online')).toBeVisible({ timeout: 20_000 });

    const marker = `E2E-MARK-${Date.now()}`;
    const input = chatInput(page);
    await input.fill(marker);
    await page.getByRole('button', { name: 'Send', exact: true }).click();

    // The user message renders via MessageContent (markdown), so the marker
    // should appear as page text once the turn is committed.
    await expect(page.getByText(marker)).toBeVisible({ timeout: 15_000 });
  });

  test('streaming response appears (mocked WS)', async ({ page }) => {
    await installMockWS(page);
    await page.goto('/');

    // Mock fires onopen synchronously-ish, so the app reports "online" and
    // enables the input without a real backend.
    const input = chatInput(page);
    await expect(input).toBeEnabled({ timeout: 10_000 });
    await input.fill(TEST_MESSAGE.content);
    await page.getByRole('button', { name: 'Send', exact: true }).click();

    // text_delta chunks concatenate into "Done streaming."
    await expect(page.getByText('Done streaming.')).toBeVisible({ timeout: 10_000 });
  });

  test('tool call cards render when received', async ({ page }) => {
    await installMockWS(page);
    await page.goto('/');

    const input = chatInput(page);
    await expect(input).toBeEnabled({ timeout: 10_000 });
    await input.fill('run a tool');
    await page.getByRole('button', { name: 'Send', exact: true }).click();

    // Tool messages render msg.tool_name in a <span>, so the name shows up
    // verbatim once the tool_call frame lands.
    await expect(page.getByText('shell')).toBeVisible({ timeout: 10_000 });
  });
});
