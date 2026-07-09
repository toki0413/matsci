import { test, expect, TEST_MESSAGE } from './fixtures';
import type { Page } from '@playwright/test';

// Same idea as chat.spec.ts's installMockWS, but the streamed frames are passed
// in so each test can drive a specific text / reasoning / tool sequence. The
// app only sends `user_input` over the wire, so the mock replies to that and
// ignores heartbeat pings.
type Frame = Record<string, any>;

function installMockWS(page: Page, frames: Frame[]) {
  return page.addInitScript((frames: Frame[]) => {
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
        // Emit one frame per ~15ms tick so the app's send() returns first and
        // the streaming batcher (rAF flush) gets a chance to render each chunk.
        frames.forEach((f, i) => setTimeout(() => {
          if (this.closed) return;
          this.onmessage?.({ data: JSON.stringify(f) });
        }, 15 * (i + 1)));
      }
      close() { this.closed = true; this.readyState = 3; }
    };
  }, frames);
}

// Install the mock, load the page, wait for the input to come online (the mock
// fires onopen so the app flips to "connected"), then submit a message to kick
// off the streamed response.
async function sendAndAwaitStream(page: Page, frames: Frame[]) {
  await installMockWS(page, frames);
  await page.goto('/');
  const input = page.locator('textarea.flex-1');
  await expect(input).toBeEnabled({ timeout: 10_000 });
  await input.fill(TEST_MESSAGE.content);
  await page.getByRole('button', { name: 'Send', exact: true }).click();
}

test.describe('text + reasoning dual-stream rendering', () => {
  test('text_delta renders in the chat area', async ({ page }) => {
    await sendAndAwaitStream(page, [
      { type: 'text_delta', text: 'STREAM-ALPHA ' },
      { type: 'text_delta', text: 'STREAM-BETA' },
      { type: 'done' },
    ]);
    await expect(page.getByText('STREAM-ALPHA')).toBeVisible({ timeout: 10_000 });
    await expect(page.getByText('STREAM-BETA')).toBeVisible({ timeout: 10_000 });
  });

  test('reasoning_delta renders in the reasoning area', async ({ page }) => {
    // No `done` frame: the streaming placeholder stays open, which keeps the
    // reasoning disclosure expanded so its content is visible for assertion.
    await sendAndAwaitStream(page, [
      { type: 'reasoning_delta', text: 'THINK-ONE' },
      { type: 'reasoning_delta', text: 'THINK-TWO' },
    ]);
    await expect(page.getByText('THINK-ONE')).toBeVisible({ timeout: 10_000 });
    await expect(page.getByText('THINK-TWO')).toBeVisible({ timeout: 10_000 });
  });

  test('tool_call renders a tool card with the tool name', async ({ page }) => {
    await sendAndAwaitStream(page, [
      { type: 'tool_call', id: 'tc-1', name: 'shell', args: { cmd: 'echo hi' } },
      { type: 'done' },
    ]);
    // Tool cards render msg.tool_name in a <span>, so the name shows verbatim.
    await expect(page.getByText('shell')).toBeVisible({ timeout: 10_000 });
  });

  test('tool_result updates the tool card with the result', async ({ page }) => {
    await sendAndAwaitStream(page, [
      { type: 'tool_call', id: 'tc-1', name: 'calc_tool', args: { expr: '1+1' } },
      { type: 'tool_result', id: 'tc-1', content: 'CALC-OUT-42' },
      { type: 'done' },
    ]);
    // Tool name appears first...
    await expect(page.getByText('calc_tool')).toBeVisible({ timeout: 10_000 });
    // ...then the result content renders once tool_result lands.
    await expect(page.getByText('CALC-OUT-42')).toBeVisible({ timeout: 10_000 });
  });

  test('done re-enables the chat input', async ({ page }) => {
    await sendAndAwaitStream(page, [
      { type: 'text_delta', text: 'DONE-MARKER' },
      { type: 'done' },
    ]);
    // The done frame ends streaming: the input is usable again and the send
    // button no longer reads the streaming indicator.
    await expect(page.locator('textarea.flex-1')).toBeEnabled({ timeout: 10_000 });
    await expect(page.getByRole('button', { name: 'Send', exact: true })).toBeVisible({ timeout: 10_000 });
  });

  test('interleaved text and reasoning deltas land in separate areas', async ({ page }) => {
    await sendAndAwaitStream(page, [
      { type: 'reasoning_delta', text: 'REASON-A' },
      { type: 'text_delta', text: 'TEXT-A' },
      { type: 'reasoning_delta', text: 'REASON-B' },
      { type: 'text_delta', text: 'TEXT-B' },
      { type: 'done' },
    ]);

    // Text deltas accumulate in the message body.
    await expect(page.getByText('TEXT-A')).toBeVisible({ timeout: 10_000 });
    await expect(page.getByText('TEXT-B')).toBeVisible({ timeout: 10_000 });

    // Reasoning lives behind the "thought process" disclosure; expand it.
    const reasoning = page.locator('details').filter({ hasText: 'thought process' });
    await reasoning.locator('summary').click();
    await expect(reasoning.getByText('REASON-A')).toBeVisible({ timeout: 10_000 });
    await expect(reasoning.getByText('REASON-B')).toBeVisible({ timeout: 10_000 });
    // Text never leaked into the reasoning block.
    await expect(reasoning.getByText('TEXT-A')).toHaveCount(0);
  });
});
