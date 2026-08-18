// Accessibility (a11y) tests — WCAG 2.1 AA compliance checks via axe-core.
// Scans key pages and reports all violations. Critical violations fail the
// test; serious/minor are logged as warnings for tracking.
//
// When all violations are fixed, tighten the threshold to also fail on
// serious violations.

import { test, expect } from './fixtures';
import AxeBuilder from '@axe-core/playwright';

test.describe('accessibility — main views', () => {
  test('chat view — axe scan reports violations', async ({ page }) => {
    await page.goto('/');
    // The /events/stream SSE keeps the page "network active" forever, so
    // networkidle never fires. Await the chat input being enabled instead —
    // that's the real mount signal the rest of the suite relies on.
    await page.waitForLoadState('load');
    await expect(page.locator('textarea.flex-1')).toBeEnabled({ timeout: 30_000 });

    const results = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
      .analyze();

    // Log all violations for visibility in CI output.
    if (results.violations.length > 0) {
      console.log(`\n[a11y] ${results.violations.length} violations found:`);
      for (const v of results.violations) {
        const impact = v.impact || 'unknown';
        console.log(`  [${impact}] ${v.id}: ${v.description}`);
        console.log(`    help: ${v.helpUrl}`);
      }
    }

    // Only critical violations block the build. Serious and below are
    // tracked as tech debt — fix them in the component, then tighten
    // this assertion to include serious.
    const critical = results.violations.filter(v => v.impact === 'critical');
    expect(critical, `${critical.length} critical a11y violations`).toHaveLength(0);
  });

  test('structure panel — axe scan reports violations', async ({ page }) => {
    await page.goto('/');
    // networkidle never fires with an active WebSocket — use 'load' instead.
    await page.waitForLoadState('load');

    const tab = page.getByRole('button', { name: 'More Tools' });
    await tab.click();
    await page.getByPlaceholder('Search tools…').fill('Structure');
    await page.getByRole('button', { name: 'Structure', exact: true }).click();
    await expect(page.getByText('No structure loaded')).toBeVisible({ timeout: 10_000 });

    const results = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa'])
      .analyze();

    if (results.violations.length > 0) {
      console.log(`\n[a11y] Structure panel: ${results.violations.length} violations:`);
      for (const v of results.violations) {
        const impact = v.impact || 'unknown';
        console.log(`  [${impact}] ${v.id}: ${v.description}`);
      }
    }

    const critical = results.violations.filter(v => v.impact === 'critical');
    expect(critical, `${critical.length} critical a11y violations`).toHaveLength(0);
  });
});

test.describe('accessibility — keyboard navigation', () => {
  test('chat input is reachable via keyboard tab', async ({ browserName, page }) => {
    // Firefox and WebKit have different tab-focus behavior; skip on
    // those until the focus management is fixed.
    test.skip(browserName === 'firefox' || browserName === 'webkit',
      'Tab order needs fixing for Firefox/WebKit focus model');

    await page.goto('/');
    // The /events/stream SSE keeps the page "network active" forever, so
    // networkidle never fires. Await the chat input being enabled instead —
    // that's the real mount signal the rest of the suite relies on.
    await page.waitForLoadState('load');
    await expect(page.locator('textarea.flex-1')).toBeEnabled({ timeout: 30_000 });

    // The chat textarea is disabled while the WS handshake is pending, and a
    // disabled control is skipped by Tab. Wait for it to be enabled first so
    // the tab-order check exercises the real focus chain.
    await expect(page.locator('textarea.flex-1')).toBeEnabled({ timeout: 30_000 });

    for (let i = 0; i < 50; i++) {
      await page.keyboard.press('Tab');
      const focused = await page.evaluate(() => {
        const el = document.activeElement;
        return el ? el.tagName + '.' + el.className : '';
      });
      if (focused.startsWith('TEXTAREA')) return;
    }
    const focused = await page.evaluate(() => {
      const el = document.activeElement;
      return el ? `${el.tagName}(${el.className})` : 'none';
    });
    throw new Error(`Chat input not reachable via Tab. Last focused: ${focused}`);
  });

  test('send button is keyboard-activatable', async ({ page }) => {
    // The Send button stays disabled until isConnected flips true. Under a
    // full-suite run the real WS can connect slowly (or the probe thread is
    // starved), so a plain goto flakes toEnabled on Firefox. This a11y test
    // is about keyboard activation, not connectivity -- stub the WS so the
    // button's enable/disable is deterministic and the keyboard path alone
    // is exercised. Mirrors the mock in chat.spec.ts.
    await page.addInitScript(() => {
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
        send() { /* not needed here; just keep the WS "open" */ }
        close() { this.closed = true; this.readyState = 3; }
      };
    });
    await page.goto('/');
    // The /events/stream SSE keeps the page "network active" forever, so
    // networkidle never fires. Await the chat input being enabled instead —
    // that's the real mount signal the rest of the suite relies on.
    await page.waitForLoadState('load');
    await expect(page.locator('textarea.flex-1')).toBeEnabled({ timeout: 30_000 });

    const input = page.locator('textarea.flex-1');
    await input.fill('keyboard test');
    expect(await input.inputValue()).toBe('keyboard test');

    const sendBtn = page.getByRole('button', { name: 'Send', exact: true });
    // fill() 只写进 DOM, disabled 由 React 受控态决定; 不等按钮真正使能就 focus,
    // 会在 disabled→enabled 的竞态窗口里拿到 "inactive" 偶发失败。先钉住使能态。
    await expect(sendBtn).toBeEnabled({ timeout: 10_000 });
    await sendBtn.focus();
    await expect(sendBtn).toBeFocused();

    await page.keyboard.press('Enter');

    // Pressing Enter must actually send: sendMessage optimistically clears the
    // input and renders the user message, so both are observable even though
    // the mock WS never replies. This is what proves the keyboard path works —
    // without it, a broken Enter handler would silently pass.
    await expect(input).toHaveValue('', { timeout: 10_000 });
    await expect(page.getByText('keyboard test')).toBeVisible({ timeout: 10_000 });
  });
});

test.describe('accessibility — ARIA and semantics', () => {
  test('page has landmark regions for screen readers', async ({ page }) => {
    await page.goto('/');
    // The /events/stream SSE keeps the page "network active" forever, so
    // networkidle never fires. Await the chat input being enabled instead —
    // that's the real mount signal the rest of the suite relies on.
    await page.waitForLoadState('load');
    await expect(page.locator('textarea.flex-1')).toBeEnabled({ timeout: 30_000 });

    // Check for landmark elements (nav, main, aside, header, etc.)
    // or ARIA roles that define page structure for screen readers.
    const landmarks = await page.evaluate(() => {
      const selectors = [
        'main', 'nav', 'aside', 'header', 'footer',
        '[role="main"]', '[role="navigation"]', '[role="complementary"]',
        '[role="banner"]', '[role="contentinfo"]',
      ];
      return selectors.reduce((count, sel) => {
        return count + document.querySelectorAll(sel).length;
      }, 0);
    });

    // At least one landmark should exist for screen reader navigation.
    expect(landmarks, 'No landmark regions found').toBeGreaterThan(0);
  });

  test('interactive elements have accessible names', async ({ page }) => {
    await page.goto('/');
    // The /events/stream SSE keeps the page "network active" forever, so
    // networkidle never fires. Await the chat input being enabled instead —
    // that's the real mount signal the rest of the suite relies on.
    await page.waitForLoadState('load');
    await expect(page.locator('textarea.flex-1')).toBeEnabled({ timeout: 30_000 });

    const issues = await page.evaluate(() => {
      const problems: string[] = [];
      document.querySelectorAll('button, a, [role="button"]').forEach(el => {
        // Accessible name = aria-label, else aria-labelledby target text,
        // else title, else the first image's alt (icon buttons), else content.
        // element.accessibleName isn't reliable on headless Chromium, so
        // approximate the accessible name computation ourselves.
        const labelledBy = el.getAttribute('aria-labelledby');
        let byId = '';
        if (labelledBy) {
          const t = document.getElementById(labelledBy);
          if (t) byId = t.textContent || '';
        }
        const img = el.querySelector('img:not([aria-hidden="true"])');
        const svgTitle = el.querySelector('svg title');
        const name =
          (el.getAttribute('aria-label') ||
            byId ||
            el.getAttribute('title') ||
            img?.getAttribute('alt') ||
            svgTitle?.textContent ||
            el.textContent ||
            '').trim();
        if (!name) {
          problems.push(`${el.tagName}#${el.id} (${el.getAttribute('class') || ''}) has no accessible name`);
        }
      });
      return problems;
    });

    expect(issues, issues.join('\n')).toHaveLength(0);
  });

  test('images have alt text', async ({ page }) => {
    await page.goto('/');
    // The /events/stream SSE keeps the page "network active" forever, so
    // networkidle never fires. Await the chat input being enabled instead —
    // that's the real mount signal the rest of the suite relies on.
    await page.waitForLoadState('load');
    await expect(page.locator('textarea.flex-1')).toBeEnabled({ timeout: 30_000 });

    const imgIssues = await page.evaluate(() => {
      const problems: string[] = [];
      document.querySelectorAll('img').forEach(el => {
        if (!el.hasAttribute('alt')) {
          problems.push(`img src=${el.getAttribute('src')?.slice(0, 50)} has no alt`);
        }
      });
      return problems;
    });

    expect(imgIssues, imgIssues.join('\n')).toHaveLength(0);
  });
});
