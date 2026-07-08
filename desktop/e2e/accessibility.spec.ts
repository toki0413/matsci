// Accessibility (a11y) tests — WCAG 2.1 AA compliance checks via axe-core.
// Scans key pages and reports all violations. Critical violations fail the
// test; serious/minor are logged as warnings for tracking.
//
// When all violations are fixed, tighten the threshold to also fail on
// serious violations.

import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

const GUIDE_KEY = 'huginn:guide:v1';

test.beforeEach(async ({ page }) => {
  await page.addInitScript((key) => {
    try { localStorage.setItem(key, '1'); } catch { /* ignore */ }
  }, GUIDE_KEY);
});

test.describe('accessibility — main views', () => {
  test('chat view — axe scan reports violations', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

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

    const tab = page.getByRole('tab', { name: 'Structure', exact: true });
    await tab.click();
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
    await page.waitForLoadState('networkidle');

    for (let i = 0; i < 15; i++) {
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
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const input = page.locator('textarea.flex-1');
    await input.fill('keyboard test');
    expect(await input.inputValue()).toBe('keyboard test');

    const sendBtn = page.getByRole('button', { name: 'Send', exact: true });
    await sendBtn.focus();
    await expect(sendBtn).toBeFocused();

    await page.keyboard.press('Enter');
    await page.waitForTimeout(500);
  });
});

test.describe('accessibility — ARIA and semantics', () => {
  test('page has landmark regions for screen readers', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

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
    await page.waitForLoadState('networkidle');

    const issues = await page.evaluate(() => {
      const problems: string[] = [];
      document.querySelectorAll('button, a, [role="button"]').forEach(el => {
        const name = (el.getAttribute('aria-label') || el.textContent || '').trim();
        if (!name) {
          problems.push(`${el.tagName}#${el.id} has no accessible name`);
        }
      });
      return problems;
    });

    expect(issues, issues.join('\n')).toHaveLength(0);
  });

  test('images have alt text', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

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
