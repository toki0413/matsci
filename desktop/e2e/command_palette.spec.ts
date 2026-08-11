import { test, expect } from './fixtures';
import type { Page } from '@playwright/test';

// The palette's search box is the single element with this placeholder, so it
// doubles as a stable "palette is open" probe.
const paletteSearch = (page: Page) => page.getByPlaceholder('Search tools…');

// Open the palette via the sidebar "More Tools" button. Ctrl+K is exercised in
// its own test; the rest use this reliable entry point.
async function openPalette(page: Page) {
  await page.getByRole('button', { name: 'More Tools' }).click();
  await expect(paletteSearch(page)).toBeVisible({ timeout: 10_000 });
}

test.describe('command palette', () => {
  test('Ctrl+K opens the palette', async ({ page }) => {
    await page.goto('/');
    await page.keyboard.press('Control+k');
    await expect(paletteSearch(page)).toBeVisible({ timeout: 10_000 });
  });

  test('search filters tools', async ({ page }) => {
    await page.goto('/');
    await openPalette(page);

    await paletteSearch(page).fill('team');
    // "Team" is the only label containing "team".
    await expect(page.getByRole('button', { name: 'Team' })).toBeVisible();
    // A sibling core tool is filtered out (leaves the DOM entirely).
    await expect(page.getByRole('button', { name: 'Coder' })).toHaveCount(0);
  });

  test('clicking a tool closes the palette and switches the tab', async ({ page }) => {
    await page.goto('/');
    await openPalette(page);

    await page.getByRole('button', { name: 'Team' }).click();

    // Palette overlay is gone.
    await expect(paletteSearch(page)).toBeHidden({ timeout: 10_000 });
    // Tab switched: the sidebar badge for a non-primary tool appears.
    await expect(page.locator('.sidebar-shell').getByText('Team')).toBeVisible({ timeout: 10_000 });
  });

  test('Escape closes the palette', async ({ page }) => {
    await page.goto('/');
    await openPalette(page);

    await page.keyboard.press('Escape');
    await expect(paletteSearch(page)).toBeHidden({ timeout: 10_000 });
  });

  test('empty search shows all tools', async ({ page }) => {
    await page.goto('/');
    await openPalette(page);
    // The search box is empty on open, so one tool from each sidebar group is
    // visible. The palette renders the sidebar groups (core/workspace/system),
    // so pick a representative tab from each. Query within the palette dialog
    // to avoid matching the sidebar's own nav buttons (e.g. "Settings").
    const palette = page.getByLabel('Command palette');
    for (const label of ['Team', 'Files', 'Settings']) {
      await expect(palette.getByRole('button', { name: label, exact: true })).toBeVisible();
    }
    // And the no-match message is not rendered.
    await expect(page.getByText('No tools match')).toHaveCount(0);
  });

  test('no match shows the "No tools match" message', async ({ page }) => {
    await page.goto('/');
    await openPalette(page);

    await paletteSearch(page).fill('zzzzz');
    await expect(page.getByText('No tools match')).toBeVisible({ timeout: 10_000 });
  });
});
