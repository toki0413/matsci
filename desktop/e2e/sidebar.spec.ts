import { test, expect } from './fixtures';
import type { Page } from '@playwright/test';

// The new chat-first sidebar. <aside class="sidebar-shell"> is the only element
// carrying that class, so scoping nav lookups to it keeps them stable even when
// the header happens to show the same label text.
const sidebar = (page: Page) => page.locator('.sidebar-shell');

// The four primary destinations rendered directly in the sidebar nav.
const PRIMARY_ITEMS = ['Chat', 'Knowledge', 'Threads', 'Settings'] as const;

test.describe('minimal sidebar', () => {
  test('renders the four primary destinations', async ({ page }) => {
    await page.goto('/');
    for (const label of PRIMARY_ITEMS) {
      await expect(
        sidebar(page).getByRole('button', { name: label, exact: true }),
      ).toBeVisible();
    }
  });

  test('"More Tools" button is visible', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByRole('button', { name: 'More Tools' })).toBeVisible();
  });

  test('clicking Knowledge opens the knowledge panel', async ({ page }) => {
    await page.goto('/');
    await sidebar(page).getByRole('button', { name: 'Knowledge', exact: true }).click();
    // KnowledgePanel renders a PanelHeader whose <h2> is hard-coded "Knowledge
    // Base"; targeting the heading avoids matching the panel's own error text
    // ("Knowledge base backend is not available...") which also shows up.
    await expect(page.getByRole('heading', { name: 'Knowledge Base', exact: true })).toBeVisible({ timeout: 10_000 });
  });

  test('clicking Settings opens the settings panel', async ({ page }) => {
    await page.goto('/');
    await sidebar(page).getByRole('button', { name: 'Settings', exact: true }).click();
    // SettingsTabNav always renders the full set of tab buttons; "credentials"
    // only lives inside the settings panel, so seeing it proves the panel mounted.
    await expect(page.getByRole('button', { name: 'credentials' })).toBeVisible({ timeout: 10_000 });
  });

  test('clicking Chat returns to the chat view', async ({ page }) => {
    await page.goto('/');
    // Detour through Settings first so we know we are not already on chat.
    await sidebar(page).getByRole('button', { name: 'Settings', exact: true }).click();
    await expect(page.getByRole('button', { name: 'credentials' })).toBeVisible({ timeout: 10_000 });

    await sidebar(page).getByRole('button', { name: 'Chat', exact: true }).click();
    // ChatPanel's input is the only textarea.flex-1 on the chat tab.
    await expect(page.locator('textarea.flex-1')).toBeVisible({ timeout: 10_000 });
  });

  test('hide/show sidebar toggle works', async ({ page }) => {
    await page.goto('/');
    await expect(sidebar(page)).toBeVisible();

    // Collapse: the aside is conditionally rendered, so it leaves the DOM.
    await page.getByTitle('Hide sidebar').click();
    await expect(sidebar(page)).toBeHidden();
    await expect(page.getByTitle('Show sidebar')).toBeVisible();

    // Expand again.
    await page.getByTitle('Show sidebar').click();
    await expect(sidebar(page)).toBeVisible();
  });

  test('active non-primary tool shows a compact badge with back-to-chat', async ({ page }) => {
    await page.goto('/');
    // Open the palette and pick a non-primary tool (Team).
    await page.getByRole('button', { name: 'More Tools' }).click();
    await page.getByPlaceholder('Search tools...').fill('Team');
    await page.getByRole('button', { name: 'Team' }).click();

    // The sidebar badge shows the active tool label plus a back-to-chat button.
    await expect(sidebar(page).getByText('Team')).toBeVisible({ timeout: 10_000 });
    await expect(sidebar(page).getByTitle('Back to chat')).toBeVisible({ timeout: 10_000 });
  });
});
