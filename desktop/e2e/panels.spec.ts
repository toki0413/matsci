import { test, expect } from './fixtures';
import type { Page } from '@playwright/test';

// The app shells are lazy-loaded, so a freshly mounted panel takes a beat to
// appear; giving each assertion the same generous timeout keeps the smoke run
// deterministic on cold CI machines.
const MOUNT_TIMEOUT = 10_000;

const sidebar = (page: Page) => page.locator('.sidebar-shell');
const palette = (page: Page) =>
  page.getByRole('dialog', { name: 'Command palette' });

// Open a tool through the "More Tools" palette. The palette filters both the
// quick-action grid and the sidebar-group tabs by the search text; non-primary
// tools (Coder/Files/Terminal/Skills...) only live in the group grid, so once
// we type an exact match there is exactly one button to click.
async function openFromPalette(page: Page, label: string) {
  await sidebar(page).getByRole('button', { name: 'More Tools' }).click();
  await palette(page).getByPlaceholder('Search tools…').fill(label);
  await palette(page).getByRole('button', { name: label, exact: true }).click();
  await expect(palette(page)).toBeHidden();
}

// The primary icon-only tabs live in the sidebar's compact icon bar; their
// only label is the aria-label/title, so select by the role name. exact:true
// keeps a name like "Tools" from also matching the "More Tools" button.
async function openFromIconBar(page: Page, label: string) {
  await sidebar(page).getByRole('button', { name: label, exact: true }).click();
}

test.describe('panel navigation', () => {
  // Panels reached through the More Tools palette. Note that "Files" renders
  // no content when the backend /workspace call fails, but its PanelHeader
  // ("Workspace") still mounts, so the heading is a reliable mount signal.
  const palettePanels: Array<[tool: string, heading: string]> = [
    ['Coder', '💻 Coder Mode'],
    ['Files', 'Workspace'],
    ['Terminal', 'Integrated Terminal'],
    ['Skills', 'Skills Marketplace'],
  ];

  for (const [tool, heading] of palettePanels) {
    test(`opens ${tool} via the More Tools palette`, async ({ page }) => {
      await page.goto('/');
      await openFromPalette(page, tool);
      await expect(
        page.getByRole('heading', { name: heading }).first(),
      ).toBeVisible({ timeout: MOUNT_TIMEOUT });
    });
  }

  // Panels that are primary icon-only tabs in the sidebar bar.
  const iconPanels: Array<[tool: string, heading: string]> = [
    ['Memory', 'Memory'],
    ['Tools', 'Available Tools'],
    ['HPC', 'HPC'],
  ];

  for (const [tool, heading] of iconPanels) {
    test(`opens ${tool} from the sidebar icon bar`, async ({ page }) => {
      await page.goto('/');
      await openFromIconBar(page, tool);
      await expect(
        page.getByRole('heading', { name: heading }).first(),
      ).toBeVisible({ timeout: MOUNT_TIMEOUT });
    });
  }
});