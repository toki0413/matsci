import { test, expect, backendReachable, BACKEND_URL, authHeaders } from './fixtures';

let backendUp = false;
test.beforeAll(async () => {
  backendUp = await backendReachable();
});

test.describe('3D viewer', () => {
  test('element table loads from /viewer3d/elements', async ({ request }) => {
    test.skip(!backendUp, 'backend not running');
    const res = await request.get(`${BACKEND_URL}/viewer3d/elements`, {
      headers: authHeaders,
    });
    expect(res.status()).toBe(200);
    const body = await res.json();
    expect(Array.isArray(body.elements)).toBe(true);
    expect(body.elements.length).toBeGreaterThan(0);
    // CPK table starts at hydrogen -- a cheap "real data" sanity check.
    expect(body.elements[0].symbol).toBe('H');
  });

  test('can navigate to a secondary panel from the sidebar', async ({ page }) => {
    await page.goto('/');
    // The app no longer has a "Structure" tab; navigate to a real secondary
    // view instead. Sidebar nav exposes Tools as a button (not role=tab).
    await page.getByRole('button', { name: 'Tools', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Available Tools' })).toBeVisible({ timeout: 10_000 });
  });
});
