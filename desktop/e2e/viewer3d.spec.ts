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

  test('can navigate to the 3D structure panel', async ({ page }) => {
    await page.goto('/');
    // Sidebar entry whose label is "Structure" (icon + text tab).
    await page.getByRole('tab', { name: 'Structure', exact: true }).click();
    // Empty-state copy from StructureViewer -- confirms the panel mounted.
    await expect(page.getByText('No structure loaded')).toBeVisible({ timeout: 10_000 });
  });
});
