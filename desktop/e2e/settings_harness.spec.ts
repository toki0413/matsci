import { test, expect, backendReachable, BACKEND_URL, authHeaders } from './fixtures';
import type { Page, APIRequestContext } from '@playwright/test';

// 端到端验证"后端设置真实反映到前端":
//   Settings → Advanced 的 HarnessPanel 从后端 GET /config/features 渲染两个
//   调优开关; 切换后 POST 回后端, 后端 is_enabled 翻转, 且 /config/harness/gates
//   反映门控实际生效状态随开关同步翻转 (后端生效断言).
// 后端不可达时整文件 skip (不硬失败), 与其它 e2e 规范一致。

let backendUp = false;

test.beforeAll(async () => {
  backendUp = await backendReachable();
});

// 前端 HarnessPanel 只展示这两个官方 harness 调优开关 (H5 显著性门 / H6 分布外留出)
const GATES = ['harness_significance_gate', 'harness_ood_holdout'] as const;

// HarnessPanel 每个 flag 卡片: div.rounded-lg 内含一个显示 name 的 font-mono span
function flagRow(page: Page, name: string) {
  return page.locator('div.rounded-lg', {
    has: page.locator(`span.font-mono:has-text("${name}")`),
  });
}

function flagCheckbox(page: Page, name: string) {
  return flagRow(page, name).locator('input[type="checkbox"]');
}

// 点击开关滑块 (label) 触发真实用户交互; sr-only input 用 force 也能点,
// 但 label 更贴近真实操作.
function flagToggle(page: Page, name: string) {
  return flagRow(page, name).locator('label');
}

// 后端门控实际生效状态 (读 significance_gate/ood_holdout 的 _harness_enabled)
async function gateEnabled(request: APIRequestContext, name: string): Promise<boolean> {
  const res = await request.get(`${BACKEND_URL}/config/harness/gates`, { headers: authHeaders });
  expect(res.status()).toBe(200);
  const body = await res.json();
  const gate = body.gates.find((g: { name: string }) => g.name === name);
  if (!gate) throw new Error(`gate not found in /config/harness/gates: ${name}`);
  return gate.enabled as boolean;
}

async function openAdvancedTab(page: Page) {
  await page.goto('/');
  await page.locator('.sidebar-shell').getByRole('button', { name: 'Settings', exact: true }).click();
  const advanced = page.getByRole('button', { name: 'Advanced', exact: true });
  await advanced.scrollIntoViewIfNeeded();
  await advanced.click();
}

test.describe('harness switches reflect backend settings (H5/H6)', () => {
  test('Advanced tab renders both harness toggles fetched from backend', async ({ page }) => {
    test.skip(!backendUp, 'backend not running');
    await openAdvancedTab(page);
    for (const name of GATES) {
      await expect(flagCheckbox(page, name)).toHaveCount(1, { timeout: 10_000 });
    }
  });

  test('toggling a switch flips backend is_enabled and gate behavior', async ({ page, request }) => {
    test.skip(!backendUp, 'backend not running');
    await openAdvancedTab(page);

    const name = GATES[0];
    const beforeUi = await flagCheckbox(page, name).isChecked();
    const beforeGate = await gateEnabled(request, name);
    // 初始一致性: 前端渲染的开关状态应与后端门控生效状态一致 (同一真源)
    expect(beforeUi).toBe(beforeGate);

    // 通过 UI 切换开关 → POST 回后端 (persist: true)
    await flagToggle(page, name).click();
    await expect
      .poll(async () => flagCheckbox(page, name).isChecked(), { timeout: 10_000 })
      .toBe(!beforeUi);

    // 后端 is_enabled 真实翻转 (独立于 UI 重新 GET)
    const afterGate = await gateEnabled(request, name);
    expect(afterGate).toBe(!beforeGate);

    // 清理: 切回原状态, 恢复配置文件中的持久化值
    await flagToggle(page, name).click();
    await expect
      .poll(async () => flagCheckbox(page, name).isChecked(), { timeout: 10_000 })
      .toBe(beforeUi);
    expect(await gateEnabled(request, name)).toBe(beforeGate);
  });

  test('both harness gates report their default state off', async ({ request }) => {
    test.skip(!backendUp, 'backend not running');
    const res = await request.get(`${BACKEND_URL}/config/harness/gates`, { headers: authHeaders });
    expect(res.status()).toBe(200);
    const body = await res.json();
    expect(body.count).toBe(GATES.length);
    for (const name of GATES) {
      const gate = body.gates.find((g: { name: string }) => g.name === name);
      expect(gate, `missing gate ${name}`).toBeTruthy();
      expect(gate.default).toBe(false);
    }
  });
});