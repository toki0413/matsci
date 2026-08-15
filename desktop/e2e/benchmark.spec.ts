import { test, expect, backendReachable } from './fixtures';

// 通过前端调用后端跑分：打开 Benchmark 面板 → 填 categories → 点 Run，
// 触发 POST /bench/run（真实后端 + 真实 LLM），验证跑分结果渲染在 UI 上。
//
// 这是"前端→后端→LLM→结果回显"的完整 E2E 链路，不是直接 curl 后端。
let backendUp = false;

test.beforeAll(async () => {
  backendUp = await backendReachable();
});

// Benchmark 面板通过 sidebar 的 "Benchmark" 工具卡片进入（More Tools 网格）。
async function openBenchmarkPanel(page: import('@playwright/test').Page) {
  await page.goto('/');
  // 打开工具网格（sidebar 的 "More Tools" 按钮）
  await page.getByRole('button', { name: 'More Tools' }).click();
  // 工具卡片按 label "Benchmark" 出现
  await page.getByRole('button', { name: 'Benchmark', exact: true }).click();
  // 面板标题出现
  await expect(page.getByText('Benchmark', { exact: true }).first()).toBeVisible({ timeout: 10_000 });
}

test.describe('benchmark via frontend', () => {
  test('Benchmark tab is reachable from the sidebar tool grid', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('button', { name: 'More Tools' }).click();
    await expect(page.getByRole('button', { name: 'Benchmark', exact: true })).toBeVisible({ timeout: 10_000 });
  });

  test('runs benchmark through the frontend and renders a report', async ({ page }) => {
    test.skip(!backendUp, 'backend not running');
    test.setTimeout(300_000);

    await openBenchmarkPanel(page);

    // 只跑一个轻量 category，避免整库跑太久。math 只有一个数值题。
    const categoriesInput = page.locator('input[placeholder*="Categories"]').first()
      .or(page.locator('input[placeholder*="类别"]').first());
    await categoriesInput.fill('math');

    // 点击 "Run benchmark"（en）或 "运行基准测试"（zh）
    const runBtn = page.getByRole('button', {
      name: /Run benchmark|运行基准测试|运行中|Running/,
    }).first();
    await runBtn.click();

    // 等待结果卡片出现：pass rate 文本（"Pass rate: N%" 或 "通过率: N%"）
    await expect(page.getByText(/Pass rate:|通过率:/)).toBeVisible({ timeout: 240_000 });

    // 报告里应列出任务结果（BenchmarkPanel 每个 task 卡片把 task_id 渲染在 font-mono span 中）
    const taskIds = page.locator('span.font-mono:visible');
    await expect(taskIds.first()).toBeVisible({ timeout: 30_000 });
  });
});