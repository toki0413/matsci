import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

// Unit/component tests (vitest). Keep separate from vite.config.ts so the
// `npm run dev` / `vite build` path stays untouched by test setup.
//
// jsdom environment for everything: the lib/hooks/components under test read
// localStorage, window, WebSocket, etc. that only exist in a DOM runtime.
export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    // colocated beside the modules they cover inside src/
    include: ['src/**/*.{test,spec}.{ts,tsx}'],
    exclude: ['e2e/**', 'node_modules/**', 'dist/**'],
    css: false,
    // 24 个 spec 文件, 默认 forks pool 按 CPU 核数全量并行拉起, 直接吃爆
    // runner (7G/2vCPU), 所有 worker "Timeout waiting for worker to respond".
    // 限制并发到 2 个进程即可稳定跑完. (Vitest 4 里 poolOptions 已废弃,
    // 并发上限是顶层 maxWorkers/minWorkers.)
    maxWorkers: 2,
    minWorkers: 1,
    coverage: {
      provider: 'v8',
      reporter: ['text', 'html'],
      include: ['src/lib/**', 'src/hooks/**', 'src/components/**'],
      exclude: ['src/**/*.{test,spec}.{ts,tsx}'],
    },
  },
});