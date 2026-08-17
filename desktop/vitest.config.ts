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
    coverage: {
      provider: 'v8',
      reporter: ['text', 'html'],
      include: ['src/lib/**', 'src/hooks/**', 'src/components/**'],
      exclude: ['src/**/*.{test,spec}.{ts,tsx}'],
    },
  },
});