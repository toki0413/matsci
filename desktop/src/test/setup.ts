import '@testing-library/jest-dom/vitest';

// Global fetch/WebSocket mocks come from the individual specs (they need
// per-test control). Keep shared cleanup here:
// - clear localStorage between tests so config/token state doesn't leak
// - reset API base back to default
afterEach(() => {
  localStorage.clear();
});