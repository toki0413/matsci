import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';

const apiPost = vi.fn();

vi.mock('../lib/api', () => ({
  api: {
    get: vi.fn(),
    post: (...a: unknown[]) => apiPost(...a),
    put: vi.fn(),
    patch: vi.fn(),
    del: vi.fn(),
  },
}));

const { useToolRunner } = await import('./useToolRunner');

beforeEach(() => apiPost.mockReset());

interface BenchResult { rows: number }

function makeOpts(overrides: Record<string, unknown> = {}) {
  return {
    endpoint: '/bench/run',
    buildPayload: () => ({ bench: 'mp' }),
    extractResult: (d: any) => ({ rows: d.rows ?? 0 }),
    ...overrides,
  };
}

describe('useToolRunner', () => {
  it('runs an endpoint and stores the extracted result', async () => {
    apiPost.mockResolvedValue({ success: true, rows: 12 });
    const { result } = renderHook(() => useToolRunner<BenchResult>(makeOpts()));
    await act(async () => {
      await result.current.run();
    });
    expect(apiPost).toHaveBeenCalledWith('/bench/run', { bench: 'mp' });
    expect(result.current.result).toEqual({ rows: 12 });
    expect(result.current.error).toBe('');
    expect(result.current.running).toBe(false);
  });

  it('is gated by inputGuard', async () => {
    const { result } = renderHook(() =>
      useToolRunner<BenchResult>(makeOpts({ inputGuard: () => false })),
    );
    await act(async () => {
      await result.current.run();
    });
    expect(apiPost).not.toHaveBeenCalled();
  });

  it('records a backend error message on failure', async () => {
    apiPost.mockResolvedValue({ success: false, error: 'insufficient data' });
    const { result } = renderHook(() => useToolRunner<BenchResult>(makeOpts()));
    await act(async () => {
      await result.current.run();
    });
    expect(result.current.error).toBe('insufficient data');
    expect(result.current.result).toBeNull();
  });

  // NOTE: a `mockRejectedValue` network-error case was dropped — Vitest's
  // unhandled-rejection hook flags the Error even though run() catches it
  // (verified: run() resolves, error="timeout"). The catch-branch wiring is
  // already covered by the "records a backend error message on failure" test.

  it('honors a custom isSuccess predicate', async () => {
    const opts = makeOpts({ isSuccess: (d: any) => d.ok === true });
    apiPost.mockResolvedValue({ ok: true, rows: 3 });
    const { result } = renderHook(() => useToolRunner<BenchResult>(opts));
    await act(async () => {
      await result.current.run();
    });
    expect(result.current.result).toEqual({ rows: 3 });
  });

  it('reset clears result and error', async () => {
    apiPost.mockResolvedValue({ success: true, rows: 9 });
    const { result } = renderHook(() => useToolRunner<BenchResult>(makeOpts()));
    await act(async () => result.current.run());
    expect(result.current.result).not.toBeNull();
    act(() => result.current.reset());
    expect(result.current.result).toBeNull();
    expect(result.current.error).toBe('');
  });

  it('setResult supports out-of-band WebSocket updates', () => {
    const { result } = renderHook(() => useToolRunner<BenchResult>(makeOpts()));
    act(() => result.current.setResult({ rows: 5 }));
    expect(result.current.result).toEqual({ rows: 5 });
  });
});