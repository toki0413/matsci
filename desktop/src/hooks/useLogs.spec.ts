import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';

// Listen is captured here so tests can emit synthetic backend-log events.
const listeners = new Map<string, (e: { payload: unknown }) => void>();
const listen = vi.fn(async (kind: string, cb: (e: { payload: unknown }) => void) => {
  listeners.set(kind, cb);
  return () => listeners.delete(kind);
});

vi.mock('@tauri-apps/api/event', () => ({ listen }));

const { useLogs } = await import('./useLogs');

beforeEach(() => {
  listeners.clear();
  listen.mockClear();
});

function emit(payload: unknown) {
  act(() => {
    listeners.get('backend-log')?.({ payload });
  });
}

describe('useLogs', () => {
  it('does not subscribe without the Tauri runtime', () => {
    delete (window as any).__TAURI_INTERNALS__;
    renderHook(() => useLogs());
    expect(listen).not.toHaveBeenCalled();
  });

  it('subscribes to backend-log events when running in Tauri', () => {
    (window as any).__TAURI_INTERNALS__ = {};
    renderHook(() => useLogs());
    expect(listen).toHaveBeenCalledWith('backend-log', expect.any(Function));
    delete (window as any).__TAURI_INTERNALS__;
  });

  it('accumulates stdout log lines with a timestamp', () => {
    (window as any).__TAURI_INTERNALS__ = {};
    const { result } = renderHook(() => useLogs());
    emit({ source: 'stdout', text: 'hello' });
    emit({ source: 'stdout', text: 'world' });
    expect(result.current.backendLogs.map((l) => l.text)).toEqual(['hello', 'world']);
    expect(result.current.backendLogs.every((l) => l.source === 'stdout')).toBe(true);
    expect(typeof result.current.backendLogs[0].time).toBe('string');
    delete (window as any).__TAURI_INTERNALS__;
  });

  it('classifies stderr sources and filters nothing by default', () => {
    (window as any).__TAURI_INTERNALS__ = {};
    const { result } = renderHook(() => useLogs());
    emit({ source: 'stderr', text: 'traceback' });
    expect(result.current.backendLogs[0].source).toBe('stderr');
    expect(result.current.logFilter).toBe('all');
    delete (window as any).__TAURI_INTERNALS__;
  });
});