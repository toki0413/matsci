import { describe, it, expect, beforeEach, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useTheme } from './useTheme';

function matchMediaMock(dark: boolean) {
  const listeners: Array<() => void> = [];
  window.matchMedia = vi.fn().mockImplementation((query: string) => ({
    matches: query.includes('dark') ? dark : !dark,
    media: query,
    addEventListener: (_: string, cb: () => void) => listeners.push(cb),
    removeEventListener: () => {},
    addListener: (_: () => void) => {},
    removeListener: () => {},
    dispatchEvent: () => false,
    onchange: null,
  }));
  return listeners;
}

beforeEach(() => {
  localStorage.clear();
  document.documentElement.classList.remove('dark');
});

describe('useTheme', () => {
  it('defaults to auto when nothing is saved', () => {
    matchMediaMock(false);
    const { result } = renderHook(() => useTheme());
    expect(result.current.theme).toBe('auto');
    // auto + light system -> not dark
    expect(result.current.isDark).toBe(false);
  });

  it('restores a saved "dark" theme from localStorage', () => {
    localStorage.setItem('huginn:theme', 'dark');
    matchMediaMock(false);
    const { result } = renderHook(() => useTheme());
    expect(result.current.theme).toBe('dark');
    expect(result.current.isDark).toBe(true);
  });

  it('ignores invalid stored values and falls back to auto', () => {
    localStorage.setItem('huginn:theme', 'neon');
    matchMediaMock(false);
    const { result } = renderHook(() => useTheme());
    expect(result.current.theme).toBe('auto');
  });

  it('applies .dark to <html> when theme is dark, and persists it', () => {
    matchMediaMock(true);
    const { result } = renderHook(() => useTheme());
    act(() => result.current.setTheme('dark'));
    expect(document.documentElement.classList.contains('dark')).toBe(true);
    expect(localStorage.getItem('huginn:theme')).toBe('dark');
  });

  it('resolves auto against the system dark preference', () => {
    matchMediaMock(true); // system prefers dark
    const { result } = renderHook(() => useTheme());
    expect(result.current.theme).toBe('auto');
    expect(result.current.isDark).toBe(true);
  });

  it('toggleTheme cycles light -> dark -> auto -> light', () => {
    matchMediaMock(false);
    const { result } = renderHook(() => useTheme());
    act(() => result.current.setTheme('light'));
    act(() => result.current.toggleTheme());
    expect(result.current.theme).toBe('dark');
    act(() => result.current.toggleTheme());
    expect(result.current.theme).toBe('auto');
    act(() => result.current.toggleTheme());
    expect(result.current.theme).toBe('light');
  });
});