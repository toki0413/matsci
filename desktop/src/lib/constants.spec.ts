import { describe, it, expect, vi, afterEach } from 'vitest';
import { formatTimeAgo } from './constants';

// formatTimeAgo is relative to Date.now(), so pin the clock to a fixed "now".
const NOW = Date.UTC(2026, 0, 15, 12, 0, 0); // 2026-01-15 12:00 UTC

afterEach(() => {
  vi.useRealTimers();
});

function at(now: number) {
  vi.useFakeTimers();
  vi.setSystemTime(new Date(now));
}

describe('formatTimeAgo', () => {
  it('labels anything under a minute as "just now"', () => {
    at(NOW);
    expect(formatTimeAgo(new Date(NOW - 30_000).toISOString())).toBe('just now');
    expect(formatTimeAgo(new Date(NOW - 59_000).toISOString())).toBe('just now');
  });

  it('formats minutes like "3m ago"', () => {
    at(NOW);
    expect(formatTimeAgo(new Date(NOW - 3 * 60_000).toISOString())).toBe('3m ago');
  });

  it('formats hours like "2h ago"', () => {
    at(NOW);
    expect(formatTimeAgo(new Date(NOW - 2 * 3_600_000).toISOString())).toBe('2h ago');
  });

  it('formats days like "1d ago" and separates from hours', () => {
    at(NOW);
    expect(formatTimeAgo(new Date(NOW - 1 * 86_400_000).toISOString())).toBe('1d ago');
  });

  it('crosses unit boundaries correctly (89min -> 1h, 23h50m -> 23h)', () => {
    at(NOW);
    expect(formatTimeAgo(new Date(NOW - 89 * 60_000).toISOString())).toBe('1h ago');
    // just under a day stays in hours; 60*24=1440s floor is day
    expect(formatTimeAgo(new Date(NOW - (24 * 3_600_000 - 60_000)).toISOString())).toBe('23h ago');
  });
});