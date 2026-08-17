import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render } from '@testing-library/react';
import { useScrollMemory } from './useScrollMemory';

// The hook uses requestAnimationFrame to restore scroll — shim it to fire synchronously.
let onFrame: (() => void) | null = null;
beforeEach(() => {
  sessionStorage.clear();
  (globalThis as any).requestAnimationFrame = (cb: () => void) => {
    onFrame = cb;
    return 1;
  };
});
afterEach(() => {
  onFrame = null;
});

function Harness() {
  const ref = useScrollMemory('test-scroll') as React.RefObject<HTMLDivElement>;
  return (
    <div ref={ref} id="box">
      Box
    </div>
  );
}

describe('useScrollMemory', () => {
  it('restores the saved scroll position for the key', () => {
    sessionStorage.setItem('scroll-memory:test-scroll', '42');
    render(<Harness />);
    const box = document.getElementById('box') as HTMLElement;
    // requestAnimationFrame fires -> scrollTop restored
    onFrame?.();
    expect(box.scrollTop).toBe(42);
  });

  it('saves the scroll position on unmount', () => {
    const { unmount } = render(<Harness />);
    const box = document.getElementById('box') as HTMLElement;
    // jsdom elements have scrollTop read-only; read it back after marking
    Object.defineProperty(box, 'scrollTop', { writable: true, value: 77 });
    unmount();
    expect(sessionStorage.getItem('scroll-memory:test-scroll')).toBe('77');
  });

  it('does nothing when there is no stored position', () => {
    render(<Harness />);
    const box = document.getElementById('box') as HTMLElement;
    box.scrollTop = 0;
    onFrame?.();
    expect(box.scrollTop).toBe(0);
  });
});