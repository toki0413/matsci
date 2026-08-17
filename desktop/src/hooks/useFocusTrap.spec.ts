import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useFocusTrap } from './useFocusTrap';

function mountContainer(html: string) {
  const container = document.createElement('div');
  container.innerHTML = html;
  // jsdom always reports offsetParent === null, so the hook's "visible only"
  // filter would drop every element. Stub a truthy offsetParent on each child
  // to simulate a laid-out (non-hidden) tree.
  container.querySelectorAll('*').forEach((el) => {
    Object.defineProperty(el, 'offsetParent', { value: container, configurable: true });
  });
  document.body.appendChild(container);
  return container;
}

beforeEach(() => {
  document.body.innerHTML = '';
});

describe('useFocusTrap', () => {
  it('moves focus to the first focusable element when active', () => {
    const container = mountContainer(
      '<input class="a" /><button>One</button><button>Two</button>',
    );
    const { result, unmount } = renderHook(() => useFocusTrap({ current: container } as any, true));
    expect(document.activeElement).toBe(container.querySelector('input'));
    unmount();
  });

  it('focuses the container itself when empty of focusables', () => {
    const container = mountContainer('<span>nothing focusable</span>');
    renderHook(() => useFocusTrap({ current: container } as any, true));
    expect(container.getAttribute('tabindex')).toBe('-1');
  });

  it('does nothing when inactive', () => {
    const container = mountContainer('<button>One</button>');
    renderHook(() => useFocusTrap({ current: container } as any, false));
    expect(container.getAttribute('tabindex')).toBeNull();
  });

  it('does nothing when the ref has no element', () => {
    expect(() =>
      renderHook(() => useFocusTrap({ current: null } as any, true)),
    ).not.toThrow();
  });

  it('wraps focus backward (Shift+Tab) from the first element to the last', () => {
    const container = mountContainer(
      '<button id="a">A</button><button id="b">B</button><button id="c">C</button>',
    );
    renderHook(() => useFocusTrap({ current: container } as any, true));
    const a = container.querySelector('#a') as HTMLElement;
    const c = container.querySelector('#c') as HTMLElement;
    (a as any).focus = vi.fn();
    (c as any).focus = vi.fn();

    // force activeElement to the first button, then send Shift+Tab
    Object.defineProperty(document, 'activeElement', { value: a, configurable: true });
    const ev = new KeyboardEvent('keydown', { key: 'Tab', shiftKey: true, bubbles: true });
    container.dispatchEvent(ev);
    expect(c.focus).toHaveBeenCalled();
    expect(a.focus).not.toHaveBeenCalled();
  });

  it('wraps focus forward (Tab) from the last element to the first', () => {
    const container = mountContainer(
      '<button id="a">A</button><button id="b">B</button><button id="c">C</button>',
    );
    renderHook(() => useFocusTrap({ current: container } as any, true));
    const a = container.querySelector('#a') as HTMLElement;
    const c = container.querySelector('#c') as HTMLElement;
    (a as any).focus = vi.fn();
    (c as any).focus = vi.fn();

    Object.defineProperty(document, 'activeElement', { value: c, configurable: true });
    const ev = new KeyboardEvent('keydown', { key: 'Tab', shiftKey: false, bubbles: true });
    container.dispatchEvent(ev);
    expect(a.focus).toHaveBeenCalled();
  });

  it('restores focus to the previously focused element on unmount', () => {
    const trigger = document.createElement('button');
    trigger.id = 'trigger';
    document.body.appendChild(trigger);
    (trigger as any).focus = vi.fn();
    const container = mountContainer('<button>One</button>');

    Object.defineProperty(document, 'activeElement', { value: trigger, configurable: true });
    const { result, unmount } = renderHook(() => useFocusTrap({ current: container } as any, true));
    // trap moved focus away from the trigger
    expect((trigger as any).focus).not.toHaveBeenCalled();
    unmount();
    expect(trigger.focus).toHaveBeenCalled();
  });
});