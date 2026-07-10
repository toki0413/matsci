import { useEffect, type RefObject } from 'react';

/**
 * Traps keyboard focus inside a container element while it's active.
 * - On mount: moves focus to the first focusable element inside the container.
 * - Tab/Shift+Tab cycles through focusable elements within the container.
 * - On unmount: focus returns to the previously focused element (the trigger).
 */
export function useFocusTrap(
  containerRef: RefObject<HTMLElement | null>,
  active: boolean,
): void {
  useEffect(() => {
    if (!active || !containerRef.current) return;

    const container = containerRef.current;
    const previouslyFocused = document.activeElement as HTMLElement | null;

    // Focusable element selectors
    const selector = [
      'a[href]',
      'button:not([disabled])',
      'input:not([disabled])',
      'textarea:not([disabled])',
      'select:not([disabled])',
      '[tabindex]:not([tabindex="-1"])',
    ].join(', ');

    // Move focus into the container
    const focusables = () =>
      Array.from(container.querySelectorAll<HTMLElement>(selector)).filter(
        (el) => el.offsetParent !== null, // visible only
      );

    // Initial focus
    const items = focusables();
    if (items.length > 0) {
      items[0].focus();
    } else {
      container.setAttribute('tabindex', '-1');
      container.focus();
    }

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Tab') return;
      const currentItems = focusables();
      if (currentItems.length === 0) {
        e.preventDefault();
        return;
      }
      const first = currentItems[0];
      const last = currentItems[currentItems.length - 1];
      const activeEl = document.activeElement;

      if (e.shiftKey) {
        // Shift+Tab: if on first, wrap to last
        if (activeEl === first || !container.contains(activeEl)) {
          e.preventDefault();
          last.focus();
        }
      } else {
        // Tab: if on last, wrap to first
        if (activeEl === last || !container.contains(activeEl)) {
          e.preventDefault();
          first.focus();
        }
      }
    };

    container.addEventListener('keydown', handleKeyDown);

    return () => {
      container.removeEventListener('keydown', handleKeyDown);
      // Restore focus to the trigger element
      previouslyFocused?.focus();
    };
  }, [active, containerRef]);
}
