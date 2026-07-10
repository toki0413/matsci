import { useEffect, useRef } from 'react';

/**
 * Remembers scroll position for a scrollable element.
 * When the component unmounts, the scroll position is saved.
 * On remount, it's restored after a short delay.
 *
 * Usage:
 *   const ref = useScrollMemory('chat-scroll');
 *   <div ref={ref} className="overflow-y-auto">...</div>
 */
export function useScrollMemory(key: string) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    // Restore saved position
    const saved = sessionStorage.getItem(`scroll-memory:${key}`);
    if (saved) {
      requestAnimationFrame(() => {
        el.scrollTop = parseInt(saved, 10);
      });
    }

    // Save on unmount
    return () => {
      sessionStorage.setItem(`scroll-memory:${key}`, String(el.scrollTop));
    };
  }, [key]);

  return ref;
}
