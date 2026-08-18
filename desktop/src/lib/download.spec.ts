import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { downloadBlob, downloadJson, downloadText } from './download';

describe('download helpers', () => {
  let revokeCalls: string[] = [];

  beforeEach(() => {
    revokeCalls = [];
    vi.spyOn(document.body, 'appendChild').mockImplementation(
      (node) => node as HTMLElement,
    );
    vi.spyOn(document.body, 'removeChild').mockImplementation(
      (node) => node as HTMLElement,
    );
    // createObjectURL returns a fake url; record when revoke is deferred.
    URL.createObjectURL = vi.fn((_b: Blob) => `blob:fake-${revokeCalls.length}`);
    URL.revokeObjectURL = vi.fn((u: string) => {
      revokeCalls.push(u);
    });
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  function captureClick(): { filename: string; callCount: () => number } {
    let clicked = 0;
    let filename = '';
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
      clicked += 1;
      filename = this.download;
    });
    return {
      get filename() {
        return filename;
      },
      callCount: () => clicked,
    };
  }

  it('downloadBlob triggers a click with the given filename, then revokes async', () => {
    const tracker = captureClick();
    downloadBlob(new Blob(['x'], { type: 'text/plain' }), 'a.txt');

    expect(tracker.callCount()).toBe(1);
    expect(tracker.filename).toBe('a.txt');

    // revoke deferred to a macrotask, not synchronous
    expect(revokeCalls).toHaveLength(0);
    vi.runAllTimers();
    expect(revokeCalls).toEqual(['blob:fake-0']);
  });

  it('downloadJson serializes data before download', () => {
    const tracker = captureClick();
    downloadJson({ a: 1 }, 'd.json');
    expect(tracker.filename).toBe('d.json');
    vi.runAllTimers();
  });

  it('downloadText sets the requested mime type', () => {
    const tracker = captureClick();
    downloadText('# hi', 'note.md', 'text/markdown');
    expect(tracker.filename).toBe('note.md');
    vi.runAllTimers();
  });
});