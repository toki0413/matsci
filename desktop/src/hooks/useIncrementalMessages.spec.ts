import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';

const apiGet = vi.fn();

vi.mock('../lib/api', () => ({
  api: {
    get: (...a: unknown[]) => apiGet(...a),
    post: vi.fn(),
    put: vi.fn(),
    patch: vi.fn(),
    del: vi.fn(),
  },
}));

const { useIncrementalMessages, blocksToMessages, compactionBlocks } = await import('./useIncrementalMessages');

const text = { kind: 'text' as const, text: 'hello', frozen: true, rev: 0, seq: 1 };
const tool = { kind: 'tool' as const, text: 'ran tool', frozen: false, rev: 0, seq: 2 };
const compact = { kind: 'compaction' as const, text: '...', frozen: true, rev: 1, seq: 3 };

beforeEach(() => apiGet.mockReset());

describe('blocksToMessages (pure)', () => {
  it('turns text/tool blocks into assistant messages', () => {
    const msgs = blocksToMessages([text, tool]);
    expect(msgs).toHaveLength(2);
    expect(msgs[0]).toMatchObject({ role: 'assistant', content: 'hello' });
    expect(msgs[1]).toMatchObject({ role: 'assistant', content: 'ran tool' });
  });

  it('turns compaction blocks into compacted dividers', () => {
    const [d] = blocksToMessages([compact]);
    expect(d.isCompacted).toBe(true);
    expect(d.content).toBe('');
  });

  it('compactionBlocks filters only compaction kind', () => {
    const out = compactionBlocks([text, compact, tool]);
    expect(out).toEqual([compact]);
  });

  it('uses empty content when a text block has no text', () => {
    const [m] = blocksToMessages([{ ...text, text: '' }]);
    expect(m.content).toBe('');
  });
});

describe('useIncrementalMessages', () => {
  it('fetches events and advances the per-thread cursor', async () => {
    apiGet.mockResolvedValue({ thread_id: 't1', blocks: [text], next_seq: 9, leaf_id: null });
    const { result } = renderHook(() => useIncrementalMessages());
    await act(async () => {
      const res = await result.current.fetchEvents('t1');
      expect(res.next_seq).toBe(9);
      expect(res.blocks).toEqual([text]);
    });
    expect(apiGet).toHaveBeenCalledWith('/threads/t1/events');
    expect(result.current.loading.t1).toBe(false);
  });

  it('passes the after cursor on incremental fetches', async () => {
    apiGet.mockResolvedValueOnce({ thread_id: 't1', blocks: [text], next_seq: 1, leaf_id: null });
    apiGet.mockResolvedValueOnce({ thread_id: 't1', blocks: [tool], next_seq: 2, leaf_id: null });
    const { result } = renderHook(() => useIncrementalMessages());
    await act(async () => result.current.fetchEvents('t1'));
    await act(async () => result.current.fetchIncremental('t1'));
    expect(apiGet).toHaveBeenLastCalledWith('/threads/t1/events?after=1');
  });

  it('hydrates the full block model via fetchThreadBlocks', async () => {
    apiGet.mockResolvedValue({ thread_id: 't1', blocks: [text, compact], next_seq: 5, leaf_id: null });
    const { result } = renderHook(() => useIncrementalMessages());
    await act(async () => {
      const blocks = await result.current.fetchThreadBlocks('t1');
      expect(blocks).toHaveLength(2);
    });
  });
});