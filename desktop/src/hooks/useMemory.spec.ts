import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';

const apiGet = vi.fn();
const apiPost = vi.fn();
const apiDel = vi.fn();
const apiPatch = vi.fn();
const toastSuccess = vi.fn();
const toastError = vi.fn();

vi.mock('../lib/api', () => ({
  api: {
    get: (...a: unknown[]) => apiGet(...a),
    post: (...a: unknown[]) => apiPost(...a),
    put: vi.fn(),
    patch: (...a: unknown[]) => apiPatch(...a),
    del: (...a: unknown[]) => apiDel(...a),
  },
}));

vi.mock('../components/Toast', () => ({
  toast: { success: (...a: unknown[]) => toastSuccess(...a), error: (...a: unknown[]) => toastError(...a) },
}));

const { useMemory } = await import('./useMemory');

const entries = [{ id: 'm1', content: '记住这个结果', category: 'fact' }];

beforeEach(() => {
  apiGet.mockReset();
  apiPost.mockReset();
  apiDel.mockReset();
  apiPatch.mockReset();
  toastSuccess.mockReset();
  toastError.mockReset();
  vi.stubGlobal('confirm', vi.fn(() => true));
});

describe('useMemory', () => {
  it('loads the first page of memories and tracks hasMore', async () => {
    apiGet.mockResolvedValue({ entries, total: 350 });
    const { result } = renderHook(() => useMemory());
    await act(async () => {
      await result.current.loadMemory();
    });
    expect(apiGet).toHaveBeenCalledWith('/memory?limit=100');
    expect(result.current.memories).toEqual(entries);
    expect(result.current.memoryHasMore).toBe(true);
    expect(result.current.memoriesLoading).toBe(false);
  });

  it('appends new entries on loadMore without duplicating', async () => {
    apiGet.mockResolvedValue({ entries: [{ id: 'a' }, { id: 'b' }], total: 200 });
    const { result } = renderHook(() => useMemory());
    await act(async () => result.current.loadMemory());
    apiGet.mockResolvedValue({ entries: [{ id: 'a' }, { id: 'b' }, { id: 'c' }], total: 200 });
    await act(async () => result.current.loadMemory(true));
    expect(result.current.memories.map((m) => m.id)).toEqual(['a', 'b', 'c']);
  });

  it('loads memory stats', async () => {
    apiGet.mockResolvedValue({ count: 5, layers: 3 });
    const { result } = renderHook(() => useMemory());
    await act(async () => {
      await result.current.loadMemoryStats();
    });
    expect(result.current.memoryStats).toEqual({ count: 5, layers: 3 });
  });

  it('searches memories via /memory/search', async () => {
    apiPost.mockResolvedValue({ results: [{ id: 'hit' }] });
    const { result } = renderHook(() => useMemory());
    await act(async () => result.current.setMemorySearch('gel'));
    await act(async () => {
      await result.current.searchMemory();
    });
    expect(apiPost).toHaveBeenCalledWith('/memory/search', { query: 'gel', top_k: 10 });
    expect(result.current.memories).toEqual([{ id: 'hit' }]);
    expect(result.current.memoryMsg).toContain('Found 1 results');
  });

  it('creates a memory: splits tags and resets the form', async () => {
    apiPost.mockResolvedValue({ success: true });
    apiGet.mockResolvedValue({ entries, total: 1 });
    const { result } = renderHook(() => useMemory());
    await act(async () => result.current.setMemoryForm({ content: '新记忆', category: 'fact', tags: 'a, b,', importance: 0.9, tier: 'high' }));
    await act(async () => {
      await result.current.createMemory();
    });
    expect(apiPost).toHaveBeenCalledWith('/memory', expect.objectContaining({ content: '新记忆', tags: ['a', 'b'], importance: 0.9 }));
    expect(toastSuccess).toHaveBeenCalledWith('Memory saved');
    expect(result.current.memoryForm.content).toBe(''); // form reset
  });

  it('reports a save failure from the backend', async () => {
    apiPost.mockResolvedValue({ success: false, error: 'invalid' });
    const { result } = renderHook(() => useMemory());
    await act(async () => result.current.setMemoryForm({ content: 'x', category: 'fact', tags: '', importance: 0.5, tier: 'mid' }));
    await act(async () => {
      await result.current.createMemory();
    });
    expect(result.current.memoryMsg).toContain('invalid');
    expect(toastError).toHaveBeenCalled();
  });

  it('deletes only after confirm() returns true', async () => {
    vi.stubGlobal('confirm', vi.fn(() => false));
    apiDel.mockResolvedValue({});
    const { result } = renderHook(() => useMemory());
    await act(async () => {
      await result.current.deleteMemory('m1');
    });
    expect(apiDel).not.toHaveBeenCalled();
  });

  it('updates a memory via PATCH', async () => {
    apiPatch.mockResolvedValue({ success: true });
    apiGet.mockResolvedValue({ entries, total: 1 });
    const { result } = renderHook(() => useMemory());
    await act(async () => {
      await result.current.updateMemory('m1', { importance: 0.8 });
    });
    expect(apiPatch).toHaveBeenCalledWith('/memory/m1', { importance: 0.8 });
    expect(toastSuccess).toHaveBeenCalledWith('Memory updated');
  });

  it('promotes a memory to long-term', async () => {
    apiPost.mockResolvedValue({ success: true });
    apiGet.mockResolvedValue({ entries, total: 1 });
    const { result } = renderHook(() => useMemory());
    await act(async () => {
      await result.current.promoteMemory('m1');
    });
    expect(apiPost).toHaveBeenCalledWith('/memory/promote/m1');
    expect(toastSuccess).toHaveBeenCalledWith('Promoted to long-term');
  });

  it('prunes expired and low-importance memories', async () => {
    apiPost.mockResolvedValue({ expired: 2, low_importance: 1 });
    apiGet.mockResolvedValue({ entries, total: 1 });
    const { result } = renderHook(() => useMemory());
    await act(async () => {
      await result.current.pruneMemory();
    });
    expect(result.current.memoryMsg).toContain('Pruned 2 expired, 1 low-importance.');
  });

  it('loads the four memory layers', async () => {
    apiGet.mockResolvedValue({ working: [], longterm: [], reflective: [], episodic: [] });
    const { result } = renderHook(() => useMemory());
    await act(async () => {
      await result.current.loadMemoryLayers();
    });
    expect(apiGet).toHaveBeenCalledWith('/memory/layers');
    expect(result.current.memoryLayers).toEqual({ working: [], longterm: [], reflective: [], episodic: [] });
    expect(result.current.memoryLayersLoading).toBe(false);
  });
});