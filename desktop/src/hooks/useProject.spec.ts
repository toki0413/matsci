import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';

const apiGet = vi.fn();
const apiPost = vi.fn();

vi.mock('../lib/api', () => ({
  api: {
    get: (...a: unknown[]) => apiGet(...a),
    post: (...a: unknown[]) => apiPost(...a),
    put: vi.fn(),
    patch: vi.fn(),
    del: vi.fn(),
  },
}));

const { useProject } = await import('./useProject');

beforeEach(() => {
  apiGet.mockReset();
  apiPost.mockReset();
});

describe('useProject', () => {
  it('loads project context and marks the source', async () => {
    apiGet.mockResolvedValue({ content: '# 项目', source: 'auto' });
    const { result } = renderHook(() => useProject());
    await act(async () => {
      await result.current.loadProjectContext();
    });
    expect(result.current.projectContext).toBe('# 项目');
    expect(result.current.projectContextSource).toBe('auto');
  });

  it('saves project context on success', async () => {
    apiPost.mockResolvedValue({ success: true });
    const { result } = renderHook(() => useProject());
    await act(async () => result.current.setProjectContext('新的上下文'));
    await act(async () => {
      await result.current.saveProjectContext();
    });
    expect(apiPost).toHaveBeenCalledWith('/project-context', { content: '新的上下文' });
    expect(result.current.projectContextMsg).toContain('Saved');
  });

  it('surfaces a save error from the backend', async () => {
    apiPost.mockResolvedValue({ success: false, error: 'too big' });
    const { result } = renderHook(() => useProject());
    await act(async () => result.current.setProjectContext('x'));
    await act(async () => {
      await result.current.saveProjectContext();
    });
    expect(result.current.projectContextMsg).toContain('too big');
  });

  it('loads codebase indexing status', async () => {
    apiGet.mockResolvedValue({ status: 'ready', files: 10 });
    const { result } = renderHook(() => useProject());
    await act(async () => {
      await result.current.loadCodebaseStatus();
    });
    expect(result.current.codebaseStatus).toEqual({ status: 'ready', files: 10 });
  });

  it('indexes the codebase and refreshes status', async () => {
    apiPost.mockResolvedValue({ success: true, indexed_files: 42, chunks: 500 });
    apiGet.mockResolvedValue({ status: 'ready' });
    const { result } = renderHook(() => useProject());
    await act(async () => {
      await result.current.indexCodebase();
    });
    expect(apiPost).toHaveBeenCalledWith('/codebase/index');
    expect(result.current.codebaseMsg).toContain('Indexed 42 files, 500 chunks');
    expect(apiGet).toHaveBeenCalledWith('/codebase');
  });

  it('searches the codebase and stores results', async () => {
    apiPost.mockResolvedValue({ results: [{ path: 'a.py' }] });
    const { result } = renderHook(() => useProject());
    await act(async () => result.current.setCodebaseQuery('retry'));
    await act(async () => {
      await result.current.searchCodebase();
    });
    expect(apiPost).toHaveBeenCalledWith('/codebase/search', { query: 'retry', top_k: 8 });
    expect(result.current.codebaseResults).toEqual([{ path: 'a.py' }]);
  });

  it('skips search when the query is blank', async () => {
    const { result } = renderHook(() => useProject());
    await act(async () => {
      await result.current.searchCodebase();
    });
    expect(apiPost).not.toHaveBeenCalled();
  });
});