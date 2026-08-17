import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';

// ---- mock the api module so no real network happens ---------------------
const apiGet = vi.fn();
const apiPost = vi.fn();
const apiDel = vi.fn();
const apiUpload = vi.fn();

vi.mock('../lib/api', () => ({
  api: {
    get: (...a: unknown[]) => apiGet(...a),
    post: (...a: unknown[]) => apiPost(...a),
    put: vi.fn(),
    patch: vi.fn(),
    del: (...a: unknown[]) => apiDel(...a),
    uploadWithProgress: (...a: unknown[]) => apiUpload(...a),
  },
}));

const { useKnowledge } = await import('./useKnowledge');

const fakeFile = new File(['x'], 'paper.pdf', { type: 'application/pdf' });

beforeEach(() => {
  apiGet.mockReset();
  apiPost.mockReset();
  apiDel.mockReset();
  apiUpload.mockReset();
  vi.useFakeTimers();
});

afterEach(() => {
  vi.runOnlyPendingTimers();
  vi.useRealTimers();
});

describe('useKnowledge', () => {
  it('loads documents and availability from /knowledge', async () => {
    apiGet.mockResolvedValue({ documents: [{ id: '1' }, { id: '2' }], available: true });
    const { result } = renderHook(() => useKnowledge());
    await act(async () => {
      await result.current.loadKnowledge();
    });
    expect(result.current.kbDocs).toHaveLength(2);
    expect(result.current.kbAvailable).toBe(true);
    expect(result.current.kbLoading).toBe(false);
  });

  it('reports an error message when the knowledge load fails', async () => {
    apiGet.mockRejectedValue(new Error('boom'));
    const { result } = renderHook(() => useKnowledge());
    await act(async () => {
      await result.current.loadKnowledge();
    });
    expect(result.current.kbMsg).toContain('boom');
  });

  it('uploads a file and refreshes the docs list on success', async () => {
    apiUpload.mockResolvedValue({ success: true, document: { chunks: 7, name: 'paper.pdf' } });
    apiGet.mockResolvedValue({ documents: [], available: true });
    const { result } = renderHook(() => useKnowledge());
    await act(async () => {
      await result.current.uploadKnowledge(fakeFile);
    });
    expect(apiUpload).toHaveBeenCalledWith(
      '/knowledge/upload',
      fakeFile,
      expect.any(Function),
    );
    expect(result.current.kbMsg).toContain('7 chunks');
    expect(result.current.uploadPct).toBe(100); // reaches 100 on completion
    // the 2s timer then resets progress back to 0
    await act(async () => {
      vi.advanceTimersByTime(2000);
    });
    expect(result.current.uploadPct).toBe(0);
  });

  it('reports an upload failure message', async () => {
    apiUpload.mockResolvedValue({ success: false, error: 'rejected' });
    const { result } = renderHook(() => useKnowledge());
    await act(async () => {
      await result.current.uploadKnowledge(fakeFile);
    });
    expect(result.current.kbMsg).toContain('rejected');
  });

  it('parses a document and reports graph node counts', async () => {
    apiUpload.mockResolvedValue({
      info_packages: 3,
      graph: { nodes: [1, 2, 3], edges: [1] },
    });
    apiGet.mockResolvedValue({ documents: [], available: true });
    const { result } = renderHook(() => useKnowledge());
    await act(async () => {
      await result.current.parseDocument(fakeFile);
    });
    expect(apiUpload).toHaveBeenCalledWith('/document/parse', fakeFile, expect.any(Function));
    expect(result.current.parseLoading).toBe(false);
    expect(result.current.kbMsg).toContain('3 info packages');
    expect(result.current.kbMsg).toContain('3 graph nodes');
  });

  it('queries knowledge and stores returned chunks', async () => {
    apiPost.mockResolvedValue({ chunks: [{ id: 'a' }, { id: 'b' }] });
    const { result } = renderHook(() => useKnowledge());
    await act(async () => result.current.setKbQuery('C-S-H gel'));
    await act(async () => {
      await result.current.queryKnowledge();
    });
    expect(apiPost).toHaveBeenCalledWith('/knowledge/query', { query: 'C-S-H gel', top_k: 5 });
    expect(result.current.kbChunks).toHaveLength(2);
    expect(result.current.kbMsg).toContain('Found 2 chunks');
  });

  it('skips the query when the query is blank', async () => {
    const { result } = renderHook(() => useKnowledge());
    await act(async () => {
      await result.current.queryKnowledge();
    });
    expect(apiPost).not.toHaveBeenCalled();
  });

  it('deletes a document then reloads', async () => {
    apiDel.mockResolvedValue({});
    apiGet.mockResolvedValue({ documents: [], available: true });
    const { result } = renderHook(() => useKnowledge());
    await act(async () => {
      await result.current.deleteKnowledge('doc-1');
    });
    expect(apiDel).toHaveBeenCalledWith('/knowledge/doc-1');
    expect(apiGet).toHaveBeenCalledWith('/knowledge');
  });

  it('ingests a URL and refreshes the docs', async () => {
    apiPost.mockResolvedValue({ success: true, source_url: 'https://x.dev' });
    apiGet.mockResolvedValue({ documents: [], available: true });
    const { result } = renderHook(() => useKnowledge());
    await act(async () => {
      await result.current.ingestUrl('https://x.dev');
    });
    expect(apiPost).toHaveBeenCalledWith('/knowledge/ingest-url', { url: 'https://x.dev' });
    expect(result.current.kbMsg).toContain('Added https://x.dev');
  });

  it('loads a provenance DAG and falls back to empty on error', async () => {
    const { result } = renderHook(() => useKnowledge());
    apiGet.mockResolvedValueOnce({ success: true, data: { nodes: [1], edges: [1] } });
    await act(async () => {
      const dag = await result.current.loadProvenanceDag();
      expect(dag.success).toBe(true);
    });
    apiGet.mockRejectedValueOnce(new Error('net'));
    await act(async () => {
      const dag = await result.current.loadProvenanceDag();
      expect(dag).toEqual({ success: false, data: { nodes: [], edges: [] } });
    });
  });
});