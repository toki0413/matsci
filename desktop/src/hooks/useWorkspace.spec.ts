import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';

// ---- mock api -----
const apiGet = vi.fn();
const apiPut = vi.fn();
vi.mock('../lib/api', () => ({
  api: {
    get: (...a: unknown[]) => apiGet(...a),
    put: (...a: unknown[]) => apiPut(...a),
    post: vi.fn(),
    patch: vi.fn(),
    del: vi.fn(),
  },
}));

// ---- mock tauri event listen so it's a no-op outside a Tauri shell -----
vi.mock('@tauri-apps/api/event', () => ({
  listen: vi.fn(async () => () => {}),
}));

const { useWorkspace } = await import('./useWorkspace');

beforeEach(() => {
  apiGet.mockReset();
  apiPut.mockReset();
  // the hook's mount effect calls /v1/fs/cwd (and reads it); when a test sets
  // apiGet to reject globally, that produces expected-but-noisy console.error
  // logs. Silence them so a passing run stays clean.
  vi.spyOn(console, 'error').mockImplementation(() => {});
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('useWorkspace', () => {
  it('loads the cwd and root directory listing on mount', async () => {
    apiGet.mockImplementation((url: string) => {
      if (url === '/v1/fs/cwd') return Promise.resolve({ path: '/root' });
      if (url === '/v1/fs/list') return Promise.resolve({ entries: [{ name: 'a.py', path: '/root/a.py', is_dir: false }] });
      return Promise.resolve({});
    });
    const { result } = renderHook(() => useWorkspace());
    await waitFor(() => expect(result.current.cwd).toBe('/root'));
    expect(result.current.dirCache['/root']).toEqual([{ name: 'a.py', path: '/root/a.py', is_dir: false }]);
    expect(result.current.expandedDirs.has('/root')).toBe(true);
  });

  it('toggles a directory open (loads it) and closed (collapses)', async () => {
    apiGet.mockResolvedValue({ entries: [] });
    const { result } = renderHook(() => useWorkspace());

    act(() => result.current.toggleDir('/root/src'));
    // toggleDir triggers state update + loadDir; give the async one a tick
    await act(async () => {});
    expect(result.current.expandedDirs.has('/root/src')).toBe(true);

    act(() => result.current.toggleDir('/root/src'));
    expect(result.current.expandedDirs.has('/root/src')).toBe(false);
  });

  it('openFile loads content and resets dirty state', async () => {
    apiGet.mockImplementation((url: string) => {
      if (url === '/v1/fs/read') return Promise.resolve({ content: 'print(1)' });
      return Promise.resolve({});
    });
    const { result } = renderHook(() => useWorkspace());
    // mark dirty first, then open a clean file
    act(() => result.current.setEditorDirty(true));
    await act(async () => {
      await result.current.openFile('/root/a.py');
    });
    expect(result.current.selectedFile).toBe('/root/a.py');
    expect(result.current.editorContent).toBe('print(1)');
    expect(result.current.editorDirty).toBe(false);
  });

  it('openFile sets an error message when the read fails', async () => {
    apiGet.mockRejectedValue(new Error('boom'));
    const { result } = renderHook(() => useWorkspace());
    await act(async () => {
      await result.current.openFile('/root/a.py');
    });
    expect(result.current.editorMsg).toContain('Failed to open file');
  });

  it('saveFile PUTs the current content and clears dirty', async () => {
    apiPut.mockResolvedValue({});
    const { result } = renderHook(() => useWorkspace());
    act(() => {
      result.current.setSelectedFile('/root/a.py');
      result.current.setEditorContent('updated');
      result.current.setEditorDirty(true);
    });
    await act(async () => {
      await result.current.saveFile();
    });
    expect(apiPut).toHaveBeenCalledWith('/v1/fs/write', { path: '/root/a.py', content: 'updated' });
    expect(result.current.editorDirty).toBe(false);
    expect(result.current.editorMsg).toBe('Saved.');
  });

  it('saveFile does nothing when nothing is selected', async () => {
    const { result } = renderHook(() => useWorkspace());
    await act(async () => {
      await result.current.saveFile();
    });
    expect(apiPut).not.toHaveBeenCalled();
  });
});