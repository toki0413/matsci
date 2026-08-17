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

const { usePlugins } = await import('./usePlugins');

beforeEach(() => {
  apiGet.mockReset();
  apiPost.mockReset();
});

describe('usePlugins', () => {
  it('loads MCP servers', async () => {
    apiGet.mockResolvedValue({ servers: [{ name: 'a' }] });
    const { result } = renderHook(() => usePlugins());
    await act(async () => {
      await result.current.loadMcp();
    });
    expect(apiGet).toHaveBeenCalledWith('/mcp/servers');
    expect(result.current.mcpServers).toEqual([{ name: 'a' }]);
  });

  it('discovers servers', async () => {
    apiGet.mockResolvedValue({ servers: [{ name: 'filesystem' }] });
    const { result } = renderHook(() => usePlugins());
    await act(async () => {
      await result.current.discoverMcp();
    });
    expect(apiGet).toHaveBeenCalledWith('/mcp/servers/discover');
    expect(result.current.discoveredServers).toEqual([{ name: 'filesystem' }]);
  });

  it('connects to a MCP server and reloads the server list', async () => {
    apiPost.mockResolvedValue({ success: true, tools: [1, 2] });
    apiGet.mockResolvedValue({ servers: [] });
    const { result } = renderHook(() => usePlugins());
    await act(async () => {
      await result.current.connectMcp({ name: 'python', command: 'python', args: [] });
    });
    expect(apiPost).toHaveBeenCalledWith('/mcp/servers/connect', { name: 'python', command: 'python', args: [] });
    expect(result.current.mcpMsg).toContain('2 tools');
    expect(apiGet).toHaveBeenCalledWith('/mcp/servers');
  });

  it('disconnects a MCP server by name', async () => {
    apiPost.mockResolvedValue({ success: true });
    apiGet.mockResolvedValue({ servers: [] });
    const { result } = renderHook(() => usePlugins());
    await act(async () => {
      await result.current.disconnectMcp('python');
    });
    expect(apiPost).toHaveBeenCalledWith('/mcp/servers/python/disconnect');
    expect(result.current.mcpMsg).toContain('Disconnected python');
  });

  it('reconnects and reports failure surfaced from backend', async () => {
    apiPost.mockResolvedValue({ success: false, error: 'gone' });
    const { result } = renderHook(() => usePlugins());
    await act(async () => {
      await result.current.reconnectMcp('db');
    });
    expect(result.current.mcpMsg).toContain('Reconnect failed');
  });

  it('calls a MCP tool and returns backend result', async () => {
    apiPost.mockResolvedValue({ result: 'ok' });
    const { result } = renderHook(() => usePlugins());
    let out: unknown;
    await act(async () => {
      out = await result.current.callMcpTool('srv', 'list', { path: '/' });
    });
    expect(apiPost).toHaveBeenCalledWith('/mcp/tools/srv/call', { tool_name: 'list', arguments: { path: '/' } });
    expect(result.current.mcpMsg).toContain('completed');
    expect((out as { result: string }).result).toBe('ok');
  });
});