import { Puzzle } from 'lucide-react';
import { PanelHeader } from '../settings-shared';
import EmptyState from '../EmptyState';

interface McpTool {
  name: string;
  description: string;
}

interface McpServer {
  name: string;
  path?: string;
  command?: string;
  args?: string[];
  tools: McpTool[];
}

interface NewMcp {
  name: string;
  command: string;
  args: string;
}

interface PluginsPanelProps {
  mcpServers: McpServer[];
  discoveredServers: McpServer[];
  mcpMsg: string;
  newMcp: NewMcp;
  setNewMcp: (v: NewMcp) => void;
  loadMcp: () => void;
  discoverMcp: () => void;
  connectMcp: (srv: any) => void;
  disconnectMcp: (name: string) => void;
}

export function PluginsPanel({
  mcpServers, discoveredServers, mcpMsg, newMcp, setNewMcp,
  loadMcp, discoverMcp, connectMcp, disconnectMcp,
}: PluginsPanelProps) {
  return (
    <div className="flex h-full flex-col">
      <PanelHeader title="Plugins / MCP Servers" className="px-6">
        <button onClick={() => { loadMcp(); discoverMcp(); }} className="btn-secondary px-3 py-1.5 text-xs">
          Refresh
        </button>
      </PanelHeader>
      <div className="flex flex-1 overflow-hidden">
        <aside className="flex w-80 flex-col border-r border-border bg-bg-secondary p-4">
          <h3 className="mb-3 text-sm font-semibold">Connect manually</h3>
          <div className="space-y-3">
            <div>
              <label className="mb-1 block text-xs text-text-secondary">Name</label>
              <input
                type="text"
                value={newMcp.name}
                onChange={(e) => setNewMcp({ ...newMcp, name: e.target.value })}
                placeholder="my-server"
                className="input text-sm"
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-text-secondary">Command</label>
              <input
                type="text"
                value={newMcp.command}
                onChange={(e) => setNewMcp({ ...newMcp, command: e.target.value })}
                placeholder="python"
                className="input text-sm"
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-text-secondary">Args (space separated)</label>
              <input
                type="text"
                value={newMcp.args}
                onChange={(e) => setNewMcp({ ...newMcp, args: e.target.value })}
                placeholder="server.py"
                className="input text-sm"
              />
            </div>
            <button
              onClick={() => {
                if (!newMcp.name.trim()) return;
                const args = newMcp.args
                  .split(" ")
                  .map((s) => s.trim())
                  .filter(Boolean);
                connectMcp({ name: newMcp.name.trim(), command: newMcp.command.trim() || "python", args });
                setNewMcp({ name: "", command: "python", args: "" });
              }}
              className="btn-primary w-full text-xs"
            >
              Connect
            </button>
          </div>

          {mcpMsg && (
            <div className="mt-4 rounded-lg border border-border bg-bg-tertiary p-2 text-xs text-text-secondary">
              {mcpMsg}
            </div>
          )}

          <h3 className="mb-2 mt-6 text-sm font-semibold">Discovered local</h3>
          <div className="flex-1 overflow-y-auto space-y-2">
            {discoveredServers.length === 0 && (
              <div className="flex flex-col items-start rounded-lg border border-dashed border-border p-3">
                <p className="text-xs font-medium text-text-secondary">No local servers found</p>
                <p className="mt-1 text-[11px] leading-relaxed text-text-muted">
                  Start an MCP server on your network, or use the form above to connect manually by name and command.
                </p>
              </div>
            )}
            {discoveredServers.map((srv) => (
              <div
                key={srv.name}
                className="rounded-lg border border-border bg-bg-tertiary p-2"
              >
                <div className="text-xs font-medium text-text-primary">{srv.name}</div>
                <div className="mt-1 truncate text-[10px] text-text-muted">{srv.path}</div>
                <button
                  onClick={() => connectMcp(srv)}
                  className="mt-2 w-full rounded bg-accent px-2 py-1 text-xs text-white hover:bg-accent/90"
                >
                  Connect
                </button>
              </div>
            ))}
          </div>
        </aside>

        <div className="flex flex-1 flex-col bg-bg-primary p-4">
          <h3 className="mb-3 text-sm font-semibold">Connected servers</h3>
          <div className="flex-1 overflow-y-auto space-y-3">
            {mcpServers.length === 0 && (
              <EmptyState
                icon={Puzzle}
                title="No plugins connected"
                subtitle="Connect an MCP server from the sidebar to extend Huginn with additional tools and capabilities."
              />
            )}
            {mcpServers.map((srv) => (
              <div key={srv.name} className="rounded-xl border border-border bg-bg-secondary p-4">
                <div className="flex items-center justify-between">
                  <div className="text-sm font-semibold text-text-primary">{srv.name}</div>
                  <button
                    onClick={() => disconnectMcp(srv.name)}
                    className="btn-secondary px-2 py-1 text-xs text-error hover:bg-error/10"
                  >
                    Disconnect
                  </button>
                </div>
                <div className="mt-2 text-xs text-text-secondary">
                  {srv.tools.length} tool{srv.tools.length === 1 ? "" : "s"}
                </div>
                <div className="mt-2 space-y-1">
                  {srv.tools.map((t) => (
                    <div
                      key={t.name}
                      className="rounded bg-bg-tertiary px-2 py-1 text-xs text-text-primary"
                    >
                      <span className="font-mono text-accent">{t.name}</span>
                      <span className="ml-2 text-text-muted">{t.description}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
