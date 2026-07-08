/**
 * usePlugins — MCP (Model Context Protocol) server management.
 *
 * Manages MCP server list, discovery, connection/disconnection.
 * Calls /mcp/servers, /mcp/servers/discover, /mcp/servers/connect,
 * and /mcp/servers/:name/disconnect endpoints.
 */
import { useState } from "react";
import { api } from "../lib/api";
import type { McpServer, DiscoveredServer } from "../types/domain";

export function usePlugins() {
  const [mcpServers, setMcpServers] = useState<McpServer[]>([]);
  const [discoveredServers, setDiscoveredServers] = useState<DiscoveredServer[]>([]);
  const [mcpMsg, setMcpMsg] = useState<string>("");
  const [newMcp, setNewMcp] = useState<{ name: string; command: string; args: string }>({
    name: "",
    command: "python",
    args: "",
  });

  const loadMcp = async () => {
    try {
      const data = await api.get<{ servers?: any[] }>("/mcp/servers");
      setMcpServers(data.servers || []);
    } catch (e: any) {
      setMcpMsg(`Failed to load MCP servers: ${e.message}`);
    }
  };

  const discoverMcp = async () => {
    try {
      const data = await api.get<{ servers?: any[] }>("/mcp/servers/discover");
      setDiscoveredServers(data.servers || []);
    } catch (e: any) {
      setMcpMsg(`Discovery failed: ${e.message}`);
    }
  };

  const connectMcp = async (server: { name: string; command: string; args: string[] }) => {
    setMcpMsg(`Connecting ${server.name}…`);
    try {
      const data = await api.post<{ success?: boolean; tools?: any[]; error?: string }>(
        "/mcp/servers/connect",
        server
      );
      if (data.success) {
        setMcpMsg(`Connected ${server.name} (${data.tools?.length || 0} tools)`);
        loadMcp();
      } else {
        setMcpMsg(`Connect failed: ${data.error}`);
      }
    } catch (e: any) {
      setMcpMsg(`Connect error: ${e.message}`);
    }
  };

  const disconnectMcp = async (name: string) => {
    setMcpMsg(`Disconnecting ${name}…`);
    try {
      const data = await api.post<{ success?: boolean; error?: string }>(
        `/mcp/servers/${name}/disconnect`
      );
      if (data.success) {
        setMcpMsg(`Disconnected ${name}`);
        loadMcp();
      } else {
        setMcpMsg(`Disconnect failed: ${data.error}`);
      }
    } catch (e: any) {
      setMcpMsg(`Disconnect error: ${e.message}`);
    }
  };

  return {
    mcpServers, discoveredServers, mcpMsg, newMcp,
    setMcpMsg, setNewMcp,
    loadMcp, discoverMcp, connectMcp, disconnectMcp,
  };
}
