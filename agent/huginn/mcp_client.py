"""MCP Client Manager — connects to external MCP servers and exposes their tools.

Supports stdio-based MCP servers (e.g., mat-db-mcp, math-anything-mcp).
Discovers tools dynamically and routes calls to the correct server.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server connection."""
    name: str
    command: str
    args: list[str]
    env: dict[str, str] | None = None


@dataclass
class MCPToolInfo:
    """Discovered MCP tool with routing info."""
    name: str
    description: str
    input_schema: dict[str, Any]
    server_name: str


class MCPClientManager:
    """Manages connections to multiple MCP servers.

    Usage:
        mgr = MCPClientManager()
        await mgr.connect(MCPServerConfig("mat-db", "python", ["server.py"]))
        tools = mgr.list_tools()
        result = await mgr.call_tool("query_materials_project", {"formula": "Si"})
        await mgr.disconnect_all()
    """

    def __init__(self):
        self._sessions: dict[str, ClientSession] = {}
        self._clients: dict[str, Any] = {}  # stdio client contexts
        self._tools: list[MCPToolInfo] = []
        self._tool_index: dict[str, MCPToolInfo] = {}
        self._initialized: set[str] = set()

    async def connect(self, config: MCPServerConfig) -> None:
        """Connect to an MCP server via stdio."""
        if config.name in self._sessions:
            logger.warning(f"MCP server '{config.name}' already connected")
            return

        params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=config.env,
        )

        try:
            client = stdio_client(params)
            read_stream, write_stream = await client.__aenter__()
            session = ClientSession(read_stream, write_stream)
            await session.__aenter__()
            await session.initialize()

            self._clients[config.name] = client
            self._sessions[config.name] = session
            self._initialized.add(config.name)

            # Discover tools
            tools_result = await session.list_tools()
            for tool in tools_result.tools:
                info = MCPToolInfo(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=tool.inputSchema or {"type": "object", "properties": {}},
                    server_name=config.name,
                )
                self._tools.append(info)
                self._tool_index[tool.name] = info

            logger.info(f"Connected to MCP server '{config.name}' with {len(tools_result.tools)} tools")

        except Exception as e:
            logger.error(f"Failed to connect to MCP server '{config.name}': {e}")
            raise

    async def disconnect(self, name: str) -> None:
        """Disconnect a specific MCP server."""
        session = self._sessions.pop(name, None)
        client = self._clients.pop(name, None)
        self._initialized.discard(name)

        # Remove tools from this server
        self._tools = [t for t in self._tools if t.server_name != name]
        self._tool_index = {k: v for k, v in self._tool_index.items() if v.server_name != name}

        if session:
            await session.__aexit__(None, None, None)
        if client:
            await client.__aexit__(None, None, None)

        logger.info(f"Disconnected MCP server '{name}'")

    async def disconnect_all(self) -> None:
        """Disconnect all MCP servers."""
        for name in list(self._sessions.keys()):
            await self.disconnect(name)

    def list_tools(self) -> list[MCPToolInfo]:
        """List all discovered MCP tools."""
        return list(self._tools)

    def get_tool_info(self, name: str) -> MCPToolInfo | None:
        return self._tool_index.get(name)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call an MCP tool by name."""
        info = self._tool_index.get(name)
        if not info:
            raise ValueError(f"MCP tool '{name}' not found. Available: {list(self._tool_index.keys())}")

        session = self._sessions.get(info.server_name)
        if not session:
            raise RuntimeError(f"MCP server '{info.server_name}' not connected")

        result = await session.call_tool(name, arguments)

        # Convert MCP result to simple dict
        outputs = []
        for content in result.content:
            if hasattr(content, "text"):
                outputs.append(content.text)
            else:
                outputs.append(str(content))

        return {
            "success": not result.isError,
            "output": "\n".join(outputs),
            "is_error": result.isError,
        }

    async def read_resource(self, uri: str, server_name: str | None = None) -> str:
        """Read an MCP resource."""
        if server_name:
            session = self._sessions.get(server_name)
            if not session:
                raise RuntimeError(f"MCP server '{server_name}' not connected")
            result = await session.read_resource(uri)
            return result.contents[0].text if result.contents else ""

        # Try all servers
        for name, session in self._sessions.items():
            try:
                result = await session.read_resource(uri)
                return result.contents[0].text if result.contents else ""
            except Exception:
                continue
        raise ValueError(f"Resource '{uri}' not found on any connected server")

    def is_connected(self, name: str) -> bool:
        return name in self._initialized

    def __del__(self):
        # Best-effort cleanup
        if self._sessions:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.disconnect_all())
                else:
                    loop.run_until_complete(self.disconnect_all())
            except Exception:
                pass
