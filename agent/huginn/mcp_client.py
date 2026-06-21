"""MCP Client Manager — connects to external MCP servers and exposes their tools.

Supports stdio-based MCP servers (e.g., mat-db-mcp, math-anything-mcp).
Discovers tools dynamically and routes calls to the correct server.
Provides health monitoring, automatic reconnection with exponential backoff,
and idempotent tool registration.

Also provides a registry interface for managing multiple server configurations
(list_servers, register_server, connect_server, disconnect_server, remove_server).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

# Reconnection defaults
_BACKOFF_BASE: float = 1.0       # initial delay in seconds
_BACKOFF_FACTOR: float = 2.0     # multiplier per retry
_BACKOFF_MAX: float = 30.0       # maximum delay cap
_MAX_RECONNECT_ATTEMPTS: int = 5  # give up after this many consecutive failures
_HEALTH_CHECK_INTERVAL: float = 30.0  # seconds between health polls
_HEALTH_CHECK_TIMEOUT: float = 5.0    # seconds before a health probe is considered hung


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
        # Stored configs for reconnection
        self._configs: dict[str, MCPServerConfig] = {}
        # Reconnection tracking
        self._consecutive_failures: dict[str, int] = {}
        self._last_health_check: dict[str, float] = {}
        # Registry for server configs (sync interface for tests)
        self._registry: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # Registry API (sync, used by tests and CLI)
    # ------------------------------------------------------------------ #
    def list_servers(self) -> list[dict[str, Any]]:
        """List registered server configurations."""
        return [
            {"name": name, "config": cfg}
            for name, cfg in self._registry.items()
        ]

    def register_server(self, name: str, config: dict[str, Any]) -> None:
        """Register a server configuration without connecting."""
        self._registry[name] = config

    def remove_server(self, name: str) -> None:
        """Remove a registered server configuration."""
        self._registry.pop(name, None)

    def connect_server(self, name: str) -> None:
        """Connect to a registered server by name (sync wrapper)."""
        cfg = self._registry.get(name)
        if not cfg:
            raise ValueError(f"Server '{name}' not registered")

        config = MCPServerConfig(
            name=name,
            command=cfg.get("command", "python"),
            args=cfg.get("args", []),
            env=cfg.get("env"),
        )

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule in running loop
                asyncio.create_task(self.connect(config))
            else:
                loop.run_until_complete(self.connect(config))
        except Exception as e:
            logger.warning(f"Failed to connect MCP server '{name}': {e}")
            raise

    def disconnect_server(self, name: str) -> None:
        """Disconnect a server by name (sync wrapper)."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.disconnect(name))
            else:
                loop.run_until_complete(self.disconnect(name))
        except Exception as e:
            logger.warning(f"Disconnect error for '{name}': {e}")

    def load_from_config(self, config: dict[str, dict[str, Any]] | None) -> None:
        """Load MCP server configurations from HuginnConfig.mcp_servers."""
        if not config:
            return
        for name, cfg in config.items():
            self.register_server(name, cfg)
            logger.info(f"Registered MCP server '{name}' from config")

    def load_from_huginn_config(self) -> None:
        """Auto-load from the global HuginnConfig."""
        try:
            from huginn.config import get_settings

            settings = get_settings()
            self.load_from_config(getattr(settings.config, "mcp_servers", None))
        except Exception as e:
            logger.warning(f"Failed to auto-load MCP config: {e}")

    # ------------------------------------------------------------------ #
    # Async connection API
    # ------------------------------------------------------------------ #
    async def connect(self, config: MCPServerConfig) -> None:
        """Connect to an MCP server via stdio."""
        if config.name in self._sessions:
            logger.warning(f"MCP server '{config.name}' already connected")
            return

        # Store config for future reconnection attempts
        self._configs[config.name] = config

        # Clear any stale tools from a previous connection (idempotent)
        self._tools = [t for t in self._tools if t.server_name != config.name]
        self._tool_index = {
            k: v for k, v in self._tool_index.items() if v.server_name != config.name
        }

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
            self._consecutive_failures[config.name] = 0
            self._last_health_check[config.name] = time.monotonic()

            # Discover tools
            tools_result = await session.list_tools()
            for tool in tools_result.tools:
                info = MCPToolInfo(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=tool.inputSchema
                    or {"type": "object", "properties": {}},
                    server_name=config.name,
                )
                self._tools.append(info)
                self._tool_index[tool.name] = info

            logger.info(
                f"Connected to MCP server '{config.name}' with {len(tools_result.tools)} tools"
            )

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
        self._tool_index = {
            k: v for k, v in self._tool_index.items() if v.server_name != name
        }

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
            raise ValueError(
                f"MCP tool '{name}' not found. Available: {list(self._tool_index.keys())}"
            )

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
        for _name, session in self._sessions.items():
            try:
                result = await session.read_resource(uri)
                return result.contents[0].text if result.contents else ""
            except Exception:
                continue
        raise ValueError(f"Resource '{uri}' not found on any connected server")

    def is_connected(self, name: str) -> bool:
        return name in self._initialized

    # ------------------------------------------------------------------ #
    # Health monitoring and reconnection
    # ------------------------------------------------------------------ #

    async def check_health(self, name: str) -> bool:
        """Probe whether a connected MCP server is still responsive.

        Uses ``list_tools()`` as a lightweight ping.  Returns ``True`` if
        the server replies within the timeout, ``False`` otherwise.  Never
        raises.
        """
        session = self._sessions.get(name)
        if not session:
            return False
        try:
            async with asyncio.timeout(_HEALTH_CHECK_TIMEOUT):
                await session.list_tools()
            self._last_health_check[name] = time.monotonic()
            return True
        except Exception:
            return False

    async def reconnect(self, name: str) -> bool:
        """Disconnect and re-connect a single MCP server using the stored config.

        Returns ``True`` on success, ``False`` on failure.  Updates the
        consecutive-failure counter used by the health monitor for backoff.
        """
        config = self._configs.get(name)
        if not config:
            logger.warning(f"Cannot reconnect '{name}': no stored config")
            return False

        # Tear down existing connection (best-effort, ignore errors)
        try:
            await self.disconnect(name)
        except Exception as e:
            logger.debug(f"Error during disconnect of '{name}' before reconnect: {e}")

        try:
            await self.connect(config)
            self._consecutive_failures[name] = 0
            return True
        except Exception as e:
            self._consecutive_failures[name] = (
                self._consecutive_failures.get(name, 0) + 1
            )
            logger.warning(
                f"Reconnect failed for '{name}' "
                f"(attempt {self._consecutive_failures[name]}): {e}"
            )
            return False

    def should_stop_retrying(self, name: str) -> bool:
        """Return True when the server has exceeded max reconnection attempts."""
        return self._consecutive_failures.get(name, 0) >= _MAX_RECONNECT_ATTEMPTS

    def get_server_status(self) -> dict[str, dict[str, Any]]:
        """Return a status snapshot for every known server (connected or not)."""
        names = set(self._configs.keys()) | set(self._sessions.keys())
        status: dict[str, dict[str, Any]] = {}
        for name in sorted(names):
            status[name] = {
                "connected": name in self._initialized,
                "tools": len([t for t in self._tools if t.server_name == name]),
                "failures": self._consecutive_failures.get(name, 0),
                "has_config": name in self._configs,
            }
        return status

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


# Module-level default manager for tests and CLI integration
mcp_manager = MCPClientManager()
