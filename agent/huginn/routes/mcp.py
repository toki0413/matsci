"""MCP / plugin management endpoints."""

from __future__ import annotations

import contextlib
import logging
import traceback
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends

from huginn.security.auth import require_admin_key as require_admin
from huginn.server_core import get_context
from huginn.tools.registry import ToolRegistry

router = APIRouter(tags=["mcp"])

logger = logging.getLogger(__name__)


@router.get("/mcp/servers")
async def list_mcp_servers() -> dict[str, Any]:
    """List connected MCP servers and their discovered tools."""
    if get_context().mcp_manager is None:
        return {"servers": [], "connected": []}
    servers = []
    for name in get_context().mcp_manager._sessions:
        tools = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in get_context().mcp_manager._tools
            if t.server_name == name
        ]
        servers.append({"name": name, "connected": True, "tools": tools})
    return {"servers": servers}


@router.get("/mcp/servers/discover")
async def discover_mcp_servers() -> dict[str, Any]:
    """Discover local MCP server directories under servers/*."""
    base = Path(__file__).parent.parent.parent / "servers"
    found = []
    if base.exists():
        for entry in base.iterdir():
            server_py = entry / "server.py"
            if entry.is_dir() and server_py.exists():
                found.append(
                    {
                        "name": entry.name,
                        "path": str(server_py),
                        "command": "python",
                        "args": [str(server_py)],
                    }
                )
    return {"servers": found}


@router.post("/mcp/servers/connect")
async def connect_mcp_server(
    params: dict[str, Any], _admin: str = Depends(require_admin)
) -> dict[str, Any]:
    """Connect to an MCP server and register its tools.

    Requires admin authentication to prevent arbitrary command execution.
    """
    if get_context().mcp_manager is None:
        from huginn.mcp_client import MCPClientManager

        get_context().mcp_manager = MCPClientManager()

    name = params.get("name", "")
    if not name:
        return {"success": False, "error": "name is required"}

    transport = params.get("transport", "stdio")

    try:
        from huginn.mcp_client import MCPServerConfig
        from huginn.tools.mcp_adapter import register_mcp_prompts, register_mcp_tools

        if transport == "sse":
            # SSE spawns no local process, so the command whitelist doesn't
            # apply — we just need a reachable URL.
            url = params.get("url")
            if not url:
                return {
                    "success": False,
                    "error": "url is required for SSE transport",
                }
            await get_context().mcp_manager.connect(
                MCPServerConfig(
                    name=name, command="", args=[], transport="sse", url=url
                )
            )
        else:
            command = params.get("command", "python")
            args = params.get("args", [])
            env = params.get("env")

            # Command whitelist: only allow known-safe interpreters
            _ALLOWED_COMMANDS = {"python", "python3", "node", "npx", "uvx"}
            if command not in _ALLOWED_COMMANDS:
                return {
                    "success": False,
                    "error": f"Command '{command}' not in allowed list: {_ALLOWED_COMMANDS}",
                }

            await get_context().mcp_manager.connect(
                MCPServerConfig(name=name, command=command, args=args, env=env)
            )

        registered = register_mcp_tools(get_context().mcp_manager, server_name=name)
        # Prompts are optional; failure to list them shouldn't sink the connect.
        with contextlib.suppress(Exception):
            await register_mcp_prompts(get_context().mcp_manager, server_name=name)
        return {
            "success": True,
            "server": name,
            "tools": [
                {"name": t.name, "description": t.description}
                for t in registered
            ],
        }
    except Exception as e:
        logger.error("unexpected error", exc_info=True)
        return {"success": False, "error": str(e)}


@router.post("/mcp/servers/{name}/disconnect", dependencies=[Depends(require_admin)])
async def disconnect_mcp_server(name: str) -> dict[str, Any]:
    """Disconnect an MCP server and unregister its tools."""
    if get_context().mcp_manager is None:
        return {"success": False, "error": "MCP manager not initialized"}

    try:
        tools_to_remove = [
            t.name for t in get_context().mcp_manager._tools if t.server_name == name
        ]
        await get_context().mcp_manager.disconnect(name)
        for tool_name in tools_to_remove:
            ToolRegistry.unregister(tool_name)
        return {"success": True, "unregistered": tools_to_remove}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/mcp/status")
async def mcp_server_status() -> dict[str, Any]:
    """Return health and connection status for all known MCP servers."""
    mgr = get_context().mcp_manager
    if mgr is None:
        return {"servers": {}, "message": "MCP manager not initialized"}
    return {"servers": mgr.get_server_status()}


@router.post("/mcp/servers/{name}/reconnect", dependencies=[Depends(require_admin)])
async def reconnect_mcp_server(name: str) -> dict[str, Any]:
    """Manually trigger a reconnection to an MCP server."""
    mgr = get_context().mcp_manager
    if mgr is None:
        return {"success": False, "error": "MCP manager not initialized"}

    try:
        from huginn.tools.mcp_adapter import register_mcp_prompts, register_mcp_tools

        success = await mgr.reconnect(name)
        if success:
            registered = register_mcp_tools(mgr, server_name=name)
            with contextlib.suppress(Exception):
                await register_mcp_prompts(mgr, server_name=name)
            return {
                "success": True,
                "server": name,
                "tools": [t.name for t in registered],
            }
        return {
            "success": False,
            "error": f"Reconnect failed for '{name}' "
            f"(failures: {mgr._consecutive_failures.get(name, 0)})",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/mcp/tools/{tool_name}/call", dependencies=[Depends(require_admin)])
async def call_mcp_tool(
    tool_name: str, args: dict[str, Any]
) -> dict[str, Any]:
    """直接调用 MCP 工具, 带 session expiry 检测和自动重连.

    用 call_tool_with_retry 替代裸 call_tool, 连续失败或 session
    过期时自动重连 server, 避免一次性调用失败.
    """
    mgr = get_context().mcp_manager
    if mgr is None:
        return {"success": False, "error": "MCP manager not initialized"}

    try:
        result = await mgr.call_tool_with_retry(tool_name, args)
        return {"success": True, "tool": tool_name, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/mcp/connect/batch")
async def connect_mcp_batch(
    params: dict[str, Any], _admin: str = Depends(require_admin)
) -> dict[str, Any]:
    """批量并发连接多个 MCP server.

    用 connect_batch 替代逐个 connect, 内部用信号量控制并发度,
    避免大量 server 同时握手把资源耗尽.
    """
    from huginn.mcp_client import MCPServerConfig

    if get_context().mcp_manager is None:
        from huginn.mcp_client import MCPClientManager

        get_context().mcp_manager = MCPClientManager()

    servers = params.get("servers", [])
    concurrency = params.get("concurrency", 4)
    if not servers:
        return {"success": False, "error": "servers list is required"}

    configs = [
        MCPServerConfig(
            name=s.get("name", ""),
            command=s.get("command", "python"),
            args=s.get("args", []),
            env=s.get("env"),
        )
        for s in servers
        if s.get("name")
    ]

    try:
        results = await get_context().mcp_manager.connect_batch(
            configs, concurrency=concurrency
        )
        return {
            "success": True,
            "connected": [
                {"name": name, "ok": ok} for name, ok in results.items()
            ],
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/mcp/prompts")
async def list_mcp_prompts() -> dict[str, Any]:
    """List prompts exposed by every connected MCP server.

    Returns ``{prompts: {server_name: [{name, description, arguments}, ...]}}``.
    """
    mgr = get_context().mcp_manager
    if mgr is None:
        return {"prompts": {}}
    try:
        return {"prompts": await mgr.list_prompts()}
    except Exception as e:
        logger.error("list_mcp_prompts failed", exc_info=True)
        return {"prompts": {}, "error": str(e)}


@router.post("/mcp/prompts/{name}/get")
async def get_mcp_prompt(
    name: str, args: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Render a prompt by name from the first server that has it.

    Body is the prompt's arguments (may be empty). Returns the concatenated
    message text under ``content``.
    """
    mgr = get_context().mcp_manager
    if mgr is None:
        return {"success": False, "error": "MCP manager not initialized"}
    try:
        text = await mgr.get_prompt(name, args)
        return {"success": True, "prompt": name, "content": text}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/mcp/resources/subscribe")
async def subscribe_mcp_resource(
    params: dict[str, Any],
) -> dict[str, Any]:
    """Subscribe to updates for an MCP resource URI.

    Body: ``{"uri": "..."}``. HTTP can't hold a client callback open, so this
    establishes the subscription on every connected server and logs updates
    server-side. In-process callers that need push delivery should call
    ``mcp_manager.subscribe_resource(uri, callback)`` directly.
    """
    mgr = get_context().mcp_manager
    if mgr is None:
        return {"success": False, "error": "MCP manager not initialized"}

    uri = params.get("uri")
    if not uri:
        return {"success": False, "error": "uri is required"}

    async def _log_update(changed_uri: str) -> None:
        logger.info("MCP resource updated: %s", changed_uri)

    try:
        await mgr.subscribe_resource(uri, _log_update)
        return {"success": True, "uri": uri}
    except Exception as e:
        return {"success": False, "error": str(e)}
