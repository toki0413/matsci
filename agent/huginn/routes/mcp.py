"""MCP / plugin management endpoints."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from fastapi import APIRouter

from huginn.server_core import get_context
from huginn.tools.registry import ToolRegistry

router = APIRouter(tags=["mcp"])


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
async def connect_mcp_server(params: dict[str, Any]) -> dict[str, Any]:
    """Connect to an MCP server and register its tools."""
    if get_context().mcp_manager is None:
        from huginn.mcp_client import MCPClientManager

        get_context().mcp_manager = MCPClientManager()

    name = params.get("name", "")
    command = params.get("command", "python")
    args = params.get("args", [])
    env = params.get("env")
    if not name:
        return {"success": False, "error": "name is required"}

    try:
        from huginn.mcp_client import MCPServerConfig
        from huginn.tools.mcp_adapter import register_mcp_tools

        await get_context().mcp_manager.connect(
            MCPServerConfig(
                name=name,
                command=command,
                args=args,
                env=env,
            )
        )
        registered = register_mcp_tools(get_context().mcp_manager)
        return {
            "success": True,
            "server": name,
            "tools": [
                {"name": t.name, "description": t.description}
                for t in registered
                if t.server_name == name
            ],
        }
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@router.post("/mcp/servers/{name}/disconnect")
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
