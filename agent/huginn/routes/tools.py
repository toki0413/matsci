"""Tool listing and direct invocation endpoints."""

from __future__ import annotations

import asyncio
import json
import traceback
from typing import Any

from fastapi import APIRouter

from huginn.server_core import (
    _EDIT_TOOLS,
    _server_allows_tool,
    get_agent_factory,
    get_context,
    get_memory_manager,
)
from huginn.tools.registry import ToolRegistry
from huginn.types import ToolContext

router = APIRouter(tags=["tools"])


@router.get("/tools")
async def list_tools() -> list[dict[str, Any]]:
    """List all available tools with their schemas."""
    return ToolRegistry.get_all_schemas()


@router.post("/tools/{tool_name}")
async def call_tool(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Call a tool directly via HTTP."""
    from huginn.types import ToolContext

    tool = ToolRegistry.get(tool_name)
    if not tool:
        return {"error": f"Tool '{tool_name}' not found"}

    if not tool.input_schema:
        return {"error": f"Tool '{tool_name}' has no input schema"}

    try:
        input_data = tool.input_schema.model_validate(args, strict=True)
    except Exception as strict_err:
        # Retry with lenient mode for backward compat, but log the strict failure
        try:
            input_data = tool.input_schema(**args)
        except Exception:
            return {"error": f"Invalid input (strict): {strict_err}"}

    allowed, reason = _server_allows_tool(tool_name, input_data)
    if not allowed:
        get_context().audit_logger.log(
            event_type="tool_call",
            actor="http",
            action=tool_name,
            details={"approved": False, "reason": reason},
            input_data=json.dumps(args, sort_keys=True, default=str),
        )
        return {"error": reason}

    context = ToolContext(
        session_id="http",
        workspace=".",
        memory_manager=get_memory_manager(),
        agent_factory=get_agent_factory(),
        audit_logger=get_context().audit_logger,
    )
    import asyncio
    if asyncio.iscoroutinefunction(tool.call):
        result = await tool.call(input_data.model_dump(), context)
    else:
        result = tool.call(input_data.model_dump(), context)

    get_context().audit_logger.log(
        event_type="tool_call",
        actor="http",
        action=tool_name,
        details={"approved": True, "success": result.success},
        input_data=json.dumps(args, sort_keys=True, default=str),
        output_data=(
            json.dumps(result.data, sort_keys=True, default=str)
            if result.data
            else None
        ),
    )

    return {
        "success": result.success,
        "data": result.data,
        "error": result.error,
    }
