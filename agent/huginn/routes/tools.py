"""Tool listing and direct invocation endpoints."""

from __future__ import annotations

import inspect
import json
from typing import Any

from fastapi import APIRouter

from huginn.core_types import ToolContext
from huginn.server_core import (
    _server_allows_tool,
    get_agent_factory,
    get_context,
    get_memory_manager,
)
from huginn.tools.registry import ToolRegistry

router = APIRouter(tags=["tools"])


@router.get("/tools")
async def list_tools() -> list[dict[str, Any]]:
    """List all available tools with their schemas."""
    return ToolRegistry.get_all_schemas()


def _validate_tool_input(tool: Any, args: dict[str, Any]) -> tuple[Any, str | None]:
    """按 strict 模式校验入参, 失败再退一步尝试宽松模式."""
    try:
        return tool.input_schema.model_validate(args, strict=True), None
    except Exception as strict_err:
        # Retry with lenient mode for backward compat, but log the strict failure
        try:
            return tool.input_schema(**args), None
        except Exception:
            return None, f"Invalid input (strict): {strict_err}"


def _audit_denied(tool_name: str, args: dict[str, Any], reason: str) -> None:
    """记录一次被拒绝的工具调用."""
    get_context().audit_logger.log(
        event_type="tool_call",
        actor="http",
        action=tool_name,
        details={"approved": False, "reason": reason},
        input_data=json.dumps(args, sort_keys=True, default=str),
    )


def _audit_success(tool_name: str, args: dict[str, Any], result: Any) -> None:
    """记录一次成功的工具调用."""
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


async def _invoke_tool(tool: Any, input_data: Any, context: ToolContext) -> Any:
    if inspect.iscoroutinefunction(tool.call):
        return await tool.call(input_data.model_dump(), context)
    return tool.call(input_data.model_dump(), context)


@router.post("/tools/{tool_name}")
async def call_tool(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Call a tool directly via HTTP."""

    tool = ToolRegistry.get(tool_name)
    if not tool:
        return {"error": f"Tool '{tool_name}' not found"}

    if not tool.input_schema:
        return {"error": f"Tool '{tool_name}' has no input schema"}

    input_data, invalid_reason = _validate_tool_input(tool, args)
    if invalid_reason is not None:
        return {"error": invalid_reason}

    allowed, reason = _server_allows_tool(tool_name, input_data)
    if not allowed:
        _audit_denied(tool_name, args, reason)
        return {"error": reason}

    context = ToolContext(
        session_id="http",
        workspace=".",
        memory_manager=get_memory_manager(),
        agent_factory=get_agent_factory(),
        audit_logger=get_context().audit_logger,
    )
    result = await _invoke_tool(tool, input_data, context)

    _audit_success(tool_name, args, result)

    return {
        "success": result.success,
        "data": result.data,
        "error": result.error,
    }
