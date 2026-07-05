"""Adapter to expose MCP tools as LangChain-compatible tools.

Allows Huginn to use tools from external MCP servers
(mat-db-mcp, math-anything-mcp) as if they were native tools.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from pydantic import BaseModel, create_model

from huginn.mcp_client import MCPClientManager
from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult

logger = logging.getLogger(__name__)

# MCP tool timeout defaults (seconds), by read-only classification.
_MCP_TIMEOUT_READONLY = 60  # read queries: 60s
_MCP_TIMEOUT_WRITE = 120  # write ops: 120s

# High-value MCP tools whose results should be promoted to memory.
_HIGH_VALUE_MCP_TOOLS = {
    "query_materials_project",
    "get_structure",
    "search_by_property",
    "query_interatomic_potentials",
    "compare_materials",
    "extract_math",
}


def _schema_to_pydantic(
    schema: dict[str, Any], model_name: str = "DynamicInput"
) -> type[BaseModel]:
    """Convert a JSON schema to a Pydantic model dynamically."""
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    fields: dict[str, tuple[type, Any]] = {}
    for name, prop in properties.items():
        json_type = prop.get("type", "string")
        if json_type == "string":
            py_type = str
        elif json_type == "integer":
            py_type = int
        elif json_type == "number":
            py_type = float
        elif json_type == "boolean":
            py_type = bool
        elif json_type == "array":
            py_type = list
        elif json_type == "object":
            py_type = dict
        else:
            py_type = str

        if name not in required:
            default = prop.get("default", None)
            fields[name] = (py_type | None, default)
        else:
            fields[name] = (py_type, ...)

    return create_model(model_name, **fields)


class MCPToolAdapter(HuginnTool):
    """Wraps an MCP tool as a HuginnTool.

    This enables seamless integration of MCP server tools into the
    Huginn tool registry and LangGraph agent.

    Includes:
    - CircuitBreaker protection (prevents cascading failures)
    - Async timeout (prevents hanging MCP calls)
    - Automatic memory promotion for high-value results
    """

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        client_manager: MCPClientManager,
    ):
        self._tool_name = name
        self._description = description
        self._client_manager = client_manager
        self.name = name
        self.description = description
        # Build Pydantic model from JSON schema
        self.input_schema = _schema_to_pydantic(input_schema, f"{name}Input")

    def is_read_only(self, args: BaseModel) -> bool:
        # Conservative default: assume MCP tools that query DB are read-only
        read_only_names = {
            "query_materials_project",
            "search_by_property",
            "get_structure",
            "query_interatomic_potentials",
            "compare_materials",
            "extract_math",
            "math_diff",
            "dimensional_analysis",
            "track_precision",
            "normalize_expression",
            "read_resource",
        }
        return self._tool_name in read_only_names

    async def call(self, args: BaseModel, context: ToolContext) -> ToolResult:
        # Determine timeout based on read/write classification
        timeout = (
            _MCP_TIMEOUT_READONLY
            if self.is_read_only(args)
            else _MCP_TIMEOUT_WRITE
        )

        # Use async CircuitBreaker if available
        try:
            from huginn.agents.circuit_breaker import async_circuit_guard

            async with async_circuit_guard(self._tool_name) as guard:
                if guard.get("blocked"):
                    return ToolResult(
                        data=None,
                        success=False,
                        error=f"MCP tool '{self._tool_name}' circuit breaker open",
                    )
                return await self._call_with_timeout(args, context, timeout)
        except ImportError:
            # CircuitBreaker not available, just use timeout
            return await self._call_with_timeout(args, context, timeout)
        except Exception as e:
            # CircuitBreaker open or other protection error
            cb_msg = "Circuit breaker open" if "circuit" in str(e).lower() else str(e)
            return ToolResult(
                data=None,
                success=False,
                error=f"MCP tool '{self._tool_name}' unavailable: {cb_msg}",
            )

    async def _call_with_timeout(
        self, args: BaseModel, context: ToolContext, timeout: int
    ) -> ToolResult:
        """Execute MCP tool call with timeout and memory promotion."""
        try:
            arguments = args.model_dump(exclude_none=True)
            result = await asyncio.wait_for(
                self._client_manager.call_tool(self._tool_name, arguments),
                timeout=timeout,
            )

            if result.get("is_error"):
                return ToolResult(
                    data=None,
                    success=False,
                    error=result.get("output", "Unknown MCP error"),
                )

            # Try to parse JSON output
            output = result.get("output", "")
            try:
                data = json.loads(output)
            except Exception:
                data = {"raw_output": output}

            # ── Promote high-value MCP results to memory ──────────
            # If this is a high-value tool (materials DB query, structure
            # retrieval, etc.), store the result in long-term memory so
            # future queries can benefit without re-calling the MCP server.
            if self._tool_name in _HIGH_VALUE_MCP_TOOLS and context.memory_manager:
                try:
                    memory_text = output[:2000] if isinstance(output, str) else json.dumps(data)[:2000]
                    context.memory_manager.add_tool_call(
                        tool_name=self._tool_name,
                        tool_args=arguments,
                        tool_result=memory_text,
                        success=True,
                    )
                except Exception:
                    # Memory promotion failure should never block tool result
                    logger.debug(
                        "MCP memory promotion failed for %s",
                        self._tool_name,
                        exc_info=True,
                    )

            return ToolResult(data=data, success=True)

        except asyncio.TimeoutError:
            return ToolResult(
                data=None,
                success=False,
                error=f"MCP tool '{self._tool_name}' timed out after {timeout}s",
            )
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"MCP tool error: {e}")


def register_mcp_tools(
    client_manager: MCPClientManager,
    server_name: str | None = None,
    whitelist: set[str] | None = None,
) -> list[HuginnTool]:
    """Discover MCP tools and register them as HuginnTools.

    If *server_name* is given, only tools from that server are registered
    (used during reconnection).  If *whitelist* is given, only tools whose
    name appears in the set are registered (used to filter ToolUniverse's
    350+ biomedical tools down to the materials-science subset).

    Existing tools with the same name are replaced and a debug message is
    logged.

    Returns the list of newly registered (or replaced) adapters.
    """
    from huginn.tools.registry import ToolRegistry

    tools: list[HuginnTool] = []
    skipped_by_whitelist = 0
    for info in client_manager.list_tools():
        if server_name and info.server_name != server_name:
            continue
        if whitelist is not None and info.name not in whitelist:
            skipped_by_whitelist += 1
            continue

        existing = ToolRegistry.get(info.name)
        if existing is not None:
            logger.debug(
                f"Replacing existing tool '{info.name}' "
                f"(server: {info.server_name})"
            )

        adapter = MCPToolAdapter(
            name=info.name,
            description=info.description,
            input_schema=info.input_schema,
            client_manager=client_manager,
        )
        ToolRegistry.register(adapter)
        tools.append(adapter)

    if whitelist is not None and skipped_by_whitelist:
        logger.info(
            f"MCP whitelist filtered out {skipped_by_whitelist} tools "
            f"from server '{server_name or 'all'}' ({len(tools)} kept)"
        )

    return tools
