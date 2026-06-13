"""Adapter to expose MCP tools as LangChain-compatible tools.

Allows MatSci-Agent to use tools from external MCP servers
(mat-db-mcp, math-anything-mcp) as if they were native tools.
"""

from __future__ import annotations

from typing import Any, Callable, Coroutine

from pydantic import BaseModel, create_model

from matsci_agent.tools.base import MatSciTool
from matsci_agent.types import ToolResult, ToolContext
from matsci_agent.mcp_client import MCPClientManager


def _schema_to_pydantic(schema: dict[str, Any], model_name: str = "DynamicInput") -> type[BaseModel]:
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


class MCPToolAdapter(MatSciTool):
    """Wraps an MCP tool as a MatSciTool.

    This enables seamless integration of MCP server tools into the
    MatSci-Agent tool registry and LangGraph agent.
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
        try:
            arguments = args.model_dump(exclude_none=True)
            result = await self._client_manager.call_tool(self._tool_name, arguments)

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

            return ToolResult(data=data, success=True)

        except Exception as e:
            return ToolResult(data=None, success=False, error=f"MCP tool error: {e}")


def register_mcp_tools(client_manager: MCPClientManager) -> list[MatSciTool]:
    """Discover MCP tools and register them as MatSciTools.

    Returns the list of adapted tools for manual registration if needed.
    """
    from matsci_agent.tools.registry import ToolRegistry

    tools: list[MatSciTool] = []
    for info in client_manager.list_tools():
        adapter = MCPToolAdapter(
            name=info.name,
            description=info.description,
            input_schema=info.input_schema,
            client_manager=client_manager,
        )
        ToolRegistry.register(adapter)
        tools.append(adapter)

    return tools


import json
