"""LangChain tool adapter — bridges MatSciTool to LangChain BaseTool.

EvoScientist/deepagents expects LangChain-compatible tools.
This adapter wraps our MatSciTool instances into StructuredTool
so they can be used in the Agent Loop.
"""

from __future__ import annotations

import asyncio
import inspect
import typing
from typing import Any, get_origin

from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from matsci_agent.tools.base import MatSciTool
from matsci_agent.types import ToolContext, ToolResult


def _wants_dict(tool: MatSciTool) -> bool:
    """Return True if ``tool.call`` expects a plain dict for ``args``."""
    try:
        hints = typing.get_type_hints(tool.call)
    except Exception:
        hints = {}
    ann = hints.get("args")
    if ann is None:
        return False
    origin = get_origin(ann)
    return origin is dict or ann is dict


class ToolAdapter:
    """Adapts MatSciTool instances to LangChain StructuredTool."""

    @staticmethod
    def adapt(
        tool: MatSciTool,
        memory_manager: Any | None = None,
        agent_factory: Any | None = None,
    ) -> StructuredTool:
        """Convert a MatSciTool to LangChain StructuredTool.

        Example:
            from matsci_agent.tools.structure_tool import StructureTool
            from matsci_agent.tools.adapter import ToolAdapter

            lc_tool = ToolAdapter.adapt(StructureTool())
            result = lc_tool.invoke({"action": "read", "file_path": "POSCAR"})
        """
        if not tool.input_schema:
            raise ValueError(f"Tool {tool.name} must have an input_schema")

        wants_dict = _wants_dict(tool)
        is_async = inspect.iscoroutinefunction(tool.call)

        def _build_inputs(**kwargs: Any) -> tuple[BaseModel | dict[str, Any], ToolContext]:
            input_data = tool.input_schema(**kwargs)
            context = ToolContext(
                session_id="default",
                workspace=".",
                memory_manager=memory_manager,
                agent_factory=agent_factory,
            )
            payload = input_data.model_dump() if wants_dict else input_data
            return payload, context

        def _serialize(result: ToolResult) -> dict[str, Any]:
            if result.success:
                return {"result": result.data}
            return {"error": result.error or "Unknown error"}

        async def _arun(**kwargs: Any) -> dict[str, Any]:
            """Async execution wrapper."""
            payload, context = _build_inputs(**kwargs)
            if is_async:
                result = await tool.call(payload, context)
            else:
                result = tool.call(payload, context)
            return _serialize(result)

        def _run(**kwargs: Any) -> dict[str, Any]:
            """Sync execution wrapper."""
            payload, context = _build_inputs(**kwargs)
            if is_async:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    result = asyncio.run(tool.call(payload, context))
                else:
                    result = loop.run_until_complete(tool.call(payload, context))
            else:
                result = tool.call(payload, context)
            return _serialize(result)

        return StructuredTool.from_function(
            name=tool.name,
            description=tool.description,
            args_schema=tool.input_schema,
            coroutine=_arun,
            func=_run,
            return_direct=False,
        )

    @classmethod
    def adapt_registry(
        cls,
        registry: Any,
        memory_manager: Any | None = None,
        agent_factory: Any | None = None,
    ) -> list[StructuredTool]:
        """Adapt all tools from a ToolRegistry."""
        tools = []
        for name in registry.list_tools():
            tool = registry.get(name)
            if tool:
                tools.append(cls.adapt(tool, memory_manager=memory_manager, agent_factory=agent_factory))
        return tools
