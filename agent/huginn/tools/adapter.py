"""LangChain tool adapter — bridges HuginnTool to LangChain BaseTool.

EvoScientist/deepagents expects LangChain-compatible tools.
This adapter wraps our HuginnTool instances into StructuredTool
so they can be used in the Agent Loop.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import typing
from typing import Any, Callable, get_origin

from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from huginn.security.audit import AuditLogger
from huginn.tools.base import HuginnTool
from huginn.types import (
    PermissionMode,
    ToolContext,
    ToolResult,
)
from huginn.permissions import PermissionConfig
from huginn.pet import get_pet_bus, PetMood
from huginn.privacy import redact_secrets
from huginn.tools.compress import compress_tool_output
from huginn.utils.tokens import rough_token_count_for_text


ApprovalCallback = Callable[[str, str], bool]
"""Callback signature: (tool_name, reason) -> approved."""


def _wants_dict(tool: HuginnTool) -> bool:
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


def _default_audit_logger() -> AuditLogger:
    """Return a default audit logger for tool invocations."""
    from pathlib import Path

    log_path = Path.home() / ".huginn" / "audit.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return AuditLogger(log_path)


class ToolAdapter:
    """Adapts HuginnTool instances to LangChain StructuredTool."""

    @staticmethod
    def adapt(
        tool: HuginnTool,
        memory_manager: Any | None = None,
        agent_factory: Any | None = None,
        permission_config: PermissionConfig | None = None,
        approval_callback: ApprovalCallback | None = None,
        audit_logger: AuditLogger | None = None,
        max_tool_output_tokens: int | None = None,
    ) -> StructuredTool:
        """Convert a HuginnTool to LangChain StructuredTool.

        Example:
            from huginn.tools.structure_tool import StructureTool
            from huginn.tools.adapter import ToolAdapter

            lc_tool = ToolAdapter.adapt(StructureTool())
            result = lc_tool.invoke({"action": "read", "file_path": "POSCAR"})
        """
        if not tool.input_schema:
            raise ValueError(f"Tool {tool.name} must have an input_schema")

        wants_dict = _wants_dict(tool)
        is_async = inspect.iscoroutinefunction(tool.call)
        permission_config = permission_config or PermissionConfig()
        audit_logger = audit_logger or _default_audit_logger()
        if max_tool_output_tokens is None:
            max_tool_output_tokens = int(os.environ.get("HUGINN_MAX_TOOL_OUTPUT_TOKENS", "25000"))

        def _check_permission(input_data: BaseModel) -> tuple[bool, str | None]:
            """Return (approved, reason_or_none)."""
            name = tool.name
            mode = permission_config.get_mode(name)

            if permission_config.auto_approve_all or mode == PermissionMode.AUTO:
                return True, None

            if mode == PermissionMode.DENY:
                return False, f"Tool '{name}' is blocked by permission policy"

            reasons: list[str] = []
            try:
                if tool.is_destructive(input_data):
                    reasons.append("this operation is destructive")
            except Exception:
                pass

            try:
                cost = tool.estimate_cost(input_data)
                if cost:
                    cpu = cost.get("cpu_hours", 0)
                    if cpu > 1:
                        reasons.append(f"estimated cost: {cpu:.1f} CPU hours")
            except Exception:
                pass

            reason = f"Tool '{name}' requires approval"
            if reasons:
                reason += f" ({', '.join(reasons)})"

            if approval_callback is not None:
                if approval_callback(name, reason):
                    return True, None
                return False, f"User denied: {reason}"

            # Non-interactive fallback: allow if HUGINN_AUTO_APPROVE is set.
            if os.environ.get("HUGINN_AUTO_APPROVE") == "1":
                return True, None

            return False, reason

        def _build_inputs(**kwargs: Any) -> tuple[BaseModel | dict[str, Any], ToolContext]:
            input_data = tool.input_schema(**kwargs)
            context = ToolContext(
                session_id="default",
                workspace=".",
                memory_manager=memory_manager,
                agent_factory=agent_factory,
                audit_logger=audit_logger,
            )
            payload = input_data.model_dump() if wants_dict else input_data
            return payload, context

        def _serialize(result: ToolResult) -> dict[str, Any]:
            data: dict[str, Any]
            if result.success:
                data = {"result": result.data}
            else:
                data = {"error": result.error or "Unknown error"}

            data = _sanitize_and_compress(data)
            return data

        def _sanitize_and_compress(obj: Any) -> Any:
            # Strings: privacy redaction first, then token-aware truncation.
            if isinstance(obj, str):
                s = obj
                if os.environ.get("HUGINN_PRIVACY_REDACT_SECRETS", "1") != "0":
                    s = redact_secrets(s)
                return compress_tool_output(
                    s, max_output_tokens=max_tool_output_tokens
                )

            # Everything else: apply structured compression (numeric summaries,
            # list head/tail, long-text truncation).
            return compress_tool_output(
                obj, max_output_tokens=max_tool_output_tokens
            )

        def _audit(
            input_data: BaseModel,
            output: dict[str, Any],
            approved: bool,
            reason: str | None,
        ) -> None:
            try:
                details: dict[str, Any] = {
                    "tool": tool.name,
                    "approved": approved,
                }
                if reason:
                    details["reason"] = reason
                audit_logger.log(
                    event_type="tool_call",
                    actor="agent",
                    action=tool.name,
                    details=details,
                    input_data=json.dumps(input_data.model_dump(), default=str, sort_keys=True),
                    output_data=json.dumps(output, default=str, sort_keys=True),
                )
            except Exception:
                # Audit failures must not break tool execution.
                pass

        def _publish(mood: PetMood, message: str, details: dict[str, Any] | None = None) -> None:
            try:
                get_pet_bus().publish(mood, message, details)
            except Exception:
                pass

        async def _arun(**kwargs: Any) -> dict[str, Any]:
            """Async execution wrapper."""
            payload, context = _build_inputs(**kwargs)
            input_data = tool.input_schema(**kwargs)

            approved, reason = _check_permission(input_data)
            if not approved:
                output = {"error": reason or f"Tool '{tool.name}' was denied"}
                _audit(input_data, output, approved=False, reason=reason)
                _publish(PetMood.ERROR, f"{tool.name} denied", {"reason": reason})
                return output

            validation = await tool.validate_input(input_data, context)
            if not validation.result:
                output = {"error": f"Input validation failed: {validation.message}"}
                _audit(input_data, output, approved=True, reason=validation.message)
                _publish(PetMood.ERROR, f"{tool.name} input invalid", {"reason": validation.message})
                return output

            _publish(PetMood.WORKING, f"Running {tool.name}…", {"tool": tool.name})
            if is_async:
                result = await tool.call(payload, context)
            else:
                result = tool.call(payload, context)
            output = _serialize(result)
            _audit(input_data, output, approved=True, reason=None)
            if result.success:
                _publish(PetMood.SUCCESS, f"{tool.name} done", {"tool": tool.name})
            else:
                _publish(PetMood.ERROR, f"{tool.name} failed", {"tool": tool.name, "error": result.error})
            return output

        def _run(**kwargs: Any) -> dict[str, Any]:
            """Sync execution wrapper."""
            payload, context = _build_inputs(**kwargs)
            input_data = tool.input_schema(**kwargs)

            approved, reason = _check_permission(input_data)
            if not approved:
                output = {"error": reason or f"Tool '{tool.name}' was denied"}
                _audit(input_data, output, approved=False, reason=reason)
                _publish(PetMood.ERROR, f"{tool.name} denied", {"reason": reason})
                return output

            validation_result = tool.validate_input(input_data, context)
            if asyncio.iscoroutine(validation_result):
                try:
                    validation = asyncio.run(validation_result)
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    try:
                        validation = loop.run_until_complete(validation_result)
                    finally:
                        loop.close()
            else:
                validation = validation_result
            if not validation.result:
                output = {"error": f"Input validation failed: {validation.message}"}
                _audit(input_data, output, approved=True, reason=validation.message)
                _publish(PetMood.ERROR, f"{tool.name} input invalid", {"reason": validation.message})
                return output

            _publish(PetMood.WORKING, f"Running {tool.name}…", {"tool": tool.name})
            if is_async:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    result = asyncio.run(tool.call(payload, context))
                else:
                    result = loop.run_until_complete(tool.call(payload, context))
            else:
                result = tool.call(payload, context)
            output = _serialize(result)
            _audit(input_data, output, approved=True, reason=None)
            if result.success:
                _publish(PetMood.SUCCESS, f"{tool.name} done", {"tool": tool.name})
            else:
                _publish(PetMood.ERROR, f"{tool.name} failed", {"tool": tool.name, "error": result.error})
            return output

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
        permission_config: PermissionConfig | None = None,
        approval_callback: ApprovalCallback | None = None,
        audit_logger: AuditLogger | None = None,
        max_tool_output_tokens: int | None = None,
    ) -> list[StructuredTool]:
        """Adapt all tools from a ToolRegistry."""
        tools = []
        for name in registry.list_tools():
            tool = registry.get(name)
            if tool:
                tools.append(
                    cls.adapt(
                        tool,
                        memory_manager=memory_manager,
                        agent_factory=agent_factory,
                        permission_config=permission_config,
                        approval_callback=approval_callback,
                        audit_logger=audit_logger,
                        max_tool_output_tokens=max_tool_output_tokens,
                    )
                )
        return tools
