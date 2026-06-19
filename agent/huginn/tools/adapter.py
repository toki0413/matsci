"""LangChain tool adapter — bridges HuginnTool to LangChain BaseTool.

EvoScientist/deepagents expects LangChain-compatible tools.
This adapter wraps our HuginnTool instances into StructuredTool
so they can be used in the Agent Loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import os
import typing
from collections.abc import Callable
from typing import Any, get_origin

from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from huginn.constraints import ConstraintAdapter
from huginn.constraints.boundaries import BoundaryEvolution, BoundaryState
from huginn.permissions import PermissionConfig
from huginn.pet import PetMood, get_pet_bus
from huginn.privacy import redact_secrets
from huginn.security.audit import AuditLogger
from huginn.telemetry import get_telemetry_collector
from huginn.tools.base import HuginnTool
from huginn.tools.compress import compress_tool_output
from huginn.types import (
    PermissionMode,
    ToolContext,
    ToolResult,
)
from huginn.utils.cache import TimedLRUCache

# Map tool names to the constraint scope used for post-call validation.
_TOOL_CONSTRAINT_SCOPES: dict[str, str] = {
    "vasp_tool": "dft",
    "qe_tool": "dft",
    "cp2k_tool": "dft",
    "lammps_tool": "md",
    "openfoam_tool": "cfd",
    "comsol_tool": "fea",
    "abaqus_tool": "fea",
}

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

    # Bounded cache for read-only tool outputs to improve cache hit rate and
    # reduce repeated token-heavy outputs in the agent context window.
    _read_only_cache: TimedLRUCache[dict[str, Any]] = TimedLRUCache(
        max_size=256, ttl=300.0
    )
    _constraint_adapter: ConstraintAdapter = ConstraintAdapter.default()

    @staticmethod
    def adapt(
        tool: HuginnTool,
        memory_manager: Any | None = None,
        agent_factory: Any | None = None,
        permission_config: PermissionConfig | None = None,
        approval_callback: ApprovalCallback | None = None,
        audit_logger: AuditLogger | None = None,
        max_tool_output_tokens: int | None = None,
        compression_max_tokens: int | None = None,
        boundary_state: BoundaryState | None = None,
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
            max_tool_output_tokens = int(
                os.environ.get("HUGINN_MAX_TOOL_OUTPUT_TOKENS", "25000")
            )
        if compression_max_tokens is None:
            compression_max_tokens = int(
                os.environ.get("HUGINN_TOOL_COMPRESSION_MAX_TOKENS", str(max_tool_output_tokens))
            )

        def _check_permission(input_data: BaseModel) -> tuple[bool, str | None]:
            """Return (approved, reason_or_none)."""
            name = tool.name
            mode = permission_config.get_mode(name)
            scope = _TOOL_CONSTRAINT_SCOPES.get(name)

            if boundary_state is not None:
                if name in boundary_state.blocked_tools:
                    return False, f"Tool '{name}' is blocked by dynamic boundary"
                if scope is not None and scope in boundary_state.blocked_scopes:
                    return (
                        False,
                        f"Tool '{name}' is blocked by dynamic boundary (scope: {scope})",
                    )
                if boundary_state.require_confirmation and mode == PermissionMode.AUTO:
                    mode = PermissionMode.ASK

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

        def _build_inputs(
            **kwargs: Any,
        ) -> tuple[BaseModel | dict[str, Any], ToolContext]:
            input_data = tool.input_schema(**kwargs)
            context = ToolContext(
                session_id="default",
                workspace=".",
                memory_manager=memory_manager,
                agent_factory=agent_factory,
                audit_logger=audit_logger,
                boundary_state=boundary_state,
            )
            payload = input_data.model_dump() if wants_dict else input_data
            return payload, context

        def _check_constraints(result: ToolResult, context: ToolContext) -> ToolResult:
            """Run domain constraints on successful tool outputs."""
            if not result.success:
                return result
            scope = _TOOL_CONSTRAINT_SCOPES.get(tool.name)
            if scope is None or not isinstance(result.data, dict):
                return result

            checks = ToolAdapter._constraint_adapter.evaluate_all(scope, result.data)
            warnings = [c for c in checks if not c.passed and c.severity != "block"]
            blocks = [c for c in checks if not c.passed and c.severity == "block"]

            if warnings:
                result.data["_constraint_warnings"] = [
                    {"name": c.name, "message": c.message, "severity": c.severity}
                    for c in warnings
                ]

            # Evolve the session boundary based on constraint outcomes.
            if context.boundary_state is not None:
                BoundaryEvolution(context.boundary_state).update(checks)
                if blocks:
                    context.boundary_state.blocked_tools.add(tool.name)
                    if scope is not None:
                        context.boundary_state.blocked_scopes.add(scope)

            if blocks:
                messages = "; ".join(f"{c.name}: {c.message}" for c in blocks)
                return ToolResult(
                    data=result.data,
                    success=False,
                    error=f"Constraint check failed: {messages}",
                )

            return result

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
                return compress_tool_output(s, max_output_tokens=compression_max_tokens)

            # Everything else: apply structured compression (numeric summaries,
            # list head/tail, long-text truncation).
            return compress_tool_output(obj, max_output_tokens=compression_max_tokens)

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
                raw_input = json.dumps(
                    input_data.model_dump(), default=str, sort_keys=True
                )
                raw_output = json.dumps(output, default=str, sort_keys=True)
                audit_logger.log(
                    event_type="tool_call",
                    actor="agent",
                    action=tool.name,
                    details=details,
                    input_data=redact_secrets(raw_input),
                    output_data=redact_secrets(raw_output),
                )
            except Exception:
                # Audit failures must not break tool execution.
                pass

        def _publish(
            mood: PetMood, message: str, details: dict[str, Any] | None = None
        ) -> None:
            with contextlib.suppress(Exception):
                get_pet_bus().publish(mood, message, details)

        def _cache_key(input_data: BaseModel) -> str:
            return f"{tool.name}:{json.dumps(input_data.model_dump(), sort_keys=True, default=str)}"

        def _get_cached(input_data: BaseModel) -> dict[str, Any] | None:
            if not getattr(tool, "read_only", False):
                return None
            return ToolAdapter._read_only_cache.get(_cache_key(input_data))

        def _set_cached(input_data: BaseModel, output: dict[str, Any]) -> None:
            if getattr(tool, "read_only", False) and output.get("error") is None:
                ToolAdapter._read_only_cache.set(_cache_key(input_data), output)

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
                _publish(
                    PetMood.ERROR,
                    f"{tool.name} input invalid",
                    {"reason": validation.message},
                )
                return output

            cached = _get_cached(input_data)
            if cached is not None:
                _audit(input_data, cached, approved=True, reason="cache_hit")
                _publish(PetMood.SUCCESS, f"{tool.name} (cached)", {"tool": tool.name})
                return cached

            _publish(PetMood.WORKING, f"Running {tool.name}…", {"tool": tool.name})
            with get_telemetry_collector().span("tool_call", tool=tool.name) as span:
                if is_async:
                    result = await tool.call(payload, context)
                else:
                    result = tool.call(payload, context)
                result = _check_constraints(result, context)
                output = _serialize(result)
                span.metadata["success"] = result.success
                if result.error:
                    span.metadata["error"] = result.error
            _audit(input_data, output, approved=True, reason=None)
            if result.success:
                _publish(PetMood.SUCCESS, f"{tool.name} done", {"tool": tool.name})
                _set_cached(input_data, output)
            else:
                _publish(
                    PetMood.ERROR,
                    f"{tool.name} failed",
                    {"tool": tool.name, "error": result.error},
                )
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
                _publish(
                    PetMood.ERROR,
                    f"{tool.name} input invalid",
                    {"reason": validation.message},
                )
                return output

            cached = _get_cached(input_data)
            if cached is not None:
                _audit(input_data, cached, approved=True, reason="cache_hit")
                _publish(PetMood.SUCCESS, f"{tool.name} (cached)", {"tool": tool.name})
                return cached

            _publish(PetMood.WORKING, f"Running {tool.name}…", {"tool": tool.name})
            with get_telemetry_collector().span("tool_call", tool=tool.name) as span:
                if is_async:
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        result = asyncio.run(tool.call(payload, context))
                    else:
                        result = loop.run_until_complete(tool.call(payload, context))
                else:
                    result = tool.call(payload, context)
                result = _check_constraints(result, context)
                output = _serialize(result)
                span.metadata["success"] = result.success
                if result.error:
                    span.metadata["error"] = result.error
            _audit(input_data, output, approved=True, reason=None)
            if result.success:
                _publish(PetMood.SUCCESS, f"{tool.name} done", {"tool": tool.name})
                _set_cached(input_data, output)
            else:
                _publish(
                    PetMood.ERROR,
                    f"{tool.name} failed",
                    {"tool": tool.name, "error": result.error},
                )
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
        compression_max_tokens: int | None = None,
        boundary_state: BoundaryState | None = None,
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
                        compression_max_tokens=compression_max_tokens,
                        boundary_state=boundary_state,
                    )
                )
        return tools
