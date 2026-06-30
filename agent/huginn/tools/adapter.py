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
import logging
import os
import time
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
from huginn.tools.compress import compress_tool_output, smart_compress_text
from huginn.tools.timeouts import get_timeout
from huginn.types import (
    PermissionMode,
    ToolContext,
    ToolResult,
)
from huginn.utils.cache import TimedLRUCache
from huginn.utils.tokens import count_tokens

logger = logging.getLogger(__name__)

# Map tool names to the constraint scope used for post-call validation.
# Populated by _rebuild_constraint_scopes() from ToolProfile metadata.
# Internal callers should prefer tool.constraint_scope directly; this dict
# is kept as a backward-compat shim for external consumers.
_TOOL_CONSTRAINT_SCOPES: dict[str, str] = {}


def _rebuild_constraint_scopes() -> None:
    """Rebuild _TOOL_CONSTRAINT_SCOPES in place from ToolProfile metadata.

    Called at the end of register_all_tools() so the scope map tracks the
    registered tools' declared constraint_scope instead of a hand-maintained
    dict.
    """
    from huginn.tools.registry import ToolRegistry

    new = {
        t.name: t.constraint_scope
        for t in ToolRegistry._tools.values()
        if t.constraint_scope is not None
    }
    _TOOL_CONSTRAINT_SCOPES.clear()
    _TOOL_CONSTRAINT_SCOPES.update(new)

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


# trajectory 持久化时单个字段的字符上限, 避免大输出把 JSON 文件撑爆
_TRAJECTORY_FIELD_LIMIT = 8192


def _truncate_for_trajectory(value: Any) -> Any:
    """递归截断 args/result, 防止大输出撑爆 trajectory 文件。

    字符串超限就截断并加标记, dict/list 递归处理, 其他类型原样返回。
    """
    if isinstance(value, str):
        if len(value) > _TRAJECTORY_FIELD_LIMIT:
            return value[:_TRAJECTORY_FIELD_LIMIT] + "...(truncated)"
        return value
    if isinstance(value, dict):
        return {k: _truncate_for_trajectory(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_truncate_for_trajectory(v) for v in value]
    return value


def _default_audit_logger() -> AuditLogger:
    """Return a default audit logger for tool invocations."""
    from pathlib import Path

    base = os.environ.get("HUGINN_CACHE_DIR")
    log_path = Path(base) / "audit.jsonl" if base else Path.home() / ".huginn" / "audit.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return AuditLogger(log_path)


# 熔断器/仪表盘开关，跑测试或 benchmark 时可以关掉
_HEALTH_MONITOR_ON = os.environ.get("HUGINN_HEALTH_MONITOR", "1") == "1"


def _breaker_blocked(tool_name: str) -> dict[str, Any] | None:
    """熔断器开着就返回错误 dict，没装或放行返回 None。"""
    if not _HEALTH_MONITOR_ON:
        return None
    try:
        from huginn.agents.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker.shared()
        if not breaker.can_call(tool_name):
            stats = breaker.get_stats(tool_name)
            return {
                "error": "circuit_open",
                "tool": tool_name,
                "retry_after": stats.get("retry_after", 0),
                "_circuit_open": True,
            }
    except Exception:
        pass
    return None


def _record_outcome(
    tool_name: str,
    success: bool,
    duration_sec: float,
    error: str | None = None,
) -> None:
    """工具执行完记一笔到熔断器 + 仪表盘，best-effort 不抛。"""
    if not _HEALTH_MONITOR_ON:
        return
    try:
        from huginn.agents.circuit_breaker import CircuitBreaker
        from huginn.agents.health_dashboard import HealthDashboard

        breaker = CircuitBreaker.shared()
        dashboard = HealthDashboard.shared()
        if success:
            breaker.record_success(tool_name)
        else:
            breaker.record_failure(tool_name, error or "")
        dashboard.record_call(
            tool_name, success, duration_sec, cache_hit=False, error=error
        )
    except Exception:
        pass


def _record_cache_hit(tool_name: str) -> None:
    """缓存命中记一笔到仪表盘（不算熔断器的成败）。"""
    if not _HEALTH_MONITOR_ON:
        return
    try:
        from huginn.agents.health_dashboard import HealthDashboard

        HealthDashboard.shared().record_call(
            tool_name, True, 0.0, cache_hit=True
        )
    except Exception:
        pass


class ToolAdapter:
    """Adapts HuginnTool instances to LangChain StructuredTool."""

    # Bounded cache for read-only tool outputs to improve cache hit rate and
    # reduce repeated token-heavy outputs in the agent context window.
    _read_only_cache: TimedLRUCache[dict[str, Any]] = TimedLRUCache(
        max_size=256, ttl=300.0
    )
    _constraint_adapter: ConstraintAdapter = ConstraintAdapter.default()
    # Class-level fallback summarizer. Instance-level ``self._summarizer``
    # takes priority so multi-agent setups don't clobber each other.
    _summarizer: Any = None

    def __init__(self) -> None:
        # Per-instance summarizer; preferred over the class-level fallback.
        self._summarizer: Any = None
        # 当前轮次的工具调用预算，由 agent 在 chat() 开始时 set 进来，
        # 结束后 clear。None 表示不限制。
        self._current_budget: Any = None
        # 最简路径决策路由, 跟 budget 同生命周期. None 表示不拦重型工具.
        self._current_router: Any = None
        # 工具调用循环检测器, 跟 budget / router 同生命周期. None 时不检测.
        # 抓 LLM 反复调同工具同参数的死循环, 跟 budget 互补.
        self._current_loop_detector: Any = None

    def set_budget(self, budget: Any) -> None:
        """设置当前轮次的工具调用预算，传 None 清除。"""
        self._current_budget = budget

    def set_router(self, router: Any) -> None:
        """设置当前轮次的 ToolCallRouter, 传 None 清除.

        router 为 None 时重型工具直接放行, 不做轻量路径 sanity check.
        """
        self._current_router = router

    def set_loop_detector(self, detector: Any) -> None:
        """设置当前轮次的 LoopDetector, 传 None 清除.

        detector 为 None 时跳过循环检测, 兼容老调用路径.
        """
        self._current_loop_detector = detector

    def set_summarizer(self, summarizer: Any) -> None:
        """Register an async summarizer for smart output compression.

        The summarizer should be an async callable that takes a text string
        and returns a summary. When set, large tool outputs will have their
        middle portion summarized instead of simply truncated.
        """
        self._summarizer = summarizer

    def adapt(
        self,
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

            lc_tool = ToolAdapter().adapt(StructureTool())
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
            scope = tool.constraint_scope

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

        def _needs_confirmation(input_data: BaseModel) -> str | None:
            """破坏性/高成本操作要用户确认, 返回提问文案; 不需要则 None."""
            try:
                if tool.is_destructive(input_data):
                    return f"工具 {tool.name} 将执行破坏性操作"
            except Exception:
                pass
            try:
                cost = tool.estimate_cost(input_data)
                if cost:
                    wt = cost.get("walltime_hours", 0)
                    cpu = cost.get("cpu_hours", 0)
                    if wt > 0.5 or cpu > 2:
                        return f"工具 {tool.name} 预计耗时 {wt:.1f}h ({cpu:.1f} CPU核时)"
            except Exception:
                pass
            return None

        async def _ask_confirmation(
            context: ToolContext, question: str
        ) -> bool:
            """通过 ClarificationManager 问用户, 返回是否确认."""
            try:
                from huginn.interaction.clarification import (
                    get_clarification_manager,
                )
                mgr = get_clarification_manager()
                tid = getattr(context, "session_id", None) or "default"
                answer = await mgr.ask(
                    thread_id=tid,
                    question=f"⚠️ {question}. 确认执行？",
                    options=["确认执行", "取消"],
                    context=f"工具 {tool.name} 调用前确认",
                    default_answer="取消",
                    timeout=120.0,
                )
                return answer == "确认执行"
            except Exception:
                # ClarificationManager 不可用时不阻断
                return True

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
                # 把 permission_config 透传给工具, 这样工具内部也能复用同一份配置
                # 做细粒度检查 (e.g. file_edit_tool 的 diff 预览强制化)
                config=permission_config,
            )
            payload = input_data.model_dump() if wants_dict else input_data
            return payload, context

        def _check_constraints(result: ToolResult, context: ToolContext) -> ToolResult:
            """Run domain constraints on successful tool outputs."""
            if not result.success:
                return result
            scope = tool.constraint_scope
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

        # Threshold (in tokens) above which a string value triggers LLM-based
        # smart compression instead of plain head/tail truncation.
        _smart_compress_threshold = max(2000, compression_max_tokens // 2)

        async def _smart_compress_output(obj: Any) -> Any:
            """Recursively compress large strings in ``obj`` using LLM summary.

            Walks dicts/lists and applies ``smart_compress_text`` to any string
            whose token count exceeds the threshold. The summarizer is resolved
            per-instance first, falling back to the class-level default so
            legacy callers without an instance still work. When neither is set,
            this is a no-op (the regular ``compress_tool_output`` path already
            handled truncation).
            """
            summarizer = self._summarizer if self._summarizer is not None else ToolAdapter._summarizer
            if summarizer is None:
                return obj
            if isinstance(obj, str):
                if count_tokens(obj) <= _smart_compress_threshold:
                    return obj
                try:
                    return await smart_compress_text(
                        obj,
                        max_tokens=_smart_compress_threshold,
                        summarizer=summarizer,
                    )
                except Exception as exc:
                    logger.debug("smart_compress_text failed for %s: %s", tool.name, exc)
                    return obj
            if isinstance(obj, dict):
                return {k: await _smart_compress_output(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [await _smart_compress_output(v) for v in obj]
            return obj

        def _sync_smart_compress(obj: Any) -> Any:
            """Sync entry point into _smart_compress_output for the _run path.

            Short-circuits when no summarizer is configured so we don't spin up
            an event loop for nothing.
            """
            summarizer = self._summarizer if self._summarizer is not None else ToolAdapter._summarizer
            if summarizer is None:
                return obj
            try:
                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    return asyncio.run(_smart_compress_output(obj))
                # We're inside a running loop already — use a fresh one.
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(_smart_compress_output(obj))
                finally:
                    loop.close()
            except Exception as exc:
                logger.debug("smart_compress sync wrapper failed for %s: %s", tool.name, exc)
                return obj

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

            # 硬确认门: 破坏性/高成本操作必须用户先点头, 不让 LLM 自己拍板.
            # HUGINN_AUTO_APPROVE=1 时跳过 (CI / benchmark 场景).
            if (
                os.environ.get("HUGINN_AUTO_APPROVE") != "1"
                and not permission_config.auto_approve_all
                and approval_callback is None
            ):
                confirm_q = _needs_confirmation(input_data)
                if confirm_q:
                    confirmed = await _ask_confirmation(context, confirm_q)
                    if not confirmed:
                        output = {
                            "error": f"用户取消了 {tool.name} 调用",
                            "_user_cancelled": True,
                        }
                        _audit(
                            input_data,
                            output,
                            approved=False,
                            reason="user_cancelled",
                        )
                        _publish(
                            PetMood.ERROR,
                            f"{tool.name} cancelled by user",
                            {"reason": confirm_q},
                        )
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
                _record_cache_hit(tool.name)
                return cached

            # 最简路径决策: 重型工具调用前先看轻量路径走没走过, 没走过就拦下
            # 把 reason 喂回 LLM, 让它要么先调轻量工具, 要么加 __confirm_heavy=true
            router = self._current_router
            if router is not None:
                allowed, router_reason = router.should_allow(
                    tool.name, kwargs, {}
                )
                if not allowed:
                    output = {
                        "error": router_reason,
                        "_router_blocked": True,
                    }
                    _audit(input_data, output, approved=False, reason="router_blocked")
                    _publish(
                        PetMood.ERROR,
                        f"{tool.name} blocked by router",
                        {"reason": router_reason},
                    )
                    return output

            # 工具调用预算检查：缓存命中不算，只有真正执行工具才计数
            budget = self._current_budget
            if budget is not None:
                if not budget.record(tool.name):
                    stop, reason = budget.should_stop()
                    output = {
                        "error": f"工具调用预算耗尽: {reason}",
                        "_budget_exceeded": True,
                    }
                    _audit(input_data, output, approved=True, reason="budget_exceeded")
                    _publish(
                        PetMood.ERROR,
                        f"{tool.name} blocked by budget",
                        {"reason": reason},
                    )
                    return output

            # 工具调用循环检测: 记一笔, 命中就拦下, 把 reason 喂回 LLM
            # 让它换思路. 缓存命中不算 (没真正执行), 跟 budget 同口径.
            loop_detector = self._current_loop_detector
            if loop_detector is not None:
                is_loop = loop_detector.record(tool.name, kwargs)
                if is_loop:
                    _, loop_reason = loop_detector.should_break()
                    output = {
                        "error": loop_reason,
                        "_loop_detected": True,
                    }
                    _audit(input_data, output, approved=False, reason="loop_detected")
                    _publish(
                        PetMood.ERROR,
                        f"{tool.name} blocked by loop detector",
                        {"reason": loop_reason},
                    )
                    return output

            # 熔断器前置检查: 连续失败的工具直接拒, 不浪费 timeout 去试
            blocked = _breaker_blocked(tool.name)
            if blocked is not None:
                _audit(input_data, blocked, approved=False, reason="circuit_open")
                _publish(
                    PetMood.ERROR,
                    f"{tool.name} circuit open",
                    {"retry_after": blocked.get("retry_after", 0)},
                )
                return blocked

            _publish(PetMood.WORKING, f"Running {tool.name}…", {"tool": tool.name})
            # 按工具类型分级超时，防止外部 API 卡死整个 agent
            timeout = get_timeout(tool.name)
            _call_start = time.time()
            with get_telemetry_collector().span("tool_call", tool=tool.name) as span:
                try:
                    if is_async:
                        result = await asyncio.wait_for(
                            tool.call(payload, context), timeout=timeout
                        )
                    else:
                        result = tool.call(payload, context)
                except asyncio.TimeoutError:
                    result = ToolResult(
                        data=None,
                        success=False,
                        error=f"{tool.name} timed out after {timeout}s",
                    )
                except Exception as exc:
                    # 非超时异常也得让熔断器/仪表盘看见, 记完再抛
                    _record_outcome(
                        tool.name, False, time.time() - _call_start, str(exc)
                    )
                    raise
                # 到这里 result 一定是 ToolResult (正常返回或超时)
                _record_outcome(
                    tool.name,
                    result.success,
                    time.time() - _call_start,
                    result.error if not result.success else None,
                )
                # 轻量工具执行完记一笔, 后续重型工具凭此放行.
                # record_light_attempt 内部会过滤非轻量工具, 这里无脑调即可.
                if router is not None:
                    router.record_light_attempt(tool.name)
                result = _check_constraints(result, context)
                output = _serialize(result)
                # LLM-based smart compression for very large text payloads.
                # Runs only when a summarizer has been registered.
                output = await _smart_compress_output(output)
                span.metadata["success"] = result.success
                # 记录 args/result 让 trajectory replay 能看到工具的真实输入输出
                span.metadata["args"] = _truncate_for_trajectory(payload)
                span.metadata["result"] = _truncate_for_trajectory(output)
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

            # 硬确认门 (同步版): 把 async _ask_confirmation 套进 event loop.
            # 跟上面 _smart_compress_output 同款套路: 先试 asyncio.run, 撞上
            # 已有 loop 就新开一个跑.
            if (
                os.environ.get("HUGINN_AUTO_APPROVE") != "1"
                and not permission_config.auto_approve_all
                and approval_callback is None
            ):
                confirm_q = _needs_confirmation(input_data)
                if confirm_q:
                    try:
                        try:
                            asyncio.get_running_loop()
                            loop = asyncio.new_event_loop()
                            try:
                                confirmed = loop.run_until_complete(
                                    _ask_confirmation(context, confirm_q)
                                )
                            finally:
                                loop.close()
                        except RuntimeError:
                            confirmed = asyncio.run(
                                _ask_confirmation(context, confirm_q)
                            )
                    except Exception:
                        # ClarificationManager 不可用时安全放行, 不阻断
                        confirmed = True
                    if not confirmed:
                        output = {
                            "error": f"用户取消了 {tool.name} 调用",
                            "_user_cancelled": True,
                        }
                        _audit(
                            input_data,
                            output,
                            approved=False,
                            reason="user_cancelled",
                        )
                        _publish(
                            PetMood.ERROR,
                            f"{tool.name} cancelled by user",
                            {"reason": confirm_q},
                        )
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
                _record_cache_hit(tool.name)
                return cached

            # 最简路径决策 (同步路径): 跟 _arun 同样的 sanity check
            router = self._current_router
            if router is not None:
                allowed, router_reason = router.should_allow(
                    tool.name, kwargs, {}
                )
                if not allowed:
                    output = {
                        "error": router_reason,
                        "_router_blocked": True,
                    }
                    _audit(input_data, output, approved=False, reason="router_blocked")
                    _publish(
                        PetMood.ERROR,
                        f"{tool.name} blocked by router",
                        {"reason": router_reason},
                    )
                    return output

            # 工具调用预算检查（同步路径）
            budget = self._current_budget
            if budget is not None:
                if not budget.record(tool.name):
                    stop, reason = budget.should_stop()
                    output = {
                        "error": f"工具调用预算耗尽: {reason}",
                        "_budget_exceeded": True,
                    }
                    _audit(input_data, output, approved=True, reason="budget_exceeded")
                    _publish(
                        PetMood.ERROR,
                        f"{tool.name} blocked by budget",
                        {"reason": reason},
                    )
                    return output

            # 工具调用循环检测 (同步路径): 跟 _arun 同样的口径, 缓存命中不算
            loop_detector = self._current_loop_detector
            if loop_detector is not None:
                is_loop = loop_detector.record(tool.name, kwargs)
                if is_loop:
                    _, loop_reason = loop_detector.should_break()
                    output = {
                        "error": loop_reason,
                        "_loop_detected": True,
                    }
                    _audit(input_data, output, approved=False, reason="loop_detected")
                    _publish(
                        PetMood.ERROR,
                        f"{tool.name} blocked by loop detector",
                        {"reason": loop_reason},
                    )
                    return output

            # 熔断器前置检查 (同步路径): 跟 _arun 同口径
            blocked = _breaker_blocked(tool.name)
            if blocked is not None:
                _audit(input_data, blocked, approved=False, reason="circuit_open")
                _publish(
                    PetMood.ERROR,
                    f"{tool.name} circuit open",
                    {"retry_after": blocked.get("retry_after", 0)},
                )
                return blocked

            _publish(PetMood.WORKING, f"Running {tool.name}…", {"tool": tool.name})
            # 同步路径也套分级超时
            timeout = get_timeout(tool.name)
            _call_start = time.time()
            with get_telemetry_collector().span("tool_call", tool=tool.name) as span:
                try:
                    if is_async:
                        try:
                            loop = asyncio.get_running_loop()
                        except RuntimeError:
                            result = asyncio.run(
                                asyncio.wait_for(
                                    tool.call(payload, context), timeout=timeout
                                )
                            )
                        else:
                            result = loop.run_until_complete(
                                asyncio.wait_for(
                                    tool.call(payload, context), timeout=timeout
                                )
                            )
                    else:
                        result = tool.call(payload, context)
                except asyncio.TimeoutError:
                    result = ToolResult(
                        data=None,
                        success=False,
                        error=f"{tool.name} timed out after {timeout}s",
                    )
                except Exception as exc:
                    # 非超时异常: 记完失败再抛, 同 _arun
                    _record_outcome(
                        tool.name, False, time.time() - _call_start, str(exc)
                    )
                    raise
                # 到这里 result 一定是 ToolResult (正常返回或超时)
                _record_outcome(
                    tool.name,
                    result.success,
                    time.time() - _call_start,
                    result.error if not result.success else None,
                )
                # 轻量工具执行完记一笔 (同步路径). 同 _arun, 无脑调即可.
                if router is not None:
                    router.record_light_attempt(tool.name)
                result = _check_constraints(result, context)
                output = _serialize(result)
                # LLM-based smart compression for very large text payloads.
                # Mirrors the async path; only runs when a summarizer is set.
                output = _sync_smart_compress(output)
                span.metadata["success"] = result.success
                span.metadata["args"] = _truncate_for_trajectory(payload)
                span.metadata["result"] = _truncate_for_trajectory(output)
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

    def adapt_registry(
        self,
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
                    self.adapt(
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
