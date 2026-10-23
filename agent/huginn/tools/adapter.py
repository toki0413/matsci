"""LangChain tool adapter — bridges HuginnTool to LangChain BaseTool.

EvoScientist/deepagents expects LangChain-compatible tools.
This adapter wraps our HuginnTool instances into StructuredTool
so they can be used in the Agent Loop.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import hashlib
import inspect
import json
import logging
import os
import re
import threading
import time
import types
import typing
from collections.abc import Callable
from pathlib import Path
from typing import Any, get_args, get_origin

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ValidationError

from huginn.constraints import ConstraintAdapter
from huginn.constraints.boundaries import BoundaryEvolution, BoundaryState
from huginn.core_types import (
    ErrorKind,
    PermissionMode,
    ToolContext,
    ToolResult,
)
from huginn.feature_flags import FeatureFlags
from huginn.permissions import PermissionConfig
from huginn.pet import PetMood, get_pet_bus
from huginn.privacy import redact_secrets
from huginn.security.audit import AuditLogger
from huginn.telemetry import get_telemetry_collector
from huginn.tools.base import HuginnTool
from huginn.tools.compress import compress_tool_output, smart_compress_text
from huginn.tools.timeouts import get_timeout
from huginn.utils.cache import TimedLRUCache
from huginn.utils.runtime import get_runtime_home
from huginn.utils.tokens import count_tokens

logger = logging.getLogger(__name__)


def _timeout_failure_hint(timeout: float | None = None) -> str:
    """超时失败的可操作提示 (与 code_lab 的超时专用口径一致).

    超时 = 算力预算耗尽, 不是崩溃. 把它译成 "sandbox/process crash" 会让 agent
    换方法而不是砍计算量, 从而在同一个太重的方案上反复撞墙 (ws11 观察到 8 次).
    """
    budget = f" (> {timeout}s)" if timeout else ""
    return (
        f"Execution timed out{budget}: compute budget exhausted, not a crash. "
        "Reduce iterations/resolution/parameter-grid size or split into smaller "
        "runs, then retry."
    )


# ── Tool economy — 让"工具瘦身"每天可见、可变、可调 ───────────────────────
# 进程内累计 (进程启动清零), 不持久化、跨 session 共享一份。纯展示埋点,
# 绝不碰执行路径。真实调用 vs 缓存/去重命中 vs 压缩省字符, 前端抽屉/路由只读。
_economy_lock = threading.Lock()
_economy: dict[str, int] = {
    "calls": 0,
    "cache_hits": 0,
    "dedupe_hits": 0,
    "chars_before": 0,
    "chars_after": 0,
}


def _economy_bump(**kw: int) -> None:
    """对若干计数器累加, 线程安全, best-effort 不抛。"""
    with _economy_lock:
        for k, v in kw.items():
            if v > 0 and k in _economy:
                _economy[k] += v


def _estimate_chars(o: Any) -> int:
    """粗略估算对象体量的字符数, 只用于省量对比, 不追求精确。"""
    try:
        return len(json.dumps(o, default=str, ensure_ascii=False))
    except Exception:
        return len(str(o))


def tool_economy_snapshot() -> dict[str, Any]:
    """返回当前工具经济快照。tokens 省量是字符差的粗略 4:1 换算。"""
    with _economy_lock:
        saved_chars = max(0, _economy["chars_before"] - _economy["chars_after"])
        return dict(_economy, chars_saved=saved_chars, tokens_saved=saved_chars // 4)


# ── 可操作错误信息 — 把裸异常转成人话 + 下一步 ─────────────────────────
# 工具失败时 _serialize 会把 result.error 原样回喂给 LLM, 常常是一长串
# 没人读的堆栈/路径。这里是模块级纯函数: 对常见模式做一次性改写, 识别
# 不了就原样返回 (不掩盖原始文本, 宁可不改也不丢信息)。原则: 只在汇聚
# 点改一处, 而不是在几十个工具里各写一遍。
_ERR_HINTS = (
    # (regex, 人话一句话, 建议下一步)
    (re.compile(r"Permission denied(?:'[^']*')?", re.I),
     "目标文件/目录没有写权限",
     "检查文件属主和读写权限, 或换到有写权限的路径"),
    (re.compile(r"FileNotFoundError|No such file|ENOENT", re.I),
     "目标路径不存在",
     "确认路径拼写和相对/绝对目录, 或先创建目录再写"),
    (re.compile(r"TimeoutError|timed out|Timeout while await|> processor timeout", re.I),
     "操作超时",
     "缩小输入规模、加大 timeout, 或把大任务拆小分步跑"),
    (re.compile(r"Too many open files|EMFILE|Resource temporarily unavailable", re.I),
     "系统文件描述符耗尽",
     "释放不再用的句柄, 减少并发, 或重启 agent 进程"),
    (re.compile(r"Broken pipe|Connection (reset|closed|refused)|Errno 32", re.I),
     "底层连接被断开",
     "重试一次; 若持续失败检查网络/服务端是否还在"),
    (re.compile(r"Out of memory|Killed|No memory|CUDA out of memory", re.I),
     "内存/显存不足",
     "减小单次数据量或 batch, 给计算程序配更小的内存占用"),
    (re.compile(r"TypeError:|ValueError:", re.I),
     "参数类型或取值不对",
     "核对传入参数的类型和取值范围, 修正后再调"),
    (re.compile(r"ModuleNotFoundError|ImportError", re.I),
     "缺少依赖模块",
     "安装缺失的依赖包后重试"),
    (re.compile(r"ENOSPC|No space left on device", re.I),
     "磁盘空间不足",
     "清理临时文件/中间产物, 释放磁盘后再跑"),
    (re.compile(r"Segmentation fault|EINSTRUCTION|assert .* failed", re.I),
     "程序崩溃(非脚本逻辑问题)",
     "多半是底层/砂箱问题, 换一种实现或换工具绕过"),
)

def humanize_tool_error(raw: str) -> str:
    """把工具返回的裸错误改写成'一句话 + 下一步'。

    只匹配上面清单里的常见模式, 命中就在开头加两行 `可操作` 提示;
    没命中就原样返回。纯函数、无副作用、不回填、不丢原文。
    """
    if not raw:
        return raw
    for pattern, hint, action in _ERR_HINTS:
        if pattern.search(raw):
            return f"[{hint}] {action}。原文: {raw}"
    return raw


# 插件事件总线 (typed Event 分发, api/event.py 的 ToolCallEvent/ToolRespondEvent).
# 跟 events/integration.py 的内部字符串事件总线是两套, 这里只管插件系统.
# 懒加载共享 registry; 拿不到 bus 时返回 None, 调用方判空跳过, 不阻断 tool.
_plugin_event_bus: Any = None


def _get_plugin_event_bus() -> Any:
    """懒加载插件 EventBus. 失败返回 None."""
    global _plugin_event_bus
    if _plugin_event_bus is not None:
        return _plugin_event_bus
    try:
        from huginn.plugins.event_bus import EventBus

        _plugin_event_bus = EventBus()
    except Exception:
        logger.debug(
            "plugin EventBus unavailable, typed tool events skipped",
            exc_info=True,
        )
        return None
    return _plugin_event_bus


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
    if isinstance(value, list | tuple):
        return [_truncate_for_trajectory(v) for v in value]
    return value


# ── 参数边界校验 (不可信的 LLM 参数 → 工具 schema) ──────────────────────
# 模型给出的参数常常"语义对、形状错": 该给数组的地方给一个 JSON 字符串
# ('["ls","-la"]'), 该给整数的地方给 "5". langchain 在 _parse_input 里用
# args_schema 严格校验, 形状不符直接抛 ValidationError, 调用在到达工具之前
# 就死了 —— 既没有修复机会, 也没有遥测. 这里补两层:
#   1) _coerce_tool_args: 按声明类型做一次可解释的反序列化, 只修形状;
#   2) _validate_tool_args: 严格校验 → 一次修复重试 → 结构化错误回执.
# 安全边界: 只接受 json / 字面量反序列化, 绝不 eval; 只改形状不猜语义 —
# 解析不了的字符串原样交给严格校验, 由它报错 (不静默改写用户数据).
_NO_PARSE = object()
# 反序列化体量上限: 防超长字符串把 json/literal 解析拖成 CPU 黑洞.
_ARG_PARSE_LIMIT = 200_000


def _jsonish(raw: str) -> Any:
    """把形状错误的字符串按 JSON / Python 字面量反序列化, 失败返回 _NO_PARSE."""
    s = raw.strip()
    if not s or len(s) > _ARG_PARSE_LIMIT:
        return _NO_PARSE
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        pass
    try:
        # 仅字面量解析 (字符串/数字/容器/布尔/None), 不执行任何代码.
        return ast.literal_eval(s)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return _NO_PARSE


def _matches_annotation(value: Any, annotation: Any) -> bool:
    """value 是否已经符合 annotation 的形状 (用于 union 短路, 避免过度改写)."""
    origin = get_origin(annotation)
    if origin is not None:
        if origin is list:
            return isinstance(value, list)
        if origin is dict:
            return isinstance(value, dict)
        if origin is tuple:
            return isinstance(value, tuple)
        if origin in (set, frozenset):
            return isinstance(value, (set, frozenset))
        return False
    try:
        return isinstance(value, annotation)
    except TypeError:
        # Literal / 复杂注解无法用 isinstance 判定 → 视作匹配, 不冒险改动.
        return True


def _coerce_for_annotation(raw: Any, annotation: Any) -> tuple[Any, str]:
    """按声明类型修一次形状. 返回 (值, 修复说明); 无需修复时说明为空串."""
    origin = get_origin(annotation)

    if origin in (types.UnionType, typing.Union):
        args = [a for a in get_args(annotation) if a is not type(None)]
        for arg in args:
            if _matches_annotation(raw, arg):
                return raw, ""
        for arg in args:
            value, note = _coerce_for_annotation(raw, arg)
            if note:
                return value, note
        return raw, ""

    if origin is dict:
        if isinstance(raw, str):
            parsed = _jsonish(raw)
            if parsed is _NO_PARSE or not isinstance(parsed, dict):
                return raw, ""
            return parsed, "str→dict (JSON 反序列化)"
        return raw, ""

    if origin in (list, tuple, set, frozenset):
        container = origin
        if isinstance(raw, str):
            parsed = _jsonish(raw)
            if parsed is _NO_PARSE or not isinstance(
                parsed, (list, tuple, set, frozenset)
            ):
                return raw, ""
            value = parsed if container is list else container(parsed)
            return value, f"str→{container.__name__} (JSON 反序列化)"
        if container is list and isinstance(raw, tuple):
            return list(raw), "tuple→list"
        if container is tuple and isinstance(raw, list):
            return tuple(raw), "list→tuple"
        if container in (set, frozenset) and isinstance(raw, list):
            return container(raw), f"list→{container.__name__}"
        return raw, ""

    if isinstance(raw, str):
        text = raw.strip()
        if annotation is bool:
            low = text.lower()
            if low in {"true", "1", "yes", "on"}:
                return True, "str→bool"
            if low in {"false", "0", "no", "off"}:
                return False, "str→bool"
            return raw, ""
        if annotation is int:
            try:
                return int(text), "str→int"
            except ValueError:
                return raw, ""
        if annotation is float:
            try:
                return float(text), "str→float"
            except ValueError:
                return raw, ""
        return raw, ""

    if annotation is float and isinstance(raw, int) and not isinstance(raw, bool):
        return float(raw), "int→float"

    return raw, ""


def _coerce_tool_args(
    tool: HuginnTool, kwargs: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """对 kwargs 逐一按 schema 声明类型修形状, 返回 (新 kwargs, 修复说明列表)."""
    schema = getattr(tool, "input_schema", None)
    fields = getattr(schema, "model_fields", None) or {}
    if not fields:
        return dict(kwargs), []

    out = dict(kwargs)
    notes: list[str] = []
    for name, field in fields.items():
        if name not in out or out[name] is None:
            continue
        annotation = getattr(field, "annotation", None)
        if annotation is None:
            continue
        try:
            value, note = _coerce_for_annotation(out[name], annotation)
        except Exception:  # 防御: 单个字段修复失败不影响其余字段
            logger.debug("arg coercion failed for %s.%s", tool.name, name, exc_info=True)
            continue
        if note:
            out[name] = value
            notes.append(f"{name}: {note}")
    return out, notes


def _annotation_example(annotation: Any, depth: int = 0) -> Any:
    """给 LLM 看的字段形状示例. 只覆盖常见类型, 认不出返回 "..."."""
    if depth > 2 or annotation is None:
        return "..."
    origin = get_origin(annotation)
    if origin in (types.UnionType, typing.Union):
        for arg in get_args(annotation):
            if arg is not type(None):
                return _annotation_example(arg, depth + 1)
        return "..."
    if origin is list:
        args = get_args(annotation)
        return [_annotation_example(args[0], depth + 1)] if args else []
    if origin is tuple:
        return []
    if origin in (set, frozenset):
        return []
    if origin is dict:
        return {}
    if origin is typing.Literal:
        args = get_args(annotation)
        return args[0] if args else "..."
    return {str: "string", int: 0, float: 0.0, bool: False, dict: {}, list: []}.get(
        annotation, "..."
    )


def _preview_value(value: Any, limit: int = 200) -> str:
    """错误回执里的输入预览: 限长, 不把整段输入回灌给模型."""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, default=str, ensure_ascii=False)
        except Exception:
            text = str(value)
    return text[:limit] + ("…" if len(text) > limit else "")


def _invalid_args_output(
    tool: HuginnTool,
    exc: BaseException,
    repairs: list[str],
) -> dict[str, Any]:
    """把参数校验失败转成模型能自修的结构化回执 (而不是裸 ValidationError)."""
    detail: list[dict[str, Any]] = []
    raw_errors = getattr(exc, "errors", None)
    if callable(raw_errors):
        try:
            for err in raw_errors()[:5]:
                loc = ".".join(str(p) for p in err.get("loc", ()))
                detail.append(
                    {
                        "field": loc or "<args>",
                        "problem": str(err.get("msg", ""))[:300],
                        "got": _preview_value(err.get("input")),
                    }
                )
        except Exception:  # 防御: 非标准异常对象, 退回字符串
            detail = []
    if not detail:
        detail = [{"field": "<args>", "problem": str(exc)[:500]}]

    fields = getattr(getattr(tool, "input_schema", None), "model_fields", None) or {}
    expected = {name: _annotation_example(getattr(f, "annotation", None)) for name, f in fields.items()}

    return {
        "error": f"invalid_tool_args: {tool.name} 参数不符合 schema, 本次调用未执行",
        "_invalid_args": True,
        "tool": tool.name,
        "detail": detail,
        "expected_shape": expected,
        "repairs_attempted": repairs,
        "hint": (
            "按 expected_shape 重新发起调用: 数组/对象字段必须是真实 JSON 数组/对象, "
            "不能是 JSON 字符串; 缺失的必填字段必须补齐."
        ),
    }


def _validate_tool_args(
    tool: HuginnTool, kwargs: dict[str, Any]
) -> tuple[Any | None, dict[str, Any] | None, list[str]]:
    """边界校验: 严格 → 一次修复重试 → 结构化错误.

    返回 ``(input_data, error_output, repairs)``。error_output 非空表示本次
    调用不能执行, 调用方应原样回给模型 (含修复建议), 而不是让异常穿透。
    """
    schema = getattr(tool, "input_schema", None)
    try:
        return schema(**kwargs), None, []  # type: ignore[misc]
    except ValidationError as first:
        repaired, notes = _coerce_tool_args(tool, kwargs)
        if notes:
            try:
                return schema(**repaired), None, notes  # type: ignore[misc]
            except ValidationError as second:
                return None, _invalid_args_output(tool, second, notes), notes
        return None, _invalid_args_output(tool, first, []), []
    except (TypeError, ValueError) as exc:
        return None, _invalid_args_output(tool, exc, []), []


def _emit_span(
    tool_name: str,
    args: Any,
    output: Any,
    *,
    stage: str,
    started: float | None = None,
    repairs: list[str] | None = None,
) -> None:
    """为"执行前就被拦下"的调用补一条 tool_call span.

    之前 span 只在真正执行工具时开, 于是参数非法 / 权限拒绝 / 预算耗尽 /
    校验失败这些路径在遥测里完全不可见 —— 失败率被系统性低估. 这里让每条
    调用路径都恰好产生一条 span, success 由回执内容判定.
    """
    try:
        with get_telemetry_collector().span("tool_call", tool=tool_name) as span:
            failed = isinstance(output, dict) and (
                "error" in output or output.get("_user_cancelled") is True
            )
            span.metadata["success"] = not failed
            span.metadata["stage"] = stage
            span.metadata["args"] = _truncate_for_trajectory(args)
            span.metadata["result"] = _truncate_for_trajectory(output)
            if started is not None:
                span.metadata["latency_ms"] = round((time.time() - started) * 1000, 1)
            if repairs:
                span.metadata["arg_repairs"] = repairs
            if failed and isinstance(output, dict):
                span.metadata["error"] = str(output.get("error") or output)[:2000]
    except Exception:  # 遥测必须 fail-open, 绝不阻断工具路径
        logger.debug("pre-execution telemetry span failed (fail-open)", exc_info=True)


class _AdapterStructuredTool(StructuredTool):
    """StructuredTool 的运行时校验让位给 adapter 的修复层.

    langchain_core 的 ``_parse_input`` 会在 wrapper 之前用 args_schema 严格
    校验 dict 参数, 形状不符直接抛 ValidationError —— 我们的修复层根本没机会
    看到参数. 这里保留 happy path 完全不变 (仍然走父类校验, 保证默认值/injected
    参数行为一致), 只在校验失败时把原始 dict 透传下去, 由 ``_validate_tool_args``
    做一次修复并提供可自修的错误回执.
    """

    def _parse_input(self, tool_input: Any, tool_call_id: str | None) -> Any:
        try:
            return super()._parse_input(tool_input, tool_call_id)
        except ValidationError as exc:
            if isinstance(tool_input, dict):
                logger.debug(
                    "deferring arg validation to adapter (%s): %s", self.name, exc
                )
                return dict(tool_input)
            raise


def _default_audit_logger() -> AuditLogger:
    """Return a default audit logger for tool invocations."""
    from pathlib import Path

    base = os.environ.get("HUGINN_CACHE_DIR")
    log_path = (
        Path(base) / "audit.jsonl" if base else get_runtime_home() / "audit.jsonl"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return AuditLogger(log_path)


# 熔断器/仪表盘开关. AV5: 默认关 — file_read_tool 偶尔失败就触发 circuit_open,
# 阻止 agent 读文件, 生产路径 (deli_research/cli/routes) 也踩同一个坑.
# 升级路径: mode-aware breaker, 按 tool 配置不同阈值 (file_read 容忍度高, hpc 容忍度低).
_HEALTH_MONITOR_ON = os.environ.get("HUGINN_HEALTH_MONITOR", "0") == "1"


def _is_heavy_tool(tool_name: str) -> bool:
    """判断是否为重型/sim 工具. 这类工具失败代价高 (DFT 跑 24h 才发现挂了),
    应默认受熔断保护, 不受全局 HUGINN_HEALTH_MONITOR=0 影响."""
    try:
        from huginn.agents.tool_call_router import HEAVY_TOOLS

        if tool_name in HEAVY_TOOLS:
            return True
    except Exception:
        logger.debug("heavy tool lookup failed", exc_info=True)
    # 兜底: 常见 sim 工具名硬编码, 防 HEAVY_TOOLS 表未重建时漏判.
    return tool_name in {
        "vasp_tool",
        "lammps_tool",
        "gaussian_tool",
        "orca_tool",
        "qe_tool",
        "cp2k_tool",
        "comsol_tool",
        "abaqus_tool",
        "openfoam_tool",
        "fenics_tool",
        "gromacs_tool",
    }


def _breaker_blocked(tool_name: str) -> dict[str, Any] | None:
    """熔断器开着就返回错误 dict，没装或放行返回 None。"""
    # 全局开关关时, 仍对 heavy/sim 工具生效 — 这类工具连续失败代价过高.
    if not _HEALTH_MONITOR_ON and not _is_heavy_tool(tool_name):
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
        logger.debug("suppressed in _breaker_blocked", exc_info=True)
    return None


def _record_outcome(
    tool_name: str,
    success: bool,
    duration_sec: float,
    error: str | None = None,
) -> None:
    """工具执行完记一笔到熔断器 + 仪表盘，best-effort 不抛。"""
    _economy_bump(calls=1)  # 真实执行计数, economy 不随健康监控开关走
    if not _HEALTH_MONITOR_ON and not _is_heavy_tool(tool_name):
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
        logger.debug("suppressed in _record_outcome", exc_info=True)


def _record_cache_hit(tool_name: str) -> None:
    """缓存命中记一笔到仪表盘（不算熔断器的成败）。"""
    _economy_bump(cache_hits=1)
    if not _HEALTH_MONITOR_ON:
        return
    try:
        from huginn.agents.health_dashboard import HealthDashboard

        HealthDashboard.shared().record_call(tool_name, True, 0.0, cache_hit=True)
    except Exception:
        logger.debug("suppressed in _record_cache_hit", exc_info=True)


# ── Hashline 自动闭环 ─────────────────────────────────────────────────────
# P0-2 wiring: 让"防并发覆盖"成为系统保证而非 LLM 自觉。
# 缓存"agent 上次看到的文件内容 hash"，编辑前自动注入 expected_hash。
# 若磁盘内容在上次 read/edit 后被外部进程改动，编辑被严格拒绝，避免覆盖他人工作。
#
# 缓存键是解析后的绝对路径。跨进程/session 共享一个简单 dict 即可——hash 校验
# 是"保守优先"：未命中缓存时不注入（行为与旧版一致），命中时注入使校验生效。
_HASHLINE_CACHE: dict[str, str] = {}
_HASHLINE_TOOLS: frozenset[str] = frozenset(
    {
        "file_read_tool",
        "file_write_tool",
        "file_edit_tool",
        "multi_edit_tool",
        "lsp_tool",
    }
)


def _hashline_abs_path(file_path: str, working_dir: str | None) -> str | None:
    """Resolve a tool's file_path to an absolute path (or None on failure)."""
    try:
        p = Path(file_path) if file_path else None
        if p is None:
            return None
        if not p.is_absolute():
            base = Path(working_dir) if working_dir else Path.cwd()
            p = base / p
        return str(p.resolve())
    except Exception:
        return None


def _hashline_hash(path: str) -> str | None:
    """Compute the same short hash file_edit_tool uses (_content_hash)."""
    try:
        text = Path(path).read_text(encoding="utf-8")
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    except Exception:
        return None


def _maybe_inject_hashline(tool_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Auto-inject expected_hash from the session-level file-hash cache.

    Only injects when the caller did not already provide expected_hash and the
    cache has a recorded hash for the target file. No cache hit → no injection
    → behavior identical to pre-wiring (backward compatible).
    """
    if tool_name not in _HASHLINE_TOOLS:
        return kwargs

    kw = dict(kwargs)
    wd = kw.get("working_dir")

    if tool_name == "file_edit_tool":
        if kw.get("expected_hash") is None:
            ap = _hashline_abs_path(kw.get("file_path", ""), wd)
            if ap and ap in _HASHLINE_CACHE:
                kw["expected_hash"] = _HASHLINE_CACHE[ap]
        return kw

    if tool_name == "multi_edit_tool":
        edits = kw.get("edits")
        if isinstance(edits, list) and edits:
            new_edits = []
            for e in edits:
                ed = e if isinstance(e, dict) else dict(e)
                if ed.get("expected_hash") is None:
                    ap = _hashline_abs_path(ed.get("file_path", ""), wd)
                    if ap and ap in _HASHLINE_CACHE:
                        ed["expected_hash"] = _HASHLINE_CACHE[ap]
                new_edits.append(ed)
            kw["edits"] = new_edits
        return kw

    return kw


def _update_hashline_cache(tool_name: str, input_data: Any, context: Any) -> None:
    """After a successful read/write/edit, record the file's current hash.

    This is the version the agent now "sees"; a later external modification
    will diverge from it and be caught by the injected expected_hash.
    """
    if tool_name not in _HASHLINE_TOOLS or not input_data:
        return
    wd = getattr(input_data, "working_dir", None) or getattr(context, "workspace", None)
    files: list[str] = []
    if tool_name == "multi_edit_tool":
        edits = getattr(input_data, "edits", None)
        if edits:
            files = [e.file_path for e in edits if getattr(e, "file_path", None)]
    else:
        fp = getattr(input_data, "file_path", None)
        if fp:
            files = [fp]
    for f in files:
        ap = _hashline_abs_path(f, wd)
        if ap is None:
            continue
        h = _hashline_hash(ap)
        if h is not None:
            _HASHLINE_CACHE[ap] = h


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
        # 同 step 工具调用去重, 跟 budget / router 同生命周期. None 时不去重.
        # 对标 Kimi Code toolDedupe: 同 (tool, args) 命中即返回上次结果.
        self._current_deduper: Any = None
        # agent 反向引用 — _serialize 把 _visual_base64 存到 agent 实例上,
        # chat 模式不污染上下文, visual_inspect 路径能拿到.
        self._agent_ref: Any = None

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

    def set_deduper(self, deduper: Any) -> None:
        """设置当前轮次的 ToolDeduper, 传 None 清除.

        deduper 为 None 时跳过去重, 兼容老调用路径.
        """
        self._current_deduper = deduper

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
                os.environ.get(
                    "HUGINN_TOOL_COMPRESSION_MAX_TOKENS", str(max_tool_output_tokens)
                )
            )

        def _check_permission(
            input_data: BaseModel, session_id: str = "default"
        ) -> tuple[bool, str | None]:
            """Return (approved, reason_or_none).

            Dangerous-command patterns are checked FIRST, before any
            auto_approve_all bypass — yolo mode must never silently
            execute rm -rf / or equivalent.
            """
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

            # Dangerous pattern check — overrides auto_approve_all.
            # This is the last-resort guard against catastrophic commands.
            args = input_data.model_dump() if hasattr(input_data, "model_dump") else {}
            from huginn.permissions import PermissionChecker

            checker = PermissionChecker(permission_config)
            is_dangerous, matched = checker._check_dangerous(name, args)
            if is_dangerous:
                reason = (
                    f"Tool '{name}' matches dangerous pattern '{matched}' — "
                    "requires explicit approval even in auto-approve mode"
                )
                if approval_callback is not None:
                    if approval_callback(name, reason):
                        return True, None
                    return False, f"User denied: {reason}"
                return False, reason

            # Second layer: command_filter for broader pattern coverage
            if name == "bash_tool":
                from huginn.security.command_filter import check_command_safety

                cmd = args.get("command", [])
                safety = check_command_safety(cmd)
                if not safety.is_safe:
                    reason = (
                        f"Command blocked by safety filter (pattern: "
                        f"{safety.matched_pattern}). Requires explicit approval."
                    )
                    if approval_callback is not None:
                        if approval_callback(name, reason):
                            return True, None
                        return False, f"User denied: {reason}"
                    return False, reason

            if permission_config.auto_approve_all or mode == PermissionMode.AUTO:
                return True, None

            if mode == PermissionMode.DENY:
                return False, f"Tool '{name}' is blocked by permission policy"

            # P4-2: Standing rules — (tool, target) 维度常驻授权.
            # ASK 模式下先查 standing rules, 命中则放行 (不调 approval_callback).
            # HUGINN_STANDING_RULES=1 启用, 默认 off (向后兼容).
            if os.environ.get("HUGINN_STANDING_RULES") == "1":
                from huginn.permissions import (
                    extract_target_from_args,
                    get_standing_rules_store,
                )

                _target = extract_target_from_args(args)
                _store = get_standing_rules_store()
                if _store.is_granted(session_id, name, _target):
                    return True, None

            reasons: list[str] = []
            try:
                if tool.is_destructive(input_data):
                    reasons.append("this operation is destructive")
            except Exception:
                logger.debug("suppressed in _check_permission", exc_info=True)

            try:
                cost = tool.estimate_cost(input_data)
                if cost:
                    cpu = cost.get("cpu_hours", 0)
                    if cpu > 1:
                        reasons.append(f"estimated cost: {cpu:.1f} CPU hours")
            except Exception:
                logger.debug("suppressed in _check_permission", exc_info=True)

            reason = f"Tool '{name}' requires approval"
            if reasons:
                reason += f" ({', '.join(reasons)})"

            if approval_callback is not None:
                if approval_callback(name, reason):
                    # P4-2: approval 通过后记录 standing rule, 下次同 tool+target 自动放行.
                    # HUGINN_STANDING_RULES=1 启用, 默认 off.
                    if os.environ.get("HUGINN_STANDING_RULES") == "1":
                        from huginn.permissions import (
                            extract_target_from_args,
                            get_standing_rules_store,
                        )

                        _target = extract_target_from_args(args)
                        get_standing_rules_store().grant(session_id, name, _target)
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
                logger.debug("suppressed in _needs_confirmation", exc_info=True)
            try:
                cost = tool.estimate_cost(input_data)
                if cost:
                    wt = cost.get("walltime_hours", 0)
                    cpu = cost.get("cpu_hours", 0)
                    if wt > 0.5 or cpu > 2:
                        return (
                            f"工具 {tool.name} 预计耗时 {wt:.1f}h ({cpu:.1f} CPU核时)"
                        )
            except Exception:
                logger.debug("suppressed in _needs_confirmation", exc_info=True)
            return None

        async def _ask_confirmation(context: ToolContext, question: str) -> bool:
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
                logger.warning("ClarificationManager unavailable, failing closed")
                return False

        def _build_inputs(
            input_data: BaseModel | dict[str, Any],
        ) -> tuple[BaseModel | dict[str, Any], ToolContext]:
            """Attach the execution context to an already-validated input.

            Schema construction moved out (``_validate_tool_args``) so a failed
            validation can be answered with a repair receipt instead of raising
            inside the wrapper.
            """
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
                # v7: 透传父 agent 的 approval_callback, subagent_tool 用它给子 agent.
                approval_callback=approval_callback,
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
                # 可操作错误: 把裸异常转成人话+下一步, 让 LLM 不再啃一长串堆栈.
                # 识别不了就原样返回, 不掩盖原文.
                data["error"] = humanize_tool_error(data["error"])
                # 兜底: 从 data 字典里抓 stderr/stdout 作为 error_detail.
                # 之前用 getattr(result, "stderr") 是死代码 — ToolResult 没这字段.
                # bash_tool/code_tool 把 stderr 放在 data 里, 这里抽出来给 agent 看.
                # ponytail: data 可能是 dict 或 None, 只在 dict 路径抽.
                _data_obj = result.data if isinstance(result.data, dict) else None
                if _data_obj:
                    _stderr = _data_obj.get("stderr") or ""
                    _stdout = _data_obj.get("stdout") or ""
                    _tail = (_stderr or _stdout or "")[-2000:]
                    if _tail and "error_detail" not in data:
                        data["error_detail"] = _tail
                # P0-7: 静默失败检测 — success=False 且无 stderr/stdout/error_detail
                # = 工具静默崩溃 (Rust sandbox 典型模式). 之前只返回 "Unknown error"
                # 让 agent 耗死重试. 加可操作提示让 agent 换路径.
                if (
                    not data.get("error_detail")
                    and data.get("error") == "Unknown error"
                ):
                    if result.error_kind == ErrorKind.TIMEOUT:
                        _d = result.data if isinstance(result.data, dict) else {}
                        data["error"] = _timeout_failure_hint(_d.get("timeout"))
                        data["timed_out"] = True
                    else:
                        data["error"] = (
                            "Unknown error (silent failure — no stderr/stdout captured). "
                            "This is likely a sandbox/process crash, not a code bug. "
                            "Try a different approach or tool."
                        )
            # ARGUS: tool 输出统一标 source_class, 下游 PhaseGate 可识别.
            # 顶层 _source_class 不进 LLM-visible content, 只作元数据.
            data["_source_class"] = "tool_output"

            # Pass through resolution requests so the agent loop can ask the user.
            if result.metadata.get("needs_resolution"):
                data["metadata"] = result.metadata
            # 透传 mock 标记: sim 工具可执行文件缺失时返回 success=True 的假数据,
            # 下游 dedupe/telemetry 需据此识别并跳过缓存. 参见 tool_dedupe._is_mock_result.
            if result.metadata.get("mock") is True:
                data["metadata"] = result.metadata

            data = _sanitize_and_compress(data)

            # 磁盘卸载: 压缩后仍然超长的输出存文件, 只留预览
            try:
                from huginn.tools.compress import offload_tool_output

                serialized = json.dumps(data, default=str, ensure_ascii=False)
                if count_tokens(serialized) > 20000:
                    preview, artifact_path = offload_tool_output(
                        serialized,
                        tool.name,
                    )
                    data = {
                        "_offloaded": True,
                        "_preview": preview,
                        "_artifact_path": artifact_path,
                        "_full_size_chars": len(serialized),
                    }
                    # ObservationPack: 同时注册稳定句柄, 模型可 recall 精确原文分页.
                    # 失败静默退化到纯预览(与 offload 同款 best-effort).
                    try:
                        from huginn.tools.observation_pack import handle_for_body

                        sess = getattr(self, "_session_id", None) or getattr(
                            getattr(self, "_agent_ref", None), "session_id", None
                        )
                        h = handle_for_body(serialized, session_id=sess)
                        if h:
                            data["_recall_handle"] = h
                    except Exception:
                        logger.debug("observation_pack handle failed (non-fatal)", exc_info=True)
            except Exception:
                logger.debug("offload_tool_output failed (non-fatal)", exc_info=True)

            # Auto-render numerical results as chart for VLM analysis
            try:
                from huginn.tools.visual_hook import enrich_with_visual

                data = enrich_with_visual(tool.name, data)
                # _visual_base64 是给多模态 LLM 看的, 但 chat 模式下 ToolMessage
                # 会把它序列化成字符串污染上下文 (base64 很长). 这里 pop 出来
                # 存到 agent 实例, visual_inspect 路径能拿到, chat 模式走 primitives.
                # DeepSeek-OCR 启发: 解码器就是 LLM, 但 base64 不该进 text context.
                # ponytail: pop + setattr. 升级: ToolMessage content 改成 multimodal list.
                b64 = data.pop("_visual_base64", None)
                if b64 and hasattr(self, "_agent_ref") and self._agent_ref is not None:
                    with contextlib.suppress(Exception):
                        self._agent_ref._last_visual_base64 = b64
            except Exception:
                logger.debug("best-effort op failed", exc_info=True)  # non-fatal

            return data

        def _sanitize_and_compress(obj: Any) -> Any:
            _eco_before = _estimate_chars(obj)
            # Strings: privacy redaction first, then token-aware truncation.
            if isinstance(obj, str):
                s = obj
                if FeatureFlags.shared().is_enabled("privacy_redact_secrets"):
                    s = redact_secrets(s)
                out = compress_tool_output(
                    s,
                    max_output_tokens=compression_max_tokens,
                    tool_name=tool.name,
                )
            else:
                # Everything else: structured compression (numeric summaries,
                # list head/tail, long-text truncation).
                out = compress_tool_output(
                    obj,
                    max_output_tokens=compression_max_tokens,
                    tool_name=tool.name,
                )
            _economy_bump(chars_before=_eco_before, chars_after=_estimate_chars(out))
            return out

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
                logger.debug("best-effort op failed", exc_info=True)

        def _publish(
            mood: PetMood, message: str, details: dict[str, Any] | None = None
        ) -> None:
            with contextlib.suppress(Exception):
                get_pet_bus().publish(mood, message, details)

        # Tool-name → fine-grained pet mood classification.
        # 用**注册名**: 早期写的 `python_tool`/`terminal_tool`/`list_tool`/`read_file`
        # 等与注册表对不上, 分类静默退化成 WORKING (MECE 审计记的允许表死项).
        _CODING_TOOLS = frozenset(
            {
                "code_tool",
                "bash_tool",
                "notebook_edit_tool",
            }
        )
        _REVIEWING_TOOLS = frozenset(
            {
                "file_read_tool",
                "grep",
                "glob",
                "web_search_tool",
                "lean_tool",
            }
        )

        def _classify_mood(tool_name: str) -> PetMood:
            if tool_name in _CODING_TOOLS:
                return PetMood.CODING
            if tool_name in _REVIEWING_TOOLS:
                return PetMood.REVIEWING
            return PetMood.WORKING

        def _publish_blocked(
            tool_name: str, input_data: Any, reason: str, context: Any
        ) -> None:
            """发布 tool.blocked 事件到事件总线 (统一走 UnifiedBus)."""
            with contextlib.suppress(Exception):
                from huginn.events.unified_bus import publish_event

                publish_event(
                    "tool.blocked",
                    {"tool": tool_name, "reason": reason},
                    thread_id=getattr(context, "thread_id", ""),
                    source="tool_adapter",
                )

        async def _dispatch_tool_call_event(
            args_dict: dict[str, Any], session_id: str
        ) -> None:
            """分发插件系统 ToolCallEvent (typed). best-effort, 失败不阻断.

            跟 _run_post_checks 里的 publish_tool_event_sync (内部 tool.call 字符串
            事件) 是两套并行的事件系统, 这里补的是 api/event.py 的 typed Event.
            """
            bus = _get_plugin_event_bus()
            if bus is None:
                return
            try:
                from huginn.api.event import EventType, ToolCallEvent

                await bus.dispatch(
                    ToolCallEvent(
                        type=EventType.ON_TOOL_CALL,
                        tool_name=tool.name,
                        args=args_dict,
                        session_id=session_id,
                    )
                )
            except Exception:
                logger.debug("ToolCallEvent dispatch failed (non-fatal)", exc_info=True)

        async def _dispatch_tool_respond_event(
            args_dict: dict[str, Any], result: Any, success: bool
        ) -> None:
            """分发插件系统 ToolRespondEvent (typed). best-effort, 失败不阻断."""
            bus = _get_plugin_event_bus()
            if bus is None:
                return
            try:
                from huginn.api.event import EventType, ToolRespondEvent

                await bus.dispatch(
                    ToolRespondEvent(
                        type=EventType.ON_TOOL_RESPOND,
                        tool_name=tool.name,
                        args=args_dict,
                        result=result,
                        success=success,
                    )
                )
            except Exception:
                logger.debug(
                    "ToolRespondEvent dispatch failed (non-fatal)", exc_info=True
                )

        async def _dispatch_tool_execute_event(
            args_dict: dict[str, Any], session_id: str
        ) -> None:
            """分发插件系统 ON_TOOL_EXECUTE (三段式中间段, 只读打日志用).

            v23 Round 8: 已实际 dispatch — 三段式 (ON_TOOL_CALL → ON_TOOL_EXECUTE
            → ON_TOOL_RESPOND) 完整闭环, 中间段监听器可收到.
            """
            bus = _get_plugin_event_bus()
            if bus is None:
                return
            try:
                from huginn.api.event import Event, EventType

                await bus.dispatch(
                    Event(
                        type=EventType.ON_TOOL_EXECUTE,
                        data={
                            "tool_name": tool.name,
                            "args": args_dict,
                            "session_id": session_id,
                        },
                    )
                )
            except Exception:
                logger.debug(
                    "ON_TOOL_EXECUTE dispatch failed (non-fatal)", exc_info=True
                )

        def _schedule_event(coro: Any) -> None:
            """从同步路径 fire-and-forget 一个 async typed Event dispatch.

            有 running loop 就 ensure_future, 否则丢 main loop, 再不行关掉协程
            避免 "coroutine was never awaited" 警告. 跟 events/integration._schedule_sync
            同套路.
            """
            try:
                asyncio.get_running_loop()
                asyncio.ensure_future(coro)
            except RuntimeError:
                try:
                    from huginn.events.integration import _main_loop

                    if _main_loop is not None and _main_loop.is_running():
                        asyncio.run_coroutine_threadsafe(coro, _main_loop)
                    else:
                        coro.close()
                except Exception:
                    logger.debug("sync typed-event schedule failed", exc_info=True)
            except Exception:
                logger.debug("typed-event schedule failed", exc_info=True)

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
            summarizer = (
                self._summarizer
                if self._summarizer is not None
                else ToolAdapter._summarizer
            )
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
                    logger.debug(
                        "smart_compress_text failed for %s: %s", tool.name, exc
                    )
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
            summarizer = (
                self._summarizer
                if self._summarizer is not None
                else ToolAdapter._summarizer
            )
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
                logger.debug(
                    "smart_compress sync wrapper failed for %s: %s", tool.name, exc
                )
                return obj

        # ── Shared pre-execution checks ──────────────────────────
        # Extracted to eliminate duplication between _run (sync) and
        # _arun (async).  Both paths run the same sequence of gates;
        # only the async-call mechanism differs for confirmation and
        # validation.

        def _run_pre_checks(
            input_data: Any,
            kwargs: dict[str, Any],
            context: Any = None,
        ) -> tuple[dict[str, Any] | None, Any, str]:
            """Run pre-execution gates: permission, cache, router, budget,
            loop detection, circuit breaker.

            Returns ``(early_return, router, stage)``.  ``early_return`` is a
            dict to return immediately if a gate blocks, or ``None``.
            ``router`` is the ToolCallRouter (may be None).  ``stage`` names the
            gate that fired so the caller can emit a span for it (blocked calls
            used to vanish from telemetry entirely).
            """
            # 1. Permission
            _sid = (
                getattr(context, "session_id", None) or "default"
                if context
                else "default"
            )
            approved, reason = _check_permission(input_data, session_id=_sid)
            if not approved:
                output = {"error": reason or f"Tool '{tool.name}' was denied"}
                _audit(input_data, output, approved=False, reason=reason)
                _publish(PetMood.ERROR, f"{tool.name} denied", {"reason": reason})
                return output, None, "permission"

            # 2. Cache (checked before confirmation — safe, no approval needed)
            cached = _get_cached(input_data)
            if cached is not None:
                _audit(input_data, cached, approved=True, reason="cache_hit")
                _publish(PetMood.SUCCESS, f"{tool.name} (cached)", {"tool": tool.name})
                _record_cache_hit(tool.name)
                return cached, None, "cache"

            # 2b. Dedupe (same step, same tool+args → return last result)
            # 跟 read_only cache 互补: cache 只对 read_only 工具生效,
            # dedupe 对所有工具生效, 防止 step 内重复调用有副作用的工具.
            deduper = self._current_deduper
            if deduper is not None:
                cached_dedup = deduper.lookup(tool.name, kwargs)
                if cached_dedup is not None:
                    _economy_bump(dedupe_hits=1)
                    _audit(input_data, cached_dedup, approved=True, reason="dedupe_hit")
                    _publish(
                        PetMood.SUCCESS, f"{tool.name} (deduped)", {"tool": tool.name}
                    )
                    return cached_dedup, None, "dedupe"

            # 3. Router (lightweight-path decision)
            router = self._current_router
            if router is not None:
                allowed, router_reason = router.should_allow(tool.name, kwargs, {})
                if not allowed:
                    output = {"error": router_reason, "_router_blocked": True}
                    _audit(input_data, output, approved=False, reason="router_blocked")
                    _publish(
                        PetMood.ERROR,
                        f"{tool.name} blocked by router",
                        {"reason": router_reason},
                    )
                    return output, router, "router"

            # 4. Budget
            budget = self._current_budget
            if budget is not None and not budget.record(tool.name):
                _, budget_reason = budget.should_stop()
                # P0-3: 加重置语义, 让 LLM 知道是本轮耗尽而非死刑.
                # 之前 "预算耗尽" 让 agent 误判 game over 直接交卷.
                output = {
                    "error": (
                        f"工具调用预算耗尽: {budget_reason}. "
                        f"本轮预算已尽, 下轮重置. 请换工具或收尾, "
                        f"不要放弃当前任务."
                    ),
                    "_budget_exceeded": True,
                }
                _audit(input_data, output, approved=True, reason="budget_exceeded")
                _publish(
                    PetMood.ERROR,
                    f"{tool.name} blocked by budget",
                    {"reason": budget_reason},
                )
                return output, router, "budget"

            # 5. Loop detection
            loop_detector = self._current_loop_detector
            if loop_detector is not None:
                is_loop = loop_detector.record(tool.name, kwargs)
                if is_loop:
                    _, loop_reason = loop_detector.should_break()
                    output = {"error": loop_reason, "_loop_detected": True}
                    _audit(input_data, output, approved=False, reason="loop_detected")
                    _publish(
                        PetMood.ERROR,
                        f"{tool.name} blocked by loop detector",
                        {"reason": loop_reason},
                    )
                    _publish_blocked(tool.name, input_data, loop_reason, context)
                    return output, router, "loop_detector"

            # 6. Circuit breaker → Degradation coordinator
            blocked = _breaker_blocked(tool.name)
            if blocked is not None:
                # 主动降级：查 degradation_chain 自动尝试替代工具
                from huginn.agents.degradation_coordinator import try_with_degradation

                degraded = try_with_degradation(tool.name, input_data, context)
                if not degraded.get("_degradation_exhausted") and not degraded.get(
                    "_no_degradation_chain"
                ):
                    # 降级成功 — 标记并返回替代结果
                    _audit(
                        input_data,
                        degraded,
                        approved=True,
                        reason=f"degraded:{degraded.get('_degraded_to', '?')}",
                    )
                    _publish(
                        PetMood.NEUTRAL,
                        f"{tool.name} degraded → {degraded.get('_degraded_to', '?')}",
                        {"quality_tier": degraded.get("_quality_tier", "")},
                    )
                    return degraded, router, "degraded"
                # 降级链耗尽或无降级链 — 返回结构化错误
                _audit(input_data, degraded, approved=False, reason="circuit_open")
                _publish(
                    PetMood.ERROR,
                    f"{tool.name} circuit open, all fallbacks exhausted",
                    {"tried": degraded.get("tried", [])},
                )
                _publish_blocked(tool.name, input_data, "circuit_open", context)
                return degraded, router, "circuit_breaker"

            return None, router, ""

        def _run_post_checks(
            input_data: Any,
            result: Any,
            output: dict[str, Any],
            context: Any,
            router: Any,
            duration: float = 0.0,
            kwargs: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            """Post-execution: constraints, audit, cache, publish."""
            if router is not None:
                router.record_light_attempt(tool.name)

            # 记录到 deduper, 后续同 step 同 (tool, args) 命中即跳过真实执行.
            # 只缓存成功结果, 错误结果不缓存 (让 LLM 重试).
            deduper = self._current_deduper
            if deduper is not None and result.success:
                try:
                    deduper.record(
                        tool.name, kwargs if kwargs is not None else input_data, output
                    )
                except Exception:
                    logger.debug("deduper record failed (non-fatal)", exc_info=True)

            # 溯源注册: 自动提取文件路径和关键属性
            if result.success:
                try:
                    from huginn.provenance import register_tool_output

                    register_tool_output(
                        tool_name=tool.name,
                        tool_input=input_data.model_dump()
                        if hasattr(input_data, "model_dump")
                        else {},
                        tool_output=output,
                    )
                except ImportError:
                    logger.debug("best-effort op failed", exc_info=True)
                except Exception:
                    logger.debug(
                        "provenance register failed (non-fatal)", exc_info=True
                    )

            # 贝叶斯技能进化: 多维反馈信号 (success + duration + info_gain)
            try:
                from huginn.skills.evolution import SkillEvolutionLayer

                _ig = len(output) if isinstance(output, dict) else 0
                SkillEvolutionLayer.shared().record_tool_call(
                    tool.name,
                    input_data.model_dump()
                    if hasattr(input_data, "model_dump")
                    else {},
                    result.success,
                    duration=duration,
                    info_gain=_ig,
                )
            except Exception:
                logger.debug("skill evolution record failed", exc_info=True)

            result = _check_constraints(result, context)
            output = _serialize(result)
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
            # 事件总线: 发布 tool.result / tool.error
            try:
                from huginn.events.integration import publish_tool_event_sync

                publish_tool_event_sync(
                    tool.name,
                    input_data.model_dump()
                    if hasattr(input_data, "model_dump")
                    else {},
                    output,
                    thread_id=context.thread_id
                    if hasattr(context, "thread_id")
                    else "",
                    error=result.error if not result.success else None,
                )
            except Exception:
                logger.debug("tool event publish failed", exc_info=True)
            # 插件事件系统: 分发 typed ToolRespondEvent (有别于内部 tool.result 字符串事件)
            try:
                _resp_args = (
                    input_data.model_dump() if hasattr(input_data, "model_dump") else {}
                )
                _schedule_event(
                    _dispatch_tool_respond_event(
                        _resp_args,
                        output,
                        result.success,
                    )
                )
            except Exception:
                logger.debug(
                    "ToolRespondEvent dispatch skipped (non-fatal)", exc_info=True
                )
            return output

        async def _arun(**kwargs: Any) -> dict[str, Any]:
            """Async execution wrapper."""
            _t_start = time.time()
            # Hashline 自动闭环: 编辑前注入 expected_hash (来自缓存的上次所见版本)
            kwargs = _maybe_inject_hashline(tool.name, kwargs)
            # 参数边界校验: 只修形状, 修不动就回结构化错误 (绝不抛到调用方)
            input_data, invalid, repairs = _validate_tool_args(tool, kwargs)
            if invalid is not None:
                _audit(kwargs, invalid, approved=True, reason="invalid_tool_args")
                _publish(
                    PetMood.ERROR, f"{tool.name} args invalid", {"tool": tool.name}
                )
                _emit_span(
                    tool.name,
                    kwargs,
                    invalid,
                    stage="schema_validation",
                    started=_t_start,
                    repairs=repairs or None,
                )
                return invalid
            if repairs:
                logger.info(
                    "repaired tool args for %s: %s", tool.name, "; ".join(repairs)
                )
            payload, context = _build_inputs(input_data)

            # Shared pre-checks (permission, cache, router, budget, loop, breaker)
            early, router, stage = _run_pre_checks(input_data, kwargs, context)
            if early is not None:
                _emit_span(
                    tool.name,
                    kwargs,
                    early,
                    stage=stage,
                    started=_t_start,
                    repairs=repairs or None,
                )
                return early

            # 插件事件系统: 分发 typed ToolCallEvent (有别于内部 tool.call 字符串事件)
            # 预置兜底值: 下面 model_dump 失败时后续 ON_TOOL_EXECUTE 仍能拿到参数,
            # 不会因未绑定局部变量抛 NameError.
            _call_args: dict[str, Any] = (
                input_data.model_dump()
                if hasattr(input_data, "model_dump")
                else dict(kwargs)
            )
            _call_sid = (
                getattr(context, "thread_id", None)
                or getattr(context, "session_id", "")
                or ""
            )
            try:
                await _dispatch_tool_call_event(_call_args, _call_sid)
            except Exception:
                logger.debug(
                    "ToolCallEvent dispatch skipped (non-fatal)", exc_info=True
                )

            # Confirmation gate (async: direct await)
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
                            input_data, output, approved=False, reason="user_cancelled"
                        )
                        _publish(
                            PetMood.ERROR,
                            f"{tool.name} cancelled by user",
                            {"reason": confirm_q},
                        )
                        _emit_span(
                            tool.name,
                            kwargs,
                            output,
                            stage="confirmation",
                            started=_t_start,
                            repairs=repairs or None,
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
                _emit_span(
                    tool.name,
                    kwargs,
                    output,
                    stage="validate_input",
                    started=_t_start,
                    repairs=repairs or None,
                )
                return output

            _publish(
                _classify_mood(tool.name), f"Running {tool.name}…", {"tool": tool.name}
            )
            # ON_TOOL_EXECUTE: 三段式中间段, 工具开始执行时发一次 (只读打日志用).
            try:
                await _dispatch_tool_execute_event(_call_args, _call_sid)
            except Exception:
                logger.debug(
                    "ON_TOOL_EXECUTE dispatch skipped (non-fatal)", exc_info=True
                )
            # 按工具类型分级超时，防止外部 API 卡死整个 agent
            timeout = get_timeout(tool.name)
            _call_start = time.time()
            # Wire Prometheus tool call counter
            try:
                from huginn.routes.metrics import track_tool_call

                track_tool_call(tool.name)
            except Exception:
                logger.debug("tool call metrics track failed", exc_info=True)
            with get_telemetry_collector().span("tool_call", tool=tool.name) as span:
                try:
                    if is_async:
                        result = await asyncio.wait_for(
                            tool.call(payload, context), timeout=timeout
                        )
                    else:
                        # G33: 同步工具调到_thread里, 不再卡死 event loop.
                        # 之前 result = tool.call(...) 直接在协程里跑, 14 个同步
                        # 工具 (file_read/vasp/qe/lammps/...) 任何一个慢调用都会
                        # 堵塞所有 WS/SSE 心跳. 现在统一走 wait_for + to_thread.
                        result = await asyncio.wait_for(
                            asyncio.to_thread(tool.call, payload, context),
                            timeout=timeout,
                        )
                except TimeoutError:
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
                # Post-execution: constraints, audit, cache, publish
                result = _check_constraints(result, context)
                # Hashline 自动闭环: 成功读写后记录该文件当前 hash (agent 所见版本)
                if result.success:
                    _update_hashline_cache(tool.name, input_data, context)
                output = _serialize(result)
                # LLM-based smart compression for very large text payloads.
                # Runs only when a summarizer has been registered.
                _eco_before = _estimate_chars(output)
                output = await _smart_compress_output(output)
                _economy_bump(chars_before=_eco_before, chars_after=_estimate_chars(output))
                if repairs:
                    span.metadata["arg_repairs"] = repairs
                span.metadata["success"] = result.success
                span.metadata["args"] = _truncate_for_trajectory(payload)
                span.metadata["result"] = _truncate_for_trajectory(output)
                span.metadata["latency_ms"] = round(
                    (time.time() - _call_start) * 1000, 1
                )
                if result.error:
                    span.metadata["error"] = result.error
            output = _run_post_checks(
                input_data,
                result,
                output,
                context,
                router,
                time.time() - _call_start,
                kwargs,
            )
            return output

        def _run(**kwargs: Any) -> dict[str, Any]:
            """Sync execution wrapper.

            Delegates pre-checks and post-checks to the shared helpers
            to avoid duplicating the 6-gate pipeline.  Only the
            async-specific calls (confirmation, validation, compression)
            are handled differently here.
            """
            # Hashline 自动闭环: 编辑前注入 expected_hash (同 async 路径)
            _t_start = time.time()
            kwargs = _maybe_inject_hashline(tool.name, kwargs)
            # 参数边界校验: 与 async 路径同一层 (只修形状, 修不动回结构化错误)
            input_data, invalid, repairs = _validate_tool_args(tool, kwargs)
            if invalid is not None:
                _audit(kwargs, invalid, approved=True, reason="invalid_tool_args")
                _publish(
                    PetMood.ERROR, f"{tool.name} args invalid", {"tool": tool.name}
                )
                _emit_span(
                    tool.name,
                    kwargs,
                    invalid,
                    stage="schema_validation",
                    started=_t_start,
                    repairs=repairs or None,
                )
                return invalid
            if repairs:
                logger.info(
                    "repaired tool args for %s: %s", tool.name, "; ".join(repairs)
                )
            payload, context = _build_inputs(input_data)

            # Shared pre-checks (permission, cache, router, budget, loop, breaker)
            early, router, stage = _run_pre_checks(input_data, kwargs, context)
            if early is not None:
                _emit_span(
                    tool.name,
                    kwargs,
                    early,
                    stage=stage,
                    started=_t_start,
                    repairs=repairs or None,
                )
                return early

            # 插件事件系统: 分发 typed ToolCallEvent (sync 路径 fire-and-forget)
            _call_args: dict[str, Any] = (
                input_data.model_dump()
                if hasattr(input_data, "model_dump")
                else dict(kwargs)
            )
            _call_sid = (
                getattr(context, "thread_id", None)
                or getattr(context, "session_id", "")
                or ""
            )
            try:
                _schedule_event(_dispatch_tool_call_event(_call_args, _call_sid))
            except Exception:
                logger.debug(
                    "ToolCallEvent dispatch skipped (non-fatal)", exc_info=True
                )

            # Confirmation gate (sync: wrap async call)
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
                        # ClarificationManager unavailable — fail closed for safety
                        # ponytail: 与 async path (L399) 保持一致, fail-closed
                        logger.warning(
                            "ClarificationManager unavailable (sync), failing closed"
                        )
                        confirmed = False
                    if not confirmed:
                        output = {
                            "error": f"用户取消了 {tool.name} 调用",
                            "_user_cancelled": True,
                        }
                        _audit(
                            input_data, output, approved=False, reason="user_cancelled"
                        )
                        _publish(
                            PetMood.ERROR,
                            f"{tool.name} cancelled by user",
                            {"reason": confirm_q},
                        )
                        _emit_span(
                            tool.name,
                            kwargs,
                            output,
                            stage="confirmation",
                            started=_t_start,
                            repairs=repairs or None,
                        )
                        return output

            # Input validation (sync: handle coroutine result)
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
                _emit_span(
                    tool.name,
                    kwargs,
                    output,
                    stage="validate_input",
                    started=_t_start,
                    repairs=repairs or None,
                )
                return output

            _publish(
                _classify_mood(tool.name), f"Running {tool.name}…", {"tool": tool.name}
            )
            # ON_TOOL_EXECUTE: 三段式中间段 (sync 路径 fire-and-forget)
            try:
                _schedule_event(_dispatch_tool_execute_event(_call_args, _call_sid))
            except Exception:
                logger.debug(
                    "ON_TOOL_EXECUTE dispatch skipped (non-fatal)", exc_info=True
                )
            timeout = get_timeout(tool.name)
            _call_start = time.time()
            # Wire Prometheus tool call counter
            try:
                from huginn.routes.metrics import track_tool_call

                track_tool_call(tool.name)
            except Exception:
                logger.debug("tool call metrics track failed", exc_info=True)
            with get_telemetry_collector().span("tool_call", tool=tool.name) as span:
                try:
                    if is_async:
                        # 用统一 helper: 已在 running loop 时跑独立线程,
                        # 之前 loop.run_until_complete 在 running loop 下
                        # 必抛 "already running", 让 sync caller 拿不到结果.
                        from huginn.utils.async_bridge import run_async

                        result = run_async(
                            asyncio.wait_for(
                                tool.call(payload, context), timeout=timeout
                            )
                        )
                    else:
                        result = tool.call(payload, context)
                except TimeoutError:
                    result = ToolResult(
                        data=None,
                        success=False,
                        error=f"{tool.name} timed out after {timeout}s",
                    )
                except Exception as exc:
                    _record_outcome(
                        tool.name, False, time.time() - _call_start, str(exc)
                    )
                    raise
                _record_outcome(
                    tool.name,
                    result.success,
                    time.time() - _call_start,
                    result.error if not result.success else None,
                )
                result = _check_constraints(result, context)
                # Hashline 自动闭环: 成功读写后记录该文件当前 hash (同 async 路径)
                if result.success:
                    _update_hashline_cache(tool.name, input_data, context)
                output = _serialize(result)
                _eco_before = _estimate_chars(output)
                output = _sync_smart_compress(output)
                _economy_bump(chars_before=_eco_before, chars_after=_estimate_chars(output))
                if repairs:
                    span.metadata["arg_repairs"] = repairs
                span.metadata["success"] = result.success
                span.metadata["args"] = _truncate_for_trajectory(payload)
                span.metadata["result"] = _truncate_for_trajectory(output)
                span.metadata["latency_ms"] = round(
                    (time.time() - _call_start) * 1000, 1
                )
                if result.error:
                    span.metadata["error"] = result.error
            output = _run_post_checks(
                input_data,
                result,
                output,
                context,
                router,
                time.time() - _call_start,
                kwargs,
            )
            return output

        return _AdapterStructuredTool.from_function(
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
