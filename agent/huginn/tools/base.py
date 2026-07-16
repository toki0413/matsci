"""Tool base class — inspired by Claude Code's Tool<T> interface.

Every tool is self-contained with:
- name, description, input/output schemas
- permission checking
- input validation
- execution logic
- result mapping
"""

from __future__ import annotations

import contextvars
import hashlib
import sys
from abc import ABC
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from huginn.phases import ResearchPhase
from huginn.tools.profile import CostTier, ToolProfile
from huginn.types import (
    PermissionMode,
    PermissionResult,
    ToolContext,
    ToolResult,
    ValidationResult,
)

import logging

logger = logging.getLogger(__name__)

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)

# values we treat as "no real description set, go read the .md file"
_PLACEHOLDER_DESCRIPTIONS = frozenset({"", "todo", "placeholder", "tbd"})


# ── provenance collector (context var) ─────────────────────────────────────
# The autoloop / engine sets a collector (any object with .append) for the
# duration of a run via set_provenance_collector. HuginnTool.call() then drops
# a ProvenanceSnapshot into it after every tool execution, so every tool gets
# traced without each subclass repeating the capture boilerplate. Defaults to
# None -> no capture, which keeps standalone tool calls zero-cost.
_provenance_collector: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "provenance_collector", default=None
)


def set_provenance_collector(collector: list | None) -> None:
    """Bind a collector list for the current async context (engine.run sets this)."""
    _provenance_collector.set(collector)


def get_provenance_collector() -> list | None:
    return _provenance_collector.get()


def _serialize_tool_args(args: Any) -> dict[str, Any]:
    """Flatten whatever a tool received as input into a JSON-ish dict for hashing.

    Most tools get a Pydantic model (model_dump), some get a plain dict, and a
    few pass arbitrary objects — the str() fallback keeps the hash stable
    without crashing the snapshot path.
    """
    if hasattr(args, "model_dump"):
        try:
            dumped = args.model_dump()
            return dumped if isinstance(dumped, dict) else {"value": dumped}
        except Exception:
            return {"args": str(args)}
    if isinstance(args, dict):
        return dict(args)
    return {"args": str(args)}


class HuginnTool(ABC, Generic[InputT, OutputT]):
    """Base class for all Huginn tools.

    Mirrors Claude Code's Tool<Input, Output, Progress> interface.
    """

    name: str = ""
    description: str = ""

    # .md description cache, keyed by tool name (qualname fallback).
    # stops us re-statting the filesystem for every instance we spin up.
    _description_cache: dict[str, str | None] = {}

    # 工具分类, 用于 tool_filter / UI 分组 / 日志统计
    # 取值: core / search / meta / sim / sci / design / cv / materials / misc
    category: str = "misc"

    # Static hints for UI / permission systems
    destructive: bool = False
    read_only: bool = False

    # Schema definitions (Pydantic v2, replacing Zod)
    input_schema: type[InputT] | None = None
    output_schema: type[OutputT] | None = None

    # 声明需要从 config 注入的构造参数: {构造参数名: config 字段名}
    # register_all_tools() 会读这个 map 自动填充 kwargs, 避免类名 if 分支
    _init_kwargs_map: dict[str, str] = {}

    # 调度元数据: 工具自声明 cost tier / phase 适用性 / constraint scope /
    # light alternatives / heavy actions. None 时走默认值 (见 _default_profile).
    # 详见 huginn/tools/profile.py
    profile: ToolProfile | None = None

    # 运行时启停 (AstrBot FunctionTool.active 模式)
    # False 时工具对 LLM 不可见, 但仍可通过 ToolRegistry.get() 直接调用
    active: bool = True

    # 后台任务标记 (AstrBot FunctionTool.is_background_task 模式)
    # True 时工具调用立即返回 task_id, 结果异步轮询
    is_background_task: bool = False

    def is_available(self) -> bool:
        """Whether this tool is currently usable.

        Active flag gates LLM visibility; is_available() gates real-world
        readiness — e.g. MCP server disconnected, optional dependency missing,
        HPC endpoint down. AnythingLLM isToolAvailable() pattern: tools that
        report False here are filtered out of the schema list the LLM sees,
        so the model never wastes a turn trying to call a dead tool.

        Default True. Override in subclasses that own external resources.
        """
        return True

    def __init__(self) -> None:
        # Subclasses that hardcode their own description are left alone.
        # Empty / placeholder -> fall back to a sibling .md file so the
        # prompt text can be tweaked without redeploying the .py.
        # ponytail: we load at construction, not at first attribute read.
        # The registry instantiates a tool exactly when it needs the
        # description, so this is lazy enough in practice; swap to a
        # property if access-time laziness ever becomes a real need.
        desc = self.description or ""
        if desc.strip().lower() in _PLACEHOLDER_DESCRIPTIONS:
            loaded = type(self)._load_description()
            if loaded:
                # instance attr shadows the empty class default
                self.description = loaded

    @classmethod
    def _load_description(cls) -> str | None:
        """Read the external .md description for this tool, if one exists.

        Two lookup spots, first match wins:
          1. beside the module that defines the class (foo.py -> foo.md)
          2. the shared descriptions/ folder, keyed by the tool name

        Cached per tool name so repeat instantiations skip the file read.
        """
        cache_key = cls.name or cls.__qualname__
        cache = HuginnTool._description_cache
        if cache_key in cache:
            return cache[cache_key]

        candidates: list[Path] = []

        # co-located with the defining module
        mod = sys.modules.get(cls.__module__)
        mod_file = getattr(mod, "__file__", None) if mod is not None else None
        if mod_file:
            candidates.append(Path(mod_file).with_suffix(".md"))

        # shared descriptions dir, keyed by tool name
        if cls.name:
            candidates.append(
                Path(__file__).resolve().parent / "descriptions" / f"{cls.name}.md"
            )

        content: str | None = None
        for path in candidates:
            try:
                if path.is_file():
                    text = path.read_text(encoding="utf-8").strip()
                    content = text or None
                    break
            except OSError:
                # unreadable / weird encoding -> try the next candidate
                continue

        cache[cache_key] = content
        return content

    @staticmethod
    def _default_profile() -> ToolProfile:
        """profile=None 时的回落值, 保持重构前的行为:
        非重型非轻量, 仅 OPEN 阶段可用, 无约束检查.
        """
        return ToolProfile(
            cost_tier="none",
            phases=frozenset(),
            constraint_scope=None,
            light_alternatives=(),
            heavy_actions=None,
        )

    @property
    def input_json_schema(self) -> dict[str, Any] | None:
        if self.input_schema:
            return self.input_schema.model_json_schema()
        return None

    def format_result(self, result: ToolResult) -> dict[str, Any]:
        """Serialize a ToolResult to JSON-safe dict (CLI-Anything --json contract).

        Validates against output_schema if defined; validation failure
        is logged but doesn't block — the data still goes through.
        """
        from huginn.types import _jsonify

        payload = result.to_dict()

        # output_schema 是可选的, 有就校验
        if self.output_schema and result.success and result.data is not None:
            try:
                if isinstance(result.data, dict):
                    self.output_schema(**result.data)
                elif not isinstance(result.data, self.output_schema):
                    # 非侵入式: 记录不匹配但不阻塞
                    pass
            except Exception:
                logger.debug("format result failed", exc_info=True)  # 校验失败不阻塞, 数据照传

        return payload

    # ── 调度元数据便利属性 ──────────────────────────────────────────
    # 透传到 profile, profile=None 时回落到默认值, 保证未声明 profile 的
    # 工具行为与重构前一致.

    @property
    def cost_tier(self) -> CostTier:
        p = self.profile if self.profile is not None else self._default_profile()
        return p.cost_tier

    @property
    def phases(self) -> frozenset[ResearchPhase] | None:
        p = self.profile if self.profile is not None else self._default_profile()
        return p.phases

    @property
    def constraint_scope(self) -> str | None:
        p = self.profile if self.profile is not None else self._default_profile()
        return p.constraint_scope

    @property
    def light_alternatives(self) -> tuple[str, ...]:
        p = self.profile if self.profile is not None else self._default_profile()
        return p.light_alternatives

    @property
    def heavy_actions(self) -> frozenset[str] | None:
        p = self.profile if self.profile is not None else self._default_profile()
        return p.heavy_actions

    async def _execute(self, args: InputT, context: ToolContext) -> ToolResult:
        """Actual tool logic. New tools override this and get provenance for free.

        Legacy tools still override ``call`` directly — that keeps working, they
        just opt out of the automatic snapshot below (and keep doing their own
        capture where they already do, e.g. vasp/lammps).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _execute() or override call()"
        )

    async def call(self, args: InputT, context: ToolContext) -> ToolResult:
        """Execute the tool and capture provenance.

        Subclasses normally override ``_execute`` (not this method) so the
        snapshot wrapper here runs for free. Tools that already override
        ``call`` keep working unchanged — their override shadows this wrapper,
        so existing manual provenance capture (vasp_tool / lammps_tool) is
        untouched and never double-captured.
        """
        try:
            result = await self._execute(args, context)
        except Exception as exc:
            # tool 运行时异常, 回写 FailureModeRegistry 给 metacog 层用.
            # metacog 不可用绝不能拖垮 tool 本身的异常路径, 全程 try 包.
            try:
                from huginn.metacog.failure_modes import DEFAULT_REGISTRY

                DEFAULT_REGISTRY.record_observation(
                    "runtime_error",
                    self.name or type(self).__name__,
                    str(exc)[:200],
                )
            except Exception:
                pass
            raise
        try:
            self._capture_provenance(args, result)
        except Exception:
            # provenance is best-effort: never let it sink a good result
            pass
        return result

    def _capture_provenance(self, args: Any, result: ToolResult) -> Any:
        """Append a ProvenanceSnapshot to the active collector, if any.

        Lightweight by design: respects the ``provenance`` feature flag and
        skips the heavy software-version sweep that ``huginn.provenance.capture``
        does (that one imports ~10 packages), so it's cheap to run on every
        tool call. Returns the snapshot or None when capture is off.
        """
        try:
            from huginn.feature_flags import FeatureFlags
            if not FeatureFlags.shared().is_enabled("provenance"):
                return None
        except Exception:
            # flag layer down — keep capturing, provenance is opt-out not opt-in
            pass

        collector = get_provenance_collector()
        if collector is None:
            return None

        from huginn.provenance import ProvenanceSnapshot

        params = _serialize_tool_args(args)
        out_repr = result.to_dict() if hasattr(result, "to_dict") else str(result)
        snapshot = ProvenanceSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            tool_name=self.name,
            tool_version=getattr(self, "version", "1.0"),
            input_params=params,
            output_hash=hashlib.sha256(str(out_repr).encode("utf-8")).hexdigest()[:16],
        )
        collector.append(snapshot)
        return snapshot

    async def check_permissions(
        self, args: InputT, context: ToolContext
    ) -> PermissionResult:
        """Check if the tool can be executed under current permissions.

        Default: allow. Override for tools that need explicit approval
        (e.g., job submission, file deletion).
        """
        return PermissionResult(mode=PermissionMode.AUTO)

    async def validate_input(
        self, args: InputT, context: ToolContext
    ) -> ValidationResult:
        """Validate input before execution. Pydantic already handles schema validation,
        but this allows additional semantic checks (e.g., file existence, path validity).
        """
        return ValidationResult(result=True)

    def is_read_only(self, args: InputT) -> bool:
        """Return True if this tool call is read-only (no side effects).
        Read-only tools can be auto-executed without user confirmation.
        """
        return self.read_only

    def is_destructive(self, args: InputT) -> bool:
        """Return True if this tool call is destructive (deletes/overwrites data).
        Destructive tools ALWAYS require explicit user confirmation.
        """
        return self.destructive

    def estimate_cost(self, args: InputT) -> dict[str, float] | None:
        """Estimate computational cost for this tool call.
        Returns dict with keys like cpu_hours, gpu_hours, walltime_hours.
        Return None if cost is negligible.
        """
        return None
