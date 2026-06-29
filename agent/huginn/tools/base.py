"""Tool base class — inspired by Claude Code's Tool<T> interface.

Every tool is self-contained with:
- name, description, input/output schemas
- permission checking
- input validation
- execution logic
- result mapping
"""

from __future__ import annotations

from abc import ABC, abstractmethod
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

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class HuginnTool(ABC, Generic[InputT, OutputT]):
    """Base class for all Huginn tools.

    Mirrors Claude Code's Tool<Input, Output, Progress> interface.
    """

    name: str = ""
    description: str = ""

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

    @abstractmethod
    async def call(self, args: InputT, context: ToolContext) -> ToolResult:
        """Execute the tool. Must be implemented by subclasses."""
        ...

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
