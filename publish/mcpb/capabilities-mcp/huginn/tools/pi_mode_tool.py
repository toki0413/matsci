"""pi_mode_tool —— 运行时开关 Pi 极简内核模式.

动作:
- on:     apply_pi_mode(): 隐藏非原语工具, 只留原语 + 自扩展
- off:    exit_pi_mode():  恢复全部工具可见
- status: 当前是否 pi 模式 + 原语/隐藏统计

不触碰中心装配: 只翻转各工具实例的 ``active`` 标志, ``get_all_schemas`` 自然按
active=False 过滤, 既有架构门禁不变.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.core_types import ToolContext, ToolResult
from huginn.tools.base import HuginnTool
from huginn.tools.profile import ToolProfile


class PiModeToolInput(BaseModel):
    action: Literal["on", "off", "status"] = Field(default="status")


class PiModeTool(HuginnTool):
    """运行时切换 Pi 极简内核模式 (精简工具可见面, 保留自扩展)."""

    name = "pi_mode_tool"
    category = "meta"
    description = (
        "Toggle Pi minimal-core mode. action='on' hides all non-primitive tools from "
        "the LLM (keep read/write/edit/bash + tool_search + capability + make_tool), "
        "recovering context for the actual task — the Pi philosophy. Tools stay "
        "callable via ToolRegistry.get, and new ones come online via make_tool. "
        "action='off' restores everything; action='status' reports current state."
    )
    input_schema = PiModeToolInput
    profile = ToolProfile(cost_tier="none", phases=None)

    async def call(self, args: Any, context: ToolContext | None = None) -> ToolResult:
        if isinstance(args, PiModeToolInput):
            data = args
        elif isinstance(args, BaseModel):
            data = PiModeToolInput(**args.model_dump())
        else:
            data = PiModeToolInput(**dict(args or {}))

        from huginn.modes.pi import apply_pi_mode, exit_pi_mode, pi_active

        if data.action == "on":
            return ToolResult(data=apply_pi_mode(), success=True)
        if data.action == "off":
            return ToolResult(data=exit_pi_mode(), success=True)
        return ToolResult(
            data={"mode": "pi" if pi_active() else "default"}, success=True
        )


__all__ = ["PiModeTool", "PiModeToolInput"]
