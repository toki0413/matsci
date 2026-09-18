"""make_tool —— Pi 式"缺啥自写工具"原语的模型侧入口.

哲学对齐 Pi: 与其在 180 个预装工具里翻, 不如让模型在遇到没有的能力时
**当场写一个**. ``make`` 校验 → 落盘 → 热注册, 下一步这条工具就能被调用.

安全: 模型自写的工具是可执行代码. 本工具走既有的权限门禁 (check_permissions),
只在显式允许下放行 (默认 ASK, 需 context.permissions 里授权) —— 这是 Pi 的
"你掌控自己造的东西"在 huginn 的落地, 而非无约束 YOLO.

actions:
- make:   name + code (+可选 description/category) → 校验/落盘/热注册
- list:   已注册的扩展工具 + 元数据
- info:   扩展目录位置 / 注册数
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.core_types import PermissionMode, PermissionResult, ToolContext, ToolResult
from huginn.tools.base import HuginnTool
from huginn.tools.extensions import build_extension, extensions_dir, list_extensions
from huginn.tools.profile import ToolProfile


class MakeToolInput(BaseModel):
    action: Literal["make", "list", "info"] = Field(default="list")
    name: str = Field(
        default="",
        description="lower snake_case tool name, 2-64 chars [a-z0-9_]. must equal "
                    "the `name` attribute declared in `code`.",
    )
    code: str = Field(
        default="",
        description=(
            "Python source defining a HuginnTool subclass whose `name` equals the "
            "requested name. Example:\n"
            "from pydantic import BaseModel\n"
            "from huginn.core_types import ToolContext, ToolResult\n"
            "from huginn.tools.base import HuginnTool\n\n"
            "class CalcTool(HuginnTool):\n"
            "    name = 'calc'\n"
            "    category = 'self'\n"
            "    def call(self, args, context: ToolContext | None = None):\n"
            "        return ToolResult(data={'ok': True})\n"
        ),
    )
    description: str | None = Field(default=None, description="(可选) 覆盖工具描述")
    category: str = Field(default="self", description="(可选) 工具分组, 默认 self")


class MakeTool(HuginnTool):
    """让模型当场自写并热注册一个工具 (Pi 自扩展原语)."""

    name = "make_tool"
    category = "meta"
    description = (
        "Write and hot-register a NEW custom tool on the fly. Use when no existing "
        "tool covers the task: provide `name` (lower snake_case) and `code` (Python "
        "defining a HuginnTool subclass whose `name` equals the requested name). "
        "The tool is compiled, persisted, and registered so it's callable next turn. "
        "action='list' enumerates model-authored tools; action='info' shows the "
        "extensions dir. Model-authored code is gated by permissions."
    )
    input_schema = MakeToolInput
    profile = ToolProfile(cost_tier="none", phases=None)

    async def check_permissions(
        self, args: Any, context: ToolContext | None
    ) -> PermissionResult:
        """缺省 ASK: 写可执行代码需显式授权 (除非 context.permissions 已放行)."""
        if context is not None:
            perms = getattr(context, "permissions", None) or {}
            get = getattr(perms, "get", lambda *_: None)
            if get("make_tool", None) == PermissionMode.AUTO:
                return PermissionResult(mode=PermissionMode.AUTO)
        return PermissionResult(
            mode=PermissionMode.ASK,
            reason="make_tool compiles & registers model-authored executable code; "
                   "approve it to enable Pi-style self-extension",
        )

    async def call(self, args: Any, context: ToolContext | None = None) -> ToolResult:
        if isinstance(args, MakeToolInput):
            data = args
        elif isinstance(args, BaseModel):
            data = MakeToolInput(**args.model_dump())
        else:
            data = MakeToolInput(**dict(args or {}))

        if data.action == "list":
            return ToolResult(
                data={"action": "list", "count": len(list_extensions()),
                      "extensions": list_extensions()},
                success=True,
            )
        if data.action == "info":
            return ToolResult(
                data={
                    "action": "info",
                    "extensions_dir": str(extensions_dir()),
                    "count": len(list_extensions()),
                    "note": "model-authored tools persist across restarts via scan_extensions",
                },
                success=True,
            )
        # make
        if not data.name.strip() or not data.code.strip():
            return ToolResult(
                data=None, success=False,
                error="make requires both 'name' and 'code'",
            )
        try:
            result = build_extension(
                data.name, data.code,
                category=data.category, description=data.description,
            )
        except ValueError as exc:
            return ToolResult(data=None, success=False, error=f"make_tool failed: {exc}")
        return ToolResult(data={"action": "make", **result}, success=True)


__all__ = ["MakeTool", "MakeToolInput"]
