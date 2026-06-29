"""personalization_tool — 让 agent 自己查/改用户语言偏好 profile.

agent 在对话里可以主动看用户偏好, 或根据用户口头反馈调整.
用户显式表达偏好时 (如 "别用X词" / "回答简短点"), agent 调 set_preference.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from pydantic import BaseModel, Field

from huginn.personalization import get_shared_style_learner
from huginn.tools.base import HuginnTool
from huginn.types import ToolResult


class PersonalizationInput(BaseModel):
    action: str = Field(
        description=(
            "操作类型: "
            "get_profile (查当前用户偏好) / "
            "get_directive (拿风格指令文本) / "
            "reset (重置所有学习结果) / "
            "set_preference (手动设某维度, 覆盖学习结果)"
        )
    )
    dimension: str | None = Field(
        default=None,
        description=(
            "要设的维度名, 仅 set_preference 用. 可选: "
            "vocabulary_level / formality / verbosity / language / "
            "response_format / code_style / avoid_terms"
        ),
    )
    value: str | None = Field(
        default=None,
        description="要设的值, 仅 set_preference 用",
    )


class PersonalizationOutput(BaseModel):
    data: dict[str, Any] | None = None
    success: bool = True
    error: str | None = None


class PersonalizationTool(HuginnTool[PersonalizationInput, PersonalizationOutput]):
    name = "personalization_tool"
    category = "meta"
    description = (
        "查询或调整 agent 的用户语言偏好 profile. "
        "agent 据此定制自己的通信风格 (用词/格式/语气/专业程度). "
        "用户显式表达偏好时 (如 '别用X词' '回答简短点'), 调 set_preference."
    )
    destructive = False
    read_only = False
    input_schema = PersonalizationInput
    output_schema = PersonalizationOutput

    async def call(self, args: PersonalizationInput, context) -> ToolResult:
        learner = get_shared_style_learner()
        try:
            if args.action == "get_profile":
                p = learner.get_profile()
                return ToolResult(
                    data=asdict(p),
                    success=True,
                )
            if args.action == "get_directive":
                return ToolResult(
                    data={"directive": learner.get_style_directive()},
                    success=True,
                )
            if args.action == "reset":
                learner.reset()
                return ToolResult(data={"reset": True}, success=True)
            if args.action == "set_preference":
                if not args.dimension or not args.value:
                    return ToolResult(
                        data=None,
                        success=False,
                        error="set_preference 需要 dimension 和 value 两个参数",
                    )
                ok = learner.set_preference(args.dimension, args.value)
                if ok:
                    return ToolResult(
                        data={"set": True, "dimension": args.dimension, "value": args.value},
                        success=True,
                    )
                return ToolResult(
                    data=None,
                    success=False,
                    error=f"无效维度: {args.dimension}",
                )
            return ToolResult(
                data=None,
                success=False,
                error=f"未知 action: {args.action}. 支持: get_profile / get_directive / reset / set_preference",
            )
        except Exception as exc:
            return ToolResult(data=None, success=False, error=str(exc))
