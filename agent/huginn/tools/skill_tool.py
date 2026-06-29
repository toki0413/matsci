"""Skill tool — lets the LLM invoke preset scientific workflow skills.

Wraps DeclarativeSkillExecutor so the agent can list, describe, and run
named skills (DFT, MD, phonon, band structure, etc.) during a conversation.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# Importing presets has a side effect: every SkillDefinition passed to
# register_skill() lands in SkillRegistry. Keep this import above the
# tool class so the registry is populated before the first call.
import huginn.skills.presets  # noqa: F401
from huginn.skills.base import DeclarativeSkillExecutor, SkillDefinition
from huginn.skills.registry import SkillRegistry
from huginn.tools.base import HuginnTool
from huginn.tools.registry import ToolRegistry
from huginn.types import ToolContext, ToolResult


class SkillToolInput(BaseModel):
    action: Literal["list", "execute", "describe"] = Field(default="list")
    skill_name: str | None = Field(
        default=None, description="Name of the skill to execute or describe"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Parameters to pass to the skill"
    )
    thread_id: str | None = Field(
        default=None, description="Thread ID for context isolation"
    )


class SkillToolOutput(BaseModel):
    success: bool
    action: str
    skill_name: str | None = None
    result: Any = None
    error: str | None = None
    available_skills: list[dict] | None = None


def _skill_summary(skill: SkillDefinition) -> dict:
    """Compact dict representation used by list and describe actions."""
    return {
        "name": skill.name,
        "description": skill.description,
        "category": skill.category,
        "tags": list(skill.tags),
        "parameters": [
            {
                "name": p.name,
                "type": p.type,
                "description": p.description,
                "required": p.required,
                "default": p.default,
            }
            for p in skill.parameters
        ],
        "required_tools": list(skill.required_tools),
        "steps": [
            {
                "name": s.name,
                "tool": s.tool,
                "output_key": s.output_key,
                "on_failure": s.on_failure,
            }
            for s in skill.steps
        ],
    }


class SkillTool(HuginnTool[SkillToolInput, SkillToolOutput]):
    """Bridge between the LLM and the declarative skill executor."""

    name = "skill"
    category = "meta"
    description = (
        "Execute preset scientific workflow skills (DFT, MD, phonon, band "
        "structure, etc.). Use 'list' action to see available skills."
    )
    destructive = False
    read_only = False  # skills run real tools, so they can have side effects
    input_schema = SkillToolInput
    output_schema = SkillToolOutput

    def __init__(self, skill_executor: DeclarativeSkillExecutor | None = None) -> None:
        super().__init__()
        if skill_executor is None:
            skill_executor = DeclarativeSkillExecutor(ToolRegistry)
        self._executor = skill_executor

    async def call(self, args: SkillToolInput, context: ToolContext) -> ToolResult:
        if args.action == "list":
            return self._list_skills()
        if args.action == "describe":
            return self._describe_skill(args)
        if args.action == "execute":
            return await self._execute_skill(args, context)

        msg = f"Unknown action: {args.action}"
        return ToolResult(
            data=SkillToolOutput(success=False, action=args.action, error=msg).model_dump(),
            success=False,
            error=msg,
        )

    # -- actions -----------------------------------------------------------

    def _list_skills(self) -> ToolResult:
        skills = SkillRegistry.get_all_definitions()
        summaries = [_skill_summary(s) for s in skills]
        out = SkillToolOutput(
            success=True,
            action="list",
            available_skills=summaries,
            result={"count": len(summaries)},
        )
        return ToolResult(data=out.model_dump(), success=True)

    def _describe_skill(self, args: SkillToolInput) -> ToolResult:
        if not args.skill_name:
            return self._missing_name("describe")

        skill = SkillRegistry.get(args.skill_name)
        if skill is None:
            return self._not_found(args.skill_name, "describe")

        out = SkillToolOutput(
            success=True,
            action="describe",
            skill_name=skill.name,
            result=_skill_summary(skill),
        )
        return ToolResult(data=out.model_dump(), success=True)

    async def _execute_skill(
        self, args: SkillToolInput, context: ToolContext
    ) -> ToolResult:
        if not args.skill_name:
            return self._missing_name("execute")

        skill = SkillRegistry.get(args.skill_name)
        if skill is None:
            return self._not_found(args.skill_name, "execute")

        # Build a context dict for the executor. Params are merged on top
        # of this inside DeclarativeSkillExecutor.execute, so user-supplied
        # values always win.
        exec_context: dict[str, Any] = {}
        if args.thread_id:
            exec_context["thread_id"] = args.thread_id
        if context.session_id:
            exec_context["session_id"] = context.session_id

        try:
            result = await self._executor.execute(skill, args.parameters, exec_context)
        except Exception as exc:
            out = SkillToolOutput(
                success=False,
                action="execute",
                skill_name=skill.name,
                error=str(exc),
            )
            return ToolResult(data=out.model_dump(), success=False, error=str(exc))

        out = SkillToolOutput(
            success=bool(result.get("success", False)),
            action="execute",
            skill_name=skill.name,
            result=result,
        )
        return ToolResult(data=out.model_dump(), success=out.success)

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _missing_name(action: str) -> ToolResult:
        msg = f"skill_name is required for {action} action"
        return ToolResult(
            data=SkillToolOutput(success=False, action=action, error=msg).model_dump(),
            success=False,
            error=msg,
        )

    @staticmethod
    def _not_found(skill_name: str, action: str) -> ToolResult:
        msg = f"Skill '{skill_name}' not found"
        return ToolResult(
            data=SkillToolOutput(
                success=False,
                action=action,
                skill_name=skill_name,
                error=msg,
            ).model_dump(),
            success=False,
            error=msg,
        )
