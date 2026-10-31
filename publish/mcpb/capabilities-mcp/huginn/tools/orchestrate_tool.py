"""Orchestrate tool — lets the lead agent spawn a team of sub-agents.

Input: objective + optional agent_ids + thread_id.
Output: task plan, per-sub-agent outputs, and synthesized summary.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from huginn.core_types import ToolResult
from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile


class OrchestrateInput(BaseModel):
    objective: str = Field(
        description="High-level goal to decompose across sub-agents."
    )
    agent_ids: list[str] = Field(
        default_factory=list,
        description="Optional list of agent profile IDs to use. If empty, all enabled profiles are available.",
    )
    thread_id: str = Field(
        default="default", description="Thread ID for shared context."
    )
    synthesize: bool = Field(
        default=True, description="Whether to ask the lead agent to synthesize outputs."
    )


class OrchestrateOutput(BaseModel):
    success: bool
    objective: str
    plan: list[dict]
    outputs: dict[str, str]
    summary: str
    error: str | None = None


def _validate_agent_ids(args: OrchestrateInput, factory) -> ToolResult | None:
    # If agent_ids provided, temporarily narrow factory profiles (orchestrator uses factory.list_profiles)
    if not args.agent_ids:
        return None
    available = {p.id for p in factory.list_profiles()}
    missing = [a for a in args.agent_ids if a not in available]
    if missing:
        return ToolResult(
            data=None,
            success=False,
            error=f"Unknown agent profiles: {missing}. Available: {sorted(available)}",
        )
    return None


def _build_output(result, synthesize: bool) -> OrchestrateOutput:
    if synthesize and result.success:
        summary = result.summary
    else:
        summary = result.summary or ""

    return OrchestrateOutput(
        success=result.success,
        objective=result.objective,
        plan=[
            {
                "task_id": t.task_id,
                "agent_id": t.agent_id,
                "status": t.status,
                "prompt": t.prompt,
            }
            for t in result.plan.tasks
        ],
        outputs=result.outputs,
        summary=summary,
        error=result.error,
    )


class OrchestrateTool(HuginnTool[OrchestrateInput, OrchestrateOutput]):
    name = "orchestrate"
    category = "meta"
    profile = ToolProfile(phases=frozenset({ResearchPhase.EXECUTION}))
    description = (
        "Decompose a complex objective into parallel subtasks and run them with "
        "specialist sub-agents (potentially using different LLM providers/models). "
        "Returns a synthesized answer."
    )
    destructive = False
    read_only = True
    input_schema = OrchestrateInput
    output_schema = OrchestrateOutput

    async def call(self, args: OrchestrateInput, context) -> ToolResult:
        if context.agent_factory is None:
            return ToolResult(
                data=None,
                success=False,
                error="Multi-agent factory not available. Configure models and agent profiles first.",
            )
        try:
            from huginn.agents.orchestrator import Orchestrator

            factory = context.agent_factory
            failure = _validate_agent_ids(args, factory)
            if failure is not None:
                return failure

            orchestrator = Orchestrator(
                factory=factory,
                memory_manager=context.memory_manager,
                max_concurrent=max(1, factory.config.max_concurrent_subagents),
            )
            result = await orchestrator.run(args.objective)

            return ToolResult(
                data=_build_output(result, args.synthesize).model_dump(),
                success=result.success,
            )
        except Exception as exc:
            return ToolResult(data=None, success=False, error=str(exc))
