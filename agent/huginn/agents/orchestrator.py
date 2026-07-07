"""Orchestrator — lead agent plans, sub-agents execute in parallel, synthesizer summarizes.

Inspired by Claude Code Agent Teams (Lead + Teammates) and Hermes PolyBrain.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass, field
from typing import Any

from huginn.agents.factory import AgentFactory
from huginn.autoloop.plan_store import Plan, PlanStep, PlanStore
from huginn.memory.longterm import LongTermMemory
from huginn.memory.manager import MemoryManager
import logging
logger = logging.getLogger(__name__)



@dataclass
class SubTask:
    task_id: str
    agent_id: str
    prompt: str
    depends_on: list[str] = field(default_factory=list)
    result: str = ""
    status: str = "pending"  # pending, running, done, error


@dataclass
class TaskPlan:
    objective: str
    tasks: list[SubTask]


@dataclass
class OrchestratorResult:
    objective: str
    plan: TaskPlan
    outputs: dict[str, str] = field(default_factory=dict)
    summary: str = ""
    success: bool = True
    error: str | None = None


_PLANNER_PROMPT = """You are a team coordinator. Given the user's objective, break it into 2-5 parallel subtasks suitable for different specialist agents.

Available agents and their roles:
{profiles}

Return ONLY a JSON object with no markdown formatting:
{{
  "tasks": [
    {{"task_id": "t1", "agent_id": "<profile_id>", "prompt": "<specific instructions>", "depends_on": []}},
    ...
  ]
}}

Rules:
- Each task must have a unique task_id and reference one of the available agent_ids.
- Keep prompts self-contained and specific.
- Use depends_on only when a task truly needs the output of another task (list task_ids).
- If the objective is simple, return a single task for the lead agent.
"""


# 结构化计划 prompt — 产出 PlanStep 列表, 会持久化到 PlanStore.
# 跟 _PLANNER_PROMPT 的区别: 输出 steps 而非 tasks, 每步带 tool/parameters,
# 方便 executor 精确执行 + 用户审查.
_PLANNER_PERSONA_PROMPT = """You are a research planner. Decompose the user's objective into a sequence of concrete, executable steps.

Available agents and their roles:
{profiles}

Return ONLY a JSON object with no markdown formatting:
{{
  "steps": [
    {{"id": "s1", "description": "<what to do>", "tool": "<tool_name or null>", "parameters": {{}}, "dependencies": [], "agent_id": "<profile_id>"}},
    ...
  ]
}}

Rules:
- Each step must have a unique id (s1, s2, ...) and reference one of the available agent_ids.
- description must be self-contained — an executor agent should run it without further context.
- tool: which tool to call (vasp_tool, lammps_tool, web_search, file_read_tool, ...) or null for a reasoning step.
- parameters: dict of tool arguments, or {{}} if no tool.
- dependencies: list of step ids that must complete before this step.
- If the objective is simple, return a single step for the lead agent.
"""


class Orchestrator:
    """Coordinates multi-agent task execution.

    With plan_store set, plan() persists structured Plans and execute()
    reads confirmed plans by id — enabling human-in-the-loop review.
    Without plan_store, falls back to the legacy in-memory TaskPlan path.
    """

    def __init__(
        self,
        factory: AgentFactory,
        memory_manager: MemoryManager | None = None,
        max_concurrent: int = 3,
        plan_store: PlanStore | None = None,
        auto_confirm: bool = False,
    ):
        self.factory = factory
        self.memory_manager = memory_manager
        self.max_concurrent = max_concurrent
        self.plan_store = plan_store
        self.auto_confirm = auto_confirm

    async def plan(
        self, objective: str, auto_confirm: bool | None = None
    ) -> Plan | TaskPlan:
        """Decompose objective into structured steps.

        With plan_store: persists a Plan (draft or confirmed), returns Plan.
        Without plan_store: legacy in-memory TaskPlan path (backward compat).
        """
        if self.plan_store is None:
            return await self._plan_legacy(objective)

        ac = auto_confirm if auto_confirm is not None else self.auto_confirm
        profiles = self._format_profiles()
        # Planner role: lead model + planner persona override
        lead = self.factory.create_lead(system_prompt_override=_PLANNER_PERSONA_PROMPT.format(profiles=profiles))
        prompt = f"\n\nObjective: {objective}"
        state = lead.invoke(prompt)
        raw = self._extract_output(state)
        steps = self._parse_steps(raw, objective)
        plan = self.plan_store.create_plan(objective, steps, auto_confirm=ac)
        if ac:
            self.plan_store.confirm_plan(plan.id)
            plan = self.plan_store.get_plan(plan.id)
        return plan

    async def _plan_legacy(self, objective: str) -> TaskPlan:
        """Legacy in-memory plan path (no persistence)."""
        profiles = self._format_profiles()
        lead = self.factory.create_lead()
        prompt = (
            _PLANNER_PROMPT.format(profiles=profiles) + f"\n\nObjective: {objective}"
        )
        state = lead.invoke(prompt)
        raw = self._extract_output(state)
        return self._parse_plan(objective, raw)

    def _format_profiles(self) -> str:
        return "\n".join(
            f"- {p.id}: {p.name or p.id} (persona={p.persona}, tools={p.tools or 'all'})"
            for p in self.factory.list_profiles()
        )

    def _parse_steps(self, raw: str, objective: str) -> list[PlanStep]:
        """Parse planner JSON output into PlanStep list, with fallback."""
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`").split("\n", 1)[-1]
        try:
            data = json.loads(text)
            steps = []
            for s in data.get("steps", []):
                steps.append(PlanStep(
                    id=s["id"],
                    description=s.get("description", ""),
                    tool=s.get("tool"),
                    parameters=s.get("parameters", {}),
                    dependencies=s.get("dependencies", []),
                    agent_id=s.get("agent_id", "lead"),
                ))
            if steps:
                return steps
        except Exception:
            logger.debug("loads failed", exc_info=True)
        # Fallback: single step for lead agent
        return [PlanStep(id="s1", description=text or objective, agent_id="lead")]

    def _plan_to_taskplan(self, plan: Plan) -> TaskPlan:
        """Convert a persisted Plan to the in-memory TaskPlan for execution."""
        tasks = []
        for step in plan.steps:
            tasks.append(SubTask(
                task_id=step.id,
                agent_id=step.agent_id,
                prompt=self._step_to_prompt(step),
                depends_on=list(step.dependencies),
            ))
        return TaskPlan(objective=plan.objective, tasks=tasks)

    @staticmethod
    def _step_to_prompt(step: PlanStep) -> str:
        """Build the executor prompt for one step."""
        parts = [f"Execute step {step.id}: {step.description}"]
        if step.tool:
            parts.append(f"Use tool: {step.tool}")
            if step.parameters:
                parts.append(f"Parameters: {json.dumps(step.parameters, ensure_ascii=False)}")
        return "\n".join(parts)

    def _sync_step_statuses(self, plan_id: str, task_plan: TaskPlan) -> None:
        """Push SubTask statuses back into the persisted Plan's steps."""
        for task in task_plan.tasks:
            fields: dict[str, Any] = {"status": task.status}
            if task.result:
                fields["result"] = task.result
            if task.status == "error":
                fields["error"] = task.result
            try:
                self.plan_store.update_step(plan_id, task.task_id, **fields)
            except Exception:
                logger.debug("update step failed", exc_info=True)

    def _parse_plan(self, objective: str, raw: str) -> TaskPlan:
        """Parse legacy planner output into a TaskPlan, with fallback."""
        text = raw.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.strip("`").split("\n", 1)[-1]
        try:
            data = json.loads(text)
            tasks = [SubTask(**t) for t in data.get("tasks", [])]
        except Exception:
            # Fallback: single task for lead agent
            tasks = [
                SubTask(
                    task_id="t1",
                    agent_id="lead",
                    prompt=text or objective,
                )
            ]
        return TaskPlan(objective=objective, tasks=tasks)

    async def execute(
        self,
        plan_or_id: Plan | TaskPlan | str,
        on_status: Any | None = None,
    ) -> OrchestratorResult:
        """Execute a plan by id, Plan object, or legacy TaskPlan.

        - str: load confirmed Plan from store, mark executing, run, mark done/failed.
        - Plan: convert to TaskPlan, run, sync step statuses.
        - TaskPlan: legacy in-memory path (no store involvement).
        """
        if isinstance(plan_or_id, str):
            return await self._execute_by_id(plan_or_id, on_status)
        if isinstance(plan_or_id, TaskPlan):
            return await self._execute_taskplan(plan_or_id, on_status)
        # Plan object
        task_plan = self._plan_to_taskplan(plan_or_id)
        result = await self._execute_taskplan(task_plan, on_status)
        if self.plan_store is not None:
            self._sync_step_statuses(plan_or_id.id, task_plan)
        return result

    async def _execute_by_id(
        self, plan_id: str, on_status: Any | None = None
    ) -> OrchestratorResult:
        """Load a confirmed plan from the store and execute it."""
        if self.plan_store is None:
            raise ValueError("plan_store is not configured, cannot execute by id")
        plan = self.plan_store.get_plan(plan_id)
        if plan is None:
            raise ValueError(f"plan not found: {plan_id}")
        if plan.status != "confirmed":
            raise ValueError(
                f"plan {plan_id} status={plan.status}, need 'confirmed' to execute"
            )
        self.plan_store.mark_executing(plan_id)
        task_plan = self._plan_to_taskplan(plan)
        result = await self._execute_taskplan(task_plan, on_status)
        self._sync_step_statuses(plan_id, task_plan)
        if result.success:
            self.plan_store.complete_plan(plan_id)
        else:
            self.plan_store.fail_plan(plan_id, result.error)
        return result

    async def _execute_taskplan(
        self, plan: TaskPlan, on_status: Any | None = None
    ) -> OrchestratorResult:
        """Execute a TaskPlan respecting dependencies and concurrency limits."""
        result = OrchestratorResult(objective=plan.objective, plan=plan)
        semaphore = asyncio.Semaphore(self.max_concurrent)
        completed: set[str] = set()
        running: dict[str, asyncio.Task] = {}

        async def run_task(task: SubTask) -> None:
            async with semaphore:
                task.status = "running"
                if on_status:
                    with contextlib.suppress(Exception):
                        on_status(task)
                try:
                    # Build context from dependencies
                    dep_context = ""
                    for dep_id in task.depends_on:
                        dep_out = result.outputs.get(dep_id, "")
                        if dep_out:
                            dep_context += f"\n\nOutput from {dep_id}:\n{dep_out}"
                    full_prompt = f"{task.prompt}{dep_context}"

                    # Each sub-agent shares the long-term memory but gets its own session
                    sub_memory = self._make_sub_memory()
                    agent = self.factory.create(
                        task.agent_id,
                        memory_manager=sub_memory,
                    )
                    state = await asyncio.to_thread(agent.invoke, full_prompt)
                    output = self._extract_output(state)
                    task.result = output
                    task.status = "done"
                    result.outputs[task.task_id] = output
                except Exception as exc:
                    task.status = "error"
                    task.result = f"Error: {exc}"
                    result.outputs[task.task_id] = task.result
                if on_status:
                    with contextlib.suppress(Exception):
                        on_status(task)
                completed.add(task.task_id)

        pending = list(plan.tasks)
        while pending or running:
            ready = [t for t in pending if set(t.depends_on) <= completed]
            for t in ready:
                pending.remove(t)
                coro = asyncio.create_task(run_task(t))
                running[t.task_id] = coro

            if not running:
                # Cycle detection / stuck
                if pending:
                    result.success = False
                    result.error = f"Dependency cycle or unreachable tasks: {[t.task_id for t in pending]}"
                    return result
                break

            done, _ = await asyncio.wait(
                running.values(),
                return_when=asyncio.FIRST_COMPLETED,
            )
            for d in done:
                for tid, task in list(running.items()):
                    if task is d:
                        del running[tid]
                        break

        result.success = all(t.status == "done" for t in plan.tasks)
        return result

    async def synthesize(self, result: OrchestratorResult) -> str:
        """Ask the lead agent to synthesize sub-agent outputs into a final answer."""
        if len(result.plan.tasks) == 1:
            return result.plan.tasks[0].result

        parts = [f"Objective: {result.objective}\n\nSub-agent outputs:"]
        for task in result.plan.tasks:
            parts.append(f"\n## {task.task_id} ({task.agent_id})\n{task.result}")
        prompt = (
            "You are the lead agent. Synthesize the following sub-agent outputs into a "
            "coherent final answer for the user. Be concise and highlight any conflicts or gaps.\n"
            + "\n".join(parts)
        )
        lead = self.factory.create_lead()
        state = await asyncio.to_thread(lead.invoke, prompt)
        summary = self._extract_output(state)
        result.summary = summary
        return summary

    async def run(
        self, objective: str, on_status: Any | None = None, auto_confirm: bool | None = None
    ) -> OrchestratorResult | Plan:
        """Plan, execute, and synthesize — or plan-only for review.

        - plan_store is None: legacy path (plan → execute → synthesize).
        - plan_store set + auto_confirm: plan → execute → synthesize in one call.
        - plan_store set + not auto_confirm: plan only, return draft Plan for review.
        """
        ac = auto_confirm if auto_confirm is not None else self.auto_confirm
        plan = await self.plan(objective, auto_confirm=ac)

        # Legacy path or auto-confirm: execute immediately
        if self.plan_store is None or ac:
            if isinstance(plan, TaskPlan):
                result = await self.execute(plan, on_status=on_status)
            else:
                # Plan object with auto_confirm — execute by id
                result = await self.execute(plan.id, on_status=on_status)
            if not result.success:
                return result
            result.summary = await self.synthesize(result)
            return result

        # Review path: return draft plan for user to confirm
        return plan

    def _make_sub_memory(self) -> MemoryManager:
        """Create a MemoryManager that shares long-term memory but has a fresh session."""
        longterm = (
            self.memory_manager.longterm if self.memory_manager else LongTermMemory()
        )
        return MemoryManager(longterm=longterm)

    @staticmethod
    def _extract_output(state: Any) -> str:
        if isinstance(state, dict):
            messages = state.get("messages", [])
            if messages:
                last = messages[-1]
                if hasattr(last, "content"):
                    return str(last.content)
                return str(last)
        return str(state)
