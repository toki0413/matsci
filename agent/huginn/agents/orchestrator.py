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
from huginn.memory.longterm import LongTermMemory
from huginn.memory.manager import MemoryManager


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


class Orchestrator:
    """Coordinates multi-agent task execution."""

    def __init__(
        self,
        factory: AgentFactory,
        memory_manager: MemoryManager | None = None,
        max_concurrent: int = 3,
    ):
        self.factory = factory
        self.memory_manager = memory_manager
        self.max_concurrent = max_concurrent

    async def plan(self, objective: str) -> TaskPlan:
        """Ask the lead agent to produce a task plan."""
        profiles = "\n".join(
            f"- {p.id}: {p.name or p.id} (persona={p.persona}, tools={p.tools or 'all'})"
            for p in self.factory.list_profiles()
        )
        lead = self.factory.create_lead()
        prompt = (
            _PLANNER_PROMPT.format(profiles=profiles) + f"\n\nObjective: {objective}"
        )
        state = lead.invoke(prompt)
        raw = (
            state.get("messages", [{}])[-1].content
            if isinstance(state, dict) and "messages" in state
            else str(state)
        )
        return self._parse_plan(objective, raw)

    def _parse_plan(self, objective: str, raw: str) -> TaskPlan:
        """Parse planner output into a TaskPlan, with fallback."""
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
        plan: TaskPlan,
        on_status: Any | None = None,
    ) -> OrchestratorResult:
        """Execute a task plan respecting dependencies and concurrency limits."""
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
        self, objective: str, on_status: Any | None = None
    ) -> OrchestratorResult:
        """Plan, execute, and synthesize in one call."""
        plan = await self.plan(objective)
        result = await self.execute(plan, on_status=on_status)
        if not result.success:
            return result
        result.summary = await self.synthesize(result)
        return result

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
