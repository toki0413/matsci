"""Multi-agent swarm orchestration.

HuginnSwarm coordinates specialized workers:

- Planner: breaks a user task into a JSON plan
- Scientist: chooses physical models and equations
- Coder: writes code / tool calls
- Critic: reviews outputs for correctness
- Executor: runs external tools (often the main HuginnAgent)

The supervisor executes plan steps respecting dependencies; independent
steps run in parallel.
"""

from __future__ import annotations

import asyncio
import enum
import json
import time
from dataclasses import dataclass, field
from typing import Any


class AgentRole(enum.StrEnum):
    PLANNER = "planner"
    SCIENTIST = "scientist"
    CODER = "coder"
    CRITIC = "critic"
    EXECUTOR = "executor"


@dataclass
class SwarmAgent:
    """A worker agent in the swarm."""

    name: str
    role: AgentRole
    agent: Any
    instructions: str = ""


@dataclass
class SwarmPlanStep:
    """One step in a swarm execution plan."""

    id: str
    role: AgentRole
    task: str
    depends_on: list[str] = field(default_factory=list)


@dataclass
class SwarmStep:
    """One step in a swarm execution trace."""

    role: AgentRole
    agent_name: str
    input_task: str
    output: str
    duration_ms: float = 0.0


class HuginnSwarm:
    """Supervisor-based multi-agent orchestrator."""

    # Default plan prompt if the user-supplied planner is unavailable.
    _PLANNER_PROMPT = (
        "You are a task planner. Break the user task into steps. "
        "Respond with a JSON array only. Each item must have:\n"
        '{"id": "step1", "role": "scientist", "task": "...", "depends_on": []}\n'
        "Available roles: scientist, coder, executor, critic. "
        "Use depends_on to declare steps that must finish before this one."
    )

    def __init__(self, workers: list[SwarmAgent]) -> None:
        self.workers = {w.role: w for w in workers}
        self.trace: list[SwarmStep] = []

    def add_worker(self, worker: SwarmAgent) -> HuginnSwarm:
        self.workers[worker.role] = worker
        return self

    async def run(
        self, task: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Run the task through the swarm and return the final result."""
        self.trace.clear()
        ctx = dict(context or {})
        ctx["original_task"] = task

        # 1. Planning
        plan_text = await self._delegate(AgentRole.PLANNER, task, ctx)
        ctx["planner_output"] = plan_text
        steps = self._parse_plan(plan_text)
        if not steps:
            steps = self._default_plan(task)
        ctx["plan"] = self._plan_to_text(steps)

        # 2. Execute planned steps respecting dependencies.
        step_outputs = await self._execute_plan(steps, ctx)

        # Map outputs to legacy context keys for convenience.
        role_outputs: dict[AgentRole, str] = {}
        for step, output in zip(steps, step_outputs):
            role_outputs[step.role] = output
        ctx["scientific_reasoning"] = role_outputs.get(AgentRole.SCIENTIST, "")
        ctx["code"] = role_outputs.get(AgentRole.CODER, "")
        ctx["execution_result"] = role_outputs.get(
            AgentRole.EXECUTOR, "No executor step completed."
        )

        # 3. Critic review (only if the plan did not already include one).
        critic_output = role_outputs.get(AgentRole.CRITIC, "")
        if not critic_output and AgentRole.CRITIC in self.workers:
            critic_input = (
                f"Task: {task}\n"
                f"Plan: {ctx['plan']}\n"
                f"Execution result: {ctx['execution_result']}"
            )
            critic_output = await self._delegate(AgentRole.CRITIC, critic_input, ctx)
        ctx["review"] = critic_output

        return {
            "task": task,
            "context": ctx,
            "trace": [self._step_to_dict(s) for s in self.trace],
            "final_output": ctx["execution_result"],
        }

    def _parse_plan(self, text: str) -> list[SwarmPlanStep]:
        """Parse planner output into steps."""
        if not text:
            return []
        # Strip markdown fences.
        if "```" in text:
            parts = text.split("```")
            if len(parts) >= 3:
                text = parts[1].strip("json").strip()
        try:
            data = json.loads(text)
        except Exception:
            return []
        if not isinstance(data, list):
            return []

        steps: list[SwarmPlanStep] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                steps.append(
                    SwarmPlanStep(
                        id=str(item.get("id", f"step{len(steps) + 1}")),
                        role=AgentRole(item.get("role", "executor")),
                        task=str(item.get("task", "")),
                        depends_on=[str(d) for d in item.get("depends_on", []) if d],
                    )
                )
            except Exception:
                continue
        return steps

    def _default_plan(self, task: str) -> list[SwarmPlanStep]:
        """Fallback plan when planner output is unusable."""
        steps: list[SwarmPlanStep] = []
        order = [
            AgentRole.SCIENTIST,
            AgentRole.CODER,
            AgentRole.EXECUTOR,
            AgentRole.CRITIC,
        ]
        prev_id: str | None = None
        for role in order:
            if role not in self.workers:
                continue
            step_id = f"{role.value}_step"
            depends = [prev_id] if prev_id else []
            steps.append(
                SwarmPlanStep(
                    id=step_id,
                    role=role,
                    task=f"{role.value.replace('_', ' ').title()} for: {task}",
                    depends_on=depends,
                )
            )
            prev_id = step_id
        return steps

    @staticmethod
    def _plan_to_text(steps: list[SwarmPlanStep]) -> str:
        lines = []
        for s in steps:
            deps = f" (after {', '.join(s.depends_on)})" if s.depends_on else ""
            lines.append(f"{s.id}: [{s.role.value}] {s.task}{deps}")
        return "\n".join(lines)

    async def _execute_plan(
        self,
        steps: list[SwarmPlanStep],
        ctx: dict[str, Any],
    ) -> list[str]:
        """Execute steps respecting dependencies; independent steps run in parallel."""
        results: dict[str, str] = {}
        pending = {s.id: s for s in steps}

        while pending:
            ready = [
                s
                for s in pending.values()
                if all(dep in results for dep in s.depends_on)
            ]
            if not ready:
                # Cyclic dependency fallback: run remaining sequentially.
                ready = list(pending.values())

            async def run_one(step: SwarmPlanStep) -> tuple[str, str]:
                worker = self.workers.get(step.role)
                if not worker:
                    return step.id, ""
                # Build input using outputs from dependencies.
                dep_text = "\n".join(
                    f"{dep}: {results[dep]}"
                    for dep in step.depends_on
                    if dep in results
                )
                task = step.task
                if dep_text:
                    task = f"{task}\n\nContext from previous steps:\n{dep_text}"
                output = await self._run_agent(worker, task, ctx)
                return step.id, output

            batch_results = await asyncio.gather(*(run_one(s) for s in ready))
            for step_id, output in batch_results:
                results[step_id] = output
                pending.pop(step_id)

        return [results[s.id] for s in steps]

    async def _delegate(self, role: AgentRole, task: str, ctx: dict[str, Any]) -> str:
        worker = self.workers.get(role)
        if not worker:
            return ""
        return await self._run_agent(worker, task, ctx)

    async def _run_agent(
        self,
        worker: SwarmAgent,
        task: str,
        ctx: dict[str, Any],
    ) -> str:
        start = time.time()
        full_prompt = (
            f"{worker.instructions}\n\n{task}" if worker.instructions else task
        )
        final_output = ""
        async for state in worker.agent.chat(
            full_prompt, thread_id=ctx.get("thread_id", "swarm")
        ):
            messages = state.get("messages", [])
            for msg in messages:
                content = getattr(msg, "content", None)
                if content:
                    final_output = str(content)
        duration_ms = round((time.time() - start) * 1000, 2)
        step = SwarmStep(
            role=worker.role,
            agent_name=worker.name,
            input_task=task,
            output=final_output,
            duration_ms=duration_ms,
        )
        self.trace.append(step)
        return final_output

    @staticmethod
    def _step_to_dict(step: SwarmStep) -> dict[str, Any]:
        return {
            "role": step.role.value,
            "agent_name": step.agent_name,
            "input_task": step.input_task,
            "output": step.output,
            "duration_ms": step.duration_ms,
        }
