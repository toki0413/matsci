"""Multi-agent swarm orchestration.

HuginnSwarm coordinates specialized workers:

- Planner: breaks a user task into steps
- Scientist: chooses physical models and equations
- Coder: writes code / tool calls
- Critic: reviews outputs for correctness
- Executor: runs external tools (often the main HuginnAgent)

This is intentionally a scaffold: the routing is rule-based today but can
be replaced with a learned supervisor later.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class AgentRole(str, enum.Enum):
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
class SwarmStep:
    """One step in a swarm execution trace."""

    role: AgentRole
    agent_name: str
    input_task: str
    output: str
    duration_ms: float = 0.0


class HuginnSwarm:
    """Rule-based multi-agent orchestrator."""

    def __init__(self, workers: list[SwarmAgent]) -> None:
        self.workers = {w.role: w for w in workers}
        self.trace: list[SwarmStep] = []

    def add_worker(self, worker: SwarmAgent) -> "HuginnSwarm":
        self.workers[worker.role] = worker
        return self

    async def run(self, task: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run the task through the swarm and return the final result."""
        self.trace.clear()
        ctx = dict(context or {})
        ctx["original_task"] = task

        # 1. Planning
        plan = await self._delegate(AgentRole.PLANNER, task, ctx)
        if not plan:
            plan = "1. Analyze the task.\n2. Execute.\n3. Review."
        ctx["plan"] = plan

        # 2. Scientific reasoning (if available)
        if AgentRole.SCIENTIST in self.workers:
            scientific_input = f"Task: {task}\nPlan: {plan}"
            scientific_output = await self._delegate(AgentRole.SCIENTIST, scientific_input, ctx)
        else:
            scientific_output = ""
        ctx["scientific_reasoning"] = scientific_output

        # 3. Coding / tool design
        if AgentRole.CODER in self.workers:
            coder_input = (
                f"Task: {task}\n"
                f"Plan: {plan}\n"
                f"Scientific reasoning: {ctx.get('scientific_reasoning', 'N/A')}"
            )
            code_output = await self._delegate(AgentRole.CODER, coder_input, ctx)
        else:
            code_output = ""
        ctx["code"] = code_output

        # 4. Execution (fallback to running the original task if no executor)
        executor = self.workers.get(AgentRole.EXECUTOR)
        if executor:
            exec_input = self._build_execution_prompt(task, ctx)
            exec_output = await self._run_agent(executor, exec_input, ctx)
        else:
            exec_output = "No executor worker configured."
        ctx["execution_result"] = exec_output

        # 5. Critic review
        if AgentRole.CRITIC in self.workers:
            critic_input = (
                f"Task: {task}\n"
                f"Plan: {plan}\n"
                f"Execution result: {exec_output}"
            )
            critic_output = await self._delegate(AgentRole.CRITIC, critic_input, ctx)
        else:
            critic_output = ""
        ctx["review"] = critic_output

        return {
            "task": task,
            "context": ctx,
            "trace": [self._step_to_dict(s) for s in self.trace],
            "final_output": exec_output,
        }

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
        import time

        start = time.time()
        full_prompt = f"{worker.instructions}\n\n{task}" if worker.instructions else task
        final_output = ""
        async for state in worker.agent.chat(full_prompt, thread_id=ctx.get("thread_id", "swarm")):
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
    def _build_execution_prompt(task: str, ctx: dict[str, Any]) -> str:
        parts = [f"Execute the following task: {task}"]
        if "plan" in ctx:
            parts.append(f"Plan:\n{ctx['plan']}")
        if "scientific_reasoning" in ctx:
            parts.append(f"Scientific reasoning:\n{ctx['scientific_reasoning']}")
        if "code" in ctx:
            parts.append(f"Code / tool design:\n{ctx['code']}")
        return "\n\n".join(parts)

    @staticmethod
    def _step_to_dict(step: SwarmStep) -> dict[str, Any]:
        return {
            "role": step.role.value,
            "agent_name": step.agent_name,
            "input_task": step.input_task,
            "output": step.output,
            "duration_ms": step.duration_ms,
        }
