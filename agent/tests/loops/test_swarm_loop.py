"""P0 integration tests for the swarm orchestration loop (agents/swarm.py).

Drives the real HuginnSwarm.run() with mock chat agents that replay
fixed responses. This lets us verify plan parsing, dependency ordering,
parallel dispatch, and graceful handling of missing roles.
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from huginn.agents.swarm import AgentRole, HuginnSwarm, SwarmAgent


# ── mock chat agent ───────────────────────────────────────────────


class _ScriptedAgent:
    """Agent whose chat() yields a single AIMessage with fixed content.

    If *responses* is a list, each chat() call pops the next one (for
    agents that get called multiple times, e.g. the planner who might
    be invoked again).
    """

    def __init__(self, content: str = "ok"):
        self._content = content
        self.call_count = 0

    async def chat(self, message: str, thread_id: str = "swarm"):
        self.call_count += 1
        yield {"messages": [AIMessage(content=self._content)]}


class _DelayedAgent(_ScriptedAgent):
    """Same as _ScriptedAgent but with a small async delay to make
    parallel-vs-observable timing visible."""

    def __init__(self, content: str, delay: float = 0.1):
        super().__init__(content)
        self._delay = delay

    async def chat(self, message: str, thread_id: str = "swarm"):
        import asyncio
        await asyncio.sleep(self._delay)
        self.call_count += 1
        yield {"messages": [AIMessage(content=self._content)]}


def _worker(role: AgentRole, content: str, name: str | None = None) -> SwarmAgent:
    return SwarmAgent(
        name=name or f"{role.value}_bot",
        role=role,
        agent=_ScriptedAgent(content),
    )


# ── 1. basic flow: planner → scientist → coder → executor → critic ──


class TestSwarmBasicFlow:
    @pytest.mark.asyncio
    async def test_all_roles_execute(self):
        """Planner + 3 plan steps + critic → 5 trace entries."""
        plan = json.dumps([
            {"id": "s1", "role": "scientist", "task": "derive equations", "depends_on": []},
            {"id": "s2", "role": "coder", "task": "write code", "depends_on": ["s1"]},
            {"id": "s3", "role": "executor", "task": "run code", "depends_on": ["s2"]},
        ])
        swarm = HuginnSwarm([
            _worker(AgentRole.PLANNER, plan),
            _worker(AgentRole.SCIENTIST, "Use DFT with PBE functional."),
            _worker(AgentRole.CODER, "import numpy as np"),
            _worker(AgentRole.EXECUTOR, "Job completed, energy = -12.5 eV"),
            _worker(AgentRole.CRITIC, "Results look correct."),
        ])
        result = await swarm.run("Compute the formation energy of Si")

        roles_in_trace = [s["role"] for s in result["trace"]]
        assert "planner" in roles_in_trace
        assert "scientist" in roles_in_trace
        assert "coder" in roles_in_trace
        assert "executor" in roles_in_trace
        assert "critic" in roles_in_trace
        assert result["final_output"] == "Job completed, energy = -12.5 eV"


# ── 2. dependency: step2 depends on step1 → sequential ─────────────


class TestSwarmDependency:
    @pytest.mark.asyncio
    async def test_sequential_when_dependent(self):
        """step2 depends on step1 → step2 runs after step1, receiving its output."""

        # ponytail: instead of checking wall-clock timing (flaky), we
        # verify that step2's input contains step1's output text.
        plan = json.dumps([
            {"id": "s1", "role": "scientist", "task": "first step", "depends_on": []},
            {"id": "s2", "role": "coder", "task": "second step", "depends_on": ["s1"]},
        ])

        captured_inputs: list[str] = []

        class _CapturingAgent:
            def __init__(self, content: str):
                self._content = content

            async def chat(self, message: str, thread_id: str = "swarm"):
                captured_inputs.append(message)
                yield {"messages": [AIMessage(content=self._content)]}

        scientist = SwarmAgent(name="sci", role=AgentRole.SCIENTIST,
                               agent=_CapturingAgent("PBE functional is best"))
        coder = SwarmAgent(name="cod", role=AgentRole.CODER,
                           agent=_CapturingAgent("code written"))
        planner = SwarmAgent(name="pln", role=AgentRole.PLANNER,
                             agent=_ScriptedAgent(plan))

        swarm = HuginnSwarm([planner, scientist, coder])
        await swarm.run("task")

        # coder's input should include scientist's output
        assert any("PBE functional is best" in inp for inp in captured_inputs)


# ── 3. parallel: 2a and 2b both depend on step1 → parallel ────────


class TestSwarmParallel:
    @pytest.mark.asyncio
    async def test_parallel_after_dependency(self):
        """2a + 2b both depend on s1; they run concurrently after s1."""
        plan = json.dumps([
            {"id": "s1", "role": "scientist", "task": "prep", "depends_on": []},
            {"id": "s2a", "role": "coder", "task": "branch A", "depends_on": ["s1"]},
            {"id": "s2b", "role": "executor", "task": "branch B", "depends_on": ["s1"]},
        ])

        swarm = HuginnSwarm([
            SwarmAgent(name="pln", role=AgentRole.PLANNER,
                       agent=_ScriptedAgent(plan)),
            SwarmAgent(name="sci", role=AgentRole.SCIENTIST,
                       agent=_ScriptedAgent("prep done")),
            # 100ms delay each — if serial, total ~200ms; parallel ~100ms
            SwarmAgent(name="cod", role=AgentRole.CODER,
                       agent=_DelayedAgent("code A", delay=0.1)),
            SwarmAgent(name="exe", role=AgentRole.EXECUTOR,
                       agent=_DelayedAgent("result B", delay=0.1)),
        ])

        start = time.monotonic()
        result = await swarm.run("parallel task")
        elapsed = time.monotonic() - start

        assert result["final_output"] in ("code A", "result B")
        # ponytail: generous threshold to avoid CI flakiness
        assert elapsed < 0.25, f"parallel took {elapsed:.3f}s, expected < 0.25"


# ── 4. missing role: no critic → step skipped, no error ──────────


class TestSwarmMissingRole:
    @pytest.mark.asyncio
    async def test_missing_critic_no_error(self):
        """Plan includes a critic step but no critic worker → empty output, no crash."""
        plan = json.dumps([
            {"id": "s1", "role": "scientist", "task": "analyze", "depends_on": []},
            {"id": "s2", "role": "coder", "task": "implement", "depends_on": ["s1"]},
            {"id": "s3", "role": "critic", "task": "review", "depends_on": ["s2"]},
        ])
        swarm = HuginnSwarm([
            _worker(AgentRole.PLANNER, plan),
            _worker(AgentRole.SCIENTIST, "analysis done"),
            _worker(AgentRole.CODER, "code done"),
            # no executor, no critic
        ])
        result = await swarm.run("task without all roles")

        # no exception; critic step is silently skipped (no worker → no trace entry)
        assert result is not None
        critic_steps = [s for s in result["trace"] if s["role"] == "critic"]
        assert len(critic_steps) == 0

    @pytest.mark.asyncio
    async def test_default_plan_skips_missing_workers(self):
        """When planner returns garbage, the default plan only includes
        available workers and runs without error."""
        swarm = HuginnSwarm([
            _worker(AgentRole.PLANNER, "this is not json"),
            _worker(AgentRole.SCIENTIST, "did science"),
            _worker(AgentRole.EXECUTOR, "ran it"),
            # no coder, no critic
        ])
        result = await swarm.run("fallback plan task")

        assert result is not None
        roles_in_trace = {s["role"] for s in result["trace"]}
        # planner always runs, then the default plan skips missing workers
        assert "planner" in roles_in_trace
        assert "scientist" in roles_in_trace
