"""Tests for the multi-agent swarm scaffold."""

from __future__ import annotations

import asyncio

import pytest

from huginn.agents.swarm import AgentRole, HuginnSwarm, SwarmAgent


class _FakeAgent:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    async def chat(self, message: str, thread_id: str = "default"):
        yield {
            "messages": [
                type("Msg", (), {"content": f"{self.reply}({message[:20]})"})()
            ]
        }


class _PlannerAgent:
    """Planner that returns a JSON plan."""

    def __init__(self, plan: str) -> None:
        self.plan = plan

    async def chat(self, message: str, thread_id: str = "default"):
        yield {"messages": [type("Msg", (), {"content": self.plan})()]}


class TestHuginnSwarm:
    @pytest.mark.asyncio
    async def test_swarm_runs_all_workers(self):
        swarm = HuginnSwarm(
            [
                SwarmAgent(
                    "planner",
                    AgentRole.PLANNER,
                    _PlannerAgent("""
                [
                    {"id": "s1", "role": "scientist", "task": "analyze", "depends_on": []},
                    {"id": "s2", "role": "coder", "task": "code", "depends_on": []},
                    {"id": "s3", "role": "executor", "task": "run", "depends_on": ["s1", "s2"]},
                    {"id": "s4", "role": "critic", "task": "review", "depends_on": ["s3"]}
                ]
                """),
                ),
                SwarmAgent("scientist", AgentRole.SCIENTIST, _FakeAgent("Science:")),
                SwarmAgent("coder", AgentRole.CODER, _FakeAgent("Code:")),
                SwarmAgent("executor", AgentRole.EXECUTOR, _FakeAgent("Exec:")),
                SwarmAgent("critic", AgentRole.CRITIC, _FakeAgent("Review:")),
            ]
        )
        result = await swarm.run("compute stress")
        assert result["task"] == "compute stress"
        assert "analyze" in result["context"]["plan"]
        assert "Science:" in result["context"]["scientific_reasoning"]
        assert "Code:" in result["context"]["code"]
        assert "Exec:" in result["final_output"]
        assert "Review:" in result["context"]["review"]
        assert len(result["trace"]) == 5

    @pytest.mark.asyncio
    async def test_independent_steps_run_in_parallel(self):
        delays = {"scientist": 0.05, "coder": 0.05}

        class _SlowAgent:
            def __init__(self, role: str) -> None:
                self.role = role

            async def chat(self, message: str, thread_id: str = "default"):
                await asyncio.sleep(delays[self.role])
                yield {
                    "messages": [type("Msg", (), {"content": f"{self.role}:done"})()]
                }

        start = asyncio.get_event_loop().time()
        swarm = HuginnSwarm(
            [
                SwarmAgent(
                    "planner",
                    AgentRole.PLANNER,
                    _PlannerAgent("""
                [
                    {"id": "s1", "role": "scientist", "task": "a", "depends_on": []},
                    {"id": "s2", "role": "coder", "task": "b", "depends_on": []},
                    {"id": "s3", "role": "executor", "task": "c", "depends_on": ["s1", "s2"]}
                ]
                """),
                ),
                SwarmAgent("scientist", AgentRole.SCIENTIST, _SlowAgent("scientist")),
                SwarmAgent("coder", AgentRole.CODER, _SlowAgent("coder")),
                SwarmAgent("executor", AgentRole.EXECUTOR, _FakeAgent("Exec:")),
            ]
        )
        await swarm.run("task")
        elapsed = asyncio.get_event_loop().time() - start
        # If serial, would be ~0.1s; parallel should be <0.09s.
        assert elapsed < 0.09

    @pytest.mark.asyncio
    async def test_missing_workers_skip_gracefully(self):
        swarm = HuginnSwarm(
            [
                SwarmAgent("planner", AgentRole.PLANNER, _FakeAgent("Plan:")),
                SwarmAgent("executor", AgentRole.EXECUTOR, _FakeAgent("Exec:")),
            ]
        )
        result = await swarm.run("simple task")
        assert "executor_step" in result["context"]["plan"]
        assert result["context"].get("scientific_reasoning") == ""
        assert result["context"].get("code") == ""
        assert "Exec:" in result["final_output"]
        assert len(result["trace"]) == 2  # planner + executor

    @pytest.mark.asyncio
    async def test_no_executor_reports_config_message(self):
        swarm = HuginnSwarm(
            [SwarmAgent("planner", AgentRole.PLANNER, _FakeAgent("Plan:"))]
        )
        result = await swarm.run("task")
        assert "No executor step completed" in result["final_output"]
