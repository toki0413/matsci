"""Tests for the multi-agent swarm scaffold."""

from __future__ import annotations

import pytest

from huginn.agents.swarm import AgentRole, HuginnSwarm, SwarmAgent


class _FakeAgent:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    async def chat(self, message: str, thread_id: str = "default"):
        yield {"messages": [type("Msg", (), {"content": f"{self.reply}({message[:20]})"})()]}


class TestHuginnSwarm:
    @pytest.mark.asyncio
    async def test_swarm_runs_all_workers(self):
        swarm = HuginnSwarm(
            [
                SwarmAgent("planner", AgentRole.PLANNER, _FakeAgent("Plan:")),
                SwarmAgent("scientist", AgentRole.SCIENTIST, _FakeAgent("Science:")),
                SwarmAgent("coder", AgentRole.CODER, _FakeAgent("Code:")),
                SwarmAgent("executor", AgentRole.EXECUTOR, _FakeAgent("Exec:")),
                SwarmAgent("critic", AgentRole.CRITIC, _FakeAgent("Review:")),
            ]
        )
        result = await swarm.run("compute stress")
        assert result["task"] == "compute stress"
        assert "Plan:" in result["context"]["plan"]
        assert "Science:" in result["context"]["scientific_reasoning"]
        assert "Code:" in result["context"]["code"]
        assert "Exec:" in result["final_output"]
        assert "Review:" in result["context"]["review"]
        assert len(result["trace"]) == 5

    @pytest.mark.asyncio
    async def test_missing_workers_skip_gracefully(self):
        swarm = HuginnSwarm(
            [
                SwarmAgent("planner", AgentRole.PLANNER, _FakeAgent("Plan:")),
                SwarmAgent("executor", AgentRole.EXECUTOR, _FakeAgent("Exec:")),
            ]
        )
        result = await swarm.run("simple task")
        assert "Plan:" in result["context"]["plan"]
        assert result["context"].get("scientific_reasoning") == ""
        assert result["context"].get("code") == ""
        assert "Exec:" in result["final_output"]
        assert len(result["trace"]) == 2

    @pytest.mark.asyncio
    async def test_no_executor_reports_config_message(self):
        swarm = HuginnSwarm(
            [SwarmAgent("planner", AgentRole.PLANNER, _FakeAgent("Plan:"))]
        )
        result = await swarm.run("task")
        assert "No executor worker configured" in result["final_output"]
