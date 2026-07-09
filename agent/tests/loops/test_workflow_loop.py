"""P0 integration tests for the workflow engine (workflows/engine.py).

Drives the real WorkflowEngine.execute() with a mock tool registry so
we can verify dependency ordering, parallel dispatch, and retry logic
without needing real computational tools.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from huginn.types import ToolContext, ToolResult
from huginn.workflows.engine import WorkflowEngine
from huginn.workflows.stages import ComputationalStage, RetryPolicy


# ── mock collaborators ─────────────────────────────────────────────


class MockTool:
    """Tool that optionally fails N times before succeeding."""

    # input_schema = None so the engine skips pydantic validation
    input_schema = None

    def __init__(self, name: str, fail_times: int = 0, delay: float = 0):
        self.name = name
        self._fail_times = fail_times
        self._delay = delay
        self.call_count = 0

    async def call(self, inputs: Any, context: ToolContext) -> ToolResult:
        self.call_count += 1
        if self._delay:
            await asyncio.sleep(self._delay)
        if self.call_count <= self._fail_times:
            return ToolResult(data=None, success=False, error=f"call {self.call_count} failed")
        return ToolResult(data={"tool": self.name, "run": self.call_count}, success=True)


class MockRegistry:
    def __init__(self):
        self._tools: dict[str, MockTool] = {}

    def register(self, tool: MockTool) -> MockTool:
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> MockTool | None:
        return self._tools.get(name)


def _ctx() -> ToolContext:
    return ToolContext(session_id="test-session", workspace="/tmp/wf-test")


# ── 1. linear pipeline: stage1 → stage2 → stage3 ──────────────────


class TestWorkflowLinear:
    @pytest.mark.asyncio
    async def test_sequential_stages(self):
        """Three stages with chain dependencies execute in order."""
        reg = MockRegistry()
        t1 = reg.register(MockTool("t1"))
        t2 = reg.register(MockTool("t2"))
        t3 = reg.register(MockTool("t3"))
        engine = WorkflowEngine(reg)

        stages = [
            ComputationalStage(id="s1", name="stage1", tool="t1", tool_input={}),
            ComputationalStage(id="s2", name="stage2", tool="t2", tool_input={}, dependencies=["s1"]),
            ComputationalStage(id="s3", name="stage3", tool="t3", tool_input={}, dependencies=["s2"]),
        ]
        result = await engine.execute(stages, _ctx())

        assert result.success
        assert result.outputs["s1"]["tool"] == "t1"
        assert result.outputs["s2"]["tool"] == "t2"
        assert result.outputs["s3"]["tool"] == "t3"
        # each tool called exactly once
        assert t1.call_count == 1
        assert t2.call_count == 1
        assert t3.call_count == 1


# ── 2. parallel stages: 2a + 2b run after stage1 ─────────────────


class TestWorkflowParallel:
    @pytest.mark.asyncio
    async def test_parallel_after_dependency(self):
        """2a and 2b both depend on s1; they should run concurrently."""
        reg = MockRegistry()
        # 100ms delay so parallel execution is measurably faster than serial
        t1 = reg.register(MockTool("t1", delay=0))
        t2a = reg.register(MockTool("t2a", delay=0.1))
        t2b = reg.register(MockTool("t2b", delay=0.1))
        engine = WorkflowEngine(reg)

        stages = [
            ComputationalStage(id="s1", name="stage1", tool="t1", tool_input={}),
            ComputationalStage(id="s2a", name="stage2a", tool="t2a", tool_input={}, dependencies=["s1"]),
            ComputationalStage(id="s2b", name="stage2b", tool="t2b", tool_input={}, dependencies=["s1"]),
        ]
        start = time.monotonic()
        result = await engine.execute(stages, _ctx())
        elapsed = time.monotonic() - start

        assert result.success
        # ponytail: timing-based assertions are flaky on CI; use a generous
        # upper bound. Serial would be ~200ms for 2a+2b; parallel should be
        # well under 180ms. If this flakes, bump the threshold.
        assert elapsed < 0.18, f"parallel stages took {elapsed:.3f}s, expected < 0.18s"
        assert t2a.call_count == 1
        assert t2b.call_count == 1


# ── 3. retry: failed stage retries up to max_retries ──────────────


class TestWorkflowRetry:
    @pytest.mark.asyncio
    async def test_retry_then_succeed(self):
        """Tool fails twice then succeeds; stage completes on 3rd attempt."""
        reg = MockRegistry()
        flaky = reg.register(MockTool("flaky", fail_times=2))
        engine = WorkflowEngine(reg)

        stage = ComputationalStage(
            id="s1", name="flaky-stage", tool="flaky", tool_input={},
            retry_policy=RetryPolicy(max_retries=3, backoff_factor=0.0, auto_diagnose=False),
        )
        result = await engine.execute([stage], _ctx())

        assert result.success
        assert result.outputs["s1"]["run"] == 3
        assert flaky.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted(self):
        """Tool keeps failing; stage fails after max_retries+1 attempts."""
        reg = MockRegistry()
        always_fail = reg.register(MockTool("bad", fail_times=99))
        engine = WorkflowEngine(reg)

        stage = ComputationalStage(
            id="s1", name="bad-stage", tool="bad", tool_input={},
            retry_policy=RetryPolicy(max_retries=2, backoff_factor=0.0, auto_diagnose=False),
        )
        result = await engine.execute([stage], _ctx())

        assert not result.success
        # 1 initial + 2 retries = 3 total calls
        assert always_fail.call_count == 3
        assert result.error is not None
