"""P0 integration tests for the high-throughput sweep loop
(tools/sci/high_throughput_tool.py).

Drives the real HighThroughputTool.call() with a mock target tool
registered in ToolRegistry. Also tests ParameterSweep.check_early_termination()
directly since the tool itself runs all jobs via asyncio.gather.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from huginn.tools.registry import ToolRegistry
from huginn.tools.sci.high_throughput_tool import HighThroughputTool
from huginn.types import ToolContext, ToolResult
from huginn.workflows.high_throughput import (
    GridSpace,
    ParameterSweep,
    RandomSpace,
)


# ── mock target tool ──────────────────────────────────────────────


class _MockCalcTool:
    """Fake computational tool that returns an energy based on encut."""

    name = "mock_calc"
    input_schema = None  # skip pydantic validation in the sweep runner

    def __init__(self, energy_fn=None):
        self._energy_fn = energy_fn or (lambda params: -1.0 * params.get("encut", 400) / 100)
        self.call_count = 0

    async def call(self, inputs: dict, context: ToolContext) -> ToolResult:
        self.call_count += 1
        energy = self._energy_fn(inputs)
        return ToolResult(data={"energy": energy, "encut": inputs.get("encut")}, success=True)


@pytest.fixture
def registered_tool(monkeypatch):
    """Register a mock tool in ToolRegistry and clean up after."""
    tool = _MockCalcTool()
    # save/restore the class-level dict so other tests aren't affected
    original = dict(ToolRegistry._tools)
    ToolRegistry.register(tool)
    yield tool
    ToolRegistry._tools = original


def _ctx() -> ToolContext:
    return ToolContext(session_id="ht-test", workspace="/tmp/ht-test")


# ── 1. grid sweep: 2 × 2 = 4 combinations ─────────────────────────


class TestHighThroughputGrid:
    @pytest.mark.asyncio
    async def test_grid_produces_all_combos(self, registered_tool):
        """{encut:[300,400], kpoints:["2 2 2","4 4 4"]} → 4 jobs."""
        tool = HighThroughputTool()
        result = await tool.call(
            {
                "tool_name": "mock_calc",
                "space_type": "grid",
                "parameter_space": {
                    "encut": [300, 400],
                    "kpoints": ["2 2 2", "4 4 4"],
                },
                "base_input": {"structure": "POSCAR"},
                "max_parallel": 2,
            },
            _ctx(),
        )

        assert result.success
        assert result.data["n_total"] == 4
        assert result.data["n_completed"] == 4
        assert result.data["n_failed"] == 0
        assert registered_tool.call_count == 4

        # verify all 4 parameter combos are present
        param_sets = [j["parameters"] for j in result.data["jobs"]]
        encuts = sorted(p["encut"] for p in param_sets)
        assert encuts == [300, 300, 400, 400]
        kpts = sorted(p["kpoints"] for p in param_sets)
        assert kpts == ["2 2 2", "2 2 2", "4 4 4", "4 4 4"]


# ── 2. random sweep: n_samples=20 → 20 samples ───────────────────


class TestHighThroughputRandom:
    @pytest.mark.asyncio
    async def test_random_produces_n_samples(self, registered_tool):
        """Random space with n_samples=20 produces exactly 20 jobs."""
        tool = HighThroughputTool()
        result = await tool.call(
            {
                "tool_name": "mock_calc",
                "space_type": "random",
                "parameter_space": {
                    "encut": [200, 600],
                },
                "n_samples": 20,
                "seed": 42,
                "max_parallel": 4,
            },
            _ctx(),
        )

        assert result.success
        assert result.data["n_total"] == 20
        assert result.data["n_completed"] == 20
        assert registered_tool.call_count == 20

        # all sampled encut values should be within the range
        encuts = [j["parameters"]["encut"] for j in result.data["jobs"]]
        assert all(200 <= e <= 600 for e in encuts)


# ── 3. early termination: condition met → stops ───────────────────


class TestHighThroughputEarlyTermination:
    def test_condition_met_returns_true(self):
        """check_early_termination returns True when condition is satisfied.

        ponytail: HighThroughputTool.call() doesn't actually call
        check_early_termination mid-sweep (it runs all jobs via gather).
        We test the ParameterSweep method directly — that's where the
        safe_eval logic lives.
        """
        space = GridSpace({"encut": [300, 400, 500, 600]})
        sweep = ParameterSweep(
            name="early_term_test",
            tool_name="mock_calc",
            parameter_space=space,
            early_termination="n_completed >= 2",
        )
        jobs = sweep.generate_jobs()
        assert len(jobs) == 4

        # complete first 2 jobs with dummy results
        sweep.update_job(jobs[0].job_id, "completed", result={"energy": -3.1})
        sweep.update_job(jobs[1].job_id, "completed", result={"energy": -4.2})

        assert sweep.check_early_termination() is True

    def test_condition_not_met_returns_false(self):
        """Returns False when fewer than threshold completed."""
        space = GridSpace({"encut": [300, 400]})
        sweep = ParameterSweep(
            name="no_early",
            tool_name="mock_calc",
            parameter_space=space,
            early_termination="n_completed >= 3",
        )
        jobs = sweep.generate_jobs()
        sweep.update_job(jobs[0].job_id, "completed", result={"energy": -1.0})

        assert sweep.check_early_termination() is False

    def test_no_expression_returns_false(self):
        """No early_termination expression → always False."""
        space = GridSpace({"encut": [300]})
        sweep = ParameterSweep(
            name="no_expr",
            tool_name="mock_calc",
            parameter_space=space,
        )
        sweep.generate_jobs()
        assert sweep.check_early_termination() is False
