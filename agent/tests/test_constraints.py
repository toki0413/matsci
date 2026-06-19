"""Tests for the constraint system."""

from __future__ import annotations

import pytest

from huginn.constraints import (
    Constraint,
    ConstraintAdapter,
    ConstraintResult,
    QualityOperator,
    SafetyOperator,
)
from huginn.constraints.boundaries import BoundaryEvolution, BoundaryState
from huginn.tools.validate_tool import ValidateTool
from huginn.types import ToolContext


class TestReferenceDomain:
    def test_constraint_evaluate(self):
        def _check(data):
            return ConstraintResult(
                name="positive",
                passed=data["x"] > 0,
                value=data["x"],
                expected=">0",
                tolerance=0,
                message="",
            )

        c = Constraint("positive", "test", "quality", "warn", _check)
        assert c.evaluate({"x": 1}).passed is True
        assert c.evaluate({"x": -1}).passed is False


class TestOperators:
    def test_safety_operator_scope_filter(self):
        def _always_pass(_data):
            return ConstraintResult("p", True, None, "", 0, "", family="safety")

        op = SafetyOperator(
            [
                Constraint("a", "dft", "safety", "warn", _always_pass),
                Constraint("b", "md", "safety", "warn", _always_pass),
            ]
        )
        results = op.evaluate({}, scope="dft")
        assert len(results) == 1
        assert results[0].family == "safety"

    def test_quality_operator_family_set(self):
        def _check(_data):
            return ConstraintResult("q", True, None, "", 0, "")

        op = QualityOperator([Constraint("q", "dft", "quality", "warn", _check)])
        results = op.evaluate({}, scope="dft")
        assert results[0].family == "quality"


class TestAdapter:
    def test_default_adapter_dft(self):
        adapter = ConstraintAdapter.default()
        safety, quality = adapter.evaluate(
            "dft",
            {"energy": -10.0, "max_force": 0.005, "band_gap": 1.0, "volume": 100.0},
        )
        assert all(r.passed for r in safety)
        assert all(r.passed for r in quality)

    def test_default_adapter_dft_fails(self):
        adapter = ConstraintAdapter.default()
        _, quality = adapter.evaluate(
            "dft", {"energy": 10.0, "max_force": 0.1, "band_gap": -0.5, "volume": -1.0}
        )
        passed = {r.name: r.passed for r in quality}
        assert passed["energy_sign"] is False
        assert passed["force_convergence"] is False
        assert passed["band_gap"] is False
        assert passed["volume_positive"] is False

    def test_default_adapter_md(self):
        adapter = ConstraintAdapter.default()
        _, quality = adapter.evaluate(
            "md",
            {
                "energy_drift_per_atom": 0.0001,
                "temperature_std": 5.0,
                "target_temperature": 300.0,
                "initial_atom_count": 10,
                "final_atom_count": 10,
                "density": 2.3,
            },
        )
        assert all(r.passed for r in quality)


class TestBoundaryEvolution:
    def test_block_failure_disables_auto_approval(self):
        state = BoundaryState(require_confirmation=False)
        evo = BoundaryEvolution(state)
        evo.update(
            [
                ConstraintResult(
                    "x", False, None, "", 0, "", severity="block", family="safety"
                )
            ]
        )
        assert evo.state.require_confirmation is True


class TestValidateTool:
    @pytest.mark.asyncio
    async def test_validate_dft(self):
        tool = ValidateTool()
        result = await tool.call(
            tool.input_schema(
                result_type="dft",
                result_data={
                    "energy": -10.0,
                    "max_force": 0.005,
                    "band_gap": 1.0,
                    "volume": 100.0,
                },
            ),
            ToolContext(session_id="test", workspace="."),
        )
        assert result.success is True
        assert result.data["all_passed"] is True

    @pytest.mark.asyncio
    async def test_validate_md_finds_failures(self):
        tool = ValidateTool()
        result = await tool.call(
            tool.input_schema(
                result_type="md",
                result_data={
                    "energy_drift_per_atom": 0.1,
                    "temperature_std": 100.0,
                    "target_temperature": 300.0,
                    "initial_atom_count": 10,
                    "final_atom_count": 8,
                    "density": 200.0,
                },
            ),
            ToolContext(session_id="test", workspace="."),
        )
        assert result.success is True
        assert result.data["all_passed"] is False
        names = {c["name"] for c in result.data["checks"]}
        assert "energy_conservation" in names
        assert "atom_count" in names
