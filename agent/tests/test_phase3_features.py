"""Tests for Phase 3 report comparison, symbolic regression compare,
workflow custom validation, and exploration query.
"""

from __future__ import annotations

import pytest
import numpy as np

from huginn.tools.report_tool import ReportTool, ReportToolInput, ReportComparator
from huginn.tools.symbolic_regression_tool import SymbolicRegressionTool, SymbolicRegressionInput
from huginn.workflows.engine import WorkflowEngine
from huginn.workflows.stages import ValidationRule
from huginn.exploration.core import BranchStatus, Branch, Decision, ExplorationSpace
from huginn.types import ToolContext


def _ctx() -> ToolContext:
    return ToolContext(session_id="test", workspace=".")


# ═══════════════════════════════════════════════════════════════════
# Report Comparison Tests
# ═══════════════════════════════════════════════════════════════════


class TestReportComparator:
    def test_two_dataset_comparison(self):
        comp = ReportComparator(["run_A", "run_B"])
        datasets = {
            "run_A": {
                "methods": {"functional": "PBE", "encut": 400},
                "results": {"energy": -100.0},
            },
            "run_B": {
                "methods": {"functional": "PBE", "encut": 520},
                "results": {"energy": -102.5},
            },
        }
        report = comp.generate(datasets)
        assert "Comparison Report" in report
        assert "Methods Comparison" in report
        assert "Results Comparison" in report
        assert "run_A" in report
        assert "run_B" in report

    def test_three_dataset_comparison(self):
        comp = ReportComparator(["A", "B", "C"])
        datasets = {
            "A": {"methods": {"encut": 300}, "results": {}},
            "B": {"methods": {"encut": 400}, "results": {}},
            "C": {"methods": {"encut": 500}, "results": {}},
        }
        report = comp.generate(datasets)
        assert "A" in report and "B" in report and "C" in report

    def test_convergence_comparison(self):
        comp = ReportComparator(["run1", "run2"])
        datasets = {
            "run1": {
                "convergence": {"energy": -100, "n_iterations": 20, "converged": True},
                "results": {},
            },
            "run2": {
                "convergence": {"energy": -102, "n_iterations": 25, "converged": True},
                "results": {},
            },
        }
        report = comp.generate(datasets)
        assert "Convergence Comparison" in report


class TestReportToolCompare:
    def setup_method(self):
        self.tool = ReportTool()
        self.ctx = _ctx()

    @pytest.mark.asyncio
    async def test_compare_no_datasets(self):
        args = ReportToolInput(action="compare")
        result = await self.tool.call(args, self.ctx)
        assert not result.success
        assert "comparison_datasets" in result.error

    @pytest.mark.asyncio
    async def test_compare_one_dataset(self):
        args = ReportToolInput(
            action="compare",
            comparison_datasets={"only_one": {"methods": {}}},
        )
        result = await self.tool.call(args, self.ctx)
        assert not result.success
        assert "2 datasets" in result.error

    @pytest.mark.asyncio
    async def test_compare_two_datasets(self):
        args = ReportToolInput(
            action="compare",
            comparison_datasets={
                "calc_A": {"methods": {"encut": 400}, "results": {"energy": -100}},
                "calc_B": {"methods": {"encut": 520}, "results": {"energy": -102}},
            },
        )
        result = await self.tool.call(args, self.ctx)
        assert result.success
        assert "report" in result.data

    @pytest.mark.asyncio
    async def test_compare_from_workflow_results(self):
        args = ReportToolInput(
            action="compare",
            workflow_results={
                "comparison_datasets": {
                    "A": {"methods": {}, "results": {}},
                    "B": {"methods": {}, "results": {}},
                }
            },
        )
        result = await self.tool.call(args, self.ctx)
        assert result.success


# ═══════════════════════════════════════════════════════════════════
# Symbolic Regression Compare Tests
# ═══════════════════════════════════════════════════════════════════


class TestSymRegCompare:
    def setup_method(self):
        self.tool = SymbolicRegressionTool()
        self.ctx = _ctx()

    @pytest.mark.asyncio
    async def test_compare_no_expression(self):
        args = SymbolicRegressionInput(
            action="compare",
            data_json={"x": [1, 2, 3], "y": [1, 4, 9]},
            target_column="y",
        )
        result = await self.tool.call(args, self.ctx)
        assert not result.success
        assert "probe_expression" in result.error

    @pytest.mark.asyncio
    async def test_compare_single_expression(self):
        args = SymbolicRegressionInput(
            action="compare",
            data_json={"x": [1, 2, 3, 4, 5], "y": [1, 4, 9, 16, 25]},
            target_column="y",
            probe_expression="x**2",
        )
        result = await self.tool.call(args, self.ctx)
        assert not result.success
        assert "2 expressions" in result.error

    @pytest.mark.asyncio
    async def test_compare_two_expressions(self):
        args = SymbolicRegressionInput(
            action="compare",
            data_json={"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [1.0, 4.0, 9.0, 16.0, 25.0]},
            target_column="y",
            probe_expression="x**2; x**2 + x",
        )
        result = await self.tool.call(args, self.ctx)
        assert result.success
        assert result.data["expressions_evaluated"] == 2
        assert result.data["best_expression"] is not None
        # x**2 should have perfect R² for y = x²
        assert result.data["best_r2"] > 0.99

    @pytest.mark.asyncio
    async def test_compare_ranking(self):
        args = SymbolicRegressionInput(
            action="compare",
            data_json={"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [2.0, 4.0, 6.0, 8.0, 10.0]},
            target_column="y",
            probe_expression="2*x; x; x**2",
        )
        result = await self.tool.call(args, self.ctx)
        assert result.success
        # 2*x should be rank 1 (perfect for y=2x)
        comparison = result.data["comparison"]
        rank1 = [c for c in comparison if c.get("rank") == 1][0]
        assert rank1["expression"] == "2*x"

    @pytest.mark.asyncio
    async def test_compare_with_error(self):
        args = SymbolicRegressionInput(
            action="compare",
            data_json={"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]},
            target_column="y",
            probe_expression="x; forbidden_func(x)",
        )
        result = await self.tool.call(args, self.ctx)
        assert result.success
        comparison = result.data["comparison"]
        error_entries = [c for c in comparison if "error" in c]
        assert len(error_entries) == 1


# ═══════════════════════════════════════════════════════════════════
# Workflow Custom Validation Tests
# ═══════════════════════════════════════════════════════════════════


class TestWorkflowCustomValidation:
    def setup_method(self):
        self.engine = WorkflowEngine(tool_registry=None)

    def test_builtin_convergence_pass(self):
        rule = ValidationRule(check="convergence")
        assert self.engine._validate({"converged": True}, rule) is True

    def test_builtin_convergence_fail(self):
        rule = ValidationRule(check="convergence")
        assert self.engine._validate({"converged": False}, rule) is False

    def test_builtin_energy_sign(self):
        rule = ValidationRule(check="energy_sign")
        assert self.engine._validate({"energy": -100}, rule) is True
        assert self.engine._validate({"energy": 50}, rule) is False

    def test_builtin_force_threshold(self):
        rule = ValidationRule(check="force_threshold", threshold=0.05)
        assert self.engine._validate({"max_force": 0.01}, rule) is True
        assert self.engine._validate({"max_force": 0.1}, rule) is False

    def test_custom_registered_validator(self):
        def check_positive(data):
            return data.get("value", 0) > 0

        self.engine.register_validator("check_positive", check_positive)
        rule = ValidationRule(check="custom", custom_fn="check_positive")
        assert self.engine._validate({"value": 5}, rule) is True
        assert self.engine._validate({"value": -1}, rule) is False

    def test_custom_with_threshold(self):
        def check_range(data, threshold=1.0):
            return abs(data.get("error", 999)) < threshold

        self.engine.register_validator("check_range", check_range)
        rule = ValidationRule(check="custom", custom_fn="check_range", threshold=0.1)
        assert self.engine._validate({"error": 0.05}, rule) is True
        assert self.engine._validate({"error": 0.5}, rule) is False

    def test_custom_unknown_validator_fails_closed(self):
        rule = ValidationRule(check="custom", custom_fn="nonexistent_fn")
        assert self.engine._validate({"data": 1}, rule) is False

    def test_custom_exception_returns_false(self):
        def bad_fn(data):
            raise ValueError("boom")

        self.engine.register_validator("bad_fn", bad_fn)
        rule = ValidationRule(check="custom", custom_fn="bad_fn")
        assert self.engine._validate({}, rule) is False

    def test_resolve_dotted_path(self):
        # Test that _resolve_custom_fn handles non-existent modules gracefully
        result = WorkflowEngine._resolve_custom_fn("nonexistent.module.fn")
        assert result is None

    def test_resolve_simple_name(self):
        result = WorkflowEngine._resolve_custom_fn("simple_name")
        assert result is None


# ═══════════════════════════════════════════════════════════════════
# Exploration Query Tests
# ═══════════════════════════════════════════════════════════════════


def _make_space() -> ExplorationSpace:
    """Create a test exploration space with branches."""
    space = ExplorationSpace(
        id="test-space",
        name="Test Space",
        objective="Find best material",
    )
    space.objectives_config = {"energy": "minimize", "band_gap": "maximize"}

    b1 = Branch(id="b1", name="PBE-Si", hypothesis="PBE functional for Si")
    b1.status = BranchStatus.COMPLETED
    b1.objectives = {"energy": -100.0, "band_gap": 1.1}
    b1.decisions = [
        Decision(
            id="d1", description="Choose functional",
            decision_type="categorical",
            chosen_option="PBE", available_options=["PBE", "HSE06"],
            rationale="Fast and reliable",
        ),
    ]

    b2 = Branch(id="b2", name="HSE06-Si", hypothesis="HSE06 for Si")
    b2.status = BranchStatus.COMPLETED
    b2.objectives = {"energy": -102.0, "band_gap": 1.5}
    b2.decisions = [
        Decision(
            id="d2", description="Choose functional",
            decision_type="categorical",
            chosen_option="HSE06", available_options=["PBE", "HSE06"],
            rationale="More accurate band gap",
        ),
    ]

    b3 = Branch(id="b3", name="LDA-test", hypothesis="LDA test")
    b3.status = BranchStatus.PRUNED
    b3.prune_reason = "Energy too high, dominated by b1 and b2"

    space.add_branch(b1)
    space.add_branch(b2)
    space.add_branch(b3)
    space.mark_pruned("b3", "Energy too high")

    space.update_pareto_front()
    return space


class TestExplorationQuery:
    def setup_method(self):
        self.space = _make_space()

    def test_pareto_query(self):
        result = self.space.query("What are the best branches?")
        assert result["type"] == "pareto_front"
        assert result["count"] >= 1

    def test_pareto_explicit(self):
        result = self.space.query("show pareto front")
        assert result["type"] == "pareto_front"

    def test_pruned_query(self):
        result = self.space.query("Why were branches pruned?")
        assert result["type"] == "pruned"
        assert result["count"] == 1
        assert result["branches"][0]["id"] == "b3"

    def test_pruned_with_filter(self):
        result = self.space.query("Why were LDA branches rejected?")
        assert result["type"] == "pruned"
        # Should filter for "lda"
        if result["count"] > 0:
            assert "lda" in result["branches"][0]["name"].lower()

    def test_path_query(self):
        result = self.space.query("What is the decision path for b1?")
        assert result["type"] == "path"
        assert result["branch_id"] == "b1"
        assert len(result["decisions"]) == 1
        assert result["decisions"][0]["chosen"] == "PBE"

    def test_path_query_fallback(self):
        result = self.space.query("Show me the decision path")
        assert result["type"] == "path"
        # Should fall back to first completed branch
        assert "branch_id" in result

    def test_status_query(self):
        result = self.space.query("What is the exploration status?")
        assert result["type"] == "status"
        assert result["total_branches"] == 3
        assert result["pruned"] == 1

    def test_list_all_query(self):
        result = self.space.query("List all branches")
        assert result["type"] == "all_branches"
        assert result["count"] == 3

    def test_unrecognized_query(self):
        result = self.space.query("xyzzy foobar baz")
        assert result["type"] == "unrecognized"
        assert "hint" in result
        assert "available_query_types" in result

    def test_optimal_keyword(self):
        result = self.space.query("optimal solutions")
        assert result["type"] == "pareto_front"

    def test_discard_keyword(self):
        result = self.space.query("discarded options")
        assert result["type"] == "pruned"
