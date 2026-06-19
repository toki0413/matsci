"""Coverage boost tests for pure-Python modules with no heavy dependencies.

These tests exercise modules that are currently 0% covered but require no
external executables (VASP, QE, LAMMPS, Lean, etc.) or heavy ML packages.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from huginn.validation.physics import PhysicsValidator, ValidationCheck
from huginn.utils.context import compact_messages, estimate_message_tokens
from huginn.utils.tokens import (
    bytes_per_token_for_extension,
    rough_token_count,
    rough_token_count_for_text,
)
from huginn.diagnostics.convergence import (
    DiagnosisReport,
    ConvergenceDiagnostician,
    VASP_FAILURE_PATTERNS,
)
from huginn.benchmark.core import (
    BenchmarkCase,
    BenchmarkSuite,
    keyword_evaluator,
    numeric_evaluator,
    SelfImprovementLoop,
)
from huginn.constraints.boundaries import BoundaryEvolution, BoundaryState
from huginn.constraints.operators import SafetyOperator, QualityOperator
from huginn.constraints.reference import Constraint, ConstraintResult
from huginn.workflows.stages import ComputationalStage, ValidationRule, RetryPolicy, WorkflowResult
from huginn.workflows.checkpoint import WorkflowCheckpoint


class TestPhysicsValidator:
    def test_validate_dft_result_all_checks(self):
        v = PhysicsValidator()
        result = {
            "energy": -123.45,
            "forces": [[0.01, 0.02, 0.03]],
            "band_gap": 1.2,
            "volume": 64.0,
            "magnetic_moments": {"Fe": 2.1},
        }
        checks = v.validate_dft_result(result)
        assert all(isinstance(c, ValidationCheck) for c in checks)
        energy_check = next(c for c in checks if c.name == "energy_sign")
        assert energy_check.passed is True

    def test_validate_dft_positive_energy(self):
        v = PhysicsValidator()
        checks = v.validate_dft_result({"energy": 5.0})
        energy_check = next(c for c in checks if c.name == "energy_sign")
        assert energy_check.passed is False

    def test_validate_dft_missing_energy(self):
        v = PhysicsValidator()
        checks = v.validate_dft_result({})
        energy_check = next(c for c in checks if c.name == "energy_sign")
        assert energy_check.passed is True  # not available = skip

    def test_validate_md_result(self):
        v = PhysicsValidator()
        result = {
            "total_energy": [-1.0, -1.01, -1.005],
            "temperature": [300, 301, 299],
            "atom_count": [10, 10, 10],
            "density": 2.5,
        }
        checks = v.validate_md_result(result)
        assert any(c.name == "energy_conservation" for c in checks)

    def test_validate_phonon(self):
        v = PhysicsValidator()
        checks = v.validate_phonon_result({"frequencies": [100, 200, 300]})
        assert checks[0].name == "imaginary_modes"

    def test_magnetic_moments_reference(self):
        assert "Fe" in PhysicsValidator.REFERENCE_MAGNETIC_MOMENTS
        assert PhysicsValidator.REFERENCE_MAGNETIC_MOMENTS["Fe"] == 2.2



class TestUtilsTokens:
    def test_rough_token_count_empty(self):
        assert rough_token_count("") == 0

    def test_rough_token_count_basic(self):
        assert rough_token_count("hello world") == 3  # 11 chars / 4 = 2.75 -> 3

    def test_bytes_per_token_json(self):
        assert bytes_per_token_for_extension("json") == 2.0
        assert bytes_per_token_for_extension("txt") == 4.0
        assert bytes_per_token_for_extension(None) == 4.0

    def test_rough_token_count_for_text(self):
        assert rough_token_count_for_text("abcd", "json") == 2
        assert rough_token_count_for_text("abcd") == 1


class TestUtilsContext:
    def test_estimate_message_tokens_empty(self):
        assert estimate_message_tokens([]) == 0

    def test_estimate_message_tokens_dict(self):
        msgs = [{"content": "hello"}, {"content": "world"}]
        assert estimate_message_tokens(msgs) > 0

    def test_estimate_message_tokens_obj(self):
        class Msg:
            content = "test"
        assert estimate_message_tokens([Msg()]) > 0

    def test_compact_messages_no_trim(self):
        msgs = [{"content": "a"}, {"content": "b"}]
        result = compact_messages(msgs, budget_tokens=1000)
        assert len(result) == 2

    def test_compact_messages_trims(self):
        msgs = [{"content": "x" * 4000}, {"content": "y" * 4000}, {"content": "keep"}]
        result = compact_messages(msgs, budget_tokens=10, keep_last_n=1)
        assert len(result) == 1
        assert result[0]["content"] == "keep"

    def test_compact_messages_zero_budget(self):
        msgs = [{"content": "a"}]
        assert compact_messages(msgs, budget_tokens=0) == msgs


class TestDiagnosticsConvergence:
    def test_known_patterns_exist(self):
        assert "EDDDAV" in VASP_FAILURE_PATTERNS
        assert "ZPOTRF" in VASP_FAILURE_PATTERNS
        assert "ZBRENT" in VASP_FAILURE_PATTERNS
        assert "TOO FEW BANDS" in VASP_FAILURE_PATTERNS

    def test_diagnose_convergence_known(self):
        d = ConvergenceDiagnostician()
        report = d.diagnose("vasp", "EDDDAV error occurred")
        assert isinstance(report, DiagnosisReport)
        assert report is not None
        assert report.auto_fixable is True

    def test_diagnose_convergence_unknown(self):
        d = ConvergenceDiagnostician()
        report = d.diagnose("vasp", "UNKNOWN_ERROR")
        assert report is None

    def test_diagnose_convergence_wrong_software(self):
        d = ConvergenceDiagnostician()
        report = d.diagnose("qe", "EDDDAV")
        assert report is None

    def test_diagnose_file_no_file(self, tmp_path: Path):
        d = ConvergenceDiagnostician()
        report = d.diagnose_from_file("vasp", tmp_path / "nonexistent")
        assert report is None

    def test_diagnose_file_with_content(self, tmp_path: Path):
        d = ConvergenceDiagnostician()
        log = tmp_path / "vasp.log"
        log.write_text("some EDDDAV error here")
        report = d.diagnose_from_file("vasp", log)
        assert report is not None
        assert "EDDDAV" in report.problem

    def test_suggest_auto_fix(self):
        d = ConvergenceDiagnostician()
        report = d.diagnose("vasp", "EDDDAV")
        fixes = d.suggest_auto_fix(report)
        assert fixes is not None
        assert "ALGO" in fixes


class TestBenchmarkCore:
    def test_keyword_evaluator_pass(self):
        case = BenchmarkCase(
            task="test",
            expected_keywords=["hello", "world"],
        )
        success, score = keyword_evaluator("Hello beautiful world!", case)
        assert success is True
        assert score == 1.0

    def test_keyword_evaluator_partial(self):
        case = BenchmarkCase(
            task="test",
            expected_keywords=["hello", "world"],
        )
        success, score = keyword_evaluator("hello", case)
        assert success is False
        assert score == 0.5

    def test_keyword_evaluator_no_keywords(self):
        case = BenchmarkCase(task="test")
        success, score = keyword_evaluator("anything", case)
        assert success is True
        assert score == 1.0

    def test_numeric_evaluator_no_expected(self):
        case = BenchmarkCase(task="test")
        success, score = numeric_evaluator("answer is 42", case)
        assert success is True  # falls back to keyword

    def test_numeric_evaluator_hit(self):
        case = BenchmarkCase(task="test", expected_value=42.0)
        success, score = numeric_evaluator("the result is 42.0", case)
        assert success is True
        assert score > 0.9

    def test_numeric_evaluator_miss(self):
        case = BenchmarkCase(task="test", expected_value=100.0)
        success, score = numeric_evaluator("the result is 42.0", case)
        assert success is False

    def test_benchmark_suite_add(self):
        suite = BenchmarkSuite()
        case = BenchmarkCase(task="t", expected_keywords=["k"])
        suite.add(case)
        assert len(suite.cases) == 1

    def test_benchmark_suite_add_defaults(self):
        suite = BenchmarkSuite()
        suite.add_defaults()
        assert len(suite.cases) >= 3

    def test_benchmark_suite_summary(self):
        suite = BenchmarkSuite()
        suite.add(BenchmarkCase(task="t1", expected_keywords=["yes"]))
        # summary() takes a list of BenchmarkResult
        from huginn.benchmark.core import BenchmarkResult
        results = [
            BenchmarkResult(case_id="c1", task="t1", success=True, score=1.0, response="yes", duration_ms=100.0),
        ]
        summary = suite.summary(results)
        assert summary["total"] == 1
        assert summary["passed"] == 1


class TestConstraints:
    def test_constraint_result_defaults(self):
        r = ConstraintResult(name="x", passed=True, value=1, expected="1", tolerance=0, message="ok")
        assert r.severity == "warn"
        assert r.family == "quality"

    def test_constraint_evaluate(self):
        def check(data):
            return ConstraintResult(name="x", passed=data["x"] > 0, value=data["x"], expected=">0", tolerance=0, message="")

        c = Constraint(name="x", scope="test", family="quality", severity="block", check=check)
        r = c.evaluate({"x": 5})
        assert r.passed is True

    def test_safety_operator_scope_filter(self):
        op = SafetyOperator()
        op.add(Constraint(name="a", scope="dft", family="safety", severity="block", check=lambda d: ConstraintResult(name="a", passed=True, value=0, expected="0", tolerance=0, message="")))
        op.add(Constraint(name="b", scope="md", family="safety", severity="block", check=lambda d: ConstraintResult(name="b", passed=True, value=0, expected="0", tolerance=0, message="")))
        results = op.evaluate({}, scope="dft")
        assert len(results) == 1
        assert results[0].name == "a"

    def test_boundary_state_allows_empty(self):
        state = BoundaryState()
        assert state.allows("anything") is True

    def test_boundary_state_allows_restricted(self):
        state = BoundaryState(allowed_executables={"python"})
        assert state.allows("python") is True
        assert state.allows("bash") is False

    def test_boundary_evolution_block(self):
        state = BoundaryState()
        evo = BoundaryEvolution(state)
        result = ConstraintResult(name="x", passed=False, value=0, expected="0", tolerance=0, message="", severity="block")
        new_state = evo.update([result])
        assert new_state.require_confirmation is True
        assert new_state.max_retries == 1

    def test_boundary_evolution_pass(self):
        state = BoundaryState()
        evo = BoundaryEvolution(state)
        result = ConstraintResult(name="x", passed=True, value=0, expected="0", tolerance=0, message="", severity="warn")
        new_state = evo.update([result])
        assert new_state.require_confirmation is True  # initial state


class TestWorkflowStages:
    def test_stage_creation(self):
        stage = ComputationalStage(
            id="s1", name="setup", tool="test", tool_input={}
        )
        assert stage.name == "setup"
        assert stage.status == "pending"

    def test_validation_rule_creation(self):
        rule = ValidationRule(check="convergence", threshold=0.01)
        assert rule.check == "convergence"
        assert rule.threshold == 0.01

    def test_retry_policy_defaults(self):
        policy = RetryPolicy()
        assert policy.max_retries == 2
        assert policy.auto_diagnose is True

    def test_workflow_result_creation(self):
        result = WorkflowResult(success=True, stages={}, outputs={})
        assert result.success is True
        assert result.error is None


class TestWorkflowCheckpoint:
    def test_checkpoint_creation(self):
        cp = WorkflowCheckpoint(stages=[], outputs={"x": 1})
        assert cp.outputs == {"x": 1}

    def test_checkpoint_save_load(self, tmp_path: Path):
        cp = WorkflowCheckpoint(stages=[], outputs={"x": 1})
        path = tmp_path / "cp.json"
        cp.save(path)
        loaded = WorkflowCheckpoint.load(path)
        assert loaded.outputs == {"x": 1}

    def test_checkpoint_default_path(self, tmp_path: Path):
        path = WorkflowCheckpoint.default_path(tmp_path, "run_123")
        assert path.name == "run_123.json"
        assert ".huginn" in str(path)
