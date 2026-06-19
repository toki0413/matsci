"""Tests for remaining uncovered paths in huginn/workflows/engine.py."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from huginn.tools.registry import ToolRegistry
from huginn.types import BudgetDecision, BudgetPolicy, ToolContext, ToolResult
from huginn.workflows.checkpoint import WorkflowCheckpoint
from huginn.workflows.engine import WorkflowEngine
from huginn.workflows.stages import ComputationalStage, RetryPolicy, ValidationRule


class _DummyTool:
    name = "dummy"

    def __init__(self):
        self.calls = 0
        self.input_schema = None

    async def call(self, tool_input, context):
        self.calls += 1
        return ToolResult(data={"value": tool_input.get("x", 0)}, success=True)


class _FailThenSucceedTool:
    name = "fail_then"

    def __init__(self, fails=1):
        self.calls = 0
        self.fails = fails
        self.input_schema = None

    async def call(self, tool_input, context):
        self.calls += 1
        if self.calls <= self.fails:
            return ToolResult(data=None, success=False, error="timeout")
        return ToolResult(data={"ok": True}, success=True)


class _FailingTool:
    name = "always_fail"

    def __init__(self, error="boom"):
        self.calls = 0
        self.error = error
        self.input_schema = None

    async def call(self, tool_input, context):
        self.calls += 1
        return ToolResult(data=None, success=False, error=self.error)


@pytest.fixture(autouse=True)
def _clear_registry():
    ToolRegistry.clear()
    yield
    ToolRegistry.clear()


class TestRetryPolicy:
    async def _run(self, engine, stages, context):
        return await engine.execute(stages, context)

    def test_stage_retry_then_success(self, tmp_path, monkeypatch):
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())
        ToolRegistry.register(_FailThenSucceedTool(fails=1))

        engine = WorkflowEngine(ToolRegistry)
        stage = ComputationalStage(
            id="s1",
            name="s1",
            tool="fail_then",
            tool_input={},
            retry_policy=RetryPolicy(
                max_retries=2,
                backoff_factor=1.0,
                retry_on=["any"],
                auto_diagnose=False,
                apply_auto_fix=False,
            ),
        )
        context = ToolContext(session_id="test", workspace=str(tmp_path))
        result = asyncio.run(self._run(engine, [stage], context))

        assert result.success is True
        assert result.stages["s1"].status == "completed"
        registered = ToolRegistry.get("fail_then")
        assert registered.calls == 2

    def test_stage_retry_exhausted(self, tmp_path, monkeypatch):
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())
        ToolRegistry.register(_FailingTool(error="timeout"))

        engine = WorkflowEngine(ToolRegistry)
        stage = ComputationalStage(
            id="s1",
            name="s1",
            tool="always_fail",
            tool_input={},
            retry_policy=RetryPolicy(
                max_retries=1,
                backoff_factor=1.0,
                retry_on=["timeout"],
                auto_diagnose=False,
                apply_auto_fix=False,
            ),
        )
        context = ToolContext(session_id="test", workspace=str(tmp_path))
        result = asyncio.run(self._run(engine, [stage], context))

        assert result.success is False
        assert result.stages["s1"].status == "failed"


class TestShouldRetry:
    def _stage(self, attempts=0, retry_on=None):
        return ComputationalStage(
            id="s1",
            name="s1",
            tool="dummy",
            tool_input={},
            retry_policy=RetryPolicy(
                max_retries=2, retry_on=retry_on or ["any"]
            ),
            attempts=attempts,
        )

    def test_max_retries_reached(self):
        engine = WorkflowEngine(None)
        assert engine._should_retry(self._stage(attempts=2), "error") is False

    def test_retry_any(self):
        engine = WorkflowEngine(None)
        assert engine._should_retry(self._stage(retry_on=["any"]), "anything") is True

    def test_retry_timeout(self):
        engine = WorkflowEngine(None)
        stage = self._stage(retry_on=["timeout"])
        assert engine._should_retry(stage, "operation timed out") is True
        assert engine._should_retry(stage, "") is False

    def test_retry_oom(self):
        engine = WorkflowEngine(None)
        stage = self._stage(retry_on=["oom"])
        assert engine._should_retry(stage, "out of memory") is True
        assert engine._should_retry(stage, "") is False

    def test_retry_remote_failure(self):
        engine = WorkflowEngine(None)
        stage = self._stage(retry_on=["remote_failure"])
        assert engine._should_retry(stage, "ssh connection refused") is True
        assert engine._should_retry(stage, "") is False

    def test_default_retry_on_non_empty_error(self):
        engine = WorkflowEngine(None)
        stage = self._stage(retry_on=["convergence_fail"])
        assert engine._should_retry(stage, "boom") is True
        assert engine._should_retry(stage, "") is False

    def test_default_non_empty_error(self):
        engine = WorkflowEngine(None)
        stage = self._stage(retry_on=["convergence_fail"])
        assert engine._should_retry(stage, "some failure") is True
        assert engine._should_retry(stage, "") is False


class TestRemoteFailure:
    def test_remote_markers(self):
        assert WorkflowEngine._is_remote_failure("slurm job killed") is True
        assert WorkflowEngine._is_remote_failure("node failure") is True
        assert WorkflowEngine._is_remote_failure("compute error") is False
        assert WorkflowEngine._is_remote_failure("") is False
        assert WorkflowEngine._is_remote_failure(None) is False


class TestEstimateStageCost:
    def _estimate(self, tool, tool_input):
        stage = ComputationalStage(
            id="s1", name="s1", tool=tool, tool_input=tool_input
        )
        return WorkflowEngine(None)._estimate_stage_cost(stage)

    def test_vasp_default(self):
        est = self._estimate("vasp_tool", {})
        assert est.cpu_hours > 0

    def test_vasp_kpoints(self):
        est = self._estimate(
            "vasp_tool",
            {
                "encut": 1040,
                "kpoints": "1 1 2 3 4",
                "nodes": 2,
                "ntasks_per_node": 8,
            },
        )
        assert est.cpu_hours > 0

    def test_lammps(self):
        est = self._estimate("lammps_tool", {"n_steps": 5000})
        assert est.cpu_hours > 0

    def test_md(self):
        est = self._estimate("aimd_tool", {"n_steps": 2000})
        assert est.cpu_hours > 0

    def test_gpu_hours(self):
        est = self._estimate("vasp_gpu_tool", {"queue": "gpu"})
        assert est.gpu_hours > 0


class TestResolveInputs:
    def test_stage_reference_with_key(self):
        engine = WorkflowEngine(None)
        resolved = engine._resolve_inputs(
            {"x": "${s1.y}"}, {"s1": {"y": 42}}
        )
        assert resolved["x"] == 42

    def test_stage_reference_whole_output(self):
        engine = WorkflowEngine(None)
        resolved = engine._resolve_inputs(
            {"x": "${s1}"}, {"s1": {"y": 42}}
        )
        assert resolved["x"] == {"y": 42}

    def test_unresolved_reference_preserved(self):
        engine = WorkflowEngine(None)
        resolved = engine._resolve_inputs({"x": "${missing.y}"}, {})
        assert resolved["x"] == "${missing.y}"


class TestDiagnoseAndFix:
    class _MockDiagnoseTool:
        name = "diagnose_tool"
        input_schema = None

        async def call(self, args, context):
            return ToolResult(
                data={
                    "findings": [
                        {"text": "SCF convergence issue"},
                        {"text": "Memory OOM"},
                    ],
                    "general_advice": ["check input"],
                    "recommended_next_steps": ["increase NELM"],
                },
                success=True,
            )

    class _MockRAGTool:
        name = "rag_tool"
        input_schema = None

        async def call(self, args, context):
            return ToolResult(
                data={"results": [{"text": "hint one"}, {"text": "hint two"}]},
                success=True,
            )

    def test_diagnose_and_fix_vasp(self, tmp_path):
        ToolRegistry.register(TestDiagnoseAndFix._MockDiagnoseTool())
        ToolRegistry.register(TestDiagnoseAndFix._MockRAGTool())

        engine = WorkflowEngine(ToolRegistry)
        stage = ComputationalStage(
            id="s1",
            name="s1",
            tool="vasp_tool",
            tool_input={"scf": "dft"},
            retry_policy=RetryPolicy(
                auto_diagnose=True, apply_auto_fix=True
            ),
        )
        context = ToolContext(session_id="test", workspace=str(tmp_path))
        result = ToolResult(data=None, success=False, error="EDDDAV")

        asyncio.run(engine._diagnose_and_fix(stage, result, context))

        assert "__diagnosis" in stage.tool_input
        overrides = stage.tool_input.get("incar_overrides", {})
        assert overrides.get("ALGO") == "Normal"
        assert overrides.get("NCORE") == 4
        assert "__sobko_hints" in stage.tool_input


class TestDetection:
    def _software(self, tool, tool_input=None):
        stage = ComputationalStage(
            id="s1", name="s1", tool=tool, tool_input=tool_input or {}
        )
        return WorkflowEngine(None)._detect_software_from_stage(stage)

    def test_detect_software_by_tool_name(self):
        assert self._software("vasp_tool") == "VASP"
        assert self._software("lammps_runner") == "LAMMPS"
        assert self._software("g16_job") == "Gaussian"
        assert self._software("pw.x") == "QuantumESPRESSO"

    def test_detect_software_from_input(self):
        assert (
            self._software("generic_tool", {"software": "vasp"}) == "Vasp"
        )

    def test_detect_calculation_type(self):
        def _calc(tool_input):
            stage = ComputationalStage(
                id="s1", name="s1", tool="t", tool_input=tool_input
            )
            return WorkflowEngine(None)._detect_calculation_type_from_stage(
                stage
            )

        assert _calc({"pbe": True}) == "DFT"
        assert _calc({"nvt": True}) == "MD"
        assert _calc({"excited": True}) == "TDDFT"
        assert _calc({"opt": True}) == "geometry_optimization"
        assert _calc({"band": True}) == "band_structure"


class TestExtractAndApplyFixes:
    def test_extract_vasp_fixes(self):
        engine = WorkflowEngine(None)
        report = {"findings": [{"text": "SCF convergence"}, {"text": "memory OOM"}]}
        fixes = engine._extract_fixes_from_diagnosis(report, "VASP")
        assert fixes["ALGO"] == "Normal"
        assert fixes["NCORE"] == 4

    def test_extract_gaussian_fixes(self):
        engine = WorkflowEngine(None)
        report = {"findings": [{"text": "SCF convergence"}]}
        fixes = engine._extract_fixes_from_diagnosis(report, "Gaussian")
        assert "scf" in fixes

    def test_extract_lammps_fixes(self):
        engine = WorkflowEngine(None)
        report = {"findings": [{"text": "timestep too large"}]}
        fixes = engine._extract_fixes_from_diagnosis(report, "LAMMPS")
        assert fixes["timestep"] == "__reduce_half"

    def test_apply_fixes_vasp(self):
        engine = WorkflowEngine(None)
        stage = ComputationalStage(
            id="s1", name="s1", tool="vasp_tool", tool_input={}
        )
        engine._apply_fixes_to_stage(stage, {"ALGO": "Normal"}, "VASP")
        assert stage.tool_input["incar_overrides"]["ALGO"] == "Normal"

    def test_apply_fixes_gaussian(self):
        engine = WorkflowEngine(None)
        stage = ComputationalStage(
            id="s1", name="s1", tool="gaussian_tool", tool_input={}
        )
        engine._apply_fixes_to_stage(stage, {"scf": "xqc"}, "Gaussian")
        assert stage.tool_input["params"]["scf"] == "xqc"

    def test_apply_fixes_lammps(self):
        engine = WorkflowEngine(None)
        stage = ComputationalStage(
            id="s1", name="s1", tool="lammps_tool", tool_input={}
        )
        engine._apply_fixes_to_stage(stage, {"timestep": "__reduce_half"}, "LAMMPS")
        assert stage.tool_input["fixes"]["timestep"] == "__reduce_half"

    def test_apply_fixes_generic(self):
        engine = WorkflowEngine(None)
        stage = ComputationalStage(
            id="s1", name="s1", tool="other_tool", tool_input={}
        )
        engine._apply_fixes_to_stage(stage, {"x": 1}, None)
        assert stage.tool_input["__auto_fixes"]["x"] == 1


class TestValidate:
    def _validate(self, data, rule):
        return WorkflowEngine(None)._validate(data, rule)

    def test_convergence(self):
        assert self._validate({"converged": True}, ValidationRule("convergence")) is True
        assert self._validate({"converged": False}, ValidationRule("convergence")) is False

    def test_energy_sign(self):
        assert self._validate({"energy": -1.0}, ValidationRule("energy_sign")) is True
        assert self._validate({"energy": 1.0}, ValidationRule("energy_sign")) is False
        assert self._validate({"x": 1}, ValidationRule("energy_sign")) is True

    def test_force_threshold(self):
        assert (
            self._validate({"max_force": 0.005}, ValidationRule("force_threshold"))
            is True
        )
        assert (
            self._validate({"max_force": 0.1}, ValidationRule("force_threshold"))
            is False
        )
        assert self._validate({}, ValidationRule("force_threshold")) is True

    def test_custom(self):
        assert self._validate({}, ValidationRule("custom")) is True


class TestResume:
    def test_resume_skips_completed(self, tmp_path):
        ToolRegistry.register(_DummyTool())
        engine = WorkflowEngine(ToolRegistry)
        context = ToolContext(session_id="test", workspace=str(tmp_path))

        stage = ComputationalStage(
            id="s1",
            name="s1",
            tool="dummy",
            tool_input={"x": 1},
            status="completed",
            result=ToolResult(data={"value": 1}),
        )
        cp_path = tmp_path / "cp.json"
        WorkflowCheckpoint(stages=[stage], outputs={"s1": {"value": 1}}).save(
            cp_path
        )

        result = asyncio.run(
            engine.resume(
                [ComputationalStage(id="s1", name="s1", tool="dummy", tool_input={"x": 1})],
                context,
                cp_path,
            )
        )
        assert result.success is True
        assert ToolRegistry.get("dummy").calls == 0


class TestBudgetWarn:
    def test_budget_warn_proceeds_with_warning(self, tmp_path, monkeypatch):
        ToolRegistry.register(_DummyTool())
        engine = WorkflowEngine(ToolRegistry)

        policy = BudgetPolicy(max_cpu_hours=1e6)
        monkeypatch.setattr(
            policy,
            "check",
            lambda _estimate: (BudgetDecision.WARN, "CPU usage high"),
        )

        stage = ComputationalStage(
            id="s1", name="s1", tool="dummy", tool_input={"x": 1}
        )
        context = ToolContext(session_id="test", workspace=str(tmp_path))
        result = asyncio.run(engine.execute([stage], context, budget_policy=policy))

        assert result.success is True
        assert "__budget_warnings" in stage.tool_input
        assert "CPU usage high" in stage.tool_input["__budget_warnings"]
