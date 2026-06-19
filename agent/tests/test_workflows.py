"""Unit tests for workflow engine."""

import asyncio

from huginn.types import ToolContext
from huginn.workflows.engine import ComputationalStage, ValidationRule, WorkflowEngine
from huginn.workflows.templates import (
    get_template,
    list_templates,
    symbolic_verify_workflow,
)


class TestWorkflowEngine:
    def test_empty_workflow(self):
        engine = WorkflowEngine(None)
        result = asyncio.run(
            engine.execute([], ToolContext(session_id="test", workspace="."))
        )
        assert result.success
        assert result.total_walltime >= 0

    def test_validation_rule_dataclass(self):
        rule = ValidationRule(check="convergence")
        assert rule.check == "convergence"
        assert rule.threshold is None


class TestSymbolicVerifyWorkflow:
    def test_template_registered(self):
        assert "symbolic_verify" in list_templates()
        tmpl = get_template("symbolic_verify")
        assert tmpl is not None

    def test_generates_two_stages(self):
        stages = symbolic_verify_workflow(
            verify_type="derivative",
            expression="x**2",
            symbols=["x"],
            variable="x",
        )
        assert len(stages) == 2
        assert stages[0].id == "symbolic_derive"
        assert stages[1].id == "lean_verify"

    def test_dependency_set(self):
        stages = symbolic_verify_workflow(verify_type="derivative")
        assert stages[1].dependencies == ["symbolic_derive"]

    def test_input_reference_resolution(self):
        stages = symbolic_verify_workflow(verify_type="constitutive", free_energy="psi")
        assert stages[0].tool_input["action"] == "constitutive"
        assert stages[0].tool_input["free_energy"] == "psi"
        assert stages[1].tool_input["symbolic_result"] == "${symbolic_derive}"
        assert stages[1].tool_input["auto_verify_action"] == "constitutive"

    def test_derivative_action_mapped(self):
        stages = symbolic_verify_workflow(verify_type="derivative")
        # symbolic_math_tool uses "differentiate", lean_tool uses "derivative"
        assert stages[0].tool_input["action"] == "differentiate"
        assert stages[1].tool_input["auto_verify_action"] == "derivative"

    def test_validation_rules_present(self):
        stages = symbolic_verify_workflow()
        assert stages[0].validation is not None
        assert stages[0].validation.check == "custom"
        assert stages[1].validation is not None
        assert stages[1].validation.custom_fn == "lean_compiles"


class _DummyTool:
    """Simple tool for workflow engine tests."""

    name = "dummy"

    def __init__(self):
        self.calls = 0
        self.input_schema = None

    async def call(self, tool_input, context):
        self.calls += 1
        from huginn.types import ToolResult

        return ToolResult(data={"value": tool_input.get("x", 0)}, success=True)


class TestWorkflowCheckpoint:
    def test_checkpoint_save_and_load(self, tmp_path):
        from huginn.types import ToolResult
        from huginn.workflows.checkpoint import WorkflowCheckpoint

        stage = ComputationalStage(
            id="s1",
            name="stage one",
            tool="dummy",
            tool_input={"x": 1},
            status="completed",
            result=ToolResult(data={"value": 1}),
            attempts=2,
        )
        cp = WorkflowCheckpoint(stages=[stage], outputs={"s1": {"value": 1}})
        path = tmp_path / "cp.json"
        cp.save(path)

        loaded = WorkflowCheckpoint.load(path)
        assert len(loaded.stages) == 1
        assert loaded.stages[0].id == "s1"
        assert loaded.stages[0].status == "completed"
        assert loaded.stages[0].result.success is True
        assert loaded.outputs == {"s1": {"value": 1}}

    def test_execute_writes_checkpoint(self, tmp_path):
        from huginn.tools.registry import ToolRegistry

        ToolRegistry.clear()
        dummy = _DummyTool()
        ToolRegistry.register(dummy)

        engine = WorkflowEngine(ToolRegistry)
        stages = [
            ComputationalStage(id="s1", name="s1", tool="dummy", tool_input={"x": 10})
        ]
        context = ToolContext(session_id="test", workspace=str(tmp_path))
        cp_path = tmp_path / "checkpoint.json"

        result = asyncio.run(engine.execute(stages, context, checkpoint_path=cp_path))

        assert result.success is True
        assert cp_path.exists()
        from huginn.workflows.checkpoint import WorkflowCheckpoint

        loaded = WorkflowCheckpoint.load(cp_path)
        assert loaded.stages[0].status == "completed"

        ToolRegistry.clear()

    def test_resume_skips_completed_stage(self, tmp_path):
        from huginn.tools.registry import ToolRegistry
        from huginn.types import ToolResult
        from huginn.workflows.checkpoint import WorkflowCheckpoint

        ToolRegistry.clear()
        dummy = _DummyTool()
        ToolRegistry.register(dummy)

        # Pre-build a checkpoint with s1 already done.
        stage1 = ComputationalStage(
            id="s1",
            name="s1",
            tool="dummy",
            tool_input={"x": 1},
            status="completed",
            result=ToolResult(data={"value": 1}),
        )
        cp = WorkflowCheckpoint(stages=[stage1], outputs={"s1": {"value": 1}})
        cp_path = tmp_path / "resume.json"
        cp.save(cp_path)

        engine = WorkflowEngine(ToolRegistry)
        # s1 template + a new pending stage would still be re-run in current
        # implementation because we overlay checkpoint state and then execute
        # the whole graph; but completed stages are skipped by the ready filter.
        stages = [
            ComputationalStage(id="s1", name="s1", tool="dummy", tool_input={"x": 1}),
        ]
        context = ToolContext(session_id="test", workspace=str(tmp_path))

        result = asyncio.run(engine.resume(stages, context, cp_path))

        assert result.success is True
        assert dummy.calls == 0  # s1 was already completed in checkpoint

        ToolRegistry.clear()


class TestWorkflowBudget:
    def test_budget_denies_expensive_stage(self, tmp_path):
        from huginn.tools.registry import ToolRegistry
        from huginn.types import BudgetPolicy

        ToolRegistry.clear()
        dummy = _DummyTool()
        ToolRegistry.register(dummy)

        policy = BudgetPolicy(max_cpu_hours=0.5)
        engine = WorkflowEngine(ToolRegistry, budget_policy=policy)
        stages = [
            ComputationalStage(
                id="big",
                name="big",
                tool="dummy",
                tool_input={"walltime_hours": 24, "nodes": 4, "ntasks_per_node": 8},
            )
        ]
        context = ToolContext(session_id="test", workspace=str(tmp_path))
        result = asyncio.run(engine.execute(stages, context))

        assert result.success is False
        assert result.stages["big"].status == "failed"
        assert "Budget denied" in (result.stages["big"].result.error or "")
        assert dummy.calls == 0

        ToolRegistry.clear()

    def test_budget_allows_cheap_stage(self, tmp_path):
        from huginn.tools.registry import ToolRegistry
        from huginn.types import BudgetPolicy

        ToolRegistry.clear()
        dummy = _DummyTool()
        ToolRegistry.register(dummy)

        policy = BudgetPolicy(max_cpu_hours=1000.0)
        engine = WorkflowEngine(ToolRegistry, budget_policy=policy)
        stages = [
            ComputationalStage(
                id="small", name="small", tool="dummy", tool_input={"x": 1}
            )
        ]
        context = ToolContext(session_id="test", workspace=str(tmp_path))
        result = asyncio.run(engine.execute(stages, context))

        assert result.success is True
        assert dummy.calls == 1

        ToolRegistry.clear()
