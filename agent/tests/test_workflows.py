"""Unit tests for workflow engine."""

import asyncio
import pytest
from matsci_agent.workflows.engine import WorkflowEngine, ComputationalStage, ValidationRule
from matsci_agent.workflows.templates import symbolic_verify_workflow, get_template, list_templates
from matsci_agent.types import ToolContext


class TestWorkflowEngine:
    def test_empty_workflow(self):
        engine = WorkflowEngine(None)
        result = asyncio.run(engine.execute([], ToolContext(session_id="test", workspace=".")))
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
