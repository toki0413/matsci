"""Tests for the k-point / ENCUT convergence test tool."""

import pytest
from pydantic import ValidationError

from huginn.tools.sim.convergence_test_tool import (
    ConvergenceTestTool,
    ConvergenceTestToolInput,
)
from huginn.types import ToolContext

CTX = ToolContext(session_id="test", workspace=".")


def test_tool_metadata():
    """The tool should declare the right name, category, and profile."""
    tool = ConvergenceTestTool()
    assert tool.name == "convergence_test_tool"
    assert tool.category == "sim"
    assert tool.profile is not None
    assert tool.profile.cost_tier == "heavy"
    assert "dft" in (tool.profile.constraint_scope or "")


def test_kpoint_convergence_input_schema():
    """kpoint_convergence should accept structure, base_incar, kpoint_series."""
    inp = ConvergenceTestToolInput(
        action="kpoint_convergence",
        structure="/fake/POSCAR",
        base_incar={"ENCUT": 520, "ISMEAR": 0},
        kpoint_series=[4, 6, 8, 12, 16],
        tolerance=0.001,
        n_atoms=2,
    )
    assert inp.kpoint_series == [4, 6, 8, 12, 16]
    assert inp.tolerance == 0.001
    assert inp.n_atoms == 2


def test_kpoint_convergence_requires_kpoint_series():
    """kpoint_convergence without kpoint_series should fail validation."""
    with pytest.raises(ValidationError, match="kpoint_series"):
        ConvergenceTestToolInput(
            action="kpoint_convergence",
            structure="/fake/POSCAR",
        )


def test_encut_convergence_input_schema():
    """encut_convergence should accept structure, base_incar, encut_series."""
    inp = ConvergenceTestToolInput(
        action="encut_convergence",
        structure="/fake/POSCAR",
        base_incar={"ISMEAR": 0},
        encut_series=[300, 400, 500, 600],
        tolerance=0.001,
    )
    assert inp.encut_series == [300, 400, 500, 600]


def test_cutoff_analysis_with_mock_data():
    """cutoff_analysis should detect convergence in a clean data set."""
    data = [
        {"parameter": "encut", "value": 300, "energy": -100.0},
        {"parameter": "encut", "value": 400, "energy": -100.5},
        {"parameter": "encut", "value": 500, "energy": -100.505},
        {"parameter": "encut", "value": 600, "energy": -100.5051},
        {"parameter": "encut", "value": 700, "energy": -100.50515},
    ]
    inp = ConvergenceTestToolInput(
        action="cutoff_analysis",
        convergence_data=data,
        tolerance=0.001,
    )
    tool = ConvergenceTestTool()
    result = tool._run_cutoff_analysis(inp)
    assert result.success
    assert result.data["converged"] is True
    assert result.data["recommended_parameter"] == 600
    assert result.data["plateau_energy"] is not None
    assert "convergence_rate" in result.data


def test_cutoff_analysis_not_converged():
    """If deltas never drop below tolerance, converged should be False."""
    data = [
        {"parameter": "kmesh", "value": 4, "energy": -100.0},
        {"parameter": "kmesh", "value": 6, "energy": -100.5},
        {"parameter": "kmesh", "value": 8, "energy": -101.0},
    ]
    inp = ConvergenceTestToolInput(
        action="cutoff_analysis",
        convergence_data=data,
        tolerance=0.001,
    )
    tool = ConvergenceTestTool()
    result = tool._run_cutoff_analysis(inp)
    assert result.success
    assert result.data["converged"] is False


def test_unknown_action_raises_error():
    """An invalid action should fail Pydantic validation."""
    with pytest.raises(ValidationError):
        ConvergenceTestToolInput(action="bogus_action")
