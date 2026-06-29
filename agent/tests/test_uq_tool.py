"""Tests for the uncertainty quantification tool."""

import asyncio

import pytest

from huginn.tools.uq_tool import UQTool, UQToolInput


def test_uq_tool_monte_carlo() -> None:
    """UQTool should propagate uncertainty through a symbolic expression."""
    tool = UQTool()
    result = asyncio.run(
        tool.call(
            {
                "action": "monte_carlo",
                "expression": "E * epsilon",
                "variables": [
                    {"name": "E", "distribution": "uniform", "low": 200e9, "high": 220e9},
                    {
                        "name": "epsilon",
                        "distribution": "normal",
                        "mean": 0.001,
                        "std": 0.0001,
                    },
                ],
                "n_samples": 2000,
                "seed": 42,
            }
        )
    )
    assert result.success is True
    data = result.data
    assert data["method"] == "monte_carlo"
    assert data["n_samples"] == 2000
    assert 190e6 < data["mean"] < 250e6
    assert data["std"] > 0
    assert "histogram" in data


def test_uq_tool_sensitivity() -> None:
    """UQTool should compute local sensitivities."""
    tool = UQTool()
    result = asyncio.run(
        tool.call(
            {
                "action": "sensitivity",
                "expression": "a**2 + 2*b",
                "variables": [
                    {"name": "a", "distribution": "uniform", "low": 1.0, "high": 3.0},
                    {"name": "b", "distribution": "uniform", "low": 0.0, "high": 2.0},
                ],
            }
        )
    )
    assert result.success is True
    data = result.data
    assert data["method"] == "sensitivity"
    assert data["nominal_output"] == pytest.approx(6.0, abs=1e-9)
    assert data["sensitivities"]["a"]["symbolic_derivative"] == pytest.approx(
        4.0, abs=1e-9
    )
    assert data["sensitivities"]["b"]["symbolic_derivative"] == pytest.approx(
        2.0, abs=1e-9
    )


def test_uq_tool_sobol() -> None:
    """UQTool should compute Sobol global sensitivity indices."""
    tool = UQTool()
    result = asyncio.run(
        tool.call(
            {
                "action": "sobol",
                "expression": "a + 10*b",
                "variables": [
                    {"name": "a", "distribution": "uniform", "low": 0.0, "high": 1.0},
                    {"name": "b", "distribution": "uniform", "low": 0.0, "high": 2.0},
                ],
                "n_samples": 2000,
                "seed": 42,
            }
        )
    )
    assert result.success is True
    data = result.data
    assert data["method"] == "sobol"
    assert data["n_samples"] == 2000
    # b dominates because coefficient is 10x larger
    assert data["S1"]["b"] > data["S1"]["a"]
    assert data["ST"]["b"] > data["ST"]["a"]


def test_uq_tool_propagate() -> None:
    """UQTool GUM 法线性误差传播."""
    tool = UQTool()
    result = asyncio.run(
        tool.call(
            {
                "action": "propagate",
                "expression": "a*b",
                "variables": {
                    "a": {"value": 2.0, "uncertainty": 0.1},
                    "b": {"value": 3.0, "uncertainty": 0.2},
                },
            }
        )
    )
    assert result.success is True
    data = result.data
    assert data["nominal_value"] == pytest.approx(6.0, abs=1e-9)
    assert data["combined_uncertainty"] == pytest.approx(0.5, abs=1e-9)
    assert data["sensitivity_coefficients"]["a"] == pytest.approx(3.0, abs=1e-9)
    assert data["sensitivity_coefficients"]["b"] == pytest.approx(2.0, abs=1e-9)


def test_uq_tool_input_schema() -> None:
    """UQToolInput should validate parameters."""
    inp = UQToolInput(
        action="monte_carlo",
        expression="x + y",
        n_samples=500,
    )
    assert inp.n_samples == 500
