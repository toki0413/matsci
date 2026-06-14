"""Tests for the Gaussian process surrogate tool."""

import numpy as np
import pytest

from huginn.tools.gp_tool import GPTool, GPToolInput


def test_gp_tool_fit_and_predict() -> None:
    """GPTool should fit a GP and predict on new points."""
    tool = GPTool()
    x = np.linspace(0, 10, 8).reshape(-1, 1).tolist()
    y = (np.sin(np.linspace(0, 10, 8)) + np.random.default_rng(0).normal(0, 0.05, 8)).tolist()
    result = tool.call({"action": "fit", "X": x, "y": y})
    assert result.success is True
    assert result.data["n_train"] == 8

    x_new = [[1.5], [3.5], [5.5]]
    pred = tool.call({"action": "predict", "X": x, "y": y, "X_new": x_new})
    assert pred.success is True
    assert len(pred.data["mean"]) == 3
    assert len(pred.data["std"]) == 3
    assert all(s >= 0 for s in pred.data["std"])


def test_gp_tool_suggest_minimize() -> None:
    """GPTool should suggest a candidate with high expected improvement."""
    tool = GPTool()
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 10, 6).reshape(-1, 1).tolist()
    y = [np.sin(v[0]) + rng.normal(0, 0.05) for v in x]

    candidates = [[i * 0.5] for i in range(21)]
    result = tool.call(
        {
            "action": "suggest",
            "X": x,
            "y": y,
            "X_new": candidates,
            "maximize": False,
        }
    )
    assert result.success is True
    assert 0 <= result.data["suggested_index"] < len(candidates)
    assert result.data["expected_improvement"] >= 0


def test_gp_tool_suggest_maximize() -> None:
    """GPTool should suggest a candidate for maximization."""
    tool = GPTool()
    x = [[0.0], [1.0], [2.0], [3.0]]
    y = [0.0, 0.5, 0.3, 0.8]
    candidates = [[0.5], [1.5], [2.5], [3.5]]
    result = tool.call(
        {
            "action": "suggest",
            "X": x,
            "y": y,
            "X_new": candidates,
            "maximize": True,
        }
    )
    assert result.success is True
    assert result.data["expected_improvement"] >= 0


def test_gp_tool_calibrate() -> None:
    """GPTool calibrate action should optimize a symbolic objective."""
    tool = GPTool()
    result = tool.call(
        {
            "action": "calibrate",
            "objective_expression": "-(x - 2.5)**2 + 5",
            "calibration_variables": [
                {"name": "x", "low": 0.0, "high": 5.0},
            ],
            "n_initial": 3,
            "n_iterations": 5,
            "maximize": True,
            "length_scale": 0.5,
            "sigma_n": 0.001,
        }
    )
    assert result.success is True
    data = result.data
    assert data["method"] == "bayesian_calibration"
    assert data["n_evaluations"] == 8
    # Best y should be close to global maximum 5.0 at x=2.5
    assert data["best_y"] > 4.5


def test_gp_tool_input_schema() -> None:
    """GPToolInput should validate parameters."""
    inp = GPToolInput(
        action="predict",
        X=[[0.0], [1.0]],
        y=[0.0, 1.0],
        X_new=[[0.5]],
        length_scale=1.5,
    )
    assert inp.action == "predict"
    assert inp.length_scale == 1.5
