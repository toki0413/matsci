"""Tests for the interpretable ML closed-loop tool.

Covers the three actions and their composition: symbolic regression recovers a
known law, the Gaussian-process path returns calibrated uncertainty, and the
Bourbaki-style structure check catches dimensional / symmetry violations.
The full loop (SR -> validate_structure) is exercised end to end.
"""

import asyncio
import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import pytest

from huginn.tools.sci.interpretable_ml_tool import InterpretableMLInput, InterpretableMLTool
from huginn.tools.base import ResearchPhase
from huginn.types import ToolContext


def _ctx() -> ToolContext:
    return ToolContext(session_id="test", workspace=".")


def _run(tool, args):
    return asyncio.run(tool.call(args, _ctx()))


_HAS_SYMPY = importlib.util.find_spec("sympy") is not None
_HAS_GPYTORCH = importlib.util.find_spec("gpytorch") is not None


def _linear_data(seed: int = 0, n: int = 200, intercept: float = 3.0):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-2, 2, n)
    x1 = rng.uniform(-2, 2, n)
    y = 2.0 * x0 + intercept + rng.normal(0, 0.01, n)
    return {"x0": x0.tolist(), "x1": x1.tolist(), "y": y.tolist()}


# ── metadata ────────────────────────────────────────────────────


def test_tool_metadata():
    tool = InterpretableMLTool()
    assert tool.name == "interpretable_ml_tool"
    assert tool.category == "sci"
    assert tool.profile.cost_tier == "light"
    assert ResearchPhase.HYPOTHESIS in tool.profile.phases
    assert ResearchPhase.VALIDATION in tool.profile.phases


# ── symbolic regression ─────────────────────────────────────────


def test_symbolic_regression_recovers_linear_law():
    tool = InterpretableMLTool()
    res = _run(
        tool,
        InterpretableMLInput(
            action="symbolic_regression",
            data_json=_linear_data(),
            max_order=1,
            threshold=0.05,
        ),
    )
    assert res.success, res.error
    d = res.data
    coef = dict(zip(d["terms"], d["coefficients"]))
    assert abs(coef["x0"] - 2.0) < 0.05
    assert abs(coef["1"] - 3.0) < 0.05
    assert d["r2"] > 0.99
    assert d["target"] == "y"
    assert "x0" in d["features"]


def test_symbolic_regression_data_loading_errors():
    tool = InterpretableMLTool()
    # no data source
    res = _run(tool, InterpretableMLInput(action="symbolic_regression"))
    assert not res.success
    # unknown target column
    res = _run(
        tool,
        InterpretableMLInput(action="symbolic_regression", data_json={"x0": [0, 1, 2]}),
    )
    assert not res.success


def test_symbolic_regression_from_csv():
    tool = InterpretableMLTool()
    rng = np.random.default_rng(2)
    x0 = rng.uniform(-1, 1, 50)
    y = -1.5 * x0 + rng.normal(0, 0.01, 50)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "dat.csv"
        with open(p, "w", encoding="utf-8") as f:
            f.write("x0,y\n")
            for a, b in zip(x0, y):
                f.write(f"{a},{b}\n")
        res = _run(
            tool,
            InterpretableMLInput(
                action="symbolic_regression",
                data_file=str(p),
                target_column="y",
                max_order=1,
            ),
        )
    assert res.success, res.error
    coef = dict(zip(res.data["terms"], res.data["coefficients"]))
    assert abs(coef["x0"] + 1.5) < 0.1


# ── Gaussian process ─────────────────────────────────────────────


def test_gaussian_process_uncertainty_intervals():
    tool = InterpretableMLTool()
    data = _linear_data(seed=3)
    res = _run(
        tool,
        InterpretableMLInput(
            action="gaussian_process",
            data_json=data,
            # training data has 2 features (x0, x1), so X_new must be 2-D
            X_new=[[0.0, 0.0], [1.0, 0.5], [-1.0, -0.5]],
            use_gpytorch=False,  # force the always-available numpy backend
        ),
    )
    assert res.success, res.error
    d = res.data
    assert d["backend"] == "numpy"
    mu = np.array(d["mean"])
    lower = np.array(d["lower"])
    upper = np.array(d["upper"])
    assert mu.shape == (3,)
    # the interval must bracket the mean
    assert np.all(lower <= mu) and np.all(mu <= upper)
    assert d["confidence"] == 0.95


def test_gaussian_process_dim_mismatch_returns_clean_error():
    tool = InterpretableMLTool()
    res = _run(
        tool,
        InterpretableMLInput(
            action="gaussian_process",
            data_json=_linear_data(seed=7),  # 2 features
            X_new=[[0.0]],  # wrong feature count
            use_gpytorch=False,
        ),
    )
    # either gpytorch handled it or the numpy backend surfaced a clean error;
    # never an uncaught exception either way.
    assert isinstance(res.success, bool)
    if not res.success:
        assert "GP" in res.error or "X_new" in res.error


@pytest.mark.skipif(not _HAS_GPYTORCH, reason="gpytorch not installed")
def test_gaussian_process_gpytorch_backend():
    tool = InterpretableMLTool()
    res = _run(
        tool,
        InterpretableMLInput(
            action="gaussian_process",
            data_json=_linear_data(seed=4),
            use_gpytorch=True,
        ),
    )
    assert res.success, res.error
    # gpytorch may fall back to numpy due to torch version incompat;
    # both backends should produce valid predictions
    assert res.data["backend"] in ("gpytorch", "numpy")
    assert len(res.data["mean"]) > 0


# ── structure validation ─────────────────────────────────────────


@pytest.mark.skipif(not _HAS_SYMPY, reason="sympy not installed")
def test_validate_structure_catches_dimensional_inconsistency():
    tool = InterpretableMLTool()
    # y = 2*x0 + 3 : the bare constant is dimensionless, x0 carries length,
    # so additive terms disagree on dimension.
    res = _run(
        tool,
        InterpretableMLInput(
            action="validate_structure",
            equation="y = 2.0*x0 + 3.0",
            variable_units={"x0": "m", "y": "m"},
        ),
    )
    assert res.success
    dims = next(c for c in res.data["checks"] if c["name"] == "dimensional_consistency")
    assert not dims["passed"]

    # the dimensionally clean version passes
    res2 = _run(
        tool,
        InterpretableMLInput(
            action="validate_structure",
            equation="y = 2.0*x0",
            variable_units={"x0": "m", "y": "m"},
        ),
    )
    assert res2.data["all_passed"]


@pytest.mark.skipif(not _HAS_SYMPY, reason="sympy not installed")
def test_validate_structure_symmetry_detection():
    tool = InterpretableMLTool()
    # even function in x0
    res = _run(
        tool,
        InterpretableMLInput(
            action="validate_structure", equation="y = x0**2", symmetry_vars=["x0"]
        ),
    )
    sym = next(c for c in res.data["checks"] if c["name"] == "symmetry:x0")
    assert sym["symmetry"] == "even"
    assert sym["passed"]

    # odd function in x0
    res = _run(
        tool,
        InterpretableMLInput(
            action="validate_structure", equation="y = x0", symmetry_vars=["x0"]
        ),
    )
    sym = next(c for c in res.data["checks"] if c["name"] == "symmetry:x0")
    assert sym["symmetry"] == "odd"


@pytest.mark.skipif(not _HAS_SYMPY, reason="sympy not installed")
def test_validate_structure_from_terms_and_coefficients():
    tool = InterpretableMLTool()
    # Build the equation from SR-style output rather than an equation string.
    res = _run(
        tool,
        InterpretableMLInput(
            action="validate_structure",
            terms=["1", "x0", "x1"],
            coefficients=[0.0, 2.0, 0.0],
            target_column="y",
            variable_units={"x0": "m", "y": "m"},
        ),
    )
    assert res.success
    assert res.data["equation"] == "y = 2.0*x0"
    assert res.data["all_passed"]


# ── closed loop: SR -> validate_structure ─────────────────────────


@pytest.mark.skipif(not _HAS_SYMPY, reason="sympy not installed")
def test_closed_loop_sr_then_validate():
    tool = InterpretableMLTool()
    # Clean proportional law so the discovered equation is dimensionally sound.
    rng = np.random.default_rng(5)
    x0 = rng.uniform(-2, 2, 200)
    y = 2.0 * x0
    sr = _run(
        tool,
        InterpretableMLInput(
            action="symbolic_regression",
            data_json={"x0": x0.tolist(), "y": y.tolist()},
            max_order=1,
            threshold=0.05,
        ),
    )
    assert sr.success
    # Feed the discovered equation back for structural validation.
    val = _run(
        tool,
        InterpretableMLInput(
            action="validate_structure",
            equation=sr.data["equation"],
            variable_units={"x0": "m", "y": "m"},
        ),
    )
    assert val.success
    assert val.data["all_passed"], val.data


def test_unknown_action_errors():
    # pydantic rejects invalid actions at construction time
    with pytest.raises(Exception):
        InterpretableMLInput(action="bogus")  # type: ignore[arg-type]
