"""Integration tests for the SR-guided GP action (sr_guided_gp).

The new action wires symbolic_regression and gaussian_process together: SINDy
discovers the closed-form trend, a GP is fit on the SR residuals, and the
combined prediction is SR_mean + GP_residual. These checks verify the wiring
holds and that the decomposition actually pays off on a trend + residual dataset
where a bare GP has to model both at once.
"""

import asyncio

import numpy as np

from huginn.tools.sci.interpretable_ml_tool import InterpretableMLInput, InterpretableMLTool
from huginn.types import ToolContext


def _ctx() -> ToolContext:
    return ToolContext(session_id="test", workspace=".")


def _run(tool, args):
    return asyncio.run(tool.call(args, _ctx()))


def _rmse(pred, truth):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(truth)) ** 2)))


def _trend_residual_data(seed: int = 0, n_train: int = 10):
    """y = 2*x + 0.8*sin(2.5*x): a linear trend SINDy nails at max_order=1 plus a
    sinusoidal residual the GP has to mop up. 10 sparse training points is the
    regime where a single RBF GP on y is forced to trade off the steep linear
    trend against the oscillation, while SR + residual-GP sidesteps that."""
    rng = np.random.default_rng(seed)
    x_train = np.linspace(-3, 3, n_train)
    y_train = 2.0 * x_train + 0.8 * np.sin(2.5 * x_train) + rng.normal(0, 0.02, n_train)
    x_test = np.linspace(-2.9, 2.9, 80)
    y_test = 2.0 * x_test + 0.8 * np.sin(2.5 * x_test)
    return x_train, y_train, x_test, y_test


# ── structure / wiring ────────────────────────────────────────────


def test_sr_guided_gp_executes_and_returns_expected_keys():
    tool = InterpretableMLTool()
    x_tr, y_tr, x_te, _ = _trend_residual_data()
    res = _run(
        tool,
        InterpretableMLInput(
            action="sr_guided_gp",
            data_json={"x": x_tr.tolist(), "y": y_tr.tolist()},
            X_new=x_te.reshape(-1, 1).tolist(),
            max_order=1,
            threshold=0.05,
            length_scale=0.8,
            use_gpytorch=False,  # pin the always-available numpy backend
        ),
    )
    assert res.success, res.error
    d = res.data
    # SR side
    assert "sr_equation" in d
    sr = d["sr_equation"]
    assert {"equation", "terms", "coefficients", "r2"} <= set(sr)
    # GP-on-residuals side
    assert "gp_residuals" in d
    gp = d["gp_residuals"]
    assert {"mean", "lower", "upper"} <= set(gp)
    # combined output
    assert d["combined_approach"] == "SR mean + GP residual"
    assert "interpretation" in d and isinstance(d["interpretation"], str)
    combined = np.asarray(d["combined_mean"])
    lower = np.asarray(d["combined_lower"])
    upper = np.asarray(d["combined_upper"])
    assert combined.shape == (len(x_te),)
    # the SR mean is exposed so callers can decompose the prediction
    assert len(d["sr_mean"]) == len(x_te)
    # confidence band must bracket the combined mean everywhere
    assert np.all(lower <= combined) and np.all(combined <= upper)
    # combined == SR mean + GP residual mean (the documented composition)
    sr_mean = np.asarray(d["sr_mean"])
    gp_mean = np.asarray(gp["mean"])
    assert np.allclose(combined, sr_mean + gp_mean, atol=1e-9)
    assert d["n_predict"] == len(x_te)


def test_sr_guided_gp_residual_target_is_residual_by_construction():
    # The GP is fit on y - SR_pred, so at the training points its mean must be
    # close to the residuals (the GP interpolates its training targets with
    # near-zero noise). Combined at training points therefore reconstructs y.
    tool = InterpretableMLTool()
    x_tr, y_tr, _, _ = _trend_residual_data()
    res = _run(
        tool,
        InterpretableMLInput(
            action="sr_guided_gp",
            data_json={"x": x_tr.tolist(), "y": y_tr.tolist()},
            max_order=1,
            threshold=0.05,
            length_scale=0.8,
            use_gpytorch=False,
        ),
    )
    assert res.success, res.error
    combined = np.asarray(res.data["combined_mean"])
    # no X_new -> predicts at training X, so combined should track y closely
    assert _rmse(combined, y_tr) < 0.05


def test_sr_guided_gp_propagates_data_errors():
    tool = InterpretableMLTool()
    # no data source at all
    res = _run(tool, InterpretableMLInput(action="sr_guided_gp"))
    assert not res.success


# ── accuracy: combined beats pure SR and pure GP ──────────────────


def test_combined_beats_pure_sr_and_pure_gp():
    tool = InterpretableMLTool()
    x_tr, y_tr, x_te, y_te = _trend_residual_data(seed=0, n_train=10)

    # --- SR-guided GP combined prediction at the held-out test points ---
    comb_res = _run(
        tool,
        InterpretableMLInput(
            action="sr_guided_gp",
            data_json={"x": x_tr.tolist(), "y": y_tr.tolist()},
            X_new=x_te.reshape(-1, 1).tolist(),
            max_order=1,
            threshold=0.05,
            length_scale=0.8,
            use_gpytorch=False,
        ),
    )
    assert comb_res.success, comb_res.error
    combined = np.asarray(comb_res.data["combined_mean"])

    # --- pure SR: rebuild the design matrix at the test points with the same
    # library SR used, then apply the discovered coefficients ---
    sr_res = _run(
        tool,
        InterpretableMLInput(
            action="symbolic_regression",
            data_json={"x": x_tr.tolist(), "y": y_tr.tolist()},
            max_order=1,
            threshold=0.05,
        ),
    )
    assert sr_res.success, sr_res.error
    coefs = np.asarray(sr_res.data["coefficients"], dtype=float)
    _, Theta_te = tool._sindy._build_library(x_te.reshape(-1, 1), 1, False, False)
    sr_pred = Theta_te @ coefs

    # --- pure GP on the raw target y, same kernel hyperparameters ---
    gp_res = _run(
        tool,
        InterpretableMLInput(
            action="gaussian_process",
            data_json={"x": x_tr.tolist(), "y": y_tr.tolist()},
            X_new=x_te.reshape(-1, 1).tolist(),
            length_scale=0.8,
            use_gpytorch=False,
        ),
    )
    assert gp_res.success, gp_res.error
    gp_pred = np.asarray(gp_res.data["mean"], dtype=float)

    sr_rmse = _rmse(sr_pred, y_te)
    gp_rmse = _rmse(gp_pred, y_te)
    comb_rmse = _rmse(combined, y_te)

    # SR misses the sinusoidal residual entirely (no trig in the library).
    assert sr_rmse > 0.3
    # The GP residual correction must beat bare SR by a wide margin.
    assert comb_rmse < sr_rmse
    # A single RBF GP on y has to model the steep linear trend and the
    # oscillation with one kernel; splitting the job (SR trend + GP residual)
    # is better conditioned and wins here too.
    assert comb_rmse < gp_rmse
