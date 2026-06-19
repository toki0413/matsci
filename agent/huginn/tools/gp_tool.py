"""Gaussian process surrogate and Bayesian optimization helper tool.

Fits a GP to tabular data (X, y), predicts mean and uncertainty, and computes
expected improvement for Bayesian optimization.  Uses scikit-learn when
available; otherwise falls back to a pure NumPy implementation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult


class CalibrationVariableSpec(BaseModel):
    name: str = Field(..., description="Variable name in objective_expression")
    low: float = Field(default=0.0)
    high: float = Field(default=1.0)


class GPToolInput(BaseModel):
    action: Literal["fit", "predict", "suggest", "calibrate"] = Field(default="fit")
    X: list[list[float]] = Field(default_factory=list, description="Training inputs")
    y: list[float] = Field(default_factory=list, description="Training targets")
    X_new: list[list[float]] = Field(
        default_factory=list, description="Prediction candidate inputs"
    )
    length_scale: float = Field(default=1.0, gt=0)
    sigma_f: float = Field(default=1.0, gt=0, description="Signal variance")
    sigma_n: float = Field(default=1e-5, ge=0, description="Observation noise")
    maximize: bool = Field(
        default=False, description="If True, maximize y; else minimize"
    )
    # Calibration / Bayesian optimization
    objective_expression: str | None = Field(
        default=None, description="SymPy expression for the true objective"
    )
    calibration_variables: list[CalibrationVariableSpec] = Field(default_factory=list)
    n_initial: int = Field(default=5, ge=1)
    n_iterations: int = Field(default=10, ge=1)
    seed: int | None = Field(default=None)
    # Real-simulator coupling
    objective_tool: str | None = Field(
        default=None,
        description="Name of a registered tool to call for each objective evaluation",
    )
    objective_path: str = Field(
        default="data.value",
        description="Dotted path into the tool result dict to extract the objective value",
    )
    tool_input_template: dict[str, Any] = Field(
        default_factory=dict,
        description="Template args passed to objective_tool; use {var_name} placeholders",
    )
    tool_working_dir: str | None = Field(default=None)
    working_dir: str | None = Field(default=None)


class NumPyGP:
    """Pure NumPy GP with squared-exponential kernel."""

    def __init__(self, length_scale: float, sigma_f: float, sigma_n: float) -> None:
        self.length_scale = length_scale
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.L: np.ndarray | None = None
        self.alpha: np.ndarray | None = None

    @staticmethod
    def _kernel(
        a: np.ndarray, b: np.ndarray, length_scale: float, sigma_f: float
    ) -> np.ndarray:
        sqdist = (
            np.sum(a**2, axis=1).reshape(-1, 1) + np.sum(b**2, axis=1) - 2 * a @ b.T
        )
        return sigma_f**2 * np.exp(-0.5 * sqdist / length_scale**2)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        K = self._kernel(self.X, self.X, self.length_scale, self.sigma_f)
        K += (self.sigma_n**2 + 1e-8) * np.eye(len(self.y))
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))

    def predict(self, X_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.X is None or self.alpha is None or self.L is None:
            raise RuntimeError("GP must be fit before prediction")
        X_new = np.asarray(X_new, dtype=float)
        K_s = self._kernel(self.X, X_new, self.length_scale, self.sigma_f)
        mu = K_s.T @ self.alpha
        v = np.linalg.solve(self.L, K_s)
        K_ss = self._kernel(X_new, X_new, self.length_scale, self.sigma_f)
        var = np.diag(K_ss) - np.sum(v**2, axis=0)
        var = np.maximum(var, 0.0)
        return mu, np.sqrt(var)


class GPTool(HuginnTool):
    """Gaussian process surrogate for data-driven modeling and Bayesian optimization."""

    name = "gp_tool"
    description = (
        "Fit a Gaussian process surrogate to data, predict with uncertainty, "
        "and compute expected improvement for Bayesian optimization."
    )
    input_schema = GPToolInput

    def __init__(self) -> None:
        super().__init__()
        self._sklearn_available = self._check_sklearn()

    @staticmethod
    def _check_sklearn() -> bool:
        try:
            import sklearn.gaussian_process  # noqa: F401

            return True
        except ImportError:
            return False

    def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = GPToolInput(**args)
        work_dir = (
            Path(input_data.working_dir) if input_data.working_dir else Path.cwd()
        )
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            if input_data.action == "predict":
                return self._predict(input_data)
            if input_data.action == "suggest":
                return self._suggest(input_data)
            if input_data.action == "calibrate":
                return self._calibrate(input_data)
            return self._fit(input_data)
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"GP tool failed: {e}")

    def _fit(self, args: GPToolInput) -> ToolResult:
        X = np.asarray(args.X, dtype=float)
        y = np.asarray(args.y, dtype=float)
        if len(X) == 0 or len(y) == 0:
            return ToolResult(
                data=None, success=False, error="X and y must be non-empty."
            )
        if len(X) != len(y):
            return ToolResult(
                data=None, success=False, error="X and y must have the same length."
            )

        # Use a tiny dummy prediction to verify the GP is usable.
        gp = self._create_gp(args)
        gp.fit(X, y)
        mu, sigma = gp.predict(X[:1])

        return ToolResult(
            data={
                "n_train": len(X),
                "input_dim": X.shape[1],
                "length_scale": args.length_scale,
                "sigma_f": args.sigma_f,
                "sigma_n": args.sigma_n,
                "backend": "sklearn" if self._sklearn_available else "numpy",
                "message": f"GP fitted on {len(X)} points using { 'sklearn' if self._sklearn_available else 'numpy' } backend.",
            },
            success=True,
        )

    def _predict(self, args: GPToolInput) -> ToolResult:
        if not args.X_new:
            return ToolResult(
                data=None, success=False, error="predict action requires X_new."
            )
        gp = self._create_gp(args)
        gp.fit(np.asarray(args.X, dtype=float), np.asarray(args.y, dtype=float))
        mu, sigma = gp.predict(np.asarray(args.X_new, dtype=float))
        return ToolResult(
            data={
                "mean": mu.tolist(),
                "std": sigma.tolist(),
                "message": f"Predictions generated for {len(args.X_new)} points.",
            },
            success=True,
        )

    def _suggest(self, args: GPToolInput) -> ToolResult:
        if not args.X_new:
            return ToolResult(
                data=None,
                success=False,
                error="suggest action requires candidate X_new.",
            )
        gp = self._create_gp(args)
        gp.fit(np.asarray(args.X, dtype=float), np.asarray(args.y, dtype=float))
        mu, sigma = gp.predict(np.asarray(args.X_new, dtype=float))

        y_train = np.asarray(args.y, dtype=float)
        incumbent = float(np.max(y_train)) if args.maximize else float(np.min(y_train))

        ei = self._expected_improvement(mu, sigma, incumbent, maximize=args.maximize)
        best_idx = int(np.argmax(ei))

        return ToolResult(
            data={
                "suggested_index": best_idx,
                "suggested_X": args.X_new[best_idx],
                "expected_improvement": float(ei[best_idx]),
                "predicted_mean": float(mu[best_idx]),
                "predicted_std": float(sigma[best_idx]),
                "all_ei": ei.tolist(),
                "message": f"Suggested candidate at index {best_idx} with EI {ei[best_idx]:.6f}.",
            },
            success=True,
        )

    def _create_gp(self, args: GPToolInput):
        if self._sklearn_available:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import (
                RBF,
                WhiteKernel,
            )
            from sklearn.gaussian_process.kernels import (
                ConstantKernel as C,
            )

            kernel = C(args.sigma_f**2, (1e-3, 1e3)) * RBF(
                args.length_scale, (1e-3, 1e3)
            ) + WhiteKernel(
                noise_level=args.sigma_n**2, noise_level_bounds=(1e-10, 1e1)
            )
            return _SklearnGPAdapter(
                GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
            )
        return NumPyGP(args.length_scale, args.sigma_f, args.sigma_n)

    def _calibrate(self, args: GPToolInput) -> ToolResult:
        """Closed-loop Bayesian calibration/optimization against a symbolic objective or a real simulator tool."""
        if not args.calibration_variables:
            return ToolResult(
                data=None,
                success=False,
                error="calibrate action requires calibration_variables.",
            )
        if not args.objective_tool and not args.objective_expression:
            return ToolResult(
                data=None,
                success=False,
                error="calibrate requires objective_expression or objective_tool.",
            )

        bounds = [(v.low, v.high) for v in args.calibration_variables]
        var_names = [v.name for v in args.calibration_variables]
        rng = np.random.default_rng(args.seed if args.seed is not None else None)

        evaluate = self._make_evaluator(args, var_names)

        # Initialize with provided data or random samples
        if args.X and args.y:
            X = np.asarray(args.X, dtype=float).tolist()
            y = np.asarray(args.y, dtype=float).tolist()
        else:
            X = self._lhs_samples(args.n_initial, bounds, rng)
            y = [evaluate(row) for row in X]

        history: list[dict[str, Any]] = [
            {"iteration": 0, "X": row, "y": val} for row, val in zip(X, y)
        ]

        for iteration in range(1, args.n_iterations + 1):
            X_arr = np.asarray(X, dtype=float)
            y_arr = np.asarray(y, dtype=float)

            gp = self._create_gp(args)
            gp.fit(X_arr, y_arr)

            # Candidate grid: random samples in bounds
            n_candidates = max(50, 10 * len(args.calibration_variables))
            candidates = self._lhs_samples(n_candidates, bounds, rng)
            cand_arr = np.asarray(candidates, dtype=float)
            mu, sigma = gp.predict(cand_arr)

            incumbent = float(np.max(y_arr)) if args.maximize else float(np.min(y_arr))
            ei = self._expected_improvement(
                mu, sigma, incumbent, maximize=args.maximize
            )
            best_idx = int(np.argmax(ei))
            next_x = candidates[best_idx]
            next_y = evaluate(next_x)

            X.append(next_x)
            y.append(next_y)
            history.append(
                {
                    "iteration": iteration,
                    "X": next_x,
                    "y": next_y,
                    "ei": float(ei[best_idx]),
                }
            )

        y_arr = np.asarray(y, dtype=float)
        best_idx = int(np.argmax(y_arr)) if args.maximize else int(np.argmin(y_arr))

        return ToolResult(
            data={
                "method": "bayesian_calibration",
                "maximize": args.maximize,
                "best_X": X[best_idx],
                "best_y": float(y_arr[best_idx]),
                "history": history,
                "n_evaluations": len(X),
                "variable_names": var_names,
                "message": (
                    f"Calibration completed after {len(X)} evaluations. "
                    f"Best objective: {y_arr[best_idx]:.6f}"
                ),
            },
            success=True,
        )

    def _make_evaluator(self, args: GPToolInput, var_names: list[str]):
        """Return a callable that evaluates the objective for a design vector."""
        if args.objective_tool:
            return lambda row: self._evaluate_with_tool(row, args, var_names)

        try:
            import sympy as sp
        except ImportError as exc:
            raise RuntimeError(
                "GP calibration requires sympy. Install with: pip install sympy"
            ) from exc

        symbols = {v.name: sp.Symbol(v.name) for v in args.calibration_variables}
        expr = sp.sympify(args.objective_expression, locals=symbols)
        f_obj = sp.lambdify(list(symbols.values()), expr, modules="numpy")
        return lambda row: float(f_obj(*[row[i] for i in range(len(row))]))

    def _evaluate_with_tool(
        self, row: list[float], args: GPToolInput, var_names: list[str]
    ) -> float:
        from huginn.tools.registry import ToolRegistry

        tool = ToolRegistry.get(args.objective_tool)
        if tool is None:
            raise RuntimeError(
                f"Objective tool '{args.objective_tool}' not found in registry."
            )

        values = {name: float(row[i]) for i, name in enumerate(var_names)}
        tool_args = self._format_template(args.tool_input_template, values, args)
        result = self._call_tool(tool, tool_args)
        if not result.success:
            raise RuntimeError(f"Objective tool failed: {result.error}")

        return float(self._extract_value(result.data, args.objective_path))

    @staticmethod
    def _format_template(
        template: dict[str, Any],
        values: dict[str, float],
        args: GPToolInput,
    ) -> dict[str, Any]:
        """Substitute {var_name} placeholders and inject working_dir if needed."""
        raw = json.dumps(template)
        raw = raw.format(**values)
        parsed = json.loads(raw)
        if args.tool_working_dir:
            parsed.setdefault("working_dir", args.tool_working_dir)
        return parsed

    def _call_tool(self, tool: Any, tool_args: dict[str, Any]) -> ToolResult:
        import asyncio
        import inspect

        if inspect.iscoroutinefunction(tool.call):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(tool.call(tool_args))
            # Already inside an async loop: schedule and run to completion
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, tool.call(tool_args))
                return future.result()
        return tool.call(tool_args)

    @staticmethod
    def _extract_value(data: Any, path: str) -> Any:
        parts = path.split(".")
        current = data
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                raise ValueError(f"Cannot extract path '{path}' from {type(data)}")
        return current

    @staticmethod
    def _lhs_samples(
        n: int, bounds: list[tuple[float, float]], rng: np.random.Generator
    ) -> list[list[float]]:
        """Generate Latin-hypercube-like samples within bounds."""
        dim = len(bounds)
        samples = np.zeros((n, dim))
        for d, (low, high) in enumerate(bounds):
            # Random permutation of strata + jitter
            cut = np.linspace(0, 1, n + 1)
            a = cut[:-1]
            b = cut[1:]
            samples[:, d] = a + (b - a) * rng.random(n)
            samples[:, d] = low + samples[:, d] * (high - low)
        # Shuffle each dimension independently to avoid correlation
        for d in range(dim):
            rng.shuffle(samples[:, d])
        return samples.tolist()

    @staticmethod
    def _expected_improvement(
        mu: np.ndarray,
        sigma: np.ndarray,
        incumbent: float,
        maximize: bool = False,
    ) -> np.ndarray:
        sigma = np.where(sigma < 1e-12, 1e-12, sigma)
        if maximize:
            z = (mu - incumbent) / sigma
            ei = (mu - incumbent) * _phi_cdf(z) + sigma * _phi_pdf(z)
        else:
            z = (incumbent - mu) / sigma
            ei = (incumbent - mu) * _phi_cdf(z) + sigma * _phi_pdf(z)
        return np.maximum(ei, 0.0)


class _SklearnGPAdapter:
    """Minimal adapter to give sklearn GP the same interface as NumPyGP."""

    def __init__(self, gpr):
        self.gpr = gpr

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.gpr.fit(X, y)

    def predict(self, X_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu, sigma = self.gpr.predict(X_new, return_std=True)
        return mu, sigma


def _phi_cdf(z: np.ndarray) -> np.ndarray:
    # Standard normal CDF.
    try:
        from scipy.special import erf
    except ImportError:
        import math

        erf = np.vectorize(math.erf)
    return 0.5 * (1 + erf(z / np.sqrt(2)))


def _phi_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
