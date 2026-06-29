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

from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile
from huginn.types import ToolContext, ToolResult


class CalibrationVariableSpec(BaseModel):
    name: str = Field(..., description="Variable name in objective_expression")
    low: float = Field(default=0.0)
    high: float = Field(default=1.0)


class GPToolInput(BaseModel):
    action: Literal[
        "fit",
        "predict",
        "suggest",
        "calibrate",
        "natural_gradient",
        "fisher_information",
        "kl_divergence",
    ] = Field(default="fit")
    X: list[list[float]] = Field(default_factory=list, description="Training inputs")
    y: list[float] = Field(default_factory=list, description="Training targets")
    X_new: list[list[float]] = Field(
        default_factory=list, description="Prediction candidate inputs"
    )
    length_scale: float = Field(default=1.0, gt=0)
    sigma_f: float = Field(default=1.0, gt=0, description="Signal variance")
    sigma_n: float = Field(default=1e-5, ge=0, description="Observation noise")
    kernel: Literal["rbf", "matern32", "matern52"] = Field(
        default="rbf", description="Covariance kernel family"
    )
    acquisition: Literal["ei", "ucb", "pi"] = Field(
        default="ei", description="Acquisition function for the suggest action"
    )
    kappa: float = Field(
        default=2.0, ge=0, description="Exploration weight for UCB acquisition"
    )
    maximize: bool = Field(
        default=False, description="If True, maximize y; else minimize"
    )
    # Natural gradient optimization
    n_steps: int = Field(
        default=20, ge=1, description="Steps for natural gradient descent"
    )
    lr: float = Field(
        default=0.01, gt=0, description="Learning rate for natural gradient descent"
    )
    # Fisher information / experimental design
    X_candidates: list[list[float]] = Field(
        default_factory=list,
        description="Candidate points for Fisher information experimental design",
    )
    # KL divergence between two posteriors
    y1: list[float] = Field(
        default_factory=list,
        description="First dataset targets for KL divergence",
    )
    y2: list[float] = Field(
        default_factory=list,
        description="Second dataset targets for KL divergence",
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
    """Pure NumPy GP with squared-exponential / Matérn kernels."""

    def __init__(
        self,
        length_scale: float,
        sigma_f: float,
        sigma_n: float,
        kernel: str = "rbf",
    ) -> None:
        self.length_scale = length_scale
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.kernel = kernel
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.L: np.ndarray | None = None
        self.alpha: np.ndarray | None = None

    @staticmethod
    def _kernel(
        a: np.ndarray,
        b: np.ndarray,
        length_scale: float,
        sigma_f: float,
        kernel: str = "rbf",
    ) -> np.ndarray:
        sqdist = (
            np.sum(a**2, axis=1).reshape(-1, 1) + np.sum(b**2, axis=1) - 2 * a @ b.T
        )
        if kernel == "matern32":
            r = np.sqrt(np.maximum(sqdist, 0.0)) * np.sqrt(3.0) / length_scale
            return sigma_f**2 * (1.0 + r) * np.exp(-r)
        if kernel == "matern52":
            r = np.sqrt(np.maximum(sqdist, 0.0)) * np.sqrt(5.0) / length_scale
            return sigma_f**2 * (1.0 + r + r**2 / 3.0) * np.exp(-r)
        # squared-exponential (RBF)
        return sigma_f**2 * np.exp(-0.5 * sqdist / length_scale**2)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        K = self._kernel(self.X, self.X, self.length_scale, self.sigma_f, self.kernel)
        K += (self.sigma_n**2 + 1e-8) * np.eye(len(self.y))
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))

    def predict(self, X_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.X is None or self.alpha is None or self.L is None:
            raise RuntimeError("GP must be fit before prediction")
        X_new = np.asarray(X_new, dtype=float)
        K_s = self._kernel(self.X, X_new, self.length_scale, self.sigma_f, self.kernel)
        mu = K_s.T @ self.alpha
        v = np.linalg.solve(self.L, K_s)
        K_ss = self._kernel(X_new, X_new, self.length_scale, self.sigma_f, self.kernel)
        var = np.diag(K_ss) - np.sum(v**2, axis=0)
        var = np.maximum(var, 0.0)
        return mu, np.sqrt(var)


class GPTool(HuginnTool):
    """Gaussian process surrogate for data-driven modeling and Bayesian optimization."""

    name = "gp_tool"
    category = "sci"
    profile = ToolProfile(phases=frozenset({ResearchPhase.VALIDATION}))
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
            if input_data.action == "natural_gradient":
                return self._natural_gradient(input_data)
            if input_data.action == "fisher_information":
                return self._fisher_information(input_data)
            if input_data.action == "kl_divergence":
                return self._kl_divergence(input_data)
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
                "kernel": args.kernel,
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

        acq_type = args.acquisition
        if acq_type == "ucb":
            acq = self._upper_confidence_bound(
                mu, sigma, args.kappa, maximize=args.maximize
            )
        elif acq_type == "pi":
            acq = self._probability_improvement(
                mu, sigma, incumbent, maximize=args.maximize
            )
        else:
            acq = self._expected_improvement(
                mu, sigma, incumbent, maximize=args.maximize
            )
        best_idx = int(np.argmax(acq))

        data: dict[str, Any] = {
            "suggested_index": best_idx,
            "suggested_X": args.X_new[best_idx],
            "acquisition_type": acq_type,
            "acquisition_value": float(acq[best_idx]),
            "predicted_mean": float(mu[best_idx]),
            "predicted_std": float(sigma[best_idx]),
            "all_acquisition": acq.tolist(),
            "message": (
                f"Suggested candidate at index {best_idx} "
                f"({acq_type}={acq[best_idx]:.6f})."
            ),
        }
        # Keep the historical EI keys around for callers that expect them.
        if acq_type == "ei":
            data["expected_improvement"] = float(acq[best_idx])
            data["all_ei"] = acq.tolist()

        return ToolResult(data=data, success=True)

    @staticmethod
    def _log_marginal_and_loo(
        log_theta: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        kernel: str,
    ) -> tuple[float, np.ndarray]:
        # Marginal log-likelihood plus per-sample LOO terms (R&W 5.12-5.13).
        # Log-space params keep the kernel hyperparameters positive and let us
        # build per-sample scores for the empirical Fisher matrix.
        sigma_f = float(np.exp(log_theta[0]))
        length_scale = float(np.exp(log_theta[1]))
        sigma_n = float(np.exp(log_theta[2]))
        gp = NumPyGP(length_scale, sigma_f, sigma_n, kernel)
        gp.fit(X, y)
        L = gp.L
        alpha = gp.alpha
        n = len(y)
        log_marg = (
            -0.5 * float(y @ alpha)
            - float(np.sum(np.log(np.diag(L))))
            - 0.5 * n * np.log(2.0 * np.pi)
        )
        # K^{-1} via two triangular solves; only the diagonal is needed for LOO.
        eye = np.eye(n)
        k_inv = np.linalg.solve(L.T, np.linalg.solve(L, eye))
        q = np.diag(k_inv)
        q = np.where(q <= 0.0, 1e-12, q)
        mu_loo = y - alpha / q
        var_loo = np.maximum(1.0 / q, 1e-12)
        log_p = -0.5 * np.log(2.0 * np.pi * var_loo) - 0.5 * (y - mu_loo) ** 2 / var_loo
        return log_marg, log_p

    def _natural_gradient(self, args: GPToolInput) -> ToolResult:
        """Optimize GP hyperparameters with natural gradient descent.

        The Fisher information matrix (estimated from per-sample LOO scores)
        preconditions the marginal-likelihood gradient so updates follow the
        statistical manifold instead of flat Euclidean parameter space.
        """
        X = np.asarray(args.X, dtype=float)
        y = np.asarray(args.y, dtype=float)
        if len(X) == 0 or len(y) == 0:
            return ToolResult(
                data=None, success=False, error="natural_gradient requires X and y."
            )
        if len(X) != len(y):
            return ToolResult(
                data=None,
                success=False,
                error="X and y must have the same length.",
            )

        kernel = args.kernel
        log_theta = np.array(
            [
                np.log(args.sigma_f),
                np.log(args.length_scale),
                np.log(args.sigma_n),
            ],
            dtype=float,
        )
        h = 1e-5
        eps = 1e-6
        trajectory: list[dict[str, Any]] = []
        converged = False
        grad_norm = float("inf")

        for step in range(args.n_steps):
            log_marg, log_p = self._log_marginal_and_loo(log_theta, X, y, kernel)
            grad = np.zeros(3)
            # Each column holds the per-sample score for one parameter.
            scores = np.zeros((len(y), 3))
            for i in range(3):
                tp = log_theta.copy()
                tm = log_theta.copy()
                tp[i] += h
                tm[i] -= h
                mp, lp = self._log_marginal_and_loo(tp, X, y, kernel)
                mm, lm = self._log_marginal_and_loo(tm, X, y, kernel)
                grad[i] = (mp - mm) / (2.0 * h)
                scores[:, i] = (lp - lm) / (2.0 * h)

            # Empirical Fisher: outer product of per-sample scores, regularized
            # before inversion for numerical stability.
            fisher = (scores.T @ scores) / len(y) + eps * np.eye(3)
            nat_grad = np.linalg.solve(fisher, grad)
            log_theta = log_theta + args.lr * nat_grad
            grad_norm = float(np.linalg.norm(grad))
            trajectory.append(
                {
                    "step": step,
                    "log_likelihood": float(log_marg),
                    "grad_norm": grad_norm,
                    "natural_grad_norm": float(np.linalg.norm(nat_grad)),
                    "sigma_f": float(np.exp(log_theta[0])),
                    "length_scale": float(np.exp(log_theta[1])),
                    "sigma_n": float(np.exp(log_theta[2])),
                }
            )
            if grad_norm < 1e-6:
                converged = True
                break

        return ToolResult(
            data={
                "optimized_sigma_f": float(np.exp(log_theta[0])),
                "optimized_length_scale": float(np.exp(log_theta[1])),
                "optimized_sigma_n": float(np.exp(log_theta[2])),
                "log_likelihood_trajectory": trajectory,
                "converged": converged,
                "n_steps_run": len(trajectory),
                "final_grad_norm": grad_norm,
                "method": "natural_gradient_descent",
                "message": (
                    f"Natural gradient optimization finished after "
                    f"{len(trajectory)} steps. Final log-likelihood: "
                    f"{trajectory[-1]['log_likelihood']:.6f}."
                ),
            },
            success=True,
        )

    def _fisher_information(self, args: GPToolInput) -> ToolResult:
        """Fisher information of GP hyperparameters over candidate points.

        Each candidate contributes an outer product of the posterior-mean
        sensitivity weighted by inverse predictive variance, supporting
        D/A/E-optimal experimental design.
        """
        X_train = np.asarray(args.X, dtype=float)
        y_train = np.asarray(args.y, dtype=float)
        if len(X_train) == 0 or len(y_train) == 0:
            return ToolResult(
                data=None,
                success=False,
                error="fisher_information requires X and y as training data.",
            )
        if not args.X_candidates:
            return ToolResult(
                data=None,
                success=False,
                error="fisher_information requires X_candidates.",
            )
        X_cand = np.asarray(args.X_candidates, dtype=float)
        kernel = args.kernel
        log_theta = np.array(
            [
                np.log(args.sigma_f),
                np.log(args.length_scale),
                np.log(args.sigma_n),
            ],
            dtype=float,
        )

        def predict(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            gp = NumPyGP(
                float(np.exp(theta[1])),
                float(np.exp(theta[0])),
                float(np.exp(theta[2])),
                kernel,
            )
            gp.fit(X_train, y_train)
            mu, std = gp.predict(X_cand)
            return mu, np.maximum(std**2, 1e-12)

        mu0, var0 = predict(log_theta)
        # Sensitivity of the posterior mean to each hyperparameter (central diff).
        h = 1e-5
        dmu = np.zeros((len(X_cand), 3))
        for i in range(3):
            tp = log_theta.copy()
            tm = log_theta.copy()
            tp[i] += h
            tm[i] -= h
            mu_p, _ = predict(tp)
            mu_m, _ = predict(tm)
            dmu[:, i] = (mu_p - mu_m) / (2.0 * h)

        n_cand = len(X_cand)
        fisher_total = np.zeros((3, 3))
        per_cand: list[float] = []
        for k in range(n_cand):
            g = dmu[k].reshape(-1, 1)
            fk = (g @ g.T) / var0[k]
            fisher_total += fk
            try:
                per_cand.append(
                    float(np.log(np.linalg.det(fk + 1e-12 * np.eye(3))))
                )
            except np.linalg.LinAlgError:
                per_cand.append(float("nan"))

        fisher_reg = fisher_total + 1e-10 * np.eye(3)
        try:
            d_opt = float(np.log(np.linalg.det(fisher_reg)))
        except np.linalg.LinAlgError:
            d_opt = float("nan")
        try:
            a_opt = float(np.trace(np.linalg.inv(fisher_reg)))
        except np.linalg.LinAlgError:
            a_opt = float("nan")
        try:
            e_opt = float(np.linalg.eigvalsh(fisher_reg)[0])
        except np.linalg.LinAlgError:
            e_opt = float("nan")

        return ToolResult(
            data={
                "fisher_matrix": fisher_total.tolist(),
                "d_optimal": d_opt,
                "a_optimal": a_opt,
                "e_optimal": e_opt,
                "per_candidate_contribution": per_cand,
                "n_candidates": n_cand,
                "message": (
                    f"Fisher information computed over {n_cand} candidates. "
                    f"D-opt={d_opt:.4f}, A-opt={a_opt:.4f}, E-opt={e_opt:.4f}."
                ),
            },
            success=True,
        )

    def _kl_divergence(self, args: GPToolInput) -> ToolResult:
        """KL divergence between two GP posteriors fit to y1 and y2.

        Predictive distributions are Gaussian at each test point, so the KL
        reduces to the univariate form and is averaged over the test set.
        """
        X = np.asarray(args.X, dtype=float)
        y1 = np.asarray(args.y1, dtype=float)
        y2 = np.asarray(args.y2, dtype=float)
        if len(X) == 0:
            return ToolResult(
                data=None, success=False, error="kl_divergence requires X."
            )
        if len(y1) == 0 or len(y2) == 0:
            return ToolResult(
                data=None,
                success=False,
                error="kl_divergence requires y1 and y2.",
            )
        if len(X) != len(y1) or len(X) != len(y2):
            return ToolResult(
                data=None,
                success=False,
                error="X, y1, and y2 must have the same length.",
            )

        X_test = np.asarray(args.X_new, dtype=float) if args.X_new else X

        gp1 = self._create_gp(args)
        gp1.fit(X, y1)
        gp2 = self._create_gp(args)
        gp2.fit(X, y2)
        mu1, std1 = gp1.predict(X_test)
        mu2, std2 = gp2.predict(X_test)
        var1 = np.maximum(std1**2, 1e-12)
        var2 = np.maximum(std2**2, 1e-12)

        kl = 0.5 * (
            np.log(var2 / var1) - 1.0 + var1 / var2 + (mu2 - mu1) ** 2 / var2
        )
        mean_kl = float(np.mean(kl))
        max_kl = float(np.max(kl))
        return ToolResult(
            data={
                "kl_divergence": mean_kl,
                "per_point_kl": kl.tolist(),
                "mean_kl": mean_kl,
                "max_kl": max_kl,
                "n_test_points": len(X_test),
                "message": (
                    f"KL divergence computed over {len(X_test)} test points. "
                    f"Mean KL={mean_kl:.6f}, max KL={max_kl:.6f}."
                ),
            },
            success=True,
        )

    def _create_gp(self, args: GPToolInput):
        if self._sklearn_available:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import (
                RBF,
                Matern,
                WhiteKernel,
            )
            from sklearn.gaussian_process.kernels import (
                ConstantKernel as C,
            )

            if args.kernel == "matern32":
                base_kernel = Matern(length_scale=args.length_scale, nu=1.5)
            elif args.kernel == "matern52":
                base_kernel = Matern(length_scale=args.length_scale, nu=2.5)
            else:
                base_kernel = RBF(length_scale=args.length_scale)

            kernel = C(args.sigma_f**2, (1e-3, 1e3)) * base_kernel + WhiteKernel(
                noise_level=args.sigma_n**2, noise_level_bounds=(1e-10, 1e1)
            )
            return _SklearnGPAdapter(
                GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
            )
        return NumPyGP(args.length_scale, args.sigma_f, args.sigma_n, args.kernel)

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

    @staticmethod
    def _probability_improvement(
        mu: np.ndarray,
        sigma: np.ndarray,
        incumbent: float,
        maximize: bool = False,
    ) -> np.ndarray:
        sigma = np.where(sigma < 1e-12, 1e-12, sigma)
        if maximize:
            z = (mu - incumbent) / sigma
        else:
            z = (incumbent - mu) / sigma
        return _phi_cdf(z)

    @staticmethod
    def _upper_confidence_bound(
        mu: np.ndarray,
        sigma: np.ndarray,
        kappa: float,
        maximize: bool = False,
    ) -> np.ndarray:
        # Maximization trusts the upper bound; minimization trusts the lower.
        if maximize:
            return mu + kappa * sigma
        return mu - kappa * sigma


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
