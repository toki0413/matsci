"""Uncertainty quantification and sensitivity analysis tool.

Provides Monte Carlo sampling and local sensitivity analysis for symbolic
material-science models.  Works with SymPy expressions so the LLM can write a
model in plain math and immediately inspect how parameter uncertainty propagates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile
from huginn.types import ToolContext, ToolResult


class VariableSpec(BaseModel):
    name: str = Field(..., description="Variable name")
    distribution: Literal["uniform", "normal", "lognormal"] = Field(default="uniform")
    low: float | None = Field(default=None)
    high: float | None = Field(default=None)
    mean: float | None = Field(default=None)
    std: float | None = Field(default=None)


class UQToolInput(BaseModel):
    action: Literal["monte_carlo", "sensitivity", "sobol", "pce", "morris", "propagate"] = Field(
        default="monte_carlo"
    )
    expression: str = Field(..., description="SymPy-compatible expression")
    # monte_carlo/sobol/pce/morris 用 list[VariableSpec];
    # propagate (GUM) 用 dict[str, {value, uncertainty}]
    variables: list[VariableSpec] | dict[str, dict[str, float]] = Field(default_factory=list)
    n_samples: int = Field(default=1000, ge=10)
    seed: int | None = Field(default=None)
    # PCE settings
    order: int = Field(default=3, ge=1, description="Total polynomial order for PCE")
    # Morris screening settings
    r: int = Field(default=10, ge=1, description="Number of Morris trajectories")
    levels: int = Field(default=4, ge=2, description="Grid levels for Morris screening")
    working_dir: str | None = Field(default=None)
    # propagate (GUM) settings
    correlations: dict[str, float] | None = Field(
        default=None,
        description="相关系数, 如 {'a_b': 0.5}; 不传默认独立",
    )
    confidence: float = Field(
        default=0.95, ge=0.0, le=1.0,
        description="置信水平, 用于算扩展不确定度 k 因子",
    )


class UQTool(HuginnTool):
    """Uncertainty quantification and sensitivity analysis for symbolic models."""

    name = "uq_tool"
    category = "sci"
    profile = ToolProfile(phases=frozenset({ResearchPhase.VALIDATION}))
    description = (
        "Uncertainty quantification and sensitivity analysis for symbolic "
        "material-science models. Supports Monte Carlo propagation, local "
        "sensitivity, Sobol global sensitivity indices, polynomial chaos "
        "expansion (PCE), Morris elementary-effects screening, and GUM "
        "first-order uncertainty propagation."
    )
    input_schema = UQToolInput

    async def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        """Async entry — 对齐基类 HuginnTool.call 的 async 契约.

        action 实现本身是同步的, 这里直接返回结果即可 (async 函数里
        return 同步值会被自动包进 coroutine).
        """
        input_data = UQToolInput(**args)
        work_dir = (
            Path(input_data.working_dir) if input_data.working_dir else Path.cwd()
        )
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            if input_data.action == "sensitivity":
                return self._sensitivity(input_data)
            if input_data.action == "sobol":
                return self._sobol(input_data)
            if input_data.action == "pce":
                return self._pce(input_data)
            if input_data.action == "morris":
                return self._morris(input_data)
            if input_data.action == "propagate":
                return self._propagate(input_data)
            return self._monte_carlo(input_data)
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"UQ tool failed: {e}")

    def _parse_expression(self, expression: str, variables: list[VariableSpec]):
        try:
            import sympy as sp
        except ImportError as exc:
            raise RuntimeError(
                "UQ tool requires sympy. Install with: pip install sympy"
            ) from exc

        symbols = {v.name: sp.Symbol(v.name) for v in variables}
        expr = sp.sympify(expression, locals=symbols)
        return sp, expr, symbols

    def _sample_variables(
        self,
        variables: list[VariableSpec],
        n_samples: int,
        seed: int | None,
    ) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        samples: dict[str, np.ndarray] = {}
        for v in variables:
            if v.distribution == "uniform":
                low = v.low if v.low is not None else 0.0
                high = v.high if v.high is not None else 1.0
                samples[v.name] = rng.uniform(low, high, size=n_samples)
            elif v.distribution == "normal":
                mean = v.mean if v.mean is not None else 0.0
                std = v.std if v.std is not None else 1.0
                samples[v.name] = rng.normal(mean, std, size=n_samples)
            elif v.distribution == "lognormal":
                mean = v.mean if v.mean is not None else 0.0
                std = v.std if v.std is not None else 1.0
                samples[v.name] = rng.lognormal(mean, std, size=n_samples)
            else:
                raise ValueError(f"Unsupported distribution: {v.distribution}")
        return samples

    def _monte_carlo(self, args: UQToolInput) -> ToolResult:
        sp, expr, symbols = self._parse_expression(args.expression, args.variables)
        samples = self._sample_variables(args.variables, args.n_samples, args.seed)

        # lambdify for fast vectorized evaluation
        f = sp.lambdify(list(symbols.values()), expr, modules="numpy")
        arrays = [samples[v.name] for v in args.variables]
        values = f(*arrays)

        # Histogram with 20 bins
        counts, bin_edges = np.histogram(values, bins=20)
        hist = {
            "bin_edges": bin_edges.tolist(),
            "counts": counts.tolist(),
        }

        return ToolResult(
            data={
                "method": "monte_carlo",
                "n_samples": args.n_samples,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
                "q05": float(np.quantile(values, 0.05)),
                "q95": float(np.quantile(values, 0.95)),
                "histogram": hist,
                "message": f"Monte Carlo completed with {args.n_samples} samples.",
            },
            success=True,
        )

    def _sensitivity(self, args: UQToolInput) -> ToolResult:
        sp, expr, symbols = self._parse_expression(args.expression, args.variables)

        # Nominal point: use mean for normal/lognormal, midpoint for uniform
        nominal: dict[str, float] = {}
        for v in args.variables:
            if v.distribution == "uniform":
                low = v.low if v.low is not None else 0.0
                high = v.high if v.high is not None else 1.0
                nominal[v.name] = (low + high) / 2.0
            elif v.distribution in ("normal", "lognormal"):
                nominal[v.name] = v.mean if v.mean is not None else 0.0
            else:
                nominal[v.name] = 0.0

        f = sp.lambdify(list(symbols.values()), expr, modules="numpy")
        nominal_values = [nominal[v.name] for v in args.variables]
        nominal_output = float(f(*nominal_values))

        sensitivities: dict[str, dict[str, float]] = {}
        for v in args.variables:
            symbol = symbols[v.name]
            deriv = sp.diff(expr, symbol)
            df = sp.lambdify(list(symbols.values()), deriv, modules="numpy")
            derivative_value = float(df(*nominal_values))

            # Perturbation size: 1% of nominal or 1e-6 if zero
            x0 = nominal[v.name]
            dx = abs(x0) * 0.01 if x0 != 0 else 1e-6
            perturbed = nominal_values.copy()
            idx = args.variables.index(v)
            perturbed[idx] = x0 + dx
            finite_diff = (float(f(*perturbed)) - nominal_output) / dx

            sensitivities[v.name] = {
                "symbolic_derivative": derivative_value,
                "finite_difference": finite_diff,
                "nominal": x0,
                "elasticity": (
                    derivative_value * x0 / nominal_output
                    if nominal_output != 0
                    else None
                ),
            }

        return ToolResult(
            data={
                "method": "sensitivity",
                "nominal_output": nominal_output,
                "nominal_inputs": nominal,
                "sensitivities": sensitivities,
                "message": "Local sensitivity analysis completed.",
            },
            success=True,
        )

    def _sample_matrix(
        self,
        variables: list[VariableSpec],
        n_samples: int,
        seed: int | None,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        k = len(variables)
        matrix = np.empty((n_samples, k))
        for idx, v in enumerate(variables):
            if v.distribution == "uniform":
                low = v.low if v.low is not None else 0.0
                high = v.high if v.high is not None else 1.0
                matrix[:, idx] = rng.uniform(low, high, size=n_samples)
            elif v.distribution == "normal":
                mean = v.mean if v.mean is not None else 0.0
                std = v.std if v.std is not None else 1.0
                matrix[:, idx] = rng.normal(mean, std, size=n_samples)
            elif v.distribution == "lognormal":
                mean = v.mean if v.mean is not None else 0.0
                std = v.std if v.std is not None else 1.0
                matrix[:, idx] = rng.lognormal(mean, std, size=n_samples)
            else:
                raise ValueError(f"Unsupported distribution: {v.distribution}")
        return matrix

    def _sobol(self, args: UQToolInput) -> ToolResult:
        """Sobol global sensitivity indices using Saltelli sampling."""
        sp, expr, symbols = self._parse_expression(args.expression, args.variables)
        f = sp.lambdify(list(symbols.values()), expr, modules="numpy")

        # Two independent sample matrices
        A = self._sample_matrix(args.variables, args.n_samples, args.seed)
        B = self._sample_matrix(
            args.variables,
            args.n_samples,
            (args.seed + 1) if args.seed is not None else None,
        )

        fA = f(*[A[:, i] for i in range(A.shape[1])])
        fB = f(*[B[:, i] for i in range(B.shape[1])])

        total_var = float(np.var(np.concatenate([fA, fB]), ddof=1))
        if total_var == 0:
            return ToolResult(
                data=None,
                success=False,
                error="Total variance is zero; Sobol indices are undefined.",
            )

        s1: dict[str, float] = {}
        st: dict[str, float] = {}
        for idx, v in enumerate(args.variables):
            AB = A.copy()
            AB[:, idx] = B[:, idx]
            fAB = f(*[AB[:, j] for j in range(AB.shape[1])])

            first = float(np.mean(fB * (fAB - fA)) / total_var)
            total = float(0.5 * np.mean((fA - fAB) ** 2) / total_var)
            s1[v.name] = max(0.0, min(1.0, first))
            st[v.name] = max(0.0, min(1.0, total))

        return ToolResult(
            data={
                "method": "sobol",
                "n_samples": args.n_samples,
                "total_variance": total_var,
                "S1": s1,
                "ST": st,
                "message": "Sobol global sensitivity analysis completed.",
            },
            success=True,
        )

    # ── Polynomial Chaos Expansion ─────────────────────────────────────

    def _pce(self, args: UQToolInput) -> ToolResult:
        """Polynomial chaos expansion with Legendre (uniform) / Hermite (normal) bases."""
        _sp, expr, symbols = self._parse_expression(args.expression, args.variables)
        f = _sp.lambdify(list(symbols.values()), expr, modules="numpy")

        try:
            import chaospy  # noqa: F401
        except ImportError:
            return self._pce_numpy(args, f)
        return self._pce_chaospy(args, f)

    def _pce_numpy(self, args: UQToolInput, f) -> ToolResult:
        import math
        from itertools import product

        from numpy.polynomial.hermite_e import hermeval
        from numpy.polynomial.legendre import legval

        n_inputs = len(args.variables)
        order = args.order
        n_samples = (order + 1) ** n_inputs

        # Draw samples in the original parameter space, then map each
        # coordinate to its standard distribution (U[-1,1] or N(0,1)) so the
        # orthogonal polynomials are evaluated on the right support.
        X = self._sample_matrix(args.variables, n_samples, args.seed)
        y = np.array([float(f(*X[i])) for i in range(n_samples)])

        std_X = np.empty_like(X)
        poly_types: list[str] = []
        for j, v in enumerate(args.variables):
            if v.distribution == "normal":
                mean = v.mean if v.mean is not None else 0.0
                std = v.std if v.std is not None else 1.0
                std = std if std > 0 else 1.0
                std_X[:, j] = (X[:, j] - mean) / std
                poly_types.append("hermite")
            else:
                # uniform (lognormal falls back to Legendre on the box too)
                low = v.low if v.low is not None else 0.0
                high = v.high if v.high is not None else 1.0
                if high == low:
                    std_X[:, j] = 0.0
                else:
                    std_X[:, j] = 2.0 * (X[:, j] - low) / (high - low) - 1.0
                poly_types.append("legendre")

        # Total-order multi-index set: {alpha in N^n : sum(alpha) <= order}
        indices = [
            np.array(alpha, dtype=int)
            for alpha in product(range(order + 1), repeat=n_inputs)
            if sum(alpha) <= order
        ]
        n_basis = len(indices)

        # Assemble the design matrix from tensor products of 1D orthonormal
        # polynomials. With orthonormal basis the constant term is the mean
        # and the sum of squared higher-order coeffs is the variance.
        Psi = np.empty((n_samples, n_basis))
        for col, alpha in enumerate(indices):
            vals = np.ones(n_samples)
            for j in range(n_inputs):
                deg = int(alpha[j])
                if deg == 0:
                    continue
                if poly_types[j] == "hermite":
                    c = np.zeros(deg + 1)
                    c[deg] = 1.0
                    vals = vals * (hermeval(std_X[:, j], c) / math.sqrt(math.factorial(deg)))
                else:
                    c = np.zeros(deg + 1)
                    c[deg] = 1.0
                    vals = vals * (legval(std_X[:, j], c) * np.sqrt(2 * deg + 1))
            Psi[:, col] = vals

        coeffs, *_ = np.linalg.lstsq(Psi, y, rcond=None)

        mean = float(coeffs[0])
        variance = float(np.sum(coeffs[1:] ** 2))
        variance = max(variance, 0.0)

        # First-order Sobol-like indices: variance contributed by each input
        # acting alone (alpha nonzero only in that component).
        sobol: dict[str, float] = {}
        denom = variance if variance > 0 else 1.0
        for j, v in enumerate(args.variables):
            var_j = 0.0
            for idx, alpha in enumerate(indices):
                if idx == 0:
                    continue
                if alpha[j] > 0 and int(np.sum(alpha)) == int(alpha[j]):
                    var_j += coeffs[idx] ** 2
            sobol[v.name] = float(var_j / denom) if variance > 0 else 0.0

        return ToolResult(
            data={
                "method": "pce",
                "backend": "numpy",
                "order": order,
                "n_samples": n_samples,
                "n_basis": n_basis,
                "coefficients": coeffs.tolist(),
                "mean": mean,
                "variance": variance,
                "std": float(np.sqrt(variance)),
                "sobol_first": sobol,
                "message": f"PCE fitted at order {order} ({n_basis} basis terms).",
            },
            success=True,
        )

    def _pce_chaospy(self, args: UQToolInput, f) -> ToolResult:
        import chaospy

        dists = []
        for v in args.variables:
            if v.distribution == "normal":
                mean = v.mean if v.mean is not None else 0.0
                std = v.std if v.std is not None else 1.0
                dists.append(chaospy.Normal(mean, std))
            elif v.distribution == "lognormal":
                mean = v.mean if v.mean is not None else 0.0
                std = v.std if v.std is not None else 1.0
                dists.append(chaospy.LogNormal(mean, std))
            else:
                low = v.low if v.low is not None else 0.0
                high = v.high if v.high is not None else 1.0
                dists.append(chaospy.Uniform(low, high))

        joint = chaospy.J(*dists)
        n_samples = (args.order + 1) ** len(args.variables)
        samples = np.asarray(joint.sample(n_samples))
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)

        evals = np.array([float(f(*samples[:, i])) for i in range(samples.shape[1])])

        expansion = chaospy.generate_expansion(args.order, joint, norm=True)
        model = chaospy.fit_regression(expansion, samples, evals)

        mean = float(chaospy.E(model, joint))
        variance = float(chaospy.Var(model, joint))
        sens = chaospy.Sens_m(model, joint)

        sobol: dict[str, float] = {}
        for i, v in enumerate(args.variables):
            val = sens[i] if hasattr(sens, "__getitem__") else sens.get(v.name, 0.0)
            sobol[v.name] = float(val)

        return ToolResult(
            data={
                "method": "pce",
                "backend": "chaospy",
                "order": args.order,
                "n_samples": int(samples.shape[1]),
                "mean": mean,
                "variance": variance,
                "std": float(np.sqrt(max(variance, 0.0))),
                "sobol_first": sobol,
                "message": f"PCE fitted at order {args.order} (chaospy backend).",
            },
            success=True,
        )

    # ── Morris elementary-effects screening ────────────────────────────

    def _morris(self, args: UQToolInput) -> ToolResult:
        """Morris elementary-effects screening over a box defined by low/high."""
        _sp, expr, symbols = self._parse_expression(args.expression, args.variables)
        f = _sp.lambdify(list(symbols.values()), expr, modules="numpy")

        k = len(args.variables)
        p = args.levels
        r = args.r
        rng = np.random.default_rng(args.seed)

        lows = np.array(
            [v.low if v.low is not None else 0.0 for v in args.variables], dtype=float
        )
        highs = np.array(
            [v.high if v.high is not None else 1.0 for v in args.variables], dtype=float
        )

        # Grid step in [0, 1]^k space.
        delta = p / (2.0 * (p - 1))

        ee_samples: list[list[float]] = [[] for _ in range(k)]
        for _ in range(r):
            Bstar = self._morris_trajectory(k, p, delta, rng)
            X = lows + Bstar * (highs - lows)
            y_vals = np.array([float(f(*X[i])) for i in range(k + 1)])

            # Each consecutive pair of rows differs in exactly one coordinate;
            # attribute the elementary effect to that variable.
            for i in range(k):
                diff = Bstar[i + 1] - Bstar[i]
                changed = int(np.argmax(np.abs(diff)))
                step = diff[changed]
                if abs(step) < 1e-12:
                    continue
                actual_step = step * (highs[changed] - lows[changed])
                ee = (y_vals[i + 1] - y_vals[i]) / actual_step
                ee_samples[changed].append(float(ee))

        results: dict[str, dict[str, float]] = {}
        for j, v in enumerate(args.variables):
            arr = np.array(ee_samples[j])
            if arr.size == 0:
                results[v.name] = {"mu": 0.0, "mu_star": 0.0, "sigma": 0.0}
                continue
            results[v.name] = {
                "mu": float(np.mean(arr)),
                "mu_star": float(np.mean(np.abs(arr))),
                "sigma": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            }

        return ToolResult(
            data={
                "method": "morris",
                "r": r,
                "levels": p,
                "elementary_effects": results,
                "message": f"Morris screening completed with {r} trajectories.",
            },
            success=True,
        )

    @staticmethod
    def _morris_trajectory(
        k: int, p: int, delta: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Build one (k+1) x k Morris trajectory in [0, 1]^k space.

        Uses the standard sampling matrix construction (Saltelli et al.,
        Global Sensitivity Analysis: The Primer) with a random base point,
        random sign diagonal, and a random column permutation.
        """
        grid = np.linspace(0.0, 1.0, p)
        # Base point must leave room for a +/- delta step inside [0, 1].
        valid = grid[grid <= 1.0 - delta + 1e-9]
        x_star = np.array([rng.choice(valid) for _ in range(k)])

        D = np.diag(rng.choice([-1.0, 1.0], size=k))
        # Strictly lower-triangular matrix of ones.
        B = np.tril(np.ones((k + 1, k)), -1)
        J = np.ones((k + 1, k))

        x_star_mat = np.tile(x_star, (k + 1, 1))
        Bstar = x_star_mat + (delta / 2.0) * ((2.0 * B - J) @ D + J)

        # Randomize the order in which variables are bumped.
        perm = rng.permutation(k)
        Bstar = Bstar[:, perm]
        return np.clip(Bstar, 0.0, 1.0)

    # ── GUM first-order uncertainty propagation ────────────────────────

    def _propagate(self, args: UQToolInput) -> ToolResult:
        """GUM 法线性误差传播.

        u_c² = Σ_i (∂f/∂x_i)² u_xi²             (无相关)
        u_c² = Σ_i Σ_j (∂f/∂x_i)(∂f/∂x_j) u_xi u_xj r_ij   (有相关)

        偏导用 sympy 算, 在标称点求值. k 因子按 confidence 近似:
        95% → 2, 99% → 3, 其它用正态分位近似.
        """
        try:
            import sympy as sp
        except ImportError as exc:
            raise RuntimeError(
                "UQ tool requires sympy. Install with: pip install sympy"
            ) from exc

        # variables 在 propagate 模式下是 dict[str, {value, uncertainty}]
        var_dict = args.variables
        if not isinstance(var_dict, dict):
            return ToolResult(
                data=None, success=False,
                error="propagate action requires variables as "
                      "dict[str, {value, uncertainty}]",
            )
        if not var_dict:
            return ToolResult(
                data=None, success=False,
                error="propagate action requires at least one variable",
            )

        # 拆出标称值和不确定度
        names = list(var_dict.keys())
        symbols = {n: sp.Symbol(n) for n in names}
        nominal: dict[str, float] = {}
        uncertainties: dict[str, float] = {}
        for n in names:
            spec = var_dict[n]
            nominal[n] = float(spec.get("value", 0.0))
            u = spec.get("uncertainty")
            uncertainties[n] = float(u) if u is not None else 0.0

        expr = sp.sympify(args.expression, locals=symbols)

        # 标称值下的表达式结果
        f = sp.lambdify([symbols[n] for n in names], expr, modules="numpy")
        nominal_value = float(f(*[nominal[n] for n in names]))

        # 偏导 ∂f/∂x_i 在标称点求值
        sens: dict[str, float] = {}
        for n in names:
            deriv = sp.diff(expr, symbols[n])
            df = sp.lambdify([symbols[n] for n in names], deriv, modules="numpy")
            sens[n] = float(df(*[nominal[n] for n in names]))

        # 解析相关系数, 支持 "a_b" / "a-b" / "ab" 这种 key
        corr: dict[tuple[str, str], float] = {}
        if args.correlations:
            for key, r_val in args.correlations.items():
                pair = self._parse_correlation_key(key, names)
                if pair is not None:
                    corr[pair] = float(r_val)

        # 合成标准不确定度 u_c
        # 无相关: Σ (∂f/∂x_i)² u_i²
        # 有相关: 再加 Σ_{i≠j} (∂f/∂x_i)(∂f/∂x_j) u_i u_j r_ij
        uc_sq = 0.0
        for n in names:
            uc_sq += (sens[n] * uncertainties[n]) ** 2
        # 相关项 (i<j, 系数 2 因为对称)
        cross_terms: list[str] = []
        for i, ni in enumerate(names):
            for nj in names[i + 1:]:
                r_ij = corr.get((ni, nj), corr.get((nj, ni), 0.0))
                if r_ij == 0.0:
                    continue
                term = 2.0 * sens[ni] * sens[nj] * uncertainties[ni] * uncertainties[nj] * r_ij
                uc_sq += term
                cross_terms.append(f"2*({sens[ni]})*({sens[nj]})*{uncertainties[ni]}*{uncertainties[nj]}*{r_ij}")
        uc_sq = max(uc_sq, 0.0)  # 数值噪声可能搞出负的
        uc = float(np.sqrt(uc_sq))

        # 各变量对 u_c² 的贡献百分比
        contribution: dict[str, float] = {}
        for n in names:
            contrib_sq = (sens[n] * uncertainties[n]) ** 2
            contribution[n] = (contrib_sq / uc_sq * 100.0) if uc_sq > 0 else 0.0

        dominant = max(contribution, key=contribution.get) if contribution else None

        # 扩展不确定度 U = k * u_c, k 按 confidence 近似
        k_factor = self._coverage_factor(args.confidence)
        expanded = k_factor * uc

        # 用的公式字符串, 方便用户确认
        if cross_terms:
            formula = (
                "u_c² = Σ (∂f/∂x_i)² u_xi² + Σ_{i≠j} (∂f/∂x_i)(∂f/∂x_j) u_xi u_xj r_ij"
            )
        else:
            formula = "u_c² = Σ (∂f/∂x_i)² u_xi²  (变量独立, 无相关项)"

        return ToolResult(
            data={
                "method": "propagate",
                "nominal_value": nominal_value,
                "combined_uncertainty": uc,
                "expanded_uncertainty": expanded,
                "k_factor": k_factor,
                "confidence": args.confidence,
                "sensitivity_coefficients": sens,
                "contribution_percent": contribution,
                "dominant_variable": dominant,
                "formula": formula,
                "correlations_used": {f"{a}_{b}": r for (a, b), r in corr.items()} if corr else None,
                "message": f"GUM propagation: nominal={nominal_value:.6g}, u_c={uc:.6g}, U={expanded:.6g} (k={k_factor}).",
            },
            success=True,
        )

    @staticmethod
    def _parse_correlation_key(
        key: str, names: list[str]
    ) -> tuple[str, str] | None:
        """把 'a_b' / 'a-b' / 'ab' 这种 key 拆成 (a, b), 拆不开返回 None.

        优先按分隔符拆, 拆不出来再试着在已知变量名里两两组合匹配.
        """
        for sep in ("_", "-", " ", ",", ":"):
            if sep in key:
                parts = key.split(sep, 1)
                if len(parts) == 2:
                    a, b = parts[0].strip(), parts[1].strip()
                    if a in names and b in names:
                        return (a, b)
        # 没分隔符, 试着直接匹配两个变量名拼接
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if key == a + b or key == b + a:
                    return (a, b)
        return None

    @staticmethod
    def _coverage_factor(confidence: float) -> float:
        """按置信水平给覆盖因子 k. 95% → 2, 99% → 3, 中间值线性插值近似."""
        # GUM 默认 k=2 (≈95.45%), 这里给几个常见档, 其它线性插
        table = [
            (0.90, 1.645),
            (0.95, 1.960),
            (0.99, 2.576),
            (0.999, 3.291),
        ]
        if confidence <= table[0][0]:
            return table[0][1]
        if confidence >= table[-1][0]:
            return table[-1][1]
        for i in range(len(table) - 1):
            c_low, k_low = table[i]
            c_high, k_high = table[i + 1]
            if c_low <= confidence <= c_high:
                # 线性插值
                t = (confidence - c_low) / (c_high - c_low)
                return k_low + t * (k_high - k_low)
        return 2.0
