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

from matsci_agent.tools.base import MatSciTool
from matsci_agent.types import ToolResult, ToolContext


class VariableSpec(BaseModel):
    name: str = Field(..., description="Variable name")
    distribution: Literal["uniform", "normal", "lognormal"] = Field(default="uniform")
    low: float | None = Field(default=None)
    high: float | None = Field(default=None)
    mean: float | None = Field(default=None)
    std: float | None = Field(default=None)


class UQToolInput(BaseModel):
    action: Literal["monte_carlo", "sensitivity", "sobol"] = Field(default="monte_carlo")
    expression: str = Field(..., description="SymPy-compatible expression")
    variables: list[VariableSpec] = Field(default_factory=list)
    n_samples: int = Field(default=1000, ge=10)
    seed: int | None = Field(default=None)
    working_dir: str | None = Field(default=None)


class UQTool(MatSciTool):
    """Uncertainty quantification and sensitivity analysis for symbolic models."""

    name = "uq_tool"
    description = (
        "Uncertainty quantification and sensitivity analysis for symbolic "
        "material-science models. Supports Monte Carlo propagation, local "
        "sensitivity, and Sobol global sensitivity indices."
    )
    input_schema = UQToolInput

    def call(self, args: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        input_data = UQToolInput(**args)
        work_dir = Path(input_data.working_dir) if input_data.working_dir else Path.cwd()
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            if input_data.action == "sensitivity":
                return self._sensitivity(input_data)
            if input_data.action == "sobol":
                return self._sobol(input_data)
            return self._monte_carlo(input_data)
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"UQ tool failed: {e}")

    def _parse_expression(self, expression: str, variables: list[VariableSpec]):
        try:
            import sympy as sp
        except ImportError as exc:
            raise RuntimeError("UQ tool requires sympy. Install with: pip install sympy") from exc

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
                "elasticity": derivative_value * x0 / nominal_output if nominal_output != 0 else None,
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
        B = self._sample_matrix(args.variables, args.n_samples, (args.seed + 1) if args.seed is not None else None)

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
