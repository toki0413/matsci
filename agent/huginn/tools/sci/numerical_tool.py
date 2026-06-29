"""Numerical solver tool — unified interface to scipy/numpy solvers.

Exposes common numerical methods to the agent through a single tool:
ODE integration, optimization, root finding, curve fitting, numerical
integration, linear systems, eigenvalues, interpolation, and FFT.

The tool accepts string representations of Python functions (via safe_eval)
so the LLM can define equations without writing files.
"""

from __future__ import annotations

import re
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from huginn.security.math_eval import safe_math_eval
from huginn.tools.base import HuginnTool, ToolProfile
from huginn.types import ToolContext, ToolResult, ValidationResult


class NumericalToolInput(BaseModel):
    action: Literal[
        "ode",
        "minimize",
        "constrained_minimize",
        "root",
        "curve_fit",
        "integrate",
        "linear_solve",
        "eigenvalues",
        "svd",
        "matrix_exp",
        "interpolate",
        "fft",
    ] = Field(...)

    # Function/expression strings
    func: str | None = Field(
        default=None,
        description="Python expression or function body. Use 'x' for scalar, 'X' for vector.",
    )
    func_fit: str | None = Field(
        default=None,
        description="For curve_fit: function signature f(x, a, b, ...).",
    )

    # Initial values / data
    x0: list[float] | float | None = Field(default=None)
    y0: list[float] | None = Field(default=None)
    t_span: list[float] | None = Field(default=None)
    t_eval: list[float] | None = Field(default=None)
    xdata: list[float] | None = Field(default=None)
    ydata: list[float] | None = Field(default=None)
    a: float | None = Field(default=None, description="Integration lower bound")
    b: float | None = Field(default=None, description="Integration upper bound")

    # Matrix / linear system
    A: list[list[float]] | None = Field(default=None)
    b_vec: list[float] | None = Field(default=None)

    # Constrained optimization
    bounds: list[list[float]] | None = Field(
        default=None,
        description="Box bounds [[lb, ub], ...] for constrained_minimize.",
    )
    constraints: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Constraints for constrained_minimize as scipy-style dicts: "
            "{'type': 'ineq'|'eq', 'fun': '<expr of X>'}."
        ),
    )

    # Interpolation
    x: list[float] | None = Field(default=None)
    y: list[float] | None = Field(default=None)
    x_new: list[float] | None = Field(default=None)
    kind: str = Field(default="linear")

    # FFT / signal
    data: list[float] | None = Field(default=None)
    sampling_rate: float = Field(default=1.0)

    # Solver options
    method: str | None = Field(default=None)
    tol: float | None = Field(default=None)
    maxiter: int | None = Field(default=None)
    n_points: int = Field(default=1000)
    args: list[float] = Field(default_factory=list)


def _normalize_expr(expr: str | None, default: str) -> str:
    """Normalize an expression string: use default if empty, replace ^ with **."""
    expr = (expr or default).strip()
    return expr.replace("^", "**")


def _compile_scalar_func(expr: str, args: tuple[float, ...] = ()) -> Any:
    """Compile a scalar expression of x into a callable."""
    expr = _normalize_expr(expr, "x")

    def fn(x: float, *_args: float) -> float:
        names: dict[str, Any] = {"x": x}
        names.update({f"a{i}": v for i, v in enumerate(args)})
        return float(safe_math_eval(expr, names))

    return fn


def _compile_ode_func(expr: str) -> Any:
    """Compile dy/dt expression of t and y (y is a list/array)."""
    expr = _normalize_expr(expr, "-y[0]")

    def fn(t: float, y: list[float]) -> list[float]:
        names: dict[str, Any] = {"t": t, "y": y}
        result = safe_math_eval(expr, names)
        if not isinstance(result, (list, tuple, np.ndarray)):
            result = [float(result)]
        return list(result)

    return fn


def _compile_vector_func(expr: str) -> Any:
    """Compile a vector expression of X (list) into a callable returning array."""
    expr = _normalize_expr(expr, "X[0]**2")

    def fn(X: list[float]) -> np.ndarray:
        names: dict[str, Any] = {
            "X": X,
            "x0": X[0],
            "x1": X[1] if len(X) > 1 else 0,
        }
        result = safe_math_eval(expr, names)
        return np.asarray(result, dtype=float)

    return fn


def _compile_constraint_func(expr: str) -> Any:
    """Compile a scalar constraint expression of X into a callable returning float.

    Used by constrained_minimize so users can pass scipy-style constraints as
    expression strings (e.g. "X[0] + X[1] - 1" for an inequality >= 0).
    """
    expr = _normalize_expr(expr, "0")

    def fn(X: list[float]) -> float:
        names: dict[str, Any] = {"X": list(X)}
        for i in range(len(X)):
            names[f"x{i}"] = X[i]
        return float(safe_math_eval(expr, names))

    return fn


def _fit_param_count(expr: str) -> int:
    """Return the number of fit parameters referenced in the expression."""
    indices = [int(m.group(1)) for m in re.finditer(r"\ba(\d+)\b", expr)]
    return max(indices) + 1 if indices else 1


def _compile_fit_func(expr: str) -> Any:
    """Compile f(x, a0, a1, ...) for curve fitting."""
    expr = _normalize_expr(expr, "a0*x + a1")

    def fn(x: np.ndarray, *params: float) -> np.ndarray:
        names: dict[str, Any] = {"x": x}
        for i, p in enumerate(params):
            names[f"a{i}"] = p
        result = safe_math_eval(expr, names)
        return np.asarray(result, dtype=float)

    return fn


class NumericalTool(HuginnTool):
    """Unified numerical solver: ODEs, optimization, roots, fitting, FFT, etc."""

    name = "numerical_tool"
    category = "sci"
    profile = ToolProfile(cost_tier="light")
    description = (
        "Solve numerical problems: integrate ODEs, minimize functions, find roots, "
        "fit curves, integrate functions, solve linear systems, compute eigenvalues, "
        "interpolate data, and compute FFTs. Pass expressions as strings using 'x', "
        "'X' (vector), 't', and 'y'."
    )
    input_schema = NumericalToolInput

    def is_read_only(self, args: NumericalToolInput) -> bool:
        return True

    async def validate_input(
        self, args: NumericalToolInput, context: ToolContext | None = None
    ) -> ValidationResult:
        if args.action in ("ode", "minimize", "constrained_minimize", "root", "integrate") and not args.func:
            return ValidationResult(
                result=False,
                message=f"{args.action} requires a func expression.",
            )
        if args.action == "constrained_minimize" and args.x0 is None:
            return ValidationResult(
                result=False,
                message="constrained_minimize requires x0.",
            )
        if args.action == "curve_fit" and (not args.func_fit or args.xdata is None or args.ydata is None):
            return ValidationResult(
                result=False,
                message="curve_fit requires func_fit, xdata, and ydata.",
            )
        if args.action == "linear_solve" and (args.A is None or args.b_vec is None):
            return ValidationResult(
                result=False,
                message="linear_solve requires A and b.",
            )
        if args.action in ("eigenvalues", "svd", "matrix_exp") and args.A is None:
            return ValidationResult(
                result=False,
                message=f"{args.action} requires A.",
            )
        if args.action == "interpolate" and (args.x is None or args.y is None):
            return ValidationResult(
                result=False,
                message="interpolate requires x and y arrays.",
            )
        if args.action == "fft" and args.data is None:
            return ValidationResult(result=False, message="fft requires data array.")
        return ValidationResult(result=True)

    async def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = NumericalToolInput(**args)

        try:
            # Actions that don't route through huginn.utils.numerical.
            if input_data.action == "svd":
                return self._svd(input_data)
            if input_data.action == "matrix_exp":
                return self._matrix_exp(input_data)
            if input_data.action == "constrained_minimize":
                return self._constrained_minimize(input_data)

            from huginn.utils.numerical import (
                eigenvalues,
                fft,
                find_root,
                fit_curve,
                integrate,
                interpolate_1d,
                minimize_func,
                solve_linear,
                solve_ode,
            )

            if input_data.action == "ode":
                if input_data.t_span is None or input_data.y0 is None:
                    return ToolResult(data=None, success=False, error="ode requires t_span and y0")
                func = _compile_ode_func(input_data.func or "-y[0]")
                result = solve_ode(
                    func,
                    tuple(input_data.t_span),  # type: ignore[arg-type]
                    input_data.y0,
                    method=input_data.method or "RK45",
                    t_eval=input_data.t_eval,
                    rtol=input_data.tol or 1e-6,
                )

            elif input_data.action == "minimize":
                if input_data.x0 is None:
                    return ToolResult(data=None, success=False, error="minimize requires x0")
                func = _compile_vector_func(input_data.func or "X[0]**2")
                result = minimize_func(
                    func,
                    input_data.x0,
                    method=input_data.method or "BFGS",
                    tol=input_data.tol or 1e-6,
                )

            elif input_data.action == "root":
                if input_data.x0 is None:
                    return ToolResult(data=None, success=False, error="root requires x0")
                func = _compile_scalar_func(input_data.func or "x**2 - 4")
                result = find_root(
                    func,
                    input_data.x0,
                    method=input_data.method or "hybr",
                    tol=input_data.tol or 1e-8,
                    maxiter=input_data.maxiter or 100,
                )

            elif input_data.action == "curve_fit":
                fit_expr = input_data.func_fit or "a0*x + a1"
                func = _compile_fit_func(fit_expr)
                result = fit_curve(
                    func,
                    np.asarray(input_data.xdata, dtype=float),
                    np.asarray(input_data.ydata, dtype=float),
                    p0=[1.0] * _fit_param_count(fit_expr),
                )

            elif input_data.action == "integrate":
                if input_data.a is None or input_data.b is None:
                    return ToolResult(data=None, success=False, error="integrate requires a and b")
                func = _compile_scalar_func(input_data.func or "x**2")
                result = integrate(
                    func,
                    input_data.a,
                    input_data.b,
                    method=input_data.method or "quad",
                    n_points=input_data.n_points,
                )

            elif input_data.action == "linear_solve":
                result = solve_linear(
                    np.asarray(input_data.A, dtype=float),
                    np.asarray(input_data.b_vec, dtype=float),
                )

            elif input_data.action == "eigenvalues":
                if input_data.A is None:
                    return ToolResult(data=None, success=False, error="eigenvalues requires A")
                result = eigenvalues(np.asarray(input_data.A, dtype=float))

            elif input_data.action == "interpolate":
                result = interpolate_1d(
                    np.asarray(input_data.x, dtype=float),
                    np.asarray(input_data.y, dtype=float),
                    kind=input_data.kind,
                    x_new=np.asarray(input_data.x_new, dtype=float) if input_data.x_new else None,
                )

            elif input_data.action == "fft":
                result = fft(
                    np.asarray(input_data.data, dtype=float),
                    sampling_rate=input_data.sampling_rate,
                )

            else:
                return ToolResult(
                    data=None,
                    success=False,
                    error=f"Unknown action: {input_data.action}",
                )

            if result.success:
                return ToolResult(data=result.to_dict(), success=True)
            return ToolResult(
                data=result.to_dict(),
                success=False,
                error=f"Numerical solver failed: {result.message}",
            )

        except Exception as e:
            return ToolResult(
                data=None,
                success=False,
                error=f"Numerical solver failed: {e}",
            )

    def _svd(self, args: NumericalToolInput) -> ToolResult:
        if args.A is None:
            return ToolResult(data=None, success=False, error="svd requires A")
        A = np.asarray(args.A, dtype=float)
        try:
            U, S, Vh = np.linalg.svd(A, full_matrices=False)
        except np.linalg.LinAlgError as e:
            return ToolResult(data=None, success=False, error=f"SVD failed: {e}")
        return ToolResult(
            data={
                "U": U.tolist(),
                "S": S.tolist(),
                "Vh": Vh.tolist(),
                "message": f"SVD completed (rank-{len(S)} approximation).",
            },
            success=True,
        )

    def _matrix_exp(self, args: NumericalToolInput) -> ToolResult:
        if args.A is None:
            return ToolResult(data=None, success=False, error="matrix_exp requires A")
        A = np.asarray(args.A, dtype=float)
        try:
            from scipy.linalg import expm

            result = expm(A)
        except ImportError:
            return ToolResult(
                data=None,
                success=False,
                error="scipy is required for matrix_exp.",
            )
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"matrix_exp failed: {e}")
        return ToolResult(
            data={
                "expm": result.tolist(),
                "message": "Matrix exponential computed.",
            },
            success=True,
        )

    def _constrained_minimize(self, args: NumericalToolInput) -> ToolResult:
        if args.x0 is None:
            return ToolResult(
                data=None, success=False, error="constrained_minimize requires x0"
            )
        func = _compile_vector_func(args.func or "X[0]**2")
        x0 = args.x0 if isinstance(args.x0, list) else [args.x0]

        bounds = None
        if args.bounds:
            bounds = [(float(b[0]), float(b[1])) for b in args.bounds]

        constraints = None
        if args.constraints:
            constraints = []
            for c in args.constraints:
                ctype = c.get("type", "ineq")
                cfunc = _compile_constraint_func(c.get("fun", "0"))
                constraints.append({"type": ctype, "fun": cfunc})

        method = args.method or "SLSQP"
        from huginn.utils.numerical import minimize_func

        result = minimize_func(
            func,
            x0,
            method=method,
            bounds=bounds,
            constraints=constraints,
            tol=args.tol or 1e-6,
        )
        if result.success:
            return ToolResult(data=result.to_dict(), success=True)
        return ToolResult(
            data=result.to_dict(),
            success=False,
            error=f"Constrained minimization failed: {result.message}",
        )
