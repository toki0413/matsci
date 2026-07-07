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
        "lp",
        "milp",
        "convex",
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

    # ── LP / MILP ──
    c_vec: list[float] | None = Field(
        default=None,
        description="Cost coefficients for lp/milp (objective: minimize c@x).",
    )
    A_ub: list[list[float]] | None = Field(
        default=None, description="Inequality constraint matrix for lp (A_ub @ x <= b_ub)."
    )
    b_ub: list[float] | None = Field(default=None)
    A_eq: list[list[float]] | None = Field(
        default=None, description="Equality constraint matrix for lp (A_eq @ x = b_eq)."
    )
    b_eq: list[float] | None = Field(default=None)
    integrality: list[int] | None = Field(
        default=None,
        description=(
            "For milp: per-variable integrality (1=integer, 0=continuous). "
            "If omitted, all variables treated as continuous (i.e. LP)."
        ),
    )
    var_bounds: list[list[float]] | None = Field(
        default=None,
        description="Per-variable bounds [[lb, ub], ...] for lp/milp. Default [0, inf).",
    )

    # ── Convex optimization (cvxpy) ──
    convex_problem: str | None = Field(
        default=None,
        description=(
            "For convex: cvxpy problem description. "
            "Format: 'minimize' or 'maximize' followed by objective expression, "
            "then 'subject to' and constraint expressions (one per line). "
            "Variables: x[0], x[1], ... or named vectors."
        ),
    )

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
        if args.action in ("lp", "milp") and args.c_vec is None:
            return ValidationResult(
                result=False, message=f"{args.action} requires c_vec (cost coefficients)."
            )
        if args.action == "convex" and not args.convex_problem:
            return ValidationResult(
                result=False, message="convex requires convex_problem (problem description)."
            )
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
            if input_data.action == "lp":
                return self._lp(input_data)
            if input_data.action == "milp":
                return self._milp(input_data)
            if input_data.action == "convex":
                return self._convex(input_data)

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

    # ── Linear Programming ──────────────────────────────────────────

    def _lp(self, args: NumericalToolInput) -> ToolResult:
        """Solve: minimize c@x  s.t.  A_ub@x <= b_ub, A_eq@x = b_eq, lb <= x <= ub."""
        try:
            from scipy.optimize import linprog
        except ImportError:
            return ToolResult(
                data=None, success=False,
                error="scipy is required for linear programming.",
            )

        c = np.asarray(args.c_vec, dtype=float)
        A_ub = np.asarray(args.A_ub, dtype=float) if args.A_ub else None
        b_ub = np.asarray(args.b_ub, dtype=float) if args.b_ub else None
        A_eq = np.asarray(args.A_eq, dtype=float) if args.A_eq else None
        b_eq = np.asarray(args.b_eq, dtype=float) if args.b_eq else None

        bounds = self._parse_var_bounds(args.var_bounds, n_vars=len(c))

        try:
            res = linprog(
                c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                bounds=bounds, method=args.method or "highs",
            )
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"LP failed: {e}")

        if res.success:
            return ToolResult(
                data={
                    "x": res.x.tolist(),
                    "fun": float(res.fun),
                    "method": "linprog (HiGHS)" if args.method is None else args.method,
                    "message": res.message or "Optimization terminated successfully.",
                    "n_iter": getattr(res, "nit", None),
                },
                success=True,
            )
        return ToolResult(
            data={"message": res.message or "LP failed", "fun": float(res.fun) if res.fun is not None else None},
            success=False,
            error=f"LP failed: {res.message}",
        )

    # ── Mixed-Integer Linear Programming ────────────────────────────

    def _milp(self, args: NumericalToolInput) -> ToolResult:
        """Solve MILP: minimize c@x with linear constraints and integer variables."""
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
        except ImportError:
            return self._milp_fallback_pulp(args)

        c = np.asarray(args.c_vec, dtype=float)
        n = len(c)

        constraints = []
        if args.A_ub is not None and args.b_ub is not None:
            # A_ub @ x <= b_ub  →  LinearConstraint(A_ub, -inf, b_ub)
            A_ub = np.asarray(args.A_ub, dtype=float)
            b_ub = np.asarray(args.b_ub, dtype=float)
            constraints.append(LinearConstraint(A_ub, -np.inf, b_ub))
        if args.A_eq is not None and args.b_eq is not None:
            A_eq = np.asarray(args.A_eq, dtype=float)
            b_eq = np.asarray(args.b_eq, dtype=float)
            constraints.append(LinearConstraint(A_eq, b_eq, b_eq))

        if args.var_bounds:
            lb = [b[0] for b in args.var_bounds]
            ub = [b[1] for b in args.var_bounds]
            bounds = Bounds(lb=lb, ub=ub)
        else:
            bounds = Bounds(lb=0, ub=np.inf)

        integrality = np.asarray(args.integrality, dtype=int) if args.integrality else None

        try:
            res = milp(
                c=c,
                constraints=constraints or None,
                integrality=integrality,
                bounds=bounds,
            )
        except Exception as e:
            return self._milp_fallback_pulp(args)

        if res.success:
            return ToolResult(
                data={
                    "x": res.x.tolist(),
                    "fun": float(res.fun),
                    "method": "milp (HiGHS)",
                    "message": res.message or "MILP solved.",
                },
                success=True,
            )
        return ToolResult(
            data={"message": res.message or "MILP failed"},
            success=False,
            error=f"MILP failed: {res.message}",
        )

    def _milp_fallback_pulp(self, args: NumericalToolInput) -> ToolResult:
        """Fallback MILP solver using PuLP when scipy.optimize.milp is unavailable."""
        try:
            import pulp
        except ImportError:
            return ToolResult(
                data=None, success=False,
                error=(
                    "MILP requires scipy>=1.9 (milp) or pulp. "
                    "Install one: pip install scipy --upgrade  OR  pip install pulp"
                ),
            )

        c = args.c_vec or []
        n = len(c)
        prob = pulp.LpProblem("milp", pulp.LpMinimize)
        xs = [pulp.LpVariable(f"x{i}", lowBound=0, cat="Integer" if
              (args.integrality and args.integrality[i]) else "Continuous")
              for i in range(n)]

        prob += pulp.lpSum(c[i] * xs[i] for i in range(n))

        if args.A_ub and args.b_ub:
            for row, bi in zip(args.A_ub, args.b_ub):
                prob += pulp.lpSum(row[j] * xs[j] for j in range(n)) <= bi
        if args.A_eq and args.b_eq:
            for row, bi in zip(args.A_eq, args.b_eq):
                prob += pulp.lpSum(row[j] * xs[j] for j in range(n)) == bi

        if args.var_bounds:
            for i, (lb, ub) in enumerate(args.var_bounds):
                if lb is not None and lb > -1e30:
                    xs[i].lowBound = lb
                if ub is not None and ub < 1e30:
                    xs[i].upBound = ub

        status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
        if status == 1:
            return ToolResult(
                data={
                    "x": [pulp.value(x) for x in xs],
                    "fun": pulp.value(prob.objective),
                    "method": "pulp (CBC)",
                    "message": "MILP solved via PuLP fallback.",
                },
                success=True,
            )
        return ToolResult(
            data={"message": f"PuLP status: {pulp.LpStatus[status]}"},
            success=False,
            error=f"MILP (PuLP) failed: {pulp.LpStatus[status]}",
        )

    # ── Convex Optimization (cvxpy) ────────────────────────────────

    def _convex(self, args: NumericalToolInput) -> ToolResult:
        """Solve a convex optimization problem using cvxpy.

        Accepts a structured problem description that the LLM can generate:
        {
            "action": "convex",
            "convex_problem": "minimize 0.5*sum_squares(A@x - b)\nsubject to\nx >= 0\nsum(x) <= 1"
        }

        For more complex problems with named variables, the LLM can pass
        Python code that constructs a cvxpy Problem directly.
        """
        try:
            import cvxpy as cp
        except ImportError:
            return ToolResult(
                data=None, success=False,
                error="cvxpy is required for convex optimization. Install: pip install cvxpy",
            )

        problem_str = args.convex_problem or ""
        lines = [l.strip() for l in problem_str.strip().splitlines() if l.strip()]

        if not lines:
            return ToolResult(data=None, success=False, error="Empty convex problem.")

        # Parse: [minimize|maximize] <objective>
        #        subject to
        #        <constraint1>
        #        <constraint2>
        sense = "minimize"
        obj_line = lines[0]
        for s in ("minimize", "maximize", "min", "max"):
            if obj_line.lower().startswith(s):
                sense = "maximize" if "max" in s.lower() else "minimize"
                obj_line = obj_line[len(s):].strip()
                break

        # Split objective and constraints
        constraints_str = []
        in_constraints = False
        for line in lines[1:]:
            if line.lower() in ("subject to", "s.t.", "st:"):
                in_constraints = True
                continue
            if in_constraints:
                constraints_str.append(line)
            else:
                obj_line += "\n" + line

        # Build cvxpy problem with variable vector x
        # Determine variable count from bounds or A_ub
        n = len(args.c_vec) if args.c_vec else (
            len(args.A_ub[0]) if args.A_ub else
            len(args.A_eq[0]) if args.A_eq else
            len(args.var_bounds or []) if args.var_bounds else 1
        )

        x = cp.Variable(n, name="x")

        # Build a safe evaluation context
        A = np.asarray(args.A_ub, dtype=float) if args.A_ub else None
        b_ub = np.asarray(args.b_ub, dtype=float) if args.b_ub else None
        A_eq = np.asarray(args.A_eq, dtype=float) if args.A_eq else None
        b_eq = np.asarray(args.b_eq, dtype=float) if args.b_eq else None

        eval_globals = {
            "cp": cp, "np": np, "x": x,
            "A": A, "b": b_ub, "A_ub": A, "b_ub": b_ub,
            "A_eq": A_eq, "b_eq": b_eq,
            "sum": cp.sum, "norm": cp.norm, "abs": cp.abs,
            "square": cp.square, "quad_form": cp.quad_form,
            "sum_squares": cp.sum_squares,
        }
        eval_locals = {}

        try:
            objective_expr = eval(obj_line, eval_globals, eval_locals)
            if sense == "maximize":
                objective = cp.Maximize(objective_expr)
            else:
                objective = cp.Minimize(objective_expr)
        except Exception as e:
            return ToolResult(
                data=None, success=False,
                error=f"Failed to parse objective '{obj_line}': {e}",
            )

        constraints = []
        for cstr in constraints_str:
            try:
                c_expr = eval(cstr, eval_globals, eval_locals)
                constraints.append(c_expr)
            except Exception as e:
                return ToolResult(
                    data=None, success=False,
                    error=f"Failed to parse constraint '{cstr}': {e}",
                )

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve()
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"cvxpy solve failed: {e}")

        if prob.status in ("optimal", "optimal_inaccurate"):
            return ToolResult(
                data={
                    "x": x.value.tolist() if x.value is not None else None,
                    "fun": float(prob.value) if prob.value is not None else None,
                    "method": "cvxpy",
                    "status": prob.status,
                    "message": f"Convex optimization solved (status={prob.status}).",
                },
                success=True,
            )
        return ToolResult(
            data={"status": prob.status, "message": f"Problem status: {prob.status}"},
            success=False,
            error=f"Convex optimization did not converge: {prob.status}",
        )

    # ── Shared helpers ──────────────────────────────────────────────

    @staticmethod
    def _parse_var_bounds(
        var_bounds: list[list[float]] | None, n_vars: int
    ) -> list[tuple[float, float]]:
        """Parse per-variable bounds for LP/MILP. Default: [0, inf)."""
        if var_bounds and len(var_bounds) >= n_vars:
            return [(float(b[0]), float(b[1])) for b in var_bounds[:n_vars]]
        return [(0.0, float("inf"))] * n_vars
