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

from huginn.core_types import ToolContext, ToolResult, ValidationResult
from huginn.security.math_eval import safe_math_eval
from huginn.tools.base import HuginnTool, ToolProfile


class NumericalToolInput(BaseModel):
    action: Literal[
        "ode",
        "minimize",
        "constrained_minimize",
        "root",
        "curve_fit",
        "eos_fit",
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
        default=None,
        description="Inequality constraint matrix for lp (A_ub @ x <= b_ub).",
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

    # ── EOS fitting ──
    eos_type: Literal["birch_murnaghan", "murnaghan", "vinet"] = Field(
        default="birch_murnaghan",
        description="Equation of state type for eos_fit.",
    )
    volumes: list[float] | None = Field(
        default=None, description="Volume data points (Å³)"
    )
    energies: list[float] | None = Field(
        default=None, description="Energy data points (eV)"
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
        if not isinstance(result, list | tuple | np.ndarray):
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

    @staticmethod
    def _validate_expression_actions(args: NumericalToolInput) -> str | None:
        """Checks for actions driven by a user expression / initial guess."""
        if (
            args.action
            in ("ode", "minimize", "constrained_minimize", "root", "integrate")
            and not args.func
        ):
            return f"{args.action} requires a func expression."
        if args.action == "constrained_minimize" and args.x0 is None:
            return "constrained_minimize requires x0."
        if args.action == "curve_fit" and (
            not args.func_fit or args.xdata is None or args.ydata is None
        ):
            return "curve_fit requires func_fit, xdata, and ydata."
        return None

    @staticmethod
    def _validate_array_actions(args: NumericalToolInput) -> str | None:
        """Checks for actions driven by matrices / arrays."""
        if args.action == "linear_solve" and (args.A is None or args.b_vec is None):
            return "linear_solve requires A and b."
        if args.action in ("eigenvalues", "svd", "matrix_exp") and args.A is None:
            return f"{args.action} requires A."
        if args.action == "interpolate" and (args.x is None or args.y is None):
            return "interpolate requires x and y arrays."
        if args.action == "fft" and args.data is None:
            return "fft requires data array."
        return None

    @staticmethod
    def _validate_problem_actions(args: NumericalToolInput) -> str | None:
        """Checks for optimisation / EOS actions."""
        if args.action in ("lp", "milp") and args.c_vec is None:
            return f"{args.action} requires c_vec (cost coefficients)."
        if args.action == "convex" and not args.convex_problem:
            return "convex requires convex_problem (problem description)."
        if args.action == "eos_fit" and (args.volumes is None or args.energies is None):
            return "eos_fit requires volumes and energies."
        return None

    async def validate_input(
        self, args: NumericalToolInput, context: ToolContext | None = None
    ) -> ValidationResult:
        for message in (
            self._validate_expression_actions(args),
            self._validate_array_actions(args),
            self._validate_problem_actions(args),
        ):
            if message is not None:
                return ValidationResult(result=False, message=message)
        return ValidationResult(result=True)

    def _dispatch_direct(self, d: NumericalToolInput) -> ToolResult | None:
        """Actions handled by their own method, bypassing the scipy library."""
        handlers = {
            "svd": self._svd,
            "matrix_exp": self._matrix_exp,
            "constrained_minimize": self._constrained_minimize,
            "lp": self._lp,
            "milp": self._milp,
            "convex": self._convex,
            "eos_fit": self._eos_fit,
        }
        handler = handlers.get(d.action)
        return handler(d) if handler is not None else None

    def _library_result(self, d: NumericalToolInput) -> tuple[Any, str | None]:
        """Route remaining actions to huginn.utils.numerical; (result, error)."""
        handlers = {
            "ode": self._num_ode,
            "minimize": self._num_minimize,
            "root": self._num_root,
            "curve_fit": self._num_curve_fit,
            "integrate": self._num_integrate,
            "linear_solve": self._num_linear_solve,
            "eigenvalues": self._num_eigenvalues,
            "interpolate": self._num_interpolate,
            "fft": self._num_fft,
        }
        handler = handlers.get(d.action)
        if handler is None:
            return None, f"Unknown action: {d.action}"
        return handler(d)

    def _num_ode(self, d: NumericalToolInput) -> tuple[Any, str | None]:
        from huginn.utils.numerical import solve_ode

        if d.t_span is None or d.y0 is None:
            return None, "ode requires t_span and y0"
        func = _compile_ode_func(d.func or "-y[0]")
        return (
            solve_ode(
                func,
                tuple(d.t_span),  # type: ignore[arg-type]
                d.y0,
                method=d.method or "RK45",
                t_eval=d.t_eval,
                rtol=d.tol or 1e-6,
            ),
            None,
        )

    def _num_minimize(self, d: NumericalToolInput) -> tuple[Any, str | None]:
        from huginn.utils.numerical import minimize_func

        if d.x0 is None:
            return None, "minimize requires x0"
        func = _compile_vector_func(d.func or "X[0]**2")
        return (
            minimize_func(func, d.x0, method=d.method or "BFGS", tol=d.tol or 1e-6),
            None,
        )

    def _num_root(self, d: NumericalToolInput) -> tuple[Any, str | None]:
        from huginn.utils.numerical import find_root

        if d.x0 is None:
            return None, "root requires x0"
        func = _compile_scalar_func(d.func or "x**2 - 4")
        return (
            find_root(
                func,
                d.x0,
                method=d.method or "hybr",
                tol=d.tol or 1e-8,
                maxiter=d.maxiter or 100,
            ),
            None,
        )

    def _num_curve_fit(self, d: NumericalToolInput) -> tuple[Any, str | None]:
        from huginn.utils.numerical import fit_curve

        fit_expr = d.func_fit or "a0*x + a1"
        func = _compile_fit_func(fit_expr)
        return (
            fit_curve(
                func,
                np.asarray(d.xdata, dtype=float),
                np.asarray(d.ydata, dtype=float),
                p0=[1.0] * _fit_param_count(fit_expr),
            ),
            None,
        )

    def _num_integrate(self, d: NumericalToolInput) -> tuple[Any, str | None]:
        from huginn.utils.numerical import integrate

        if d.a is None or d.b is None:
            return None, "integrate requires a and b"
        func = _compile_scalar_func(d.func or "x**2")
        return (
            integrate(
                func,
                d.a,
                d.b,
                method=d.method or "quad",
                n_points=d.n_points,
            ),
            None,
        )

    def _num_linear_solve(self, d: NumericalToolInput) -> tuple[Any, str | None]:
        from huginn.utils.numerical import solve_linear

        return (
            solve_linear(
                np.asarray(d.A, dtype=float),
                np.asarray(d.b_vec, dtype=float),
            ),
            None,
        )

    def _num_eigenvalues(self, d: NumericalToolInput) -> tuple[Any, str | None]:
        from huginn.utils.numerical import eigenvalues

        if d.A is None:
            return None, "eigenvalues requires A"
        return eigenvalues(np.asarray(d.A, dtype=float)), None

    def _num_interpolate(self, d: NumericalToolInput) -> tuple[Any, str | None]:
        from huginn.utils.numerical import interpolate_1d

        return (
            interpolate_1d(
                np.asarray(d.x, dtype=float),
                np.asarray(d.y, dtype=float),
                kind=d.kind,
                x_new=np.asarray(d.x_new, dtype=float) if d.x_new else None,
            ),
            None,
        )

    def _num_fft(self, d: NumericalToolInput) -> tuple[Any, str | None]:
        from huginn.utils.numerical import fft

        return (
            fft(
                np.asarray(d.data, dtype=float),
                sampling_rate=d.sampling_rate,
            ),
            None,
        )

    async def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = NumericalToolInput(**args)

        try:
            direct = self._dispatch_direct(input_data)
            if direct is not None:
                return direct

            result, missing = self._library_result(input_data)
            if missing is not None:
                return ToolResult(data=None, success=False, error=missing)

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

    @staticmethod
    def _lp_arrays(
        args: NumericalToolInput,
    ) -> tuple[np.ndarray, Any, Any, Any, Any]:
        """(c, A_ub, b_ub, A_eq, b_eq) as float arrays or None."""
        return (
            np.asarray(args.c_vec, dtype=float),
            np.asarray(args.A_ub, dtype=float) if args.A_ub else None,
            np.asarray(args.b_ub, dtype=float) if args.b_ub else None,
            np.asarray(args.A_eq, dtype=float) if args.A_eq else None,
            np.asarray(args.b_eq, dtype=float) if args.b_eq else None,
        )

    @staticmethod
    def _lp_result(res: Any, method: str | None) -> ToolResult:
        """Package a linprog result."""
        if res.success:
            return ToolResult(
                data={
                    "x": res.x.tolist(),
                    "fun": float(res.fun),
                    "method": "linprog (HiGHS)" if method is None else method,
                    "message": res.message or "Optimization terminated successfully.",
                    "n_iter": getattr(res, "nit", None),
                },
                success=True,
            )
        return ToolResult(
            data={
                "message": res.message or "LP failed",
                "fun": float(res.fun) if res.fun is not None else None,
            },
            success=False,
            error=f"LP failed: {res.message}",
        )

    def _lp(self, args: NumericalToolInput) -> ToolResult:
        """Solve: minimize c@x  s.t.  A_ub@x <= b_ub, A_eq@x = b_eq, lb <= x <= ub."""
        try:
            from scipy.optimize import linprog
        except ImportError:
            return ToolResult(
                data=None,
                success=False,
                error="scipy is required for linear programming.",
            )

        c, A_ub, b_ub, A_eq, b_eq = self._lp_arrays(args)
        bounds = self._parse_var_bounds(args.var_bounds, n_vars=len(c))

        try:
            res = linprog(
                c,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method=args.method or "highs",
            )
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"LP failed: {e}")

        return self._lp_result(res, args.method)

    # ── Mixed-Integer Linear Programming ────────────────────────────

    @staticmethod
    def _milp_constraints(args: NumericalToolInput) -> list[Any]:
        """Linear inequality / equality constraints as scipy LinearConstraint."""
        from scipy.optimize import LinearConstraint

        constraints: list[Any] = []
        if args.A_ub is not None and args.b_ub is not None:
            # A_ub @ x <= b_ub  →  LinearConstraint(A_ub, -inf, b_ub)
            constraints.append(
                LinearConstraint(
                    np.asarray(args.A_ub, dtype=float),
                    -np.inf,
                    np.asarray(args.b_ub, dtype=float),
                )
            )
        if args.A_eq is not None and args.b_eq is not None:
            b_eq = np.asarray(args.b_eq, dtype=float)
            constraints.append(
                LinearConstraint(np.asarray(args.A_eq, dtype=float), b_eq, b_eq)
            )
        return constraints

    @staticmethod
    def _milp_bounds(args: NumericalToolInput) -> Any:
        """Variable box bounds; defaults to x >= 0."""
        from scipy.optimize import Bounds

        if args.var_bounds:
            lb = [b[0] for b in args.var_bounds]
            ub = [b[1] for b in args.var_bounds]
            return Bounds(lb=lb, ub=ub)
        return Bounds(lb=0, ub=np.inf)

    def _milp(self, args: NumericalToolInput) -> ToolResult:
        """Solve MILP: minimize c@x with linear constraints and integer variables."""
        try:
            from scipy.optimize import milp
        except ImportError:
            return self._milp_fallback_pulp(args)

        c = np.asarray(args.c_vec, dtype=float)
        constraints = self._milp_constraints(args)
        bounds = self._milp_bounds(args)
        integrality = (
            np.asarray(args.integrality, dtype=int) if args.integrality else None
        )

        try:
            res = milp(
                c=c,
                constraints=constraints or None,
                integrality=integrality,
                bounds=bounds,
            )
        except Exception:
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

    @staticmethod
    def _pulp_variables(c: list[float], integrality: list[int] | None) -> list[Any]:
        """One non-negative LpVariable per decision variable."""
        import pulp

        return [
            pulp.LpVariable(
                f"x{i}",
                lowBound=0,
                cat="Integer" if (integrality and integrality[i]) else "Continuous",
            )
            for i in range(len(c))
        ]

    @staticmethod
    def _pulp_solve_constraints(
        prob: Any, xs: list[Any], args: NumericalToolInput
    ) -> None:
        """Attach inequality / equality constraints to the PuLP problem."""
        import pulp

        n = len(xs)
        if args.A_ub and args.b_ub:
            for row, bi in zip(args.A_ub, args.b_ub):
                prob += pulp.lpSum(row[j] * xs[j] for j in range(n)) <= bi
        if args.A_eq and args.b_eq:
            for row, bi in zip(args.A_eq, args.b_eq):
                prob += pulp.lpSum(row[j] * xs[j] for j in range(n)) == bi

    @staticmethod
    def _pulp_apply_bounds(xs: list[Any], var_bounds: list[Any]) -> None:
        """Override per-variable bounds when the user supplied finite ones."""
        for i, (lb, ub) in enumerate(var_bounds):
            if lb is not None and lb > -1e30:
                xs[i].lowBound = lb
            if ub is not None and ub < 1e30:
                xs[i].upBound = ub

    def _milp_fallback_pulp(self, args: NumericalToolInput) -> ToolResult:
        """Fallback MILP solver using PuLP when scipy.optimize.milp is unavailable."""
        try:
            import pulp
        except ImportError:
            return ToolResult(
                data=None,
                success=False,
                error=(
                    "MILP requires scipy>=1.9 (milp) or pulp. "
                    "Install one: pip install scipy --upgrade  OR  pip install pulp"
                ),
            )

        c = args.c_vec or []
        prob = pulp.LpProblem("milp", pulp.LpMinimize)
        xs = self._pulp_variables(c, args.integrality)
        prob += pulp.lpSum(c[i] * xs[i] for i in range(len(c)))

        self._pulp_solve_constraints(prob, xs, args)
        if args.var_bounds:
            self._pulp_apply_bounds(xs, args.var_bounds)

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

    @staticmethod
    def _convex_parse(problem_str: str) -> tuple[str, str, list[str]]:
        """Split the problem text into (sense, objective, constraint lines)."""
        lines = [ln.strip() for ln in problem_str.strip().splitlines() if ln.strip()]
        if not lines:
            return "minimize", "", []

        # Parse: [minimize|maximize] <objective>
        #        subject to
        #        <constraint1>
        sense = "minimize"
        obj_line = lines[0]
        for s in ("minimize", "maximize", "min", "max"):
            if obj_line.lower().startswith(s):
                sense = "maximize" if "max" in s.lower() else "minimize"
                obj_line = obj_line[len(s) :].strip()
                break

        constraints_str: list[str] = []
        in_constraints = False
        for line in lines[1:]:
            if line.lower() in ("subject to", "s.t.", "st:"):
                in_constraints = True
                continue
            if in_constraints:
                constraints_str.append(line)
            else:
                obj_line += "\n" + line
        return sense, obj_line, constraints_str

    @staticmethod
    def _convex_n_vars(args: NumericalToolInput) -> int:
        """Infer the variable count from whichever array was supplied."""
        if args.c_vec:
            return len(args.c_vec)
        if args.A_ub:
            return len(args.A_ub[0])
        if args.A_eq:
            return len(args.A_eq[0])
        return len(args.var_bounds) if args.var_bounds else 1

    @staticmethod
    def _convex_globals(cp: Any, x: Any, args: NumericalToolInput) -> dict[str, Any]:
        """Eval namespace exposing cvxpy helpers + problem arrays, no builtins."""
        A = np.asarray(args.A_ub, dtype=float) if args.A_ub else None
        b_ub = np.asarray(args.b_ub, dtype=float) if args.b_ub else None
        A_eq = np.asarray(args.A_eq, dtype=float) if args.A_eq else None
        b_eq = np.asarray(args.b_eq, dtype=float) if args.b_eq else None
        # ponytail: eval with __builtins__={} blocks common escapes but
        # numpy/cvxpy attribute chains can still reach os/subprocess.
        # Acceptable for local single-user tool; switch to Docker sandbox
        # (security/docker_sandbox.py) if untrusted users share the instance.
        return {
            "__builtins__": {},  # sandbox: no __import__, no open, no exec
            "cp": cp,
            "np": np,
            "x": x,
            "A": A,
            "b": b_ub,
            "A_ub": A,
            "b_ub": b_ub,
            "A_eq": A_eq,
            "b_eq": b_eq,
            "sum": cp.sum,
            "norm": cp.norm,
            "abs": cp.abs,
            "square": cp.square,
            "quad_form": cp.quad_form,
            "sum_squares": cp.sum_squares,
        }

    @staticmethod
    def _convex_objective(
        cp: Any, obj_line: str, sense: str, eval_globals: dict[str, Any]
    ) -> tuple[Any, str | None]:
        """Compile the objective expression into a cvxpy Minimize/Maximize."""
        try:
            objective_expr = eval(obj_line, eval_globals, {})
        except Exception as e:
            return None, f"Failed to parse objective '{obj_line}': {e}"
        if sense == "maximize":
            return cp.Maximize(objective_expr), None
        return cp.Minimize(objective_expr), None

    @staticmethod
    def _convex_constraints(
        constraints_str: list[str], eval_globals: dict[str, Any]
    ) -> tuple[list[Any], str | None]:
        """Compile each constraint line; return (constraints, error)."""
        constraints: list[Any] = []
        for cstr in constraints_str:
            try:
                constraints.append(eval(cstr, eval_globals, {}))
            except Exception as e:
                return [], f"Failed to parse constraint '{cstr}': {e}"
        return constraints, None

    @staticmethod
    def _convex_solve(cp: Any, prob: Any, x: Any) -> ToolResult:
        """Solve the cvxpy problem and package the outcome."""
        try:
            prob.solve()
        except Exception as e:
            return ToolResult(
                data=None, success=False, error=f"cvxpy solve failed: {e}"
            )

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
                data=None,
                success=False,
                error="cvxpy is required for convex optimization. Install: pip install cvxpy",
            )

        sense, obj_line, constraints_str = self._convex_parse(args.convex_problem or "")
        if not obj_line:
            return ToolResult(data=None, success=False, error="Empty convex problem.")

        x = cp.Variable(self._convex_n_vars(args), name="x")
        eval_globals = self._convex_globals(cp, x, args)

        objective, err = self._convex_objective(cp, obj_line, sense, eval_globals)
        if err is not None:
            return ToolResult(data=None, success=False, error=err)

        constraints, err = self._convex_constraints(constraints_str, eval_globals)
        if err is not None:
            return ToolResult(data=None, success=False, error=err)

        return self._convex_solve(cp, cp.Problem(objective, constraints), x)

    # ── Equation of State fitting ───────────────────────────────────

    # 1 eV/Å³ = 160.21766208 GPa
    _EV_PER_A3_TO_GPA = 160.21766208

    @staticmethod
    def _eos_validate(args: NumericalToolInput) -> str | None:
        """Return an error message when E-V data cannot be fitted, else None."""
        if not args.volumes or not args.energies:
            return "eos_fit requires volumes and energies."
        if len(args.volumes) != len(args.energies):
            return "volumes and energies must have the same length."
        if len(args.volumes) < 4:
            return f"EOS fit needs ≥4 points, got {len(args.volumes)}."
        return None

    @staticmethod
    def _eos_guess(V: np.ndarray, E: np.ndarray) -> list[float]:
        """Initial [E0, V0, B0, B0'] guesses derived from the data itself."""
        i_min = int(np.argmin(E))
        V0_guess = float(V[i_min])
        E0_guess = float(E[i_min])
        # B0 ~ V0 * d²E/dV² at the minimum (eV/Å³)
        # np.gradient handles non-uniform spacing
        if len(V) >= 3:
            d2E = np.gradient(np.gradient(E, V), V)
            B0_guess = max(V0_guess * float(d2E[i_min]), 1e-4)
        else:
            B0_guess = 1e-3
        B0p_guess = 4.0  # typical value for crystalline solids
        return [E0_guess, V0_guess, B0_guess, B0p_guess]

    @staticmethod
    def _eos_param_dicts(
        popt: Any, pcov: np.ndarray
    ) -> tuple[dict[str, float], dict[str, float]]:
        """(parameters, 1σ uncertainties) keyed by E0/V0/B0/B0_prime."""
        names = ("E0", "V0", "B0", "B0_prime")
        perr = np.sqrt(np.diag(pcov))
        return (
            {n: float(popt[i]) for i, n in enumerate(names)},
            {n: float(perr[i]) for i, n in enumerate(names)},
        )

    @staticmethod
    def _eos_fit_stats(
        eos_func: Any, V: np.ndarray, E: np.ndarray, popt: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """R² plus the smoothed V/E curve used for plotting."""
        E_pred = eos_func(V, *popt)
        ss_res = float(np.sum((E - E_pred) ** 2))
        ss_tot = float(np.sum((E - np.mean(E)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 1.0
        V_smooth = np.linspace(float(V.min()), float(V.max()), 200)
        return r_squared, V_smooth, eos_func(V_smooth, *popt)

    def _eos_result(
        self,
        eos_type: str,
        eos_func: Any,
        V: np.ndarray,
        E: np.ndarray,
        popt: np.ndarray,
        pcov: np.ndarray,
    ) -> ToolResult:
        """Package fitted parameters, uncertainties, R² and the smooth curve."""
        params, uncertainties = self._eos_param_dicts(popt, pcov)
        E0_fit, V0_fit, B0_fit, B0p_fit = params.values()
        r_squared, V_smooth, E_smooth = self._eos_fit_stats(eos_func, V, E, popt)

        B0_gpa = B0_fit * self._EV_PER_A3_TO_GPA

        return ToolResult(
            data={
                "eos_type": eos_type,
                "E0": E0_fit,
                "V0": V0_fit,
                "B0_GPa": B0_gpa,
                "B0_eV_per_A3": B0_fit,
                "B0_prime": B0p_fit,
                "r_squared": r_squared,
                "parameters": params,
                "uncertainties": uncertainties,
                "fit_curve": {
                    "volumes": V_smooth.tolist(),
                    "energies": E_smooth.tolist(),
                },
                "input_data": {
                    "volumes": V.tolist(),
                    "energies": E.tolist(),
                },
                "message": (
                    f"{eos_type} EOS fit: B0={B0_gpa:.2f} GPa, "
                    f"V0={V0_fit:.4f} Å³, E0={E0_fit:.6f} eV, "
                    f"B0'={B0p_fit:.3f}, R²={r_squared:.6f}"
                ),
            },
            success=True,
        )

    @staticmethod
    def _eos_curve_fit(
        curve_fit: Any, eos_func: Any, V: np.ndarray, E: np.ndarray, p0: list[float]
    ) -> tuple[Any, Any, str | None]:
        """Run the bounded curve fit; return (popt, pcov, error)."""
        # keep V0 inside the data range, B0 positive, B0' physical
        bounds = (
            [-np.inf, float(V.min()) * 0.5, 1e-8, 0.1],
            [np.inf, float(V.max()) * 2.0, np.inf, 20.0],
        )
        try:
            popt, pcov = curve_fit(eos_func, V, E, p0=p0, bounds=bounds, maxfev=10000)
        except Exception as e:
            return None, None, f"EOS curve_fit failed: {e}"
        return popt, pcov, None

    def _eos_fit(self, args: NumericalToolInput) -> ToolResult:
        """Fit an equation of state to energy-volume data.

        Supports third-order Birch-Murnaghan, Murnaghan, and Vinet EOS.
        All internal math uses eV and Å³; B0 is reported in both eV/Å³ and GPa.
        """
        error = self._eos_validate(args)
        if error is not None:
            return ToolResult(data=None, success=False, error=error)

        try:
            from scipy.optimize import curve_fit
        except ImportError:
            return ToolResult(
                data=None,
                success=False,
                error="scipy is required for eos_fit.",
            )

        V = np.asarray(args.volumes, dtype=float)
        E = np.asarray(args.energies, dtype=float)

        # sort by volume so gradient estimation is sane
        order = np.argsort(V)
        V, E = V[order], E[order]

        eos_type = args.eos_type
        eos_func = self._get_eos_function(eos_type)
        if eos_func is None:
            return ToolResult(
                data=None,
                success=False,
                error=f"Unknown EOS type: {eos_type}",
            )

        popt, pcov, error = self._eos_curve_fit(
            curve_fit, eos_func, V, E, self._eos_guess(V, E)
        )
        if error is not None:
            return ToolResult(data=None, success=False, error=error)

        return self._eos_result(eos_type, eos_func, V, E, popt, pcov)

    @staticmethod
    def _get_eos_function(eos_type: str):
        """Return the EOS callable f(V, E0, V0, B0, B0p) or None if unknown.

        B0 is in eV/Å³ internally; V in Å³; E in eV.
        """
        if eos_type == "birch_murnaghan":
            # 3rd-order Birch-Murnaghan EOS.
            # Ref: Birch (1947); Poirier "Introduction to the Physics of the
            # Earth's Interior"; Mouhat & Coudert PRB 2014 也用此形式.
            # 标准定义 (Wikipedia / pymatgen / ase 一致):
            #   η = (V0/V)^(1/3),   x = η² - 1
            #   E = E0 + (9/16) * B0 * V0 * [B0' * x³ + 6 * x²]
            #
            # 旧实现三处错误 (审计 14号报告):
            #   1. η 取反: (V/V0) → 应为 (V0/V), 物理意义相反
            #   2. x³ 系数 (B0'-4) → 应为 B0', 把 B0'=4 时三次项消成 0
            #   3. x² 系数 2 → 应为 6, 偏小 3 倍
            # 旧测试通过的唯一原因是 test_eos_fit.py 用同一错误公式生成数据
            # (循环论证), 且 B0P_TRUE=4.0 恰好消掉三次项.
            def eos(V, E0, V0, B0, B0p):
                eta = (V0 / V) ** (1.0 / 3.0)
                x = eta**2 - 1.0
                return E0 + (9.0 / 16.0) * B0 * V0 * (B0p * x**3 + 6.0 * x**2)

            return eos

        if eos_type == "murnaghan":

            def eos(V, E0, V0, B0, B0p):
                r = V0 / V
                return E0 + (B0 * V0 / B0p) * (
                    r ** (B0p - 1.0) / (B0p - 1.0) - B0p / (B0p - 1.0) + V / V0
                )

            return eos

        if eos_type == "vinet":

            def eos(V, E0, V0, B0, B0p):
                eta = (V / V0) ** (1.0 / 3.0)
                xi = 1.5 * (B0p - 1.0) * (1.0 - eta)
                return E0 + 4.0 * B0 * V0 / (B0p - 1.0) ** 2 * (
                    1.0 - (1.0 - xi) * np.exp(xi)
                )

            return eos

        return None

    # ── Shared helpers ──────────────────────────────────────────────

    @staticmethod
    def _parse_var_bounds(
        var_bounds: list[list[float]] | None, n_vars: int
    ) -> list[tuple[float, float]]:
        """Parse per-variable bounds for LP/MILP. Default: [0, inf)."""
        if var_bounds and len(var_bounds) >= n_vars:
            return [(float(b[0]), float(b[1])) for b in var_bounds[:n_vars]]
        return [(0.0, float("inf"))] * n_vars
