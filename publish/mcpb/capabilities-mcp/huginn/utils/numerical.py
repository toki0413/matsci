"""Unified numerical solver interface for scientific computing.

Wraps scipy's numerical methods behind a consistent API with uniform error
handling and result formatting. All methods return a ``SolverResult`` with
the solution, success flag, diagnostics, and optional metadata.

Usage:
    from huginn.utils.numerical import solve_ode, minimize_func, find_root

    # Solve an ODE: dy/dt = -y, y(0) = 1
    result = solve_ode(lambda t, y: -y, t_span=(0, 10), y0=[1.0])
    # result.values → array of y(t) at result.t_points

    # Minimize a function
    result = minimize_func(lambda x: (x[0] - 3)**2 + (x[1] + 1)**2, x0=[0, 0])
    # result.x → [3, -1], result.fun → 0

    # Find a root
    result = find_root(lambda x: x**2 - 4, x0=1.0)
    # result.x → 2.0
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SolverResult:
    """Uniform result container for all numerical solvers."""

    success: bool
    method: str
    values: Any = None  # The primary result (array, scalar, dict, etc.)
    message: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict suitable for JSON / tool output."""
        def _clean(v: Any) -> Any:
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, (np.integer, np.floating)):
                return float(v)
            if isinstance(v, dict):
                return {k: _clean(val) for k, val in v.items()}
            if isinstance(v, (list, tuple)):
                return [_clean(x) for x in v]
            return v

        return {
            "success": self.success,
            "method": self.method,
            "values": _clean(self.values),
            "message": self.message,
            "diagnostics": _clean(self.diagnostics),
            "metadata": _clean(self.metadata),
        }


# ── ODE solvers ───────────────────────────────────────────────────────

def solve_ode(
    func: Callable,
    t_span: tuple[float, float],
    y0: list[float] | np.ndarray,
    method: str = "RK45",
    t_eval: list[float] | np.ndarray | None = None,
    dense_output: bool = False,
    max_step: float | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    args: tuple = (),
) -> SolverResult:
    """Solve an initial value problem dy/dt = f(t, y).

    Wraps scipy.integrate.solve_ivp with uniform error handling.
    """
    try:
        from scipy.integrate import solve_ivp

        kwargs: dict[str, Any] = {
            "dense_output": dense_output,
            "rtol": rtol,
            "atol": atol,
            "args": args,
        }
        if t_eval is not None:
            kwargs["t_eval"] = t_eval
        if max_step is not None:
            kwargs["max_step"] = max_step
        sol = solve_ivp(
            func,
            t_span,
            y0,
            method=method,
            **kwargs,
        )
        return SolverResult(
            success=sol.success,
            method=f"ode_{method}",
            values={"t": sol.t, "y": sol.y},
            message=sol.message or ("Integration completed" if sol.success else "Integration failed"),
            diagnostics={
                "n_steps": len(sol.t),
                "t_final": float(sol.t[-1]) if len(sol.t) > 0 else None,
            },
        )
    except ImportError:
        return SolverResult(success=False, method="ode", message="scipy not available")
    except Exception as e:
        return SolverResult(success=False, method=f"ode_{method}", message=str(e))


# ── Optimization ──────────────────────────────────────────────────────

def minimize_func(
    func: Callable,
    x0: list[float] | np.ndarray,
    method: str = "BFGS",
    jac: Callable | None = None,
    bounds: list[tuple[float, float]] | None = None,
    constraints: dict | list | None = None,
    tol: float = 1e-6,
    options: dict | None = None,
    args: tuple = (),
) -> SolverResult:
    """Minimize a scalar function of one or more variables.

    Wraps scipy.optimize.minimize with uniform error handling.
    """
    try:
        from scipy.optimize import minimize

        result = minimize(
            func,
            x0,
            method=method,
            jac=jac,
            bounds=bounds,
            constraints=constraints,
            tol=tol,
            options=options or {},
            args=args,
        )
        return SolverResult(
            success=result.success,
            method=f"minimize_{method}",
            values={"x": result.x, "fun": float(result.fun)},
            message=result.message,
            diagnostics={
                "nit": getattr(result, "nit", None),
                "nfev": getattr(result, "nfev", None),
                "njev": getattr(result, "njev", None),
            },
        )
    except ImportError:
        return SolverResult(success=False, method="minimize", message="scipy not available")
    except Exception as e:
        return SolverResult(success=False, method=f"minimize_{method}", message=str(e))


# ── Root finding ──────────────────────────────────────────────────────

def find_root(
    func: Callable,
    x0: float | list[float],
    method: str = "hybr",
    bracket: tuple[float, float] | None = None,
    tol: float = 1e-8,
    maxiter: int = 100,
    args: tuple = (),
) -> SolverResult:
    """Find a root of a function.

    For scalar functions, uses root_scalar (with bracket if provided).
    For vector functions, uses scipy.optimize.root.
    """
    try:
        from scipy.optimize import root, root_scalar

        if isinstance(x0, (int, float)):
            # Scalar root finding
            kwargs: dict[str, Any] = {"x0": float(x0), "maxiter": maxiter, "xtol": tol}
            if bracket is not None:
                kwargs["bracket"] = bracket
                kwargs["method"] = "brentq"
            else:
                kwargs["method"] = method if method != "hybr" else "newton"

            sol = root_scalar(func, args=args, **kwargs)
            return SolverResult(
                success=sol.converged,
                method=f"root_scalar_{sol.method}",
                values={"root": float(sol.root)},
                message=sol.flag or ("Converged" if sol.converged else "Did not converge"),
                diagnostics={"iterations": getattr(sol, "iterations", None)},
            )
        else:
            # Vector root finding
            sol = root(func, x0, method=method, tol=tol, options={"maxiter": maxiter}, args=args)
            return SolverResult(
                success=sol.success,
                method=f"root_{method}",
                values={"x": sol.x},
                message="Converged" if sol.success else "Did not converge",
                diagnostics={"nfev": getattr(sol, "nfev", None)},
            )
    except ImportError:
        return SolverResult(success=False, method="root", message="scipy not available")
    except Exception as e:
        return SolverResult(success=False, method="root", message=str(e))


# ── Curve fitting ─────────────────────────────────────────────────────

def fit_curve(
    func: Callable,
    xdata: np.ndarray,
    ydata: np.ndarray,
    p0: list[float] | None = None,
    bounds: tuple[list, list] | None = None,
    maxfev: int = 1000,
) -> SolverResult:
    """Fit a function to data using least squares.

    Wraps scipy.optimize.curve_fit.
    """
    try:
        from scipy.optimize import curve_fit

        popt, pcov = curve_fit(
            func,
            xdata,
            ydata,
            p0=p0,
            bounds=bounds or (-np.inf, np.inf),
            maxfev=maxfev,
        )
        perr = np.sqrt(np.diag(pcov))
        residuals = ydata - func(xdata, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((ydata - np.mean(ydata))**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return SolverResult(
            success=True,
            method="curve_fit",
            values={"params": popt, "param_errors": perr, "covariance": pcov},
            message="Fit completed",
            diagnostics={"r_squared": float(r_squared), "n_points": len(xdata)},
        )
    except ImportError:
        return SolverResult(success=False, method="curve_fit", message="scipy not available")
    except Exception as e:
        return SolverResult(success=False, method="curve_fit", message=str(e))


# ── Numerical integration ─────────────────────────────────────────────

def integrate(
    func: Callable,
    a: float,
    b: float,
    method: str = "quad",
    n_points: int = 1000,
    args: tuple = (),
) -> SolverResult:
    """Numerically integrate a function over [a, b].

    Methods: 'quad' (adaptive), 'trapezoid', 'simpson'.
    """
    try:
        if method == "quad":
            from scipy.integrate import quad

            value, error = quad(func, a, b, args=args)
            return SolverResult(
                success=True,
                method="integrate_quad",
                values={"integral": float(value)},
                message="Integration completed",
                diagnostics={"error_estimate": float(error)},
            )
        elif method in ("trapezoid", "simpson"):
            from scipy.integrate import simpson, trapezoid

            x = np.linspace(a, b, n_points)
            y = np.array([func(xi, *args) for xi in x])
            if method == "trapezoid":
                value = trapezoid(y, x)
            else:
                value = simpson(y, x=x)
            return SolverResult(
                success=True,
                method=f"integrate_{method}",
                values={"integral": float(value)},
                message="Integration completed",
                diagnostics={"n_points": n_points},
            )
        else:
            return SolverResult(success=False, method="integrate", message=f"Unknown method: {method}")
    except ImportError:
        return SolverResult(success=False, method="integrate", message="scipy not available")
    except Exception as e:
        return SolverResult(success=False, method=f"integrate_{method}", message=str(e))


# ── Linear algebra ────────────────────────────────────────────────────

def solve_linear(
    A: np.ndarray,
    b: np.ndarray,
) -> SolverResult:
    """Solve the linear system Ax = b."""
    try:
        from scipy.linalg import solve

        x = solve(A, b)
        return SolverResult(
            success=True,
            method="linalg_solve",
            values={"x": x},
            message="Linear system solved",
            diagnostics={"matrix_shape": list(A.shape)},
        )
    except ImportError:
        # Fallback to numpy
        try:
            x = np.linalg.solve(A, b)
            return SolverResult(
                success=True,
                method="linalg_solve_numpy",
                values={"x": x},
                message="Linear system solved (numpy fallback)",
                diagnostics={"matrix_shape": list(A.shape)},
            )
        except Exception as e:
            return SolverResult(success=False, method="linalg_solve", message=str(e))
    except Exception as e:
        return SolverResult(success=False, method="linalg_solve", message=str(e))


def eigenvalues(
    A: np.ndarray,
    eigvals_only: bool = True,
) -> SolverResult:
    """Compute eigenvalues (and optionally eigenvectors) of a matrix."""
    try:
        if eigvals_only:
            w = np.linalg.eigvals(A)
            return SolverResult(
                success=True,
                method="eigvals",
                values={"eigenvalues": w},
                message="Eigenvalues computed",
            )
        else:
            w, v = np.linalg.eig(A)
            return SolverResult(
                success=True,
                method="eig",
                values={"eigenvalues": w, "eigenvectors": v},
                message="Eigenvalues and eigenvectors computed",
            )
    except Exception as e:
        return SolverResult(success=False, method="eig", message=str(e))


# ── Interpolation ─────────────────────────────────────────────────────

def interpolate_1d(
    x: np.ndarray,
    y: np.ndarray,
    kind: str = "linear",
    x_new: np.ndarray | None = None,
    fill_value: float | str = "extrapolate",
) -> SolverResult:
    """1D interpolation of data points.

    Args:
        kind: 'linear', 'cubic', 'quadratic', 'nearest'
        x_new: New x-coordinates to evaluate at. If None, returns the interpolator.
    """
    try:
        from scipy.interpolate import interp1d

        f = interp1d(x, y, kind=kind, fill_value=fill_value)
        if x_new is not None:
            y_new = f(x_new)
            return SolverResult(
                success=True,
                method=f"interp1d_{kind}",
                values={"x_new": x_new, "y_new": y_new},
                message="Interpolation completed",
            )
        return SolverResult(
            success=True,
            method=f"interp1d_{kind}",
            values={"interpolator": "created"},
            message="Interpolator created (call with x_new to evaluate)",
        )
    except ImportError:
        return SolverResult(success=False, method="interp1d", message="scipy not available")
    except Exception as e:
        return SolverResult(success=False, method=f"interp1d_{kind}", message=str(e))


# ── FFT ───────────────────────────────────────────────────────────────

def fft(
    data: np.ndarray,
    sampling_rate: float = 1.0,
) -> SolverResult:
    """Compute the FFT of a 1D signal and return frequencies + magnitudes."""
    try:
        n = len(data)
        fft_vals = np.fft.fft(data)
        freqs = np.fft.fftfreq(n, d=1.0 / sampling_rate)
        magnitudes = np.abs(fft_vals)

        # Only return the positive half (Nyquist)
        positive_mask = freqs >= 0
        return SolverResult(
            success=True,
            method="fft",
            values={
                "frequencies": freqs[positive_mask],
                "magnitudes": magnitudes[positive_mask],
                "phases": np.angle(fft_vals[positive_mask]),
            },
            message="FFT completed",
            diagnostics={"n_samples": n, "sampling_rate": sampling_rate},
        )
    except Exception as e:
        return SolverResult(success=False, method="fft", message=str(e))
