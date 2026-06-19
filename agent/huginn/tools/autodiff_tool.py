"""Automatic Differentiation Tool — sensitivity analysis and gradient computation.

Uses JAX for high-performance automatic differentiation of physical models.
Enables:
  - Gradient-based optimization of material parameters
  - Sensitivity analysis (how does output change with input?)
  - Hessian computation for stability analysis
  - Jacobian for coupled systems

This bridges symbolic math (exact derivatives) with numerical computation
(efficient evaluation on large datasets).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult


class AutoDiffInput(BaseModel):
    action: str = Field(
        ..., description="gradient | hessian | jacobian | sensitivity | optimize"
    )
    function_type: str = Field(
        default="custom", description="Custom function name or 'custom'"
    )
    function_params: dict[str, Any] = Field(
        default_factory=dict, description="Parameters defining the function"
    )
    variables: dict[str, list[float]] = Field(
        default_factory=dict, description="Variable values for evaluation"
    )
    target_variable: str | None = Field(
        default=None, description="Variable to differentiate with respect to"
    )
    step_size: float = Field(
        default=1e-5, description="Finite difference step size (fallback)"
    )
    use_jax: bool = Field(default=True, description="Use JAX autodiff if available")


class AutoDiffTool(HuginnTool):
    """Automatic differentiation for materials science models.

    Computes gradients, Hessians, and Jacobians for:
    - Fitting potential parameters to DFT data
    - Optimizing alloy compositions
    - Sensitivity analysis of constitutive models
    - Stability analysis (eigenvalues of Hessian)
    """

    name = "autodiff_tool"
    description = (
        "Compute gradients, Hessians, and Jacobians using automatic differentiation. "
        "Supports built-in material models (EOS, elastic, potential) and custom functions."
    )
    input_schema = AutoDiffInput

    def __init__(self):
        super().__init__()
        self._jax_available = self._check_jax()
        self._built_in_functions = self._register_functions()

    def _check_jax(self) -> bool:
        import importlib.util

        return importlib.util.find_spec("jax") is not None

    def _register_functions(self) -> dict[str, Callable]:
        """Register built-in material science functions."""
        functions = {}

        # Birch-Murnaghan EOS: E(V) = E0 + B0*V0/BP * [f^BP * (BP-1) + 1] * exp(-f)
        # where f = (V/V0)^(-1/3) - 1
        def birch_murnaghan(V, E0=0.0, B0=100.0, V0=100.0, BP=4.0):
            # Guard against division by zero or negative V0
            V0 = np.where(np.asarray(V0) <= 0, 1e-3, V0)
            f = (V / V0) ** (-1.0 / 3.0) - 1.0
            # numpy float exponent on negative base yields nan; handle sign explicitly
            bp_int = int(round(BP))
            f_arr = np.asarray(f, dtype=float)
            f_pow = np.empty_like(f_arr, dtype=float)
            pos = f_arr >= 0
            f_pow[pos] = f_arr[pos] ** BP
            neg = ~pos
            f_pow[neg] = np.sign(f_arr[neg]) ** bp_int * np.abs(f_arr[neg]) ** BP
            return E0 + B0 * V0 / BP * (f_pow * (BP - 1) + 1) * np.exp(-f)

        functions["birch_murnaghan"] = birch_murnaghan

        # Murnaghan EOS: E(V) = E0 + B0*V/BP * [ (V0/V)^BP / (BP-1) + 1 ] - B0*V0/(BP-1)
        def murnaghan(V, E0=0.0, B0=100.0, V0=100.0, BP=4.0):
            return (
                E0 + B0 * V / BP * ((V0 / V) ** BP / (BP - 1) + 1) - B0 * V0 / (BP - 1)
            )

        functions["murnaghan"] = murnaghan

        # Vinet EOS
        def vinet(V, E0=0.0, B0=100.0, V0=100.0, BP=4.0):
            x = (V / V0) ** (1.0 / 3.0)
            eta = 1.5 * (BP - 1)
            return E0 + 2 * B0 * V0 / (BP - 1) ** 2 * (
                1 - (1 + eta * (x - 1)) * np.exp(-eta * (x - 1))
            )

        functions["vinet"] = vinet

        # Neo-Hookean hyperelastic energy: Ψ = C10*(I1-3) + D1*(J-1)²
        def neo_hookean(I1, J, C10=0.5, D1=2.0):
            return C10 * (I1 - 3) + D1 * (J - 1) ** 2

        functions["neo_hookean"] = neo_hookean

        # Mooney-Rivlin: Ψ = C10*(I1-3) + C01*(I2-3) + D1*(J-1)²
        def mooney_rivlin(I1, I2, J, C10=0.5, C01=0.1, D1=2.0):
            return C10 * (I1 - 3) + C01 * (I2 - 3) + D1 * (J - 1) ** 2

        functions["mooney_rivlin"] = mooney_rivlin

        # Lennard-Jones potential: E(r) = 4ε[(σ/r)^12 - (σ/r)^6]
        def lennard_jones(r, epsilon=1.0, sigma=1.0):
            sr6 = (sigma / r) ** 6
            return 4 * epsilon * (sr6**2 - sr6)

        functions["lennard_jones"] = lennard_jones

        # Morse potential: E(r) = D_e * [1 - exp(-a*(r-r_e))]²
        def morse(r, De=1.0, a=1.0, re=1.0):
            return De * (1 - np.exp(-a * (r - re))) ** 2

        functions["morse"] = morse

        return functions

    def is_read_only(self, args: AutoDiffInput) -> bool:
        return True

    async def call(self, args: AutoDiffInput, context: ToolContext) -> ToolResult:
        action = args.action.lower()

        try:
            if action == "gradient":
                return self._compute_gradient(args)
            if action == "hessian":
                return self._compute_hessian(args)
            if action == "jacobian":
                return self._compute_jacobian(args)
            if action == "sensitivity":
                return self._compute_sensitivity(args)
            if action == "optimize":
                return self._optimize_parameters(args)

            return ToolResult(
                data=None, success=False, error=f"Unknown action: {args.action}"
            )
        except Exception as e:
            return ToolResult(
                data=None, success=False, error=f"AutoDiff error: {str(e)}"
            )

    # ------------------------------------------------------------------
    # Core computations
    # ------------------------------------------------------------------

    def _compute_gradient(self, args: AutoDiffInput) -> ToolResult:
        """Compute gradient of a function with respect to specified variables."""
        fn = self._get_function(args)
        var_names = list(args.variables.keys())
        var_values = [np.array(v) for v in args.variables.values()]

        if self._jax_available and args.use_jax:
            import jax
            import jax.numpy as jnp

            # Try JAX autodiff; fall back to finite differences on failure.
            try:
                # Build a fully-JAX version of the function so autodiff works.
                def jax_fn(x_dict):
                    kw = {k: x_dict[k] for k in var_names}
                    kw.update(args.function_params)
                    out = fn(**kw)
                    # Collapse to scalar
                    arr = jnp.asarray(out)
                    return arr.sum() if arr.ndim > 0 else arr

                x0 = {name: jnp.array(v).sum() for name, v in args.variables.items()}
                grad_fn = jax.grad(jax_fn)
                grads = grad_fn(x0)

                result = {
                    "gradients": {name: float(grads[name]) for name in var_names},
                    "method": "JAX autodiff",
                }
            except Exception:
                # Fallback to finite differences
                pass
        if "result" not in locals():
            # Finite difference fallback
            result = {"gradients": {}, "method": "finite difference"}
            for name, values in args.variables.items():
                h = args.step_size
                # Simple central difference — ensure numpy arrays for arithmetic
                plus = {k: np.array(v) for k, v in args.variables.items()}
                minus = {k: np.array(v) for k, v in args.variables.items()}
                plus[name] = np.array([v + h for v in values])
                minus[name] = np.array([v - h for v in values])
                f_plus = fn(**plus, **args.function_params)
                f_minus = fn(**minus, **args.function_params)
                diff = np.squeeze(np.asarray((f_plus - f_minus) / (2 * h)))
                result["gradients"][name] = float(diff)

        return ToolResult(data=result, success=True)

    def _compute_hessian(self, args: AutoDiffInput) -> ToolResult:
        """Compute Hessian matrix (second derivatives)."""
        fn = self._get_function(args)
        var_names = list(args.variables.keys())

        if self._jax_available and args.use_jax:
            import jax
            import jax.numpy as jnp

            def jax_fn(x_dict):
                kwargs = {**x_dict, **args.function_params}
                return fn(**kwargs)

            # Compute Hessian using JAX
            # For scalar output, hessian is ∂²f/∂xᵢ∂xⱼ
            x0 = {name: jnp.array(v[0]) for name, v in args.variables.items()}
            hess_fn = jax.hessian(jax_fn)
            H = hess_fn(x0)

            # Convert to matrix format
            n = len(var_names)
            hess_matrix = [
                [float(H[var_names[i]][var_names[j]]) for j in range(n)]
                for i in range(n)
            ]

            # Compute eigenvalues for stability analysis
            H_np = np.array(hess_matrix)
            eigvals = np.linalg.eigvalsh(H_np)

            result = {
                "hessian_matrix": hess_matrix,
                "variables": var_names,
                "eigenvalues": [float(ev) for ev in eigvals],
                "positive_definite": all(ev > 0 for ev in eigvals),
                "method": "JAX hessian",
            }

            if not result["positive_definite"]:
                result["stability_warning"] = (
                    "Hessian has negative eigenvalues — system is at a saddle point or maximum"
                )

            return ToolResult(data=result, success=True)
        else:
            return ToolResult(
                data=None,
                success=False,
                error="Hessian computation requires JAX. Install with: pip install jax jaxlib",
            )

    def _compute_jacobian(self, args: AutoDiffInput) -> ToolResult:
        """Compute Jacobian for vector-valued functions."""
        # For now, return finite difference approximation
        fn = self._get_function(args)
        var_names = list(args.variables.keys())
        var_values = {name: np.array(v) for name, v in args.variables.items()}

        h = args.step_size
        jac = {}

        for name in var_names:
            plus = dict(var_values)
            minus = dict(var_values)
            plus[name] = plus[name] + h
            minus[name] = minus[name] - h
            f_plus = fn(**plus, **args.function_params)
            f_minus = fn(**minus, **args.function_params)
            diff = np.squeeze(np.asarray((f_plus - f_minus) / (2 * h)))
            jac[name] = float(diff)

        return ToolResult(
            data={
                "jacobian": jac,
                "variables": var_names,
                "method": "finite difference",
            },
            success=True,
        )

    def _compute_sensitivity(self, args: AutoDiffInput) -> ToolResult:
        """Compute normalized sensitivity coefficients: S = (∂f/∂x) * (x/f)."""
        grad_result = self._compute_gradient(args)
        if not grad_result.success:
            return grad_result

        gradients = grad_result.data["gradients"]
        fn = self._get_function(args)
        f0 = fn(
            **{name: np.array(v) for name, v in args.variables.items()},
            **args.function_params,
        )
        f0 = float(np.squeeze(np.asarray(f0)))

        sensitivities = {}
        for name, grad in gradients.items():
            x0 = args.variables[name][0] if args.variables[name] else 1.0
            if abs(f0) > 1e-10 and abs(x0) > 1e-10:
                sensitivities[name] = float(grad * x0 / f0)
            else:
                sensitivities[name] = None

        f0_scalar = f0
        return ToolResult(
            data={
                "sensitivities": sensitivities,
                "function_value": f0_scalar,
                "gradients": gradients,
                "interpretation": "|S| > 1: highly sensitive; |S| < 0.1: weakly sensitive",
            },
            success=True,
        )

    def _optimize_parameters(self, args: AutoDiffInput) -> ToolResult:
        """Simple gradient descent optimization of model parameters."""
        fn = self._get_function(args)
        params = dict(args.function_params)
        target_data = args.variables.get("target", [])
        input_data = {k: v for k, v in args.variables.items() if k != "target"}

        if not target_data:
            return ToolResult(
                data=None,
                success=False,
                error="No target data provided for optimization",
            )

        # Simple gradient descent on sum of squared errors
        lr = 0.01
        n_iter = 100
        param_names = list(params.keys())

        input_rows = list(zip(*input_data.values()))
        n_data = len(target_data)

        for _ in range(n_iter):
            # Compute loss and gradients
            predictions = [
                fn(
                    **{
                        k: np.array([v_i])
                        for k, v_i in zip(input_data.keys(), input_rows[i])
                    },
                    **params,
                )
                for i in range(n_data)
            ]
            loss = float(
                sum(
                    float(np.squeeze(p - t)) ** 2
                    for p, t in zip(predictions, target_data)
                )
            )

            # Finite difference gradients for each parameter
            for p_name in param_names:
                h = 1e-6
                params_plus = dict(params)
                params_minus = dict(params)
                params_plus[p_name] += h
                params_minus[p_name] -= h

                loss_plus = float(
                    sum(
                        float(
                            np.squeeze(
                                fn(
                                    **{
                                        k: np.array([v_i])
                                        for k, v_i in zip(
                                            input_data.keys(), input_rows[i]
                                        )
                                    },
                                    **params_plus,
                                )
                                - target_data[i]
                            )
                        )
                        ** 2
                        for i in range(n_data)
                    )
                )
                loss_minus = float(
                    sum(
                        float(
                            np.squeeze(
                                fn(
                                    **{
                                        k: np.array([v_i])
                                        for k, v_i in zip(
                                            input_data.keys(), input_rows[i]
                                        )
                                    },
                                    **params_minus,
                                )
                                - target_data[i]
                            )
                        )
                        ** 2
                        for i in range(n_data)
                    )
                )

                grad = (loss_plus - loss_minus) / (2 * h)
                params[p_name] -= lr * grad
                # Project parameters that must remain positive (physical constraints)
                if p_name in (
                    "V0",
                    "V",
                    "r",
                    "re",
                    "sigma",
                    "B0",
                    "BP",
                    "C10",
                    "D1",
                    "C01",
                    "De",
                    "a",
                    "epsilon",
                ):
                    params[p_name] = max(params[p_name], 1e-6)

        return ToolResult(
            data={
                "optimized_params": params,
                "final_loss": float(loss),
                "iterations": n_iter,
                "method": "gradient descent",
            },
            success=True,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_function(self, args: AutoDiffInput) -> Callable:
        """Get the function to differentiate."""
        fn_name = args.function_type.lower()
        if fn_name in self._built_in_functions:
            return self._built_in_functions[fn_name]

        # Try to parse as sympy expression
        if args.function_params.get("expression"):
            import sympy as sp

            expr_str = args.function_params["expression"]
            expr = sp.sympify(expr_str)
            return lambda **kwargs: float(expr.subs(kwargs))

        raise ValueError(
            f"Unknown function: {fn_name}. Available: {list(self._built_in_functions.keys())}"
        )
