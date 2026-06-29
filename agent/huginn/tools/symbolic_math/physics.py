"""物理量类 action: dimensional_analysis / dft / thermodynamics / probability / unified."""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any

import sympy as sp

from huginn.types import ToolResult

if TYPE_CHECKING:
    from .tool import SymbolicMathInput


def dimensional_analysis(args: "SymbolicMathInput") -> ToolResult:
    """量纲分析, 走 DimensionalValidator."""
    from huginn.execution.dimensional_validator import DimensionalValidator

    validator = DimensionalValidator()
    target = (args.target or "validate_expression").lower()

    # 模式 1: check_equation — 表达式形如 "210 GPa = 500 MPa / 0.001"
    if target == "check_equation":
        expr = args.expression or ""
        if "=" not in expr:
            return ToolResult(
                data=None,
                success=False,
                error="Equation must contain '=' for check_equation",
            )
        lhs_str, rhs_str = expr.split("=", 1)
        lhs_parts = [p.strip() for p in re.split(r"[+/\-]", lhs_str) if p.strip()]
        rhs_parts = [p.strip() for p in re.split(r"[+/\-]", rhs_str) if p.strip()]
        result = validator.check_equation(
            lhs_parts, rhs_parts, equation_name=args.expression
        )
        return ToolResult(
            data={
                "consistent": result.consistent,
                "equation": result.equation,
                "lhs_dimensions": result.lhs_dimensions,
                "rhs_dimensions": result.rhs_dimensions,
                "notes": result.notes,
            },
            success=True,
        )

    # 模式 2: buckingham_pi — expression 是逗号分隔的 "name:unit" 列表
    if target == "buckingham_pi":
        expr = args.expression or ""
        variables = []
        for part in expr.split(","):
            part = part.strip()
            if ":" in part:
                name, unit = part.split(":", 1)
                variables.append((name.strip(), unit.strip()))
            elif " " in part:
                # e.g. "210 GPa" — 忽略数值, 只要单位
                _, unit = part.rsplit(" ", 1)
                variables.append((f"var_{len(variables)}", unit.strip()))
        pi_groups = validator.buckingham_pi(variables, target="")
        return ToolResult(
            data={
                "pi_groups": pi_groups,
                "n_variables": len(variables),
            },
            success=True,
        )

    # 模式 3: validate_expression — 解析表达式里的每个量
    expr = args.expression or ""

    quantities = re.findall(r"[+-]?\d+\.?\d*(?:[eE][+-]?\d+)?\s+\w+(?:/\w+)?", expr)
    parsed = []
    for q in quantities:
        try:
            val, unit, dims = validator.parse_quantity(q)
            parsed.append(
                {"quantity": q, "value": val, "unit": unit, "dimensions": dims}
            )
        except ValueError:
            parsed.append({"quantity": q, "error": "Unknown unit"})

    return ToolResult(
        data={
            "quantities": parsed,
            "expression": expr,
        },
        success=True,
    )


def dft(args: "SymbolicMathInput") -> ToolResult:
    """密度泛函理论相关计算.

    支持:
      fermi_energy | free_electron_dos | particle_in_box
      | tight_binding_band | lda_xc_energy
    """
    target = (args.target or "fermi_energy").lower()

    params: dict[str, float] = {}
    if args.expression:
        for part in args.expression.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                params[k.strip()] = float(v.strip())

    m = params.get("m", 1.0)  # 电子质量 (原子单位)
    hbar = params.get("hbar", 1.0)  # 约化 Planck 常数 (原子单位)

    if target == "fermi_energy":
        n = params.get("n", 0.05)
        ef = (hbar**2 / (2.0 * m)) * (3.0 * math.pi**2 * n) ** (2.0 / 3.0)
        return ToolResult(
            data={
                "fermi_energy": ef,
                "fermi_wavevector": (3.0 * math.pi**2 * n) ** (1.0 / 3.0),
                "density": n,
            },
            success=True,
        )

    if target == "free_electron_dos":
        n = params.get("n", 0.05)
        epsilon = params.get("epsilon", 1.0)
        prefactor = 1.0 / (2.0 * math.pi**2)
        mass_term = (2.0 * m / hbar**2) ** 1.5
        dos = prefactor * mass_term * math.sqrt(epsilon)
        return ToolResult(
            data={
                "dos": dos,
                "energy": epsilon,
                "density": n,
            },
            success=True,
        )

    if target == "particle_in_box":
        L = params.get("L", 10.0)
        N = int(params.get("N", 3))
        levels = []
        for nn in range(1, N + 1):
            e = (nn**2 * math.pi**2 * hbar**2) / (2.0 * m * L**2)
            levels.append({"n": nn, "energy": e})
        return ToolResult(
            data={
                "levels": levels,
                "box_length": L,
            },
            success=True,
        )

    if target == "tight_binding_band":
        epsilon0 = params.get("epsilon0", 0.0)
        t = params.get("t", 1.0)
        a = params.get("a", 1.0)
        nK = int(params.get("nK", 50))
        band = []
        for idx in range(nK):
            k = -math.pi / a + 2.0 * math.pi / a * (idx / (nK - 1.0))
            e = epsilon0 - 2.0 * t * math.cos(k * a)
            band.append({"k": k, "energy": e})
        return ToolResult(
            data={
                "band": band,
                "epsilon0": epsilon0,
                "t": t,
                "a": a,
            },
            success=True,
        )

    if target == "lda_xc_energy":
        n = params.get("n", 0.05)
        ex = -(3.0 / 4.0) * (3.0 * n / math.pi) ** (1.0 / 3.0)
        rs = (3.0 / (4.0 * math.pi * n)) ** (1.0 / 3.0)
        A, B, C, D = 0.0311, -0.048, 0.0020, -0.0116
        if rs < 1.0:
            ec = A * math.log(rs) + B + C * rs * math.log(rs) + D * rs
        else:
            ec = -0.1423 / (rs + 1.0529 * math.sqrt(rs) + 0.3334)
        return ToolResult(
            data={
                "exchange_energy_density": ex,
                "correlation_energy_density": ec,
                "xc_energy_density": ex + ec,
                "density": n,
                "rs": rs,
            },
            success=True,
        )

    return ToolResult(
        data=None, success=False, error=f"Unknown dft target: {target}"
    )


def thermodynamics(args: "SymbolicMathInput") -> ToolResult:
    """热力学计算.

    支持:
      ideal_gas | van_der_waals | helmholtz_energy | gibbs_energy
      | chemical_potential | clausius_clapeyron | partition_function
    """
    target = (args.target or "ideal_gas").lower()

    params: dict[str, float] = {}
    if args.expression:
        for part in args.expression.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                params[k.strip()] = float(v.strip())

    R = 8.314462618  # J/(mol·K)

    if target == "ideal_gas":
        n = params.get("n", 1.0)
        T = params.get("T", 273.15)
        V = params.get("V", 0.022414)
        P = n * R * T / V
        U = 1.5 * n * R * T
        return ToolResult(
            data={
                "pressure": P,
                "internal_energy": U,
                "volume": V,
                "temperature": T,
                "moles": n,
            },
            success=True,
        )

    if target == "van_der_waals":
        n = params.get("n", 1.0)
        T = params.get("T", 273.15)
        V = params.get("V", 0.022414)
        a = params.get("a", 0.364)
        b = params.get("b", 4.27e-5)
        P = n * R * T / (V - n * b) - a * n * n / (V * V)
        return ToolResult(
            data={
                "pressure": P,
                "vdw_a": a,
                "vdw_b": b,
                "critical_temperature": 8.0 * a / (27.0 * R * b),
                "critical_pressure": a / (27.0 * b * b),
            },
            success=True,
        )

    if target == "helmholtz_energy":
        n = params.get("n", 1.0)
        T = params.get("T", 300.0)
        V1 = params.get("V1", 1.0)
        V2 = params.get("V2", 2.0)
        U = 1.5 * n * R * T
        deltaS = n * R * math.log(V2 / V1)
        F = U - T * deltaS
        return ToolResult(
            data={
                "helmholtz_energy": F,
                "internal_energy": U,
                "entropy_change": deltaS,
            },
            success=True,
        )

    if target == "gibbs_energy":
        n = params.get("n", 1.0)
        T = params.get("T", 300.0)
        P = params.get("P", 101325.0)
        P0 = params.get("P0", 101325.0)
        H = 2.5 * n * R * T
        S = n * R * math.log(P0 / P)
        G = H - T * S
        return ToolResult(
            data={
                "gibbs_energy": G,
                "enthalpy": H,
                "entropy": S,
            },
            success=True,
        )

    if target == "chemical_potential":
        mu0 = params.get("mu0", 0.0)
        T = params.get("T", 300.0)
        P = params.get("P", 101325.0)
        P0 = params.get("P0", 101325.0)
        mu = mu0 + R * T * math.log(P / P0)
        return ToolResult(
            data={
                "chemical_potential": mu,
                "temperature": T,
                "pressure": P,
            },
            success=True,
        )

    if target == "clausius_clapeyron":
        T = params.get("T", 373.15)
        L = params.get("L", 40700.0)
        deltaV = params.get("deltaV", 18.0e-6)
        slope = L / (T * deltaV)
        return ToolResult(
            data={
                "slope_dPdT": slope,
                "latent_heat": L,
                "temperature": T,
                "delta_volume": deltaV,
            },
            success=True,
        )

    if target == "partition_function":
        m = params.get("m", 9.11e-31)
        T = params.get("T", 300.0)
        V = params.get("V", 1.0)
        kB_SI = 1.380649e-23
        thermal_wavelength = math.sqrt(2.0 * math.pi * m * kB_SI * T)
        Z1 = V / (thermal_wavelength**3)
        return ToolResult(
            data={
                "single_partition_function": Z1,
                "thermal_wavelength": thermal_wavelength,
            },
            success=True,
        )

    return ToolResult(
        data=None, success=False, error=f"Unknown thermodynamics target: {target}"
    )


def probability(args: "SymbolicMathInput") -> ToolResult:
    """概率与高斯过程计算.

    支持:
      normal_pdf | normal_cdf | gp_kernel | monte_carlo_integral
      | bayesian_update_normal
    """
    target = (args.target or "normal_pdf").lower()

    params: dict[str, float] = {}
    if args.expression:
        for part in args.expression.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                params[k.strip()] = float(v.strip())

    if target == "normal_pdf":
        mu = params.get("mu", 0.0)
        sigma = params.get("sigma", 1.0)
        x = params.get("x", 0.0)
        z = (x - mu) / sigma
        pdf = math.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))
        return ToolResult(
            data={
                "pdf": pdf,
                "mu": mu,
                "sigma": sigma,
                "x": x,
            },
            success=True,
        )

    if target == "normal_cdf":
        mu = params.get("mu", 0.0)
        sigma = params.get("sigma", 1.0)
        x = params.get("x", 0.0)
        z = (x - mu) / sigma
        # Abramowitz & Stegun 近似
        t = 1.0 / (1.0 + 0.2316419 * abs(z))
        poly = t * (
            0.319381530
            + t
            * (
                -0.356563782
                + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))
            )
        )
        phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        cdf = phi * poly if z < 0.0 else 1.0 - phi * poly
        return ToolResult(
            data={
                "cdf": cdf,
                "mu": mu,
                "sigma": sigma,
                "x": x,
            },
            success=True,
        )

    if target == "gp_kernel":
        kernel_type = (args.equations or ["rbf"])[0] if args.equations else "rbf"
        s = params.get("sigma", 1.0)
        lengthscale = params.get("lengthscale", 1.0)
        x1 = params.get("x1", 0.0)
        x2 = params.get("x2", 1.0)
        d = x1 - x2
        if kernel_type == "rbf":
            k = s * s * math.exp(-0.5 * d * d / (lengthscale * lengthscale))
        elif kernel_type == "matern32":
            sqrt3r_over_l = math.sqrt(3.0) * abs(d) / lengthscale
            k = s * s * (1.0 + sqrt3r_over_l) * math.exp(-sqrt3r_over_l)
        elif kernel_type == "matern52":
            sqrt5r_over_l = math.sqrt(5.0) * abs(d) / lengthscale
            k = (
                s
                * s
                * (
                    1.0
                    + sqrt5r_over_l
                    + (5.0 / 3.0) * sqrt5r_over_l * sqrt5r_over_l
                )
                * math.exp(-sqrt5r_over_l)
            )
        else:
            return ToolResult(
                data=None,
                success=False,
                error=f"Unknown kernel_type: {kernel_type}",
            )
        return ToolResult(
            data={
                "kernel_value": k,
                "kernel_type": kernel_type,
                "x1": x1,
                "x2": x2,
            },
            success=True,
        )

    if target == "monte_carlo_integral":
        a = params.get("a", 0.0)
        b = params.get("b", 1.0)
        n_samples = int(params.get("n", 100))
        # 默认 f(x) = x^2
        h = b - a
        total = 0.0
        for i in range(1, n_samples + 1):
            x = a + h * i / (n_samples + 1.0)
            total += x * x
        integral = h * total / n_samples
        return ToolResult(
            data={
                "integral": integral,
                "exact": (b**3 - a**3) / 3.0,
                "n_samples": n_samples,
            },
            success=True,
        )

    if target == "bayesian_update_normal":
        mu0 = params.get("mu0", 0.0)
        tau0_sq = params.get("tau0", 1.0)
        sigma_sq = params.get("sigma", 0.25)
        data_mean = params.get("data_mean", 2.0)
        n = params.get("n", 10.0)
        precision = n / sigma_sq + 1.0 / tau0_sq
        tau_n_sq = 1.0 / precision
        mu_n = tau_n_sq * (n * data_mean / sigma_sq + mu0 / tau0_sq)
        return ToolResult(
            data={
                "posterior_mean": mu_n,
                "posterior_variance": tau_n_sq,
                "prior_mean": mu0,
                "prior_variance": tau0_sq,
            },
            success=True,
        )

    return ToolResult(
        data=None, success=False, error=f"Unknown probability target: {target}"
    )


def unified(args: "SymbolicMathInput") -> ToolResult:
    """桥接 huginn.unified: 推导方程 + 多尺度桥.

    Targets:
      - list: 返回可用的 unified 模型.
      - derive: 从命名模型推导控制方程.
      - bridge: 跑多尺度桥 (dft-to-md 或 cauchy-born).
    """
    from huginn.unified import derive_equations, discretize, solve, solve_and_plot
    from huginn.unified.bridge import (
        ConstitutiveModel,
        cauchy_born_elasticity,
        dft_potential_to_md,
        md_stress_to_continuum,
    )
    from huginn.unified.models import get_model, list_models

    target = (args.target or "derive").lower()

    if target == "list":
        return ToolResult(data={"models": list_models()}, success=True)

    if target == "derive":
        model_name = args.expression
        if not model_name:
            return ToolResult(
                data=None,
                success=False,
                error="expression must be a unified model name for target=derive",
            )
        factory = get_model(model_name)
        if factory is None:
            return ToolResult(
                data=None,
                success=False,
                error=f"Unknown unified model: {model_name}. Available: {', '.join(list_models())}",
            )
        problem = factory()
        result = derive_equations(problem)

        def _serialize(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _serialize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_serialize(v) for v in obj]
            if hasattr(obj, "lhs") and hasattr(obj, "rhs"):  # Eq
                return {"lhs": str(obj.lhs), "rhs": str(obj.rhs)}
            return str(obj)

        equations = _serialize(result.get("equations", {}))
        energy_expr = None
        energy_latex = None
        if problem.energy is not None:
            energy_expr = str(problem.energy.expression)
            energy_latex = sp.latex(problem.energy.expression)

        return ToolResult(
            data={
                "model": model_name,
                "principle": result.get("principle"),
                "energy_expression": energy_expr,
                "energy_latex": energy_latex,
                "equations": equations,
            },
            success=True,
        )

    if target == "bridge":
        bridge_name = args.expression
        if not bridge_name:
            return ToolResult(
                data=None,
                success=False,
                error="expression must be a bridge name for target=bridge",
            )
        bridge_name = bridge_name.lower().replace("-", "_")

        if bridge_name == "dft_to_md":
            from huginn.unified.models import one_d_kohn_sham_dft

            dft_problem = one_d_kohn_sham_dft()
            bridge_result = dft_potential_to_md(dft_problem)
            potential = bridge_result.get("potential")
            return ToolResult(
                data={
                    "bridge": "dft_to_md",
                    "potential_name": potential.name if potential else None,
                    "parameters": potential.parameters if potential else {},
                },
                success=True,
            )

        if bridge_name == "cauchy_born":
            potential_expr = args.free_energy or args.expression
            if not potential_expr:
                return ToolResult(
                    data=None,
                    success=False,
                    error="For cauchy-born bridge, free_energy must be the potential expression",
                )
            sym_dict = {s: sp.Symbol(s) for s in args.symbols}
            sym_dict.update(
                {
                    "sin": sp.sin,
                    "cos": sp.cos,
                    "tan": sp.tan,
                    "exp": sp.exp,
                    "log": sp.log,
                    "sqrt": sp.sqrt,
                    "pi": sp.pi,
                }
            )
            try:
                expr = sp.sympify(potential_expr, locals=sym_dict)
            except Exception as e:
                return ToolResult(
                    data=None,
                    success=False,
                    error=f"Failed to parse potential expression: {e}",
                )
            potential = ConstitutiveModel(
                name="user_potential",
                expression=expr,
                parameters={
                    s: str(sym_dict[s]) for s in args.symbols if s in sym_dict
                },
            )
            bridge_result = cauchy_born_elasticity(potential)
            return ToolResult(
                data={
                    "bridge": "cauchy_born",
                    "result": {k: str(v) for k, v in bridge_result.items()},
                },
                success=True,
            )

        if bridge_name == "md_to_stress":
            bridge_result = md_stress_to_continuum()
            return ToolResult(
                data={
                    "bridge": "md_to_stress",
                    "result": {k: str(v) for k, v in bridge_result.items()},
                },
                success=True,
            )

        if bridge_name == "md_to_elasticity":
            # cauchy_born 的别名, 用原子对势
            potential_expr = args.free_energy or args.expression
            if not args.symbols or "r" not in args.symbols or not potential_expr:
                return ToolResult(
                    data=None,
                    success=False,
                    error="md_to_elasticity requires symbols including 'r' and free_energy containing the potential expression",
                )
            sym_dict = {s: sp.Symbol(s) for s in args.symbols}
            sym_dict.update(
                {
                    "sin": sp.sin,
                    "cos": sp.cos,
                    "tan": sp.tan,
                    "exp": sp.exp,
                    "log": sp.log,
                    "sqrt": sp.sqrt,
                    "pi": sp.pi,
                }
            )
            try:
                expr = sp.sympify(potential_expr, locals=sym_dict)
            except Exception as e:
                return ToolResult(
                    data=None,
                    success=False,
                    error=f"Failed to parse potential expression: {e}",
                )
            potential = ConstitutiveModel(
                name="user_pair_potential",
                expression=expr,
                parameters={
                    s: str(sym_dict[s]) for s in args.symbols if s in sym_dict
                },
            )
            bridge_result = cauchy_born_elasticity(potential)
            return ToolResult(
                data={
                    "bridge": "md_to_elasticity",
                    "result": {k: str(v) for k, v in bridge_result.items()},
                },
                success=True,
            )

        return ToolResult(
            data=None,
            success=False,
            error=f"Unknown bridge: {bridge_name}. Supported: dft_to_md, md_to_stress, md_to_elasticity",
        )

    if target == "discretize":
        model_name = args.expression
        if not model_name:
            return ToolResult(
                data=None,
                success=False,
                error="expression must be a unified model name for target=discretize",
            )
        factory = get_model(model_name)
        if factory is None:
            return ToolResult(
                data=None,
                success=False,
                error=f"Unknown unified model: {model_name}. Available: {', '.join(list_models())}",
            )
        problem = factory()
        method = (args.variable or "fem").lower()
        n = args.order if args.order >= 1 else 10
        try:
            disc = discretize(problem, method=method, n=n)
        except Exception as e:
            return ToolResult(
                data=None,
                success=False,
                error=f"Discretization failed: {e}",
            )
        return ToolResult(
            data={
                "model": model_name,
                "method": disc["method"],
                "n": n,
                "n_dof": disc["n_dof"],
                "stiffness_matrix": disc["stiffness_matrix"],
                "load_vector": disc["load_vector"],
                "mesh": disc["mesh"],
            },
            success=True,
        )

    if target == "solve":
        model_name = args.expression
        if not model_name:
            return ToolResult(
                data=None,
                success=False,
                error="expression must be a unified model name for target=solve",
            )
        factory = get_model(model_name)
        if factory is None:
            return ToolResult(
                data=None,
                success=False,
                error=f"Unknown unified model: {model_name}. Available: {', '.join(list_models())}",
            )
        problem = factory()
        method = (args.variable or "fem").lower()
        n = args.order if args.order >= 1 else 10
        try:
            sol = solve(problem, method=method, n=n)
        except Exception as e:
            return ToolResult(
                data=None,
                success=False,
                error=f"Solve failed: {e}",
            )
        return ToolResult(
            data={
                "model": model_name,
                "method": sol["method"],
                "n": n,
                "n_dof": sol["n_dof"],
                "mesh": sol["mesh"],
                "solution": sol["solution"],
                "residual": sol["residual"],
            },
            success=True,
        )

    if target == "solve_and_plot":
        model_name = args.expression
        if not model_name:
            return ToolResult(
                data=None,
                success=False,
                error="expression must be a unified model name for target=solve_and_plot",
            )
        factory = get_model(model_name)
        if factory is None:
            return ToolResult(
                data=None,
                success=False,
                error=f"Unknown unified model: {model_name}. Available: {', '.join(list_models())}",
            )
        problem = factory()
        method = (args.variable or "fem").lower()
        n = args.order if args.order >= 1 else 10
        output_path = args.output_path or "unified_solution.png"
        try:
            sol = solve_and_plot(
                problem, method=method, n=n, output_path=output_path
            )
        except Exception as e:
            return ToolResult(
                data=None,
                success=False,
                error=f"solve_and_plot failed: {e}",
            )
        return ToolResult(
            data={
                "model": model_name,
                "method": sol["method"],
                "n": n,
                "plot_path": sol["plot_path"],
                "n_dof": sol["n_dof"],
                "residual": sol["residual"],
            },
            success=True,
        )

    return ToolResult(
        data=None,
        success=False,
        error=f"Unknown unified target: {target}. Supported: list, derive, bridge, discretize, solve, solve_and_plot",
    )
