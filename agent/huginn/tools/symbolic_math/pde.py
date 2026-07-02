"""PDE 类 action: classify / separation / characteristics / discretize.

覆盖二阶线性 PDE 分类 (椭圆/抛物/双曲)、分离变量法、一阶 PDE 特征线法、
有限差分 stencil 自动生成. 全部用 SymPy 符号推导, 不调用数值库.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sympy as sp

from huginn.types import ToolResult

from ._parsers import parse_symbols, safe_parse

if TYPE_CHECKING:
    from .tool import SymbolicMathInput


def _build_function(expr_str: str, sym_dict: dict[str, sp.Symbol]) -> sp.Expr:
    """把字符串解析成 SymPy 表达式, 复用 safe_parse."""
    return safe_parse(expr_str, sym_dict)


def classify(args: "SymbolicMathInput") -> ToolResult:
    """按判别式 B^2 - 4AC 给二阶线性 PDE 分类.

    PDE 形式: A u_xx + 2B u_xy + C u_yy + (低阶项) = 0
    用户在 expression 里给 A、B、C 三项, 用 ; 分隔, 例如 "1;0;1" 表示拉普拉斯.
    也可以在 symbols 里给 A,B,C 然后直接传 "A;B;C".
    """
    sym_dict = parse_symbols(args.symbols, args.assumptions)
    raw = (args.expression or "").strip()
    if not raw:
        return ToolResult(
            data=None,
            success=False,
            error="classify needs expression like '1;0;1' (A;B;C)",
        )

    parts = [p.strip() for p in raw.split(";")]
    if len(parts) != 3:
        return ToolResult(
            data=None,
            success=False,
            error="Expected 3 semicolon-separated values A;B;C",
        )

    try:
        A = _build_function(parts[0], sym_dict)
        B = _build_function(parts[1], sym_dict)
        C = _build_function(parts[2], sym_dict)
    except Exception as exc:
        return ToolResult(data=None, success=False, error=f"Parse error: {exc}")

    # 判别式: 计算前先替换任何符号到自由变量, 用 sp.N 试求值
    disc_expr = sp.expand(B**2 - 4 * A * C)
    try:
        disc_val = complex(sp.N(disc_expr))
    except Exception:
        # 符号判别式: 看是否能定号
        disc_val = None

    if disc_val is not None:
        if abs(disc_val.imag) > 1e-12:
            classification = "indeterminate (complex discriminant)"
        elif disc_val.real < -1e-12:
            classification = "elliptic"
        elif abs(disc_val.real) < 1e-12:
            classification = "parabolic"
        else:
            classification = "hyperbolic"
    else:
        classification = "symbolic (sign depends on parameters)"

    examples = {
        "elliptic": "Laplace / Poisson: u_xx + u_yy = 0",
        "parabolic": "Heat: u_t - k u_xx = 0",
        "hyperbolic": "Wave: u_tt - c^2 u_xx = 0",
    }

    return ToolResult(
        data={
            "A": str(A),
            "B": str(B),
            "C": str(C),
            "discriminant": str(disc_expr),
            "discriminant_value": disc_val,
            "classification": classification,
            "canonical_form": examples.get(
                classification.split()[0], "n/a"
            ),
        },
        success=True,
    )


def separation(args: "SymbolicMathInput") -> ToolResult:
    """分离变量法: u(x,t) = X(x) T(t).

    target 选模式:
      - heat:    u_t = k u_xx      →  X'' + λ X = 0,  T' + λ k T = 0
      - wave:    u_tt = c^2 u_xx   →  X'' + λ X = 0,  T'' + λ c^2 T = 0
      - laplace: u_xx + u_yy = 0   →  X'' + λ X = 0,  Y'' - λ Y = 0
    返回分离后的两个 ODE 和通解结构.
    """
    sym_dict = parse_symbols(args.symbols, args.assumptions)
    target = (args.target or "heat").lower()
    lam = sp.Symbol("lambda", positive=True)

    x = sym_dict.get("x", sp.Symbol("x"))
    t = sym_dict.get("t", sp.Symbol("t"))
    X = sp.Function("X")
    T = sp.Function("T")

    ode_X_str = "X''(x) + λ X(x) = 0"
    ode_T_str: str
    general_form: str
    eigenvalues: str
    eigenfunctions: str

    if target == "heat":
        k = sym_dict.get("k", sp.Symbol("k", positive=True))
        # u_t = k u_xx;  X T' = k X'' T  →  X''/X = T'/(kT) = -λ
        ode_T_str = f"T'(t) + {lam} * {k} * T(t) = 0"
        general_form = "u(x,t) = Σ_n A_n sin(sqrt(λ_n) x) exp(-λ_n k t)"
        eigenvalues = "λ_n = (n π / L)^2  (Dirichlet BC on [0, L])"
        eigenfunctions = "X_n(x) = sin(n π x / L)"
    elif target == "wave":
        c = sym_dict.get("c", sp.Symbol("c", positive=True))
        # u_tt = c^2 u_xx;  X T'' = c^2 X'' T  →  X''/X = T''/(c^2 T) = -λ
        ode_T_str = f"T''(t) + {lam} * {c}**2 * T(t) = 0"
        general_form = (
            "u(x,t) = Σ_n [A_n cos(c sqrt(λ_n) t) + B_n sin(c sqrt(λ_n) t)] "
            "sin(sqrt(λ_n) x)"
        )
        eigenvalues = "λ_n = (n π / L)^2  (Dirichlet BC on [0, L])"
        eigenfunctions = "X_n(x) = sin(n π x / L)"
    elif target == "laplace":
        y = sym_dict.get("y", sp.Symbol("y"))
        Y = sp.Function("Y")
        # u_xx + u_yy = 0;  X''/X = -Y''/Y = -λ
        ode_T_str = f"Y''(y) - {lam} * Y(y) = 0"
        general_form = (
            "u(x,y) = Σ_n [A_n cosh(sqrt(λ_n) y) + B_n sinh(sqrt(λ_n) y)] "
            "sin(sqrt(λ_n) x)"
        )
        eigenvalues = "λ_n = (n π / a)^2  (Dirichlet BC on x in [0, a])"
        eigenfunctions = "X_n(x) = sin(n π x / a)"
        ode_X_str = "X''(x) + λ X(x) = 0"
    else:
        return ToolResult(
            data=None,
            success=False,
            error=f"Unknown separation target: {target}",
        )

    return ToolResult(
        data={
            "pde_type": target,
            "ansatz": "u = X(x) * T(t)" if target != "laplace" else "u = X(x) * Y(y)",
            "spatial_ode": ode_X_str,
            "temporal_ode": ode_T_str,
            "separation_constant": "lambda",
            "general_solution": general_form,
            "eigenvalues": eigenvalues,
            "eigenfunctions": eigenfunctions,
        },
        success=True,
    )


def characteristics(args: "SymbolicMathInput") -> ToolResult:
    """一阶 PDE 的特征线法.

    覆盖两类:
      - target='first_order_linear': a(x,y) u_x + b(x,y) u_y = f(x,y,u)
        特征方程 dx/ds = a, dy/ds = b, du/ds = f
      - target='transport': c u_x + u_t = 0  (常系数输运方程)
        特征线 x - c t = const, 通解 u(x,t) = F(x - c t)
    expression 给 a, b, f (用 ; 分隔), 或者 c (transport 模式).
    """
    sym_dict = parse_symbols(args.symbols, args.assumptions)
    target = (args.target or "first_order_linear").lower()
    raw = (args.expression or "").strip()

    if target == "transport":
        # 输运方程: u_t + c u_x = 0
        try:
            c = _build_function(raw, sym_dict) if raw else sp.Symbol("c")
        except Exception as exc:
            return ToolResult(data=None, success=False, error=f"Parse error: {exc}")
        x = sym_dict.get("x", sp.Symbol("x"))
        t = sym_dict.get("t", sp.Symbol("t"))
        return ToolResult(
            data={
                "pde": f"u_t + ({c}) u_x = 0",
                "characteristic_ode": "dx/dt = c",
                "invariant": "x - c t = const",
                "general_solution": f"u(x,t) = F(x - ({c}) t)",
                "characteristic_curves": f"x(t) = x0 + ({c}) t",
            },
            success=True,
        )

    if target == "first_order_linear":
        if not raw:
            return ToolResult(
                data=None,
                success=False,
                error="first_order_linear needs expression 'a;b;f'",
            )
        parts = [p.strip() for p in raw.split(";")]
        if len(parts) != 3:
            return ToolResult(
                data=None,
                success=False,
                error="Expected 3 semicolon-separated values a;b;f",
            )
        try:
            a_coef = _build_function(parts[0], sym_dict)
            b_coef = _build_function(parts[1], sym_dict)
            f_src = _build_function(parts[2], sym_dict)
        except Exception as exc:
            return ToolResult(data=None, success=False, error=f"Parse error: {exc}")
        return ToolResult(
            data={
                "pde": f"({a_coef}) u_x + ({b_coef}) u_y = {f_src}",
                "characteristic_system": [
                    "dx/ds = a(x,y)",
                    "dy/ds = b(x,y)",
                    "du/ds = f(x,y,u)",
                ],
                "a": str(a_coef),
                "b": str(b_coef),
                "f": str(f_src),
                "method": "Solve ODE system along characteristics, then invert mapping.",
            },
            success=True,
        )

    return ToolResult(
        data=None, success=False, error=f"Unknown characteristics target: {target}"
    )


def discretize(args: "SymbolicMathInput") -> ToolResult:
    """有限差分 stencil 生成.

    target 选模式:
      - 'laplacian_2d': 5 点 stencil for Δu on uniform grid (dx=dy=h)
      - 'laplacian_3d': 7 点 stencil
      - 'heat_ftcs': forward time centered space (显式热传导)
      - 'wave_explicit': 中心时间 + 中心空间 (显式波动)
    expression 给网格步长 h (默认 'h'), 时间步长用 dt (默认 'dt'), 波速 c.
    """
    sym_dict = parse_symbols(args.symbols, args.assumptions)
    target = (args.target or "laplacian_2d").lower()
    h = sym_dict.get("h", sp.Symbol("h"))
    dt = sym_dict.get("dt", sp.Symbol("dt"))

    if target == "laplacian_2d":
        # (u_{i-1,j} + u_{i+1,j} + u_{i,j-1} + u_{i,j+1} - 4 u_{i,j}) / h^2
        stencil = (
            "(u_{i-1,j} + u_{i+1,j} + u_{i,j-1} + u_{i,j+1} - 4 u_{i,j}) / h^2"
        )
        return ToolResult(
            data={
                "scheme": "5-point Laplacian (2D)",
                "stencil": stencil,
                "order": "O(h^2)",
                "stencil_points": [
                    "center:  -4/h^2",
                    "left:    +1/h^2",
                    "right:   +1/h^2",
                    "down:    +1/h^2",
                    "up:      +1/h^2",
                ],
                "stability": "Unconditionally stable for elliptic (Poisson) solve.",
            },
            success=True,
        )

    if target == "laplacian_3d":
        stencil = (
            "(u_{i-1,j,k} + u_{i+1,j,k} + u_{i,j-1,k} + u_{i,j+1,k} + "
            "u_{i,j,k-1} + u_{i,j,k+1} - 6 u_{i,j,k}) / h^2"
        )
        return ToolResult(
            data={
                "scheme": "7-point Laplacian (3D)",
                "stencil": stencil,
                "order": "O(h^2)",
                "stencil_points": [
                    "center:  -6/h^2",
                    "±x:      +1/h^2 (each)",
                    "±y:      +1/h^2 (each)",
                    "±z:      +1/h^2 (each)",
                ],
            },
            success=True,
        )

    if target == "heat_ftcs":
        # Forward Time Centered Space: u^{n+1}_i = u^n_i + (α dt / h^2)(u^n_{i-1} - 2u^n_i + u^n_{i+1})
        alpha = sym_dict.get("alpha", sp.Symbol("alpha"))
        r = alpha * dt / h**2
        return ToolResult(
            data={
                "scheme": "FTCS (Forward Time Centered Space)",
                "stencil": f"u^(n+1)_i = u^n_i + ({r}) (u^n_{{i-1}} - 2 u^n_i + u^n_{{i+1}})",
                "order": "O(dt, h^2)",
                "courant_number": f"r = α dt / h^2 = {r}",
                "stability": "Stable iff r = α dt / h^2 <= 1/2 (von Neumann)",
            },
            success=True,
        )

    if target == "wave_explicit":
        c = sym_dict.get("c", sp.Symbol("c"))
        CFL = c * dt / h
        return ToolResult(
            data={
                "scheme": "Central-difference (explicit wave)",
                "stencil": (
                    f"u^(n+1)_i = 2 u^n_i - u^(n-1)_i + ({CFL})^2 "
                    f"(u^n_{{i-1}} - 2 u^n_i + u^n_{{i+1}})"
                ),
                "order": "O(dt^2, h^2)",
                "courant_number": f"CFL = c dt / h = {CFL}",
                "stability": "Stable iff CFL = c dt / h <= 1 (CFL condition)",
            },
            success=True,
        )

    return ToolResult(
        data=None, success=False, error=f"Unknown discretize target: {target}"
    )
