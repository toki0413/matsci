"""变分法类 action: euler_lagrange / functional_derivative / isoperimetric / noether.

Euler-Lagrange 方程 δS/δu = ∂L/∂u - d/dx(∂L/∂u') = 0
泛函导数 δF/δu
等周问题 (带约束的变分)
Noether 定理 (对称性 → 守恒流)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from huginn.types import ToolResult

from ._parsers import parse_symbols, safe_parse

if TYPE_CHECKING:
    from .tool import SymbolicMathInput


def _substitute_primes(expr_str: str, u_name: str) -> str:
    """把 u' 这种撇号导数记号替换成合法标识符 __u_prime__."""
    return expr_str.replace(f"{u_name}'", "__u_prime__")


def _substitute_partials(expr_str: str, u_name: str, x_names: list[str]) -> str:
    """把 u_x, u_y 这种偏导记号替换成 __u_x__, __u_y__ 占位符."""
    out = expr_str
    for xn in x_names:
        out = out.replace(f"{u_name}_{xn}", f"__u_{xn}__")
    return out


def euler_lagrange(args: "SymbolicMathInput") -> ToolResult:
    """Euler-Lagrange 方程: 给定 L(u, u', x) 推导 ∂L/∂u - d/dx(∂L/∂u') = 0.

    expression 给 L, symbols 必须含 u (场) 和 x (自变量).
    也支持场依赖多个自变量 (多变量 L(u, ∂u/∂x_i, x_i)) — 用 multivar 模式,
    此时 symbols 里前 N-1 个是 x_i, 最后一个是 u.
    """
    sym_dict = parse_symbols(args.symbols, args.assumptions)
    L_str = (args.expression or "").strip()
    if not L_str:
        return ToolResult(
            data=None, success=False, error="euler_lagrange needs L in expression"
        )

    target = (args.target or "single").lower()
    if target == "single":
        u_name = args.variable or "u"
        x_name = "x"
        u = sym_dict.get(u_name)
        x = sym_dict.get(x_name)
        if u is None or x is None:
            return ToolResult(
                data=None,
                success=False,
                error=f"Need symbols [{u_name}, {x_name}]",
            )
        # 把 L 当成 u(x) 的函数, u' 用 Derivative(u, x)
        u_func = sp.Function(u_name)(x)
        u_prime = sp.Derivative(u_func, x)
        # 字符串里 u' 不是合法标识符, 先替换成占位符
        L_sub = _substitute_primes(L_str, u_name)
        try:
            L = safe_parse(L_sub, {**sym_dict, u_name: u_func, "__u_prime__": u_prime})
        except Exception as exc:
            return ToolResult(data=None, success=False, error=f"Parse error: {exc}")

        dL_du = sp.diff(L, u_func)
        dL_du_prime = sp.diff(L, u_prime)
        ddx_dL_du_prime = sp.diff(dL_du_prime, x)
        el = sp.simplify(dL_du - ddx_dL_du_prime)
        return ToolResult(
            data={
                "lagrangian": str(L),
                "lagrangian_latex": sp.latex(L),
                "dL_du": str(dL_du),
                "dL_du_prime": str(dL_du_prime),
                "d_dx_dL_du_prime": str(ddx_dL_du_prime),
                "euler_lagrange": f"{sp.simplify(el)} = 0",
                "euler_lagrange_latex": sp.latex(el),
            },
            success=True,
        )

    if target == "multivar":
        # 多变量: L(u, ∂u/∂x_i, x_i);  EL = ∂L/∂u - Σ_i d/dx_i (∂L/∂(∂u/∂x_i)) = 0
        if not args.symbols:
            return ToolResult(
                data=None, success=False, error="multivar needs symbols [x1, x2, ..., u]"
            )
        var_names = list(args.symbols)
        u_name = args.variable or var_names[-1]
        x_names = [n for n in var_names if n != u_name]
        if not x_names:
            return ToolResult(
                data=None, success=False, error="Need at least one spatial variable"
            )
        x_syms = [sym_dict[n] for n in x_names]
        u_func = sp.Function(u_name)(*x_syms)
        # 替换 u_x, u_y 占位符
        L_sub = _substitute_partials(L_str, u_name, x_names)
        local = {**sym_dict, u_name: u_func}
        for xn in x_names:
            local[f"__u_{xn}__"] = sp.Derivative(u_func, sym_dict[xn])
        try:
            L = safe_parse(L_sub, local)
        except Exception as exc:
            return ToolResult(data=None, success=False, error=f"Parse error: {exc}")

        dL_du = sp.diff(L, u_func)
        el_sum = dL_du
        for xn, xs in zip(x_names, x_syms):
            du_i = sp.Derivative(u_func, xs)
            dL_du_i = sp.diff(L, du_i)
            term = sp.diff(dL_du_i, xs)
            el_sum -= term
        el = sp.simplify(el_sum)
        return ToolResult(
            data={
                "lagrangian": str(L),
                "lagrangian_latex": sp.latex(L),
                "spatial_vars": x_names,
                "field": u_name,
                "euler_lagrange": f"{el} = 0",
                "euler_lagrange_latex": sp.latex(el),
                "note": "Multi-variable Euler-Lagrange with Σ_i d/dx_i (∂L/∂(∂u/∂x_i)).",
            },
            success=True,
        )

    return ToolResult(
        data=None, success=False, error=f"Unknown euler_lagrange target: {target}"
    )


def functional_derivative(args: "SymbolicMathInput") -> ToolResult:
    """泛函导数 δF/δu.

    F[u] = ∫ L(u, u', x) dx  →  δF/δu = ∂L/∂u - d/dx(∂L/∂u')
    实际上等价于 EL 方程的左端. 这里作为独立接口方便调用.
    """
    import copy

    args = copy.copy(args)
    args.target = "single"
    res = euler_lagrange(args)
    if not res.success:
        return res
    # 重命名键
    data = dict(res.data)
    data["functional_derivative"] = data.pop("euler_lagrange")
    data["functional_derivative_latex"] = data.pop("euler_lagrange_latex")
    data["note"] = "δF/δu equals the Euler-Lagrange left-hand side."
    return ToolResult(data=data, success=True)


def isoperimetric(args: "SymbolicMathInput") -> ToolResult:
    """等周问题: 在约束 ∫ G dx = const 下极值化 ∫ F dx.

    用 Lagrange 乘子 λ 构造增广泛函 ∫ (F - λ G) dx, 然后对增广 L 解 EL 方程.
    expression 用 ';' 分隔 F 和 G, 例如 "sqrt(1 + u'^2);u^2".
    symbols 必须含 u 和 x, λ 自动加入.
    """
    sym_dict = parse_symbols(args.symbols, args.assumptions)
    raw = (args.expression or "").strip()
    if not raw:
        return ToolResult(
            data=None, success=False, error="isoperimetric needs 'F;G' in expression"
        )
    parts = [p.strip() for p in raw.split(";")]
    if len(parts) != 2:
        return ToolResult(
            data=None,
            success=False,
            error="Expected 2 semicolon-separated values F;G",
        )

    u_name = args.variable or "u"
    x_name = "x"
    u = sym_dict.get(u_name)
    x = sym_dict.get(x_name)
    if u is None or x is None:
        return ToolResult(
            data=None, success=False, error=f"Need symbols [{u_name}, {x_name}]"
        )
    u_func = sp.Function(u_name)(x)
    u_prime = sp.Derivative(u_func, x)
    lam = sp.Symbol("lambda")

    F_sub = _substitute_primes(parts[0], u_name)
    G_sub = _substitute_primes(parts[1], u_name)
    try:
        F = safe_parse(F_sub, {**sym_dict, u_name: u_func, "__u_prime__": u_prime})
        G = safe_parse(G_sub, {**sym_dict, u_name: u_func, "__u_prime__": u_prime})
    except Exception as exc:
        return ToolResult(data=None, success=False, error=f"Parse error: {exc}")

    L_aug = sp.simplify(F - lam * G)
    dL_du = sp.diff(L_aug, u_func)
    dL_du_prime = sp.diff(L_aug, u_prime)
    ddx_dL_du_prime = sp.diff(dL_du_prime, x)
    el = sp.simplify(dL_du - ddx_dL_du_prime)

    return ToolResult(
        data={
            "functional_F": str(F),
            "constraint_G": str(G),
            "lagrange_multiplier": "lambda",
            "augmented_lagrangian": str(L_aug),
            "augmented_lagrangian_latex": sp.latex(L_aug),
            "euler_lagrange": f"{el} = 0",
            "euler_lagrange_latex": sp.latex(el),
            "note": "Solve EL for u, then enforce ∫ G dx = const to determine λ.",
        },
        success=True,
    )


def noether(args: "SymbolicMathInput") -> ToolResult:
    """Noether 定理: 连续对称性 → 守恒流.

    若 L 在变换 u → u + ε η(x) 下不变 (δL = 0), 则
      J = η · ∂L/∂u'  守恒 (dJ/dx = 0).

    expression 给 L, target 给对称类型:
      - 'translation': u → u + ε  (η = 1)
      - 'scaling':     u → (1+ε) u  (η = u)
      - 'custom':      η 由 args.sub_action 给出
    """
    sym_dict = parse_symbols(args.symbols, args.assumptions)
    L_str = (args.expression or "").strip()
    if not L_str:
        return ToolResult(
            data=None, success=False, error="noether needs L in expression"
        )

    u_name = args.variable or "u"
    x_name = "x"
    u = sym_dict.get(u_name)
    x = sym_dict.get(x_name)
    if u is None or x is None:
        return ToolResult(
            data=None, success=False, error=f"Need symbols [{u_name}, {x_name}]"
        )

    target = (args.target or "translation").lower()
    if target == "translation":
        eta = sp.Integer(1)
        symmetry = "u → u + ε"
    elif target == "scaling":
        eta = u
        symmetry = "u → (1 + ε) u"
    elif target == "custom":
        eta_str = args.sub_action or "1"
        try:
            eta = safe_parse(eta_str, sym_dict)
        except Exception as exc:
            return ToolResult(data=None, success=False, error=f"Parse eta: {exc}")
        symmetry = f"u → u + ε ({eta})"
    else:
        return ToolResult(
            data=None, success=False, error=f"Unknown noether target: {target}"
        )

    u_func = sp.Function(u_name)(x)
    u_prime = sp.Derivative(u_func, x)
    L_sub = _substitute_primes(L_str, u_name)
    try:
        L = safe_parse(L_sub, {**sym_dict, u_name: u_func, "__u_prime__": u_prime})
    except Exception as exc:
        return ToolResult(data=None, success=False, error=f"Parse error: {exc}")

    dL_du_prime = sp.diff(L, u_prime)
    # 把 eta 里的 u 也替换成 u_func
    eta_sub = eta.subs(u, u_func) if hasattr(eta, "subs") else eta
    J = sp.simplify(eta_sub * dL_du_prime)

    return ToolResult(
        data={
            "lagrangian": str(L),
            "symmetry": symmetry,
            "eta": str(eta),
            "conserved_current": f"J = {J}",
            "conserved_current_latex": sp.latex(J),
            "conservation_law": "dJ/dx = 0",
            "note": "Noether: δL = 0 under continuous symmetry ⇒ J = η ∂L/∂u' is conserved.",
        },
        success=True,
    )
