"""微分几何类 action: metric / geodesic / curvature / lie_derivative / connection.

全部基于 SymPy 符号推导, 不依赖外部微分几何库.
覆盖: 度量张量 → Christoffel 记号 → Riemann/Ricci 曲率;
      测地线方程; 高斯曲率; 李导数; Levi-Civita 联络系数.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sympy as sp

from huginn.core_types import ToolResult

from ._parsers import parse_symbols, safe_parse

if TYPE_CHECKING:
    from .tool import SymbolicMathInput


def _parse_metric_matrix(
    args: SymbolicMathInput,
) -> tuple[list[sp.Expr], list[sp.Symbol]]:
    """从 args.matrix (list[list[str]]) 解析出度规矩阵和坐标变量."""
    sym_dict = parse_symbols(args.symbols, args.assumptions)
    if not args.matrix:
        raise ValueError("Need metric matrix in args.matrix")
    coords = [sym_dict[n] for n in args.symbols]
    n = len(args.symbols)
    if len(args.matrix) != n or any(len(row) != n for row in args.matrix):
        raise ValueError(
            f"Metric must be {n}x{n}, got {len(args.matrix)}x{len(args.matrix[0]) if args.matrix else 0}"
        )
    g = [[safe_parse(str(cell), sym_dict) for cell in row] for row in args.matrix]
    return g, coords


def _christoffel(
    g: list[list[sp.Expr]], coords: list[sp.Symbol]
) -> list[list[list[sp.Expr]]]:
    """计算第二类 Christoffel 记号 Γ^k_{ij}."""
    n = len(coords)
    g_inv = sp.Matrix(g).inv()
    Gamma = [[[sp.Integer(0) for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                s = sp.Integer(0)
                for l in range(n):  # noqa: E741
                    # Γ^k_{ij} = (1/2) g^{kl} (∂_i g_{jl} + ∂_j g_{il} - ∂_l g_{ij})
                    s += g_inv[k, l] * (
                        sp.diff(g[j][l], coords[i])
                        + sp.diff(g[i][l], coords[j])
                        - sp.diff(g[i][j], coords[l])
                    )
                Gamma[k][i][j] = sp.simplify(s / 2)
    return Gamma


def _metric_base(g: Any, coords: Any, n: int) -> dict:
    """度规结果公共字段: metric / coords / dim."""
    return {
        "metric": [[str(c) for c in row] for row in g],
        "coords": [str(c) for c in coords],
        "dim": n,
    }


def _christoffel_nonzero(Gamma: Any, coords: Any, n: int) -> list[dict]:
    """输出非零 Christoffel 记号."""
    nonzero = []
    for k in range(n):
        for i in range(n):
            for j in range(i, n):
                val = Gamma[k][i][j]
                if val != 0:
                    nonzero.append(
                        {
                            "index": f"\u0393^{coords[k]}_{{{coords[i]}{coords[j]}}}",
                            "value": str(val),
                            "latex": sp.latex(val),
                        }
                    )
    return nonzero


def _riemann_tensor(Gamma: Any, coords: Any, n: int) -> Any:
    """Riemann 张量 R^\u03c1_{\u03c3\u03bc\u03bd} =
    \u2202_\u03bc \u0393^\u03c1_{\u03bd\u03c3} - \u2202_\u03bd \u0393^\u03c1_{\u03bc\u03c3}
    + \u0393^\u03c1_{\u03bc\u03bb} \u0393^\u03bb_{\u03bd\u03c3}
    - \u0393^\u03c1_{\u03bd\u03bb} \u0393^\u03bb_{\u03bc\u03c3}."""
    Riemann = [
        [[[sp.Integer(0) for _ in range(n)] for _ in range(n)] for _ in range(n)]
        for _ in range(n)
    ]
    for rho in range(n):
        for sigma in range(n):
            for mu in range(n):
                for nu in range(n):
                    val = sp.diff(Gamma[rho][nu][sigma], coords[mu]) - sp.diff(
                        Gamma[rho][mu][sigma], coords[nu]
                    )
                    for lam in range(n):
                        val += (
                            Gamma[rho][mu][lam] * Gamma[lam][nu][sigma]
                            - Gamma[rho][nu][lam] * Gamma[lam][mu][sigma]
                        )
                    Riemann[rho][sigma][mu][nu] = sp.simplify(val)
    return Riemann


def _ricci_tensor(Riemann: Any, n: int) -> Any:
    """Ricci 张量 R_{\u03c3\u03bd} = R^\u03c1_{\u03c3\u03c1\u03bd}."""
    Ricci = [[sp.Integer(0) for _ in range(n)] for _ in range(n)]
    for sigma in range(n):
        for nu in range(n):
            s = sp.Integer(0)
            for rho in range(n):
                s += Riemann[rho][sigma][rho][nu]
            Ricci[sigma][nu] = sp.simplify(s)
    return Ricci


def _scalar_curvature(g_inv: Any, Ricci: Any, n: int) -> Any:
    """标量曲率 R = g^{\u03c3\u03bd} R_{\u03c3\u03bd}."""
    scalar = sp.Integer(0)
    for sigma in range(n):
        for nu in range(n):
            scalar += g_inv[sigma, nu] * Ricci[sigma][nu]
    return sp.simplify(scalar)


def _curvature_data(g: Any, coords: Any, n: int, Gamma: Any, target: str) -> dict:
    """Ricci / 标量曲率 (target == all 时附 Christoffel 全矩阵)."""
    Riemann = _riemann_tensor(Gamma, coords, n)
    Ricci = _ricci_tensor(Riemann, n)
    g_inv = sp.Matrix(g).inv()
    scalar = _scalar_curvature(g_inv, Ricci, n)
    data: dict = {
        **_metric_base(g, coords, n),
        "ricci_tensor": [[str(Ricci[i][j]) for j in range(n)] for i in range(n)],
        "ricci_scalar": str(scalar),
        "ricci_scalar_latex": sp.latex(scalar),
    }
    if target == "all":
        data["christoffel"] = [
            [[str(Gamma[k][i][j]) for j in range(n)] for i in range(n)]
            for k in range(n)
        ]
    return data


def metric(args: SymbolicMathInput) -> ToolResult:
    """从度规矩阵计算 Christoffel 记号、Ricci 张量、标量曲率.

    args.matrix: n×n 对称矩阵 (list[list[str]])
    args.symbols: 坐标变量名 (e.g. ["t", "r", "theta", "phi"])
    args.target: 'christoffel' (默认) | 'ricci' | 'scalar' | 'all'
    """
    try:
        g, coords = _parse_metric_matrix(args)
    except Exception as exc:
        return ToolResult(data=None, success=False, error=str(exc))

    target = (args.target or "christoffel").lower()
    n = len(coords)
    Gamma = _christoffel(g, coords)

    if target == "christoffel":
        nonzero = _christoffel_nonzero(Gamma, coords, n)
        data = _metric_base(g, coords, n)
        data["christoffel_nonzero"] = nonzero
        data["n_nonzero"] = len(nonzero)
        return ToolResult(data=data, success=True)

    if target in ("ricci", "scalar", "all"):
        return ToolResult(
            data=_curvature_data(g, coords, n, Gamma, target), success=True
        )

    return ToolResult(
        data=None, success=False, error=f"Unknown metric target: {target}"
    )


def geodesic(args: SymbolicMathInput) -> ToolResult:
    """从度规推导测地线方程 d²x^k/ds² + Γ^k_{ij} dx^i/ds dx^j/ds = 0."""
    try:
        g, coords = _parse_metric_matrix(args)
    except Exception as exc:
        return ToolResult(data=None, success=False, error=str(exc))

    n = len(coords)
    Gamma = _christoffel(g, coords)
    equations = []
    for k in range(n):
        s = sp.Integer(0)
        for i in range(n):
            for j in range(n):
                s += (
                    Gamma[k][i][j]
                    * sp.Symbol(f"d{coords[i]}/ds")
                    * sp.Symbol(f"d{coords[j]}/ds")
                )
        eq = f"d²{coords[k]}/ds² + {sp.simplify(s)} = 0"
        equations.append(
            {"coord": str(coords[k]), "equation": eq, "latex": sp.latex(s)}
        )

    return ToolResult(
        data={
            "metric": [[str(c) for c in row] for row in g],
            "coords": [str(c) for c in coords],
            "geodesic_equations": equations,
            "n_equations": n,
        },
        success=True,
    )


def _cross3(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _curvature_coefficients(X, Y, Z, u, v):
    """返回 (E, F, G, L, M, N)；退化曲面返回 (None, 错误结果)."""
    # 第一基本形式 E, F, G
    r_u = [sp.diff(X, u), sp.diff(Y, u), sp.diff(Z, u)]
    r_v = [sp.diff(X, v), sp.diff(Y, v), sp.diff(Z, v)]
    E = sum(a * b for a, b in zip(r_u, r_u))
    F = sum(a * b for a, b in zip(r_u, r_v))
    G = sum(a * b for a, b in zip(r_v, r_v))

    # 第二基本形式 L, M, N  (用 r_uu, r_uv, r_vv 与法向 n = r_u × r_v)
    r_uu = [sp.diff(c, u, 2) for c in [X, Y, Z]]
    r_uv = [sp.diff(c, u, v) for c in [X, Y, Z]]
    r_vv = [sp.diff(c, v, 2) for c in [X, Y, Z]]

    # 叉积 r_u × r_v
    n_vec = _cross3(r_u, r_v)
    n_norm_sq = sum(c * c for c in n_vec)
    if n_norm_sq == 0:
        return None, ToolResult(
            success=False,
            error="退化曲面: r_u × r_v = 0, 法向量不存在",
        )
    n_norm = sp.sqrt(n_norm_sq)

    # L = r_uu · n, M = r_uv · n, N = r_vv · n (用未归一化法向)
    L = sum(a * b for a, b in zip(r_uu, n_vec)) / n_norm
    M = sum(a * b for a, b in zip(r_uv, n_vec)) / n_norm
    N = sum(a * b for a, b in zip(r_vv, n_vec)) / n_norm

    return (E, F, G, L, M, N), None


def _curvature_result(X, Y, Z, E, F, G, L, M, N, K, H) -> ToolResult:
    return ToolResult(
        data={
            "parameterization": {"X": str(X), "Y": str(Y), "Z": str(Z)},
            "first_fundamental_form": {
                "E": str(sp.simplify(E)),
                "F": str(sp.simplify(F)),
                "G": str(sp.simplify(G)),
            },
            "second_fundamental_form": {
                "L": str(sp.simplify(L)),
                "M": str(sp.simplify(M)),
                "N": str(sp.simplify(N)),
            },
            "gaussian_curvature": str(K),
            "mean_curvature": str(H),
            "gaussian_curvature_latex": sp.latex(K),
        },
        success=True,
    )


def curvature(args: SymbolicMathInput) -> ToolResult:
    """计算曲面高斯曲率 K 和平均曲率 H.

    输入参数化曲面 r(u, v) = (X, Y, Z), 用 args.expression 给 'X;Y;Z',
    args.symbols 给 ['u', 'v'].
    """
    sym_dict = parse_symbols(args.symbols, args.assumptions)
    raw = (args.expression or "").strip()
    parts = [p.strip() for p in raw.split(";")]
    if len(parts) != 3:
        return ToolResult(
            data=None,
            success=False,
            error="curvature needs 'X;Y;Z' parameterization in expression",
        )
    if len(args.symbols) != 2:
        return ToolResult(
            data=None, success=False, error="Need exactly 2 parameters (u, v)"
        )
    u, v = sym_dict[args.symbols[0]], sym_dict[args.symbols[1]]
    try:
        X = safe_parse(parts[0], sym_dict)
        Y = safe_parse(parts[1], sym_dict)
        Z = safe_parse(parts[2], sym_dict)
    except Exception as exc:
        return ToolResult(data=None, success=False, error=f"Parse error: {exc}")

    coefficients, error = _curvature_coefficients(X, Y, Z, u, v)
    if error is not None:
        return error
    E, F, G, L, M, N = coefficients

    # 高斯曲率 K = (LN - M²) / (EG - F²)
    # 平均曲率 H = (EN - 2FM + GL) / (2(EG - F²))
    denom = sp.simplify(E * G - F**2)
    if denom == 0:
        return ToolResult(
            success=False,
            error="退化第一基本形式: EG - F² = 0, 曲率未定义",
        )
    K = sp.simplify((L * N - M**2) / denom)
    H = sp.simplify((E * N - 2 * F * M + G * L) / (2 * denom))

    return _curvature_result(X, Y, Z, E, F, G, L, M, N, K, H)


def _field_parse_error(message: str):
    """构造 (None, None, 错误结果) 三元组."""
    return None, None, ToolResult(data=None, success=False, error=message)


def _flatten_field_matrix(matrix) -> list[str]:
    """把 list[list[str]] 每行展平成一维分量列表."""
    x_raw: list[str] = []
    for row in matrix:
        if isinstance(row, list):
            x_raw.extend(row)
        else:
            x_raw.append(str(row))
    return x_raw


def _lie_derivative_fields(args, sym_dict, n):
    """解析 X 向量场与 Y 向量场；失败返回 (None, None, 错误结果)."""
    if not args.matrix:
        return _field_parse_error("X field (matrix) is empty")
    x_raw = _flatten_field_matrix(args.matrix)
    if len(x_raw) != n:
        return _field_parse_error(
            f"X field (matrix) must have {n} components, got {len(x_raw)}"
        )
    raw = (args.expression or "").strip()
    parts = [p.strip() for p in raw.split(";")]
    if len(parts) != n:
        return _field_parse_error(
            f"Y field (expression) must have {n} components separated by ';'"
        )
    try:
        X = [safe_parse(str(c), sym_dict) for c in x_raw]
        Y = [safe_parse(p, sym_dict) for p in parts]
    except Exception as exc:
        return _field_parse_error(f"Parse error: {exc}")
    return X, Y, None


def _lie_bracket(X, Y, coords):
    # [X, Y]^i = Σ_j (X^j ∂_j Y^i - Y^j ∂_j X^i)
    n = len(coords)
    bracket = []
    for i in range(n):
        s = sp.Integer(0)
        for j in range(n):
            s += X[j] * sp.diff(Y[i], coords[j]) - Y[j] * sp.diff(X[i], coords[j])
        bracket.append(sp.simplify(s))
    return bracket


def lie_derivative(args: SymbolicMathInput) -> ToolResult:
    """李导数 L_X Y = [X, Y] = X·∇Y - Y·∇X (向量场李括号).

    args.symbols: 坐标变量名 (e.g. ["x", "y"])
    args.matrix: X 向量场分量 (list[str], 长度 n)
    args.expression: Y 向量场分量, 用 ';' 分隔
    """
    sym_dict = parse_symbols(args.symbols, args.assumptions)
    n = len(args.symbols)
    coords = [sym_dict[name] for name in args.symbols]
    X, Y, error = _lie_derivative_fields(args, sym_dict, n)
    if error is not None:
        return error

    bracket = _lie_bracket(X, Y, coords)

    return ToolResult(
        data={
            "X": [str(x) for x in X],
            "Y": [str(y) for y in Y],
            "coords": [str(c) for c in coords],
            "lie_bracket": [str(b) for b in bracket],
            "lie_bracket_latex": [sp.latex(b) for b in bracket],
            "note": "L_X Y = [X, Y] is the Lie derivative of Y along X.",
        },
        success=True,
    )


def connection(args: SymbolicMathInput) -> ToolResult:
    """计算 Levi-Civita 联络系数 (Christoffel 第一类 + 第二类)."""
    try:
        g, coords = _parse_metric_matrix(args)
    except Exception as exc:
        return ToolResult(data=None, success=False, error=str(exc))

    n = len(coords)
    g_matrix = sp.Matrix(g)
    g_matrix.inv()
    # 第二类
    Gamma2 = _christoffel(g, coords)
    # 第一类 Γ_{kij} = g_{kl} Γ^l_{ij}
    Gamma1 = [[[sp.Integer(0) for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                s = sp.Integer(0)
                for l in range(n):  # noqa: E741
                    s += g[k][l] * Gamma2[l][i][j]
                Gamma1[k][i][j] = sp.simplify(s)

    return ToolResult(
        data={
            "metric": [[str(c) for c in row] for row in g],
            "coords": [str(c) for c in coords],
            "dim": n,
            "christoffel_first_kind": [
                [[str(Gamma1[k][i][j]) for j in range(n)] for i in range(n)]
                for k in range(n)
            ],
            "christoffel_second_kind": [
                [[str(Gamma2[k][i][j]) for j in range(n)] for i in range(n)]
                for k in range(n)
            ],
            "note": "Γ_{kij} = g_{kl} Γ^l_{ij};  Γ^k_{ij} = (1/2) g^{kl}(∂_i g_{jl} + ∂_j g_{il} - ∂_l g_{ij})",
        },
        success=True,
    )
