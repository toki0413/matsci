"""代数类 action: solve / eigenvalue / linear_algebra."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sympy as sp

from huginn.core_types import ToolResult

from ._parsers import parse_symbols, safe_parse

if TYPE_CHECKING:
    from .tool import SymbolicMathInput


def solve(args: SymbolicMathInput) -> ToolResult:
    sym_dict = parse_symbols(args.symbols, args.assumptions)
    equations = []
    for eq_str in args.equations or []:
        if "=" in eq_str:
            lhs, rhs = eq_str.split("=", 1)
            equations.append(
                sp.Eq(
                    safe_parse(lhs.strip(), sym_dict),
                    safe_parse(rhs.strip(), sym_dict),
                )
            )
        else:
            equations.append(safe_parse(eq_str, sym_dict))

    solve_for = [sym_dict[s] for s in args.symbols if s in sym_dict]
    solutions = sp.solve(equations, solve_for, dict=True)

    return ToolResult(
        data={
            "equations": [str(e) for e in equations],
            "solve_for": [str(s) for s in solve_for],
            "solutions": (
                [{str(k): str(v) for k, v in sol.items()} for sol in solutions]
                if solutions
                else []
            ),
        },
        success=True,
    )


def eigenvalue(args: SymbolicMathInput) -> ToolResult:
    sym_dict = parse_symbols(args.symbols, args.assumptions)
    if not args.matrix:
        return ToolResult(data=None, success=False, error="No matrix provided")

    M = sp.Matrix(
        [
            [safe_parse(entry, sym_dict) for entry in row]
            for row in args.matrix
        ]
    )

    eigenvals = M.eigenvals()

    return ToolResult(
        data={
            "matrix": str(M),
            "eigenvalues": [
                {"value": str(val), "multiplicity": mult}
                for val, mult in eigenvals.items()
            ],
            "trace": str(sp.trace(M)),
            "determinant": str(sp.det(M)),
        },
        success=True,
    )


def linear_algebra(args: SymbolicMathInput) -> ToolResult:
    """数值线性代数: lu_decompose / cholesky / jacobi_solve / gauss_seidel_solve
    / cg_solve / mat_vec_mul / cond_number.
    """
    target = (args.target or "lu_decompose").lower()

    if not args.matrix:
        return ToolResult(
            data=None, success=False, error="matrix required for linear_algebra"
        )
    M = sp.Matrix([[sp.sympify(entry) for entry in row] for row in args.matrix])
    n = M.rows
    if M.rows != M.cols:
        return ToolResult(data=None, success=False, error="Matrix must be square")

    b_vec = None
    if args.expression:
        b_vec = sp.Matrix([sp.sympify(x) for x in args.expression.split(",")])

    if target == "lu_decompose":
        return _la_lu(M, n)

    if target == "cholesky":
        return _la_cholesky(M, n)

    if target == "mat_vec_mul":
        return _la_mat_vec(M, b_vec, n)

    if target in ("jacobi_solve", "gauss_seidel_solve", "cg_solve"):
        return _la_solve(M, b_vec, n, target)

    if target == "cond_number":
        return _la_cond(M, n)

    return ToolResult(
        data=None, success=False, error=f"Unknown linear_algebra target: {target}"
    )


def _matrix_strings(mat: Any, n: int) -> list[list[str]]:
    """SymPy Matrix → 字符串二维表 (输出用)."""
    return [[str(mat[i, j]) for j in range(n)] for i in range(n)]


def _la_lu(M: Any, n: int) -> ToolResult:
    L, U, _ = M.LUdecomposition()
    return ToolResult(
        data={"L": _matrix_strings(L, n), "U": _matrix_strings(U, n), "size": n},
        success=True,
    )


def _la_cholesky(M: Any, n: int) -> ToolResult:
    if not M.is_symmetric():
        return ToolResult(
            data=None, success=False, error="Matrix must be symmetric for Cholesky"
        )
    try:
        L = M.cholesky()
    except ValueError as e:
        return ToolResult(data=None, success=False, error=f"Cholesky failed: {e}")
    return ToolResult(data={"L": _matrix_strings(L, n), "size": n}, success=True)


def _la_mat_vec(M: Any, b_vec: Any, n: int) -> ToolResult:
    if b_vec is None or b_vec.rows != n:
        return ToolResult(
            data=None,
            success=False,
            error="Vector required (via expression) with matching size",
        )
    result = M * b_vec
    return ToolResult(
        data={"result": [str(result[i]) for i in range(n)], "size": n},
        success=True,
    )


def _la_solve(M: Any, b_vec: Any, n: int, target: str) -> ToolResult:
    """Jacobi / Gauss-Seidel / 共轭梯度三种迭代解法."""
    if b_vec is None or b_vec.rows != n:
        return ToolResult(
            data=None,
            success=False,
            error="Vector b required (via expression) with matching size",
        )
    A = [[float(M[i, j]) for j in range(n)] for i in range(n)]
    b = [float(b_vec[i]) for i in range(n)]
    x = [0.0] * n
    if target == "jacobi_solve":
        x = _jacobi_solve(A, b, n, x)
    elif target == "gauss_seidel_solve":
        x = _gauss_seidel_solve(A, b, n, x)
    else:
        x = _cg_solve(A, b, n, x)
    return ToolResult(data={"solution": x, "size": n, "solver": target}, success=True)


_LA_MAX_ITER = 100
_LA_TOL = 1e-10


def _jacobi_solve(
    A: list[list[float]], b: list[float], n: int, x: list[float]
) -> list[float]:
    """Jacobi 迭代: x_new[i] = (b[i] − Σ_{j≠i} A[i][j]·x[j]) / A[i][i]."""
    for _ in range(_LA_MAX_ITER):
        x_new = [
            (b[i] - sum(A[i][j] * x[j] for j in range(n) if j != i)) / A[i][i]
            for i in range(n)
        ]
        if max(abs(x_new[i] - x[i]) for i in range(n)) < _LA_TOL:
            break
        x = x_new
    return x


def _gauss_seidel_solve(
    A: list[list[float]], b: list[float], n: int, x: list[float]
) -> list[float]:
    """Gauss-Seidel: 用本轮已更新的 x[j] (j<i) 和上一轮的 x_old[j] (j>i)."""
    for _ in range(_LA_MAX_ITER):
        x_old = x[:]
        for i in range(n):
            sum_l = sum(A[i][j] * x[j] for j in range(i))
            sum_u = sum(A[i][j] * x_old[j] for j in range(i + 1, n))
            x[i] = (b[i] - sum_l - sum_u) / A[i][i]
        if max(abs(x[i] - x_old[i]) for i in range(n)) < _LA_TOL:
            break
    return x


def _cg_solve(
    A: list[list[float]], b: list[float], n: int, x: list[float]
) -> list[float]:
    """共轭梯度法 (对称正定矩阵)."""
    r = [b[i] - sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]
    p = r[:]
    rs_old = sum(ri * ri for ri in r)
    for _ in range(_LA_MAX_ITER):
        ap = [sum(A[i][j] * p[j] for j in range(n)) for i in range(n)]
        p_ap = sum(p[i] * ap[i] for i in range(n))
        if p_ap == 0.0:
            break
        alpha = rs_old / p_ap
        x = [x[i] + alpha * p[i] for i in range(n)]
        r = [r[i] - alpha * ap[i] for i in range(n)]
        rs_new = sum(ri * ri for ri in r)
        if rs_new**0.5 < _LA_TOL:
            break
        beta = rs_new / rs_old
        p = [r[i] + beta * p[i] for i in range(n)]
        rs_old = rs_new
    return x


def _la_cond(M: Any, n: int) -> ToolResult:
    return ToolResult(
        data={"cond_number": str(M.condition_number()), "size": n},
        success=True,
    )
