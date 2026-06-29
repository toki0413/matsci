"""代数类 action: solve / eigenvalue / linear_algebra."""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from huginn.types import ToolResult

from ._parsers import parse_symbols, safe_parse

if TYPE_CHECKING:
    from .tool import SymbolicMathInput


def solve(args: "SymbolicMathInput") -> ToolResult:
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


def eigenvalue(args: "SymbolicMathInput") -> ToolResult:
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


def linear_algebra(args: "SymbolicMathInput") -> ToolResult:
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
        L, U, _ = M.LUdecomposition()
        return ToolResult(
            data={
                "L": [[str(L[i, j]) for j in range(n)] for i in range(n)],
                "U": [[str(U[i, j]) for j in range(n)] for i in range(n)],
                "size": n,
            },
            success=True,
        )

    if target == "cholesky":
        if not M.is_symmetric():
            return ToolResult(
                data=None,
                success=False,
                error="Matrix must be symmetric for Cholesky",
            )
        try:
            L = M.cholesky()
        except ValueError as e:
            return ToolResult(
                data=None, success=False, error=f"Cholesky failed: {e}"
            )
        return ToolResult(
            data={
                "L": [[str(L[i, j]) for j in range(n)] for i in range(n)],
                "size": n,
            },
            success=True,
        )

    if target == "mat_vec_mul":
        if b_vec is None or b_vec.rows != n:
            return ToolResult(
                data=None,
                success=False,
                error="Vector required (via expression) with matching size",
            )
        result = M * b_vec
        return ToolResult(
            data={
                "result": [str(result[i]) for i in range(n)],
                "size": n,
            },
            success=True,
        )

    if target in ("jacobi_solve", "gauss_seidel_solve", "cg_solve"):
        if b_vec is None or b_vec.rows != n:
            return ToolResult(
                data=None,
                success=False,
                error="Vector b required (via expression) with matching size",
            )
        max_iter = 100
        tol = 1e-10
        A_float = [[float(M[i, j]) for j in range(n)] for i in range(n)]
        b_float = [float(b_vec[i]) for i in range(n)]
        x = [0.0] * n

        if target == "jacobi_solve":
            for _ in range(max_iter):
                x_new = [
                    (
                        b_float[i]
                        - sum(A_float[i][j] * x[j] for j in range(n) if j != i)
                    )
                    / A_float[i][i]
                    for i in range(n)
                ]
                if max(abs(x_new[i] - x[i]) for i in range(n)) < tol:
                    break
                x = x_new
        elif target == "gauss_seidel_solve":
            for _ in range(max_iter):
                x_old = x[:]
                for i in range(n):
                    sumL = sum(A_float[i][j] * x[j] for j in range(i))
                    sumU = sum(A_float[i][j] * x_old[j] for j in range(i + 1, n))
                    x[i] = (b_float[i] - sumL - sumU) / A_float[i][i]
                if max(abs(x[i] - x_old[i]) for i in range(n)) < tol:
                    break
        elif target == "cg_solve":
            r = [
                b_float[i] - sum(A_float[i][j] * x[j] for j in range(n))
                for i in range(n)
            ]
            p = r[:]
            rs_old = sum(ri * ri for ri in r)
            for _ in range(max_iter):
                Ap = [sum(A_float[i][j] * p[j] for j in range(n)) for i in range(n)]
                pAp = sum(p[i] * Ap[i] for i in range(n))
                if pAp == 0.0:
                    break
                alpha = rs_old / pAp
                x = [x[i] + alpha * p[i] for i in range(n)]
                r = [r[i] - alpha * Ap[i] for i in range(n)]
                rs_new = sum(ri * ri for ri in r)
                if rs_new**0.5 < tol:
                    break
                beta = rs_new / rs_old
                p = [r[i] + beta * p[i] for i in range(n)]
                rs_old = rs_new

        return ToolResult(
            data={
                "solution": x,
                "size": n,
                "solver": target,
            },
            success=True,
        )

    if target == "cond_number":
        cond = M.condition_number()
        return ToolResult(
            data={
                "cond_number": str(cond),
                "size": n,
            },
            success=True,
        )

    return ToolResult(
        data=None, success=False, error=f"Unknown linear_algebra target: {target}"
    )
