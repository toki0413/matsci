"""微积分类 action: differentiate / integrate / simplify / taylor / series."""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from huginn.types import ToolResult

from ._parsers import parse_symbols, safe_parse

if TYPE_CHECKING:
    from .tool import SymbolicMathInput


def differentiate(args: "SymbolicMathInput") -> ToolResult:
    sym_dict = parse_symbols(args.symbols, args.assumptions)
    expr = safe_parse(args.expression or "", sym_dict)
    var = sym_dict.get(args.variable)
    if var is None:
        return ToolResult(
            data=None,
            success=False,
            error=f"Variable {args.variable} not in symbols",
        )

    result = sp.diff(expr, var, args.order)
    return ToolResult(
        data={
            "input": str(expr),
            "variable": str(var),
            "order": args.order,
            "result": str(result),
            "latex": sp.latex(result),
        },
        success=True,
    )


def integrate(args: "SymbolicMathInput") -> ToolResult:
    sym_dict = parse_symbols(args.symbols, args.assumptions)
    expr = safe_parse(args.expression or "", sym_dict)
    var = sym_dict.get(args.variable)
    if var is None:
        return ToolResult(
            data=None,
            success=False,
            error=f"Variable {args.variable} not in symbols",
        )

    result = sp.integrate(expr, var)
    return ToolResult(
        data={
            "input": str(expr),
            "variable": str(var),
            "result": str(result),
            "latex": sp.latex(result),
        },
        success=True,
    )


def simplify(args: "SymbolicMathInput") -> ToolResult:
    sym_dict = parse_symbols(args.symbols, args.assumptions)
    expr = safe_parse(args.expression or "", sym_dict)
    simplified = sp.simplify(expr)
    return ToolResult(
        data={
            "original": str(expr),
            "simplified": str(simplified),
            "latex": sp.latex(simplified),
        },
        success=True,
    )


def taylor(args: "SymbolicMathInput") -> ToolResult:
    sym_dict = parse_symbols(args.symbols, args.assumptions)
    expr = safe_parse(args.expression or "", sym_dict)
    var = sym_dict.get(args.variable)
    if var is None:
        return ToolResult(
            data=None,
            success=False,
            error=f"Variable {args.variable} not in symbols",
        )

    point = args.point or {}
    expansion_point = point.get(args.variable, 0)
    series = sp.series(expr, var, expansion_point, args.order + 1).removeO()

    return ToolResult(
        data={
            "input": str(expr),
            "variable": str(var),
            "order": args.order,
            "expansion_point": expansion_point,
            "series": str(series),
            "latex": sp.latex(series),
        },
        success=True,
    )


def series(args: "SymbolicMathInput") -> ToolResult:
    """符号级数展开."""
    sym_dict = parse_symbols(args.symbols, args.assumptions)
    expr = safe_parse(args.expression or "", sym_dict)
    var = sym_dict.get(args.variable)
    if var is None:
        return ToolResult(
            data=None,
            success=False,
            error=f"Variable {args.variable} not in symbols",
        )

    point = args.point or {}
    x0 = point.get(args.variable, 0)
    n = args.order

    s = sp.series(expr, var, x0, n + 1).removeO()
    return ToolResult(
        data={
            "original": str(expr),
            "variable": str(var),
            "point": x0,
            "order": n,
            "expansion": str(s),
            "latex": sp.latex(s),
        },
        success=True,
    )
