"""Restricted expression evaluation replacing raw eval().

Only allows mathematical and boolean operations — no attribute access
on arbitrary objects, no imports, no calls to untrusted functions.
"""

from __future__ import annotations

import ast
import operator
import sys
import warnings
from typing import Any


class SafeEvalError(Exception):
    """Raised when an expression violates the safe-eval policy."""


# Allowed AST node types
_ALLOWED_NODES = {
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.FloorDiv,
    ast.USub,
    ast.UAdd,
    ast.Not,
    ast.Invert,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Is,
    ast.IsNot,
    ast.In,
    ast.NotIn,
    ast.IfExp,
    ast.Tuple,
    ast.List,
    ast.Set,
    ast.Dict,
    ast.Subscript,
    ast.Index,
    ast.Slice,
    ast.ExtSlice,
    ast.Call,
    ast.Attribute,
    ast.keyword,
    ast.Starred,
}

# ast.Num is deprecated in Python 3.14+; keep for backward compat on older versions
if sys.version_info < (3, 14):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        _ALLOWED_NODES.add(ast.Num)

# Allowed built-in functions
_ALLOWED_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "len": len,
    "max": max,
    "min": min,
    "pow": pow,
    "round": round,
    "sum": sum,
    "True": True,
    "False": False,
    "None": None,
}

# Binary operators
_BIN_OPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}

# Unary operators
_UNARY_OPS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
    ast.Invert: operator.invert,
}

# Boolean operators
_BOOL_OPS: dict[type[ast.boolop], Any] = {
    ast.And: all,
    ast.Or: any,
}

# Comparison operators
_COMPARE_OPS: dict[type[ast.cmpop], Any] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}


def safe_eval(expr: str, locals_dict: dict[str, Any] | None = None) -> Any:
    """Evaluate a mathematical/boolean expression safely.

    Raises SafeEvalError if the expression contains forbidden constructs.
    """
    locals_dict = locals_dict or {}

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise SafeEvalError(f"Invalid syntax: {e}") from e

    def _eval(node: ast.AST) -> Any:
        if type(node) not in _ALLOWED_NODES:
            raise SafeEvalError(
                f"Forbidden expression construct: {type(node).__name__}"
            )

        if isinstance(node, ast.Expression):
            return _eval(node.body)

        if isinstance(node, ast.Constant):
            return node.value

        if sys.version_info < (3, 14):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                if isinstance(node, ast.Num):  # pragma: no cover (Python <3.8 compat)
                    return node.n

        if isinstance(node, ast.Name):
            if node.id in locals_dict:
                return locals_dict[node.id]
            if node.id in _ALLOWED_BUILTINS:
                return _ALLOWED_BUILTINS[node.id]
            raise SafeEvalError(f"Undefined name: {node.id}")

        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            op_type = type(node.op)
            if op_type not in _BIN_OPS:
                raise SafeEvalError(f"Unsupported binary operator: {op_type.__name__}")
            return _BIN_OPS[op_type](left, right)

        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            op_type = type(node.op)
            if op_type not in _UNARY_OPS:
                raise SafeEvalError(f"Unsupported unary operator: {op_type.__name__}")
            return _UNARY_OPS[op_type](operand)

        if isinstance(node, ast.BoolOp):
            values = [_eval(v) for v in node.values]
            op_type = type(node.op)
            if op_type not in _BOOL_OPS:
                raise SafeEvalError(f"Unsupported boolean operator: {op_type.__name__}")
            return _BOOL_OPS[op_type](values)

        if isinstance(node, ast.Compare):
            left = _eval(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                right = _eval(comparator)
                op_type = type(op)
                if op_type not in _COMPARE_OPS:
                    raise SafeEvalError(f"Unsupported comparison: {op_type.__name__}")
                if not _COMPARE_OPS[op_type](left, right):
                    return False
                left = right
            return True

        if isinstance(node, ast.IfExp):
            return _eval(node.body) if _eval(node.test) else _eval(node.orelse)

        if isinstance(node, ast.Tuple):
            return tuple(_eval(elt) for elt in node.elts)

        if isinstance(node, ast.List):
            return [_eval(elt) for elt in node.elts]

        if isinstance(node, ast.Dict):
            return {_eval(k): _eval(v) for k, v in zip(node.keys, node.values)}

        if isinstance(node, ast.Set):
            return {_eval(elt) for elt in node.elts}

        if isinstance(node, ast.Subscript):
            value = _eval(node.value)
            slice_val: Any
            if isinstance(node.slice, ast.Constant):
                slice_val = node.slice.value
            elif isinstance(node.slice, ast.Index):  # pragma: no cover
                slice_val = _eval(node.slice.value)
            else:
                slice_val = _eval(node.slice)
            return value[slice_val]

        if isinstance(node, ast.Call):
            raise SafeEvalError("Function calls are forbidden in safe_eval")

        if isinstance(node, ast.Attribute):
            raise SafeEvalError("Attribute access is forbidden in safe_eval")

        raise SafeEvalError(f"Unsupported node: {type(node).__name__}")

    return _eval(tree)
