"""Restricted mathematical expression evaluation with a numpy whitelist.

This module extends :mod:`huginn.security.safe_eval` with just enough
functionality to evaluate the scientific expressions the agent generates for
:huginn.tools.numerical_tool`.  Only whitelisted numpy callables and constants
are exposed; arbitrary attribute access and function calls remain forbidden.
"""

from __future__ import annotations

import ast
import sys
import warnings
from typing import Any

import numpy as np

from huginn.security.safe_eval import (
    _ALLOWED_BUILTINS,
    _ALLOWED_NODES,
    _BIN_OPS,
    _BOOL_OPS,
    _COMPARE_OPS,
    _UNARY_OPS,
    SafeEvalError,
)

# Numpy functions/constants that are safe to expose inside mathematical
# expressions.  We deliberately keep this list small and reviewable.
_ALLOWED_NUMPY_NAMES = (
    "sin",
    "cos",
    "tan",
    "exp",
    "log",
    "log10",
    "sqrt",
    "abs",
    "absolute",
    "power",
    "square",
    "cbrt",
    "sinh",
    "cosh",
    "tanh",
    "arcsin",
    "arccos",
    "arctan",
    "arctan2",
    "pi",
    "e",
)

_ALLOWED_NUMPY: dict[str, Any] = {
    name: getattr(np, name) for name in _ALLOWED_NUMPY_NAMES if hasattr(np, name)
}

# Pre-compute the identities of all callables we are willing to execute.
# Any ast.Call whose resolved callable is not in this set is rejected.
_ALLOWED_CALLABLES: set[int] = set()
for _obj in list(_ALLOWED_BUILTINS.values()) + list(_ALLOWED_NUMPY.values()):
    if callable(_obj):
        _ALLOWED_CALLABLES.add(id(_obj))

# Math expressions need Call/Attribute for np.<func>; everything else is
# inherited from the base safe_eval whitelist.
_MATH_ALLOWED_NODES = _ALLOWED_NODES | {ast.Call, ast.Attribute, ast.keyword}


def safe_math_eval(expr: str, locals_dict: dict[str, Any] | None = None) -> Any:
    """Evaluate a mathematical expression safely with a numpy whitelist.

    Supports basic arithmetic, subscripts, comparisons, conditionals, and
    calls to a small whitelist of numpy functions (``np.sin``, ``np.exp``,
    etc.).  Arbitrary imports, attribute access, and function calls are
    rejected.

    Args:
        expr: The expression to evaluate.
        locals_dict: Additional names available to the expression.  Values
            that are callable are *not* whitelisted automatically.

    Returns:
        The evaluated result.

    Raises:
        SafeEvalError: If the expression is invalid or uses a forbidden
            construct.
    """
    locals_dict = locals_dict or {}

    # Build the evaluation environment.  ``np`` is present so attribute
    # access can be validated against the numpy whitelist.
    env: dict[str, Any] = {"np": np, **_ALLOWED_NUMPY, **_ALLOWED_BUILTINS, **locals_dict}

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise SafeEvalError(f"Invalid syntax: {e}") from e

    def _eval(node: ast.AST) -> Any:
        if type(node) not in _MATH_ALLOWED_NODES:
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
                if isinstance(node, ast.Num):  # pragma: no cover
                    return node.n

        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            raise SafeEvalError(f"Undefined name: {node.id}")

        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            op_type = type(node.op)
            if op_type not in _BIN_OPS:
                raise SafeEvalError(
                    f"Unsupported binary operator: {op_type.__name__}"
                )
            return _BIN_OPS[op_type](left, right)

        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            op_type = type(node.op)
            if op_type not in _UNARY_OPS:
                raise SafeEvalError(
                    f"Unsupported unary operator: {op_type.__name__}"
                )
            return _UNARY_OPS[op_type](operand)

        if isinstance(node, ast.BoolOp):
            values = [_eval(v) for v in node.values]
            op_type = type(node.op)
            if op_type not in _BOOL_OPS:
                raise SafeEvalError(
                    f"Unsupported boolean operator: {op_type.__name__}"
                )
            return _BOOL_OPS[op_type](values)

        if isinstance(node, ast.Compare):
            left = _eval(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                right = _eval(comparator)
                op_type = type(op)
                if op_type not in _COMPARE_OPS:
                    raise SafeEvalError(
                        f"Unsupported comparison: {op_type.__name__}"
                    )
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
            return {
                _eval(k): _eval(v)
                for k, v in zip(node.keys, node.values)
                if k is not None
            }

        if isinstance(node, ast.Set):
            return {_eval(elt) for elt in node.elts}

        if isinstance(node, ast.Subscript):
            value = _eval(node.value)
            slice_val: Any
            if isinstance(node.slice, ast.Constant):
                slice_val = node.slice.value
            elif hasattr(ast, "Index") and isinstance(node.slice, ast.Index):
                slice_val = _eval(node.slice.value)
            else:
                slice_val = _eval(node.slice)
            return value[slice_val]

        if isinstance(node, ast.Attribute):
            obj = _eval(node.value)
            # Only permit ``np.<whitelisted>``.
            if obj is np and node.attr in _ALLOWED_NUMPY:
                return _ALLOWED_NUMPY[node.attr]
            raise SafeEvalError(f"Attribute access is not allowed: {node.attr}")

        if isinstance(node, ast.Call):
            func = _eval(node.func)
            if not callable(func):
                raise SafeEvalError("Cannot call non-callable")
            if id(func) not in _ALLOWED_CALLABLES:
                raise SafeEvalError("Function call is not allowed")
            args = [_eval(a) for a in node.args]
            kwargs = {
                kw.arg: _eval(kw.value)
                for kw in node.keywords
                if kw.arg is not None
            }
            return func(*args, **kwargs)

        raise SafeEvalError(f"Unsupported node: {type(node).__name__}")

    return _eval(tree)
