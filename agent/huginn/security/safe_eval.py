"""Restricted expression evaluation replacing raw eval().

Only allows mathematical and boolean operations — no attribute access
on arbitrary objects, no imports, no calls to untrusted functions.

Extensible via :class:`SafeExprWalker`: the shared AST traversal lives here,
and callers (e.g. ``huginn.security.math_eval``) plug in their own
name/call/attribute policy without re-copying the walker.
"""

from __future__ import annotations

import ast
import operator
import sys
import warnings
from typing import Any, Callable


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
    ast.Slice,
    ast.Call,
    ast.Attribute,
    ast.keyword,
    ast.Starred,
}

# ast.Index and ast.ExtSlice were removed in Python 3.12.
# In 3.9+, subscripts use the expression directly (no Index wrapper).
# Use getattr for cross-version compatibility.
for _deprecated_node in ("Index", "ExtSlice"):
    _node = getattr(ast, _deprecated_node, None)
    if _node is not None:
        _ALLOWED_NODES.add(_node)

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


class SafeExprWalker:
    """Reusable safe AST evaluator (the single walker for safe_eval / math_eval).

    The arithmetic/boolean/container/subscript traversal is shared here.
    Extensions plug in via three hooks, letting sub-policies change only
    *what names, calls and attributes are allowed* without copying the walker:

    - ``name_extra(id) -> (found, value)``: extra names resolvable by a policy
      (e.g. numpy bare names and the ``np`` module in math_eval). Locals and
      ``_ALLOWED_BUILTINS`` are resolved first.
    - ``resolve_call(node, eval_fn) -> Any``: returns the call result.
      ``None`` (default) forbids all function calls.
    - ``resolve_attribute(node, eval_fn) -> Any``: returns the attribute value.
      ``None`` (default) forbids all attribute access.
    """

    def __init__(
        self,
        *,
        allowed_nodes: set[type[ast.AST]] | None = None,
        name_extra: Callable[[str], tuple[bool, Any]] | None = None,
        resolve_call: Callable[[ast.AST, Callable[[ast.AST], Any]], Any] | None = None,
        resolve_attribute: Callable[[ast.AST, Callable[[ast.AST], Any]], Any] | None = None,
    ) -> None:
        self._nodes = allowed_nodes if allowed_nodes is not None else _ALLOWED_NODES
        self._name_extra = name_extra
        self._resolve_call = resolve_call
        self._resolve_attribute = resolve_attribute

    def __call__(self, expr: str, locals_dict: dict[str, Any] | None = None) -> Any:
        locals_dict = locals_dict or {}

        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as e:
            raise SafeEvalError(f"Invalid syntax: {e}") from e

        return self._eval(tree, locals_dict)

    def _eval(self, node: ast.AST, locals_dict: dict[str, Any], _depth: int = 0) -> Any:
        if _depth > 128:
            raise SafeEvalError("Expression exceeds maximum nesting depth")

        if type(node) not in self._nodes:
            raise SafeEvalError(
                f"Forbidden expression construct: {type(node).__name__}"
            )

        def sub(n: ast.AST) -> Any:
            return self._eval(n, locals_dict, _depth + 1)

        if isinstance(node, ast.Expression):
            return sub(node.body)

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
            if self._name_extra is not None:
                found, value = self._name_extra(node.id)
                if found:
                    return value
            raise SafeEvalError(f"Undefined name: {node.id}")

        if isinstance(node, ast.BinOp):
            left = sub(node.left)
            right = sub(node.right)
            op_type = type(node.op)
            if op_type not in _BIN_OPS:
                raise SafeEvalError(f"Unsupported binary operator: {op_type.__name__}")
            return _BIN_OPS[op_type](left, right)

        if isinstance(node, ast.UnaryOp):
            operand = sub(node.operand)
            op_type = type(node.op)
            if op_type not in _UNARY_OPS:
                raise SafeEvalError(f"Unsupported unary operator: {op_type.__name__}")
            return _UNARY_OPS[op_type](operand)

        if isinstance(node, ast.BoolOp):
            values = [sub(v) for v in node.values]
            op_type = type(node.op)
            if op_type not in _BOOL_OPS:
                raise SafeEvalError(f"Unsupported boolean operator: {op_type.__name__}")
            return _BOOL_OPS[op_type](values)

        if isinstance(node, ast.Compare):
            left = sub(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                right = sub(comparator)
                op_type = type(op)
                if op_type not in _COMPARE_OPS:
                    raise SafeEvalError(f"Unsupported comparison: {op_type.__name__}")
                if not _COMPARE_OPS[op_type](left, right):
                    return False
                left = right
            return True

        if isinstance(node, ast.IfExp):
            return sub(node.body) if sub(node.test) else sub(node.orelse)

        if isinstance(node, ast.Tuple):
            return tuple(sub(elt) for elt in node.elts)

        if isinstance(node, ast.List):
            return [sub(elt) for elt in node.elts]

        if isinstance(node, ast.Dict):
            return {sub(k): sub(v) for k, v in zip(node.keys, node.values) if k is not None}

        if isinstance(node, ast.Set):
            return {sub(elt) for elt in node.elts}

        if isinstance(node, ast.Subscript):
            value = sub(node.value)
            slice_val: Any
            if isinstance(node.slice, ast.Constant):
                slice_val = node.slice.value
            elif hasattr(ast, "Index") and isinstance(node.slice, ast.Index):
                # Python < 3.9 wraps subscript index in ast.Index
                slice_val = sub(node.slice.value)
            else:
                slice_val = sub(node.slice)
            return value[slice_val]

        if isinstance(node, ast.Call):
            if self._resolve_call is None:
                raise SafeEvalError("Function calls are forbidden in safe_eval")
            return self._resolve_call(node, sub)

        if isinstance(node, ast.Attribute):
            if self._resolve_attribute is None:
                raise SafeEvalError("Attribute access is forbidden in safe_eval")
            return self._resolve_attribute(node, sub)

        raise SafeEvalError(f"Unsupported node: {type(node).__name__}")


def safe_eval(expr: str, locals_dict: dict[str, Any] | None = None) -> Any:
    """Evaluate a mathematical/boolean expression safely.

    Raises SafeEvalError if the expression contains forbidden constructs.
    """
    return SafeExprWalker()(expr, locals_dict)