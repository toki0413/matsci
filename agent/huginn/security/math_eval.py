"""Restricted mathematical expression evaluation with a numpy whitelist.

This module configures :class:`huginn.security.safe_eval.SafeExprWalker`
with just enough functionality to evaluate the scientific expressions the
agent generates for :huginn.tools.numerical_tool` and the symbolic
regression tool. Only whitelisted numpy callables/constants are exposed;
arbitrary attribute access and function calls remain forbidden.

The AST traversal itself lives in :mod:`huginn.security.safe_eval` — this
module only supplies the numpy *policy* (extra names, call whitelist, and
``np.<whitelisted>`` attribute access), so it stays thin instead of copying
the walker.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from huginn.security.safe_eval import (
    _ALLOWED_BUILTINS,
    SafeEvalError,
    SafeExprWalker,
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
# Builtin callables come from safe_eval's shared table; numpy callables above.
_ALLOWED_CALLABLES: set[int] = {
    id(obj)
    for obj in list(_ALLOWED_BUILTINS.values()) + list(_ALLOWED_NUMPY.values())
    if callable(obj)
}


def _name_extra(name: str) -> tuple[bool, Any]:
    """Expose the ``np`` module and bare numpy names (sin, pi, ...)."""
    if name == "np":
        return True, np
    if name in _ALLOWED_NUMPY:
        return True, _ALLOWED_NUMPY[name]
    return False, None


def _resolve_call(node: Any, eval_fn: Any) -> Any:
    func = eval_fn(node.func)
    if not callable(func):
        raise SafeEvalError("Cannot call non-callable")
    if id(func) not in _ALLOWED_CALLABLES:
        raise SafeEvalError("Function call is not allowed")
    args = [eval_fn(a) for a in node.args]
    kwargs = {kw.arg: eval_fn(kw.value) for kw in node.keywords if kw.arg is not None}
    return func(*args, **kwargs)


def _resolve_attribute(node: Any, eval_fn: Any) -> Any:
    obj = eval_fn(node.value)
    # Only permit ``np.<whitelisted>``.
    if obj is np and node.attr in _ALLOWED_NUMPY:
        return _ALLOWED_NUMPY[node.attr]
    raise SafeEvalError(f"Attribute access is not allowed: {node.attr}")


_WALKER = SafeExprWalker(
    name_extra=_name_extra,
    resolve_call=_resolve_call,
    resolve_attribute=_resolve_attribute,
)


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
    return _WALKER(expr, locals_dict)