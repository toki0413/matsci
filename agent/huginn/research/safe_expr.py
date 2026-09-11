"""共享安全数学表达式求值器 (统一 deep_research / llm_generate_scm 的 `_safe_eval_expr`).

原由: 深研 HTTP 端点与因果 SCM 方程各持一份"白名单 math"求值器, 名字同为
``_safe_eval_expr`` 却实现分叉——一份是手写 AST 遍历(不 eval), 一份是
``ast.walk`` 校验 + ``eval``。收敛到本模块是唯一实现, 两处共用。

安全模型: 不求值字符串、不 ``exec``/``eval`` —— 把表达式解析成 AST, 只手写
遍历白名单节点(数字常量/命名变量/算术/比较链/白名单 math 函数)。
任何越界节点或不可解析输入抛 ``ValueError``, 不自耗静默吞掉。
"""

from __future__ import annotations

import ast
import math
import numbers
import operator

# 白名单 math 常量 (可作裸名字: `pi`, `e`, `tau`, `inf`)
_CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}

# 白名单 math 函数 (只能经 Call 调用, 且不允许 keyword 参数)
_FUNCS: dict[str, object] = {
    "abs": abs, "min": min, "max": max, "round": round,
    "sqrt": math.sqrt, "exp": math.exp, "log": math.log, "log10": math.log10,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "floor": math.floor, "ceil": math.ceil, "fabs": math.fabs, "pow": math.pow,
}

_BIN_OPS: dict[type, object] = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type, object] = {ast.UAdd: operator.pos, ast.USub: operator.neg}

_MAX_DEPTH = 64


def safe_math_eval(
    expr: str,
    variables: dict[str, float],
    *,
    constants: dict[str, float] | None = None,
) -> float:
    """安全求值 math 表达式, 返回 float.

    - 只允许: 数字/白名单 math 常量/命名变量/算术/比较链/白名单 math 函数。
    - ``constants`` 是调用方注入的额外常量(如因果 SCM 的 ``R``), 会覆盖同名变量。
    - 任何越界或不可解析输入抛 ``ValueError``。
    """
    ctx = dict(variables)
    if constants:
        ctx.update(constants)
    if not isinstance(expr, str) or not expr.strip():
        raise ValueError("expression must be a non-empty string")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as syn:
        raise ValueError(f"expression not parseable: {syn.msg}") from syn
    return float(_walk(tree, ctx, 0))


def _walk(node: ast.AST, ctx: dict[str, float], depth: int) -> float:
    if depth > _MAX_DEPTH:
        raise ValueError("expression too deep")
    if isinstance(node, ast.Expression):
        return _walk(node.body, ctx, depth + 1)
    if isinstance(node, ast.Constant):
        # Py3.8+ 数值统一走 Constant (ast.Num 已在 3.14 移除)
        if isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node.value, numbers.Real):
            return float(node.value)
        raise ValueError("literal not supported")
    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return float(_CONSTANTS[node.id])
        if node.id in ctx:
            return float(ctx[node.id])
        raise ValueError(f"unknown variable: {node.id}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](  # type: ignore[misc]
            _walk(node.left, ctx, depth + 1),
            _walk(node.right, ctx, depth + 1),
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](  # type: ignore[misc]
            _walk(node.operand, ctx, depth + 1)
        )
    if isinstance(node, ast.Call):
        fn = node.func
        if not isinstance(fn, ast.Name) or fn.id not in _FUNCS:
            raise ValueError("only whitelisted math functions")
        if node.keywords:
            raise ValueError("keyword arguments not allowed")
        args = [_walk(a, ctx, depth + 1) for a in node.args]
        return float(_FUNCS[fn.id](*args))  # type: ignore[operator]
    if isinstance(node, ast.Compare):
        left = _walk(node.left, ctx, depth + 1)
        for opn, cmp_node in zip(node.ops, node.comparators):
            right = _walk(cmp_node, ctx, depth + 1)
            if type(opn) is ast.Lt and not (left < right):
                return 0.0
            if type(opn) is ast.LtE and not (left <= right):
                return 0.0
            if type(opn) is ast.Gt and not (left > right):
                return 0.0
            if type(opn) is ast.GtE and not (left >= right):
                return 0.0
            if type(opn) is ast.Eq and left != right:
                return 0.0
            if type(opn) is ast.NotEq and left == right:
                return 0.0
            left = right
        return 1.0
    raise ValueError(f"expression node not allowed: {type(node).__name__}")