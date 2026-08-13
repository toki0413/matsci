"""math_eval.py 全分支测试 — 受限数学表达式求值、numpy 白名单、错误处理."""

from __future__ import annotations

import pytest

from huginn.security.math_eval import safe_math_eval
from huginn.security.safe_eval import SafeEvalError


# ── 算术 / 常量 ──────────────────────────────────────────────────────────

def test_integer_arith():
    assert safe_math_eval("1 + 2 * 3") == 7


def test_float_division():
    assert abs(safe_math_eval("3 / 2") - 1.5) < 1e-12


def test_unary_neg():
    assert safe_math_eval("-5") == -5


def test_constant_pi():
    assert abs(safe_math_eval("np.pi") - 3.14159265) < 1e-6


def test_constant_e():
    assert abs(safe_math_eval("np.e") - 2.71828) < 1e-4


# ── 比较 / 布尔 / 条件 ───────────────────────────────────────────────────

def test_comparison():
    assert safe_math_eval("1 < 2") is True


def test_chained_comparison():
    assert safe_math_eval("1 < 2 < 3") is True
    assert safe_math_eval("1 < 5 < 3") is False


def test_bool_and():
    assert safe_math_eval("True and False") is False


def test_if_exp():
    assert safe_math_eval("1 if 2 > 1 else 0") == 1
    assert safe_math_eval("1 if 0 > 1 else 0") == 0


# ── 容器 ─────────────────────────────────────────────────────────────────

def test_tuple():
    assert safe_math_eval("(1, 2, 3)") == (1, 2, 3)


def test_list():
    assert safe_math_eval("[1, 2, 3]") == [1, 2, 3]


def test_subscript():
    assert safe_math_eval("[1, 2, 3][1]") == 2


# ── numpy 白名单调用 ─────────────────────────────────────────────────────

def test_numpy_sin_call():
    assert abs(safe_math_eval("np.sin(0)")) < 1e-12


def test_numpy_cos_call():
    assert abs(safe_math_eval("np.cos(0)") - 1.0) < 1e-12


def test_numpy_sqrt_call():
    assert safe_math_eval("np.sqrt(9)") == 3.0


def test_builtin_abs():
    assert safe_math_eval("abs(-4)") == 4


def test_builtin_round_kwargs():
    assert safe_math_eval("round(3.14159, ndigits=2)") == 3.14


# ── 名称解析 ─────────────────────────────────────────────────────────────

def test_undefined_name():
    with pytest.raises(SafeEvalError, match="Undefined name"):
        safe_math_eval("foo + 1")


def test_locals_injected():
    assert safe_math_eval("a + b", {"a": 2, "b": 3}) == 5


def test_locals_non_callable_not_whitelisted():
    # callable 注入不应被自动放行
    with pytest.raises(SafeEvalError, match="Function call"):
        safe_math_eval("f(1)", {"f": lambda x: x})


# ── 拒绝路径 ─────────────────────────────────────────────────────────────

def test_call_not_allowed_function():
    with pytest.raises(SafeEvalError, match="Function call is not allowed"):
        safe_math_eval("f(1)", {"f": lambda x: x})


def test_attribute_access_forbidden():
    with pytest.raises(SafeEvalError, match="Attribute access is not allowed"):
        safe_math_eval("np.random.rand()")


def test_forbidden_construct():
    with pytest.raises(SafeEvalError, match="Forbidden expression construct"):
        safe_math_eval("lambda x: x")


def test_import_inside_expr_forbidden():
    with pytest.raises(SafeEvalError):
        safe_math_eval("__import__('os')")


def test_comprehension_forbidden():
    with pytest.raises(SafeEvalError):
        safe_math_eval("[x for x in range(3)]")


def test_invalid_syntax():
    with pytest.raises(SafeEvalError, match="Invalid syntax"):
        safe_math_eval("1 +")