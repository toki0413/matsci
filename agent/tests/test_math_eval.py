"""Tests for the restricted mathematical expression evaluator."""

from __future__ import annotations

import math

import numpy as np
import pytest

from huginn.security.math_eval import SafeEvalError, safe_math_eval


class TestSafeMathEval:
    def test_simple_arithmetic(self):
        assert safe_math_eval("1 + 2 * 3") == 7
        assert safe_math_eval("(1 + 2) * 3") == 9
        assert safe_math_eval("2 ** 10") == 1024

    def test_locals(self):
        assert safe_math_eval("x**2 + 2*x + 1", {"x": 3}) == 16
        assert safe_math_eval("a + b", {"a": 1.5, "b": 2.5}) == 4.0

    def test_numpy_functions(self):
        assert math.isclose(safe_math_eval("np.sin(np.pi / 2)"), 1.0)
        assert math.isclose(safe_math_eval("np.cos(0)"), 1.0)
        assert math.isclose(safe_math_eval("np.exp(1)"), math.e)
        assert math.isclose(safe_math_eval("np.log(np.e)"), 1.0)
        assert math.isclose(safe_math_eval("np.sqrt(16)"), 4.0)

    def test_subscripts(self):
        assert safe_math_eval("X[0] + X[1]", {"X": [1, 2, 3]}) == 3
        assert safe_math_eval("y[0] * t", {"y": [4, 5], "t": 2}) == 8

    def test_comparisons_and_conditionals(self):
        assert safe_math_eval("x > 2", {"x": 3}) is True
        assert safe_math_eval("x if x > 0 else -x", {"x": -5}) == 5

    def test_rejects_import(self):
        with pytest.raises(SafeEvalError):
            safe_math_eval("__import__('os').system('ls')")

    def test_rejects_open(self):
        with pytest.raises(SafeEvalError):
            safe_math_eval("open('secret.txt').read()")

    def test_rejects_arbitrary_attribute(self):
        with pytest.raises(SafeEvalError):
            safe_math_eval("x.__class__", {"x": 1})

    def test_rejects_non_whitelisted_numpy_attribute(self):
        with pytest.raises(SafeEvalError):
            safe_math_eval("np.load('evil.npy')")

    def test_rejects_lambda(self):
        with pytest.raises(SafeEvalError):
            safe_math_eval("(lambda: 1)()")

    def test_rejects_arbitrary_call(self):
        with pytest.raises(SafeEvalError):
            safe_math_eval("foo()", {"foo": lambda: 1})


class TestSafeMathEvalVectors:
    def test_vector_expression(self):
        result = safe_math_eval("np.sqrt(X[0]**2 + X[1]**2)", {"X": [3.0, 4.0]})
        assert math.isclose(result, 5.0)

    def test_list_literal(self):
        result = safe_math_eval("[x, x**2, x**3]", {"x": 2})
        assert result == [2, 4, 8]

    def test_numpy_constants(self):
        assert math.isclose(safe_math_eval("np.pi"), np.pi)
        assert math.isclose(safe_math_eval("np.e"), np.e)


# ── 全分支扩展 (原 test_math_eval_ext.py) ────────────────────────────────

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


def test_tuple():
    assert safe_math_eval("(1, 2, 3)") == (1, 2, 3)


def test_list():
    assert safe_math_eval("[1, 2, 3]") == [1, 2, 3]


def test_subscript():
    assert safe_math_eval("[1, 2, 3][1]") == 2


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


def test_undefined_name():
    with pytest.raises(SafeEvalError, match="Undefined name"):
        safe_math_eval("foo + 1")


def test_locals_injected():
    assert safe_math_eval("a + b", {"a": 2, "b": 3}) == 5


def test_locals_non_callable_not_whitelisted():
    with pytest.raises(SafeEvalError, match="Function call"):
        safe_math_eval("f(1)", {"f": lambda x: x})


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
