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
