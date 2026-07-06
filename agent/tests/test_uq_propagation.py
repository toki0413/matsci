"""Tests for huginn.autoloop.uq_propagation.

Covers linear (GUM) propagation, Monte Carlo propagation, and the ChainTracker
that stitches tool outputs into a cumulative uncertainty chain.
"""

from __future__ import annotations

import math

import pytest

from huginn.autoloop.uq_propagation import (
    ChainTracker,
    UQChain,
    UQState,
    linear_propagate,
    monte_carlo_propagate,
)


# ── UQState basics ─────────────────────────────────────────────────────────


class TestUQState:
    def test_defaults_and_clamp(self):
        s = UQState(value=3.0)
        assert s.sigma == 0.0
        assert s.method == "exact"
        # negative sigma is meaningless -> clamped to 0
        bad = UQState(value=1.0, sigma=-2.0)
        assert bad.sigma == 0.0


# ── linear_propagate ────────────────────────────────────────────────────────


class TestLinearPropagate:
    def test_zero_sensitivity_no_intrinsic(self):
        # dB/dA == 0 and no intrinsic noise -> output is exact
        inp = UQState(value=10.0, sigma=0.5, source="A")
        out = linear_propagate(inp, sensitivity_fn=0.0, output_value=20.0)
        assert out.sigma == pytest.approx(0.0)
        assert out.value == 20.0
        assert out.method == "linear"
        assert out.source == "A"

    def test_zero_sensitivity_with_intrinsic(self):
        # sigma_out should collapse to the intrinsic floor
        inp = UQState(value=10.0, sigma=0.5)
        out = linear_propagate(inp, sensitivity_fn=0.0, output_value=20.0,
                              intrinsic_sigma=0.4)
        assert out.sigma == pytest.approx(0.4)

    def test_known_sensitivity(self):
        # sigma_out = sqrt((2*0.5)^2 + 0.3^2) = sqrt(1.09)
        inp = UQState(value=10.0, sigma=0.5)
        out = linear_propagate(inp, sensitivity_fn=2.0, output_value=20.0,
                              intrinsic_sigma=0.3)
        assert out.sigma == pytest.approx(math.sqrt(1.09))
        assert out.value == 20.0

    def test_callable_sensitivity_evaluated_at_input(self):
        # a callable dB/dA(x) should match the scalar at the input value
        inp = UQState(value=10.0, sigma=0.5)
        scalar_out = linear_propagate(inp, sensitivity_fn=2.0, output_value=20.0)
        fn_out = linear_propagate(inp, sensitivity_fn=lambda x: 2.0,
                                  output_value=20.0)
        assert fn_out.sigma == pytest.approx(scalar_out.sigma)


# ── monte_carlo_propagate ──────────────────────────────────────────────────


class TestMonteCarloPropagate:
    def test_constant_function_zero_sigma(self):
        # f(x) = 42 for all x -> output spread is exactly 0
        inp = UQState(value=5.0, sigma=0.5)
        out = monte_carlo_propagate(inp, fn=lambda x: 42.0, n_samples=200)
        assert out.sigma == pytest.approx(0.0)
        assert out.value == pytest.approx(42.0)
        assert out.method == "monte_carlo"

    def test_zero_input_sigma_exact(self):
        # degenerate input distribution -> no spread, value == fn(input)
        inp = UQState(value=5.0, sigma=0.0)
        out = monte_carlo_propagate(inp, fn=lambda x: 3.0 * x, n_samples=100)
        assert out.sigma == pytest.approx(0.0)
        assert out.value == pytest.approx(15.0)

    def test_linear_function_propagates_slope(self):
        # y = 3x, x ~ N(10, 0.5) -> std(y) = 3 * 0.5 = 1.5, mean(y) = 30
        inp = UQState(value=10.0, sigma=0.5)
        out = monte_carlo_propagate(inp, fn=lambda x: 3.0 * x, n_samples=8000)
        assert out.sigma == pytest.approx(1.5, rel=0.08)
        assert out.value == pytest.approx(30.0, rel=0.02)


# ── ChainTracker / UQChain ──────────────────────────────────────────────────


class TestChainTracker:
    def test_accumulates_states(self):
        tracker = ChainTracker()
        tracker.add("tool_a", UQState(1.0, 0.1, "a"),
                    UQState(2.0, 0.2, "a", "linear"))
        tracker.add("tool_b", UQState(2.0, 0.2, "a"),
                    UQState(4.0, 0.4, "b", "linear"))
        assert len(tracker.steps) == 2
        names = [name for name, _inp, _out in tracker.steps]
        assert names == ["tool_a", "tool_b"]

    def test_summary_includes_cumulative_uncertainty(self):
        tracker = ChainTracker()
        tracker.add("tool_a", UQState(1.0, 0.1, "a"),
                    UQState(2.0, 0.2, "a", "linear"))
        tracker.add("tool_b", UQState(2.0, 0.2, "a"),
                    UQState(4.0, 0.4, "b", "linear"))
        chain = tracker.summary()
        assert isinstance(chain, UQChain)
        assert len(chain.states) == 2
        # cumulative sigma is the last link's sigma (each link folds upstream in)
        assert chain.cumulative_sigma == pytest.approx(0.4)
        assert chain.states[-1].source == "tool_b"

    def test_empty_chain_cumulative_zero(self):
        chain = UQChain()
        assert chain.cumulative_sigma == 0.0
        assert chain.summary()["n_steps"] == 0

    def test_uqchain_propagate_appends_linear(self):
        chain = UQChain()
        inp = UQState(value=10.0, sigma=0.5, source="A")
        out = chain.propagate(inp, sensitivity=2.0, output_value=20.0,
                             intrinsic_sigma=0.3)
        assert out.sigma == pytest.approx(math.sqrt(1.09))
        assert len(chain.states) == 1
        assert chain.cumulative_sigma == pytest.approx(math.sqrt(1.09))
