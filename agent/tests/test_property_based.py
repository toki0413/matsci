"""Property-based tests for key matsci-agent modules using hypothesis.

Covered properties:
- DempsterShaferCombiner: sum invariant, commutativity, associativity, boundary
  cases (reinforce pass/fail, total-uncertainty identity, total-conflict fallback)
- MultiFidelityTool nested DOE: HF ⊂ LF nesting, shape consistency, point bounds
- WetlabRpcTool protocol validation: valid params accepted, out-of-range / NaN / inf
  rejected, missing required fields rejected
- PDE classify: classification matches discriminant sign, discriminant value finite
  (the PDE module does symbolic analysis only — no numerical solve — so "boundary
  condition" properties don't apply here)
- Variational euler_lagrange: polynomial Lagrangians produce valid EL expressions
  (symbolic computation — no NaN/inf possible, "convergence" is trivially satisfied)
- DiffGeo metric: symmetric positive-definite metrics succeed, output preserves
  symmetry, determinant stays positive
"""

from __future__ import annotations

import asyncio
import math

import numpy as np
import pytest
import sympy as sp
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from huginn.autoloop.phase_gate import DempsterShaferCombiner
from huginn.tools.sci.multi_fidelity_tool import MultiFidelityTool
from huginn.tools.symbolic_math.tool import SymbolicMathInput, SymbolicMathTool
from huginn.tools.wetlab_rpc_tool import PROTOCOLS, WetlabInput, WetlabRpcTool
from huginn.types import ToolContext


# ── shared helpers ───────────────────────────────────────────────────────────


def _make_ctx() -> ToolContext:
    """Lightweight ToolContext for symbolic-math tool calls."""
    return ToolContext(session_id="prop-test", workspace=".")


# ── Dempster-Shafer strategies ───────────────────────────────────────────────


@st.composite
def mass_tuple(draw):
    """Draw a (pass, fail, unc) triple that sums to 1.0.

    Uniform over the simplex — three random weights normalised to unit sum.
    """
    a = draw(st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False))
    b = draw(st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False))
    c = draw(st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False))
    total = a + b + c
    assume(total > 1e-10)  # skip the degenerate all-zero draw
    return (a / total, b / total, c / total)


def _conflict(m1, m2):
    """Dempster conflict K = m_pass1*m_fail2 + m_fail1*m_pass2. K>=1 is total."""
    return m1[0] * m2[1] + m1[1] * m2[0] >= 1.0


# ── Wetlab protocol strategies ───────────────────────────────────────────────


def _valid_value_for_spec(spec):
    """Build a hypothesis strategy for one protocol param given its schema spec."""
    ptype = spec["type"]
    rng = spec.get("range")
    if ptype == "int" and rng:
        return st.integers(min_value=int(rng[0]), max_value=int(rng[1]))
    if ptype == "float" and rng:
        return st.floats(
            min_value=rng[0], max_value=rng[1],
            allow_nan=False, allow_infinity=False,
        )
    if ptype == "list[float]" and rng:
        # range applies per-element (validation only checks scalars, but
        # we still draw in-range values so the submission is realistic)
        return st.lists(
            st.floats(
                min_value=rng[0], max_value=rng[1],
                allow_nan=False, allow_infinity=False,
            ),
            min_size=2, max_size=2,
        )
    if ptype == "str":
        return st.text(min_size=1, max_size=20)
    return st.just(1.0)


@st.composite
def valid_submission(draw):
    """Draw (protocol, valid_params, valid_sample) with all required fields filled."""
    protocol = draw(st.sampled_from(sorted(PROTOCOLS.keys())))
    schema = PROTOCOLS[protocol]["params"]
    params = {}
    for name, spec in schema.items():
        if spec.get("required", False):
            params[name] = draw(_valid_value_for_spec(spec))
    sample = {f: f"{f}_val" for f in PROTOCOLS[protocol]["sample_fields"]}
    return protocol, params, sample


@st.composite
def submission_with_bad_numeric(draw):
    """Take a valid submission and corrupt one scalar numeric param.

    Returns (protocol, params, sample, target_name) where target has been
    replaced with an out-of-range, NaN, or inf value.
    """
    protocol, params, sample = draw(valid_submission())
    schema = PROTOCOLS[protocol]["params"]
    # Only scalar numerics are range-checked by the validator — list params
    # skip the isinstance(value, (int, float)) gate.
    numeric_required = [
        n for n, s in schema.items()
        if n in params and s.get("range") and s["type"] in ("int", "float")
    ]
    assume(numeric_required)
    target = draw(st.sampled_from(numeric_required))
    spec = schema[target]
    low, high = spec["range"]
    bad_type = draw(st.sampled_from(["below", "above", "nan", "inf"]))
    if bad_type == "below":
        bad_value = low - abs(low) - 1
    elif bad_type == "above":
        bad_value = high + abs(high) + 1
    elif bad_type == "nan":
        bad_value = float("nan")
    else:
        bad_value = float("inf")
    params = dict(params)
    params[target] = bad_value
    return protocol, params, sample, target


@st.composite
def submission_missing_one_required(draw):
    """Take a valid submission and drop one required param."""
    protocol, params, sample = draw(valid_submission())
    schema = PROTOCOLS[protocol]["params"]
    required = [n for n, s in schema.items() if s.get("required", False)]
    target = draw(st.sampled_from(required))
    params = dict(params)
    del params[target]
    return protocol, params, sample, target


# ── PDE / Variational / DiffGeo strategies ───────────────────────────────────


@st.composite
def pde_coefficients(draw):
    """Integer A, B, C for PDE classification (skip the all-zero degenerate)."""
    A = draw(st.integers(min_value=-5, max_value=5))
    B = draw(st.integers(min_value=-5, max_value=5))
    C = draw(st.integers(min_value=-5, max_value=5))
    assume(not (A == 0 and B == 0 and C == 0))
    return A, B, C


@st.composite
def polynomial_lagrangian(draw):
    """L = a/2 * u'^2 + b/2 * u^2 with small integer coefficients.

    a stays positive so the kinetic term is always present — keeps the EL
    expression non-trivial and avoids the u''=0 edge case.
    """
    a = draw(st.integers(min_value=1, max_value=10))
    b = draw(st.integers(min_value=-10, max_value=10))
    return f"{a}/2 * u'**2 + {b}/2 * u**2", a, b


@st.composite
def diagonal_pd_metric_2d(draw):
    """Diagonal positive-definite 2×2 metric as a string matrix for diffgeo."""
    a = draw(st.floats(min_value=0.1, max_value=10, allow_nan=False, allow_infinity=False))
    b = draw(st.floats(min_value=0.1, max_value=10, allow_nan=False, allow_infinity=False))
    return [[str(a), "0"], ["0", str(b)]], a, b


# ════════════════════════════════════════════════════════════════════════════
# DempsterShaferCombiner
# ════════════════════════════════════════════════════════════════════════════


class TestDempsterShaferProperties:
    """Property-based tests for Dempster-Shafer evidence combination."""

    @given(mass_tuple(), mass_tuple())
    @settings(max_examples=50)
    def test_sum_invariant(self, m1, m2):
        # Combined masses always sum to 1.0 (after normalisation or conflict fallback)
        result = DempsterShaferCombiner.combine_pair(m1, m2)
        assert sum(result) == pytest.approx(1.0, abs=1e-9)

    @given(mass_tuple(), mass_tuple())
    @settings(max_examples=50)
    def test_commutativity(self, m1, m2):
        # Order of combination must not matter
        r1 = DempsterShaferCombiner.combine_pair(m1, m2)
        r2 = DempsterShaferCombiner.combine_pair(m2, m1)
        assert r1[0] == pytest.approx(r2[0], abs=1e-9)
        assert r1[1] == pytest.approx(r2[1], abs=1e-9)
        assert r1[2] == pytest.approx(r2[2], abs=1e-9)

    @given(mass_tuple(), mass_tuple(), mass_tuple())
    @settings(max_examples=50)
    def test_associativity(self, a, b, c):
        # Skip total-conflict pairings — the (0,1,0) fallback is not associative
        assume(not _conflict(a, b))
        assume(not _conflict(b, c))
        assume(not _conflict(a, c))
        left = DempsterShaferCombiner.combine_pair(
            DempsterShaferCombiner.combine_pair(a, b), c
        )
        right = DempsterShaferCombiner.combine_pair(
            a, DempsterShaferCombiner.combine_pair(b, c)
        )
        assert left[0] == pytest.approx(right[0], abs=1e-6)
        assert left[1] == pytest.approx(right[1], abs=1e-6)
        assert left[2] == pytest.approx(right[2], abs=1e-6)

    def test_reinforce_pass(self):
        # Two certain-pass sources → still certain pass
        r = DempsterShaferCombiner.combine_pair((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        assert r == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)

    def test_reinforce_fail(self):
        r = DempsterShaferCombiner.combine_pair((0.0, 1.0, 0.0), (0.0, 1.0, 0.0))
        assert r == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)

    @given(mass_tuple())
    @settings(max_examples=50)
    def test_total_uncertainty_identity(self, x):
        # Combining with full uncertainty is a no-op
        r = DempsterShaferCombiner.combine_pair((0.0, 0.0, 1.0), x)
        assert r[0] == pytest.approx(x[0], abs=1e-9)
        assert r[1] == pytest.approx(x[1], abs=1e-9)
        assert r[2] == pytest.approx(x[2], abs=1e-9)

    def test_total_conflict_returns_all_fail(self):
        # pass vs fail with no uncertainty → combiner falls back to all-fail
        r = DempsterShaferCombiner.combine_pair((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        assert r == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)


# ════════════════════════════════════════════════════════════════════════════
# MultiFidelityTool nested DOE
# ════════════════════════════════════════════════════════════════════════════


class TestNestedDoeProperties:
    """Property-based tests for nested design-of-experiments (Qian 2009)."""

    @given(
        n_hf=st.integers(min_value=1, max_value=10),
        n_lf=st.integers(min_value=1, max_value=30),
        dim=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=50)
    def test_nested_doe_properties(self, n_hf, n_lf, dim, seed):
        assume(n_hf <= n_lf)  # nested design requires HF ⊆ LF
        tool = MultiFidelityTool()
        result = asyncio.run(tool.call({
            "action": "nested_doe",
            "n_hf": n_hf,
            "n_lf": n_lf,
            "dim": dim,
            "seed": seed,
        }))
        assert result.success, f"nested_doe failed: {result.error}"
        data = result.data

        X_hf = np.array(data["X_hf"])
        X_lf = np.array(data["X_lf"])

        # Shape consistency
        assert X_hf.shape == (n_hf, dim)
        assert X_lf.shape == (n_lf, dim)

        # HF ⊂ LF: the first n_hf LF points are exactly the HF points
        np.testing.assert_array_almost_equal(X_lf[:n_hf], X_hf)

        # Default bounds are [0, 1] — LHS stays in standardised space
        assert np.all(X_lf >= 0.0 - 1e-9)
        assert np.all(X_lf <= 1.0 + 1e-9)
        assert np.all(X_hf >= 0.0 - 1e-9)
        assert np.all(X_hf <= 1.0 + 1e-9)


# ════════════════════════════════════════════════════════════════════════════
# WetlabRpcTool protocol validation
# ════════════════════════════════════════════════════════════════════════════


class TestWetlabProtocolProperties:
    """Property-based tests for wetlab protocol parameter validation."""

    @given(valid_submission())
    @settings(max_examples=50)
    @pytest.mark.asyncio
    async def test_valid_submission_accepted(self, submission):
        protocol, params, sample = submission
        tool = WetlabRpcTool()
        args = WetlabInput(
            action="submit_protocol",
            protocol=protocol,
            params=params,
            sample=sample,
        )
        result = await tool.call(args, context=None)
        assert result.success, (
            f"valid {protocol} submission rejected: {result.error}"
        )

    @given(submission_with_bad_numeric())
    @settings(max_examples=50)
    @pytest.mark.asyncio
    async def test_bad_numeric_rejected(self, submission):
        # Covers out-of-range, NaN, and inf — all three must be rejected
        protocol, params, sample, target = submission
        tool = WetlabRpcTool()
        args = WetlabInput(
            action="submit_protocol",
            protocol=protocol,
            params=params,
            sample=sample,
        )
        result = await tool.call(args, context=None)
        assert not result.success, (
            f"{protocol}.{target} bad value should be rejected"
        )
        assert target in result.error

    @given(submission_missing_one_required())
    @settings(max_examples=50)
    @pytest.mark.asyncio
    async def test_missing_required_rejected(self, submission):
        protocol, params, sample, target = submission
        tool = WetlabRpcTool()
        args = WetlabInput(
            action="submit_protocol",
            protocol=protocol,
            params=params,
            sample=sample,
        )
        result = await tool.call(args, context=None)
        assert not result.success
        assert target in result.error


# ════════════════════════════════════════════════════════════════════════════
# PDE classify
# ════════════════════════════════════════════════════════════════════════════


class TestPdeClassifyProperties:
    """Property-based tests for PDE classification.

    The pde module does symbolic analysis (classification, separation,
    characteristics, stencil generation) — there's no numerical PDE solve,
    so "boundary condition" properties don't apply. We focus on classification
    correctness and finiteness of the discriminant value.
    """

    @given(pde_coefficients())
    @settings(max_examples=50)
    @pytest.mark.asyncio
    async def test_classification_matches_discriminant(self, coeffs):
        A, B, C = coeffs
        tool = SymbolicMathTool()
        ctx = _make_ctx()
        args = SymbolicMathInput(
            action="pde_classify",
            expression=f"{A};{B};{C}",
        )
        result = await tool.call(args, ctx)
        assert result.success, f"classify failed: {result.error}"

        disc = B * B - 4 * A * C
        classification = result.data["classification"]

        if disc < 0:
            assert classification == "elliptic"
        elif disc == 0:
            assert classification == "parabolic"
        else:
            assert classification == "hyperbolic"

        # discriminant_value must be finite (no NaN/inf) for integer inputs
        dv = result.data["discriminant_value"]
        assert dv is not None
        assert math.isfinite(dv.real)
        assert abs(dv.imag) < 1e-12


# ════════════════════════════════════════════════════════════════════════════
# Variational euler_lagrange
# ════════════════════════════════════════════════════════════════════════════


class TestVariationalProperties:
    """Property-based tests for Euler-Lagrange derivation.

    The variational module is fully symbolic — "convergence" in the numerical
    sense doesn't apply. We verify that reasonable Lagrangians produce valid
    EL expressions and check mathematical correctness against a hand-computed
    reference.
    """

    @given(polynomial_lagrangian())
    @settings(max_examples=50)
    @pytest.mark.asyncio
    async def test_polynomial_lagrangian_succeeds(self, lagrangian):
        L_str, a, b = lagrangian
        tool = SymbolicMathTool()
        ctx = _make_ctx()
        args = SymbolicMathInput(
            action="euler_lagrange",
            expression=L_str,
            symbols=["u", "x"],
            variable="u",
        )
        result = await tool.call(args, ctx)
        assert result.success, f"EL failed for L={L_str}: {result.error}"

        el_str = result.data["euler_lagrange"]
        assert isinstance(el_str, str)
        assert len(el_str) > 0
        # EL must reference the field u — either as u(x) or as a Derivative
        assert "u" in el_str
        # Kinetic term a/2 * u'^2 always produces a second-derivative term
        assert "Derivative" in el_str or "u''" in el_str

        # Mathematical correctness: parse the EL string back and compare
        # against the hand-computed reference b*u(x) - a*u''(x) = 0.
        el_body = el_str.split(" = ")[0].strip()
        x = sp.Symbol("x")
        u = sp.Function("u")
        el_expr = sp.sympify(el_body, locals={"u": u, "x": x})
        u_func = u(x)
        expected = b * u_func - a * sp.Derivative(u_func, (x, 2))
        diff = sp.simplify(el_expr - expected)
        assert diff == 0, (
            f"EL mismatch for L={L_str}: got {el_expr}, expected {expected}"
        )


# ════════════════════════════════════════════════════════════════════════════
# DiffGeo metric
# ════════════════════════════════════════════════════════════════════════════


class TestDiffgeoMetricProperties:
    """Property-based tests for differential geometry metric computation.

    Properties:
    - Symmetric positive-definite metrics produce valid Christoffel symbols
    - Output metric preserves input symmetry (g_ij == g_ji)
    - Determinant stays positive for PD inputs
    """

    @given(diagonal_pd_metric_2d())
    @settings(max_examples=50)
    @pytest.mark.asyncio
    async def test_symmetric_pd_metric_succeeds(self, metric_data):
        matrix, a, b = metric_data
        tool = SymbolicMathTool()
        ctx = _make_ctx()
        args = SymbolicMathInput(
            action="diffgeo_metric",
            target="christoffel",
            symbols=["x", "y"],
            matrix=matrix,
        )
        result = await tool.call(args, ctx)
        assert result.success, f"metric failed: {result.error}"

        # Output metric is 2×2
        out = result.data["metric"]
        assert len(out) == 2
        assert len(out[0]) == 2

        # Symmetry: g_01 == g_10 (parse via sympy to handle string repr differences)
        g01 = sp.sympify(out[0][1])
        g10 = sp.sympify(out[1][0])
        assert sp.simplify(g01 - g10) == 0

        # Determinant positive: det = g00 * g11 - g01^2 > 0 for PD input
        g00 = sp.sympify(out[0][0])
        g11 = sp.sympify(out[1][1])
        det = sp.simplify(g00 * g11 - g01 ** 2)
        det_val = float(sp.N(det))
        assert det_val > 0, f"det={det_val} should be positive for PD metric"

        # n_nonzero is a non-negative integer
        assert isinstance(result.data["n_nonzero"], int)
        assert result.data["n_nonzero"] >= 0