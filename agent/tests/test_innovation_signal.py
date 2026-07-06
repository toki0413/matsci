"""Innovation signal detector + LiteratureGrader tests.

Verifies that significant deviations from literature are flagged as
potential innovations (not just errors), and that the LiteratureGrader
scores them correctly.
"""
from __future__ import annotations

import pytest

from huginn.validation.innovation_signal import (
    DeviationLevel,
    InnovationSignal,
    InnovationSignalDetector,
)
from huginn.validation.grader import LiteratureGrader, default_registry


# ── InnovationSignalDetector ──────────────────────────────────


def test_deviation_within_1sigma_is_negligible():
    """Agent value within 1 sigma -> NEGLIGIBLE, not an innovation signal."""
    det = InnovationSignalDetector()
    # literature: mean=10, std~1
    lit = [9.0, 10.0, 11.0, 10.5, 9.5]
    sig = det.detect("band_gap", 10.2, lit)
    assert sig.level == DeviationLevel.NEGLIGIBLE
    assert sig.deviation_sigma < 1.0
    assert sig.is_innovation_signal is False


def test_deviation_beyond_2sigma_is_innovation():
    """Agent value > 2 sigma from consensus -> SIGNIFICANT, innovation signal."""
    det = InnovationSignalDetector()
    lit = [9.0, 10.0, 11.0, 10.5, 9.5]  # mean~10, std~0.79
    sig = det.detect("band_gap", 12.0, lit)
    assert sig.deviation_sigma > 2.0
    assert sig.deviation_sigma < 3.0
    assert sig.level == DeviationLevel.SIGNIFICANT
    assert sig.is_innovation_signal is True


def test_deviation_beyond_3sigma_is_extraordinary():
    """Agent value > 3 sigma -> EXTRAORDINARY level."""
    det = InnovationSignalDetector()
    lit = [1.0, 1.1, 0.9, 1.05, 0.95]  # mean~1, std~0.065
    sig = det.detect("energy", 3.0, lit)
    assert sig.deviation_sigma > 3.0
    assert sig.level == DeviationLevel.EXTRAORDINARY


def test_nan_value_is_not_innovation():
    """NaN / inf are physically implausible -> never an innovation signal."""
    det = InnovationSignalDetector()
    lit = [1.0, 2.0, 3.0, 2.5, 1.5]
    sig = det.detect("energy", float("nan"), lit)
    assert sig.is_innovation_signal is False

    sig_inf = det.detect("energy", float("inf"), lit)
    assert sig_inf.is_innovation_signal is False


def test_possible_explanations_generated():
    """When deviation is significant, possible_explanations should be non-empty."""
    det = InnovationSignalDetector()
    lit = [10.0, 10.1, 9.9, 10.05, 9.95]
    sig = det.detect("band_gap", 15.0, lit)
    assert len(sig.possible_explanations) > 0
    # Innovation signals should mention "discovery"
    joined = " ".join(sig.possible_explanations).lower()
    assert "discover" in joined or "novel" in joined


def test_empty_literature_handled_gracefully():
    """Empty or single-value literature list should not crash."""
    det = InnovationSignalDetector()
    sig = det.detect("band_gap", 1.5, [])
    assert sig.is_innovation_signal is False
    assert sig.level == DeviationLevel.NEGLIGIBLE

    sig_one = det.detect("band_gap", 1.5, [1.0])
    assert sig_one.is_innovation_signal is False


# ── LiteratureGrader ──────────────────────────────────────────


def _make_signal(
    prop: str,
    agent_val: float,
    consensus: float,
    spread: float,
    sigma: float,
    level: DeviationLevel,
    is_signal: bool = False,
) -> InnovationSignal:
    """Build an InnovationSignal with explicit fields for grader tests."""
    return InnovationSignal(
        property_name=prop,
        agent_value=agent_val,
        literature_consensus=consensus,
        literature_spread=spread,
        deviation_sigma=sigma,
        level=level,
        possible_explanations=[],
        is_innovation_signal=is_signal,
    )


def test_literature_grader_within_2sigma_scores_1():
    """Within 2 sigma -> score 1.0 (agrees with literature)."""
    g = LiteratureGrader()
    data = {
        "literature_comparison": {
            "band_gap": _make_signal(
                "band_gap", 1.1, 1.0, 0.1, 1.0,
                DeviationLevel.NEGLIGIBLE,
            ),
        }
    }
    res = g.evaluate(data)
    assert res.score == pytest.approx(1.0)
    assert res.passed is True


def test_literature_grader_2to3_sigma_scores_0_5():
    """2-3 sigma without innovation signal -> score 0.5 (interesting)."""
    g = LiteratureGrader()
    data = {
        "literature_comparison": {
            "band_gap": _make_signal(
                "band_gap", 2.25, 1.0, 0.5, 2.5,
                DeviationLevel.SIGNIFICANT,
                is_signal=False,
            ),
        }
    }
    res = g.evaluate(data)
    assert res.score == pytest.approx(0.5)


def test_literature_grader_beyond_3sigma_implausible_scores_0():
    """>3 sigma and not an innovation signal -> score 0.0 (likely error)."""
    g = LiteratureGrader()
    data = {
        "literature_comparison": {
            "band_gap": _make_signal(
                "band_gap", 2.5, 1.0, 0.5, 3.0,
                DeviationLevel.EXTRAORDINARY,
                is_signal=False,
            ),
        }
    }
    res = g.evaluate(data)
    assert res.score == pytest.approx(0.0)


def test_literature_grader_innovation_keeps_0_5():
    """Innovation signal (even > 3 sigma) keeps score at 0.5, not 0.0."""
    g = LiteratureGrader()
    data = {
        "literature_comparison": {
            "band_gap": _make_signal(
                "band_gap", 5.0, 1.0, 0.5, 8.0,
                DeviationLevel.EXTRAORDINARY,
                is_signal=True,
            ),
        }
    }
    res = g.evaluate(data)
    assert res.score == pytest.approx(0.5)
    assert res.passed is True


def test_literature_grader_no_data_neutral():
    """No literature_comparison in data -> neutral 1.0."""
    g = LiteratureGrader()
    res = g.evaluate({"tool_name": "vasp_tool"})
    assert res.score == pytest.approx(1.0)
    assert res.passed is True


def test_literature_grader_accepts_dict_signals():
    """LiteratureGrader should accept plain dicts (from serialization)."""
    g = LiteratureGrader()
    data = {
        "literature_comparison": {
            "band_gap": {
                "property_name": "band_gap",
                "agent_value": 1.1,
                "literature_consensus": 1.0,
                "literature_spread": 0.1,
                "deviation_sigma": 1.0,
                "level": "NEGLIGIBLE",
                "possible_explanations": [],
                "is_innovation_signal": False,
            },
        }
    }
    res = g.evaluate(data)
    assert res.score == pytest.approx(1.0)


def test_default_registry_includes_literature():
    """default_registry() should register the literature grader."""
    reg = default_registry()
    assert "literature" in reg.names()


def test_literature_grader_mixed_signals_average():
    """Multiple properties: average of individual scores."""
    g = LiteratureGrader()
    data = {
        "literature_comparison": {
            "band_gap": _make_signal(
                "band_gap", 1.1, 1.0, 0.1, 1.0,
                DeviationLevel.NEGLIGIBLE,  # score 1.0
            ),
            "energy": _make_signal(
                "energy", 5.0, 1.0, 0.5, 8.0,
                DeviationLevel.EXTRAORDINARY, is_signal=True,  # score 0.5
            ),
        }
    }
    res = g.evaluate(data)
    # (1.0 + 0.5) / 2 = 0.75
    assert res.score == pytest.approx(0.75)
