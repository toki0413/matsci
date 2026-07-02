"""Tests for multi-fidelity fusion tool (W3 M2).

Locks behaviour:
- register_source: valid/invalid/override
- fit_surrogate: single source, two-level autoregressive, rho estimation
- propagate: prediction + uncertainty
- select_next: cost-aware EI selection
- ToolProfile: light cost, PLANNING+VALIDATION+OPEN phases
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from huginn.phases import ResearchPhase
from huginn.tools.multi_fidelity_tool import (
    FidelitySource,
    MultiFidelityInput,
    MultiFidelitySurrogate,
    MultiFidelityTool,
)
from huginn.tools.registry import ToolRegistry
from huginn.types import ToolContext, ToolResult


@pytest.fixture(autouse=True)
def _clean_registry():
    ToolRegistry.clear()
    yield
    ToolRegistry.clear()


def _call(tool, args, ctx=None):
    return asyncio.run(tool.call(args, ctx))


# ── register_source ──────────────────────────────────────────────────────────


class TestRegisterSource:
    def test_register_valid_source(self):
        tool = MultiFidelityTool()
        result = _call(tool, {
            "action": "register_source",
            "name": "empirical",
            "level": 0,
            "cost": 0.01,
            "X": [[0.0], [1.0], [2.0]],
            "y": [0.0, 1.0, 4.0],
        })
        assert result.success is True
        assert result.data["name"] == "empirical"
        assert result.data["level"] == 0
        assert result.data["cost"] == 0.01
        assert result.data["n_points"] == 3

    def test_register_missing_name(self):
        tool = MultiFidelityTool()
        result = _call(tool, {
            "action": "register_source",
            "level": 0,
            "X": [[0.0]],
            "y": [0.0],
        })
        assert result.success is False
        assert "name" in result.error

    def test_register_missing_level(self):
        tool = MultiFidelityTool()
        result = _call(tool, {
            "action": "register_source",
            "name": "src",
            "X": [[0.0]],
            "y": [0.0],
        })
        assert result.success is False
        assert "level" in result.error

    def test_register_mismatched_lengths(self):
        tool = MultiFidelityTool()
        result = _call(tool, {
            "action": "register_source",
            "name": "src",
            "level": 0,
            "X": [[0.0], [1.0]],
            "y": [0.0],
        })
        assert result.success is False
        assert "长度不一致" in result.error

    def test_register_empty_data(self):
        tool = MultiFidelityTool()
        result = _call(tool, {
            "action": "register_source",
            "name": "src",
            "level": 0,
            "X": [],
            "y": [],
        })
        assert result.success is False

    def test_register_same_name_overrides(self):
        tool = MultiFidelityTool()
        _call(tool, {
            "action": "register_source",
            "name": "src",
            "level": 0,
            "cost": 1.0,
            "X": [[0.0]],
            "y": [0.0],
        })
        result = _call(tool, {
            "action": "register_source",
            "name": "src",
            "level": 1,
            "cost": 2.0,
            "X": [[0.0], [1.0]],
            "y": [0.0, 1.0],
        })
        assert result.success is True
        assert result.data["level"] == 1
        assert result.data["n_points"] == 2
        assert len(tool._sources) == 1

    def test_register_default_cost(self):
        tool = MultiFidelityTool()
        result = _call(tool, {
            "action": "register_source",
            "name": "src",
            "level": 0,
            "X": [[0.0]],
            "y": [0.0],
        })
        assert result.success is True
        assert result.data["cost"] == 1.0


# ── fit_surrogate ────────────────────────────────────────────────────────────


class TestFitSurrogate:
    def test_fit_single_source(self):
        tool = MultiFidelityTool()
        _call(tool, {
            "action": "register_source",
            "name": "low",
            "level": 0,
            "X": [[0.0], [1.0], [2.0], [3.0]],
            "y": [0.0, 1.0, 4.0, 9.0],
        })
        result = _call(tool, {"action": "fit_surrogate"})
        assert result.success is True
        assert result.data["fitted"] is True
        assert result.data["n_sources"] == 1
        assert result.data["has_delta"] is False

    def test_fit_two_sources_autoregressive(self):
        tool = MultiFidelityTool()
        # low fidelity: y = x
        _call(tool, {
            "action": "register_source",
            "name": "low",
            "level": 0,
            "cost": 0.1,
            "X": [[i * 0.5] for i in range(8)],
            "y": [i * 0.5 for i in range(8)],
        })
        # high fidelity: y = 2*x (rho should be ~2)
        _call(tool, {
            "action": "register_source",
            "name": "high",
            "level": 1,
            "cost": 10.0,
            "X": [[i * 0.5] for i in range(8)],
            "y": [2.0 * i * 0.5 for i in range(8)],
        })
        result = _call(tool, {"action": "fit_surrogate"})
        assert result.success is True
        assert result.data["n_sources"] == 2
        assert result.data["has_delta"] is True
        assert result.data["rho"] > 1.5  # should be close to 2

    def test_fit_no_sources(self):
        tool = MultiFidelityTool()
        result = _call(tool, {"action": "fit_surrogate"})
        assert result.success is False
        assert "未注册" in result.error

    def test_fit_invalidates_previous_surrogate(self):
        tool = MultiFidelityTool()
        _call(tool, {
            "action": "register_source",
            "name": "low",
            "level": 0,
            "X": [[0.0], [1.0]],
            "y": [0.0, 1.0],
        })
        _call(tool, {"action": "fit_surrogate"})
        assert tool._surrogate is not None
        # re-register invalidates
        _call(tool, {
            "action": "register_source",
            "name": "low",
            "level": 0,
            "X": [[0.0], [1.0], [2.0]],
            "y": [0.0, 1.0, 4.0],
        })
        assert tool._surrogate is None


# ── propagate ────────────────────────────────────────────────────────────────


class TestPropagate:
    def test_propagate_after_fit(self):
        tool = MultiFidelityTool()
        _call(tool, {
            "action": "register_source",
            "name": "low",
            "level": 0,
            "X": [[i * 0.3] for i in range(10)],
            "y": [np.sin(i * 0.3) for i in range(10)],
        })
        _call(tool, {"action": "fit_surrogate"})
        result = _call(tool, {
            "action": "propagate",
            "X_new": [[0.5], [1.0], [1.5]],
        })
        assert result.success is True
        assert result.data["n_points"] == 3
        assert len(result.data["mu"]) == 3
        assert len(result.data["sigma"]) == 3
        # sigma should be non-negative
        assert all(s >= 0 for s in result.data["sigma"])

    def test_propagate_without_fit(self):
        tool = MultiFidelityTool()
        _call(tool, {
            "action": "register_source",
            "name": "low",
            "level": 0,
            "X": [[0.0]],
            "y": [0.0],
        })
        result = _call(tool, {
            "action": "propagate",
            "X_new": [[0.5]],
        })
        assert result.success is False
        assert "未拟合" in result.error

    def test_propagate_missing_X_new(self):
        tool = MultiFidelityTool()
        _call(tool, {
            "action": "register_source",
            "name": "low",
            "level": 0,
            "X": [[0.0], [1.0]],
            "y": [0.0, 1.0],
        })
        _call(tool, {"action": "fit_surrogate"})
        result = _call(tool, {"action": "propagate"})
        assert result.success is False

    def test_propagate_two_level_uncertainty(self):
        """Two-level surrogate should have uncertainty from both GPs."""
        tool = MultiFidelityTool()
        _call(tool, {
            "action": "register_source",
            "name": "low",
            "level": 0,
            "cost": 0.1,
            "X": [[i * 0.3] for i in range(10)],
            "y": [np.sin(i * 0.3) for i in range(10)],
        })
        _call(tool, {
            "action": "register_source",
            "name": "high",
            "level": 1,
            "cost": 10.0,
            "X": [[i * 0.3] for i in range(5)],
            "y": [np.sin(i * 0.3) + 0.01 * i for i in range(5)],
        })
        _call(tool, {"action": "fit_surrogate"})
        result = _call(tool, {
            "action": "propagate",
            "X_new": [[0.5], [1.5]],
        })
        assert result.success is True
        # high-level prediction should have non-zero sigma
        assert all(s > 0 for s in result.data["sigma"])


# ── select_next ──────────────────────────────────────────────────────────────


class TestSelectNext:
    def test_select_returns_candidates(self):
        tool = MultiFidelityTool()
        _call(tool, {
            "action": "register_source",
            "name": "low",
            "level": 0,
            "cost": 1.0,
            "X": [[i * 0.3] for i in range(10)],
            "y": [np.sin(i * 0.3) for i in range(10)],
        })
        _call(tool, {"action": "fit_surrogate"})
        result = _call(tool, {
            "action": "select_next",
            "candidates": [[0.5], [1.0], [1.5], [2.0]],
            "n_select": 2,
        })
        assert result.success is True
        assert len(result.data["selected"]) == 2
        assert result.data["n_candidates"] == 4
        # each selected item has required fields
        for sel in result.data["selected"]:
            assert "candidate_idx" in sel
            assert "fidelity" in sel
            assert "acquisition" in sel
            assert "ei" in sel

    def test_select_without_fit(self):
        tool = MultiFidelityTool()
        result = _call(tool, {
            "action": "select_next",
            "candidates": [[0.5]],
        })
        assert result.success is False
        assert "未拟合" in result.error

    def test_select_missing_candidates(self):
        tool = MultiFidelityTool()
        _call(tool, {
            "action": "register_source",
            "name": "low",
            "level": 0,
            "X": [[0.0], [1.0]],
            "y": [0.0, 1.0],
        })
        _call(tool, {"action": "fit_surrogate"})
        result = _call(tool, {"action": "select_next"})
        assert result.success is False

    def test_select_cost_aware_prefers_cheap_fidelity(self):
        """With two sources, low-cost source should have higher acquisition for same EI."""
        tool = MultiFidelityTool()
        _call(tool, {
            "action": "register_source",
            "name": "cheap",
            "level": 0,
            "cost": 0.1,
            "X": [[i * 0.3] for i in range(10)],
            "y": [np.sin(i * 0.3) for i in range(10)],
        })
        _call(tool, {
            "action": "register_source",
            "name": "expensive",
            "level": 1,
            "cost": 100.0,
            "X": [[i * 0.3] for i in range(5)],
            "y": [np.sin(i * 0.3) for i in range(5)],
        })
        _call(tool, {"action": "fit_surrogate"})
        result = _call(tool, {
            "action": "select_next",
            "candidates": [[0.5]],
            "n_select": 2,
        })
        assert result.success is True
        # top pick should be the cheap source (higher acquisition for same EI)
        assert result.data["selected"][0]["fidelity"] == "cheap"

    def test_select_n_select_limits_results(self):
        tool = MultiFidelityTool()
        _call(tool, {
            "action": "register_source",
            "name": "low",
            "level": 0,
            "X": [[i * 0.3] for i in range(10)],
            "y": [np.sin(i * 0.3) for i in range(10)],
        })
        _call(tool, {"action": "fit_surrogate"})
        result = _call(tool, {
            "action": "select_next",
            "candidates": [[0.5], [1.0], [1.5]],
            "n_select": 1,
        })
        assert result.success is True
        assert len(result.data["selected"]) == 1


# ── tool profile ─────────────────────────────────────────────────────────────


class TestToolProfile:
    def test_profile_metadata(self):
        assert MultiFidelityTool.name == "multi_fidelity_tool"
        assert MultiFidelityTool.category == "sci"
        assert MultiFidelityTool.profile.cost_tier == "light"
        assert ResearchPhase.PLANNING in MultiFidelityTool.profile.phases
        assert ResearchPhase.VALIDATION in MultiFidelityTool.profile.phases
        assert ResearchPhase.OPEN in MultiFidelityTool.profile.phases
        assert MultiFidelityTool.read_only is True
