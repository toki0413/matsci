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
from huginn.tools.registry import ToolRegistry
from huginn.tools.sci.multi_fidelity_tool import (
    MultiFidelityTool,
)


@pytest.fixture(autouse=True)
def _isolate_registry():
    before = ToolRegistry.snapshot()
    yield
    ToolRegistry.restore(before)


@pytest.fixture
def tool() -> MultiFidelityTool:
    return MultiFidelityTool()


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

    def test_propagate_missing_x_new(self):
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


# ── bayesian_calibrate ─────────────────────────────────────────


class TestBayesianCalibrate:
    @pytest.mark.asyncio
    async def test_basic_calibration_returns_posterior(self, tool):
        # 1D 输入, 1D theta
        # y_lf = x, y_hf = 2*x + 1 (rho=2, delta=1, theta 不影响)
        X_lf = [[0.1], [0.3], [0.5], [0.7], [0.9]]
        y_lf = [0.1, 0.3, 0.5, 0.7, 0.9]
        X_hf = [[0.2], [0.4], [0.6]]
        y_hf = [1.4, 1.8, 2.2]  # 2*x + 1
        result = await tool.call({
            "action": "bayesian_calibrate",
            "X_lf": X_lf,
            "y_lf": y_lf,
            "X_hf": X_hf,
            "y_hf": y_hf,
            "theta_prior_low": [0.0],
            "theta_prior_high": [1.0],
            "n_mcmc_samples": 500,
            "n_burnin": 100,
            "seed": 42,
        })
        assert result.success, f"calibration failed: {result.error}"
        data = result.data
        assert "posterior_mean" in data
        assert "posterior_std" in data
        assert "posterior_samples" in data
        assert data["n_post_burnin"] == 400
        assert 0.0 <= data["acceptance_rate"] <= 1.0
        assert data["method"] == "kennedy_ohagan_2000_metropolis_hastings"

    @pytest.mark.asyncio
    async def test_posterior_within_prior_bounds(self, tool):
        # theta 后验样本必须在 [theta_low, theta_high] 内
        result = await tool.call({
            "action": "bayesian_calibrate",
            "X_lf": [[0.1], [0.5], [0.9]],
            "y_lf": [0.1, 0.5, 0.9],
            "X_hf": [[0.3], [0.7]],
            "y_hf": [0.6, 1.4],
            "theta_prior_low": [0.2],
            "theta_prior_high": [0.8],
            "n_mcmc_samples": 300,
            "n_burnin": 50,
            "seed": 7,
        })
        assert result.success
        samples = np.array(result.data["posterior_samples"])
        assert np.all(samples >= 0.2 - 1e-9)
        assert np.all(samples <= 0.8 + 1e-9)

    @pytest.mark.asyncio
    async def test_missing_data_fails(self, tool):
        result = await tool.call({
            "action": "bayesian_calibrate",
            "X_lf": [[0.1]],
            "y_lf": [0.1],
            # X_hf/y_hf 缺失
            "theta_prior_low": [0.0],
            "theta_prior_high": [1.0],
        })
        assert not result.success
        assert "X_hf/y_hf/X_lf/y_lf" in result.error

    @pytest.mark.asyncio
    async def test_missing_prior_fails(self, tool):
        result = await tool.call({
            "action": "bayesian_calibrate",
            "X_lf": [[0.1]],
            "y_lf": [0.1],
            "X_hf": [[0.2]],
            "y_hf": [0.4],
            # theta_prior 缺失
        })
        assert not result.success
        assert "theta_prior" in result.error

    @pytest.mark.asyncio
    async def test_acceptance_rate_reasonable(self, tool):
        # 接受率应在合理范围 — 用较大 sigma_n 让后验有展宽, MCMC 能探索
        result = await tool.call({
            "action": "bayesian_calibrate",
            "X_lf": [[0.1], [0.3], [0.5], [0.7], [0.9]],
            "y_lf": [0.1, 0.3, 0.5, 0.7, 0.9],
            "X_hf": [[0.2], [0.4], [0.6], [0.8]],
            "y_hf": [0.4, 0.8, 1.2, 1.6],
            "theta_prior_low": [0.0],
            "theta_prior_high": [1.0],
            "n_mcmc_samples": 1000,
            "n_burnin": 200,
            "proposal_std": 0.05,
            "sigma_n": 0.1,  # 较大噪声让后验有展宽, MCMC 能探索
            "seed": 123,
        })
        assert result.success
        # 宽松边界 — MCMC 接受率与问题结构相关, 只要 > 0 说明在探索
        assert 0.01 < result.data["acceptance_rate"] < 0.99

    @pytest.mark.asyncio
    async def test_rho_estimated(self, tool):
        # y_hf = 3 * y_lf → rho 应当接近 3
        result = await tool.call({
            "action": "bayesian_calibrate",
            "X_lf": [[0.1], [0.3], [0.5], [0.7], [0.9]],
            "y_lf": [0.1, 0.3, 0.5, 0.7, 0.9],
            "X_hf": [[0.2], [0.4], [0.6]],
            "y_hf": [0.6, 1.2, 1.8],  # 3 * x
            "theta_prior_low": [0.0],
            "theta_prior_high": [0.01],  # 很窄, theta 几乎不影响
            "n_mcmc_samples": 500,
            "n_burnin": 100,
            "seed": 42,
        })
        assert result.success
        # rho 是最小二乘估计, 应当在 3 附近
        assert abs(result.data["rho"] - 3.0) < 0.5


# ── nested_doe ─────────────────────────────────────────────────


class TestNestedDoe:
    @pytest.mark.asyncio
    async def test_basic_nested_design(self, tool):
        result = await tool.call({
            "action": "nested_doe",
            "n_hf": 5,
            "n_lf": 20,
            "dim": 2,
            "bounds_low": [0.0, 0.0],
            "bounds_high": [1.0, 1.0],
            "seed": 42,
        })
        assert result.success
        data = result.data
        assert data["n_hf"] == 5
        assert data["n_lf"] == 20
        assert data["dim"] == 2
        assert data["nested"] is True
        assert len(data["X_hf"]) == 5
        assert len(data["X_lf"]) == 20
        assert len(data["X_hf"][0]) == 2
        assert len(data["X_lf"][0]) == 2

    @pytest.mark.asyncio
    async def test_hf_subset_of_lf(self, tool):
        # HF 点必须是 LF 点的前 n_hf 个 (嵌套性)
        result = await tool.call({
            "action": "nested_doe",
            "n_hf": 3,
            "n_lf": 10,
            "dim": 2,
            "seed": 7,
        })
        assert result.success
        X_hf = np.array(result.data["X_hf"])
        X_lf = np.array(result.data["X_lf"])
        # 前 n_hf 个 LF 点 = HF 点
        np.testing.assert_array_almost_equal(X_lf[:3], X_hf)

    @pytest.mark.asyncio
    async def test_points_within_bounds(self, tool):
        result = await tool.call({
            "action": "nested_doe",
            "n_hf": 4,
            "n_lf": 12,
            "dim": 3,
            "bounds_low": [-1.0, 0.0, 2.0],
            "bounds_high": [1.0, 5.0, 3.0],
            "seed": 11,
        })
        assert result.success
        X_lf = np.array(result.data["X_lf"])
        assert np.all(X_lf[:, 0] >= -1.0 - 1e-9)
        assert np.all(X_lf[:, 0] <= 1.0 + 1e-9)
        assert np.all(X_lf[:, 1] >= 0.0 - 1e-9)
        assert np.all(X_lf[:, 1] <= 5.0 + 1e-9)
        assert np.all(X_lf[:, 2] >= 2.0 - 1e-9)
        assert np.all(X_lf[:, 2] <= 3.0 + 1e-9)

    @pytest.mark.asyncio
    async def test_n_hf_greater_than_n_lf_fails(self, tool):
        result = await tool.call({
            "action": "nested_doe",
            "n_hf": 10,
            "n_lf": 5,
            "dim": 2,
        })
        assert not result.success
        assert "n_hf" in result.error

    @pytest.mark.asyncio
    async def test_seed_reproducibility(self, tool):
        # 同种子应当生成相同设计
        args = {
            "action": "nested_doe",
            "n_hf": 4,
            "n_lf": 10,
            "dim": 2,
            "seed": 99,
        }
        r1 = await tool.call(args)
        r2 = await tool.call(args)
        assert r1.success and r2.success
        np.testing.assert_array_almost_equal(
            np.array(r1.data["X_lf"]), np.array(r2.data["X_lf"])
        )

    @pytest.mark.asyncio
    async def test_space_filling_min_distance(self, tool):
        # 空间填充: min-distance 不应太小 (LHS 性质)
        result = await tool.call({
            "action": "nested_doe",
            "n_hf": 5,
            "n_lf": 20,
            "dim": 2,
            "bounds_low": [0.0, 0.0],
            "bounds_high": [1.0, 1.0],
            "seed": 42,
        })
        assert result.success
        # 20 个点在 [0,1]^2, LHS 的 min-distance 通常 > 0.05
        assert result.data["lf_min_distance"] > 0.02


# ── variance_reduction ─────────────────────────────────────────


class TestVarianceReduction:
    @pytest.mark.asyncio
    async def test_basic_variance_reduction(self, tool):
        # y_hf = y_lf + noise, 高相关 → 方差应当缩减
        rng = np.random.default_rng(42)
        n = 100
        y_lf = rng.uniform(0, 10, n)
        y_hf = y_lf + rng.normal(0, 0.1, n)  # 高度相关
        result = await tool.call({
            "action": "variance_reduction",
            "y_hf_samples": y_hf.tolist(),
            "y_lf_samples": y_lf.tolist(),
        })
        assert result.success
        data = result.data
        assert data["reduction_ratio"] > 0.5  # 高相关 → 大幅缩减
        assert data["variance_reduced"] < data["variance_original"]

    @pytest.mark.asyncio
    async def test_optimal_beta_calculated(self, tool):
        rng = np.random.default_rng(7)
        n = 50
        y_lf = rng.uniform(-1, 1, n)
        y_hf = 2 * y_lf + rng.normal(0, 0.05, n)
        result = await tool.call({
            "action": "variance_reduction",
            "y_hf_samples": y_hf.tolist(),
            "y_lf_samples": y_lf.tolist(),
        })
        assert result.success
        # 最优 beta 应接近 2 (因为 y_hf ≈ 2*y_lf)
        assert abs(result.data["beta_optimal"] - 2.0) < 0.3

    @pytest.mark.asyncio
    async def test_manual_beta(self, tool):
        rng = np.random.default_rng(11)
        n = 30
        y_lf = rng.uniform(0, 5, n)
        y_hf = y_lf + rng.normal(0, 0.2, n)
        result = await tool.call({
            "action": "variance_reduction",
            "y_hf_samples": y_hf.tolist(),
            "y_lf_samples": y_lf.tolist(),
            "beta": 0.5,  # 手动指定, 非最优
        })
        assert result.success
        assert result.data["beta"] == 0.5
        # 手动非最优 beta 的缩减比应当 < 最优 beta 的缩减比
        optimal_result = await tool.call({
            "action": "variance_reduction",
            "y_hf_samples": y_hf.tolist(),
            "y_lf_samples": y_lf.tolist(),
        })
        assert optimal_result.data["reduction_ratio"] >= result.data["reduction_ratio"]

    @pytest.mark.asyncio
    async def test_uncorrelated_no_reduction(self, tool):
        # y_hf 和 y_lf 不相关 → 缩减比应当接近 0 或负
        rng = np.random.default_rng(99)
        n = 200
        y_hf = rng.normal(0, 1, n)
        y_lf = rng.normal(0, 1, n)  # 独立
        result = await tool.call({
            "action": "variance_reduction",
            "y_hf_samples": y_hf.tolist(),
            "y_lf_samples": y_lf.tolist(),
        })
        assert result.success
        # 不相关时, 最优 beta ≈ 0, 缩减比 ≈ 0 (或略负, 噪声)
        assert result.data["reduction_ratio"] < 0.1

    @pytest.mark.asyncio
    async def test_length_mismatch_fails(self, tool):
        result = await tool.call({
            "action": "variance_reduction",
            "y_hf_samples": [1.0, 2.0, 3.0],
            "y_lf_samples": [1.0, 2.0],  # 长度不同
        })
        assert not result.success
        assert "长度" in result.error

    @pytest.mark.asyncio
    async def test_too_few_samples_fails(self, tool):
        result = await tool.call({
            "action": "variance_reduction",
            "y_hf_samples": [1.0],
            "y_lf_samples": [1.0],
        })
        assert not result.success
        assert "2" in result.error

    @pytest.mark.asyncio
    async def test_estimate_close_to_hf_mean(self, tool):
        # 控制变量估计应当接近 y_hf 的真实均值
        rng = np.random.default_rng(55)
        n = 500
        y_lf = rng.uniform(0, 10, n)
        y_hf = y_lf + rng.normal(0, 0.5, n)
        result = await tool.call({
            "action": "variance_reduction",
            "y_hf_samples": y_hf.tolist(),
            "y_lf_samples": y_lf.tolist(),
        })
        assert result.success
        # 估计值应当接近 hf_only 估计 (都是无偏的, 只是方差不同)
        assert abs(result.data["estimate"] - result.data["estimate_hf_only"]) < 1.0
