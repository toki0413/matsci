"""E3: 多保真融合深化 — Kennedy-O'Hagan Bayesian calibration + nested DOE + variance reduction.

覆盖:
- bayesian_calibrate: MCMC 采样, 后验统计, 接受率, 边界处理
- nested_doe: 嵌套 LHS, HF ⊂ LF, 空间填充度
- variance_reduction: 控制变量, 最优 beta, 方差缩减比
- 错误处理: 缺数据/n_hf>n_lf/长度不匹配
"""

from __future__ import annotations

import numpy as np
import pytest

from huginn.tools.sci.multi_fidelity_tool import (
    MultiFidelityInput,
    MultiFidelityTool,
)


@pytest.fixture
def tool():
    return MultiFidelityTool()


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
