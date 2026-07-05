"""SINDy 动力学发现工具测试.

阻尼振荡器真值: dx0/dt = x1, dx1/dt = -x0 - 0.2*x1
验证 discover 能稀疏恢复这两条方程, validate 能积分回轨迹, 含噪仍稳健,
数据点太少时优雅失败而非崩溃.
"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from huginn.tools.sci.dynamics_discovery_tool import (
    DynamicsDiscoveryInput,
    DynamicsDiscoveryTool,
)
from huginn.types import ToolContext


def _damped_oscillator(n: int = 600, noise: float = 0.0, seed: int = 0):
    """生成阻尼振荡器轨迹 + 可选高斯噪声. 返回 (t, X)."""
    rng = np.random.default_rng(seed)

    def rhs(_t, x):
        return [x[1], -1.0 * x[0] - 0.2 * x[1]]

    t = np.linspace(0, 30, n)
    sol = solve_ivp(rhs, (t[0], t[-1]), [1.0, 0.0], t_eval=t, rtol=1e-8, atol=1e-10)
    X = sol.y.T
    if noise > 0:
        X = X + rng.normal(0, noise, X.shape)
    return t, X


def _ctx() -> ToolContext:
    return ToolContext(session_id="test-dyn", workspace=".")


def _discover(t, X, **kw):
    tool = DynamicsDiscoveryTool()
    args = DynamicsDiscoveryInput(
        data_json={"t": t.tolist(), "x0": X[:, 0].tolist(), "x1": X[:, 1].tolist()},
        max_order=2,
        threshold=0.05,
        smooth=True,
        **kw,
    )
    return asyncio.run(tool.call(args, _ctx()))


def test_discover_damped_oscillator():
    """干净数据 -> 正确发现 dx0/dt=x1, dx1/dt=-x0-0.2x1."""
    t, X = _damped_oscillator(n=600, noise=0.0)
    res = _discover(t, X)
    assert res.success, res.error
    data = res.data
    # 系数对齐到 terms, 检查关键项
    coefs_x0 = dict(zip(data["terms"], data["coefficients"]["x0"]))
    coefs_x1 = dict(zip(data["terms"], data["coefficients"]["x1"]))
    # dx0/dt = 1.0 * x1
    assert abs(coefs_x0["x1"] - 1.0) < 0.05, f"x1 coef={coefs_x0['x1']}"
    # dx1/dt = -1.0 * x0 - 0.2 * x1
    assert abs(coefs_x1["x0"] + 1.0) < 0.05, f"x0 coef={coefs_x1['x0']}"
    assert abs(coefs_x1["x1"] + 0.2) < 0.05, f"x1 coef={coefs_x1['x1']}"
    # 拟合质量
    assert data["r2_score"]["x0"] > 0.9
    assert data["r2_score"]["x1"] > 0.9
    assert data["n_samples"] == 600


def test_validate_equation():
    """用 discover 出的方程积分, 跟真实轨迹对比, R2 应该高."""
    t, X = _damped_oscillator(n=600, noise=0.01, seed=1)
    disc = _discover(t, X)
    assert disc.success
    d = disc.data

    tool = DynamicsDiscoveryTool()
    args = DynamicsDiscoveryInput(
        action="validate",
        data_json={"t": t.tolist(), "x0": X[:, 0].tolist(), "x1": X[:, 1].tolist()},
        terms=d["terms"],
        coefficients=d["coefficients"],
    )
    res = asyncio.run(tool.call(args, _ctx()))
    assert res.success, res.error
    # 积分轨迹应贴近原数据
    assert res.data["r2_score"]["x0"] > 0.9
    assert res.data["r2_score"]["x1"] > 0.9
    assert res.data["n_samples"] == 600


def test_noisy_data():
    """1% 噪声下仍能恢复正确的方程结构 (关键系数符号 + 量级)."""
    t, X = _damped_oscillator(n=800, noise=0.01, seed=2)
    res = _discover(t, X)
    assert res.success, res.error
    d = res.data
    coefs_x0 = dict(zip(d["terms"], d["coefficients"]["x0"]))
    coefs_x1 = dict(zip(d["terms"], d["coefficients"]["x1"]))
    # 噪声下容差放宽, 但结构必须对
    assert abs(coefs_x0["x1"] - 1.0) < 0.1
    assert abs(coefs_x1["x0"] + 1.0) < 0.1
    assert abs(coefs_x1["x1"] + 0.2) < 0.08
    assert d["r2_score"]["x0"] > 0.85
    assert d["r2_score"]["x1"] > 0.85


def test_insufficient_data():
    """数据点太少 (<5) 时应优雅失败, 不能抛未捕获异常."""
    t, X = _damped_oscillator(n=4, noise=0.0)
    assert X.shape[0] < 5
    res = _discover(t, X)
    # 必须返回失败结果, 而不是抛异常
    assert not res.success
    assert res.error is not None
    assert "sample" in res.error.lower() or "derivative" in res.error.lower()


def test_unknown_action():
    """未知 action 返回失败, 不崩."""
    tool = DynamicsDiscoveryTool()
    args = DynamicsDiscoveryInput(action="bogus")
    res = asyncio.run(tool.call(args, _ctx()))
    assert not res.success
    assert "unknown action" in res.error.lower()
