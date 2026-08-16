"""Attractor-geometry identifiability ceiling tests.

源自 Gallo et al. (arXiv:2607.18490): 单个 λ_min(M) (不变测度矩矩阵最小特征值)
在跑系统发现算法之前就设定了辨识上限. 这里验证:

- moment_matrix / identifiability_ceiling 数值正确;
- 恒定轨迹 (不覆盖函数空间) → deficient (λ_min≈0, 几何不可辨识);
- 混沌轨迹 (Lorenz 投影) 铺开吸引子 → 明显高于恒定情形 (感知 λ_min 随覆盖增长);
- dynamics_discovery_tool 接线: discover 输出 identifiability 块 + 新增
  identifiability 预飞 action, 且 preflight 不跑任何回归.
"""
from __future__ import annotations

import asyncio

import numpy as np
from scipy.integrate import solve_ivp

from huginn.core_types import ToolContext
from huginn.tools.sci.dynamics_discovery_tool import (
    DynamicsDiscoveryInput,
    DynamicsDiscoveryTool,
)
from huginn.validation.identifiability import (
    assess_trajectory,
    identifiability_ceiling,
    moment_matrix,
    trajectory_coverage,
)

# ── 轨迹生成器 ─────────────────────────────────────────────────────────────

def _make_traj(x0: np.ndarray, x1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(x0), len(x1))
    t = np.linspace(0, float(n), n)
    return t, np.column_stack([x0[:n], x1[:n]])


def _constant_traj(n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0, 1, n)
    X = np.column_stack([3.0 * np.ones(n), 5.0 * np.ones(n)])
    return t, X


def _chaotic_traj(n: int = 3000) -> tuple[np.ndarray, np.ndarray]:
    """Lorenz 混沌吸引子 (x,y) 投影. 混沌铺开吸引子 → λ_min(M) 应明显抬高."""
    def rhs(_t, y):
        x, y_, z = y
        return [10 * (y_ - x), x * (28 - z) - y_, x * y_ - (8 / 3) * z]

    t = np.linspace(0, n * 0.01, n)
    sol = solve_ivp(
        rhs, (0, n * 0.01), [1.0, 1.0, 1.0], t_eval=t,
        rtol=1e-8, atol=1e-10, method="DOP853",
    )
    return t, sol.y[:2].T


# ── 核心模块数值 ───────────────────────────────────────────────────────────

def test_moment_matrix_matches_gram() -> None:
    Theta = np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
    M = moment_matrix(Theta)
    expected = np.array([[1.0, 3.0], [3.0, 29.0 / 3.0]])
    np.testing.assert_allclose(M, expected, rtol=1e-12)


def test_identifiability_ceiling_basic_attrs() -> None:
    Theta = np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
    ceiling = identifiability_ceiling(Theta)
    assert ceiling.lambda_min > 0
    assert ceiling.lambda_max > ceiling.lambda_min
    assert 0.0 < ceiling.lambda_min_rel < 1.0
    assert ceiling.n_terms == 2
    assert ceiling.n_samples == 3


def test_constant_trajectory_is_deficient() -> None:
    """恒定轨迹: 候选库全共线 (常数列), 矩矩阵降秩 → λ_min≈0 → 几何不可辨识."""
    t, X = _constant_traj()
    ceiling = assess_trajectory(t, X, max_order=2)
    assert ceiling.level == "deficient"
    assert ceiling.lambda_min_rel < 1e-6
    # 6 个多项式的候选库里只有 1 个覆盖方向 (常数方向).
    assert ceiling.coverage_ratio <= 1.0 / 6 + 1e-9


def test_chaos_raises_lambda_min_above_constant() -> None:
    """论文核心: 混沌铺开吸引子抬高 λ_min(M). Lorenz 应显著高于恒定轨迹."""
    _, X_const = _constant_traj()
    t_const = np.linspace(0, 1, X_const.shape[0])
    c_const = assess_trajectory(t_const, X_const, max_order=2)

    t_chaos, X_chaos = _chaotic_traj()
    c_chaos = assess_trajectory(t_chaos, X_chaos, max_order=2)

    # 混沌 (可辨识) 顶点明显高于恒定 (不可辨识), 且混沌情形不落于 deficient.
    assert c_chaos.lambda_min_rel > c_const.lambda_min_rel * 1e3
    assert c_chaos.level != "deficient"
    # 覆盖度也应更高: 混沌轨迹铺开函数空间.
    assert c_chaos.coverage_ratio > c_const.coverage_ratio


def test_assess_trajectory_handles_single_variable() -> None:
    t = np.linspace(0, 1, 100)
    X = np.cos(4 * t).reshape(-1, 1)
    # 单变量不应崩, 且返回合理规模.
    ceiling = assess_trajectory(t, X, max_order=2)
    assert ceiling.n_terms == 3  # 1 + x + x^2
    assert ceiling.level in {"adequate", "limited", "deficient"}


def test_trajectory_coverage_stats() -> None:
    t, X = _chaotic_traj(n=500)
    stats = trajectory_coverage(t, X)
    assert stats["n_samples"] == 500
    assert stats["range"] > 0
    assert stats["volatility"] >= 0


# ── 工具接线 ───────────────────────────────────────────────────────────────

def _ctx() -> ToolContext:
    return ToolContext(session_id="test-ident", workspace=".")


def test_discover_returns_identifiability_block() -> None:
    """discover 结果带 identifiability 块, agent 可据此判断要不要信任方程."""
    t, X = _chaotic_traj(n=1500)
    tool = DynamicsDiscoveryTool()
    args = DynamicsDiscoveryInput(
        data_json={"t": t.tolist(), "x0": X[:, 0].tolist(), "x1": X[:, 1].tolist()},
        max_order=2, threshold=0.05, smooth=True,
    )
    res = asyncio.run(tool.call(args, _ctx()))
    assert res.success, res.error
    ident = res.data["identifiability"]
    assert ident is not None
    for key in ("lambda_min", "lambda_min_rel", "level", "coverage_ratio", "note"):
        assert key in ident, f"missing {key}"


def test_preflight_identifiability_action_runs_no_regression() -> None:
    """identifiability action 是预飞: 不跑回归, 返回天花板 + 行动建议."""
    t, X = _constant_traj()
    tool = DynamicsDiscoveryTool()
    args = DynamicsDiscoveryInput(
        action="identifiability",
        data_json={"t": t.tolist(), "x0": X[:, 0].tolist(), "x1": X[:, 1].tolist()},
        max_order=2,
    )
    res = asyncio.run(tool.call(args, _ctx()))
    assert res.success, res.error
    assert res.data["level"] == "deficient"
    assert "recommended_action" in res.data
    # 预飞标记
    assert res.metadata.get("preflight") is True
    assert res.metadata.get("no_regression_run") is True
    assert "honest_boundary" in res.metadata


def test_preflight_requires_no_terms() -> None:
    """identifiability 主要比 discover 更轻: 不需要 terms/coefficients."""
    t, X = _constant_traj()
    tool = DynamicsDiscoveryTool()
    args = DynamicsDiscoveryInput(
        action="identifiability",
        data_json={"t": t.tolist(), "x0": X[:, 0].tolist(), "x1": X[:, 1].tolist()},
    )
    res = asyncio.run(tool.call(args, _ctx()))
    assert res.success, res.error
    assert "terms" not in res.data  # 预飞只算几何, 不产出候选项
