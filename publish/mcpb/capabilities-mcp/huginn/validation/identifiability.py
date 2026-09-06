"""System-discovery identifiability ceiling — attractor geometry pre-flight check.

源自 Gallo, Anselmi & Lazzari, "Attractor Geometry Determines the
Identifiability Limits of System Discovery" (arXiv:2607.18490, 2026). 核心主张:

  符号发现/系统辨识 (SINDy / 进化符号回归) 的天花板不由算法或数据量决定,
  而由吸引子几何决定. 一个数字, λ_min(M), 即"不变测度矩矩阵"的最小特征值,
  在运行任何回归之前, 就设定好了能恢复的辨识上限:

    - λ_min(M) → 0: 吸引子没有覆盖函数空间, 恢复对任何算法 (稀疏/组合) 都不可行,
      R² 再高也可能只是对当前轨迹的过拟合, 解不唯一 (non-identifiable);
    - λ_min(M) 增长: 稀疏与进化两种算法都随之改善.

  源自 Birkhoff 遍历定理, 从一条短参考轨迹任何回归前就能算. 混沌通过铺开吸引子
  抬高 λ_min(M), 但也会放大噪声; 噪声进 SINDy 的回归瓶颈是线性的、进进化回归的
  判别通道是超线性的, 所以"更深的混沌"不总是更好.

设计 (落地到 Huginn):
  - moment_matrix(Theta): 不变测度矩矩阵  M = Thetaᵀ Theta / N (N=样本数),
    即候选函数库在轨迹上的样本 Gram 矩阵, 对角元是各库函数遍历均值, 整体衡量
    吸引子对函数空间的覆盖程度.
  - identifiability_ceiling(Theta): 取 M 最小特征值 λ_min, 归一化到相对量
    (λ_min / λ_max) 再分级: deficient / limited / adequate. 预飞阶段据此告诉 agent
    该不该信任将要跑出来的方程, 而不是只看拟合 R².
  - assess_trajectory(t, X, ...): 高层便捷入口, 用与 dynamics_discovery_tool
    _build_library 相同的候选库构造 Theta, 返回辨识度天花板 + 覆盖度 stats.

ceiling / 诚实边界:
  - 只反映"吸引子几何允许恢复什么", 不保证"算法真能恢复": λ_min 高是必要条件
    不是充分条件.
  - 归一化阈值是启发式 (相对最大特征值比例), 不是定理; 阈值本身可随域调节.
  - 不处理噪声的算法依赖差异 (SINDy 线性 vs 进化超线性), 部分留到 research-note:
    噪声鉴别的算法依赖差异在落地时尚未建模.
  - 纯 numpy 实现, 无 scipy/sklearn 依赖, 确定性可测.
  升级路径: 接入真实 SINDy/PySR benchmark 标定阈值; 给每个库函数算 Fisher 段贡献.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# 归一化阈值: λ_min_rel = λ_min / λ_max. 低于 deficient 视为"几何上不可辨识".
# 启发式标定 (论文用 Lorenz-84 驱动固定点/极限环/混沌, 固定点与非混沌极限环
# 的 λ_min 相对值远低于混沌; 这里的边界取数量级分离经验值).
_DEFICIENT_EIG = 1e-6
_LIMITED_EIG = 1e-3


@dataclass(frozen=True)
class IdentifiabilityCeiling:
    """预飞辨识度天花板 — 一个数字 + 一个分级, 跑回归之前就能算."""

    lambda_min: float       # M 最小特征值 (绝对量)
    lambda_max: float       # M 最大特征值
    lambda_min_rel: float   # λ_min / λ_max (归一化相对量, 跨尺度可比)
    level: str              # "adequate" | "limited" | "deficient"
    coverage_ratio: float   # 特征值 > 容差的比例 (吸引子对函数空间的覆盖度)
    n_terms: int            # 候选库规模 p
    n_samples: int          # 参考轨迹样本数 N
    note: str               # 人类可读的结论/行动建议


def _relative_level(lam_rel: float) -> str:
    if lam_rel >= _LIMITED_EIG:
        return "adequate"
    if lam_rel >= _DEFICIENT_EIG:
        return "limited"
    return "deficient"


def moment_matrix(Theta: np.ndarray) -> np.ndarray:
    """不变测度矩矩阵  M = Thetaᵀ Theta / N.

    Theta 形状 (N, p) = 候选库在轨迹 N 个采样点的取值 (含常数列). M 的 (i,j)
    元是 ∫ φ_i φ_j dμ 的遍历平均近似: M[i,j] = (1/N) Σ_n φ_i(x_n) φ_j(x_n).
    M 的特征值刻画"吸引子沿每个库函数方向覆盖了多少"。
    """
    Theta = np.asarray(Theta, dtype=np.float64)
    if Theta.ndim != 2:
        raise ValueError(f"Theta must be 2D (N, p), got {Theta.ndim}D")
    n = Theta.shape[0]
    if n == 0:
        raise ValueError("Theta has zero rows")
    return Theta.T @ Theta / float(n)


def identifiability_ceiling(
    Theta: np.ndarray, *, level_thresholds: tuple[float, float] | None = None
) -> IdentifiabilityCeiling:
    """从候选库 Theta 计算辨识度天花板.

    取不变测度矩矩阵 M = Thetaᵀ Theta / N 的最小/最大特征值, 归一化成相对量
    λ_min_rel = λ_min/λ_max 分级:

      - adequate  (>= limited_threshold): 吸引子覆盖函数空间足够, 算法才有机会独特恢复;
      - limited   (>= deficient_threshold): 部分方向欠覆盖, 恢复受限;
      - deficient (<  deficient_threshold): λ_min≈0, 几何上不可辨识, 任何算法都无解,
        此轨迹的 R² 高不足以背书方程唯一性.

    Args:
        Theta: (N, p) 候选库取值矩阵.
        level_thresholds: 自定义 (deficient, limited) 相对阈值; 默认 (_DEFICIENT_EIG,

    Returns:
        IdentifiabilityCeiling.
    """
    M = moment_matrix(Theta)
    if M.size == 0:
        return IdentifiabilityCeiling(
            lambda_min=0.0, lambda_max=0.0, lambda_min_rel=0.0,
            level="deficient", coverage_ratio=0.0, n_terms=0,
            n_samples=Theta.shape[0], note="empty candidate library",
        )
    ev = np.linalg.eigvalsh(M)
    lam_max = float(ev[-1])
    lam_min = float(ev[0])
    lam_rel = (lam_min / lam_max) if lam_max > 0 else 0.0

    def_t, lim_t = level_thresholds or (_DEFICIENT_EIG, _LIMITED_EIG)
    level = _relative_level(lam_rel) if level_thresholds is None else _level_with(lam_rel, def_t, lim_t)

    # 覆盖度: 特征值显著大于 0 (= 被吸引子采样到的库函数方向) 的比例.
    tol = lim_t * lam_max if lam_max > 0 else 0.0
    n_covered = int(np.sum(ev > tol))
    coverage = n_covered / float(ev.size) if ev.size else 0.0

    note = _build_note(level, lam_rel, coverage, ev.size, Theta.shape[0])
    return IdentifiabilityCeiling(
        lambda_min=float(lam_min),
        lambda_max=lam_max,
        lambda_min_rel=float(lam_rel),
        level=level,
        coverage_ratio=float(coverage),
        n_terms=int(ev.size),
        n_samples=int(Theta.shape[0]),
        note=note,
    )


def _level_with(lam_rel: float, def_t: float, lim_t: float) -> str:
    if lam_rel >= lim_t:
        return "adequate"
    if lam_rel >= def_t:
        return "limited"
    return "deficient"


def _build_note(level: str, lam_rel: float, coverage: float, p: int, n: int) -> str:
    if level == "adequate":
        return (
            "Attractor covers function space well (lambda_min/max ="
            f" {lam_rel:.2e}); recovery is geometrically permitted. Run the"
            " regression — a sparse/unique fit is achievable in principle."
        )
    if level == "limited":
        return (
            "Partial coverage (lambda_min/max = {lam_rel:.2e}, coverage"
            f" {coverage:.0%} of {p} terms). Some directions are under-sampled;"
            " a high R2 may still be non-unique. Consider a nonlinear regime or"
            " more diverse initial conditions before trusting uniqueness."
        ).replace("{lam_rel:.2e}", f"{lam_rel:.2e}")
    return (
        "Attractor barely covers function space (lambda_min/max ="
        f" {lam_rel:.2e}). Any recovery from this trajectory is geometrically"
        " ill-posed: an apparent fit is non-unique and may not reflect the true"
        " governing equation. R2 alone does NOT certify the discovered model."
    )


def assess_trajectory(
    t: np.ndarray,
    X: np.ndarray,
    *,
    max_order: int = 2,
    include_trig: bool = False,
    include_exp: bool = False,
    level_thresholds: tuple[float, float] | None = None,
) -> IdentifiabilityCeiling:
    """预飞便捷入口: 给一段参考轨迹 → 候选库 → 辨识度天花板.

    与 dynamics_discovery_tool._build_library 构造相同的多项式 + sin/cos + exp
    候选库 (跨模块行为一致), 再算 λ_min(M) 天花板. 不跑任何回归.

    Args:
        t: (N,) 时间序列.
        X: (N, m) 状态轨迹.
        max_order/include_trig/include_exp: 候选库控制, 与发现工具同语义.

    Returns:
        IdentifiabilityCeiling.
    """
    t = np.asarray(t, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (N, m), got {X.ndim}D")
    Theta = _build_library(X, max_order, include_trig, include_exp)
    return identifiability_ceiling(Theta, level_thresholds=level_thresholds)


def trajectory_coverage(t: np.ndarray, X: np.ndarray) -> dict[str, Any]:
    """Birkhoff 视角的简单覆盖统计: 轨迹在状态空间铺开程度 (无需候选库).

    辅助诊断: 用数值导数幅度、状态幅度、样本数给出一个粗糙的覆盖画像, 供
    research-note 与工具侧的人类可读说明. 不作为正式分级依据.
    """
    t = np.asarray(t, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] < 2:
        return {"n_samples": int(X.shape[0]), "range": 0.0, "volatility": 0.0}
    dt = float(np.mean(np.diff(t)))
    dX = np.diff(X, axis=0) / dt if dt > 0 else np.gradient(X, axis=0)
    return {
        "n_samples": int(X.shape[0]),
        "range": float(np.ptp(X)),
        "volatility": float(np.mean(np.abs(dX))),
    }


# ── 候选库: 与 dynamics_discovery_tool 保持一致 ─────────────────────────
def _build_library(
    X: np.ndarray, max_order: int, trig: bool, exp: bool
) -> np.ndarray:
    """构造候选库 Theta (N, p): 常数 + 多项式(含交叉) + 可选 sin/cos/exp."""
    from itertools import combinations_with_replacement

    n, m = X.shape
    cols: list[np.ndarray] = [np.ones(n)]
    for deg in range(1, max_order + 1):
        for combo in combinations_with_replacement(range(m), deg):
            cols.append(np.prod(X[:, list(combo)], axis=1))
    if trig:
        for i in range(m):
            cols.append(np.sin(X[:, i]))
            cols.append(np.cos(X[:, i]))
    if exp:
        for i in range(m):
            cols.append(np.exp(X[:, i]))
    return np.column_stack(cols)
