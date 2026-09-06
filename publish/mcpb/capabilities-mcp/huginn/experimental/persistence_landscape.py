"""Persistent homology landscape — hypothesis prediction 向量当 point cloud,
用 GUDHI Vietoris-Rips 算 persistence diagram, 看 hypothesis landscape topology.

Phase 1-6 的 fisher_distance 只看 pairwise 距离, 看不到 hypothesis space 的 topology
(cluster 结构 / hole / void). 这里走 Open Problem 7.4 探索路径: 把每个 hypothesis 的
predictions dict 拍成等长向量, Rips complex 出 persistence diagram, 看 persistent
feature 跟 posterior 高的 hypothesis 对不对得上 — 期望 true hypothesis 在 prediction
空间里孤立, 跨尺度存活 (高 death).

研究探索层, 允许失败: hypothesis <3 / prediction 维度 <2 / GUDHI 没装时 persistent
homology 没意义, 降级到 fisher_distance (Phase 1-6 工程近似继续用). 不改 Phase 1-6
任何模块, 只复用 Phase 7.1 的 compute_persistent_homology.

# 架构状态: 研究探索层 — 未接入主循环, 保留作为 future hook. 如需启用, 在 huginn/events/unified_bus.py 订阅 cognitive.* 事件并接入.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

# 依赖的 HypothesisManifold / simplicial_homology 留在 metacog 包, 这里用绝对导入.
from huginn.metacog.hypothesis_manifold import (  # noqa: E402
    Hypothesis,
    HypothesisManifold,
    Observation,
)
from huginn.metacog.simplicial_homology import (  # noqa: E402
    compute_persistent_homology,
    is_gudhi_available,
)

# ---------- point cloud 构造 ----------

def hypothese_to_pointcloud(
    manifold: HypothesisManifold,
) -> tuple[np.ndarray, list[str]]:
    """每个 hypothesis 的 predictions dict 拍成等长向量, 返回 N × D matrix.

    missing key 填 0 — 跟 hypothesis_manifold._gaussian_log_likelihood 的 uniform
    fallback 一致 (没预测 = 0, sigma 默认 1). 维度 D = 所有 hypothesis predictions
    key 的并集, 字母序确定.

    Returns:
        (points, h_ids) — points shape (N, D), h_ids[i] 对应 points[i].
        空 manifold 返回 (shape (0,0), []).
    """
    h_items = list(manifold._hyp.items())
    if not h_items:
        return np.zeros((0, 0), dtype=float), []

    all_keys = sorted({k for _, h in h_items for k in h.predictions})
    h_ids = [h_id for h_id, _ in h_items]

    if not all_keys:
        # predictions 全空: 退化成 1D 0向量, 形状合法但 Rips 没意义 (compute_landscape 会降级)
        return np.zeros((len(h_items), 1), dtype=float), h_ids

    mat = np.zeros((len(h_items), len(all_keys)), dtype=float)
    for i, (_, h) in enumerate(h_items):
        for j, k in enumerate(all_keys):
            mat[i, j] = float(h.predictions.get(k, 0.0))
    return mat, h_ids


# ---------- landscape ----------

@dataclass
class PersistenceLandscape:
    """hypothesis landscape 的 persistence diagram + 摘要.

    diagram: [(dim, birth, death), ...] — 来自 GUDHI RipsComplex (Phase 7.1)
    point_cloud: N × D matrix, 调试 + correlation 用
    h_ids: 每个 point 对应的 h_id (diagram 跟 h 关联用)
    per_h_death: 每个 hypothesis 作为 0-dim feature 的 death scale. 大 = 该 h 跟
        最近 neighbor 远, 在 prediction 空间孤立 (persistent). ponytail: 这是
        0-dim death 的解析定义 (Rips 0-dim death = nearest neighbor distance),
        不依赖 GUDHI 内部 vertex index 映射
    degraded: True 表示 GUDHI 没装 / 数据太少, diagram 是退化结果
    degrade_reason: 降级原因
    """
    diagram: list[tuple[int, float, float]]
    point_cloud: np.ndarray
    h_ids: list[str]
    per_h_death: list[float]
    degraded: bool
    degrade_reason: str | None = None

    @property
    def n_persistent_h0(self) -> int:
        """0-dim persistent feature 数 (跨尺度存活的 cluster)."""
        return sum(
            1 for d, b, de in self.diagram
            if d == 0 and (math.isinf(de) or de - b > 0.0)
        )


def compute_landscape(
    manifold: HypothesisManifold,
    max_dim: int = 2,
    max_edge_length: float | None = None,
) -> PersistenceLandscape:
    """算 hypothesis landscape 的 persistence diagram.

    复用 Phase 7.1 的 compute_persistent_homology (GUDHI RipsComplex).
    降级条件: GUDHI 没装 / hypothesis <3 / prediction 维度 <2 — 这种情况
    persistent homology 没意义, 返回 degraded=True.

    ponytail: witness complex (GUDHI WitnessComplex) 也是 spec 提到的路径
    (hypothesis 当 landmark, observation 当 witness), 但 observation 通常 <5 个,
    witness complex 退化. Rips 更稳, 升级路径: observation 多时换 witness.
    """
    pts, h_ids = hypothese_to_pointcloud(manifold)
    n = len(h_ids)
    dim = pts.shape[1] if pts.size else 0

    # 降级检查 — persistent homology 在这些情形下没意义
    if not is_gudhi_available():
        return PersistenceLandscape(
            diagram=[], point_cloud=pts, h_ids=h_ids,
            per_h_death=[float("inf")] * n,
            degraded=True, degrade_reason="gudhi_not_available",
        )
    if n < 3:
        return PersistenceLandscape(
            diagram=[], point_cloud=pts, h_ids=h_ids,
            per_h_death=[float("inf")] * n,
            degraded=True, degrade_reason=f"too_few_hypotheses_n={n}",
        )
    if dim < 2:
        return PersistenceLandscape(
            diagram=[], point_cloud=pts, h_ids=h_ids,
            per_h_death=[float("inf")] * n,
            degraded=True, degrade_reason=f"prediction_dim_too_low_d={dim}",
        )

    # Phase 7.1 的 GUDHI Rips
    diag = compute_persistent_homology(
        pts, max_dim=max_dim, max_edge_length=max_edge_length
    )

    # 0-dim death = nearest neighbor distance (Rips 0-dim 的解析定义)
    # ponytail: O(n²) pdist, n>2000 时换 sample. 跟 Phase 7.1 同样上限.
    from scipy.spatial.distance import pdist, squareform
    d_matrix = squareform(pdist(pts))
    np.fill_diagonal(d_matrix, np.inf)
    per_h_death = [float(d_matrix[i].min()) for i in range(n)]

    return PersistenceLandscape(
        diagram=diag, point_cloud=pts, h_ids=h_ids,
        per_h_death=per_h_death,
        degraded=False, degrade_reason=None,
    )


# ---------- correlation: persistent feature vs posterior ----------

@dataclass
class CorrelationResult:
    """persistent feature (0-dim death) 跟 posterior 的一致性结果.

    Hypothesis (spec Open Problem 7.4): persistent feature (高 death = h 在
    prediction 空间孤立) 跟 posterior 高 (h 跟 obs 一致) 应该一致 — true
    hypothesis 跟其他错 hypothesis 距离远, 所以 death 大且 posterior 高.

    consistent = (top_persistent_h == top_posterior_h) and (spearman_rho > 0)
    ponytail: 不要求 rho 高, N 小 spearman 噪声大; top 一致 + 同向就够了
    """
    degraded: bool
    degrade_reason: str | None
    persistent_ranking: list[str]   # h_ids 按 death 降序
    posterior_ranking: list[str]    # h_ids 按 posterior 降序
    spearman_rho: float             # [-1, 1]
    top_persistent_h: str | None
    top_posterior_h: str | None
    consistent: bool
    note: str = ""


def _spearman_rho(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation. ponytail: stdlib only, 不依赖 scipy.stats.

    转成 rank (ties 用 average rank) 后算 Pearson. N<2 返回 0.
    """
    n = len(x)
    if n < 2:
        return 0.0

    def rank(vals: list[float]) -> list[float]:
        idx = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[idx[j + 1]] == vals[idx[i]]:
                j += 1
            r = (i + j) / 2.0 + 1.0  # average rank, 1-indexed
            for k in range(i, j + 1):
                ranks[idx[k]] = r
            i = j + 1
        return ranks

    rx = rank(x)
    ry = rank(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    if dx == 0.0 or dy == 0.0:
        return 0.0
    return num / (dx * dy)


def correlate_persistence_with_posterior(
    landscape: PersistenceLandscape,
    manifold: HypothesisManifold,
    obs: Iterable[Observation],
) -> CorrelationResult:
    """persistent feature 跟 posterior 一致性.

    成功判据 (spec Open Problem 7.4): persistence diagram 识别 persistent feature,
    跟 posterior 高的 h 一致. 用 0-dim death (h 跟最近 neighbor 的距离) 作
    persistence score: death 大 → h 在 prediction 空间孤立 → persistent.
    期望: true h 跟其他错 h 距离远 → death 大 → posterior 高.

    Returns:
        CorrelationResult. degraded=True 时 consistent=False, note 说明降级原因.
    """
    obs = list(obs)

    if landscape.degraded:
        return CorrelationResult(
            degraded=True,
            degrade_reason=landscape.degrade_reason,
            persistent_ranking=[],
            posterior_ranking=[],
            spearman_rho=0.0,
            top_persistent_h=None,
            top_posterior_h=None,
            consistent=False,
            note=f"landscape degraded ({landscape.degrade_reason}), 降级到 fisher_distance",
        )

    h_ids = landscape.h_ids
    post = manifold.posterior(obs)
    post_vec = [post.get(h, 0.0) for h in h_ids]
    death_vec = landscape.per_h_death

    # 降序排, zip 后按数值大的在前
    persistent_ranking = [h for _, h in sorted(zip(death_vec, h_ids), reverse=True)]
    posterior_ranking = [h for _, h in sorted(zip(post_vec, h_ids), reverse=True)]

    rho = _spearman_rho(death_vec, post_vec)

    top_persistent = persistent_ranking[0] if persistent_ranking else None
    top_posterior = posterior_ranking[0] if posterior_ranking else None

    # 成功判据: top 一致 + spearman 同向
    consistent = (top_persistent == top_posterior) and (rho > 0.0)

    note = (
        f"top_persistent={top_persistent} (death={max(death_vec):.3f}), "
        f"top_posterior={top_posterior} (post={max(post_vec):.3f}), "
        f"spearman_rho={rho:.3f}"
    )

    return CorrelationResult(
        degraded=False,
        degrade_reason=None,
        persistent_ranking=persistent_ranking,
        posterior_ranking=posterior_ranking,
        spearman_rho=rho,
        top_persistent_h=top_persistent,
        top_posterior_h=top_posterior,
        consistent=consistent,
        note=note,
    )


# ---------- Self-check ----------

def _selfcheck() -> None:
    """验证: point cloud 构造 / persistence diagram 非空 / persistent 跟 posterior 一致.

    成功判据 (spec Open Problem 7.4): persistence diagram 识别 persistent feature
    跟 posterior 高的 h 一致.
    """
    print("=== persistence_landscape self-check ===")
    print()

    # Test 1: point cloud 构造 — missing key 填 0, key 字母序
    m = HypothesisManifold()
    m.add(Hypothesis("h1", "a", predictions={"x": 1.0, "y": 2.0}))
    m.add(Hypothesis("h2", "b", predictions={"x": 1.5, "z": 3.0}))
    m.add(Hypothesis("h3", "c", predictions={"x": 5.0, "y": 8.0, "z": 9.0}))
    pts, h_ids = hypothese_to_pointcloud(m)
    assert pts.shape == (3, 3), f"expected (3,3), got {pts.shape}"
    assert h_ids == ["h1", "h2", "h3"]
    # keys sorted: x, y, z; missing → 0
    assert pts[0, 0] == 1.0 and pts[0, 1] == 2.0 and pts[0, 2] == 0.0
    assert pts[1, 0] == 1.5 and pts[1, 1] == 0.0 and pts[1, 2] == 3.0
    assert pts[2, 0] == 5.0 and pts[2, 1] == 8.0 and pts[2, 2] == 9.0
    print(f"✓ Test 1 point cloud 构造: shape={pts.shape}, h_ids={h_ids}")
    print("  keys sorted (x,y,z); missing 填 0 ✓")

    # Test 2: edge case — 1 个 hypothesis 降级
    m1 = HypothesisManifold()
    m1.add(Hypothesis("only", "solo", predictions={"x": 1.0}))
    land1 = compute_landscape(m1)
    assert land1.degraded and "too_few" in (land1.degrade_reason or "")
    print(f"✓ Test 2 edge case 1 hypothesis: degraded ({land1.degrade_reason})")

    # Test 3: edge case — predictions 全空降级
    m_empty = HypothesisManifold()
    for i in range(3):
        m_empty.add(Hypothesis(f"h{i}", f"empty {i}"))
    land_e = compute_landscape(m_empty)
    assert land_e.degraded and "dim_too_low" in (land_e.degrade_reason or "")
    print(f"✓ Test 3 edge case empty predictions: degraded ({land_e.degrade_reason})")

    # Test 4: 成功判据 — true h 跟 obs 一致, 其他 decoy 分散
    # true h 预测 (1.0, 2.0), 4 个 decoy 预测分散在 (5,20)~(8,50) 区域
    # true 跟 decoy 距离大, decoy 互相之间也远 — true 应该最 persistent
    m5 = HypothesisManifold()
    m5.add(Hypothesis("true", "true hypothesis", predictions={"out": 1.0, "out2": 2.0}, n_params=1))
    m5.add(Hypothesis("d1", "decoy 1", predictions={"out": 5.0, "out2": 20.0}, n_params=1))
    m5.add(Hypothesis("d2", "decoy 2", predictions={"out": 6.0, "out2": 30.0}, n_params=1))
    m5.add(Hypothesis("d3", "decoy 3", predictions={"out": 7.0, "out2": 40.0}, n_params=1))
    m5.add(Hypothesis("d4", "decoy 4", predictions={"out": 8.0, "out2": 50.0}, n_params=1))

    obs = [
        Observation("out", 1.0, sigma=0.1),
        Observation("out2", 2.0, sigma=0.1),
    ]

    land = compute_landscape(m5, max_dim=1)
    print("\nTest 4 成功判据 (5 hypothesis: 1 true + 4 decoy):")
    print(f"  landscape degraded={land.degraded}, n_hypotheses={len(land.h_ids)}")
    print("  per_h_death (h_id → nearest neighbor dist):")
    for h_id, d in zip(land.h_ids, land.per_h_death):
        print(f"    {h_id}: {d:.3f}")

    if not land.degraded:
        print(f"  diagram: {len(land.diagram)} features (dim, birth, death)")
        for dim, b, de in land.diagram[:10]:
            ds = "inf" if math.isinf(de) else f"{de:.3f}"
            print(f"    dim={dim}, birth={b:.3f}, death={ds}")
        # diagram 非空
        assert len(land.diagram) > 0, "persistence diagram 应该非空"
        n_h0 = sum(1 for d, _, _ in land.diagram if d == 0)
        assert n_h0 >= 1, f"应该有 0-dim feature, got {n_h0}"
        print(f"  → persistence diagram 非空: {n_h0} 个 0-dim feature ✓")

    post = m5.posterior(obs)
    print(f"  posterior: {post}")

    result = correlate_persistence_with_posterior(land, m5, obs)
    print("\n  CorrelationResult:")
    print(f"    degraded: {result.degraded}")
    print(f"    persistent_ranking: {result.persistent_ranking}")
    print(f"    posterior_ranking: {result.posterior_ranking}")
    print(f"    spearman_rho: {result.spearman_rho:.3f}")
    print(f"    top_persistent_h: {result.top_persistent_h}")
    print(f"    top_posterior_h: {result.top_posterior_h}")
    print(f"    consistent: {result.consistent}")
    print(f"    note: {result.note}")

    # posterior 最高的必须是 true (无论 persistent 判据如何)
    assert result.top_posterior_h == "true", (
        f"posterior 最高应该是 true h, got {result.top_posterior_h}"
    )

    if result.consistent:
        print("\n✓ Test 4 成功判据达成:")
        print(f"  top_persistent={result.top_persistent_h} == top_posterior={result.top_posterior_h}")
        print(f"  spearman_rho={result.spearman_rho:.3f} > 0")
        print("  → persistent feature 跟 posterior 高的 h 一致")
        print("  → Open Problem 7.4 探索成功")
    else:
        print(f"\n? Test 4 部分一致 (top_persistent={result.top_persistent_h}, "
              f"top_posterior={result.top_posterior_h}, rho={result.spearman_rho:.3f})")
        print("  → 记录结果, fisher_distance 近似继续可用 (spec 允许探索失败)")

    # Test 5: 紧密 cluster (判据可能弱化, 但不应 crash)
    m_close = HypothesisManifold()
    m_close.add(Hypothesis("a", "a", predictions={"x": 1.0, "y": 1.0}, n_params=1))
    m_close.add(Hypothesis("b", "b", predictions={"x": 1.1, "y": 1.1}, n_params=1))
    m_close.add(Hypothesis("c", "c", predictions={"x": 1.2, "y": 1.2}, n_params=1))
    land_close = compute_landscape(m_close)
    obs_close = [Observation("x", 1.0, sigma=0.05), Observation("y", 1.0, sigma=0.05)]
    result_close = correlate_persistence_with_posterior(land_close, m_close, obs_close)
    assert not result_close.degraded, "紧密 cluster 不应降级"
    assert result_close.top_posterior_h == "a"
    print(f"\n✓ Test 5 紧密 cluster: top_posterior={result_close.top_posterior_h}, "
          f"consistent={result_close.consistent}")

    # Test 6: 降级时 correlate 也降级
    result_deg = correlate_persistence_with_posterior(land1, m1, [Observation("x", 1.0)])
    assert result_deg.degraded and not result_deg.consistent
    print("✓ Test 6 降级传播: landscape degraded → correlate degraded, consistent=False")

    print()
    print("=== persistence_landscape self-check 完成 ===")


if __name__ == "__main__":
    _selfcheck()
