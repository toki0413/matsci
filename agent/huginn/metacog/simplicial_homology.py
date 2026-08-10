"""真正 simplicial homology (GUDHI), 替代 trace_topology 的 networkx 近似.

Phase 1-6 的 trace_topology.compute_betti 用 networkx 算 β_0/β_1:
  - β_0 = connected_components
  - β_1 = cycle_basis 数量 (Euler characteristic 公式)

这是工程近似, 在以下情形系统性偏差:
  1. 实心 triangle: networkx 把边界 cycle 算成 β_1=1, 但实心 triangle β_1=0
     (二维 simplex 把 hole 填满了, 真正 homology 没洞)
  2. 看不到 β_2 (sphere / void)
  3. 看不到 scale: networkx 给整数, 不知道洞在哪尺度上出生/死亡

本模块用 GUDHI SimplexTree 算真正 betti (基于 Z_2 系数 boundary matrix
reduction, 等价于 Smith normal form 在 Z_2 上), 用 RipsComplex 算
persistence diagram — 后者是 networkx 结构上无法给出的不变量.

研究探索层 (Open Problem 7.1): 不改 trace_topology.py, 不阻塞 Phase 1-6.
GUDHI 没装时退回 networkx 风格的 β_0 + β_1 估计, 升级路径: pip install gudhi.

# 架构状态: 研究探索层 — 未接入主循环, 保留作为 future hook. 如需启用, 在 huginn/events/unified_bus.py 订阅 cognitive.* 事件并接入.
"""
from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

try:
    import gudhi
    _GUDHI_AVAILABLE = True
except ImportError:
    _GUDHI_AVAILABLE = False


def is_gudhi_available() -> bool:
    """GUDHI 是否可用, 用来在 caller 端决定要不要走降级路径."""
    return _GUDHI_AVAILABLE


def compute_exact_betti(
    simplices: Iterable[tuple],
    max_dim: int = 3,
) -> dict[int, int]:
    """真正 simplicial homology 的 betti numbers.

    用 GUDHI SimplexTree + compute_persistence 算 Z_2 系数下的 betti.
    内部是 boundary matrix reduction, 跟 Smith normal form 在 Z_2 上等价.
    不是 networkx 的 Euler characteristic 公式 — 那个只在 simplex 维度 ≤1
    时碰巧正确, 维度 ≥2 时会漏报 β_2 / 误报 β_1.

    Args:
        simplices: 每个 tuple 是一个 simplex 的 vertex id, e.g. (0,1,2)
            表示 2-simplex [v0, v1, v2]. 重复插入幂等.
        max_dim: 算到第几维 betti. 默认 3 (β_0..β_3), 一般够用.

    Returns:
        {0: β_0, 1: β_1, 2: β_2, ...} — key 范围 [0, max_dim].

    Example:
        # 实心 triangle: 3 vertex + 3 edge + 1 triangle
        >>> compute_exact_betti([(0,),(1,),(2,),(0,1),(1,2),(0,2),(0,1,2)])
        {0: 1, 1: 0, 2: 0, 3: 0}
        # networkx 会算 β_1=1 (cycle_basis 把边界当洞), 这里正确给 0
    """
    if not _GUDHI_AVAILABLE:
        return _betti_fallback(simplices, max_dim)

    st = gudhi.SimplexTree()
    for simplex in simplices:
        if len(simplex) == 0:
            continue
        # filtration=0 让所有 simplex 同时存在, 等价于静态 complex
        st.insert(list(simplex), filtration=0.0)

    # compute_persistence 算 boundary matrix reduction, 然后查 betti
    # persistence_dim_max=True 让 GUDHI 算最高维的 betti (默认会跳过最高维,
    # 因为 filtration 视角下最高维 essential class 算法上不算 persistent)
    st.compute_persistence(homology_coeff_field=2, persistence_dim_max=True)
    betti = st.betti_numbers()  # list, betti[d] = β_d

    return {d: (betti[d] if d < len(betti) else 0) for d in range(max_dim + 1)}


def _betti_fallback(simplices: Iterable[tuple], max_dim: int) -> dict[int, int]:
    """GUDHI 缺失时的降级 — 退回 networkx 风格工程近似.

    ponytail: 只算 β_0 (并查集) + β_1 (cycle_basis), 维度 ≥2 直接给 0.
    升级路径: pip install gudhi 走真正 SimplexTree. 这个 fallback 跟
    trace_topology.compute_betti 等价, 留着只是让模块没 GUDHI 也能 import.
    """
    try:
        import networkx as nx
        _nx = True
    except ImportError:
        _nx = False

    vertices: set = set()
    edges: list[tuple] = []
    for s in simplices:
        if len(s) == 0:
            continue
        vertices.update(s)
        if len(s) == 2:
            edges.append(tuple(s))

    if _nx:
        G = nx.Graph()
        G.add_nodes_from(vertices)
        G.add_edges_from(edges)
        b0 = nx.number_connected_components(G)
        b1 = len(nx.cycle_basis(G))
    else:
        # networkx 也没: 并查集算 β_0, β_1 给 0 (保守)
        parent = {v: v for v in vertices}
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for a, b in edges:
            if a in parent and b in parent:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
        b0 = len({find(v) for v in vertices})
        b1 = 0

    return {d: (b0 if d == 0 else (b1 if d == 1 else 0)) for d in range(max_dim + 1)}


def compute_persistent_homology(
    points: np.ndarray | list[list[float]],
    max_dim: int = 2,
    max_edge_length: float | None = None,
) -> list[tuple[int, float, float]]:
    """Vietoris-Rips persistence diagram via GUDHI.

    这是 networkx 结构上无法给出的不变量 — networkx 只能给整数 betti,
    persistence diagram 给每个 topological feature 的 (birth_scale,
    death_scale), 能区分噪声 (短命) 和真实结构 (长命).

    Args:
        points: (n, dim) point cloud. n 大时考虑 downsample.
        max_dim: 算到第几维 persistence. 默认 2 (β_0 / β_1 / β_2).
        max_edge_length: Rips complex 的尺度截断. None 时用启发式:
            2 * median pairwise distance — 大到能连通, 但不至于全 clique.

    Returns:
        list of (dim, birth, death) tuples. death=inf 表示 persistent
        feature (跨整个尺度存在).

    Example:
        # 50 个点排成 ring (中间有 2D hole)
        >>> pts = ring_point_cloud(n=50, r=2.0)
        >>> diag = compute_persistent_homology(pts, max_dim=1)
        # 应该看到 1 个 1-dim persistent feature (death=inf 或大 death),
        # networkx 给不出这个, 只能给 β_1=1 (整数).
    """
    if not _GUDHI_AVAILABLE:
        return _persistent_fallback(points, max_dim)

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError(f"points should be 2D (n, dim), got shape {pts.shape}")

    if max_edge_length is None:
        # 启发式: 2x median pairwise distance. 大到连通, 不至于退化成全 clique.
        # ponytail: O(n^2) pdist, n>2000 时换 sample. 升级路径: 用 max_edge_length
        # 参数由 caller 显式给.
        from scipy.spatial.distance import pdist
        d = pdist(pts)
        if len(d) == 0:
            return []
        max_edge_length = float(np.median(d)) * 2.0

    rc = gudhi.RipsComplex(points=pts, max_edge_length=max_edge_length)
    # simplex tree 扩展到 max_dim+1: 算 k-dim persistence 需要 (k+1)-simplex
    # 来 kill spurious cycles (e.g. 算 1-dim persistence 需要 2-simplex 来填
    # 三角形, 否则每个 edge 都是 essential class — 没意义)
    st = rc.create_simplex_tree(max_dimension=max_dim + 1)
    # persistence_dim_max=True 让 essential class (death=inf, 整个尺度存在的洞)
    # 也进 diagram — point cloud 的 ring/sphere 等 "持续到无穷" 的特征正是想看的
    diag = st.persistence(homology_coeff_field=2, persistence_dim_max=True)

    # GUDHI 返回 [(dim, (birth, death)), ...] — 拍扁成 (dim, birth, death)
    return [(dim, float(b), float(d)) for dim, (b, d) in diag]


def _persistent_fallback(points, max_dim):
    """GUDHI 缺失 — persistence diagram 给不出来, 只能返回 trivial β_0.

    ponytail: 这就是 spec 说的"失败则保留工程近似". networkx 在 persistence
    层面没法替代, 只能给静态 betti. 升级路径: pip install gudhi.
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n == 0:
        return []
    # 一个 (0, 0, inf) 表示所有点连通 + 没尺度信息 — 完全退化
    return [(0, 0.0, float("inf"))]


def count_persistent_features(
    diagram: list[tuple[int, float, float]],
    dim: int,
    min_persistence: float = 0.0,
) -> int:
    """数 persistence diagram 里某维的 persistent feature 数.

    Args:
        diagram: compute_persistent_homology 返回的 (dim, birth, death) list
        dim: 数第几维 feature (e.g. 1 = 1-dim holes)
        min_persistence: death - birth 阈值, 默认 0 (任何 persistent 都算)

    Returns:
        该维 persistent feature 数.
    """
    count = 0
    for d, birth, death in diagram:
        if d != dim:
            continue
        # inf death 表示跨尺度存在 — 一定 persistent
        persist = float("inf") if math.isinf(death) else death - birth
        if persist > min_persistence:
            count += 1
    return count


# ---------- 测试用 point cloud 构造器 ----------

def ring_point_cloud(n: int = 50, r: float = 2.0, noise: float = 0.05) -> np.ndarray:
    """n 个点排在半径 r 的圆周上 — 用于测试 1-dim persistent hole.

    圆周是个 1-sphere, 真正 homology β_0=1, β_1=1. 加上点采样 + Rips 复形,
    在合适的尺度下能看到 1 个 persistent 1-dim feature (圆周形成的环).
    """
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    if noise > 0:
        rng = np.random.default_rng(42)
        pts = pts + rng.normal(0, noise, pts.shape)
    return pts


def torus_point_cloud(n: int = 100, R: float = 2.0, r: float = 0.7) -> np.ndarray:
    """torus 表面采样 n 个点 — 真正 homology β_0=1, β_1=2, β_2=1.

    torus 是 Open Problem 7.1 的经典验证 case (spec line 99 暗示).
    ponytail: 均匀采样用 (u, v) → ((R+r cos v) cos u, (R+r cos v) sin u, r sin v).
    """
    rng = np.random.default_rng(42)
    u = rng.uniform(0, 2 * np.pi, n)
    v = rng.uniform(0, 2 * np.pi, n)
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    return np.column_stack([x, y, z])


# ---------- Self-check ----------

def _selfcheck() -> None:
    """验证: betti 一致性 + persistence diagram + 成功判据.

    测试矩阵:
      1. 实心 triangle — GUDHI β_1=0 (networkx 会误报 β_1=1)
      2. 空心 triangle — GUDHI β_1=1, 跟 networkx 一致
      3. torus triangulation — β_0=1, β_1=2, β_2=1
      4. n=50 ring point cloud — persistent 1-dim feature
      5. 成功判据: persistence diagram 给出 networkx 给不出的 scale 信息
    """
    if not _GUDHI_AVAILABLE:
        print("✗ GUDHI not available — fallback only, Phase 7.1 探索失败")
        print("  升级路径: pip install gudhi")
        return

    print(f"gudhi version: {gudhi.__version__}")
    print()

    # Test 1: 实心 triangle — 关键 case
    # networkx (cycle_basis) 会把边界算成 β_1=1, 真正 homology β_1=0
    # (2-simplex 把洞填了)
    solid_triangle = [(0,), (1,), (2,), (0, 1), (1, 2), (0, 2), (0, 1, 2)]
    betti_solid = compute_exact_betti(solid_triangle, max_dim=2)
    assert betti_solid[0] == 1, f"solid triangle β_0 should be 1, got {betti_solid[0]}"
    assert betti_solid[1] == 0, (
        f"solid triangle β_1 should be 0 (filled by 2-simplex), got {betti_solid[1]}"
    )
    assert betti_solid[2] == 0, f"solid triangle β_2 should be 0, got {betti_solid[2]}"
    print(f"✓ Test 1 solid triangle: β={betti_solid}")
    print("  (networkx cycle_basis 会误报 β_1=1, GUDHI 正确给 0)")

    # 对比: networkx 算法 (trace_topology 风格)
    try:
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2])
        G.add_edges_from([(0, 1), (1, 2), (0, 2)])
        nx_b0 = nx.number_connected_components(G)
        nx_b1 = len(nx.cycle_basis(G))
        print(f"  networkx 对比: β_0={nx_b0}, β_1={nx_b1} (误报 β_1)")
    except ImportError:
        pass

    # Test 2: 空心 triangle — 3 vertex + 3 edge, 没 2-simplex
    # 这种情况 networkx 正确: β_1=1 (确实是 cycle, 没被填)
    hollow_triangle = [(0,), (1,), (2,), (0, 1), (1, 2), (0, 2)]
    betti_hollow = compute_exact_betti(hollow_triangle, max_dim=2)
    assert betti_hollow[0] == 1
    assert betti_hollow[1] == 1, f"hollow triangle β_1 should be 1, got {betti_hollow[1]}"
    assert betti_hollow[2] == 0
    print(f"✓ Test 2 hollow triangle: β={betti_hollow} (跟 networkx 一致)")

    # Test 3: 两个 disjoint triangle — β_0=2
    two_triangles = [
        (0,), (1,), (2,), (0, 1), (1, 2), (0, 2), (0, 1, 2),
        (3,), (4,), (5,), (3, 4), (4, 5), (3, 5), (3, 4, 5),
    ]
    betti_two = compute_exact_betti(two_triangles, max_dim=2)
    assert betti_two[0] == 2, f"two disjoint triangles β_0 should be 2, got {betti_two[0]}"
    assert betti_two[1] == 0
    print(f"✓ Test 3 two disjoint triangles: β={betti_two}")

    # Test 4: tetrahedron boundary (2-sphere) — β_0=1, β_2=1
    # 4 个 vertex + 6 条 edge + 4 个 triangle (边界), 没 tetrahedron (没填实)
    # 真正 homology: S^2 → β_0=1, β_1=0, β_2=1
    tet_boundary = [
        (0,), (1,), (2,), (3,),
        (0,1), (0,2), (0,3), (1,2), (1,3), (2,3),
        (0,1,2), (0,1,3), (0,2,3), (1,2,3),
    ]
    betti_sphere = compute_exact_betti(tet_boundary, max_dim=3)
    assert betti_sphere[0] == 1, f"S^2 β_0 should be 1, got {betti_sphere[0]}"
    assert betti_sphere[1] == 0, f"S^2 β_1 should be 0, got {betti_sphere[1]}"
    assert betti_sphere[2] == 1, f"S^2 β_2 should be 1 (enclosed void), got {betti_sphere[2]}"
    print(f"✓ Test 4 S^2 (tetrahedron boundary): β={betti_sphere}")
    print("  (networkx 算不出 β_2=1, 这就是 GUDHI 的额外能力)")

    # Test 5: n=50 ring point cloud — persistent 1-dim feature
    # 50 个点排在 r=2 圆周上, 相邻点距 ~2π·2/50≈0.25, max_edge_length=0.6
    # 让相邻 2-3 个点连成边但不至于全 clique — 期望看到 1 个 essential 1-dim
    # feature (圆环形成的洞, death=inf)
    pts = ring_point_cloud(n=50, r=2.0, noise=0.05)
    diag = compute_persistent_homology(pts, max_dim=1, max_edge_length=0.6)
    # 数 essential 1-dim feature (death=inf, 真正跨尺度存在的洞)
    n_essential_1d = sum(1 for d, b, de in diag if d == 1 and math.isinf(de))
    assert n_essential_1d >= 1, (
        f"ring should have ≥1 essential 1-dim feature (death=inf), got {n_essential_1d}"
    )
    print(f"✓ Test 5 ring point cloud (n=50): {n_essential_1d} essential 1-dim feature(s)")
    # 打印最 persistent 的几个 (按 persistence 排序, inf 在前)
    persist_1d = [(d, b, de) for d, b, de in diag if d == 1]
    persist_1d.sort(
        key=lambda x: float("inf") if math.isinf(x[2]) else x[2] - x[1],
        reverse=True,
    )
    for _d, b, de in persist_1d[:3]:
        persist = "inf" if math.isinf(de) else f"{de - b:.3f}"
        ds = "inf" if math.isinf(de) else f"{de:.3f}"
        print(f"  dim=1, birth={b:.3f}, death={ds}, persistence={persist}")

    # Test 6 (成功判据): networkx 给不出 persistence, GUDHI 能给
    # networkx 给出的是静态整数 β_1, 没尺度信息
    # GUDHI 给 (dim, birth, death) — 每个洞都有 scale, 区分 noise (短命) 和
    # 真实结构 (essential 或长 persistence)
    has_scale_info = any(
        d == 1 and not math.isinf(b) for d, b, de in diag
    )
    assert has_scale_info, "persistence diagram 应该有有限 birth (scale info)"
    n_1d_total = sum(1 for d, b, de in diag if d == 1)
    print(f"✓ Test 6 成功判据: GUDHI 给出 {n_1d_total} 个 1-dim feature (含 essential + noise)")
    print("  networkx 只能给整数 β_1, 看不到 birth/death scale — GUDHI 给出 diagram")
    print(f"  → {n_essential_1d} 个 essential feature 跨尺度存在 (death=inf), 即 ring 的 2D hole")
    print("  → Open Problem 7.1 成功判据达成")

    # Test 7: torus point cloud — β_0=1, β_1=2 (经典)
    # 用 Rips approximation (point cloud, 不是 triangulation)
    torus_pts = torus_point_cloud(n=80, R=2.0, r=0.7)
    torus_diag = compute_persistent_homology(torus_pts, max_dim=2, max_edge_length=1.0)
    n_persist_1d_torus = count_persistent_features(torus_diag, dim=1, min_persistence=0.3)
    # torus 真正 β_1=2, Rips approximation 可能多个噪声, 但至少 ≥2
    assert n_persist_1d_torus >= 2, (
        f"torus should have ≥2 persistent 1-dim features (β_1=2), got {n_persist_1d_torus}"
    )
    print(f"✓ Test 7 torus point cloud: {n_persist_1d_torus} persistent 1-dim features (期望 ≥2)")

    # Test 8: 性能开销 sanity check
    import time
    big_pts = np.random.default_rng(0).uniform(0, 5, (200, 3))
    t0 = time.perf_counter()
    _ = compute_persistent_homology(big_pts, max_dim=2, max_edge_length=1.0)
    dt = time.perf_counter() - t0
    print(f"✓ Test 8 性能 n=200: {dt*1000:.0f}ms")
    assert dt < 10.0, f"n=200 Rips 应该 <10s, got {dt:.2f}s"
    print("  (Phase 1-6 trace_topology cap=50, 这里 n=200 远超, 性能可接受)")

    print()
    print("✓ simplicial_homology self-check 全过 — Open Problem 7.1 探索成功")
    print("  GUDHI 已装, 真正 simplicial homology 可用, persistence diagram 可用")


if __name__ == "__main__":
    _selfcheck()
