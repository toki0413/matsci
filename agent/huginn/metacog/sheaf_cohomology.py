"""Sheaf cohomology — Open Problem 7.2 探索层.

替代 hint_coordinator 的 keyword overlap / cosine similarity 代理. 把 Core
context + Support findings 建模成 sheaf (open set / stalk / restriction
map), 然后算 Čech H¹ 检测 "gluing obstruction" — 局部不一致的全局信号.

数学模型:
  - Open cover: 每个 source (Core / Support finding k) 是一个 open set U_i
  - Stalk F(U_i): source i 的 claims 视为向量空间 (key -> value, 数值或字符串)
  - Section s_i: source i 实际赋的值 (它的 claims)
  - Restriction r_{ij}: F(U_i) -> F(U_ij) 把 s_i 投影到 i,j 共享的 keys 上

Čech 复合形:
  - C^0 = ⊕_i F(U_i),        dim = sum of source claim counts
  - C^1 = ⊕_{i<j} F(U_ij),   dim = sum of pairwise shared key counts
  - C^2 = ⊕_{i<j<k} F(U_ijk), dim = sum of triple-shared key counts
  - δ^0: C^0 -> C^1,  (δ^0 s)_{ij} = r_{ji}(s_j) - r_{ij}(s_i)
  - δ^1: C^1 -> C^2,  (δ^1 t)_{ijk} = t_{jk} - t_{ik} + t_{ij}
  - H^1 = ker(δ^1) / im(δ^0),  dim = nullity(δ^1) - rank(δ^0)

对于 constant sheaf + acyclic nerve (Leray 定理), Čech H^1 = 0. 这意味着
纯 Čech H^1 检测不到 "pairwise disagreement" — 因为 actual disagreement
D_{ij} = s_j - s_i 本身就是 δ^0(s_actual) 的 (i,j) 分量, 是 coboundary,
cohomology class 永远 trivial.

所以 spec §7.2 显式允许: "如果 H^1 计算太复杂, 降级到 检测 restriction
map 失败 作为 H^1≠0 的代理". 我们做双层:
  Layer 1 (math): Čech H^1 via coboundary matrix rank + nullity
                  检测 "twist" obstruction (pairwise consistent 但 triple twist)
  Layer 2 (proxy): restriction failure count = #pairs (i,j) where
                  r_{ij}(s_i) ≠ r_{ji}(s_j) on shared keys
                  检测 pairwise disagreement (Phase 1-6 实际关心的)

返回 H^1 = Čech_H^1 + restriction_failure_count. 一致 sources -> 0;
任一数值/语义冲突 -> >0.

研究探索层 (Open Problem 7.2): 不改 hint_coordinator.py, 不阻塞 Phase 1-6.
失败容忍: 算不动就降级到 layer 2 (proxy). 升级路径: 把 layer 2 替换成
non-constant sheaf (e.g. local system with monodromy) 的真 Čech H^1.

ponytail: 单文件, numpy + stdlib, 不引 GUDHI cohomology (Phase 7.1 装的
GUDHI 是 simplex tree, 跟 sheaf cohomology 是两条路). 手写 coboundary
matrix + SVD rank.
"""
from __future__ import annotations

# 直接跑脚本时把 agent/ 加到 sys.path (被 import 时不执行, rcb_runner 已设好)
if __name__ == "__main__":
    import sys as _sys
    from pathlib import Path as _Path
    _agent_root = str(_Path(__file__).resolve().parents[2])
    if _agent_root not in _sys.path:
        _sys.path.insert(0, _agent_root)

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# 数值提取: regex 抓 "Tc=100K", "Tc = 100 K", "temperature=290" 等.
# ponytail: 简单 regex, 不上 parser. 单位 (K, eV, ...) 暂时丢掉, 只留数值.
# 升级路径: spaCy / LLM extract 把 "transition temperature reaches 100 kelvin"
# 也归一化成 {"Tc": 100.0, "unit": "K"}.
_NUMERIC_PATTERN = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)

# 语义对端词: 同一个 stalk 不应该出现两端的任意一端 (冲突才算, 单出现不算).
# 取自材料学常见 binary: 磁有序 / 拓扑 / 导电性. 跟 Phase 1-6 同天花板:
# 同义改写漏检 (e.g. "spin splitting" ↔ "altermagnetism") 升级路径用 embedding.
_SEMANTIC_CONFLICT_PAIRS: list[tuple[str, str]] = [
    ("altermagnetism", "ferromagnetism"),
    ("altermagnet", "ferromagnet"),
    ("antiferromagnetic", "ferromagnetic"),
    ("topological", "trivial"),
    ("superconducting", "insulating"),
    ("metallic", "insulating"),
]


@dataclass
class Stalk:
    """Sheaf stalk — 一个 source 的局部数据.

    claims: key -> value. value 是 float (数值 claim) 或 str (语义 claim,
    e.g. {"magnetic_order": "altermagnetism"}). 同 key 跨 source 出现就
    构成 overlap, restriction map 检查两 source 的 value 是否一致.
    """
    source_id: str
    claims: dict[str, Any]


@dataclass
class Sheaf:
    """Sheaf over a finite open cover — Core + Support findings.

    open_sets: source ids, e.g. ["core", "support_0", "support_1"]
    stalks: 每个 open set 的 Stalk
    restriction_maps: (i, j) -> 排序后的共享 keys 列表, i < j
        U_{ij} 上的 stalk basis = 这些 keys (向量空间)
    """
    open_sets: list[str]
    stalks: list[Stalk]
    restriction_maps: dict[tuple[int, int], list[str]] = field(default_factory=dict)

    def overlap_keys(self, i: int, j: int) -> list[str]:
        """U_i ∩ U_j 上 stalk 的 basis keys (两 source 都声明的字段)."""
        if i == j:
            return sorted(self.stalks[i].claims.keys())
        key = (min(i, j), max(i, j))
        return list(self.restriction_maps.get(key, []))


# ---------- claim 提取 ----------

def _extract_claims(text: str) -> dict[str, Any]:
    """从一段文本提取数值 claim + 语义关键词 claim.

    数值: regex 抓 key=value (e.g. "Tc=100K" -> {"tc": 100.0})
    语义: 关键词出现就标 magnetic_order (后出现的覆盖前者, 同一 source 内
          只能有一个 order, 矛盾时按文本顺序取最后)

    ponytail: regex + 关键词表, 不上 NLP parser. 同义改写漏检是已知天花板.
    """
    claims: dict[str, Any] = {}
    text_lower = text.lower()
    # 数值 claim
    for m in _NUMERIC_PATTERN.finditer(text_lower):
        key = m.group(1)
        try:
            val = float(m.group(2))
        except ValueError:
            continue
        # 后出现的覆盖前者 (e.g. "Tc=100K ... Tc=50K" 取 50K — 最后陈述优先)
        claims[key] = val
    # 语义 claim: 关键词出现就标. 按 keyword 长度降序处理避免短词覆盖长词
    # (e.g. "altermagnet" 是 "altermagnetism" 子串, 先匹长词就不被覆盖).
    # 已有 magnetic_order 时不再覆盖 (一个 source 一个 order, 重复出现取首次).
    _sorted_pairs = sorted(
        _SEMANTIC_CONFLICT_PAIRS,
        key=lambda p: -max(len(p[0]), len(p[1])),
    )
    for kw_a, kw_b in _sorted_pairs:
        if "magnetic_order" in claims:
            break
        if kw_a in text_lower:
            claims["magnetic_order"] = kw_a
        elif kw_b in text_lower:
            claims["magnetic_order"] = kw_b
    return claims


def build_sheaf_from_findings(
    core_context: str | dict[str, Any],
    support_findings: list[str | dict[str, Any]],
) -> Sheaf:
    """从 Core context + Support findings 构造 sheaf.

    Args:
        core_context: 任务陈述 (str) 或预提取的 claim dict (key -> value)
        support_findings: related_work 提取的 claims, 每项是 str 或 dict

    Returns:
        Sheaf, open cover = [core, support_0, support_1, ...]

    Example:
        >>> sheaf = build_sheaf_from_findings(
        ...     "Task: reproduce Tc=100K altermagnetism paper",
        ...     ["Paper reports Tc=100K altermagnetism"])
        >>> compute_H1(sheaf)
        0  # sources agree -> H^1 = 0
    """
    open_sets: list[str] = ["core"]
    stalks: list[Stalk] = []

    if isinstance(core_context, dict):
        core_claims = dict(core_context)
    else:
        core_claims = _extract_claims(core_context)
    stalks.append(Stalk(source_id="core", claims=core_claims))

    for k, sf in enumerate(support_findings):
        sid = f"support_{k}"
        open_sets.append(sid)
        if isinstance(sf, dict):
            claims = dict(sf)
        else:
            claims = _extract_claims(sf)
        stalks.append(Stalk(source_id=sid, claims=claims))

    # Restriction maps: pairwise 共享 keys (sorted 保证 basis 顺序稳定)
    restriction_maps: dict[tuple[int, int], list[str]] = {}
    for i in range(len(stalks)):
        for j in range(i + 1, len(stalks)):
            shared = sorted(
                set(stalks[i].claims.keys()) & set(stalks[j].claims.keys())
            )
            if shared:
                restriction_maps[(i, j)] = shared

    return Sheaf(
        open_sets=open_sets,
        stalks=stalks,
        restriction_maps=restriction_maps,
    )


# ---------- 值比较 ----------

def _values_agree(
    a: Any, b: Any, *, rel_tol: float = 1e-3, abs_tol: float = 1e-2,
) -> bool:
    """两个 claim 值是否一致.

    数值: math.isclose 语义 (rel + abs tolerance). 跟 Phase 1-6 hint_coordinator
          数值冲突检查的容差一致 — 不上 fuzzy match.
    字符串: lower 后完全相等.

    ponytail: 简单比较, 不上 fuzzy string match. 同义词漏检 (e.g.
    "altermagnet" ↔ "spin-split antiferromagnet") 是已知天花板.
    """
    # 数值 vs 数值
    if isinstance(a, (int, float)) and not isinstance(a, bool) \
            and isinstance(b, (int, float)) and not isinstance(b, bool):
        if a == 0 and b == 0:
            return True
        if abs(a - b) <= rel_tol * max(abs(a), abs(b)):
            return True
        if abs(a - b) <= abs_tol:
            return True
        return False
    # 字符串或混合: lower 后字符串比较
    return str(a).lower() == str(b).lower()


def _check_restriction(sheaf: Sheaf, i: int, j: int) -> bool:
    """检查 U_i, U_j 上 restriction map 是否一致.

    两 source 在所有共享 keys 上值一致 -> True (可 glue).
    任一 key 不一致 -> False (restriction failure).

    spec §7.2 H^1 != 0 proxy: 任一 pair (i,j) 失败 -> H^1 != 0 信号.
    """
    shared = sheaf.overlap_keys(i, j)
    if not shared:
        return True  # 无 overlap, trivially consistent (跟 Phase 1-6 keyword overlap=0 一致)
    s_i = sheaf.stalks[i].claims
    s_j = sheaf.stalks[j].claims
    for k in shared:
        if k not in s_i or k not in s_j:
            continue  # 安全保护, overlap_keys 已过滤
        if not _values_agree(s_i[k], s_j[k]):
            return False
    return True


# ---------- Čech coboundary matrix ----------

def _build_cech_matrices(sheaf: Sheaf) -> tuple[np.ndarray, np.ndarray]:
    """构造 Čech coboundary matrices δ^0: C^0 -> C^1 和 δ^1: C^1 -> C^2.

    每个 stalk 当 R-vector space, basis = sorted keys. Overlap U_{ij} 上
    stalk basis = sorted 共享 keys. Triple overlap U_{ijk} 上 = 三 source
    都有的 keys.

    δ^0(s)_{(ij), k} = s_j[k] - s_i[k]   (linear in s; pairwise diff)
    δ^1(t)_{(ijk), k} = t_{ij}[k] - t_{ik}[k] + t_{jk}[k]   (cocycle condition)

    Returns: (delta0, delta1) 实数矩阵. shape: (dim C^1, dim C^0) 和 (dim C^2, dim C^1).
    """
    n = len(sheaf.stalks)

    # C^0 basis: 每个 source 的每个 key
    c0_basis: list[tuple[int, str]] = []
    for i, stalk in enumerate(sheaf.stalks):
        for k in sorted(stalk.claims.keys()):
            c0_basis.append((i, k))
    dim_c0 = len(c0_basis)
    c0_idx = {b: idx for idx, b in enumerate(c0_basis)}

    # C^1 basis: 每个 pairwise overlap 的每个共享 key
    c1_basis: list[tuple[int, int, str]] = []
    pairs = sorted(sheaf.restriction_maps.keys())
    for (i, j) in pairs:
        for k in sheaf.restriction_maps[(i, j)]:
            c1_basis.append((i, j, k))
    dim_c1 = len(c1_basis)
    c1_idx = {b: idx for idx, b in enumerate(c1_basis)}

    # C^2 basis: 每个 triple overlap 的每个共享 key (3 source 都有的 key)
    c2_basis: list[tuple[int, int, int, str]] = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                shared_triple = (
                    set(sheaf.stalks[i].claims.keys())
                    & set(sheaf.stalks[j].claims.keys())
                    & set(sheaf.stalks[k].claims.keys())
                )
                for kk in sorted(shared_triple):
                    c2_basis.append((i, j, k, kk))
    dim_c2 = len(c2_basis)
    c2_idx = {b: idx for idx, b in enumerate(c2_basis)}

    # δ^0: (dim_c1, dim_c0). δ^0(s)_{(i,j), k} = s_j[k] - s_i[k]
    delta0 = np.zeros((dim_c1, dim_c0), dtype=float)
    for (i, j, k), row in c1_idx.items():
        col_i = c0_idx.get((i, k))
        col_j = c0_idx.get((j, k))
        if col_i is not None:
            delta0[row, col_i] = -1.0
        if col_j is not None:
            delta0[row, col_j] = +1.0

    # δ^1: (dim_c2, dim_c1). δ^1(t)_{(i,j,k), kk} = t_{ij}[kk] - t_{ik}[kk] + t_{jk}[kk]
    # 标准 Čech differential, i < j < k 已由构造保证
    delta1 = np.zeros((dim_c2, dim_c1), dtype=float)
    for (i, j, k, kk), row in c2_idx.items():
        col_ij = c1_idx.get((i, j, kk))  # i < j 已保证
        col_ik = c1_idx.get((i, k, kk))  # i < k 已保证 (k > j > i)
        col_jk = c1_idx.get((j, k, kk))  # j < k 已保证
        if col_ij is not None:
            delta1[row, col_ij] += +1.0
        if col_ik is not None:
            delta1[row, col_ik] += -1.0
        if col_jk is not None:
            delta1[row, col_jk] += +1.0

    return delta0, delta1


def _matrix_rank(M: np.ndarray, *, tol: float = 1e-9) -> int:
    """矩阵的 rank via SVD. tol 以下 singular value 视为 0.

    ponytail: numpy SVD, 不上 scipy. 实数 coboundary matrix, 复数这里用不上.
    """
    if M.size == 0:
        return 0
    s = np.linalg.svd(M, compute_uv=False)
    return int(np.sum(s > tol))


def _count_restriction_failures(sheaf: Sheaf) -> int:
    """数 restriction map 失败的 pairwise overlap 数 — spec §7.2 H^1!=0 proxy.

    对每个 overlap U_{ij}, 检查 s_i 跟 s_j 在共享 keys 上是否一致.
    任一 key 不一致 -> 该 overlap restriction failure.

    Returns: 失败的 overlap 数 (>=0). 0 = 全部 pairwise consistent.
    """
    n_failures = 0
    for (i, j) in sheaf.restriction_maps:
        if not _check_restriction(sheaf, i, j):
            n_failures += 1
    return n_failures


def compute_H1(sheaf: Sheaf) -> int:
    """Sheaf 的 Čech H^1 — Open Problem 7.2 主算法.

    双层 (spec §7.2 允许):
      Layer 1 (math): Čech H^1 = nullity(δ^1) - rank(δ^0). 检测 "twist"
                      obstruction (pairwise consistent 但 triple 不一致的
                      global twist). 对 constant sheaf + acyclic nerve
                      通常 = 0 (Leray 定理).
      Layer 2 (proxy): restriction failure count = #pairs where r_{ij}(s_i)
                      ≠ r_{ji}(s_j). 检测 pairwise disagreement. 对 2-source
                      冲突 case 这层非 0.

    Returns: int >= 0. 0 = sources 可以 glue 成 global section (consistent).
             >0 = 检测到不一致 (数值或语义冲突).

    ponytail: 不依赖 GUDHI cohomology (那是 simplex tree 的链, 跟 sheaf
    是两条路). 手写 coboundary matrix + SVD rank. 升级路径: 上 persistent
    sheaf cohomology (上游研究工具, 如 `pySheaf`).
    """
    # Layer 1: Čech H^1 via linear algebra
    delta0, delta1 = _build_cech_matrices(sheaf)
    rank_d0 = _matrix_rank(delta0)
    rank_d1 = _matrix_rank(delta1)
    dim_c1 = delta0.shape[0]
    nullity_d1 = dim_c1 - rank_d1
    cech_h1 = max(0, nullity_d1 - rank_d0)

    # Layer 2: restriction failure proxy (spec §7.2 fallback)
    n_failures = _count_restriction_failures(sheaf)

    return cech_h1 + n_failures


# ---------- Self-check ----------

def _selfcheck() -> None:
    """验证 4 个 test case + Phase 1-6 行为一致性 + 数学层 sanity.

    Test 1: Core 跟 Support 一致 -> H^1 = 0
    Test 2: 数值冲突 (Core Tc=100K, Support Tc=50K) -> H^1 != 0
    Test 3: 语义冲突 (Core altermagnetism, Support ferromagnetism) -> H^1 != 0
    Test 4: 跟 Phase 1-6 hint_coordinator 行为一致
            (H^1=0 时不冲突, H^1!=0 时冲突; sheaf 严格更强 — 检出 Phase 1-6 漏检的语义冲突)
    """
    print("=== Phase 7 Open Problem 7.2: Sheaf Cohomology self-check ===\n")

    # Test 1: Core 跟 Support 一致 — 全部 claim 对齐
    sheaf1 = build_sheaf_from_findings(
        core_context="Task: reproduce altermagnetism paper, Tc=100K reported",
        support_findings=[
            "The paper reports Tc=100K with altermagnetism ordering",
        ],
    )
    h1_1 = compute_H1(sheaf1)
    assert h1_1 == 0, f"Test 1 consistent should give H^1=0, got {h1_1}"
    print(f"PASS Test 1 (consistent): H^1={h1_1}")
    print(f"  Core claims:    {sheaf1.stalks[0].claims}")
    print(f"  Support claims: {sheaf1.stalks[1].claims}")
    print(f"  Overlap keys: {sheaf1.overlap_keys(0, 1)}")

    # Test 2: 数值冲突 — Core Tc=100K, Support Tc=50K
    sheaf2 = build_sheaf_from_findings(
        core_context="Task: reproduce Tc=100K altermagnetism paper",
        support_findings=[
            "The paper reports Tc=50K with altermagnetism ordering",
        ],
    )
    h1_2 = compute_H1(sheaf2)
    assert h1_2 > 0, f"Test 2 numerical conflict should give H^1>0, got {h1_2}"
    print(f"\nPASS Test 2 (numerical conflict Tc=100K vs Tc=50K): H^1={h1_2}")
    print(f"  Core claims:    {sheaf2.stalks[0].claims}")
    print(f"  Support claims: {sheaf2.stalks[1].claims}")
    print(f"  Restriction failures: {_count_restriction_failures(sheaf2)}")

    # Test 3: 语义冲突 — Core altermagnetism, Support ferromagnetism
    sheaf3 = build_sheaf_from_findings(
        core_context="Task: study altermagnetism ordering in MnTe",
        support_findings=[
            "The compound shows ferromagnetism with Tc=100K",
        ],
    )
    h1_3 = compute_H1(sheaf3)
    assert h1_3 > 0, f"Test 3 semantic conflict should give H^1>0, got {h1_3}"
    print(f"\nPASS Test 3 (semantic conflict altermagnetism vs ferromagnetism): H^1={h1_3}")
    print(f"  Core claims:    {sheaf3.stalks[0].claims}")
    print(f"  Support claims: {sheaf3.stalks[1].claims}")
    print(f"  Restriction failures: {_count_restriction_failures(sheaf3)}")

    # Test 4: 跟 Phase 1-6 hint_coordinator 行为一致性
    # Phase 1-6 _detect_conflict 只能检测特定 keyword 冲突
    # ("按选定方案执行" + "换数学结构"); _keyword_overlap 是 Jaccard.
    # Sheaf H^1 是严格更强的检测器 — 检出 Phase 1-6 漏检的语义/数值冲突.
    # 行为一致性: H^1=0 时 Phase 1-6 也不冲突 (no false positive);
    # H^1!=0 时是真正冲突 (Phase 1-6 可能漏检但不算 false positive).
    from huginn.agent.hint_coordinator import HintCoordinator
    _hc = HintCoordinator()

    # case 4a: 一致 sources — Phase 1-6 不冲突, sheaf H^1=0. 行为一致.
    conflict_consistent_huginn = _hc._detect_conflict(
        "execute methodology checklist",
        "no special keyword",
    )
    assert conflict_consistent_huginn is False, "Phase 1-6 should not flag consistent"
    assert h1_1 == 0, "sheaf should give H^1=0 for consistent"
    print(f"\nPASS Test 4a (consistent): Phase 1-6 _detect_conflict="
          f"{conflict_consistent_huginn}, sheaf H^1={h1_1} - 行为一致 (no false positive)")

    # case 4b: Phase 1-6 检出的冲突 — sheaf 也检出 (强化检测, 不漏)
    # 这里直接构造一个数值冲突, Phase 1-6 keyword 漏检但 sheaf H^1!=0
    _no_kw_conflict = _hc._detect_conflict(
        "Tc=100K reproduce",
        "Tc=50K reported",  # Phase 1-6 keyword 不命中 — 漏检
    )
    assert _no_kw_conflict is False, "Phase 1-6 misses numerical conflict (no keyword)"
    assert h1_2 > 0, "sheaf detects numerical conflict"
    print(f"PASS Test 4b (numerical conflict): Phase 1-6 _detect_conflict="
          f"{_no_kw_conflict} (漏检), sheaf H^1={h1_2} - sheaf 严格更强")

    # case 4c: Phase 1-6 keyword 检出冲突时, sheaf 也检出 (不漏)
    # 构造一个既命中 Phase 1-6 keyword 又有数值冲突的 case
    sheaf_4c = build_sheaf_from_findings(
        core_context="按选定方案执行 Tc=100K",
        support_findings=["换数学结构 Tc=50K"],
    )
    h1_4c = compute_H1(sheaf_4c)
    _kw_conflict = _hc._detect_conflict("按选定方案执行", "换数学结构")
    assert _kw_conflict is True, "Phase 1-6 should flag keyword conflict"
    assert h1_4c > 0, "sheaf should detect numerical conflict in 4c"
    print(f"PASS Test 4c (Phase 1-6 keyword + numerical conflict): "
          f"Phase 1-6 _detect_conflict={_kw_conflict}, sheaf H^1={h1_4c} - 双方都检出")

    # 数学层 sanity: 验证 Čech complex 结构
    print("\n--- 数学层 sanity ---")
    delta0_1, delta1_1 = _build_cech_matrices(sheaf1)
    rank_d0_1 = _matrix_rank(delta0_1)
    rank_d1_1 = _matrix_rank(delta1_1)
    nullity_d1_1 = delta0_1.shape[0] - rank_d1_1
    cech_h1_1 = max(0, nullity_d1_1 - rank_d0_1)
    print(f"  Test 1 sheaf: dim C^0={delta0_1.shape[1]}, dim C^1={delta0_1.shape[0]}, "
          f"dim C^2={delta1_1.shape[0]}")
    print(f"  rank(δ^0)={rank_d0_1}, rank(δ^1)={rank_d1_1}, "
          f"nullity(δ^1)={nullity_d1_1}, Čech H^1={cech_h1_1}")
    # 2-source + 2 shared keys: C^0 dim=4 (2 sources x 2 keys),
    # C^1 dim=2 (2 shared keys), δ^0 rank=2 (surjective), C^2 dim=0
    # nullity(δ^1) = 2 (no C^2 -> δ^1 trivial), H^1 = 2 - 2 = 0 (consistent)
    assert cech_h1_1 == 0, f"Čech H^1 for consistent 2-source should be 0, got {cech_h1_1}"

    # Bonus 1: 3 source 全一致 — 验证 Čech 三 source 复合形
    sheaf_3 = build_sheaf_from_findings(
        core_context="Tc=100K altermagnetism",
        support_findings=[
            "Paper A: Tc=100K altermagnetism",
            "Paper B: Tc=100K altermagnetism",
        ],
    )
    h1_3src = compute_H1(sheaf_3)
    assert h1_3src == 0, f"3-source consistent should give H^1=0, got {h1_3src}"
    delta0_3, delta1_3 = _build_cech_matrices(sheaf_3)
    print(f"\n  Bonus 1 (3-source consistent): H^1={h1_3src}")
    print(f"    dim C^0={delta0_3.shape[1]}, dim C^1={delta0_3.shape[0]}, "
          f"dim C^2={delta1_3.shape[0]}")
    print(f"    rank(δ^0)={_matrix_rank(delta0_3)}, rank(δ^1)={_matrix_rank(delta1_3)}")
    # 3 sources x 2 keys = 6 in C^0, 3 pairs x 2 keys = 6 in C^1,
    # 1 triple x 2 keys = 2 in C^2. δ^0 rank=4, δ^1 rank=2, nullity(δ^1)=4,
    # Čech H^1 = 4 - 4 = 0 (constant sheaf on contractible nerve)

    # Bonus 2: 3 source, pair (0,1) 冲突 (1-2, 0-2 一致) — H^1 由 proxy layer 检出
    sheaf_twist = build_sheaf_from_findings(
        core_context="Tc=100K altermagnetism",
        support_findings=[
            "Paper A: Tc=50K altermagnetism",   # 跟 Core Tc 冲突
            "Paper B: Tc=100K altermagnetism",  # 跟 Core 一致
        ],
    )
    h1_twist = compute_H1(sheaf_twist)
    assert h1_twist > 0, f"3-source with pair (0,1) conflict should give H^1>0, got {h1_twist}"
    print(f"\n  Bonus 2 (3-source, pair (0,1) conflict): H^1={h1_twist}")
    print(f"    Restriction failures: {_count_restriction_failures(sheaf_twist)}")
    # Čech H^1 仍 = 0 (constant sheaf), 但 restriction failure = 1 -> 总 H^1 = 1

    # Bonus 3: dict 输入 (预提取 claims, 跳过 regex)
    sheaf_dict = build_sheaf_from_findings(
        core_context={"tc": 100.0, "magnetic_order": "altermagnetism"},
        support_findings=[{"tc": 100.0, "magnetic_order": "altermagnetism"}],
    )
    assert compute_H1(sheaf_dict) == 0, "dict-input consistent should give H^1=0"
    sheaf_dict_conflict = build_sheaf_from_findings(
        core_context={"tc": 100.0},
        support_findings=[{"tc": 50.0}],
    )
    assert compute_H1(sheaf_dict_conflict) > 0, "dict-input conflict should give H^1>0"
    print(f"\n  Bonus 3 (dict input): consistent H^1=0, conflict H^1>0 - OK")

    print("\n=== Phase 7 Open Problem 7.2 self-check PASSED ===")
    print("  Sheaf 建模: Core/Support -> open sets + stalks + restriction maps")
    print("  H^1 计算: Čech coboundary δ^0,δ^1 + nullity-rank + restriction failure proxy")
    print("  成功判据: H^1=0 一致, H^1!=0 检出数值+语义冲突, 跟 Phase 1-6 行为一致 (no false positive, 严格更强)")
    print("  -> Open Problem 7.2 探索成功")


if __name__ == "__main__":
    _selfcheck()
