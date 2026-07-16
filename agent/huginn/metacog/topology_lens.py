"""拓扑透镜 — 高阶网络视角的元认知判据.

把 Fujita & Smarandache (2026) 的高阶网络四大族系 + 6 个心智模型提炼成
agent 可调用的判据函数. 不是完整 SKILL, 是元认知层的结构透镜.

接入点:
- red_team._review_hypothesis: 调 needs_downward_closure 判断假设的证据
  网络是否需要单纯复形结构 (有闭包才能用 Hodge/Betti)
- equivalence_auditor: 用 hodge_signature 判断两个假设是否拓扑等价
  (调和分量不同 = 拓扑不同 = 真不同, 不是换名)
- hypothesis_loop.refine: 用 topology_permits 判断目标动力学模式是否被
  当前网络拓扑允许 (β_k=0 时该模式不可能出现, 假设需重构)

设计原则 (ponytail):
- 判据是启发式, 不是定理. 返回 (verdict, reason) 让 LLM/规则做最终判定.
- 不依赖网络库 (networkx/gudhi), 用 frozenset + dict 算近似判据.
  升级路径: 接 gudhi 算真实 Betti 数, 但当前判据足够元认知用.
- 6 个心智模型 → 4 个判据函数 (合并表达力-可处理性到 classify_system).

诚实边界:
- 高阶网络视角是认知透镜, 不是物理真理. 判据返回 "建议" 不是 "约束".
- β_k 近似用连通分量数, 不是真实同调群. 升级: 接 gudhi.
- Hodge 签名用证据拓扑的度分布近似, 不是真实 Hodge 分解.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# ── 四族分类 ──────────────────────────────────────────────────

Family = Literal[
    "combinatorial",   # 超图/MetaGraph — 无向下闭包, 建模自由
    "topological",     # 单纯复形/CW — 有闭包, 有同调/Hodge/Betti
    "factorization",   # 因子图/多层/时间 — 分解/层/时间
    "semantic",        # 知识图/Petri — 语义/逻辑
]


@dataclass
class FamilyVerdict:
    """classify_system 的返回: 推荐族 + 理由 + 升级路径."""

    family: Family
    reason: str
    # 是否需要向下闭包 (topological 族需要, 其他不需要)
    needs_closure: bool
    # 升级路径: 如果当前族不够, 下一步该试什么
    upgrade_hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "reason": self.reason,
            "needs_closure": self.needs_closure,
            "upgrade_hint": self.upgrade_hint,
        }


def classify_system(
    interactions: list[set[str]],
    need_topological_invariants: bool = False,
    need_orthogonal_decomposition: bool = False,
) -> FamilyVerdict:
    """四族透镜: 根据交互模式 + 需求推荐网络族.

    interactions: 交互组列表, 每组是参与的实体名集合.
      例: [{"C3S","H2O","C-S-H","CH"}] 是一个化学反应 (4-uniform 超边)
    need_topological_invariants: 是否需要 Betti 数/同调 (孔隙网络渗流需要)
    need_orthogonal_decomposition: 是否需要 Hodge 梯度/旋度/调和三分量

    判据 (按 skill 的 10 条决策启发式):
    1. 需要正交分解 → topological (只有单纯复形给 Hodge)
    2. 需要拓扑不变量 → topological (需要向下闭包)
    3. 交互自然多路且子群不交互 → combinatorial (化学反应)
    4. 交互有层级包含 → topological 或 combinatorial (看闭包)
    5. 默认 → combinatorial (最简, O(|V|+|E|))

    ponytail: 判据是启发式, 不是定理. 真实物理系统可能混合多族.
    """
    if need_orthogonal_decomposition:
        return FamilyVerdict(
            family="topological",
            reason="需要 Hodge 正交分解 (梯度/旋度/调和), 只有单纯复形给三分量",
            needs_closure=True,
            upgrade_hint="若数据不足以支撑闭包, 退化到超图 + 启发式调和检测",
        )

    if need_topological_invariants:
        return FamilyVerdict(
            family="topological",
            reason="需要 Betti 数/同调群, 强制向下闭包 (单纯复形)",
            needs_closure=True,
            upgrade_hint="孔隙/力链等物理连通 → 闭包自然成立; 化学反应 → 用超图",
        )

    # 检查交互是否自然多路 (任一交互 size>2)
    max_size = max((len(s) for s in interactions), default=0)
    if max_size > 2:
        return FamilyVerdict(
            family="combinatorial",
            reason=f"检测到 {max_size}-路交互, 子群未必交互 → 超图 (无闭包)",
            needs_closure=False,
            upgrade_hint="若发现子群必然也交互, 升级到单纯复形获得同调工具",
        )

    # 默认: 简单图/超图都行, 选超图 (更通用, 成本相同)
    return FamilyVerdict(
        family="combinatorial",
        reason="交互规模 ≤2, 用超图 (与简单图同成本, 更通用)",
        needs_closure=False,
        upgrade_hint="若出现 3+ 路交互且子群也交互, 升级到单纯复形",
    )


# ── 向下闭包判据 ──────────────────────────────────────────────

@dataclass
class ClosureCheck:
    """needs_downward_closure 的返回."""

    needs_closure: bool
    reason: str
    # 闭包是否自然成立 (物理保证) vs 需要强制添加
    natural: bool = False
    # 闭包缺失会丢失什么能力
    lost_capability: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "needs_closure": self.needs_closure,
            "reason": self.reason,
            "natural": self.natural,
            "lost_capability": self.lost_capability,
        }


def needs_downward_closure(
    interactions: list[set[str]],
    physical_connectivity: bool = False,
) -> ClosureCheck:
    """向下闭包判据: k 路交互是否强制所有 <k 子群也交互.

    physical_connectivity: 交互是否源于物理连通 (孔隙/力链/流).
      True → 闭包自然成立 (3 孔连通则任 2 孔连通)
      False → 闭包未必成立 (化学反应子群不是反应)

    判据 (skill 启发式 9):
    - 物理连通 → 闭包自然 → 单纯复形, 获得同调/Hodge/Betti
    - 化学/逻辑交互 → 闭包不成立 → 超图, 保留建模自由
    - 混合 → 用 path complex 或 Dowker complex (中间地带)
    """
    if physical_connectivity:
        return ClosureCheck(
            needs_closure=True,
            reason="物理连通保证子群必然交互 → 闭包自然成立",
            natural=True,
            lost_capability="不用闭包会丢失同调群/Betti 数/Hodge 分解",
        )

    # 非物理连通: 检查是否真的需要闭包
    max_size = max((len(s) for s in interactions), default=0)
    if max_size <= 2:
        return ClosureCheck(
            needs_closure=False,
            reason="交互规模 ≤2, 无闭包需求 (简单图天然闭包)",
        )

    return ClosureCheck(
        needs_closure=False,
        reason=f"{max_size}-路交互但非物理连通, 子群未必交互 → 超图保留自由",
        natural=False,
        lost_capability="放弃闭包 = 放弃同调工具; 若后续需要 Betti 数, 转单纯复形",
    )


# ── Hodge 签名 (等价性审计用) ─────────────────────────────────

@dataclass
class HodgeSignature:
    """假设/证据网络的近似 Hodge 签名.

    用于等价性审计: 两个假设的签名不同 → 拓扑不同 → 真不同 (非换名).
    签名相同 → 可能等价, 需更深判断.

    ponytail: 用度分布 + 环检测近似, 不是真实 Hodge 分解.
    升级: 接 gudhi 算真实 Betti 数和 Hodge Laplacian 特征值.
    """

    # 顶点数 (假设/证据节点数)
    n_vertices: int
    # 边数 (支持/反驳关系)
    n_edges: int
    # 近似 β₁: 独立环数 = E - V + 连通分量数 (图论欧拉示性数)
    beta1_approx: int
    # 度分布的熵 (高熵 = 均匀, 低熵 = 集中)
    degree_entropy: float
    # 是否有调和分量 (β₁>0 → 有拓扑环, 存在调和 1-形式)
    has_harmonic: bool

    def differs_from(self, other: HodgeSignature) -> tuple[bool, str]:
        """两个签名是否拓扑不同.

        返回 (is_different, reason). 不同 = 真不同, 相同 = 可能等价.
        判据: β₁ 不同 或 度熵差异 > 0.3 → 拓扑不同.
        """
        if self.beta1_approx != other.beta1_approx:
            return True, (
                f"β₁ 不同 ({self.beta1_approx} vs {other.beta1_approx}) → "
                f"独立环数不同 → 拓扑不同 → 非换名归约"
            )
        entropy_diff = abs(self.degree_entropy - other.degree_entropy)
        if entropy_diff > 0.3:
            return True, (
                f"度熵差异 {entropy_diff:.2f} > 0.3 → 证据分布结构不同 → "
                f"非换名归约"
            )
        if self.has_harmonic != other.has_harmonic:
            return True, (
                f"调和分量存在性不同 ({self.has_harmonic} vs {other.has_harmonic}) → "
                f"拓扑环结构不同 → 非换名归约"
            )
        return False, "签名相似, 可能等价, 需更深判断"


def hodge_signature(
    nodes: list[str],
    edges: list[tuple[str, str]],
) -> HodgeSignature:
    """给假设/证据网络算近似 Hodge 签名.

    nodes: 节点 ID 列表 (假设/证据)
    edges: 边列表 (支持/反驳关系, 无向)

    近似方法 (ponytail: 不依赖 gudhi):
    - β₁ ≈ E - V + C (C=连通分量数, 图论欧拉示性数)
    - 度熵 = -Σ p(d) log p(d), p(d)=度数 d 的节点占比
    - has_harmonic = β₁ > 0

    升级: 接 gudhi 算 Vietoris-Rips 复形的真实 Betti 数.
    """
    n_v = len(nodes)
    n_e = len(edges)
    if n_v == 0:
        return HodgeSignature(0, 0, 0, 0.0, False)

    # 连通分量 (Union-Find, 不引 networkx)
    parent = {n: n for n in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in edges:
        if a in parent and b in parent:
            union(a, b)

    components = len({find(n) for n in nodes})
    beta1 = n_e - n_v + components
    if beta1 < 0:
        beta1 = 0  # 森的 β₁=0

    # 度分布熵
    from collections import Counter
    import math
    degree = Counter()
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
    if degree:
        total = sum(degree.values())
        entropy = -sum(
            (c / total) * math.log2(c / total)
            for c in degree.values()
        )
    else:
        entropy = 0.0

    return HodgeSignature(
        n_vertices=n_v,
        n_edges=n_e,
        beta1_approx=beta1,
        degree_entropy=entropy,
        has_harmonic=beta1 > 0,
    )


# ── 拓扑许可动力学 ────────────────────────────────────────────

@dataclass
class TopologyPermit:
    """topology_permits 的返回: 拓扑是否允许目标动力学模式."""

    permitted: bool
    reason: str
    # 当前拓扑的近似 Betti 数 (β₀=连通分量, β₁=独立环)
    beta0: int = 0
    beta1: int = 0
    # 若不许可, 需要什么拓扑变化
    required_change: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "permitted": self.permitted,
            "reason": self.reason,
            "beta0": self.beta0,
            "beta1": self.beta1,
            "required_change": self.required_change,
        }


def topology_permits(
    nodes: list[str],
    edges: list[tuple[str, str]],
    target_mode: Literal[
        "synchronization",    # 同步: 需要 β₁ > 0 (拓扑环)
        "percolation",         # 渗流: 需要 β₀ > 1 (多连通分量) 或 β₀=1 全连通
        "circulation",         # 环流: 需要 β₁ > 0 (独立环)
        "global_consistency",  # 全局一致: 需要层粘合 H¹=0
    ],
) -> TopologyPermit:
    """拓扑许可动力学: 目标模式是否被当前网络拓扑允许.

    skill Model 4 的核心: β_k=0 时 k 维信号不可能同步, 无论耦合强度.
    推广: 某些动力学模式需要特定拓扑特征.

    target_mode:
    - synchronization: k 维同步需要 β_k > 0
    - percolation: 渗流需要多连通 (β₀>1) 或全连通 (β₀=1)
    - circulation: 环流需要独立环 (β₁>0)
    - global_consistency: 全局一致需要层粘合无障碍 (近似: β₁=0)

    ponytail: 用图论近似 Betti 数, 不是真实同调. 升级: 接 gudhi.
    ponytail: "拓扑许可动力学"定理对 Kuramoto 成立, 推广到任意 PDE 是假设.
    """
    sig = hodge_signature(nodes, edges)
    # β₀ = 连通分量数 = n_vertices - n_edges + beta1 (欧拉示性数反推)
    # 但更直接: β₀ = 连通分量数, 重新算
    parent = {n: n for n in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in edges:
        if a in parent and b in parent:
            union(a, b)

    beta0 = len({find(n) for n in nodes}) if nodes else 0
    beta1 = sig.beta1_approx

    if target_mode == "synchronization":
        # 同步需要 β₁ > 0 (拓扑环允许同步模式)
        if beta1 > 0:
            return TopologyPermit(
                permitted=True,
                reason=f"β₁={beta1} > 0, 拓扑环允许同步模式",
                beta0=beta0, beta1=beta1,
            )
        return TopologyPermit(
            permitted=False,
            reason=f"β₁=0, 无拓扑环, 同步模式不可能 (无论耦合强度)",
            beta0=beta0, beta1=beta1,
            required_change="引入环结构 (新增反馈边) 使 β₁ > 0",
        )

    if target_mode == "circulation":
        # 环流需要独立环
        if beta1 > 0:
            return TopologyPermit(
                permitted=True,
                reason=f"β₁={beta1} > 0, 独立环允许环流模式",
                beta0=beta0, beta1=beta1,
            )
        return TopologyPermit(
            permitted=False,
            reason=f"β₁=0, 无独立环, 环流不可能 (标准扩散模型已足够)",
            beta0=beta0, beta1=beta1,
            required_change="若物理确有环流, 需补反馈边; 否则用标准 Fick 扩散",
        )

    if target_mode == "percolation":
        # 渗流: β₀>1 (多连通, 部分渗流) 或 β₀=1 (全连通, 完全渗流)
        if beta0 >= 1:
            status = "全连通" if beta0 == 1 else f"{beta0} 个连通分量"
            return TopologyPermit(
                permitted=True,
                reason=f"β₀={beta0} ({status}), 渗流可能",
                beta0=beta0, beta1=beta1,
            )
        return TopologyPermit(
            permitted=False,
            reason="β₀=0, 无节点, 渗流无意义",
            beta0=beta0, beta1=beta1,
        )

    if target_mode == "global_consistency":
        # 全局一致: 层粘合 H¹=0, 近似用 β₁=0 (无拓扑障碍)
        if beta1 == 0:
            return TopologyPermit(
                permitted=True,
                reason=f"β₁=0, 无拓扑障碍, 局部模型可全局粘合 (H¹≈0)",
                beta0=beta0, beta1=beta1,
            )
        return TopologyPermit(
            permitted=False,
            reason=f"β₁={beta1} > 0, 存在拓扑障碍, 局部模型无法一致粘合 (H¹≠0)",
            beta0=beta0, beta1=beta1,
            required_change="修正局部模型使粘合条件一致, 或显式处理障碍",
        )

    # 不该到这里
    return TopologyPermit(
        permitted=False,
        reason=f"未知 target_mode: {target_mode}",
        beta0=beta0, beta1=beta1,
    )


# ── 层粘合障碍检测 ────────────────────────────────────────────

@dataclass
class GluingObstruction:
    """局部模型能否全局粘合的检测结果."""

    can_glue: bool
    reason: str
    # 障碍类型 (若不能粘合)
    obstruction: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "can_glue": self.can_glue,
            "reason": self.reason,
            "obstruction": self.obstruction,
        }


def gluing_obstruction(
    local_models: list[dict[str, Any]],
    overlap_pairs: list[tuple[int, int]],
) -> GluingObstruction:
    """检测局部模型能否全局粘合 (Čech H¹ 近似).

    local_models: 每个局部模型的描述 dict (含 'consistent_with' 字段
      列出它声称能与之粘合的其他模型 index)
    overlap_pairs: 实际有重叠的模型对 (i, j)

    近似判据 (ponytail: 不算真实 Čech 上同调):
    - 如果所有重叠对的 consistent_with 互相一致 → H¹=0, 可粘合
    - 若存在重叠对 (i,j) 但 i 没声明与 j 一致 → 障碍, H¹≠0

    升级: 接 sheaf cohomology 计算真实 H¹.
    """
    if not local_models:
        return GluingObstruction(can_glue=True, reason="无局部模型, 平凡可粘合")

    # 检查每个重叠对是否双向一致
    for i, j in overlap_pairs:
        if i >= len(local_models) or j >= len(local_models):
            continue
        mi = local_models[i]
        mj = local_models[j]
        ci = mi.get("consistent_with", [])
        cj = mj.get("consistent_with", [])
        if j not in ci or i not in cj:
            return GluingObstruction(
                can_glue=False,
                reason=f"模型 {i} 与 {j} 重叠但未双向声明一致 → 拓扑障碍",
                obstruction=f"H¹≠0: 对 ({i},{j}) 粘合条件不一致",
            )

    return GluingObstruction(
        can_glue=True,
        reason=f"{len(overlap_pairs)} 个重叠对全部双向一致, H¹≈0, 可全局粘合",
    )


# ── 综合诊断 (给 red_team / equivalence_auditor 用) ────────────

@dataclass
class TopologyDiagnosis:
    """一次完整拓扑诊断的结果."""

    family: FamilyVerdict
    closure: ClosureCheck
    signature: HodgeSignature | None = None
    permit: TopologyPermit | None = None
    gluing: GluingObstruction | None = None
    # 给 LLM 审查的文本提示
    advisory: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family.to_dict(),
            "closure": self.closure.to_dict(),
            "signature": self.signature.__dict__ if self.signature else None,
            "permit": self.permit.to_dict() if self.permit else None,
            "gluing": self.gluing.to_dict() if self.gluing else None,
            "advisory": self.advisory,
        }


def diagnose(
    interactions: list[set[str]],
    physical_connectivity: bool = False,
    need_topological_invariants: bool = False,
    need_orthogonal_decomposition: bool = False,
    nodes: list[str] | None = None,
    edges: list[tuple[str, str]] | None = None,
    target_mode: Literal["synchronization", "percolation", "circulation", "global_consistency"] | None = None,
    local_models: list[dict[str, Any]] | None = None,
    overlap_pairs: list[tuple[int, int]] | None = None,
) -> TopologyDiagnosis:
    """一次完整拓扑诊断: 族 + 闭包 + 签名 + 许可 + 粘合.

    所有参数可选, 传什么算什么. 给 red_team / equivalence_auditor 一站式调用.
    """
    fam = classify_system(
        interactions, need_topological_invariants, need_orthogonal_decomposition
    )
    clo = needs_downward_closure(interactions, physical_connectivity)
    sig = hodge_signature(nodes, edges) if nodes else None
    perm = topology_permits(nodes, edges, target_mode) if (nodes and target_mode) else None
    glu = gluing_obstruction(local_models, overlap_pairs) if local_models else None

    parts = [f"[族] {fam.family}: {fam.reason}"]
    parts.append(f"[闭包] {'需要' if clo.needs_closure else '不需要'}: {clo.reason}")
    if sig:
        parts.append(
            f"[签名] β₀≈{sig.n_vertices - sig.n_edges + sig.beta1_approx} "
            f"β₁≈{sig.beta1_approx} 度熵={sig.degree_entropy:.2f} "
            f"调和={'有' if sig.has_harmonic else '无'}"
        )
    if perm:
        parts.append(f"[许可] {'允许' if perm.permitted else '不允许'}: {perm.reason}")
    if glu:
        parts.append(f"[粘合] {'可' if glu.can_glue else '不可'}: {glu.reason}")

    return TopologyDiagnosis(
        family=fam,
        closure=clo,
        signature=sig,
        permit=perm,
        gluing=glu,
        advisory="\n".join(parts),
    )


# ── 自检 (ponytail: 非平凡逻辑留 runnable check) ──────────────


def _selfcheck() -> None:
    # 1. 化学反应 → combinatorial, 无闭包
    rxn = [{"C3S", "H2O", "C-S-H", "CH"}]
    v = classify_system(rxn)
    assert v.family == "combinatorial", f"反应应归 combinatorial, got {v.family}"
    c = needs_downward_closure(rxn, physical_connectivity=False)
    assert not c.needs_closure, "化学反应不需要闭包"

    # 2. 孔隙网络 → topological, 闭包自然
    pore = [{"p1", "p2", "p3"}]
    v2 = classify_system(pore, need_topological_invariants=True)
    assert v2.family == "topological", f"孔隙应归 topological, got {v2.family}"
    c2 = needs_downward_closure(pore, physical_connectivity=True)
    assert c2.needs_closure and c2.natural, "孔隙闭包应自然成立"

    # 3. Hodge 签名: 三角形 (有环) vs 树 (无环)
    tri = hodge_signature(["a", "b", "c"], [("a", "b"), ("b", "c"), ("a", "c")])
    assert tri.beta1_approx == 1, f"三角形 β₁ 应=1, got {tri.beta1_approx}"
    assert tri.has_harmonic, "三角形应有调和分量"
    tree = hodge_signature(["a", "b", "c"], [("a", "b"), ("b", "c")])
    assert tree.beta1_approx == 0, f"树 β₁ 应=0, got {tree.beta1_approx}"
    assert not tree.has_harmonic, "树不应有调和分量"

    # 4. 签名差异检测: 环 vs 树 → 不同
    diff, reason = tri.differs_from(tree)
    assert diff, "环与树拓扑不同, 应检出差异"
    assert "β₁" in reason

    # 5. 拓扑许可: 环允许同步, 树不允许
    p_sync_tri = topology_permits(["a", "b", "c"], [("a", "b"), ("b", "c"), ("a", "c")], "synchronization")
    assert p_sync_tri.permitted, "环应允许同步"
    p_sync_tree = topology_permits(["a", "b", "c"], [("a", "b"), ("b", "c")], "synchronization")
    assert not p_sync_tree.permitted, "树不应允许同步"
    assert "β₁=0" in p_sync_tree.reason

    # 6. 粘合障碍: 双向一致 → 可粘合
    models = [
        {"consistent_with": [1]},
        {"consistent_with": [0]},
    ]
    g = gluing_obstruction(models, [(0, 1)])
    assert g.can_glue, "双向一致应可粘合"
    models_bad = [
        {"consistent_with": []},
        {"consistent_with": [0]},
    ]
    g2 = gluing_obstruction(models_bad, [(0, 1)])
    assert not g2.can_glue, "单向一致应不可粘合"

    # 7. diagnose 综合
    d = diagnose(
        interactions=[{"a", "b", "c"}],
        physical_connectivity=True,
        nodes=["a", "b", "c"],
        edges=[("a", "b"), ("b", "c"), ("a", "c")],
        target_mode="synchronization",
    )
    assert d.permit and d.permit.permitted, "三角环应允许同步"
    assert "族" in d.advisory and "闭包" in d.advisory

    print("topology_lens selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
