"""相位级全序门 + 增量边重算 (Phase Total-Order Gate & Incremental Edge Recompute).

借鉴 Ghidra ``AnalysisPriority`` 与 ``Analyzer.added/removed`` 的两种工程思想, 把它落到
科研管线的**规划后置原语**上 (默认零行为变化, 由调用方显式启用):

1. **相位级全序门 (phase gate)**
   Ghidra 给每个 Analyzer 声明一个具名全序相位 (FORMAT→BLOCK→DISASSEMBLY→CODE→
   FUNCTION→REFERENCE→DATA→…), 越高优先级越"确定/基础", 越低越"投机/高阶"。相位决定
   结算次序 —— 基础相位先结算, 高阶相位后结算。本模块把匿名拓扑层映射到这个相位全序,
   使"层"获得语义身份 (不再只是拓扑索引), 供调度与重算引用。

2. **增量边重算 (incremental recompute)**
   Ghidra 不做"上游变了就整层重跑", 而是只对 ``removed/added`` 集合覆盖的**受影响子集**
   做局部重分析。本模块提供: 给定"某上游实验结果已变更/失效"的集合, 求 DAG 中**真正受影响
   的下游闭包子集**, 重算只落到这个子集, 而非整个相位层/全部后序。

红线 (与 P-A/P-B/P-C 一致, 不可逾越):
  - 本模块只做**调度/范围决策**, 不改动任何真实实验结果的数值 —— 重算推导出的"受影响
    子集"仍由调用方用真实 run 去执行, 本模块不伪造证据;
  - 受影响集是**确定性可达闭包** (DAG 边驱动的纯计算), 不做"值不值得研究"的质量判断;
  - 分配·默认相位映射缺省只把每个拓扑层当作自身唯一相位 (逐层), 不改变现有层义 ——
    启用相位命名只是"给层贴标签", 不改变层的结构。
"""
from __future__ import annotations

from typing import Any, Iterable


# ═══════════════════ 1. 相位级全序 (Phase Priority Total Order) ═══════════════════

# 科研域相位全序 (沿用 Ghidra"越高越基础/确定"的直觉自下而上排):
#   0 FORMAT_FORM       需求/问题定形 (最基础, 先结算)
#   1 DATA_ACQUISITION  数据/证据获取
#   2 HYPOTHESIS        假说生成
#   3 EXPERIMENT        实验执行
#   4 SYNTHESIS         综合/结论 (最投机, 后结算)
# 调用方可传自定义 "实验名 -> 相位索引" 映射; 缺省把每个拓扑层映射为递增的独一相位。
PHASE_NAMES: tuple[str, ...] = (
    "FORMAT_FORM", "DATA_ACQUISITION", "HYPOTHESIS",
    "EXPERIMENT", "SYNTHESIS",
)

# 缺省相位分配: 拓扑层 idx -> phase idx (逐层自 0 递增, 不改变层结构)
DEFAULT_PHASE_BY_LAYER: tuple[int, ...] = tuple(range(len(PHASE_NAMES)))


def assign_phases(
    layers: list[list[str]],
    phase_by_layer: dict[int, int] | Iterable[int] | None = None,
) -> dict[str, int]:
    """把拓扑层映射成相位, 返回 {实验名 -> 相位索引}.

    layers:          拓扑层划分 [[层0...], [层1...], ...].
    phase_by_layer:  层索引 -> 相位索引 的映射 (迭代器按层序配对亦可). 缺省逐层自增,
                     即把每层当作其自身相位 —— 不改变现有任何层义 (默认零行为变化).
                     相位索引越靠前 = 越基础/确定, 越靠后 = 越投机/高阶.
    """
    if phase_by_layer is None:
        pb: dict[int, int] = {i: min(i, len(PHASE_NAMES) - 1) for i in range(len(layers))}
    elif isinstance(phase_by_layer, dict):
        pb = {int(k): int(v) for k, v in phase_by_layer.items()}
    else:
        pb = {i: int(p) for i, p in enumerate(phase_by_layer)}
    mapping: dict[str, int] = {}
    for i, layer in enumerate(layers):
        phase = pb.get(i, min(i, len(PHASE_NAMES) - 1))
        for name in layer:
            mapping[name] = phase
    return mapping


def phase_total_order(phases: dict[str, int] | None = None) -> list[int]:
    """相位全序 (升序, 基础先): 实际出现的相位索引去重排序."""
    if not phases:
        return list(range(len(PHASE_NAMES)))
    return sorted(set(phases.values()))


# ═══════════════════ 2. 增量边重算 (Incremental Downstream Recompute) ═══════════════════

def downstream_closure(
    deps: list[tuple[str, str]],
    changed: Iterable[str],
) -> set[str]:
    """求 change 集合的下游闭包 (即增量重算的**受影响子集**).

    deps:    有向依赖边 [(A, B)] —— A 是 B 的上游 (B 依赖 A). 与 TaskDAG.dependencies 同序.
    changed: 结果已变更/失效的上游实验名集合.

    返回: 这些上游的**直接+传递下游** (不包含 changed 自身), 即"如果它们变了, 谁必须
          重新结算". 这是确定性可达闭包计算: 从 changed 出发沿依赖边向下 BFS, 不判质量.
    """
    adj: dict[str, list[str]] = {}
    for u, v in deps:
        adj.setdefault(u, []).append(v)
    changed = list(changed)
    affected: set[str] = set()
    stack = list(changed)
    while stack:
        node = stack.pop()
        for nxt in adj.get(node, []):
            if nxt not in affected:
                affected.add(nxt)
                stack.append(nxt)
    return affected


def affected_within(
    deps: list[tuple[str, str]],
    changed: Iterable[str],
    scope: Iterable[str],
) -> list[str]:
    """把下游闭包限制在某个执行范围内 (避免重算范围扩大到闭包外).

    返回 scope 内受 changed 影响的实验名 (保 scope 的顺序). 常用于: 只在当前相位层
    尚未结算的实验里重算, 不回溯已经结算的层.
    """
    affected = downstream_closure(deps, changed)
    return [s for s in scope if s in affected]


def recompute_plan(
    deps: list[tuple[str, str]],
    layers: list[list[str]],
    changed: Iterable[str],
    *,
    phase_by_layer: dict[int, int] | Iterable[int] | None = None,
) -> dict[str, Any]:
    """一键: 给定上游变更, 求"按相位/层范围应增量重算的子集".

    这是 contribution 的**单入口** —— 调度方拿它决定"哪些实验该重跑", 再自己去执行.
    返回纯调度决策 (不执行任何真实计算, 不伪造证据).

    deps:    有向依赖边 [(A, B)].
    layers:  拓扑层划分.
    changed: 结果已变更/失效的上游集合.
    phase_by_layer: 相位映射 (缺省逐层自增).

    返回:
      changed          输入的上游变更集.
      affected_set     下游闭包 (含跨层), 去重.
      recompute        建议重算子集: 受影响 且 位于本层后的实验 (不回溯已结算基础层).
      phases           {实验名 -> 相位索引}.
      total_order      相位全序.
      note             为什么这样重算 (审计/可证伪).
    """
    phases = assign_phases(layers, phase_by_layer)
    affected = downstream_closure(deps, changed)
    # 分层: 受影响但只重算"相对被变更上游相位更靠后"的实验 (Ghidra added/removed 只在
    # 低相位触发, 不再回溯已确定的基础相位). changed 自身不重列.
    layer_of: dict[str, int] = {}
    for i, layer in enumerate(layers):
        for name in layer:
            layer_of[name] = i
    recompute: list[str] = []
    for u in sorted(affected):
        if u in changed:
            continue
        # 任何受影响节点都在下游闭包里 —— 若它所在层 > 上游起始层至少一层, 排除同层
        u_layer = layer_of.get(u, 0)
        up_layer = min((layer_of.get(c, 0) for c in changed), default=0)
        if u_layer > up_layer:
            recompute.append(u)
    return {
        "changed": sorted(changed),
        "affected_set": sorted(affected),
        "recompute": recompute,
        "phases": phases,
        "total_order": phase_total_order(phases),
        "note": ("增量重算: 仅受影响且位于上游相位之后的下游子集; "
                 "范围由 DAG 边确定性推出, 不做质量判断, 不伪造证据."),
    }


__all__ = [
    "PHASE_NAMES", "DEFAULT_PHASE_BY_LAYER",
    "assign_phases", "phase_total_order",
    "downstream_closure", "affected_within", "recompute_plan",
]


# ── selfcheck ──────────────────────────────────────────────

if __name__ == "__main__":
    # DAG: A→B, A→C, B→D, C→D, D→E
    deps = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", "E")]
    layers = [["A"], ["B", "C"], ["D"], ["E"]]

    # 1. 缺省相位: 逐层自增 (0,1,2,3), 层义不变 (默认零行为变化)
    ph = assign_phases(layers)
    assert ph == {"A": 0, "B": 1, "C": 1, "D": 2, "E": 3}, ph
    assert phase_total_order(ph) == [0, 1, 2, 3], phase_total_order(ph)
    print("[ok] 缺省相位逐层自增, 层义不变:", ph)

    # 2. 自定义相位: A/B 数据基座(相位1), C 假说(相位2), D/E 综合(相位4)
    ph2 = assign_phases(layers, {0: 1, 1: 2, 2: 4, 3: 4})
    assert ph2["A"] == 1 and ph2["B"] == 2 and ph2["D"] == 4, ph2
    assert phase_total_order(ph2) == [1, 2, 4], phase_total_order(ph2)
    print("[ok] 自定义相位:", ph2, "全序:", phase_total_order(ph2))

    # 3. 下游闭包: B 变更 → 受影响的 D, E (传递), 不含 C (B 不分叉到 C)
    affB = downstream_closure(deps, ["B"])
    assert affB == {"D", "E"}, affB
    # D 变更 → E
    affD = downstream_closure(deps, ["D"])
    assert affD == {"E"}, affD
    # A 变更 → B,C,D,E (全下游, 不含 A 自身)
    affA = downstream_closure(deps, ["A"])
    assert affA == {"B", "C", "D", "E"}, affA
    print("[ok] 下游闭包: B→{D,E}, D→{E}, A→{B,C,D,E}")

    # 4. recompute_plan: C 变更 → 只重算 C 之后的层 (D,E), 不回算已结算 A/B
    rp = recompute_plan(deps, layers, ["C"])
    assert rp["affected_set"] == ["D", "E"], rp["affected_set"]
    assert rp["recompute"] == ["D", "E"], rp["recompute"]
    assert rp["note"], "note 应存在(审计面)"
    print("[ok] recompute_plan(C 变更) -> affected={D,E}, recompute={D,E}")

    # 5. B 变更同层 (B,C 同层, B→D→E): 只重算层2起 (D,E)
    rp2 = recompute_plan(deps, layers, ["B"])
    assert rp2["recompute"] == ["D", "E"], rp2["recompute"]
    print("[ok] recompute_plan(B 变更) -> recompute={D,E} (同层 C 不回溯)")

    # 6. no-change: changed 对不受其影响的节点 -> 空重算 (不误伤)
    rp3 = recompute_plan(deps, layers, ["E"])
    assert rp3["affected_set"] == [] and rp3["recompute"] == [], rp3
    print("[ok] recompute_plan(E 变更) -> affected 为空 (E 无下游)")

    # 7. 纯函数确定性/可证伪: 同一输入两次结果一致
    assert recompute_plan(deps, layers, ["A"]) == recompute_plan(deps, layers, ["A"])
    print("[phase_recompute] self-check OK (7/7)")