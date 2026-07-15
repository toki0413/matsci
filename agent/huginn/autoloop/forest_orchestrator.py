"""森林编排器 — 随机森林启发的多 engine 并行探索.

随机森林的核心不是"更多树", 而是"有差异的树" + "独立训练后聚合".
映射到研究探索:
- 多棵树 → 多个并行 AutoloopEngine (各自独立 HypothesisGraph)
- Bagging → 每个 engine 从不同模态角度出发 (VISReg 切片思想)
- 投票/平均 → DS 证据合成 (DempsterShaferCombiner)
- 交叉授粉 → 成熟 engine 间交换 frontier 假设 (比随机森林更强)

当前实现: 方案 B (独立跑完后 DS 合成). 交叉授粉 (方案 A) 留作升级口.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from huginn.autoloop.engine import AutoloopEngine, AutoloopResult
from huginn.autoloop.hypothesis_loop import HypothesisGraph, HypothesisNode
from huginn.autoloop.phase_gate import DempsterShaferCombiner

logger = logging.getLogger(__name__)


# VISReg 启发: 每个 engine 从不同模态角度出发, 对应"特征随机选择"
# 材料科学的正交模态: 电子结构 / 晶格动力学 / 热力学 / 缺陷化学
_MODAL_PERSPECTIVES = [
    "从电子结构角度分析",
    "从晶格动力学角度分析",
    "从热力学稳定性角度分析",
    "从缺陷化学角度分析",
    "从输运性质角度分析",
]


@dataclass
class TreeResult:
    """单棵树 (engine) 的结果."""

    tree_id: str
    objective: str
    result: AutoloopResult | None
    hypothesis_count: int
    supported_count: int
    refuted_count: int
    # 该 engine 的置信度三元组 (m_pass, m_fail, m_unc)
    ds_mass: tuple[float, float, float]
    # 假设节点快照 (含 status / evidence), 供森林层合并用.
    # ponytail: 拷贝一份, 避免 engine 被 GC 后图也丢.
    nodes: list[HypothesisNode] = field(default_factory=list)


@dataclass
class ForestResult:
    """森林聚合结果."""

    objective: str
    trees: list[TreeResult]
    # DS 合成后的全局置信度
    combined_mass: tuple[float, float, float]
    # 森林多样性: 不同 engine 的假设重叠度 (0=完全独立, 1=完全相同)
    diversity: float
    # 聚合结论
    consensus: str
    passed: bool
    # 合并后的假设图: 把各棵树的 supported/refuted 节点汇到一张图,
    # evidence 里标 tree_id 来源. 下游 engine 可直接接续探索.
    merged_graph: HypothesisGraph = field(default_factory=HypothesisGraph)
    # 给下游 engine _speculator_hint 的回流水印.
    speculator_hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "n_trees": len(self.trees),
            "combined": {
                "pass": round(self.combined_mass[0], 3),
                "fail": round(self.combined_mass[1], 3),
                "uncertain": round(self.combined_mass[2], 3),
            },
            "diversity": round(self.diversity, 3),
            "consensus": self.consensus,
            "passed": self.passed,
            "trees": [
                {
                    "tree_id": t.tree_id,
                    "hypothesis_count": t.hypothesis_count,
                    "supported": t.supported_count,
                    "refuted": t.refuted_count,
                    "ds_mass": {
                        "pass": round(t.ds_mass[0], 3),
                        "fail": round(t.ds_mass[1], 3),
                        "uncertain": round(t.ds_mass[2], 3),
                    },
                }
                for t in self.trees
            ],
        }


class ForestOrchestrator:
    """随机森林启发的多 engine 并行探索编排器.

    ponytail: 当前实现方案 B (独立跑完后 DS 合成).
    升级路径 (方案 A): 在 engine.run 的每轮迭代后调 _cross_pollinate,
    把成熟 engine 的 frontier 假设传给其他 engine. 需要 engine 支持
    中途注入假设 (当前 run() 是阻塞的, 需改为 yield 式).
    """

    def __init__(
        self,
        n_trees: int = 3,
        ds_threshold: float = 0.6,
    ):
        self.n_trees = max(2, min(n_trees, 5))  # ponytail: 上限 5, 防资源爆炸
        self.ds_threshold = ds_threshold

    async def run_forest(
        self,
        objective: str,
        max_iterations: int = 15,
        engine_factory: Any = None,
    ) -> ForestResult:
        """并行跑 N 棵树, DS 合成结果.

        每个 engine 从不同模态角度出发 (VISReg 切片), 独立探索后聚合.
        """
        perspectives = _MODAL_PERSPECTIVES[: self.n_trees]
        # 每个 engine 的 objective 加模态前缀, 强制从不同角度切入
        objectives = [
            f"{p}: {objective}" for p in perspectives
        ]

        # 并行创建 + 跑 N 个 engine
        tasks = []
        for i, obj in enumerate(objectives):
            tasks.append(self._run_single_tree(f"tree_{i}", obj, max_iterations, engine_factory))

        tree_results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常
        valid_results: list[TreeResult] = []
        for i, tr in enumerate(tree_results):
            if isinstance(tr, Exception):
                logger.warning("tree_%d failed: %s", i, tr)
                valid_results.append(TreeResult(
                    tree_id=f"tree_{i}",
                    objective=objectives[i],
                    result=None,
                    hypothesis_count=0,
                    supported_count=0,
                    refuted_count=0,
                    ds_mass=(0.0, 0.0, 1.0),  # 全不确定
                ))
            else:
                valid_results.append(tr)

        # DS 合成
        masses = [t.ds_mass for t in valid_results]
        combined = DempsterShaferCombiner.combine(masses)

        # 多样性: 假设重叠度 (Jaccard 相似度的补)
        diversity = self._compute_diversity(valid_results)

        # 聚合结论
        m_pass, m_fail, m_unc = combined
        if m_pass >= self.ds_threshold:
            consensus = f"森林共识: 支持 (m_pass={m_pass:.2f}, {len(valid_results)} 棵树)"
            passed = True
        elif m_fail >= self.ds_threshold:
            consensus = f"森林共识: 反驳 (m_fail={m_fail:.2f}, {len(valid_results)} 棵树)"
            passed = False
        else:
            consensus = f"森林共识: 不确定 (m_unc={m_unc:.2f}, 需更多证据)"
            passed = False

        return ForestResult(
            objective=objective,
            trees=valid_results,
            combined_mass=combined,
            diversity=diversity,
            consensus=consensus,
            passed=passed,
            merged_graph=self._merge_trees_into_graph(valid_results),
            speculator_hint=self._build_speculator_hint(
                consensus, combined, diversity, valid_results
            ),
        )

    def _merge_trees_into_graph(
        self, trees: list[TreeResult]
    ) -> HypothesisGraph:
        """把各棵树的 supported/refuted 节点汇到一张新图.

        evidence 里标 tree_id 来源, 让下游能看出某条假设是被哪棵树验证的.
        ponytail: 只搬 supported/refuted, untested 节点不带结论不搬.
        升级路径: 用语义相似度合并重复假设 (当前按 statement 去重).
        """
        merged = HypothesisGraph()
        seen: set[str] = set()  # 按 statement 去重, 避免重复 add
        for t in trees:
            for node in t.nodes:
                if node.status not in ("supported", "refuted"):
                    continue
                if node.statement in seen:
                    continue
                seen.add(node.statement)
                try:
                    nid = merged.add_hypothesis(
                        statement=node.statement,
                        rationale=f"[from {t.tree_id}] {node.rationale}",
                        testable_prediction=node.testable_prediction,
                    )
                    if nid is None:
                        continue
                    ev = dict(node.evidence)
                    ev["tree_id"] = t.tree_id
                    if node.status == "supported":
                        merged.support(nid, ev)
                    else:
                        merged.refute(nid, ev)
                except Exception:
                    logger.debug("merge node from %s failed", t.tree_id, exc_info=True)
        return merged

    @staticmethod
    def _build_speculator_hint(
        consensus: str,
        combined: tuple[float, float, float],
        diversity: float,
        trees: list[TreeResult],
    ) -> str:
        """给下游 engine 的回流水印: consensus + DS 质量 + 各树结论摘要."""
        m_pass, m_fail, m_unc = combined
        parts = [
            f"[forest consensus] {consensus}",
            f"[DS mass] pass={m_pass:.2f} fail={m_fail:.2f} unc={m_unc:.2f}",
            f"[diversity] {diversity:.2f} (0=独立, 1=重复)",
        ]
        for t in trees:
            parts.append(
                f"[{t.tree_id}] supported={t.supported_count} "
                f"refuted={t.refuted_count} m_pass={t.ds_mass[0]:.2f}"
            )
        return "\n".join(parts)

    async def _run_single_tree(
        self,
        tree_id: str,
        objective: str,
        max_iterations: int,
        engine_factory: Any,
    ) -> TreeResult:
        """跑单棵树 (一个 engine)."""
        engine = engine_factory() if engine_factory else AutoloopEngine()
        engine.forest_id = tree_id  # 标记归属森林

        result = await engine.run(objective, max_iterations=max_iterations)

        # 从 hypothesis_graph 提取统计
        graph = engine.hypothesis_graph
        all_nodes = graph.all_nodes()
        supported = graph.supported()
        refuted = graph.refuted()

        # 计算 DS 三元组: supported → m_pass, refuted → m_fail, 其余 → m_unc
        total = len(all_nodes) or 1
        m_pass = len(supported) / total
        m_fail = len(refuted) / total
        m_unc = 1.0 - m_pass - m_fail

        return TreeResult(
            tree_id=tree_id,
            objective=objective,
            result=result,
            hypothesis_count=len(all_nodes),
            supported_count=len(supported),
            refuted_count=len(refuted),
            ds_mass=(m_pass, m_fail, m_unc),
            nodes=list(all_nodes),
        )

    def _compute_diversity(self, trees: list[TreeResult]) -> float:
        """计算森林多样性: 假设重叠度的补.

        ponytail: 用 supported 假设的 Jaccard 相似度.
        0 = 完全独立 (理想), 1 = 完全相同 (退化成单树).
        升级路径: 用假设的语义相似度而非二值匹配.
        """
        if len(trees) < 2:
            return 0.0
        # 没有 result 的树跳过
        valid = [t for t in trees if t.result is not None]
        if len(valid) < 2:
            return 0.0
        # ponytail: 用 supported_count 的差异度作为多样性代理
        # 真正的多样性应该比假设内容, 但那需要语义比较, 留作升级
        counts = [t.supported_count for t in valid]
        if max(counts) == 0:
            return 0.0
        return 1.0 - (min(counts) / max(counts))

    async def _cross_pollinate(
        self,
        engines: list[AutoloopEngine],
    ) -> int:
        """交叉授粉: 把成熟 engine 的 frontier 假设传给其他 engine.

        升级路径 (方案 A): 在 engine.run 的每轮迭代后调用.
        当前 run() 是阻塞的, 这个方法预留但不调用.

        返回交换的假设数.
        """
        # 检查各 engine 的连通分量成熟度
        n_exchanged = 0
        for i, src in enumerate(engines):
            src_graph = src.hypothesis_graph
            src_frontier = src_graph.frontier()
            if not src_frontier:
                continue
            for j, dst in enumerate(engines):
                if i == j:
                    continue
                dst_graph = dst.hypothesis_graph
                # 把 src 的 frontier 假设加到 dst 的图中
                for hyp in src_frontier[:3]:  # ponytail: 每次最多传 3 个
                    try:
                        dst_graph.add_hypothesis(
                            statement=hyp.statement,
                            rationale=f"[cross-pollinated from {src.forest_id}] {hyp.rationale}",
                            testable_prediction=hyp.testable_prediction,
                            parent_id=None,  # 作为新根节点接入
                        )
                        n_exchanged += 1
                    except Exception:
                        logger.debug("cross-pollinate failed", exc_info=True)
        return n_exchanged


# ── 自检 (ponytail: 非平凡逻辑留一个可运行检查) ──────────────────


def _selfcheck():
    """森林编排器自检 — 不需要真 LLM, 只测 DS 合成 + 多样性."""
    orch = ForestOrchestrator(n_trees=3, ds_threshold=0.6)

    # 模拟 3 棵树的结果
    trees = [
        TreeResult("t0", "obj", None, 5, 3, 1, (0.6, 0.2, 0.2)),
        TreeResult("t1", "obj", None, 4, 2, 1, (0.5, 0.25, 0.25)),
        TreeResult("t2", "obj", None, 6, 1, 3, (0.17, 0.5, 0.33)),
    ]
    masses = [t.ds_mass for t in trees]
    combined = DempsterShaferCombiner.combine(masses)
    assert combined[0] > 0, "DS 合成 m_pass 不应为 0"
    assert abs(sum(combined) - 1.0) < 0.01, f"DS 合成应归一化, got {sum(combined)}"

    diversity = orch._compute_diversity(trees)
    assert 0.0 <= diversity <= 1.0, f"多样性应在 [0,1], got {diversity}"

    # 验证: 高一致 → 高 m_pass
    strong_trees = [
        TreeResult("t0", "obj", None, 5, 4, 0, (0.8, 0.0, 0.2)),
        TreeResult("t1", "obj", None, 5, 3, 1, (0.6, 0.2, 0.2)),
    ]
    strong_combined = DempsterShaferCombiner.combine([t.ds_mass for t in strong_trees])
    assert strong_combined[0] > 0.7, f"高一致应高 m_pass, got {strong_combined[0]}"

    # 验证: 高冲突 → m_fail 上升或 m_unc 上升
    conflict_trees = [
        TreeResult("t0", "obj", None, 5, 4, 0, (0.8, 0.0, 0.2)),
        TreeResult("t1", "obj", None, 5, 0, 4, (0.0, 0.8, 0.2)),
    ]
    conflict_combined = DempsterShaferCombiner.combine([t.ds_mass for t in conflict_trees])
    # 高冲突时 DS 的 K 很大, 归一化后 m_unc 应显著
    assert conflict_combined[2] > 0.3 or conflict_combined[1] > 0.3, \
        f"高冲突应升高 m_unc 或 m_fail, got {conflict_combined}"

    # 回流检查: merged_graph 应收下 supported/refuted 节点, speculator_hint 非空
    from huginn.autoloop.hypothesis_loop import HypothesisNode
    merge_trees = [
        TreeResult(
            "t0", "obj", None, 2, 1, 1, (0.6, 0.4, 0.0),
            nodes=[
                HypothesisNode(id="a", statement="H_supported", status="supported"),
                HypothesisNode(id="b", statement="H_refuted", status="refuted"),
            ],
        ),
        TreeResult(
            "t1", "obj", None, 1, 1, 0, (0.9, 0.1, 0.0),
            nodes=[
                HypothesisNode(id="c", statement="H_supported", status="supported"),  # 重复, 应被去重
            ],
        ),
    ]
    merged = orch._merge_trees_into_graph(merge_trees)
    assert len(merged.all_nodes()) == 2, f"去重后应 2 个, got {len(merged.all_nodes())}"
    assert len(merged.supported()) == 1 and len(merged.refuted()) == 1, \
        "supported/refuted 各 1"
    hint = orch._build_speculator_hint("test consensus", (0.6, 0.4, 0.0), 0.5, merge_trees)
    assert "forest consensus" in hint and "DS mass" in hint, "hint 缺字段"

    print("PASS: forest_orchestrator DS 合成 + 多样性 + 回流")


if __name__ == "__main__":
    _selfcheck()
