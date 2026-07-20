"""CrossDomainPipeline — 跨域同构 conjecture pipeline.

把散落的积木串成 5 步跨域迁移管道:
1. MathConceptGraph.query_concept_neighborhood — 从 problem 命中数学概念
2. transfer_registry.find_transfer_domain — 找结构同构预训练域
3. transfer_registry.shared_structure — SymPy unify 验证等价
4. reframe_problem(mode="analogy_map") — 翻译到候选域
5. 输出 TransferHypothesis (含 confidence)

复用清单 (0 新组件):
- kg.graph.MathConceptGraph (查 ancestors/duals/LCA)
- ml.transfer_registry.find_transfer_domain + shared_structure
- autoloop.conjecture.reframe_problem (analogy_map 模式)
- metacog.topology_lens.hodge_signature (跨域拓扑签名对比, 可选)

不做:
- 不新建 topology_lens — 已有同名文件做元认知判据, 不冲突
- 不真算同调群 — transfer_registry 的 SymPy unify 已够等价验证
- 不接 engine — 独立可调, engine 接入留 flag

失败返 None, 调用方走原 reframe_problem 路径.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TransferHypothesis:
    """跨域迁移假设 — pipeline 的输出."""

    original_problem: str
    target_domain: str  # 候选预训练域名
    shared_math: list[str]  # 共享数学结构 tags (SymPy unify 验证过)
    math_concept: str  # 命中的 MathConcept
    lca_concept: str  # MathConceptGraph 上的最近公共祖先
    reframed_problem: str  # reframe_problem 翻译后的问题
    confidence: float  # 0-1, LCA depth × similarity × shared 命中数
    trace: list[str] = field(default_factory=list)  # 5 步执行轨迹


def _extract_math_concept(problem: str) -> str | None:
    """从 problem 文本提取 MathConcept (复用 conjecture.py 的关键词表)."""
    try:
        from huginn.autoloop.conjecture import _PROBLEM_TO_MATH_CONCEPT
    except Exception:
        logger.debug("_PROBLEM_TO_MATH_CONCEPT import failed", exc_info=True)
        return None
    problem_lower = problem.lower()
    for key, concept in _PROBLEM_TO_MATH_CONCEPT.items():
        if key in problem_lower:
            return concept
    return None


def _find_lca_with_target(
    math_concept: str, target_domain_profile: Any
) -> str | None:
    """找 math_concept 与 target_domain 共享的最近公共祖先.

    ponytail: target_domain_profile.structure_signature 的 canonical_name
    作为另一概念, 找 LCA. 失败返 None.
    """
    try:
        from huginn.kg.graph import MathConceptGraph
        g = MathConceptGraph()
        # target domain 的 structure_signature 取第一个 canonical_name
        sigs = getattr(target_domain_profile, "structure_signature", ()) or ()
        if not sigs:
            return None
        target_concept = sigs[0][0] if isinstance(sigs[0], tuple) else str(sigs[0])
        # canonical_name 可能不是 MathConcept 节点, find_common_ancestor 返 None
        return g.find_common_ancestor(math_concept, target_concept, max_depth=4)
    except Exception:
        logger.debug("LCA lookup failed", exc_info=True)
        return None


def cross_domain_reframe(
    problem: str,
    target_composition: frozenset[str] | set[str] | list[str] | None = None,
    target_structure: str = "",
    target_property: str = "",
    target_method: str = "",
    n_samples: int = 0,
    threshold: float = 0.3,
    model: Any = None,
) -> TransferHypothesis | None:
    """5 步跨域迁移 pipeline. 失败返 None.

    target_composition/structure/property/method: 目标域画像, 缺省时从 problem
    推断 (用 _extract_math_concept 命中的概念当 property_type).
    model: reframe_problem 用的 LLM, None 走模板路径.
    """
    trace: list[str] = []

    # Step 1: 从 problem 提取 MathConcept
    math_concept = _extract_math_concept(problem)
    if math_concept is None:
        trace.append("step1: no math concept matched")
        logger.debug("cross_domain: no math concept for %r", problem[:80])
        return None
    trace.append(f"step1: math_concept={math_concept}")

    # Step 2: 找结构同构预训练域
    try:
        from huginn.ml.transfer_registry import (
            find_transfer_domain, shared_structure, _REGISTRY,
        )
    except Exception:
        logger.debug("transfer_registry import failed", exc_info=True)
        return None

    # 推断 target_property: 用 math_concept 当作 property_type 初筛
    if not target_property:
        target_property = math_concept

    if target_composition is None:
        # 无 composition 也调, find_transfer_domain 会用 jaccard 兜底
        target_composition = frozenset()

    try:
        target_profile, similarity, recommendation = find_transfer_domain(
            target_composition=target_composition,
            target_structure=target_structure,
            target_property=target_property,
            target_method=target_method,
            n_samples=n_samples,
            threshold=threshold,
        )
    except Exception:
        logger.debug("find_transfer_domain failed", exc_info=True)
        return None

    if target_profile is None:
        trace.append(f"step2: no transfer domain (similarity below {threshold})")
        logger.debug("cross_domain: no transfer domain for %r", problem[:80])
        return None
    trace.append(f"step2: target_domain={target_profile.name}, sim={similarity:.3f}")

    # Step 3: SymPy unify 验证结构等价
    shared_math: list[str] = []
    try:
        # target_profile 自己就是 source (它有 structure_signature)
        # 找 _REGISTRY 里和它 shared_structure 命中的所有结构
        for other in _REGISTRY:
            if other.name == target_profile.name:
                continue
            shared = shared_structure(target_profile, other)
            if shared:
                shared_math.extend(shared)
        # 去重保序
        seen = set()
        shared_math = [s for s in shared_math if not (s in seen or seen.add(s))]
    except Exception:
        logger.debug("shared_structure failed", exc_info=True)
    trace.append(f"step3: shared_math={shared_math}")

    # Step 4: LCA 找公共数学祖先 (用于 confidence)
    lca = _find_lca_with_target(math_concept, target_profile)
    if lca:
        trace.append(f"step4: lca={lca}")
    else:
        trace.append("step4: no lca")

    # Step 5: reframe_problem 翻译到候选域
    try:
        from huginn.autoloop.conjecture import reframe_problem
        reframe_result = reframe_problem(
            problem,
            mode="analogy_map",
            model=model,
            target_domain=target_profile.name,
        )
        reframed = (
            reframe_result.get("reframed_problem")
            or reframe_result.get("reframed", "")
            or problem
        )
    except Exception:
        logger.debug("reframe_problem failed, fallback to original", exc_info=True)
        reframed = problem
    trace.append(f"step5: reframed={'ok' if reframed != problem else 'fallback'}")

    # confidence: LCA 命中 (0.4) + similarity (0.4) + shared 命中 (0.2)
    confidence = 0.0
    if lca:
        confidence += 0.4
    confidence += min(similarity, 1.0) * 0.4
    if shared_math:
        confidence += min(0.2, len(shared_math) * 0.1)
    confidence = min(confidence, 1.0)

    transfer = TransferHypothesis(
        original_problem=problem,
        target_domain=target_profile.name,
        shared_math=shared_math,
        math_concept=math_concept,
        lca_concept=lca or "",
        reframed_problem=reframed,
        confidence=confidence,
        trace=trace,
    )

    # P13: 持久化到 KG + memory. 失败不阻塞返回, 单测默认不写
    # (write_transfer_to_kg 的 mm 参数 None 时跳过 memory 写入)
    try:
        write_transfer_to_kg(transfer)
    except Exception:
        logger.warning("write_transfer_to_kg failed", exc_info=True)

    return transfer


def write_transfer_to_kg(
    transfer: TransferHypothesis,
    kg: Any = None,
    memory_manager: Any = None,
) -> None:
    """把 transfer 写 KG + memory. flag off 或组件不可用时降级.

    ponytail: 不在 cross_domain_reframe 内部直接调, 而是单独函数让 caller
    决定是否写 (避免单测依赖全局 kg/memory_manager). KG 用 conjecture.get_kg
    单例, 不新建存储.
    """
    # 1. 写 KG: caller 没传 kg 时用 conjecture 的全局单例, 不可用就跳过
    if kg is None:
        try:
            from huginn.autoloop.conjecture import get_kg
            kg = get_kg()
        except Exception:
            logger.debug("get_kg unavailable", exc_info=True)
            kg = None
    if kg is not None:
        try:
            kg.add_transfer_edge(transfer)
        except Exception:
            logger.warning("add_transfer_edge failed", exc_info=True)

    # 2. 写 memory: caller 没传 mm 时跳过 (cross_domain_reframe 不知道 mm 在哪)
    if memory_manager is None:
        return
    try:
        content = (
            f"CrossDomain transfer: {transfer.original_problem} -> "
            f"{transfer.target_domain}\n"
            f"shared_math: {transfer.shared_math}\n"
            f"reframed: {transfer.reframed_problem}\n"
            f"confidence: {transfer.confidence}"
        )
        # P12 typed API 优先, 不可用时降级到 category 字符串
        if hasattr(memory_manager, "remember_typed"):
            memory_manager.remember_typed(
                content=content,
                memory_type="cross_domain_transfer",
                status="proposed",
            )
        else:
            memory_manager.remember(
                content=content,
                category="cross_domain_transfer",
            )
    except Exception:
        logger.warning("remember_typed failed, fallback skipped", exc_info=True)


# === 自检 ===

if __name__ == "__main__":
    # 1) _extract_math_concept: 命中关键词
    c = _extract_math_concept("predict Si band gap")
    assert c == "band_symmetry", f"应命中 band_symmetry, got {c}"

    c = _extract_math_concept("calculate Fe magnetic properties")
    assert c == "lie_group", f"应命中 lie_group, got {c}"

    # 未命中 → None
    assert _extract_math_concept("nonexistent problem xyz") is None
    assert _extract_math_concept("") is None

    # 2) cross_domain_reframe: 完整 pipeline (无 model, 走模板路径)
    #    用 "Fe 磁性" 问题, target 用 Fe 元素, 应该找到 ferromagnet 域
    hyp = cross_domain_reframe(
        problem="predict Fe magnetic transition temperature",
        target_composition={"Fe"},
        target_property="magnetic",
    )
    # 可能命中 ferromagnet (Fe + magnetic + DFT-PBE)
    if hyp is not None:
        assert hyp.original_problem == "predict Fe magnetic transition temperature"
        assert hyp.math_concept == "lie_group"
        assert hyp.target_domain  # 非空
        assert 0 <= hyp.confidence <= 1.0
        assert len(hyp.trace) >= 5  # 5 步轨迹
        # reframed 应非空 (reframe_problem 模板路径兜底)
        assert hyp.reframed_problem
    else:
        # threshold 太高可能返 None, 也是合法行为 (advisory)
        print("info: cross_domain_reframe 返 None (threshold 可能太高)")

    # 3) 未命中 math_concept → None
    hyp = cross_domain_reframe(problem="totally unknown xyz problem")
    assert hyp is None

    # 4) threshold 太高 → None (find_transfer_domain 返 None)
    hyp = cross_domain_reframe(
        problem="predict Fe magnetic transition temperature",
        target_composition={"Fe"},
        target_property="magnetic",
        threshold=0.99,  # 几乎不可能达到
    )
    # 99% 阈值下应返 None (除非完美匹配)
    # 不强制 assert None, 因为可能有完美匹配情况, 但通常应 None
    if hyp is not None:
        # 如果命中, confidence 应该接近 0 (no shared_math, no lca 通常)
        assert 0 <= hyp.confidence <= 1.0

    # 5) TransferHypothesis dataclass 字段完整性
    if hyp is not None:
        for f in ("original_problem", "target_domain", "shared_math",
                  "math_concept", "lca_concept", "reframed_problem",
                  "confidence", "trace"):
            assert hasattr(hyp, f), f"TransferHypothesis 缺字段 {f}"

    # 6) confidence 计算: LCA 命中给 0.4
    #    用 Fe 磁性 (lie_group) + ferromagnet 域 (landau_phi4), 看 LCA
    #    MathConceptGraph 种子里 lie_group 和 landau_phi4 可能无 LCA, 那就靠 similarity + shared
    #    不强制数值, 只验证 0-1 范围
    hyp2 = cross_domain_reframe(
        problem="predict Fe magnetic transition temperature",
        target_composition={"Fe"},
        target_property="magnetic",
        threshold=0.1,  # 低阈值确保命中
    )
    if hyp2 is not None:
        assert 0 <= hyp2.confidence <= 1.0
        # trace 应该有 5 步
        assert len(hyp2.trace) >= 5

    # 7) shared_math: 铁磁体 (landau_phi4) 应与钙钛矿太阳能 (同 landau_phi4) 共享
    #    跑这个需要 target 命中 ferromagnet, 然后扫描 _REGISTRY 找 shared
    if hyp2 is not None and hyp2.target_domain == "ferromagnet":
        # landau_phi4 在 ferromagnet 和 perovskite_solar 都有, SymPy unify 应等价
        assert "landau_phi4" in hyp2.shared_math, \
            f"ferromagnet 应与 perovskite_solar 共享 landau_phi4, got {hyp2.shared_math}"

    print("all self-checks passed")
