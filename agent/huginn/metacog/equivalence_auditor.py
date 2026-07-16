"""等价性审计 agent — 检测 "换名归约" 伪进展.

核心机制: 每个探索 agent 返回的 "发现" 必须经过审计, 检查它是否真的
推进了问题, 还是把问题换了个名字.

为什么需要: prompt 里最锋利的一句——"一条最终到达与原猜想强度等价的引理
的路线, 除非它能为该引理提供真正新的证明, 否则不算接近完成."

材料科学里的等价性陷阱:
- 性质归约: "解决稳定性" → 归约为 "形成能预测" (后者仍难)
- 表述换名: "新机制" → 只是 CALPHAD 外推换了变量名
- 伪相关: "结构-性质映射" → 实际同源变量, 非因果
- 基准循环: "超越 SOTA" → 用同分布基准, 无外推证据
- 维度隐藏: "降维后预测" → 信息丢失未补偿

两种模式:
- rule-based (默认): 关键词模式匹配已知陷阱
- LLM-enhanced (model 传入): 用 LLM 做更深的语义判断, 失败降级到规则
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)


Verdict = Literal["advances", "equivalent_renaming", "undetermined"]


@dataclass
class EquivalenceVerdict:
    """等价性审计结果."""

    verdict: Verdict
    # 若 equivalent_renaming: 指出等价于什么
    reduction_target: str = ""
    # 若 advances: 仍缺什么机制才算完整发现
    missing_mechanism: str = ""
    # 具体证据 (引理/方程/反例描述), 拒绝状态报告
    evidence: list[str] = field(default_factory=list)
    # 命中的陷阱类别
    trap_category: str = ""

    @property
    def is_equivalent_renaming(self) -> bool:
        return self.verdict == "equivalent_renaming"

    @property
    def is_advancement(self) -> bool:
        return self.verdict == "advances"

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "reduction_target": self.reduction_target,
            "missing_mechanism": self.missing_mechanism,
            "evidence": list(self.evidence),
            "trap_category": self.trap_category,
        }


# 已知等价性陷阱的规则模式
# ponytail: 用关键词对 + 阈值, 不用复杂 NLP. LLM 增强负责语义判断.
_TRAP_PATTERNS: list[dict[str, Any]] = [
    {
        "category": "property_reduction",
        "description": "性质归约: 把难问题归约为另一个未解问题",
        # candidate 提到目标性质, 但 reduction_chain 出现归约目标
        "candidate_markers": ["稳定性", "stability", "稳定性预测"],
        "reduction_markers": ["形成能", "formation energy", "convex hull"],
        "target": "formation_energy_prediction",
    },
    {
        "category": "renaming",
        "description": "表述换名: 同一方法换了变量名",
        "candidate_markers": ["新机制", "new mechanism", "novel approach"],
        "reduction_markers": ["calphad", "外推", "extrapolat", "substitution"],
        "target": "calphad_extrapolation",
    },
    {
        "category": "pseudo_correlation",
        "description": "伪相关: 同源变量伪装成因果映射",
        "candidate_markers": ["映射", "mapping", "结构-性质", "structure-property"],
        "reduction_markers": ["同源", "same source", "合成条件", "synthesis"],
        "target": "spurious_correlation",
    },
    {
        "category": "benchmark_circularity",
        "description": "基准循环: 同分布基准伪装成 SOTA 突破",
        "candidate_markers": ["超越", "sota", "state-of-the-art", "超越基准"],
        "reduction_markers": ["同分布", "in-distribution", "iid", "同源 benchmark"],
        "target": "in_distribution_benchmark",
    },
    {
        "category": "dimension_hiding",
        "description": "维度隐藏: 降维丢失物理约束未补偿",
        "candidate_markers": ["降维", "dimension reduction", "pca", "autoencoder"],
        "reduction_markers": ["物理约束", "physics constraint", "守恒", "conservation"],
        "target": "constraint_loss",
    },
]


class EquivalenceAuditor:
    """等价性审计 agent.

    用法:
        auditor = EquivalenceAuditor()
        verdict = auditor.audit(
            candidate_finding="我们解决了稳定性预测问题",
            original_problem="预测材料稳定性",
            reduction_chain="通过形成能预测间接得到稳定性",
        )
        if verdict.is_equivalent_renaming:
            # 不计入进展, 不更新 current_preferred_hypothesis
    """

    def __init__(self, model: Any | None = None) -> None:
        self._model = model

    def audit(
        self,
        candidate_finding: str,
        original_problem: str,
        reduction_chain: str = "",
    ) -> EquivalenceVerdict:
        """审计一个候选发现是否为换名归约."""
        # 规则先跑, LLM 增强作为补充
        rule_verdict = self._rule_based_audit(
            candidate_finding, original_problem, reduction_chain
        )

        if self._model is not None and self._is_real_model():
            try:
                llm_verdict = self._llm_audit(
                    candidate_finding, original_problem, reduction_chain
                )
                # LLM 判 equivalent_renaming 优先 (更敏感)
                if llm_verdict.is_equivalent_renaming and not rule_verdict.is_equivalent_renaming:
                    return llm_verdict
                # 否则规则优先 (确定性高)
                return rule_verdict if rule_verdict.verdict != "undetermined" else llm_verdict
            except Exception:
                logger.debug("LLM audit failed, falling back to rules", exc_info=True)

        return rule_verdict

    def _rule_based_audit(
        self,
        candidate: str,
        problem: str,
        reduction: str,
    ) -> EquivalenceVerdict:
        """规则审计: 扫描已知陷阱模式."""
        cand_lower = candidate.lower()
        prob_lower = problem.lower()
        red_lower = (reduction or "").lower()

        for pattern in _TRAP_PATTERNS:
            cand_hit = any(m.lower() in cand_lower for m in pattern["candidate_markers"])
            red_hit = any(m.lower() in red_lower for m in pattern["reduction_markers"])

            if cand_hit and red_hit:
                return EquivalenceVerdict(
                    verdict="equivalent_renaming",
                    reduction_target=pattern["target"],
                    trap_category=pattern["category"],
                    evidence=[pattern["description"]],
                    missing_mechanism=(
                        f"需要为 {pattern['target']} 提供真正新的机制, "
                        "而不是归约到它"
                    ),
                )

        # 没命中陷阱, 但也没法确认为进展 → undetermined
        # (规则无法判断"真的推进了", 只有 LLM 能判断)
        if candidate.strip() and problem.strip():
            return EquivalenceVerdict(
                verdict="undetermined",
                missing_mechanism="规则无法确认是否真推进, 需 LLM 增强或人工审查",
            )
        return EquivalenceVerdict(verdict="undetermined", missing_mechanism="输入不足")

    def _is_real_model(self) -> bool:
        """检测是不是 MagicMock (测试注入的)."""
        return not hasattr(self._model, "_mock_name")

    def _llm_audit(
        self,
        candidate: str,
        problem: str,
        reduction: str,
    ) -> EquivalenceVerdict:
        """LLM 增强审计. 失败抛异常, 调用方 try/except 降级."""
        from langchain_core.messages import HumanMessage, SystemMessage

        prompt = (
            f"原始问题: {problem}\n"
            f"候选发现: {candidate}\n"
            f"归约链: {reduction or '(无)'}\n\n"
            "判断这个候选发现是否为'换名归约'——把问题换了个名字但没真正推进.\n"
            "输出 JSON: {\"verdict\": \"advances|equivalent_renaming|undetermined\", "
            "\"reduction_target\": \"若换名归约, 等价于什么\", "
            "\"missing_mechanism\": \"若 advances, 还缺什么\", "
            "\"evidence\": [\"具体引理/方程/反例\"]}\n"
            "拒绝状态报告和模糊乐观, 只要具体证据."
        )
        messages = [
            SystemMessage(content=(
                "你是等价性审计员. 检查候选发现是否真的推进了问题, "
                "还是把问题换了个名字. 材料科学常见陷阱: "
                "性质归约 / 表述换名 / 伪相关 / 基准循环 / 维度隐藏."
            )),
            HumanMessage(content=prompt),
        ]
        resp = self._model.invoke(messages)
        text = str(resp.content).strip()
        return self._parse_llm_verdict(text)

    @staticmethod
    def _parse_llm_verdict(text: str) -> EquivalenceVerdict:
        """解析 LLM 返回的 JSON. 解析失败返回 undetermined."""
        try:
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text)
            v = str(data.get("verdict", "undetermined"))
            if v not in ("advances", "equivalent_renaming", "undetermined"):
                v = "undetermined"
            return EquivalenceVerdict(
                verdict=v,  # type: ignore[arg-type]
                reduction_target=str(data.get("reduction_target", "")),
                missing_mechanism=str(data.get("missing_mechanism", "")),
                evidence=[str(e) for e in data.get("evidence", [])],
                trap_category=str(data.get("trap_category", "")),
            )
        except (json.JSONDecodeError, TypeError, AttributeError):
            return EquivalenceVerdict(verdict="undetermined", missing_mechanism="LLM 解析失败")

    # ── 拓扑审计 (高阶网络视角) ──────────────────────────────────

    def audit_topology(
        self,
        candidate_nodes: list[str],
        candidate_edges: list[tuple[str, str]],
        original_nodes: list[str],
        original_edges: list[tuple[str, str]],
    ) -> EquivalenceVerdict:
        """拓扑审计: 用 Hodge 签名判断候选发现与原问题的证据网络是否拓扑等价.

        高阶网络视角: 两个假设的证据网络拓扑不同 → 真不同 (非换名).
        拓扑相同 → 可能等价, 需更深判断.

        判据:
        - β₁ 不同 → 拓扑不同 → advances (真推进)
        - 度熵差异 > 0.3 → 证据分布不同 → advances
        - 调和分量存在性不同 → 拓扑环结构不同 → advances
        - 全相同 → equivalent_renaming (可能换名)

        ponytail: 用图论近似 Betti 数, 不是真实同调. 升级: 接 gudhi.
        """
        from huginn.metacog.topology_lens import hodge_signature

        sig_cand = hodge_signature(candidate_nodes, candidate_edges)
        sig_orig = hodge_signature(original_nodes, original_edges)

        is_diff, reason = sig_cand.differs_from(sig_orig)
        if is_diff:
            return EquivalenceVerdict(
                verdict="advances",
                missing_mechanism=f"拓扑结构不同: {reason}",
                evidence=[f"候选 β₁={sig_cand.beta1_approx}, 原 β₁={sig_orig.beta1_approx}"],
                trap_category="topological_distinction",
            )
        return EquivalenceVerdict(
            verdict="equivalent_renaming",
            reduction_target="same_topology",
            evidence=[f"拓扑签名相同 (β₁={sig_cand.beta1_approx}, 度熵≈{sig_cand.degree_entropy:.2f})"],
            trap_category="topological_equivalence",
            missing_mechanism="拓扑签名相同, 可能是换名归约, 需更深判断",
        )


# ── 自检 ─────────────────────────────────────────────────────────

def _selfcheck() -> None:
    auditor = EquivalenceAuditor()

    # 1. 性质归约: candidate 说稳定性, reduction 说形成能 → 等价归约
    v1 = auditor.audit(
        candidate_finding="我们解决了稳定性预测问题",
        original_problem="预测材料稳定性",
        reduction_chain="通过形成能预测间接得到稳定性",
    )
    assert v1.is_equivalent_renaming, f"应判为换名归约, got {v1.verdict}"
    assert v1.reduction_target == "formation_energy_prediction"
    assert v1.trap_category == "property_reduction"

    # 2. 基准循环: candidate 说 SOTA, reduction 说同分布基准
    v2 = auditor.audit(
        candidate_finding="我们的模型超越了 SOTA",
        original_problem="预测形成能",
        reduction_chain="在同分布 in-distribution benchmark 上测试",
    )
    assert v2.is_equivalent_renaming
    assert v2.trap_category == "benchmark_circularity"

    # 3. 没陷阱也没明确进展 → undetermined
    v3 = auditor.audit(
        candidate_finding="我们用了 GP 拟合",
        original_problem="预测形成能",
        reduction_chain="",
    )
    assert v3.verdict == "undetermined"

    # 4. 空输入 → undetermined
    v4 = auditor.audit("", "", "")
    assert v4.verdict == "undetermined"

    print("equivalence_auditor selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
