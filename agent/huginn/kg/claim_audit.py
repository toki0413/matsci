"""Claim audit layer — 挑战检测 + 传播 + 自指环 (debug 人类科学认知的编排层).

把三个已存在的底层能力串成一个"知识审计"入口:
  - ClaimHypergraph (huginn/kg/hypergraph.py): n 元 AND/OR 结论-前提超边,
    impact_propagation 挑战传播, citation_cycles 自指环检测.
  - ProjectKnowledgeGraph.add_claim (huginn/kg/graph.py): 把结论登记为一等
    KG 节点 (depends_on / supports_claim 边), 供 hybrid_retrieve 召回.
  - sheaf_cohomology.compute_H1 (huginn/metacog/sheaf_cohomology.py): 严格
    Čech H¹ 粘合障碍, 检测多源证据的局部/全局不一致 (挑战源头).

流程 (用户愿景: "像 debug 软件一样 debug 人类已有的科学认知"):
  1. ingest_findings: 一组"来源 → 结论"evidence 进来 → 先 sheaf H¹ 检测
     是否有跨源冲突 (challenge detection), 冲突的结论标记 contested.
  2. register_claim: 把结论 + 前提写进二元 KG + 超图 (AND/OR 语义).
  3. challenge: 某结论被新证据挑战 → 超图前向传播, 计算受影响下游, 并回写
     每个受影响结论的 KG 状态 (recheck/weakened) — 让现有检索也能看到.
  4. audit_citation_cycles: 找出"引用链惯性"的自指环 (β₁ 调和分量), 建议 recheck.

设计约束:
  - 纯编排层: 不重复实现任何底层逻辑, 只做薄胶水 + 状态回写.
  - 全组件可选: 缺 KB / 缺 KG / 缺 sheaf 时降级为 advisory 输出, 不 raise.
  - 依赖方向: kg → metacog (sheaf) 单向; claim_audit 是唯一同时触碰两者的模块.
  - ponytail: 无新依赖; 规模上限沿用底层 (超图 < 10K 边 / KG < 500 节点).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from huginn.kg.graph import ProjectKnowledgeGraph
from huginn.kg.hypergraph import (
    IMPACT_RECHECK,
    IMPACT_ROOT,
    IMPACT_WEAKENED,
    STATUS_ACCEPTED,
    STATUS_CONTESTED,
    STATUS_RECHECK,
    STATUS_SUPERSEDED,
    ClaimHypergraph,
)

if TYPE_CHECKING:
    from huginn.kg.hypergraph import ClaimHyperedge

# sheaf H¹ 阈值: >0 即检测到不一致. 语义冲突 (pairwise) 在 Layer 2 计 1,
# 纯 twist (triple) 在 Layer 1. 用 >0 判定, 不用绝对值.
_H1_CONTEST_THRESHOLD = 0


def _canonical_claim(claim_text: str) -> str:
    """结论文本规范化: 去空白 + 折叠空格, 供超图/KG 节点去重."""
    return " ".join(claim_text.strip().split())


class ClaimAuditor:
    """知识审计编排器 — 结论抽取/登记/挑战传播/自指环检测的统一入口.

    Usage::

        auditor = ClaimAuditor(workspace)
        auditor.register_claim("材料X在低温下呈 altermagnetism",
                               literature_id="doi:...", premises=[...],
                               mode="AND", evidence_strength=0.9)
        report = auditor.challenge("材料X在低温下呈 altermagnetism")
        cycles = auditor.audit_citation_cycles()
    """

    def __init__(
        self,
        root: Path | str,
        kg: ProjectKnowledgeGraph | None = None,
        hypergraph: ClaimHypergraph | None = None,
    ) -> None:
        self.root = Path(root)
        self.kg = kg if kg is not None else ProjectKnowledgeGraph(self.root)
        self.hypergraph = (
            hypergraph if hypergraph is not None else ClaimHypergraph(self.root)
        )

    # ── 1. 挑战检测 (sheaf H¹) ──────────────────────────────────────

    def detect_conflict(
        self,
        findings: list[dict[str, Any] | str],
        core: dict[str, Any] | str | None = None,
    ) -> dict[str, Any]:
        """对多来源 evidence 计算 Čech H¹, 判定是否有挑战信号.

        findings: 每个来源的 claim 数据 (dict: key->value 或 str).
        core: 任务/基线结论 (可选), 作为 open cover 的 core source.

        返回 {consistent, h1, contested_claims: [...]}. sheaf 不可用时
        (ImportError/numpy 缺失) 降级为 "sheaf_unavailable", 不 raise.
        """
        try:
            from huginn.metacog.sheaf_cohomology import (
                build_sheaf_from_findings,
                compute_H1,
            )

            sheaf = build_sheaf_from_findings(
                core if core is not None else {}, findings
            )
            h1 = int(compute_H1(sheaf))
        except Exception as exc:  # pragma: no cover - 依赖缺失降级
            return {
                "consistent": None,
                "h1": None,
                "contested_claims": [],
                "error": f"sheaf_unavailable: {type(exc).__name__}: {exc}",
            }

        # Layer 2 restriction failure 暴露了具体哪对 source 冲突 — 这里只
        # 输出整体一致性与涉及数, 细粒度定位交给调用方 (literature/auto_ingest).
        contested = [] if h1 <= _H1_CONTEST_THRESHOLD else ["<multi-source-conflict>"]
        return {
            "consistent": h1 <= _H1_CONTEST_THRESHOLD,
            "h1": h1,
            "contested_claims": contested,
        }

    # ── 2. 结论登记 (KG + 超图双写) ─────────────────────────────────

    def register_claim(
        self,
        claim_text: str,
        *,
        literature_id: str = "",
        premises: list[str] | None = None,
        mode: str = "AND",
        condition: str = "",
        evidence_strength: float = 0.5,
        source: str = "claim_audit",
        status: str = STATUS_ACCEPTED,
    ) -> dict[str, Any]:
        """把一条结论登记进二元 KG 与超图, 返回合并后的记录.

        先写 KG (hybrid_retrieve 能召回), 再写超图 (AND/OR + 传播语义).
        超图侧对同一 (conclusion, mode, condition) 幂等更新, 不会重复堆边.
        """
        claim_text = _canonical_claim(claim_text)
        premises = [_canonical_claim(p) for p in (premises or []) if p.strip()]

        kg_node_id = self.kg.add_claim(
            claim_text,
            literature_id=literature_id,
            premises=premises,
            evidence_strength=evidence_strength,
            condition=condition,
            status=status,
            source=source,
        )
        hyperedge: ClaimHyperedge = self.hypergraph.add_claim(
            claim_text,
            premises,
            mode=mode,
            condition=condition,
            evidence_strength=evidence_strength,
            source=source or literature_id or "auto",
            status=status,
        )
        self.hypergraph.save()
        self.kg.save()
        return {
            "claim": claim_text,
            "kg_node_id": kg_node_id,
            "hyperedge": {
                "mode": hyperedge.mode,
                "condition": hyperedge.condition,
                "status": hyperedge.status,
                "evidence_strength": hyperedge.evidence_strength,
                "n_premises": len(hyperedge.premises),
            },
        }

    # ── 3. 挑战传播 (超图前向 + KG 状态回写) ───────────────────────

    def challenge(
        self, claim_text: str, max_hops: int = 10
    ) -> dict[str, Any]:
        """挑战一条结论, 沿超边前向传播, 并把受影响结论的状态回写 KG.

        回写规则 (与 impact_propagation 一致):
          - root (被挑战源头): status → recheck (若还没被 superseded)
          - AND 下游 (recheck): KG 节点 status → recheck
          - OR 下游 (weakened): KG 节点 status → contested (削弱但保留)
          回写是尽力而为 — KG 里不存在的节点跳过, 不 raise.
        """
        claim_text = _canonical_claim(claim_text)
        report = self.hypergraph.impact_propagation(claim_text, max_hops=max_hops)

        # 回写 KG 节点状态, 让现有检索/可视化能看到挑战结果
        status_map: dict[str, str] = {}
        if report.get("root"):
            status_map[claim_text] = STATUS_RECHECK
        for a in report.get("affected", []):
            status_map[a["claim"]] = (
                STATUS_RECHECK
                if a["impact"] == IMPACT_RECHECK
                else STATUS_CONTESTED
            )

        updated = []
        for cid, st in status_map.items():
            cid_node = f"Claim:{cid}"
            if cid_node not in self.kg._graph:
                continue
            node = self.kg._graph.nodes[cid_node]
            if node.get("status") == STATUS_SUPERSEDED:
                continue  # 已被更权威结论取代的, 不再降级
            if node.get("status") != st:
                node["status"] = st
                updated.append(cid)
        if updated:
            self.kg.save()

        # 超图侧: 让被挑战的结论本身进入 recheck (供 stats/循环审计可见)
        self.hypergraph.update_status(claim_text, STATUS_RECHECK)
        # 受影响下游同样同步超图边状态 (recheck/contested) — 否则 contested
        # 标签 (rag_bridge / context_builder 读的是超图边状态) 会读到过期值.
        for a in report.get("affected", []):
            target = (
                STATUS_RECHECK
                if a["impact"] == IMPACT_RECHECK
                else STATUS_CONTESTED
            )
            self.hypergraph.update_status(a["claim"], target)

        return {
            "challenged": claim_text,
            "root": report.get("root"),
            "affected": report.get("affected", []),
            "kg_status_updated": updated,
            "hops": report.get("hops", 0),
        }

    # ── 4. 自指环审计 (引用链惯性) ─────────────────────────────────

    def audit_citation_cycles(self) -> dict[str, Any]:
        """列出超图里的自指环 (强连通分量 > 1 节点).

        返回 {cycles: [[...], ...], n_cycles, advice}: 环上的结论互为前提,
        是 Hodge 分解的"调和分量" — 建议标记 recheck (可能只是引用链惯性,
        而非独立证据支撑).
        """
        cycles = self.hypergraph.citation_cycles()
        flagged: list[str] = []
        for cyc in cycles:
            for c in cyc:
                if self.kg.get_claim(c) is not None:
                    flagged.append(c)
        if flagged:
            for c in flagged:
                node = self.kg.get_claim(c)
                if node is not None and node.get("status") != STATUS_SUPERSEDED:
                    self.kg._graph.nodes[f"Claim:{c}"]["status"] = STATUS_RECHECK
            self.kg.save()
        return {
            "cycles": cycles,
            "n_cycles": len(cycles),
            "flagged_claims": flagged,
            "advice": (
                "环上结论互为依赖且无外部锚点 — 标记 recheck, 建议寻找环外 "
                "独立证据或源头假设复核."
            ),
        }

    # ── 顶层流水线: findings 进来 → 冲突检测 + 登记 ────────────────

    def ingest_findings(
        self,
        findings: list[dict[str, Any]],
        *,
        core: dict[str, Any] | str | None = None,
        source_prefix: str = "audit",
        default_status: str = STATUS_ACCEPTED,
    ) -> dict[str, Any]:
        """批量接入文献/蒸馏 findings, 执行完整审计流程.

        每个 finding 形如 {"claim": ..., "premises": [...], "mode": "AND",
        "evidence_strength": 0.8, "literature_id": "doi:...", "source": ...}.

        流程:
          1. 若 findings 是跨源 evidence → 先 detect_conflict (挑战检测),
             有冲突的结论用 contested 状态登记.
          2. 逐条 register_claim.
          3. 返回冲突信号 + 登记统计.

        返回 {"conflict": {...}, "registered": N, "claims": [...]}.
        """
        conflict = self.detect_conflict(findings, core=core)
        if conflict.get("error"):
            logger = __import__("logging").getLogger(__name__)
            logger.debug("claim audit conflict detection degraded: %s", conflict["error"])

        claims = []
        for i, f in enumerate(findings):
            claim_text = _canonical_claim(str(f.get("claim", "")).strip())
            if not claim_text:
                continue
            status = (
                default_status
                if not conflict.get("contested_claims") or conflict.get("consistent")
                else STATUS_CONTESTED
            )
            rec = self.register_claim(
                claim_text,
                literature_id=str(f.get("literature_id", "") or ""),
                premises=[
                    _canonical_claim(str(p))
                    for p in (f.get("premises") or [])
                    if str(p).strip()
                ],
                mode=str(f.get("mode", "AND")),
                condition=str(f.get("condition", "") or ""),
                evidence_strength=float(f.get("evidence_strength", 0.5)),
                source=str(f.get("source", "") or f"{source_prefix}:{i}"),
                status=status,
            )
            claims.append(rec)

        return {
            "conflict": {
                "consistent": conflict.get("consistent"),
                "h1": conflict.get("h1"),
                "n_contested": len(conflict.get("contested_claims", [])),
            },
            "registered": len(claims),
            "claims": claims,
        }

    def stats(self) -> dict[str, Any]:
        return {"hypergraph": self.hypergraph.stats(), "kg": self.kg.stats()}


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        auditor = ClaimAuditor(tmp)

        # 1. 无冲突登记 (AND 链: X 依赖 A+B; Z 依赖 X)
        r = auditor.register_claim(
            "材料X在低温下呈 altermagnetism",
            literature_id="doi:1",
            premises=["假设A", "Fact:B"],
            mode="AND",
            condition="低温区间",
            evidence_strength=0.9,
        )
        assert r["kg_node_id"] == "Claim:材料X在低温下呈 altermagnetism", r
        assert r["hyperedge"]["mode"] == "AND"
        assert r["hyperedge"]["n_premises"] == 2
        print("1. register_claim 双写 OK")

        auditor.register_claim(
            "下游结论Z成立", premises=["材料X在低温下呈 altermagnetism"],
            mode="AND", source="doi:3",
        )

        # 2. sheaf 冲突检测: 同 key 不同值 → H¹ > 0
        dc = auditor.detect_conflict(
            [
                {"Tc": 100.0, "magnetic_order": "altermagnetism"},
                {"Tc": 50.0, "magnetic_order": "altermagnetism"},
            ]
        )
        assert dc["consistent"] is False and dc["h1"] > 0, dc
        print("2. detect_conflict (数值冲突) OK")

        # 3. 挑战源头 A → X (AND recheck) → Z (AND recheck)
        rep = auditor.challenge("假设A")
        assert rep["root"]["impact"] == IMPACT_ROOT, rep
        by_claim = {a["claim"]: a["impact"] for a in rep["affected"]}
        assert by_claim["材料X在低温下呈 altermagnetism"] == IMPACT_RECHECK, by_claim
        assert by_claim["下游结论Z成立"] == IMPACT_RECHECK, by_claim
        # KG 状态回写
        x_node = auditor.kg.get_claim("材料X在低温下呈 altermagnetism")
        assert x_node["status"] == "recheck", x_node
        print("3. challenge 传播 + KG 回写 OK")

        # 4. 自指环审计
        auditor.register_claim(
            "环结论P", premises=["环结论Q"], mode="AND", source="doi:x",
        )
        auditor.register_claim(
            "环结论Q", premises=["环结论P"], mode="AND", source="doi:y",
        )
        cyc = auditor.audit_citation_cycles()
        assert cyc["n_cycles"] >= 1, cyc
        assert any("环结论P" in c and "环结论Q" in c for c in cyc["cycles"]), cyc
        print("4. 自指环审计 OK")

        # 5. ingest_findings 顶层流水线 (跨源冲突 → contested 登记)
        res = auditor.ingest_findings(
            [
                {"claim": "新实验观测到 Tc=80K", "Tc": 80.0,
                 "premises": ["样本制备S"], "mode": "AND",
                 "evidence_strength": 0.7, "literature_id": "doi:new"},
            ],
            core={"Tc": 100.0},
        )
        assert res["registered"] == 1, res
        assert res["conflict"]["consistent"] is False, res
        assert res["claims"][0]["hyperedge"]["status"] == "contested", res
        print("5. ingest_findings 顶层流水线 OK")

        print("all claim-audit checks passed")
