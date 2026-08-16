"""Tests for the scientific-claim hypergraph layer.

Covers:
  - ClaimHypergraph: n-ary AND/OR dependency, challenge propagation, citation
    cycles, persistence.
  - ProjectKnowledgeGraph claim nodes: add_claim / get_claim / query_claim_deps.
  - ClaimAuditor: register_claim double-write, sheaf conflict detection,
    challenge propagation + KG status writeback, citation-cycle audit,
    ingest_findings pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from huginn.kg import ClaimAuditor, ClaimHypergraph, ProjectKnowledgeGraph
from huginn.kg.entities import EntityType, Relation
from huginn.kg.hypergraph import (
    IMPACT_RECHECK,
    IMPACT_ROOT,
    IMPACT_WEAKENED,
    STATUS_ACCEPTED,
    STATUS_CONTESTED,
    STATUS_RECHECK,
    STATUS_SUPERSEDED,
)


class TestClaimHypergraph:
    def test_add_claim_and_roundtrip(self, tmp_path: Path) -> None:
        chg = ClaimHypergraph(tmp_path)
        he = chg.add_claim(
            "结论X", ["假设A", "Fact:B"], mode="AND",
            condition="低温区间", evidence_strength=0.9, source="doi:1",
        )
        assert he.mode == "AND" and he.condition == "低温区间"
        assert set(he.premises) == {"假设A", "Fact:B"}

        # 同一 (conclusion, mode, condition) 幂等更新, 不重复堆边
        chg.add_claim(
            "结论X", ["假设A", "Fact:B"], mode="AND",
            condition="低温区间", evidence_strength=0.95, source="doi:2",
        )
        edges = chg.get_edges_for_conclusion("结论X")
        assert len(edges) == 1
        assert edges[0].evidence_strength == pytest.approx(0.95)

        # 持久化 round-trip
        chg.save()
        chg2 = ClaimHypergraph(tmp_path)
        assert len(chg2._edges) == 1
        assert chg2.get_edges_for_conclusion("结论X")[0].premises == ["假设A", "Fact:B"]

    def test_impact_propagation_and_or(self, tmp_path: Path) -> None:
        chg = ClaimHypergraph(tmp_path)
        chg.add_claim("结论X", ["假设A", "Fact:B"], mode="AND", source="doi:1")
        chg.add_claim("结论Y", ["Fact:B", "Fact:B2"], mode="OR", source="doi:2")
        chg.add_claim("结论Z", ["结论X"], mode="AND", source="doi:3")

        # 挑战源头 A → X (AND recheck) → Z (AND recheck)
        r = chg.impact_propagation("假设A")
        assert r["root"]["claim"] == "假设A" and r["root"]["impact"] == IMPACT_ROOT
        by_claim = {a["claim"]: a["impact"] for a in r["affected"]}
        assert by_claim["结论X"] == IMPACT_RECHECK
        assert by_claim["结论Z"] == IMPACT_RECHECK
        assert "结论Y" not in by_claim  # Y 的 AND 前提里没有 A

        # 挑战 Fact:B → X (AND recheck) + Y (OR weakened)
        r2 = chg.impact_propagation("Fact:B")
        by_claim2 = {a["claim"]: a["impact"] for a in r2["affected"]}
        assert by_claim2["结论X"] == IMPACT_RECHECK
        assert by_claim2["结论Y"] == IMPACT_WEAKENED

    def test_citation_cycles(self, tmp_path: Path) -> None:
        chg = ClaimHypergraph(tmp_path)
        chg.add_claim("环P", ["环Q"], mode="AND")
        chg.add_claim("环Q", ["环P"], mode="AND")
        cycles = chg.citation_cycles()
        assert len(cycles) == 1
        assert set(cycles[0]) == {"环P", "环Q"}

    def test_status_transition(self, tmp_path: Path) -> None:
        chg = ClaimHypergraph(tmp_path)
        chg.add_claim("结论X", ["假设A"], mode="AND")
        chg.update_status("结论X", STATUS_CONTESTED)
        assert chg.get_edges_for_conclusion("结论X")[0].status == STATUS_CONTESTED
        assert chg.stats()["statuses"][STATUS_CONTESTED] == 1

    def test_invalid_mode_and_empty_premises_rejected(self, tmp_path: Path) -> None:
        chg = ClaimHypergraph(tmp_path)
        with pytest.raises(ValueError):
            chg.add_claim("X", ["A"], mode="NAND")
        with pytest.raises(ValueError):
            chg.add_claim("X", [], mode="AND")


class TestProjectKnowledgeGraphClaims:
    def test_add_claim_creates_node_and_edges(self, tmp_path: Path) -> None:
        kg = ProjectKnowledgeGraph(tmp_path)
        eid = kg.add_claim(
            "材料X呈 altermagnetism", literature_id="lit_1",
            premises=["假设A", "Fact:B"], evidence_strength=0.9,
        )
        assert eid == "Claim:材料X呈 altermagnetism"
        node = kg.get_claim("材料X呈 altermagnetism")
        assert node is not None and node["type"] == EntityType.CLAIM
        assert node["evidence_strength"] == pytest.approx(0.9)

        # 前提 → 结论 depends_on 边
        deps = kg.query_claim_deps("材料X呈 altermagnetism", direction="backward")
        dep_relations = {e["relation"] for e in deps["edges"]}
        assert Relation.DEPENDS_ON in dep_relations

    def test_add_claim_repeat_increases_mentions(self, tmp_path: Path) -> None:
        kg = ProjectKnowledgeGraph(tmp_path)
        kg.add_claim("C", literature_id="lit_1", evidence_strength=0.5)
        kg.add_claim("C", literature_id="lit_2", evidence_strength=0.8)
        node = kg.get_claim("C")
        assert node["mentions"] == 2
        assert node["evidence_strength"] == pytest.approx(0.8)  # 取更强证据

    def test_query_claim_deps_missing(self, tmp_path: Path) -> None:
        kg = ProjectKnowledgeGraph(tmp_path)
        res = kg.query_claim_deps("不存在的结论")
        assert res["nodes"] == [] and res["edges"] == []


class TestClaimAuditor:
    def test_register_claim_double_write(self, tmp_path: Path) -> None:
        auditor = ClaimAuditor(tmp_path)
        rec = auditor.register_claim(
            "结论X", literature_id="doi:1", premises=["假设A", "Fact:B"],
            mode="AND", evidence_strength=0.9,
        )
        assert rec["kg_node_id"] == "Claim:结论X"
        assert rec["hyperedge"]["mode"] == "AND"
        # KG 与超图都写入了
        assert auditor.kg.get_claim("结论X") is not None
        assert len(auditor.hypergraph.get_edges_for_conclusion("结论X")) == 1

    def test_detect_conflict_via_sheaf(self, tmp_path: Path) -> None:
        auditor = ClaimAuditor(tmp_path)
        # 数值冲突 → H¹ > 0
        dc = auditor.detect_conflict(
            [{"Tc": 100.0, "magnetic_order": "altermagnetism"},
             {"Tc": 50.0, "magnetic_order": "altermagnetism"}]
        )
        assert dc["consistent"] is False and dc["h1"] is not None and dc["h1"] > 0

        # 一致 → H¹ = 0
        dc_ok = auditor.detect_conflict(
            [{"Tc": 100.0}, {"Tc": 100.0}]
        )
        assert dc_ok["consistent"] is True and dc_ok["h1"] == 0

    def test_challenge_propagates_and_writes_back_kg(self, tmp_path: Path) -> None:
        auditor = ClaimAuditor(tmp_path)
        auditor.register_claim("结论X", premises=["假设A"], mode="AND")
        auditor.register_claim("结论Z", premises=["结论X"], mode="AND")

        rep = auditor.challenge("假设A")
        assert rep["root"]["impact"] == IMPACT_ROOT
        by_claim = {a["claim"]: a["impact"] for a in rep["affected"]}
        assert by_claim["结论X"] == IMPACT_RECHECK
        assert by_claim["结论Z"] == IMPACT_RECHECK

        # KG 状态回写
        assert auditor.kg.get_claim("结论X")["status"] == STATUS_RECHECK
        assert auditor.kg.get_claim("结论Z")["status"] == STATUS_RECHECK
        assert rep["kg_status_updated"]  # 非空

    def test_challenge_skips_superseded(self, tmp_path: Path) -> None:
        auditor = ClaimAuditor(tmp_path)
        auditor.register_claim("结论X", premises=["假设A"], mode="AND")
        # 手动把 X 标记为 superseded
        auditor.kg._graph.nodes["Claim:结论X"]["status"] = STATUS_SUPERSEDED
        rep = auditor.challenge("假设A")
        # 被取代的结论不再被降级为 recheck
        assert "结论X" not in rep["kg_status_updated"]

    def test_audit_citation_cycles(self, tmp_path: Path) -> None:
        auditor = ClaimAuditor(tmp_path)
        auditor.register_claim("环P", premises=["环Q"], mode="AND")
        auditor.register_claim("环Q", premises=["环P"], mode="AND")
        res = auditor.audit_citation_cycles()
        assert res["n_cycles"] >= 1
        assert any({"环P", "环Q"} <= set(c) for c in res["cycles"])

    def test_ingest_findings_marks_contested_on_conflict(self, tmp_path: Path) -> None:
        auditor = ClaimAuditor(tmp_path)
        res = auditor.ingest_findings(
            [
                {"claim": "新实验观测到 Tc=80K", "Tc": 80.0,
                 "premises": ["样本S"], "mode": "AND", "evidence_strength": 0.7},
            ],
            core={"Tc": 100.0},
        )
        assert res["registered"] == 1
        assert res["conflict"]["consistent"] is False
        # 冲突时登记状态为 contested
        assert res["claims"][0]["hyperedge"]["status"] == STATUS_CONTESTED


class TestRagBridgeContestedTagging:
    def test_match_contested_claims(self, tmp_path: Path) -> None:
        from huginn.perception.doc_types import InformationPackage
        from huginn.perception.rag_bridge import RAGBridge

        auditor = ClaimAuditor(tmp_path)
        auditor.register_claim("该材料的 Tc 存在争议", premises=["A"], mode="AND")
        auditor.challenge("A")  # 让该结论进入 recheck

        pkg = InformationPackage(
            package_id="p1",
            claims=[{"conclusion": "该材料的 Tc 存在争议", "metric": "Tc"}],
        )
        bridge = RAGBridge(kb=None, auditor=auditor)
        hits = bridge._match_contested_claims(pkg)
        assert any("Tc 存在争议" in h for h in hits)

        # 无 auditor → 空
        assert RAGBridge(kb=None)._match_contested_claims(pkg) == []


class TestContextBuilderContestedNote:
    def test_contested_kb_note(self, tmp_path: Path) -> None:
        from huginn.context_builder import ContextBuilder

        auditor = ClaimAuditor(tmp_path)
        auditor.register_claim("某材料低温下呈 altermagnetism", premises=["A"], mode="AND")
        auditor.challenge("A")

        builder = ContextBuilder(memory_manager=None, workspace=str(tmp_path))
        note = builder._contested_kb_note("材料 altermagnetism")
        assert "Knowledge Audit Note" in note
        assert "altermagnetism" in note

        # 无争议 query → 空
        assert builder._contested_kb_note("完全无关的词xyz") == ""
