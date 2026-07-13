"""R5 Hypothesis loop 测试 — 假设图 + 失败驱动修正.

覆盖: 节点增删 / 状态转移 (support/refute/supersede) / derive 边 /
      refine_failed (模板 + LLM mock) / 衍生链查询 / 序列化 / 异常.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from huginn.autoloop.hypothesis_loop import (
    HypothesisEdge,
    HypothesisGraph,
    HypothesisGraphError,
    HypothesisNode,
)


# ── 节点基本操作 ─────────────────────────────────────────────────────────────


class TestAddHypothesis:
    def test_add_returns_id(self):
        g = HypothesisGraph()
        hid = g.add_hypothesis("如果掺杂增加, 带隙减小")
        assert hid.startswith("h_")
        node = g.get(hid)
        assert node.statement == "如果掺杂增加, 带隙减小"
        assert node.status == "untested"
        assert node.created_at != ""

    def test_add_empty_statement_raises(self):
        g = HypothesisGraph()
        with pytest.raises(HypothesisGraphError, match="不能为空"):
            g.add_hypothesis("   ")

    def test_add_with_parent_creates_derive_edge(self):
        g = HypothesisGraph()
        h1 = g.add_hypothesis("原始假设")
        h2 = g.add_hypothesis("衍生假设", parent_id=h1)
        edges = g.edges()
        derive_edges = [e for e in edges if e.edge_type == "derive"]
        assert len(derive_edges) == 1
        assert derive_edges[0].from_id == h1
        assert derive_edges[0].to_id == h2
        assert g.get(h2).parent_id == h1

    def test_add_with_missing_parent_raises(self):
        g = HypothesisGraph()
        with pytest.raises(HypothesisGraphError, match="不存在"):
            g.add_hypothesis("x", parent_id="h_nope")

    def test_add_with_missing_parent_no_orphan(self):
        """失败时不留孤儿节点 (code review #1 fix)."""
        g = HypothesisGraph()
        with pytest.raises(HypothesisGraphError, match="不存在"):
            g.add_hypothesis("x", parent_id="h_nope")
        assert len(g.all_nodes()) == 0
        assert len(g.edges()) == 0

    def test_get_missing_raises(self):
        g = HypothesisGraph()
        with pytest.raises(HypothesisGraphError, match="不存在"):
            g.get("h_nope")


# ── 状态转移 ────────────────────────────────────────────────────────────────


class TestStatusTransitions:
    def test_support_marks_supported(self):
        g = HypothesisGraph()
        h = g.add_hypothesis("假设 A")
        g.support(h, evidence={"p_value": 0.01})
        assert g.get(h).status == "supported"
        assert g.get(h).evidence["p_value"] == 0.01

    def test_refute_marks_refuted(self):
        g = HypothesisGraph()
        h = g.add_hypothesis("假设 B")
        g.refute(h, evidence={"result": "相反趋势"})
        assert g.get(h).status == "refuted"

    def test_cannot_support_after_refute(self):
        g = HypothesisGraph()
        h = g.add_hypothesis("假设 C")
        g.refute(h, evidence={})
        with pytest.raises(HypothesisGraphError, match="不能再标记为 supported"):
            g.support(h, evidence={})

    def test_cannot_refute_after_support(self):
        g = HypothesisGraph()
        h = g.add_hypothesis("假设 D")
        g.support(h, evidence={})
        with pytest.raises(HypothesisGraphError, match="不能再标记为 refuted"):
            g.refute(h, evidence={})

    def test_supersede(self):
        g = HypothesisGraph()
        h = g.add_hypothesis("假设 E")
        g.refute(h, evidence={})  # supersede 要求节点先测过
        g.supersede(h)
        assert g.get(h).status == "superseded"

    def test_supersede_rejects_untested(self):
        # 源状态校验: 没测过的节点不应被取代
        g = HypothesisGraph()
        h = g.add_hypothesis("未测试的假设")
        with pytest.raises(HypothesisGraphError, match="未测试"):
            g.supersede(h)


# ── 查询 ────────────────────────────────────────────────────────────────────


class TestQueries:
    def test_frontier_returns_untested(self):
        g = HypothesisGraph()
        h1 = g.add_hypothesis("A")
        h2 = g.add_hypothesis("B")
        h3 = g.add_hypothesis("C")
        g.support(h1, evidence={})
        g.refute(h2, evidence={})
        frontier = g.frontier()
        assert len(frontier) == 1
        assert frontier[0].id == h3

    def test_supported_and_refuted_lists(self):
        g = HypothesisGraph()
        h1 = g.add_hypothesis("A")
        h2 = g.add_hypothesis("B")
        g.support(h1, evidence={})
        g.refute(h2, evidence={})
        assert len(g.supported()) == 1
        assert len(g.refuted()) == 1

    def test_children(self):
        g = HypothesisGraph()
        h1 = g.add_hypothesis("parent")
        h2 = g.add_hypothesis("child1", parent_id=h1)
        h3 = g.add_hypothesis("child2", parent_id=h1)
        kids = g.children(h1)
        assert len(kids) == 2
        kid_stmts = {k.statement for k in kids}
        assert kid_stmts == {"child1", "child2"}

    def test_derivation_chain(self):
        g = HypothesisGraph()
        h1 = g.add_hypothesis("root")
        h2 = g.add_hypothesis("mid", parent_id=h1)
        h3 = g.add_hypothesis("leaf", parent_id=h2)
        chain = g.derivation_chain(h3)
        assert [n.statement for n in chain] == ["root", "mid", "leaf"]


# ── refine_failed ───────────────────────────────────────────────────────────


class TestRefineFailed:
    def test_refine_requires_refuted_status(self):
        g = HypothesisGraph()
        h = g.add_hypothesis("未测试假设")
        with pytest.raises(HypothesisGraphError, match="只有 refuted 才能 refine"):
            g.refine_failed(h, evidence={})

    def test_refine_template_mode(self):
        """无 model 时走模板拼接."""
        g = HypothesisGraph()
        h = g.add_hypothesis(
            "如果掺杂浓度增加, 那么带隙会减小, 这个关系在所有温度范围内都成立",
            testable_prediction="带隙随掺杂浓度单调减小",
        )
        g.refute(h, evidence={"result": "带隙在高掺杂下反而增加"})
        new_id = g.refine_failed(h, evidence={"result": "带隙在高掺杂下反而增加"})
        new_node = g.get(new_id)
        assert new_node.status == "untested"
        assert new_node.parent_id == h
        assert "修正假设" in new_node.statement
        # 旧节点变 superseded
        assert g.get(h).status == "superseded"
        # refinement_basis 记了 findings
        assert len(new_node.refinement_basis) > 0

    def test_refine_creates_derive_edge(self):
        g = HypothesisGraph()
        h = g.add_hypothesis("如果 X 则 Y, 在所有条件下成立")
        g.refute(h, evidence={})
        new_id = g.refine_failed(h, evidence={})
        derive_edges = [e for e in g.edges() if e.edge_type == "derive"
                        and e.from_id == h and e.to_id == new_id]
        assert len(derive_edges) == 1

    def test_refine_with_mock_model_skips_llm(self):
        """MagicMock model 应该走模板, 不走 LLM 路径."""
        g = HypothesisGraph()
        h = g.add_hypothesis("如果 X 则 Y, 在所有条件下成立")
        g.refute(h, evidence={})
        mock_model = MagicMock()
        new_id = g.refine_failed(h, evidence={}, model=mock_model)
        new_node = g.get(new_id)
        # MagicMock 被识别, 走模板不走 LLM
        assert "修正假设" in new_node.statement
        mock_model.ainvoke.assert_not_called()
        mock_model.invoke.assert_not_called()

    def test_refine_chain_multiple_rounds(self):
        """连续修正: refute → refine → refute → refine."""
        g = HypothesisGraph()
        h1 = g.add_hypothesis("假设 v1, 在所有条件下成立")
        g.refute(h1, evidence={"round": 1})
        h2 = g.refine_failed(h1, evidence={"round": 1})
        g.refute(h2, evidence={"round": 2})
        h3 = g.refine_failed(h2, evidence={"round": 2})
        chain = g.derivation_chain(h3)
        assert len(chain) == 3
        assert g.get(h1).status == "superseded"
        assert g.get(h2).status == "superseded"
        assert g.get(h3).status == "untested"


# ── pivot 战略转向 ──────────────────────────────────────────────────────────


class TestPivot:
    def test_pivot_template_creates_new_untested_node(self):
        """无 model 时走模板, 新假设是 untested."""
        g = HypothesisGraph()
        h = g.add_hypothesis("假设 A: X 增加则 Y 增加")
        g.refute(h, evidence={"err": "Y 反而减少"})
        new_id = g.pivot(h, evidence={"err": "Y 反而减少"})
        new_node = g.get(new_id)
        assert new_node.status == "untested"
        # pivot 不继承 parent (不是 derive 关系)
        assert new_node.parent_id is None

    def test_pivot_creates_pivot_edge_not_derive(self):
        """pivot 关系用 "pivot" 类型, 不污染 children() 的 derive 过滤."""
        g = HypothesisGraph()
        h = g.add_hypothesis("原假设")
        g.refute(h, evidence={})
        new_id = g.pivot(h, evidence={})
        pivot_edges = [e for e in g.edges()
                       if e.edge_type == "pivot"
                       and e.from_id == h and e.to_id == new_id]
        assert len(pivot_edges) == 1
        assert pivot_edges[0].evidence["reason"] == "max_refines_reached"
        # 不应该有 derive 边连这两点
        derive_edges = [e for e in g.edges()
                        if e.edge_type == "derive"
                        and e.from_id == h and e.to_id == new_id]
        assert derive_edges == []

    def test_pivot_does_not_pollute_derivation_chain(self):
        """derivation_chain 只跟 derive 边, pivot 关系不算衍生."""
        g = HypothesisGraph()
        h = g.add_hypothesis("root")
        g.refute(h, evidence={})
        new_id = g.pivot(h, evidence={})
        chain = g.derivation_chain(new_id)
        # pivot 后新节点没 derive 祖先, chain 只含自己
        assert chain == [g.get(new_id)]

    def test_pivot_records_pivot_event(self):
        g = HypothesisGraph()
        h = g.add_hypothesis("失败假设")
        g.refute(h, evidence={})
        new_id = g.pivot(h, evidence={})
        piv_evs = [e for e in g.events() if e["event"] == "pivot"]
        assert len(piv_evs) == 1
        assert piv_evs[0]["node_id"] == new_id
        assert piv_evs[0]["from_node"] == h
        assert piv_evs[0]["failed_count"] >= 1

    def test_pivot_with_mock_model_skips_llm(self):
        """MagicMock model 应该走模板, 不调 LLM."""
        g = HypothesisGraph()
        h = g.add_hypothesis("原方向假设")
        g.refute(h, evidence={})
        mock_model = MagicMock()
        new_id = g.pivot(h, evidence={}, model=mock_model)
        mock_model.ainvoke.assert_not_called()
        mock_model.invoke.assert_not_called()
        # 模板生成, 新假设不为空
        assert g.get(new_id).statement != ""

    def test_pivot_does_not_supersede_failed_node(self):
        """pivot 不会自动 supersede 失败节点 — 它已 refuted, 状态足够明确."""
        g = HypothesisGraph()
        h = g.add_hypothesis("原方向假设")
        g.refute(h, evidence={})
        g.pivot(h, evidence={})
        assert g.get(h).status == "refuted"


# ── 序列化 ──────────────────────────────────────────────────────────────────


class TestSerialization:
    def test_roundtrip(self):
        g = HypothesisGraph()
        h1 = g.add_hypothesis("假设 A", rationale="因为 X", testable_prediction="应观察到 Y")
        h2 = g.add_hypothesis("假设 B", parent_id=h1)
        g.support(h1, evidence={"p": 0.01})
        g.refute(h2, evidence={"r": "相反"})
        d = g.to_dict()
        g2 = HypothesisGraph.from_dict(d)
        assert len(g2.all_nodes()) == 2
        assert g2.get(h1).status == "supported"
        assert g2.get(h2).status == "refuted"
        assert g2.get(h1).rationale == "因为 X"
        edges2 = g2.edges()
        assert len(edges2) >= 3  # 1 derive + 1 support + 1 refute

    def test_empty_graph_roundtrip(self):
        g = HypothesisGraph()
        d = g.to_dict()
        g2 = HypothesisGraph.from_dict(d)
        assert g2.all_nodes() == []
        assert g2.edges() == []


# ── 边 ──────────────────────────────────────────────────────────────────────


class TestEdges:
    def test_support_edge(self):
        g = HypothesisGraph()
        h = g.add_hypothesis("X")
        g.support(h, evidence={"val": 1})
        support_edges = [e for e in g.edges() if e.edge_type == "support"]
        assert len(support_edges) == 1
        assert support_edges[0].evidence["val"] == 1

    def test_refute_edge(self):
        g = HypothesisGraph()
        h = g.add_hypothesis("Y")
        g.refute(h, evidence={"val": 2})
        refute_edges = [e for e in g.edges() if e.edge_type == "refute"]
        assert len(refute_edges) == 1
        assert refute_edges[0].evidence["val"] == 2

    def test_edge_to_dict(self):
        e = HypothesisEdge(from_id="a", to_id="b", edge_type="derive", evidence={"k": 1})
        d = e.to_dict()
        assert d["from_id"] == "a"
        assert d["to_id"] == "b"
        assert d["edge_type"] == "derive"
        assert d["evidence"] == {"k": 1}

    def test_node_to_dict_from_dict(self):
        n = HypothesisNode(
            id="h1", statement="test", rationale="r",
            testable_prediction="p", status="supported",
            parent_id="h0", evidence={"k": 1},
            created_at="2026-07-01T00:00:00Z",
            refinement_basis=[{"category": "x"}],
        )
        d = n.to_dict()
        n2 = HypothesisNode.from_dict(d)
        assert n2.id == "h1"
        assert n2.statement == "test"
        assert n2.status == "supported"
        assert n2.refinement_basis == [{"category": "x"}]


# ── 双覆盖查询 ─────────────────────────────────────────────────────────────


class TestDualCoverage:
    def test_dual_covered_false_with_no_support(self):
        g = HypothesisGraph()
        h = g.add_hypothesis("H")
        assert g.dual_covered(h) is False

    def test_dual_covered_false_with_single_modality(self):
        g = HypothesisGraph()
        h = g.add_hypothesis("H")
        g.support(h, evidence={"modality": "deductive"})
        assert g.dual_covered(h) is False

    def test_dual_covered_true_with_two_modalities(self):
        g = HypothesisGraph()
        h = g.add_hypothesis("H")
        g.support(h, evidence={"modality": "deductive"})
        g.support(h, evidence={"modality": "numeric"})
        assert g.dual_covered(h) is True

    def test_dual_covered_false_with_same_modality_twice(self):
        g = HypothesisGraph()
        h = g.add_hypothesis("H")
        g.support(h, evidence={"modality": "deductive"})
        g.support(h, evidence={"modality": "deductive"})
        assert g.dual_covered(h) is False

    def test_dual_covered_false_with_same_data_source(self):
        """两条不同 modality 但同一 data_source → 假双覆盖 (IPI 防御).
        攻击场景: 两条 support 边都来自同一被污染的 CIF."""
        g = HypothesisGraph()
        h = g.add_hypothesis("H")
        g.support(h, evidence={
            "modality": "deductive", "data_source": "polluted_cif"
        })
        g.support(h, evidence={
            "modality": "numeric", "data_source": "polluted_cif"
        })
        assert g.dual_covered(h) is False

    def test_dual_covered_true_with_different_data_sources(self):
        """两条不同 modality 且不同 data_source → 真双覆盖."""
        g = HypothesisGraph()
        h = g.add_hypothesis("H")
        g.support(h, evidence={
            "modality": "deductive", "data_source": "symbolic_tool"
        })
        g.support(h, evidence={
            "modality": "numeric", "data_source": "gp_tool"
        })
        assert g.dual_covered(h) is True

    def test_dual_covered_backward_compat_no_data_source(self):
        """没有 data_source 字段时, 退回只检查 modality (向后兼容)."""
        g = HypothesisGraph()
        h = g.add_hypothesis("H")
        g.support(h, evidence={"modality": "deductive"})
        g.support(h, evidence={"modality": "numeric"})
        assert g.dual_covered(h) is True

    def test_needs_dual_coverage_root_node_false(self):
        """单节点图无割点."""
        g = HypothesisGraph()
        h = g.add_hypothesis("H")
        assert g.needs_dual_coverage(h) is False

    def test_needs_dual_coverage_chain_middle_true(self):
        """链 H1→H2→H3: H2 是割点 (删后 H1 与 H3 断开)."""
        g = HypothesisGraph()
        h1 = g.add_hypothesis("H1")
        h2 = g.add_hypothesis("H2", parent_id=h1)
        h3 = g.add_hypothesis("H3", parent_id=h2)
        assert g.needs_dual_coverage(h2) is True
        # 端点不是割点
        assert g.needs_dual_coverage(h1) is False
        assert g.needs_dual_coverage(h3) is False

    def test_needs_dual_coverage_star_center_true(self):
        """星形: 中心 H1 是割点 (删后所有叶子断开)."""
        g = HypothesisGraph()
        h1 = g.add_hypothesis("H1")
        h2 = g.add_hypothesis("H2", parent_id=h1)
        h3 = g.add_hypothesis("H3", parent_id=h1)
        assert g.needs_dual_coverage(h1) is True
        assert g.needs_dual_coverage(h2) is False
        assert g.needs_dual_coverage(h3) is False

    def test_needs_dual_coverage_tree_leaf_false(self):
        """树形叶子不是割点."""
        g = HypothesisGraph()
        h1 = g.add_hypothesis("H1")
        h2 = g.add_hypothesis("H2", parent_id=h1)
        h3 = g.add_hypothesis("H3", parent_id=h1)
        assert g.needs_dual_coverage(h2) is False
        assert g.needs_dual_coverage(h3) is False


# ── 事件日志 ───────────────────────────────────────────────────────────────


class TestEventLog:
    """Session event log: 结构化事件可回放/调试 (Anthropic Managed Agents 模式)."""

    def test_add_records_event(self):
        g = HypothesisGraph()
        h = g.add_hypothesis("H", parent_id=None)
        evs = g.events()
        assert len(evs) == 1
        assert evs[0]["event"] == "add"
        assert evs[0]["node_id"] == h
        assert "ts" in evs[0]

    def test_support_records_modality_and_data_source(self):
        g = HypothesisGraph()
        h = g.add_hypothesis("H")
        g.support(h, evidence={
            "modality": "deductive", "data_source": "symbolic_tool",
        })
        sup_evs = [e for e in g.events() if e["event"] == "support"]
        assert len(sup_evs) == 1
        assert sup_evs[0]["modality"] == "deductive"
        assert sup_evs[0]["data_source"] == "symbolic_tool"

    def test_refute_records_reason(self):
        g = HypothesisGraph()
        h = g.add_hypothesis("H")
        g.refute(h, evidence={"errors": "带隙反而增加"})
        ref_evs = [e for e in g.events() if e["event"] == "refute"]
        assert len(ref_evs) == 1
        assert "带隙反而增加" in ref_evs[0]["reason"]

    def test_supersede_records_event(self):
        g = HypothesisGraph()
        h = g.add_hypothesis("H")
        g.refute(h, evidence={})  # supersede 要求节点先测过
        g.supersede(h)
        sup_evs = [e for e in g.events() if e["event"] == "supersede"]
        assert len(sup_evs) == 1
        assert sup_evs[0]["node_id"] == h

    def test_refine_failed_records_refine_event_with_from_node(self):
        g = HypothesisGraph()
        h = g.add_hypothesis("如果 X 则 Y, 在所有条件下成立")
        g.refute(h, evidence={})
        new_id = g.refine_failed(h, evidence={})
        refine_evs = [e for e in g.events() if e["event"] == "refine"]
        assert len(refine_evs) == 1
        assert refine_evs[0]["node_id"] == new_id
        assert refine_evs[0]["from_node"] == h
        assert refine_evs[0]["findings_count"] >= 0

    def test_events_are_append_only_and_ordered(self):
        """完整生命周期事件按发生顺序排列."""
        g = HypothesisGraph()
        h1 = g.add_hypothesis("H1")
        g.refute(h1, evidence={"errors": "fail1"})
        h2 = g.refine_failed(h1, evidence={"errors": "fail1"})
        ev_types = [e["event"] for e in g.events()]
        # add(h1) → refute(h1) → add(h2) → refine → supersede(h1)
        assert ev_types == ["add", "refute", "add", "refine", "supersede"]
        # ts 单调不减
        timestamps = [e["ts"] for e in g.events()]
        assert timestamps == sorted(timestamps)

    def test_events_returns_copy(self):
        """events() 返回副本, 修改不影响内部状态."""
        g = HypothesisGraph()
        g.add_hypothesis("H")
        evs = g.events()
        evs.clear()
        assert len(g.events()) == 1

    def test_empty_graph_events(self):
        assert HypothesisGraph().events() == []
