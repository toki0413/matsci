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
        g.supersede(h)
        assert g.get(h).status == "superseded"


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
