"""P0 tests for HypothesisGraph — node lifecycle, status transitions, serialization."""

from __future__ import annotations

import pytest

from huginn.autoloop.hypothesis_loop import (
    HypothesisGraph,
    HypothesisGraphError,
    HypothesisNode,
)


# ── add_hypothesis ──────────────────────────────────────────────────


class TestAddHypothesis:
    def test_returns_id_prefixed_h(self):
        g = HypothesisGraph()
        hid = g.add_hypothesis("doping increases conductivity")
        assert hid.startswith("h_")
        assert g.get(hid).statement == "doping increases conductivity"

    def test_new_node_is_untested(self):
        g = HypothesisGraph()
        hid = g.add_hypothesis("h1")
        assert g.get(hid).status == "untested"

    def test_empty_statement_raises(self):
        g = HypothesisGraph()
        with pytest.raises(HypothesisGraphError, match="不能为空"):
            g.add_hypothesis("   ")

    def test_parent_creates_derive_edge(self):
        g = HypothesisGraph()
        h1 = g.add_hypothesis("parent")
        h2 = g.add_hypothesis("child", parent_id=h1)
        assert g.get(h2).parent_id == h1
        derive_edges = [e for e in g.edges() if e.edge_type == "derive"]
        assert len(derive_edges) == 1
        assert derive_edges[0].from_id == h1
        assert derive_edges[0].to_id == h2

    def test_missing_parent_raises(self):
        g = HypothesisGraph()
        with pytest.raises(HypothesisGraphError, match="不存在"):
            g.add_hypothesis("x", parent_id="h_nope")

    def test_missing_parent_leaves_no_orphan(self):
        g = HypothesisGraph()
        with pytest.raises(HypothesisGraphError):
            g.add_hypothesis("x", parent_id="h_nope")
        assert len(g.all_nodes()) == 0
        assert len(g.edges()) == 0


# ── support / refute ────────────────────────────────────────────────


class TestSupportRefute:
    def test_support_sets_status_and_evidence(self):
        g = HypothesisGraph()
        hid = g.add_hypothesis("h")
        g.support(hid, evidence={"p": 0.01})
        assert g.get(hid).status == "supported"
        assert g.get(hid).evidence["p"] == 0.01

    def test_support_adds_support_edge(self):
        g = HypothesisGraph()
        hid = g.add_hypothesis("h")
        g.support(hid, evidence={})
        support_edges = [e for e in g.edges() if e.edge_type == "support"]
        assert len(support_edges) == 1

    def test_refute_sets_status_and_evidence(self):
        g = HypothesisGraph()
        hid = g.add_hypothesis("h")
        g.refute(hid, evidence={"reason": "wrong sign"})
        assert g.get(hid).status == "refuted"
        assert g.get(hid).evidence["reason"] == "wrong sign"

    def test_refute_adds_refute_edge(self):
        g = HypothesisGraph()
        hid = g.add_hypothesis("h")
        g.refute(hid, evidence={})
        refute_edges = [e for e in g.edges() if e.edge_type == "refute"]
        assert len(refute_edges) == 1

    def test_cannot_support_after_refute(self):
        g = HypothesisGraph()
        hid = g.add_hypothesis("h")
        g.refute(hid, evidence={})
        with pytest.raises(HypothesisGraphError, match="已被反驳"):
            g.support(hid, evidence={})

    def test_cannot_refute_after_support(self):
        g = HypothesisGraph()
        hid = g.add_hypothesis("h")
        g.support(hid, evidence={})
        with pytest.raises(HypothesisGraphError, match="已被支持"):
            g.refute(hid, evidence={})

    def test_support_merges_evidence(self):
        g = HypothesisGraph()
        hid = g.add_hypothesis("h")
        g.support(hid, evidence={"a": 1})
        g.support(hid, evidence={"b": 2})
        # ponytail: second support adds another edge but doesn't change status
        assert g.get(hid).evidence == {"a": 1, "b": 2}


# ── supersede ──────────────────────────────────────────────────────


class TestSupersede:
    def test_supersede_marks_status(self):
        g = HypothesisGraph()
        hid = g.add_hypothesis("h")
        g.supersede(hid)
        assert g.get(hid).status == "superseded"

    def test_supersede_missing_raises(self):
        g = HypothesisGraph()
        with pytest.raises(HypothesisGraphError):
            g.supersede("h_nope")


# ── refine_failed ───────────────────────────────────────────────────


class TestRefineFailed:
    def test_refine_creates_new_node_and_supersedes_old(self):
        g = HypothesisGraph()
        hid = g.add_hypothesis("如果掺杂增加, 带隙减小")
        g.refute(hid, evidence={"result": "bandgap increased"})
        new_id = g.refine_failed(hid, evidence={"result": "bandgap increased"})
        # new node is untested, parent is the refuted one
        assert g.get(new_id).status == "untested"
        assert g.get(new_id).parent_id == hid
        # old node is now superseded
        assert g.get(hid).status == "superseded"
        # refinement_basis is populated
        assert len(g.get(new_id).refinement_basis) > 0

    def test_refine_requires_refuted_status(self):
        g = HypothesisGraph()
        hid = g.add_hypothesis("h")
        with pytest.raises(HypothesisGraphError, match="只有 refuted"):
            g.refine_failed(hid, evidence={})

    def test_refine_missing_raises(self):
        g = HypothesisGraph()
        with pytest.raises(HypothesisGraphError):
            g.refine_failed("h_nope", evidence={})


# ── query helpers ──────────────────────────────────────────────────


class TestQueries:
    def test_frontier_returns_untested(self):
        g = HypothesisGraph()
        h1 = g.add_hypothesis("h1")
        h2 = g.add_hypothesis("h2")
        g.support(h1, evidence={})
        frontier = g.frontier()
        ids = {n.id for n in frontier}
        assert h2 in ids
        assert h1 not in ids

    def test_supported_returns_supported_only(self):
        g = HypothesisGraph()
        h1 = g.add_hypothesis("h1")
        h2 = g.add_hypothesis("h2")
        g.support(h1, evidence={})
        g.refute(h2, evidence={})
        supported = g.supported()
        assert len(supported) == 1
        assert supported[0].id == h1

    def test_refuted_returns_refuted_only(self):
        g = HypothesisGraph()
        h1 = g.add_hypothesis("h1")
        h2 = g.add_hypothesis("h2")
        g.refute(h2, evidence={})
        refuted = g.refuted()
        assert len(refuted) == 1
        assert refuted[0].id == h2

    def test_children_finds_derive_descendants(self):
        g = HypothesisGraph()
        h1 = g.add_hypothesis("parent")
        h2 = g.add_hypothesis("child1", parent_id=h1)
        h3 = g.add_hypothesis("child2", parent_id=h1)
        kids = g.children(h1)
        child_ids = {n.id for n in kids}
        assert child_ids == {h2, h3}


# ── to_dict / from_dict round-trip ──────────────────────────────────


class TestSerialization:
    def test_round_trip_preserves_nodes_and_edges(self):
        g = HypothesisGraph()
        h1 = g.add_hypothesis("假设 A", rationale="r1", testable_prediction="p1")
        h2 = g.add_hypothesis("假设 B", parent_id=h1)
        g.support(h1, evidence={"score": 0.9})
        g.refute(h2, evidence={"reason": "no effect"})

        data = g.to_dict()
        g2 = HypothesisGraph.from_dict(data)

        assert len(g2.all_nodes()) == 2
        assert len(g2.edges()) == len(g.edges())

        n1 = g2.get(h1)
        assert n1.statement == "假设 A"
        assert n1.status == "supported"
        assert n1.evidence["score"] == 0.9
        assert n1.rationale == "r1"
        assert n1.testable_prediction == "p1"

        n2 = g2.get(h2)
        assert n2.status == "refuted"
        assert n2.parent_id == h1

    def test_round_trip_preserves_edge_types(self):
        g = HypothesisGraph()
        h1 = g.add_hypothesis("h1")
        h2 = g.add_hypothesis("h2")
        g.support(h1, evidence={})
        g.refute(h2, evidence={})
        data = g.to_dict()
        g2 = HypothesisGraph.from_dict(data)
        edge_types = {e.edge_type for e in g2.edges()}
        assert "support" in edge_types
        assert "refute" in edge_types

    def test_empty_graph_serialization(self):
        g = HypothesisGraph()
        data = g.to_dict()
        assert data == {"nodes": [], "edges": []}
        g2 = HypothesisGraph.from_dict(data)
        assert len(g2.all_nodes()) == 0

    def test_node_to_from_dict_round_trip(self):
        node = HypothesisNode(
            id="h_test",
            statement="test statement",
            rationale="why",
            testable_prediction="predict X",
            status="supported",
            parent_id="h_parent",
            evidence={"key": "val"},
            created_at="2024-01-01T00:00:00Z",
            refinement_basis=[{"desc": "finding"}],
        )
        d = node.to_dict()
        restored = HypothesisNode.from_dict(d)
        assert restored.id == node.id
        assert restored.statement == node.statement
        assert restored.status == node.status
        assert restored.parent_id == node.parent_id
        assert restored.evidence == node.evidence
        assert restored.refinement_basis == node.refinement_basis
