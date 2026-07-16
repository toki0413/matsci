"""Darwin ratchet 测试 — 假设质量棘轮 + early stop.

4 测:
  1. score 计算: supported/testable/diversity/topology_richness 四维
  2. 棘轮只保留改进: score 退化时不更新 best_score
  3. 连续 2 轮 Δ<0.5 → early stop (_should_stop=True)
  4. topology_richness 维: 有环图比树图得分高
"""
from __future__ import annotations

from huginn.autoloop.hypothesis_loop import HypothesisEdge, HypothesisNode, HypothesisGraph
from huginn.autoloop.engine import AutoloopEngine


def _make_engine():
    # 跳过 __init__ (会触发 get_model 报 ValueError), 只塞 _darwin_ratchet_check 需要的字段
    eng = AutoloopEngine.__new__(AutoloopEngine)
    eng.hypothesis_graph = HypothesisGraph()
    eng._darwin_best_score = 0.0
    eng._darwin_stagnation = 0
    eng._darwin_last_score = 0.0
    eng._iteration = 0
    eng._should_stop = False
    return eng


def _make_node(
    nid: str, statement: str, status: str = "untested",
    testable: str = "",
) -> HypothesisNode:
    return HypothesisNode(
        id=nid, statement=statement, status=status,
        testable_prediction=testable,
    )


def _edge(a: str, b: str, et: str = "support") -> HypothesisEdge:
    return HypothesisEdge(from_id=a, to_id=b, edge_type=et)


class TestDarwinRatchet:
    """_darwin_ratchet_check 纯逻辑测试."""

    def test_score_calculation_basic(self):
        """3 节点无边: 1 supported + 2 untested, 全有 testable, 全唯一.
        supported_ratio=1/3, testable_ratio=3/3, diversity=3/3, topology=0 (无环).
        score = (1/3 + 1 + 1 + 0) / 4 * 10 = 5.83
        """
        engine = _make_engine()
        graph = engine.hypothesis_graph
        graph._nodes = {
            "a": _make_node("a", "H1", "supported", "pred1"),
            "b": _make_node("b", "H2", "untested", "pred2"),
            "c": _make_node("c", "H3", "untested", "pred3"),
        }
        engine._iteration = 1

        engine._darwin_ratchet_check()

        assert engine._darwin_best_score > 0
        assert 5.0 < engine._darwin_best_score < 6.5  # ≈5.83
        assert engine._darwin_stagnation == 0  # 第一轮, 无 stagnation

    def test_ratchet_keeps_best_on_regression(self):
        """第一轮高分, 第二轮退化 → best 保留第一轮."""
        engine = _make_engine()
        graph = engine.hypothesis_graph

        # 第一轮: 全 supported, 全 testable, 全唯一, 无环 → topology=0
        # score = (1 + 1 + 1 + 0) / 4 * 10 = 7.5
        graph._nodes = {
            "a": _make_node("a", "H1", "supported", "p1"),
            "b": _make_node("b", "H2", "supported", "p2"),
        }
        engine._iteration = 1
        engine._darwin_ratchet_check()
        best_after_r1 = engine._darwin_best_score
        assert best_after_r1 > 6.0  # ≈7.5

        # 第二轮: 全 refute (退化), 无 testable → 低分
        graph._nodes = {
            "a": _make_node("a", "H1", "refuted", ""),
            "b": _make_node("b", "H2", "refuted", ""),
        }
        engine._iteration = 2
        engine._darwin_ratchet_check()

        assert engine._darwin_best_score == best_after_r1
        assert engine._darwin_stagnation >= 1

    def test_early_stop_on_stagnation(self):
        """连续 2 轮 Δ<0.5 + iteration>2 → _should_stop=True."""
        engine = _make_engine()
        graph = engine.hypothesis_graph

        graph._nodes = {
            "a": _make_node("a", "H1", "supported", "p1"),
        }
        engine._iteration = 1
        engine._darwin_ratchet_check()
        assert not engine._should_stop

        graph._nodes = {
            "a": _make_node("a", "H1", "supported", "p1"),
            "b": _make_node("b", "H2", "untested", ""),
        }
        engine._iteration = 2
        engine._darwin_ratchet_check()
        assert not engine._should_stop  # stagnation=1, 还没到 2

        graph._nodes = {
            "a": _make_node("a", "H1", "supported", "p1"),
            "b": _make_node("b", "H2", "untested", ""),
            "c": _make_node("c", "H3", "untested", ""),
        }
        engine._iteration = 3
        engine._darwin_ratchet_check()
        assert engine._should_stop, "连续 2 轮低增益应触发 early stop"

    def test_no_early_stop_when_improving(self):
        """每轮都改进 (Δ>=0.5) → 不触发 early stop."""
        engine = _make_engine()
        graph = engine.hypothesis_graph

        graph._nodes = {"a": _make_node("a", "H1", "untested", "p1")}
        engine._iteration = 1
        engine._darwin_ratchet_check()
        assert not engine._should_stop

        graph._nodes = {"a": _make_node("a", "H1", "supported", "p1")}
        engine._iteration = 2
        engine._darwin_ratchet_check()
        assert not engine._should_stop
        assert engine._darwin_stagnation == 0

    def test_empty_graph_noop(self):
        """空图 → no-op, 不崩溃."""
        engine = _make_engine()
        engine._iteration = 1
        engine._darwin_ratchet_check()
        assert engine._darwin_best_score == 0.0
        assert not engine._should_stop

    def test_topology_richness_boosts_score(self):
        """有环图 (β₁>0) 比同节点同 supported 比例的树图得分高."""
        engine_tree = _make_engine()
        engine_tree.hypothesis_graph._nodes = {
            "a": _make_node("a", "H1", "supported", "p1"),
            "b": _make_node("b", "H2", "supported", "p2"),
            "c": _make_node("c", "H3", "supported", "p3"),
        }
        engine_tree._iteration = 1
        engine_tree._darwin_ratchet_check()
        tree_score = engine_tree._darwin_best_score

        engine_cyclic = _make_engine()
        engine_cyclic.hypothesis_graph._nodes = {
            "a": _make_node("a", "H1", "supported", "p1"),
            "b": _make_node("b", "H2", "supported", "p2"),
            "c": _make_node("c", "H3", "supported", "p3"),
        }
        engine_cyclic.hypothesis_graph._edges = [
            _edge("a", "b"), _edge("b", "c"), _edge("c", "a"),
        ]
        engine_cyclic._iteration = 1
        engine_cyclic._darwin_ratchet_check()
        cyclic_score = engine_cyclic._darwin_best_score

        assert cyclic_score > tree_score, (
            f"有环图应比树图得分高: cyclic={cyclic_score}, tree={tree_score}"
        )
