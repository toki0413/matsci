"""Darwin ratchet 测试 — 假设质量棘轮 + early stop.

4 测:
  1. score 计算: supported/testable/diversity/topology_richness 四维
  2. 棘轮只保留改进: score 退化时不更新 best_score
  3. 连续 2 轮 Δ<0.5 → early stop (_should_stop=True)
  4. topology_richness 维: 有环图比树图得分高
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from huginn.autoloop.hypothesis_loop import HypothesisEdge, HypothesisNode, HypothesisGraph
from huginn.autoloop.engine import AutoloopEngine


# ponytail: CI 无 HUGINN_MODEL 时 get_model 抛 ValueError, stub 掉
@pytest.fixture(autouse=True)
def _stub_model(monkeypatch):
    monkeypatch.setattr("huginn.autoloop.engine.get_model", lambda s: MagicMock())


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
        engine = AutoloopEngine()
        graph = engine.hypothesis_graph
        # 直接塞节点, 绕过 add_hypothesis 的复杂逻辑
        graph._nodes = {
            "a": _make_node("a", "H1", "supported", "pred1"),
            "b": _make_node("b", "H2", "untested", "pred2"),
            "c": _make_node("c", "H3", "untested", "pred3"),
        }
        engine._iteration = 1

        engine._darwin_ratchet_check()

        # best_score 应等于算出的 score
        assert engine._darwin_best_score > 0
        assert 5.0 < engine._darwin_best_score < 6.5  # ≈5.83
        assert engine._darwin_stagnation == 0  # 第一轮, 无 stagnation

    def test_ratchet_keeps_best_on_regression(self):
        """第一轮高分, 第二轮退化 → best 保留第一轮."""
        engine = AutoloopEngine()
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

        # best 应保留第一轮的高分, 不被第二轮拉低
        assert engine._darwin_best_score == best_after_r1
        # stagnation 应增加 (Δ<0.5)
        assert engine._darwin_stagnation >= 1

    def test_early_stop_on_stagnation(self):
        """连续 2 轮 Δ<0.5 + iteration>2 → _should_stop=True."""
        engine = AutoloopEngine()
        graph = engine.hypothesis_graph

        # 第 1 轮: 稳定状态
        graph._nodes = {
            "a": _make_node("a", "H1", "supported", "p1"),
        }
        engine._iteration = 1
        engine._darwin_ratchet_check()
        assert not engine._should_stop

        # 第 2 轮: 微小变化 (Δ<0.5)
        graph._nodes = {
            "a": _make_node("a", "H1", "supported", "p1"),
            "b": _make_node("b", "H2", "untested", ""),  # 加一个 untested, score 略降
        }
        engine._iteration = 2
        engine._darwin_ratchet_check()
        assert not engine._should_stop  # stagnation=1, 还没到 2

        # 第 3 轮: 继续微小变化
        graph._nodes = {
            "a": _make_node("a", "H1", "supported", "p1"),
            "b": _make_node("b", "H2", "untested", ""),
            "c": _make_node("c", "H3", "untested", ""),
        }
        engine._iteration = 3
        engine._darwin_ratchet_check()
        # stagnation>=2 且 iteration>2 → early stop
        assert engine._should_stop, "连续 2 轮低增益应触发 early stop"

    def test_no_early_stop_when_improving(self):
        """每轮都改进 (Δ>=0.5) → 不触发 early stop."""
        engine = AutoloopEngine()
        graph = engine.hypothesis_graph

        # 第 1 轮: 1 个 untested
        graph._nodes = {"a": _make_node("a", "H1", "untested", "p1")}
        engine._iteration = 1
        engine._darwin_ratchet_check()
        assert not engine._should_stop

        # 第 2 轮: 变成 supported (大改进, Δ>=0.5)
        graph._nodes = {"a": _make_node("a", "H1", "supported", "p1")}
        engine._iteration = 2
        engine._darwin_ratchet_check()
        assert not engine._should_stop
        assert engine._darwin_stagnation == 0  # 改进了, stagnation 清零

    def test_empty_graph_noop(self):
        """空图 → no-op, 不崩溃."""
        engine = AutoloopEngine()
        engine._iteration = 1
        engine._darwin_ratchet_check()
        assert engine._darwin_best_score == 0.0
        assert not engine._should_stop

    def test_topology_richness_boosts_score(self):
        """有环图 (β₁>0) 比同节点同 supported 比例的树图得分高.

        3 节点 + 3 边构成三角形 (β₁=1), 比 3 节点无边 (β₁=0) 得分高.
        ponytail: 这是 topology_richness 维度的最小可运行验证.
        """
        engine_tree = AutoloopEngine()
        engine_tree.hypothesis_graph._nodes = {
            "a": _make_node("a", "H1", "supported", "p1"),
            "b": _make_node("b", "H2", "supported", "p2"),
            "c": _make_node("c", "H3", "supported", "p3"),
        }
        engine_tree._iteration = 1
        engine_tree._darwin_ratchet_check()
        tree_score = engine_tree._darwin_best_score

        engine_cyclic = AutoloopEngine()
        engine_cyclic.hypothesis_graph._nodes = {
            "a": _make_node("a", "H1", "supported", "p1"),
            "b": _make_node("b", "H2", "supported", "p2"),
            "c": _make_node("c", "H3", "supported", "p3"),
        }
        # 三角形: A→B→C→A, β₁ = 3 - 3 + 1 = 1
        engine_cyclic.hypothesis_graph._edges = [
            _edge("a", "b"), _edge("b", "c"), _edge("c", "a"),
        ]
        engine_cyclic._iteration = 1
        engine_cyclic._darwin_ratchet_check()
        cyclic_score = engine_cyclic._darwin_best_score

        # topology_richness: 树=0/3=0, 三角形=1/3≈0.33
        # 树 score = (1+1+1+0)/4*10 = 7.5
        # 环 score = (1+1+1+0.33)/4*10 = 8.33
        assert cyclic_score > tree_score, (
            f"有环图应比树图得分高: cyclic={cyclic_score}, tree={tree_score}"
        )
