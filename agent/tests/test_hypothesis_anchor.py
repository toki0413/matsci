"""假设落地锚的单测: 验证证据来源 → 锚 / 覆盖率 / 持久化."""

from __future__ import annotations

from huginn.autoloop.hypothesis_loop import (
    HypothesisGraph,
    HypothesisNode,
    _dominant_grounding,
)


def test_dominant_grounding_reads_source_class():
    assert _dominant_grounding({"source_class": "tool_output"}) == "tool_output"
    assert _dominant_grounding({"source_class": "external_content"}) == "external_content"
    # 嵌套 evidence 也扫得到
    assert (
        _dominant_grounding({"outer": {"source_class": "external_content"}})
        == "external_content"
    )
    # 无 source_class → "" (未知, 不臆造)
    assert _dominant_grounding({"modality": "execution"}) == ""


def test_support_stamps_grounding():
    g = HypothesisGraph()
    h = g.add_hypothesis("掺杂增加带隙减小", rationale="r")
    g.support(h, {"tests_passed": True, "source_class": "tool_output"})
    assert g.get(h).grounding == "tool_output"
    assert g.grounded_ratio() == 1.0


def test_agent_generated_is_not_grounded():
    g = HypothesisGraph()
    h = g.add_hypothesis("纯推理假设", rationale="r")
    g.support(h, {"source_class": "agent_generated"})
    # 记下来源类别, 但不计入落地 (自说自话不算有据)
    assert g.get(h).grounding == "agent_generated"
    assert g.grounded_ratio() == 0.0


def test_grounded_ratio_is_fraction_and_empty_is_zero():
    g = HypothesisGraph()
    a = g.add_hypothesis("有据假设 a")
    b = g.add_hypothesis("无据假设 b")
    g.support(a, {"source_class": "external_content"})
    # b 保持 untested (无 grounding)
    assert g.get(b).grounding == ""
    assert g.grounded_ratio() == 0.5
    assert HypothesisGraph().grounded_ratio() == 0.0


def test_refute_also_stamps_grounding():
    g = HypothesisGraph()
    h = g.add_hypothesis("可能被实测反驳的假设")
    g.refute(h, {"errors": "band_gap 反向", "source_class": "tool_output"})
    assert g.get(h).grounding == "tool_output"
    assert g.grounded_ratio() == 1.0


def test_grounding_survives_roundtrip():
    n = HypothesisNode(id="h1", statement="s", grounding="external_content")
    assert HypothesisNode.from_dict(n.to_dict()).grounding == "external_content"
    # 旧快照缺 grounding 字段 → 默认 "" (向后兼容)
    legacy = HypothesisNode.from_dict({"id": "h2", "statement": "s"})
    assert legacy.grounding == ""
