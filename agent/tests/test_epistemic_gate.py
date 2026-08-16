"""Epistemic gate (IOED) 测试 — 强断言 + 证据/不确定性标注的组合判定.

IOED (illusion of explanatory depth): agent 自信断言但缺证据且不标不确定.
detect_epistemic_gap 在 autoloop._validate 阶段兜底暴露这种知识缺口.
"""

from __future__ import annotations

from huginn.validation.epistemic import detect_epistemic_gap


def test_confident_claim_without_evidence_flags_gap():
    """强断言 + 无证据 + 无不确定性标注 → 标记 overconfidence 缺口."""
    r = detect_epistemic_gap(
        {"final_answer": "this proves that the defect causes the degradation"},
        {},
    )
    assert r is not None
    assert r["overconfidence"] is True
    assert r["has_evidence"] is False
    assert "unknown" in r["advice"]  # 建议明确 established/estimated/unknown


def test_uncertainty_hedge_suppresses_gap():
    """已标注不确定性 → 不触发缺口 (agent 已自知之明)."""
    r = detect_epistemic_gap(
        {"final_answer": "this might be caused by the defect, but I am not sure"},
        {},
    )
    assert r is None


def test_evidence_present_gives_mild_hint_not_overconfidence():
    """有证据支撑 → 仅轻度提示补局限说明, 不标记 overconfidence."""
    r = detect_epistemic_gap(
        {"final_answer": "the calculated bandgap is 1.2 eV, which demonstrates "
         "the material is a semiconductor"},
        {"r_phys": 0.85},
    )
    assert r is not None
    assert r["overconfidence"] is False
    assert r["has_evidence"] is True
    assert "补一句" in r["advice"]


def test_empty_result_no_gap():
    """空结果 → 不触发 (无断言可查)."""
    assert detect_epistemic_gap(None, {}) is None
    assert detect_epistemic_gap({}, {}) is None


def test_never_raises_on_garbage():
    """异常输入 → 返 None 而非抛异常 (防御性兜底)."""
    assert detect_epistemic_gap(object(), object()) is None
    assert detect_epistemic_gap(12345, ["x"]) is None
