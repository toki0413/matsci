"""UnifiedEvaluator — 统一评估器接口测试.

覆盖:
  - UnifiedEvaluationResult 钳值/默认字段
  - from_mcda / from_goal_judge / from_grader / from_step_evaluator 适配
  - evaluate() 聚合 (多子结果 / 单分支 / 全异常 / 空输入)
  - duck-typing (dataclass + dict 两种形态)
  - 异常输入不抛 (各分支 try/except 兜底)
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from huginn.evaluation.unified_evaluator import (
    UnifiedEvaluationResult,
    UnifiedEvaluator,
)

# ── UnifiedEvaluationResult ────────────────────────────────────────


def test_result_clamps_high_score():
    r = UnifiedEvaluationResult(score=1.5, achieved=True)
    assert r.score == 1.0


def test_result_clamps_low_score():
    r = UnifiedEvaluationResult(score=-0.5)
    assert r.score == 0.0


def test_result_default_lists_are_empty():
    r = UnifiedEvaluationResult()
    assert r.evidence == []
    assert r.gaps == []
    assert r.category == "unified"
    assert r.source == "UnifiedEvaluator"


def test_result_default_lists_are_independent():
    """两个实例的 evidence 不能共享同一 list (dataclass field 默认值陷阱)."""
    a = UnifiedEvaluationResult()
    b = UnifiedEvaluationResult()
    a.evidence.append("x")
    assert b.evidence == []


# ── from_mcda ──────────────────────────────────────────────────────


def test_from_mcda_object_takes_top_ranked():
    mcda = SimpleNamespace(
        method="entropy-topsis",
        scores={"A": 0.8, "B": 0.3},
        ranking=["A", "B"],
        weights={"c1": 0.5},
    )
    u = UnifiedEvaluator.from_mcda(mcda)
    assert u.score == 0.8
    assert u.achieved is True
    assert u.category == "mcda"
    assert u.source == "entropy-topsis"


def test_from_mcda_dict_with_explicit_alternative():
    mcda = {
        "method": "ahp",
        "scores": {"A": 0.8, "B": 0.3},
        "ranking": ["A", "B"],
    }
    u = UnifiedEvaluator.from_mcda(mcda, alternative="B")
    assert u.score == 0.3
    assert u.achieved is False
    assert any("0.300" in g for g in u.gaps)


def test_from_mcda_none_does_not_raise():
    """None 输入走 getattr 默认值, 不抛, 返回零分结果."""
    u = UnifiedEvaluator.from_mcda(None)
    assert u.score == 0.0
    assert u.achieved is False
    # source = method 默认 "mcda" (无异常, 走默认路径)
    assert u.source == "mcda"


# ── from_goal_judge ────────────────────────────────────────────────


def test_from_goal_judge_dict_achieved():
    gj = {
        "achieved": True,
        "score": 0.85,
        "evidence": ["数值合理"],
        "gaps": [],
    }
    u = UnifiedEvaluator.from_goal_judge(gj)
    assert u.score == 0.85
    assert u.achieved is True
    assert u.evidence == ["数值合理"]
    assert u.source == "GoalJudge"


def test_from_goal_judge_object_with_gaps():
    gj = SimpleNamespace(achieved=False, score=0.2, evidence=[], gaps=["无机制解释"])
    u = UnifiedEvaluator.from_goal_judge(gj)
    assert u.achieved is False
    assert u.gaps == ["无机制解释"]


# ── from_grader ────────────────────────────────────────────────────


def test_from_grader_passed():
    gr = SimpleNamespace(
        name="physics", score=0.6, passed=True,
        checks=[{"issue": "ok"}], message="no findings",
    )
    u = UnifiedEvaluator.from_grader(gr)
    assert u.score == 0.6
    assert u.achieved is True
    assert u.source == "physics"
    assert any("no findings" in e for e in u.evidence)


def test_from_grader_failed_has_gaps():
    gr = SimpleNamespace(
        name="dimensional", score=0.0, passed=False,
        checks=[], message="量纲不一致",
    )
    u = UnifiedEvaluator.from_grader(gr)
    assert u.achieved is False
    assert u.gaps


def test_from_grader_dict_serialized():
    u = UnifiedEvaluator.from_grader({
        "name": "materials_bounds", "score": 0.0, "passed": False,
        "checks": [{"violation": "band_gap_eV=50 outside [0, 10]"}],
        "message": "band_gap_eV=50 outside [0, 10]",
    })
    assert u.source == "materials_bounds"
    assert any("band_gap" in e for e in u.evidence)


# ── from_step_evaluator ────────────────────────────────────────────


def test_from_step_on_track_true():
    step = SimpleNamespace(
        step_id=3, on_track="true", structure_check="passed",
        evidence_quality="high", deviation="", target_chain_ref="T1",
    )
    u = UnifiedEvaluator.from_step_evaluator(step)
    assert u.score == 1.0
    assert u.achieved is True


def test_from_step_unsure_with_deviation():
    step = SimpleNamespace(
        step_id=4, on_track="unsure", structure_check="not_applicable",
        evidence_quality="unknown", deviation="机械信号不足", target_chain_ref=None,
    )
    u = UnifiedEvaluator.from_step_evaluator(step)
    assert u.score == 0.5
    assert u.achieved is False
    assert "机械信号不足" in u.gaps


def test_from_step_false_with_struct_failure():
    step = SimpleNamespace(
        step_id=5, on_track="false", structure_check="failed",
        evidence_quality="low", deviation="脱轨", target_chain_ref=None,
    )
    u = UnifiedEvaluator.from_step_evaluator(step)
    assert u.score == 0.0
    assert "结构不变量检查失败" in u.gaps
    assert "证据质量低" in u.gaps


# ── evaluate() 聚合 ────────────────────────────────────────────────


def test_evaluate_empty_returns_default():
    u = UnifiedEvaluator().evaluate({})
    assert u.score == 0.0
    assert u.achieved is False
    assert u.source == "default"


def test_evaluate_aggregates_multiple_sources():
    ctx = {
        "goal_judge": {"achieved": True, "score": 0.8, "evidence": ["e1"], "gaps": []},
        "grader": [
            SimpleNamespace(name="physics", score=0.7, passed=True, checks=[], message="ok"),
            SimpleNamespace(name="dim", score=0.0, passed=False, checks=[], message="bad"),
        ],
        "step": SimpleNamespace(
            step_id=1, on_track="true", structure_check="passed",
            evidence_quality="high", deviation="", target_chain_ref="T",
        ),
    }
    u = UnifiedEvaluator().evaluate(ctx)
    # grader 内部聚合 (0.7+0.0)/2=0.35; 整体 (0.8+0.35+1.0)/3 ≈ 0.717
    assert 0.5 < u.score < 0.9
    assert u.category == "unified"
    assert "GoalJudge" in u.source and "physics" in u.source
    assert u.achieved is True


def test_evaluate_single_grader_branch():
    u = UnifiedEvaluator().evaluate({
        "grader": [SimpleNamespace(
            name="hallucination", score=1.0, passed=True, checks=[], message="clean"
        )],
    })
    assert u.score == 1.0
    assert u.achieved is True
    assert u.source == "hallucination"


def test_evaluate_all_invalid_inputs_returns_default():
    """所有分支异常时不抛, 返回默认结果."""
    u = UnifiedEvaluator().evaluate({
        "mcda": object(),
        "goal_judge": [1, 2, 3],
        "grader": 12345,
        "step": 99,
    })
    assert isinstance(u, UnifiedEvaluationResult)
    assert u.source == "default"


def test_evaluate_evidence_has_source_prefix():
    """聚合后 evidence 应带 [source] 前缀避免混淆."""
    ctx = {
        "goal_judge": {"achieved": True, "score": 1.0, "evidence": ["e1"], "gaps": []},
    }
    u = UnifiedEvaluator().evaluate(ctx)
    assert any("[GoalJudge]" in e for e in u.evidence)


# ── 异常输入 ──────────────────────────────────────────────────────


@pytest.mark.parametrize("bad", [None, 12345, [1, 2, 3], "string"])
def test_from_grader_bad_input_does_not_raise(bad):
    """无效输入走 getattr 默认值, 不抛, 返回零分结果."""
    u = UnifiedEvaluator.from_grader(bad)
    assert isinstance(u, UnifiedEvaluationResult)
    # 无异常时 source = name, 默认 "grader" (getattr 兜底)
    assert u.source == "grader"
    assert u.score == 0.0
    assert u.achieved is False


@pytest.mark.parametrize("bad", [None, 12345, [1, 2, 3], "string"])
def test_from_step_evaluator_bad_input_does_not_raise(bad):
    """无效输入走 getattr 默认值, 不抛, 返回 unsure (0.5) 默认结果."""
    u = UnifiedEvaluator.from_step_evaluator(bad)
    assert isinstance(u, UnifiedEvaluationResult)
    # 无异常时 source = "StepEvaluator" (默认)
    assert u.source == "StepEvaluator"
    # on_track 默认 "unsure" → score 0.5
    assert u.score == 0.5


def test_threshold_can_be_customized():
    """低门槛下, 中等分应判 achieved."""
    ev = UnifiedEvaluator(threshold=0.3)
    u = ev.evaluate({
        "goal_judge": {"achieved": False, "score": 0.4, "evidence": [], "gaps": ["g"]},
    })
    # score 0.4 >= 0.3 阈值 → achieved True (单分支聚合 = 自身)
    assert u.achieved is True
