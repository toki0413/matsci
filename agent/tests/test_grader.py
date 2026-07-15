"""Grader 层测试 — PhysicsGrader / DimensionalGrader / Registry / GraderResult.

Grader 把 PhysicsAuditor 和 DimensionalValidator 包成统一的
(score, passed, reward) 接口, 喂给 autoloop 奖励回流通道.
"""
from __future__ import annotations

import pytest

from huginn.validation.grader import (
    DimensionalGrader,
    GraderRegistry,
    GraderResult,
    PhysicsGrader,
    ValidityJudge,
    default_registry,
)


# ── PhysicsGrader (包 PhysicsAuditor) ──────────────────────────


def test_physics_grader_clean_result():
    """物理上合理的结果 -> score=1.0, passed=True."""
    g = PhysicsGrader()
    data = {
        "tool_name": "vasp_tool",
        "action": "relax",
        "parsed": {
            "energy": -24.0,        # 负, 合理
            "volume": 50.0,         # 正
            "band_gap": 1.1,        # 正
            "converged": True,
        },
        "input_params": {"n_atoms": 2},
    }
    res = g.evaluate(data)
    assert res.name == "physics"
    assert res.score == pytest.approx(1.0)
    assert res.passed is True
    # reward 默认等于 score
    assert res.reward == res.score


def test_physics_grader_unphysical():
    """正能量 / 负体积 -> error findings -> passed=False, score<1."""
    g = PhysicsGrader()
    data = {
        "tool_name": "vasp_tool",
        "action": "relax",
        "parsed": {
            "energy": 10.0,         # 正能量 -> error (energy/atom>0)
            "volume": -5.0,         # 负体积 -> error
            "converged": True,
        },
        "input_params": {"n_atoms": 2},
    }
    res = g.evaluate(data)
    assert res.passed is False
    assert res.score < 1.0
    # 有 error 级 finding
    severities = [c["severity"] for c in res.checks]
    assert "error" in severities


def test_physics_grader_warning_only():
    """只有 warning 没 error -> passed=True (无硬错), score 折半."""
    g = PhysicsGrader()
    data = {
        "tool_name": "vasp_tool",
        "action": "relax",
        "parsed": {
            "energy": -24.0,
            "volume": 50.0,
            "band_gap": 20.0,        # 很大 -> warning
            "converged": True,
        },
        "input_params": {"n_atoms": 2},
    }
    res = g.evaluate(data)
    # 没硬错 -> passed=True, 但 warning 拉低 score
    assert res.passed is True
    assert 0.0 < res.score < 1.0


# ── DimensionalGrader (包 DimensionalValidator) ────────────────


def test_dimensional_grader_consistent():
    """量纲一致 -> score=1.0, passed=True."""
    g = DimensionalGrader()
    res = g.evaluate({
        "lhs_quantities": ["1.0 N"],
        "rhs_quantities": ["1.0 kg*m/s2"],
        "equation_name": "F=ma",
    })
    assert res.name == "dimensional"
    assert res.score == 1.0
    assert res.passed is True
    assert res.reward == 1.0
    assert len(res.checks) == 1


def test_dimensional_grader_inconsistent():
    """量纲不一致 -> score=0.0, passed=False."""
    g = DimensionalGrader()
    res = g.evaluate({
        "lhs_quantities": ["1.0 N"],
        "rhs_quantities": ["1.0 m"],   # 力 != 长度
        "equation_name": "bad",
    })
    assert res.score == 0.0
    assert res.passed is False


def test_dimensional_grader_missing_input():
    """缺 lhs/rhs -> 优雅失败, score=0."""
    g = DimensionalGrader()
    res = g.evaluate({"equation_name": "incomplete"})
    assert res.score == 0.0
    assert res.passed is False
    assert "missing" in res.message.lower()


# ── GraderRegistry ──────────────────────────────────────────────


def test_grader_registry_register_and_evaluate_all():
    """register 两个 grader, evaluate_all 返回同构结果列表."""
    reg = GraderRegistry()
    reg.register("physics", PhysicsGrader())
    reg.register("dimensional", DimensionalGrader())
    assert set(reg.names()) == {"physics", "dimensional"}

    data = {
        "tool_name": "vasp_tool",
        "action": "relax",
        "parsed": {"energy": -10.0, "volume": 50.0, "converged": True},
        "input_params": {"n_atoms": 2},
        "lhs_quantities": ["1.0 N"],
        "rhs_quantities": ["1.0 kg*m/s2"],
    }
    results = reg.evaluate_all(data)
    assert len(results) == 2
    names = {r.name for r in results}
    assert names == {"physics", "dimensional"}
    # 两个 grader 都通过
    assert all(r.passed for r in results)
    assert all(r.score == 1.0 for r in results)


def test_grader_registry_empty():
    """空 registry evaluate_all 返回空列表."""
    reg = GraderRegistry()
    assert reg.evaluate_all({}) == []
    assert reg.names() == []


def test_default_registry():
    """default_registry 预注册 physics + dimensional + validity."""
    reg = default_registry()
    assert "physics" in reg.names()
    assert "dimensional" in reg.names()
    assert "validity" in reg.names()


# ── GraderResult ────────────────────────────────────────────────


def test_grader_result_reward_defaults_to_score():
    """不显式给 reward 时, reward 等于 score."""
    r = GraderResult(name="x", score=0.75, passed=True)
    assert r.reward == pytest.approx(0.75)


def test_grader_result_explicit_reward():
    """显式给 reward 时, 保留显式值 (调用方加权场景)."""
    r = GraderResult(name="x", score=0.5, passed=True, reward=2.0)
    assert r.reward == 2.0


def test_grader_result_zero_score_no_reward():
    """score=0 时 reward 保持 0 (不凭空产生)."""
    r = GraderResult(name="x", score=0.0, passed=False)
    assert r.reward == 0.0


# ── ValidityJudge (post-hoc LLM judge, 规则降级) ─────────────


def test_validity_judge_no_code_returns_clean():
    """无 code 可审 -> 走规则降级, clean."""
    j = ValidityJudge(model=None)
    res = j.evaluate({"agent_code": "", "conversation_log": ""})
    assert res.name == "validity"
    assert res.score == 1.0
    assert res.passed is True
    assert "no code" in res.message.lower()


def test_validity_judge_rule_fallback_detects_hardcoded():
    """LLM 不可用时, 正则扫到硬编码 band_gap=5.0 -> invalid."""
    j = ValidityJudge(model=None)
    code = """
import numpy as np
def predict_band_gap(features):
    # hardcoded
    band_gap = 5.0
    return np.array([band_gap])
"""
    res = j.evaluate({"agent_code": code, "conversation_log": "", "output_summary": "band_gap=5.0"})
    assert res.score == 0.0
    assert res.passed is False
    assert len(res.checks) >= 1
    msg_lower = res.message.lower()
    assert "hardcoded" in msg_lower or "shortcut" in msg_lower


def test_validity_judge_rule_fallback_clean_code():
    """干净代码 (从数据 fit) -> 规则降级也判 valid."""
    j = ValidityJudge(model=None)
    code = """
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0).fit(X_train, y_train)
y_pred = model.predict(X_test)
"""
    res = j.evaluate({"agent_code": code, "conversation_log": "", "output_summary": "y_pred"})
    assert res.score == 1.0
    assert res.passed is True


def test_validity_judge_parse_verdict_valid_json():
    """_parse_verdict 能解析嵌套 JSON."""
    j = ValidityJudge(model=None)
    content = 'noise {"is_valid": true, "reason": "looks genuine"} trailing'
    parsed = j._parse_verdict(content)
    assert parsed is not None
    is_valid, reason = parsed
    assert is_valid is True
    assert "genuine" in reason


def test_validity_judge_parse_verdict_invalid_json():
    """_parse_verdict 解析 is_valid=false."""
    j = ValidityJudge(model=None)
    content = '{"is_valid": false, "reason": "hardcoded value"}'
    parsed = j._parse_verdict(content)
    assert parsed is not None
    is_valid, reason = parsed
    assert is_valid is False
    assert "hardcoded" in reason


def test_validity_judge_parse_verdict_no_json():
    """无 JSON -> None (走规则降级)."""
    j = ValidityJudge(model=None)
    assert j._parse_verdict("no json here") is None


def test_validity_judge_in_default_registry():
    """default_registry(model=None) 注册的 validity 走规则降级."""
    reg = default_registry(model=None)
    assert "validity" in reg.names()
    results = reg.evaluate_all({"agent_code": "band_gap = 5.0 # hardcoded"})
    valid_res = next(r for r in results if r.name == "validity")
    assert valid_res.passed is False
