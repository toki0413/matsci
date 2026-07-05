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
    """default_registry 预注册 physics + dimensional."""
    reg = default_registry()
    assert "physics" in reg.names()
    assert "dimensional" in reg.names()


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
