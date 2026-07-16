"""KRCL plan_check 自检 — 反向校验 JSON 解析 + trivial plan 跳过 + 闭环 refine.

只测纯函数 (无 LLM 调用), 真实 LLM 路径靠 autoloop 集成测试覆盖.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


# ── _parse_plan_check (括号配平 JSON 解析) ──────────────────────


def _make_engine():
    """造一个最小 AutoloopEngine 实例, 只为了调 _parse_plan_check.

    ponytail: 不走完整 __init__, 用 __new__ 绕过依赖注入.
    """
    from huginn.autoloop.engine import AutoloopEngine
    eng = AutoloopEngine.__new__(AutoloopEngine)
    return eng


def test_parse_plan_check_valid_json():
    """valid JSON + 完整字段 -> 正常解析."""
    eng = _make_engine()
    resp = '{"is_valid": false, "reason": "missing SCF", "missing_steps": ["SCF"], "risks": ["band gap wrong"]}'
    res = eng._parse_plan_check(resp)
    assert res["is_valid"] is False
    assert res["reason"] == "missing SCF"
    assert res["missing_steps"] == ["SCF"]


def test_parse_plan_check_nested_json():
    """嵌套 JSON 也能解析 (括号配平法)."""
    eng = _make_engine()
    resp = 'noise {"is_valid": true, "reason": "ok", "missing_steps": [], "risks": [{"type": "low"}]} trailing'
    res = eng._parse_plan_check(resp)
    assert res["is_valid"] is True
    assert res["risks"] == [{"type": "low"}]


def test_parse_plan_check_missing_fields():
    """JSON 缺字段 -> setdefault 补全."""
    eng = _make_engine()
    resp = '{"is_valid": false, "reason": "bad mode"}'
    res = eng._parse_plan_check(resp)
    assert res["is_valid"] is False
    assert res["missing_steps"] == []
    assert res["risks"] == []


def test_parse_plan_check_no_json():
    """无 JSON -> is_valid=True (跳过, 不阻塞)."""
    eng = _make_engine()
    res = eng._parse_plan_check("no json here")
    assert res["is_valid"] is True
    assert "skip" in res["reason"]


def test_parse_plan_check_broken_json():
    """JSON 解析失败 -> is_valid=True (不阻塞)."""
    eng = _make_engine()
    res = eng._parse_plan_check('{"is_valid": false, "reason":}')
    assert res["is_valid"] is True
    assert "skip" in res["reason"] or "failed" in res["reason"]


def test_parse_plan_check_no_closing_brace():
    """JSON 没闭合 -> is_valid=True."""
    eng = _make_engine()
    res = eng._parse_plan_check('{"is_valid": false, "reason": "missing')
    assert res["is_valid"] is True


# ── _plan_check_and_refine (闭环 + trivial plan 跳过) ─────────


def test_trivial_plan_skips_check():
    """description 太短 (<20 chars) 跳过校验, 不调 LLM."""
    eng = _make_engine()
    eng._plan_check = AsyncMock()
    plan = {"mode": "coder", "description": "short"}  # 5 chars
    result = asyncio.get_event_loop().run_until_complete(
        eng._plan_check_and_refine(plan, "hypothesis", {})
    )
    assert result is plan  # 原样返回
    eng._plan_check.assert_not_called()


def test_valid_plan_passes_first_try():
    """is_valid=True -> 直接返回, plan 带 plan_check 字段."""
    eng = _make_engine()
    eng._plan_check = AsyncMock(return_value={
        "is_valid": True, "reason": "ok", "missing_steps": [], "risks": []
    })
    plan = {"mode": "coder", "description": "run SCF then band calculation"}
    result = asyncio.get_event_loop().run_until_complete(
        eng._plan_check_and_refine(plan, "calc band gap", {})
    )
    assert result["plan_check"]["is_valid"] is True
    assert "plan_check_warning" not in result


def test_invalid_plan_refines_once_then_gives_up():
    """两次 is_valid=False -> 1 次 refine 后标 warning 继续 (不阻塞)."""
    eng = _make_engine()
    eng._plan_check = AsyncMock(return_value={
        "is_valid": False, "reason": "missing SCF",
        "missing_steps": ["SCF"], "risks": [],
    })
    eng._refine_plan = AsyncMock(return_value={
        "mode": "workflow", "description": "SCF then band",
    })
    eng._override_plan_mode = MagicMock(side_effect=lambda p: p)
    plan = {"mode": "coder", "description": "run band calculation directly"}
    result = asyncio.get_event_loop().run_until_complete(
        eng._plan_check_and_refine(plan, "calc band gap", {})
    )
    # 重试 1 次, 总共调 _plan_check 2 次
    assert eng._plan_check.call_count == 2
    assert eng._refine_plan.call_count == 1
    # 最终 plan 带 warning, 不阻塞
    assert "plan_check_warning" in result
    assert "missing SCF" in result["plan_check_warning"]


def test_llm_failure_returns_plan():
    """_plan_check 抛异常 -> 直接返回原 plan (不阻塞)."""
    eng = _make_engine()
    eng._plan_check = AsyncMock(side_effect=Exception("LLM down"))
    plan = {"mode": "coder", "description": "run SCF calculation on Si"}
    result = asyncio.get_event_loop().run_until_complete(
        eng._plan_check_and_refine(plan, "calc band gap", {})
    )
    assert result is plan
    assert "plan_check" not in result


# ── _build_plan_check_prompt (纯函数) ──────────────────────────


def test_build_prompt_includes_key_fields():
    """prompt 包含 hypothesis / mode / description / prediction."""
    eng = _make_engine()
    eng._speculator_hint = ""
    plan = {
        "mode": "coder",
        "description": "run VASP SCF on Si",
        "expected_prediction": "band_gap=1.1eV",
    }
    prompt = eng._build_plan_check_prompt(plan, "calc Si band gap", {})
    assert "calc Si band gap" in prompt
    assert "coder" in prompt
    assert "run VASP SCF on Si" in prompt
    assert "band_gap=1.1eV" in prompt
