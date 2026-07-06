"""Tests for enhanced ClarificationManager + ClarifyQuestionsHook.

Covers:
- should_ask_contextual (cooldown, failures, cost, timeout rate)
- generate_question (template fallback, LLM path skipped on mock)
- _record_stats tracking
- ClarifyQuestionsHook cooldown (re-askable after cooldown expires)
- ClarifyQuestionsHook cost/multi-path detection
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from huginn.interaction.clarification import ClarificationManager
from huginn.hooks.clarify_questions_hook import ClarifyQuestionsHook


# ── should_ask_contextual ──────────────────────────────────────


def test_contextual_blocks_on_cooldown():
    """同 thread 60s 内不重复问."""
    mgr = ClarificationManager()
    # 模拟刚刚问过
    mgr._last_ask_time["t1"] = time.time()
    assert not mgr.should_ask_contextual("cost_confirm", {"thread_id": "t1"})


def test_contextual_allows_after_cooldown():
    """冷却期过后允许再问."""
    mgr = ClarificationManager()
    mgr._last_ask_time["t1"] = time.time() - 61
    assert mgr.should_ask_contextual("cost_confirm", {"thread_id": "t1"})


def test_contextual_forces_on_consecutive_failures():
    """连续 3+ 次失败时强制触发, 绕过 cooldown."""
    mgr = ClarificationManager()
    mgr._last_ask_time["t1"] = time.time()  # 刚问过
    ctx = {"thread_id": "t1", "consecutive_failures": 3}
    assert mgr.should_ask_contextual("validation_fail", ctx)


def test_contextual_forces_on_high_cost():
    """高成本 (>=1h) 时强制触发."""
    mgr = ClarificationManager()
    mgr._last_ask_time["t1"] = time.time()
    ctx = {"thread_id": "t1", "cost_estimate_hours": 2.0}
    assert mgr.should_ask_contextual("cost_confirm", ctx)


def test_contextual_reduces_on_high_timeout_rate():
    """超时率 > 70% 的类型降频."""
    mgr = ClarificationManager()
    mgr._stats["task_vague"] = {"asked": 10, "answered": 2, "timed_out": 8}
    assert not mgr.should_ask_contextual("task_vague", {"thread_id": "t2"})


def test_contextual_respects_pending_limit():
    """3+ 个 pending 时不允许再问."""
    mgr = ClarificationManager()
    mgr._by_thread["t3"] = ["q1", "q2", "q3"]
    assert not mgr.should_ask_contextual("cost_confirm", {"thread_id": "t3"})


# ── generate_question ──────────────────────────────────────────


def test_generate_question_cost_confirm():
    """模板: cost_confirm 类型生成确认问题."""
    mgr = ClarificationManager()
    q, opts, default = mgr.generate_question({
        "question_type": "cost_confirm",
        "phase": "plan",
        "summary": "VASP relaxation",
        "tool": "vasp",
    })
    assert "确认" in q or "执行" in q
    assert "确认执行" in opts
    assert default == "确认执行"


def test_generate_question_multi_path():
    """模板: multi_path 类型生成选择问题."""
    mgr = ClarificationManager()
    paths = ["GGA-PBE", "HSE06"]
    q, opts, default = mgr.generate_question({
        "question_type": "multi_path",
        "phase": "plan",
        "summary": "choose functional",
        "options_hint": paths,
    })
    assert "GGA-PBE" in q
    assert opts == paths
    assert default == "GGA-PBE"


def test_generate_question_validation_fail():
    """模板: validation_fail 类型生成方向问题."""
    mgr = ClarificationManager()
    q, opts, default = mgr.generate_question({
        "question_type": "validation_fail",
        "phase": "validate",
        "summary": "tests failed: energy not converged",
        "consecutive_failures": 3,
    })
    assert "3" in q
    assert "修正假设重新实验" in opts
    assert default == "继续当前路径"


def test_generate_question_falls_back_on_mock_model():
    """mock model (有 _mock_name) 不走 LLM 路径, 降级到模板."""
    mgr = ClarificationManager()
    mock_model = MagicMock()
    mock_model._mock_name = "MockModel"
    q, opts, default = mgr.generate_question(
        {"question_type": "cost_confirm", "summary": "VASP", "tool": "vasp"},
        model=mock_model,
    )
    # 走了模板
    assert "确认" in q or "执行" in q


# ── _record_stats ──────────────────────────────────────────────


def test_record_stats_tracks_outcomes():
    """ask 后记录 answered/timed_out."""
    mgr = ClarificationManager(default_timeout=0.1)

    # 模拟超时: 不 resolve, 等 timeout
    async def _timeout_ask():
        return await mgr.ask(
            thread_id="stats_t",
            question="test",
            default_answer="default",
            timeout=0.1,
            metadata={"question_type": "cost_confirm"},
        )

    asyncio.run(_timeout_ask())
    stats = mgr._stats.get("cost_confirm", {})
    assert stats.get("asked", 0) >= 1
    assert stats.get("timed_out", 0) >= 1


# ── ClarifyQuestionsHook cooldown ──────────────────────────────


def test_hook_cooldown_blocks_within_period():
    """冷却期内不重复追问."""
    hook = ClarifyQuestionsHook()
    # 第一次触发
    ctx1 = MagicMock()
    ctx1.metadata = {
        "user_message": "帮我分析一下",
        "thread_id": "thread_cooldown",
    }
    asyncio.run(hook(ctx1))
    # 冷却期内第二次: 不应追问
    ctx2 = MagicMock()
    ctx2.metadata = {
        "user_message": "帮我研究一下",
        "thread_id": "thread_cooldown",
    }
    result = asyncio.run(hook(ctx2))
    assert result is None
    assert "clarify_questions" not in ctx2.metadata


def test_hook_cooldown_allows_after_expiry():
    """冷却期过后允许追问."""
    hook = ClarifyQuestionsHook()
    hook._COOLDOWN_SEC = 0  # 立即过期
    ctx1 = MagicMock()
    ctx1.metadata = {
        "user_message": "帮我分析一下",
        "thread_id": "thread_expiry",
    }
    asyncio.run(hook(ctx1))
    # 冷却已过, 第二次应追问
    ctx2 = MagicMock()
    ctx2.metadata = {
        "user_message": "帮我研究一下",
        "thread_id": "thread_expiry",
    }
    asyncio.run(hook(ctx2))
    assert "clarify_questions" in ctx2.metadata


# ── ClarifyQuestionsHook cost/multi-path ───────────────────────


def test_hook_detects_expensive_tool():
    """VASP 关键词触发 cost 追问."""
    hook = ClarifyQuestionsHook()
    hook._COOLDOWN_SEC = 0
    questions = hook._detect_and_generate("帮我跑一下 VASP")
    assert any("成本" in q or "确认" in q for q in questions)


def test_hook_detects_multi_path():
    """还是 触发 multi_path 追问."""
    hook = ClarifyQuestionsHook()
    hook._COOLDOWN_SEC = 0
    questions = hook._detect_and_generate("用 GGA-PBE 还是 HSE06 做弛豫？")
    assert any("路径" in q or "方案" in q for q in questions)


def test_hook_no_trigger_on_clear_request():
    """明确请求不触发追问."""
    hook = ClarifyQuestionsHook()
    hook._COOLDOWN_SEC = 0
    questions = hook._detect_and_generate(
        "用 VASP 对 SiO2 做 ENCUT=520 eV 的结构弛豫，输出 CONTCAR"
    )
    # 有具体材料名和参数, 不触发 cost 追问
    assert not any("成本" in q for q in questions)
