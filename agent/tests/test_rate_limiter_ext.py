"""rate_limiter.py 全分支测试 — 三道闸门、会话隔离、记账、预警、单例."""

from __future__ import annotations

import time

import pytest

from huginn.security.rate_limiter import (
    RateLimitConfig,
    RateLimitExceeded,
    TokenRateLimiter,
    _build_from_env,
    _detect_model_name,
    _estimate_input_tokens,
    _extract_usage,
    get_rate_limiter,
)

# ── disabled ─────────────────────────────────────────────────────────────

def test_disabled_always_allows():
    l = TokenRateLimiter(RateLimitConfig(enabled=False))
    ok, _ = l.check_allowed("m", 10**9)
    assert ok is True
    l.record_usage("m", 10**9, 10**9)
    assert l.get_stats()["total_tokens"] == 0


# ── 三道闸门 ─────────────────────────────────────────────────────────────

def test_turn_limit_gate():
    l = TokenRateLimiter(RateLimitConfig(max_tokens_per_turn=100))
    ok, reason = l.check_allowed("m", 101)
    assert ok is False
    assert "单轮 token 超限" in reason


def test_second_limit_gate():
    l = TokenRateLimiter(RateLimitConfig(max_tokens_per_second=100))
    ok, reason = l.check_allowed("m", 101)
    assert ok is False
    assert "秒级 token 超限" in reason


def test_cost_limit_gate():
    l = TokenRateLimiter(RateLimitConfig(max_total_cost_usd=1.0))
    l._total_cost = 1.0
    ok, reason = l.check_allowed("m", 100)
    assert ok is False
    assert "总成本超限" in reason


def test_ok_when_under_all_limits():
    l = TokenRateLimiter()
    ok, reason = l.check_allowed("m", 100)
    assert ok is True
    assert reason == ""


# ── 会话隔离 ─────────────────────────────────────────────────────────────

def test_per_session_isolation():
    l = TokenRateLimiter(RateLimitConfig(max_tokens_per_turn=100))
    l.record_usage("m", 90, 0, thread_id="t1")
    # t1 已用 90, 再加 20 超 100
    ok, _ = l.check_allowed("m", 20, thread_id="t1")
    assert ok is False
    # t2 独立, 不受影响
    ok2, _ = l.check_allowed("m", 20, thread_id="t2")
    assert ok2 is True


def test_second_window_prunes():
    l = TokenRateLimiter(RateLimitConfig(max_tokens_per_second=100))
    l.record_usage("m", 60, 0)
    # 手动把窗口时间拨到 2 秒前 → 应被剪掉
    s = l._get_session("default")
    s["second_window"][0] = (time.time() - 2.0, 60)
    ok, _ = l.check_allowed("m", 50)
    assert ok is True


# ── record_usage / stats ─────────────────────────────────────────────────

def test_record_usage_totals():
    l = TokenRateLimiter()
    l.record_usage("m", 10, 20, cost=0.5)
    stats = l.get_stats()
    assert stats["total_tokens"] == 30
    assert stats["total_cost"] == pytest.approx(0.5)


def test_record_usage_per_model():
    l = TokenRateLimiter()
    l.record_usage("m1", 10, 5, cost=0.1)
    l.record_usage("m1", 10, 5, cost=0.1)
    stats = l.get_stats()
    assert stats["per_model"]["m1"]["calls"] == 2
    assert stats["per_model"]["m1"]["cost"] == pytest.approx(0.2)


def test_record_usage_per_session_turn():
    l = TokenRateLimiter()
    l.record_usage("m", 10, 0, thread_id="t1")
    l.record_usage("m", 10, 0, thread_id="t2")
    stats = l.get_stats()
    assert stats["turn_tokens"] == 20
    assert len(stats["active_sessions"]) == 2


def test_get_stats_limits():
    l = TokenRateLimiter(RateLimitConfig(max_tokens_per_turn=50))
    stats = l.get_stats()
    assert stats["limits"]["max_tokens_per_turn"] == 50


# ── reset ────────────────────────────────────────────────────────────────

def test_reset_turn_clears_session_only():
    l = TokenRateLimiter()
    l.record_usage("m", 100, 0, cost=1.0)
    l.reset_turn()
    stats = l.get_stats()
    assert stats["turn_tokens"] == 0
    assert stats["total_tokens"] == 100  # 全局不动


def test_reset_all_clears_everything():
    l = TokenRateLimiter()
    l.record_usage("m", 100, 0, cost=1.0)
    l.reset_all()
    stats = l.get_stats()
    assert stats["total_tokens"] == 0
    assert stats["total_cost"] == 0
    assert stats["active_sessions"] == {}


# ── 预警 ─────────────────────────────────────────────────────────────────

def test_warning_threshold_zero_disables():
    l = TokenRateLimiter(RateLimitConfig(warning_threshold=0.0))
    ok, _ = l.check_allowed("m", 100)
    assert ok is True


def test_warning_only_once_per_dimension():
    l = TokenRateLimiter(RateLimitConfig(max_tokens_per_turn=1000, warning_threshold=0.5))
    # 第一次触发预警
    l.check_allowed("m", 600)
    s = l._get_session("default")
    assert "turn" in s["warned"]
    # reset_turn 清空 warned
    l.reset_turn()
    s = l._get_session("default")
    assert "turn" not in s["warned"]


# ── RateLimitExceeded ────────────────────────────────────────────────────

def test_rate_limit_exceeded_reason():
    e = RateLimitExceeded("msg", reason="cost_limit")
    assert e.reason == "cost_limit"
    assert str(e) == "msg"


def test_rate_limit_exceeded_default_reason():
    e = RateLimitExceeded("msg")
    assert e.reason == "limit_exceeded"


# ── 辅助函数 ─────────────────────────────────────────────────────────────

def test_detect_model_name_from_attr():
    class M:
        model_name = "claude"

    assert _detect_model_name(M()) == "claude"


def test_detect_model_name_fallback_class():
    assert _detect_model_name(object()) == "object"


def test_estimate_input_tokens_str():
    n = _estimate_input_tokens("hello world")
    assert n >= 1


def test_estimate_input_tokens_list():
    n = _estimate_input_tokens(["aaaa", "bbbb"])
    assert n >= 2


def test_estimate_input_tokens_empty():
    assert _estimate_input_tokens("") == 1


def test_extract_usage_from_usage_metadata():
    class Msg:
        usage_metadata = {"input_tokens": 5, "output_tokens": 3}

    assert _extract_usage(Msg()) == (5, 3)


def test_extract_usage_from_response_metadata_dict():
    class Msg:
        response_metadata = {"token_usage": {"prompt_tokens": 7, "completion_tokens": 2}}

    msg = Msg()
    assert _extract_usage(msg) == (7, 2)


def test_extract_usage_from_dict():
    assert _extract_usage({"input_tokens": 4, "output_tokens": 1}) == (4, 1)


def test_extract_usage_none():
    assert _extract_usage(None) == (0, 0)


# ── 单例 / env ───────────────────────────────────────────────────────────

def test_get_rate_limiter_singleton():
    a = get_rate_limiter()
    b = get_rate_limiter()
    assert a is b


def test_build_from_env(monkeypatch):
    monkeypatch.setenv("HUGINN_RATE_LIMIT_TOKENS_PER_TURN", "100")
    monkeypatch.setenv("HUGINN_RATE_LIMIT_ENABLED", "0")
    l = _build_from_env()
    assert l.config.max_tokens_per_turn == 100
    assert l.config.enabled is False
