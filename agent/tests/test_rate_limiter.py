"""Token 限流器的测试.

覆盖三道闸门 (单轮 / 秒级 / 总成本) 的拦截逻辑, 以及用量提取、
成本追踪、滑动窗口裁剪和单例行为. 纯标准库, 不依赖真实 LLM 调用.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

import huginn.security.rate_limiter as rl_mod
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

# ── check_allowed ──────────────────────────────────────────


def test_check_allowed_under_limit() -> None:
    # 全新限流器, 1000 token 远低于默认上限
    limiter = TokenRateLimiter(RateLimitConfig())
    ok, reason = limiter.check_allowed("test-model", 1000)
    assert ok is True
    assert reason == ""


def test_check_allowed_over_turn_limit() -> None:
    # 单轮上限设 100, 已用 100, 再来 1 个就超
    cfg = RateLimitConfig(
        max_tokens_per_turn=100,
        max_tokens_per_second=100_000,  # 抬高秒级, 别让它先拦
        max_total_cost_usd=1000.0,
    )
    limiter = TokenRateLimiter(cfg)
    limiter.record_usage("test-model", input_tokens=100, output_tokens=0)
    ok, reason = limiter.check_allowed("test-model", 1)
    assert ok is False
    assert "单轮" in reason


# ── record_usage / get_stats ───────────────────────────────


def test_record_usage() -> None:
    limiter = TokenRateLimiter(RateLimitConfig())
    limiter.record_usage("test-model", input_tokens=100, output_tokens=50)
    stats = limiter.get_stats()
    assert stats["turn_tokens"] == 150
    assert stats["total_tokens"] == 150
    per_model = stats["per_model"]["test-model"]
    assert per_model["input_tokens"] == 100
    assert per_model["output_tokens"] == 50
    assert per_model["calls"] == 1


# ── reset ──────────────────────────────────────────────────


def test_reset_turn() -> None:
    cfg = RateLimitConfig(max_tokens_per_turn=100, max_tokens_per_second=100_000)
    limiter = TokenRateLimiter(cfg)
    limiter.record_usage("m", 100, 0)
    ok, _ = limiter.check_allowed("m", 1)
    assert ok is False
    # 新一轮 turn, 单轮计数清零
    limiter.reset_turn()
    ok, _ = limiter.check_allowed("m", 1)
    assert ok is True


def test_reset_all() -> None:
    limiter = TokenRateLimiter(RateLimitConfig())
    limiter.record_usage("m", 100, 50, cost=0.05)
    limiter.reset_all()
    stats = limiter.get_stats()
    assert stats["turn_tokens"] == 0
    assert stats["total_tokens"] == 0
    assert stats["total_cost"] == 0.0
    assert stats["per_model"] == {}


# ── 滑动窗口 ──────────────────────────────────────────────


def test_sliding_window() -> None:
    limiter = TokenRateLimiter(RateLimitConfig())
    # 手动塞一条 2 秒前的旧记录进 per-session 窗口
    old_ts = time.time() - 2.0
    with limiter._lock:
        s = limiter._get_session("default")
        s["second_window"].append((old_ts, 5000))
    # 记一笔新的, record_usage 内部会调 _prune_session_window 把旧的裁掉
    limiter.record_usage("m", 100, 50)
    stats = limiter.get_stats()
    # 旧的 5000 应该被清了, 只剩新的 150
    assert stats["active_sessions"]["default"]["tokens_per_second"] == 150


# ── 成本追踪 ──────────────────────────────────────────────


def test_cost_tracking() -> None:
    limiter = TokenRateLimiter(RateLimitConfig())
    limiter.record_usage("m", 100, 50, cost=0.05)
    assert limiter.get_stats()["total_cost"] == pytest.approx(0.05)
    limiter.record_usage("m", 100, 50, cost=0.03)
    assert limiter.get_stats()["total_cost"] == pytest.approx(0.08)


# ── 用量提取 ──────────────────────────────────────────────


def test_extract_usage_langchain() -> None:
    # 模拟 LangChain AIMessage, 带 usage_metadata
    msg = SimpleNamespace(
        usage_metadata={"input_tokens": 100, "output_tokens": 50}
    )
    in_tok, out_tok = _extract_usage(msg)
    assert in_tok == 100
    assert out_tok == 50


def test_extract_usage_anthropic() -> None:
    # 模拟 Anthropic 风格返回, usage 塞在 response_metadata 顶层
    msg = SimpleNamespace(
        response_metadata={"input_tokens": 200, "output_tokens": 100}
    )
    in_tok, out_tok = _extract_usage(msg)
    assert in_tok == 200
    assert out_tok == 100


# ── 单例 ──────────────────────────────────────────────────


def test_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rl_mod, "_singleton", None)
    a = get_rate_limiter()
    b = get_rate_limiter()
    assert a is b


# ── 全分支扩展 (原 test_rate_limiter_ext.py) ──────────────────────────────

def test_disabled_always_allows():
    l = TokenRateLimiter(RateLimitConfig(enabled=False))
    ok, _ = l.check_allowed("m", 10**9)
    assert ok is True
    l.record_usage("m", 10**9, 10**9)
    assert l.get_stats()["total_tokens"] == 0


def test_turn_limit_gate():
    l = TokenRateLimiter(RateLimitConfig(max_tokens_per_turn=100))
    ok, reason = l.check_allowed("m", 101)
    assert ok is False
    assert "单轮 token 超限" in reason


def test_second_limit_gate():
    # 秒级闸门按"最近 1s 实际消耗"判定. 窗口内已实际消费 120 > 上限 100,
    # 下一个请求会被拦.
    l = TokenRateLimiter(RateLimitConfig(max_tokens_per_second=100))
    l.record_usage("m", 120, 0)
    ok, reason = l.check_allowed("m", 1)
    assert ok is False
    assert "秒级" in reason


def test_second_limit_does_not_block_large_single_request():
    # 回归 (长程研究 / extreme): 上下文大, 单次请求输入 9000 > 秒级默认 5000,
    # 但窗口近 1s 实际消耗为 0 —— 这是"一个大的慢请求", 不是"每秒失控速率",
    # 必须放行. 之前把 est 加进秒级判定, 导致长程任务几分钟内被误拦.
    l = TokenRateLimiter(RateLimitConfig(max_tokens_per_second=5000))
    ok, reason = l.check_allowed("m", 9000)
    assert ok is True
    assert reason == ""


def test_second_limit_still_catches_fast_burst():
    # 秒级闸门仍要拦住"快速循环": 窗口内连续完成多个请求, 实际消耗超过上限.
    l = TokenRateLimiter(RateLimitConfig(max_tokens_per_second=100))
    l.record_usage("m", 60, 0)
    l.record_usage("m", 60, 0)  # 窗口内累计 120 > 100
    ok, reason = l.check_allowed("m", 1)
    assert ok is False
    assert "秒级" in reason


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


def test_per_session_isolation():
    l = TokenRateLimiter(RateLimitConfig(max_tokens_per_turn=100))
    l.record_usage("m", 90, 0, thread_id="t1")
    ok, _ = l.check_allowed("m", 20, thread_id="t1")
    assert ok is False
    ok2, _ = l.check_allowed("m", 20, thread_id="t2")
    assert ok2 is True


def test_second_window_prunes():
    l = TokenRateLimiter(RateLimitConfig(max_tokens_per_second=100))
    l.record_usage("m", 60, 0)
    s = l._get_session("default")
    s["second_window"][0] = (time.time() - 2.0, 60)
    ok, _ = l.check_allowed("m", 50)
    assert ok is True


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


def test_warning_threshold_zero_disables():
    l = TokenRateLimiter(RateLimitConfig(warning_threshold=0.0))
    ok, _ = l.check_allowed("m", 100)
    assert ok is True


def test_warning_only_once_per_dimension():
    l = TokenRateLimiter(RateLimitConfig(max_tokens_per_turn=1000, warning_threshold=0.5))
    l.check_allowed("m", 600)
    s = l._get_session("default")
    assert "turn" in s["warned"]
    l.reset_turn()
    s = l._get_session("default")
    assert "turn" not in s["warned"]


def test_rate_limit_exceeded_reason():
    e = RateLimitExceeded("msg", reason="cost_limit")
    assert e.reason == "cost_limit"
    assert str(e) == "msg"


def test_rate_limit_exceeded_default_reason():
    e = RateLimitExceeded("msg")
    assert e.reason == "limit_exceeded"


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
