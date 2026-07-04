"""Token 限流器的测试.

覆盖三道闸门 (单轮 / 秒级 / 总成本) 的拦截逻辑, 以及用量提取、
成本追踪、滑动窗口裁剪和单例行为. 纯标准库, 不依赖真实 LLM 调用.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from huginn.security.rate_limiter import (
    RateLimitConfig,
    TokenRateLimiter,
    _extract_usage,
    get_rate_limiter,
)
import huginn.security.rate_limiter as rl_mod


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
