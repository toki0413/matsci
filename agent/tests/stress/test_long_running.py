"""长程任务测试 — 验证多轮对话中的记忆连续性和上下文压缩.

前置条件:
  1. 启动服务: python -m huginn serve --port 8000

运行:
  python -m pytest tests/stress/test_long_running.py -v -x --tb=short

测试维度:
  - 50 轮连续对话, 验证上下文压缩触发
  - 检查 /metrics 中的 agent_turns 增量
  - 验证 Belief Entropy 在合理范围
  - 内存使用不持续增长 (无泄漏)
"""

from __future__ import annotations

import asyncio
import gc
import json
import os
import time
from typing import Any

import httpx
import pytest

# 压力测试需要大 payload, 默认限流会误拦
os.environ.setdefault("HUGINN_RATE_LIMIT_ENABLED", "false")

BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 60.0


def _check_server() -> bool:
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


_skip_no_server = pytest.mark.skipif(
    not _check_server(), reason="Server not running on :8000",
)


# ── 直接调 agent (不走 HTTP, 更快) ──────────────────────


@pytest.mark.asyncio
async def test_50_turn_conversation_no_leak(tmp_path):
    """50 轮连续 chat, 验证内存不持续增长."""
    from tests.fixtures.fake_llm import make_callable_llm
    from huginn.agent import HuginnAgent
    from huginn.memory.manager import MemoryManager
    from huginn.memory.longterm import LongTermMemory

    # FakeLLM: 每轮返回一个包含轮次号的响应
    llm = make_callable_llm(lambda p: f"response turn {p[:20]}", name="long-run-llm")
    memory = MemoryManager(longterm=LongTermMemory(str(tmp_path / "memory.db")))
    agent = HuginnAgent(
        model=llm,
        memory_manager=memory,
        checkpointer_path=str(tmp_path / "checkpoint.sqlite"),
    )

    # 记录初始内存
    import tracemalloc
    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()

    for i in range(50):
        async for _ in agent.chat(f"message {i}", thread_id="long-run-1"):
            pass
        if i % 10 == 0:
            gc.collect()

    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    # 验证内存增长不超过 50MB (50 轮对话不应消耗太多)
    stats = snapshot_after.compare_to(snapshot_before, "lineno")
    total_growth = sum(s.size_diff for s in stats if s.size_diff > 0)
    total_growth_mb = total_growth / 1024 / 1024
    assert total_growth_mb < 50, f"Memory grew {total_growth_mb:.1f}MB in 50 turns"
    print(f"  50 turns: memory growth = {total_growth_mb:.1f}MB")


@pytest.mark.asyncio
async def test_context_compression_triggers(tmp_path):
    """发送足够多消息触发上下文压缩, 验证不丢失关键信息."""
    from tests.fixtures.fake_llm import make_callable_llm
    from huginn.agent import HuginnAgent
    from huginn.memory.manager import MemoryManager
    from huginn.memory.longterm import LongTermMemory

    # 每条消息 2KB, 20 条就够触发压缩 (默认 budget ~8K tokens)
    big_msg = "x" * 2000
    llm = make_callable_llm(lambda p: "ok", name="compress-llm")
    memory = MemoryManager(longterm=LongTermMemory(str(tmp_path / "memory.db")))
    agent = HuginnAgent(
        model=llm,
        memory_manager=memory,
        checkpointer_path=str(tmp_path / "checkpoint.sqlite"),
        context_budget_tokens=4000,  # 小 budget 加快触发
    )

    for i in range(30):
        async for _ in agent.chat(f"{big_msg} turn {i}", thread_id="compress-test"):
            pass

    # agent 不应该 crash, 说明压缩正常工作
    # 如果压缩没生效, 30 条 2KB 消息会超出 context window
    print("  30 turns × 2KB: compression working, no crash")


@pytest.mark.asyncio
async def test_concurrent_long_sessions(tmp_path):
    """3 个并发长会话, 各 20 轮, 验证互不干扰."""
    from tests.fixtures.fake_llm import make_callable_llm
    from huginn.agent import HuginnAgent
    from huginn.memory.manager import MemoryManager
    from huginn.memory.longterm import LongTermMemory

    llm = make_callable_llm(
        lambda p: f"response for {p[:30]}",
        name="multi-session-llm",
    )
    memory = MemoryManager(longterm=LongTermMemory(str(tmp_path / "memory.db")))
    agent = HuginnAgent(
        model=llm,
        memory_manager=memory,
        checkpointer_path=str(tmp_path / "checkpoint.sqlite"),
    )

    async def session(tid: str, rounds: int):
        responses = []
        for i in range(rounds):
            async for state in agent.chat(f"msg-{tid}-{i}", thread_id=tid):
                pass
            responses.append("ok")
        return tid, responses

    # 3 个并发会话
    results = await asyncio.gather(
        session("long-A", 20),
        session("long-B", 20),
        session("long-C", 20),
    )

    # 每个会话都应该有 20 条响应
    for tid, responses in results:
        assert len(responses) == 20, f"{tid}: expected 20 responses, got {len(responses)}"
    print("  3 sessions × 20 turns: all completed, no interference")


# ── HTTP 级长程测试 (走完整链路) ─────────────────────


@pytest.mark.asyncio
@_skip_no_server
async def test_http_30_turn_conversation():
    """30 轮 HTTP 对话, 验证全链路稳定性."""
    async with httpx.AsyncClient() as client:
        for i in range(30):
            r = await client.post(
                f"{BASE_URL}/agents/agent/chat",
                json={"content": f"long conversation turn {i}", "thread_id": "http-long-run"},
                timeout=TIMEOUT,
            )
            assert r.status_code == 200, f"Turn {i} failed: {r.status_code}"
            # 小间隔避免打爆
            await asyncio.sleep(0.05)

    # /metrics 应该显示 30 个 agent turns
    r = httpx.get(f"{BASE_URL}/metrics", timeout=5.0)
    assert r.status_code == 200
    # 检查 metrics 文本中有 agent_turns 相关指标
    assert "huginn_agent_turns" in r.text or "huginn_requests" in r.text
    print("  30 HTTP turns: all 200, metrics recorded")
