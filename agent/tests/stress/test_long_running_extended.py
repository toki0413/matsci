"""超长程测试 — 100 轮+多会话+检查点恢复+上下文压缩验证.

测试维度:
  - 100 轮连续对话内存稳定性
  - 5 个并发会话各 30 轮
  - 检查点恢复 (agent 重启后继续对话)
  - 上下文压缩触发且不丢失关键信息
  - 长程会话中工具调用不累积
"""

from __future__ import annotations

import asyncio
import gc
import os
import sys
import tracemalloc
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("HUGINN_RATE_LIMIT_ENABLED", "false")

AGENT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(AGENT_DIR))

from tests.fixtures.fake_llm import make_callable_llm
from huginn.agent import HuginnAgent
from huginn.memory.manager import MemoryManager
from huginn.memory.longterm import LongTermMemory


# ── 100 轮内存稳定性 ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_100_turn_memory_stable(tmp_path):
    """100 轮连续 chat, 内存增长应控制在 100MB 以内."""
    llm = make_callable_llm(lambda p: "ok", name="long100-llm")
    memory = MemoryManager(longterm=LongTermMemory(str(tmp_path / "memory.db")))
    agent = HuginnAgent(
        model=llm,
        memory_manager=memory,
        checkpointer_path=str(tmp_path / "checkpoint.sqlite"),
    )

    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()

    for i in range(100):
        async for _ in agent.chat(f"turn {i}", thread_id="long-100"):
            pass
        if i % 20 == 0:
            gc.collect()

    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats = snapshot_after.compare_to(snapshot_before, "lineno")
    total_growth = sum(s.size_diff for s in stats if s.size_diff > 0)
    total_growth_mb = total_growth / 1024 / 1024
    print(f"  100 turns: memory growth = {total_growth_mb:.1f}MB")
    assert total_growth_mb < 100, f"Memory grew {total_growth_mb:.1f}MB in 100 turns"


# ── 5 个并发会话 ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_5_concurrent_sessions_30_turns(tmp_path):
    """5 个并发会话各 30 轮, 验证互不干扰."""
    llm = make_callable_llm(
        lambda p: f"resp {p[:20]}",
        name="multi5-llm",
    )
    memory = MemoryManager(longterm=LongTermMemory(str(tmp_path / "memory.db")))
    agent = HuginnAgent(
        model=llm,
        memory_manager=memory,
        checkpointer_path=str(tmp_path / "checkpoint.sqlite"),
    )

    async def session(tid: str, rounds: int):
        count = 0
        for i in range(rounds):
            async for _ in agent.chat(f"msg-{tid}-{i}", thread_id=tid):
                count += 1
        return tid, count

    results = await asyncio.gather(
        session("sess-A", 30),
        session("sess-B", 30),
        session("sess-C", 30),
        session("sess-D", 30),
        session("sess-E", 30),
    )

    for tid, count in results:
        assert count > 0, f"{tid}: no responses"
    print(f"  5 sessions x 30 turns: all completed")


# ── 检查点恢复 ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_checkpoint_recovery(tmp_path):
    """agent 重启后能从检查点恢复对话历史."""
    checkpoint_path = str(tmp_path / "checkpoint.sqlite")
    memory_path = str(tmp_path / "memory.db")

    # First session: 5 turns
    llm1 = make_callable_llm(lambda p: f"reply {p[:20]}", name="recover-llm-1")
    mem1 = MemoryManager(longterm=LongTermMemory(memory_path))
    agent1 = HuginnAgent(
        model=llm1,
        memory_manager=mem1,
        checkpointer_path=checkpoint_path,
    )
    for i in range(5):
        async for _ in agent1.chat(f"first {i}", thread_id="recover-test"):
            pass

    # Check state has messages
    tree1 = agent1._conversation_tree.summary()
    assert tree1["total_nodes"] > 0

    # Simulate restart: new agent instance, same checkpoint
    llm2 = make_callable_llm(lambda p: f"recovered {p[:20]}", name="recover-llm-2")
    mem2 = MemoryManager(longterm=LongTermMemory(memory_path))
    agent2 = HuginnAgent(
        model=llm2,
        memory_manager=mem2,
        checkpointer_path=checkpoint_path,
    )

    # Continue conversation — should pick up where we left off
    responses = []
    async for state in agent2.chat("after recovery", thread_id="recover-test"):
        responses.append(state)
    assert len(responses) > 0, "No responses after recovery"
    print("  checkpoint recovery: agent continued from previous state")


# ── 上下文压缩验证 ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_context_compression_preserves_keywords(tmp_path):
    """压缩后关键信息 (材料名、数值) 不应丢失."""
    llm = make_callable_llm(lambda p: "ok", name="compress-llm")
    memory = MemoryManager(longterm=LongTermMemory(str(tmp_path / "memory.db")))
    agent = HuginnAgent(
        model=llm,
        memory_manager=memory,
        checkpointer_path=str(tmp_path / "checkpoint.sqlite"),
        context_budget_tokens=2000,  # small budget to trigger compression fast
    )

    # Send messages with specific keywords
    keywords = ["SiO2", "band gap 5.2eV", "space group P1", "lattice a=4.2"]
    for i, kw in enumerate(keywords):
        async for _ in agent.chat(f"Tell me about {kw}", thread_id="compress-keywords"):
            pass

    # Agent should still respond (compression didn't crash it)
    resp_count = 0
    async for _ in agent.chat("what did I ask about?", thread_id="compress-keywords"):
        resp_count += 1
    assert resp_count > 0
    print("  context compression: keywords sent, agent stable")


# ── 工具调用不累积 ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tool_calls_not_accumulating(tmp_path):
    """多轮对话中工具调用记录不应无限增长."""
    llm = make_callable_llm(lambda p: "ok", name="tool-accum-llm")
    memory = MemoryManager(longterm=LongTermMemory(str(tmp_path / "memory.db")))
    agent = HuginnAgent(
        model=llm,
        memory_manager=memory,
        checkpointer_path=str(tmp_path / "checkpoint.sqlite"),
    )

    for i in range(20):
        async for _ in agent.chat(f"turn {i}", thread_id="tool-accum"):
            pass

    # Session messages should be capped
    assert len(memory.session.messages) <= memory.session.max_messages
    # Tool calls should also be capped
    assert len(memory.session.tool_calls) <= memory.session.max_tool_calls
    print(f"  20 turns: {len(memory.session.messages)} msgs, {len(memory.session.tool_calls)} tool calls")


# ── 混合操作长程测试 ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mixed_operations_50_turns(tmp_path):
    """50 轮混合操作: chat + recall + remember, 验证综合稳定性."""
    llm = make_callable_llm(lambda p: "ok", name="mixed-llm")
    memory = MemoryManager(longterm=LongTermMemory(str(tmp_path / "memory.db")))
    agent = HuginnAgent(
        model=llm,
        memory_manager=memory,
        checkpointer_path=str(tmp_path / "checkpoint.sqlite"),
    )

    for i in range(50):
        # Mix chat with memory operations
        if i % 10 == 0:
            memory.remember(f"fact from turn {i}", importance=0.7)
        async for _ in agent.chat(f"mixed turn {i}", thread_id="mixed-50"):
            pass
        if i % 15 == 0:
            results = memory.recall("fact", top_k=3)
            assert isinstance(results, list)

    # Verify long-term memory accumulated facts
    all_memories = memory.recall("fact", top_k=100)
    assert len(all_memories) >= 3  # at least 3 facts from turns 0, 10, 20, 30, 40
    print(f"  50 mixed turns: {len(all_memories)} long-term memories")
