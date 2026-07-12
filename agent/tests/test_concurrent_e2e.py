"""Concurrent end-to-end tests for HuginnAgent.

These are the first tests in the codebase that simulate multiple
simultaneous users hitting a shared agent singleton.  They exercise
the contextvars-based thread isolation, per-session rate limiting,
checkpoint isolation, and contextvar cleanup under realistic
concurrent load.

Run with::

    python -m pytest tests/test_concurrent_e2e.py -v --tb=short --no-cov
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

import huginn.security.rate_limiter as rl_mod
from huginn.agent import HuginnAgent
from huginn.memory.longterm import LongTermMemory
from huginn.memory.manager import MemoryManager
from huginn.security.rate_limiter import RateLimitConfig, TokenRateLimiter
from huginn.utils.session_context import get_thread_id
from tests.fixtures.fake_llm import make_callable_llm


# ── shared helpers ───────────────────────────────────────────


@tool
def _echo(text: str) -> str:
    """Bounce the input back — gives the agent graph at least one bound tool."""
    return text


@pytest.fixture(autouse=True)
def _fresh_rate_limiter():
    """Wipe the global rate-limiter singleton before and after each test.

    Without this, token counts from one test bleed into the next via the
    module-level _singleton in rate_limiter.py.
    """
    rl_mod._singleton = None
    yield
    rl_mod._singleton = None


def _build_agent(tmp_path: Any, llm: Any | None = None, **kwargs: Any) -> HuginnAgent:
    """Build an isolated HuginnAgent backed by a FakeLLM.

    Deliberately omits ``checkpointer_path`` so the agent falls back to
    the in-memory checkpointer (InMemorySaver).  This keeps the graph on
    the async ``astream`` path, which is required for contextvars to
    propagate into model calls and middleware.
    """
    if llm is None:
        # The callable reads get_thread_id() at call-time, so each
        # concurrent task gets a response tagged with its own thread_id.
        llm = make_callable_llm(
            lambda prompt: f"response for thread {get_thread_id()}"
        )
    memory = MemoryManager(
        longterm=LongTermMemory(str(tmp_path / "memory.db")),
    )
    return HuginnAgent(
        model=llm,
        tools=[_echo],
        memory_manager=memory,
        # Disable optional subsystems that slow down tests or pull in
        # external resources (ChromaDB, knowledge graph, etc.).
        telemetry_enabled=False,
        kg_enabled=False,
        kb_enabled=False,
        auto_approve=True,
        **kwargs,
    )


async def _consume(
    agent: HuginnAgent, message: str, thread_id: str = "default"
) -> dict[str, Any]:
    """Drain the async chat stream and return the last state with messages."""
    final_state: dict[str, Any] | None = None
    async for state in agent.chat(message, thread_id=thread_id):
        if isinstance(state, dict) and "messages" in state:
            final_state = state
    assert final_state is not None, f"chat() yielded no state with messages for thread {thread_id}"
    return final_state


# ── tests ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_concurrent_thread_id_isolation(tmp_path):
    """Ten concurrent chat() calls with distinct thread_ids must each
    receive a response tagged with their own thread_id — no leakage.

    The FakeLLM callable reads get_thread_id() at call-time.  If the
    contextvar is correctly isolated per asyncio task, each response
    is an exact match for ``f"response for thread {tid}"``.
    """
    agent = _build_agent(tmp_path)
    try:
        thread_ids = [f"iso-{i}" for i in range(10)]

        coros = [
            _consume(agent, f"hello from {tid}", thread_id=tid)
            for tid in thread_ids
        ]
        results = await asyncio.gather(*coros)

        assert len(results) == 10
        for tid, state in zip(thread_ids, results):
            messages = state["messages"]
            ai_msgs = [m for m in messages if isinstance(m, AIMessage)]
            assert ai_msgs, f"no AIMessage in final state for {tid}"
            expected = f"response for thread {tid}"
            assert ai_msgs[-1].content == expected, (
                f"thread_id leakage: expected '{expected}', "
                f"got '{ai_msgs[-1].content}'"
            )
    finally:
        agent.close()


@pytest.mark.asyncio
async def test_concurrent_memory_isolation(tmp_path):
    """Five concurrent chat sessions must keep their per-thread state
    isolated at the model-call level.

    The FakeLLM callable returns ``f"response for thread {get_thread_id()}"``
    — if the contextvar is correctly scoped per asyncio task, each
    thread's AI response is an exact match and no thread sees another
    thread's id in its response.

    Note: the agent's ``_conversation_tree`` is a shared instance-level
    structure, so HumanMessages from prior turns may appear in the
    graph input across threads.  This test focuses on the model-output
    isolation that the contextvars fix guarantees.
    """
    agent = _build_agent(tmp_path)
    try:
        thread_ids = [f"mem-{i}" for i in range(5)]
        # Each thread sends a unique payload we can later grep for.
        payloads = {tid: f"secret-payload-{tid}" for tid in thread_ids}

        coros = [
            _consume(agent, payloads[tid], thread_id=tid)
            for tid in thread_ids
        ]
        results = await asyncio.gather(*coros)

        for tid, state in zip(thread_ids, results):
            msgs = state["messages"]

            # 1. The thread's own user message must be present.
            human_contents = [
                getattr(m, "content", "")
                for m in msgs
                if isinstance(m, HumanMessage)
            ]
            assert any(
                payloads[tid] in c for c in human_contents
            ), f"thread {tid} is missing its own user message"

            # 2. The AI response must carry this thread's id — exact match,
            #    so "mem-1" can't hide inside "mem-10".
            ai_msgs = [m for m in msgs if isinstance(m, AIMessage)]
            assert ai_msgs, f"thread {tid} has no AIMessage"
            expected = f"response for thread {tid}"
            assert ai_msgs[-1].content == expected, (
                f"thread {tid}: expected '{expected}', "
                f"got '{ai_msgs[-1].content}'"
            )

            # 3. No other thread's id may appear in the AI response.
            for other_tid in thread_ids:
                if other_tid == tid:
                    continue
                assert other_tid not in ai_msgs[-1].content, (
                    f"response contamination: thread {tid} contains "
                    f"id from {other_tid}"
                )
    finally:
        agent.close()


def test_per_session_rate_limiting():
    """A TokenRateLimiter with a tight turn budget must block session A
    once it exhausts its quota while session B — on a different
    thread_id — remains completely unaffected."""
    cfg = RateLimitConfig(
        max_tokens_per_turn=100,
        # Push the per-second ceiling high so only the turn gate fires.
        max_tokens_per_second=1_000_000,
        max_total_cost_usd=10_000.0,
    )
    limiter = TokenRateLimiter(cfg)

    # Session A: burn through the entire turn budget in one record.
    limiter.record_usage(
        "test-model", input_tokens=100, output_tokens=0, thread_id="A"
    )

    # Session A is now over budget — even 1 more token is rejected.
    ok_a, reason_a = limiter.check_allowed("test-model", 1, thread_id="A")
    assert ok_a is False
    assert "单轮" in reason_a

    # Session B has its own counter and should sail through.
    ok_b, reason_b = limiter.check_allowed("test-model", 50, thread_id="B")
    assert ok_b is True
    assert reason_b == ""

    # Session B can keep spending; session A stays blocked.
    limiter.record_usage(
        "test-model", input_tokens=50, output_tokens=0, thread_id="B"
    )
    ok_b2, _ = limiter.check_allowed("test-model", 1, thread_id="B")
    assert ok_b2 is True

    ok_a2, _ = limiter.check_allowed("test-model", 1, thread_id="A")
    assert ok_a2 is False


@pytest.mark.asyncio
async def test_concurrent_agent_singleton_safety(tmp_path):
    """The production pattern: ``get_agent()`` returns a shared singleton.
    Twenty concurrent chat() calls with different thread_ids must show
    zero cross-contamination — each response is an exact match for its
    own thread_id.

    Uses exact matching (``==``) instead of substring (``in``) so that
    ids like "single-1" don't false-positive against "single-10".
    """
    agent = _build_agent(tmp_path)
    try:
        thread_ids = [f"single-{i}" for i in range(20)]

        coros = [
            _consume(agent, f"msg for {tid}", thread_id=tid)
            for tid in thread_ids
        ]
        results = await asyncio.gather(*coros)

        assert len(results) == 20
        for tid, state in zip(thread_ids, results):
            msgs = state["messages"]
            ai_contents = [
                getattr(m, "content", "")
                for m in msgs
                if isinstance(m, AIMessage)
            ]
            assert ai_contents, f"no AIMessage for {tid}"
            expected = f"response for thread {tid}"
            assert ai_contents[-1] == expected, (
                f"singleton contamination: expected '{expected}', "
                f"got '{ai_contents[-1]}'"
            )
    finally:
        agent.close()


@pytest.mark.asyncio
async def test_contextvars_cleanup(tmp_path):
    """After concurrent tasks finish, the thread_id contextvar must be
    reset to None — no leaked state survives the task boundary.

    asyncio.gather wraps each coroutine in its own Task, and each Task
    copies the parent context.  set_thread_id() inside chat() mutates
    only the child copy, so the parent (this test) should be untouched."""
    # Sanity: contextvar is None before we start.
    assert get_thread_id() is None

    agent = _build_agent(tmp_path)
    try:
        thread_ids = [f"cleanup-{i}" for i in range(5)]
        coros = [
            _consume(agent, f"msg for {tid}", thread_id=tid)
            for tid in thread_ids
        ]
        await asyncio.gather(*coros)
    finally:
        agent.close()

    # After all concurrent tasks have completed, the contextvar in
    # *this* task must still be None.
    assert get_thread_id() is None, (
        "contextvar leaked: thread_id is not None after concurrent tasks"
    )
