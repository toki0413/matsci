"""Behaviour check for ``_stream_agent_response``.

Locks down the exact WS message sequence the refactor is meant to
preserve: tool_call / tool_result / task_progress / text_delta / done
ordering for a normal chat turn, the plan_result frame for the
plan_confirm path, and the original fall-through quirk where a
clarification ``break`` still lets the trailing ``done`` fire.

Plain asserts, no fixtures. Run standalone:

    python tests/test_ws_stream.py
"""

import asyncio
import os
import sys
from types import SimpleNamespace

# Make ``huginn`` importable when running this file directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402

from huginn.routes.ws import _stream_agent_response  # noqa: E402


class FakeWS:
    def __init__(self):
        self.sent = []

    async def send_json(self, msg):
        self.sent.append(msg)


class FakeAgent:
    """Yields a fixed list of states from its async ``chat`` generator."""

    def __init__(self, states):
        self.states = list(states)

    async def chat(self, content, thread_id):
        for s in self.states:
            yield s


def _types(seq):
    return [m.get("type") for m in seq]


def test_normal_chat_stream_emits_expected_sequence():
    ws = FakeWS()
    states = [
        {"messages": [AIMessage(content="", tool_calls=[
            {"id": "tc1", "name": "search", "args": {"q": "x"}}])]},
        {"messages": [ToolMessage(
            content="job_id: 12345 submitted", tool_call_id="tc1")]},
        {"messages": [AIMessage(content="Hello world")]},
    ]
    agent = FakeAgent(states)
    cfg = SimpleNamespace(rag_enabled=False, workspace=".")

    full = asyncio.run(_stream_agent_response(
        ws, agent, "hi", "t1", cfg,
        auto_checkpoint=False, handle_clarification=True,
    ))

    # full_response is the last assistant content seen.
    assert full == "Hello world"
    # governance 事件是 advisory, 过滤后断言核心序列
    core = [m for m in ws.sent if m.get("type") != "governance"]
    assert _types(core) == [
        "tool_call", "tool_result", "task_progress", "text_delta", "done",
    ]
    assert core[0] == {
        "type": "tool_call", "id": "tc1", "name": "search", "args": {"q": "x"},
        "thread_id": "t1",
    }
    assert core[1]["type"] == "tool_result"
    assert core[1]["id"] == "tc1"
    # HPC-job detection inside the tool result path.
    assert core[2]["type"] == "task_progress"
    assert core[2]["job_id"] == "12345"
    assert core[3] == {"type": "text_delta", "text": "Hello world", "thread_id": "t1"}


def test_plan_stream_emits_plan_result_before_done():
    ws = FakeWS()
    states = [{"messages": [AIMessage(content="done", tool_calls=[])]}]
    agent = FakeAgent(states)
    cfg = SimpleNamespace(rag_enabled=False, workspace=".")

    asyncio.run(_stream_agent_response(
        ws, agent, "plan ctx", "t1", cfg,
        auto_checkpoint=False, handle_clarification=False,
        plan_result={"plan_id": "p1", "acceptance_criteria": [{"criterion": "c1"}]},
        sediment_question=None,
    ))

    assert _types(ws.sent) == ["text_delta", "plan_result", "done"]
    pr = ws.sent[1]
    assert pr["plan_id"] == "p1"
    assert pr["all_passed"] is True
    assert pr["criteria"][0]["criterion"] == "c1"


def test_clarification_break_still_fires_trailing_done():
    # Preserves the pre-refactor fall-through: a clarification ``break``
    # does NOT skip the post-loop ``done``, so the client receives two
    # done frames (clarification's own + the trailing one). Changing
    # this to a single ``done`` would alter the on-the-wire behaviour.
    ws = FakeWS()
    states = [{
        "needs_clarification": True,
        "clarify_questions": ["which?"],
        "messages": [AIMessage(content="which one?")],
    }]
    agent = FakeAgent(states)
    cfg = SimpleNamespace(rag_enabled=False, workspace=".")

    asyncio.run(_stream_agent_response(
        ws, agent, "hi", "t1", cfg,
        auto_checkpoint=False, handle_clarification=True,
    ))

    assert _types(ws.sent) == ["clarification_request", "text_delta", "done", "done"]
    assert ws.sent[0]["questions"] == ["which?"]
    assert ws.sent[1] == {"type": "text_delta", "text": "which one?", "thread_id": "t1"}


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    if failures:
        raise SystemExit(f"{failures} check(s) failed")
    print("all checks passed")
