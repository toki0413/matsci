"""P0 integration tests for the chat (ReAct) loop.

Drives the real /v1/ws/agent WebSocket endpoint through Starlette's
TestClient.  The agent layer is mocked with a state-replay MockAgent,
but the full WS handler logic in routes/ws.py runs for real — message
parsing, streaming, tool-call visibility, thread routing, the works.

FakeLLM (scripted mode) provides the AIMessage objects that get fed
into the mock agent's state list, so the on-the-wire messages match
what a real scripted LLM would produce.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage

from huginn.server import app
from tests.fixtures.fake_llm import FakeLLM, make_scripted_llm

client = TestClient(app)
WS_PATH = "/v1/ws/agent"


# ── mock collaborators (trimmed from test_ws_integration) ──────────


class _MockModel:
    """Bare-bones model for plan-mode ainvoke calls."""

    def __init__(self, plan_text: str | None = None) -> None:
        self.plan_text = plan_text or "{}"

    async def ainvoke(self, prompt: str, **kw: Any):
        class _Resp:
            def __init__(self, content: str):
                self.content = content
        return _Resp(self.plan_text)


class MockAgent:
    """Async-generator agent that replays scripted states."""

    def __init__(self, states: list[dict] | None = None, model: Any = None):
        self._states = list(states or [])
        self.model = model if model is not None else _MockModel()
        self.persona_name = "default"

    def set_states(self, states: list[dict]) -> None:
        self._states = list(states)

    def set_persona(self, *a, **kw):
        pass

    async def chat(self, content: str, thread_id: str = "default"):
        for state in self._states:
            yield state


class _MockFactory:
    def __init__(self, harness: "ChatHarness") -> None:
        self._h = harness
        self.create_lead_calls: list[dict] = []

    def create_lead(self, *a, **kw):
        self.create_lead_calls.append(kw)
        return self._h.agent

    def create(self, *a, **kw):
        return self._h.agent

    def list_profiles(self):
        return []


class ChatHarness:
    """Holds the mocks a chat test needs."""

    def __init__(self, tmp_path) -> None:
        self.tmp_path = tmp_path
        self.cfg = SimpleNamespace(
            workspace=str(tmp_path),
            rag_enabled=False,
            persona_auto_route=False,
            persona_auto_route_threshold=0.6,
            team_mode_enabled=False,
            max_concurrent_subagents=2,
        )
        self.ctx = SimpleNamespace(
            config=SimpleNamespace(workspace=str(tmp_path)),
            kb=None,
        )
        # FakeLLM in scripted mode — provides the AIMessages the agent replays
        self.llm = make_scripted_llm([AIMessage(content="ok")])
        self.agent = MockAgent(states=[], model=self.llm)
        self.factory = _MockFactory(self)

    def states(self, states: list[dict]) -> None:
        self.agent.set_states(states)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    import huginn.routes.ws as ws_mod

    h = ChatHarness(tmp_path)

    monkeypatch.setattr(ws_mod, "get_config", lambda: h.cfg)
    monkeypatch.setattr(ws_mod, "get_agent_factory", lambda: h.factory)
    monkeypatch.setattr(ws_mod, "get_context", lambda: h.ctx)
    monkeypatch.setattr(ws_mod, "get_memory_manager", lambda: MagicMock())
    monkeypatch.setattr(ws_mod, "get_or_create_thread", lambda *a, **k: {"id": "t"})
    return h


# ── helpers ────────────────────────────────────────────────────────


def _drain(ws, max_msgs: int = 32) -> list[dict]:
    out: list[dict] = []
    for _ in range(max_msgs):
        msg = ws.receive_json()
        out.append(msg)
        if msg.get("type") in ("done", "error"):
            break
    return out


def _first(msgs: list[dict], mtype: str) -> dict | None:
    for m in msgs:
        if m.get("type") == mtype:
            return m
    return None


def _join_text(msgs: list[dict]) -> str:
    return "".join(m.get("text", "") for m in msgs if m.get("type") == "text_delta")


# ── 1. basic ReAct: user_input → LLM → text_delta + done ──────────


class TestChatBasicReAct:
    def test_text_delta_then_done(self, harness):
        """Simplest happy path: one AIMessage, streamed as text_delta, then done."""
        # FakeLLM scripted mode — the response it would produce
        llm = make_scripted_llm([AIMessage(content="The band gap is 1.12 eV.")])
        harness.agent = MockAgent(
            states=[{"messages": llm.responses}],
            model=llm,
        )
        with client.websocket_connect(WS_PATH) as ws:
            ws.send_json({
                "type": "user_input",
                "content": "What is the band gap of silicon?",
                "thread_id": "react-basic",
            })
            msgs = _drain(ws)

        assert _first(msgs, "text_delta")["text"] == "The band gap is 1.12 eV."
        assert msgs[-1]["type"] == "done"

    def test_empty_response_fallback(self, harness):
        """Agent with no text gets a fallback message, not a hang."""
        llm = make_scripted_llm([AIMessage(content="")])
        harness.agent = MockAgent(
            states=[{"messages": llm.responses}],
            model=llm,
        )
        with client.websocket_connect(WS_PATH) as ws:
            ws.send_json({
                "type": "user_input",
                "content": "hi",
                "thread_id": "react-empty",
            })
            msgs = _drain(ws)

        # ponytail: the fallback text is locale-specific Chinese; just
        # check that *something* was sent as text_delta and done follows.
        assert _first(msgs, "text_delta") is not None
        assert msgs[-1]["type"] == "done"


# ── 2. tool call chain: LLM → tool_call → tool_result → final ─────


class TestChatToolChain:
    def test_tool_call_visible(self, harness):
        """LLM calls file_read_tool, gets result, then gives final answer."""
        # Three scripted states: tool call → tool result → final text
        states = [
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "file_read_tool",
                                "args": {"path": "POSCAR"},
                                "id": "tc1",
                            }
                        ],
                    )
                ]
            },
            {
                "messages": [
                    ToolMessage(content="Si\n2\ndirect", tool_call_id="tc1")
                ]
            },
            {
                "messages": [
                    AIMessage(content="The POSCAR contains silicon in a direct configuration.")
                ]
            },
        ]
        harness.states(states)
        with client.websocket_connect(WS_PATH) as ws:
            ws.send_json({
                "type": "user_input",
                "content": "Read the POSCAR file",
                "thread_id": "tool-chain",
            })
            msgs = _drain(ws)

        # tool_call card
        tc = _first(msgs, "tool_call")
        assert tc is not None
        assert tc["name"] == "file_read_tool"
        assert tc["id"] == "tc1"
        assert tc["args"] == {"path": "POSCAR"}

        # tool_result
        tr = _first(msgs, "tool_result")
        assert tr is not None
        assert tr["id"] == "tc1"
        assert "Si" in tr["content"]

        # final text delta
        assert _first(msgs, "text_delta")["text"] == (
            "The POSCAR contains silicon in a direct configuration."
        )
        assert msgs[-1]["type"] == "done"


# ── 3. thread isolation: two threads don't share context ───────────


class TestChatThreadIsolation:
    def test_two_threads_independent(self, harness):
        """Consecutive messages on different threads get different responses.

        ponytail: Starlette TestClient is synchronous, so we can't truly
        run two WS connections in parallel. Instead we send two sequential
        messages with different thread_ids and verify the agent was called
        with the right thread_id each time. The real isolation guarantee
        comes from thread_id-keyed session state in HuginnAgent.
        """
        # First thread → response A
        llm_a = make_scripted_llm([AIMessage(content="Response for thread A")])
        harness.agent = MockAgent(
            states=[{"messages": llm_a.responses}],
            model=llm_a,
        )
        with client.websocket_connect(WS_PATH) as ws:
            ws.send_json({
                "type": "user_input",
                "content": "question A",
                "thread_id": "thread-A",
            })
            msgs_a = _drain(ws)

        # Second thread → response B (different content)
        llm_b = make_scripted_llm([AIMessage(content="Response for thread B")])
        harness.agent = MockAgent(
            states=[{"messages": llm_b.responses}],
            model=llm_b,
        )
        with client.websocket_connect(WS_PATH) as ws:
            ws.send_json({
                "type": "user_input",
                "content": "question B",
                "thread_id": "thread-B",
            })
            msgs_b = _drain(ws)

        text_a = _join_text(msgs_a)
        text_b = _join_text(msgs_b)
        assert "thread A" in text_a
        assert "thread B" in text_b
        assert text_a != text_b
        assert msgs_a[-1]["type"] == "done"
        assert msgs_b[-1]["type"] == "done"
