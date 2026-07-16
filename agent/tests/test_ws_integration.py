"""WebSocket integration tests for the Huginn agent.

Drives the real ``/v1/ws/agent`` endpoint through Starlette's TestClient
while mocking the LLM/agent layer, so nothing hits a live model. Each
test opens its own connection, runs a short exchange, and tears down — no
shared state between tests.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage

from huginn.server import app

client = TestClient(app)

WS_PATH = "/v1/ws/agent"


# ── fake collaborators ──────────────────────────────────────────────


class _FakeUserStore:
    """No per-user keys; force the shared API-key comparison path."""

    def get_user_by_api_key(self, key: str):
        return None

    def get_user(self, uid: str):
        return None


class _MockModel:
    """Just enough of the chat model for plan mode (``ainvoke``)."""

    def __init__(self, plan_text: str | None = None) -> None:
        self.plan_text = plan_text or "{}"

    async def ainvoke(self, prompt: str, **kwargs: Any):
        class _Resp:
            def __init__(self, content: str) -> None:
                self.content = content

        return _Resp(self.plan_text)


class MockAgent:
    """Async-generator agent that replays a scripted list of states."""

    def __init__(
        self,
        states: list[dict] | None = None,
        model: Any = None,
    ) -> None:
        self._states = list(states or [])
        # ``agent.model is None`` short-circuits the handler with an error,
        # so always hand it something truthy.
        self.model = model if model is not None else _MockModel()
        self.persona_name = "default"

    def set_states(self, states: list[dict]) -> None:
        self._states = list(states)

    def set_plan(self, plan_text: str) -> None:
        self.model.plan_text = plan_text

    def set_persona(self, *args: Any, **kwargs: Any) -> None:
        pass

    # ws_helpers._handle_plan_confirm 调这两个, mock 里空实现就行
    def enter_plan_execution(self) -> None:
        pass

    def exit_plan_execution(self) -> None:
        pass

    async def chat(self, content: str, thread_id: str = "default"):
        for state in self._states:
            yield state


class _MockFactory:
    """Factory stub that always returns the harness's current agent."""

    def __init__(self, harness: "WSHarness") -> None:
        self._h = harness
        self.create_lead_calls: list[dict] = []

    def create_lead(self, *args: Any, **kwargs: Any) -> MockAgent:
        self.create_lead_calls.append(kwargs)
        return self._h.agent

    def create(self, *args: Any, **kwargs: Any) -> MockAgent:
        return self._h.agent

    def list_profiles(self) -> list:
        return []


class WSHarness:
    """Bundles the mocks a chat test needs and exposes hooks to
    reconfigure the scripted agent output per test."""

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
        self.agent = MockAgent(states=[], model=_MockModel())
        self.factory = _MockFactory(self)

    def states(self, states: list[dict]) -> None:
        self.agent.set_states(states)

    def plan(self, plan_text: str) -> None:
        self.agent.set_plan(plan_text)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """Patch the ws.py entry points with mocks and yield the harness.

    Patching happens in the ``huginn.routes.ws`` namespace because that's
    where the names were imported into at module load time.
    """
    import huginn.routes.ws as ws_mod

    h = WSHarness(tmp_path)

    async def _get_agent():
        # Read ``h.agent`` at call time so tests can swap the instance
        # after the fixture has wired up the patches.
        return h.agent

    monkeypatch.setattr(ws_mod, "get_config", lambda: h.cfg)
    monkeypatch.setattr(ws_mod, "get_agent", _get_agent)
    monkeypatch.setattr(ws_mod, "get_agent_factory", lambda: h.factory)
    monkeypatch.setattr(ws_mod, "get_context", lambda: h.ctx)
    monkeypatch.setattr(ws_mod, "get_memory_manager", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(
        ws_mod, "get_or_create_thread", lambda *a, **k: {"id": "t"}
    )
    return h


# ── small receive helpers ────────────────────────────────────────────


def _drain(ws, max_msgs: int = 32) -> list[dict]:
    """Pull messages until we hit ``done`` or ``error`` (or the limit)."""
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
    return "".join(
        m.get("text", "") for m in msgs if m.get("type") == "text_delta"
    )


# A structured plan the mock model returns during plan mode. Includes
# acceptance_criteria so the confirm path emits a plan_result message.
_PLAN_JSON = json.dumps(
    {
        "steps": [
            {
                "name": "Gather inputs",
                "description": "collect the data to analyze",
                "tool": "agent",
                "estimated_time": "10s",
            }
        ],
        "acceptance_criteria": [
            {
                "criterion": "data analyzed",
                "how_to_verify": "check the summary output",
            }
        ],
        "tools_needed": [],
        "summary": "Analyze the data step by step",
    },
    ensure_ascii=False,
)


# ── 1. connection / auth tests ──────────────────────────────────────


class TestWSConnection:
    def test_ws_connect(self, monkeypatch):
        # Turn off dev mode so the shared API key is the only way in.
        monkeypatch.delenv("HUGINN_DEV_MODE", raising=False)
        monkeypatch.setenv("HUGINN_API_KEY", "test-key")
        from huginn.security import auth as auth_mod

        monkeypatch.setattr(auth_mod, "get_user_store", lambda: _FakeUserStore())

        with client.websocket_connect(
            WS_PATH, headers={"X-HUGINN-API-KEY": "test-key"}
        ) as ws:
            ws.send_json({"type": "ping"})
            assert ws.receive_json()["type"] == "pong"

    def test_ws_connect_no_key(self, monkeypatch):
        monkeypatch.delenv("HUGINN_DEV_MODE", raising=False)
        monkeypatch.setenv("HUGINN_API_KEY", "test-key")
        from huginn.security import auth as auth_mod

        monkeypatch.setattr(auth_mod, "get_user_store", lambda: _FakeUserStore())

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(WS_PATH):
                pass

    def test_ws_connect_invalid_key(self, monkeypatch):
        monkeypatch.delenv("HUGINN_DEV_MODE", raising=False)
        monkeypatch.setenv("HUGINN_API_KEY", "test-key")
        from huginn.security import auth as auth_mod

        monkeypatch.setattr(auth_mod, "get_user_store", lambda: _FakeUserStore())

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                WS_PATH, headers={"X-HUGINN-API-KEY": "totally-wrong"}
            ):
                pass


# ── 2. chat message tests ───────────────────────────────────────────


class TestWSChat:
    def test_ws_chat_basic(self, harness):
        harness.states([{"messages": [AIMessage(content="Hello back!")]}])
        with client.websocket_connect(WS_PATH) as ws:
            ws.send_json(
                {"type": "user_input", "content": "Hi there", "thread_id": "chat-basic"}
            )
            msgs = _drain(ws)
        assert _first(msgs, "text_delta")["text"] == "Hello back!"
        assert msgs[-1]["type"] == "done"

    def test_ws_chat_empty_content(self, harness):
        # The mock agent refuses empty input; the server should surface
        # that as an ``error`` message rather than hanging or crashing.
        class _RejectEmpty(MockAgent):
            async def chat(self, content, thread_id="default"):
                if not content.strip():
                    raise ValueError("empty content")
                async for s in super().chat(content, thread_id):
                    yield s

        harness.agent = _RejectEmpty(model=_MockModel())
        with client.websocket_connect(WS_PATH) as ws:
            ws.send_json(
                {"type": "user_input", "content": "", "thread_id": "chat-empty"}
            )
            msgs = _drain(ws)
        err = _first(msgs, "error")
        assert err is not None
        assert "empty content" in err["error"]

    def test_ws_chat_with_config(self, harness):
        harness.states(
            [{"messages": [AIMessage(content="Configured response")]}]
        )
        with client.websocket_connect(WS_PATH) as ws:
            ws.send_json(
                {
                    "type": "user_input",
                    "content": "Hello with a custom config override please",
                    "thread_id": "chat-cfg",
                    "thinking": "medium",
                    "max_tokens": 1000,
                }
            )
            msgs = _drain(ws)
        assert _first(msgs, "text_delta")["text"] == "Configured response"
        assert msgs[-1]["type"] == "done"
        # The factory should have been asked to build a fresh lead agent
        # carrying the overrides we sent.
        assert harness.factory.create_lead_calls
        call = harness.factory.create_lead_calls[-1]
        assert call.get("thinking") == "medium"
        assert call.get("max_tokens") == 1000


# ── 3. plan mode tests ──────────────────────────────────────────────


class TestWSPlanMode:
    def test_ws_plan_mode(self, harness):
        harness.plan(_PLAN_JSON)
        with client.websocket_connect(WS_PATH) as ws:
            ws.send_json(
                {
                    "type": "user_input",
                    "content": "/plan analyze the dataset thoroughly",
                    "thread_id": "plan-1",
                }
            )
            msg = ws.receive_json()
        assert msg["type"] == "plan"
        assert "plan_id" in msg
        assert "steps" in msg["plan"]

    def test_ws_plan_confirm(self, harness):
        harness.plan(_PLAN_JSON)
        harness.states(
            [{"messages": [AIMessage(content="Plan executed successfully")]}]
        )
        with client.websocket_connect(WS_PATH) as ws:
            ws.send_json(
                {
                    "type": "user_input",
                    "content": "/plan analyze the dataset thoroughly",
                    "thread_id": "plan-confirm",
                }
            )
            plan_msg = ws.receive_json()
            assert plan_msg["type"] == "plan"
            ws.send_json(
                {
                    "type": "plan_confirm",
                    "plan_id": plan_msg["plan_id"],
                    "confirmed": True,
                }
            )
            msgs = _drain(ws)
        assert _first(msgs, "text_delta")["text"] == "Plan executed successfully"
        assert msgs[-1]["type"] == "done"
        # Plan carried acceptance_criteria, so a plan_result should land
        # before the terminal done.
        assert _first(msgs, "plan_result") is not None

    def test_ws_plan_cancel(self, harness):
        harness.plan(_PLAN_JSON)
        with client.websocket_connect(WS_PATH) as ws:
            ws.send_json(
                {
                    "type": "user_input",
                    "content": "/plan analyze the dataset thoroughly",
                    "thread_id": "plan-cancel",
                }
            )
            plan_msg = ws.receive_json()
            ws.send_json(
                {
                    "type": "plan_confirm",
                    "plan_id": plan_msg["plan_id"],
                    "confirmed": False,
                }
            )
            msgs = _drain(ws)
        assert "cancelled" in _join_text(msgs).lower()
        assert msgs[-1]["type"] == "done"


# ── 4. clarification tests ─────────────────────────────────────────


class TestWSClarification:
    def test_ws_clarification_request_flow(self, harness):
        # When the agent flags ambiguity, the server emits a
        # clarification_request carrying the questions.
        harness.states(
            [
                {
                    "needs_clarification": True,
                    "clarify_questions": [
                        {"id": "q1", "question": "Which material?"}
                    ],
                    "messages": [
                        AIMessage(content="Which material are you asking about?")
                    ],
                }
            ]
        )
        with client.websocket_connect(WS_PATH) as ws:
            ws.send_json(
                {
                    "type": "user_input",
                    "content": "tell me about the band gap",
                    "thread_id": "clar-1",
                }
            )
            msgs = _drain(ws)
        req = _first(msgs, "clarification_request")
        assert req is not None
        assert req["questions"][0]["id"] == "q1"
        assert _first(msgs, "text_delta") is not None
        assert msgs[-1]["type"] == "done"

    def test_ws_clarification_response(self, harness, monkeypatch):
        # Client answers a clarification. The server doesn't reply, so we
        # ping afterwards to confirm the connection survived intact.
        resolved: dict = {}

        class _FakeMgr:
            def resolve(self, question_id, answer):
                resolved["question_id"] = question_id
                resolved["answer"] = answer

            def resolve_thread(self, thread_id, answer):
                resolved["thread_id"] = thread_id
                resolved["answer"] = answer

        fake = _FakeMgr()
        import huginn.interaction.clarification as clar_mod

        # ponytail: get_clarification_manager 缓存了 _singleton, patch 类没用
        monkeypatch.setattr(clar_mod, "get_clarification_manager", lambda: fake)

        with client.websocket_connect(WS_PATH) as ws:
            ws.send_json(
                {
                    "type": "clarification_response",
                    "question_id": "q1",
                    "answer": "Silicon",
                    "thread_id": "clar-1",
                }
            )
            ws.send_json({"type": "ping"})
            assert ws.receive_json()["type"] == "pong"
        assert resolved.get("question_id") == "q1"
        assert resolved.get("answer") == "Silicon"


# ── 5. ping / pong ──────────────────────────────────────────────────


class TestWSPing:
    def test_ws_ping(self):
        with client.websocket_connect(WS_PATH) as ws:
            ws.send_json({"type": "ping"})
            msg = ws.receive_json()
            assert msg["type"] == "pong"


# ── 6. thought loop termination ─────────────────────────────────────


class TestWSThoughtLoop:
    def test_ws_thought_loop(self, harness):
        harness.states([{"thought_loop_terminated": True}])
        with client.websocket_connect(WS_PATH) as ws:
            ws.send_json(
                {
                    "type": "user_input",
                    "content": "keep going in circles",
                    "thread_id": "loop-1",
                }
            )
            msgs = _drain(ws)
        assert "思考循环" in _join_text(msgs)
        assert msgs[-1]["type"] == "done"


# ── 7. tool call visibility ────────────────────────────────────────


class TestWSToolCall:
    def test_ws_tool_call_visible(self, harness):
        harness.states(
            [
                {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "calculator",
                                    "args": {"expression": "2+2"},
                                    "id": "tc1",
                                }
                            ],
                        )
                    ]
                },
                {"messages": [ToolMessage(content="4", tool_call_id="tc1")]},
                {"messages": [AIMessage(content="The answer is 4.")]},
            ]
        )
        with client.websocket_connect(WS_PATH) as ws:
            ws.send_json(
                {"type": "user_input", "content": "What is 2+2?", "thread_id": "tool-1"}
            )
            msgs = _drain(ws)
        tc = _first(msgs, "tool_call")
        assert tc is not None
        assert tc["name"] == "calculator"
        assert tc["id"] == "tc1"
        assert tc["args"] == {"expression": "2+2"}
        tr = _first(msgs, "tool_result")
        assert tr is not None
        assert tr["id"] == "tc1"
        assert tr["content"] == "4"
        assert _first(msgs, "text_delta")["text"] == "The answer is 4."
        assert msgs[-1]["type"] == "done"
