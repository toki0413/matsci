"""AgentSession —— 库式会话内核门面的测试.

用 stub agent (duck-type) 验证封装契约, 不构造真 LLM agent —— 保持 hermetic.
"""
from __future__ import annotations

import asyncio

import pytest

from huginn.agent.agent_session import AgentSession
from huginn.tools.registry import ToolRegistry


class _StubAgent:
    """最小 duck-type: 具备 AgentSession 依赖的全部接口."""

    def __init__(self) -> None:
        self.thread_id = "stub-thread"
        self.system_prompt = "old"
        self._agent_graph = "graph-object"
        self._invalidate_count = 0
        self._closed = False

        def _noop() -> None:
            self._invalidate_count += 1

        self._invalidate_tool_description_cache = _noop

    def invoke(self, message, thread_id="default"):
        return {"message": message, "thread_id": thread_id, "final": True}

    def chat(self, message, thread_id="default"):
        async def _gen():
            yield {"type": "message", "content": message, "thread_id": thread_id}

        return _gen()

    def fork_conversation(self, from_node_id=None):
        return {"forked": from_node_id}

    def switch_branch(self, node_id):
        return {"switched_to": node_id}

    def conversation_branches(self):
        return {"branches": 2}

    def close(self):
        self._closed = True


@pytest.fixture
def _reg():
    snap = ToolRegistry.snapshot()
    yield
    ToolRegistry.restore(snap)


def test_attach_wraps_agent_and_defaults_thread_id():
    s = AgentSession.attach(_StubAgent())
    assert s.thread_id == "stub-thread"    # 优先用 agent 自带 thread_id


def test_prompt_returns_final_state():
    s = AgentSession.attach(_StubAgent())
    r = s.prompt("你好", thread_id="t1")
    assert r["final"] is True
    assert r["message"] == "你好"
    assert r["thread_id"] == "t1"


def test_prompt_uses_session_default_thread():
    s = AgentSession.attach(_StubAgent())
    r = s.prompt("hi")
    assert r["thread_id"] == "stub-thread"


def test_astream_yields_events():
    s = AgentSession.attach(_StubAgent())

    async def _collect():
        return [ev async for ev in s.astream("ping")]

    evs = asyncio.run(_collect())
    assert evs == [{"type": "message", "content": "ping", "thread_id": "stub-thread"}]


def test_fork_rewind_branches_delegate_to_tree():
    s = AgentSession.attach(_StubAgent())
    assert s.fork(from_node_id="n5") == {"forked": "n5"}
    assert s.rewind("n2") == {"switched_to": "n2"}
    assert s.branches() == {"branches": 2}


def test_set_system_prompt_invalidates_graph_and_cache():
    s = AgentSession.attach(_StubAgent())
    s.set_system_prompt("  new system  ")
    assert s.agent.system_prompt == "new system"
    assert s.agent._agent_graph is None
    assert s.agent._invalidate_count == 1


def test_mode_switch_pi_and_status(_reg, tmp_path):
    from huginn.modes.pi import PRIMITIVES

    # 注入一个非原语工具, 让 pi 模式能隐藏它
    from huginn.tools.extensions import build_extension

    build_extension(
        "parsec_flux",
        "from huginn.tools.base import HuginnTool\n"
        "class P(HuginnTool):\n"
        "  name='parsec_flux'\n"
        "  def call(self,args,context=None):\n"
        "    from huginn.core_types import ToolResult\n"
        "    return ToolResult(data={})\n",
        base=tmp_path,
    )
    s = AgentSession.attach(_StubAgent())
    info = s.mode("pi")
    assert info["mode"] == "pi"
    assert info["hidden"] >= 1
    assert s.mode("status") == {"mode": "pi"}
    # 原语必须可见
    for prim in PRIMITIVES:
        t = ToolRegistry.get(prim)
        if t is not None:
            assert t.active is True
    assert s.mode("default")["mode"] == "default"


def test_mode_rejects_unknown():
    s = AgentSession.attach(_StubAgent())
    with pytest.raises(ValueError):
        s.mode("bogus")


def test_context_manager_closes():
    stub = _StubAgent()
    with AgentSession.attach(stub) as s:
        assert s.agent is stub
    assert stub._closed is True


# ── DSH 对齐: Code Mode + trajectory ─────────────────────────────

def test_set_code_mode_toggles_agent_loop():
    stub = _StubAgent()
    stub.mode = "tool_call"
    s = AgentSession.attach(stub)
    assert s.set_code_mode(True) == {"code_mode": True, "agent_mode": "code_act"}
    assert stub.mode == "code_act"
    assert s.set_code_mode(False) == {"code_mode": False, "agent_mode": "tool_call"}


def test_trajectory_groups_events_by_source():
    stub = _StubAgent()

    class FakeLog:
        def events_on_path(self, leaf_id=None):
            return [
                {"kind": "tool_call", "seq": 1},
                {"kind": "tool_result", "seq": 2},
                {"kind": "reasoning", "seq": 3},
                {"kind": "phase_change", "seq": 4},
                {"kind": "unknown_kind", "seq": 5},
            ]

    s = AgentSession.attach(stub)
    tr = s.trajectory(log=FakeLog())
    assert tr["thread_id"] == "stub-thread"
    assert tr["count"] == 5
    assert {k: len(v) for k, v in tr["by_source"].items()} == {
        "tools": 2, "model": 1, "governance": 1, "custom": 1,
    }


def test_trajectory_without_log_fails_open():
    s = AgentSession.attach(_StubAgent())
    tr = s.trajectory()   # 无真事件日志也应返回空结构, 不抛
    assert tr["count"] == 0
    assert tr["by_source"] == {}
