"""Tests for the final-answer synthesis fallback (_synthesize_closing_answer).

Covers:
  - Already has assistant text -> no fallback (empty list).
  - No assistant text + working model -> synthesized AIMessage appended.
  - No model available -> no fallback (empty list), never raises.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from huginn.agent import HuginnAgent


def _make_agent(model=None) -> HuginnAgent:
    agent = HuginnAgent.__new__(HuginnAgent)
    agent._main_fallback_override = None
    agent._csm = None
    if model is not None:
        agent.model = model
        agent.model_router = None
    else:
        agent.model = None
        agent.model_router = None
    return agent


def _run(agent, final_state):
    return asyncio.run(agent._synthesize_closing_answer(final_state))


class _FakeModel:
    def __init__(self, text: str):
        self._text = text

    async def ainvoke(self, messages):
        return AIMessage(content=self._text)


def test_already_has_text_answer_no_fallback():
    """已有非空助手文本 -> 不追加兜底."""
    agent = _make_agent(_FakeModel("won't be used"))
    state = {"messages": [HumanMessage(content="hi"), AIMessage(content="final answer")]}
    out = _run(agent, state)
    assert out == []


def test_no_text_with_working_model_appends_synthesis():
    """无助手文本 + 模型可用 -> 追加合成 AIMessage."""
    agent = _make_agent(_FakeModel("synthesized closing summary"))
    # 纯工具轮: 助手消息只有空 content 与 tool_calls, 无文本答案.
    empty_ai = AIMessage(content="")
    empty_ai.tool_calls = [{"name": "x", "args": {}, "id": "call_1"}]
    state = {"messages": [HumanMessage(content="调研一下带隙"), empty_ai]}
    out = _run(agent, state)
    assert len(out) == 1
    assert isinstance(out[0], AIMessage)
    assert "synthesized closing summary" in out[0].content


def test_no_model_returns_empty():
    """无模型 -> 空列表, 不抛异常."""
    agent = _make_agent(None)
    state = {"messages": [HumanMessage(content="hi")]}
    out = _run(agent, state)
    assert out == []


def test_exception_is_suppressed():
    """模型 ainvoke 抛异常 -> 静默降级为空列表."""
    failing = MagicMock()
    failing.ainvoke.side_effect = RuntimeError("boom")

    async def _bad_ainvoke(*a, **k):
        raise RuntimeError("boom")

    failing.ainvoke = _bad_ainvoke
    agent = _make_agent(failing)
    state = {"messages": [HumanMessage(content="hi")]}
    out = _run(agent, state)
    assert out == []


def test_empty_transcript_returns_empty():
    """Transcript 为空 (无 user/assistant 文本) -> 空列表."""
    text_model = MagicMock()
    text_model.ainvoke.return_value = AIMessage(content="should not trigger")
    agent = _make_agent(text_model)
    state = {"messages": []}
    out = _run(agent, state)
    assert out == []


def test_synthesis_uses_agent_model_via_select_model():
    """select_model('agent') 返回的模型被用于合成."""
    agent = _make_agent(_FakeModel("summary via select_model"))
    state = {"messages": [HumanMessage(content="hello")]}
    with patch.object(agent, "select_model", return_value=_FakeModel("summary via select_model")):
        out = _run(agent, state)
    assert len(out) == 1
    assert "summary via select_model" in out[0].content
