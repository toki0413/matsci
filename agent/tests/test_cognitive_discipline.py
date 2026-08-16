"""Tests for the event-driven cognitive discipline guard (M2)."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from huginn import cognitive_discipline as cd
from huginn.plugins.model_tier import ModelTier, get_store


@pytest.fixture(autouse=True)
def _reset_tier():
    get_store().set_tier(ModelTier.FULL)
    yield
    get_store().set_tier(ModelTier.FULL)


def test_discipline_mode_reflects_tier():
    assert cd.discipline_mode() == "always"
    get_store().set_tier(ModelTier.BALANCED)
    assert cd.discipline_mode() == "event"
    get_store().set_tier(ModelTier.MINIMAL)
    assert cd.discipline_mode() == "event"


def test_deviation_kind_detects_tool_failure():
    tm = ToolMessage(content="Error: simulation crashed", tool_call_id="tc1")
    assert cd.deviation_kind(tm) == "tool_failure"


def test_deviation_kind_none_on_ok_or_non_tool():
    assert cd.deviation_kind(ToolMessage(content="done", tool_call_id="tc1")) is None
    assert cd.deviation_kind(AIMessage(content="final answer")) is None
    assert cd.deviation_kind(None) is None


def test_event_reminder_known_and_unknown():
    assert "[Discipline]" in cd.event_reminder("tool_failure")
    assert cd.event_reminder("bogus") == ""


def test_inject_discipline_reminder_in_event_mode():
    get_store().set_tier(ModelTier.MINIMAL)
    msgs = [
        AIMessage(content="let me run", tool_calls=[{"id": "tc1", "name": "x", "args": {}}]),
        ToolMessage(content="Error: boom", tool_call_id="tc1"),
    ]
    out = cd.inject_discipline_reminder(list(msgs))
    assert len(out) == len(msgs) + 1
    assert isinstance(out[-1], HumanMessage)
    assert "[Discipline]" in out[-1].content


def test_inject_discipline_reminder_noop_in_always_mode():
    # full 档 = always: 常驻纪律已覆盖, 不额外注入.
    msgs = [ToolMessage(content="Error: boom", tool_call_id="tc1")]
    assert cd.inject_discipline_reminder(list(msgs)) == msgs


def test_inject_discipline_reminder_noop_without_deviation():
    get_store().set_tier(ModelTier.MINIMAL)
    msgs = [ToolMessage(content="all good", tool_call_id="tc1")]
    assert cd.inject_discipline_reminder(list(msgs)) == msgs


def test_inject_discipline_reminder_noop_when_reminder_present():
    get_store().set_tier(ModelTier.MINIMAL)
    reminded = HumanMessage(content="[Discipline] ...")
    msgs = [
        ToolMessage(content="Error: boom", tool_call_id="tc1"),
        reminded,
    ]
    assert cd.inject_discipline_reminder(list(msgs)) == msgs


def test_inject_discipline_reminder_empty():
    get_store().set_tier(ModelTier.MINIMAL)
    assert cd.inject_discipline_reminder([]) == []