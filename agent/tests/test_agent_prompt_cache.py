"""Tests for HuginnAgent prompt-caching message layout."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from huginn.agent import HuginnAgent
from huginn.memory.manager import MemoryManager, MemoryConfig
from huginn.memory.longterm import LongTermMemory
from huginn.utils.prompt_cache import PromptCacheBuilder


class _CaptureGraph:
    """Minimal stand-in for a compiled LangGraph that records its inputs."""

    def __init__(self) -> None:
        self.captured_inputs: dict[str, Any] | None = None

    async def astream(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        stream_mode: str,
    ):
        self.captured_inputs = inputs
        yield {"messages": [AIMessage(content="ok")]}


def _make_agent(
    system_prompt: str = "static system",
    begin_dialogs: list[tuple[str, str]] | None = None,
    memory_manager: MemoryManager | None = None,
    cache_control: bool = True,
) -> HuginnAgent:
    return HuginnAgent(
        model=None,
        tools=[],
        system_prompt=system_prompt,
        begin_dialogs=begin_dialogs,
        memory_manager=memory_manager,
        prompt_cache_control=cache_control,
    )


class TestPromptCacheBuilder:
    def test_static_prefix_marked_for_cache(self):
        builder = PromptCacheBuilder(
            system_prompt="static",
            cache_control=True,
        )
        prefix = builder.build_state_modifier()
        assert len(prefix) == 1
        assert prefix[0].content == "static"
        assert prefix[0].additional_kwargs.get("cache_control") == {"type": "ephemeral"}

    def test_input_message_order_with_memory(self):
        builder = PromptCacheBuilder(
            system_prompt="static",
            begin_dialogs=[("user", "hi"), ("assistant", "hello")],
            cache_control=True,
        )
        msgs = builder.build_input_messages("memory fact", "current question")
        assert isinstance(msgs[0], HumanMessage)
        assert msgs[0].content == "hi"
        assert isinstance(msgs[1], AIMessage)
        assert msgs[1].content == "hello"
        assert isinstance(msgs[2], SystemMessage)
        assert "memory fact" in msgs[2].content
        assert isinstance(msgs[3], HumanMessage)
        assert msgs[3].content == "current question"
        # The last static block (the assistant begin-dialog) is cache-tagged.
        assert msgs[1].additional_kwargs.get("cache_control") == {"type": "ephemeral"}

    def test_cache_control_can_be_disabled(self):
        builder = PromptCacheBuilder(
            system_prompt="static",
            begin_dialogs=[("user", "hi")],
            cache_control=False,
        )
        msgs = builder.build_input_messages("", "q")
        assert not any(
            "cache_control" in m.additional_kwargs for m in msgs
        )

    def test_full_messages_combines_prefix_and_input(self):
        builder = PromptCacheBuilder(
            system_prompt="static", begin_dialogs=[("assistant", "hi")]
        )
        full = builder.build_full_messages("mem", "q")
        assert isinstance(full[0], SystemMessage)
        assert full[0].content == "static"
        assert isinstance(full[1], AIMessage)


class TestHuginnAgentPromptCache:
    def test_memory_is_not_in_system_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            longterm = LongTermMemory(db_path=Path(tmp) / "memory.db")
            memory = MemoryManager(longterm=longterm)
            memory.remember(
                "materials science computation: Si band gap is 1.1 eV",
                category="fact",
            )

            agent = _make_agent(memory_manager=memory)
            prefix = agent._build_state_modifier()
            assert len(prefix) == 1
            assert "Si band gap" not in prefix[0].content

            memory_text = agent._build_memory_text()
            assert "Si band gap" in memory_text

    def test_chat_input_message_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            longterm = LongTermMemory(db_path=Path(tmp) / "memory.db")
            memory = MemoryManager(longterm=longterm)
            memory.remember(
                "materials science computation: Ti is hcp", category="fact"
            )

            agent = _make_agent(
                begin_dialogs=[("user", "hello"), ("assistant", "hi there")],
                memory_manager=memory,
                cache_control=True,
            )
            graph = _CaptureGraph()
            agent._agent_graph = graph

            asyncio.run(_consume_chat(agent, "compute stress"))

            assert graph.captured_inputs is not None
            msgs = graph.captured_inputs["messages"]
            assert isinstance(msgs[0], HumanMessage)
            assert msgs[0].content == "hello"
            assert isinstance(msgs[1], AIMessage)
            assert msgs[1].content == "hi there"
            assert isinstance(msgs[2], SystemMessage)
            assert "Ti is hcp" in msgs[2].content
            assert isinstance(msgs[3], HumanMessage)
            assert msgs[3].content == "compute stress"

    def test_no_memory_message_when_recall_empty(self):
        agent = _make_agent(begin_dialogs=[("assistant", "hi")])
        msgs = agent._build_input_messages("q")
        assert len(msgs) == 2
        assert isinstance(msgs[0], AIMessage)
        assert isinstance(msgs[1], HumanMessage)

    def test_cache_control_env_default(self, monkeypatch):
        monkeypatch.setenv("HUGINN_PROMPT_CACHE_CONTROL", "0")
        agent = HuginnAgent(model=None, tools=[])
        assert agent.prompt_cache_control is False
        prefix = agent._build_state_modifier()
        assert "cache_control" not in prefix[0].additional_kwargs


async def _consume_chat(agent: HuginnAgent, message: str) -> None:
    async for _ in agent.chat(message):
        pass
