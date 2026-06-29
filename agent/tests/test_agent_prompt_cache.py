"""Tests for HuginnAgent prompt-caching message layout."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from huginn.agent import HuginnAgent
from huginn.memory.longterm import LongTermMemory
from huginn.memory.manager import MemoryManager
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
        assert not any("cache_control" in m.additional_kwargs for m in msgs)

    def test_provider_specific_cache_control(self):
        anthropic = PromptCacheBuilder(
            system_prompt="static",
            cache_control=True,
            provider="anthropic",
        )
        openai = PromptCacheBuilder(
            system_prompt="static",
            cache_control=True,
            provider="openai",
        )
        assert anthropic.build_state_modifier()[0].additional_kwargs.get(
            "cache_control"
        ) == {"type": "ephemeral"}
        assert "cache_control" not in openai.build_state_modifier()[0].additional_kwargs

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
            # The memory content should share keywords with the user query so
            # dynamic semantic recall picks it up.
            memory.remember(
                "compute stress for Ti: Ti has an hcp structure under pressure",
                category="fact",
            )

            agent = _make_agent(
                begin_dialogs=[("user", "hello"), ("assistant", "hi there")],
                memory_manager=memory,
                cache_control=True,
            )
            graph = _CaptureGraph()
            agent._agent_graph = graph

            asyncio.run(_consume_chat(agent, "compute stress for Ti"))

            assert graph.captured_inputs is not None
            msgs = graph.captured_inputs["messages"]
            assert isinstance(msgs[0], HumanMessage)
            assert msgs[0].content == "hello"
            assert isinstance(msgs[1], AIMessage)
            assert msgs[1].content == "hi there"
            assert isinstance(msgs[2], SystemMessage)
            assert "hcp structure" in msgs[2].content
            assert isinstance(msgs[3], HumanMessage)
            assert msgs[3].content == "compute stress for Ti"

    def test_memory_recall_uses_current_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            longterm = LongTermMemory(db_path=Path(tmp) / "memory.db")
            memory = MemoryManager(longterm=longterm)
            memory.remember("band gap of Si is 1.1 eV", category="fact")

            agent = _make_agent(memory_manager=memory)
            text = agent._build_memory_text(query="band gap of Si")
            assert "1.1 eV" in text

            # Different query should not recall the Si fact.
            text = agent._build_memory_text(query="lattice parameter of Ti")
            assert "1.1 eV" not in text

    def test_no_memory_message_when_recall_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            longterm = LongTermMemory(db_path=Path(tmp) / "memory.db")
            memory = MemoryManager(longterm=longterm)
            agent = _make_agent(
                begin_dialogs=[("assistant", "hi")],
                memory_manager=memory,
            )
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

    def test_tool_description_cache(self):
        class _FakeTool:
            description = "fake tool"

        agent = _make_agent()
        assert agent._get_tool_description_text() == ""
        agent.register_tool(_FakeTool())
        # _get_tool_description_text serializes full JSON schema (name +
        # description + parameters) for accurate token estimation; the cache
        # must be invalidated and rebuilt on re-registration.
        text1 = agent._get_tool_description_text()
        assert "fake tool" in text1
        agent.register_tool(_FakeTool())
        text2 = agent._get_tool_description_text()
        assert text2.count("fake tool") == 2

    def test_extract_cache_stats_from_response_metadata(self):
        agent = _make_agent()
        msgs = [
            AIMessage(
                content="ok",
                response_metadata={
                    "cache_creation_input_tokens": 100,
                    "cache_read_input_tokens": 500,
                    "output_tokens": 42,
                },
            )
        ]
        stats = agent._extract_cache_stats(msgs)
        assert stats["cache_creation_input_tokens"] == 100
        assert stats["cache_read_input_tokens"] == 500
        assert stats["output_tokens"] == 42

    def test_extract_cache_stats_from_usage_dict(self):
        agent = _make_agent()
        msgs = [
            AIMessage(
                content="ok",
                response_metadata={
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5}
                },
            )
        ]
        stats = agent._extract_cache_stats(msgs)
        assert stats["usage_prompt_tokens"] == 10
        assert stats["usage_completion_tokens"] == 5


async def _consume_chat(agent: HuginnAgent, message: str) -> None:
    async for _ in agent.chat(message):
        pass
