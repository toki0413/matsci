"""End-to-end integration tests for the core HuginnAgent flow.

These tests wire together a fake LLM, real LangGraph ReAct executor,
memory, telemetry, and checkpointing to verify the whole agent lifecycle
without requiring external API keys.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from huginn.agent import HuginnAgent
from huginn.memory.longterm import LongTermMemory
from huginn.memory.manager import MemoryManager
from huginn.telemetry import get_telemetry_collector


class FakeToolCallingModel(BaseChatModel):
    """Deterministic chat model that returns a scripted sequence of messages."""

    responses: list[AIMessage]
    calls: list[list[Any]] | None = None
    _index: int = 0

    def __init__(self, responses: list[AIMessage], **kwargs: Any) -> None:
        super().__init__(responses=responses, **kwargs)
        self.calls = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(messages)
        response = self.responses[self._index]
        self._index += 1
        return ChatResult(generations=[ChatGeneration(message=response)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop, run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"

    def bind_tools(self, tools, **kwargs):
        return self


@tool
def e2e_calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression."""
    with get_telemetry_collector().span("e2e_calculator", expression=expression):
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return f"Result: {result}"
        except Exception as exc:
            return f"Error: {exc}"


def _build_agent(tmp_path: Any, responses: list[AIMessage], **kwargs: Any) -> HuginnAgent:
    """Build an isolated HuginnAgent with a fake model."""
    model = FakeToolCallingModel(responses=responses)
    memory = MemoryManager(
        longterm=LongTermMemory(str(tmp_path / "memory.db")),
    )
    return HuginnAgent(
        model=model,
        tools=[e2e_calculator],
        memory_manager=memory,
        checkpointer_path=str(tmp_path / "checkpoint.sqlite"),
        **kwargs,
    )


async def _consume(agent: HuginnAgent, message: str, thread_id: str = "e2e") -> dict[str, Any]:
    """Consume the async chat stream and return the last state."""
    final_state = None
    async for state in agent.chat(message, thread_id=thread_id):
        final_state = state
    return final_state


class TestAgentFlow:
    @pytest.mark.asyncio
    async def test_tool_call_and_final_answer(self, tmp_path):
        responses = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "e2e_calculator",
                        "args": {"expression": "2 + 3"},
                        "id": "tc1",
                    }
                ],
            ),
            AIMessage(content="The answer is 5."),
        ]
        agent = _build_agent(tmp_path, responses)
        try:
            final_state = await _consume(agent, "What is 2 + 3?")
            messages = final_state["messages"]
            assert any(
                getattr(m, "content", "") == "The answer is 5." for m in messages
            )
            # Memory captured the assistant/tool interaction.
            session_messages = agent.memory.session.messages
            assert any(m.role == "assistant" for m in session_messages)
            assert any(tc.tool_name == "e2e_calculator" for tc in agent.memory.session.tool_calls)

            # Telemetry recorded the turn and the nested tool span.
            spans = agent.telemetry_spans()
            names = {s["name"] for s in _flatten_spans(spans)}
            assert "agent_turn" in names
            assert "e2e_calculator" in names
        finally:
            agent.close()

    @pytest.mark.asyncio
    async def test_memory_recall_in_prompt(self, tmp_path):
        responses = [
            AIMessage(content="Recalling what I know about band gaps."),
        ]
        agent = _build_agent(tmp_path, responses)
        try:
            agent.remember(
                "Silicon has an indirect band gap of ~1.12 eV.",
                category="material_fact",
                tags=["silicon", "band_gap"],
                importance=0.9,
            )
            await _consume(agent, "indirect band gap")

            # The fake model should have been called with the recalled memory.
            assert len(responses) == 1  # sanity
            calls = agent.model.calls
            assert calls
            memory_found = any(
                "Silicon has an indirect band gap" in getattr(m, "content", "")
                for msg_list in calls
                for m in msg_list
            )
            assert memory_found

            # Explicit recall also returns the stored fact.
            recalled = agent.recall("indirect band gap")
            assert any("1.12 eV" in r["content"] for r in recalled)
        finally:
            agent.close()

    @pytest.mark.asyncio
    async def test_checkpoint_restore_across_instances(self, tmp_path):
        # First agent: tool call that returns a result.
        responses1 = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "e2e_calculator",
                        "args": {"expression": "10 * 10"},
                        "id": "tc1",
                    }
                ],
            ),
            AIMessage(content="The result is 100."),
        ]
        agent1 = _build_agent(tmp_path, responses1)
        try:
            await _consume(agent1, "Compute 10 times 10.", thread_id="restore_thread")
        finally:
            agent1.close()

        # Second agent: same checkpoint, new model; should see prior messages.
        responses2 = [AIMessage(content="Previously we got 100.")]
        agent2 = _build_agent(tmp_path, responses2)
        try:
            final_state = await _consume(agent2, "What did we get?", thread_id="restore_thread")
            messages = final_state["messages"]
            # The new user message is present, and prior assistant message should be loaded.
            roles = [type(m).__name__ for m in messages]
            assert "HumanMessage" in roles
            assert "AIMessage" in roles
            assert any("100" in getattr(m, "content", "") for m in messages)
        finally:
            agent2.close()

    def test_synchronous_invoke(self, tmp_path):
        responses = [AIMessage(content="Sync answer.")]
        agent = _build_agent(tmp_path, responses)
        try:
            result = agent.invoke("hello")
            messages = result["messages"]
            assert any(getattr(m, "content", "") == "Sync answer." for m in messages)
        finally:
            agent.close()


def _flatten_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for span in spans:
        result.append(span)
        result.extend(_flatten_spans(span.get("children", [])))
    return result
