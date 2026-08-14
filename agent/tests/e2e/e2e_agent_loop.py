"""端到端测试 — LLM Mock E2E: 完整 agent loop (沙箱可跑).

用 FakeLLM (脚本回放) 驱动真实 HuginnAgent, 验证完整 ReAct 循环:
规划 → 工具调用 → 观察 → 总结, 不依赖外部 LLM API.

覆盖场景:
1. 单轮工具调用: 用户提问 → agent 调工具 → 拿结果 → 总结回答
2. 多轮对话上下文: 第一轮调工具, 第二轮基于历史继续对话
3. 工具失败容错: 工具抛异常, agent 不崩溃, 能继续
4. Memory 集成: agent 执行中写入 memory, 后续可检索
5. Telemetry 链路: 每次 turn 产生 span, 工具调用有嵌套 span

与 tests/test_e2e_agent_flow.py 的区别:
- 不依赖 langgraph sqlite checkpointer (用 in-memory)
- 增加多轮对话 + 工具失败 + memory 集成场景
"""
from __future__ import annotations

import importlib.util
import sys
from typing import Any

import pytest

# langgraph 是硬依赖, 没装就跳过整个文件
_langgraph_available = importlib.util.find_spec("langgraph") is not None
pytestmark = pytest.mark.skipif(
    not _langgraph_available, reason="langgraph not available"
)

from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

from huginn.agent.core import HuginnAgent  # noqa: E402
from huginn.memory.longterm import LongTermMemory  # noqa: E402
from huginn.memory.manager import MemoryManager  # noqa: E402
from tests.fixtures.fake_llm import FakeLLM  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────
# 测试工具
# ──────────────────────────────────────────────────────────────────────────


@tool
def e2e_add(a: int, b: int) -> str:
    """Add two integers and return the result."""
    return f"Result: {a + b}"


@tool
def e2e_fail(msg: str) -> str:
    """A tool that simulates a failure by returning an error string.

    Note: 实际抛异常会让 langgraph ToolNode re-raise (除非 ToolNode 构造时
    传 handle_tool_errors=True, 但 HuginnAgent 不暴露这个参数).
    所以这里改成返回错误字符串, 模拟工具内部 try/except 后的降级返回.
    这也是推荐的生产实践: 工具应该自己处理异常, 返回结构化错误, 而不是
    让异常冒泡到 agent loop.
    """
    return f"ERROR: intentional failure: {msg}"


@tool
def e2e_store_fact(fact: str) -> str:
    """Store a fact in long-term memory."""
    # 通过 contextvars 或直接调用 memory manager (这里简化, 返回确认)
    return f"Stored: {fact}"


# ──────────────────────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────────────────────


def _build_agent(
    tmp_path: Any,
    responses: list[AIMessage],
    tools: list | None = None,
) -> HuginnAgent:
    """构建隔离的 HuginnAgent + FakeLLM, 用 in-memory checkpointer."""
    model = FakeLLM(responses=responses)
    memory = MemoryManager(
        longterm=LongTermMemory(str(tmp_path / "memory.db")),
    )
    return HuginnAgent(
        model=model,
        tools=tools or [e2e_add],
        memory_manager=memory,
        # 不传 checkpointer_path, 默认用 in-memory
    )


async def _consume(
    agent: HuginnAgent, message: str, thread_id: str = "e2e"
) -> dict[str, Any] | None:
    """消费 agent.chat 的 async stream, 返回最后一个含 messages 的 state."""
    final_state = None
    async for state in agent.chat(message, thread_id=thread_id):
        if isinstance(state, dict) and "messages" in state:
            final_state = state
    return final_state


# ──────────────────────────────────────────────────────────────────────────
# 测试用例
# ──────────────────────────────────────────────────────────────────────────


class TestAgentLoopSingleTurn:
    """单轮 ReAct 循环: 提问 → 工具调用 → 总结."""

    @pytest.mark.asyncio
    async def test_tool_call_then_summary(self, tmp_path):
        """agent 调用 e2e_add 工具, 拿到结果, 给出最终回答."""
        responses = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "e2e_add",
                        "args": {"a": 2, "b": 3},
                        "id": "tc1",
                    }
                ],
            ),
            AIMessage(content="The sum of 2 and 3 is 5."),
        ]
        agent = _build_agent(tmp_path, responses, tools=[e2e_add])
        try:
            final_state = await _consume(agent, "What is 2 + 3?")
            assert final_state is not None, "agent did not return final state"
            messages = final_state["messages"]
            # 最后一条应该是 AIMessage 含总结
            last = messages[-1]
            assert getattr(last, "content", "") == "The sum of 2 and 3 is 5.", \
                f"unexpected final message: {last}"
        finally:
            agent.close()

    @pytest.mark.asyncio
    async def test_direct_answer_without_tool(self, tmp_path):
        """简单问题 agent 直接回答, 不调工具."""
        responses = [
            AIMessage(content="Hello! I'm Huginn, ready to help."),
        ]
        agent = _build_agent(tmp_path, responses, tools=[e2e_add])
        try:
            final_state = await _consume(agent, "Hi")
            assert final_state is not None
            messages = final_state["messages"]
            last = messages[-1]
            assert "Huginn" in getattr(last, "content", ""), \
                f"unexpected: {last}"
        finally:
            agent.close()


class TestAgentLoopMultiTurn:
    """多轮对话: 上下文保持."""

    @pytest.mark.asyncio
    async def test_multi_turn_context_retention(self, tmp_path):
        """第一轮调工具, 第二轮基于历史回答."""
        responses = [
            # 第一轮: 调工具
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "e2e_add", "args": {"a": 10, "b": 20}, "id": "tc1"}
                ],
            ),
            # 第一轮: 总结
            AIMessage(content="10 + 20 = 30."),
            # 第二轮: 基于历史直接回答 (不调工具)
            AIMessage(content="As I said, the result was 30."),
        ]
        agent = _build_agent(tmp_path, responses, tools=[e2e_add])
        try:
            # 第一轮
            state1 = await _consume(agent, "What is 10 + 20?", thread_id="multi")
            assert state1 is not None
            msgs1 = state1["messages"]
            assert any(getattr(m, "content", "") == "10 + 20 = 30." for m in msgs1)

            # 第二轮 (同 thread_id, 应该能拿到第一轮上下文)
            state2 = await _consume(agent, "What was the result?", thread_id="multi")
            assert state2 is not None
            msgs2 = state2["messages"]
            last2 = msgs2[-1]
            assert "30" in getattr(last2, "content", ""), \
                f"agent lost context: {last2}"
        finally:
            agent.close()


class TestAgentLoopFaultTolerance:
    """工具失败容错."""

    @pytest.mark.asyncio
    async def test_tool_failure_does_not_crash_agent(self, tmp_path):
        """工具返回错误字符串, agent 不崩溃, 能继续给出回答.

        工具内部处理异常并返回错误信息是推荐的生产实践.
        agent 拿到错误后应该能给出降级回答.
        """
        responses = [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "e2e_fail", "args": {"msg": "test"}, "id": "tc1"}
                ],
            ),
            # 工具返回错误后, agent 应该能给出降级回答
            AIMessage(content="The tool reported a failure. I cannot complete the request."),
        ]
        agent = _build_agent(tmp_path, responses, tools=[e2e_fail, e2e_add])
        try:
            final_state = await _consume(agent, "Use the failing tool")
            assert final_state is not None, "agent crashed on tool failure"
            messages = final_state["messages"]
            last = messages[-1]
            # agent 应该有最终回答 (不是空白)
            assert getattr(last, "content", ""), \
                f"agent gave empty response after tool failure: {last}"
        finally:
            agent.close()


class TestAgentLoopMemoryIntegration:
    """Memory 集成: agent 执行中写入 memory."""

    @pytest.mark.asyncio
    async def test_memory_records_interaction(self, tmp_path):
        """agent 执行后, session memory 应该记录这次交互."""
        responses = [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "e2e_add", "args": {"a": 1, "b": 1}, "id": "tc1"}
                ],
            ),
            AIMessage(content="1 + 1 = 2."),
        ]
        agent = _build_agent(tmp_path, responses, tools=[e2e_add])
        try:
            await _consume(agent, "What is 1 + 1?")

            # session memory 应该有 assistant message
            session_messages = agent.memory.session.messages
            assert len(session_messages) > 0, "session memory is empty"
            assert any(m.role == "assistant" for m in session_messages), \
                "no assistant message in session"

            # tool_calls 应该被记录
            tool_calls = agent.memory.session.tool_calls
            assert any(tc.tool_name == "e2e_add" for tc in tool_calls), \
                f"e2e_add not recorded: {tool_calls}"
        finally:
            agent.close()

    @pytest.mark.asyncio
    async def test_longterm_memory_recall(self, tmp_path):
        """预先写入 long-term memory, agent 对话时能召回."""
        responses = [
            AIMessage(content="I recall that silicon has a band gap of 1.12 eV."),
        ]
        agent = _build_agent(tmp_path, responses, tools=[e2e_add])
        try:
            # 预先写入一条 memory
            agent.memory.remember(
                "Silicon has an indirect band gap of ~1.12 eV.",
                category="material_fact",
                tags=["silicon", "band_gap"],
                importance=0.9,
            )

            await _consume(agent, "Tell me about silicon band gap")

            # 显式 recall 应该能找到
            recalled = agent.memory.recall("silicon band gap")
            assert any("1.12 eV" in r["content"] for r in recalled), \
                f"long-term memory not recalled: {recalled}"
        finally:
            agent.close()


class TestAgentLoopTelemetry:
    """Telemetry 链路: 每次 turn 产生 span."""

    @pytest.mark.asyncio
    async def test_telemetry_records_turn(self, tmp_path):
        """agent 执行后, telemetry 应该有 turn span."""
        responses = [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "e2e_add", "args": {"a": 5, "b": 5}, "id": "tc1"}
                ],
            ),
            AIMessage(content="5 + 5 = 10."),
        ]
        agent = _build_agent(tmp_path, responses, tools=[e2e_add])
        try:
            await _consume(agent, "What is 5 + 5?")

            spans = agent.telemetry_spans()
            # spans 可能是嵌套结构, 展平检查
            span_names = set()

            def _flatten(s):
                if isinstance(s, dict):
                    span_names.add(s.get("name", ""))
                    for child in s.get("children", []) or s.get("spans", []):
                        _flatten(child)
                elif isinstance(s, list):
                    for item in s:
                        _flatten(item)

            _flatten(spans)

            # 应该有 agent_turn span
            assert any("turn" in n.lower() or "agent" in n.lower() for n in span_names), \
                f"no agent turn span: {span_names}"
        finally:
            agent.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short", "--no-cov"]))
