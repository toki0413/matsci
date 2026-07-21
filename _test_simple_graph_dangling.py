"""Smoke test: _build_simple_graph 走 fallback 路径时 dangling tool_calls 真的被修.

复现: deepagents 未装 → build_graph 走 _build_simple_graph → create_react_agent
无 middleware → dangling tool_calls 400. 修复: pre_model_hook 返回
llm_input_messages (不经 add_messages reducer, 顺序保留).

测试场景:
1. _patch_messages 被调时 dangling tool_calls 被插 synthetic ToolMessage
2. AIMessage + dangling → patch 后序列对 (ToolMessage 紧跟 AIMessage)
3. file multimodal block 被剥成 text 占位
4. pre_model_hook 包装: 模拟 state, 验证返回的 llm_input_messages 已 patch
5. create_react_agent 用 pre_model_hook 能正常 build (真实 model 类型校验过)
"""
import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).parent / "agent"
sys.path.insert(0, str(AGENT_ROOT))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from huginn.agent.middlewares import FixDanglingToolCallsMiddleware


def test_patch_inserts_synthetic_tool_message_in_order():
    """AIMessage(tool_calls=[X1, X2]) 只 X1 有 ToolMessage → patch 后 X1 紧跟
    AIMessage, X2 紧跟 X1 (synthetic), 然后才是后续消息. 顺序对."""
    mw = FixDanglingToolCallsMiddleware()
    ai = AIMessage(
        content="let me read",
        tool_calls=[
            {"id": "X1", "name": "bash", "args": {"cmd": "ls"}},
            {"id": "X2", "name": "bash", "args": {"cmd": "cat"}},
        ],
    )
    tm1 = ToolMessage(content="file_a\nfile_b", tool_call_id="X1", name="bash")
    user = HumanMessage(content="next question")

    messages = [ai, tm1, user]
    patched = mw._patch_messages(list(messages))

    assert len(patched) == 4, f"expected 4 messages after patch, got {len(patched)}: {patched}"
    assert patched[0] is ai, "AIMessage should stay at position 0"
    assert patched[1] is tm1, "existing ToolMessage should stay at position 1"
    assert isinstance(patched[2], ToolMessage), f"position 2 should be synthetic ToolMessage, got {type(patched[2])}"
    assert patched[2].tool_call_id == "X2", f"synthetic should be for X2, got {patched[2].tool_call_id}"
    assert patched[3] is user, "HumanMessage should be at the end"
    print("OK test_patch_inserts_synthetic_tool_message_in_order")


def test_patch_no_change_when_all_answered():
    """所有 tool_calls 都有 ToolMessage → patch 是 no-op, 长度不变."""
    mw = FixDanglingToolCallsMiddleware()
    ai = AIMessage(
        content="let me read",
        tool_calls=[{"id": "X1", "name": "bash", "args": {"cmd": "ls"}}],
    )
    tm = ToolMessage(content="ok", tool_call_id="X1", name="bash")
    messages = [ai, tm]
    patched = mw._patch_messages(list(messages))
    assert len(patched) == 2, f"no-op expected, got {len(patched)}"
    print("OK test_patch_no_change_when_all_answered")


def test_patch_handles_file_blocks():
    """AIMessage content 里有 file block → 转 text 占位, 不让 DeepSeek 400."""
    mw = FixDanglingToolCallsMiddleware()
    ai = AIMessage(
        content=[
            {"type": "text", "text": "reading file"},
            {"type": "file", "mime_type": "application/pdf", "base64": "abc123"},
        ],
        tool_calls=[],
    )
    patched = mw._patch_messages([ai])
    assert isinstance(patched[0].content, list), "content still list"
    assert patched[0].content[1]["type"] == "text", f"file block should become text, got {patched[0].content[1]}"
    assert "omitted" in patched[0].content[1]["text"], f"placeholder text missing, got {patched[0].content[1]['text']}"
    print("OK test_patch_handles_file_blocks")


def test_pre_model_hook_returns_patched_llm_input():
    """模拟 _build_simple_graph 里的 _pre_model_hook: 喂带 dangling 的 state,
    返回的 llm_input_messages 应该已 patch, 且不污染原 state messages."""
    mw = FixDanglingToolCallsMiddleware()

    def _pre_model_hook(state):
        msgs = state.get("messages", []) or []
        patched = mw._patch_messages(list(msgs))
        return {"llm_input_messages": patched}

    ai = AIMessage(
        content="let me read",
        tool_calls=[{"id": "X1", "name": "bash", "args": {"cmd": "ls"}}],
    )
    user = HumanMessage(content="next")
    state = {"messages": [ai, user]}

    result = _pre_model_hook(state)

    # llm_input_messages 应该被 patch (插了 synthetic ToolMessage)
    assert "llm_input_messages" in result, f"missing llm_input_messages key: {result}"
    llm_msgs = result["llm_input_messages"]
    assert len(llm_msgs) == 3, f"expected 3 after patch, got {len(llm_msgs)}: {llm_msgs}"
    assert isinstance(llm_msgs[1], ToolMessage), f"position 1 should be ToolMessage, got {type(llm_msgs[1])}"
    assert llm_msgs[1].tool_call_id == "X1"

    # 原 state.messages 不被污染 (只读 list(state["messages"]) 的 copy)
    assert len(state["messages"]) == 2, "original state messages should not be modified"
    assert "messages" not in result, "pre_model_hook 不应返回 messages key (会触发 add_messages reducer)"

    print("OK test_pre_model_hook_returns_patched_llm_input")


def test_create_react_agent_accepts_pre_model_hook():
    """真实 create_react_agent 接受 pre_model_hook + prompt, model 类型校验过.
    用真实 ChatModel (fake 但 Runnable 兼容) 验证不再因 _PatchedModel 非 Runnable 报错."""
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langgraph.prebuilt import create_react_agent

    # FakeMessagesListChatModel 是 Runnable, 通过 create_react_agent 类型校验
    fake_model = FakeMessagesListChatModel(responses=[AIMessage(content="ok")])

    def _hook(state):
        return {"llm_input_messages": state.get("messages", [])}

    agent = create_react_agent(
        model=fake_model,
        tools=[],
        prompt=SystemMessage(content="test"),
        pre_model_hook=_hook,
        checkpointer=None,
    )
    assert agent is not None, "agent build failed"
    print("OK test_create_react_agent_accepts_pre_model_hook")


if __name__ == "__main__":
    test_patch_inserts_synthetic_tool_message_in_order()
    test_patch_no_change_when_all_answered()
    test_patch_handles_file_blocks()
    test_pre_model_hook_returns_patched_llm_input()
    test_create_react_agent_accepts_pre_model_hook()
    print("[simple_graph_dangling] all tests OK")
