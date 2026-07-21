"""验证 conversation_tree_to_messages 剥掉 dangling tool_calls.

构造 3 种场景:
  1. 正常: AIMessage(tool_calls) 后跟 ToolMessage → 保留 tool_calls
  2. dangling: AIMessage(tool_calls) 后没 ToolMessage → 剥掉 tool_calls
  3. 部分 dangling: AIMessage 2 个 tool_calls, 只 1 个有 ToolMessage → 保留 1 个
"""
import sys
sys.path.insert(0, "agent")

from huginn.utils.conversation_tree import ConversationTree
from huginn.context_builder import ContextBuilder


def _make_cb(tree):
    """构造一个最小 ContextBuilder, 只用 conversation_tree 字段."""
    cb = ContextBuilder.__new__(ContextBuilder)
    cb._conversation_tree = tree
    return cb


def _add(tree, role, content, **meta):
    return tree.add_message(role, content, metadata=meta or None)


def test_normal_tool_calls_kept():
    """AIMessage(tool_calls) + ToolMessage → 保留 tool_calls."""
    tree = ConversationTree()
    _add(tree, "user", "hello")
    _add(tree, "assistant", "let me check", tool_calls=[
        {"id": "call_1", "name": "bash_tool", "args": {"command": ["ls"]}}
    ])
    _add(tree, "tool", "file.txt", tool_call_id="call_1", name="bash_tool")
    _add(tree, "user", "thanks")
    cb = _make_cb(tree)
    msgs = cb.conversation_tree_to_messages()
    # user, assistant(tool_calls), tool, user = 4 条 (最后 user 被排除, 因为是当前消息)
    # 实际: path[:-1] 排除最后一个 node
    assert len(msgs) == 3, f"normal: expected 3 msgs, got {len(msgs)}"
    ai = msgs[1]
    assert ai.tool_calls, f"normal: assistant should keep tool_calls, got {ai}"
    assert ai.tool_calls[0]["id"] == "call_1"
    print("OK test_normal_tool_calls_kept")


def test_dangling_tool_calls_stripped():
    """AIMessage(tool_calls) 后无 ToolMessage → 剥掉 tool_calls."""
    tree = ConversationTree()
    _add(tree, "user", "hello")
    _add(tree, "assistant", "let me check", tool_calls=[
        {"id": "call_dangling", "name": "bash_tool", "args": {"command": ["ls"]}}
    ])
    # 没有 ToolMessage — 模拟 timeout 中断
    _add(tree, "user", "next message after interruption")
    cb = _make_cb(tree)
    msgs = cb.conversation_tree_to_messages()
    # user, assistant, user = 3 (最后 user 排除 → 2)
    assert len(msgs) == 2, f"dangling: expected 2 msgs, got {len(msgs)}"
    ai = msgs[1]
    assert not ai.tool_calls, f"dangling: assistant tool_calls should be stripped, got {ai.tool_calls}"
    # content 保留
    assert "let me check" in ai.content
    print("OK test_dangling_tool_calls_stripped")


def test_partial_dangling():
    """2 个 tool_calls, 只 1 个有 ToolMessage → 保留有对应的那个."""
    tree = ConversationTree()
    _add(tree, "user", "hello")
    _add(tree, "assistant", "two calls", tool_calls=[
        {"id": "call_a", "name": "bash_tool", "args": {}},
        {"id": "call_b", "name": "code_tool", "args": {}},
    ])
    _add(tree, "tool", "bash result", tool_call_id="call_a", name="bash_tool")
    # call_b 没 ToolMessage
    _add(tree, "user", "next")
    cb = _make_cb(tree)
    msgs = cb.conversation_tree_to_messages()
    ai = msgs[1]
    assert len(ai.tool_calls) == 1, f"partial: expected 1 kept, got {len(ai.tool_calls)}"
    assert ai.tool_calls[0]["id"] == "call_a"
    print("OK test_partial_dangling")


if __name__ == "__main__":
    test_normal_tool_calls_kept()
    test_dangling_tool_calls_stripped()
    test_partial_dangling()
    print("[dangling_tool_calls_fix] all tests OK")
