"""huginn/agent/middlewares.py 单元测试.

补充 test_middleware_vision_cv.py 之外的覆盖:
- 单位匹配逻辑 (_NUMERIC_PATTERN / _CLAIM_KEYWORDS)
- DeliverableCoverage 检查 (横向 _check_coverage + 纵向 _check_layer_coverage)
- keyword 提取 (_extract_keywords)
- _patch_messages 的 orphan tool call 修复 (重点 — deepagents compaction
  残留 AIMessage.tool_calls 没 ToolMessage 响应 → DeepSeek 400, 这是之前
  修过 deprecation 的代码)
- _reorder_tool_messages (重排 + 补 cancelled ToolMessage)
- RateLimitMiddleware._estimate_tokens + wrap_model_call (mock limiter)
- _parse_data_url / _omitted_block / _compute_strip_flag 等纯函数

测试原则:
- 不依赖真实 LLM 调用 — build_cv_context / get_rate_limiter 用 mock
- 不修改 huginn/agent/middlewares.py 源文件
- 中文注释, 英文标识符
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from huginn.agent.middlewares import (
    DeliverableCoverageMiddleware,
    FixDanglingToolCallsMiddleware,
    RateLimitMiddleware,
)

# ── 辅助: 轻量 request double ────────────────────────────────────
# wrap_model_call / before_agent 接收的 request 只需要两个能力:
#   .messages 属性可读
#   .override(messages=...) 返回新 request
# 这里用一个 dataclass-like 替身, 避开 deepagents AgentRequest 重依赖.


class _FakeRequest:
    """deepagents AgentRequest 的最小替身, 仅供 middleware 单测使用."""

    def __init__(self, messages: list) -> None:
        self.messages = list(messages)

    def override(self, **changes: Any) -> _FakeRequest:
        # 只支持 messages 覆盖, 其它字段 YAGNI
        return _FakeRequest(changes.get("messages", self.messages))


def _ai_with_tool_calls(*tcs: dict, content: str = "") -> AIMessage:
    """构造带 tool_calls 的 AIMessage."""
    return AIMessage(content=content, tool_calls=list(tcs))


def _tc(name: str, tid: str, args: dict | None = None) -> dict:
    """构造一个 tool_call dict."""
    return {"name": name, "args": args or {}, "id": tid}


# ───────────────────────────────────────────────────────────────────
# FixDanglingToolCallsMiddleware._compute_strip_flag (静态纯函数)
# ───────────────────────────────────────────────────────────────────


class TestComputeStripFlag:
    """_compute_strip_flag: 按 model capability 决定是否剥离 multimodal block."""

    def test_empty_model_name_fail_closed_strip(self):
        """空 model name → fail-closed, 剥离 (True)."""
        # 生产 core.py 默认 model_name="" 走的就是这条路径
        assert FixDanglingToolCallsMiddleware._compute_strip_flag("") is True

    def test_vision_true_model_keeps_multimodal(self):
        """gpt-4o 在 registry 是 vision=True → 不剥离 (False)."""
        assert FixDanglingToolCallsMiddleware._compute_strip_flag("gpt-4o") is False

    def test_vision_false_model_strips_multimodal(self):
        """deepseek-chat 在 registry 是 vision=False → 剥离 (True)."""
        assert (
            FixDanglingToolCallsMiddleware._compute_strip_flag("deepseek-chat")
            is True
        )

    def test_unknown_model_name_fail_closed(self):
        """未知名 → fail-closed, 剥离 (避免不支持 vision 的 model 400)."""
        assert (
            FixDanglingToolCallsMiddleware._compute_strip_flag("totally-bogus-xyz")
            is True
        )

    def test_registry_exception_fail_closed(self, monkeypatch):
        """get_model_capabilities 抛异常时 → fail-closed 剥离.

        测的是 try/except 兜底, 避免注册表加载失败把整个 middleware 干挂.
        """
        import huginn.models.registry as _reg

        def _boom(_name: str):
            raise RuntimeError("registry exploded")

        monkeypatch.setattr(_reg, "get_model_capabilities", _boom)
        # 注意: middlewares.py 内是 lazy import, 所以 monkeypatch 模块对象即可
        assert (
            FixDanglingToolCallsMiddleware._compute_strip_flag("deepseek-chat")
            is True
        )

    def test_init_caches_strip_flag(self):
        """__init__ 应该把 _compute_strip_flag 结果缓存到 _strip_multimodal."""
        mw_text = FixDanglingToolCallsMiddleware("deepseek-chat")
        mw_vision = FixDanglingToolCallsMiddleware("gpt-4o")
        mw_unknown = FixDanglingToolCallsMiddleware("")
        assert mw_text._strip_multimodal is True
        assert mw_vision._strip_multimodal is False
        assert mw_unknown._strip_multimodal is True


# ───────────────────────────────────────────────────────────────────
# FixDanglingToolCallsMiddleware._parse_data_url (静态纯函数)
# ───────────────────────────────────────────────────────────────────


class TestParseDataUrl:
    """_parse_data_url: 拆 data:{mime};base64,{payload} → (mime, payload)."""

    def test_standard_png_data_url(self):
        url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
        mime, payload = FixDanglingToolCallsMiddleware._parse_data_url(url)
        assert mime == "image/png"
        assert payload == "iVBORw0KGgoAAAANSUhEUg=="

    def test_jpeg_data_url(self):
        url = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ=="
        mime, payload = FixDanglingToolCallsMiddleware._parse_data_url(url)
        assert mime == "image/jpeg"
        assert payload == "/9j/4AAQSkZJRgABAQ=="

    def test_non_data_url_returns_empty(self):
        """非 data: 前缀 → ('', '')."""
        mime, payload = FixDanglingToolCallsMiddleware._parse_data_url(
            "https://example.com/foo.png"
        )
        assert (mime, payload) == ("", "")

    def test_no_comma_returns_empty(self):
        """没逗号分隔 → ('', '')."""
        mime, payload = FixDanglingToolCallsMiddleware._parse_data_url(
            "data:image/png;base64"
        )
        assert (mime, payload) == ("", "")

    def test_empty_url_returns_empty(self):
        mime, payload = FixDanglingToolCallsMiddleware._parse_data_url("")
        assert (mime, payload) == ("", "")

    def test_no_semicolon_uses_whole_rest_as_mime(self):
        """没有 ; 时 header[5:] 整段当 mime (兼容非标准格式)."""
        url = "data:image/gif,R0lGODlh"
        mime, payload = FixDanglingToolCallsMiddleware._parse_data_url(url)
        # "image/gif" (没 ; base64 时 _rest = "image/gif")
        assert mime == "image/gif"
        assert payload == "R0lGODlh"


# ───────────────────────────────────────────────────────────────────
# FixDanglingToolCallsMiddleware._omitted_block (静态纯函数)
# ───────────────────────────────────────────────────────────────────


class TestOmittedBlock:
    """_omitted_block: 把 multimodal block 退回 [content omitted] text 占位."""

    def test_image_url_block_becomes_text(self):
        block = {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
        out = FixDanglingToolCallsMiddleware._omitted_block(block)
        assert out["type"] == "text"
        assert "content omitted" in out["text"]
        assert "image_url" in out["text"]
        assert "mime=image/png" not in out["text"]  # mime_type 缺失 → 空串

    def test_includes_mime_type_and_b64_length(self):
        block = {
            "type": "image",
            "mime_type": "image/png",
            "base64": "abcd1234",  # 8 chars
        }
        out = FixDanglingToolCallsMiddleware._omitted_block(block)
        assert out["type"] == "text"
        assert "mime=image/png" in out["text"]
        assert "8 chars base64" in out["text"]
        assert "model does not support multimodal" in out["text"]

    def test_missing_type_uses_unknown(self):
        block = {"base64": "xyz"}
        out = FixDanglingToolCallsMiddleware._omitted_block(block)
        assert out["type"] == "text"
        assert "[unknown content omitted" in out["text"]

    def test_empty_base64_shows_zero_chars(self):
        block = {"type": "file", "mime_type": "application/pdf"}
        out = FixDanglingToolCallsMiddleware._omitted_block(block)
        assert "0 chars base64" in out["text"]


# ───────────────────────────────────────────────────────────────────
# FixDanglingToolCallsMiddleware._reorder_tool_messages
# (重点: orphan tool call 修复 + 顺序重排)
# ───────────────────────────────────────────────────────────────────


class TestReorderToolMessages:
    """_reorder_tool_messages: 让 ToolMessage 紧跟 AIMessage(tool_calls),
    并在 force_cancel_orphans=True 时给 orphan tool_call 补 cancelled ToolMessage.
    """

    def test_empty_messages_returns_same_list(self):
        mw = FixDanglingToolCallsMiddleware("")
        empty: list = []
        # 空列表直接返回 (identity)
        assert mw._reorder_tool_messages(empty) is empty

    def test_correct_order_no_change(self):
        """顺序已经正确 (ToolMessage 紧跟 AIMessage) → 内容/顺序保持.

        注意: 实现里 dedup 逻辑会 rebuilt (因为 trailing ToolMessage 被视为
        "已被 AIMessage 吸收" 的 duplicate, 触发 _changed=True), 所以测内容
        等价而非对象 identity.
        """
        mw = FixDanglingToolCallsMiddleware("")
        ai = _ai_with_tool_calls(_tc("foo", "tc1"))
        tool = ToolMessage(content="result", name="foo", tool_call_id="tc1")
        msgs = [ai, tool]
        out = mw._reorder_tool_messages(msgs)
        # 顺序 + 内容保持: AIMessage 在前, 对应 ToolMessage 紧跟
        assert len(out) == 2
        assert isinstance(out[0], AIMessage)
        assert isinstance(out[1], ToolMessage)
        assert out[1].tool_call_id == "tc1"
        assert out[1].content == "result"

    def test_misordered_tool_message_gets_reordered(self):
        """ToolMessage 被 SystemMessage 隔开 → 重排让 ToolMessage 紧跟 AIMessage.

        场景: langgraph add_messages reducer ID 冲突或 ctx_builder 注入
        SystemMessage 时会把 ToolMessage 推离 AIMessage, DeepSeek 会 400.
        """
        mw = FixDanglingToolCallsMiddleware("")
        ai = _ai_with_tool_calls(_tc("foo", "tc1"))
        tool = ToolMessage(content="result", name="foo", tool_call_id="tc1")
        sys_msg = SystemMessage(content="reminder")
        msgs = [ai, sys_msg, tool]
        out = mw._reorder_tool_messages(msgs)
        # 不是原 list (发生了重排)
        assert out is not msgs
        # ToolMessage 应该紧跟 AIMessage, SystemMessage 在后
        assert isinstance(out[0], AIMessage)
        assert isinstance(out[1], ToolMessage)
        assert out[1].tool_call_id == "tc1"
        assert isinstance(out[2], SystemMessage)

    def test_orphan_tool_call_cancelled_when_forced(self):
        """orphan AIMessage.tool_calls (无对应 ToolMessage) + force=True → 补 cancelled.

        这是 deepagents summarization compaction 后的典型残留:
        AIMessage 还在但 ToolMessage 被 drop, DeepSeek 严格要求每个 tool_call_id
        都要有对应 ToolMessage 否则 400.
        """
        mw = FixDanglingToolCallsMiddleware("")
        ai = _ai_with_tool_calls(_tc("foo", "orphan-1"))
        msgs = [ai]
        out = mw._reorder_tool_messages(msgs, force_cancel_orphans=True)
        assert out is not msgs  # 发生了变化
        assert len(out) == 2
        assert isinstance(out[0], AIMessage)
        assert isinstance(out[1], ToolMessage)
        assert out[1].tool_call_id == "orphan-1"
        assert "cancelled" in out[1].content.lower()
        assert "summarization compaction" in out[1].content

    def test_orphan_tool_call_not_cancelled_by_default(self):
        """force_cancel_orphans=False (默认) 时不补 cancelled ToolMessage."""
        mw = FixDanglingToolCallsMiddleware("")
        ai = _ai_with_tool_calls(_tc("foo", "orphan-2"))
        msgs = [ai]
        out = mw._reorder_tool_messages(msgs)  # force=False
        # 没发生任何变化 → 返回原 list
        assert out is msgs

    def test_duplicate_tool_messages_deduplicated(self):
        """同一 tool_call_id 出现两次 ToolMessage → 只保留一条避免重复.

        实现语义: tool_msgs_by_id 后面同 id 覆盖前面 (last wins),
        AIMessage 吸收时取的是 last; 之后遍历到所有同 id ToolMessage 都
        因 used_tool_ids 标记而跳过.
        """
        mw = FixDanglingToolCallsMiddleware("")
        ai = _ai_with_tool_calls(_tc("foo", "dup-1"))
        tool1 = ToolMessage(content="result-1", name="foo", tool_call_id="dup-1")
        tool2 = ToolMessage(content="result-2", name="foo", tool_call_id="dup-1")
        msgs = [ai, tool1, tool2]
        out = mw._reorder_tool_messages(msgs)
        assert out is not msgs  # dedup 触发 _changed
        # 只保留一条 ToolMessage (last wins: tool2 被 tool_msgs_by_id 记录)
        tool_msgs = [m for m in out if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].content == "result-2"

    def test_multiple_tool_calls_each_gets_response(self):
        """一个 AIMessage 带 2 个 tool_calls → 2 个 ToolMessage 都紧跟其后."""
        mw = FixDanglingToolCallsMiddleware("")
        ai = _ai_with_tool_calls(
            _tc("foo", "tc-a"), _tc("bar", "tc-b")
        )
        tool_a = ToolMessage(content="ra", name="foo", tool_call_id="tc-a")
        tool_b = ToolMessage(content="rb", name="bar", tool_call_id="tc-b")
        # 故意打乱顺序 (b 在 a 前面)
        msgs = [ai, tool_b, tool_a]
        out = mw._reorder_tool_messages(msgs)
        # 应按 AIMessage.tool_calls 顺序排列 (a, b)
        assert isinstance(out[0], AIMessage)
        assert out[1].tool_call_id == "tc-a"
        assert out[2].tool_call_id == "tc-b"

    def test_orphan_tool_call_with_unknown_name_uses_unknown(self):
        """orphan tool_call 没有 name 字段时 → cancelled msg 用 'unknown'."""
        mw = FixDanglingToolCallsMiddleware("")
        # name 缺失
        ai = _ai_with_tool_calls({"name": "", "args": {}, "id": "no-name-tc"})
        msgs = [ai]
        out = mw._reorder_tool_messages(msgs, force_cancel_orphans=True)
        tool_msg = next(m for m in out if isinstance(m, ToolMessage))
        # name="" → "unknown" 兜底
        assert tool_msg.name == "unknown"

    def test_tool_call_without_id_skipped(self):
        """tool_call 没 id 字段 → 跳过 (不补 cancelled, 不匹配 ToolMessage)."""
        mw = FixDanglingToolCallsMiddleware("")
        ai = _ai_with_tool_calls({"name": "foo", "args": {}, "id": None})
        msgs = [ai]
        out = mw._reorder_tool_messages(msgs, force_cancel_orphans=True)
        # 没有 id, 不会触发 orphan 修复
        assert len(out) == 1
        assert isinstance(out[0], AIMessage)

    def test_invalid_tool_calls_also_repaired(self):
        """AIMessage.invalid_tool_calls 里的 tool_call 也应被响应/补 cancelled.

        DeepSeek 偶尔会把 malformed tool_call 塞到 invalid_tool_calls, 同样
        会触发 400, 必须一并修复. LangChain 要求 invalid_tool_calls 的 args
        是 string (malformed 无法解析成 dict).
        """
        mw = FixDanglingToolCallsMiddleware("")
        ai = AIMessage(
            content="",
            tool_calls=[],
            invalid_tool_calls=[{"name": "bad", "args": "", "id": "bad-1"}],
        )
        msgs = [ai]
        out = mw._reorder_tool_messages(msgs, force_cancel_orphans=True)
        assert len(out) == 2
        assert isinstance(out[1], ToolMessage)
        assert out[1].tool_call_id == "bad-1"

    def test_orphan_tool_message_without_ai_kept_in_place(self):
        """孤儿 ToolMessage (没对应 AIMessage) → 保留原位, 不删除."""
        mw = FixDanglingToolCallsMiddleware("")
        tool = ToolMessage(content="orphan", name="foo", tool_call_id="no-ai")
        msgs = [SystemMessage("sys"), tool]
        out = mw._reorder_tool_messages(msgs)
        # ToolMessage 不在任何 AIMessage 后面, used_tool_ids 不会标记它 → 保留原位
        assert out is msgs  # 没变化


# ───────────────────────────────────────────────────────────────────
# FixDanglingToolCallsMiddleware._patch_messages
# (重点 — orphan tool call 修复的入口)
# ───────────────────────────────────────────────────────────────────


class TestPatchMessagesOrphanRepair:
    """_patch_messages: 完整的 orphan tool call 修复流程.

    重点测试场景: deepagents summarization compaction 后, AIMessage.tool_calls
    残留但对应 ToolMessage 被 drop, 直接发给 DeepSeek 会 400. middleware 必须补
    cancelled ToolMessage 让消息序列合法.
    """

    def test_empty_messages_returns_empty(self):
        mw = FixDanglingToolCallsMiddleware("")
        assert mw._patch_messages([]) == []

    def test_vision_false_orphan_tool_call_cancelled(self, monkeypatch):
        """vision=False 路径: orphan AIMessage.tool_calls → 补 cancelled ToolMessage.

        模拟生产场景: deepseek-chat 模型 + compaction 后只剩 AIMessage,
        ToolMessage 被 summarizer 删掉. middleware 必须补 cancelled.
        """
        # vision=False 路径不需要调 build_cv_context (没 image block)
        mw = FixDanglingToolCallsMiddleware("deepseek-chat")
        assert mw._strip_multimodal is True

        ai = _ai_with_tool_calls(_tc("foo", "orphan-x"))
        out = mw._patch_messages([ai])

        # 应该多出一条 cancelled ToolMessage
        assert len(out) == 2
        assert isinstance(out[1], ToolMessage)
        assert out[1].tool_call_id == "orphan-x"
        assert "cancelled" in out[1].content.lower()

    def test_vision_false_correct_order_unchanged_when_no_image(self):
        """vision=False 但消息全是 text + 顺序正确 → 不应改 (除了 strip 走一遍).

        无 image block, content 是 str, _patch_messages 不应破坏消息结构.
        """
        mw = FixDanglingToolCallsMiddleware("deepseek-chat")
        ai = _ai_with_tool_calls(_tc("foo", "ok-1"))
        tool = ToolMessage(content="r", name="foo", tool_call_id="ok-1")
        msgs = [ai, tool]
        out = mw._patch_messages(msgs)
        # 顺序正确 + 无 orphan + 无 image → 应保持
        assert len(out) == 2
        assert isinstance(out[0], AIMessage)
        assert isinstance(out[1], ToolMessage)

    def test_vision_true_orphan_not_cancelled_via_default_path(self):
        """vision=True 路径: _reorder_tool_messages 不带 force_cancel_orphans.

        注意: 这是当前实现的行为. vision=True 模型走 _reorder_tool_messages
        默认 force=False, orphan 不补 cancelled. 测试锁住这个行为, 防止
        后续重构误改.
        """
        mw = FixDanglingToolCallsMiddleware("gpt-4o")
        assert mw._strip_multimodal is False

        ai = _ai_with_tool_calls(_tc("foo", "orphan-vision-true"))
        out = mw._patch_messages([ai])
        # vision=True 不补 cancelled (force=False)
        # AIMessage 没变化, _changed=False → 返回原 list
        assert out == [ai] or len(out) == 1
        assert isinstance(out[0], AIMessage)

    def test_vision_true_misordered_tool_messages_reordered(self):
        """vision=True 路径: ToolMessage 错位时仍重排 (不补 orphan 但修顺序)."""
        mw = FixDanglingToolCallsMiddleware("gpt-4o")
        ai = _ai_with_tool_calls(_tc("foo", "reorder-me"))
        tool = ToolMessage(content="r", name="foo", tool_call_id="reorder-me")
        sys_msg = SystemMessage(content="interruption")
        msgs = [ai, sys_msg, tool]
        out = mw._patch_messages(msgs)
        # 重排了, ToolMessage 紧跟 AIMessage
        assert isinstance(out[0], AIMessage)
        assert isinstance(out[1], ToolMessage)
        assert isinstance(out[2], SystemMessage)

    def test_no_tool_calls_no_modification(self):
        """消息没任何 tool_calls → 不修改 (vision=False text-only 路径)."""
        mw = FixDanglingToolCallsMiddleware("deepseek-chat")
        msgs = [
            HumanMessage(content="hi"),
            AIMessage(content="hello"),
        ]
        out = mw._patch_messages(msgs)
        # 没改: 没工具调用, 没 image block, content 都是 str
        # 返回原 list (因为 _changed_blocks=False)
        assert out is msgs

    def test_text_only_messages_returned_unchanged(self):
        """所有消息都是纯 text → 返回原 list (零开销)."""
        mw = FixDanglingToolCallsMiddleware("deepseek-chat")
        msgs = [
            SystemMessage(content="system"),
            HumanMessage(content="user"),
            AIMessage(content="assistant"),
        ]
        out = mw._patch_messages(msgs)
        assert out is msgs

    def test_orphan_with_human_message_after_ai(self):
        """AIMessage(tool_calls) + HumanMessage + 无 ToolMessage → 补 cancelled.

        生产场景: 用户在 tool 调用未完成时打断, 留下 orphan AIMessage.tool_calls.
        """
        mw = FixDanglingToolCallsMiddleware("deepseek-chat")
        ai = _ai_with_tool_calls(_tc("search", "aborted-1"))
        human = HumanMessage(content="never mind, do this instead")
        out = mw._patch_messages([ai, human])

        # 应该补 cancelled ToolMessage, 顺序: AI, Tool(cancelled), Human
        assert len(out) == 3
        assert isinstance(out[0], AIMessage)
        assert isinstance(out[1], ToolMessage)
        assert isinstance(out[2], HumanMessage)
        assert out[1].tool_call_id == "aborted-1"
        assert "cancelled" in out[1].content.lower()

    def test_mixed_orphan_and_answered_tool_calls(self):
        """一个 AIMessage 带 2 个 tool_calls, 1 个有 ToolMessage 1 个 orphan.

        场景: compaction 选择性 drop (有的 ToolMessage 留下, 有的被删).
        middleware 应保留已有的, 给 orphan 补 cancelled.
        """
        mw = FixDanglingToolCallsMiddleware("deepseek-chat")
        ai = _ai_with_tool_calls(
            _tc("foo", "answered"), _tc("bar", "orphan-z")
        )
        answered_tool = ToolMessage(
            content="real result", name="foo", tool_call_id="answered"
        )
        out = mw._patch_messages([ai, answered_tool])

        # 应有 3 条: AIMessage, answered ToolMessage, cancelled ToolMessage
        assert len(out) == 3
        assert isinstance(out[0], AIMessage)
        # answered 应该紧跟 (顺序保持)
        tool_msgs = [m for m in out if isinstance(m, ToolMessage)]
        ids = {t.tool_call_id for t in tool_msgs}
        assert ids == {"answered", "orphan-z"}
        # orphan 那条 content 应是 cancelled
        cancelled = next(t for t in tool_msgs if t.tool_call_id == "orphan-z")
        assert "cancelled" in cancelled.content.lower()


# ───────────────────────────────────────────────────────────────────
# FixDanglingToolCallsMiddleware wrap_model_call / before_agent
# (集成 _patch_messages 的薄包装层, 用 _FakeRequest 验证调用链)
# ───────────────────────────────────────────────────────────────────


class TestWrapModelCallDelegation:
    """wrap_model_call / before_agent 把 _patch_messages 结果通过 override 传下去."""

    def test_wrap_model_call_overrides_messages_and_calls_handler(self):
        mw = FixDanglingToolCallsMiddleware("deepseek-chat")
        ai = _ai_with_tool_calls(_tc("foo", "orphan-wrap"))
        req = _FakeRequest([ai])
        called_with: list = []

        def handler(r):
            called_with.append(r)
            return "ok"

        result = mw.wrap_model_call(req, handler)
        assert result == "ok"
        assert called_with and called_with[0] is not req
        # handler 拿到的 request.messages 应已被 patch (含 cancelled ToolMessage)
        patched_msgs = called_with[0].messages
        assert any(
            isinstance(m, ToolMessage) and m.tool_call_id == "orphan-wrap"
            for m in patched_msgs
        )

    def test_before_agent_overrides_messages(self):
        mw = FixDanglingToolCallsMiddleware("deepseek-chat")
        ai = _ai_with_tool_calls(_tc("foo", "orphan-ba"))
        req = _FakeRequest([ai])
        captured: list = []

        def handler(r):
            captured.append(r)
            return "done"

        mw.before_agent(req, handler)
        assert captured
        patched = captured[0].messages
        assert any(
            isinstance(m, ToolMessage) and m.tool_call_id == "orphan-ba"
            for m in patched
        )

    def test_before_agent_no_handler_returns_none(self):
        """before_agent 没 handler → 返回 None (兼容 deepagents 协议)."""
        mw = FixDanglingToolCallsMiddleware("deepseek-chat")
        req = _FakeRequest([HumanMessage("hi")])
        assert mw.before_agent(req, None) is None

    def test_before_agent_empty_messages_skips_patch(self):
        """request.messages 为空 → 不 patch (避免无意义 override)."""
        mw = FixDanglingToolCallsMiddleware("")
        req = _FakeRequest([])
        called = {"n": 0}

        def handler(r):
            called["n"] += 1
            return "ok"

        mw.before_agent(req, handler)
        # handler 仍被调用, 但 messages 是空 (无 orphan 需修)
        assert called["n"] == 1

    def test_wrap_tool_call_passthrough(self):
        """wrap_tool_call 不做 patch, 直接转发."""
        mw = FixDanglingToolCallsMiddleware("deepseek-chat")
        sentinel = object()
        req = _FakeRequest([])
        out = mw.wrap_tool_call(req, lambda r: sentinel)
        assert out is sentinel

    def test_awrap_model_call_delegates(self):
        """awrap_model_call 异步包装 _patch_messages."""
        import asyncio

        mw = FixDanglingToolCallsMiddleware("deepseek-chat")
        ai = _ai_with_tool_calls(_tc("foo", "async-orphan"))
        req = _FakeRequest([ai])

        async def handler(r):
            return r.messages

        loop = asyncio.new_event_loop()
        try:
            patched = loop.run_until_complete(mw.awrap_model_call(req, handler))
        finally:
            loop.close()
        assert any(
            isinstance(m, ToolMessage) and m.tool_call_id == "async-orphan"
            for m in patched
        )


# ───────────────────────────────────────────────────────────────────
# RateLimitMiddleware._estimate_tokens (纯函数, 不依赖 LLM)
# ───────────────────────────────────────────────────────────────────


class TestEstimateTokens:
    """_estimate_tokens: 按 content 形态算 token 估算 (4 char ≈ 1 token).

    重点: list[multipart] content 不能 str(list) 兜底, 否则一条 multimodal
    message 会把 list repr 算进去撑爆 estimate 触发误拦.
    """

    @staticmethod
    def _make_mw() -> RateLimitMiddleware:
        """构造一个 RateLimitMiddleware, mock 掉 get_rate_limiter 避免 singleton 副作用."""
        from huginn.security.rate_limiter import TokenRateLimiter
        mw = RateLimitMiddleware.__new__(RateLimitMiddleware)
        mw._limiter = TokenRateLimiter()
        return mw

    def test_str_content(self):
        mw = self._make_mw()
        msg = HumanMessage(content="hello world!")  # 12 chars
        assert mw._estimate_tokens([msg]) == 12 // 4  # = 3

    def test_empty_messages_returns_at_least_one(self):
        mw = self._make_mw()
        # max(0//4, 1) = 1
        assert mw._estimate_tokens([]) == 1
        assert mw._estimate_tokens(None) == 1

    def test_list_text_blocks_aggregated(self):
        """list content 时累加每个 block 的 text, 不算 list repr."""
        mw = self._make_mw()
        msg = HumanMessage(content=[
            {"type": "text", "text": "abcd"},   # 4 chars
            {"type": "text", "text": "efgh"},   # 4 chars
        ])
        # 8 chars // 4 = 2 tokens
        assert mw._estimate_tokens([msg]) == 2

    def test_list_multimodal_block_does_not_explode(self):
        """multimodal block (image_url) 不应让 estimate 暴增.

        回归: 旧版 str(content) 兜底会把整个 list repr (含 base64) 算进去,
        一条 multimodal message 撑到 12M chars 触发误拦.
        """
        mw = self._make_mw()
        big_b64 = "x" * 100_000
        msg = HumanMessage(content=[
            {"type": "text", "text": "short"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{big_b64}"}},
        ])
        # text=5 chars, image_url block 走 str(block.get("text", block.get("content", "")))
        # → "" (没 text/content 字段) → 0 chars. 总 = 5 chars // 4 = 1
        # 关键: 不会算 100k base64
        estimate = mw._estimate_tokens([msg])
        assert estimate < 100, f"multimodal 不应爆增: got {estimate}"
        assert estimate >= 1

    def test_none_content_falls_back_to_repr_capped(self):
        """content=None 时用 str(msg) 但限长 1000, 防止巨大 msg 撑爆 estimate."""
        mw = self._make_mw()
        # LangChain 消息 content 不会是 None, 但 defensive 代码兜底. 用一个
        # 自定义类让 str(msg) 很长, 验证被 1000 截断.
        class _LongMsg:
            content = None

            def __str__(self) -> str:
                return "x" * 5000

        estimate = mw._estimate_tokens([_LongMsg()])
        # min(len(str(msg)), 1000) = 1000, 1000 // 4 = 250
        assert estimate == 250

    def test_non_str_non_list_content_uses_str(self):
        """content 是其它类型 (如 int) → str() 后算长度."""
        mw = self._make_mw()
        fake_msg = SimpleNamespace(content=12345)  # str → "12345" 5 chars
        assert mw._estimate_tokens([fake_msg]) == 1  # 5 // 4 = 1

    def test_dict_block_without_text_uses_content_field(self):
        """block 是 dict 但没 text 字段 → 退到 content 字段."""
        mw = self._make_mw()
        msg = HumanMessage(content=[
            {"type": "custom", "content": "abcdefgh"},  # 8 chars
        ])
        assert mw._estimate_tokens([msg]) == 2  # 8 // 4

    def test_non_dict_block_in_list_uses_str(self):
        """list 里有非 dict 元素 (如 str) → str() 后算长度."""
        mw = self._make_mw()
        # 直接传 list[str] content (LangChain 偶尔出现)
        msg = HumanMessage(content=["abcd", "efgh"])  # 4+4 = 8 chars
        assert mw._estimate_tokens([msg]) == 2


# ───────────────────────────────────────────────────────────────────
# RateLimitMiddleware.wrap_model_call (用 mock limiter)
# ───────────────────────────────────────────────────────────────────


class TestRateLimitWrapModelCall:
    """wrap_model_call: check_allowed 拦截 + handler 转发 + record_usage 记账."""

    def _make_mw(self, monkeypatch, *, allowed: bool, record_calls=None):
        """构造带 mock limiter 的 RateLimitMiddleware."""
        # 注意: lambda 不能用 `if record_calls` 短路 (空 list falsy 会跳过 append),
        # 直接 append 即可 — record_calls 默认是空 list, 调用方传非空 list 进来.
        def _record(*a, **k):
            if record_calls is not None:
                record_calls.append((a, k))

        limiter = SimpleNamespace(
            check_allowed=lambda *a, **k: (allowed, "blocked-test" if not allowed else ""),
            record_usage=_record,
        )
        # 直接 __new__ 跳过 __init__ 的 get_rate_limiter 调用
        mw = RateLimitMiddleware.__new__(RateLimitMiddleware)
        mw._limiter = limiter
        return mw

    def test_allowed_calls_handler_and_records(self, monkeypatch):
        record_calls: list = []
        mw = self._make_mw(monkeypatch, allowed=True, record_calls=record_calls)

        # mock _extract_usage 让 record_usage 拿到非零 token
        monkeypatch.setattr(mw, "_extract_usage", lambda r: (100, 50))

        sentinel = object()
        req = _FakeRequest([HumanMessage("hi")])
        out = mw.wrap_model_call(req, lambda r: sentinel)
        assert out is sentinel
        assert record_calls, "应该调 record_usage 记账"
        # record_usage("agent", in_tok, out_tok, thread_id=...)
        args, kwargs = record_calls[0]
        assert args[0] == "agent"
        assert args[1] == 100  # in_tok
        assert args[2] == 50  # out_tok

    def test_blocked_raises_rate_limit_exceeded(self, monkeypatch):
        from huginn.security.rate_limiter import RateLimitExceeded

        mw = self._make_mw(monkeypatch, allowed=False)
        record_calls: list = []
        # 即使 blocked 也设个 record lambda, 验证不被调用
        mw._limiter.record_usage = lambda *a, **k: record_calls.append((a, k))
        monkeypatch.setattr(mw, "_extract_usage", lambda r: (1, 1))

        req = _FakeRequest([HumanMessage("hi")])
        with pytest.raises(RateLimitExceeded) as exc_info:
            mw.wrap_model_call(req, lambda r: pytest.fail("handler 不应被调"))
        # reason 属性应标记 limit_exceeded
        assert exc_info.value.reason == "limit_exceeded"
        assert not record_calls, "blocked 时不应记账"

    def test_blocked_uses_default_thread_id(self, monkeypatch):
        """无 thread_id context → 用 'default' 兜底."""
        captured: dict = {}

        def fake_check(*a, **k):
            captured.update(k)
            return (False, "blocked")

        mw = RateLimitMiddleware.__new__(RateLimitMiddleware)
        mw._limiter = SimpleNamespace(
            check_allowed=fake_check, record_usage=lambda *a, **k: None
        )

        from huginn.security.rate_limiter import RateLimitExceeded
        req = _FakeRequest([HumanMessage("hi")])
        with pytest.raises(RateLimitExceeded):
            mw.wrap_model_call(req, lambda r: None)
        # 默认 thread_id
        assert captured.get("thread_id") == "default"

    def test_wrap_tool_call_passthrough(self):
        """RateLimitMiddleware 的 wrap_tool_call 不限流, 直接转发."""
        mw = RateLimitMiddleware.__new__(RateLimitMiddleware)
        mw._limiter = SimpleNamespace(
            check_allowed=lambda *a, **k: (True, ""),
            record_usage=lambda *a, **k: None,
        )
        sentinel = object()
        out = mw.wrap_tool_call(_FakeRequest([]), lambda r: sentinel)
        assert out is sentinel


# ───────────────────────────────────────────────────────────────────
# DeliverableCoverageMiddleware — 单位匹配 / 数值模式
# ───────────────────────────────────────────────────────────────────


class TestNumericPatternUnits:
    """_NUMERIC_PATTERN: 行里有真实数值才算 '真的分析了'.

    重点: future work / caveat 段落无数值 → missing. 单位匹配是判定的关键一环.
    """

    @staticmethod
    def _mw() -> DeliverableCoverageMiddleware:
        return DeliverableCoverageMiddleware()

    def test_decimal_number_matches(self):
        assert self._mw()._NUMERIC_PATTERN.search("μ = 5.2e-20 eV")

    def test_scientific_notation_matches(self):
        assert self._mw()._NUMERIC_PATTERN.search("g < 1.3e-17 GeV⁻¹")

    def test_multiplication_with_ten_power_matches(self):
        assert self._mw()._NUMERIC_PATTERN.search("λ ≲ 4.6 × 10⁻⁶")

    def test_comparison_operator_matches(self):
        # 模式 \d+\s*(?:<|≤|>|≥|±) 要求数字在比较符前面
        assert self._mw()._NUMERIC_PATTERN.search("mass 5 < 10 upper bound")
        assert self._mw()._NUMERIC_PATTERN.search("value 5 > 3 measured")

    def test_pm_operator_matches(self):
        assert self._mw()._NUMERIC_PATTERN.search("value = 100 ± 0.02")

    def test_number_with_unit_ev_matches(self):
        # 单位匹配大小写敏感 (pattern 全小写), "eV" 不匹配 "ev" — 用小写
        assert self._mw()._NUMERIC_PATTERN.search("mass 5.2e-20 ev at 95% CL")

    def test_number_with_unit_gev_matches(self):
        # 科学计数法优先匹配 (1.3e-17)
        assert self._mw()._NUMERIC_PATTERN.search("g < 1.3e-17 gev⁻¹")

    def test_number_with_unit_kg_matches(self):
        assert self._mw()._NUMERIC_PATTERN.search("m = 75 kg")

    def test_number_with_unit_hz_matches(self):
        # 大小写敏感, 用小写 hz
        assert self._mw()._NUMERIC_PATTERN.search("freq = 60 hz")

    def test_number_with_unit_year_matches(self):
        # 大小写敏感, pattern 是 myr/gyr/yr 全小写
        assert self._mw()._NUMERIC_PATTERN.search("3 myr lifetime")
        assert self._mw()._NUMERIC_PATTERN.search("lifetime 100 gyr")

    def test_plain_text_no_number_does_not_match(self):
        """定性描述 / future work 无数值 → 不匹配."""
        assert not self._mw()._NUMERIC_PATTERN.search(
            "we leave the coupling analysis for future work"
        )

    def test_list_numbering_does_not_match(self):
        """列表编号 '1. ' 不应匹配 (排除 false positive)."""
        assert not self._mw()._NUMERIC_PATTERN.search("1. first item")
        # "1." 单独结尾也不应匹配
        assert not self._mw()._NUMERIC_PATTERN.search("see section 1.")

    def test_single_digit_does_not_match(self):
        """单位数 (1 位整数) 不应匹配 (排除 figure 1 这类)."""
        # 注意: pattern 要求 2 位+ 整数, 单位数不匹配
        assert not self._mw()._NUMERIC_PATTERN.search("Figure 1 shows")

    def test_two_digit_integer_matches(self):
        """2 位+ 整数匹配 (排除列表编号后)."""
        assert self._mw()._NUMERIC_PATTERN.search("ROC-AUC = 48")

    def test_claim_keywords_unit_indicators(self):
        """_CLAIM_KEYWORDS 应识别 GeV⁻¹ / eV / dimensionless 等单位声明."""
        kw = self._mw()._CLAIM_KEYWORDS
        assert kw.search("g < 1.3 GeV⁻¹")
        assert kw.search("energy in eV")
        assert kw.search("dimensionless coupling")
        assert kw.search("upper limit on the mass")
        assert kw.search("point estimate of g")
        assert kw.search("95% CL")
        assert kw.search("1 sigma uncertainty")
        assert kw.search("constrained to be small")
        assert kw.search("bounded by data")
        assert kw.search("excluded at 95% CL")
        assert kw.search("consistent with the data")


# ───────────────────────────────────────────────────────────────────
# DeliverableCoverageMiddleware._extract_keywords
# ───────────────────────────────────────────────────────────────────


class TestExtractKeywords:
    """_extract_keywords: 从短语提取所有非停用词作为关键词 + 整个短语."""

    @staticmethod
    def _mw() -> DeliverableCoverageMiddleware:
        return DeliverableCoverageMiddleware()

    def test_simple_phrase(self):
        kws = self._mw()._extract_keywords("ulb masses")
        # 拆词 + 整个短语
        assert "ulb" in kws
        assert "masses" in kws
        assert "ulb masses" in kws

    def test_stopwords_filtered(self):
        kws = self._mw()._extract_keywords("the upper limits on mass")
        # the/on 被过滤, upper/limits/mass 保留 (limits 因 > 2 字符)
        assert "upper" in kws
        assert "limits" in kws
        assert "mass" in kws
        assert "the" not in kws
        assert "on" not in kws

    def test_short_words_filtered(self):
        """≤2 字符的词被过滤 (避免匹配到无关短词)."""
        kws = self._mw()._extract_keywords("d wave anisotropy")
        # d 太短被过滤, wave/anisotropy 保留
        assert "wave" in kws
        assert "anisotropy" in kws
        assert "d" not in kws

    def test_hyphen_split(self):
        """连字符拆分: 'self-interaction' → self + interaction + 整个短语."""
        kws = self._mw()._extract_keywords("self-interaction coupling")
        assert "self" in kws
        assert "interaction" in kws
        assert "coupling" in kws
        assert "self-interaction coupling" in kws

    def test_slash_split(self):
        """斜杠拆分: 'metal/insulator' → metal + insulator + 整个短语."""
        kws = self._mw()._extract_keywords("metal/insulator")
        assert "metal" in kws
        assert "insulator" in kws
        assert "metal/insulator" in kws

    def test_empty_phrase_returns_self(self):
        """全是停用词 → 返回 [phrase] 兜底 (避免空 keyword 漏匹配)."""
        kws = self._mw()._extract_keywords("the of and")
        assert kws == ["the of and"]

    def test_compound_d_g_i_wave(self):
        """d/g/i-wave anisotropy 拆分: d/g/i 太短过滤, wave 保留."""
        kws = self._mw()._extract_keywords("d/g/i-wave anisotropy")
        assert "wave" in kws
        assert "anisotropy" in kws
        # d/g/i 各 1 字符 → 过滤
        assert "d" not in kws
        assert "g" not in kws
        assert "i" not in kws


# ───────────────────────────────────────────────────────────────────
# DeliverableCoverageMiddleware._extract_quantities
# ───────────────────────────────────────────────────────────────────


class TestExtractQuantities:
    """_extract_quantities: 从 INSTRUCTIONS text 提取 'X and Y' 物理量对."""

    @staticmethod
    def _mw() -> DeliverableCoverageMiddleware:
        return DeliverableCoverageMiddleware()

    def test_derive_x_and_y(self):
        inst = "derive upper limits on ulb masses and self-interaction coupling strengths."
        qs = self._mw()._extract_quantities(inst)
        # 应提取 mass 和 coupling 相关
        joined = " ".join(qs)
        assert "mass" in joined or "masses" in joined
        assert "coupling" in joined or "self-interaction" in joined

    def test_classify_x_and_y_pattern(self):
        """第二条 pattern: classifications such as X and Y."""
        inst = (
            "with classifications such as metal/insulator and d/g/i-wave anisotropy."
        )
        qs = self._mw()._extract_quantities(inst)
        joined = " ".join(qs)
        assert "metal" in joined or "insulator" in joined
        assert "anisotropy" in joined or "wave" in joined

    def test_meta_terms_filtered(self):
        """related work / data / deliverables 等 meta 词应被过滤."""
        inst = "study the related work and data analysis."
        qs = self._mw()._extract_quantities(inst)
        # 不应提取 "related work" / "data analysis"
        for q in qs:
            assert "related work" not in q
            assert "data" not in q

    def test_no_quantities_returns_empty(self):
        """无 'X and Y' 模式 → 返回 []."""
        inst = "Just a plain sentence without any quantity pair."
        qs = self._mw()._extract_quantities(inst)
        assert qs == []

    def test_multiple_quantity_pairs(self):
        """一段 instructions 可能有多对 'X and Y'."""
        inst = (
            "derive the mass and coupling. "
            "calculate the energy and momentum."
        )
        qs = self._mw()._extract_quantities(inst)
        # 应至少 2 对
        assert len(qs) >= 2

    def test_quantity_length_constraints(self):
        """提取的 quantity 长度应在 [3, 80] 区间."""
        inst = "derive a and bb and cccccc and dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd."
        qs = self._mw()._extract_quantities(inst)
        for q in qs:
            assert 3 <= len(q) <= 80


# ───────────────────────────────────────────────────────────────────
# DeliverableCoverageMiddleware._check_coverage (横向覆盖)
# ───────────────────────────────────────────────────────────────────


class TestCheckCoverage:
    """_check_coverage: 行级 ±1 窗口数值检测, 返回 report.md 缺失的物理量列表."""

    @staticmethod
    def _mw() -> DeliverableCoverageMiddleware:
        return DeliverableCoverageMiddleware()

    def _inst(self) -> str:
        # Astronomy_000 风格 INSTRUCTIONS
        return (
            "To constrain the properties of ultralight bosons by developing "
            "a novel Bayesian statistical framework. The goal is to derive "
            "statistically rigorous upper limits on ULB masses and "
            "self-interaction coupling strengths, thereby using astrophysical "
            "data to probe fundamental particle physics."
        )

    def test_completely_missing_quantity_reported(self):
        """report 完全没提 coupling → 报 missing."""
        report = (
            "# Report\n## Results\n"
            "We compute ULB mass μ < 5.2e-20 eV at 95% CL.\n"
        )
        missing = self._mw()._check_coverage(self._inst(), report)
        assert any("coupling" in m or "self-interaction" in m for m in missing), \
            f"coupling 完全缺失应报 missing: {missing}"

    def test_future_work_paragraph_without_number_is_missing(self):
        """keyword 出现在 future work 段但 ±1 窗口无数值 → missing."""
        report = (
            "# Report\n## Results\n"
            "ULB mass μ < 5.2e-20 eV at 95% CL.\n\n"
            "## Future Work\n"
            "Including self-interaction coupling would allow further constraints.\n"
        )
        missing = self._mw()._check_coverage(self._inst(), report)
        # coupling 在 future work, ±1 窗口没数值 → missing
        assert any("coupling" in m or "self-interaction" in m for m in missing), \
            f"future work 无数值应报 missing: {missing}"

    def test_keyword_with_numeric_is_covered(self):
        """keyword 出现且 ±1 行窗口内有数值 → covered (不报 missing)."""
        report = (
            "# Report\n## Results\n"
            "ULB mass μ < 5.2e-20 eV at 95% CL.\n"
            "Self-interaction coupling g < 1.3e-17 GeV⁻¹.\n"
        )
        missing = self._mw()._check_coverage(self._inst(), report)
        assert missing == [], f"全部 covered 不应报: {missing}"

    def test_no_required_quantities_returns_empty(self):
        """INSTRUCTIONS 提取不出 quantity → 直接返回 [] (不查 report)."""
        mw = self._mw()
        # 没有 'X and Y' 模式
        missing = mw._check_coverage("plain text no quantities here.", "any report")
        assert missing == []

    def test_pm_one_line_window(self):
        """数值在 keyword 下一行 (±1 窗口) → 算 covered."""
        report = (
            "## Results\n"
            "Self-interaction coupling strength:\n"   # keyword 行, 无数值
            "g < 1.3e-17 GeV⁻¹ at 95% CL.\n"          # 下一行有数值
        )
        missing = self._mw()._check_coverage(self._inst(), report)
        # coupling 行无数值但下一行有 → ±1 窗口内 → covered
        coupling_missing = [m for m in missing if "coupling" in m or "self-interaction" in m]
        assert coupling_missing == [], \
            f"±1 行窗口有数值应 covered: {coupling_missing}"

    def test_material_000_classification_missing(self):
        """Material_000: report 缺 metal/insulator 分类 → 报 missing."""
        inst = (
            "50 newly discovered altermagnets with classifications such "
            "as metal/insulator and d/g/i-wave anisotropy."
        )
        report = (
            "# Report\n## Results\n"
            "ROC-AUC = 0.486 on candidate set.\n"
        )
        missing = self._mw()._check_coverage(inst, report)
        assert len(missing) >= 1, f"缺分类应报 missing: {missing}"

    def test_material_000_classification_with_numbers_covered(self):
        """Material_000: report 有分类数值 (32 metals 18 insulators) → covered."""
        inst = (
            "50 newly discovered altermagnets with classifications such "
            "as metal/insulator and d/g/i-wave anisotropy."
        )
        report = (
            "# Report\n## Classification Results\n"
            "Of 50 discovered altermagnets, 32 are metals and 18 are insulators.\n"
            "Anisotropy analysis: 12 d-wave, 8 g-wave, 10 i-wave patterns.\n"
        )
        missing = self._mw()._check_coverage(inst, report)
        assert missing == [], f"全 covered 不应报: {missing}"


# ───────────────────────────────────────────────────────────────────
# DeliverableCoverageMiddleware._check_layer_coverage (纵向分层)
# ───────────────────────────────────────────────────────────────────


class TestCheckLayerCoverage:
    """_check_layer_coverage: 5 层 (concept/equation/method/data/claim) 缺失检测."""

    @staticmethod
    def _mw() -> DeliverableCoverageMiddleware:
        return DeliverableCoverageMiddleware()

    def test_concept_missing_short_circuits(self):
        """concept 缺 (keyword 完全没出现) → 只返回 ['concept'] (短路)."""
        report = "# Report\n## Results\nNothing relevant here.\n"
        layers = self._mw()._check_layer_coverage("self-interaction coupling", report)
        assert layers == ["concept"]

    def test_concept_present_other_layers_missing(self):
        """concept 在但其它层缺 → 返回多个层名 (不含 concept)."""
        report = (
            "# Report\n"
            "We mention self-interaction coupling briefly.\n"
            "## Results\n"
            "Some unrelated data here.\n"
        )
        layers = self._mw()._check_layer_coverage("self-interaction coupling", report)
        # concept 在 (keyword 出现) → 不在 layers 里
        assert "concept" not in layers
        # 缺 method/data/claim (没 ## Methodology / ## Discussion)
        assert "method" in layers
        assert "data" in layers
        assert "claim" in layers

    def test_full_coverage_returns_empty(self):
        """五层齐全 → []."""
        report = (
            "# Report\n\n"
            "## Methodology\n"
            "We apply Bayesian marginalization to constrain self-interaction coupling. "
            "The self-interaction coupling is parameterized as $\\lambda = g^2/(2m^2)$.\n\n"
            "## Results\n"
            "Self-interaction coupling g < 1.3e-17 GeV⁻¹ at 95% CL.\n\n"
            "## Discussion\n"
            "We report upper limits on the self-interaction coupling strength g "
            "in GeV⁻¹ at 95% confidence level.\n"
        )
        layers = self._mw()._check_layer_coverage("self-interaction coupling", report)
        assert layers == [], f"五层齐全不应报: {layers}"

    def test_claim_layer_missing_without_unit_keyword(self):
        """claim 层缺 GeV⁻¹/upper limit 等声明关键词 → 报 claim 缺."""
        report = (
            "# Report\n"
            "## Methodology\n"
            "self-interaction coupling derivation here.\n\n"
            "## Results\n"
            "self-interaction coupling value 1.3e-17 measured.\n\n"
            "## Discussion\n"
            "We discuss the self-interaction coupling qualitatively.\n"
        )
        layers = self._mw()._check_layer_coverage("self-interaction coupling", report)
        # Discussion 段有 coupling 关键词, 但没 upper limit/GeV⁻¹ 等 → claim 缺
        assert "claim" in layers

    def test_equation_layer_missing_when_no_formula(self):
        """report 没公式 → equation 层缺."""
        report = (
            "# Report\n"
            "## Methodology\n"
            "self-interaction coupling derived numerically.\n\n"
            "## Results\n"
            "self-interaction coupling g < 1.3e-17.\n\n"
            "## Discussion\n"
            "upper limit on self-interaction coupling at 95% CL in GeV⁻¹.\n"
        )
        layers = self._mw()._check_layer_coverage("self-interaction coupling", report)
        assert "equation" in layers

    def test_data_layer_missing_when_no_numeric_in_results(self):
        """## Results 段有 keyword 但无数值 → data 层缺."""
        report = (
            "# Report\n"
            "## Methodology\n"
            "self-interaction coupling method.\n\n"
            "## Results\n"
            "We analyze the self-interaction coupling qualitatively, no numbers yet.\n\n"
            "## Discussion\n"
            "self-interaction coupling upper limit at 95% CL GeV⁻¹.\n"
        )
        layers = self._mw()._check_layer_coverage("self-interaction coupling", report)
        assert "data" in layers

    def test_method_layer_missing_when_no_methodology_section(self):
        """report 没 ## Methodology 段 → method 层缺."""
        report = (
            "# Report\n"
            "We mention self-interaction coupling.\n"
            "## Results\n"
            "self-interaction coupling g < 1.3e-17.\n\n"
            "## Discussion\n"
            "upper limit on self-interaction coupling at 95% CL GeV⁻¹.\n"
        )
        layers = self._mw()._check_layer_coverage("self-interaction coupling", report)
        assert "method" in layers


# ───────────────────────────────────────────────────────────────────
# DeliverableCoverageMiddleware._check_layer_gaps
# ───────────────────────────────────────────────────────────────────


class TestCheckLayerGaps:
    """_check_layer_gaps: 横向 covered 的 quantity 才查纵向 (避免重复提醒)."""

    @staticmethod
    def _mw() -> DeliverableCoverageMiddleware:
        return DeliverableCoverageMiddleware()

    def _inst(self) -> str:
        return (
            "derive upper limits on ULB masses and self-interaction "
            "coupling strengths, thereby probing particle physics."
        )

    def test_horizontal_missing_quantity_excluded_from_gaps(self):
        """横向 missing 的 quantity 不进 gaps (A 已经会报, v13 不重复)."""
        # report 完全没 coupling → 横向 missing
        report = (
            "# Report\n## Results\n"
            "ULB mass μ < 5.2e-20 eV at 95% CL.\n"
        )
        gaps = self._mw()._check_layer_gaps(self._inst(), report)
        # coupling 横向 missing → 不应出现在 gaps 里
        coupling_in_gaps = [q for q, _ in gaps if "coupling" in q or "self-interaction" in q]
        assert coupling_in_gaps == [], \
            f"横向 missing 的 quantity 不应在 gaps: {coupling_in_gaps}"

    def test_horizontal_covered_but_layer_missing_in_gaps(self):
        """横向 covered 但纵向缺层 → 进 gaps."""
        # coupling 横向 covered (有 keyword + 数值) 但没 ## Methodology / ## Discussion
        report = (
            "# Report\n## Results\n"
            "ULB mass μ < 5.2e-20 eV.\n"
            "Self-interaction coupling g < 1.3e-17 GeV⁻¹.\n"
        )
        gaps = self._mw()._check_layer_gaps(self._inst(), report)
        coupling_gaps = [(q, l) for q, l in gaps
                         if "coupling" in q or "self-interaction" in q]
        assert coupling_gaps, f"横向 covered 纵向缺层应进 gaps: {gaps}"
        # 缺 method/claim (没那些 section)
        _, layers = coupling_gaps[0]
        assert "method" in layers or "claim" in layers

    def test_no_required_returns_empty(self):
        gaps = self._mw()._check_layer_gaps("no quantities here", "any report")
        assert gaps == []

    def test_full_coverage_no_gaps(self):
        report = (
            "# Report\n\n"
            "## Methodology\n"
            "We constrain ULB mass and self-interaction coupling via Bayesian marginalization. "
            "The self-interaction coupling is parameterized as $\\lambda = g^2/(2m^2)$.\n\n"
            "## Results\n"
            "ULB mass μ < 5.2e-20 eV at 95% CL.\n"
            "Self-interaction coupling g < 1.3e-17 GeV⁻¹.\n\n"
            "## Discussion\n"
            "We report upper limits on ULB mass and self-interaction coupling "
            "strength g in GeV⁻¹ at 95% confidence level.\n"
        )
        gaps = self._mw()._check_layer_gaps(self._inst(), report)
        assert gaps == [], f"全 covered 不应报 gaps: {gaps}"


# ───────────────────────────────────────────────────────────────────
# DeliverableCoverageMiddleware 消息构建器
# ───────────────────────────────────────────────────────────────────


class TestMessageBuilders:
    """_build_frontier_msg / _build_layer_frontier_msg / _build_planning_msg."""

    @staticmethod
    def _mw() -> DeliverableCoverageMiddleware:
        return DeliverableCoverageMiddleware()

    def test_build_frontier_msg_basic(self):
        msg = self._mw()._build_frontier_msg(["self-interaction coupling strengths"])
        assert "FRONTIER TASK" in msg
        assert "DeliverableCoverageMiddleware" in msg
        assert "self-interaction coupling strengths" in msg
        assert "0 分" in msg
        assert "report.md" in msg

    def test_build_frontier_msg_multiple_quantities(self):
        msg = self._mw()._build_frontier_msg(["mass", "coupling", "energy"])
        for q in ("mass", "coupling", "energy"):
            assert q in msg

    def test_build_layer_frontier_msg_includes_required_wording(self):
        """v13 layer frontier msg 必须含反 future-work / 单位 / 换算公式 / 0 分."""
        msg = self._mw()._build_layer_frontier_msg([
            ("self-interaction coupling strengths", ["method", "data", "claim"]),
        ])
        assert "FRONTIER TASK" in msg
        assert "self-interaction coupling strengths" in msg
        assert "method" in msg
        assert "data" in msg
        assert "claim" in msg
        # 关键 wording (来自 _self_check 场景 11)
        assert "Future Work" in msg
        assert "GeV⁻¹" in msg
        assert "1/f_a" in msg
        assert "related_work" in msg
        assert "0 分" in msg
        assert "preliminary" in msg
        assert "exclusion probability curve" in msg
        assert "QCD axion" in msg

    def test_build_layer_frontier_msg_per_layer_action(self):
        """每层应有具体可操作提示 (layer_action dict)."""
        msg = self._mw()._build_layer_frontier_msg([
            ("mass", ["concept", "equation", "data", "method", "claim"]),
        ])
        assert "在 report 任意位置引入该物理概念" in msg  # concept
        assert "定义式或约束方程" in msg  # equation
        assert "## Results 段给出该量的具体数值" in msg  # data
        assert "## Methodology 段描述该量的推导" in msg  # method
        assert "## Discussion 或 ## Conclusion 段给出 upper limit" in msg  # claim

    def test_build_planning_msg_includes_quantities(self):
        msg = self._mw()._build_planning_msg(["ulb masses", "self-interaction coupling strengths"])
        assert "PLANNING HINT" in msg
        assert "ulb masses" in msg
        assert "self-interaction coupling strengths" in msg
        assert "Future Work" in msg
        assert "1/f_a" in msg
        assert "GeV⁻¹" in msg
        assert "preliminary" in msg
        assert "exclusion probability curve" in msg
        assert "QCD axion" in msg

    def test_build_planning_msg_empty_quantities(self):
        """空 quantities → 仍能渲染 (不崩)."""
        msg = self._mw()._build_planning_msg([])
        assert "PLANNING HINT" in msg
        # 没有具体量但模板框架应还在
        assert "规划要求" in msg


# ───────────────────────────────────────────────────────────────────
# DeliverableCoverageMiddleware._inject_frontier (集成, 用 tmp_path 隔离)
# ───────────────────────────────────────────────────────────────────


class TestInjectFrontier:
    """_inject_frontier: 读 INSTRUCTIONS.md + report.md, 注入 frontier/planning msg.

    用 tmp_path + monkeypatch.chdir 隔离 cwd, 不污染真实工作目录.
    """

    @staticmethod
    def _mw() -> DeliverableCoverageMiddleware:
        return DeliverableCoverageMiddleware()

    @staticmethod
    def _instructions_text() -> str:
        return (
            "# Task\n"
            "The goal is to derive upper limits on ULB masses and "
            "self-interaction coupling strengths, thereby probing particle physics.\n"
        )

    def test_no_instructions_does_nothing(self, tmp_path, monkeypatch):
        """INSTRUCTIONS.md 不存在 → 不修改 messages."""
        monkeypatch.chdir(tmp_path)
        mw = self._mw()
        msgs = [HumanMessage("hi")]
        req = _FakeRequest(msgs)
        mw._inject_frontier(req)
        assert req.messages == msgs

    def test_instructions_without_quantities_does_nothing(self, tmp_path, monkeypatch):
        """INSTRUCTIONS.md 存在但提取不出 quantity → 不注入."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "INSTRUCTIONS.md").write_text("plain text no X and Y.", encoding="utf-8")
        mw = self._mw()
        msgs = [HumanMessage("hi")]
        req = _FakeRequest(msgs)
        mw._inject_frontier(req)
        assert req.messages == msgs

    def test_no_report_injects_planning_hint(self, tmp_path, monkeypatch):
        """report.md 不存在 → 注入 planning hint (写 report 前给 agent 提示)."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "INSTRUCTIONS.md").write_text(
            self._instructions_text(), encoding="utf-8"
        )
        # report/ 目录不存在
        mw = self._mw()
        original = [HumanMessage("hi")]
        req = _FakeRequest(original)
        mw._inject_frontier(req)

        # 应在原 messages 前面插一条 SystemMessage (planning)
        assert len(req.messages) == 2
        assert isinstance(req.messages[0], SystemMessage)
        assert "PLANNING HINT" in req.messages[0].content
        # 原 messages 保留在后面
        assert req.messages[1] is original[0]

    def test_report_with_missing_injects_frontier_task(self, tmp_path, monkeypatch):
        """report.md 存在但缺量 → 注入 frontier task (横向 + 纵向)."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "INSTRUCTIONS.md").write_text(
            self._instructions_text(), encoding="utf-8",
        )
        # report 只覆盖 mass, 缺 coupling
        report_dir = tmp_path / "report"
        report_dir.mkdir()
        (report_dir / "report.md").write_text(
            "# Report\n## Results\n"
            "ULB mass μ < 5.2e-20 eV at 95% CL.\n",
            encoding="utf-8",
        )

        mw = self._mw()
        req = _FakeRequest([HumanMessage("hi")])
        mw._inject_frontier(req)

        # 应注入 SystemMessage, 内容是 frontier task (因为 report 已存在)
        assert len(req.messages) == 2
        assert isinstance(req.messages[0], SystemMessage)
        content = req.messages[0].content
        # 横向 missing 的注入 (coupling 缺)
        assert "FRONTIER TASK" in content or "PLANNING HINT" in content

    def test_full_coverage_no_injection(self, tmp_path, monkeypatch):
        """report.md 全覆盖 → 不注入 (没 missing 也没 gaps)."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "INSTRUCTIONS.md").write_text(
            self._instructions_text(), encoding="utf-8"
        )
        report_dir = tmp_path / "report"
        report_dir.mkdir()
        (report_dir / "report.md").write_text(
            "# Report\n\n"
            "## Methodology\n"
            "We constrain ULB mass and self-interaction coupling via Bayesian marginalization. "
            "self-interaction coupling: $\\lambda = g^2/(2m^2)$.\n\n"
            "## Results\n"
            "ULB mass μ < 5.2e-20 eV at 95% CL.\n"
            "Self-interaction coupling g < 1.3e-17 GeV⁻¹.\n\n"
            "## Discussion\n"
            "upper limit on ULB mass and self-interaction coupling strength g "
            "in GeV⁻¹ at 95% confidence level.\n",
            encoding="utf-8",
        )

        mw = self._mw()
        original = [HumanMessage("hi")]
        req = _FakeRequest(original)
        mw._inject_frontier(req)
        # 全 covered → 不应改 messages
        assert req.messages == original

    def test_exception_silently_skipped(self, tmp_path, monkeypatch):
        """_inject_frontier 内部异常不应传播 (try/except 兜底)."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "INSTRUCTIONS.md").write_text(
            self._instructions_text(), encoding="utf-8"
        )
        mw = self._mw()
        original = [HumanMessage("hi")]
        req = _FakeRequest(original)

        # 让 _extract_quantities 抛异常
        monkeypatch.setattr(
            mw, "_extract_quantities", lambda t: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        # 不应抛
        mw._inject_frontier(req)
        # messages 不变
        assert req.messages == original

    def test_messages_none_does_nothing(self, tmp_path, monkeypatch):
        """request.messages is None → 不修改 (defensive)."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "INSTRUCTIONS.md").write_text(
            self._instructions_text(), encoding="utf-8"
        )
        mw = self._mw()
        req = SimpleNamespace(messages=None)
        # 不应抛
        mw._inject_frontier(req)
        assert req.messages is None


# ───────────────────────────────────────────────────────────────────
# DeliverableCoverageMiddleware.wrap_model_call (集成层)
# ───────────────────────────────────────────────────────────────────


class TestCoverageWrapModelCall:
    """wrap_model_call: 调 _inject_frontier 后转发给 handler."""

    def test_wrap_model_call_delegates_to_handler(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # 无 INSTRUCTIONS → _inject_frontier no-op
        mw = DeliverableCoverageMiddleware()
        sentinel = object()
        req = _FakeRequest([HumanMessage("hi")])
        out = mw.wrap_model_call(req, lambda r: sentinel)
        assert out is sentinel

    def test_wrap_tool_call_passthrough(self):
        mw = DeliverableCoverageMiddleware()
        sentinel = object()
        out = mw.wrap_tool_call(_FakeRequest([]), lambda r: sentinel)
        assert out is sentinel

    def test_awrap_model_call_delegates(self):
        import asyncio

        mw = DeliverableCoverageMiddleware()
        req = _FakeRequest([HumanMessage("hi")])

        async def handler(r):
            return "async-ok"

        loop = asyncio.new_event_loop()
        try:
            out = loop.run_until_complete(mw.awrap_model_call(req, handler))
        finally:
            loop.close()
        assert out == "async-ok"


if __name__ == "__main__":
    # 不走 pytest 也能跑: 简单 smoke 各纯函数测试
    import sys

    print("[smoke] _parse_data_url:",
          FixDanglingToolCallsMiddleware._parse_data_url("data:image/png;base64,abc"))
    print("[smoke] _omitted_block:",
          FixDanglingToolCallsMiddleware._omitted_block({"type": "image_url"}))
    mw = DeliverableCoverageMiddleware()
    print("[smoke] _extract_keywords('ulb masses'):",
          mw._extract_keywords("ulb masses"))
    sys.exit(0)
