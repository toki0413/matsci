"""``huginn.agent.streaming`` 的直接单元测试.

覆盖范围:
- 模块级纯函数: ``_strip_dangling_tool_calls`` / ``_load_root_markers`` /
  ``_thinking_scale_timeout`` / ``_thinking_stream_idle`` / ``_is_root_message``
- 异步 watchdog: ``_astream_with_watchdog`` (超时 / 透传 / 空流)
- ``StreamingMixin`` 纯逻辑方法: ``_extract_last_ai_content`` / ``_check_phase_transition``
- 消息压缩: ``compact_messages`` (drop-oldest / root 保护 / thinking 块保护 /
  tool_result_ttl 清理 / tool_call 原子性) — streaming 主循环依赖的核心压缩逻辑
- root 标记重构 (Task B): ``_trim_checkpointer_messages`` 按 metadata 标记 root,
  无标记时回退到按位置
- red_team 数据落盘: ``_dump_completion_records`` (jsonl 落盘) +
  ``_process_stream_state`` 抓 record (落盘的数据源)

所有 LLM / graph 调用均用 mock 替代, 不发真实请求.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from huginn.agent.streaming import (
    _STREAM_IDLE_TIMEOUT,
    StreamingMixin,
    _astream_with_watchdog,
    _compute_common_prefix,
    _dump_completion_records,
    _is_root_message,
    _load_root_markers,
    _reconstruct_completion_records,
    _strip_dangling_tool_calls,
    _thinking_scale_timeout,
    _thinking_stream_idle,
)
from huginn.utils.context import (
    _msg_role,
    compact_messages,
)

# ── 辅助: 构造带 id 的大消息, 用于触发 compaction ─────────────────────────


def _big(content: str = "word " * 300, *, mid: str | None = None) -> HumanMessage:
    """构造一条 token 较多的 HumanMessage, 可选 id."""
    return HumanMessage(content=content, id=mid)


def _big_ai(content: str = "answer " * 300, *, mid: str | None = None) -> AIMessage:
    return AIMessage(content=content, id=mid)


# ═══════════════════════════════════════════════════════════════════════════
# _strip_dangling_tool_calls — dangling tool_call 剥离 (C1 修复)
# ═══════════════════════════════════════════════════════════════════════════


class TestStripDanglingToolCalls:
    def test_empty_list_returns_zero(self):
        # 空列表: 不崩, 返回 0
        assert _strip_dangling_tool_calls([]) == 0

    def test_all_answered_no_strip(self):
        # 全应答: AIMessage 的 tool_call 有对应 ToolMessage, 不剥
        msgs = [
            AIMessage(
                content="ok",
                tool_calls=[{"id": "tc1", "name": "f", "args": {}}],
            ),
            ToolMessage(content="r1", tool_call_id="tc1"),
        ]
        n = _strip_dangling_tool_calls(msgs)
        assert n == 0
        assert len(msgs[0].tool_calls) == 1  # tool_calls 不变

    def test_all_dangling_degrades_to_pure_ai(self):
        # 全 dangling: 无对应 ToolMessage, 退化为纯 AIMessage (保留 content)
        msgs = [
            AIMessage(
                content="d",
                tool_calls=[{"id": "tc2", "name": "f", "args": {}}],
            ),
        ]
        n = _strip_dangling_tool_calls(msgs)
        assert n == 1
        assert msgs[0].tool_calls == []  # tool_calls 清空
        assert msgs[0].content == "d"  # content 保留

    def test_partial_dangling_keeps_answered(self):
        # 部分 dangling: 保留已应答的, 剥掉未应答的
        msgs = [
            AIMessage(
                content="p",
                tool_calls=[
                    {"id": "tc3", "name": "f", "args": {}},
                    {"id": "tc4", "name": "g", "args": {}},
                ],
            ),
            ToolMessage(content="r3", tool_call_id="tc3"),
        ]
        n = _strip_dangling_tool_calls(msgs)
        assert n == 1  # 只剥掉 tc4
        assert len(msgs[0].tool_calls) == 1
        assert msgs[0].tool_calls[0]["id"] == "tc3"  # 保留已应答的 tc3


# ═══════════════════════════════════════════════════════════════════════════
# _load_root_markers — env 读取 root 内容 marker
# ═══════════════════════════════════════════════════════════════════════════


class TestLoadRootMarkers:
    def test_default_markers_when_env_unset(self, monkeypatch):
        # env 未设 → 返回默认 marker 列表 (分号分隔解析)
        monkeypatch.delenv("HUGINN_ROOT_MARKERS", raising=False)
        markers = _load_root_markers()
        assert markers is not None
        assert "## Methodology Checklist" in markers
        assert "## Selected Execution Plan" in markers

    def test_env_override(self, monkeypatch):
        # env 覆盖: 自定义 marker
        monkeypatch.setenv("HUGINN_ROOT_MARKERS", "## My Checklist;## Winner")
        markers = _load_root_markers()
        assert markers == ["## My Checklist", "## Winner"]

    def test_empty_env_returns_none(self, monkeypatch):
        # env 为空串 → 返回 None (关闭 marker 保护)
        monkeypatch.setenv("HUGINN_ROOT_MARKERS", "   ")
        assert _load_root_markers() is None

    def test_env_with_blank_segments_filtered(self, monkeypatch):
        # 分号间空段被过滤
        monkeypatch.setenv("HUGINN_ROOT_MARKERS", "## A;;  ;## B")
        assert _load_root_markers() == ["## A", "## B"]


# ═══════════════════════════════════════════════════════════════════════════
# _thinking_scale_timeout / _thinking_stream_idle — thinking 强度档位超时
# ═══════════════════════════════════════════════════════════════════════════


class TestThinkingTimeouts:
    @pytest.mark.parametrize("tier", ["high", "deep", "max", "extreme"])
    def test_scale_timeout_high_tier(self, monkeypatch, tier):
        # 深度推理档 → ainvoke 超时 900s
        monkeypatch.setenv("HUGINN_THINKING", tier)
        assert _thinking_scale_timeout() == 900.0

    @pytest.mark.parametrize("tier", ["medium", "med", "normal"])
    def test_scale_timeout_medium_tier(self, monkeypatch, tier):
        monkeypatch.setenv("HUGINN_THINKING", tier)
        assert _thinking_scale_timeout() == 600.0

    @pytest.mark.parametrize("tier", ["", "low", "off", "anything"])
    def test_scale_timeout_default_tier(self, monkeypatch, tier):
        # 其它/未设 → 默认 300s
        monkeypatch.setenv("HUGINN_THINKING", tier)
        assert _thinking_scale_timeout() == 300.0

    @pytest.mark.parametrize("tier", ["high", "deep", "max", "extreme"])
    def test_stream_idle_high_tier(self, monkeypatch, tier):
        # 深度推理档 → 流式空闲超时 180s
        monkeypatch.setenv("HUGINN_THINKING", tier)
        assert _thinking_stream_idle() == 180.0

    @pytest.mark.parametrize("tier", ["medium", "med", "normal"])
    def test_stream_idle_medium_tier(self, monkeypatch, tier):
        monkeypatch.setenv("HUGINN_THINKING", tier)
        assert _thinking_stream_idle() == 120.0

    def test_stream_idle_default_returns_module_constant(self, monkeypatch):
        # 默认档 → 回退到模块常量 _STREAM_IDLE_TIMEOUT (不重读 env)
        monkeypatch.setenv("HUGINN_THINKING", "")
        assert _thinking_stream_idle() == _STREAM_IDLE_TIMEOUT


# ═══════════════════════════════════════════════════════════════════════════
# _astream_with_watchdog — 流式空闲超时 watchdog (A3)
# ═══════════════════════════════════════════════════════════════════════════


class TestAstreamWatchdog:
    async def test_normal_stream_passthrough(self):
        # chunks 及时到达 → 全部透传, 不超时
        async def _fast():
            for i in range(5):
                await asyncio.sleep(0.01)
                yield i

        out = []
        async for item in _astream_with_watchdog(_fast(), idle_timeout=2.0):
            out.append(item)
        assert out == [0, 1, 2, 3, 4]

    async def test_idle_timeout_raises(self):
        # chunk 间隔超过 idle_timeout → asyncio.TimeoutError
        async def _slow():
            yield "first"
            await asyncio.sleep(5.0)  # 远超 timeout
            yield "second"

        with pytest.raises(asyncio.TimeoutError):
            async for _ in _astream_with_watchdog(_slow(), idle_timeout=0.3):
                pass

    async def test_empty_stream_yields_nothing(self):
        # 空流: 立即结束, 不超时
        async def _empty():
            return
            yield  # 让函数成为 async generator (不会执行到这)

        out = []
        async for item in _astream_with_watchdog(_empty(), idle_timeout=1.0):
            out.append(item)
        assert out == []

    async def test_first_chunk_yielded_before_timeout(self):
        # 首 chunk 透传, 第二个超时 — 验证已 yield 的不丢
        async def _one_then_slow():
            yield "fast"
            await asyncio.sleep(3.0)
            yield "slow"

        seen = []
        try:
            async for item in _astream_with_watchdog(
                _one_then_slow(), idle_timeout=0.3
            ):
                seen.append(item)
        except TimeoutError:
            pass
        assert seen == ["fast"]


# ═══════════════════════════════════════════════════════════════════════════
# _is_root_message — root metadata 标记判定 (Task B 核心)
# ═══════════════════════════════════════════════════════════════════════════


class TestIsRootMessage:
    def test_additional_kwargs_is_root(self):
        # additional_kwargs["is_root"]=True → root
        m = HumanMessage(content="checklist")
        m.additional_kwargs["is_root"] = True
        assert _is_root_message(m) is True

    def test_additional_kwargs_root_shorthand(self):
        # 简写 "root" 也识别
        m = HumanMessage(content="plan")
        m.additional_kwargs["root"] = True
        assert _is_root_message(m) is True

    def test_no_marker_returns_false(self):
        # 无标记 → False (回退到位置判断由调用方负责)
        assert _is_root_message(HumanMessage(content="plain")) is False

    def test_falsy_marker_returns_false(self):
        # 标记为 False/None → 不算 root
        m = HumanMessage(content="x")
        m.additional_kwargs["is_root"] = False
        assert _is_root_message(m) is False

    def test_metadata_dict_attribute(self):
        # 拥有 metadata dict 属性的对象 (如自定义消息/namespace) → 读 metadata
        obj = SimpleNamespace(metadata={"is_root": True}, additional_kwargs={})
        assert _is_root_message(obj) is True

    def test_non_dict_containers_skipped(self):
        # metadata/additional_kwargs 不是 dict (None / 缺失) → 安全跳过, 不崩
        obj = SimpleNamespace(metadata=None, additional_kwargs=None)
        assert _is_root_message(obj) is False

    def test_plain_dict_message(self):
        # dict 形态消息无 additional_kwargs 属性 → getattr 返回 None, 不算 root
        assert _is_root_message({"role": "user", "content": "x"}) is False


# ═══════════════════════════════════════════════════════════════════════════
# StreamingMixin._extract_last_ai_content — 从 graph state 取最近 AI 文本
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractLastAiContent:
    def test_string_content(self):
        # 字符串 content → 直接返回
        state = {"messages": [HumanMessage(content="q"), AIMessage(content="hello")]}
        assert StreamingMixin._extract_last_ai_content(state) == "hello"

    def test_list_content_with_text_blocks(self):
        # content 是 list[dict] (如 Anthropic 多块响应) → 拼接 text 字段
        ai = AIMessage(content=[
            {"type": "thinking", "thinking": "reason"},
            {"type": "text", "text": "the answer"},
        ])
        state = {"messages": [ai]}
        assert StreamingMixin._extract_last_ai_content(state) == "the answer"

    def test_no_ai_message_returns_empty(self):
        # 无 AIMessage → 空串
        state = {"messages": [HumanMessage(content="q")]}
        assert StreamingMixin._extract_last_ai_content(state) == ""

    def test_empty_messages_returns_empty(self):
        assert StreamingMixin._extract_last_ai_content({"messages": []}) == ""

    def test_picks_last_ai_when_multiple(self):
        # 多条 AIMessage → 取最后一条
        state = {
            "messages": [
                AIMessage(content="first"),
                HumanMessage(content="q2"),
                AIMessage(content="second"),
            ]
        }
        assert StreamingMixin._extract_last_ai_content(state) == "second"

    def test_missing_messages_key_returns_empty(self):
        assert StreamingMixin._extract_last_ai_content({}) == ""


# ═══════════════════════════════════════════════════════════════════════════
# StreamingMixin._check_phase_transition — 解析 [PHASE:xxx] marker
# ═══════════════════════════════════════════════════════════════════════════


class TestCheckPhaseTransition:
    def _mixin(self) -> StreamingMixin:
        # _check_phase_transition 不访问实例属性, 给个裸实例即可
        return StreamingMixin()

    def test_valid_marker_returns_phase(self):
        # 合法 phase 名 → 返回对应 ResearchPhase
        phase = self._mixin()._check_phase_transition("done.\n[PHASE: REPORTING]\n")
        assert phase is not None
        assert phase.value == "reporting"

    def test_marker_case_insensitive(self):
        # marker 大小写不敏感
        phase = self._mixin()._check_phase_transition("[phase: hypothesis]")
        assert phase is not None
        assert phase.value == "hypothesis"

    def test_no_marker_returns_none(self):
        assert self._mixin()._check_phase_transition("just a normal reply") is None

    def test_invalid_phase_name_returns_none(self):
        # 有 marker 但 phase 名不存在 → None (不抛 KeyError)
        assert self._mixin()._check_phase_transition("[PHASE: NONSENSE]") is None

    def test_all_valid_phases_parse(self):
        # 所有 ResearchPhase 值都能被解析
        for name in (
            "literature", "hypothesis", "planning",
            "execution", "validation", "reporting", "open",
        ):
            phase = self._mixin()._check_phase_transition(f"[PHASE: {name}]")
            assert phase is not None
            assert phase.value == name


# ═══════════════════════════════════════════════════════════════════════════
# compact_messages — 消息压缩逻辑 (streaming 主循环依赖)
# ═══════════════════════════════════════════════════════════════════════════


class TestCompactMessages:
    def test_zero_budget_passthrough(self):
        # budget<=0 → 不压缩, 原样返回
        msgs = [HumanMessage(content="a"), AIMessage(content="b")]
        assert compact_messages(msgs, budget_tokens=0) is msgs

    def test_under_budget_passthrough(self):
        # total <= budget → 原样返回 (顺序不变)
        msgs = [HumanMessage(content="hi"), AIMessage(content="hello")]
        out = compact_messages(msgs, budget_tokens=10**6, keep_last_n=1)
        assert out == msgs

    def test_drop_oldest_until_under_budget(self):
        # 超预算 → 从最旧开始丢, 保留 keep_last_n 条尾部
        msgs = [
            HumanMessage(content="old " * 200, id="m0"),
            AIMessage(content="older " * 200, id="m1"),
            HumanMessage(content="keep me", id="m2"),
        ]
        out = compact_messages(msgs, budget_tokens=50, keep_last_n=1, tool_result_ttl=0)
        # 最后一条必保留
        assert out[-1].content == "keep me"
        # 最旧的那条应被丢
        assert all(getattr(m, "id", None) != "m0" for m in out)

    def test_keep_root_n_position_based(self):
        # keep_root_n > 0 → 前 N 条 root 永不丢 (位置标记)
        root0 = HumanMessage(content="task " * 200, id="root0")
        root1 = SystemMessage(content="checklist " * 200, id="root1")
        body = [AIMessage(content="body " * 200, id=f"b{i}") for i in range(3)]
        tail = HumanMessage(content="tail", id="tail")
        msgs = [root0, root1, *body, tail]
        out = compact_messages(
            msgs, budget_tokens=50, keep_last_n=1, keep_root_n=2, tool_result_ttl=0
        )
        # 前 2 条 root 必须保留
        out_ids = {getattr(m, "id", None) for m in out}
        assert "root0" in out_ids
        assert "root1" in out_ids

    def test_root_content_markers_protect_mid_list(self):
        # root_content_markers: 按内容 marker 标 root, 中间消息含 marker 也保护
        # (这是 "前缀合并" 的等价物 — 关键早期消息跨压缩保留, 不靠位置)
        marker = "## Methodology Checklist"
        checklist = HumanMessage(content=f"{marker}\n- step1\n" + "x " * 200, id="chk")
        msgs = [
            HumanMessage(content="q " * 200, id="m0"),
            checklist,
            AIMessage(content="a " * 200, id="m1"),
            HumanMessage(content="latest", id="tail"),
        ]
        out = compact_messages(
            msgs, budget_tokens=50, keep_last_n=1,
            keep_root_n=0, root_content_markers=[marker], tool_result_ttl=0,
        )
        # 含 marker 的消息必须保留
        assert any(getattr(m, "id", None) == "chk" for m in out)

    def test_thinking_block_protected(self):
        # 含 thinking/redacted_thinking 块的 AIMessage 永不裁剪 (丢 signature 会 400)
        thinking_ai = AIMessage(content=[
            {"type": "thinking", "thinking": "reason", "signature": "sig"},
            {"type": "text", "text": "ans"},
        ], id="think")
        msgs = [
            HumanMessage(content="task " * 200, id="m0"),
            thinking_ai,
            AIMessage(content="plain " * 200, id="m1"),
            HumanMessage(content="q", id="tail"),
        ]
        out = compact_messages(msgs, budget_tokens=10, keep_last_n=1, tool_result_ttl=0)
        assert any(getattr(m, "id", None) == "think" for m in out)

    def test_tool_result_ttl_clears_old_tool_content(self):
        # tool_result_ttl: 超过 TTL 的旧 tool 消息 content 被替换为清除标记
        big_tool = ToolMessage(content="result " * 200, tool_call_id="c1", id="t0")
        msgs = [
            HumanMessage(content="task", id="h0"),
            AIMessage(content="call", tool_calls=[{"id": "c1", "name": "f", "args": {}}], id="a0"),
            big_tool,
            AIMessage(content="a2 " * 200, id="a1"),
            AIMessage(content="a3 " * 200, id="a2"),
            HumanMessage(content="latest", id="tail"),
        ]
        out = compact_messages(
            msgs, budget_tokens=10**6, keep_last_n=2, tool_result_ttl=1,
        )
        # 旧 tool 消息 content 应被清除
        cleared = [m for m in out if getattr(m, "tool_call_id", None) == "c1"]
        assert cleared, "tool 消息应仍在列表里 (顺序不断)"
        assert "[cleared: tool result over TTL]" in cleared[0].content

    def test_tool_call_atomicity_no_orphan_tool(self):
        # 保留区不能以孤儿 ToolMessage 开头 (它的 AIMessage 被丢 → API 400)
        msgs = [
            HumanMessage(content="task " * 300, id="h0"),
            AIMessage(
                content="call it " * 200,
                tool_calls=[{"id": "c1", "name": "f", "args": {}}],
                id="a0",
            ),
            ToolMessage(content="r " * 300, tool_call_id="c1", id="t0"),
            AIMessage(content="done", id="a1"),
        ]
        out = compact_messages(msgs, budget_tokens=50, keep_last_n=2, tool_result_ttl=0)
        roles = [_msg_role(m) for m in out]
        # 保留区第一条不能是 tool (孤儿)
        if roles:
            assert roles[0] != "tool", f"orphan ToolMessage survived: {roles}"
        # 最后的 done 必须在
        assert out[-1].content == "done"


# ═══════════════════════════════════════════════════════════════════════════
# _trim_checkpointer_messages — root 标记重构 (Task B)
# ═══════════════════════════════════════════════════════════════════════════


def _make_mock_graph(messages: list) -> MagicMock:
    """构造 mock graph: get_state 返回含 messages 的 snapshot, update_state 记录调用."""
    snapshot = SimpleNamespace(values={"messages": messages})
    graph = MagicMock()
    graph.get_state = MagicMock(return_value=snapshot)
    graph.update_state = MagicMock()
    return graph


def _dropped_ids(graph: MagicMock) -> list[str]:
    """从 graph.update_state 调用里取出被 drop 的 message id 列表."""
    assert graph.update_state.called, "update_state 应被调用以执行删除"
    # update_state(config, {"messages": removals})
    payload = graph.update_state.call_args.args[1]
    removals = payload["messages"]
    return [r.id for r in removals]


class TestTrimCheckpointerRootMarking:
    async def test_budget_passthrough_no_drop(self):
        # total <= budget → 不删, update_state 不调用
        self_obj = SimpleNamespace(context_budget_tokens=10**9)
        msgs = [_big(mid="m0"), _big_ai(mid="m1"), _big(mid="m2"), _big_ai(mid="m3"), _big(mid="m4")]
        graph = _make_mock_graph(msgs)
        n = await StreamingMixin._trim_checkpointer_messages(self_obj, {}, graph, {})
        assert n == 0
        graph.update_state.assert_not_called()

    async def test_context_budget_le_zero_returns_zero(self):
        # context_budget_tokens <= 0 → 直接返回 0
        self_obj = SimpleNamespace(context_budget_tokens=0)
        graph = _make_mock_graph([_big(mid="m0")] * 5)
        n = await StreamingMixin._trim_checkpointer_messages(self_obj, {}, graph, {})
        assert n == 0

    async def test_fallback_position_when_no_metadata(self, monkeypatch):
        # 无 metadata 标记 → 回退到按位置 (前 keep_root_n 条 protected)
        # 与历史行为一致: 前 2 条 root 不丢, 从第 3 条开始丢
        monkeypatch.setenv("HUGINN_KEEP_ROOT_N", "2")
        msgs = [
            _big(mid="root0"),       # idx 0 — 位置 root
            _big_ai(mid="root1"),    # idx 1 — 位置 root
            _big(mid="body0"),       # idx 2 — 可丢
            _big_ai(mid="body1"),    # idx 3 — 可丢
            _big(mid="tail0"),       # idx 4 — keep_last_n 尾部
            _big_ai(mid="tail1"),    # idx 5
            _big(mid="tail2"),       # idx 6
            _big_ai(mid="tail3"),    # idx 7
        ]
        self_obj = SimpleNamespace(context_budget_tokens=50)
        graph = _make_mock_graph(msgs)
        n = await StreamingMixin._trim_checkpointer_messages(self_obj, {}, graph, {})
        dropped = _dropped_ids(graph)
        # 前 2 条 root 不应被丢
        assert "root0" not in dropped
        assert "root1" not in dropped
        # body 区 (idx 2,3) 应被丢 (budget 极小, 全丢)
        assert "body0" in dropped
        assert "body1" in dropped
        assert n == len(dropped)

    async def test_metadata_root_protects_mid_list_message(self, monkeypatch):
        # Task B 核心: metadata 标记的 root 在任意位置都受保护, 不靠位置
        monkeypatch.setenv("HUGINN_KEEP_ROOT_N", "0")  # 关掉位置回退, 只看 metadata
        # 把 idx 2 标记为 root (中间位置)
        marked = _big(mid="midroot")
        marked.additional_kwargs["is_root"] = True
        msgs = [
            _big(mid="m0"),       # idx 0 — 可丢
            _big_ai(mid="m1"),    # idx 1 — 可丢
            marked,               # idx 2 — metadata root, 受保护
            _big(mid="m2"),       # idx 3 — 可丢
            _big_ai(mid="tail0"),  # idx 4 — keep_last_n 尾部
            _big(mid="tail1"),    # idx 5
            _big_ai(mid="tail2"),  # idx 6
            _big(mid="tail3"),    # idx 7
        ]
        self_obj = SimpleNamespace(context_budget_tokens=50)
        graph = _make_mock_graph(msgs)
        n = await StreamingMixin._trim_checkpointer_messages(self_obj, {}, graph, {})
        dropped = _dropped_ids(graph)
        # 被标记的中间 root 绝不能被丢
        assert "midroot" not in dropped
        # 它前后的非 root 消息应被丢 (budget 极小)
        assert "m0" in dropped
        assert "m1" in dropped
        assert "m2" in dropped

    async def test_metadata_root_overrides_position(self, monkeypatch):
        # 有 metadata 标记时, 位置回退不生效: 即便 keep_root_n=2 也不保前 2 条
        monkeypatch.setenv("HUGINN_KEEP_ROOT_N", "2")
        # 只把 idx 4 标 root (尾部, 本来就被 keep_last_n 保护); 前 2 条不标
        marked = _big(mid="onlyroot")
        marked.additional_kwargs["is_root"] = True
        msgs = [
            _big(mid="m0"),       # idx 0 — 无标记, 有 metadata 标记时不再被位置保护
            _big_ai(mid="m1"),    # idx 1 — 无标记
            _big(mid="m2"),       # idx 2 — 可丢
            _big_ai(mid="m3"),    # idx 3 — 可丢
            marked,               # idx 4 — metadata root (也在 keep_last_n 尾部)
            _big(mid="tail1"),    # idx 5
            _big_ai(mid="tail2"),  # idx 6
            _big(mid="tail3"),    # idx 7
        ]
        self_obj = SimpleNamespace(context_budget_tokens=50)
        graph = _make_mock_graph(msgs)
        await StreamingMixin._trim_checkpointer_messages(self_obj, {}, graph, {})
        dropped = _dropped_ids(graph)
        # 前 2 条 (m0,m1) 无 metadata 标记 → 有标记时不走位置回退, m0/m1 可被丢
        # (m1 在 body 区 idx 1 < body_end=4, 可丢)
        assert "m0" in dropped

    async def test_system_message_not_removed(self, monkeypatch):
        # SystemMessage 即使在可丢区也不进 drop_ids (不该删 system)
        monkeypatch.setenv("HUGINN_KEEP_ROOT_N", "0")
        sys_msg = SystemMessage(content="sys " * 300, id="sys0")
        msgs = [
            _big(mid="m0"),
            sys_msg,              # idx 1 — 可丢区, 但是 SystemMessage → 不删
            _big_ai(mid="m2"),
            _big(mid="m3"),
            _big_ai(mid="tail0"),
            _big(mid="tail1"),
            _big_ai(mid="tail2"),
            _big(mid="tail3"),
        ]
        self_obj = SimpleNamespace(context_budget_tokens=50)
        graph = _make_mock_graph(msgs)
        await StreamingMixin._trim_checkpointer_messages(self_obj, {}, graph, {})
        dropped = _dropped_ids(graph)
        assert "sys0" not in dropped

    async def test_too_few_messages_returns_zero(self):
        # len(msgs) <= 4 → 不删
        self_obj = SimpleNamespace(context_budget_tokens=50)
        graph = _make_mock_graph([_big(mid=f"m{i}") for i in range(4)])
        n = await StreamingMixin._trim_checkpointer_messages(self_obj, {}, graph, {})
        assert n == 0
        graph.update_state.assert_not_called()

    async def test_no_droppable_returns_zero(self, monkeypatch):
        # 所有非尾部消息都是 root → 无可丢候选 → 返回 0
        monkeypatch.setenv("HUGINN_KEEP_ROOT_N", "0")
        r0 = _big(mid="r0")
        r0.additional_kwargs["is_root"] = True
        r1 = _big_ai(mid="r1")
        r1.additional_kwargs["is_root"] = True
        r2 = _big(mid="r2")
        r2.additional_kwargs["is_root"] = True
        r3 = _big_ai(mid="r3")
        r3.additional_kwargs["is_root"] = True
        msgs = [r0, r1, r2, r3, _big(mid="tail0"), _big_ai(mid="tail1"),
                _big(mid="tail2"), _big_ai(mid="tail3")]
        self_obj = SimpleNamespace(context_budget_tokens=50)
        graph = _make_mock_graph(msgs)
        n = await StreamingMixin._trim_checkpointer_messages(self_obj, {}, graph, {})
        assert n == 0
        graph.update_state.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════
# P0 / P1① — checkpointer 消息数上限兜底 + 无 usage 降级 (第三方审计修复)
# ═══════════════════════════════════════════════════════════════════════════


class TestCheckpointerCountCap:
    """P0: 无条件按消息数上限修剪 checkpointer, 与 token budget/usage 解耦."""

    def test_max_messages_default(self):
        self_obj = object.__new__(StreamingMixin)
        assert StreamingMixin._checkpointer_max_messages(self_obj) == 120

    def test_max_messages_env_override(self, monkeypatch):
        monkeypatch.setenv("HUGINN_CHECKPOINTER_MAX_MESSAGES", "80")
        self_obj = object.__new__(StreamingMixin)
        assert StreamingMixin._checkpointer_max_messages(self_obj) == 80

    def test_max_messages_disabled_when_zero(self, monkeypatch):
        monkeypatch.setenv("HUGINN_CHECKPOINTER_MAX_MESSAGES", "0")
        self_obj = object.__new__(StreamingMixin)
        assert StreamingMixin._checkpointer_max_messages(self_obj) is None

    async def test_trims_by_count_even_within_budget(self, monkeypatch):
        # 核心: token 总量压在预算内 (不超 G34 预算), 但条数超上限 → 仍按条数删.
        # 用足够小的 budget 让 "token 预算" 判据不触发, 只用消息数上限触发.
        monkeypatch.setenv("HUGINN_KEEP_ROOT_N", "0")
        # budget 极大 → token 判据必然 pass; 只有 max_messages 生效
        self_obj = SimpleNamespace(context_budget_tokens=10**12)
        # 8 条, keep_last_n=4 → body 区 4 条 (idx0..3), max_messages=6
        msgs = [
            _big(mid=f"m{i}") for i in range(4)   # body 可丢
        ] + [
            _big_ai(mid=f"tail{i}") for i in range(4)  # keep_last_n 尾部
        ]
        graph = _make_mock_graph(msgs)
        n = await StreamingMixin._trim_checkpointer_messages(
            self_obj, {}, graph, config={}, max_messages=6
        )
        # 8 → 收敛到 ≤6, body 丢 2 条
        assert n == 2
        dropped = _dropped_ids(graph)
        assert dropped == ["m0", "m1"]

    async def test_no_trim_when_within_count(self, monkeypatch):
        monkeypatch.setenv("HUGINN_KEEP_ROOT_N", "0")
        self_obj = SimpleNamespace(context_budget_tokens=10**12)
        msgs = [_big(mid=f"m{i}") for i in range(2)] + [
            _big_ai(mid=f"tail{i}") for i in range(4)
        ]
        graph = _make_mock_graph(msgs)
        n = await StreamingMixin._trim_checkpointer_messages(
            self_obj, {}, graph, config={}, max_messages=6
        )
        assert n == 0
        graph.update_state.assert_not_called()

    async def test_enforce_count_cap_respects_trim_strategy(self, monkeypatch):
        # 策略不含 trim → _enforce_checkpointer_count_cap 不动作
        monkeypatch.setenv("HUGINN_COMPACT_STRATEGY", "summarize")
        monkeypatch.setenv("HUGINN_KEEP_ROOT_N", "0")
        msgs = [
            _big(mid=f"m{i}") for i in range(2)
        ] + [
            _big_ai(mid=f"tail{i}") for i in range(4)
        ]
        graph = _make_mock_graph(msgs)
        self_obj = object.__new__(StreamingMixin)
        self_obj.checkpointer = object()
        self_obj.context_budget_tokens = 10**12
        turn_span = SimpleNamespace(metadata={})
        n = await StreamingMixin._enforce_checkpointer_count_cap(
            self_obj, graph, {}, turn_span
        )
        assert n == 0

    async def test_enforce_count_cap_skips_when_no_checkpointer(self):
        self_obj = SimpleNamespace(checkpointer=None)
        n = await StreamingMixin._enforce_checkpointer_count_cap(
            self_obj, None, None, SimpleNamespace(metadata={})
        )
        assert n == 0

    async def test_enforce_count_cap_trims(self, monkeypatch):
        monkeypatch.setenv("HUGINN_KEEP_ROOT_N", "0")
        monkeypatch.setenv("HUGINN_CHECKPOINTER_MAX_MESSAGES", "6")
        self_obj = object.__new__(StreamingMixin)
        self_obj.checkpointer = object()
        self_obj.context_budget_tokens = 10**12
        msgs = [
            _big(mid=f"m{i}") for i in range(4)   # body 可丢
        ] + [
            _big_ai(mid=f"tail{i}") for i in range(4)  # keep_last_n 尾部
        ]
        graph = _make_mock_graph(msgs)
        turn_span = SimpleNamespace(metadata={})
        n = await StreamingMixin._enforce_checkpointer_count_cap(
            self_obj, graph, {}, turn_span
        )
        assert n == 2
        assert turn_span.metadata["checkpointer_p0_trimmed"] == 2


class TestDumpCompletionRecords:
    def test_empty_records_returns_none(self, monkeypatch, tmp_path):
        # 空 records → 直接返回 None, 不写文件
        monkeypatch.setenv("HUGINN_CACHE_DIR", str(tmp_path))
        assert _dump_completion_records([], "thread1", "turn1") is None
        # 不应创建任何文件
        assert not (tmp_path / "completions").exists()

    def test_writes_jsonl_per_turn(self, monkeypatch, tmp_path):
        # 非空 records → 落盘为 <cache>/completions/<thread>/<turn>.jsonl
        monkeypatch.setenv("HUGINN_CACHE_DIR", str(tmp_path))
        records = [
            {"type": "assistant", "content": "hello", "tool_calls": None, "ts": 1.0},
            {"type": "tool", "content": "result", "tool_call_id": "c1", "name": "f", "ts": 2.0},
        ]
        path = _dump_completion_records(records, "threadA", "turnA")
        assert path is not None
        assert path == tmp_path / "completions" / "threadA" / "turnA.jsonl"
        assert path.exists()
        # 每行一个 JSON, 顺序与 records 一致
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["type"] == "assistant"
        assert json.loads(lines[1])["type"] == "tool"
        assert json.loads(lines[1])["tool_call_id"] == "c1"

    def test_unicode_content_preserved(self, monkeypatch, tmp_path):
        # ensure_ascii=False → 中文/特殊字符原样保留 (red_team 数据常含中文)
        monkeypatch.setenv("HUGINN_CACHE_DIR", str(tmp_path))
        records = [{"type": "assistant", "content": "你好世界 🌍", "ts": 1.0}]
        path = _dump_completion_records(records, "t", "turn")
        text = path.read_text(encoding="utf-8")
        assert "你好世界 🌍" in text

    def test_each_turn_separate_file_no_prefix_merging(self, monkeypatch, tmp_path):
        # 当前行为: 每 turn 独立文件, 不做跨 turn 前缀合并 (prefix_merging 是升级路径)
        monkeypatch.setenv("HUGINN_CACHE_DIR", str(tmp_path))
        recs = [{"type": "assistant", "content": "x", "ts": 1.0}]
        p1 = _dump_completion_records(recs, "t", "turn1")
        p2 = _dump_completion_records(recs, "t", "turn2")
        assert p1 != p2
        assert p1.exists() and p2.exists()
        # 两个文件各自独立, 没有合并成一个
        assert len(list((tmp_path / "completions" / "t").glob("*.jsonl"))) == 2

    def test_failure_returns_none_silently(self, monkeypatch, tmp_path):
        # 落盘失败 (路径不可写) → 静默返回 None, 不抛
        monkeypatch.setenv("HUGINN_CACHE_DIR", str(tmp_path))
        # 把 cache dir 指到一个文件 (不是目录) 让 mkdir 失败
        blocker = tmp_path / "blocker"
        blocker.write_text("x")
        monkeypatch.setenv("HUGINN_CACHE_DIR", str(blocker))
        result = _dump_completion_records(
            [{"type": "assistant", "content": "x", "ts": 1.0}], "t", "turn"
        )
        assert result is None


# _compute_common_prefix / _reconstruct_completion_records — P2 prefix_merging
# ═══════════════════════════════════════════════════════════════════════════


class TestPrefixMerging:
    def test_common_prefix_counts_type_and_content(self):
        a = [
            {"type": "assistant", "content": "a"},
            {"type": "tool", "content": "r"},
            {"type": "assistant", "content": "b"},
        ]
        b = [
            {"type": "assistant", "content": "a"},
            {"type": "tool", "content": "r"},
            {"type": "assistant", "content": "DIFF"},
        ]
        assert _compute_common_prefix(a, b) == 2

    def test_common_prefix_ignores_ts_noise(self):
        a = [{"type": "assistant", "content": "a", "ts": 1.0}]
        b = [{"type": "assistant", "content": "a", "ts": 9.9}]
        assert _compute_common_prefix(a, b) == 1

    def test_common_prefix_zero_when_no_overlap(self):
        assert _compute_common_prefix([{"type": "assistant", "content": "a"}],
                                      [{"type": "tool", "content": "r"}]) == 0

    def test_dump_dedups_prefix_and_reconstruct(self, monkeypatch, tmp_path):
        # 同一 thread 连续两个 turn, 后者共享前者前 2 条 → 写 prefix_ref + 增量
        monkeypatch.setenv("HUGINN_CACHE_DIR", str(tmp_path))
        turn1 = [
            {"type": "assistant", "content": "a", "ts": 1.0},
            {"type": "tool", "content": "r", "ts": 2.0},
            {"type": "assistant", "content": "b", "ts": 3.0},
        ]
        turn2 = [
            {"type": "assistant", "content": "a", "ts": 4.0},
            {"type": "tool", "content": "r", "ts": 5.0},
            {"type": "assistant", "content": "c", "ts": 6.0},
        ]
        p1 = _dump_completion_records(turn1, "t", "turn1")
        p2 = _dump_completion_records(turn2, "t", "turn2")
        assert p1 is not None and p2 is not None
        # turn2 首行是 prefix_ref
        lines = p2.read_text(encoding="utf-8").strip().split("\n")
        first = json.loads(lines[0])
        assert first["type"] == "prefix_ref"
        assert first["prev_file"] == "turn1.jsonl"
        assert first["prefix_len"] == 2
        # 增量只有 1 条 (后缀 c)
        assert len(lines) == 2
        assert json.loads(lines[1])["content"] == "c"
        # 读侧重建 = 完整序列 (type+content; 前缀 ts 复用 prev_file 原值, 属预期)
        rebuilt = _reconstruct_completion_records(p2)
        assert [r["type"] for r in rebuilt] == ["assistant", "tool", "assistant"]
        assert [r["content"] for r in rebuilt] == ["a", "r", "c"]

    def test_no_dedup_below_threshold(self, monkeypatch, tmp_path):
        # 公共前缀 < 2 → 不写 prefix_ref, 全量落盘
        monkeypatch.setenv("HUGINN_CACHE_DIR", str(tmp_path))
        recs = [{"type": "assistant", "content": "x", "ts": 1.0}]
        _dump_completion_records(recs, "t", "turn1")
        p2 = _dump_completion_records(recs, "t", "turn2")
        lines = p2.read_text(encoding="utf-8").strip().split("\n")
        assert json.loads(lines[0])["type"] != "prefix_ref"


# ═══════════════════════════════════════════════════════════════════════════
# _process_stream_state — red_team record 抓取 (落盘数据源)
# ═══════════════════════════════════════════════════════════════════════════


def _make_streaming_self() -> MagicMock:
    """构造最小 mock self, 满足 _process_stream_state 的属性访问.

    memory / _conversation_tree / _session_state 用 MagicMock 记录调用;
    _extract_cache_stats 返回 {} (falsy) 以跳过 cache_stats/pet 分支.
    """
    self = MagicMock()
    self._state_msg_offsets = {}
    self._thought_detector = None
    self._thought_loop_terminated = False
    self._break_after_tool = False
    self._break_flag = False
    self._last_cache_stats = None
    self._pending_tool_inputs = {}
    # 返回空 dict (falsy) → 跳过 cache_stats 块, 不触发 pet.publish / span.metadata
    self._extract_cache_stats = MagicMock(return_value={})
    return self


class TestProcessStreamStateRecords:
    def test_ai_message_appends_assistant_record(self):
        # AIMessage → 抓 assistant record (red_team 消费)
        self = _make_streaming_self()
        records: list[dict[str, Any]] = []
        state = {"messages": [AIMessage(content="hello world", id="ai1")]}
        StreamingMixin._process_stream_state(
            self, state, MagicMock(), "thread1", MagicMock(), records
        )
        assert len(records) == 1
        rec = records[0]
        assert rec["type"] == "assistant"
        assert rec["content"] == "hello world"
        # AIMessage.tool_calls 默认是空 list (无 tool_call 时), 这里只需断言为空
        assert not rec["tool_calls"]
        assert "ts" in rec
        # memory / conversation_tree 都被写入
        self.memory.add_message.assert_called_with("assistant", "hello world")

    def test_ai_with_tool_calls_records_tool_calls(self):
        # AIMessage 带 tool_calls → record 里 tool_calls 字段保留 + memory.add_tool_call
        self = _make_streaming_self()
        records: list[dict[str, Any]] = []
        ai = AIMessage(
            content="calling tool",
            tool_calls=[{"id": "tc1", "name": "bash_tool", "args": {"cmd": "ls"}}],
            id="ai2",
        )
        state = {"messages": [ai]}
        StreamingMixin._process_stream_state(
            self, state, MagicMock(), "thread1", MagicMock(), records
        )
        rec = records[0]
        assert rec["tool_calls"] is not None
        assert rec["tool_calls"][0]["name"] == "bash_tool"
        self.memory.add_tool_call.assert_called_once()
        # tool_input 缓存到 _pending_tool_inputs, 供后续 ToolMessage 用
        assert self._pending_tool_inputs.get("tc1") == {"cmd": "ls"}

    def test_tool_message_appends_tool_record(self):
        # ToolMessage → 抓 tool record + session_state.add_tool_result
        self = _make_streaming_self()
        # 先放一个 AIMessage 带 tool_call, 让 _pending_tool_inputs 有缓存
        self._pending_tool_inputs = {"tc9": {"cmd": "pwd"}}
        records: list[dict[str, Any]] = []
        tm = ToolMessage(content="output", tool_call_id="tc9", name="bash_tool", id="tm1")
        state = {"messages": [tm]}
        StreamingMixin._process_stream_state(
            self, state, MagicMock(), "thread1", MagicMock(), records
        )
        rec = records[0]
        assert rec["type"] == "tool"
        assert rec["content"] == "output"
        assert rec["tool_call_id"] == "tc9"
        assert rec["name"] == "bash_tool"
        # session_state 收到 tool_result, 且 tool_input 从缓存回填
        self._session_state.add_tool_result.assert_called_once()
        result_arg = self._session_state.add_tool_result.call_args.args[0]
        assert result_arg["tool_input"] == {"cmd": "pwd"}

    def test_offset_skips_already_processed(self):
        # _state_msg_offsets 已记录偏移 → 只处理新消息, 不重复抓
        self = _make_streaming_self()
        self._state_msg_offsets = {"thread1": 1}  # 已处理 1 条
        records: list[dict[str, Any]] = []
        state = {
            "messages": [
                AIMessage(content="old", id="ai_old"),  # 已处理, 跳过
                AIMessage(content="new", id="ai_new"),
            ]
        }
        StreamingMixin._process_stream_state(
            self, state, MagicMock(), "thread1", MagicMock(), records
        )
        assert len(records) == 1
        assert records[0]["content"] == "new"
        # offset 推进到全量
        assert self._state_msg_offsets["thread1"] == 2

    def test_records_none_does_not_crash(self):
        # records=None (调用方不抓) → 不崩
        self = _make_streaming_self()
        state = {"messages": [AIMessage(content="x", id="ai")]}
        StreamingMixin._process_stream_state(
            self, state, MagicMock(), "thread1", MagicMock(), None
        )
        self.memory.add_message.assert_called_once()


class TestReasoningCapturedToTrace:
    """护栏: 真实 COT (reasoning_content) 必须落进 session.reasoning_trace.

    防止将来某天把 reasoning_content 只转发给前端就完事, 而下游蒸馏
    (knowledge_distiller / evolution)_read 到空的 reasoning_trace。
    """

    def test_reasoning_content_persisted_to_memory(self):
        # AIMessage 带 reasoning_content → _process_stream_state 必须调
        # memory.add_reasoning, 让 COT 资产化进 reasoning_trace。
        self = _make_streaming_self()
        ai = AIMessage(content="final answer", id="ai_r1")
        ai.additional_kwargs["reasoning_content"] = (
            "先用 PBE 试探, 因为 LDA 低估带隙, 再对比 GGA."
        )
        state = {"messages": [ai]}
        StreamingMixin._process_stream_state(
            self, state, MagicMock(), "thread1", MagicMock(), None
        )
        self.memory.add_reasoning.assert_called_once()
        self.memory.add_reasoning.assert_called_with(
            "先用 PBE 试探, 因为 LDA 低估带隙, 再对比 GGA."
        )

    def test_no_reasoning_skips_add_reasoning(self):
        # 无 reasoning_content (如 OpenAI/Anthropic) → 不调 add_reasoning, 不误写.
        self = _make_streaming_self()
        state = {"messages": [AIMessage(content="plain answer", id="ai_r2")]}
        StreamingMixin._process_stream_state(
            self, state, MagicMock(), "thread1", MagicMock(), None
        )
        self.memory.add_reasoning.assert_not_called()
