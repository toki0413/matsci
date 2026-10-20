"""Anthropic Messages 协议适配器 (请求转换 + 流式事件生成, 成对收口).

形状对齐 deepseek-recipe ``protocol/anthropic/messages``: 请求方向把 ``system`` /
``messages`` / ``tools`` 归一进共享 IR; 响应方向把有序 ``OutputChunk`` 流映射成
Anthropic 的 content block 事件序列, 非流式口径复用同一份 block 聚合状态生成
``message`` 终态体.

协议差异在转换边界吸收: Anthropic 把工具结果放在 user 消息的 ``tool_result`` 块里,
IR 里落成 ``Role.TOOL`` 消息; 工具调用是 assistant 的 ``tool_use`` 块, IR 里落成
``Message.tool_calls``. 仅抄形状/协议语义, 不抄 DeepSeek 专属实现.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from huginn.protocols.ir import (
    ChatRequest,
    ContentPart,
    Message,
    PartKind,
    Role,
    ToolCall,
    ToolDef,
)
from huginn.protocols.primitives import (
    CONTROL_MARKERS,
    normalize_image,
    parse_tool_call_interval,
    validate_json_schema,
)
from huginn.protocols.stream import (
    ChunkGenerator,
    ChunkKind,
    Marker,
    OutputChunk,
    OutputParser,
    ProtocolEvent,
    ProtocolRequest,
    ToolCallParser,
)

__all__ = ["AnthropicMessagesChunkGenerator", "AnthropicMessagesRequest"]

# Anthropic 只有 user / assistant 两种消息角色; system 是顶层字段.
_ROLE_MAP: dict[str, Role] = {
    "user": Role.USER,
    "assistant": Role.ASSISTANT,
}

# 请求侧工具选择语义 -> IR 归一字符串.
_TOOL_CHOICE_TYPES: dict[str, str] = {
    "auto": "auto",
    "any": "required",
    "none": "none",
}

# 后端 finish_reason -> Anthropic stop_reason (响应侧反向归一).
_STOP_REASONS: dict[str, str] = {
    "stop": "end_turn",
    "end_turn": "end_turn",
    "length": "max_tokens",
    "max_tokens": "max_tokens",
    "tool_calls": "tool_use",
    "tool_use": "tool_use",
    "stop_sequence": "stop_sequence",
    "content_filter": "refusal",
}


class AnthropicMessagesRequest(ProtocolRequest):
    """``/v1/messages`` 请求: convert() + chunk_generator() 成对绑定."""

    def __init__(
        self,
        payload: dict,
        *,
        markers: tuple[Marker, ...] = CONTROL_MARKERS,
        tool_call_parser: ToolCallParser = parse_tool_call_interval,
    ) -> None:
        if not isinstance(payload, dict):
            raise ValueError("Messages 请求体必须是 JSON 对象")
        self._payload = payload
        self._markers = tuple(markers)
        self._tool_call_parser = tool_call_parser

    # ── 请求方向 ────────────────────────────────────────────────────

    def convert(self) -> ChatRequest:
        raw = self._payload.get("messages")
        if not isinstance(raw, list) or not raw:
            raise ValueError("messages 必须是非空列表")

        messages: list[Message] = []
        system = self._system()
        if system:
            messages.append(
                Message(
                    role=Role.SYSTEM,
                    parts=(ContentPart(PartKind.TEXT, text=system),),
                )
            )

        pending: tuple[str, ...] = ()
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError("每条 message 必须是 JSON 对象")
            raw_role = str(item.get("role", ""))
            role = _ROLE_MAP.get(raw_role)
            if role is None:
                raise ValueError(f"未知 role: {raw_role!r} (Anthropic 只有 user/assistant)")

            blocks = self._blocks(item.get("content"), raw_role)
            if role is Role.ASSISTANT:
                if pending:
                    raise ValueError(f"工具轮次不完整 (有调用缺输出): {list(pending)}")
                calls = tuple(self._tool_call(b) for b in blocks if b.get("type") == "tool_use")
                parts = tuple(self._part(b) for b in blocks if b.get("type") != "tool_use")
                messages.append(
                    Message(role=Role.ASSISTANT, parts=parts, tool_calls=calls)
                )
                pending = tuple(call.call_id for call in calls)
                continue

            # user: tool_result 块落成 IR 的 TOOL 消息, 其余内容落成 USER 消息.
            results = [b for b in blocks if b.get("type") == "tool_result"]
            others = [b for b in blocks if b.get("type") != "tool_result"]
            if pending:
                if not results:
                    raise ValueError(f"工具轮次不完整 (有调用缺输出): {list(pending)}")
                provided = {str(b.get("tool_use_id", "")) for b in results}
                missing = [cid for cid in pending if cid not in provided]
                if missing:
                    raise ValueError(f"工具轮次不完整 (有调用缺输出): {missing}")
                unknown = sorted(provided - set(pending))
                if unknown:
                    raise ValueError(f"tool_result 引用了不存在的调用: {unknown}")
            elif results:
                raise ValueError("tool_result 引用了不存在的调用")
            for block in results:
                messages.append(self._tool_result(block))
            if others:
                messages.append(
                    Message(
                        role=Role.USER,
                        parts=tuple(self._part(b) for b in others),
                    )
                )
            pending = ()

        if pending:
            raise ValueError(f"工具轮次不完整 (有调用缺输出): {list(pending)}")

        return ChatRequest(
            messages=tuple(messages),
            tools=tuple(self._tool_defs()),
            tool_choice=self._tool_choice(),
            max_tokens=self._as_int(self._payload.get("max_tokens")),
            temperature=self._as_float(self._payload.get("temperature")),
            stop=self._stop(),
            stream=bool(self._payload.get("stream", False)),
            response_format=None,
            metadata=dict(self._payload.get("metadata") or {}),
        )

    # ── 响应方向 ────────────────────────────────────────────────────

    def chunk_generator(self) -> ChunkGenerator:
        return AnthropicMessagesChunkGenerator(
            message_id="msg_huginn",
            model=str(self._payload.get("model", "huginn")),
        )

    def output_parser(self) -> OutputParser:
        return OutputParser(self._markers, tool_call_parser=self._tool_call_parser)

    def sse_frame(self, event: ProtocolEvent) -> str:
        """Anthropic SSE: 每个事件同时给 ``event:`` 名与 ``data:`` JSON."""
        data = json.dumps(event.data, ensure_ascii=False)
        return f"event: {event.event}\ndata: {data}\n\n"

    # ── 内部: 请求解析 ──────────────────────────────────────────────

    def _system(self) -> str:
        raw = self._payload.get("system")
        if raw is None:
            return ""
        if isinstance(raw, str):
            return raw
        if not isinstance(raw, list):
            raise ValueError("system 必须是字符串或文本块列表")
        return "".join(str(b.get("text", "")) for b in self._blocks(raw, "system"))

    def _blocks(self, content: object, role: str) -> list[dict]:
        if content is None:
            return []
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        if not isinstance(content, list):
            raise ValueError(f"{role} 消息的 content 必须是字符串或内容块列表")
        for block in content:
            if not isinstance(block, dict):
                raise ValueError("每个 content 块必须是 JSON 对象")
        return content

    def _part(self, block: dict) -> ContentPart:
        btype = block.get("type")
        if btype == "text":
            return ContentPart(PartKind.TEXT, text=str(block.get("text", "")))
        if btype == "thinking":
            return ContentPart(
                PartKind.REASONING, text=str(block.get("thinking", ""))
            )
        if btype == "image":
            return ContentPart(PartKind.IMAGE, image=normalize_image(block))
        raise ValueError(f"不支持的 content 块类型: {btype!r}")

    def _tool_call(self, block: dict) -> ToolCall:
        call_id = str(block.get("id") or "").strip()
        if not call_id:
            raise ValueError("tool_use 块缺少 id")
        name = str(block.get("name") or "")
        if not name:
            raise ValueError("tool_use 块缺少 name")
        arguments = block.get("input", {})
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        return ToolCall(call_id=call_id, name=name, arguments=arguments)

    def _tool_result(self, block: dict) -> Message:
        call_id = str(block.get("tool_use_id") or "").strip()
        if not call_id:
            raise ValueError("tool_result 块缺少 tool_use_id")
        content = block.get("content")
        if isinstance(content, str):
            parts = (ContentPart(PartKind.TEXT, text=content),)
        else:
            parts = tuple(
                self._part(b)
                for b in self._blocks(content, "tool_result")
                if b.get("type") != "tool_result"
            )
        return Message(role=Role.TOOL, parts=parts, tool_call_id=call_id)

    def _tool_defs(self) -> list[ToolDef]:
        raw = self._payload.get("tools") or []
        if not isinstance(raw, list):
            raise ValueError("tools 必须是列表")
        defs: list[ToolDef] = []
        for tool in raw:
            if not isinstance(tool, dict):
                raise ValueError("每个 tool 必须是 JSON 对象")
            if tool.get("type") not in (None, "custom"):
                raise ValueError(f"不支持的工具类型: {tool.get('type')!r}")
            name = str(tool.get("name") or "")
            if not name:
                raise ValueError("tool 缺少 name")
            defs.append(
                ToolDef(
                    name=name,
                    description=str(tool.get("description", "")),
                    parameters=validate_json_schema(
                        tool.get("input_schema") or {"type": "object"}
                    ),
                )
            )
        return defs

    def _tool_choice(self) -> str | None:
        raw = self._payload.get("tool_choice")
        if raw is None:
            return None
        if not isinstance(raw, dict):
            return str(raw)
        choice = str(raw.get("type") or "auto")
        if choice == "tool":
            name = str(raw.get("name") or "")
            if not name:
                raise ValueError("tool_choice.type=tool 时必须给 name")
            return name
        mapped = _TOOL_CHOICE_TYPES.get(choice)
        if mapped is None:
            raise ValueError(f"不支持的 tool_choice.type: {choice!r}")
        return mapped

    def _stop(self) -> tuple[str, ...]:
        raw = self._payload.get("stop_sequences")
        if raw is None:
            return ()
        if isinstance(raw, str):
            return (raw,)
        return tuple(str(s) for s in raw)

    @staticmethod
    def _as_int(value: object) -> int | None:
        return None if value is None else int(value)  # type: ignore[arg-type]

    @staticmethod
    def _as_float(value: object) -> float | None:
        return None if value is None else float(value)  # type: ignore[arg-type]


@dataclass
class _MsgState:
    """流式/非流式共用的 content block 聚合状态."""

    model: str
    message_id: str
    blocks: list[dict] = field(default_factory=list)
    open_index: int | None = None
    open_type: str | None = None
    stop_reason: str | None = None


class AnthropicMessagesChunkGenerator(ChunkGenerator):
    """``OutputChunk`` -> Anthropic content block 事件 (唯一映射实现)."""

    def __init__(
        self, *, message_id: str = "msg_huginn", model: str = "huginn"
    ) -> None:
        self._message_id = message_id
        self._model = model

    def generate(self, chunks: Iterable[OutputChunk]) -> Iterator[ProtocolEvent]:
        state = self._state()
        yield self._event("message_start", message=self._message(state))
        stopped = False
        for chunk in chunks:
            for event in self._map(chunk, state):
                stopped = stopped or event.event == "message_stop"
                yield event
        if not stopped:  # 防御: chunk 流未给 FINISH 时也要收口, 不泄漏半截流.
            yield from self._finish(state, None)

    def final_payload(self, chunks: Iterable[OutputChunk]) -> dict:
        """非流式: 与流式同一 block 聚合状态, 只是不发 delta 事件."""
        state = self._state()
        for chunk in chunks:
            self._ingest(chunk, state)
        return self._message(state)

    # ── 内部 ────────────────────────────────────────────────────────

    def _state(self) -> _MsgState:
        return _MsgState(model=self._model, message_id=self._message_id)

    def _event(self, name: str, **data: object) -> ProtocolEvent:
        return ProtocolEvent(name, {"type": name, **data})

    def _map(self, chunk: OutputChunk, state: _MsgState) -> list[ProtocolEvent]:
        if chunk.kind is ChunkKind.TEXT:
            return self._text(chunk, state, "text", "text_delta")
        if chunk.kind is ChunkKind.REASONING:
            return self._text(chunk, state, "thinking", "thinking_delta")
        if chunk.kind is ChunkKind.TOOL_CALL_DONE and chunk.tool_call is not None:
            return self._tool_use(chunk, state)
        if chunk.kind is ChunkKind.FINISH:
            return self._finish(state, chunk.finish_reason)
        # TOOL_CALL_DELTA 是区间内原始文本, 完整参数已由 TOOL_CALL_DONE 承载.
        return []

    def _text(
        self,
        chunk: OutputChunk,
        state: _MsgState,
        block_type: str,
        delta_type: str,
    ) -> list[ProtocolEvent]:
        events: list[ProtocolEvent] = []
        if state.open_type != block_type:
            events.extend(self._close(state))
            index = self._open(state, block_type)
            events.append(
                self._event(
                    "content_block_start",
                    index=index,
                    content_block={"type": block_type, block_type: ""},
                )
            )
        index = state.open_index
        assert index is not None
        state.blocks[index][block_type] += chunk.text
        events.append(
            self._event(
                "content_block_delta",
                index=index,
                delta={"type": delta_type, delta_type: chunk.text},
            )
        )
        return events

    def _tool_use(self, chunk: OutputChunk, state: _MsgState) -> list[ProtocolEvent]:
        call = chunk.tool_call
        assert call is not None
        events = self._close(state)
        index = self._open(state, "tool_use")
        events.append(
            self._event(
                "content_block_start",
                index=index,
                content_block={
                    "type": "tool_use",
                    "id": call.call_id,
                    "name": call.name,
                    "input": {},
                },
            )
        )
        state.blocks[index].update(
            {"id": call.call_id, "name": call.name, "input": _as_input(call.arguments)}
        )
        events.append(
            self._event(
                "content_block_delta",
                index=index,
                delta={"type": "input_json_delta", "partial_json": call.arguments},
            )
        )
        events.append(self._event("content_block_stop", index=index))
        state.open_index = None
        state.open_type = None
        return events

    def _finish(
        self, state: _MsgState, finish_reason: str | None
    ) -> list[ProtocolEvent]:
        events = self._close(state)
        if any(b.get("type") == "tool_use" for b in state.blocks):
            stop_reason = "tool_use"
        else:
            stop_reason = _STOP_REASONS.get(finish_reason or "stop", "end_turn")
        state.stop_reason = stop_reason
        events.append(
            self._event(
                "message_delta",
                delta={"stop_reason": stop_reason, "stop_sequence": None},
                usage={"output_tokens": 0},
            )
        )
        events.append(self._event("message_stop"))
        return events

    def _open(self, state: _MsgState, block_type: str) -> int:
        index = len(state.blocks)
        block: dict = {"type": block_type}
        if block_type == "text":
            block["text"] = ""
        elif block_type == "thinking":
            block["thinking"] = ""
        state.blocks.append(block)
        state.open_index = index
        state.open_type = block_type
        return index

    def _close(self, state: _MsgState) -> list[ProtocolEvent]:
        if state.open_index is None:
            return []
        index = state.open_index
        state.open_index = None
        state.open_type = None
        return [self._event("content_block_stop", index=index)]

    def _ingest(self, chunk: OutputChunk, state: _MsgState) -> None:
        """非流式聚合: 只改 block 状态, 不产事件 (与流式共用 block 形状)."""
        if chunk.kind is ChunkKind.TEXT:
            self._accumulate(state, "text", chunk.text)
        elif chunk.kind is ChunkKind.REASONING:
            self._accumulate(state, "thinking", chunk.text)
        elif chunk.kind is ChunkKind.TOOL_CALL_DONE and chunk.tool_call is not None:
            call = chunk.tool_call
            state.blocks.append(
                {
                    "type": "tool_use",
                    "id": call.call_id,
                    "name": call.name,
                    "input": _as_input(call.arguments),
                }
            )
            # 工具块本身是完整的, 后续文本必须另起 block (与流式 _close 同口径).
            state.open_index = None
            state.open_type = None
        elif chunk.kind is ChunkKind.FINISH:
            if any(b.get("type") == "tool_use" for b in state.blocks):
                state.stop_reason = "tool_use"
            else:
                state.stop_reason = _STOP_REASONS.get(
                    chunk.finish_reason or "stop", "end_turn"
                )

    @staticmethod
    def _accumulate(state: _MsgState, block_type: str, text: str) -> None:
        if state.open_type != block_type:
            state.blocks.append({"type": block_type, block_type: ""})
            state.open_index = len(state.blocks) - 1
            state.open_type = block_type
        index = state.open_index
        assert index is not None
        state.blocks[index][block_type] += text

    def _message(self, state: _MsgState) -> dict:
        return {
            "id": state.message_id,
            "type": "message",
            "role": "assistant",
            "model": state.model,
            "content": [
                {k: v for k, v in block.items() if not k.startswith("_")}
                for block in state.blocks
            ],
            "stop_reason": state.stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }


def _as_input(arguments: str) -> dict:
    """工具调用参数串 -> Anthropic ``input`` 对象 (非法 JSON 原样保留)."""
    if not arguments.strip():
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return {"raw": arguments}
    return parsed if isinstance(parsed, dict) else {"value": parsed}
