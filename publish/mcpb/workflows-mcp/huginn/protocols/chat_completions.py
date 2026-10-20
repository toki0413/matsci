"""OpenAI Chat Completions 协议适配器 (请求转换 + 流式事件生成, 成对收口).

形状对齐 deepseek-recipe ``protocol/openai/chat_completion/{request,response}``:
请求方向把 ``messages`` / ``tools`` 归一进共享 IR; 响应方向把有序
``OutputChunk`` 流映射成 Chat Completions 的 delta chunk, 非流式口径复用同一
聚合状态生成 ``chat.completion`` 终态体.

仅抄形状/协议语义, 排除 DeepSeek 专属实现 (prompt 渲染 / tokenizer / DSML 标签).
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

__all__ = ["ChatCompletionsChunkGenerator", "ChatCompletionsRequest"]

_ROLE_MAP: dict[str, Role] = {
    "system": Role.SYSTEM,
    "developer": Role.SYSTEM,
    "user": Role.USER,
    "assistant": Role.ASSISTANT,
    "tool": Role.TOOL,
}


class ChatCompletionsRequest(ProtocolRequest):
    """``/v1/chat/completions`` 请求: convert() + chunk_generator() 成对绑定."""

    def __init__(
        self,
        payload: dict,
        *,
        markers: tuple[Marker, ...] = CONTROL_MARKERS,
        tool_call_parser: ToolCallParser = parse_tool_call_interval,
    ) -> None:
        if not isinstance(payload, dict):
            raise ValueError("Chat Completions 请求体必须是 JSON 对象")
        self._payload = payload
        self._markers = tuple(markers)
        self._tool_call_parser = tool_call_parser

    # ── 请求方向 ────────────────────────────────────────────────────

    def convert(self) -> ChatRequest:
        raw = self._payload.get("messages")
        if not isinstance(raw, list) or not raw:
            raise ValueError("messages 必须是非空列表")

        messages: list[Message] = []
        awaiting: dict[str, str] = {}

        for item in raw:
            if not isinstance(item, dict):
                raise ValueError("每条 message 必须是 JSON 对象")
            raw_role = str(item.get("role", ""))
            role = _ROLE_MAP.get(raw_role)
            if role is None:
                raise ValueError(f"未知 role: {raw_role!r}")

            if role is Role.TOOL:
                call_id = str(item.get("tool_call_id") or "").strip()
                if not call_id:
                    raise ValueError("tool 消息必须带 tool_call_id")
                if call_id not in awaiting:
                    raise ValueError(f"tool 消息引用了不存在的调用: {call_id!r}")
                awaiting.pop(call_id)
                messages.append(
                    Message(
                        role=Role.TOOL,
                        parts=self._parts(item.get("content")),
                        tool_call_id=call_id,
                        name=item.get("name"),
                    )
                )
                continue

            # 非 tool 消息出现时, 上一轮工具调用必须已全部有结果.
            if awaiting:
                raise ValueError(f"工具轮次不完整 (有调用缺输出): {sorted(awaiting)}")

            tool_calls = self._tool_calls(item.get("tool_calls"))
            messages.append(
                Message(
                    role=role,
                    parts=self._parts(item.get("content")),
                    tool_calls=tool_calls,
                )
            )
            awaiting = {call.call_id: call.name for call in tool_calls}

        if awaiting:
            raise ValueError(f"工具轮次不完整 (有调用缺输出): {sorted(awaiting)}")

        return ChatRequest(
            messages=tuple(messages),
            tools=tuple(self._tool_defs()),
            tool_choice=self._tool_choice(),
            max_tokens=self._as_int(
                self._payload.get("max_completion_tokens")
                if self._payload.get("max_completion_tokens") is not None
                else self._payload.get("max_tokens")
            ),
            temperature=self._as_float(self._payload.get("temperature")),
            stop=self._stop(),
            stream=bool(self._payload.get("stream", False)),
            response_format=self._payload.get("response_format"),
            metadata=dict(self._payload.get("metadata") or {}),
        )

    # ── 响应方向 ────────────────────────────────────────────────────

    def chunk_generator(self) -> ChunkGenerator:
        return ChatCompletionsChunkGenerator(
            model=str(self._payload.get("model", "huginn"))
        )

    def output_parser(self) -> OutputParser:
        return OutputParser(self._markers, tool_call_parser=self._tool_call_parser)

    def sse_terminator(self) -> str:
        # Chat Completions 流的结束标志是 `data: [DONE]` (非 JSON).
        return "data: [DONE]\n\n"

    # ── 内部: 请求解析 ──────────────────────────────────────────────

    def _parts(self, content: object) -> tuple[ContentPart, ...]:
        if content is None:
            return ()
        if isinstance(content, str):
            return (ContentPart(PartKind.TEXT, text=content),)
        if not isinstance(content, list):
            raise ValueError("content 必须是字符串、内容块列表或 null")
        parts: list[ContentPart] = []
        for block in content:
            if not isinstance(block, dict):
                raise ValueError("每个 content 块必须是 JSON 对象")
            btype = block.get("type")
            if btype in ("text", "input_text", "output_text"):
                parts.append(ContentPart(PartKind.TEXT, text=str(block.get("text", ""))))
            elif btype in ("image_url", "image"):
                parts.append(ContentPart(PartKind.IMAGE, image=normalize_image(block)))
            elif btype == "refusal":
                parts.append(
                    ContentPart(PartKind.REFUSAL, refusal=str(block.get("refusal", "")))
                )
            else:
                raise ValueError(f"不支持的 content 块类型: {btype!r}")
        return tuple(parts)

    def _tool_calls(self, raw: object) -> tuple[ToolCall, ...]:
        if raw is None:
            return ()
        if not isinstance(raw, list):
            raise ValueError("tool_calls 必须是列表")
        calls: list[ToolCall] = []
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError("每个 tool_call 必须是 JSON 对象")
            fn = item.get("function")
            if not isinstance(fn, dict):
                raise ValueError("tool_call 缺少 function 对象")
            name = str(fn.get("name") or "")
            if not name:
                raise ValueError("tool_call.function 缺少 name")
            arguments = fn.get("arguments", "")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            call_id = str(item.get("id") or "").strip()
            if not call_id:
                raise ValueError("tool_call 缺少 id")
            calls.append(
                ToolCall(call_id=call_id, name=name, arguments=arguments)
            )
        return tuple(calls)

    def _tool_defs(self) -> list[ToolDef]:
        raw = self._payload.get("tools") or []
        if not isinstance(raw, list):
            raise ValueError("tools 必须是列表")
        defs: list[ToolDef] = []
        for tool in raw:
            if not isinstance(tool, dict):
                raise ValueError("每个 tool 必须是 JSON 对象")
            if tool.get("type") not in (None, "function"):
                raise ValueError(f"不支持的工具类型: {tool.get('type')!r}")
            fn = tool.get("function")
            if not isinstance(fn, dict):
                raise ValueError("tool 缺少 function 对象")
            name = str(fn.get("name") or "")
            if not name:
                raise ValueError("tool.function 缺少 name")
            defs.append(
                ToolDef(
                    name=name,
                    description=str(fn.get("description", "")),
                    parameters=validate_json_schema(
                        fn.get("parameters") or {"type": "object"}
                    ),
                )
            )
        return defs

    def _tool_choice(self) -> str | None:
        raw = self._payload.get("tool_choice")
        if raw is None:
            return None
        if isinstance(raw, dict):
            fn = raw.get("function")
            name = fn.get("name") if isinstance(fn, dict) else None
            return str(name) if name else "auto"
        return str(raw)

    def _stop(self) -> tuple[str, ...]:
        raw = self._payload.get("stop")
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
class _ChatState:
    """流式/非流式共用的聚合状态 (同一 chunk 流的两种口径)."""

    model: str
    completion_id: str
    created: int
    text: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str | None = None


class ChatCompletionsChunkGenerator(ChunkGenerator):
    """``OutputChunk`` -> Chat Completions delta chunk (唯一映射实现)."""

    def __init__(
        self,
        *,
        completion_id: str = "chatcmpl-huginn",
        model: str = "huginn",
        created: int = 0,
    ) -> None:
        self._completion_id = completion_id
        self._model = model
        self._created = created

    def generate(self, chunks: Iterable[OutputChunk]) -> Iterator[ProtocolEvent]:
        state = self._state()
        # 首帧先声明 assistant 角色 (OpenAI 流式约定).
        yield self._chunk_event(state, {"role": "assistant", "content": ""}, None)
        for chunk in chunks:
            yield from self._map(chunk, state)

    def final_payload(self, chunks: Iterable[OutputChunk]) -> dict:
        state = self._state()
        for chunk in chunks:
            self._absorb(chunk, state)
        return self._completion(state)

    # ── 内部 ────────────────────────────────────────────────────────

    def _state(self) -> _ChatState:
        return _ChatState(
            model=self._model,
            completion_id=self._completion_id,
            created=self._created,
        )

    def _chunk_event(
        self,
        state: _ChatState,
        delta: dict,
        finish_reason: str | None,
    ) -> ProtocolEvent:
        return ProtocolEvent(
            "chat.completion.chunk",
            {
                "id": state.completion_id,
                "object": "chat.completion.chunk",
                "created": state.created,
                "model": state.model,
                "choices": [
                    {"index": 0, "delta": delta, "finish_reason": finish_reason}
                ],
            },
        )

    def _map(
        self, chunk: OutputChunk, state: _ChatState
    ) -> list[ProtocolEvent]:
        if chunk.kind is ChunkKind.FINISH:
            state.finish_reason = self._finish_reason(chunk, state)
            return [self._chunk_event(state, {}, state.finish_reason)]
        self._absorb(chunk, state)
        if chunk.kind is ChunkKind.TEXT:
            return [self._chunk_event(state, {"content": chunk.text}, None)]
        if chunk.kind is ChunkKind.REASONING:
            return [self._chunk_event(state, {"reasoning_content": chunk.text}, None)]
        if chunk.kind is ChunkKind.TOOL_CALL_DONE and chunk.tool_call is not None:
            call = chunk.tool_call
            return [
                self._chunk_event(
                    state,
                    {
                        "tool_calls": [
                            {
                                "index": len(state.tool_calls) - 1,
                                "id": call.call_id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": call.arguments,
                                },
                            }
                        ]
                    },
                    None,
                )
            ]
        # TOOL_CALL_DELTA 是区间内原始文本, 完整参数已由 TOOL_CALL_DONE 承载.
        return []

    def _absorb(self, chunk: OutputChunk, state: _ChatState) -> None:
        if chunk.kind is ChunkKind.TEXT:
            state.text.append(chunk.text)
        elif chunk.kind is ChunkKind.REASONING:
            state.reasoning.append(chunk.text)
        elif chunk.kind is ChunkKind.TOOL_CALL_DONE and chunk.tool_call is not None:
            call = chunk.tool_call
            state.tool_calls.append(
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
            )

    def _finish_reason(self, chunk: OutputChunk, state: _ChatState) -> str:
        if state.tool_calls:
            return "tool_calls"
        return chunk.finish_reason or "stop"

    def _completion(self, state: _ChatState) -> dict:
        message: dict = {
            "role": "assistant",
            "content": "".join(state.text) or None,
        }
        if state.tool_calls:
            message["tool_calls"] = list(state.tool_calls)
        if state.reasoning:
            message["reasoning_content"] = "".join(state.reasoning)
        return {
            "id": state.completion_id,
            "object": "chat.completion",
            "created": state.created,
            "model": state.model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": state.finish_reason or "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
