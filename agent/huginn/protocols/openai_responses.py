"""OpenAI Responses 协议适配器 (请求转换 + 流式事件生成, 成对收口).

请求方向 ``ResponsesRequest.convert()`` 对齐 deepseek-recipe
``protocol/openai/responses/request/convert.rs``; 响应方向
``ResponsesChunkGenerator`` 对齐 ``responses/response/chunk_generator.rs``.

仅抄形状/协议语义, 排除 DeepSeek 专属实现 (prompt 渲染 / tokenizer / DSML 标签).
标记字面量是本适配器**自定义**的 huginn 控制 token, 与 DeepSeek DSML 无关.
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
    ToolKind,
    ToolResult,
)
from huginn.protocols.primitives import (
    CONTROL_MARKERS,
    flatten_namespace,
    normalize_image,
    parse_tool_call_interval,
    resolve_tool_call_id,
    unflatten_namespace,
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

__all__ = [
    "RESPONSES_MARKERS",
    "ResponsesChunkGenerator",
    "ResponsesRequest",
]

# huginn 自定义的响应控制标记 (非 DeepSeek DSML), 三协议共用同一份定义.
RESPONSES_MARKERS: tuple[Marker, ...] = CONTROL_MARKERS

# apply_patch 型自定义工具硬编码为内置函数形状 (对齐 convert.rs 的硬编码语义).
APPLY_PATCH_SCHEMA: dict = {
    "type": "object",
    "properties": {"input": {"type": "string"}},
    "required": ["input"],
}

_ROLE_MAP: dict[str, Role] = {
    "user": Role.USER,
    "assistant": Role.ASSISTANT,
    "system": Role.SYSTEM,
    "developer": Role.SYSTEM,
    "tool": Role.TOOL,
}


class ResponsesRequest(ProtocolRequest):
    """``/v1/responses`` 请求: convert() + chunk_generator() 成对绑定."""

    def __init__(
        self,
        payload: dict,
        *,
        markers: tuple[Marker, ...] = RESPONSES_MARKERS,
        allow_web_search: bool = False,
        tool_call_parser: ToolCallParser = parse_tool_call_interval,
    ) -> None:
        if not isinstance(payload, dict):
            raise ValueError("Responses 请求体必须是 JSON 对象")
        self._payload = payload
        self._markers = tuple(markers)
        self._allow_web_search = allow_web_search
        self._tool_call_parser = tool_call_parser

    # ── 请求方向 ────────────────────────────────────────────────────

    def convert(self) -> ChatRequest:
        messages: list[Message] = []
        calls: dict[str, ToolCall] = {}
        order: list[str] = []
        outputs: dict[str, ToolResult] = {}

        for item in self._items():
            itype = item.get("type")
            if itype == "message":
                messages.append(self._message(item))
            elif itype == "reasoning":
                msg = self._reasoning(item)
                if msg is not None:
                    messages.append(msg)
            elif itype in ("function_call", "custom_tool_call"):
                call = self._tool_call(item)
                if call.call_id not in calls:
                    order.append(call.call_id)
                calls[call.call_id] = call
            elif itype in ("function_call_output", "custom_tool_call_output"):
                result = self._tool_result(item)
                outputs[result.call_id] = result
            else:
                raise ValueError(f"不支持的 Responses input 项类型: {itype!r}")

        # 工具调用与输出成组: 有调用缺输出 = 半截工具轮次 -> 拒绝.
        missing = [cid for cid in order if cid not in outputs]
        if missing:
            raise ValueError(f"工具轮次不完整 (有调用缺输出): {missing}")
        if order:
            messages.append(
                Message(
                    role=Role.ASSISTANT,
                    tool_calls=tuple(calls[cid] for cid in order),
                )
            )
            for cid in order:
                result = outputs[cid]
                messages.append(
                    Message(
                        role=Role.TOOL,
                        parts=(ContentPart(PartKind.TEXT, text=result.output),),
                        tool_call_id=cid,
                        name=result.name,
                    )
                )

        return ChatRequest(
            messages=tuple(messages),
            tools=tuple(self._tool_defs()),
            tool_choice=self._payload.get("tool_choice"),
            max_tokens=self._as_int(self._payload.get("max_output_tokens")),
            temperature=self._as_float(self._payload.get("temperature")),
            stop=self._stop(),
            stream=bool(self._payload.get("stream", False)),
            response_format=self._payload.get("response_format"),
            metadata=dict(self._payload.get("metadata") or {}),
        )

    # ── 响应方向 ────────────────────────────────────────────────────

    def chunk_generator(self) -> ChunkGenerator:
        return ResponsesChunkGenerator(
            response_id="resp_huginn",
            model=str(self._payload.get("model", "huginn")),
        )

    def output_parser(self) -> OutputParser:
        """本请求对应的输出解析器 (标记 + 工具调用区间语法)."""
        return OutputParser(self._markers, tool_call_parser=self._tool_call_parser)

    # ── 内部: 请求解析 ──────────────────────────────────────────────

    def _items(self) -> list[dict]:
        raw = self._payload.get("input")
        if raw is None:
            raise ValueError("input 必须提供")
        if isinstance(raw, str):
            if not raw.strip():
                raise ValueError("input 不能为空字符串")
            return [{"type": "message", "role": "user", "content": raw}]
        if not isinstance(raw, list):
            raise ValueError("input 必须是字符串或 input 项列表")
        if not raw:
            raise ValueError("input 必须是非空列表")
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError("每个 input 项必须是 JSON 对象")
        return raw

    def _message(self, item: dict) -> Message:
        raw_role = str(item.get("role", "user"))
        role = _ROLE_MAP.get(raw_role)
        if role is None:
            raise ValueError(f"未知 role: {raw_role!r}")
        content = item.get("content")
        if isinstance(content, str):
            parts = (ContentPart(PartKind.TEXT, text=content),)
        else:
            parts = tuple(self._part(c) for c in content or [])
        return Message(role=role, parts=parts)

    def _part(self, block: dict) -> ContentPart:
        ctype = block.get("type")
        if ctype in ("input_text", "output_text", "text"):
            return ContentPart(PartKind.TEXT, text=str(block.get("text", "")))
        if ctype in ("input_image", "image"):
            return ContentPart(PartKind.IMAGE, image=normalize_image(block))
        if ctype == "refusal":
            return ContentPart(PartKind.REFUSAL, refusal=str(block.get("refusal", "")))
        raise ValueError(f"不支持的 content 块类型: {ctype!r}")

    def _reasoning(self, item: dict) -> Message | None:
        text = str(item.get("text") or "")
        summary = item.get("summary")
        if not text and isinstance(summary, list):
            text = "".join(
                str(s.get("text", "")) for s in summary if isinstance(s, dict)
            )
        if not text:
            return None
        return Message(
            role=Role.ASSISTANT, parts=(ContentPart(PartKind.REASONING, text=text),)
        )

    def _tool_call(self, item: dict) -> ToolCall:
        seed = json.dumps(item, sort_keys=True, ensure_ascii=False)
        call_id = resolve_tool_call_id(item.get("call_id") or item.get("id"), seed=seed)
        namespace = item.get("namespace")
        name = str(item.get("name", ""))
        if item.get("type") == "function_call":
            arguments = item.get("arguments", "")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
        else:  # custom_tool_call: 自由文本 input 包成统一 JSON 形状
            arguments = json.dumps(
                {"input": str(item.get("input", ""))}, ensure_ascii=False
            )
        return ToolCall(
            call_id=call_id,
            name=flatten_namespace(namespace, name),
            arguments=arguments,
            namespace=namespace,
        )

    def _tool_result(self, item: dict) -> ToolResult:
        seed = json.dumps(item, sort_keys=True, ensure_ascii=False)
        call_id = resolve_tool_call_id(item.get("call_id"), seed=seed)
        output = item.get("output", "")
        if not isinstance(output, str):
            output = json.dumps(output, ensure_ascii=False)
        return ToolResult(
            call_id=call_id, name=str(item.get("name", "")), output=output
        )

    def _tool_defs(self) -> list[ToolDef]:
        tools = self._payload.get("tools") or []
        if not isinstance(tools, list):
            raise ValueError("tools 必须是列表")
        defs: list[ToolDef] = []
        for tool in tools:
            ttype = tool.get("type")
            if ttype == "function":
                defs.append(
                    ToolDef(
                        name=flatten_namespace(
                            tool.get("namespace"), str(tool.get("name", ""))
                        ),
                        description=str(tool.get("description", "")),
                        parameters=validate_json_schema(
                            tool.get("parameters") or {"type": "object"}
                        ),
                        kind=ToolKind.FUNCTION,
                    )
                )
            elif ttype == "custom":
                name = str(tool.get("name", ""))
                if name == "apply_patch":
                    defs.append(
                        ToolDef(
                            name="apply_patch",
                            description="内置补丁应用工具",
                            parameters=APPLY_PATCH_SCHEMA,
                            kind=ToolKind.FUNCTION,
                        )
                    )
                else:
                    defs.append(
                        ToolDef(
                            name=flatten_namespace(tool.get("namespace"), name),
                            description=str(tool.get("description", "")),
                            kind=ToolKind.CUSTOM,
                        )
                    )
            elif ttype == "web_search":
                if not self._allow_web_search:
                    raise ValueError(
                        "web_search 工具不受支持 (本后端无内建 web_search 工具面)"
                    )
                defs.append(ToolDef(name="web_search", kind=ToolKind.FUNCTION))
            else:
                raise ValueError(f"不支持的工具类型: {ttype!r}")
        return defs

    def _stop(self) -> tuple[str, ...]:
        raw = self._payload.get("stop") or self._payload.get("stop_sequences") or ()
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
class _GenState:
    """流式生成器的可变状态 (seq 全局递增 + 已产出 output 项)."""

    seq: int = 0
    output: list[dict] = field(default_factory=list)
    open_item: dict | None = None
    tool_raw: list[str] = field(default_factory=list)


class ResponsesChunkGenerator(ChunkGenerator):
    """``OutputChunk`` -> Responses SSE 事件 (唯一映射实现)."""

    def __init__(
        self, *, response_id: str = "resp_huginn", model: str = "huginn"
    ) -> None:
        self._response_id = response_id
        self._model = model

    def generate(self, chunks: Iterable[OutputChunk]) -> Iterator[ProtocolEvent]:
        state = _GenState()
        yield self._event(
            state,
            "response.created",
            response={
                "id": self._response_id,
                "model": self._model,
                "status": "in_progress",
            },
        )
        for chunk in chunks:
            yield from self._map(chunk, state)

    def final_payload(self, chunks: Iterable[OutputChunk]) -> dict:
        """非流式: 终端事件 (``response.completed``) 的 data 即完整响应体."""
        last: ProtocolEvent | None = None
        for event in self.generate(chunks):
            last = event
        return last.data if last is not None else {}

    # ── 内部 ────────────────────────────────────────────────────────

    def _event(self, state: _GenState, name: str, **data: object) -> ProtocolEvent:
        payload = {"type": name, "sequence_number": state.seq, **data}
        state.seq += 1
        return ProtocolEvent(name, payload)

    def _map(self, chunk: OutputChunk, state: _GenState) -> list[ProtocolEvent]:
        if chunk.kind is ChunkKind.TEXT:
            return self._text(chunk, state, item_type="message")
        if chunk.kind is ChunkKind.REASONING:
            return self._text(chunk, state, item_type="reasoning")
        if chunk.kind is ChunkKind.TOOL_CALL_DELTA:
            state.tool_raw.append(chunk.text)
            return []
        if chunk.kind is ChunkKind.TOOL_CALL_DONE:
            return self._tool_call(chunk, state)
        if chunk.kind is ChunkKind.FINISH:
            return self._finish(chunk, state)
        return []

    def _text(
        self, chunk: OutputChunk, state: _GenState, *, item_type: str
    ) -> list[ProtocolEvent]:
        events: list[ProtocolEvent] = []
        open_item = state.open_item
        if open_item is None or open_item["type"] != item_type:
            events.extend(self._close_open_item(state))
            index = len(state.output)
            open_item = {
                "id": f"{self._response_id}_{item_type}_{index}",
                "type": item_type,
                "status": "in_progress",
                "_index": index,
                "_buf": [],
            }
            state.open_item = open_item
            events.append(
                self._event(
                    state,
                    "response.output_item.added",
                    output_index=index,
                    item=self._public(open_item),
                )
            )
        open_item["_buf"].append(chunk.text)
        part = "output_text" if item_type == "message" else "reasoning_text"
        delta_event = (
            "response.output_text.delta"
            if item_type == "message"
            else "response.reasoning_text.delta"
        )
        events.append(
            self._event(
                state,
                delta_event,
                item_id=open_item["id"],
                output_index=open_item["_index"],
                content_index=0,
                part_type=part,
                delta=chunk.text,
            )
        )
        return events

    def _tool_call(self, chunk: OutputChunk, state: _GenState) -> list[ProtocolEvent]:
        call = chunk.tool_call
        if call is None:
            return []
        state.tool_raw.clear()
        events = self._close_open_item(state)
        index = len(state.output)
        namespace, local = unflatten_namespace(call.name)
        item: dict = {
            "id": f"{self._response_id}_fc_{index}",
            "type": "function_call",
            "status": "completed",
            "call_id": call.call_id,
            "name": local,
            "arguments": call.arguments,
        }
        if namespace is not None:
            item["namespace"] = namespace
        events.append(
            self._event(
                state,
                "response.output_item.added",
                output_index=index,
                item={**item, "status": "in_progress"},
            )
        )
        events.append(
            self._event(
                state,
                "response.function_call_arguments.delta",
                item_id=item["id"],
                output_index=index,
                delta=call.arguments,
            )
        )
        state.output.append(item)
        events.append(
            self._event(
                state, "response.output_item.done", output_index=index, item=item
            )
        )
        return events

    def _finish(self, chunk: OutputChunk, state: _GenState) -> list[ProtocolEvent]:
        events = self._close_open_item(state)
        events.append(
            self._event(
                state,
                "response.completed",
                response={
                    "id": self._response_id,
                    "model": self._model,
                    "status": "completed",
                    "finish_reason": chunk.finish_reason or "stop",
                    "output": list(state.output),
                },
            )
        )
        return events

    def _close_open_item(self, state: _GenState) -> list[ProtocolEvent]:
        open_item = state.open_item
        if open_item is None:
            return []
        state.open_item = None
        public = self._public(open_item)
        state.output.append(public)
        return [
            self._event(
                state,
                "response.output_item.done",
                output_index=open_item["_index"],
                item=public,
            )
        ]

    @staticmethod
    def _public(item: dict) -> dict:
        public = {k: v for k, v in item.items() if not k.startswith("_")}
        if item["type"] in ("message", "reasoning"):
            part = "output_text" if item["type"] == "message" else "reasoning_text"
            public["content"] = [{"type": part, "text": "".join(item.get("_buf", []))}]
        return public
