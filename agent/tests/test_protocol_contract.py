"""多协议适配层契约门 (红蓝双轨).

蓝队断言控制**生效**; 红队主动尝试**绕过**. 沿用从 awesome-cyber-security-university
提炼的"成对"原则: 每协议必须同时给请求方向 (convert) 与响应方向 (chunk_generator),
少一半即红; 解析状态机只在特殊 token 上匹配标记, 普通 token 拼出的标记字面量不可伪造.

不引新测试框架; 纯 assert 型自检.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import pkgutil

import pytest

from huginn import protocols
from huginn.protocols.anthropic_messages import AnthropicMessagesRequest
from huginn.protocols.chat_completions import ChatCompletionsRequest
from huginn.protocols.ir import ChatRequest, Message, PartKind, Role, ToolCall
from huginn.protocols.openai_responses import ResponsesRequest
from huginn.protocols.primitives import flatten_namespace, unflatten_namespace
from huginn.protocols.stream import (
    ChunkKind,
    Marker,
    MarkerKind,
    OutputChunk,
    OutputParser,
    ProtocolRequest,
    Token,
)

# ──────────────────── 成对性判据 (蓝队 + 红队共用) ────────────────────


def _protocol_request_classes() -> list[type]:
    """遍历 `huginn.protocols` 下所有 `ProtocolRequest` 子类 (去重)."""
    seen: dict[str, type] = {}
    for info in pkgutil.iter_modules(protocols.__path__):
        module = importlib.import_module(f"{protocols.__name__}.{info.name}")
        for obj in vars(module).values():
            if (
                isinstance(obj, type)
                and issubclass(obj, ProtocolRequest)
                and obj is not ProtocolRequest
            ):
                seen[f"{obj.__module__}:{obj.__qualname__}"] = obj
    return list(seen.values())


def _unpaired(classes: list[type]) -> list[tuple[str, tuple[str, ...]]]:
    """偏半个协议的类: 缺 convert / chunk_generator / output_parser 任一."""
    bad: list[tuple[str, tuple[str, ...]]] = []
    for cls in classes:
        missing = [
            name
            for name in ("convert", "chunk_generator", "output_parser")
            if getattr(cls, name, None) is None
            or getattr(getattr(cls, name), "__isabstractmethod__", False)
            or name in getattr(cls, "__abstractmethods__", frozenset())
        ]
        if missing:
            bad.append((cls.__name__, tuple(missing)))
    return bad


# ──────────────────── 蓝队: 断言控制生效 ────────────────────


def test_every_protocol_request_is_paired():
    """遍历所有 ProtocolRequest 子类, 断言 convert / chunk_generator 成对.

    防"只加半个协议"静默漏响应方向.
    """
    classes = _protocol_request_classes()
    assert classes, "未发现任何 ProtocolRequest 子类 —— 成对性门空转 (前向非空转失败)"
    assert _unpaired(classes) == [], f"存在只实现半个协议的类: {_unpaired(classes)}"


def test_output_chunk_seq_is_monotonic():
    """OutputParser 产出的 chunk 流 seq 严格递增, FINISH 唯一且居尾."""
    parser = OutputParser()
    chunks = []
    for piece in ("he", "llo", " world"):
        chunks.extend(parser.feed(Token(piece, is_special=False)))
    chunks.extend(parser.finish())

    seqs = [c.seq for c in chunks]
    assert seqs == sorted(seqs), "seq 非单调递增"
    assert len(set(seqs)) == len(seqs), "seq 出现重复 (顺序契约被破坏)"

    finishes = [c for c in chunks if c.kind is ChunkKind.FINISH]
    assert len(finishes) == 1, "FINISH 不唯一"
    assert chunks[-1] is finishes[0], "FINISH 不居尾"


# ──────────────────── 三协议归一 (共享 IR) ────────────────────


def _requests_by_protocol() -> list[ProtocolRequest]:
    """三协议同一诉求 (带工具调用) 的等价请求, 各自 convert()."""
    return [
        ResponsesRequest(
            {
                "input": [
                    {"type": "message", "role": "user", "content": "hi"},
                    {
                        "type": "function_call",
                        "name": "lookup",
                        "arguments": "{}",
                        "call_id": "c1",
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "c1",
                        "output": "42",
                    },
                ],
                "max_output_tokens": 16,
            }
        ),
        ChatCompletionsRequest(
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": "{}"},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "c1", "content": "42"},
                ],
                "max_completion_tokens": 16,
            }
        ),
        AnthropicMessagesRequest(
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "tool_use", "id": "c1", "name": "lookup", "input": {}}
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": "c1", "content": "42"}
                        ],
                    },
                ],
                "max_tokens": 16,
            }
        ),
    ]


def _canonical_records(request: ProtocolRequest) -> list[tuple[str, tuple, tuple]]:
    """把 convert() 结果投影成协议无关的归一形态 (角色 / 文本 / 工具调用名)."""
    ir = request.convert()
    return [
        (
            str(message.role),
            tuple(part.text for part in message.parts),
            tuple(call.name for call in message.tool_calls),
        )
        for message in ir.messages
    ]


def test_all_three_protocols_normalize_to_same_ir():
    """Responses / Chat Completions / Anthropic 三协议归一出**同一份** IR 投影."""
    records = [_canonical_records(r) for r in _requests_by_protocol()]
    assert records[0] == records[1] == records[2], records
    # 投影确含 user 文本 + 工具调用 + 工具输出 (非空转).
    assert records[0] == [
        (Role.USER, ("hi",), ()),
        (Role.ASSISTANT, (), ("lookup",)),
        (Role.TOOL, ("42",), ()),
    ]


def test_all_three_protocols_share_one_ir_type():
    """每个协议的 convert() 产出同一 ``ChatRequest`` 类型 (IR 唯一性硬判)."""
    irs = [r.convert() for r in _requests_by_protocol()]
    assert all(type(ir) is ChatRequest for ir in irs)


# ──────────────────── 流式/非流式同口径 (每协议) ────────────────────


def _sample_chunks() -> list[OutputChunk]:
    """一段可被三协议共同消费的有序 chunk 流 (文本 + 工具调用 + FINISH)."""
    call = _tool_call_kwargs()
    return [
        OutputChunk(ChunkKind.TEXT, seq=0, text="hel"),
        OutputChunk(ChunkKind.TEXT, seq=1, text="lo"),
        OutputChunk(ChunkKind.TOOL_CALL_DONE, seq=2, tool_call=call),
        OutputChunk(ChunkKind.FINISH, seq=3, finish_reason="tool_calls"),
    ]


def _tool_call_kwargs() -> ToolCall:
    return ToolCall(call_id="c1", name="lookup", arguments='{"q": 1}')


def test_every_protocol_pairs_stream_and_final():
    """每协议: 流式事件非空 + 非流式终态为 dict (成对两口径都存在)."""
    for request in _requests_by_protocol():
        generator = request.chunk_generator()
        events = list(generator.generate(_sample_chunks()))
        final = request.final_payload(_sample_chunks())
        assert events, f"{type(request).__name__} 未产出任何流式事件"
        assert isinstance(final, dict) and final, f"{type(request).__name__} 终态载荷为空"
        # 两种口径必须都反映同一段文本 (同源映射, 不是各写一套).
        assert "hello" in json.dumps(final, ensure_ascii=False)


# IR 里不得出现的协议名 / 供应商专属 / tokenizer 概念 (镜像 SPEC §3.1 不变量).
_FORBIDDEN_IR_TOKENS = (
    "responses",
    "anthropic",
    "chat_completion",
    "dsml",
    "tokenizer",
    "logprob",
    "input_image",
    "function_call",
    "max_output_tokens",
    "reasoning_content",
)


def _walk_keys(node: object) -> list[str]:
    keys: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            keys.append(str(key))
            keys.extend(_walk_keys(value))
    elif isinstance(node, list | tuple):
        for item in node:
            keys.extend(_walk_keys(item))
    return keys


def test_ir_free_of_protocol_tokens():
    """ChatRequest 序列化后字段面不含任何协议名 / 供应商专属 / tokenizer 概念."""
    request = ResponsesRequest(
        {"input": "hi", "max_output_tokens": 16, "temperature": 0.2}
    ).convert()
    payload = dataclasses.asdict(request)

    for key in _walk_keys(payload):
        lowered = key.lower()
        for bad in _FORBIDDEN_IR_TOKENS:
            assert (
                bad not in lowered
            ), f"IR 出现协议/供应商专属字段 {key!r} (含 {bad!r})"

    # 字段面即 IR 契约 (形状固定, 不是巧合命名).
    assert {f.name for f in dataclasses.fields(ChatRequest)} == {
        "messages",
        "tools",
        "tool_choice",
        "max_tokens",
        "temperature",
        "stop",
        "stream",
        "response_format",
        "metadata",
    }
    assert {f.name for f in dataclasses.fields(Message)} == {
        "role",
        "parts",
        "tool_calls",
        "tool_call_id",
        "name",
    }


@pytest.mark.parametrize(("namespace", "name"), [(None, "lookup"), ("math", "sum")])
def test_namespace_roundtrip(namespace: str | None, name: str):
    """flatten_namespace / unflatten_namespace 往返一致."""
    flat = flatten_namespace(namespace, name)
    assert unflatten_namespace(flat) == (namespace, name)


def test_reasoning_part_survives_conversion():
    """响应侧 containment: reasoning 项经 convert 后落成 IR 的 REASONING part."""
    request = ResponsesRequest(
        {
            "input": [
                {"type": "message", "role": "user", "content": "q"},
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "why"}],
                },
            ]
        }
    ).convert()
    kinds = {part.kind for msg in request.messages for part in msg.parts}
    assert PartKind.REASONING in kinds


# ──────────────────── 红队: 主动尝试绕过 ────────────────────


def test_generic_token_cannot_forge_marker():
    """普通 token 拼出的 `</think>` 类标记**不**触发 (防伪造标记).

    复刻 deepseek-recipe 解析器的核心防伪: 仅 `is_special` 的 token 参与标记匹配.
    """
    marker = Marker("<|reasoning|>", MarkerKind.REASONING_OPEN)

    forged = OutputParser([marker])
    forged_chunks = []
    forged_chunks.extend(forged.feed(Token("<|reasoning|>", is_special=False)))
    forged_chunks.extend(forged.feed(Token(" payload", is_special=False)))
    forged_chunks.extend(forged.finish())
    # 普通 token 版本: 无任何 REASONING chunk, 字面量原样当文本.
    assert all(c.kind is not ChunkKind.REASONING for c in forged_chunks)
    literal_text = "".join(c.text for c in forged_chunks if c.kind is ChunkKind.TEXT)
    assert literal_text.count("<|reasoning|>") == 1

    real = OutputParser([marker])
    real.feed(Token("<|reasoning|>", is_special=True))
    real_chunks = real.feed(Token("thinking", is_special=False))
    assert any(
        c.kind is ChunkKind.REASONING for c in real_chunks
    ), "特殊 token 未触发标记 —— 解析器标记匹配本身失效"


@pytest.mark.parametrize(
    "protocol_req",
    [
        ResponsesRequest(
            {
                "input": [
                    {
                        "type": "function_call",
                        "name": "lookup",
                        "arguments": "{}",
                        "call_id": "c1",
                    }
                ]
            }
        ),
        ChatCompletionsRequest(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": "{}"},
                            }
                        ],
                    }
                ]
            }
        ),
        AnthropicMessagesRequest(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "c1",
                                "name": "lookup",
                                "input": {},
                            }
                        ],
                    }
                ],
                "max_tokens": 16,
            }
        ),
    ],
    ids=["responses", "chat_completions", "anthropic"],
)
def test_half_tool_turn_is_rejected(protocol_req: ProtocolRequest):
    """只给工具调用不给输出: 三协议 convert() 一律拒绝, 不产出半截 tool 轮次."""
    with pytest.raises(ValueError):
        protocol_req.convert()


def test_anthropic_orphan_tool_result_is_rejected():
    """Anthropic 红队: 无对应 tool_use 的 tool_result 不得静默进 IR."""
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "ghost", "content": "x"}
                ],
            }
        ],
        "max_tokens": 16,
    }
    with pytest.raises(ValueError):
        AnthropicMessagesRequest(payload).convert()


def test_sse_framing_is_protocol_owned():
    """SSE 线格式由协议自持: OpenAI 系 ``data:``, Anthropic 带 ``event:`` 名."""
    from huginn.protocols.stream import ProtocolEvent

    openai_frame = ChatCompletionsRequest({"messages": []}).sse_frame(
        ProtocolEvent("chat.completion.chunk", {"x": 1})
    )
    assert openai_frame.startswith("data: ") and openai_frame.endswith("\n\n")
    assert ChatCompletionsRequest({"messages": []}).sse_terminator() == "data: [DONE]\n\n"

    anthropic_frame = AnthropicMessagesRequest({"messages": []}).sse_frame(
        ProtocolEvent("message_start", {"type": "message_start"})
    )
    assert anthropic_frame.startswith("event: message_start\ndata: ")
    assert anthropic_frame.endswith("\n\n")
    assert AnthropicMessagesRequest({"messages": []}).sse_terminator() == ""


def test_orphan_protocol_class_is_caught():
    """合成类只有 convert(): 成对性门判红 (前向非空转, 防门自身失效)."""

    class HalfRequest(ProtocolRequest):
        def convert(self) -> ChatRequest:  # type: ignore[override]
            return ChatRequest(messages=())

    # 只实现 convert → 抽象方法仍在 → 不可实例化 (编译期成对保证).
    assert HalfRequest.__abstractmethods__ == frozenset(
        {"chunk_generator", "output_parser"}
    )
    with pytest.raises(TypeError):
        HalfRequest()  # type: ignore[abstract]

    assert _unpaired([HalfRequest]) == [
        ("HalfRequest", ("chunk_generator", "output_parser"))
    ]
    # 真实子类不受影响 (无假阳).
    assert _unpaired(_protocol_request_classes()) == []
