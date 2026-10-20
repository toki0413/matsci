"""多协议适配层契约门 (红蓝双轨).

蓝队断言控制**生效**; 红队主动尝试**绕过**. 沿用从 awesome-cyber-security-university
提炼的"成对"原则: 每协议必须同时给请求方向 (convert) 与响应方向 (chunk_generator),
少一半即红; 解析状态机只在特殊 token 上匹配标记, 普通 token 拼出的标记字面量不可伪造.

不引新测试框架; 纯 assert 型自检.
"""

from __future__ import annotations

import dataclasses
import importlib
import pkgutil

import pytest

from huginn import protocols
from huginn.protocols.ir import ChatRequest, Message, PartKind
from huginn.protocols.openai_responses import ResponsesRequest
from huginn.protocols.primitives import flatten_namespace, unflatten_namespace
from huginn.protocols.stream import (
    ChunkKind,
    Marker,
    MarkerKind,
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
    """偏半个协议的类: 缺 convert 或 chunk_generator (含仍为抽象方法的情形)."""
    bad: list[tuple[str, tuple[str, ...]]] = []
    for cls in classes:
        missing = [
            name
            for name in ("convert", "chunk_generator")
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
    elif isinstance(node, (list, tuple)):
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


def test_half_tool_turn_is_rejected():
    """只给工具调用不给输出: convert() 拒绝, 而非产出半截 tool 轮次."""
    payload = {
        "input": [
            {
                "type": "function_call",
                "name": "lookup",
                "arguments": "{}",
                "call_id": "c1",
            }
        ]
    }
    with pytest.raises(ValueError):
        ResponsesRequest(payload).convert()


def test_orphan_protocol_class_is_caught():
    """合成类只有 convert(): 成对性门判红 (前向非空转, 防门自身失效)."""

    class HalfRequest(ProtocolRequest):
        def convert(self) -> ChatRequest:  # type: ignore[override]
            return ChatRequest(messages=())

    # 只实现 convert → 抽象方法仍在 → 不可实例化 (编译期成对保证).
    assert HalfRequest.__abstractmethods__ == frozenset({"chunk_generator"})
    with pytest.raises(TypeError):
        HalfRequest()  # type: ignore[abstract]

    assert _unpaired([HalfRequest]) == [("HalfRequest", ("chunk_generator",))]
    # 真实子类不受影响 (无假阳).
    assert _unpaired([ResponsesRequest]) == []
