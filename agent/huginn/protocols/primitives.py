"""跨协议共享校验原语 (协议层唯一实现点).

三协议里同语义的重复逻辑收口在此: 工具调用 id 确定化 / 参数 schema 浅校验 /
图像归一 / 命名空间工具名双向映射. 形状对齐 deepseek-recipe 的
``validation.rs`` / ``tool_call_id.rs`` / ``json_schema.rs`` / ``image.rs``.

另含三协议共用的**响应控制标记**与工具调用区间语法: 标记字面量是 huginn 自定义
控制 token (非 DeepSeek DSML), 由适配器整体覆盖.
"""

from __future__ import annotations

import hashlib
import json

from huginn.protocols.ir import ImageRef, ToolCall
from huginn.protocols.stream import Marker, MarkerKind

__all__ = [
    "CONTROL_MARKERS",
    "flatten_namespace",
    "normalize_image",
    "parse_tool_call_interval",
    "resolve_tool_call_id",
    "unflatten_namespace",
    "validate_json_schema",
]

_NS_SEP = "::"

CONTROL_MARKERS: tuple[Marker, ...] = (
    Marker("<|reasoning|>", MarkerKind.REASONING_OPEN),
    Marker("<|/reasoning|>", MarkerKind.REASONING_CLOSE),
    Marker("<|tool_calls|>", MarkerKind.TOOL_CALLS_OPEN),
    Marker("<|/tool_calls|>", MarkerKind.TOOL_CALLS_CLOSE),
)


def parse_tool_call_interval(raw: str) -> ToolCall:
    """工具调用区间文本 -> ToolCall.

    区间语法 (三协议共用): JSON 对象 ``{"name": .., "arguments": <obj|str>,
    "call_id": ..?, "namespace": ..?}``.
    """
    payload = json.loads(raw) if raw.strip() else {}
    if not isinstance(payload, dict):
        raise ValueError("tool call 区间必须是 JSON 对象")
    arguments = payload.get("arguments", "")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    seed = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return ToolCall(
        call_id=resolve_tool_call_id(payload.get("call_id"), seed=seed),
        name=flatten_namespace(payload.get("namespace"), str(payload.get("name", ""))),
        arguments=arguments,
        namespace=payload.get("namespace"),
    )


def resolve_tool_call_id(raw: object, *, seed: str = "") -> str:
    """工具调用 id 确定化.

    Responses 允许省略 ``call_id``; 归一后后端只见唯一确定形式, 同 seed 必得同 id
    (可复现). 显式给值时原样保留 (仅去首尾空白).
    """
    text = str(raw).strip() if raw is not None else ""
    if text:
        return text
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    return f"call_{digest}"


def validate_json_schema(schema: object) -> dict:
    """工具参数 JSON Schema 的浅校验 + 规范化 (补齐 object 型缺省的 properties)."""
    if not isinstance(schema, dict):
        raise ValueError(
            f"tool parameters 必须是 JSON Schema 对象, 实际是 {type(schema).__name__}"
        )
    if schema.get("type") == "object" and "properties" not in schema:
        return {**schema, "properties": {}}
    return dict(schema)


def normalize_image(part: object) -> ImageRef:
    """图像输入归一: dict(url / image_url / data / base64 / Anthropic source) 或 ImageRef -> ImageRef."""
    if isinstance(part, ImageRef):
        return part
    if not isinstance(part, dict):
        raise ValueError("image part 必须是映射 (dict) 或 ImageRef")
    url_obj = part.get("image_url", part.get("url"))
    if isinstance(url_obj, dict):
        url_obj = url_obj.get("url")
    data = part.get("data") or part.get("base64")
    # Anthropic Messages: {"type": "image", "source": {"type": ..., ...}}.
    source = part.get("source")
    if isinstance(source, dict):
        if source.get("type") == "url":
            url_obj = source.get("url")
        elif source.get("type") == "base64":
            data = source.get("data")
        else:
            raise ValueError(f"不支持的 image source 类型: {source.get('type')!r}")
    if not url_obj and not data:
        raise ValueError("image part 需要 url 或 base64 data 之一")
    return ImageRef(url=url_obj, data=data, detail=str(part.get("detail", "auto")))


def flatten_namespace(namespace: str | None, name: str) -> str:
    """命名空间工具名扁平化: ``(ns, name) -> "ns::name"``; 无 ns 时原样返回."""
    ns = (namespace or "").strip()
    return f"{ns}{_NS_SEP}{name}" if ns else name


def unflatten_namespace(flat: str) -> tuple[str | None, str]:
    """扁平名反向映射: ``"ns::name" -> (ns, name)``; 无分隔符时 ``(None, flat)``."""
    ns, sep, name = flat.partition(_NS_SEP)
    return (ns, name) if sep else (None, flat)
