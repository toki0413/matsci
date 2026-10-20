"""协议适配层的扁平共享 IR (唯一对话中间表示).

对齐 deepseek-recipe 的形状: 每个对外协议 (OpenAI Responses / Chat Completions,
Anthropic Messages) 各自 ``convert()`` 到本 IR, 后端只认本 IR, 协议差异在转换边界
被吸收. IR 里**不出现**协议名 / 供应商专属字段 / tokenizer 概念 —— 该不变量由
``tests/test_protocol_contract.py`` 硬判.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

__all__ = [
    "ChatRequest",
    "ContentPart",
    "ImageRef",
    "Message",
    "PartKind",
    "Role",
    "ToolCall",
    "ToolDef",
    "ToolKind",
    "ToolResult",
]


class Role(StrEnum):
    """IR 消息角色 (三协议归一后的最小集)."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class PartKind(StrEnum):
    """内容块类型."""

    TEXT = "text"
    REASONING = "reasoning"
    IMAGE = "image"
    REFUSAL = "refusal"


class ToolKind(StrEnum):
    """工具形态: 结构化函数 vs 自由文本自定义工具."""

    FUNCTION = "function"
    CUSTOM = "custom"


@dataclass(frozen=True)
class ImageRef:
    """图像引用: url 或 base64 data 二选一."""

    url: str | None = None
    data: str | None = None
    detail: str = "auto"


@dataclass(frozen=True)
class ContentPart:
    """一条消息内的单个内容块."""

    kind: PartKind
    text: str = ""
    refusal: str = ""
    image: ImageRef | None = None


@dataclass(frozen=True)
class ToolCall:
    """模型请求的一次工具调用 (arguments 为原始 JSON 字符串)."""

    call_id: str
    name: str
    arguments: str = ""
    namespace: str | None = None


@dataclass(frozen=True)
class ToolResult:
    """一次工具调用的执行结果."""

    call_id: str
    name: str = ""
    output: str = ""


@dataclass(frozen=True)
class Message:
    """IR 消息: 内容块 + (可选) 工具调用 / 工具结果回指."""

    role: Role
    parts: tuple[ContentPart, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None


@dataclass(frozen=True)
class ToolDef:
    """工具声明 (parameters 为 JSON Schema)."""

    name: str
    description: str = ""
    parameters: dict = field(default_factory=dict)
    kind: ToolKind = ToolKind.FUNCTION


@dataclass(frozen=True)
class ChatRequest:
    """后端唯一认得的请求形状 (协议无关)."""

    messages: tuple[Message, ...]
    tools: tuple[ToolDef, ...] = ()
    tool_choice: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    stop: tuple[str, ...] = ()
    stream: bool = False
    response_format: dict | None = None
    metadata: dict = field(default_factory=dict)
