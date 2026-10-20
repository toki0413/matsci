"""多协议适配层 (缺口①).

对外协议 (OpenAI Responses / Chat Completions, Anthropic Messages) 各自
``convert()`` 到一份扁平共享 IR, 后端只认 IR; 响应方向每个协议成对实现
``ChunkGenerator``. 形状对齐 deepseek-recipe, 只抄形状, 不抄 DeepSeek 专属实现.

- ``ir``: 扁平共享 IR (唯一对话中间表示);
- ``primitives``: 跨协议共享校验原语;
- ``stream``: ``OutputChunk`` 顺序契约 + 两个成对 ABC + ``OutputParser`` KMP 状态机;
- ``openai_responses`` / ``chat_completions`` / ``anthropic_messages``:
  三协议适配器 (convert + chunk_generator 成对, 共享 IR / parser / 标记集).

契约由 ``tests/test_protocol_contract.py`` 红蓝双轨硬判.
"""

from __future__ import annotations

from huginn.protocols import ir, primitives, stream
from huginn.protocols.anthropic_messages import (
    AnthropicMessagesChunkGenerator,
    AnthropicMessagesRequest,
)
from huginn.protocols.chat_completions import (
    ChatCompletionsChunkGenerator,
    ChatCompletionsRequest,
)
from huginn.protocols.ir import (
    ChatRequest,
    ContentPart,
    ImageRef,
    Message,
    PartKind,
    Role,
    ToolCall,
    ToolDef,
    ToolKind,
    ToolResult,
)
from huginn.protocols.openai_responses import (
    RESPONSES_MARKERS,
    ResponsesChunkGenerator,
    ResponsesRequest,
)
from huginn.protocols.stream import (
    ChunkGenerator,
    ChunkKind,
    Marker,
    MarkerKind,
    OutputChunk,
    OutputParser,
    ProtocolEvent,
    ProtocolRequest,
    Token,
)

__all__ = [
    "RESPONSES_MARKERS",
    "AnthropicMessagesChunkGenerator",
    "AnthropicMessagesRequest",
    "ChatCompletionsChunkGenerator",
    "ChatCompletionsRequest",
    "ChatRequest",
    "ChunkGenerator",
    "ChunkKind",
    "ContentPart",
    "ImageRef",
    "Marker",
    "MarkerKind",
    "Message",
    "OutputChunk",
    "OutputParser",
    "PartKind",
    "ProtocolEvent",
    "ProtocolRequest",
    "Role",
    "ResponsesChunkGenerator",
    "ResponsesRequest",
    "Token",
    "ToolCall",
    "ToolDef",
    "ToolKind",
    "ToolResult",
    "ir",
    "primitives",
    "stream",
]
