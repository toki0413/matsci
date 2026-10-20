"""流式输出契约 + 每协议成对收口 + 输出解析状态机.

三条契约 (形状对齐 deepseek-recipe, 只抄形状不抄 DeepSeek 专属实现):

1. **``OutputChunk`` 顺序契约** —— 后端到协议事件的所有差异归一到一条有序 chunk
   流; ``OutputParser`` 全局分配单调递增 ``seq``; ``ChunkGenerator`` 只做
   chunk -> 协议事件映射, 不做解析.
2. **``ProtocolRequest`` 成对收口** —— 同一请求类型**同时**实现请求方向
   ``convert()`` 与响应方向 ``chunk_generator()``, 新增协议不可能只加半个.
3. **标记只在特殊 token 上匹配** —— ``OutputParser`` 的 KMP 自动机仅在
   ``Token.is_special`` 的字符流上推进, 普通 token 拼出标记字面量**不**触发
   (防"普通 token 伪造标记"; 见 tests/test_protocol_contract.py 红队用例).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from huginn.protocols.ir import ChatRequest, ToolCall

__all__ = [
    "ChunkGenerator",
    "ChunkKind",
    "Marker",
    "MarkerKind",
    "OutputChunk",
    "OutputParser",
    "ProtocolEvent",
    "ProtocolRequest",
    "Token",
    "ToolCallParser",
]


class ChunkKind(StrEnum):
    """后端输出的一小段之类型."""

    TEXT = "text"
    REASONING = "reasoning"
    TOOL_CALL_DELTA = "tool_call_delta"
    TOOL_CALL_DONE = "tool_call_done"
    FINISH = "finish"


@dataclass(frozen=True)
class OutputChunk:
    """后端输出的一小段, 带全局单调 ``seq`` (顺序契约的承载)."""

    kind: ChunkKind
    seq: int
    text: str = ""
    tool_call: ToolCall | None = None
    finish_reason: str | None = None


@dataclass(frozen=True)
class Token:
    """模型原始 token.

    ``is_special`` 标记该 token 是否为控制 token —— 仅特殊 token 参与标记匹配,
    这是"普通 token 不能伪造标记"的根.
    """

    piece: str
    is_special: bool = False


@dataclass(frozen=True)
class ProtocolEvent:
    """一个协议事件 (事件名 + data), 是 ``ChunkGenerator`` 的唯一产出."""

    event: str
    data: dict = field(default_factory=dict)


class _Mode(StrEnum):
    """解析器内部模式机 (与 ``ChunkKind`` 正交).

    ``TOOL_CALLS`` 只是"当前处于工具调用区间内"的模式, 不是一种 chunk 类型 ——
    区间内文本落成 ``TOOL_CALL_DELTA``, 故不并进 ``ChunkKind``.
    """

    TEXT = "text"
    REASONING = "reasoning"
    TOOL_CALLS = "tool_calls"


class MarkerKind(StrEnum):
    """解析器识别的控制标记语义 (标记字面量由协议适配器提供, 不硬编码)."""

    REASONING_OPEN = "reasoning_open"
    REASONING_CLOSE = "reasoning_close"
    TOOL_CALLS_OPEN = "tool_calls_open"
    TOOL_CALLS_CLOSE = "tool_calls_close"


@dataclass(frozen=True)
class Marker:
    """一条控制标记: 字面量 piece + 语义 kind."""

    piece: str
    kind: MarkerKind


ToolCallParser = Callable[[str], ToolCall]


class ChunkGenerator(ABC):
    """每协议唯一实现点: 只做 ``OutputChunk`` -> 协议事件 映射, 不做解析."""

    @abstractmethod
    def generate(self, chunks: Iterable[OutputChunk]) -> Iterator[ProtocolEvent]: ...


class ProtocolRequest(ABC):
    """每协议唯一实现点: 请求转换 + 响应生成**成对**绑定在同一类型上."""

    @abstractmethod
    def convert(self) -> ChatRequest: ...

    @abstractmethod
    def chunk_generator(self) -> ChunkGenerator: ...


class _MarkerAutomaton:
    """单标记的 KMP 自动机 (``state`` = 已消耗特殊 token 流的最长可匹配前缀长度)."""

    __slots__ = ("marker", "_fail", "state")

    def __init__(self, marker: Marker) -> None:
        self.marker = marker
        self._fail = _failure_table(marker.piece)
        self.state = 0

    def advance(self, ch: str) -> bool:
        """推进一个字符; 完整命中返回 True 并复位."""
        piece = self.marker.piece
        while self.state > 0 and ch != piece[self.state]:
            self.state = self._fail[self.state - 1]
        if ch == piece[self.state]:
            self.state += 1
        if self.state == len(piece):
            self.state = 0
            return True
        return False


def _failure_table(piece: str) -> list[int]:
    """KMP 失败函数."""
    table = [0] * len(piece)
    k = 0
    for i in range(1, len(piece)):
        while k > 0 and piece[i] != piece[k]:
            k = table[k - 1]
        if piece[i] == piece[k]:
            k += 1
        table[i] = k
    return table


class OutputParser:
    """裸 token 流 -> 有序 ``OutputChunk`` (唯一解析点).

    - 模式机 ``TEXT -> REASONING`` 及 ``TEXT <-> TOOL_CALLS``;
    - 标记仅在特殊 token 上匹配 (KMP 自动机), 普通 token 直接作内容;
    - 未完成的标记前缀被暂存, 半个标记不会被当内容吐出;
    - ``finish()`` 补发唯一的 FINISH 并居尾.
    """

    def __init__(
        self,
        markers: Sequence[Marker] = (),
        *,
        tool_call_parser: ToolCallParser | None = None,
    ) -> None:
        self._automata = [_MarkerAutomaton(m) for m in markers]
        self._tool_call_parser = tool_call_parser
        self._mode = _Mode.TEXT
        self._held = ""
        self._tool_buf = ""
        self._seq = 0
        self._finished = False

    def feed(self, token: Token) -> list[OutputChunk]:
        """喂一个 token, 返回本次产出的 chunk (可能为空)."""
        if self._finished:
            raise RuntimeError("OutputParser 已 finish, 不能再 feed")
        if not token.is_special:
            out = self._flush_held()
            out.extend(self._as_content(token.piece))
            return out
        return self._feed_special(token.piece)

    def finish(self, finish_reason: str = "stop") -> list[OutputChunk]:
        """结束: 冲掉暂存的特殊 token 前缀, 补发唯一 FINISH."""
        if self._finished:
            raise RuntimeError("OutputParser 已 finish")
        self._finished = True
        out = self._drain_held()
        out.append(self._chunk(ChunkKind.FINISH, finish_reason=finish_reason))
        return out

    # ── 内部 ────────────────────────────────────────────────────────

    def _feed_special(self, piece: str) -> list[OutputChunk]:
        out: list[OutputChunk] = []
        for ch in piece:
            self._held += ch
            fired = self._advance_all(ch)
            if fired is not None:
                out.extend(self._on_marker(fired))
                self._held = ""
                self._reset_automata()
            else:
                out.extend(self._flush_held())
        return out

    def _advance_all(self, ch: str) -> Marker | None:
        fired: Marker | None = None
        for aut in self._automata:
            if aut.advance(ch) and (
                fired is None or len(aut.marker.piece) > len(fired.piece)
            ):
                fired = aut.marker
        return fired

    def _flush_held(self) -> list[OutputChunk]:
        """把暂存里确定不是标记前缀的部分当内容吐出, 只留最长可匹配后缀."""
        if not self._held:
            return []
        keep = max((aut.state for aut in self._automata), default=0)
        emit = self._held[: len(self._held) - keep]
        self._held = self._held[len(self._held) - keep :]
        return self._as_content(emit)

    def _drain_held(self) -> list[OutputChunk]:
        """finish 时把暂存全部当内容吐出 (不再保留前缀)."""
        text = self._held
        self._held = ""
        self._reset_automata()
        return self._as_content(text)

    def _as_content(self, text: str) -> list[OutputChunk]:
        if not text:
            return []
        if self._mode is _Mode.TOOL_CALLS:
            self._tool_buf += text
            return [self._chunk(ChunkKind.TOOL_CALL_DELTA, text=text)]
        kind = ChunkKind.TEXT if self._mode is _Mode.TEXT else ChunkKind.REASONING
        return [self._chunk(kind, text=text)]

    def _on_marker(self, marker: Marker) -> list[OutputChunk]:
        if marker.kind is MarkerKind.REASONING_OPEN:
            self._mode = _Mode.REASONING
            return []
        if marker.kind is MarkerKind.REASONING_CLOSE:
            self._mode = _Mode.TEXT
            return []
        if marker.kind is MarkerKind.TOOL_CALLS_OPEN:
            self._mode = _Mode.TOOL_CALLS
            self._tool_buf = ""
            return []
        # TOOL_CALLS_CLOSE
        self._mode = _Mode.TEXT
        raw, self._tool_buf = self._tool_buf, ""
        if self._tool_call_parser is None:
            return []
        return [
            self._chunk(ChunkKind.TOOL_CALL_DONE, tool_call=self._tool_call_parser(raw))
        ]

    def _chunk(
        self,
        kind: ChunkKind,
        *,
        text: str = "",
        tool_call: ToolCall | None = None,
        finish_reason: str | None = None,
    ) -> OutputChunk:
        chunk = OutputChunk(
            kind=kind,
            seq=self._seq,
            text=text,
            tool_call=tool_call,
            finish_reason=finish_reason,
        )
        self._seq += 1
        return chunk

    def _reset_automata(self) -> None:
        for aut in self._automata:
            aut.state = 0
