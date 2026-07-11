"""富字符串类型 — 受 Scrapling TextHandler 启发, 简化版.

给字符串加 .re() / .re_first() / .clean() / .json() 方法,
链式提取时不用反复写 re.findall / json.loads.

设计原则: 不继承 str (避免 lxml 那些坑), 只是一个轻量 wrapper.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterator


class TextHandler:
    """字符串 wrapper, 提供 regex / clean / json 链式操作.

    用法:
        text = TextHandler("  Hello World 123  ")
        text.clean()                    # -> "Hello World 123"
        text.re(r"\\d+")                # -> ["123"]
        text.re_first(r"\\w+")          # -> "Hello"
    """

    __slots__ = ("_text",)

    def __init__(self, text: str = "") -> None:
        self._text = text if isinstance(text, str) else str(text)

    # ── 基本操作 ──────────────────────────────────────

    def __str__(self) -> str:
        return self._text

    def __repr__(self) -> str:
        return f"TextHandler({self._text!r})"

    def __len__(self) -> int:
        return len(self._text)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TextHandler):
            return self._text == other._text
        if isinstance(other, str):
            return self._text == other
        return False

    def __hash__(self) -> int:
        return hash(self._text)

    def __getitem__(self, key: int | slice) -> TextHandler:
        return TextHandler(self._text[key])

    def __iter__(self) -> Iterator[str]:
        return iter(self._text)

    def __contains__(self, item: str) -> bool:
        return item in self._text

    # ── 字符串方法代理 ────────────────────────────────

    @property
    def raw(self) -> str:
        return self._text

    def strip(self, chars: str | None = None) -> TextHandler:
        return TextHandler(self._text.strip(chars))

    def lower(self) -> TextHandler:
        return TextHandler(self._text.lower())

    def upper(self) -> TextHandler:
        return TextHandler(self._text.upper())

    def replace(self, old: str, new: str, count: int = -1) -> TextHandler:
        return TextHandler(self._text.replace(old, new, count))

    def split(self, sep: str | None = None, maxsplit: int = -1) -> list[TextHandler]:
        return [TextHandler(s) for s in self._text.split(sep, maxsplit)]

    # ── 核心增强 ──────────────────────────────────────

    def clean(self) -> TextHandler:
        """去除多余空白: tab/newline 转空格, 连续空格合并, 首尾 strip."""
        cleaned = re.sub(r"\s+", " ", self._text).strip()
        return TextHandler(cleaned)

    def re(
        self,
        pattern: str | re.Pattern,
        case_sensitive: bool = True,
    ) -> list[TextHandler]:
        """正则提取所有匹配, 返回 TextHandler 列表."""
        flags = 0 if case_sensitive else re.IGNORECASE
        if isinstance(pattern, str):
            compiled = re.compile(pattern, flags)
        else:
            compiled = pattern
        matches = compiled.findall(self._text)
        # findall 在有 group 时返回 tuple, 展平
        result: list[TextHandler] = []
        for m in matches:
            if isinstance(m, tuple):
                for g in m:
                    if g:
                        result.append(TextHandler(g))
            elif m:
                result.append(TextHandler(m))
        return result

    def re_first(
        self,
        pattern: str | re.Pattern,
        default: str | None = None,
        case_sensitive: bool = True,
    ) -> TextHandler | None:
        """正则提取第一个匹配, 没有返回 default."""
        results = self.re(pattern, case_sensitive=case_sensitive)
        return results[0] if results else (TextHandler(default) if default else None)

    def re_match(self, pattern: str | re.Pattern, case_sensitive: bool = True) -> bool:
        """检查是否匹配, 不提取内容."""
        flags = 0 if case_sensitive else re.IGNORECASE
        if isinstance(pattern, str):
            return bool(re.search(pattern, self._text, flags))
        return bool(pattern.search(self._text))

    def json(self) -> Any:
        """解析 JSON, 失败抛 ValueError."""
        try:
            return json.loads(self._text)
        except json.JSONDecodeError as e:
            # 尝试从 { 或 [ 开始截取
            for start_char in ("{", "["):
                start = self._text.find(start_char)
                if start != -1:
                    end_char = "}" if start_char == "{" else "]"
                    end = self._text.rfind(end_char)
                    if end > start:
                        try:
                            return json.loads(self._text[start : end + 1])
                        except json.JSONDecodeError:
                            continue
            raise ValueError(f"Invalid JSON: {e}") from e

    def truncate(self, max_len: int, suffix: str = "...") -> TextHandler:
        """截断到指定长度, 超出加省略号."""
        if len(self._text) <= max_len:
            return TextHandler(self._text)
        return TextHandler(self._text[: max_len - len(suffix)] + suffix)

    # 兼容 Scrapy/parsel 风格
    def get(self, default: Any = None) -> TextHandler:
        return self

    def getall(self) -> list[TextHandler]:
        return [self]


class AttributesHandler:
    """只读属性映射, 支持 .search_values() 和 .json_string.

    用法:
        attrs = AttributesHandler({"class": "btn primary", "href": "/page"})
        attrs["class"]              # -> "btn primary"
        attrs.get("href", "/")      # -> "/page"
        attrs.search_values("btn")  # -> [{"class": "btn primary"}]
    """

    __slots__ = ("_data",)

    def __init__(self, mapping: dict[str, str] | None = None) -> None:
        self._data: dict[str, TextHandler] = {}
        if mapping:
            for k, v in mapping.items():
                self._data[k] = v if isinstance(v, TextHandler) else TextHandler(str(v))

    def __getitem__(self, key: str) -> TextHandler:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"AttributesHandler({dict(self._data)!r})"

    def get(self, key: str, default: Any = None) -> TextHandler | Any:
        return self._data.get(key, default)

    def search_values(self, keyword: str, partial: bool = True) -> list[dict[str, TextHandler]]:
        """按值搜索属性, 返回匹配的 {key: value} 列表."""
        results: list[dict[str, TextHandler]] = []
        for k, v in self._data.items():
            if partial:
                if keyword in str(v):
                    results.append({k: v})
            else:
                if keyword == str(v):
                    results.append({k: v})
        return results

    @property
    def raw(self) -> dict[str, str]:
        """返回原始 dict (TextHandler -> str)."""
        return {k: str(v) for k, v in self._data.items()}

    def to_dict(self) -> dict[str, str]:
        return self.raw
