"""模型面表示层 —— 把 system prompt 从字符串升维为可断言的有序段集合.

每个段携带 name 与 source. 拼接只发生在 Surface.render() 一处, 因此
「系统实际告诉了模型什么」成为可枚举、可断言、可度量字符数的对象;
此前各组装根各自 `+=` 字符串, 段名在 join 时丢失, 门只能断言生产者的声明.

本模块零 huginn 依赖 (只 import 标准库), 门的离线部分据此可以跑.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

# 段间分隔符. 与既有主路径的拼接一致, 迁移时保证字节不变.
SEPARATOR = "\n\n"


@dataclass(frozen=True)
class SurfaceSection:
    """模型面的一段. source 是产出来源 (注册表名或模块名), 用于溯源."""

    name: str
    source: str
    text: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("surface section name must be non-empty")
        if not self.source.strip():
            raise ValueError(f"surface section {self.name!r} missing source")


class Surface:
    """有序段集合; render() 是唯一拼接点.

    空段 (text 为空) 不参与拼接, 也不计入字符数 —— 与「该段未生效」同义.
    """

    __slots__ = ("_sections",)

    def __init__(self, sections: Iterable[SurfaceSection]) -> None:
        self._sections = tuple(sections)

    @property
    def sections(self) -> tuple[SurfaceSection, ...]:
        return self._sections

    def names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self._sections)

    def sources(self) -> tuple[str, ...]:
        """产出来源去重保序."""
        seen: set[str] = set()
        out: list[str] = []
        for s in self._sections:
            if s.source not in seen:
                seen.add(s.source)
                out.append(s.source)
        return tuple(out)

    def text_of(self, name: str) -> str:
        """某段文本; 未注册返回空串 (调用方据此判「该段是否生效」)."""
        for s in self._sections:
            if s.name == name:
                return s.text
        return ""

    def render(self) -> str:
        return SEPARATOR.join(s.text for s in self._sections if s.text)

    def char_count(self) -> int:
        """实际发送给模型的字符数 (与 render 同口径)."""
        return len(self.render())


def compose(sections: Iterable[SurfaceSection]) -> Surface:
    """校验后组装. 同名段重复是接线错误, 抛 ValueError 而不静默去重."""
    items = tuple(sections)
    seen: set[str] = set()
    for s in items:
        if s.name in seen:
            raise ValueError(f"duplicate surface section: {s.name!r}")
        seen.add(s.name)
    return Surface(items)


__all__ = ["SEPARATOR", "Surface", "SurfaceSection", "compose"]
