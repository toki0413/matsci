"""落地锚 — 知识可验证来源的结构化表示.

迁移与"落地"的单位不是文本相似度, 而是锚: 同一锚 (同一 run / 工具会话 / 蒸馏
条目 / 实测值 / 引用) 上的知识跨场景可复用 —— 换一个领域时, 只要锚相同, 旧知识
就能挂到新问题上被复查. 本模块只做**结构化**识别: 从记忆已有的 ``source`` /
``run_id`` 字段解析, 不对自由文本 ``content`` 做语义猜测.

诚实边界: 猜出来的锚是假信号 —— 会像 "structure_desc 全 0" 一样看着有机制、
实际无信息. 故只认白名单前缀 (现网 ``source`` 语法), 自由文本一律判定"无锚"
(宁缺勿猜, 覆盖率低是一个**可观测的结论**, 不是要粉饰的缺陷).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

# ``source`` 前缀 → 锚类型. 与现网写入语法对齐:
#   memory/manager.py:      "session:<sid>/tool:<name>"
#   evolution/knowledge_distiller.py: "distiller:<kid>"
# 其余 (run:/exec:/measure:/cite:/doi:) 为约定语法, 生产端按需写入.
_SOURCE_PREFIX_KINDS: tuple[tuple[str, str], ...] = (
    ("run:", "run"),
    ("session:", "session"),
    ("tool:", "tool"),
    ("distiller:", "distiller"),
    ("exec:", "exec"),
    ("measure:", "measure"),
    ("cite:", "cite"),
    ("doi:", "cite"),
)

ANCHOR_KINDS: tuple[str, ...] = (
    "run",
    "session",
    "tool",
    "distiller",
    "exec",
    "measure",
    "cite",
)


@dataclass(frozen=True)
class Anchor:
    """一个可回查的验证锚: ``kind`` 是来源类别, ``ref`` 是可定位的串."""

    kind: str
    ref: str

    def token(self) -> str:
        return f"{self.kind}:{self.ref}"


def parse_anchor(entry: Mapping[str, Any]) -> Anchor | None:
    """从记忆条目的结构化字段解析锚; 自由文本不猜, 无锚返回 ``None``."""
    src = str(entry.get("source") or "").strip()
    for prefix, kind in _SOURCE_PREFIX_KINDS:
        if src.startswith(prefix) and src[len(prefix):].strip():
            # ref 保留完整 source —— "session:x/tool:y" 的次级信息对回查有用.
            return Anchor(kind=kind, ref=src)
    run_id = str(entry.get("run_id") or "").strip()
    if run_id:
        return Anchor(kind="run", ref=run_id)
    return None


def has_anchor(entry: Mapping[str, Any]) -> bool:
    """该条目是否带可验证锚."""
    return parse_anchor(entry) is not None


def anchored_ratio(entries: Iterable[Mapping[str, Any]]) -> float:
    """锚覆盖率 (0-1). 空集 → 0.0 (无信号, 不臆造)."""
    rows = list(entries)
    if not rows:
        return 0.0
    return sum(1 for e in rows if has_anchor(e)) / len(rows)
