"""索引 + 主题文件两层结构 — 索引按主题文件汇总，主题文件按类型分目录存放。

目录布局示例::

    <memory_dir>/
        user/
            workflow.md
            preferences.md
        project/
            tio2_defect.md
        reference/
            vsp_pseudopotentials.md
        calculation/
            si_band_gap.md
        INDEX.md   # 由 build_memory_index 生成

索引文件本身也会经过 :func:`truncate_entrypoint` 截断，避免无限增长。
"""

from __future__ import annotations

from pathlib import Path

from huginn.memory.truncation import truncate_entrypoint
from huginn.memory.types import MemoryType


def build_memory_index(topic_files: list[Path], memory_dir: Path) -> str:
    """从主题文件构建索引，带行/字节限制。

    每个主题文件的第一行（通常是 ``# 标题``）会被提取为索引条目的显示文本，
    链接指向相对 ``memory_dir`` 的路径。最终输出会通过 :func:`truncate_entrypoint`
    做双重截断保护。
    """
    lines = ["# Memory Index", ""]
    for f in sorted(topic_files):
        rel = f.relative_to(memory_dir)
        first_line = f.read_text(encoding="utf-8").split("\n")[0].lstrip("# ")
        lines.append(f"- [{first_line}]({rel})")
    content = "\n".join(lines)
    truncated, _, _ = truncate_entrypoint(content)
    return truncated


def get_topic_file_path(
    memory_type: MemoryType, topic: str, memory_dir: Path
) -> Path:
    """根据记忆类型和主题获取文件路径。

    主题字符串会被清洗为文件安全名：保留字母数字、连字符、下划线，
    其余字符统一替换为 ``_``。类型子目录会按需创建。
    """
    type_dir = memory_dir / memory_type.value
    type_dir.mkdir(parents=True, exist_ok=True)
    # 清理文件名：保留字母数字与 - _，其他字符替换为 _
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in topic)
    return type_dir / f"{safe_name}.md"


__all__ = ["build_memory_index", "get_topic_file_path"]
