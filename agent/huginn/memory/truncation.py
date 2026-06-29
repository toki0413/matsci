"""双重截断保护 — 防止 entrypoint（如 MEMORY.md）无限增长。

Claude Code 的 memdir 在拼装 entrypoint 时同时限制行数和字节数，
我们沿用同一套阈值：200 行 / 25_000 字节。两者任一超限都会触发截断，
并且字节截断会回退到最后一个完整换行，避免产生半截行。
"""

from __future__ import annotations

MAX_ENTRYPOINT_LINES = 200
MAX_ENTRYPOINT_BYTES = 25_000


def truncate_entrypoint(raw: str) -> tuple[str, bool, bool]:
    """按行数和字节数双重截断，保留完整行。

    返回三元组 ``(truncated_content, was_line_truncated, was_byte_truncated)``：

    - ``was_line_truncated``：原始行数超过 ``MAX_ENTRYPOINT_LINES``。
    - ``was_byte_truncated``：原始字节数/字符数超过 ``MAX_ENTRYPOINT_BYTES``。

    截断顺序：先按行数截断，再按字节截断。字节截断会在阈值内回退到最后一个
    换行符，保证末尾是完整一行；如果找不到换行符（极短单行），就硬截断。
    """
    trimmed = raw.strip()
    lines = trimmed.split("\n")
    line_truncated = len(lines) > MAX_ENTRYPOINT_LINES
    byte_truncated = len(trimmed) > MAX_ENTRYPOINT_BYTES

    if line_truncated:
        lines = lines[:MAX_ENTRYPOINT_LINES]
    result = "\n".join(lines)
    if len(result) > MAX_ENTRYPOINT_BYTES:
        # 按字节截断但回退到最后一个完整行
        cut = result[:MAX_ENTRYPOINT_BYTES].rfind("\n")
        result = result[:cut] if cut > 0 else result[:MAX_ENTRYPOINT_BYTES]
    return result, line_truncated, byte_truncated


__all__ = ["MAX_ENTRYPOINT_LINES", "MAX_ENTRYPOINT_BYTES", "truncate_entrypoint"]
