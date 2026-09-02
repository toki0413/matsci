"""超长用户粘贴文本自动转临时文件 (paste-to-file).

对齐 Claude Code 的 paste-to-file: 用户直接粘贴的超长纯文本不再内联进
prompt, 而是落盘到 tool_artifacts/ 临时文件, 只把一段简短的"已保存"提示
送往模型, 避免撑爆上下文或被 limits.py 的请求体上限拒绝.

开关与阈值均可配置 (env):
- HUGINN_PASTE_OFFLOAD          : 总开关. 默认 "1" (开启); "0"/"false"/"off" 关闭.
- HUGINN_PASTE_OFFLOAD_THRESHOLD: 触发落盘的字符数阈值. 默认 20000.

复用 compress.py 的 offload 目录逻辑 (_offload_dir), 不重复造目录.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path

# 复用 compress.py 的 offload 目录逻辑 (_offload_dir), 不重复造目录.
from huginn.tools.compress import _offload_dir

logger = logging.getLogger(__name__)

# 默认触发落盘的字符数阈值.
_DEFAULT_THRESHOLD_CHARS = 20000


def _enabled() -> bool:
    """读取 HUGINN_PASTE_OFFLOAD env, 默认开启."""
    return os.environ.get("HUGINN_PASTE_OFFLOAD", "1").strip().lower() not in (
        "0",
        "false",
        "off",
        "no",
    )


def _threshold_chars() -> int:
    """读取 HUGINN_PASTE_OFFLOAD_THRESHOLD env, 非法值回退默认."""
    raw = os.environ.get("HUGINN_PASTE_OFFLOAD_THRESHOLD", "").strip()
    try:
        return max(1, int(raw)) if raw else _DEFAULT_THRESHOLD_CHARS
    except ValueError:
        return _DEFAULT_THRESHOLD_CHARS


def _write_paste(text: str, basename: str = "pasted") -> Path:
    """把超长文本写入 tool_artifacts/ 临时文件, 返回路径."""
    ts = int(time.time())
    short_id = uuid.uuid4().hex[:8]
    artifact = _offload_dir() / f"{ts}-{basename}-{short_id}.txt"
    artifact.write_text(text, encoding="utf-8")
    return artifact


def maybe_offload_pasted_text(
    text: str,
    threshold_chars: int | None = None,
    enabled: bool | None = None,
    basename: str = "pasted",
) -> tuple[str, str | None]:
    """把超长用户粘贴文本落盘, 返回 (替换后的消息, 落盘路径或 None).

    - 关闭开关或未超阈值: 原样返回 ``(text, None)``, 不改变原有行为.
    - 超过阈值: 写入 ``tool_artifacts/`` 临时 .txt, 消息体替换成一段
      "已保存"提示 (含路径与字符数), 需要全文时由 ``file_read_tool`` 读取,
      文件不再内联进 prompt.

    ``threshold_chars`` / ``enabled`` 用于直接传参覆盖 env; 不传则读 env.
    """
    if not isinstance(text, str) or not text:
        return text, None

    threshold = _threshold_chars() if threshold_chars is None else threshold_chars
    if enabled is not None:
        if enabled is False:
            return text, None
    elif not _enabled():
        return text, None

    if len(text) <= threshold:
        return text, None

    artifact = _write_paste(text, basename=basename)
    preview = (
        f"[已保存超长文本到 {artifact}（{len(text)}字符）；"
        f"如需全文用 file_read_tool 读取]"
    )
    logger.info("offloaded long pasted text (%d chars) to %s", len(text), artifact)
    return preview, str(artifact)


__all__ = ["maybe_offload_pasted_text"]
