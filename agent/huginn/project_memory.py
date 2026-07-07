"""项目记忆加载器 — 读取工作区 AGENTS.md 作为项目契约注入系统提示词。

与 project_context.py 互补: project_context 优先读 .huginn.md, 回退 AGENTS.md;
本模块专门负责 AGENTS.md 契约(cc-haha 风格的项目记忆), 独立缓存避免互相污染。
"""

from __future__ import annotations

import functools
from pathlib import Path
import logging
logger = logging.getLogger(__name__)


AGENTS_MD_FILENAME = "AGENTS.md"

# 防止把超大 AGENTS.md 灌进系统提示词, 64KB 足够覆盖绝大多数项目契约
_MAX_AGENTS_MD_BYTES = 64 * 1024


@functools.lru_cache(maxsize=8)
def load_agents_md(workspace: str | Path) -> str | None:
    """读取工作区根目录的 AGENTS.md 内容。

    - 文件不存在或读取失败返回 None, 调用方按需跳过
    - lru_cache 缓存, 同一 workspace 只读一次磁盘
    - 超过 _MAX_AGENTS_MD_BYTES 截断, 避免上下文膨胀
    """
    path = Path(workspace).expanduser().resolve() / AGENTS_MD_FILENAME
    if not path.exists() or not path.is_file():
        return None
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if len(raw.encode("utf-8")) > _MAX_AGENTS_MD_BYTES:
        # 截到字节边界附近, 留个尾巴提示被截断了
        raw = raw[: _MAX_AGENTS_MD_BYTES // 3] + "\n\n... (AGENTS.md truncated)"
    return raw.strip() or None


@functools.lru_cache(maxsize=8)
def load_project_context(workspace: str | Path) -> dict[str, str]:
    """返回项目记忆上下文 dict, 供系统提示词拼装使用。

    缺失字段不会出现在结果里, 调用方按需取用。
    """
    ctx: dict[str, str] = {}
    agents_md = load_agents_md(workspace)
    if agents_md:
        ctx["agents_md"] = agents_md
    # project_name 取 workspace 目录名, 算是个轻量的项目标识
    try:
        project_name = Path(workspace).resolve().name
        if project_name:
            ctx["project_name"] = project_name
    except Exception:
        logger.debug("Path failed", exc_info=True)
    return ctx


__all__ = [
    "load_agents_md",
    "load_project_context",
]
