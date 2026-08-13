"""Shared MemoryManager construction.

多入口 (FastAPI server / CLI / RCB harness) 之前各自手写 MemoryManager 装配,
改一个能力要同步多处. 这里提供唯一构造入口, 传参即覆盖差异:

- server_core: 主题记忆走 memory_md_path, 可注入带语义检索的 longterm.
- RCB harness: 主题记忆走 memory_dir, 注入 model 做 insight 抽取.

两端都通过 ``build_memory_manager`` 装配, 消除重复.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from huginn.memory.manager import MemoryManager


def build_memory_manager(
    *,
    memory_dir: str | Path | None = None,
    memory_md_path: str | Path | None = None,
    llm: Any | None = None,
    longterm: Any | None = None,
    enable_semantic_search: bool = True,
) -> Any:
    """Build a MemoryManager from the shared construction path.

    Args:
        memory_dir: 主题记忆目录 (RCB 用). 也可直接传 longterm 覆盖.
        memory_md_path: 主题记忆对应的 MEMORY.md (server 用).
        llm: 可选的 insight 抽取模型 (MemoryManager.set_llm).
        longterm: 可选的已构造 LongTermMemory (如带向量检索的). 传了则优先用.
        enable_semantic_search: 是否启用主题语义检索.

    任何单点装配失败都留空 longterm/llm, 不抛 — memory 是增强, 失败不阻塞主流程.
    """
    from huginn.memory.manager import MemoryConfig, MemoryManager

    config = MemoryConfig(
        enable_semantic_search=enable_semantic_search,
        memory_md_path=Path(memory_md_path) if memory_md_path else None,
        memory_dir=Path(memory_dir) if memory_dir else None,
    )
    return MemoryManager(config=config, longterm=longterm, llm=llm)