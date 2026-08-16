"""Persistent checkpointing for HuginnAgent conversations.

By default LangGraph agents use an in-memory checkpointer, which means all
conversation state is lost when the process restarts. This module provides a
small factory that switches to SQLite persistence when a path is configured.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from huginn.utils.runtime import get_runtime_home

if TYPE_CHECKING:
    from huginn.persistence import CheckpointerBackend

logger = logging.getLogger(__name__)


def _preset_incremental_vacuum(path: str | Path) -> None:
    """P2①: 新建 checkpointer 库前预置 auto_vacuum=INCREMENTAL.

    SQLite 的 ``auto_vacuum`` 在建表/写入后设置对已存在数据无效 (需全量
    VACUUM 迁移). 因此对**全新文件**在建库表结构前预置该 PRAGMA, 让删行后
    释放的页进入 freepages 记账, 后续 ``incremental_vacuum`` 在线增量回收.

    仅对尚不存在的文件生效; 已存在文件 / ``:memory:`` 静默跳过 — 避免对在用
    库执行可能锁表/需要临时磁盘的迁移.
    """
    p = Path(path)
    if p.exists():
        return
    try:
        import sqlite3

        conn = sqlite3.connect(str(p))
        try:
            conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
            conn.commit()
        finally:
            conn.close()
    except Exception:
        logger.warning(
            "preset incremental_vacuum skipped (best-effort)", exc_info=True
        )


def _enable_incremental_vacuum(saver: Any) -> None:
    """P2①: 在线回收 checkpointer 库的 freepages (incremental_vacuum).

    长周期/多线程叠加时, RemoveMessage 只删 row 不回收磁盘页, SQLite 文件会
    被历史空洞撑大 (bench orchestrator C2 的全量 VACUUM 只在运行结束后跑).
    对已预置 ``auto_vacuum=INCREMENTAL`` 的库, ``incremental_vacuum`` 在线
    增量回收, 不需锁表卡 graph.

    无 conn / 空 freepages 等场景静默降级不崩.
    """
    conn = getattr(saver, "conn", None)
    if conn is None:
        conn = getattr(saver, "_conn", None)
    if conn is None:
        logger.debug("checkpointer incremental_vacuum: no sqlite conn")
        return
    try:
        conn.execute("PRAGMA incremental_vacuum")
        logger.debug("checkpointer incremental_vacuum reclaimed freepages")
    except Exception:
        logger.warning(
            "incremental_vacuum failed (best-effort)", exc_info=True
        )


@contextmanager
def persistent_checkpointer(
    path: str | Path | None = None,
) -> Generator[Any, None, None]:
    """Context manager yielding a SQLite-backed SqliteSaver.

    The database connection is closed when the context exits, preventing
    resource leaks in long-running processes.
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    if path is None:
        path = os.environ.get("HUGINN_CHECKPOINTER_PATH")
    if path is None:
        default = get_runtime_home() / "checkpoints.sqlite"
        default.parent.mkdir(parents=True, exist_ok=True)
        path = default

    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    _preset_incremental_vacuum(path)

    with SqliteSaver.from_conn_string(str(path)) as saver:
        _enable_incremental_vacuum(saver)
        yield saver


def create_checkpointer(
    path: str | Path | None = None,
    backend: CheckpointerBackend | None = None,
) -> Any:
    """Create a LangGraph checkpointer.

    * ``backend`` is given -> use the provided persistence backend.
    * ``path`` is given -> SQLite-backed persistent checkpointer.
    * ``path`` is ``":memory:"`` -> SQLite in-memory checkpointer.
    * ``path`` is None -> use ``HUGINN_CHECKPOINTER_PATH`` env var if set,
      otherwise a default SQLite file under ``~/.huginn/checkpoints.sqlite``.

    The returned object is a ``langgraph.checkpoint.sqlite.SqliteSaver``.
    """
    if backend is not None:
        return backend.get()

    if path is None:
        path = os.environ.get("HUGINN_CHECKPOINTER_PATH")
    if path is None:
        default = get_runtime_home() / "checkpoints.sqlite"
        default.parent.mkdir(parents=True, exist_ok=True)
        path = default

    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    _preset_incremental_vacuum(path)

    from langgraph.checkpoint.sqlite import SqliteSaver

    # ``from_conn_string`` is a context-manager factory; we enter it once and
    # keep the saver alive for the lifetime of the agent. Store the cm on the
    # saver so it can be closed properly — leaks SQLite connections otherwise.
    cm = SqliteSaver.from_conn_string(str(path))
    saver = cm.__enter__()
    _enable_incremental_vacuum(saver)
    # Keep ref so __exit__ can be called during shutdown (ponytail: prevents
    # SQLite handle accumulation across agent rebuilds)
    saver._context_manager = cm  # type: ignore[attr-defined]
    return saver


def create_in_memory_checkpointer() -> Any:
    """Create an in-memory checkpointer for tests or ephemeral agents."""
    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver()
