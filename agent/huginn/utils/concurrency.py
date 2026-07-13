"""并发原语: track_task + atomic_write.

track_task: 把 fire-and-forget asyncio.create_task 注册到模块级 set,
  防止 CPython 在 task 完成前 GC 掉 Task 对象 (asyncio 官方文档明确警告这点).
atomic_write: tmp 文件 + fsync + os.replace 原子写入, 防止崩溃留半截文件.

不做 StateGuard / async_state_lock 之类的统一抽象: 项目里 threading.RLock
(server_core._state_lock 跨 5 模块共用) 和散落的 asyncio.Lock 都工作正常,
加封装只会让调用方多一层间接, 没有真实收益.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import tempfile
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

# 全局 task 注册表. add 后 task done 触发自动 discard, 不需要手动清理.
# ponytail: 不做 per-module 注册表 (routes/ws_helpers + routes/autoloop +
# scheduling/scheduler 三套并行 set/dict 太重复), 一把全局 set 已足够,
# task 完成自动消失. 需要按 name 查 task 的场景仍然可以自己另存 dict.
_pending_tasks: set[asyncio.Task[Any]] = set()


def track_task(
    coro: Coroutine[Any, Any, Any], *, name: str | None = None
) -> asyncio.Task[Any]:
    """创建并追踪一个后台 task, 阻止 CPython 在完成前 GC 掉它.

    asyncio.create_task 文档明确警告: 调用方必须保留 Task 引用, 否则解释器
    可能在 task 完成前回收它, 静默取消后台工作. 本 helper 把 task 加进
    模块级 set, done callback 自动 discard, 调用方一行 fire-and-forget 也安全.

    返回 Task 以便调用方 await / cancel / add_done_callback.
    """
    task = asyncio.create_task(coro, name=name) if name else asyncio.create_task(coro)
    _pending_tasks.add(task)
    task.add_done_callback(_pending_tasks.discard)
    return task


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """原子写入 bytes: 写同目录 tmp + fsync + os.replace.

    崩溃在 tmp 写到一半只会留下半截 tmp 文件, target 不受影响. os.replace
    在同一文件系统上原子 (POSIX rename + Windows MoveFileEx 都是原子语义).
    tmp 必须和 target 同目录, 否则跨文件系统 os.replace 退化成 copy+unlink.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(p.parent), prefix=f".{p.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, p)
    except Exception:
        with contextlib.suppress(Exception):
            os.unlink(tmp_path)
        raise


def atomic_write_text(
    path: str | Path, text: str, encoding: str = "utf-8"
) -> None:
    """原子写入文本 (默认 UTF-8)."""
    atomic_write_bytes(path, text.encode(encoding))


def pending_tasks_snapshot() -> set[asyncio.Task[Any]]:
    """返回当前 _pending_tasks 的拷贝. 测试 / shutdown 用."""
    return set(_pending_tasks)
