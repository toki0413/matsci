"""Sync→async 桥接 helper.

背景: sync wrapper 想调 async 函数, 经典写法是::

    try:
        loop = asyncio.get_running_loop()
        loop.run_until_complete(coro)   # 已在 loop 中 -> 抛 "already running"
    except RuntimeError:
        asyncio.run(coro)               # 没 loop -> OK

但只要 caller 已经在 running loop 里 (FastAPI / WebSocket handler 常见),
``loop.run_until_complete`` 必然抛 "This event loop is already running",
然后被同一个 except 抓住, ``asyncio.run`` 又因为仍有 running loop 而抛
"cannot be called from a running event loop" — 这次没人接, 直接传给 caller.

本 helper 在 running loop 中时改用独立线程跑 ``asyncio.run``, 修掉这个坑.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")

# 共享一个 1-worker 线程池, 避免每次 sync call 都付 thread 创建开销.
# ponytail: 单 worker 串行足够 — sync wrapper 本身就是阻塞语义, 没必要并行.
_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="async-bridge")


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """同步运行 coroutine 并返回结果.

    没有 running loop 时直接 ``asyncio.run``.
    有 running loop 时提交到独立线程跑 ``asyncio.run``, 否则会抛
    "cannot be called from a running event loop".
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # 没人占用 loop, 直接跑
        return asyncio.run(coro)
    # 已在 running loop — 在独立线程里跑独立 loop, 阻塞等结果
    return _pool.submit(asyncio.run, coro).result()
