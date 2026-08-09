"""Trace context — 一次 agent run 的统一 trace_id.

用 contextvars.ContextVar 存当前 trace_id, async/task 安全. 一次 run 启动时
调 new_trace_id() 生成, 之后 audit/engine_state/checkpoint/task_metrics 序列化
时各自从 get_trace_id() 读, 串联成一条可追溯的链. subprocess 边界用
clear_trace_id() 清掉 (子进程不继承父进程的 contextvar 值, 但显式清更安全).

ponytail: 只用 stdlib (uuid + contextvars), 不引新依赖. trace_id 取 uuid4 hex
前 12 位 — 同一进程内冲突概率足够低, 跨进程由 uuid4 全局唯一兜底.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar

_trace_id_var: ContextVar[str | None] = ContextVar("huginn_trace_id", default=None)


def new_trace_id() -> str:
    """生成新 trace_id 并 set 到当前 context, 返回该 id.

    一次 agent run 启动时调一次, 之后同 context 内的所有 get_trace_id() 都拿到它.
    """
    tid = uuid.uuid4().hex[:12]
    _trace_id_var.set(tid)
    return tid


def get_trace_id() -> str | None:
    """读当前 context 的 trace_id. 没设过返回 None."""
    return _trace_id_var.get()


def clear_trace_id() -> None:
    """清掉当前 context 的 trace_id.

    subprocess 边界 / 新 worker 起步时调, 防止旧 trace_id 串到新 run.
    """
    _trace_id_var.set(None)


__all__ = ["new_trace_id", "get_trace_id", "clear_trace_id"]
