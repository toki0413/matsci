"""长任务进度跟踪 —— WorkflowEngine / AutoloopEngine 共用.

设计: ProgressTracker 是一个进程级任务表, 引擎在 stage/phase 开始
和结束时调 update / complete / fail. 前端通过 GET /tasks 或
GET /tasks/stream (SSE) 拿进度.

支持多任务并发: 每个 task_id 独立维护状态, 互不干扰. task_id 由
调用方生成, 通常用 "{session_id}:{engine_kind}:{uuid8}" 格式.

ETA 计算: 简单线性外推 —— 用已用时间 / 已完成步数 * 剩余步数.
没有历史数据时返回 None, 不瞎猜.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TaskProgress:
    """单个任务的进度状态."""

    task_id: str
    description: str
    total_steps: int
    current_step: int = 0
    status: str = "pending"  # pending | running | completed | failed | cancelled
    eta_seconds: Optional[float] = None
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    # 阶段标签: 比如 ["scf", "relax", "band", "dos"], 给前端展示用
    stage_labels: list[str] = field(default_factory=list)
    # 当前阶段的人类可读状态, 比如 "正在跑 SCF 第 3 步"
    current_label: str = ""
    # 失败时的错误信息
    error: str = ""
    # 引擎类型: workflow / autoloop / custom, 给前端分组用
    engine_kind: str = ""
    # 自由扩展字段, 引擎想塞什么就塞什么
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def percentage(self) -> float:
        """完成百分比, 0-100. total_steps=0 时返回 0."""
        if self.total_steps <= 0:
            return 0.0
        return round(100.0 * self.current_step / self.total_steps, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "total_steps": self.total_steps,
            "current_step": self.current_step,
            "percentage": self.percentage,
            "status": self.status,
            "eta_seconds": self.eta_seconds,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "stage_labels": list(self.stage_labels),
            "current_label": self.current_label,
            "error": self.error,
            "engine_kind": self.engine_kind,
            "metadata": dict(self.metadata),
        }


class ProgressTracker:
    """跟踪多个长任务的进度, 支持并发.

    线程安全: 内部用 threading.Lock 保护 dict 操作. 引擎的 update
    可能在不同协程 / 线程里调 (比如 WorkflowEngine 的并行 stage),
    必须加锁.

    SSE 推送: 每次 update 都把新状态塞进事件队列, to_event_stream
    消费者按 FIFO 读. 多个 SSE 客户端各自订阅同一份队列, 互不干扰
    (实际实现是每个客户端拿一份快照 + 轮询, 简单起见).
    """

    def __init__(self) -> None:
        self._tasks: dict[str, TaskProgress] = {}
        self._lock = threading.Lock()
        # SSE 推送用: 每次更新压一条, 客户端从队列读
        self._events: list[dict[str, Any]] = []
        self._event_signal = asyncio.Event()
        # 保留最近 N 条事件, 避免内存无限增长
        self._max_events = 1000
        # 标记是否已关闭 (通常不关, 进程级单例)
        self._closed = False

    # ── 任务生命周期 ───────────────────────────────────────────

    def start_task(
        self,
        task_id: str,
        description: str,
        total_steps: int,
        stage_labels: list[str] | None = None,
        engine_kind: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> TaskProgress:
        """登记一个新任务. 同 task_id 已存在会被覆盖 (视为重启)."""
        task = TaskProgress(
            task_id=task_id,
            description=description,
            total_steps=max(0, int(total_steps)),
            stage_labels=list(stage_labels) if stage_labels else [],
            engine_kind=engine_kind,
            metadata=dict(metadata) if metadata else {},
            status="running",
            current_step=0,
            started_at=time.time(),
            updated_at=time.time(),
        )
        with self._lock:
            self._tasks[task_id] = task
        self._emit(task)
        return task

    def update(
        self,
        task_id: str,
        current_step: int | None = None,
        status: str | None = None,
        eta: Optional[float] = None,
        current_label: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Optional[TaskProgress]:
        """更新任务进度. 任务不存在返回 None."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            if current_step is not None:
                task.current_step = max(0, int(current_step))
            if status is not None:
                task.status = status
            if eta is not None:
                task.eta_seconds = float(eta)
            elif task.total_steps > 0 and task.current_step > 0:
                # 没显式给 eta 就线性外推一下
                elapsed = time.time() - task.started_at
                per_step = elapsed / task.current_step
                remaining = task.total_steps - task.current_step
                task.eta_seconds = max(0.0, per_step * remaining)
            if current_label is not None:
                task.current_label = current_label
            if metadata:
                task.metadata.update(metadata)
            task.updated_at = time.time()
        self._emit(task)
        return task

    def complete(
        self, task_id: str, result: Any = None
    ) -> Optional[TaskProgress]:
        """标记任务完成."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            task.status = "completed"
            task.current_step = task.total_steps
            task.completed_at = time.time()
            task.updated_at = task.completed_at
            task.eta_seconds = 0.0
            if result is not None:
                # result 可能很大, 只存摘要, 不存全量
                task.metadata["result_summary"] = _summarize_result(result)
        self._emit(task)
        return task

    def fail(self, task_id: str, error: str) -> Optional[TaskProgress]:
        """标记任务失败."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            task.status = "failed"
            task.error = str(error)[:500]
            task.completed_at = time.time()
            task.updated_at = task.completed_at
        self._emit(task)
        return task

    def cancel(self, task_id: str) -> Optional[TaskProgress]:
        """标记任务取消 (区别于 fail: 用户主动停, 不是出错)."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            task.status = "cancelled"
            task.completed_at = time.time()
            task.updated_at = task.completed_at
        self._emit(task)
        return task

    # ── 查询 ───────────────────────────────────────────────────

    def get_status(self, task_id: str) -> Optional[dict[str, Any]]:
        """拿单个任务的状态. 不存在返回 None."""
        with self._lock:
            task = self._tasks.get(task_id)
            return task.to_dict() if task else None

    def list_active(self) -> list[dict[str, Any]]:
        """列出所有未完成的任务 (running / pending)."""
        with self._lock:
            return [
                t.to_dict()
                for t in self._tasks.values()
                if t.status in ("running", "pending")
            ]

    def list_all(self) -> list[dict[str, Any]]:
        """列出所有任务 (含已完成). 按更新时间倒序."""
        with self._lock:
            tasks = list(self._tasks.values())
        tasks.sort(key=lambda t: t.updated_at, reverse=True)
        return [t.to_dict() for t in tasks]

    # ── SSE 流 ────────────────────────────────────────────────

    async def to_event_stream(self) -> Any:
        """转成 SSE 异步生成器. 每次 update 都吐一条.

        注意: 这里返回的是 dict 异步生成器, 路由层负责序列化成 SSE 字符串
        (用 json.dumps + event: 格式). 这样 ProgressTracker 不耦合 SSE
        协议细节, 也能被非 HTTP 调用方复用.
        """
        import json

        # 先吐一份当前全量快照, 让新接入的客户端立刻看到所有任务
        for task_dict in self.list_all():
            yield f"event: snapshot\ndata: {json.dumps(task_dict, ensure_ascii=False)}\n\n"

        last_emitted_count = 0
        while not self._closed:
            with self._lock:
                new_events = self._events[last_emitted_count:]
                last_emitted_count = len(self._events)
            for evt in new_events:
                yield f"event: update\ndata: {json.dumps(evt, ensure_ascii=False)}\n\n"
            # 等新事件
            self._event_signal.clear()
            # 用 timeout 避免 signal 永远不 set (比如没人 update 时)
            try:
                await asyncio.wait_for(self._event_signal.wait(), timeout=15.0)
            except asyncio.TimeoutError:
                # 超时也吐一条心跳, 保持 SSE 连接活着
                yield f"event: heartbeat\ndata: {json.dumps({'ts': time.time()})}\n\n"

    # ── 内部 ───────────────────────────────────────────────────

    def _emit(self, task: TaskProgress) -> None:
        """把任务状态压进事件队列, 唤醒 SSE 消费者."""
        with self._lock:
            self._events.append(task.to_dict())
            # 超过上限就丢最早的, 别无限增长
            if len(self._events) > self._max_events:
                # 删掉前 10%, 保留最近的
                drop = len(self._events) - self._max_events
                del self._events[:drop]
        # set 要在锁外调, 避免在锁里唤醒协程
        self._event_signal.set()


def _summarize_result(result: Any, max_len: int = 500) -> str:
    """把任意 result 转成短摘要, 存进 metadata."""
    try:
        if isinstance(result, str):
            text = result
        elif isinstance(result, dict):
            # 优先取 success / error / summary 这种关键字段
            keys = ("success", "error", "summary", "message", "result")
            parts = [f"{k}={result[k]}" for k in keys if k in result]
            text = ", ".join(parts) if parts else str(result)
        else:
            text = str(result)
        if len(text) > max_len:
            text = text[:max_len] + f"...<+{len(text) - max_len} chars>"
        return text
    except Exception:
        return "<unsummarizable>"


# ── 进程级单例 ──────────────────────────────────────────────
#
# WorkflowEngine / AutoloopEngine 实例化时不强制注入 tracker,
# 调用方可以传自己的 (比如测试用), 默认走进程级单例. 这样多个
# 引擎实例的进度都汇总到一处, 前端 /tasks 一次拿全.

_singleton: ProgressTracker | None = None
_singleton_lock = threading.Lock()


def get_progress_tracker() -> ProgressTracker:
    """拿进程级 ProgressTracker 单例."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = ProgressTracker()
    return _singleton
