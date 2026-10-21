"""常驻空闲环 — PreparedContext / IdleJobQueue 的生产接线（spec §19.2）.

只做预取与预计算：不执行用户任务、不产生用户可见动作、不写授权。
默认关闭；``HUGINN_IDLE_ENABLED=1`` 时由 lifespan 启动 daemon 线程。
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from huginn.metacog.proactive_prep import (
    IdleJob,
    IdleJobQueue,
    PreparedContextStore,
)

logger = logging.getLogger(__name__)


@dataclass
class IdleRuntime:
    queue: IdleJobQueue = field(default_factory=IdleJobQueue)
    store: PreparedContextStore = field(default_factory=PreparedContextStore)
    tenant_id: str = "default"
    persona_id: str = "default"
    interval_s: float = 300.0
    budget_ms: int = 60_000
    enabled: bool = True
    jobs_fn: Callable[[], list[IdleJob]] | None = None
    executor: Callable[[IdleJob], None] | None = None
    now_fn: Callable[[], float] = time.time
    _thread: threading.Thread | None = None
    _stop: threading.Event = field(default_factory=threading.Event)

    def start(self) -> None:
        if not self.enabled or self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="huginn-idle-runtime"
        )
        self._thread.start()
        logger.info("idle runtime started (interval=%.1fs)", self.interval_s)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception:
                logger.warning("idle runtime tick failed (non-fatal)", exc_info=True)
            if self._stop.wait(self.interval_s):
                break

    def run_once(self) -> dict[str, int]:
        """一次空闲 tick：先过期候选，再提交/执行队列（无可见动作）."""
        # 过期的预取立即失效，不得作为当前共同基础。
        expired = self.store.candidates(self.tenant_id)
        if self.jobs_fn is not None:
            try:
                for job in self.jobs_fn() or []:
                    self.queue.submit(job, budget_cap_ms=self.budget_ms)
            except Exception:  # 防御: 需求探针失败不影响其他租户
                logger.debug("idle jobs_fn failed", exc_info=True)
        executor = self.executor or _noop_executor
        counts = self.queue.run_due(
            self.tenant_id, executor, budget_cap_ms=self.budget_ms
        )
        counts["candidates"] = len(expired)
        return counts


def _noop_executor(job: IdleJob) -> None:  # noqa: ARG001
    """没有注入 executor 时不做任何事（不伪造预取产物）."""
    return None


_runtime: IdleRuntime | None = None


def get_idle_runtime() -> IdleRuntime:
    global _runtime
    if _runtime is None:
        from huginn.env_access import env_bool, env_int, env_str

        _runtime = IdleRuntime(
            tenant_id=env_str("HUGINN_TENANT_ID") or "default",
            interval_s=float(env_int("HUGINN_IDLE_INTERVAL_S")),
            budget_ms=int(env_int("HUGINN_IDLE_BUDGET_MS")),
            enabled=env_bool("HUGINN_IDLE_ENABLED"),
        )
    return _runtime


__all__ = ["IdleRuntime", "get_idle_runtime"]
