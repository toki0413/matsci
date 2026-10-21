"""VIGIL 生产运行时 — 兄弟监控线程 + 心跳 + drift_events 落库.

只做监控，不执行用户任务；线程失败不影响主任务。
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from huginn.metacog.drift_gate import DriftSignal
from huginn.metacog.drift_ledger import DriftEvent, get_drift_ledger
from huginn.metacog.vigil import MonitorStatus, VigilMonitor

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class VigilRuntime:
    monitor: VigilMonitor = field(default_factory=lambda: VigilMonitor("vigil"))
    heartbeat_interval_s: float = 30.0
    enabled: bool = True
    _thread: threading.Thread | None = None
    _stop: threading.Event = field(default_factory=threading.Event)

    def start(self) -> None:
        if not self.enabled:
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="vigil-runtime"
        )
        self._thread.start()
        logger.info("vigil runtime started (interval=%.1fs)", self.heartbeat_interval_s)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        logger.info("vigil runtime stopped")

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception:
                logger.warning("vigil runtime tick failed (non-fatal)", exc_info=True)
            if self._stop.wait(self.heartbeat_interval_s):
                break

    def run_once(self, now: str | datetime | None = None) -> MonitorStatus:
        self.monitor.heartbeat(now)
        return self.monitor.tick(now)

    def status(self) -> MonitorStatus:
        return self.monitor.status()

    def record_job(self, success: bool) -> MonitorStatus:
        return self.monitor.record_job(success)

    def record_detector(self, name: str, ok: bool) -> MonitorStatus:
        return self.monitor.record_detector(name, ok)

    def record_signals(
        self,
        signals: list[DriftSignal],
        *,
        tenant_id: str,
        session_id: str,
        ledger: Any = None,
    ) -> list[str]:
        """把 Smith 漂移信号落成 drift_events；不阻断任务."""
        target = ledger or get_drift_ledger()
        ids: list[str] = []
        for signal in signals:
            event = DriftEvent(
                tenant_id=tenant_id,
                session_id=session_id,
                detector=signal.detector,
                baseline_id=f"{signal.detector}:{signal.reason[:40]}",
                observed=float(signal.observed),
                threshold=float(signal.baseline),
                level=signal.level,
                evidence={
                    "delta": signal.delta,
                    "reason": signal.reason,
                },
            )
            ids.append(target.record(event))
        return ids


_runtime: VigilRuntime | None = None


def get_vigil_runtime() -> VigilRuntime:
    global _runtime
    if _runtime is None:
        from huginn.env_access import env_bool, env_float, env_int

        _runtime = VigilRuntime(
            monitor=VigilMonitor(
                "vigil",
                error_threshold=env_float("HUGINN_VIGIL_ERROR_THRESHOLD"),
            ),
            heartbeat_interval_s=float(env_int("HUGINN_VIGIL_HEARTBEAT_S")),
            enabled=env_bool("HUGINN_VIGIL_ENABLED"),
        )
    return _runtime


__all__ = ["VigilRuntime", "get_vigil_runtime"]
