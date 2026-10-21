"""VIGIL 生产运行时 — 兄弟监控线程 + 心跳 + drift_events 落库.

只做监控，不执行用户任务；线程失败不影响主任务。
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from huginn.metacog.drift_gate import DriftGate, DriftSignal
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
    # 独立进程模式：本地不起线程，status() 读 daemon 写的原子心跳文件。
    external: bool = False
    heartbeat_path: str | None = None
    drift_gate: DriftGate = field(default_factory=DriftGate)
    _thread: threading.Thread | None = None
    _stop: threading.Event = field(default_factory=threading.Event)

    def start(self) -> None:
        if not self.enabled:
            return
        if self.external:
            # 由 MonitorSupervisor 管理子进程；本地不重复启动线程。
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
        status = self.monitor.tick(now)
        self._write_heartbeat(status)
        # 行为指纹周期比较：只写 review 级 drift_events，不阻断任务.
        try:
            from huginn.env_access import env_str
            from huginn.research.behavior_fingerprint import get_behavior_monitor

            get_behavior_monitor().maybe_record_drift(
                tenant_id=env_str("HUGINN_TENANT_ID") or "default",
                persona_id="default",
            )
        except Exception:  # 防御: 指纹监控失败不影响 VIGIL 状态
            logger.debug("behavior fingerprint tick skipped", exc_info=True)
        return status

    def _write_heartbeat(self, status: MonitorStatus) -> None:
        if not self.heartbeat_path:
            return
        try:
            from huginn.metacog.vigil_process import write_heartbeat

            write_heartbeat(status, path=self.heartbeat_path)
        except Exception:  # 防御: 心跳写失败不影响监控/主任务
            logger.debug("vigil heartbeat write failed", exc_info=True)

    def status(self) -> MonitorStatus:
        """永不抛异常；内部失败返回 L4/unknown（I33/I35）."""
        try:
            if self.external and self.heartbeat_path:
                from huginn.metacog.vigil_process import (
                    read_heartbeat,
                    status_from_heartbeat,
                )

                record = read_heartbeat(self.heartbeat_path)
                return status_from_heartbeat(
                    record, stale_after_s=self.heartbeat_interval_s * 3.0
                )
            return self.monitor.status()
        except Exception:  # 防御: 监控自身失败必须显式降级，不穿透任务循环
            logger.debug("vigil status failed", exc_info=True)
            return MonitorStatus(
                monitor_id="vigil",
                tenant_id="default",
                state="L4",
                trust="unknown",
                coverage={},
                last_heartbeat="",
                degraded_reason="status_failure",
            )

    def record_job(self, success: bool) -> MonitorStatus:
        return self.monitor.record_job(success)

    def record_detector(self, name: str, ok: bool) -> MonitorStatus:
        return self.monitor.record_detector(name, ok)

    def drift_status(self) -> dict[str, Any]:
        """当前漂移闸状态；任务侧只读，不提供自清入口."""
        return {
            "level": self.drift_gate.level,
            "can_execute": self.drift_gate.can_execute(),
        }

    def clear_drift(self, *, actor: str, confirmation: bool = False) -> bool:
        """显式清除 pause：actor=agent/task 一律拒绝（I29）."""
        return bool(self.drift_gate.clear(actor=actor, confirmation=confirmation))

    def record_signals(
        self,
        signals: list[DriftSignal],
        *,
        tenant_id: str,
        session_id: str,
        ledger: Any = None,
    ) -> list[str]:
        """把 Smith 漂移信号落成 drift_events；同时单调升级 DriftGate.

        告警级别只升不降；``pause`` 只能由监控环/用户显式确认清除，任务 Agent
        不能自清（I29）。写账失败不影响任务。
        """
        self.drift_gate.update(signals)
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
        from huginn.metacog.vigil_process import heartbeat_path

        _runtime = VigilRuntime(
            monitor=VigilMonitor(
                "vigil",
                error_threshold=env_float("HUGINN_VIGIL_ERROR_THRESHOLD"),
            ),
            heartbeat_interval_s=float(env_int("HUGINN_VIGIL_HEARTBEAT_S")),
            enabled=env_bool("HUGINN_VIGIL_ENABLED"),
            external=env_bool("HUGINN_VIGIL_PROCESS"),
            heartbeat_path=str(heartbeat_path()),
        )
    return _runtime


__all__ = ["VigilRuntime", "get_vigil_runtime"]
