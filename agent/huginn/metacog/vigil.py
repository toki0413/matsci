"""VIGIL 兄弟监控运行时 — L0–L4 降级状态机（无 I/O、无线程）.

核心约束（spec §19.12）：
- 监控失败不阻塞主任务；状态机只影响监控自身；
- 降级必须可见：每次迁移产生 DegradationEvent；
- 重启有界退避；crash loop 停在 L3，不得无限重启；
- 队列有界，满了按 DROPPED 丢弃，不背压任务路径。

调用方负责驱动 `heartbeat` / `record_job` / `tick`；本模块不做真实调度。
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

MonitorStateName = Literal["L0", "L1", "L2", "L3", "L4"]
TrustLevel = Literal["full", "partial", "blind", "unknown"]


class MonitorState(StrEnum):
    L0 = "L0"  # full
    L1 = "L1"  # partial
    L2 = "L2"  # blind
    L3 = "L3"  # offline
    L4 = "L4"  # quarantined


_TRUST: dict[str, TrustLevel] = {
    "L0": "full",
    "L1": "partial",
    "L2": "blind",
    "L3": "unknown",
    "L4": "unknown",
}


def _now(now: str | datetime | None = None) -> datetime:
    if now is None:
        return datetime.now(UTC)
    if isinstance(now, datetime):
        return now if now.tzinfo else now.replace(tzinfo=UTC)
    dt = datetime.fromisoformat(str(now).strip().replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


@dataclass(frozen=True)
class MonitorStatus:
    monitor_id: str
    tenant_id: str
    state: MonitorStateName
    trust: TrustLevel
    coverage: dict[str, bool]
    last_heartbeat: str
    degraded_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "monitor_id": self.monitor_id,
            "tenant_id": self.tenant_id,
            "state": self.state,
            "trust": self.trust,
            "coverage": dict(self.coverage),
            "last_heartbeat": self.last_heartbeat,
            "degraded_reason": self.degraded_reason,
        }


@dataclass(frozen=True)
class DegradationEvent:
    monitor_id: str
    tenant_id: str
    from_state: str
    to_state: str
    trigger: str
    created_at: str


@dataclass
class MonitorCircuitBreaker:
    """监控自身 job 的熔断器：1 分钟错误率 > 20% → open 5 分钟 → half-open 放 1 个 canary.

    只保护监控自己的处理队列，不参与任务授权/预算/沙箱联锁。
    """

    error_threshold: float = 0.20
    window_s: float = 60.0
    open_s: float = 300.0
    _events: deque[tuple[float, bool]] = field(default_factory=deque)
    _state: Literal["closed", "open", "half_open"] = "closed"
    _opened_at: float = 0.0
    _canary_in_flight: bool = False

    def record(self, success: bool, *, now: float) -> str:
        now = float(now)
        self._events.append((now, bool(success)))
        cutoff = now - self.window_s
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()
        if self._state == "half_open":
            if success:
                self._state, self._canary_in_flight = "closed", False
                self._events.clear()
            else:
                self._state, self._opened_at, self._canary_in_flight = (
                    "open", now, False,
                )
            return self._state
        if self._state == "open":
            return self._state
        if self._events:
            failures = sum(1 for _, ok in self._events if not ok)
            if failures / len(self._events) > self.error_threshold:
                self._state, self._opened_at = "open", now
        return self._state

    def allow(self, *, now: float) -> bool:
        now = float(now)
        if self._state == "closed":
            return True
        if self._state == "open":
            if now - self._opened_at >= self.open_s:
                self._state = "half_open"
                self._canary_in_flight = True
                return True
            return False
        # half_open：只放一个 canary
        if not self._canary_in_flight:
            self._canary_in_flight = True
            return True
        return False

    def state(self) -> str:
        return self._state


@dataclass
class VigilMonitor:
    monitor_id: str
    tenant_id: str = "default"
    window: int = 20
    error_threshold: float = 0.20
    max_queue: int = 1000
    max_restarts: int = 3
    restart_window_s: float = 600.0
    heartbeat_timeout_s: float = 60.0
    breaker: MonitorCircuitBreaker = field(
        default_factory=lambda: MonitorCircuitBreaker()
    )

    _state: MonitorState = MonitorState.L0
    _coverage: dict[str, bool] = field(default_factory=lambda: {"default": True})
    _events: deque[bool] = field(default_factory=deque)
    _queue: deque[str] = field(default_factory=deque)
    _restart_times: list[datetime] = field(default_factory=list)
    _last_heartbeat: datetime = field(default_factory=lambda: datetime.now(UTC))
    _degraded_reason: str = ""
    _degradation_events: list[DegradationEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._events = deque(maxlen=max(1, int(self.window)))

    # ── transitions ────────────────────────────────────────────────
    def _transition(self, to: MonitorState, trigger: str) -> None:
        if to == self._state:
            return
        event = DegradationEvent(
            monitor_id=self.monitor_id,
            tenant_id=self.tenant_id,
            from_state=self._state.value,
            to_state=to.value,
            trigger=trigger,
            created_at=datetime.now(UTC).isoformat(),
        )
        self._degradation_events.append(event)
        self._state = to
        self._degraded_reason = trigger if to != MonitorState.L0 else ""

    # ── health inputs ──────────────────────────────────────────────
    def heartbeat(self, now: str | datetime | None = None) -> MonitorStatus:
        self._last_heartbeat = _now(now)
        if self._state == MonitorState.L3:
            self._transition(MonitorState.L1, "heartbeat_resumed")
        return self.status()

    def register_detector(self, name: str, ok: bool = True) -> None:
        self._coverage[str(name)] = bool(ok)

    def record_detector(self, name: str, ok: bool) -> MonitorStatus:
        self._coverage[str(name)] = bool(ok)
        if self._state == MonitorState.L4:
            return self.status()
        if not any(self._coverage.values()):
            self._transition(MonitorState.L2, "all_detectors_down")
        elif self._state == MonitorState.L2:
            self._transition(MonitorState.L1, "detector_recovered")
        return self.status()

    def record_job(self, success: bool, now: str | datetime | None = None) -> MonitorStatus:
        if self._state == MonitorState.L4:
            return self.status()
        now_dt = _now(now)
        self._last_heartbeat = now_dt
        self.breaker.record(bool(success), now=now_dt.timestamp())
        self._events.append(bool(success))
        if not any(self._coverage.values()):
            self._transition(MonitorState.L2, "all_detectors_down")
            return self.status()
        failures = sum(1 for ok in self._events if not ok)
        rate = failures / max(1, len(self._events))
        if rate > self.error_threshold:
            self._transition(MonitorState.L1, "error_rate")
        elif self._state in (MonitorState.L1, MonitorState.L2):
            self._transition(MonitorState.L0, "recovered")
        return self.status()

    def tick(self, now: str | datetime | None = None) -> MonitorStatus:
        if self._state in (MonitorState.L3, MonitorState.L4):
            return self.status()
        age = (_now(now) - self._last_heartbeat).total_seconds()
        if age > self.heartbeat_timeout_s:
            self._transition(MonitorState.L3, "heartbeat_timeout")
        return self.status()

    def quarantine(self, reason: str = "self_misbehavior") -> MonitorStatus:
        self._transition(MonitorState.L4, reason)
        return self.status()

    def restart(self, now: str | datetime | None = None) -> bool:
        """有界重启：超过窗口预算 → 停在 L3，返回 False."""
        if self._state == MonitorState.L0:
            return True
        now_dt = _now(now)
        cutoff = now_dt.timestamp() - self.restart_window_s
        self._restart_times = [
            t for t in self._restart_times if t.timestamp() >= cutoff
        ]
        if len(self._restart_times) >= self.max_restarts:
            self._transition(MonitorState.L3, "restart_budget_exhausted")
            return False
        self._restart_times.append(now_dt)
        self._transition(MonitorState.L1, "restart")
        return True

    # ── queue ──────────────────────────────────────────────────────
    def submit(self, job_id: str, now: float | None = None) -> str:
        if self._state in (MonitorState.L2, MonitorState.L3, MonitorState.L4):
            return "DROPPED"
        if not self.breaker.allow(now=time.time() if now is None else now):
            return "DROPPED"
        if len(self._queue) >= self.max_queue:
            return "DROPPED"
        self._queue.append(str(job_id))
        return str(job_id)

    def pop_job(self) -> str | None:
        return self._queue.popleft() if self._queue else None

    # ── read ───────────────────────────────────────────────────────
    @property
    def degradation_events(self) -> list[DegradationEvent]:
        return list(self._degradation_events)

    def breaker_state(self) -> str:
        return self.breaker.state()

    def status(self) -> MonitorStatus:
        """永不抛异常；内部异常返回 trust=unknown."""
        try:
            state = self._state.value
            return MonitorStatus(
                monitor_id=self.monitor_id,
                tenant_id=self.tenant_id,
                state=state,
                trust=_TRUST.get(state, "unknown"),
                coverage=dict(self._coverage),
                last_heartbeat=self._last_heartbeat.isoformat(),
                degraded_reason=self._degraded_reason,
            )
        except Exception:
            return MonitorStatus(
                monitor_id=str(getattr(self, "monitor_id", "unknown")),
                tenant_id=str(getattr(self, "tenant_id", "unknown")),
                state="L4",
                trust="unknown",
                coverage={},
                last_heartbeat="",
                degraded_reason="status_failure",
            )


__all__ = [
    "DegradationEvent",
    "MonitorState",
    "MonitorStateName",
    "MonitorStatus",
    "TrustLevel",
    "VigilMonitor",
]
