"""VIGIL 独立进程 + watchdog（spec §19.12.5）.

把监控从「同进程 daemon 线程」升级为「独立子进程 + 外部 watchdog」：

- **心跳文件**：daemon 每 tick 原子写 ``monitor_health.json``；主进程只读，
  不共享可变状态、不持有监控锁。
- **watchdog 在监控之外**：主进程内的 :class:`MonitorSupervisor` 只读心跳和
  子进程退出码；崩溃按 1/2/4/8 分钟有界退避重启，单窗口最多 3 次，
  超过即停在 L3（禁止 crash loop）。
- **crash-only**：daemon 随时可被 kill；重启后从心跳重建，不补发错过的 finding。
- **L4**：心跳自报 L4（死循环/输出 schema 非法/越权写）→ kill + 退避重启，
  其输出不得进入任何账本。
- 默认关闭（``HUGINN_VIGIL_PROCESS=0``），保持既有同进程线程行为；打开后
  :class:`VigilRuntime` 不启动本地线程，``status()`` 改读心跳文件。

本模块只做进程管理与心跳读写，不做诊断；诊断仍在 ``vigil.py``。
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from huginn.metacog.vigil import DegradationEvent, MonitorStatus

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _iso_to_ts(value: str) -> float | None:
    try:
        dt = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        if not dt.tzinfo:
            dt = dt.replace(tzinfo=UTC)
        return dt.timestamp()
    except (TypeError, ValueError):
        return None


def default_heartbeat_path() -> Path:
    from huginn.utils.runtime import get_runtime_home

    return get_runtime_home() / "monitor_health.json"


def heartbeat_path() -> Path:
    """心跳文件路径；``HUGINN_VIGIL_HEARTBEAT_FILE`` 可覆盖."""
    from huginn.env_access import env_str

    override = env_str("HUGINN_VIGIL_HEARTBEAT_FILE").strip()
    return Path(override) if override else default_heartbeat_path()


def write_heartbeat(
    status: MonitorStatus | dict[str, Any],
    *,
    path: str | Path | None = None,
) -> str:
    """原子写心跳（tmp + rename），读者永远看到完整 JSON."""
    payload = status.to_dict() if isinstance(status, MonitorStatus) else dict(status)
    payload["written_at"] = _now_iso()
    target = Path(path) if path else heartbeat_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, target)
    return str(target)


def read_heartbeat(path: str | Path | None = None) -> dict[str, Any] | None:
    target = Path(path) if path else heartbeat_path()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def heartbeat_age_s(
    record: dict[str, Any] | None, *, now_ts: float | None = None
) -> float | None:
    if not record:
        return None
    written = _iso_to_ts(
        str(record.get("written_at") or record.get("last_heartbeat") or "")
    )
    if written is None:
        return None
    return max(0.0, float(now_ts if now_ts is not None else time.time()) - written)


def status_from_heartbeat(
    record: dict[str, Any] | None,
    *,
    stale_after_s: float = 90.0,
    now_ts: float | None = None,
) -> MonitorStatus:
    """把心跳映射成 MonitorStatus；缺失/过期一律 L3 + unknown（不静默正常）."""
    age = heartbeat_age_s(record, now_ts=now_ts)
    if record is None:
        return MonitorStatus(
            monitor_id="vigil",
            tenant_id="default",
            state="L3",
            trust="unknown",
            coverage={},
            last_heartbeat="",
            degraded_reason="heartbeat missing",
        )
    if age is None or age > stale_after_s:
        return MonitorStatus(
            monitor_id="vigil",
            tenant_id="default",
            state="L3",
            trust="unknown",
            coverage={},
            last_heartbeat=str(record.get("last_heartbeat") or ""),
            degraded_reason=(
                f"heartbeat stale ({age if age is None else round(age, 1)}s)"
            ),
        )
    state = str(record.get("state") or "L0")
    trust = str(record.get("trust") or "unknown")
    coverage = record.get("coverage")
    return MonitorStatus(
        monitor_id=str(record.get("monitor_id") or "vigil"),
        tenant_id=str(record.get("tenant_id") or "default"),
        state=state,  # type: ignore[arg-type]
        trust=trust,  # type: ignore[arg-type]
        coverage=dict(coverage) if isinstance(coverage, dict) else {},
        last_heartbeat=str(record.get("last_heartbeat") or ""),
        degraded_reason=str(record.get("degraded_reason") or ""),
    )


@dataclass(frozen=True)
class SupervisorConfig:
    stale_after_s: float = 90.0
    max_restarts: int = 3
    restart_window_s: float = 600.0
    backoff_s: tuple[float, ...] = (60.0, 120.0, 240.0, 480.0)


@dataclass
class SupervisorResult:
    state: str
    trust: str
    restarts: int
    next_restart_in_s: float | None = None
    degraded_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "trust": self.trust,
            "restarts": self.restarts,
            "next_restart_in_s": self.next_restart_in_s,
            "degraded_reason": self.degraded_reason,
        }


@dataclass
class MonitorSupervisor:
    """监控之外的 watchdog：只读心跳/退出码，不阻塞、不共享写锁."""

    spawn: Callable[[], Any]
    config: SupervisorConfig = field(default_factory=SupervisorConfig)
    read: Callable[[], dict[str, Any] | None] = read_heartbeat
    now: Callable[[], float] = time.time
    on_degradation: Callable[[DegradationEvent], None] | None = None

    _proc: Any = None
    _state: str = "L3"
    _reason: str = "not started"
    _restart_times: list[float] = field(default_factory=list)
    _pending_restart_at: float | None = None
    _started_at: float | None = None

    def start(self) -> None:
        if self._proc is None:
            self._spawn()

    def stop(self, timeout: float = 5.0) -> None:
        proc = self._proc
        self._proc = None
        self._pending_restart_at = None
        if proc is None:
            return
        try:
            proc.terminate()
            try:
                proc.wait(timeout=timeout)
            except Exception:
                proc.kill()
        except Exception:  # 防御: 停止失败不影响主进程退出
            logger.debug("vigil daemon stop failed", exc_info=True)

    def _spawn(self) -> None:
        try:
            self._proc = self.spawn()
            self._started_at = self.now()
            self._pending_restart_at = None
            self._state, self._reason = "L2", "starting"
        except Exception as exc:  # 启动失败按 crash 处理
            self._proc = None
            self._state, self._reason = "L3", f"spawn failed: {exc}"
            logger.warning("vigil daemon spawn failed: %s", exc)

    def tick(self) -> SupervisorResult:
        """一次 watchdog 检查：崩溃/心跳过期/L4 → 有界退避重启."""
        now = self.now()
        record = self.read()

        proc = self._proc
        crashed = proc is not None and _poll_dead(proc)
        l4 = bool(record and str(record.get("state")) == "L4")
        age = heartbeat_age_s(record, now_ts=now)
        stale = record is not None and age is not None and age > self.config.stale_after_s
        missing = (
            record is None
            and self._started_at is not None
            and now - self._started_at > self.config.stale_after_s
        )

        failure = None
        if crashed:
            failure = "crash"
        elif l4:
            failure = "runaway"
        elif stale:
            failure = "heartbeat_stale"
        elif missing:
            failure = "timeout"

        if failure is not None:
            from_state = self._state
            self._reap(proc)
            # 有界重启：窗口内超过 max_restarts → 停在 L3，禁止 crash loop。
            self._restart_times = [
                ts
                for ts in self._restart_times
                if now - ts <= self.config.restart_window_s
            ]
            if len(self._restart_times) >= self.config.max_restarts:
                self._state, self._reason = "L3", f"{failure}: restart budget exhausted"
                self._pending_restart_at = None
                self._emit(failure, from_state, "L3")
                return self._result(now)
            backoff = self.config.backoff_s[
                min(len(self._restart_times), len(self.config.backoff_s) - 1)
            ]
            self._restart_times.append(now)
            self._pending_restart_at = now + backoff
            self._state, self._reason = "L2", f"{failure}: restart in {backoff:.0f}s"
            self._emit(failure, from_state, "L2")
            return self._result(now)

        if self._proc is None:
            if self._pending_restart_at is None or now >= self._pending_restart_at:
                self._spawn()
            else:
                return self._result(now)

        if record is not None and not stale:
            state = str(record.get("state") or "L0")
            if state != self._state:
                previous = self._state
                self._state = state
                self._reason = str(record.get("degraded_reason") or "")
                self._emit("heartbeat", previous, state)
            return SupervisorResult(
                state=state,
                trust=str(record.get("trust") or "unknown"),
                restarts=len(self._restart_times),
                degraded_reason=self._reason,
            )
        return self._result(now)

    def status(self) -> dict[str, Any]:
        return self._result(self.now()).to_dict()

    def _result(self, now: float) -> SupervisorResult:
        trust = {"L0": "full", "L1": "partial", "L2": "blind"}.get(
            self._state, "unknown"
        )
        next_in = (
            max(0.0, self._pending_restart_at - now)
            if self._pending_restart_at is not None
            else None
        )
        return SupervisorResult(
            state=self._state,
            trust=trust,
            restarts=len(self._restart_times),
            next_restart_in_s=next_in,
            degraded_reason=self._reason,
        )

    def _reap(self, proc: Any) -> None:
        self._proc = None
        if proc is None:
            return
        try:
            if not _poll_dead(proc):
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except Exception:
                    proc.kill()
        except Exception:  # 防御
            logger.debug("vigil daemon reap failed", exc_info=True)

    def _emit(self, trigger: str, from_state: str, to_state: str) -> None:
        callback = self.on_degradation
        if callback is None:
            return
        event = DegradationEvent(
            monitor_id="vigil",
            tenant_id="default",
            from_state=from_state,
            to_state=to_state,
            trigger=trigger,
            created_at=_now_iso(),
        )
        try:
            callback(event)
        except Exception:  # 防御: 审计回调失败不影响 watchdog
            logger.debug("degradation callback failed", exc_info=True)


def _poll_dead(proc: Any) -> bool:
    try:
        return proc.poll() is not None
    except Exception:
        return True


def _subprocess_spawn() -> subprocess.Popen[Any]:
    """默认 spawn：当前 Python 解释器启动 ``huginn.metacog.vigil_daemon``."""
    argv = [sys.executable, "-m", "huginn.metacog.vigil_daemon"]
    return subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        env={**os.environ, "HUGINN_VIGIL_ENABLED": "1"},
    )


_supervisor: MonitorSupervisor | None = None


def get_vigil_supervisor() -> MonitorSupervisor:
    global _supervisor
    if _supervisor is None:
        from huginn.env_access import env_float, env_int

        _supervisor = MonitorSupervisor(
            spawn=_subprocess_spawn,
            config=SupervisorConfig(
                stale_after_s=env_float("HUGINN_VIGIL_HEARTBEAT_S") * 3.0,
                max_restarts=env_int("HUGINN_VIGIL_RESTART_MAX"),
            ),
            read=lambda: read_heartbeat(heartbeat_path()),
        )
    return _supervisor


__all__ = [
    "MonitorSupervisor",
    "SupervisorConfig",
    "SupervisorResult",
    "default_heartbeat_path",
    "heartbeat_age_s",
    "heartbeat_path",
    "get_vigil_supervisor",
    "read_heartbeat",
    "status_from_heartbeat",
    "write_heartbeat",
]
