"""VIGIL 独立进程/watchdog 测试：心跳、有界重启、L4、可见降级."""
from __future__ import annotations

from datetime import UTC, datetime

from huginn.metacog.vigil import VigilMonitor
from huginn.metacog.vigil_process import (
    MonitorSupervisor,
    SupervisorConfig,
    read_heartbeat,
    status_from_heartbeat,
    write_heartbeat,
)
from huginn.metacog.vigil_runtime import VigilRuntime


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat()


def _record(ts: float, state: str = "L0") -> dict:
    return {
        "monitor_id": "vigil",
        "tenant_id": "default",
        "state": state,
        "trust": {"L0": "full", "L1": "partial", "L2": "blind"}.get(state, "unknown"),
        "coverage": {"default": True},
        "last_heartbeat": _iso(ts),
        "written_at": _iso(ts),
        "degraded_reason": "",
    }


class _FakeProc:
    def __init__(self) -> None:
        self.dead = False
        self.terminated = False
        self.killed = False

    def poll(self):
        return 0 if self.dead else None

    def terminate(self):
        self.terminated = True
        self.dead = True

    def wait(self, timeout=None):  # noqa: ARG002
        return 0

    def kill(self):
        self.killed = True
        self.dead = True


def test_heartbeat_roundtrip_and_stale_is_l3(tmp_path):
    path = tmp_path / "monitor_health.json"
    record = _record(1000.0)
    write_heartbeat(record, path=path)
    loaded = read_heartbeat(path)
    assert loaded is not None
    assert loaded["state"] == "L0"
    assert loaded["written_at"], "写入时必须补 written_at"
    assert not (tmp_path / "monitor_health.json.tmp").exists(), "原子写不应留 tmp"

    written_ts = datetime.fromisoformat(loaded["written_at"]).timestamp()
    fresh = status_from_heartbeat(loaded, stale_after_s=90, now_ts=written_ts + 10)
    assert fresh.state == "L0" and fresh.trust == "full"

    stale = status_from_heartbeat(
        loaded, stale_after_s=90, now_ts=written_ts + 1000
    )
    assert stale.state == "L3", "心跳过期必须可见降级，不能静默正常"
    assert stale.trust == "unknown"
    assert "stale" in stale.degraded_reason

    missing = status_from_heartbeat(None, stale_after_s=90, now_ts=1200.0)
    assert missing.state == "L3" and missing.trust == "unknown"


def test_supervisor_bounded_restart_backoff():
    clock = [1000.0]
    procs: list[_FakeProc] = []

    def spawn():
        proc = _FakeProc()
        procs.append(proc)
        return proc

    events: list[str] = []
    supervisor = MonitorSupervisor(
        spawn=spawn,
        config=SupervisorConfig(
            stale_after_s=90,
            max_restarts=3,
            restart_window_s=600,
            backoff_s=(60, 120, 240),
        ),
        read=lambda: _record(clock[0]),
        now=lambda: clock[0],
        on_degradation=lambda event: events.append(event.trigger),
    )
    supervisor.start()
    assert len(procs) == 1
    assert supervisor.tick().state == "L0"

    # 第 1 次 crash → 退避 60s，不立即重启
    procs[-1].dead = True
    result = supervisor.tick()
    assert result.state == "L2"
    assert result.restarts == 1
    assert result.next_restart_in_s == 60

    clock[0] += 61
    assert supervisor.tick().state == "L0"
    assert len(procs) == 2, "退避到期后应重启一次"

    # 第 2/3 次 crash：退避 120/240
    procs[-1].dead = True
    assert supervisor.tick().restarts == 2
    clock[0] += 121
    supervisor.tick()
    assert len(procs) == 3

    procs[-1].dead = True
    assert supervisor.tick().restarts == 3
    clock[0] += 241
    supervisor.tick()
    assert len(procs) == 4

    # 第 4 次 crash：窗口内预算耗尽 → 停 L3，禁止 crash loop
    procs[-1].dead = True
    result = supervisor.tick()
    assert result.state == "L3"
    assert "exhausted" in result.degraded_reason
    assert len(procs) == 4, "预算耗尽后不得继续重启"
    assert events.count("crash") >= 1
    assert events[-1] == "crash"


def test_supervisor_quarantines_l4_heartbeat():
    clock = [1000.0]
    procs: list[_FakeProc] = []
    triggers: list[str] = []

    def spawn():
        proc = _FakeProc()
        procs.append(proc)
        return proc

    supervisor = MonitorSupervisor(
        spawn=spawn,
        config=SupervisorConfig(stale_after_s=90, max_restarts=3),
        read=lambda: _record(clock[0], state="L4"),
        now=lambda: clock[0],
        on_degradation=lambda event: triggers.append(event.trigger),
    )
    supervisor.start()
    result = supervisor.tick()
    assert result.state == "L2"
    assert procs[0].dead is True, "L4 必须 kill"
    assert "runaway" in triggers


def test_runtime_external_mode_reads_heartbeat(tmp_path):
    path = tmp_path / "monitor_health.json"
    runtime = VigilRuntime(
        monitor=VigilMonitor("vigil"),
        enabled=True,
        external=True,
        heartbeat_path=str(path),
    )
    runtime.start()
    assert runtime._thread is None, "外部进程模式不得在本地起线程"

    runtime.run_once()
    assert path.exists()
    status = runtime.status()
    assert status.state == "L0"
    assert status.trust == "full"

    runtime.stop()
