"""I35–I40: 监控降级不阻塞、不抛异常、可观测、有界重启、租户隔离."""
from __future__ import annotations

import time

from huginn.core_types import RiskLevel
from huginn.metacog.vigil import VigilMonitor
from huginn.metacog.vigil_runtime import VigilRuntime
from huginn.normative.admission import AdmissionRequest
from huginn.normative.ledger import NormativeLedger
from huginn.normative.system3 import admit_with_trust


class _BrokenMonitor:
    def status(self):
        raise RuntimeError("monitor exploded")

    def record_job(self, success: bool):  # noqa: ARG002
        raise RuntimeError("monitor exploded")


def test_no_exception_no_block():
    """I35: status() 永不 raise；监控内部失败只返回降级状态."""
    runtime = VigilRuntime(monitor=_BrokenMonitor(), enabled=False)  # type: ignore[arg-type]
    status = runtime.status()
    assert status.state == "L4"
    assert status.trust == "unknown"
    assert "status_failure" in status.degraded_reason


def test_overhead_and_locking():
    """I36: 监控读路径开销小，且不持有任务侧写锁."""
    runtime = VigilRuntime(monitor=VigilMonitor("v1", window=20), enabled=False)
    start = time.perf_counter()
    for _ in range(5000):
        runtime.status()
        runtime.record_job(True)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"5000 次监控读路径耗时 {elapsed:.3f}s"
    assert not hasattr(runtime, "_task_lock"), "监控不得共享任务写锁"


def test_interlock_independence(tmp_path):
    """I37: 监控不是授权/预算/沙箱的唯一执行点."""
    runtime = VigilRuntime(monitor=_BrokenMonitor(), enabled=False)  # type: ignore[arg-type]
    assert runtime.status().trust == "unknown"
    for forbidden in ("authorize", "execute", "kill_task", "write_authorization"):
        assert not hasattr(runtime, forbidden), f"监控不得提供 {forbidden}"

    ledger = NormativeLedger(tmp_path / "norm.db")
    view = ledger.project("t1", "s1")
    request = AdmissionRequest(
        tenant_id="t1", session_id="s1", action_name="write",
        risk=RiskLevel.HIGH,
    )
    # 监控不可用，但准入仍由规范账独立判定为 ask
    decision = admit_with_trust(request, view, trust="unknown")
    assert decision.mode == "ask"
    assert decision.required_acts


def test_visible_degradation():
    """I38: trust != full 必留降级事件；状态损坏视为 L4/unknown."""
    monitor = VigilMonitor("v1", window=5, error_threshold=0.2)
    for _ in range(2):
        monitor.record_job(False)
    assert monitor.status().trust != "full"
    assert monitor.degradation_events, "降级必须可见，禁止静默"

    monitor._state = object()  # type: ignore[assignment]
    broken = monitor.status()
    assert broken.state == "L4" and broken.trust == "unknown"


def test_bounded_restart():
    """I39: 有界退避；预算耗尽停 L3，不得无限重启."""
    monitor = VigilMonitor("v1", max_restarts=2, restart_window_s=600)
    monitor.quarantine("runaway")
    assert monitor.restart("2026-01-01T00:00:00+00:00") is True
    monitor.quarantine("runaway")
    assert monitor.restart("2026-01-01T00:00:01+00:00") is True
    monitor.quarantine("runaway")
    assert monitor.restart("2026-01-01T00:00:02+00:00") is False
    assert monitor.status().state == "L3"
    assert any(
        e.trigger == "restart_budget_exhausted"
        for e in monitor.degradation_events
    )


def test_tenant_isolation_and_quarantine():
    """I40: 租户各自独立；越权/失控 → L4，且队列拒绝服务."""
    a = VigilMonitor("a", tenant_id="tenant-a")
    b = VigilMonitor("b", tenant_id="tenant-b")
    a.quarantine("unauthorized_write")
    assert a.status().state == "L4"
    assert b.status().state == "L0", "单租户故障不得污染其他租户"
    assert a.submit("job") == "DROPPED"
    assert b.submit("job") == "job"


def test_monitor_latency_p99_small():
    """I36 补充: 单次 status 延迟远低于 5% 任务预算（此处用绝对上界近似）."""
    monitor = VigilMonitor("v1")
    samples: list[float] = []
    for _ in range(500):
        start = time.perf_counter()
        monitor.status()
        samples.append(time.perf_counter() - start)
    samples.sort()
    p99 = samples[int(len(samples) * 0.99)]
    assert p99 < 0.005, f"status p99={p99 * 1000:.2f}ms 过高"
