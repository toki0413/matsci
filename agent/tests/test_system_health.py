"""Tests for the system resource health monitor.

Covers threshold detection, sustained CPU window, diagnosis logic,
top_processes caching, auto-remediation gating, and graceful degradation
when psutil is unavailable.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from huginn.diagnostics.system_health import (
    AnomalyEvent,
    SystemHealthMonitor,
    SystemMetrics,
    ThresholdPolicy,
)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_singleton():
    """每个测试用独立的 monitor 实例，避免单例状态串扰。"""
    SystemHealthMonitor._singleton = None
    yield
    SystemHealthMonitor._singleton = None


@pytest.fixture
def strict_monitor() -> SystemHealthMonitor:
    """阈值很低的 monitor，方便触发异常。"""
    policy = ThresholdPolicy(
        cpu_percent=10.0,
        cpu_sustained_seconds=1.0,
        memory_percent=10.0,
        disk_percent=10.0,
        swap_percent=10.0,
        cpu_critical_percent=90.0,
        memory_critical_percent=90.0,
        disk_critical_percent=95.0,
    )
    return SystemHealthMonitor(policy=policy, poll_interval=0.1)


def _fake_metrics(
    cpu=50.0,
    mem=50.0,
    swap=5.0,
    disk_pct=50.0,
    mem_total=16384.0,
) -> SystemMetrics:
    """构造一个假的 SystemMetrics，跳过 psutil。"""
    return SystemMetrics(
        timestamp=time.time(),
        cpu_percent=cpu,
        memory_percent=mem,
        memory_used_mb=mem * mem_total / 100,
        memory_total_mb=mem_total,
        swap_percent=swap,
        swap_used_mb=swap * 4096 / 100,
        disk={
            "/": {
                "percent": disk_pct,
                "used_gb": disk_pct * 100 / 100,
                "free_gb": (100 - disk_pct) * 100 / 100,
                "total_gb": 100.0,
            }
        },
        load_avg=(0.5, 0.5, 0.5),
        psutil_available=True,
    )


# ── ThresholdPolicy ──────────────────────────────────────────────────


class TestThresholdPolicy:
    def test_defaults_are_balanced(self):
        p = ThresholdPolicy()
        assert p.cpu_percent == 85.0
        assert p.cpu_sustained_seconds == 30.0
        assert p.memory_percent == 85.0
        assert p.disk_percent == 90.0
        assert p.swap_percent == 80.0


# ── SystemMetrics ────────────────────────────────────────────────────


class TestSystemMetrics:
    def test_to_dict_rounds_values(self):
        m = _fake_metrics(cpu=73.456, mem=60.789)
        d = m.to_dict()
        assert d["cpu_percent"] == 73.5
        assert d["memory_percent"] == 60.8
        assert d["psutil_available"] is True

    def test_to_dict_disk_rounding(self):
        m = _fake_metrics(disk_pct=42.567)
        d = m.to_dict()
        assert d["disk"]["/"]["percent"] == 42.57


# ── AnomalyEvent ─────────────────────────────────────────────────────


class TestAnomalyEvent:
    def test_to_dict_includes_all_fields(self):
        ev = AnomalyEvent(
            timestamp=1000.0,
            resource="cpu",
            severity="warning",
            value=90.0,
            threshold=85.0,
            message="high cpu",
            evidence={"top_processes": []},
            recommendations=["reduce parallelism"],
            auto_fixed=False,
        )
        d = ev.to_dict()
        assert d["resource"] == "cpu"
        assert d["severity"] == "warning"
        assert d["value"] == 90.0
        assert d["auto_fixed"] is False
        assert d["recommendations"] == ["reduce parallelism"]


# ── Diagnosis logic ──────────────────────────────────────────────────


class TestDiagnosis:
    def test_diagnose_cpu_above_threshold(self, strict_monitor):
        m = _fake_metrics(cpu=95.0, mem=5.0, swap=5.0, disk_pct=5.0)
        strict_monitor._latest = m
        events = strict_monitor.diagnose()
        cpu_events = [e for e in events if e.resource == "cpu"]
        assert len(cpu_events) == 1
        assert cpu_events[0].severity == "critical"
        assert cpu_events[0].value == 95.0
        assert len(cpu_events[0].recommendations) > 0

    def test_diagnose_cpu_below_threshold(self, strict_monitor):
        m = _fake_metrics(cpu=5.0, mem=5.0, swap=5.0, disk_pct=5.0)
        strict_monitor._latest = m
        events = strict_monitor.diagnose()
        assert events == []

    def test_diagnose_memory_warning(self, strict_monitor):
        m = _fake_metrics(cpu=5.0, mem=50.0, swap=5.0, disk_pct=5.0)
        strict_monitor._latest = m
        events = strict_monitor.diagnose()
        mem_events = [e for e in events if e.resource == "memory"]
        assert len(mem_events) == 1
        assert mem_events[0].severity == "warning"

    def test_diagnose_memory_critical(self, strict_monitor):
        m = _fake_metrics(cpu=5.0, mem=95.0, swap=5.0, disk_pct=5.0)
        strict_monitor._latest = m
        events = strict_monitor.diagnose()
        mem_events = [e for e in events if e.resource == "memory"]
        assert len(mem_events) == 1
        assert mem_events[0].severity == "critical"

    def test_diagnose_disk_multiple_partitions(self, strict_monitor):
        m = _fake_metrics(cpu=5.0, mem=5.0, swap=5.0, disk_pct=50.0)
        m.disk["/data"] = {
            "percent": 50.0,
            "used_gb": 50.0,
            "free_gb": 50.0,
            "total_gb": 100.0,
        }
        strict_monitor._latest = m
        events = strict_monitor.diagnose()
        disk_events = [e for e in events if e.resource == "disk"]
        assert len(disk_events) == 2

    def test_diagnose_swap(self, strict_monitor):
        m = _fake_metrics(cpu=5.0, mem=5.0, swap=50.0, disk_pct=5.0)
        strict_monitor._latest = m
        events = strict_monitor.diagnose()
        swap_events = [e for e in events if e.resource == "swap"]
        assert len(swap_events) == 1

    def test_diagnose_returns_multiple_anomalies(self, strict_monitor):
        m = _fake_metrics(cpu=95.0, mem=95.0, swap=50.0, disk_pct=50.0)
        strict_monitor._latest = m
        events = strict_monitor.diagnose()
        resources = {e.resource for e in events}
        assert resources == {"cpu", "memory", "disk", "swap"}

    def test_diagnose_psutil_unavailable_returns_empty(self, strict_monitor):
        m = _fake_metrics()
        m.psutil_available = False
        strict_monitor._latest = m
        events = strict_monitor.diagnose()
        assert events == []


# ── Top processes ────────────────────────────────────────────────────


class TestTopProcesses:
    def test_top_processes_returns_cache(self, strict_monitor):
        strict_monitor._top_cpu_cache = [
            {"pid": 1, "name": "a", "cpu_percent": 50.0, "memory_percent": 1.0, "rss_mb": 10.0},
            {"pid": 2, "name": "b", "cpu_percent": 30.0, "memory_percent": 2.0, "rss_mb": 20.0},
        ]
        result = strict_monitor.top_processes(n=1, by="cpu")
        assert len(result) == 1
        assert result[0]["pid"] == 1

    def test_top_processes_memory(self, strict_monitor):
        strict_monitor._top_mem_cache = [
            {"pid": 1, "name": "a", "cpu_percent": 50.0, "memory_percent": 5.0, "rss_mb": 100.0},
        ]
        result = strict_monitor.top_processes(n=5, by="memory")
        assert len(result) == 1
        assert result[0]["rss_mb"] == 100.0

    def test_top_processes_empty_cache(self, strict_monitor):
        result = strict_monitor.top_processes(n=10, by="cpu")
        assert result == []


# ── Recent anomalies ─────────────────────────────────────────────────


class TestRecentAnomalies:
    def test_recent_anomalies_returns_newest_first(self, strict_monitor):
        from collections import deque

        ev1 = AnomalyEvent(timestamp=1.0, resource="cpu", severity="warning", value=90, threshold=85, message="m1")
        ev2 = AnomalyEvent(timestamp=2.0, resource="memory", severity="warning", value=90, threshold=85, message="m2")
        strict_monitor._anomalies = deque([ev1, ev2], maxlen=100)
        result = strict_monitor.recent_anomalies(limit=10)
        assert result[0].timestamp == 2.0
        assert result[1].timestamp == 1.0

    def test_recent_anomalies_limit(self, strict_monitor):
        from collections import deque

        events = [
            AnomalyEvent(timestamp=float(i), resource="cpu", severity="warning", value=90, threshold=85, message=f"m{i}")
            for i in range(50)
        ]
        strict_monitor._anomalies = deque(events, maxlen=100)
        result = strict_monitor.recent_anomalies(limit=5)
        assert len(result) == 5
        assert result[0].timestamp == 49.0


# ── Sustained CPU window ─────────────────────────────────────────────


class TestSustainedCpuWindow:
    def test_cpu_anomaly_needs_full_window(self):
        """CPU 没填满窗口不算持续异常。"""
        policy = ThresholdPolicy(
            cpu_percent=10.0,
            cpu_sustained_seconds=1.0,
            memory_percent=99.0,
            disk_percent=99.0,
            swap_percent=99.0,
        )
        monitor = SystemHealthMonitor(policy=policy, poll_interval=0.5)
        # window_size = int(1.0 / 0.5) = 2, 只放 1 个点不够
        monitor._cpu_window.append(50.0)
        monitor._latest = _fake_metrics(cpu=50.0, mem=5.0, swap=5.0, disk_pct=5.0)
        # diagnose() 不看窗口, 直接看当前值; 但 _check_and_record 看窗口
        # 这里测 _check_and_record 不会记录 cpu 事件
        monitor._anomalies.clear()
        monitor._check_and_record(monitor._latest)
        cpu_anomalies = [e for e in monitor._anomalies if e.resource == "cpu"]
        assert len(cpu_anomalies) == 0

    def test_cpu_anomaly_triggers_when_window_full(self):
        """窗口填满且全超阈值, CPU 异常触发。"""
        policy = ThresholdPolicy(
            cpu_percent=10.0,
            cpu_sustained_seconds=1.0,
            memory_percent=99.0,
            disk_percent=99.0,
            swap_percent=99.0,
        )
        monitor = SystemHealthMonitor(policy=policy, poll_interval=0.5)
        # window_size = 2, 放 2 个点
        monitor._cpu_window.append(50.0)
        monitor._cpu_window.append(60.0)
        monitor._latest = _fake_metrics(cpu=60.0, mem=5.0, swap=5.0, disk_pct=5.0)
        monitor._anomalies.clear()
        monitor._check_and_record(monitor._latest)
        cpu_anomalies = [e for e in monitor._anomalies if e.resource == "cpu"]
        assert len(cpu_anomalies) == 1


# ── Auto-remediation gating ──────────────────────────────────────────


class TestAutoRemediation:
    def test_auto_fix_disabled_by_default(self, strict_monitor):
        """system_health_auto_fix 默认关, 不做任何修复。"""
        from huginn.feature_flags import FeatureFlags

        ff = FeatureFlags.shared()
        ff.reset("system_health_auto_fix")
        assert not ff.is_enabled("system_health_auto_fix")

        ev = AnomalyEvent(
            timestamp=time.time(),
            resource="cpu",
            severity="warning",
            value=95.0,
            threshold=85.0,
            message="high cpu",
        )
        strict_monitor._maybe_auto_fix(ev)
        assert ev.auto_fixed is False
        assert ev.auto_fix_detail == ""

    def test_auto_fix_cpu_trips_circuit_breaker(self, strict_monitor, monkeypatch):
        """auto_fix 开了 + 有不健康工具 → 强制熔断。"""
        from huginn.feature_flags import FeatureFlags

        ff = FeatureFlags.shared()
        ff.enable("system_health_auto_fix")
        ff.enable("system_health_monitor")
        try:
            # mock HealthDashboard 返回一个 unhealthy 工具
            from huginn.agents import health_dashboard as hd_mod

            class FakeDash:
                def get_all(self):
                    return [{"tool": "fake_heavy_tool", "verdict": "unhealthy"}]

            monkeypatch.setattr(hd_mod.HealthDashboard, "shared", lambda: FakeDash())

            ev = AnomalyEvent(
                timestamp=time.time(),
                resource="cpu",
                severity="critical",
                value=95.0,
                threshold=85.0,
                message="high cpu",
            )
            strict_monitor._maybe_auto_fix(ev)
            assert ev.auto_fixed is True
            assert "fake_heavy_tool" in ev.auto_fix_detail

            # 确认熔断器真的被 force_open 了
            from huginn.agents.circuit_breaker import CircuitBreaker

            breaker = CircuitBreaker.shared()
            assert breaker.get_state("fake_heavy_tool") == "open"
            breaker.reset("fake_heavy_tool")
        finally:
            ff.disable("system_health_auto_fix")

    def test_auto_fix_cooldown(self, strict_monitor, monkeypatch):
        """同一资源在冷却期内不重复修。"""
        from huginn.feature_flags import FeatureFlags

        ff = FeatureFlags.shared()
        ff.enable("system_health_auto_fix")
        ff.enable("system_health_monitor")
        try:
            from huginn.agents import health_dashboard as hd_mod

            class FakeDash:
                def get_all(self):
                    return [{"tool": "cool_tool", "verdict": "unhealthy"}]

            monkeypatch.setattr(hd_mod.HealthDashboard, "shared", lambda: FakeDash())

            ev1 = AnomalyEvent(
                timestamp=time.time(),
                resource="cpu",
                severity="critical",
                value=95.0,
                threshold=85.0,
                message="high cpu 1",
            )
            ev2 = AnomalyEvent(
                timestamp=time.time(),
                resource="cpu",
                severity="critical",
                value=96.0,
                threshold=85.0,
                message="high cpu 2",
            )
            strict_monitor._maybe_auto_fix(ev1)
            assert ev1.auto_fixed is True

            strict_monitor._maybe_auto_fix(ev2)
            assert ev2.auto_fixed is False

            from huginn.agents.circuit_breaker import CircuitBreaker

            CircuitBreaker.shared().reset("cool_tool")
        finally:
            ff.disable("system_health_auto_fix")


# ── psutil unavailable ───────────────────────────────────────────────


class TestPsutilUnavailable:
    def test_snapshot_returns_unavailable(self, monkeypatch):
        """psutil 没装时 snapshot 返回 psutil_available=False。"""
        from huginn.diagnostics import system_health as mod

        monkeypatch.setattr(mod, "_PSUTIL_OK", False)
        SystemHealthMonitor._singleton = None
        monitor = SystemHealthMonitor()
        m = monitor.snapshot()
        assert m.psutil_available is False
        assert m.cpu_percent == 0.0

    def test_start_does_not_crash_without_psutil(self, monkeypatch):
        from huginn.diagnostics import system_health as mod

        monkeypatch.setattr(mod, "_PSUTIL_OK", False)
        SystemHealthMonitor._singleton = None
        monitor = SystemHealthMonitor()
        monitor.start()  # 应该 no-op, 不报错
        assert not monitor.is_running()

    def test_diagnose_returns_empty_without_psutil(self, monkeypatch):
        from huginn.diagnostics import system_health as mod

        monkeypatch.setattr(mod, "_PSUTIL_OK", False)
        SystemHealthMonitor._singleton = None
        monitor = SystemHealthMonitor()
        assert monitor.diagnose() == []
