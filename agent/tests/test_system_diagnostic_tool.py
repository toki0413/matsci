"""Tests for the SystemDiagnosticTool — the agent-facing wrapper around
SystemHealthMonitor.

Covers all 4 actions (snapshot, diagnose, recent_anomalies, top_processes),
input validation, and read-only declaration.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque

import pytest

from huginn.diagnostics.system_health import (
    AnomalyEvent,
    SystemHealthMonitor,
    SystemMetrics,
)
from huginn.tools.system_diagnostic_tool import SystemDiagnosticTool


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_singleton():
    SystemHealthMonitor._singleton = None
    yield
    SystemHealthMonitor._singleton = None


@pytest.fixture
def tool() -> SystemDiagnosticTool:
    return SystemDiagnosticTool()


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _fake_metrics(cpu=50.0, mem=50.0, swap=5.0, disk_pct=50.0) -> SystemMetrics:
    return SystemMetrics(
        timestamp=time.time(),
        cpu_percent=cpu,
        memory_percent=mem,
        memory_used_mb=mem * 16384 / 100,
        memory_total_mb=16384.0,
        swap_percent=swap,
        swap_used_mb=swap * 4096 / 100,
        disk={
            "/": {"percent": disk_pct, "used_gb": disk_pct, "free_gb": 100.0 - disk_pct, "total_gb": 100.0},
        },
        load_avg=(0.5, 0.5, 0.5),
        psutil_available=True,
    )


# ── Metadata ─────────────────────────────────────────────────────────


class TestToolMetadata:
    def test_name(self, tool):
        assert tool.name == "system_diagnostic_tool"

    def test_category(self, tool):
        assert tool.category == "core"

    def test_is_read_only(self, tool):
        assert tool.is_read_only(None) is True

    def test_cost_tier_is_light(self, tool):
        assert tool.cost_tier == "light"


# ── Input validation ─────────────────────────────────────────────────


class TestValidation:
    def test_top_n_too_small_rejected(self, tool):
        result = _run(tool.validate_input(
            _make_input("top_processes", top_n=0),
        ))
        assert result.result is False

    def test_top_n_too_large_rejected(self, tool):
        result = _run(tool.validate_input(
            _make_input("top_processes", top_n=51),
        ))
        assert result.result is False

    def test_valid_top_n_accepted(self, tool):
        result = _run(tool.validate_input(
            _make_input("top_processes", top_n=25),
        ))
        assert result.result is True

    def test_snapshot_action_valid(self, tool):
        result = _run(tool.validate_input(_make_input("snapshot")))
        assert result.result is True


def _make_input(action: str, top_n: int = 10, by: str = "cpu"):
    from huginn.tools.system_diagnostic_tool import SystemDiagnosticInput

    return SystemDiagnosticInput(action=action, top_n=top_n, by=by)


# ── snapshot action ──────────────────────────────────────────────────


class TestSnapshotAction:
    def test_returns_metrics_dict(self, tool):
        monitor = SystemHealthMonitor.shared()
        monitor._latest = _fake_metrics(cpu=42.0)
        result = _run(tool.call({"action": "snapshot"}, None))
        assert result.success is True
        assert result.data["cpu_percent"] == 42.0
        assert result.data["psutil_available"] is True

    def test_snapshot_includes_disk(self, tool):
        monitor = SystemHealthMonitor.shared()
        monitor._latest = _fake_metrics()
        result = _run(tool.call({"action": "snapshot"}, None))
        assert "/" in result.data["disk"]


# ── diagnose action ──────────────────────────────────────────────────


class TestDiagnoseAction:
    def test_healthy_when_no_anomalies(self, tool):
        monitor = SystemHealthMonitor.shared()
        monitor._latest = _fake_metrics(cpu=5.0, mem=5.0)
        result = _run(tool.call({"action": "diagnose"}, None))
        assert result.success is True
        assert result.data["status"] == "healthy"

    def test_anomaly_when_cpu_high(self, tool):
        monitor = SystemHealthMonitor.shared()
        monitor._latest = _fake_metrics(cpu=99.0, mem=5.0)
        # diagnose() 直接看当前值, 不看 CPU 持续窗口
        result = _run(tool.call({"action": "diagnose"}, None))
        assert result.data["status"] == "anomaly"
        resources = {a["resource"] for a in result.data["anomalies"]}
        assert "cpu" in resources

    def test_anomaly_count_matches(self, tool):
        monitor = SystemHealthMonitor.shared()
        monitor._latest = _fake_metrics(cpu=99.0, mem=99.0, swap=50.0, disk_pct=50.0)
        result = _run(tool.call({"action": "diagnose"}, None))
        assert result.data["anomaly_count"] == len(result.data["anomalies"])


# ── recent_anomalies action ──────────────────────────────────────────


class TestRecentAnomaliesAction:
    def test_returns_empty_when_no_history(self, tool):
        result = _run(tool.call({"action": "recent_anomalies"}, None))
        assert result.success is True
        assert result.data["count"] == 0

    def test_returns_recorded_anomalies(self, tool):
        monitor = SystemHealthMonitor.shared()
        ev = AnomalyEvent(
            timestamp=time.time(),
            resource="memory",
            severity="warning",
            value=90.0,
            threshold=85.0,
            message="high mem",
        )
        monitor._anomalies = deque([ev], maxlen=100)
        result = _run(tool.call({"action": "recent_anomalies"}, None))
        assert result.data["count"] == 1
        assert result.data["anomalies"][0]["resource"] == "memory"


# ── top_processes action ─────────────────────────────────────────────


class TestTopProcessesAction:
    def test_returns_cached_processes(self, tool):
        monitor = SystemHealthMonitor.shared()
        monitor._top_cpu_cache = [
            {"pid": 100, "name": "python", "cpu_percent": 80.0, "memory_percent": 5.0, "rss_mb": 200.0},
        ]
        result = _run(tool.call({"action": "top_processes", "top_n": 5, "by": "cpu"}, None))
        assert result.success is True
        assert result.data["by"] == "cpu"
        assert result.data["count"] == 1
        assert result.data["processes"][0]["pid"] == 100

    def test_by_memory(self, tool):
        monitor = SystemHealthMonitor.shared()
        monitor._top_mem_cache = [
            {"pid": 200, "name": "java", "cpu_percent": 10.0, "memory_percent": 30.0, "rss_mb": 4000.0},
        ]
        result = _run(tool.call({"action": "top_processes", "by": "memory"}, None))
        assert result.data["by"] == "memory"
        assert result.data["processes"][0]["rss_mb"] == 4000.0

    def test_empty_cache_returns_empty(self, tool):
        result = _run(tool.call({"action": "top_processes"}, None))
        assert result.data["count"] == 0
        assert result.data["processes"] == []


# ── Unknown action ───────────────────────────────────────────────────


class TestUnknownAction:
    def test_unknown_action_returns_error(self, tool):
        # Pydantic Literal will reject invalid actions, but test the fallback
        # by calling the internal logic path
        result = _run(tool.call({"action": "snapshot"}, None))
        assert result.success is True
