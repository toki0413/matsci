"""Diagnostics endpoint — unified observability dashboard.

Aggregates all health signals into a single endpoint:
  * System resources (CPU, memory, disk via psutil)
  * Tool health (per-tool success rates from HealthDashboard)
  * Circuit breaker states
  * Active sessions and thread count
  * Telemetry span summary
  * Plugin status (loaded/failed)

GET /diagnostics         — full report
GET /diagnostics/tools   — tool health breakdown
GET /diagnostics/circuit  — circuit breaker states
GET /diagnostics/trace    — recent telemetry spans
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any

from fastapi import APIRouter, Depends, Response

from huginn.security.auth import require_admin_key

router = APIRouter(tags=["diagnostics"])


@router.get("/diagnostics")
async def diagnostics(response: Response) -> dict[str, Any]:
    """Unified diagnostics report covering all subsystems."""
    report: dict[str, Any] = {
        "timestamp": time.time(),
        "python_version": sys.version.split()[0],
        "pid": os.getpid(),
        "thread_count": threading.active_count(),
    }

    # ── System resources ──────────────────────────────────────────
    report["system"] = _collect_system_metrics()

    # ── Tool health ───────────────────────────────────────────────
    report["tools"] = _collect_tool_health()

    # ── Circuit breakers ───────────────────────────────────────────
    report["circuit_breakers"] = _collect_circuit_states()

    # ── Telemetry summary ──────────────────────────────────────────
    report["telemetry"] = _collect_telemetry_summary()

    # ── Plugin status ──────────────────────────────────────────────
    report["plugins"] = _collect_plugin_status()

    # ── Overall health verdict ─────────────────────────────────────
    report["verdict"] = _compute_verdict(report)

    return report


@router.get("/diagnostics/tools")
async def diagnostics_tools() -> dict[str, Any]:
    """Per-tool health breakdown from HealthDashboard."""
    return {"tools": _collect_tool_health()}


@router.get("/diagnostics/circuit")
async def diagnostics_circuit() -> dict[str, Any]:
    """Circuit breaker states for all tools."""
    return {"circuit_breakers": _collect_circuit_states()}


@router.get("/diagnostics/trace")
async def diagnostics_trace(limit: int = 20) -> dict[str, Any]:
    """Recent telemetry spans for request tracing."""
    return {"trace": _collect_trace(limit)}


# ── Collectors ────────────────────────────────────────────────────


def _collect_system_metrics() -> dict[str, Any]:
    """Gather CPU/memory/disk metrics via psutil if available."""
    try:
        from huginn.diagnostics.system_health import SystemHealthMonitor

        monitor = SystemHealthMonitor.shared()
        snapshot = monitor.snapshot()
        if snapshot and snapshot.psutil_available:
            return {
                "cpu_percent": snapshot.cpu_percent,
                "memory_percent": snapshot.memory_percent,
                "memory_used_mb": round(snapshot.memory_used_mb, 1),
                "memory_total_mb": round(snapshot.memory_total_mb, 1),
                "swap_percent": snapshot.swap_percent,
                "disk": snapshot.disk,
                "load_avg": snapshot.load_avg,
            }
    except Exception:
        pass

    # Fallback: basic process info (cross-platform)
    try:
        import psutil as _ps

        proc = _ps.Process(os.getpid())
        mem_info = proc.memory_info()
        return {
            "process_rss_mb": round(mem_info.rss / 1024 / 1024, 1),
            "psutil_available": True,
        }
    except Exception:
        return {"psutil_available": False}


def _collect_tool_health() -> dict[str, Any]:
    """Per-tool health from the shared HealthDashboard."""
    try:
        from huginn.agents.health_dashboard import HealthDashboard

        dash = HealthDashboard.shared()
        all_health: dict[str, Any] = {}
        # Get all tool names from the internal records
        with dash._lock:
            tool_names = list(dash._records.keys())
        for name in tool_names:
            all_health[name] = dash.get_health(name)
        return all_health
    except Exception:
        return {}


def _collect_circuit_states() -> dict[str, Any]:
    """Circuit breaker states for all tracked tools."""
    try:
        from huginn.agents.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker.shared()
        states: dict[str, Any] = {}
        with breaker._lock:
            for tool_name, st in breaker._states.items():
                states[tool_name] = {
                    "state": st.state,
                    "failure_count": st.failure_count,
                    "last_failure": st.last_failure_time,
                    "half_open_trials": st.half_open_trials,
                }
        return states
    except Exception:
        return {}


def _collect_telemetry_summary() -> dict[str, Any]:
    """Summary of recent telemetry spans."""
    try:
        from huginn.telemetry import get_telemetry_collector

        collector = get_telemetry_collector()
        return collector.summary()
    except Exception:
        return {"available": False}


def _collect_trace(limit: int = 20) -> list[dict[str, Any]]:
    """Recent telemetry spans as a list."""
    try:
        from huginn.telemetry import get_telemetry_collector

        collector = get_telemetry_collector()
        roots = collector.to_dict()
        # Flatten the tree into a list for easy consumption
        flat: list[dict[str, Any]] = []

        def flatten(spans: list[dict[str, Any]], depth: int = 0) -> None:
            for s in spans:
                entry = {
                    "name": s.get("name"),
                    "duration_ms": round(s.get("duration_ms", 0), 2),
                    "depth": depth,
                    "metadata": s.get("metadata", {}),
                }
                flat.append(entry)
                flatten(s.get("children", []), depth + 1)

        flatten(roots[-limit:])
        return flat
    except Exception:
        return []


def _collect_plugin_status() -> dict[str, Any]:
    """Status of loaded plugins from the shared registry."""
    try:
        from huginn.plugins.registry import get_shared_registry

        registry = get_shared_registry()
        handlers = registry.list_handlers()
        plugins: dict[str, Any] = {}
        for h in handlers:
            name = getattr(h, "plugin_name", "unknown")
            if name not in plugins:
                plugins[name] = {
                    "enabled": not getattr(h, "disabled", False),
                    "handler_count": 0,
                    "priority": getattr(h, "priority", 0),
                }
            plugins[name]["handler_count"] += 1
        return plugins
    except Exception:
        return {}


def _compute_verdict(report: dict[str, Any]) -> str:
    """Compute an overall health verdict from subsystem signals."""
    issues: list[str] = []

    # Check circuit breakers
    cb = report.get("circuit_breakers", {})
    open_count = sum(1 for v in cb.values() if v.get("state") == "open")
    if open_count > 0:
        issues.append(f"{open_count} circuit(s) open")

    # Check tool health
    tools = report.get("tools", {})
    unhealthy = sum(1 for v in tools.values() if v.get("verdict") == "unhealthy")
    if unhealthy > 0:
        issues.append(f"{unhealthy} tool(s) unhealthy")

    # Check system resources
    sys_metrics = report.get("system", {})
    if sys_metrics.get("cpu_percent", 0) > 90:
        issues.append("CPU > 90%")
    if sys_metrics.get("memory_percent", 0) > 90:
        issues.append("Memory > 90%")

    if not issues:
        return "healthy"
    if len(issues) <= 2:
        return "degraded"
    return "unhealthy"
