"""Lightweight telemetry / tracing for HuginnAgent.

Records durations and metadata for agent turns, LLM calls, and tool calls.
Designed to be zero-overhead when not inspected: spans are kept in memory
and can be exported to logs, pet bus, or OpenTelemetry later.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _get_process_memory_mb() -> float:
    """Best-effort current process RSS in MB.

    Tries psutil first (accurate cross-platform RSS), then ``resource``
    (Unix only), then ``tracemalloc`` (Python allocations only). Returns
    0.0 if nothing works — never raises.
    """
    # psutil is the most reliable but is an optional dependency.
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024.0 / 1024.0
    except Exception:
        logger.debug("psutil 不可用, 尝试下一个内存采样方式", exc_info=True)

    # resource is stdlib on Unix; ru_maxrss is KB on Linux, bytes on macOS.
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        logger.debug("resource 模块不可用, 尝试下一个内存采样方式", exc_info=True)

    # tracemalloc only sees Python-level allocations, not native extensions,
    # but it's a usable fallback when the others are unavailable.
    try:
        import tracemalloc

        if tracemalloc.is_tracing():
            current, _ = tracemalloc.get_traced_memory()
            return current / 1024.0 / 1024.0
    except Exception:
        logger.debug("tracemalloc 当前内存采样失败", exc_info=True)

    return 0.0


def _get_peak_memory_mb(start_mb: float, end_mb: float) -> float:
    """Best-effort peak memory in MB.

    Uses the tracemalloc high-water mark when tracing is active, otherwise
    falls back to the larger of the two point samples taken at span start
    and end.
    """
    try:
        import tracemalloc

        if tracemalloc.is_tracing():
            _, peak = tracemalloc.get_traced_memory()
            return peak / 1024.0 / 1024.0
    except Exception:
        logger.debug("tracemalloc 峰值内存采样失败", exc_info=True)
    return max(start_mb, end_mb)


@dataclass
class TelemetrySpan:
    """A single timed operation."""

    name: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    duration_ms: float = 0.0
    # Sampled at start/end so callers can see how much a span grew the heap.
    memory_start_mb: float = field(default_factory=_get_process_memory_mb)
    memory_end_mb: float = 0.0
    memory_peak_mb: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    children: list[TelemetrySpan] = field(default_factory=list)

    def finish(self, **metadata: Any) -> None:
        """Mark the span as finished and merge extra metadata."""
        self.end_time = time.time()
        self.duration_ms = round((self.end_time - self.start_time) * 1000, 2)
        self.memory_end_mb = _get_process_memory_mb()
        self.memory_peak_mb = _get_peak_memory_mb(
            self.memory_start_mb, self.memory_end_mb
        )
        self.metadata.update(metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "span_id": self.span_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "memory_start_mb": self.memory_start_mb,
            "memory_end_mb": self.memory_end_mb,
            "memory_peak_mb": self.memory_peak_mb,
            "metadata": self.metadata,
            "children": [c.to_dict() for c in self.children],
        }


class TelemetryCollector:
    """In-memory collector for Huginn telemetry spans."""

    def __init__(self) -> None:
        self._roots: list[TelemetrySpan] = []
        self._current_stack: list[TelemetrySpan] = []

    @contextmanager
    def span(
        self,
        name: str,
        **metadata: Any,
    ) -> Generator[TelemetrySpan, None, None]:
        """Context manager that records a span and attaches it to the active parent."""
        span = TelemetrySpan(name=name, metadata=dict(metadata))
        parent = self._current_stack[-1] if self._current_stack else None
        if parent is not None:
            parent.children.append(span)
        else:
            self._roots.append(span)

        self._current_stack.append(span)
        try:
            yield span
        finally:
            span.finish()
            self._current_stack.pop()

    def current_span(self) -> TelemetrySpan | None:
        """Return the currently active span, if any."""
        return self._current_stack[-1] if self._current_stack else None

    def add_event(self, name: str, **metadata: Any) -> None:
        """Add a zero-duration event under the current span."""
        parent = self.current_span()
        event = TelemetrySpan(name=name, metadata=dict(metadata))
        event.finish()
        if parent is not None:
            parent.children.append(event)
        else:
            self._roots.append(event)

    def to_dict(self) -> list[dict[str, Any]]:
        """Return all root spans as dicts."""
        return [r.to_dict() for r in self._roots]

    def summary(self) -> dict[str, Any]:
        """Return a coarse summary: counts, durations, and memory by span name."""
        totals: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        memory_deltas: dict[str, list[float]] = defaultdict(list)
        memory_peaks: dict[str, list[float]] = defaultdict(list)

        def walk(span: TelemetrySpan) -> None:
            totals[span.name] += span.duration_ms
            counts[span.name] += 1
            memory_deltas[span.name].append(
                span.memory_end_mb - span.memory_start_mb
            )
            memory_peaks[span.name].append(span.memory_peak_mb)
            for child in span.children:
                walk(child)

        for root in self._roots:
            walk(root)

        return {
            "total_spans": sum(counts.values()),
            "by_name": {
                name: {
                    "count": counts[name],
                    "duration_ms": totals[name],
                    "avg_memory_delta_mb": round(
                        sum(memory_deltas[name]) / counts[name], 3
                    ),
                    "max_memory_peak_mb": round(max(memory_peaks[name]), 3),
                }
                for name in counts
            },
        }

    def memory_snapshot(self) -> dict[str, Any]:
        """Point-in-time memory snapshot for ad-hoc monitoring.

        Returns current RSS, plus traced current/peak when tracemalloc is
        active. Safe to call from a periodic task or around heavy ops.
        """
        snapshot: dict[str, Any] = {"rss_mb": _get_process_memory_mb()}
        try:
            import tracemalloc

            if tracemalloc.is_tracing():
                current, peak = tracemalloc.get_traced_memory()
                snapshot["traced_current_mb"] = current / 1024.0 / 1024.0
                snapshot["traced_peak_mb"] = peak / 1024.0 / 1024.0
        except Exception:
            logger.debug("memory_snapshot tracemalloc 采样失败", exc_info=True)
        return snapshot

    def clear(self) -> None:
        """Drop all recorded spans."""
        self._roots.clear()
        self._current_stack.clear()


# Per-context collector. In a single-threaded/async agent this is usually one
# collector; multi-tenant servers can override with a per-request collector.
_telemetry_ctx: ContextVar[TelemetryCollector | None] = ContextVar(
    "huginn_telemetry_collector", default=None
)


def get_telemetry_collector() -> TelemetryCollector:
    """Get or create the current telemetry collector."""
    collector = _telemetry_ctx.get()
    if collector is None:
        collector = TelemetryCollector()
        _telemetry_ctx.set(collector)
    return collector


def set_telemetry_collector(collector: TelemetryCollector | None) -> None:
    """Replace the current telemetry collector."""
    _telemetry_ctx.set(collector)


class NullTelemetryCollector(TelemetryCollector):
    """Telemetry collector that discards all spans.

    Used when telemetry is disabled so that instrumented call sites can keep
    using ``span()`` and ``add_event()`` without branching.
    """

    def __init__(self) -> None:
        super().__init__()

    @contextmanager
    def span(
        self,
        name: str,
        **metadata: Any,
    ) -> Generator[TelemetrySpan, None, None]:
        # Yield a throwaway finished span so callers can still inspect metadata.
        span = TelemetrySpan(name=name, metadata=dict(metadata))
        span.finish()
        try:
            yield span
        finally:
            pass

    def add_event(self, name: str, **metadata: Any) -> None:
        pass

    def to_dict(self) -> list[dict[str, Any]]:
        return []

    def summary(self) -> dict[str, Any]:
        return {"total_spans": 0, "by_name": {}}

    def clear(self) -> None:
        pass


# ── Trajectory 序列化 ────────────────────────────────────────────────


def _extract_tool_calls(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """从 spans 树里按出现顺序抽出 tool_call 序列。

    工具调用 span 由 ToolAdapter 在执行工具时创建, metadata 里带 tool 名、
    success 状态, 可选带 args / result / error。这里做 best-effort 提取,
    缺的字段就留 None, 不报错。
    """
    calls: list[dict[str, Any]] = []
    step = 0

    def walk(span: dict[str, Any]) -> None:
        nonlocal step
        if span.get("name") == "tool_call":
            step += 1
            meta = span.get("metadata", {}) or {}
            calls.append(
                {
                    "step": step,
                    "tool": meta.get("tool", "unknown"),
                    "args": meta.get("args"),
                    "result": meta.get("result"),
                    "duration_ms": span.get("duration_ms", 0.0),
                    "success": meta.get("success", True),
                    "error": meta.get("error"),
                    "span_id": span.get("span_id"),
                }
            )
        for child in span.get("children", []) or []:
            walk(child)

    for root in spans:
        walk(root)
    return calls


def save_trajectory(
    collector: TelemetryCollector,
    path: str | Path,
    metadata: dict | None = None,
) -> Path:
    """把 telemetry 数据 + 工具调用轨迹保存为 JSON 文件。

    文件结构:
        {
            "version": "1.0",
            "timestamp": "2025-01-15T14:30:00",
            "metadata": {...},
            "spans": [...],          # collector.to_dict()
            "tool_calls": [...],     # 从 spans 里提取的工具调用序列
            "summary": {...},        # collector.summary()
        }

    args/result 只有在 ToolAdapter 把它们写进 span.metadata 时才有值,
    否则是 None。这是为了让回放能看到工具决策的输入输出, 而不是只看耗时。
    """
    spans = collector.to_dict()
    payload = {
        "version": "1.0",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "metadata": dict(metadata) if metadata else {},
        "spans": spans,
        "tool_calls": _extract_tool_calls(spans),
        "summary": collector.summary(),
    }

    out = Path(path)
    # 父目录不存在就建一下, 免得调用方还得自己 mkdir
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return out


def load_trajectory(path: str | Path) -> dict[str, Any]:
    """加载 save_trajectory 写出的轨迹文件。

    返回的 dict 结构跟 save_trajectory 的 payload 一致, 直接取
    ``data["tool_calls"]`` 就能拿到工具调用序列。
    """
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
