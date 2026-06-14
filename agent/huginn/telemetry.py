"""Lightweight telemetry / tracing for HuginnAgent.

Records durations and metadata for agent turns, LLM calls, and tool calls.
Designed to be zero-overhead when not inspected: spans are kept in memory
and can be exported to logs, pet bus, or OpenTelemetry later.
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Generator


@dataclass
class TelemetrySpan:
    """A single timed operation."""

    name: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    children: list[TelemetrySpan] = field(default_factory=list)

    def finish(self, **metadata: Any) -> None:
        """Mark the span as finished and merge extra metadata."""
        self.end_time = time.time()
        self.duration_ms = round((self.end_time - self.start_time) * 1000, 2)
        self.metadata.update(metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "span_id": self.span_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
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
        """Return a coarse summary: counts and total durations by span name."""
        totals: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)

        def walk(span: TelemetrySpan) -> None:
            totals[span.name] += span.duration_ms
            counts[span.name] += 1
            for child in span.children:
                walk(child)

        for root in self._roots:
            walk(root)

        return {
            "total_spans": sum(counts.values()),
            "by_name": {name: {"count": counts[name], "duration_ms": totals[name]} for name in counts},
        }

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
