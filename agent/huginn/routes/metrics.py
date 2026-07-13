"""Prometheus metrics for the Huginn API server.

Exposes a ``/metrics`` endpoint in the Prometheus text exposition format,
plus the metric objects the request middleware and other instrumentation
points (agent turns, LLM tokens, tool calls, ...) increment.

We prefer the ``prometheus_client`` library when it is installed.  A small
pure-Python fallback that emits the same text format kicks in otherwise, so
the endpoint keeps working on minimal installs without the dependency.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import Response

# ---------------------------------------------------------------------------
# Metric primitives — either the real library or a self-contained fallback.
# ---------------------------------------------------------------------------
try:
    from prometheus_client import (  # type: ignore[import-not-found]
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    _HAS_PROMETHEUS = True
except ImportError:  # pragma: no cover - only hit when prometheus_client is absent
    _HAS_PROMETHEUS = False

    _REGISTRY: list["_FallbackMetric"] = []

    # Same buckets prometheus_client uses by default for request durations.
    _DEFAULT_BUCKETS = (
        0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25,
        0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 5.0, 7.5, 10.0,
    )

    class _FallbackMetric:
        metric_type = "untyped"

        def __init__(self, name: str, documentation: str, labelnames=(), **kwargs) -> None:
            self.name = name
            self.documentation = documentation
            self.labelnames = tuple(labelnames)
            self._lock = threading.Lock()
            self._children: dict[tuple, dict] = {}
            _REGISTRY.append(self)

        def _key(self, labels: dict[str, Any]) -> tuple:
            return tuple(str(labels.get(k, "")) for k in self.labelnames)

        def _child(self, key: tuple) -> dict:
            with self._lock:
                ch = self._children.get(key)
                if ch is None:
                    ch = self._new_child()
                    self._children[key] = ch
                return ch

        def _new_child(self) -> dict:  # pragma: no cover - overridden
            return {}

        def labels(self, **kw: Any) -> "_Child":
            return _Child(self, self._key(kw))

    class _Child:
        """A label-bound view onto a metric."""

        def __init__(self, metric: "_FallbackMetric", key: tuple) -> None:
            self._m = metric
            self._k = key

        def inc(self, amount: float = 1) -> None:
            self._m._inc(self._k, amount)

        def dec(self, amount: float = 1) -> None:
            self._m._dec(self._k, amount)

        def set(self, value: float) -> None:  # noqa: A003 - mirror lib API
            self._m._set(self._k, value)

        def observe(self, value: float) -> None:
            self._m._observe(self._k, value)

    class Counter(_FallbackMetric):  # type: ignore[no-redef]
        metric_type = "counter"

        def _new_child(self) -> dict:
            return {"value": 0.0}

        def _inc(self, key: tuple, amount: float) -> None:
            ch = self._child(key)
            ch["value"] += float(amount)

        def inc(self, amount: float = 1, **kw: Any) -> None:
            if kw:
                self.labels(**kw).inc(amount)
            else:
                self._inc((), amount)

    class Gauge(_FallbackMetric):  # type: ignore[no-redef]
        metric_type = "gauge"

        def _new_child(self) -> dict:
            return {"value": 0.0}

        def _inc(self, key: tuple, amount: float) -> None:
            self._child(key)["value"] += float(amount)

        def _dec(self, key: tuple, amount: float) -> None:
            self._child(key)["value"] -= float(amount)

        def _set(self, key: tuple, value: float) -> None:
            self._child(key)["value"] = float(value)

        def inc(self, amount: float = 1, **kw: Any) -> None:
            if kw:
                self.labels(**kw).inc(amount)
            else:
                self._inc((), amount)

        def dec(self, amount: float = 1, **kw: Any) -> None:
            if kw:
                self.labels(**kw).dec(amount)
            else:
                self._dec((), amount)

        def set(self, value: float, **kw: Any) -> None:  # noqa: A003
            if kw:
                self.labels(**kw).set(value)
            else:
                self._set((), value)

    class Histogram(_FallbackMetric):  # type: ignore[no-redef]
        metric_type = "histogram"

        def _new_child(self) -> dict:
            return {
                "buckets": [0] * len(_DEFAULT_BUCKETS),
                "sum": 0.0,
                "count": 0,
            }

        def _observe(self, key: tuple, value: float) -> None:
            ch = self._child(key)
            v = float(value)
            for i, bound in enumerate(_DEFAULT_BUCKETS):
                if v <= bound:
                    ch["buckets"][i] += 1
            ch["sum"] += v
            ch["count"] += 1

        def observe(self, value: float, **kw: Any) -> None:
            if kw:
                self.labels(**kw).observe(value)
            else:
                self._observe((), value)

    def _escape_label(value: Any) -> str:
        return (
            str(value)
            .replace("\\", "\\\\")
            .replace("\n", "\\n")
            .replace('"', '\\"')
        )

    def _format_labels(labelnames: tuple[str, ...], values: tuple) -> str:
        if not labelnames:
            return ""
        parts = [f'{n}="{_escape_label(v)}"' for n, v in zip(labelnames, values)]
        return "{" + ",".join(parts) + "}"

    def generate_latest() -> bytes:  # type: ignore[no-redef]
        lines: list[str] = []
        for metric in _REGISTRY:
            lines.append(f"# HELP {metric.name} {metric.documentation}")
            lines.append(f"# TYPE {metric.name} {metric.metric_type}")
            for key, state in metric._children.items():
                if isinstance(metric, Histogram):
                    for bound, count in zip(_DEFAULT_BUCKETS, state["buckets"]):
                        lbl = _format_labels(
                            metric.labelnames + ("le",), key + (str(bound),)
                        )
                        lines.append(f"{metric.name}_bucket{lbl} {count}")
                    inf_lbl = _format_labels(
                        metric.labelnames + ("le",), key + ("+Inf",)
                    )
                    lines.append(f"{metric.name}_bucket{inf_lbl} {state['count']}")
                    base = _format_labels(metric.labelnames, key)
                    lines.append(f"{metric.name}_sum{base} {state['sum']}")
                    lines.append(f"{metric.name}_count{base} {state['count']}")
                else:
                    base = _format_labels(metric.labelnames, key)
                    lines.append(f"{metric.name}{base} {state['value']}")
        return ("\n".join(lines) + "\n").encode("utf-8")

    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

# HTTP traffic — filled in by the request middleware in server.py.
REQUESTS_TOTAL = Counter(
    "huginn_requests_total",
    "Total HTTP requests processed by the server.",
    labelnames=("method", "path", "status"),
)
REQUEST_DURATION = Histogram(
    "huginn_request_duration_seconds",
    "HTTP request latency in seconds.",
    labelnames=("method", "path"),
)

# WebSocket fan-out.  Inc/dec from the WS handlers in routes/ws.py.
ACTIVE_WS_CONNECTIONS = Gauge(
    "huginn_active_websocket_connections",
    "Currently open WebSocket connections.",
)

# Agent activity.  Incremented per chat turn / tool call.
AGENT_TURNS_TOTAL = Counter(
    "huginn_agent_turns_total",
    "Agent chat turns processed.",
    labelnames=("thread_id",),
)
TOOL_CALLS_TOTAL = Counter(
    "huginn_tool_calls_total",
    "Tool invocations issued by the agent.",
    labelnames=("tool_name",),
)

# LLM consumption.  ``kind`` is prompt or completion tokens.
LLM_TOKENS_TOTAL = Counter(
    "huginn_llm_tokens_total",
    "LLM tokens consumed, partitioned by prompt/completion.",
    labelnames=("model", "kind"),
)
LLM_COST_USD = Gauge(
    "huginn_llm_cost_usd",
    "Accumulated LLM cost in USD.",
    labelnames=("model",),
)
# TPS / TTFT 实时监控: 评估流式生成速率 + 首 token 延迟.
# ponytail: Histogram buckets 覆盖 1 tok/s (极慢) 到 5000 tok/s (极快).
LLM_TPS = Histogram(
    "huginn_llm_tps",
    "LLM tokens-per-second during streaming (chunk_chars/4 / elapsed).",
    labelnames=("model",),
    buckets=(1, 5, 10, 20, 30, 50, 80, 120, 200, 500, 1000, 5000),
)
LLM_TTFT_SECONDS = Histogram(
    "huginn_llm_ttft_seconds",
    "LLM time-to-first-token in seconds.",
    labelnames=("model",),
    buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 3, 5, 10, 30, 60),
)

# Operational gauges.
DB_CONNECTIONS_ACTIVE = Gauge(
    "huginn_db_connections_active",
    "Active SQLite database connections.",
)

# Bumped by the rate-limit middleware when a request is rejected.
RATE_LIMIT_BLOCKED_TOTAL = Counter(
    "huginn_rate_limit_blocked_total",
    "Requests rejected by the per-IP rate limiter.",
    labelnames=("session",),
)

# Prompt cache prefix stability. A "hit" means the static prefix
# (system prompt + begin-dialogs) was the same as the previous turn,
# so the LLM provider can reuse its KV cache. A "miss" means the prefix
# changed (persona switch, rebuild, first call).
PROMPT_CACHE_HITS_TOTAL = Counter(
    "huginn_prompt_cache_hits_total",
    "Prompt cache prefix hits (stable prefix reused).",
)
PROMPT_CACHE_MISSES_TOTAL = Counter(
    "huginn_prompt_cache_misses_total",
    "Prompt cache prefix misses (new or changed prefix).",
)


# ---------------------------------------------------------------------------
# Small convenience helpers for future instrumentation points
# ---------------------------------------------------------------------------

# Approximate USD per 1M tokens for common models. Used when the router
# doesn't have explicit cost data. ponytail: coarse rates, good enough
# for cost dashboards — exact billing comes from the provider invoice.
_MODEL_COST_RATES: dict[str, tuple[float, float]] = {
    # (input_per_1m, output_per_1m) in USD
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-opus": (15.00, 75.00),
    "deepseek-chat": (0.14, 0.28),
    "deepseek-reasoner": (0.55, 2.19),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-pro": (1.25, 5.00),
}


def _lookup_cost(model: str) -> tuple[float, float]:
    """Best-effort cost lookup by model name prefix."""
    if not model:
        return (0.0, 0.0)
    lower = model.lower()
    for prefix, rates in _MODEL_COST_RATES.items():
        if prefix in lower:
            return rates
    return (0.0, 0.0)


def track_llm_usage(model: str, stats: dict[str, Any]) -> None:
    """Wire LLM token usage and cost to Prometheus metrics.

    Call this after extracting cache_stats from an LLM response.
    Safe to call with partial/empty stats — no-ops on missing fields.
    """
    try:
        input_tokens = int(stats.get("input_tokens", 0) or stats.get("usage_input_tokens", 0) or 0)
        output_tokens = int(stats.get("output_tokens", 0) or stats.get("usage_output_tokens", 0) or 0)
        cache_read = int(stats.get("cache_read_input_tokens", 0) or 0)
        cache_creation = int(stats.get("cache_creation_input_tokens", 0) or 0)

        # Total input = fresh input + cache reads (already billed at cache rate)
        total_input = input_tokens + cache_read + cache_creation
        if total_input:
            LLM_TOKENS_TOTAL.labels(model=model, kind="prompt").inc(total_input)
        if output_tokens:
            LLM_TOKENS_TOTAL.labels(model=model, kind="completion").inc(output_tokens)

        # Cost: cache reads are ~10% of normal input cost
        cost_in, cost_out = _lookup_cost(model)
        if cost_in or cost_out:
            cost = (
                input_tokens / 1_000_000 * cost_in
                + cache_read / 1_000_000 * cost_in * 0.1
                + cache_creation / 1_000_000 * cost_in * 1.25
                + output_tokens / 1_000_000 * cost_out
            )
            if cost > 0:
                LLM_COST_USD.labels(model=model).inc(cost)

        # Track cache hit/miss
        if cache_read > 0:
            PROMPT_CACHE_HITS_TOTAL.inc()
        if cache_creation > 0 or (total_input > 0 and cache_read == 0 and cache_creation == 0):
            PROMPT_CACHE_MISSES_TOTAL.inc()
    except Exception:
        pass  # metrics are best-effort, never break the agent


def track_tool_call(tool_name: str) -> None:
    """Increment the tool call counter."""
    try:
        TOOL_CALLS_TOTAL.labels(tool_name=tool_name).inc()
    except Exception:
        pass


def track_llm_tps(model: str, ttft_ms: int, tps: float) -> None:
    """Wire streaming TPS / TTFT to Prometheus. Best-effort, never raises."""
    try:
        LLM_TPS.labels(model=model).observe(tps)
        if ttft_ms > 0:
            LLM_TTFT_SECONDS.labels(model=model).observe(ttft_ms / 1000.0)
    except Exception:
        pass


def track_agent_turn(thread_id: str) -> None:
    """Increment the agent turn counter."""
    try:
        AGENT_TURNS_TOTAL.labels(thread_id=thread_id).inc()
    except Exception:
        pass


def track_websocket_connection() -> None:
    """Call when a WS client connects."""
    ACTIVE_WS_CONNECTIONS.inc()


def untrack_websocket_connection() -> None:
    """Call when a WS client disconnects."""
    ACTIVE_WS_CONNECTIONS.dec()


def _route_path(request: Request) -> str:
    """Return the matched route template, falling back to the raw path.

    Using the template (e.g. ``/threads/{thread_id}``) keeps label cardinality
    bounded instead of one series per concrete id.
    """
    route = request.scope.get("route")
    template = getattr(route, "path", None)
    return template or request.url.path


async def http_metrics_dispatch(request: Request, call_next):
    """Starlette ``BaseHTTPMiddleware`` dispatch recording request metrics.

    Wraps every HTTP exchange: counts the request and observes its latency,
    bucketed by method and (templated) path so the series stay bounded.
    """
    # Don't let Prometheus self-scrapes inflate the counters.
    if request.url.path == "/metrics":
        return await call_next(request)

    start = time.perf_counter()
    status = "0"
    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    except Exception:
        # An exception bubbled out of the handler stack — record it as a 500
        # so the failure shows up in the request counter before re-raising.
        status = "500"
        raise
    finally:
        duration = time.perf_counter() - start
        path = _route_path(request)
        method = request.method
        REQUESTS_TOTAL.labels(method=method, path=path, status=status).inc()
        REQUEST_DURATION.labels(method=method, path=path).observe(duration)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
async def metrics() -> Response:
    """Expose all registered metrics in the Prometheus text format."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
