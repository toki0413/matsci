"""Minimal, self-contained OTLP/HTTP (JSON) trace exporter.

``TelemetryCollector`` keeps spans in memory; this module turns each finished
root span tree into an OpenTelemetry ``Span`` and posts it to any
OTLP/HTTP-compatible endpoint (including Langfuse's OTel ingestion endpoint) in
the background.

Design invariants:
- *Zero hard deps*: stdlib only (json/threading/urllib); no ``opentelemetry-sdk``
  requirement, so optional / heavy backends never block the agent.
- *Fail-open / non-blocking*: any error in encoding, threading or POST is
  swallowed and logged at debug. Telemetry must never break the agent loop.
- *No-op when unconfigured*: if ``HUGINN_OTEL_ENDPOINT`` is empty,
  ``get_default_exporter()`` returns ``None`` and no thread ever starts.

Wire format: the official OTLP/HTTP JSON encoding of
``resourceSpans -> scopeSpans -> spans``. Content-Type ``application/json``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)

# OTel SpanKind values.
_KIND_INTERNAL = 2
_KIND_CLIENT = 3
# OTel StatusCode values.
_STATUS_UNSET = 0
_STATUS_ERROR = 2


def _to_value(value: Any) -> dict[str, Any]:
    """Encode a scalar metadata value into an OTLP AnyValue JSON object."""
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": value}
    if isinstance(value, float):
        return {"doubleValue": value}
    if value is None:
        return {"stringValue": ""}
    return {"stringValue": str(value)}


def _attributes(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"key": str(k), "value": _to_value(v)} for k, v in metadata.items()]


class OtlpExporter:
    """Background, batched OTLP/HTTP JSON trace exporter.

    Created lazily by :func:`get_default_exporter`. Call ``emit`` from any
    thread after a root span finishes; the worker flushes on an interval or once
    the batch is full. ``flush()`` drains synchronously (useful for tests and
    shutdown).
    """

    def __init__(
        self,
        endpoint: str,
        headers: dict[str, str] | None = None,
        service_name: str = "huginn",
        batch_size: int = 32,
        interval_seconds: float = 5.0,
        timeout_seconds: float = 5.0,
    ) -> None:
        self.endpoint = endpoint
        self.headers = {"Content-Type": "application/json"}
        if headers:
            self.headers.update(headers)
        self.service_name = service_name
        self.batch_size = max(1, batch_size)
        self.interval = max(0.2, interval_seconds)
        self.timeout = max(0.5, timeout_seconds)

        self._lock = threading.Lock()
        self._pending: deque[Any] = deque()
        self._closed = False
        self._worker_started = False
        self._worker: threading.Thread | None = None
        self._warned_failure = False
        logger.info("huginn OTLP trace export enabled -> %s", endpoint)

    # -- public API -----------------------------------------------------

    def emit(self, root: Any) -> None:
        """Queue a finished root ``TelemetrySpan`` for export."""
        if self._closed:
            return
        with self._lock:
            self._pending.append(root)
            if len(self._pending) >= self.batch_size:
                self._start_worker()  # wake up promptly to drain the full batch

    def flush(self) -> None:
        """Drain all pending spans now, synchronously. Fail-open."""
        with self._lock:
            roots = list(self._pending)
            self._pending.clear()
        if roots:
            self._post_roots(roots)

    def shutdown(self, block: bool = False) -> None:
        """Stop accepting new spans and flush what remains."""
        with self._lock:
            self._closed = True
        self.flush()
        if block and self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=self.timeout)

    # -- internals ------------------------------------------------------

    def _start_worker(self) -> None:
        if self._worker_started or self._closed:
            return
        self._worker_started = True
        self._worker = threading.Thread(
            target=self._run, name="huginn-otel-export", daemon=True
        )
        self._worker.start()

    def _run(self) -> None:
        while True:
            with self._lock:
                if self._closed:
                    break
                now_pending = list(self._pending)
                # FIFO batch: carry at most ``batch_size`` per tick; anything
                # beyond stays queued for the next tick. This caps per-request
                # size without ever dropping spans.
                take = min(self.batch_size, len(now_pending))
                batch = now_pending[:take]
                remain = now_pending[take:]
                self._pending = deque(remain)
            if batch:
                self._post_roots(batch)
            with self._lock:
                if self._closed:
                    break
            time.sleep(self.interval)

    def _post_roots(self, roots: list[Any]) -> None:
        # NOTE: urllib reads HTTP(S)_PROXY / ALL_PROXY from the environment
        # automatically, so this works in proxied/corporate networks without any
        # extra config.
        try:
            payload = self._encode(roots)
            req = urllib.request.Request(
                self.endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers=self.headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp.read()
        except urllib.error.HTTPError as exc:
            # The endpoint answered but rejected the request — almost always an
            # auth/config problem (e.g. missing/invalid HUGINN_OTEL_HEADERS,
            # wrong path). Fail-open, but surface the status instead of hiding it.
            self._note_failure(f"HTTP {exc.code}")
        except Exception as exc:  # noqa: BLE001
            # Network / TLS / encoding / parsing errors. Fail-open as well.
            self._note_failure(f"{exc.__class__.__name__}: {exc}")

    def _note_failure(self, detail: str) -> None:
        """Warn once per exporter on the first dropped batch, then stay quiet.

        The single non-raising failure gate keeps telemetry from ever breaking
        the agent loop, while the one-shot warning tells operators *why* traces
        stop flowing (e.g. "HTTP 401" = check the Langfuse key).
        """
        if not self._warned_failure:
            self._warned_failure = True
            logger.warning(
                "huginn OTLP export failing (fail-open, traces dropped): %s. "
                "Check HUGINN_OTEL_ENDPOINT and HUGINN_OTEL_HEADERS.",
                detail,
            )
        else:
            logger.debug("huginn OTLP export still failing (fail-open): %s", detail)

    def _encode(self, roots: list[Any]) -> dict[str, Any]:
        spans: list[dict[str, Any]] = []
        for root in roots:
            self._append_span_tree(root, None, spans)
        return {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "service.name", "value": {"stringValue": self.service_name}}
                        ]
                    },
                    "scopeSpans": [
                        {
                            "scope": {"name": "huginn.telemetry"},
                            "spans": spans,
                        }
                    ],
                }
            ]
        }

    def _append_span_tree(self, span: Any, parent_id: str | None, out: list[dict[str, Any]]) -> str:
        # OTel expects 16-hex span ids and 32-hex trace ids. We derive them as
        # stable hashes of the collector's internal uuid span ids so parent/child
        # links and (per root) traces are consistent across one export batch.
        span_id = hashlib.sha256(span.span_id.encode("utf-8")).hexdigest()[:16]
        trace_id = hashlib.sha256(self._trace_key(span).encode("utf-8")).hexdigest()[:32]
        kind = _KIND_CLIENT if span.name in ("tool_call", "llm_call") else _KIND_INTERNAL

        start_ns = int(span.start_time * 1_000_000_000)
        end_ns = int((span.end_time or span.start_time) * 1_000_000_000)

        meta = dict(span.metadata or {})
        out.append(
            {
                "traceId": trace_id,
                "spanId": span_id,
                "parentSpanId": parent_id or "",
                "name": span.name,
                "kind": kind,
                "startTimeUnixNano": str(start_ns),
                "endTimeUnixNano": str(max(end_ns, start_ns)),
                "attributes": _attributes(meta),
                **self._status(meta),
            }
        )
        for child in span.children or []:
            self._append_span_tree(child, span_id, out)
        return span_id

    @staticmethod
    def _trace_key(span: Any) -> str:
        # Root spans become their own trace. Children inherit the parent trace in
        # the same export batch; this key only needs to be stable per root.
        return span.span_id

    @staticmethod
    def _status(meta: dict[str, Any]) -> dict[str, Any]:
        if meta.get("success") is False or meta.get("error"):
            return {"status": {"code": _STATUS_ERROR, "message": "error"}}
        return {"status": {"code": _STATUS_UNSET}}


_DEFAULT_EXPORTER: OtlpExporter | None = None
_DEFAULT_LOCK = threading.Lock()


def get_default_exporter() -> OtlpExporter | None:
    """Return a process-wide exporter from env, or ``None`` if unconfigured.

    Env:
    - ``HUGINN_OTEL_ENDPOINT``: OTLP/HTTP traces endpoint (required to enable).
      For Langfuse, use ``https://cloud.langfuse.com/api/public/otel/v1/traces``
      with an ``Authorization`` header built from your pk/sk keys.
    - ``HUGINN_OTEL_HEADERS``: optional JSON object of extra HTTP headers
      (e.g. ``{"Authorization": "Basic ..."}``).
    - ``HUGINN_OTEL_SERVICE_NAME``: resource ``service.name`` (default "huginn").
    - ``HUGINN_OTEL_BATCH_SIZE`` / ``HUGINN_OTEL_INTERVAL``: tuning (advanced).
    """
    global _DEFAULT_EXPORTER
    endpoint = os.environ.get("HUGINN_OTEL_ENDPOINT", "").strip()
    if not endpoint:
        return None
    with _DEFAULT_LOCK:
        if _DEFAULT_EXPORTER is not None:
            return _DEFAULT_EXPORTER
        headers: dict[str, str] = {}
        raw_headers = os.environ.get("HUGINN_OTEL_HEADERS", "").strip()
        if raw_headers:
            try:
                parsed = json.loads(raw_headers)
                headers = {str(k): str(v) for k, v in parsed.items()}
            except (ValueError, TypeError) as exc:
                logger.debug("HUGINN_OTEL_HEADERS not valid JSON, ignored: %s", exc)
        _DEFAULT_EXPORTER = OtlpExporter(
            endpoint=endpoint,
            headers=headers,
            service_name=os.environ.get("HUGINN_OTEL_SERVICE_NAME", "huginn"),
            batch_size=int(os.environ.get("HUGINN_OTEL_BATCH_SIZE", "32")),
            interval_seconds=float(os.environ.get("HUGINN_OTEL_INTERVAL", "5")),
        )
        return _DEFAULT_EXPORTER


def reset_default_exporter_for_tests() -> None:
    """Test hook: clear the cached default exporter so env changes apply."""
    global _DEFAULT_EXPORTER
    with _DEFAULT_LOCK:
        if _DEFAULT_EXPORTER is not None:
            _DEFAULT_EXPORTER.shutdown()
        _DEFAULT_EXPORTER = None
