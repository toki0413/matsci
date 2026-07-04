"""Structured JSON logging for Huginn.

Provides a ``JsonFormatter`` that emits one log line per record as a JSON
object, plus ``setup_json_logging()`` to wire it into the root logger.
Request-scoped context (request id, agent thread id) is pulled from
contextvars so every log line in a request carries the same correlation id.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

# Request-scoped context.  Set by the RequestID middleware (request_id) and
# by agent turn handlers (thread_id).  Defaults to empty string so logs are
# always emitted even when no request is active.
request_id_var: ContextVar[str] = ContextVar("huginn_request_id", default="")
thread_id_var: ContextVar[str] = ContextVar("huginn_thread_id", default="")

# Attributes the stdlib always puts on a LogRecord; anything else is treated
# as caller-supplied "extra" and surfaced verbatim in the JSON payload.
_RECORD_BUILTINS = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
        "message",
        "asctime",
        "request_info",
    }
)


def _format_timestamp(created: float) -> str:
    """ISO-8601 UTC with millisecond precision and a trailing ``Z``."""
    dt = datetime.fromtimestamp(created, tz=timezone.utc)
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _json_default(obj: Any) -> Any:
    """Last-resort serializer — keep logs working for odd objects."""
    try:
        return repr(obj)
    except Exception:
        return "<unserializable>"


class JsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def __init__(self, *, timestamp_key: str = "timestamp", ensure_ascii: bool = False) -> None:
        super().__init__()
        self._timestamp_key = timestamp_key
        self._ensure_ascii = ensure_ascii

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            self._timestamp_key: _format_timestamp(record.created),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
            "thread_id": thread_id_var.get(),
        }

        # Merge in any extra= fields the caller attached to the record.
        for key, value in record.__dict__.items():
            if key in _RECORD_BUILTINS or key.startswith("_"):
                continue
            payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        return json.dumps(payload, default=_json_default, ensure_ascii=self._ensure_ascii)


_CONFIGURED = False


def setup_json_logging(level: str | int | None = None) -> bool:
    """Configure the root logger to emit JSON lines.

    Enabled by default; set ``HUGINN_JSON_LOGS=0`` to opt out and keep the
    stdlib's default text formatting.  Safe to call more than once — repeat
    calls are no-ops once the handler is in place.

    Returns ``True`` if JSON logging was configured, ``False`` if skipped.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return True

    if os.environ.get("HUGINN_JSON_LOGS", "1") == "0":
        return False

    if level is None:
        level = os.environ.get("HUGINN_LOG_LEVEL", "INFO").upper()

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    # Replace pre-existing handlers so we don't double-log the same record.
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)

    # Quiet chatty third-party loggers without silencing them entirely.
    for noisy in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    return True
