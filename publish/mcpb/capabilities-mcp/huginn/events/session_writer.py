"""Session event writer — the write side of the event-sourcing loop.

The read side is ``GET /threads/{id}/events`` (serves the UiProjection from a
``SessionEventLog``). For that endpoint to return real data, the agent loop
must *write* events as turns execute. This module provides a thin, guarded
writer so agent code can append session events without erroring the loop.

Storage is keyed by ``thread_id`` at the same default path the events endpoint
reads, so a written event is immediately visible to ``/events`` and therefore
to the frontend's incremental block model (compaction dividers, etc.).

Caching: a per-thread ``SessionEventLog`` is kept in a module-level dict so
appends are O(1) instead of re-reading the whole file each time. The cache is
not evicted (sessions are bounded by the TTL sweeper); rebooting the process
re-opens from disk via ``open``.
"""

from __future__ import annotations

import logging
from typing import Any

from huginn.events.session_log import (
    EVENT_COMPACTION,
    EVENT_CONTEXT_INJECTION,
    EVENT_MESSAGE,
    EVENT_TOOL_CALL,
    EVENT_TOOL_RESULT,
    SessionEventLog,
)

logger = logging.getLogger(__name__)

# thread_id -> SessionEventLog (lazily opened, appended incrementally).
_logs: dict[str, SessionEventLog] = {}


def _log_for(thread_id: str) -> SessionEventLog:
    log = _logs.get(thread_id)
    if log is None:
        log = SessionEventLog.open(thread_id)
        _logs[thread_id] = log
    return log


def append_session_event(thread_id: str, kind: str, payload: dict[str, Any]) -> None:
    """Append one session event for a thread. Never raises — a failure to
    write must not break the agent loop."""
    try:
        _log_for(thread_id).append(kind, payload)
    except Exception:
        logger.debug("session event append failed for %s (%s)", thread_id, kind, exc_info=True)


def record_user_message(thread_id: str, content: str) -> None:
    append_session_event(thread_id, EVENT_MESSAGE, {"role": "user", "content": content})


def record_assistant_message(thread_id: str, content: str) -> None:
    append_session_event(thread_id, EVENT_MESSAGE, {"role": "assistant", "content": content})


def record_tool_call(thread_id: str, tool_call_id: str, name: str, args: Any) -> None:
    append_session_event(
        thread_id,
        EVENT_TOOL_CALL,
        {"tool_call_id": tool_call_id, "name": name, "args": args},
    )


def record_tool_result(thread_id: str, tool_call_id: str, content: str) -> None:
    append_session_event(
        thread_id,
        EVENT_TOOL_RESULT,
        {"tool_call_id": tool_call_id, "content": content},
    )


def record_compaction(thread_id: str, summary: str) -> None:
    append_session_event(thread_id, EVENT_COMPACTION, {"summary": summary})


def record_injection(
    thread_id: str, source: str, payload: dict[str, Any] | None = None
) -> None:
    """记录一次上下文注入 (DSH trajectory 按来源审计).

    ``source`` 明确"谁塞进来的" (如 ``context_builder.plan``、``plugin:xxx``),
    供 trajectory 按来源回溯模型看到了什么. fail-open: 写失败不阻断主流程.
    """
    append_session_event(
        thread_id, EVENT_CONTEXT_INJECTION,
        {"source": source, **(payload or {})},
    )
