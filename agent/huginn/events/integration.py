"""Convenience helpers for publishing events from existing code.

These wrap the EventBus with domain-specific publish functions so call
sites don't need to know about AgentEvent construction or event type
strings. Every function is a no-op if the events package is unavailable
or the bus isn't initialized — call them freely, they won't break your
flow.

Usage (from async agent code):
    from huginn.events.integration import publish_tool_event
    await publish_tool_event("vasp_tool", {"structure": "Si.cif"}, result, thread_id)

Usage (from sync code — fire and forget):
    from huginn.events.integration import publish_tool_event_sync
    publish_tool_event_sync("vasp_tool", args, result, thread_id)

The _sync variants schedule the publish on the running event loop if
one exists, otherwise they drop the event (observability is best-effort,
not a data path).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# Reference to the main event loop, set at startup.
# Used by _schedule_sync when no loop is running in the current thread.
_main_loop: asyncio.AbstractEventLoop | None = None


def set_main_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Store the main event loop for cross-thread event scheduling."""
    global _main_loop
    _main_loop = loop


# Set to False to globally disable event publishing (e.g. in tests).
_ENABLED = True


def _get_bus():
    """Return the shared EventBus, or None if it can't be created."""
    if not _ENABLED:
        return None
    try:
        from huginn.events.event_bus import EventBus
        return EventBus.shared()
    except Exception:
        return None


async def _publish(event_type: str, data: dict, thread_id: str = "", source: str = "") -> None:
    """Build and publish an AgentEvent. Never raises."""
    bus = _get_bus()
    if bus is None:
        return
    try:
        from huginn.events.event_bus import AgentEvent
        await bus.publish(AgentEvent(
            type=event_type,
            timestamp=time.time(),
            data=data,
            thread_id=thread_id,
            source=source,
        ))
    except Exception:
        logger.debug("event publish failed: %s", event_type, exc_info=True)


# ── Tool events ──────────────────────────────────────────────────────

async def publish_tool_event(
    tool_name: str,
    tool_input: Any,
    result: Any,
    thread_id: str = "",
    error: str | None = None,
) -> None:
    """Publish a tool call + result event pair.

    Call this after a tool finishes (success or failure). If error is
    not None, a tool.error event is published instead of tool.result.
    tool_input and result are passed through as-is — make sure they're
    JSON-serializable or the SSE/audit consumers will str() them.
    """
    call_data = {"tool": tool_name, "input": tool_input}
    await _publish("tool.call", call_data, thread_id, source="tool_adapter")

    if error is not None:
        await _publish("tool.error", {
            "tool": tool_name, "input": tool_input, "error": error,
        }, thread_id, source="tool_adapter")
    else:
        await _publish("tool.result", {
            "tool": tool_name, "result": result,
        }, thread_id, source="tool_adapter")


# ── Compaction events ────────────────────────────────────────────────

async def publish_compact_event(
    before_pct: float,
    after_pct: float,
    thread_id: str = "",
) -> None:
    """Publish compaction start + end events.

    before_pct / after_pct are context window utilization (0-100).
    Also fires a context.overflow if before_pct > 90.
    """
    await _publish("compact.start", {
        "context_pct": before_pct,
    }, thread_id, source="context_manager")

    if before_pct > 90:
        await _publish("context.overflow", {
            "context_pct": before_pct,
        }, thread_id, source="context_manager")

    await _publish("compact.end", {
        "before_pct": before_pct,
        "after_pct": after_pct,
    }, thread_id, source="context_manager")


# ── Pipeline events ─────────────────────────────────────────────────

async def publish_pipeline_event(
    suggestion: str,
    thread_id: str = "",
    stage: str | None = None,
) -> None:
    """Publish a pipeline suggestion / stage change event.

    If stage is given, publishes a stage_change event; otherwise a
    suggestion event.
    """
    if stage is not None:
        await _publish("pipeline.stage_change", {
            "suggestion": suggestion, "stage": stage,
        }, thread_id, source="pipeline")
    else:
        await _publish("pipeline.suggest", {
            "suggestion": suggestion,
        }, thread_id, source="pipeline")


# ── Session events ──────────────────────────────────────────────────

async def publish_session_event(
    action: str,  # "start" or "end"
    thread_id: str = "",
    metadata: dict | None = None,
) -> None:
    """Publish a session start/end event."""
    event_type = "session.start" if action == "start" else "session.end"
    data = metadata or {}
    await _publish(event_type, data, thread_id, source="session")


# ── Sync wrappers ───────────────────────────────────────────────────
# For call sites that aren't async (e.g. ToolAdapter._run). These try to
# schedule on the running loop; if no loop is running, the event is
# silently dropped — observability should never block the agent.

def _schedule_sync(coro) -> None:
    """Fire-and-forget an async publish from sync code."""
    try:
        loop = asyncio.get_running_loop()
        asyncio.ensure_future(coro)
    except RuntimeError:
        # No running loop in this thread — schedule on the main loop
        if _main_loop and _main_loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, _main_loop)
        else:
            # no loop available, close the coroutine to avoid
            # "coroutine was never awaited" warnings
            coro.close()
    except Exception:
        logger.debug("sync event schedule failed", exc_info=True)


def publish_tool_event_sync(
    tool_name: str,
    tool_input: Any,
    result: Any,
    thread_id: str = "",
    error: str | None = None,
) -> None:
    _schedule_sync(publish_tool_event(tool_name, tool_input, result, thread_id, error))


def publish_compact_event_sync(
    before_pct: float,
    after_pct: float,
    thread_id: str = "",
) -> None:
    _schedule_sync(publish_compact_event(before_pct, after_pct, thread_id))


def publish_pipeline_event_sync(
    suggestion: str,
    thread_id: str = "",
    stage: str | None = None,
) -> None:
    _schedule_sync(publish_pipeline_event(suggestion, thread_id, stage))


def publish_session_event_sync(
    action: str,
    thread_id: str = "",
    metadata: dict | None = None,
) -> None:
    _schedule_sync(publish_session_event(action, thread_id, metadata))


def disable_events() -> None:
    """Globally disable event publishing. Useful for tests."""
    global _ENABLED
    _ENABLED = False


def enable_events() -> None:
    """Re-enable event publishing after disable_events()."""
    global _ENABLED
    _ENABLED = True
