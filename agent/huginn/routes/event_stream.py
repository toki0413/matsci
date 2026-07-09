"""Unified event-bus SSE stream.

Subscribes to the process-wide EventBus and forwards every agent lifecycle
event (tool calls, compaction, pipeline transitions, ...) to the client as
Server-Sent Events. Reuses EventBus.sse_stream(), which already manages a
per-consumer asyncio.Queue and cleans it up when the client disconnects.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from huginn.events.event_bus import EventBus

router = APIRouter(tags=["events"])


@router.get("/events/stream")
async def event_stream() -> StreamingResponse:
    """Live SSE feed of all agent lifecycle events."""
    bus = EventBus.shared()

    return StreamingResponse(
        bus.sse_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
