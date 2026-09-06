"""Unified event-bus SSE stream.

Subscribes to the process-wide EventBus and forwards every agent lifecycle
event (tool calls, compaction, pipeline transitions, ...) to the client as
Server-Sent Events. Reuses EventBus.sse_stream(), which already manages a
per-consumer asyncio.Queue and cleans it up when the client disconnects.
"""

from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from huginn.events.event_bus import EventBus

router = APIRouter(tags=["events"])


@router.get("/events/recent")
async def recent_events(
    n: int = Query(200, ge=1, le=3000),
    event_type: str | None = Query(None),
) -> dict:
    """Replay recent EventBus events (for the frontend audit view).

    AgentEvent 是带类型/时间戳/来源的结构化记录, 序列化成与 /events/stream
    SSE 帧一致的格式, 前端同一解析器可直接消费回放 + 实时两条链路.
    """
    bus = EventBus.shared()
    raw = bus.recent_events(n=n, event_type=event_type)
    events = [
        {
            "type": e.type,
            "ts": e.timestamp,
            "thread_id": e.thread_id,
            "source": e.source,
            "data": e.data,
        }
        for e in raw
    ]
    return {"events": events, "count": len(events)}


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
