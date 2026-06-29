"""Server-sent events stream."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from huginn.pet import get_pet_bus

router = APIRouter(tags=["events"])


@router.get("/events")
async def events_stream() -> StreamingResponse:
    """Server-sent events stream for the desktop pet and activity indicators."""
    bus = get_pet_bus()
    queue, unsubscribe = await bus.queue()

    async def generator() -> AsyncIterator[str]:
        # Send current state immediately.
        yield f"data: {json.dumps({'type': 'state', 'state': bus.state.to_dict()})}\n\n"
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    payload = {
                        "type": "event",
                        "mood": event.mood.value,
                        "message": event.message,
                        "details": event.details,
                        "timestamp": event.timestamp,
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                except TimeoutError:
                    # Keep connection alive with the latest state.
                    yield f"data: {json.dumps({'type': 'heartbeat', 'state': bus.state.to_dict()})}\n\n"
        finally:
            # Remove the queue so it doesn't accumulate events after disconnect
            unsubscribe()

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
