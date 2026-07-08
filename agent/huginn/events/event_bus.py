"""Unified event bus for agent lifecycle observability.

Inspired by OpenCode's bus/ + event-v2-bridge pattern.
All agent actions (tool calls, compaction, pipeline transitions,
campaign events) publish events here. Subscribers can:
  - Push to SSE for real-time UI updates
  - Write to audit log for provenance
  - Trigger hooks for reactive behavior

Events are typed and structured, not just log strings.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any, AsyncIterator, Callable

from huginn.events.event_types import ALL

logger = logging.getLogger(__name__)


@dataclass
class AgentEvent:
    """A typed event in the agent lifecycle.

    type is a dotted string (see event_types.py) so external consumers
    can prefix-match without importing Python. data is free-form — put
    whatever the event needs, but keep it JSON-serializable so SSE and
    audit log don't choke.
    """

    type: str
    timestamp: float
    data: dict[str, Any] = field(default_factory=dict)
    thread_id: str = ""
    source: str = ""  # which component emitted this

    def to_sse(self) -> str:
        """Serialize to an SSE frame: ``event: <type>\\ndata: <json>\\n\\n``.

        Matches the format used by interaction/streaming.py so the
        frontend can consume both streams with one parser.
        """
        payload = {
            "type": self.type,
            "ts": self.timestamp,
            "thread_id": self.thread_id,
            "source": self.source,
            "data": self.data,
        }
        return f"event: {self.type}\ndata: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EventBus:
    """Async event bus with subscription support.

    Design notes:
      - Subscribers are plain callables (sync or async). Async ones are
        scheduled via ensure_future; sync ones run inline. This keeps
        the bus usable from both async agent code and sync helpers.
      - SSE streams get their own asyncio.Queue per consumer. When the
        consumer disconnects (generator is GC'd or breaks), the queue
        is cleaned up on the next publish that finds it full/closed.
      - History is a bounded deque, not a list — O(1) eviction on both
        ends, no reindexing.
    """

    _instance: EventBus | None = None

    @classmethod
    def shared(cls) -> EventBus:
        """Process-wide singleton. Lazily created on first access."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, history_size: int = 1000) -> None:
        # ponytail: subscribers stored as dict[type, list[callable]].
        # "*" key catches everything. Linear scan per publish is fine
        # because subscriber counts are small (<20 in practice).
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)
        self._sse_queues: list[asyncio.Queue[AgentEvent | None]] = []
        self._history: deque[AgentEvent] = deque(maxlen=history_size)
        self._history_size = history_size

    async def publish(self, event: AgentEvent) -> None:
        """Publish an event to all subscribers + SSE queues.

        Safe to call from any async context. Subscriber exceptions are
        caught and logged — one bad subscriber must not break the bus.
        """
        # History first — even if a subscriber blows up, the event is recorded.
        self._history.append(event)

        # Fan out to typed subscribers + wildcard subscribers.
        callbacks = self._subscribers.get(event.type, []) + self._subscribers.get(ALL, [])
        for cb in callbacks:
            try:
                result = cb(event)
                if asyncio.iscoroutine(result):
                    # Fire-and-forget async subscriber. We don't await
                    # because a slow subscriber would block the publisher.
                    asyncio.ensure_future(result)
            except Exception:
                logger.exception("subscriber %r failed for event %s", cb, event.type)

        # Push to SSE queues. None sentinel means "stream closed" to consumers.
        dead: list[asyncio.Queue] = []
        for q in self._sse_queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Consumer is too slow. Drop oldest and push — better to
                # lose an old event than block the whole bus.
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                except Exception:
                    dead.append(q)
            except Exception:
                dead.append(q)

        if dead:
            self._sse_queues = [q for q in self._sse_queues if q not in dead]

    def subscribe(self, event_type: str, callback: Callable) -> Callable:
        """Subscribe to events of a specific type (or "*" for all).

        Returns an unsubscribe function — call it to remove the callback.
        Keeping the unsubscribe pattern explicit (instead of relying on
        weakrefs) because most subscribers are closures/methods that
        would never get GC'd otherwise.
        """
        self._subscribers[event_type].append(callback)

        def _unsubscribe() -> None:
            try:
                self._subscribers[event_type].remove(callback)
            except ValueError:
                pass

        return _unsubscribe

    async def sse_stream(self) -> AsyncIterator[str]:
        """Get an SSE-formatted stream of events.

        Creates a dedicated asyncio.Queue for this consumer. The
        generator yields SSE strings (see AgentEvent.to_sse). When the
        consumer stops iterating (break / disconnect / GC), the queue
        is removed from the bus.

        Yields forever — callers should wrap with a timeout or
        disconnect detection at the HTTP layer.
        """
        q: asyncio.Queue[AgentEvent | None] = asyncio.Queue(maxsize=256)
        self._sse_queues.append(q)
        try:
            while True:
                event = await q.get()
                if event is None:
                    # Sentinel: bus is shutting down
                    break
                yield event.to_sse()
        finally:
            # Clean up so we don't leak queues when clients disconnect.
            try:
                self._sse_queues.remove(q)
            except ValueError:
                pass

    def recent_events(
        self, n: int = 50, event_type: str | None = None
    ) -> list[AgentEvent]:
        """Get recent events from history (for audit/debugging).

        Returns the last ``n`` events, optionally filtered by type.
        Most-recent-first ordering.
        """
        events = list(self._history)
        if event_type is not None:
            events = [e for e in events if e.type == event_type]
        return events[-n:][::-1] if n < len(events) else list(reversed(events))

    def clear_history(self) -> None:
        """Drop all history. SSE queues and subscribers are untouched."""
        self._history.clear()

    def shutdown(self) -> None:
        """Signal all SSE streams to close. Call on agent shutdown."""
        for q in self._sse_queues:
            try:
                q.put_nowait(None)
            except Exception:
                pass
        self._sse_queues.clear()
