"""Desktop pet state machine and event bus.

A lightweight ambient companion (like the Rising Antivirus lion) that reacts
to agent activity: thinking, working, success, error, idle.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class PetMood(str, Enum):
    """Possible pet moods."""

    IDLE = "idle"
    THINKING = "thinking"
    WORKING = "working"
    SUCCESS = "success"
    ERROR = "error"
    SLEEPING = "sleeping"


@dataclass
class PetEvent:
    """A single event the pet reacts to."""

    timestamp: float
    mood: PetMood
    message: str
    details: dict[str, Any] = field(default_factory=dict)


class PetState:
    """Reactive pet state."""

    def __init__(self) -> None:
        self.mood = PetMood.IDLE
        self.last_event: PetEvent | None = None
        self.idle_since = time.time()

    def update(self, event: PetEvent) -> None:
        self.mood = event.mood
        self.last_event = event
        self.idle_since = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "mood": self.mood.value,
            "message": self.last_event.message if self.last_event else "Hi!",
            "idle_seconds": time.time() - self.idle_since,
        }


class PetEventBus:
    """Async publish/subscribe bus for pet events."""

    def __init__(self) -> None:
        self._subs: list[Callable[[PetEvent], None]] = []
        self._queues: list[asyncio.Queue[PetEvent]] = []
        self._state = PetState()

    def subscribe(self, callback: Callable[[PetEvent], None]) -> Callable[[], None]:
        """Register a synchronous callback. Returns unsubscribe function."""
        self._subs.append(callback)

        def unsubscribe() -> None:
            try:
                self._subs.remove(callback)
            except ValueError:
                pass

        return unsubscribe

    async def queue(self) -> asyncio.Queue[PetEvent]:
        """Return an async queue that receives all future events."""
        q: asyncio.Queue[PetEvent] = asyncio.Queue()
        self._queues.append(q)
        return q

    def publish(self, mood: PetMood, message: str, details: dict[str, Any] | None = None) -> None:
        event = PetEvent(
            timestamp=time.time(),
            mood=mood,
            message=message,
            details=details or {},
        )
        self._state.update(event)
        for cb in self._subs:
            try:
                cb(event)
            except Exception:
                pass
        for q in self._queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

    @property
    def state(self) -> PetState:
        return self._state


# Global singleton used by the agent runtime and server.
_global_bus: PetEventBus | None = None


def get_pet_bus() -> PetEventBus:
    """Return the global pet event bus."""
    global _global_bus
    if _global_bus is None:
        _global_bus = PetEventBus()
    return _global_bus
