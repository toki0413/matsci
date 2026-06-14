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
    """Reactive pet state with active-task tracking."""

    MAX_RECENT = 8

    def __init__(self) -> None:
        self.mood = PetMood.IDLE
        self.last_event: PetEvent | None = None
        self.idle_since = time.time()
        self.active_tasks = 0
        self.recent_events: list[dict[str, Any]] = []
        self.name = "Toki"
        self.personality = "cheerful"

    def configure(self, name: str | None = None, personality: str | None = None) -> None:
        if name:
            self.name = name
        if personality:
            self.personality = personality

    def update(self, event: PetEvent) -> None:
        self.mood = event.mood
        self.last_event = event
        self.idle_since = time.time()
        self._update_active_tasks(event)
        self._record_recent(event)

    def _update_active_tasks(self, event: PetEvent) -> None:
        """Heuristically track how many tasks are currently running."""
        mood = event.mood
        details = event.details

        # Team task lifecycle
        status = details.get("status")
        if status == "running":
            self.active_tasks += 1
            return
        if status in ("done", "error") and self.active_tasks > 0:
            self.active_tasks -= 1
            return

        # Tool / general work lifecycle
        if mood == PetMood.WORKING:
            self.active_tasks += 1
        elif mood in (PetMood.SUCCESS, PetMood.ERROR) and self.active_tasks > 0:
            self.active_tasks -= 1

        # Thinking is not counted as an independent task

    def _record_recent(self, event: PetEvent) -> None:
        self.recent_events.append(
            {
                "timestamp": event.timestamp,
                "mood": event.mood.value,
                "message": event.message,
                "details": event.details,
            }
        )
        if len(self.recent_events) > self.MAX_RECENT:
            self.recent_events.pop(0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mood": self.mood.value,
            "message": self.last_event.message if self.last_event else "Hi!",
            "idle_seconds": time.time() - self.idle_since,
            "active_tasks": self.active_tasks,
            "recent_events": self.recent_events,
            "name": self.name,
            "personality": self.personality,
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

    def configure(self, name: str | None = None, personality: str | None = None) -> None:
        self._state.configure(name, personality)

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


def configure_pet(name: str | None = None, personality: str | None = None) -> None:
    """Configure the global pet name and personality."""
    get_pet_bus().configure(name, personality)
