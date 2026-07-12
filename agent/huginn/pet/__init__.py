"""Desktop pet state machine and event bus.

A lightweight ambient companion (like the Rising Antivirus lion) that reacts
to agent activity: thinking, working, success, error, idle.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

RAVEN_NAME = "渡鸦"
RAVEN_IMAGE_PATH = Path(__file__).parent.parent / "assets" / "raven.png"

RAVEN_ASCII_FALLBACK = r"""
           .-'''''-.
          /         \
         |  o     o  |
          \    ^    /
           '  '-'  '
          /|         |\
         / |         | \
        /__|_________|__\
           |  |  |  |
          /   |  |   \
         /    |  |    \
        ---   |  |   ---
""".strip()


def _image_to_ascii(image_path: Path, width: int = 50) -> str:
    """Convert an image to a grayscale ASCII art representation."""
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            img = img.convert("L")
            aspect = img.height / img.width
            height = int(width * aspect * 0.5)
            img = img.resize((width, max(1, height)), Image.Resampling.LANCZOS)
            pixels = list(img.tobytes())

        chars = " .:-=+*#%@"
        ascii_chars = [
            chars[min(pixel // (256 // len(chars)), len(chars) - 1)] for pixel in pixels
        ]
        lines = [
            "".join(ascii_chars[i : i + width])
            for i in range(0, len(ascii_chars), width)
        ]
        return "\n".join(lines)
    except Exception:
        return RAVEN_ASCII_FALLBACK


def get_pet_avatar() -> str:
    """Return the raven avatar: ASCII art rendered from the project image."""
    if RAVEN_IMAGE_PATH.exists():
        return _image_to_ascii(RAVEN_IMAGE_PATH, width=36)
    return RAVEN_ASCII_FALLBACK


class PetMood(StrEnum):
    """Possible pet moods."""

    IDLE = "idle"
    THINKING = "thinking"
    CODING = "coding"
    REVIEWING = "reviewing"
    WORKING = "working"
    SUCCESS = "success"
    ERROR = "error"
    SLEEPING = "sleeping"
    HAPPY = "happy"
    HUNGRY = "hungry"


@dataclass
class PetEvent:
    """A single event the pet reacts to."""

    timestamp: float
    mood: PetMood
    message: str
    details: dict[str, Any] = field(default_factory=dict)


# ── Accessory registry ──
ACCESSORY_REGISTRY: dict[str, dict[str, Any]] = {
    "crown": {"label": "Crown", "min_level": 5},
    "glasses": {"label": "Glasses", "min_level": 3},
    "scarf": {"label": "Scarf", "min_level": 7},
}

XP_PER_LEVEL_BASE = 100
XP_PER_SUCCESS = 15
HUNGER_DECAY_PER_MIN = 4
MOOD_DECAY_PER_MIN = 2


def _xp_for_level(level: int) -> int:
    """Return the XP required to complete the given level."""
    return int(XP_PER_LEVEL_BASE * (1.15 ** (level - 1)))


class PetState:
    """Reactive pet state with active-task tracking, XP, and vitals."""

    MAX_RECENT = 8

    def __init__(self) -> None:
        self.mood = PetMood.IDLE
        self.last_event: PetEvent | None = None
        self.idle_since = time.time()
        self.active_tasks = 0
        self.recent_events: list[dict[str, Any]] = []
        self.name = RAVEN_NAME
        self.personality = "cheerful"
        self.avatar = get_pet_avatar()
        # Gamification fields
        self.experience = 0
        self.level = 1
        self.hunger = 80
        self.happiness = 80
        self.accessories: list[str] = []
        self._last_decay = time.time()

    def configure(
        self,
        name: str | None = None,
        personality: str | None = None,
        avatar: str | None = None,
    ) -> None:
        if name:
            self.name = name
        if personality:
            self.personality = personality
        if avatar is not None:
            self.avatar = avatar

    def update(self, event: PetEvent) -> None:
        self.mood = event.mood
        self.last_event = event
        self.idle_since = time.time()
        self._update_active_tasks(event)
        self._award_xp_on_success(event)
        self._record_recent(event)
        self._apply_decay()

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

    def _award_xp_on_success(self, event: PetEvent) -> None:
        """Grant XP when a SUCCESS event arrives."""
        if event.mood == PetMood.SUCCESS:
            self.experience += XP_PER_SUCCESS
            # Check level-up
            while self.experience >= _xp_for_level(self.level):
                self.experience -= _xp_for_level(self.level)
                self.level += 1

    def _apply_decay(self) -> None:
        """Gradually reduce hunger and happiness over time."""
        now = time.time()
        elapsed_min = (now - self._last_decay) / 60.0
        if elapsed_min >= 0.5:  # Apply every 30 seconds minimum
            self.hunger = max(0, int(self.hunger - HUNGER_DECAY_PER_MIN * elapsed_min))
            self.happiness = max(
                0, int(self.happiness - MOOD_DECAY_PER_MIN * elapsed_min)
            )
            self._last_decay = now

    def toggle_accessory(self, accessory_id: str) -> None:
        """Toggle an accessory on/off. Respects level requirements."""
        if accessory_id in self.accessories:
            self.accessories.remove(accessory_id)
        else:
            reg = ACCESSORY_REGISTRY.get(accessory_id)
            if reg and self.level >= reg["min_level"]:
                self.accessories.append(accessory_id)

    def feed(self, amount: int = 25) -> None:
        """Increase hunger (satiety) by the given amount."""
        self.hunger = min(100, self.hunger + amount)

    def pet_stroke(self, amount: int = 15) -> None:
        """Increase happiness by the given amount."""
        self.happiness = min(100, self.happiness + amount)

    def reset_progress(self) -> None:
        """Reset all gamification progress."""
        self.experience = 0
        self.level = 1
        self.hunger = 80
        self.happiness = 80
        self.accessories = []

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
            "avatar": self.avatar,
            "experience": self.experience,
            "level": self.level,
            "hunger": self.hunger,
            "happiness": self.happiness,
            "accessories": self.accessories,
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
            with contextlib.suppress(ValueError):
                self._subs.remove(callback)

        return unsubscribe

    async def queue(self) -> tuple[asyncio.Queue[PetEvent], Callable[[], None]]:
        """Return an async queue that receives all future events.

        The second return value is an unsubscribe function — call it when
        the consumer disconnects so the queue is removed and garbage
        collected instead of accumulating events forever.
        """
        q: asyncio.Queue[PetEvent] = asyncio.Queue(maxsize=256)
        self._queues.append(q)

        def unsubscribe() -> None:
            with contextlib.suppress(ValueError):
                self._queues.remove(q)

        return q, unsubscribe

    def publish(
        self, mood: PetMood, message: str, details: dict[str, Any] | None = None
    ) -> None:
        event = PetEvent(
            timestamp=time.time(),
            mood=mood,
            message=message,
            details=details or {},
        )
        self._state.update(event)
        for cb in self._subs:
            with contextlib.suppress(Exception):
                cb(event)
        for q in self._queues:
            with contextlib.suppress(asyncio.QueueFull):
                q.put_nowait(event)

    def configure(
        self,
        name: str | None = None,
        personality: str | None = None,
        avatar: str | None = None,
    ) -> None:
        self._state.configure(name, personality, avatar=avatar)

    def feed(self, amount: int = 25) -> None:
        """Feed the pet."""
        self._state.feed(amount)
        self.publish(PetMood.HAPPY, "Fed! Feeling good.")

    def pet_stroke(self, amount: int = 15) -> None:
        """Pet the pet."""
        self._state.pet_stroke(amount)
        self.publish(PetMood.HAPPY, "That feels nice!")

    def toggle_accessory(self, accessory_id: str) -> None:
        """Toggle an accessory."""
        self._state.toggle_accessory(accessory_id)

    def reset_progress(self) -> None:
        """Reset gamification progress."""
        self._state.reset_progress()

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


def configure_pet(
    name: str | None = None,
    personality: str | None = None,
    avatar: str | None = None,
) -> None:
    """Configure the global pet name and personality."""
    get_pet_bus().configure(name, personality, avatar=avatar)


def feed_pet(amount: int = 25) -> None:
    """Feed the global pet."""
    get_pet_bus().feed(amount)


def pet_stroke(amount: int = 15) -> None:
    """Pet the global pet."""
    get_pet_bus().pet_stroke(amount)


def toggle_pet_accessory(accessory_id: str) -> None:
    """Toggle an accessory on the global pet."""
    get_pet_bus().toggle_accessory(accessory_id)


def reset_pet_progress() -> None:
    """Reset gamification progress on the global pet."""
    get_pet_bus().reset_progress()
