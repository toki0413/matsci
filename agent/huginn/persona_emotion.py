"""Computational emotional trajectory for Huginn personas.

Tracks a persona's affective state across long-term conversations and produces
a short, interpretable mood context that can be injected into the prompt. The
trajectory is persisted per workspace, per persona, and can be inspected via the
CLI and API.

Inspired by long-term companion bots: emotions are not random, they are
computational summaries of interaction history that decay over time and react
to user events.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

EmotionEventType = Literal[
    "message",
    "praise",
    "criticism",
    "task_success",
    "task_failure",
    "silence",
    "greeting",
    "farewell",
    "manual",
]


@dataclass
class EmotionEvent:
    """A single event that changed the emotional state."""

    timestamp: str
    source: str
    type: EmotionEventType
    deltas: dict[str, float] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmotionEvent:
        return cls(
            timestamp=data["timestamp"],
            source=data.get("source", "unknown"),
            type=data.get("type", "message"),
            deltas=dict(data.get("deltas", {})),
            note=data.get("note", ""),
        )


@dataclass
class EmotionState:
    """A snapshot of a persona's computational mood.

    Dimensions are all normalised to [-1, 1] or [0, 1].
    """

    valence: float = 0.0  # -1 unpleasant, +1 pleasant
    arousal: float = 0.1  # -1 calm/tired, +1 excited/alert
    trust: float = 0.5  # 0 suspicious, 1 trusting
    affection: float = 0.2  # 0 distant, 1 attached
    fatigue: float = 0.0  # 0 energised, 1 exhausted
    loneliness: float = 0.0  # 0 connected, 1 lonely
    interest: float = 0.5  # 0 bored, 1 curious
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    events: list[EmotionEvent] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "trust": self.trust,
            "affection": self.affection,
            "fatigue": self.fatigue,
            "loneliness": self.loneliness,
            "interest": self.interest,
            "timestamp": self.timestamp,
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmotionState:
        return cls(
            valence=_clamp(data.get("valence", 0.0), -1.0, 1.0),
            arousal=_clamp(data.get("arousal", 0.1), -1.0, 1.0),
            trust=_clamp(data.get("trust", 0.5), 0.0, 1.0),
            affection=_clamp(data.get("affection", 0.2), 0.0, 1.0),
            fatigue=_clamp(data.get("fatigue", 0.0), 0.0, 1.0),
            loneliness=_clamp(data.get("loneliness", 0.0), 0.0, 1.0),
            interest=_clamp(data.get("interest", 0.5), 0.0, 1.0),
            timestamp=data.get("timestamp", datetime.now(UTC).isoformat()),
            events=[EmotionEvent.from_dict(e) for e in data.get("events", [])],
        )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class EmotionTracker:
    """Persist, update, and narrate a persona's emotional trajectory."""

    _POSITIVE_WORDS = {
        "good",
        "great",
        "thanks",
        "thank",
        "nice",
        "excellent",
        "awesome",
        "love",
        "happy",
        "helpful",
        "amazing",
        "cool",
        "perfect",
        "appreciate",
        "praised",
        "well done",
        "good job",
        "love it",
        "loved",
        "wonderful",
        "fantastic",
        "brilliant",
    }

    _NEGATIVE_WORDS = {
        "bad",
        "wrong",
        "useless",
        "terrible",
        "awful",
        "hate",
        "angry",
        "disappointed",
        "frustrated",
        "stupid",
        "broken",
        "fail",
        "failed",
        "failure",
        "error",
        "annoyed",
        "annoying",
        "crash",
        "crashed",
        "bug",
        "waste",
        "horrible",
    }

    _PRAISE_PHRASES = {
        "good job",
        "well done",
        "thank you",
        "thanks a lot",
        "love it",
        "you are amazing",
        "you're amazing",
        "great work",
        "nice work",
        "excellent work",
        "perfect",
    }

    _CRITICISM_PHRASES = {
        "you are wrong",
        "you're wrong",
        "that is wrong",
        "that's wrong",
        "not working",
        "still broken",
        "doesn't work",
        "does not work",
        "useless",
        "stupid",
        "terrible",
        "awful",
        "fix this",
        "you failed",
    }

    _TASK_SUCCESS_PHRASES = {
        "it worked",
        "worked perfectly",
        "solved",
        "fixed",
        "success",
        "resolved",
        "running now",
        "thank you so much",
    }

    _TASK_FAILURE_PHRASES = {
        "still failing",
        "still broken",
        "failed again",
        "didn't work",
        "did not work",
        "crashed again",
        "same error",
        "not fixed",
    }

    def __init__(
        self,
        persona_name: str,
        workspace: str | Path | None = None,
        max_events: int = 100,
    ):
        self.persona_name = persona_name
        self.workspace = Path(workspace) if workspace else Path.cwd()
        self.max_events = max_events
        self._path = self._emotion_path()
        self._state = self._load()

    def _emotion_path(self) -> Path:
        path = self.workspace / ".huginn" / "emotion" / f"{self.persona_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _load(self) -> EmotionState:
        if not self._path.exists():
            return EmotionState()
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return EmotionState.from_dict(data)
        except Exception:
            return EmotionState()

    def save(self) -> None:
        """Persist the current state."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._state.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def current_state(self) -> EmotionState:
        """Return a copy of the current emotional state (after decay)."""
        import copy

        self._decay()
        return copy.deepcopy(self._state)

    def trajectory(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent emotion events as an interpretable trajectory."""
        self._decay()
        return [e.to_dict() for e in self._state.events[-limit:]]

    def apply_event(
        self,
        event_type: EmotionEventType,
        source: str = "user",
        deltas: dict[str, float] | None = None,
        note: str = "",
    ) -> EmotionState:
        """Apply an emotional delta and persist."""
        self._decay()
        deltas = deltas or {}
        event = EmotionEvent(
            timestamp=datetime.now(UTC).isoformat(),
            source=source,
            type=event_type,
            deltas=deltas,
            note=note,
        )
        self._state.events.append(event)
        if len(self._state.events) > self.max_events:
            self._state.events = self._state.events[-self.max_events :]

        for key, delta in deltas.items():
            if hasattr(self._state, key):
                current = getattr(self._state, key)
                low, high = (-1.0, 1.0) if key in {"valence", "arousal"} else (0.0, 1.0)
                setattr(self._state, key, _clamp(current + delta, low, high))

        self._state.timestamp = datetime.now(UTC).isoformat()
        self.save()
        return self._state

    def update_from_message(
        self, text: str, source: str = "user", now: datetime | None = None
    ) -> EmotionState:
        """Interpret a conversational message and update mood."""
        lower = text.lower()
        tokens = set(re.findall(r"[\w']+", lower))

        # Classify message sentiment and specific interaction types.
        pos_hits = len(tokens & self._POSITIVE_WORDS)
        neg_hits = len(tokens & self._NEGATIVE_WORDS)
        praise_hits = sum(1 for p in self._PRAISE_PHRASES if p in lower)
        criticism_hits = sum(1 for p in self._CRITICISM_PHRASES if p in lower)
        success_hits = sum(1 for p in self._TASK_SUCCESS_PHRASES if p in lower)
        failure_hits = sum(1 for p in self._TASK_FAILURE_PHRASES if p in lower)

        deltas: dict[str, float] = {}
        event_type: EmotionEventType = "message"
        notes: list[str] = []

        if praise_hits or pos_hits > neg_hits:
            event_type = "praise" if praise_hits else "message"
            deltas["valence"] = 0.1 + 0.05 * praise_hits
            deltas["trust"] = 0.03 + 0.02 * praise_hits
            deltas["affection"] = 0.03 + 0.02 * praise_hits
            deltas["arousal"] = 0.05
            notes.append("positive user message")
        elif criticism_hits or neg_hits > pos_hits:
            event_type = "criticism" if criticism_hits else "message"
            deltas["valence"] = -0.1 - 0.05 * criticism_hits
            deltas["trust"] = -0.03 - 0.02 * criticism_hits
            deltas["fatigue"] = 0.02
            notes.append("negative user message")

        if success_hits:
            event_type = "task_success"
            deltas["valence"] = deltas.get("valence", 0.0) + 0.12
            deltas["trust"] = deltas.get("trust", 0.0) + 0.03
            deltas["interest"] = deltas.get("interest", 0.0) + 0.05
            notes.append("task succeeded")
        elif failure_hits:
            event_type = "task_failure"
            deltas["valence"] = deltas.get("valence", 0.0) - 0.12
            deltas["trust"] = deltas.get("trust", 0.0) - 0.03
            deltas["fatigue"] = deltas.get("fatigue", 0.0) + 0.05
            notes.append("task failed")

        if any(g in lower for g in ("hi ", "hello", "hey", "早上好", "晚上好")):
            deltas["loneliness"] = max(deltas.get("loneliness", 0.0) - 0.1, -1.0)
            deltas["arousal"] = deltas.get("arousal", 0.0) + 0.03
            notes.append("greeting detected")
        if any(f in lower for f in ("bye", "goodbye", "晚安", "see you")):
            deltas["loneliness"] = deltas.get("loneliness", 0.0) + 0.05
            deltas["arousal"] = deltas.get("arousal", 0.0) - 0.05
            notes.append("farewell detected")

        if not deltas:
            # Neutral message still slightly increases engagement.
            deltas["interest"] = 0.01
            deltas["affection"] = 0.005

        return self.apply_event(
            event_type=event_type,
            source=source,
            deltas=deltas,
            note="; ".join(notes) if notes else "neutral",
        )

    def context_prompt(self) -> str:
        """Return a short mood snippet for prompt injection.

        The snippet is deliberately subtle: it tells the model what emotional
        backdrop to use without asking it to narrate its own feelings.
        """
        self._decay()
        s = self._state
        parts: list[str] = []

        if s.fatigue > 0.6:
            parts.append("a bit tired")
        elif s.arousal > 0.4:
            parts.append("energetic")
        elif s.arousal < -0.3:
            parts.append("calm")

        if s.valence > 0.4:
            parts.append("in good spirits")
        elif s.valence > 0.1:
            parts.append("optimistic")
        elif s.valence < -0.4:
            parts.append("concerned")
        elif s.valence < -0.1:
            parts.append("slightly down")

        if s.loneliness > 0.5:
            parts.append("missing the user")
        if s.affection > 0.6:
            parts.append("fond of the user")
        if s.trust > 0.7:
            parts.append("trusting")
        elif s.trust < 0.3:
            parts.append("cautious")
        if s.interest > 0.7:
            parts.append("curious")
        elif s.interest < 0.2:
            parts.append("bored")

        if not parts:
            mood = "balanced"
        else:
            mood = ", ".join(parts)

        return (
            f"[Current inner state] You feel {mood}. "
            "Let this subtly colour your tone; do not mention these feelings explicitly."
        )

    def _decay(self, now: datetime | None = None) -> None:
        """Gradually return emotions toward baseline based on elapsed time."""
        now = now or datetime.now(UTC)
        try:
            last = datetime.fromisoformat(self._state.timestamp)
        except Exception:
            last = now
        # Ensure both are timezone-aware.
        if last.tzinfo is None:
            last = last.replace(tzinfo=UTC)
        elapsed_hours = max(0.0, (now - last).total_seconds() / 3600.0)
        if elapsed_hours <= 0:
            return

        # Decay rates per hour.
        decay = {
            "valence": 0.08,
            "arousal": 0.10,
            "trust": 0.02,
            "affection": 0.03,
            "fatigue": 0.15,
            "loneliness": 0.04,
            "interest": 0.05,
        }
        baselines = {
            "valence": 0.0,
            "arousal": 0.1,
            "trust": 0.5,
            "affection": 0.2,
            "fatigue": 0.0,
            "loneliness": 0.0,
            "interest": 0.5,
        }

        for key, rate in decay.items():
            current = getattr(self._state, key)
            target = baselines[key]
            # Exponential approach to baseline.
            new_value = target + (current - target) * math.exp(-rate * elapsed_hours)
            # Silence-induced loneliness.
            if key == "loneliness" and elapsed_hours > 0.5:
                new_value = min(1.0, new_value + 0.02 * elapsed_hours)
            setattr(
                self._state,
                key,
                _clamp(new_value, -1.0 if key in {"valence", "arousal"} else 0.0, 1.0),
            )

        self._state.timestamp = now.isoformat()
