"""Side conversation channel — parallel Q&A without interrupting the main task.

Mirrors Codex CLI's /side: while the autoloop or main agent is busy, the user
can post side questions via HTTP. The agent drains the pending queue when it
hits an idle point (e.g. autoloop's perceive-skip sleep) and posts answers
back. The user polls for the answer.

Thread-safe via threading.Lock so FastAPI handlers and the agent loop
(which may run in different threads / tasks) can both touch it. No
asyncio.Event blocking — the agent just drains on its own schedule.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SideQuestion:
    """One side-conversation turn: user question + (later) agent answer."""

    id: str
    question: str
    answer: str | None = None
    created_at: str = field(default_factory=_now_iso)
    answered_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_answered(self) -> bool:
        return self.answer is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "created_at": self.created_at,
            "answered_at": self.answered_at,
            "is_answered": self.is_answered,
        }


class SideChannel:
    """Non-blocking side-question queue.

    submit()  — HTTP handler adds a question, gets an ID back.
    drain()   — agent reads pending questions (non-destructive snapshot).
    respond() — agent posts an answer, moves question pending → answered.
    get()     — HTTP handler polls for a single question's status/answer.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: dict[str, SideQuestion] = {}
        self._answered: dict[str, SideQuestion] = {}

    def submit(self, question: str, metadata: dict[str, Any] | None = None) -> SideQuestion:
        with self._lock:
            sq = SideQuestion(
                id=f"side_{uuid.uuid4().hex[:8]}",
                question=question,
                metadata=dict(metadata or {}),
            )
            self._pending[sq.id] = sq
            return sq

    def drain(self) -> list[SideQuestion]:
        """Snapshot of pending questions. Non-destructive — questions stay
        pending until respond() moves them to answered."""
        with self._lock:
            return list(self._pending.values())

    def respond(self, question_id: str, answer: str) -> SideQuestion | None:
        with self._lock:
            sq = self._pending.pop(question_id, None)
            if sq is None:
                return None
            sq.answer = answer
            sq.answered_at = _now_iso()
            self._answered[question_id] = sq
            return sq

    def get(self, question_id: str) -> SideQuestion | None:
        with self._lock:
            if question_id in self._answered:
                return self._answered[question_id]
            return self._pending.get(question_id)

    def list_all(self) -> list[SideQuestion]:
        with self._lock:
            return list(self._pending.values()) + list(self._answered.values())

    def list_pending(self) -> list[SideQuestion]:
        with self._lock:
            return list(self._pending.values())

    def list_answered(self) -> list[SideQuestion]:
        with self._lock:
            return list(self._answered.values())

    def clear(self) -> None:
        with self._lock:
            self._pending.clear()
            self._answered.clear()

    @property
    def n_pending(self) -> int:
        with self._lock:
            return len(self._pending)

    @property
    def n_answered(self) -> int:
        with self._lock:
            return len(self._answered)


# ── shared singleton (same pattern as InterruptManager) ─────────────────────

_shared: SideChannel | None = None
_shared_lock = threading.Lock()


def get_shared_side_channel() -> SideChannel:
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = SideChannel()
        return _shared


def set_shared_side_channel(channel: SideChannel | None) -> None:
    """Inject a fresh channel (tests) or None to reset."""
    global _shared
    with _shared_lock:
        _shared = channel


__all__ = [
    "SideQuestion",
    "SideChannel",
    "get_shared_side_channel",
    "set_shared_side_channel",
]
