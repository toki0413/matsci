"""Persistent goal store for long-term iterative task management.

Goals survive across sessions. A goal carries:
- text: what the user wants to achieve
- sub_goals: dynamically appended constraints (/subgoal)
- iteration: how many autoloop rounds have run
- status: active | paused | completed

Persistence reuses PlanStore's file lock + atomic write pattern.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from huginn.autoloop.plan_store import _file_lock, _now_iso


@dataclass
class Goal:
    """A persistent research/development goal.

    lifecycle: active → paused → active → completed
    sub_goals can be appended at any time without restarting.
    """

    id: str
    text: str
    sub_goals: list[str] = field(default_factory=list)
    iteration: int = 0
    status: str = "active"  # active | paused | completed
    created_at: str = ""
    updated_at: str = ""
    session_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = _now_iso()
        if not self.updated_at:
            self.updated_at = self.created_at

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Goal:
        return cls(
            id=data["id"],
            text=data.get("text", ""),
            sub_goals=data.get("sub_goals", []),
            iteration=data.get("iteration", 0),
            status=data.get("status", "active"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            session_id=data.get("session_id", ""),
            metadata=data.get("metadata", {}),
        )


class GoalStore:
    """JSON-backed goal store. Thread-safe, atomic writes.

    File: $HUGINN_CACHE_DIR/goals.json (or .huginn/goals.json)
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or self._default_path()
        self._goals: dict[str, Goal] = {}
        self._lock = threading.Lock()
        self._load()

    @staticmethod
    def _default_path() -> Path:
        base = os.environ.get("HUGINN_CACHE_DIR")
        if base:
            return Path(base) / "goals.json"
        return Path(".huginn") / "goals.json"

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with _file_lock(self._path):
                data = json.loads(self._path.read_text(encoding="utf-8"))
            for row in data.get("goals", []):
                g = Goal.from_dict(row)
                self._goals[g.id] = g
        except (json.JSONDecodeError, TypeError, KeyError, OSError):
            pass

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {"goals": [g.to_dict() for g in self._goals.values()]}
        payload = json.dumps(data, ensure_ascii=False, indent=2)
        with _file_lock(self._path):
            tmp = self._path.with_name(f".{self._path.name}.{os.getpid()}.tmp")
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    f.write(payload)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(str(tmp), str(self._path))
            except OSError:
                tmp.unlink(missing_ok=True)
                raise

    # ── CRUD ──────────────────────────────────────────────────────

    def create_goal(self, text: str, session_id: str = "") -> Goal:
        goal = Goal(
            id=f"goal_{uuid.uuid4().hex[:8]}",
            text=text,
            session_id=session_id,
        )
        with self._lock:
            self._goals[goal.id] = goal
            self._save()
        return goal

    def get_goal(self, goal_id: str) -> Goal | None:
        with self._lock:
            return self._goals.get(goal_id)

    def get_active(self) -> Goal | None:
        """Return the current active goal, if any."""
        with self._lock:
            for g in self._goals.values():
                if g.status == "active":
                    return g
        return None

    def list_goals(self, status: str | None = None) -> list[Goal]:
        with self._lock:
            if status is None:
                return list(self._goals.values())
            return [g for g in self._goals.values() if g.status == status]

    def update_goal(self, goal_id: str, **fields: Any) -> Goal:
        with self._lock:
            goal = self._goals.get(goal_id)
            if goal is None:
                raise KeyError(f"goal not found: {goal_id}")
            for key, value in fields.items():
                if hasattr(goal, key):
                    setattr(goal, key, value)
            goal.updated_at = _now_iso()
            self._save()
            return goal

    def add_sub_goal(self, goal_id: str, text: str) -> Goal:
        with self._lock:
            goal = self._goals.get(goal_id)
            if goal is None:
                raise KeyError(f"goal not found: {goal_id}")
            goal.sub_goals.append(text)
            goal.updated_at = _now_iso()
            self._save()
            return goal

    def increment_iteration(self, goal_id: str) -> Goal:
        with self._lock:
            goal = self._goals.get(goal_id)
            if goal is None:
                raise KeyError(f"goal not found: {goal_id}")
            goal.iteration += 1
            goal.updated_at = _now_iso()
            self._save()
            return goal

    def pause(self, goal_id: str) -> Goal:
        return self.update_goal(goal_id, status="paused")

    def resume(self, goal_id: str) -> Goal:
        return self.update_goal(goal_id, status="active")

    def complete(self, goal_id: str) -> Goal:
        return self.update_goal(goal_id, status="completed")

    def clear(self, goal_id: str) -> None:
        with self._lock:
            self._goals.pop(goal_id, None)
            self._save()


# ── Module-level singleton (lazy) ─────────────────────────────────

_store: GoalStore | None = None
_store_lock = threading.Lock()


def get_goal_store() -> GoalStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = GoalStore()
    return _store


# ── Self-check ────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        store = GoalStore(Path(tmp) / "test_goals.json")
        g = store.create_goal("test objective")
        assert g.status == "active"
        assert g.iteration == 0

        store.add_sub_goal(g.id, "constraint 1")
        assert len(store.get_goal(g.id).sub_goals) == 1

        store.increment_iteration(g.id)
        assert store.get_goal(g.id).iteration == 1

        store.pause(g.id)
        assert store.get_goal(g.id).status == "paused"
        assert store.get_active() is None

        store.resume(g.id)
        assert store.get_active() is not None
        assert store.get_active().id == g.id

        store.complete(g.id)
        assert store.get_goal(g.id).status == "completed"

        # persistence
        store2 = GoalStore(Path(tmp) / "test_goals.json")
        assert len(store2.list_goals()) == 1
        assert store2.get_goal(g.id).text == "test objective"

    print("goal_store self-check OK")
