"""Persistent goal scheduler for the autoloop.

A Goal ties an objective to machine-checkable success criteria. The engine
checks ``check_completion(goal, validation)`` after each learn phase and stops
early when all criteria are satisfied — so a loop doesn't burn all 20
iterations after it already hit its target.

GoalScheduler persists goals to ``$HUGINN_CACHE_DIR/goals.json`` so a goal
survives across CLI invocations. The engine itself only needs the Goal
dataclass + check_completion; the scheduler handles create/list/complete/fail
for the CLI layer.

Completion check is deliberately simple: serialize the validation dict to a
lowercase string, then check each criterion (also lowercased) appears as a
substring. No LLM call, no regex — callers pick keyword criteria that map to
real validation output (e.g. ``["tests_passed": true, "r_phys > 0.8"]`` →
criteria ``["tests_passed", "r_phys"]``).
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Goal:
    """A persistent research goal with machine-checkable completion criteria.

    success_criteria: list of strings — each must appear (case-insensitive
        substring) in the serialized validation result for the goal to count
        as completed.
    status: pending | active | completed | failed
    completion_condition: optional human-readable description, not used by
        check_completion (kept for UI / report display).
    """

    id: str
    objective: str
    success_criteria: list[str]
    status: str = "pending"
    max_iterations: int = 20
    created_at: str = ""
    updated_at: str = ""
    completed_at: str | None = None
    completion_condition: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = _now_iso()
        if not self.updated_at:
            self.updated_at = self.created_at


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class GoalScheduler:
    """Persistent goal store backed by a JSON file.

    The file lives at ``$HUGINN_CACHE_DIR/goals.json`` (or ``.huginn/goals.json``
    if the env var is unset). Tests inject a temp path via the ``path`` arg.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or self._default_path()
        self._goals: dict[str, Goal] = {}
        self._load()

    @staticmethod
    def _default_path() -> Path:
        base = os.environ.get("HUGINN_CACHE_DIR")
        if base:
            return Path(base) / "goals.json"
        return Path(".huginn") / "goals.json"

    # ── persistence ───────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            for row in data.get("goals", []):
                goal = Goal(**row)
                self._goals[goal.id] = goal
        except (json.JSONDecodeError, TypeError, KeyError):
            # corrupt file — start fresh, don't crash the engine
            pass

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {"goals": [g.__dict__ for g in self._goals.values()]}
        self._path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ── CRUD ──────────────────────────────────────────────────────

    def create_goal(
        self,
        objective: str,
        success_criteria: list[str] | None = None,
        max_iterations: int = 20,
        completion_condition: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Goal:
        goal = Goal(
            id=f"goal_{uuid.uuid4().hex[:8]}",
            objective=objective,
            success_criteria=list(success_criteria or []),
            max_iterations=max_iterations,
            completion_condition=completion_condition,
            metadata=dict(metadata or {}),
        )
        self._goals[goal.id] = goal
        self._save()
        return goal

    def get_goal(self, goal_id: str) -> Goal | None:
        return self._goals.get(goal_id)

    def list_goals(self, status: str | None = None) -> list[Goal]:
        if status is None:
            return list(self._goals.values())
        return [g for g in self._goals.values() if g.status == status]

    def update_goal(self, goal_id: str, **fields: Any) -> Goal:
        goal = self._goals.get(goal_id)
        if goal is None:
            raise KeyError(f"goal not found: {goal_id}")
        for key, value in fields.items():
            if hasattr(goal, key):
                setattr(goal, key, value)
        goal.updated_at = _now_iso()
        self._save()
        return goal

    def activate_goal(self, goal_id: str) -> Goal:
        return self.update_goal(goal_id, status="active")

    def complete_goal(self, goal_id: str) -> Goal:
        return self.update_goal(
            goal_id, status="completed", completed_at=_now_iso()
        )

    def fail_goal(self, goal_id: str, reason: str | None = None) -> Goal:
        meta = dict(self._goals[goal_id].metadata) if goal_id in self._goals else {}
        if reason:
            meta["failure_reason"] = reason
        return self.update_goal(goal_id, status="failed", metadata=meta)

    def delete_goal(self, goal_id: str) -> bool:
        if goal_id not in self._goals:
            return False
        del self._goals[goal_id]
        self._save()
        return True

    # ── completion check ──────────────────────────────────────────

    @staticmethod
    def check_completion(goal: Goal, validation: Any) -> bool:
        """True if every success_criterion appears in the validation result.

        Serializes validation to a lowercase string (JSON or str()), then
        checks each criterion (lowercased) as a substring. Returns False if
        there are no criteria or validation is None.

        累积模式: 如果 goal.metadata 里有 _validation_history, 会把本次 validation
        追加进去, 然后检查是否任一轮命中了所有 criteria (而非只看最后一轮).
        """
        if not goal.success_criteria or validation is None:
            return False
        try:
            blob = json.dumps(validation, ensure_ascii=False, default=str).lower()
        except (TypeError, ValueError):
            blob = str(validation).lower()

        # 累积验证历史: 不只看最后一轮, 所有轮的命中情况都算
        history = goal.metadata.setdefault("_validation_history", [])
        history.append(blob)

        # 每个 criterion 只要在任一轮命中就算通过
        return all(
            any(criterion.lower() in v for v in history)
            for criterion in goal.success_criteria
        )


__all__ = ["Goal", "GoalScheduler"]
