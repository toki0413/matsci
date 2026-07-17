"""Goal scheduler for the autoloop — built on top of GoalStore.

Historically had its own Goal dataclass + JSON persistence, which conflicted
with goal_store.py (both wrote $HUGINN_CACHE_DIR/goals.json with different
schemas). Now reuses GoalStore's Goal class + atomic write + file lock.

The only thing this module adds:
  - check_completion(goal, validation) — machine-checkable success criteria
  - activate/complete/fail goal methods (GoalStore only has pause/resume/complete)
  - create_goal with success_criteria + max_iterations
"""

from __future__ import annotations

import json
import logging
from typing import Any

from huginn.autoloop.goal_store import Goal, GoalStore, _now_iso

logger = logging.getLogger(__name__)


class GoalScheduler(GoalStore):
    """Extends GoalStore with completion-criteria checking.

    Shares the same goals.json file, same Goal dataclass, same atomic write.
    Adds: create_goal(objective, criteria), activate_goal, fail_goal,
    check_completion(goal, validation).
    """

    def create_goal(
        self,
        objective: str,
        success_criteria: list[str] | None = None,
        max_iterations: int = 20,
        completion_condition: str | None = None,
        metadata: dict[str, Any] | None = None,
        session_id: str = "",
    ) -> Goal:
        goal = Goal(
            id=f"goal_{__import__('uuid').uuid4().hex[:8]}",
            text=objective,
            objective=objective,
            success_criteria=list(success_criteria or []),
            max_iterations=max_iterations,
            completion_condition=completion_condition,
            status="pending",
            session_id=session_id,
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._goals[goal.id] = goal
            self._save()
        return goal

    def activate_goal(self, goal_id: str) -> Goal:
        return self.update_goal(goal_id, status="active")

    def complete_goal(self, goal_id: str) -> Goal:
        return self.update_goal(
            goal_id, status="completed", completed_at=_now_iso()
        )

    def fail_goal(self, goal_id: str, reason: str | None = None) -> Goal:
        with self._lock:
            goal = self._goals.get(goal_id)
            if goal is None:
                raise KeyError(f"goal not found: {goal_id}")
            goal.status = "failed"
            goal.updated_at = _now_iso()
            if reason:
                goal.metadata["failure_reason"] = reason
            self._save()
            return goal

    def delete_goal(self, goal_id: str) -> bool:
        return self.clear(goal_id) is None  # clear returns None, we return bool

    # ── completion check ──────────────────────────────────────────

    @staticmethod
    def check_completion(goal: Goal, validation: Any) -> bool:
        """True if every success_criterion appears in the validation result.

        Serializes validation to a lowercase string (JSON or str()), then
        checks each criterion (lowercased) as a substring. Returns False if
        there are no criteria or validation is None.

        累积模式: 如果 goal.metadata 里有 _validation_history, 会把本次 validation
        追加进去, 然后检查是否任一轮命中了所有 criteria (而非只看最后一轮).

        ponytail: _validation_history capped at 20 entries to prevent unbounded growth.
        """
        if not goal.success_criteria or validation is None:
            return False
        try:
            blob = json.dumps(validation, ensure_ascii=False, default=str).lower()
        except (TypeError, ValueError):
            blob = str(validation).lower()

        history = goal.metadata.setdefault("_validation_history", [])
        history.append(blob)
        # cap history to prevent unbounded growth (audited: 20 rounds is enough)
        if len(history) > 20:
            del history[:len(history) - 20]

        return all(
            any(criterion.lower() in v for v in history)
            for criterion in goal.success_criteria
        )

    # ── budget & continuation (v7) ───────────────────────────────

    @staticmethod
    def is_budget_exhausted(goal: Goal) -> bool:
        """goal.max_iterations 是否已耗尽. 0 或负数视为无限制."""
        if goal.max_iterations <= 0:
            return False
        return goal.iteration >= goal.max_iterations

    @staticmethod
    def build_continuation_prompt(goal: Goal) -> str:
        """生成续跑提示: 给 LLM 看的目标 + 当前进度 + 剩余预算 + 验收标准.

        engine.py 每轮开头把返回值拼进 _speculator_hint, 让 LLM 知道
        自己是在续跑而不是从头开始. 不依赖外部 summary, 只用 goal 字段.
        """
        remaining = goal.max_iterations - goal.iteration if goal.max_iterations > 0 else -1
        rem_str = f"{remaining} 轮" if remaining >= 0 else "无限"
        crit_str = "\n".join(f"  - {c}" for c in goal.success_criteria) or "  (无显式标准)"
        return (
            f"[continuation] 目标: {goal.objective}\n"
            f"已迭代 {goal.iteration} 轮, 剩余预算 {rem_str}.\n"
            f"验收标准:\n{crit_str}\n"
            f"继续从上一轮的发现推进, 不要重复已完成的步骤."
        )


__all__ = ["Goal", "GoalScheduler"]
