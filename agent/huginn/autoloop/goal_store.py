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

    Unified goal: supports both GoalStore (iteration tracking, unknowns)
    and GoalScheduler (success_criteria, completion check) use cases.
    """

    id: str
    text: str
    sub_goals: list[str] = field(default_factory=list)
    iteration: int = 0
    status: str = "active"  # active | paused | completed | pending | failed
    created_at: str = ""
    updated_at: str = ""
    session_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    # Unknown tracking — 诊断指标, 不是奖励信号.
    unknowns: list[dict[str, Any]] = field(default_factory=list)
    unknowns_discovered: int = 0
    unknowns_resolved: int = 0
    # Scheduler fields (optional — used by GoalScheduler)
    objective: str = ""  # alias for text when used via scheduler
    success_criteria: list[str] = field(default_factory=list)
    max_iterations: int = 20
    completed_at: str | None = None
    completion_condition: str | None = None

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = _now_iso()
        if not self.updated_at:
            self.updated_at = self.created_at
        # text ↔ objective sync: whichever was set, mirror to the other
        if not self.text and self.objective:
            self.text = self.objective
        elif not self.objective and self.text:
            self.objective = self.text

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
            unknowns=data.get("unknowns", []),
            unknowns_discovered=data.get("unknowns_discovered", 0),
            unknowns_resolved=data.get("unknowns_resolved", 0),
            objective=data.get("objective", ""),
            success_criteria=data.get("success_criteria", []),
            max_iterations=data.get("max_iterations", 20),
            completed_at=data.get("completed_at"),
            completion_condition=data.get("completion_condition"),
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
        except (json.JSONDecodeError, TypeError, KeyError, OSError) as e:
            # don't silently swallow — log so user knows their goals file is corrupted
            import logging
            logging.getLogger(__name__).warning(
                "goals.json load failed (%s), starting fresh", e
            )

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

    def add_unknown(
        self, goal_id: str, text: str, unknown_type: str = "unknown_unknown"
    ) -> str | None:
        """记录一个新发现的 unknown.

        unknown_type:
          - "known_unknown": 知道自己不知道 (e.g. "PBE 会低估带隙但我不知道具体差多少")
          - "unknown_unknown": 完全没想到的盲区 (e.g. "GaN 有两种结构我之前完全没考虑")
          - "blind_spot": 盲区扫描发现的 (pre-implementation)

        返回 unknown_id, 可用于后续 resolve_unknown.
        不追踪 discovery rate — 只追踪 resolution ratio 和 half-life.
        """
        import uuid as _uuid
        with self._lock:
            goal = self._goals.get(goal_id)
            if goal is None:
                return None
            uid = f"unk_{_uuid.uuid4().hex[:8]}"
            goal.unknowns.append({
                "id": uid,
                "text": text,
                "type": unknown_type,
                "discovered_at": _now_iso(),
                "resolved_at": None,
                "iteration": goal.iteration,
            })
            goal.unknowns_discovered += 1
            goal.updated_at = _now_iso()
            self._save()
            return uid

    def resolve_unknown(self, goal_id: str, unknown_id: str) -> bool:
        """标记一个 unknown 已解决. 返回是否找到并更新了."""
        with self._lock:
            goal = self._goals.get(goal_id)
            if goal is None:
                return False
            for u in goal.unknowns:
                if u["id"] == unknown_id and u["resolved_at"] is None:
                    u["resolved_at"] = _now_iso()
                    goal.unknowns_resolved += 1
                    goal.updated_at = _now_iso()
                    self._save()
                    return True
            return False

    def unknown_stats(self, goal_id: str) -> dict[str, Any]:
        """诊断统计: 消解率 + 半衰期. 纯观察量, 不作为奖励信号."""
        with self._lock:
            goal = self._goals.get(goal_id)
            if goal is None:
                return {}
        total = goal.unknowns_discovered
        resolved = goal.unknowns_resolved
        ratio = resolved / total if total > 0 else 0.0
        # half-life: 已解决的 unknown 从发现到解决的中位时间 (hours)
        from datetime import datetime
        durations: list[float] = []
        for u in goal.unknowns:
            if u["resolved_at"] and u["discovered_at"]:
                try:
                    d = datetime.fromisoformat(u["discovered_at"])
                    r = datetime.fromisoformat(u["resolved_at"])
                    durations.append((r - d).total_seconds() / 3600)
                except Exception:
                    pass
        durations.sort()
        median_h = durations[len(durations) // 2] if durations else None
        open_count = total - resolved
        return {
            "discovered": total,
            "resolved": resolved,
            "open": open_count,
            "resolution_ratio": round(ratio, 2),
            "median_half_life_h": round(median_h, 1) if median_h else None,
        }

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
