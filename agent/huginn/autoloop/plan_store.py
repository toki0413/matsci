"""Persistent plan store for the planner/executor separation.

Plans are structured artifacts (objective + ordered steps with dependencies)
that survive across CLI invocations and HTTP requests. The planner role writes
a draft plan here; the user (or auto_confirm) promotes it to confirmed; the
executor role reads confirmed plans and runs them step by step.

Persistence model clones GoalScheduler: JSON file at
``$HUGINN_CACHE_DIR/plans.json`` (or ``.huginn/plans.json``), thread-safe via
a single lock, corrupt-file recovery by starting fresh.

Plan lifecycle:
    draft → confirmed → executing → completed
                       └→ failed
    draft → abandoned   (rejected)
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
import uuid
from dataclasses import asdict, dataclass, field
from huginn.utils.common import now_iso
from pathlib import Path
from typing import Any


@dataclass
class PlanStep:
    """One executable step inside a plan.

    dependencies: list of step ids that must reach status=done before this
        step can start. The orchestrator's existing dependency resolver
        handles ordering.
    agent_id: which agent profile runs this step. Defaults to "lead".
    """

    id: str
    description: str
    tool: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    agent_id: str = "lead"
    status: str = "pending"  # pending | running | done | error | skipped
    result: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlanStep:
        return cls(
            id=data["id"],
            description=data.get("description", ""),
            tool=data.get("tool"),
            parameters=data.get("parameters", {}),
            dependencies=data.get("dependencies", []),
            agent_id=data.get("agent_id", "lead"),
            status=data.get("status", "pending"),
            result=data.get("result", ""),
            error=data.get("error"),
        )


@dataclass
class Plan:
    """A persisted plan with structured steps.

    status transitions:
        draft → confirmed (user approves, or auto_confirm=True)
        confirmed → executing (executor starts)
        executing → completed | failed
        draft → abandoned (user rejects)
    """

    id: str
    objective: str
    steps: list[PlanStep]
    status: str = "draft"
    auto_confirm: bool = False
    created_at: str = ""
    confirmed_at: str | None = None
    completed_at: str | None = None
    reject_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = now_iso()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "objective": self.objective,
            "steps": [s.to_dict() for s in self.steps],
            "status": self.status,
            "auto_confirm": self.auto_confirm,
            "created_at": self.created_at,
            "confirmed_at": self.confirmed_at,
            "completed_at": self.completed_at,
            "reject_reason": self.reject_reason,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Plan:
        steps = [PlanStep.from_dict(s) for s in data.get("steps", [])]
        return cls(
            id=data["id"],
            objective=data.get("objective", ""),
            steps=steps,
            status=data.get("status", "draft"),
            auto_confirm=data.get("auto_confirm", False),
            created_at=data.get("created_at", ""),
            confirmed_at=data.get("confirmed_at"),
            completed_at=data.get("completed_at"),
            reject_reason=data.get("reject_reason"),
            metadata=data.get("metadata", {}),
        )


@contextlib.contextmanager
def _file_lock(path: Path):
    """Cross-process lock for the plans file.

    msvcrt.locking on Windows (blocks with retries), fcntl.flock on Unix
    (blocks indefinitely). Prevents agent / autoloop / server processes
    from racing on plans.json reads and writes.
    """
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)
        try:
            lock_path.unlink()
        except OSError:
            pass


class PlanStore:
    """Persistent plan store backed by a JSON file.

    File location: ``$HUGINN_CACHE_DIR/plans.json`` (or ``.huginn/plans.json``).
    Tests inject a temp path via the ``path`` arg.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or self._default_path()
        self._plans: dict[str, Plan] = {}
        self._lock = threading.Lock()
        self._load()

    @staticmethod
    def _default_path() -> Path:
        base = os.environ.get("HUGINN_CACHE_DIR")
        if base:
            return Path(base) / "plans.json"
        return Path(".huginn") / "plans.json"

    # ── persistence ───────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with _file_lock(self._path):
                data = json.loads(self._path.read_text(encoding="utf-8"))
            for row in data.get("plans", []):
                plan = Plan.from_dict(row)
                self._plans[plan.id] = plan
        except (json.JSONDecodeError, TypeError, KeyError, OSError):
            # corrupt file or lock failure — start fresh, don't crash
            pass

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {"plans": [p.to_dict() for p in self._plans.values()]}
        payload = json.dumps(data, ensure_ascii=False, indent=2)
        # write to temp, fsync, then atomic rename — prevents half-written files
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

    def create_plan(
        self,
        objective: str,
        steps: list[PlanStep],
        auto_confirm: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> Plan:
        plan = Plan(
            id=f"plan_{uuid.uuid4().hex[:8]}",
            objective=objective,
            steps=list(steps),
            auto_confirm=auto_confirm,
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._plans[plan.id] = plan
            self._save()
        return plan

    def get_plan(self, plan_id: str) -> Plan | None:
        with self._lock:
            return self._plans.get(plan_id)

    def list_plans(self, status: str | None = None) -> list[Plan]:
        with self._lock:
            if status is None:
                return list(self._plans.values())
            return [p for p in self._plans.values() if p.status == status]

    def update_plan(self, plan_id: str, **fields: Any) -> Plan:
        with self._lock:
            plan = self._plans.get(plan_id)
            if plan is None:
                raise KeyError(f"plan not found: {plan_id}")
            for key, value in fields.items():
                if hasattr(plan, key):
                    setattr(plan, key, value)
            self._save()
            return plan

    def confirm_plan(self, plan_id: str) -> Plan:
        return self.update_plan(
            plan_id, status="confirmed", confirmed_at=now_iso()
        )

    def reject_plan(self, plan_id: str, reason: str | None = None) -> Plan:
        return self.update_plan(
            plan_id,
            status="abandoned",
            reject_reason=reason,
        )

    def mark_executing(self, plan_id: str) -> Plan:
        return self.update_plan(plan_id, status="executing")

    def complete_plan(self, plan_id: str) -> Plan:
        return self.update_plan(
            plan_id, status="completed", completed_at=now_iso()
        )

    def fail_plan(self, plan_id: str, reason: str | None = None) -> Plan:
        meta = dict(self._plans.get(plan_id).metadata) if plan_id in self._plans else {}
        if reason:
            meta["failure_reason"] = reason
        return self.update_plan(
            plan_id, status="failed", metadata=meta
        )

    def update_step(self, plan_id: str, step_id: str, **fields: Any) -> Plan:
        with self._lock:
            plan = self._plans.get(plan_id)
            if plan is None:
                raise KeyError(f"plan not found: {plan_id}")
            for step in plan.steps:
                if step.id == step_id:
                    for key, value in fields.items():
                        if hasattr(step, key):
                            setattr(step, key, value)
                    break
            self._save()
            return plan

    def delete_plan(self, plan_id: str) -> bool:
        with self._lock:
            if plan_id not in self._plans:
                return False
            del self._plans[plan_id]
            self._save()
            return True

    def export_markdown(self, plan_id: str, path: "Path | None" = None) -> "Path":
        """导出 plan 为 markdown 文件, 供 chat mode 引用 active step.

        Anthropic Context Management 2026: plan 持久化到文件, chat 上下文只引用
        当前 active step, 不引用完整 plan. 降低 token 占用 + mode 间隔离.
        ponytail: 纯展示层, 不改 JSON 持久化语义. 升级: 按 step 状态动态裁剪.
        """
        from pathlib import Path as _Path
        plan = self.get_plan(plan_id)
        if plan is None:
            raise KeyError(f"plan not found: {plan_id}")
        if path is None:
            cache = _Path.home() / ".huginn" / "plans_md"
            cache.mkdir(parents=True, exist_ok=True)
            path = cache / f"{plan_id}.md"
        else:
            path = _Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            f"# Plan: {plan.objective}",
            f"",
            f"- **ID**: `{plan.id}`",
            f"- **Status**: {plan.status}",
            f"- **Created**: {plan.created_at}",
        ]
        if plan.confirmed_at:
            lines.append(f"- **Confirmed**: {plan.confirmed_at}")
        if plan.completed_at:
            lines.append(f"- **Completed**: {plan.completed_at}")
        if plan.reject_reason:
            lines.append(f"- **Reject Reason**: {plan.reject_reason}")
        lines.append("")
        lines.append("## Steps")
        lines.append("")
        lines.append("| # | Step | Tool | Status | Result |")
        lines.append("|---|------|------|--------|--------|")
        for i, step in enumerate(plan.steps, 1):
            result = ""
            if step.result:
                result = str(step.result)[:80].replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {i} | {step.description} | {step.tool or '-'} | {step.status} | {result} |"
            )
        lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
        return path


__all__ = ["Plan", "PlanStep", "PlanStore"]
