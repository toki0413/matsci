"""Dynamic workflow orchestration — parallel subtask execution.

Mirrors Claude Code's Dynamic Workflows: the agent writes a declarative script
with N independent subtasks (each a tool call), the orchestrator runs them
concurrently (asyncio.gather + semaphore, default 8), aggregates results, and
returns. Failed subtasks are marked but don't crash the whole workflow — the
main agent decides whether to retry or skip.

This is deliberately lighter than workflows/engine.py (which does topological
stage dependencies for DFT/MD pipelines). Here subtasks are independent peers;
if you need sequencing, write multiple scripts or chain them in the agent loop.

Two entry points:
- WorkflowOrchestrator.run(script) — async, awaits all subtasks, returns
  WorkflowResult. Used by the autoloop engine's EXECUTION phase directly.
- WorkflowRegistry shared singleton — tracks running workflows by id, so the
  workflow_tool (submit_script / status / cancel / collect) can manage them
  from the chat/HTTP layer without holding a reference.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from huginn.tools.registry import ToolRegistry
from huginn.types import ToolContext
import logging
logger = logging.getLogger(__name__)



def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── dataclasses ──────────────────────────────────────────────────────────────


@dataclass
class Subtask:
    """One independent tool call inside a workflow script."""

    id: str
    tool_name: str
    args: dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class WorkflowScript:
    """Declarative workflow: N independent subtasks + concurrency cap."""

    id: str
    objective: str
    subtasks: list[Subtask]
    max_concurrent: int = 8

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowScript":
        """Parse a script dict (from LLM JSON / tool args).

        Expected shape:
            {"objective": "...", "max_concurrent": 8, "subtasks": [
                {"id": "s1", "tool": "vasp_tool", "args": {...}, "description": "..."},
                ...
            ]}
        """
        objective = str(data.get("objective", ""))
        max_concurrent = int(data.get("max_concurrent", 8))
        raw_subtasks = data.get("subtasks", [])
        subtasks: list[Subtask] = []
        for i, st in enumerate(raw_subtasks):
            if not isinstance(st, dict):
                continue
            sid = str(st.get("id", f"sub_{i+1}"))
            tool_name = str(st.get("tool", st.get("tool_name", "")))
            if not tool_name:
                continue
            subtasks.append(
                Subtask(
                    id=sid,
                    tool_name=tool_name,
                    args=dict(st.get("args", {})),
                    description=str(st.get("description", "")),
                )
            )
        return cls(
            id=f"wf_{uuid.uuid4().hex[:8]}",
            objective=objective,
            subtasks=subtasks,
            max_concurrent=max(1, min(max_concurrent, 64)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "objective": self.objective,
            "max_concurrent": self.max_concurrent,
            "n_subtasks": len(self.subtasks),
            "subtasks": [
                {
                    "id": s.id,
                    "tool": s.tool_name,
                    "args": s.args,
                    "description": s.description,
                }
                for s in self.subtasks
            ],
        }


@dataclass
class SubtaskResult:
    """Outcome of one subtask."""

    id: str
    tool_name: str
    status: str = "pending"  # pending | running | completed | failed
    output: Any = None
    error: str = ""
    started_at: str = ""
    completed_at: str = ""

    @property
    def is_done(self) -> bool:
        return self.status in ("completed", "failed")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tool": self.tool_name,
            "status": self.status,
            "output": _serialize(self.output),
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


@dataclass
class WorkflowResult:
    """Aggregated outcome of a whole workflow run."""

    id: str
    objective: str
    status: str = "pending"  # pending | running | completed | failed | cancelled
    subtask_results: dict[str, SubtaskResult] = field(default_factory=dict)
    started_at: str = ""
    completed_at: str = ""

    @property
    def n_completed(self) -> int:
        return sum(1 for r in self.subtask_results.values() if r.status == "completed")

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.subtask_results.values() if r.status == "failed")

    @property
    def n_total(self) -> int:
        return len(self.subtask_results)

    @property
    def success(self) -> bool:
        """True if at least one subtask completed and no catastrophic failure."""
        return self.status == "completed" and self.n_completed > 0

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "objective": self.objective,
            "status": self.status,
            "n_total": self.n_total,
            "n_completed": self.n_completed,
            "n_failed": self.n_failed,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.summary(),
            "subtasks": {
                sid: r.to_dict() for sid, r in self.subtask_results.items()
            },
        }


def _serialize(obj: Any) -> Any:
    """Best-effort JSON-safe serialization for result outputs."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    # ToolResult / dataclass / object → try dict, fall back to str
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            logger.debug("to dict failed", exc_info=True)
    if hasattr(obj, "__dict__"):
        try:
            return {k: _serialize(v) for k, v in vars(obj).items() if not k.startswith("_")}
        except Exception:
            logger.debug("serialize failed", exc_info=True)
    return str(obj)


# ── orchestrator ─────────────────────────────────────────────────────────────


class WorkflowOrchestrator:
    """Runs a WorkflowScript's subtasks concurrently, aggregates results.

    Each subtask is a tool call looked up via ToolRegistry. Concurrency is
    capped by an asyncio.Semaphore (default 8). Failed subtasks are caught
    and marked — they don't cancel sibling subtasks.
    """

    def __init__(
        self,
        max_concurrent: int = 8,
        registry: ToolRegistry | None = None,
    ) -> None:
        self.max_concurrent = max(1, min(max_concurrent, 64))
        # ToolRegistry is a class-level singleton, but allow injection for tests
        self._registry = registry or ToolRegistry

    async def run(
        self,
        script: WorkflowScript,
        context: ToolContext | None = None,
    ) -> WorkflowResult:
        """Execute all subtasks concurrently, return aggregated result.

        context: ToolContext passed to each tool.call(). If None, a minimal
        context is built from the script id.
        """
        result = WorkflowResult(
            id=script.id,
            objective=script.objective,
            status="running",
            started_at=_now_iso(),
        )
        # pre-register all subtask results as pending
        for st in script.subtasks:
            result.subtask_results[st.id] = SubtaskResult(
                id=st.id, tool_name=st.tool_name, status="pending"
            )

        if not script.subtasks:
            result.status = "completed"
            result.completed_at = _now_iso()
            return result

        sem = asyncio.Semaphore(min(script.max_concurrent, self.max_concurrent))
        ctx = context or ToolContext(
            session_id=script.id, workspace=".", config=None
        )

        tasks = [
            asyncio.create_task(self._run_subtask(st, result, sem, ctx))
            for st in script.subtasks
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            result.status = "cancelled"
            # cancel any still-running subtasks
            for t in tasks:
                if not t.done():
                    t.cancel()
            raise
        finally:
            if result.status == "running":
                # all subtasks settled → mark completed (even if some failed)
                result.status = "completed"
                result.completed_at = _now_iso()
        return result

    async def _run_subtask(
        self,
        subtask: Subtask,
        result: WorkflowResult,
        sem: asyncio.Semaphore,
        ctx: ToolContext,
    ) -> None:
        """Run one subtask under the semaphore, update result in place."""
        sr = result.subtask_results[subtask.id]
        async with sem:
            sr.status = "running"
            sr.started_at = _now_iso()
            tool = self._registry.get(subtask.tool_name)
            if tool is None:
                sr.status = "failed"
                sr.error = f"tool '{subtask.tool_name}' not registered"
                sr.completed_at = _now_iso()
                return
            try:
                tool_result = await tool.call(subtask.args, ctx)
                sr.output = tool_result
                sr.status = "completed"
            except Exception as exc:
                sr.status = "failed"
                sr.error = f"{type(exc).__name__}: {exc}"
            finally:
                sr.completed_at = _now_iso()


# ── shared registry (for the tool / HTTP layer) ──────────────────────────────


class WorkflowRegistry:
    """Tracks running workflows by id, for the workflow_tool interface.

    The orchestrator's run() is async and blocking; the tool layer wants to
    submit a script, get an id back immediately, then poll status / collect.
    This registry bridges the two: submit() kicks off a background task IF
    a running event loop exists (engine / chat path); collect() either awaits
    that task or runs the workflow synchronously if no task was started.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._scripts: dict[str, WorkflowScript] = {}
        self._results: dict[str, WorkflowResult] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._contexts: dict[str, ToolContext | None] = {}
        self._orchestrator = WorkflowOrchestrator()

    def submit(
        self,
        script: WorkflowScript,
        context: ToolContext | None = None,
    ) -> WorkflowResult:
        """Store script + create pending result. Start background execution
        if a running event loop exists. Returns the (pending) result immediately."""
        result = WorkflowResult(
            id=script.id,
            objective=script.objective,
            status="pending",
        )
        for st in script.subtasks:
            result.subtask_results[st.id] = SubtaskResult(
                id=st.id, tool_name=st.tool_name, status="pending"
            )
        with self._lock:
            self._scripts[script.id] = script
            self._results[script.id] = result
            self._contexts[script.id] = context

        # 在运行中的事件循环里开后台任务 (engine / chat path). 没有运行中
        # 的循环时 (同步测试代码) 不开, collect() 会同步跑.
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(self._run_and_update(script.id))
            with self._lock:
                self._tasks[script.id] = task
        except RuntimeError:
            pass
        return result

    async def _run_and_update(self, workflow_id: str) -> WorkflowResult:
        """Run the workflow via orchestrator, copy fields into shared result."""
        with self._lock:
            script = self._scripts.get(workflow_id)
            context = self._contexts.get(workflow_id)
            result = self._results.get(workflow_id)
        if script is None or result is None:
            return WorkflowResult(id=workflow_id, objective="", status="failed")
        orch_result = await self._orchestrator.run(script, context)
        result.status = orch_result.status
        result.subtask_results = orch_result.subtask_results
        result.started_at = orch_result.started_at
        result.completed_at = orch_result.completed_at
        return result

    async def collect(
        self, workflow_id: str, timeout: float | None = None
    ) -> WorkflowResult | None:
        """Wait for (or start) the workflow, return its result.

        If submit() started a background task, awaits it. If no task was
        started (no loop was available at submit time), runs the workflow
        synchronously now. Returns None if workflow_id not found.
        """
        with self._lock:
            result = self._results.get(workflow_id)
            task = self._tasks.get(workflow_id)
        if result is None:
            return None
        if result.status in ("completed", "failed", "cancelled"):
            return result
        if task is not None and not task.done():
            # 后台任务还在跑 — 等它
            try:
                await asyncio.wait_for(task, timeout=timeout)
            except asyncio.TimeoutError:
                pass
        else:
            # task 是 None (submit 时没 loop) 或者 task 已死 (loop 关了, 任务被
            # 取消但 result 还停在 pending) — 同步跑一遍补上结果
            await self._run_and_update(workflow_id)
        return self._results.get(workflow_id)

    def get(self, workflow_id: str) -> WorkflowResult | None:
        with self._lock:
            return self._results.get(workflow_id)

    def get_script(self, workflow_id: str) -> WorkflowScript | None:
        with self._lock:
            return self._scripts.get(workflow_id)

    def list_active(self) -> list[str]:
        with self._lock:
            return [
                wid for wid, result in self._results.items()
                if result.status not in ("completed", "failed", "cancelled")
            ]

    def cancel(self, workflow_id: str) -> bool:
        with self._lock:
            result = self._results.get(workflow_id)
            task = self._tasks.get(workflow_id)
        if result is None:
            return False
        if result.status in ("completed", "failed", "cancelled"):
            return False
        if task is not None and not task.done():
            task.cancel()
        result.status = "cancelled"
        return True

    def clear(self) -> None:
        """Remove completed/failed workflows. For test cleanup."""
        with self._lock:
            to_remove = [
                wid for wid, result in self._results.items()
                if result.status in ("completed", "failed", "cancelled")
            ]
            for wid in to_remove:
                self._scripts.pop(wid, None)
                self._results.pop(wid, None)
                self._tasks.pop(wid, None)
                self._contexts.pop(wid, None)


# ── shared singleton ─────────────────────────────────────────────────────────

_shared: WorkflowRegistry | None = None
_shared_lock = threading.Lock()


def get_shared_workflow_registry() -> WorkflowRegistry:
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = WorkflowRegistry()
        return _shared


def set_shared_workflow_registry(registry: WorkflowRegistry | None) -> None:
    """Inject a fresh registry (tests) or None to reset."""
    global _shared
    with _shared_lock:
        _shared = registry


__all__ = [
    "Subtask",
    "WorkflowScript",
    "SubtaskResult",
    "WorkflowResult",
    "WorkflowOrchestrator",
    "WorkflowRegistry",
    "get_shared_workflow_registry",
    "set_shared_workflow_registry",
]
