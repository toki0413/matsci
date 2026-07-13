"""Long-horizon task state tracker.

Externalizes agent progress to a JSON file per thread, so that:
1. The agent can resume from where it left off after context compaction
2. Users can inspect what the agent has done so far
3. Long-running research pipelines maintain state across reconnections

Inspired by OpenHands' StateTracker + external progress.md pattern.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger("huginn.task_state")


@dataclass
class TaskStep:
    """A single step in the task trajectory."""
    step_id: int
    action: str  # what the agent did (tool call, reasoning, etc.)
    tool: str = ""
    result: str = ""
    status: str = "pending"  # pending / running / done / failed
    timestamp: float = field(default_factory=time.time)
    findings: str = ""  # key insight from this step


@dataclass
class TaskState:
    """Full state of a research task."""
    thread_id: str
    goal: str = ""
    mode: str = "chat"  # chat / plan / research
    steps: list[TaskStep] = field(default_factory=list)
    key_findings: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    iteration: int = 0

    def add_step(self, action: str, tool: str = "") -> TaskStep:
        step = TaskStep(step_id=len(self.steps), action=action, tool=tool)
        self.steps.append(step)
        self.iteration += 1
        self.updated_at = time.time()
        return step

    def complete_step(self, step_id: int, result: str = "", findings: str = ""):
        for s in self.steps:
            if s.step_id == step_id:
                s.status = "done"
                s.result = result[:500]
                if findings:
                    s.findings = findings[:500]
                    self.key_findings.append(findings[:200])
                self.updated_at = time.time()
                break

    def summary(self) -> str:
        """Short summary for context injection."""
        done = sum(1 for s in self.steps if s.status == "done")
        total = len(self.steps)
        lines = [
            f"Task: {self.goal}",
            f"Mode: {self.mode} | Iteration: {self.iteration} | Steps: {done}/{total}",
        ]
        if self.key_findings:
            lines.append("Key findings:")
            for f in self.key_findings[-5:]:
                lines.append(f"  - {f}")
        if self.open_questions:
            lines.append("Open questions:")
            for q in self.open_questions[-3:]:
                lines.append(f"  - {q}")
        # recent steps so the agent knows what it already did
        if self.steps:
            lines.append("Recent steps:")
            for s in self.steps[-5:]:
                tool_info = f" [{s.tool}]" if s.tool else ""
                lines.append(f"  {s.action}{tool_info} -> {s.status}")
        return "\n".join(lines)


class TaskStateTracker:
    """Manages task state files per thread.

    Files are stored in HUGINN_CACHE_DIR/task_state/{thread_id}.json
    """

    def __init__(self, cache_dir: str | None = None):
        base = cache_dir or os.environ.get(
            "HUGINN_CACHE_DIR",
            os.path.join(os.path.expanduser("~"), ".huginn"),
        )
        self.state_dir = Path(base) / "task_state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, TaskState] = {}

    def get(self, thread_id: str) -> TaskState:
        """Load or create task state for a thread."""
        if thread_id in self._cache:
            return self._cache[thread_id]

        path = self.state_dir / f"{thread_id}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                state = TaskState(
                    thread_id=data.get("thread_id", thread_id),
                    goal=data.get("goal", ""),
                    mode=data.get("mode", "chat"),
                    steps=[TaskStep(**s) for s in data.get("steps", [])],
                    key_findings=data.get("key_findings", []),
                    open_questions=data.get("open_questions", []),
                    created_at=data.get("created_at", time.time()),
                    updated_at=data.get("updated_at", time.time()),
                    iteration=data.get("iteration", 0),
                )
            except Exception as e:
                logger.warning(f"Failed to load task state {thread_id}: {e}")
                state = TaskState(thread_id=thread_id)
        else:
            state = TaskState(thread_id=thread_id)

        self._cache[thread_id] = state
        return state

    def save(self, thread_id: str):
        """Persist task state to disk."""
        state = self._cache.get(thread_id)
        if not state:
            return
        path = self.state_dir / f"{thread_id}.json"
        try:
            data = asdict(state)
            # 原子写: task_state 在 autoloop 每步都写, 半截 JSON 会让重启后状态丢失.
            from huginn.utils.concurrency import atomic_write_text
            atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.warning(f"Failed to save task state {thread_id}: {e}")

    def record_step(
        self,
        thread_id: str,
        action: str,
        tool: str = "",
        result: str = "",
        findings: str = "",
        status: str = "done",
    ):
        """Record a step and persist."""
        state = self.get(thread_id)
        step = state.add_step(action, tool)
        if status == "done":
            state.complete_step(step.step_id, result, findings)
        else:
            step.status = status
        self.save(thread_id)

    def context_block(self, thread_id: str) -> str:
        """Return a context block for injection into the agent's prompt."""
        state = self.get(thread_id)
        if not state.steps:
            return ""
        return f"\n--- Task State ---\n{state.summary()}\n--- End Task State ---\n"


_tracker: TaskStateTracker | None = None


def get_tracker() -> TaskStateTracker:
    global _tracker
    if _tracker is None:
        _tracker = TaskStateTracker()
    return _tracker
