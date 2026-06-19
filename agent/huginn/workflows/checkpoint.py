"""Workflow checkpointing for long-running computational pipelines.

A checkpoint captures the current state of all workflow stages and their
outputs so that execution can resume after a process crash or a long remote
job wait.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from huginn.types import ToolResult
from huginn.workflows.stages import ComputationalStage, RetryPolicy, ValidationRule


def _stage_to_dict(stage: ComputationalStage) -> dict[str, Any]:
    """Serialize a ComputationalStage to a plain dict."""
    result: dict[str, Any] | None = None
    if stage.result is not None:
        result = {
            "data": stage.result.data,
            "success": stage.result.success,
            "error": stage.result.error,
            "new_messages": stage.result.new_messages,
            "side_effects": stage.result.side_effects,
        }
    return {
        "id": stage.id,
        "name": stage.name,
        "tool": stage.tool,
        "tool_input": stage.tool_input,
        "dependencies": stage.dependencies,
        "validation": asdict(stage.validation) if stage.validation else None,
        "retry_policy": asdict(stage.retry_policy),
        "status": stage.status,
        "result": result,
        "attempts": stage.attempts,
        "started_at": stage.started_at.isoformat() if stage.started_at else None,
        "completed_at": (
            stage.completed_at.isoformat() if stage.completed_at else None
        ),
    }


def _stage_from_dict(data: dict[str, Any]) -> ComputationalStage:
    """Restore a ComputationalStage from a serialized dict."""
    validation = data.get("validation")
    retry = data.get("retry_policy") or {}
    result_data = data.get("result")
    result = None
    if result_data is not None:
        result = ToolResult(
            data=result_data.get("data"),
            success=result_data.get("success", True),
            error=result_data.get("error"),
            new_messages=result_data.get("new_messages", []),
            side_effects=result_data.get("side_effects", []),
        )
    return ComputationalStage(
        id=data["id"],
        name=data["name"],
        tool=data["tool"],
        tool_input=data.get("tool_input", {}),
        dependencies=data.get("dependencies", []),
        validation=ValidationRule(**validation) if validation else None,
        retry_policy=RetryPolicy(**retry),
        status=data.get("status", "pending"),
        result=result,
        attempts=data.get("attempts", 0),
        started_at=(
            datetime.fromisoformat(data["started_at"])
            if data.get("started_at")
            else None
        ),
        completed_at=(
            datetime.fromisoformat(data["completed_at"])
            if data.get("completed_at")
            else None
        ),
    )


class WorkflowCheckpoint:
    """Serializable snapshot of a workflow execution."""

    def __init__(
        self,
        stages: list[ComputationalStage],
        outputs: dict[str, Any],
        created_at: datetime | None = None,
    ):
        self.stages = stages
        self.outputs = outputs
        self.created_at = created_at or datetime.now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "created_at": self.created_at.isoformat(),
            "stages": [_stage_to_dict(s) for s in self.stages],
            "outputs": self.outputs,
        }

    def save(self, path: str | Path) -> None:
        """Atomically write the checkpoint to disk."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        tmp.replace(target)

    @classmethod
    def load(cls, path: str | Path) -> WorkflowCheckpoint:
        """Load a checkpoint from disk."""
        target = Path(path)
        data = json.loads(target.read_text(encoding="utf-8"))
        stages = [_stage_from_dict(s) for s in data.get("stages", [])]
        outputs = data.get("outputs", {})
        created_at = (
            datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else datetime.now()
        )
        return cls(stages=stages, outputs=outputs, created_at=created_at)

    @classmethod
    def default_path(cls, workspace: str | Path, run_id: str) -> Path:
        """Return the default checkpoint path for a run."""
        return Path(workspace) / ".huginn" / "workflows" / f"{run_id}.json"
