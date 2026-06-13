"""Execution Orchestrator — turns workflow stages into real actions.

Replaces the "describe what to do" pattern with "actually do it".

Key capabilities:
  1. Dependency resolution: stages run in topological order
  2. Async execution: I/O-bound stages run in parallel where possible
  3. Progress tracking: each stage reports status, stdout, stderr
  4. Failure isolation: one stage failure doesn't cascade unless specified
  5. Checkpointing: save/resume long-running workflows
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass
class StageResult:
    """Result of executing a single workflow stage."""
    stage_id: str
    stage_name: str
    tool_name: str
    success: bool
    output_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    stdout: str = ""
    stderr: str = ""
    walltime_seconds: float = 0.0
    started_at: str = ""
    finished_at: str = ""
    retry_count: int = 0
    auto_fixed: bool = False
    fix_applied: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WorkflowExecutionRecord:
    """Complete record of a workflow execution."""
    workflow_name: str
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    finished_at: Optional[str] = None
    stage_results: List[StageResult] = field(default_factory=list)
    overall_success: bool = False
    working_directory: str = ""


class ExecutionOrchestrator:
    """Orchestrates the execution of multi-stage computational workflows.

    Usage:
        orch = ExecutionOrchestrator(working_dir="./my_calc")
        stages = [
            {"id": "relax", "tool": "vasp_tool", "action": "relax", "params": {...}},
            {"id": "band", "tool": "vasp_tool", "action": "band", "params": {...},
             "depends_on": ["relax"]},
        ]
        record = await orch.run(stages)
        print(record.overall_success)
    """

    def __init__(
        self,
        working_dir: str = "",
        tool_registry: Optional[Dict[str, Callable]] = None,
        enable_autofix: bool = True,
        max_retries: int = 2,
    ):
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.tool_registry = tool_registry or {}
        self.enable_autofix = enable_autofix
        self.max_retries = max_retries
        self._execution_history: List[WorkflowExecutionRecord] = []

    def register_tool(self, name: str, fn: Callable) -> None:
        """Register a tool function for execution."""
        self.tool_registry[name] = fn

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    async def run(
        self,
        stages: List[Dict[str, Any]],
        workflow_name: str = "unnamed_workflow",
    ) -> WorkflowExecutionRecord:
        """Execute a workflow defined as a list of stages.

        Args:
            stages: List of stage dicts with keys:
                id, tool, action, params, depends_on (optional)
            workflow_name: Identifier for this workflow run
        """
        record = WorkflowExecutionRecord(
            workflow_name=workflow_name,
            working_directory=str(self.working_dir),
        )

        # Build dependency graph
        graph = self._build_dependency_graph(stages)
        completed: set[str] = set()
        results_by_id: dict[str, StageResult] = {}

        # Execute stages in waves (topological order via BFS)
        pending = set(s["id"] for s in stages)
        while pending:
            # Find stages whose dependencies are all satisfied
            ready = [
                sid for sid in pending
                if all(d in completed for d in graph.get(sid, []))
            ]
            if not ready:
                # Cycle detected or missing dependencies
                unresolved = [sid for sid in pending if sid not in ready]
                for sid in unresolved:
                    stage = self._find_stage(stages, sid)
                    result = StageResult(
                        stage_id=sid,
                        stage_name=stage.get("name", sid),
                        tool_name=stage.get("tool", "unknown"),
                        success=False,
                        error_message="Dependencies unresolved (possible cycle or missing stage)",
                    )
                    record.stage_results.append(result)
                    completed.add(sid)
                break

            # Execute ready stages concurrently
            stage_objs = [self._find_stage(stages, sid) for sid in ready]
            coros = [self._execute_stage(s, results_by_id) for s in stage_objs]
            stage_results = await asyncio.gather(*coros, return_exceptions=True)

            for sid, res in zip(ready, stage_results):
                pending.remove(sid)
                completed.add(sid)

                if isinstance(res, Exception):
                    result = StageResult(
                        stage_id=sid,
                        stage_name=self._find_stage(stages, sid).get("name", sid),
                        tool_name=self._find_stage(stages, sid).get("tool", "unknown"),
                        success=False,
                        error_message=str(res),
                    )
                else:
                    result = res

                results_by_id[sid] = result
                record.stage_results.append(result)

                # If autofix enabled and stage failed, try to fix and retry
                if self.enable_autofix and not result.success and result.retry_count < self.max_retries:
                    fixed = await self._attempt_autofix(result, self._find_stage(stages, sid))
                    if fixed:
                        # Re-execute with fix
                        retry_result = await self._execute_stage(
                            self._find_stage(stages, sid),
                            results_by_id,
                            retry_count=result.retry_count + 1,
                        )
                        retry_result.retry_count = result.retry_count + 1
                        retry_result.auto_fixed = True
                        retry_result.fix_applied = result.fix_applied
                        results_by_id[sid] = retry_result
                        # Replace in record
                        record.stage_results[-1] = retry_result

        record.finished_at = datetime.now().isoformat()
        record.overall_success = all(r.success for r in record.stage_results)
        self._execution_history.append(record)
        self._save_checkpoint(record)
        return record

    async def _execute_stage(
        self,
        stage: Dict[str, Any],
        previous_results: Dict[str, StageResult],
        retry_count: int = 0,
    ) -> StageResult:
        """Execute a single stage."""
        stage_id = stage["id"]
        stage_name = stage.get("name", stage_id)
        tool_name = stage.get("tool", "unknown")
        action = stage.get("action", "")
        params = dict(stage.get("params", {}))

        # Substitute dependency outputs into params
        params = self._resolve_param_refs(params, previous_results)

        started = datetime.now().isoformat()
        t0 = time.time()

        # Find and call the tool
        tool_fn = self.tool_registry.get(tool_name)
        if tool_fn is None:
            return StageResult(
                stage_id=stage_id,
                stage_name=stage_name,
                tool_name=tool_name,
                success=False,
                error_message=f"Tool '{tool_name}' not found in registry",
                started_at=started,
                finished_at=datetime.now().isoformat(),
                walltime_seconds=time.time() - t0,
            )

        try:
            # Call tool (sync or async)
            if asyncio.iscoroutinefunction(tool_fn):
                output = await tool_fn(action=action, **params)
            else:
                output = tool_fn(action=action, **params)

            walltime = time.time() - t0
            return StageResult(
                stage_id=stage_id,
                stage_name=stage_name,
                tool_name=tool_name,
                success=True,
                output_data=self._serialize_output(output),
                started_at=started,
                finished_at=datetime.now().isoformat(),
                walltime_seconds=walltime,
                retry_count=retry_count,
            )
        except Exception as e:
            walltime = time.time() - t0
            return StageResult(
                stage_id=stage_id,
                stage_name=stage_name,
                tool_name=tool_name,
                success=False,
                error_message=str(e),
                started_at=started,
                finished_at=datetime.now().isoformat(),
                walltime_seconds=walltime,
                retry_count=retry_count,
            )

    # ------------------------------------------------------------------
    # Auto-fix integration
    # ------------------------------------------------------------------

    async def _attempt_autofix(
        self,
        failed_result: StageResult,
        stage: Dict[str, Any],
    ) -> bool:
        """Attempt to automatically fix a failed stage."""
        # Import autofix logic
        try:
            from matsci_agent.execution.autofix import AutoFixLoop
            fixer = AutoFixLoop()
            fixed_params = fixer.apply_fix(
                tool_name=failed_result.tool_name,
                error=failed_result.error_message or "",
                current_params=stage.get("params", {}),
            )
            if fixed_params:
                stage["params"] = fixed_params
                failed_result.fix_applied = str(fixed_params)
                return True
        except Exception:
            pass
        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_dependency_graph(self, stages: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        graph = {}
        for s in stages:
            deps = s.get("depends_on", [])
            graph[s["id"]] = deps if isinstance(deps, list) else [deps] if deps else []
        return graph

    def _find_stage(self, stages: List[Dict[str, Any]], stage_id: str) -> Dict[str, Any]:
        for s in stages:
            if s.get("id") == stage_id:
                return s
        return {"id": stage_id, "tool": "unknown", "params": {}}

    def _resolve_param_refs(
        self,
        params: Dict[str, Any],
        previous_results: Dict[str, StageResult],
    ) -> Dict[str, Any]:
        """Replace ${stage_id.output_key} references with actual values."""
        resolved = {}
        for key, val in params.items():
            if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
                # Format: ${stage_id.output_key}
                inner = val[2:-1]
                if "." in inner:
                    sid, out_key = inner.split(".", 1)
                    if sid in previous_results:
                        resolved[key] = previous_results[sid].output_data.get(out_key, val)
                    else:
                        resolved[key] = val
                else:
                    resolved[key] = previous_results.get(inner, StageResult(inner, "", "", False)).output_data
            else:
                resolved[key] = val
        return resolved

    def _serialize_output(self, output: Any) -> Dict[str, Any]:
        """Convert tool output to a serializable dict."""
        if isinstance(output, dict):
            return output
        if hasattr(output, "model_dump"):
            return output.model_dump()
        if hasattr(output, "__dict__"):
            return output.__dict__
        return {"raw": str(output)}

    def _save_checkpoint(self, record: WorkflowExecutionRecord) -> None:
        """Save execution record to disk for resumability."""
        checkpoint_dir = self.working_dir / ".checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = checkpoint_dir / f"{record.workflow_name}_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(record), f, ensure_ascii=False, indent=2)

    def list_checkpoints(self) -> List[Path]:
        """List available checkpoint files."""
        checkpoint_dir = self.working_dir / ".checkpoints"
        if not checkpoint_dir.exists():
            return []
        return sorted(checkpoint_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    def load_checkpoint(self, path: Path) -> Optional[WorkflowExecutionRecord]:
        """Load a workflow execution from checkpoint."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return WorkflowExecutionRecord(**data)
