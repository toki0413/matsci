"""Execution orchestrator, exploration, and diagnosis endpoints."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from huginn.core_types import ToolContext
from huginn.security.auth import require_api_key, require_capability
from huginn.server_core import get_agent_factory, get_context, get_memory_manager

logger = logging.getLogger(__name__)

# G32: /execute 跑工作流编排 = 特权操作, 既要鉴权也要 capability 校验.
# 之前这层完全裸奔 (router 无 dependencies), 是 fail-secure 修复的最大漏洞.
router = APIRouter(
    tags=["execution"],
    dependencies=[Depends(require_api_key), Depends(require_capability("execute"))],
)


def _validate_working_dir(raw_path: str) -> str:
    """Ensure working_dir stays within the workspace boundary."""
    try:
        workspace = Path(get_context().config.workspace).resolve()
    except Exception:
        workspace = Path.cwd()

    resolved = Path(raw_path).resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError:
        raise HTTPException(
            status_code=403,
            detail="working_dir is outside the workspace boundary",
        ) from None
    return str(resolved)


@router.post("/execute")
async def execute_stages(params: dict[str, Any]) -> dict[str, Any]:
    """Execute a list of workflow stages via the execution orchestrator."""
    from huginn.execution.orchestrator import ExecutionOrchestrator

    stages = params.get("stages", [])
    working_dir = _validate_working_dir(params.get("working_dir", "."))
    name = params.get("name", "execute")

    orch = ExecutionOrchestrator(working_dir=working_dir)
    # 注: 不调用 orch.register_tool() — orchestrator 默认接全局 ToolRegistry
    # 类, register_tool 仅对 dict 模式有效, 这里逐个注册纯属无效且每个都打
    # warning。run() 内部用 ToolRegistry.get() 取工具, 由 _invoke_tool 桥接。

    try:
        record = await orch.run(stages, workflow_name=name)
        return {
            "success": record.overall_success,
            "stages": [r.to_dict() for r in record.stage_results],
        }
    except Exception as e:
        logger.error("unexpected error", exc_info=True)
        return {"success": False, "error": str(e)}


@router.post("/explore")
async def explore_http(params: dict[str, Any]) -> dict[str, Any]:
    """Run a design-space exploration via HTTP."""
    from huginn.exploration.orchestrator import ExplorationOrchestrator
    from huginn.exploration.strategies import ParetoPruningStrategy

    objective = params.get("objective", "")
    max_iterations = int(params.get("max_iterations", 20))
    max_branches = int(params.get("max_branches", 10))
    initial_branches = params.get(
        "initial_branches",
        [{"name": "baseline", "hypothesis": f"Baseline for: {objective}"}],
    )
    objectives_config = params.get("objectives_config", {"score": "maximize"})
    cfg = get_context().config

    try:
        orch = ExplorationOrchestrator(
            strategy=ParetoPruningStrategy(max_active=max_branches),
            max_parallel=min(cfg.max_parallel_branches, max_branches),
        )
        result = await orch.explore(
            objective=objective,
            initial_branches=initial_branches,
            objectives_config=objectives_config,
            max_iterations=max_iterations,
        )
        return {
            "success": True,
            "n_branches_explored": result.n_branches_explored,
            "n_branches_pruned": result.n_branches_pruned,
            "pareto_front": result.pareto_front,
            "best_branch": result.best_branch,
            "convergence_reason": result.convergence_reason,
        }
    except Exception as e:
        logger.error("unexpected error", exc_info=True)
        return {"success": False, "error": str(e)}


@router.post("/diagnose")
async def diagnose_error(params: dict[str, Any]) -> dict[str, Any]:
    """Diagnose a computational chemistry/MD error."""
    from huginn.tools.diagnose_tool import DiagnoseInput, DiagnoseTool

    try:
        tool = DiagnoseTool()
        input_data = DiagnoseInput(
            error_message=params.get("error_message", ""),
            software=params.get("software"),
            calculation_type=params.get("calculation_type"),
            context=params.get("context"),
        )
        context = ToolContext(
            session_id="http",
            workspace=".",
            memory_manager=get_memory_manager(),
            agent_factory=get_agent_factory(),
            audit_logger=get_context().audit_logger,
        )
        result = await tool.call(input_data, context)
        return {"success": result.success, "data": result.data}
    except Exception as e:
        logger.error("unexpected error", exc_info=True)
        return {"success": False, "error": str(e)}
