"""Execution orchestrator, exploration, and diagnosis endpoints."""

from __future__ import annotations

import traceback
from typing import Any

from fastapi import APIRouter

from huginn.server_core import get_agent_factory, get_context, get_memory_manager
from huginn.tools.registry import ToolRegistry
from huginn.types import ToolContext

router = APIRouter(tags=["execution"])


@router.post("/execute")
async def execute_stages(params: dict[str, Any]) -> dict[str, Any]:
    """Execute a list of workflow stages via the execution orchestrator."""
    from huginn.execution.orchestrator import ExecutionOrchestrator

    stages = params.get("stages", [])
    working_dir = params.get("working_dir", ".")
    name = params.get("name", "execute")

    def _wrap_tool(tool):
        async def _run(action: str = "", **tool_params):
            if tool.input_schema:
                input_data = tool.input_schema(action=action, **tool_params)
            else:
                input_data = {"action": action, **tool_params}
            context = ToolContext(
                session_id="http",
                workspace=working_dir,
                memory_manager=get_memory_manager(),
                agent_factory=get_agent_factory(),
                audit_logger=get_context().audit_logger,
            )
            result = await tool.call(input_data, context)
            if not result.success:
                raise RuntimeError(result.error or f"{tool.name} failed")
            return result.data

        return _run

    orch = ExecutionOrchestrator(working_dir=working_dir)
    for tool_name in ToolRegistry.list_tools():
        tool = ToolRegistry.get(tool_name)
        if tool:
            orch.register_tool(tool_name, _wrap_tool(tool))

    try:
        record = await orch.run(stages, workflow_name=name)
        return {
            "success": record.overall_success,
            "stages": [r.to_dict() for r in record.stage_results],
        }
    except Exception as e:
        traceback.print_exc()
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
        traceback.print_exc()
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
        traceback.print_exc()
        return {"success": False, "error": str(e)}
