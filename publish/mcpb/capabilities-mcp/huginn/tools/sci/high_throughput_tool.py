"""High-throughput screening tool — parameter sweeps and design-of-experiments.

Lets the agent run structured parameter sweeps (grid, random, Latin hypercube)
over any registered HuginnTool. Useful for convergence tests, property
screening, and structure-property mapping.

Usage:
    {
      "tool_name": "vasp_tool",
      "space_type": "grid",
      "parameter_space": {"encut": [300, 400, 500], "kpoints": ["2 2 2", "4 4 4"]},
      "base_input": {"structure": "POSCAR", "mode": "scf"},
      "max_parallel": 2
    }
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.core_types import ToolContext, ToolResult, ValidationResult
from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile
from huginn.workflows.high_throughput import (
    GridSpace,
    LatinHypercubeSpace,
    ParameterSweep,
    RandomSpace,
)


class HighThroughputToolInput(BaseModel):
    tool_name: str = Field(..., description="Name of the registered tool to sweep")
    space_type: Literal["grid", "random", "lhs"] = Field(default="grid")
    parameter_space: dict[str, list[Any]] = Field(
        default_factory=dict,
        description=(
            "For grid: {param: [values]}. For random/lhs: {param: [min, max]}."
        ),
    )
    base_input: dict[str, Any] = Field(
        default_factory=dict,
        description="Base input merged with each parameter combination.",
    )
    n_samples: int = Field(default=20, description="Number of samples for random/lhs")
    seed: int | None = Field(
        default=None, description="Random seed for reproducibility"
    )
    max_parallel: int = Field(default=1, description="Max concurrent tool calls")
    early_termination: str | None = Field(
        default=None,
        description="Optional safe_eval expression to stop early.",
    )


def _build_space(
    input_data: HighThroughputToolInput,
) -> tuple[Any, ToolResult | None]:
    """按 space_type 构造参数空间; 未知类型返回 (None, 错误)."""
    if input_data.space_type == "grid":
        return GridSpace(input_data.parameter_space), None
    if input_data.space_type == "random":
        ranges = {
            k: (float(v[0]), float(v[1])) for k, v in input_data.parameter_space.items()
        }
        return (
            RandomSpace(ranges, n_samples=input_data.n_samples, seed=input_data.seed),
            None,
        )
    if input_data.space_type == "lhs":
        ranges = {
            k: (float(v[0]), float(v[1])) for k, v in input_data.parameter_space.items()
        }
        return (
            LatinHypercubeSpace(
                ranges, n_samples=input_data.n_samples, seed=input_data.seed
            ),
            None,
        )
    return None, ToolResult(
        data=None,
        success=False,
        error=f"Unknown space_type: {input_data.space_type}",
    )


def _sweep_data(
    input_data: HighThroughputToolInput,
    sweep: Any,
    jobs: list[Any],
    aggregated: dict[str, Any],
) -> dict[str, Any]:
    return {
        "sweep_name": sweep.name,
        "tool": input_data.tool_name,
        "n_total": len(jobs),
        "n_completed": len(sweep.completed_jobs),
        "n_failed": len(sweep.failed_jobs),
        "summary": aggregated.get("summary_stats", {}),
        "best_results": aggregated.get("best_results", {}),
        "jobs": [j.to_dict() for j in jobs],
    }


async def _execute_jobs(
    tool: Any,
    sweep: Any,
    jobs: list[Any],
    context: ToolContext | None,
    max_parallel: int,
) -> None:
    """并发执行所有 job, 结果写回 sweep."""
    semaphore = asyncio.Semaphore(max(1, max_parallel))

    async def _run_job(job):
        async with semaphore:
            try:
                payload = job.full_input
                if hasattr(tool, "input_schema") and tool.input_schema is not None:
                    payload = tool.input_schema(**payload).model_dump()
                if inspect.iscoroutinefunction(tool.call):
                    result = await tool.call(payload, context)
                else:
                    result = tool.call(payload, context)
                sweep.update_job(
                    job.job_id,
                    "completed" if result.success else "failed",
                    result=result.data if result.success else None,
                    error=result.error if not result.success else None,
                )
            except Exception as exc:
                sweep.update_job(job.job_id, "failed", error=str(exc))

    await asyncio.gather(*[_run_job(job) for job in jobs])


class HighThroughputTool(HuginnTool):
    """Run parameter sweeps over any registered tool."""

    name = "high_throughput_tool"
    category = "sci"
    profile = ToolProfile(phases=frozenset({ResearchPhase.EXECUTION}))
    description = (
        "Run a parameter sweep or screening campaign over a registered tool. "
        "Supports grid, random, and Latin hypercube sampling. Returns aggregated "
        "statistics (min/max/mean) and best results per metric."
    )
    input_schema = HighThroughputToolInput

    def is_read_only(self, args: HighThroughputToolInput) -> bool:
        return False

    async def validate_input(
        self, args: HighThroughputToolInput, context: ToolContext | None = None
    ) -> ValidationResult:
        from huginn.tools.registry import ToolRegistry

        if not args.tool_name:
            return ValidationResult(result=False, message="tool_name is required")
        if args.tool_name not in ToolRegistry.list_tools():
            return ValidationResult(
                result=False,
                message=f"Tool '{args.tool_name}' not found in registry.",
            )
        if args.space_type in ("random", "lhs"):
            for _key, vals in args.parameter_space.items():
                if len(vals) != 2 or not all(isinstance(v, int | float) for v in vals):
                    return ValidationResult(
                        result=False,
                        message=f"{args.space_type} requires [min, max] numeric range for each parameter.",
                    )
        return ValidationResult(result=True)

    async def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        from huginn.tools.registry import ToolRegistry

        input_data = HighThroughputToolInput(**args)

        try:
            space, failure = _build_space(input_data)
            if failure is not None:
                return failure

            sweep = ParameterSweep(
                name=f"ht_{input_data.tool_name}",
                tool_name=input_data.tool_name,
                parameter_space=space,
                base_input=input_data.base_input,
                max_parallel=input_data.max_parallel,
                early_termination=input_data.early_termination,
            )
            jobs = sweep.generate_jobs()

            tool = ToolRegistry.get(input_data.tool_name)
            if tool is None:
                return ToolResult(
                    data=None,
                    success=False,
                    error=f"Tool '{input_data.tool_name}' not found",
                )

            # Execute jobs, respecting max_parallel
            await _execute_jobs(tool, sweep, jobs, context, input_data.max_parallel)

            aggregated = sweep.aggregate_results()
            return ToolResult(
                data=_sweep_data(input_data, sweep, jobs, aggregated),
                success=True,
            )

        except Exception as exc:
            return ToolResult(
                data=None, success=False, error=f"High-throughput sweep failed: {exc}"
            )
