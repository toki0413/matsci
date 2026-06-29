"""Visualization tool — generate plots from benchmark/evolution/exploration reports.

Wraps the helper functions in ``huginn.visualize`` so they can be invoked as a
registered tool inside skills and agent workflows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile
from huginn.types import ToolContext, ToolResult


class VisualizeToolInput(BaseModel):
    action: Literal["benchmark", "evolution", "exploration"] = Field(
        ..., description="Which report type to visualize"
    )
    report_path: str | None = Field(
        default=None, description="Path to a JSON report file"
    )
    report_data: dict[str, Any] | None = Field(
        default=None, description="In-memory report dictionary"
    )
    output_path: str = Field(..., description="Where to save the figure")
    plot_type: str = Field(
        default="auto",
        description="Plot subtype (action-specific; e.g. 'bar', 'summary', '2d')",
    )


class VisualizeTool(HuginnTool):
    """Generate figures from structured reports."""

    name = "visualize_tool"
    category = "cv"
    profile = ToolProfile(phases=frozenset({ResearchPhase.REPORTING}))
    description = (
        "Create matplotlib figures from benchmark, evolution, or exploration reports"
    )
    input_schema = VisualizeToolInput
    read_only = True

    def is_read_only(self, args: VisualizeToolInput) -> bool:
        return True

    async def call(self, args: VisualizeToolInput, context: ToolContext) -> ToolResult:
        try:
            report = self._load_report(args)
        except Exception as exc:
            return ToolResult(
                data=None,
                success=False,
                error=f"Failed to load report: {exc}",
            )

        from huginn import visualize

        plotters = {
            "benchmark": visualize.plot_benchmark_report,
            "evolution": visualize.plot_evolution_report,
            "exploration": visualize.plot_exploration_result,
        }
        plotter = plotters.get(args.action)
        if plotter is None:
            return ToolResult(
                data=None,
                success=False,
                error=f"Unknown visualization action: {args.action}",
            )

        try:
            output = plotter(
                report,
                output_path=args.output_path,
                plot_type=args.plot_type,
            )
        except Exception as exc:
            return ToolResult(
                data=None,
                success=False,
                error=f"Visualization failed: {exc}",
            )

        path = Path(output) if output is not None else Path(args.output_path)
        return ToolResult(
            data={"output_path": str(path), "exists": path.exists()},
            success=path.exists(),
        )

    def _load_report(self, args: VisualizeToolInput) -> dict[str, Any]:
        if args.report_data is not None:
            return args.report_data
        if args.report_path:
            path = Path(args.report_path)
            return json.loads(path.read_text(encoding="utf-8"))
        raise ValueError("Either report_path or report_data must be provided")
