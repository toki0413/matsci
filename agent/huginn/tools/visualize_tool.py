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
    action: Literal[
        "benchmark", "evolution", "exploration",
        "band_structure", "dos", "phonon", "structure_3d",
    ] = Field(..., description="Which report type to visualize")
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
    # Phase 4c materials 数据 (band_structure/dos/phonon/structure_3d 用)
    bands_data: list[dict[str, Any]] | None = Field(
        default=None,
        description="band_structure: 每个 band 含 kpoints + energies",
    )
    kpath: list[str] | None = Field(
        default=None, description="band_structure/phonon: 高对称点标签"
    )
    fermi: float = Field(default=0.0, description="band_structure/dos: 费米能级 (eV)")
    dos_data: dict[str, Any] | None = Field(
        default=None,
        description="dos: {total: [floats], orbital_s: [floats], ...}",
    )
    energy: list[float] | None = Field(
        default=None, description="dos: 能量轴 (eV)"
    )
    branches: list[dict[str, Any]] | None = Field(
        default=None,
        description="phonon: 每个 branch 含 qpoints + frequencies",
    )
    structure: dict[str, Any] | None = Field(
        default=None,
        description="structure_3d: {lattice, species, coords, bonds}",
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
        from huginn import visualize

        # materials actions: 不走 report 路径, 直接用专用字段
        materials_plotters = {
            "band_structure": lambda: visualize.plot_band_structure(
                args.bands_data or [], args.kpath or [], args.fermi, args.output_path
            ),
            "dos": lambda: visualize.plot_dos(
                args.dos_data or {}, args.energy, args.fermi, args.output_path
            ),
            "phonon": lambda: visualize.plot_phonon_dispersion(
                args.branches or [], args.kpath, args.output_path
            ),
            "structure_3d": lambda: visualize.plot_structure_3d(
                args.structure or {}, args.output_path
            ),
        }
        if args.action in materials_plotters:
            try:
                output = materials_plotters[args.action]()
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

        # report-based 路径 (benchmark/evolution/exploration)
        try:
            report = self._load_report(args)
        except Exception as exc:
            return ToolResult(
                data=None,
                success=False,
                error=f"Failed to load report: {exc}",
            )

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
