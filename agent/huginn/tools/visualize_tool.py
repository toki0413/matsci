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
            # figure_ir 接入: 生成结构化元数据让 agent 精确引用图表结构
            # ponytail: 只给 materials actions (数值数据明确), report-based 留 v8
            ir_meta = self._build_figure_ir(args)
            return ToolResult(
                data={"output_path": str(path), "exists": path.exists(), "figure_ir": ir_meta},
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
        # v8: figure_ir 接 report-based 路径 — 从 report dict 提取通用字段生成 IR
        # 不同 action 的 report 结构异构, 但都有 scores/metrics/timeline 类字段
        ir_meta = self._build_report_figure_ir(args.action, report)
        return ToolResult(
            data={"output_path": str(path), "exists": path.exists(), "figure_ir": ir_meta},
            success=path.exists(),
        )

    def _load_report(self, args: VisualizeToolInput) -> dict[str, Any]:
        if args.report_data is not None:
            return args.report_data
        if args.report_path:
            path = Path(args.report_path)
            return json.loads(path.read_text(encoding="utf-8"))
        raise ValueError("Either report_path or report_data must be provided")

    def _build_figure_ir(self, args: VisualizeToolInput) -> dict[str, Any]:
        """从 args 构造 figure_ir 元数据 (chart_type + axes + series 摘要).

        ponytail: 失败只 warn, 不阻塞主路径. ir_to_structured 让 agent 精确
        引用图表结构, 配合 visual_hook.enrich_with_visual 的 _visual_primitives
        形成"图 + 结构化元数据 + 坐标原语"三路视觉资产.
        """
        try:
            from huginn.vision import figure_ir
            chart_type_map = {
                "band_structure": "line", "dos": "line",
                "phonon": "line", "structure_3d": "scatter",
            }
            ct = chart_type_map.get(args.action, "line")
            x_label = {"band_structure": "k-path", "dos": "Energy (eV)",
                       "phonon": "q-path"}.get(args.action, "")
            y_label = {"band_structure": "Energy (eV)", "dos": "DOS",
                       "phonon": "Frequency (THz)"}.get(args.action, "")
            series = []
            if args.action == "band_structure" and args.bands_data:
                series = [{"label": f"band_{i}", "data": []}
                          for i in range(len(args.bands_data))]
            elif args.action == "dos" and args.dos_data:
                series = [{"label": k, "data": []}
                          for k in args.dos_data.keys()]
            elif args.action == "phonon" and args.branches:
                series = [{"label": f"branch_{i}", "data": []}
                          for i in range(len(args.branches))]
            ir = figure_ir.to_ir(
                data={}, chart_type=ct, title=args.action,
                x_label=x_label, y_label=y_label, series=series,
            )
            return figure_ir.ir_to_structured(ir)
        except Exception as exc:
            return {"error": f"figure_ir build failed: {exc}"}

    def _build_report_figure_ir(
        self, action: str, report: dict[str, Any]
    ) -> dict[str, Any]:
        """v8: 从 report dict 生成 figure_ir 元数据 (benchmark/evolution/exploration).

        B5 增强: 预定义 key 找不到时, 递归扫描 report dict 找所有数值字段.
        不漏任何数值数据, 适配异构 report 结构.

        ponytail: 失败只 warn, 不阻塞. 未知字段返回 minimal IR.
        """
        try:
            from huginn.vision import figure_ir
            chart_type = "bar"
            x_label, y_label = "", ""
            series: list[dict[str, Any]] = []
            if action == "benchmark":
                chart_type = "bar"
                x_label = "Method"
                y_label = "Score"
                scores = report.get("scores") or report.get("metrics") or {}
                if isinstance(scores, dict):
                    series = [{"label": k, "data": [float(v)]}
                              for k, v in list(scores.items())[:10]
                              if isinstance(v, (int, float))]
                elif isinstance(scores, list):
                    series = [{"label": f"item_{i}", "data": [float(v)]}
                              for i, v in enumerate(scores[:10])
                              if isinstance(v, (int, float))]
                # B5: 预定义 key 没找到 → 递归扫描
                if not series:
                    found = self._scan_numeric_fields(report, max_items=10)
                    if found:
                        series = [{"label": k, "data": [v]} for k, v in found]
            elif action == "evolution":
                chart_type = "line"
                x_label = "Generation"
                y_label = "Fitness"
                timeline = report.get("timeline") or report.get("generations") or []
                if isinstance(timeline, list):
                    fitness_vals = self._extract_timeline_values(timeline)
                    if fitness_vals:
                        series = [{"label": "best_fitness", "data": fitness_vals}]
                # B5: timeline 没找到 → 递归找 list of numbers
                if not series:
                    found_list = self._scan_numeric_list(report, max_items=50)
                    if found_list:
                        series = [{"label": "values", "data": found_list}]
            elif action == "exploration":
                chart_type = "scatter"
                x_label = "Parameter 1"
                y_label = "Objective"
                candidates = report.get("candidates") or report.get("results") or []
                if isinstance(candidates, list):
                    pts = self._extract_candidate_values(candidates)
                    if pts:
                        series = [{"label": "candidates", "data": pts}]
                # B5: candidates 没找到 → 递归找 list of numbers
                if not series:
                    found_list = self._scan_numeric_list(report, max_items=50)
                    if found_list:
                        series = [{"label": "values", "data": found_list}]
            ir = figure_ir.to_ir(
                data={}, chart_type=chart_type, title=action,
                x_label=x_label, y_label=y_label, series=series,
            )
            return figure_ir.ir_to_structured(ir)
        except Exception as exc:
            return {"error": f"report figure_ir build failed: {exc}"}

    def _scan_numeric_fields(
        self, d: dict[str, Any], max_items: int = 10, prefix: str = "",
        _depth: int = 0,
    ) -> list[tuple[str, float]]:
        """B5: 递归扫描 dict 找数值字段. 返回 [(label, value), ...].

        B7: 加 _depth 上限 8 — 防止病态嵌套 (实际 report dict < 5 层, 8 给余量).
        ponytail: 不上 visited set — Python dict 通常无环, 限深足够.
        """
        if _depth > 8:
            return []
        found: list[tuple[str, float]] = []
        for k, v in d.items():
            if len(found) >= max_items:
                return found
            label = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                found.append((label, float(v)))
            elif isinstance(v, dict):
                found.extend(self._scan_numeric_fields(
                    v, max_items - len(found), label, _depth + 1,
                ))
        return found

    def _scan_numeric_list(self, d: Any, max_items: int = 50) -> list[float]:
        """B5: 递归找最长的 list of numbers.

        B7: 限深 5 — list of list of list 极罕见, 深度 >5 大概率是病态结构.
        """
        best: list[float] = []

        def _scan(d: Any, depth: int) -> None:
            nonlocal best
            if depth > 5 or len(best) >= max_items:
                return
            if isinstance(d, list):
                nums = [float(x) for x in d
                        if isinstance(x, (int, float)) and not isinstance(x, bool)]
                if len(nums) > len(best):
                    best = nums[:max_items]
                for item in d:
                    if len(best) >= max_items:
                        return
                    _scan(item, depth + 1)
            elif isinstance(d, dict):
                for v in d.values():
                    if len(best) >= max_items:
                        return
                    _scan(v, depth + 1)

        _scan(d, 0)
        return best

    def _extract_timeline_values(self, timeline: list) -> list[float]:
        """从 timeline list 提取 fitness 值."""
        vals = []
        for gen in timeline[:50]:
            if isinstance(gen, dict):
                v = gen.get("best_fitness") or gen.get("fitness") or gen.get("score")
                if isinstance(v, (int, float)):
                    vals.append(float(v))
            elif isinstance(gen, (int, float)):
                vals.append(float(gen))
        return vals

    def _extract_candidate_values(self, candidates: list) -> list[float]:
        """从 candidates list 提取 objective 值."""
        pts = []
        for c in candidates[:50]:
            if isinstance(c, dict):
                v = c.get("objective") or c.get("score") or c.get("value")
                if isinstance(v, (int, float)):
                    pts.append(float(v))
            elif isinstance(c, (int, float)):
                pts.append(float(c))
        return pts
