"""ImageAnalysisTool 主体 — 材料科学图像分析.

8 个 action (sem/tem/eds/particles/defect/phase_field/plot_extract/deplot_chart)
在 scenes_*.py 里实现, call() 按 action lazy import 对应模块, 结果统一
经过 _maybe_save 写文件.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult, ValidationResult

logger = logging.getLogger(__name__)


class ImageAnalysisInput(BaseModel):
    image_path: str = Field(..., description="图片路径, 支持 PNG/JPG/TIFF/BMP")
    action: Literal[
        "sem_analysis",
        "tem_lattice",
        "eds_mapping",
        "particle_stats",
        "defect_detect",
        "phase_field",
        "plot_extract",
        "deplot_chart",
    ] = Field(..., description="图像分析动作")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Action 专用参数. sem_analysis: pixel_size_nm/contrast_threshold; "
            "tem_lattice: pixel_size_nm/fft_threshold; "
            "eds_mapping: element_colors/overlap_threshold/color_tolerance; "
            "particle_stats: min_area_px/max_area_px/binning/invert/pixel_size_nm; "
            "defect_detect: defect_type/sensitivity/min_defect_area_px; "
            "phase_field: n_phases/interface_width_px/pixel_size_nm; "
            "plot_extract: x_min/x_max/y_min/y_max(可选, 缺失时自动OCR检测)/"
            "x_axis_type/y_axis_type/curve_color(可填auto做多曲线)/"
            "axis_box/color_tolerance; "
            "deplot_chart: max_new_tokens(可选, 默认512)"
        ),
    )
    output_path: str | None = Field(
        default=None,
        description="可选, 把结果以 JSON 形式保存到该路径",
    )


class ImageAnalysisTool(HuginnTool):
    """材料科学图像分析: SEM/TEM/EDS/粒度/缺陷/相场/图表提取."""

    name = "image_analysis_tool"
    category = "cv"
    description = (
        "Analyze materials science microscopy images: SEM morphology, TEM "
        "lattice fringes (FFT + d-spacing), EDS element mapping, particle "
        "size statistics (D10/D50/D90 + lognormal fit), defect detection "
        "(crack/pore/inclusion), phase-field post-processing, curve data "
        "extraction from published plots (single/multi-curve + auto axis "
        "detection), and chart-to-table conversion via Google DePlot. "
        "cv2/skimage are optional; numpy+PIL+scipy fallbacks are always "
        "available."
    )
    input_schema = ImageAnalysisInput
    read_only = True

    def is_read_only(self, args: ImageAnalysisInput) -> bool:
        return True

    async def validate_input(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ValidationResult:
        # routes/tools.py 传进来的是 model_dump() 后的 dict, 这里转回 model
        input_data = ImageAnalysisInput(**args)
        if not Path(input_data.image_path).exists():
            return ValidationResult(
                result=False, message=f"图片不存在: {input_data.image_path}"
            )
        return ValidationResult(result=True)

    async def call(
        self, args: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        input_data = ImageAnalysisInput(**args)
        try:
            if input_data.action == "sem_analysis":
                from huginn.tools.image_analysis.scenes_sem import sem_analysis
                result = sem_analysis(input_data)
            elif input_data.action == "tem_lattice":
                from huginn.tools.image_analysis.scenes_tem import tem_lattice
                result = tem_lattice(input_data)
            elif input_data.action == "eds_mapping":
                from huginn.tools.image_analysis.scenes_eds import eds_mapping
                result = eds_mapping(input_data)
            elif input_data.action == "particle_stats":
                from huginn.tools.image_analysis.scenes_particles import particle_stats
                result = particle_stats(input_data)
            elif input_data.action == "defect_detect":
                from huginn.tools.image_analysis.scenes_defect import defect_detect
                result = defect_detect(input_data)
            elif input_data.action == "phase_field":
                from huginn.tools.image_analysis.scenes_phase_field import phase_field
                result = phase_field(input_data)
            elif input_data.action == "plot_extract":
                from huginn.tools.image_analysis.scenes_plot_extract import plot_extract
                result = plot_extract(input_data)
            elif input_data.action == "deplot_chart":
                from huginn.tools.image_analysis.scenes_deplot import deplot_chart
                result = deplot_chart(input_data)
            else:
                return ToolResult(
                    data=None, success=False, error=f"未知 action: {input_data.action}"
                )

            # 统一写文件: action 返回 success=True 且有 data 时才写
            if result.success and result.data:
                self._maybe_save(input_data.output_path, result.data)
            return result
        except Exception as exc:
            logger.warning("image_analysis_tool %s 失败: %s", input_data.action, exc, exc_info=True)
            return ToolResult(data=None, success=False, error=str(exc))

    def _maybe_save(self, output_path: str | None, data: dict[str, Any]) -> None:
        if output_path:
            Path(output_path).write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
