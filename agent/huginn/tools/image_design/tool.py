"""ImageDesignTool 主体: 7 个 action 分发到对应场景模块.

matplotlib 工具在 _mpl_utils, 各场景在 scenes_*. call() 按 args.action
lazy import 对应场景模块, 避免 matplotlib 在包 import 时被强制初始化.
"""
from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult, ValidationResult

logger = logging.getLogger(__name__)


class ImageDesignInput(BaseModel):
    action: Literal[
        "particle_distribution",
        "xrd_pattern",
        "stress_strain",
        "band_diagram",
        "microstructure_annotate",
        "eds_overlay",
        "tem_fft_annotate",
    ] = Field(..., description="输出设计动作")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Action 专用参数, 见各 action 文档",
    )
    output_path: str = Field(..., description="输出图片路径, 支持 PNG/SVG")
    data: dict[str, Any] | None = Field(
        default=None,
        description="可选, 直接传分析数据; 没传则从 parameters 取",
    )


class ImageDesignTool(HuginnTool):
    """材料科学输出设计: 粒度分布 / XRD / 应力-应变 / 能带 / 显微标注 / EDS / TEM FFT."""

    name = "image_design_tool"
    category = "cv"
    description = (
        "Generate publication-ready materials science figures: particle "
        "distribution, XRD patterns, stress-strain curves, band diagrams, "
        "microstructure annotation, EDS overlay, TEM FFT annotation. "
        "Outputs PNG/SVG + JSON metadata."
    )
    input_schema = ImageDesignInput
    read_only = True

    def is_read_only(self, args: ImageDesignInput) -> bool:
        return True

    async def validate_input(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ValidationResult:
        # routes/tools.py 传进来的是 model_dump() 后的 dict
        input_data = ImageDesignInput(**args)
        if not input_data.output_path:
            return ValidationResult(result=False, message="output_path 不能为空")
        return ValidationResult(result=True)

    async def call(
        self, args: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        input_data = ImageDesignInput(**args)
        try:
            if input_data.action == "particle_distribution":
                from .scenes_particles import particle_distribution

                return particle_distribution(input_data)
            if input_data.action == "xrd_pattern":
                from .scenes_xrd import xrd_pattern

                return xrd_pattern(input_data)
            if input_data.action == "stress_strain":
                from .scenes_stress import stress_strain

                return stress_strain(input_data)
            if input_data.action == "band_diagram":
                from .scenes_band import band_diagram

                return band_diagram(input_data)
            if input_data.action == "microstructure_annotate":
                from .scenes_micro import microstructure_annotate

                return microstructure_annotate(input_data)
            if input_data.action == "eds_overlay":
                from .scenes_eds import eds_overlay

                return eds_overlay(input_data)
            if input_data.action == "tem_fft_annotate":
                from .scenes_tem import tem_fft_annotate

                return tem_fft_annotate(input_data)
        except Exception as exc:
            logger.warning(
                "image_design_tool %s 失败: %s",
                input_data.action,
                exc,
                exc_info=True,
            )
            return ToolResult(data=None, success=False, error=str(exc))

        return ToolResult(
            data=None, success=False, error=f"未知 action: {input_data.action}"
        )
