"""ImageAnalysisTool 主体 — 材料科学图像分析.

8 个 action (sem/tem/eds/particles/defect/phase_field/plot_extract/deplot_chart)
在 scenes_*.py 里实现, call() 按 action lazy import 对应模块, 结果统一
经过 _maybe_save 写文件.
"""
from __future__ import annotations

import ast
import json
import logging
import shutil
import sys
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
        "code_verify",
        "compare_to_target",
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
            "deplot_chart: max_new_tokens(可选, 默认512); "
            "code_verify: analysis_result(来自 sem/tem/eds 等 action 的输出 dict)/"
            "original_action(可选, 默认 sem_analysis)/timeout(可选, 默认 30s)"
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
        # adapter 可能传 ImageAnalysisInput 实例 (已解析) 或 dict (model_dump 后)
        input_data = args if isinstance(args, ImageAnalysisInput) else ImageAnalysisInput(**args)
        if not Path(input_data.image_path).exists():
            return ValidationResult(
                result=False, message=f"图片不存在: {input_data.image_path}"
            )
        return ValidationResult(result=True)

    async def call(
        self, args: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        input_data = args if isinstance(args, ImageAnalysisInput) else ImageAnalysisInput(**args)
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
            elif input_data.action == "code_verify":
                result = self._run_code_verify(input_data)
            elif input_data.action == "compare_to_target":
                result = self._run_compare_to_target(input_data)
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

    # -- code_verify: SWE-Vision 范式, 用代码验证视觉判断 --

    def _run_compare_to_target(self, input_data: ImageAnalysisInput) -> ToolResult:
        """对比 agent 生成图 vs paper 目标图, 返回 CV 四算子相似度分数.

        打通工作流断层: agent 生成图后能自评, 知道跟 paper 目标图差多远,
        决定是否重画. CV 算子跟 RCBench 评分用同一套 (cv_compare 模块).
        """
        target_path = input_data.parameters.get("target_path")
        if not target_path:
            return ToolResult(
                data=None, success=False,
                error="compare_to_target 需要 parameters.target_path",
            )
        target = Path(target_path)
        if not target.exists():
            return ToolResult(
                data=None, success=False,
                error=f"target 图不存在: {target_path}",
            )

        # candidate_paths 优先, 缺省用 image_path
        candidate_paths = input_data.parameters.get("candidate_paths")
        if candidate_paths:
            candidates = [Path(p) for p in candidate_paths]
        else:
            candidates = [Path(input_data.image_path)]

        from huginn.tools.image_analysis.cv_compare import cv_best_match
        result = cv_best_match(target, candidates)
        return ToolResult(
            data={
                "target_path": str(target),
                "best_score": result.get("score"),
                "best_path": result.get("best_path"),
                "details": result.get("details", []),
                "message": (
                    f"Best CV similarity: {result.get('score')}/100. "
                    f"算子: SSIM(结构)+HSV histogram(配色)+HOG(形状)+ORB(关键点). "
                    f"<40 建议重画, 40-70 可改进, >70 基本达标."
                ),
            },
            success=result.get("score") is not None,
        )

    def _run_code_verify(self, input_data: ImageAnalysisInput) -> ToolResult:
        """根据 analysis_result 生成验证代码, 在沙箱里跑, 返回 measured vs claimed.

        失败不阻塞, 返回 error 让 agent 决定是否重试.
        """
        analysis_result = input_data.parameters.get("analysis_result") or {}
        original_action = input_data.parameters.get("original_action", "sem_analysis")
        timeout = float(input_data.parameters.get("timeout", 30.0))

        # 目前只实现了 SEM 的验证代码生成, 其他 action 复用 SEM 路径
        # 后续扩展 tem/eds 时在这里加分支即可
        from huginn.tools.image_analysis.scenes_sem import generate_verification_code

        code = generate_verification_code(input_data.image_path, analysis_result)

        # ast.parse 验证语法, 生成代码本身不应该有语法错误
        try:
            ast.parse(code)
        except SyntaxError as exc:
            return ToolResult(
                data=None, success=False,
                error=f"Verification code syntax error: {exc}",
            )

        return self._execute_in_sandbox(code, timeout)

    def _execute_in_sandbox(self, code: str, timeout: float) -> ToolResult:
        """在 SandboxExecutor 子进程里跑验证代码, 解析 __VERIFY_RESULT__ 标记.

        不走 restricted_python (会拦 Image.open), 改用 ast.parse + 沙箱超时.
        """
        from huginn.security.sandbox import SandboxConfig, SandboxExecutor

        sandbox = SandboxExecutor(SandboxConfig(
            allowed_executables={"python", "python3"},
            default_timeout=timeout,
            max_timeout=max(timeout, 60.0),
        ))

        work_dir = Path(self._tmp_dir())
        work_dir.mkdir(parents=True, exist_ok=True)
        script = work_dir / "_verify_code.py"
        script.write_text(code, encoding="utf-8")

        # sidecar 打包后 PATH 里可能没 python, fallback 到当前解释器
        python_exe = (
            shutil.which("python")
            or shutil.which("python3")
            or sys.executable
            or "python"
        )

        sb = sandbox.run(
            [python_exe, str(script)],
            cwd=str(work_dir),
            timeout=timeout,
            capture_output=True,
            text=True,
        )

        if not sb.success:
            # 代码执行失败 (包括 AssertionError) — 不阻塞, 把 stderr 带回去
            return ToolResult(
                data={
                    "verified": False,
                    "stdout": sb.stdout,
                    "stderr": sb.stderr,
                    "returncode": sb.returncode,
                },
                success=False,
                error=f"Verification code failed: {sb.stderr[:500] if sb.stderr else sb.stdout[:500]}",
            )

        # 解析 __VERIFY_RESULT__ 标记
        parsed = None
        for line in reversed(sb.stdout.splitlines()):
            if line.startswith("__VERIFY_RESULT__:"):
                payload = line[len("__VERIFY_RESULT__:"):]
                try:
                    parsed = json.loads(payload)
                except json.JSONDecodeError:
                    parsed = {"raw": payload}
                break

        return ToolResult(
            data={
                "verified": parsed.get("all_match", False) if parsed else False,
                "checks": parsed.get("checks", []) if parsed else [],
                "measured": parsed.get("measured", {}) if parsed else {},
                "claimed": parsed.get("claimed", {}) if parsed else {},
                "stdout": sb.stdout,
            },
            success=True,
        )

    @staticmethod
    def _tmp_dir() -> str:
        """临时目录给沙箱脚本用, 跑完不清理 (OS temp 反正会定期清)."""
        import tempfile
        return tempfile.mkdtemp(prefix="huginn_verify_")
