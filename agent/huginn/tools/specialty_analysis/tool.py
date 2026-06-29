"""结构力学专门分析工具 — 屈曲/模态/疲劳/断裂.

独立接口 (不依赖 abaqus_tool), 纯 numpy/scipy 后端.
LLM 拿到 K/M 矩阵后手动拼接调用; 疲劳/断裂走解析公式.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from huginn.tools.base import HuginnTool, ToolProfile
from huginn.types import ToolContext, ToolResult


class SpecialtyAnalysisInput(BaseModel):
    action: Literal[
        "eigenvalue_buckling",
        "modal_lanczos",
        "fatigue_sn",
        "fatigue_crack_growth",
        "fracture_lefm",
    ] = Field(...)

    # eigenvalue_buckling / modal_lanczos: K/M 矩阵
    stiffness_matrix: list[list[float]] | None = Field(
        default=None, description="系统刚度矩阵 K (n x n)"
    )
    geometric_stiffness: list[list[float]] | None = Field(
        default=None, description="几何刚度矩阵 K_G (n x n), 用于屈曲"
    )
    mass_matrix: list[list[float]] | None = Field(
        default=None, description="质量矩阵 M (n x n), 用于模态"
    )
    num_modes: int = Field(default=5, ge=1, le=200)
    shift: float | None = Field(
        default=None, description="Lanczos shift-invert 平移点 (rad²/s²)"
    )

    # fatigue_sn
    sn_params: dict[str, float] = Field(
        default_factory=dict,
        description="Basquin: {sigma_f_prime, b}",
    )
    stress_amplitude: float = Field(default=0.0, gt=0, description="应力幅 σ_a, Pa")
    mean_stress: float = Field(default=0.0, description="平均应力 σ_m, Pa")
    mean_stress_theory: Literal[
        "goodman", "soderberg", "gerber", "morrow"
    ] = "goodman"
    material_uts: float | None = Field(default=None, gt=0, description="极限抗拉强度, Pa")
    material_yield: float | None = Field(default=None, gt=0, description="屈服强度, Pa")
    cycles_limit: int = Field(default=10**6, ge=1)

    # fatigue_crack_growth
    paris_params: dict[str, float] = Field(
        default_factory=dict,
        description="Paris: {C, m}",
    )
    dk_range: float = Field(default=0.0, gt=0, description="ΔK 应力强度因子幅, Pa·√m")
    a_init: float = Field(default=0.0, gt=0, description="初始裂纹长度, m")
    a_final: float = Field(default=0.0, gt=0, description="最终裂纹长度, m")

    # fracture_lefm
    crack_type: Literal[
        "edge", "interior", "surface", "three_point_bend", "compact_tension"
    ] = "edge"
    crack_length: float = Field(default=0.0, gt=0, description="裂纹长度 a, m")
    applied_stress: float = Field(default=0.0, description="施加应力 σ, Pa")
    geometry_factor: float | None = Field(
        default=1.12, description="几何因子 Y (None 用 crack_type 查表)"
    )
    k_ic: float | None = Field(default=None, gt=0, description="断裂韧性 K_IC, Pa·√m")
    youngs_modulus: float | None = Field(default=None, gt=0, description="E, Pa")
    poissons_ratio: float = Field(default=0.3, ge=-1, lt=0.5)

    @model_validator(mode="after")
    def _check_action_fields(self) -> "SpecialtyAnalysisInput":
        if self.action == "eigenvalue_buckling":
            if self.stiffness_matrix is None or self.geometric_stiffness is None:
                raise ValueError(
                    "eigenvalue_buckling requires 'stiffness_matrix' and 'geometric_stiffness'"
                )
        elif self.action == "modal_lanczos":
            if self.stiffness_matrix is None or self.mass_matrix is None:
                raise ValueError(
                    "modal_lanczos requires 'stiffness_matrix' and 'mass_matrix'"
                )
        elif self.action == "fatigue_sn":
            if not self.sn_params:
                raise ValueError("fatigue_sn requires 'sn_params' {sigma_f_prime, b}")
            if self.stress_amplitude <= 0:
                raise ValueError("fatigue_sn requires 'stress_amplitude' > 0")
        elif self.action == "fatigue_crack_growth":
            if not self.paris_params:
                raise ValueError("fatigue_crack_growth requires 'paris_params' {C, m}")
            if self.dk_range <= 0:
                raise ValueError("fatigue_crack_growth requires 'dk_range' > 0")
            if self.a_final <= self.a_init:
                raise ValueError("fatigue_crack_growth requires a_final > a_init")
        elif self.action == "fracture_lefm":
            if self.crack_length <= 0:
                raise ValueError("fracture_lefm requires 'crack_length' > 0")
            if self.applied_stress <= 0:
                raise ValueError("fracture_lefm requires 'applied_stress' > 0")
        return self


class SpecialtyAnalysisTool(HuginnTool):
    """屈曲 (eigenvalue)/模态 (Lanczos)/疲劳 (Basquin+Paris)/断裂 (LEFM) 专门分析."""

    name = "specialty_analysis_tool"
    category = "sim"
    profile = ToolProfile(constraint_scope="fea")
    description = (
        "Specialty structural analyses: eigenvalue buckling (K_G φ = λ K φ), "
        "modal analysis (Lanczos shift-invert), fatigue (Basquin S-N + Paris "
        "crack growth), and LEFM fracture (K_I/J/G vs K_IC). Pure numpy/scipy."
    )
    read_only = True
    destructive = False
    input_schema = SpecialtyAnalysisInput

    def estimate_cost(self, args: SpecialtyAnalysisInput) -> dict[str, float] | None:
        return {"cpu_hours": 0.0, "walltime_hours": 0.05}

    async def call(
        self, args: SpecialtyAnalysisInput, context: ToolContext
    ) -> ToolResult:
        try:
            if args.action == "eigenvalue_buckling":
                from .buckling import eigenvalue_buckling
                return eigenvalue_buckling(args)
            if args.action == "modal_lanczos":
                from .modal import modal_lanczos
                return modal_lanczos(args)
            if args.action == "fatigue_sn":
                from .fatigue import fatigue_sn
                return fatigue_sn(args)
            if args.action == "fatigue_crack_growth":
                from .fatigue import fatigue_crack_growth
                return fatigue_crack_growth(args)
            if args.action == "fracture_lefm":
                from .fracture import fracture_lefm
                return fracture_lefm(args)
            return ToolResult(
                data=None, success=False,
                error=f"Unknown action: {args.action}",
            )
        except Exception as exc:
            return ToolResult(
                data=None, success=False,
                error=f"Specialty analysis failed: {exc}",
            )
