"""结构力学解析求解工具 — 梁/板/壳/应力集中.

纯 numpy/scipy 实现, 不依赖外部求解器. 适合 LLM 在不想拉 Abaqus 时
快速算一个解析解 (挠度/频率/临界载荷/应力集中系数).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from huginn.tools.base import HuginnTool, ToolProfile
from huginn.types import ToolContext, ToolResult


class BeamSpec(BaseModel):
    """等截面梁参数. SI 单位 (Pa, m, kg/m^3)."""

    youngs_modulus: float = Field(..., gt=0, description="E, Pa")
    poissons_ratio: float = Field(default=0.0, ge=-1, lt=0.5)
    length: float = Field(..., gt=0, description="L, m")
    second_moment: float | None = Field(
        default=None, gt=0, description="I, m^4. 不给则按 section_dims 算"
    )
    section_type: Literal["rectangular", "circular", "general"] = "general"
    section_dims: dict[str, float] = Field(
        default_factory=dict,
        description="rectangular: {b,h}; circular: {d}; general 留空并直接给 second_moment",
    )
    density: float = Field(default=7850.0, gt=0, description="rho, kg/m^3")
    area: float | None = Field(
        default=None, gt=0, description="A, m^2. 不给则按 section_dims 算"
    )

    def resolved_I(self) -> float:
        if self.second_moment is not None:
            return self.second_moment
        if self.section_type == "rectangular":
            b = self.section_dims["b"]
            h = self.section_dims["h"]
            return b * h**3 / 12.0
        if self.section_type == "circular":
            d = self.section_dims["d"]
            return 3.141592653589793 * d**4 / 64.0
        raise ValueError("second_moment 未给且 section_type 无法推出 I")

    def resolved_A(self) -> float:
        if self.area is not None:
            return self.area
        if self.section_type == "rectangular":
            return self.section_dims["b"] * self.section_dims["h"]
        if self.section_type == "circular":
            d = self.section_dims["d"]
            return 3.141592653589793 * d**2 / 4.0
        raise ValueError("area 未给且 section_type 无法推出 A")


class PlateSpec(BaseModel):
    """等厚度板参数. SI 单位."""

    youngs_modulus: float = Field(..., gt=0)
    poissons_ratio: float = Field(default=0.3, ge=-1, lt=0.5)
    thickness: float = Field(..., gt=0, description="h, m")
    density: float = Field(default=7850.0, gt=0)

    def D(self) -> float:
        # 板弯曲刚度 D = E h^3 / (12 (1 - nu^2))
        return (
            self.youngs_modulus
            * self.thickness**3
            / (12.0 * (1.0 - self.poissons_ratio**2))
        )


class ShellSpec(BaseModel):
    """圆柱壳参数. SI 单位."""

    youngs_modulus: float = Field(..., gt=0)
    poissons_ratio: float = Field(default=0.3, ge=-1, lt=0.5)
    radius: float = Field(..., gt=0, description="R, m")
    length: float = Field(..., gt=0, description="L, m")
    thickness: float = Field(..., gt=0, description="h, m")
    density: float = Field(default=7850.0, gt=0)


class StructuralAnalyticalInput(BaseModel):
    action: Literal[
        "beam_static",
        "beam_modal",
        "beam_buckling",
        "plate_static",
        "plate_modal",
        "plate_buckling",
        "shell_buckling",
        "shell_modal",
        "stress_concentration",
    ] = Field(...)
    beam: BeamSpec | None = None
    plate: PlateSpec | None = None
    shell: ShellSpec | None = None

    theory: Literal[
        "euler_bernoulli",
        "timoshenko",
        "kirchhoff",
        "mindlin",
        "donnell",
        "flugge",
    ] = Field(default="euler_bernoulli")
    boundary: Literal[
        "simply_supported",
        "cantilever",
        "fixed_fixed",
        "fixed_pinned",
        "clamped",
        "free",
    ] = Field(default="simply_supported")
    n_modes: int = Field(default=5, ge=1, le=50)

    # beam_static
    loads: list[dict[str, Any]] = Field(
        default_factory=list,
        description="[{type: point|udl|moment, value, position}] (SI: N, N/m, N·m)",
    )
    n_points: int = Field(default=101, ge=11, le=10001)

    # plate
    plate_shape: Literal["rectangular", "circular"] = "rectangular"
    plate_dims: dict[str, float] = Field(
        default_factory=dict, description="rectangular: {a,b}; circular: {radius}"
    )
    plate_bc: Literal["simply_supported", "clamped", "free"] = "simply_supported"
    plate_load: dict[str, Any] = Field(
        default_factory=dict,
        description="{type: uniform|point, magnitude, position?}",
    )

    # shell
    shell_load_type: Literal["axial", "external_pressure", "torsion"] = "axial"

    # stress_concentration
    notch_type: Literal[
        "hole", "elliptical_hole", "fillet", "groove", "shoulder"
    ] = "hole"
    notch_geometry: dict[str, float] = Field(
        default_factory=dict,
        description="hole: {d, D}; elliptical_hole: {a, b}; fillet/groove/shoulder: {d, D, r}",
    )
    notch_load: Literal["tension", "bending", "torsion"] = "tension"
    material_uts: float | None = Field(default=None, gt=0, description="极限抗拉强度, Pa")

    unit: str = Field(default="SI")

    @model_validator(mode="after")
    def _check_action_fields(self) -> "StructuralAnalyticalInput":
        # 必填 spec
        if self.action.startswith("beam_") and self.beam is None:
            raise ValueError(f"action '{self.action}' requires 'beam' (BeamSpec)")
        if self.action.startswith("plate_") and self.plate is None:
            raise ValueError(f"action '{self.action}' requires 'plate' (PlateSpec)")
        if self.action.startswith("shell_") and self.shell is None:
            raise ValueError(f"action '{self.action}' requires 'shell' (ShellSpec)")
        if self.action == "stress_concentration" and not self.notch_geometry:
            raise ValueError("stress_concentration requires 'notch_geometry'")

        # theory 与 action 类型匹配
        if self.action.startswith("beam_") and self.theory not in (
            "euler_bernoulli",
            "timoshenko",
        ):
            raise ValueError(
                "beam_* requires theory in (euler_bernoulli, timoshenko)"
            )
        if self.action.startswith("plate_") and self.theory not in (
            "kirchhoff",
            "mindlin",
        ):
            raise ValueError("plate_* requires theory in (kirchhoff, mindlin)")
        if self.action.startswith("shell_") and self.theory not in (
            "donnell",
            "flugge",
        ):
            raise ValueError("shell_* requires theory in (donnell, flugge)")

        # 子项必填
        if self.action in ("plate_static", "plate_modal", "plate_buckling"):
            if self.plate_shape == "rectangular" and not self.plate_dims.get("a"):
                raise ValueError("rectangular plate needs plate_dims.a (and .b)")
            if self.plate_shape == "circular" and not self.plate_dims.get("radius"):
                raise ValueError("circular plate needs plate_dims.radius")
        if self.action == "beam_static" and not self.loads:
            raise ValueError("beam_static requires non-empty 'loads'")

        return self


class StructuralAnalyticalTool(HuginnTool):
    """梁/板/壳/应力集中的解析求解器, 纯 numpy/scipy."""

    name = "structural_analytical_tool"
    category = "sim"
    profile = ToolProfile(constraint_scope="fea")
    description = (
        "Analytical solvers for beam (Euler-Bernoulli/Timoshenko), plate "
        "(Kirchhoff/Mindlin), shell (Donnell/Flügge), and stress "
        "concentration factors. Pure numpy/scipy, no external solver."
    )
    read_only = True
    destructive = False
    input_schema = StructuralAnalyticalInput

    def estimate_cost(self, args: StructuralAnalyticalInput) -> dict[str, float] | None:
        return {"cpu_hours": 0.0, "walltime_hours": 0.05}

    async def call(
        self, args: StructuralAnalyticalInput, context: ToolContext
    ) -> ToolResult:
        try:
            if args.action in ("beam_static", "beam_modal", "beam_buckling"):
                from .beams import beam_buckling, beam_modal, beam_static

                if args.action == "beam_static":
                    return beam_static(args)
                if args.action == "beam_modal":
                    return beam_modal(args)
                return beam_buckling(args)

            if args.action in ("plate_static", "plate_modal", "plate_buckling"):
                from .plates import plate_buckling, plate_modal, plate_static

                if args.action == "plate_static":
                    return plate_static(args)
                if args.action == "plate_modal":
                    return plate_modal(args)
                return plate_buckling(args)

            if args.action in ("shell_buckling", "shell_modal"):
                from .shells import shell_buckling, shell_modal

                if args.action == "shell_buckling":
                    return shell_buckling(args)
                return shell_modal(args)

            if args.action == "stress_concentration":
                from .stress_concentration import stress_concentration

                return stress_concentration(args)

            return ToolResult(
                data=None, success=False,
                error=f"Unknown action: {args.action}",
            )
        except Exception as exc:
            return ToolResult(
                data=None, success=False,
                error=f"Structural analytical failed: {exc}",
            )
