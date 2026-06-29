"""轻量 FEM 工具 — scikit-fem 封装.

2D 线性平面应力/应变: 静力/模态/屈曲. scikit-fem 缺失时 optional_modules 自动跳过.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult

# scikit-fem 在 tool.py 顶部导入 — 缺失时 ImportError 让 optional_modules 跳过注册
try:
    import skfem  # noqa: F401
    _SKFEM_AVAILABLE = True
except ImportError:
    _SKFEM_AVAILABLE = False


class FEMInput(BaseModel):
    action: Literal[
        "mesh_from_geometry",
        "static_linear",
        "modal",
        "buckling",
    ] = Field(...)

    # mesh_from_geometry
    shape: Literal["rectangle", "circle"] = "rectangle"
    dims: dict[str, float] = Field(
        default_factory=dict,
        description="rectangle: {L, H}; circle: {R}",
    )
    n_div: int = Field(default=10, ge=2, le=100)

    # static_linear / modal / buckling 共用
    material: dict[str, float] = Field(
        default_factory=dict,
        description="{E, nu, rho?, thickness?}",
    )
    loads: list[dict[str, Any]] = Field(
        default_factory=list,
        description="[{type: pressure|point, value, region: left|right|bottom|top}]",
    )
    boundary_conditions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="[{region, dofs: [0,1], value: 0.0}]",
    )
    num_modes: int = Field(default=5, ge=1, le=50)

    @model_validator(mode="after")
    def _check_action_fields(self) -> "FEMInput":
        if self.action == "mesh_from_geometry" and not self.dims:
            raise ValueError("mesh_from_geometry requires 'dims'")
        if self.action in ("static_linear", "modal", "buckling"):
            if not self.material:
                raise ValueError(f"{self.action} requires 'material' {{E, nu}}")
            if not self.boundary_conditions:
                raise ValueError(f"{self.action} requires at least one boundary_condition")
        if self.action == "static_linear" and not self.loads:
            raise ValueError("static_linear requires at least one load")
        return self


class FEMTool(HuginnTool):
    """轻量 FEM (scikit-fem) 封装: 2D 线性静力/模态/屈曲."""

    name = "fem_tool"
    category = "sim"
    description = (
        "Lightweight FEM via scikit-fem: 2D linear static/modal/buckling "
        "for plane stress/strain. Generates mesh from geometry, assembles "
        "K/M matrices, solves. Falls back gracefully if scikit-fem missing."
    )
    read_only = True
    destructive = False
    input_schema = FEMInput

    def estimate_cost(self, args: FEMInput) -> dict[str, float] | None:
        return {"cpu_hours": 0.0, "walltime_hours": 0.1}

    async def call(
        self, args: FEMInput, context: ToolContext
    ) -> ToolResult:
        if not _SKFEM_AVAILABLE:
            return ToolResult(
                data=None,
                success=False,
                error=(
                    "scikit-fem not installed. Install with: pip install scikit-fem. "
                    "fem_tool registration will be skipped on next startup."
                ),
            )

        try:
            # 先生成网格 (mesh_from_geometry 以外的 action 也需要)
            if args.action == "mesh_from_geometry":
                from .mesh import mesh_from_geometry
                return mesh_from_geometry(args)

            # 给 args 挂上 _mesh_result 以便后续 action 复用
            from .mesh import mesh_from_geometry
            mesh_result = mesh_from_geometry(args)
            if not mesh_result.success:
                return mesh_result
            # 用 object.__setattr__ 绕过 pydantic (pydantic v2 BaseModel 默认 immutable)
            object.__setattr__(args, "_mesh_result", mesh_result)

            if args.action == "static_linear":
                from .static import static_linear
                return static_linear(args)
            if args.action == "modal":
                from .modal import modal
                return modal(args)
            if args.action == "buckling":
                from .buckling import buckling
                return buckling(args)

            return ToolResult(
                data=None, success=False,
                error=f"Unknown action: {args.action}",
            )
        except Exception as exc:
            return ToolResult(
                data=None, success=False,
                error=f"FEM tool failed: {exc}",
            )
