"""轻量 FEM 工具 — scikit-fem 封装.

2D 线性平面应力/应变: 静力/模态/屈曲. scikit-fem 缺失时 optional_modules 自动跳过.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from huginn.core_types import ToolContext, ToolResult
from huginn.tools.base import HuginnTool, ToolProfile

# scikit-fem 在 tool.py 顶部导入 — 缺失时 ImportError 让 optional_modules 跳过注册
try:
    import skfem  # noqa: F401
    _SKFEM_AVAILABLE = True
except ImportError:
    _SKFEM_AVAILABLE = False


def _fem_dimensional_precheck(args: FEMInput) -> dict:
    """FEM 求解前的量纲/合理性前置自检(契约层, 非学习).

    FEM 输入是裸数值、单位隐式假设 SI, 因此这里用**假定 SI 的单位映射 + 解析解表达式**
    对接量纲契约层:

      - 材料合理性硬门槛: nu 无量纲须落物理允许域 (-1, 0.5); E/rho/thickness/几何尺寸 > 0;
      - 静力弯曲刚度 D=E·h³ 与模态频率 ω∝√(E/ρ)·1/L 的量纲自检(用
        ``check_expression_dimensions`` 静态度量)。

    Returns: {"ok": bool, "checks": [dict], "error": str|None} —— ok=False 时 caller 应硬拒。
    量纲引擎不可用 → checks 空、ok=True(线上无 sympy 时不阻断 FEM, 如实现"诚实边界").
    """
    checks: list[dict] = []

    # ── 材料/几何合理性 (针对带单位字段的工程硬门槛, 零表达式) ──
    mat = dict(args.material or {})
    nu = mat.get("nu")
    if nu is not None and not (-1.0 < nu < 0.5):
        checks.append({
            "name": "material.nu",
            "unit": "1", "ok": False,
            "error": f"泊松比 nu={nu} 超物理允许域 (-1, 0.5)",
        })
    for key, lo in (("E", 0.0), ("rho", 0.0), ("thickness", 0.0)):
        if key in mat and float(mat[key]) <= lo:
            checks.append({
                "name": f"material.{key}", "unit": _FEM_UNIT.get(key),
                "ok": False, "error": f"{key}={mat[key]} 须 > 0",
            })
    for k, v in (args.dims or {}).items():
        if v is not None and float(v) <= 0:
            checks.append({
                "name": f"dims.{k}", "unit": "m",
                "ok": False, "error": f"几何尺寸 {k}={v} 须 > 0",
            })

    # ── 解析解量纲自检 (求解器实际算的物理量) ──
    try:
        from huginn.research.external_validator import check_expression_dimensions
    except Exception:  # noqa: BLE001
        return {"ok": not any(c["ok"] is False for c in checks), "checks": checks,
                "error": None}

    sym_units = {
        "E": _FEM_UNIT.get("E", "Pa"),
        "rho": _FEM_UNIT.get("rho", "kg/m3"),
        "nu": "1",
        "L": "m",
        "h": "m",
    }
    # 静力弯曲刚度量纲: D = E·h³/(1-ν²) → N·m (M·L²·T⁻² · L · ...), expected N*m
    if args.action == "static_linear":
        checks.append({
            "name": "bending_stiffness",
            "expr": "E * h**3 / (1 - nu**2)",
            **check_expression_dimensions("E * h**3 / (1 - nu**2)", sym_units, "N*m"),
        })
    # 模态频率量纲: ω ∝ √(E/ρ) · 1/L → 1/s
    if args.action == "modal":
        checks.append({
            "name": "modal_frequency",
            "expr": "sqrt(E / rho) / L",
            **check_expression_dimensions("sqrt(E / rho) / L", sym_units, "1/s"),
        })

    hard_fail = any(c.get("ok") is False and c.get("error")
                    and "量纲引擎" not in c["error"] for c in checks)
    return {"ok": not hard_fail, "checks": checks,
            "error": None if not hard_fail else "FEM 物理量纲/合理性前置自检未通过"}


# FEM 带单位字段 → SI 单位标签 (隐式假设的实验输入)
_FEM_UNIT = {
    "E": "Pa", "nu": "1", "rho": "kg/m3", "thickness": "m",
    "L": "m", "H": "m", "R": "m",
}


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
    def _check_action_fields(self) -> FEMInput:
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
    profile = ToolProfile(constraint_scope="fea")
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

            # 求解前的量纲/合理性自检(契约层, 非学习): nu 域 / 正值 / 解析解量纲.
            pre = _fem_dimensional_precheck(args)
            if not pre["ok"]:
                return ToolResult(
                    data={"precheck": pre["checks"]},
                    success=False,
                    error=pre["error"] or "FEM 量纲/合理性自检未通过",
                )

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
