"""SymbolicMathTool 主体: 21 个 action 分发, 各 action 实现在对应子模块."""

from __future__ import annotations

from typing import Any

import sympy as sp
from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile
from huginn.types import ToolContext, ToolResult
import logging
logger = logging.getLogger(__name__)



class SymbolicMathInput(BaseModel):
    action: str = Field(
        ...,
        description="derive | solve | integrate | differentiate | taylor | eigenvalue | "
        "constitutive | weak_form | simplify | series | tensor_ops | tensor_calculus | "
        "dimensional_analysis | linear_algebra | dft | thermodynamics | probability | unified | "
        "pde_classify | pde_separation | pde_characteristics | pde_discretize | "
        "euler_lagrange | functional_derivative | isoperimetric | noether | "
        "diffgeo_metric | diffgeo_geodesic | diffgeo_curvature | diffgeo_lie_derivative | diffgeo_connection",
    )
    expression: str | None = Field(
        default=None, description="Mathematical expression as string"
    )
    symbols: list[str] = Field(default_factory=list, description="List of symbol names")
    variable: str | None = Field(
        default=None, description="Variable for differentiation/integration"
    )
    target: str | None = Field(
        default=None,
        description="Target for constitutive derivation (e.g., 'stress_from_psi')",
    )
    free_energy: str | None = Field(
        default=None, description="Free energy expression for constitutive derivation"
    )
    equations: list[str] | None = Field(
        default_factory=list, description="List of equations for solving"
    )
    order: int = Field(
        default=1, ge=1, le=10, description="Order for Taylor/differentiation"
    )
    point: dict[str, float] | None = Field(
        default=None, description="Expansion point for Taylor series"
    )
    matrix: list[list[str]] | None = Field(
        default=None, description="Matrix as list of string expressions"
    )
    assumptions: dict[str, str] = Field(
        default_factory=dict,
        description="SymPy assumptions: {symbol: 'positive'|'real'|'complex'}",
    )
    tensor_type: str | None = Field(
        default=None,
        description="stress | strain | stiffness | compliance (for tensor_calculus)",
    )
    voigt_vector: list[float] | None = Field(
        default=None,
        description="Voigt vector components [v11, v22, v33, v23, v13, v12] or 21-element stiffness",
    )
    rotation_matrix: list[list[float]] | None = Field(
        default=None, description="3×3 rotation matrix for tensor rotation"
    )
    output_path: str | None = Field(
        default=None, description="Output path for visualize / plot actions"
    )
    sub_action: str | None = Field(
        default=None,
        description="Sub-action for actions like tensor_calculus (e.g. 'einstein_sum')",
    )
    indices: list[str] | None = Field(
        default=None,
        description="Explicit free-index order for einstein_sum",
    )
    sum_indices: list[str] | None = Field(
        default=None,
        description="Explicit summation indices for einstein_sum",
    )
    metric: dict[str, Any] | None = Field(
        default=None,
        description="Metric tensor for raising/lowering indices in einstein_sum",
    )
    simplify: bool = Field(
        default=True,
        description="Whether to call sympy.simplify on the einstein_sum result",
    )


class SymbolicMathTool(HuginnTool):
    """材料科学符号数学计算."""

    name = "symbolic_math_tool"
    category = "sci"
    profile = ToolProfile(
        cost_tier="light",
        phases=frozenset({
            ResearchPhase.LITERATURE,
            ResearchPhase.HYPOTHESIS,
            ResearchPhase.PLANNING,
            ResearchPhase.VALIDATION,
            ResearchPhase.REPORTING,
        }),
    )
    description = (
        "Perform symbolic mathematics using SymPy: differentiation, integration, "
        "equation solving, matrix operations, constitutive relation derivation, "
        "and weak form verification for finite element methods."
    )
    input_schema = SymbolicMathInput

    def is_read_only(self, args: SymbolicMathInput) -> bool:
        return True

    async def call(self, args: SymbolicMathInput, context: ToolContext) -> ToolResult:
        action = args.action.lower()

        try:
            if action in ("differentiate", "derivative"):
                from .calculus import differentiate
                return differentiate(args)
            if action == "integrate":
                from .calculus import integrate
                return integrate(args)
            if action == "solve":
                from .algebra import solve
                return solve(args)
            if action == "simplify":
                from .calculus import simplify
                return simplify(args)
            if action == "taylor":
                from .calculus import taylor
                return taylor(args)
            if action == "eigenvalue":
                from .algebra import eigenvalue
                return eigenvalue(args)
            if action == "constitutive":
                from .fem import constitutive
                return constitutive(args)
            if action == "weak_form":
                from .fem import weak_form
                return weak_form(args)
            if action == "series":
                from .calculus import series
                return series(args)
            if action == "tensor_ops":
                from .tensor import tensor_ops
                return tensor_ops(args)
            if action == "tensor_calculus":
                from .tensor import tensor_calculus
                return tensor_calculus(args)
            if action == "dimensional_analysis":
                from .physics import dimensional_analysis
                return dimensional_analysis(args)
            if action == "linear_algebra":
                from .algebra import linear_algebra
                return linear_algebra(args)
            if action == "dft":
                from .physics import dft
                result = dft(args)
                return self._augment_with_kb_constants(result, context, "dft")
            if action == "thermodynamics":
                from .physics import thermodynamics
                result = thermodynamics(args)
                return self._augment_with_kb_constants(result, context, "thermodynamics")
            if action == "probability":
                from .physics import probability
                return probability(args)
            if action == "unified":
                from .physics import unified
                return unified(args)
            if action == "pde_classify":
                from .pde import classify
                return classify(args)
            if action == "pde_separation":
                from .pde import separation
                return separation(args)
            if action == "pde_characteristics":
                from .pde import characteristics
                return characteristics(args)
            if action == "pde_discretize":
                from .pde import discretize
                return discretize(args)
            if action == "euler_lagrange":
                from .variational import euler_lagrange
                return euler_lagrange(args)
            if action == "functional_derivative":
                from .variational import functional_derivative
                return functional_derivative(args)
            if action == "isoperimetric":
                from .variational import isoperimetric
                return isoperimetric(args)
            if action == "noether":
                from .variational import noether
                return noether(args)
            if action == "diffgeo_metric":
                from .diffgeo import metric
                return metric(args)
            if action == "diffgeo_geodesic":
                from .diffgeo import geodesic
                return geodesic(args)
            if action == "diffgeo_curvature":
                from .diffgeo import curvature
                return curvature(args)
            if action == "diffgeo_lie_derivative":
                from .diffgeo import lie_derivative
                return lie_derivative(args)
            if action == "diffgeo_connection":
                from .diffgeo import connection
                return connection(args)
            if action == "derive":
                # derive 是 EL 方程的别名: 从拉氏量推导运动方程
                from .variational import euler_lagrange
                return euler_lagrange(args)

            return ToolResult(
                data=None,
                success=False,
                error=f"Unknown action: {args.action}",
            )
        except Exception as e:
            return ToolResult(
                data=None,
                success=False,
                error=f"Symbolic computation error: {str(e)}",
            )

    @staticmethod
    def _augment_with_kb_constants(
        result: ToolResult, context: ToolContext, domain: str
    ) -> ToolResult:
        """A6: dft/thermodynamics 返回后查 KB 校验物理常数 (ℏ, k_B, R 等).
        命中就把 reference 文本写进 data["kb_verified_constants"], 不改计算结果.
        KB 不可用/空/查询失败都静默跳过."""
        if not result.success or not isinstance(result.data, dict):
            return result
        try:
            workspace = getattr(context, "workspace", None) or "."
            from huginn.knowledge.store import get_knowledge_base

            kb = get_knowledge_base(str(workspace))
            if kb.count() == 0:
                return result
            # 按 domain 拿不同常数集合的 KB 参考
            query_map = {
                "dft": "physical constants hbar electron mass planck",
                "thermodynamics": "physical constants gas constant boltzmann R",
            }
            chunks = kb.query(query_map.get(domain, "physical constants"), top_k=2)
            refs = [
                {"text": (c.get("text") or "")[:200], "source": c.get("source", "")}
                for c in chunks
                if c.get("text")
            ]
            if refs:
                result.data["kb_verified_constants"] = refs
        except Exception:
            logger.debug("augment with kb constants failed", exc_info=True)
        return result
