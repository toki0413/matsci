"""SymbolicMathTool 主体: 21 个 action 分发, 各 action 实现在对应子模块."""

from __future__ import annotations

from typing import Any

import sympy as sp
from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult


class SymbolicMathInput(BaseModel):
    action: str = Field(
        ...,
        description="derive | solve | integrate | differentiate | taylor | eigenvalue | "
        "constitutive | weak_form | simplify | series | tensor_ops | tensor_calculus | "
        "dimensional_analysis | linear_algebra | dft | thermodynamics | probability | unified",
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
                return dft(args)
            if action == "thermodynamics":
                from .physics import thermodynamics
                return thermodynamics(args)
            if action == "probability":
                from .physics import probability
                return probability(args)
            if action == "unified":
                from .physics import unified
                return unified(args)

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
