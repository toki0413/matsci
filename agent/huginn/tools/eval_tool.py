"""Eval tool — safe expression evaluation.

Wraps safe_math_eval for LLM-accessible arithmetic/logic evaluation.
Supports numpy functions (np.sin, np.exp, etc.) and basic Python builtins.
No imports, no attribute access on arbitrary objects.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult


class EvalToolInput(BaseModel):
    action: Literal["eval"] = Field(default="eval")
    expression: str = Field(
        ...,
        description="Expression to evaluate (e.g. '3.14 * 2**10', 'np.sin(np.pi/4)')",
    )
    variables: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional variables available in the expression",
    )


class EvalTool(HuginnTool):
    """Evaluate mathematical/boolean expressions safely."""

    name = "eval_tool"
    category = "core"
    description = (
        "Evaluate a mathematical or boolean expression safely. "
        "Supports numpy functions (np.sin, np.exp, np.sqrt, etc.), "
        "arithmetic, comparisons, and conditional expressions. "
        "No imports, no attribute access on arbitrary objects."
    )
    input_schema = EvalToolInput
    destructive = False
    read_only = True

    def is_read_only(self, args: EvalToolInput) -> bool:
        return True

    async def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = EvalToolInput(**args)

        try:
            from huginn.security.math_eval import safe_math_eval
            from huginn.security.safe_eval import SafeEvalError

            result = safe_math_eval(input_data.expression, input_data.variables)
        except SafeEvalError as e:
            return ToolResult(data=None, success=False, error=str(e))
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"Evaluation error: {e}")

        return ToolResult(
            data={
                "expression": input_data.expression,
                "result": result,
                "result_type": type(result).__name__,
            },
            success=True,
        )
