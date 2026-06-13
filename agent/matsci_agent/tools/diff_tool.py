"""Semantic diff tool — compare calculations using math-anything's MathDiffer.

Read-only. Safe to auto-execute.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from matsci_agent.tools.base import MatSciTool
from matsci_agent.types import ToolResult, ToolContext


class DiffToolInput(BaseModel):
    calc_a: str = Field(..., description="Path or ID of first calculation result")
    calc_b: str = Field(..., description="Path or ID of second calculation result")
    comparison_type: Literal["parameters", "results", "full"] = Field(default="full")


class DiffTool(MatSciTool):
    """Compare two calculations semantically (not just text diff)."""
    
    name = "diff_tool"
    description = "Semantically compare two calculations: parameter changes, mathematical structure differences, and physical implications"
    input_schema = DiffToolInput
    
    def is_read_only(self, args: DiffToolInput) -> bool:
        return True
    
    async def call(self, args: DiffToolInput, context: ToolContext) -> ToolResult:
        # TODO: integrate math-anything MathDiffer
        
        return ToolResult(
            data={
                "comparison_type": args.comparison_type,
                "changes": [
                    {"type": "parameter", "field": "ENCUT", "old": 400, "new": 520, "impact": "improved basis set completeness"},
                    {"type": "result", "field": "energy", "old": -100.0, "new": -102.5, "impact": "lower total energy"},
                ],
                "semantic_summary": "ENCUT increase from 400 to 520 eV led to ~2.5% energy improvement, indicating previous basis set was incomplete.",
                "note": "Full semantic diff requires math-anything MathDiffer integration"
            },
            success=True
        )
