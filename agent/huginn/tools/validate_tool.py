"""Physical validation tool — run all physics validators on a calculation result.

Read-only. Safe to auto-execute.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from huginn.constraints import ConstraintAdapter
from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult


class ValidateToolInput(BaseModel):
    result_type: Literal["dft", "md", "phonon"] = Field(...)
    result_data: dict = Field(..., description="Calculation result dict to validate")


class ValidateTool(HuginnTool):
    """Validate physical reasonableness of calculation results."""

    name = "validate_tool"
    description = (
        "Run physical validation checks on calculation results: energy signs, "
        "convergence, band gaps, force thresholds, etc."
    )
    input_schema = ValidateToolInput

    def __init__(self) -> None:
        super().__init__()
        self._adapter = ConstraintAdapter.default()

    def is_read_only(self, args: ValidateToolInput) -> bool:
        return True

    async def call(self, args: ValidateToolInput, context: ToolContext) -> ToolResult:
        results = self._adapter.evaluate_all(args.result_type, args.result_data)
        all_passed = all(r.passed for r in results)

        return ToolResult(
            data={
                "all_passed": all_passed,
                "checks": [
                    {
                        "name": r.name,
                        "passed": r.passed,
                        "value": r.value,
                        "expected": r.expected,
                        "message": r.message,
                        "severity": r.severity,
                        "family": r.family,
                    }
                    for r in results
                ],
                "summary": f"{sum(r.passed for r in results)}/{len(results)} checks passed",
            },
            success=True,
        )
