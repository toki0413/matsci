"""Physical validation tool — run all physics validators on a calculation result.

Read-only. Safe to auto-execute.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolResult, ToolContext


class ValidateToolInput(BaseModel):
    result_type: Literal["dft", "md", "phonon"] = Field(...)
    result_data: dict = Field(..., description="Calculation result dict to validate")


class ValidateTool(HuginnTool):
    """Validate physical reasonableness of calculation results."""
    
    name = "validate_tool"
    description = "Run physical validation checks on calculation results: energy signs, convergence, band gaps, force thresholds, etc."
    input_schema = ValidateToolInput
    
    def is_read_only(self, args: ValidateToolInput) -> bool:
        return True
    
    async def call(self, args: ValidateToolInput, context: ToolContext) -> ToolResult:
        from huginn.validation.physics import PhysicsValidator
        
        validator = PhysicsValidator()
        
        if args.result_type == "dft":
            checks = validator.validate_dft_result(args.result_data)
        elif args.result_type == "md":
            checks = validator.validate_md_result(args.result_data)
        elif args.result_type == "phonon":
            checks = validator.validate_phonon_result(args.result_data)
        else:
            return ToolResult(
                data=None,
                success=False,
                error=f"Unknown result_type: {args.result_type}"
            )
        
        all_passed = all(c.passed for c in checks)
        
        return ToolResult(
            data={
                "all_passed": all_passed,
                "checks": [
                    {
                        "name": c.name,
                        "passed": c.passed,
                        "value": c.value,
                        "expected": c.expected,
                        "message": c.message
                    }
                    for c in checks
                ],
                "summary": f"{sum(c.passed for c in checks)}/{len(checks)} checks passed"
            },
            success=True
        )
