"""Machine Learning Potential tool — train and use NEP, SNAP, GAP, ACE potentials.

Can be expensive. ASK permission mode by default.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import HandleType, ToolContext, ToolResult, ValidationResult
from huginn.validation.handle_validator import HandleValidator


class PotentialToolInput(BaseModel):
    action: Literal["train", "validate", "inference", "prepare_dataset"] = Field(...)
    potential_type: Literal["nep", "snap", "gap", "ace"] = Field(default="nep")
    dataset_path: str | None = Field(default=None)
    structure_path: str | None = Field(default=None)
    trained_potential_path: str | None = Field(default=None)
    config: dict = Field(default_factory=dict)


class PotentialTool(HuginnTool):
    """Train and use machine learning potentials for molecular dynamics."""

    name = "potential_tool"
    category = "core"
    description = "Train and apply ML potentials (NEP, SNAP, GAP, ACE) for fast molecular dynamics simulations"
    input_schema = PotentialToolInput

    def estimate_cost(self, args: PotentialToolInput) -> dict[str, float] | None:
        if args.action == "train":
            return {"cpu_hours": 48, "gpu_hours": 12, "walltime_hours": 24}
        return {"cpu_hours": 1, "walltime_hours": 1}

    async def validate_input(
        self, args: PotentialToolInput, context: ToolContext
    ) -> ValidationResult:
        """Pre-flight: verify dataset/structure paths when provided."""
        if args.dataset_path:
            vr = HandleValidator.validate(HandleType.FILE_PATH, args.dataset_path, context)
            if not vr.result:
                return ValidationResult(
                    result=False,
                    message=f"Dataset file not found: {args.dataset_path}",
                    error_code=404,
                )
        if args.structure_path:
            vr = HandleValidator.validate(HandleType.FILE_PATH, args.structure_path, context)
            if not vr.result:
                return ValidationResult(
                    result=False,
                    message=f"Structure file not found: {args.structure_path}",
                    error_code=404,
                )
        if args.trained_potential_path and args.action in ("inference", "validate"):
            vr = HandleValidator.validate(
                HandleType.FILE_PATH, args.trained_potential_path, context
            )
            if not vr.result:
                return ValidationResult(
                    result=False,
                    message=f"Trained potential not found: {args.trained_potential_path}",
                    error_code=404,
                )
        return ValidationResult(result=True)

    async def call(self, args: PotentialToolInput, context: ToolContext) -> ToolResult:
        if args.action == "prepare_dataset":
            return ToolResult(
                data={
                    "dataset_path": "dataset.extxyz",
                    "num_structures": 1000,
                    "note": "Mock dataset",
                },
                success=True,
            )

        elif args.action == "train":
            return ToolResult(
                data={
                    "potential_path": f"trained_{args.potential_type}.xml",
                    "training_rmse_energy": 0.005,
                    "training_rmse_force": 0.1,
                    "note": "Mock training result",
                },
                success=True,
            )

        elif args.action == "validate":
            return ToolResult(
                data={
                    "test_rmse_energy": 0.008,
                    "test_rmse_force": 0.15,
                    "note": "Mock validation result",
                },
                success=True,
            )

        elif args.action == "inference":
            return ToolResult(
                data={"energy": -123.45, "forces": "[[...]]", "note": "Mock inference"},
                success=True,
            )

        return ToolResult(
            data=None, success=False, error=f"Unknown action: {args.action}"
        )
