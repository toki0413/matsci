"""Structure analysis tool — read, analyze, and transform crystal structures.

A read-only tool for structural analysis. Safe to auto-execute.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import HandleType, ToolContext, ToolResult, ValidationResult
from huginn.validation.handle_validator import HandleValidator


class StructureToolInput(BaseModel):
    action: Literal["read", "analyze", "convert", "compare"] = Field(...)
    file_path: str = Field(
        ..., description="Path to structure file (POSCAR, CIF, XYZ, etc.)"
    )
    output_format: Literal["poscar", "cif", "xyz", "json"] | None = Field(default=None)
    reference_path: str | None = Field(default=None, description="For compare action")


class StructureToolOutput(BaseModel):
    formula: str | None = None
    spacegroup: str | None = None
    lattice_params: dict[str, float] | None = None
    num_atoms: int | None = None
    volume: float | None = None
    density: float | None = None
    warnings: list[str] = []


class StructureTool(HuginnTool):
    """Tool for reading and analyzing crystal structures."""

    name = "structure_tool"
    description = (
        "Read, analyze, and convert crystal structure files (POSCAR, CIF, XYZ)"
    )
    input_schema = StructureToolInput
    output_schema = StructureToolOutput

    def is_read_only(self, args: StructureToolInput) -> bool:
        return True

    async def validate_input(
        self, args: StructureToolInput, context: ToolContext
    ) -> ValidationResult:
        """Pre-flight: verify structure file exists."""
        vr = HandleValidator.validate(HandleType.FILE_PATH, args.file_path, context)
        if not vr.result:
            return ValidationResult(
                result=False,
                message=f"Structure file not found: {args.file_path}",
                error_code=404,
            )
        if args.reference_path:
            vr2 = HandleValidator.validate(HandleType.FILE_PATH, args.reference_path, context)
            if not vr2.result:
                return ValidationResult(
                    result=False,
                    message=f"Reference file not found: {args.reference_path}",
                    error_code=404,
                )
        return ValidationResult(result=True)

    async def call(self, args: StructureToolInput, context: ToolContext) -> ToolResult:
        path = Path(args.file_path)

        if not path.exists():
            return ToolResult(data=None, success=False, error=f"File not found: {path}")

        try:
            # Try to use pymatgen if available
            try:
                from pymatgen.core import Structure

                structure = Structure.from_file(str(path))

                output = StructureToolOutput(
                    formula=structure.formula,
                    spacegroup=(
                        structure.get_space_group_info()[0]
                        if hasattr(structure, "get_space_group_info")
                        else None
                    ),
                    lattice_params={
                        "a": structure.lattice.a,
                        "b": structure.lattice.b,
                        "c": structure.lattice.c,
                        "alpha": structure.lattice.alpha,
                        "beta": structure.lattice.beta,
                        "gamma": structure.lattice.gamma,
                    },
                    num_atoms=len(structure),
                    volume=structure.volume,
                    density=structure.density,
                )

                return ToolResult(data=output.model_dump(), success=True)

            except ImportError:
                # Fallback: basic file info
                content = path.read_text(encoding="utf-8", errors="ignore")
                lines = content.strip().split("\n")

                output = StructureToolOutput(
                    warnings=["pymatgen not installed — providing basic file info only"]
                )

                # Simple POSCAR detection
                if (
                    path.name.upper().startswith("POSCAR") or path.suffix == ".vasp"
                ) and len(lines) >= 6:
                    try:
                        num_atoms = sum(int(x) for x in lines[5].split())
                        output.num_atoms = num_atoms
                    except ValueError:
                        pass

                return ToolResult(data=output.model_dump(), success=True)

        except Exception as e:
            return ToolResult(
                data=None, success=False, error=f"Failed to parse structure: {e}"
            )
