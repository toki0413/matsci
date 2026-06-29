"""Mathematical structure extraction tool — wraps math-anything extractors.

A read-only tool for extracting mathematical semantics from simulation inputs.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile
from huginn.types import ToolContext, ToolResult


class ExtractToolInput(BaseModel):
    engine: Literal[
        "vasp", "lammps", "abaqus", "ansys", "comsol", "gromacs", "multiwfn"
    ] = Field(...)
    file_paths: dict[str, str] = Field(
        ...,
        description="Map of file types to paths, e.g. {'incar': 'INCAR', 'poscar': 'POSCAR'}",
    )


class ExtractTool(HuginnTool):
    """Extract mathematical structure from computational engine input files."""

    name = "extract_tool"
    category = "core"
    profile = ToolProfile(phases=frozenset({ResearchPhase.LITERATURE, ResearchPhase.REPORTING}))
    description = "Extract mathematical semantics (equations, constraints, approximations) from simulation input files using math-anything"
    input_schema = ExtractToolInput

    def is_read_only(self, args: ExtractToolInput) -> bool:
        return True

    async def call(self, args: ExtractToolInput, context: ToolContext) -> ToolResult:
        try:
            # Try to use math-anything if available
            try:
                from math_anything import MathAnything

                ma = MathAnything()
                result = ma.extract(args.engine, args.file_paths)

                return ToolResult(
                    data={
                        "engine": args.engine,
                        "mathematical_structure": result.schema.get(
                            "mathematical_structure", {}
                        ),
                        "constraints": result.schema.get("symbolic_constraints", []),
                        "approximations": result.schema.get("approximations", []),
                        "warnings": result.warnings,
                    },
                    success=result.success,
                    error="; ".join(result.errors) if result.errors else None,
                )

            except ImportError:
                # Fallback: basic parameter listing
                return ToolResult(
                    data={
                        "engine": args.engine,
                        "files": args.file_paths,
                        "note": "math-anything not installed — install it for full semantic extraction",
                    },
                    success=True,
                    error=None,
                )

        except Exception as e:
            return ToolResult(data=None, success=False, error=f"Extraction failed: {e}")
