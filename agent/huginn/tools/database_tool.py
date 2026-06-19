"""Materials database query tool — search and retrieve data from MP, AFLOW, NOMAD.

Read-only tool. Safe to auto-execute.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult


class DatabaseToolInput(BaseModel):
    database: Literal["materials_project", "aflow", "nomad", "oqmd"] = Field(...)
    query_type: Literal["search", "get_structure", "get_properties", "compare"] = Field(
        ...
    )
    formula: str | None = Field(default=None)
    material_id: str | None = Field(default=None)
    properties: list[str] = Field(default_factory=list)


class DatabaseTool(HuginnTool):
    """Query materials databases for structures and properties."""

    name = "database_tool"
    description = "Search materials databases (Materials Project, AFLOW, NOMAD, OQMD) for structures, properties, and literature data"
    input_schema = DatabaseToolInput

    def is_read_only(self, args: DatabaseToolInput) -> bool:
        return True

    async def call(self, args: DatabaseToolInput, context: ToolContext) -> ToolResult:
        # TODO: implement actual database API calls using pymatgen/aflowlib/nomad-client

        if args.query_type == "search":
            if not args.formula:
                return ToolResult(
                    data=None, success=False, error="formula is required for search"
                )

            # Mock result
            return ToolResult(
                data={
                    "database": args.database,
                    "query": args.formula,
                    "results": [
                        {
                            "material_id": f"{args.database.upper()}-1234",
                            "formula": args.formula,
                            "spacegroup": "Fm-3m",
                        }
                    ],
                    "note": "Database integration not yet implemented — returning mock data",
                },
                success=True,
            )

        elif args.query_type == "get_structure":
            if not args.material_id:
                return ToolResult(
                    data=None,
                    success=False,
                    error="material_id is required for get_structure",
                )

            return ToolResult(
                data={
                    "material_id": args.material_id,
                    "lattice": {"a": 4.2, "b": 4.2, "c": 4.2},
                    "note": "Mock structure data",
                },
                success=True,
            )

        elif args.query_type == "get_properties":
            return ToolResult(
                data={
                    "material_id": args.material_id,
                    "properties": {
                        "band_gap": 1.5,
                        "formation_energy": -2.3,
                        "bulk_modulus": 150,
                    },
                    "note": "Mock properties data",
                },
                success=True,
            )

        return ToolResult(
            data=None, success=False, error=f"Unknown query_type: {args.query_type}"
        )
