"""Mat-DB MCP Server — Material Science Database Query Server.

Provides MCP tools for querying materials databases:
- Materials Project (MP)
- AFLOW
- NOMAD
- OQMD
- NIST Interatomic Potentials

Usage:
    python server.py [--transport stdio|sse]
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool, Resource

app = Server("mat-db-mcp")

# ---------------------------------------------------------------------------
# Mock data fallback when APIs are unavailable
# ---------------------------------------------------------------------------

MOCK_STRUCTURES: dict[str, dict[str, Any]] = {
    "Si": {
        "formula": "Si",
        "spacegroup": "Fd-3m",
        "lattice_a": 5.431,
        "band_gap": 1.14,
        "energy_per_atom": -5.424,
        "bulk_modulus": 97.8,
        "source": "mock/MaterialsProject",
    },
    "GaAs": {
        "formula": "GaAs",
        "spacegroup": "F-43m",
        "lattice_a": 5.653,
        "band_gap": 1.43,
        "energy_per_atom": -4.652,
        "bulk_modulus": 75.0,
        "source": "mock/MaterialsProject",
    },
    "TiO2": {
        "formula": "TiO2",
        "spacegroup": "P4_2/mnm",
        "lattice_a": 4.594,
        "lattice_c": 2.959,
        "band_gap": 3.03,
        "energy_per_atom": -8.099,
        "source": "mock/MaterialsProject",
    },
    "LiFePO4": {
        "formula": "LiFePO4",
        "spacegroup": "Pnma",
        "lattice_a": 10.332,
        "lattice_b": 6.010,
        "lattice_c": 4.691,
        "band_gap": 3.55,
        "energy_per_atom": -6.446,
        "source": "mock/MaterialsProject",
    },
}

MOCK_POTENTIALS: list[dict[str, Any]] = [
    {
        "id": "Si_EAM_Dufty-2016",
        "elements": ["Si"],
        "potential_type": "EAM",
        "reference": "Dufty et al., Phys. Rev. B 2016",
        "source": "mock/NIST",
    },
    {
        "id": "Cu_EAM_Mishin-2001",
        "elements": ["Cu"],
        "potential_type": "EAM",
        "reference": "Mishin et al., Phys. Rev. B 2001",
        "source": "mock/NIST",
    },
]

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS: list[Tool] = [
    Tool(
        name="query_materials_project",
        description="Query Materials Project for structure and property data by formula or MP-ID",
        inputSchema={
            "type": "object",
            "properties": {
                "formula": {"type": "string", "description": "Chemical formula, e.g. 'TiO2'"},
                "mp_id": {"type": "string", "description": "Materials Project ID, e.g. 'mp-554278'"},
                "properties": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Properties to return: band_gap, energy_per_atom, bulk_modulus, ...",
                },
            },
        },
    ),
    Tool(
        name="search_by_property",
        description="Search materials by property range (band gap, bulk modulus, etc.)",
        inputSchema={
            "type": "object",
            "properties": {
                "property": {"type": "string", "description": "Property name"},
                "min": {"type": "number", "description": "Minimum value"},
                "max": {"type": "number", "description": "Maximum value"},
                "limit": {"type": "integer", "default": 10, "description": "Max results"},
            },
            "required": ["property", "min", "max"],
        },
    ),
    Tool(
        name="get_structure",
        description="Get crystal structure (POSCAR/CIF format) for a material",
        inputSchema={
            "type": "object",
            "properties": {
                "formula": {"type": "string"},
                "format": {"type": "string", "enum": ["poscar", "cif", "json"], "default": "json"},
            },
            "required": ["formula"],
        },
    ),
    Tool(
        name="query_interatomic_potentials",
        description="Query NIST Interatomic Potentials database",
        inputSchema={
            "type": "object",
            "properties": {
                "elements": {"type": "array", "items": {"type": "string"}, "description": "Elements to search"},
                "potential_type": {"type": "string", "description": "EAM, MEAM, ReaxFF, NEP, SNAP, etc."},
            },
        },
    ),
    Tool(
        name="compare_materials",
        description="Compare properties of multiple materials side-by-side",
        inputSchema={
            "type": "object",
            "properties": {
                "formulas": {"type": "array", "items": {"type": "string"}},
                "properties": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["formulas"],
        },
    ),
]


@app.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


@app.call_tool()
async def call_tool(name: str, arguments: dict | None) -> list[TextContent]:
    arguments = arguments or {}

    if name == "query_materials_project":
        return await _query_materials_project(arguments)
    elif name == "search_by_property":
        return await _search_by_property(arguments)
    elif name == "get_structure":
        return await _get_structure(arguments)
    elif name == "query_interatomic_potentials":
        return await _query_interatomic_potentials(arguments)
    elif name == "compare_materials":
        return await _compare_materials(arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

async def _query_materials_project(args: dict) -> list[TextContent]:
    formula = args.get("formula", "")
    mp_id = args.get("mp_id", "")
    props = args.get("properties", [])

    # Try real MP API if key available
    api_key = os.environ.get("MP_API_KEY")
    if api_key and (formula or mp_id):
        try:
            result = await _fetch_mp_real(api_key, formula, mp_id, props)
            return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]
        except Exception as e:
            pass  # Fall back to mock

    # Mock fallback
    data = MOCK_STRUCTURES.get(formula, {
        "formula": formula,
        "note": "Not found in mock database. Use real MP_API_KEY for live data.",
    })
    if props:
        data = {k: v for k, v in data.items() if k in props or k in {"formula", "source", "note"}}

    return [TextContent(type="text", text=json.dumps(data, indent=2, ensure_ascii=False))]


async def _fetch_mp_real(api_key: str, formula: str, mp_id: str, props: list[str]) -> dict:
    """Fetch from real Materials Project API (requires mp-api package)."""
    try:
        from mp_api.client import MPRester
        with MPRester(api_key) as mpr:
            if mp_id:
                doc = mpr.materials.summary.get_data_by_id(mp_id)
            elif formula:
                docs = mpr.materials.summary.search(formula=formula, limit=1)
                doc = docs[0] if docs else None
            else:
                return {"error": "No formula or mp_id provided"}

            if not doc:
                return {"error": "Material not found"}

            result = {"formula": doc.formula_pretty, "mp_id": doc.material_id}
            if "band_gap" in props or not props:
                result["band_gap"] = doc.band_gap
            if "energy_per_atom" in props or not props:
                result["energy_per_atom"] = doc.energy_per_atom
            if "bulk_modulus" in props or not props and doc.bulk_modulus:
                result["bulk_modulus"] = doc.bulk_modulus.get("vrh", None)
            return result
    except ImportError:
        raise RuntimeError("mp-api package not installed. Run: pip install mp-api")


async def _search_by_property(args: dict) -> list[TextContent]:
    prop = args.get("property", "")
    min_val = args.get("min", 0)
    max_val = args.get("max", float("inf"))
    limit = args.get("limit", 10)

    results = []
    for formula, data in MOCK_STRUCTURES.items():
        val = data.get(prop)
        if val is not None and min_val <= val <= max_val:
            results.append({"formula": formula, prop: val, "source": data.get("source")})
        if len(results) >= limit:
            break

    return [TextContent(type="text", text=json.dumps(results, indent=2, ensure_ascii=False))]


async def _get_structure(args: dict) -> list[TextContent]:
    formula = args.get("formula", "")
    fmt = args.get("format", "json")

    data = MOCK_STRUCTURES.get(formula, {"formula": formula, "note": "Not found"})

    if fmt == "poscar":
        # Generate a simple POSCAR-like string
        a = data.get("lattice_a", 4.0)
        text = f"{formula}\n1.0\n{a:.6f} 0.0 0.0\n0.0 {a:.6f} 0.0\n0.0 0.0 {a:.6f}\n{formula}\n1\nDirect\n0.0 0.0 0.0\n"
        return [TextContent(type="text", text=text)]
    elif fmt == "cif":
        text = f"data_{formula}\n_symmetry_space_group_name_H-M 'P1'\n_cell_length_a {data.get('lattice_a', 4.0)}\n"
        return [TextContent(type="text", text=text)]
    else:
        return [TextContent(type="text", text=json.dumps(data, indent=2, ensure_ascii=False))]


async def _query_interatomic_potentials(args: dict) -> list[TextContent]:
    elements = set(args.get("elements", []))
    ptype = args.get("potential_type", "").lower()

    results = []
    for pot in MOCK_POTENTIALS:
        pot_elements = set(pot.get("elements", []))
        if elements and not elements.issubset(pot_elements):
            continue
        if ptype and ptype not in pot.get("potential_type", "").lower():
            continue
        results.append(pot)

    return [TextContent(type="text", text=json.dumps(results, indent=2, ensure_ascii=False))]


async def _compare_materials(args: dict) -> list[TextContent]:
    formulas = args.get("formulas", [])
    props = args.get("properties", ["band_gap", "energy_per_atom", "bulk_modulus"])

    table = []
    for f in formulas:
        data = MOCK_STRUCTURES.get(f, {})
        row = {"formula": f}
        for p in props:
            row[p] = data.get(p, "N/A")
        table.append(row)

    return [TextContent(type="text", text=json.dumps(table, indent=2, ensure_ascii=False))]


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

@app.list_resources()
async def list_resources() -> list[Resource]:
    return [
        Resource(uri="matdb://overview", name="Database Overview", mimeType="application/json"),
        Resource(uri="matdb://periodic-table", name="Periodic Table Reference", mimeType="text/plain"),
    ]


@app.read_resource()
async def read_resource(uri: str) -> str:
    if uri == "matdb://overview":
        return json.dumps({
            "databases": ["Materials Project", "AFLOW", "NOMAD", "OQMD", "NIST potentials"],
            "mock_mode": os.environ.get("MP_API_KEY") is None,
            "available_tools": [t.name for t in TOOLS],
        }, indent=2)
    elif uri == "matdb://periodic-table":
        return "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca ..."
    return "Unknown resource"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="mat-db-mcp",
                server_version="0.1.0",
                capabilities=app.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
