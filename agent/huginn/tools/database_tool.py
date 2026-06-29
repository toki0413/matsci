"""Materials database query tool — search and retrieve data from MP, AFLOW, NOMAD, OQMD.

Read-only tool. Safe to auto-execute.
Supports real API calls to Materials Project and OQMD when API keys are available;
falls back to structured mock data for AFLOW/NOMAD or when keys are missing.
"""

from __future__ import annotations

import os
from typing import Any, Literal
from urllib.parse import urlencode

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
    api_key: str | None = Field(
        default=None, description="API key override (falls back to env vars)"
    )
    limit: int = Field(default=10, ge=1, le=100)


class DatabaseTool(HuginnTool):
    """Query materials databases for structures and properties."""

    name = "database_tool"
    category = "core"
    description = "Search materials databases (Materials Project, AFLOW, NOMAD, OQMD) for structures, properties, and literature data"
    input_schema = DatabaseToolInput

    def is_read_only(self, args: DatabaseToolInput) -> bool:
        return True

    def _get_api_key(self, args: DatabaseToolInput) -> str | None:
        env_map = {
            "materials_project": "MP_API_KEY",
            "oqmd": "OQMD_API_KEY",
            "aflow": "AFLOW_API_KEY",
            "nomad": "NOMAD_API_KEY",
        }
        return args.api_key or os.environ.get(env_map.get(args.database, ""), None)

    async def call(self, args: DatabaseToolInput, context: ToolContext) -> ToolResult:
        api_key = self._get_api_key(args)

        if args.query_type == "compare":
            return await self._compare(args, api_key)

        handler = {
            "materials_project": self._query_mp,
            "oqmd": self._query_oqmd,
            "aflow": self._query_aflow,
            "nomad": self._query_nomad,
        }.get(args.database)

        if handler is None:
            return ToolResult(
                data=None, success=False, error=f"Unknown database: {args.database}"
            )
        return await handler(args, api_key)

    # ── Materials Project ───────────────────────────────────────────

    async def _query_mp(
        self, args: DatabaseToolInput, api_key: str | None
    ) -> ToolResult:
        if not api_key:
            return self._mock_result(args, "MP_API_KEY not set")

        try:
            import aiohttp
        except ImportError:
            return self._mock_result(args, "aiohttp not installed")

        base = "https://api.materialsproject.org"
        params: dict[str, Any] = {"API_KEY": api_key, "limit": args.limit}

        if args.query_type == "search":
            if not args.formula:
                return ToolResult(
                    data=None, success=False, error="formula is required for search"
                )
            if args.formula.lower().startswith("mp-"):
                params["material_ids"] = args.formula
            else:
                params["formula"] = args.formula
            fields = args.properties or [
                "material_id", "formula_pretty", "energy_per_atom",
                "band_gap", "symmetry",
            ]
            params["fields"] = ",".join(fields)
            url = f"{base}/materials/summary/?{urlencode(params, doseq=True)}"

        elif args.query_type == "get_structure":
            if not args.material_id:
                return ToolResult(
                    data=None, success=False, error="material_id required"
                )
            params["fields"] = "structure,material_id,formula_pretty"
            url = f"{base}/materials/core/{args.material_id}?{urlencode(params)}"

        elif args.query_type == "get_properties":
            mid = args.material_id or args.formula
            if not mid:
                return ToolResult(
                    data=None, success=False, error="material_id or formula required"
                )
            fields = args.properties or [
                "material_id", "energy_per_atom", "band_gap",
                "formation_energy_per_atom", "bulk_modulus",
            ]
            params["fields"] = ",".join(fields)
            if mid.lower().startswith("mp-"):
                params["material_ids"] = mid
            else:
                params["formula"] = mid
            url = f"{base}/materials/summary/?{urlencode(params, doseq=True)}"
        else:
            return ToolResult(data=None, success=False, error=f"Unknown query_type: {args.query_type}")

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    return ToolResult(
                        data=None, success=False,
                        error=f"MP API error {resp.status}: {text[:200]}",
                    )
                data = await resp.json()

        records = []
        for item in data.get("data", []):
            records.append({
                "material_id": item.get("material_id"),
                "formula": item.get("formula_pretty"),
                "energy_per_atom": item.get("energy_per_atom"),
                "band_gap": item.get("band_gap"),
                "spacegroup": (item.get("symmetry") or {}).get("symbol"),
            })

        return ToolResult(
            data={
                "database": "materials_project",
                "query_type": args.query_type,
                "count": len(records),
                "records": records,
            },
            success=True,
        )

    # ── OQMD ────────────────────────────────────────────────────────

    async def _query_oqmd(
        self, args: DatabaseToolInput, api_key: str | None
    ) -> ToolResult:
        if not api_key:
            return self._mock_result(args, "OQMD_API_KEY not set")

        try:
            import aiohttp
        except ImportError:
            return self._mock_result(args, "aiohttp not installed")

        base = "http://oqmd.org/oqmdapi"
        headers = {"X-API-KEY": api_key}
        params: dict[str, Any] = {"limit": args.limit}

        if args.query_type == "search":
            if not args.formula:
                return ToolResult(
                    data=None, success=False, error="formula is required for search"
                )
            params["composition"] = args.formula
            fields = args.properties or ["name", "entry_id", "band_gap", "delta_e"]
            params["fields"] = ",".join(fields)
            url = f"{base}/entry?{urlencode(params, doseq=True)}"

        elif args.query_type == "get_structure":
            if not args.material_id:
                return ToolResult(
                    data=None, success=False, error="material_id (entry_id) required"
                )
            url = f"{base}/entry/{args.material_id}?fields=structure,name,entry_id,band_gap,delta_e"

        elif args.query_type == "get_properties":
            mid = args.material_id or args.formula
            if not mid:
                return ToolResult(
                    data=None, success=False, error="material_id or formula required"
                )
            if "=" in mid or "<" in mid or ">" in mid:
                params["filter"] = mid
            else:
                params["composition"] = mid
            fields = args.properties or ["name", "entry_id", "band_gap", "delta_e"]
            params["fields"] = ",".join(fields)
            url = f"{base}/entry?{urlencode(params, doseq=True)}"
        else:
            return ToolResult(data=None, success=False, error=f"Unknown query_type: {args.query_type}")

        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    return ToolResult(
                        data=None, success=False,
                        error=f"OQMD API error {resp.status}: {text[:200]}",
                    )
                data = await resp.json()

        records = []
        items = data.get("data", [])
        if isinstance(items, dict):
            items = [items]
        for item in items:
            records.append({
                "entry_id": item.get("entry_id"),
                "formula": item.get("name"),
                "band_gap": item.get("band_gap"),
                "formation_energy": item.get("delta_e"),
            })

        return ToolResult(
            data={
                "database": "oqmd",
                "query_type": args.query_type,
                "count": len(records),
                "records": records,
            },
            success=True,
        )

    # ── AFLOW (structured fallback) ────────────────────────────────

    async def _query_aflow(
        self, args: DatabaseToolInput, api_key: str | None
    ) -> ToolResult:
        # AFLOW REST API is limited; use aiohttp if key available, else mock
        if api_key:
            try:
                import aiohttp

                base = "https://aflow.org/API/aflow.shtml"
                params: dict[str, Any] = {
                    "keywords": args.formula or args.material_id or "*",
                    "limit": args.limit,
                }
                url = f"{base}?{urlencode(params)}"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            text = await resp.text()
                            return ToolResult(
                                data={
                                    "database": "aflow",
                                    "query_type": args.query_type,
                                    "raw_response_preview": text[:500],
                                    "note": "AFLOW response parsed as raw text",
                                },
                                success=True,
                            )
                        return ToolResult(
                            data=None, success=False,
                            error=f"AFLOW API error {resp.status}",
                        )
            except ImportError:
                pass
        return self._mock_result(args, "AFLOW API key not set or aiohttp unavailable")

    # ── NOMAD (structured fallback) ────────────────────────────────

    async def _query_nomad(
        self, args: DatabaseToolInput, api_key: str | None
    ) -> ToolResult:
        if api_key:
            try:
                import aiohttp

                base = "https://nomad-lab.eu/prod/v1/api/v1"
                headers = {"Authorization": f"Bearer {api_key}"}
                query: dict[str, Any] = {}
                if args.formula:
                    query["formula"] = args.formula
                params = {"page_size": args.limit}
                url = f"{base}/entries?{urlencode(params)}"
                async with aiohttp.ClientSession(headers=headers) as session:
                    async with session.post(url, json=query) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            records = []
                            for item in data.get("data", []):
                                records.append({
                                    "entry_id": item.get("entry_id"),
                                    "formula": item.get("formula"),
                                })
                            return ToolResult(
                                data={
                                    "database": "nomad",
                                    "query_type": args.query_type,
                                    "count": len(records),
                                    "records": records,
                                },
                                success=True,
                            )
                        return ToolResult(
                            data=None, success=False,
                            error=f"NOMAD API error {resp.status}",
                        )
            except ImportError:
                pass
        return self._mock_result(args, "NOMAD API key not set or aiohttp unavailable")

    # ── Compare ─────────────────────────────────────────────────────

    async def _compare(
        self, args: DatabaseToolInput, api_key: str | None
    ) -> ToolResult:
        """Compare properties across databases for the same material."""
        if not args.formula and not args.material_id:
            return ToolResult(
                data=None, success=False,
                error="formula or material_id required for compare",
            )

        # Query each available database
        results: dict[str, Any] = {}
        for db_name in ["materials_project", "oqmd"]:
            sub_args = DatabaseToolInput(
                database=db_name,
                query_type="get_properties",
                formula=args.formula,
                material_id=args.material_id,
                properties=args.properties,
                api_key=None,  # will use env
                limit=5,
            )
            key = self._get_api_key(sub_args)
            if key:
                if db_name == "materials_project":
                    r = await self._query_mp(sub_args, key)
                else:
                    r = await self._query_oqmd(sub_args, key)
                if r.success:
                    results[db_name] = r.data

        if not results:
            return ToolResult(
                data={
                    "status": "no_databases_available",
                    "note": "Set MP_API_KEY and/or OQMD_API_KEY to enable cross-database comparison",
                },
                success=False,
                error="No database API keys configured for comparison",
            )

        return ToolResult(
            data={
                "comparison": results,
                "databases_queried": list(results.keys()),
                "query_formula": args.formula,
                "query_material_id": args.material_id,
            },
            success=True,
        )

    # ── Mock fallback ──────────────────────────────────────────────

    def _mock_result(self, args: DatabaseToolInput, reason: str) -> ToolResult:
        mock_data: dict[str, Any] = {
            "database": args.database,
            "query_type": args.query_type,
            "mock": True,
            "reason": reason,
        }
        if args.query_type == "search":
            mock_data["results"] = [{
                "material_id": f"{args.database.upper()}-MOCK",
                "formula": args.formula,
                "spacegroup": "Fm-3m",
            }]
        elif args.query_type == "get_structure":
            mock_data["lattice"] = {"a": 4.2, "b": 4.2, "c": 4.2}
        elif args.query_type == "get_properties":
            mock_data["properties"] = {
                "band_gap": 1.5,
                "formation_energy": -2.3,
                "bulk_modulus": 150,
            }
        return ToolResult(
            data=mock_data,
            success=True,
        )
