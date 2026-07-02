"""Materials database query tool — search and retrieve data from MP, AFLOW, NOMAD, OQMD.

Read-only tool. Safe to auto-execute.
Supports real API calls to Materials Project and OQMD when API keys are available;
falls back to structured mock data for AFLOW/NOMAD or when keys are missing.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Literal
from urllib.parse import urlencode

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile
from huginn.types import ToolContext, ToolResult


def _formula_to_elements(formula: str) -> list[str]:
    """粗略从化学式抽元素符号, 给 NOMAD query DSL 用. 'SiO2' -> ['Si','O']."""
    import re

    # 匹配大写开头 + 可选小写的元素符号
    symbols = re.findall(r"[A-Z][a-z]?", formula)
    return symbols


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
    profile = ToolProfile(phases=frozenset({ResearchPhase.LITERATURE, ResearchPhase.HYPOTHESIS}))
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

    # ── AFLOW ───────────────────────────────────────────────────────
    # AFLOW REST (aflowlib.duke.edu) 是公开的, 不需要 key.

    async def _query_aflow(
        self, args: DatabaseToolInput, api_key: str | None
    ) -> ToolResult:
        try:
            import aiohttp
        except ImportError:
            return self._mock_result(args, "aiohttp not installed")

        # AFLOW 的 query 语法: composition(化学式) 或 catalog(材料 id)
        target = args.formula or args.material_id
        if not target:
            return ToolResult(
                data=None, success=False, error="formula or material_id required for AFLOW"
            )

        # 公开端点, 不需要 key
        base = "http://aflowlib.duke.edu/aflowlib"
        params: dict[str, Any] = {"limit": args.limit}
        if target.lower().startswith("aflow:"):
            params["aflow"] = target
        else:
            params["composition"] = target
        url = f"{base}?{urlencode(params, doseq=True)}"

        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        return ToolResult(
                            data=None,
                            success=False,
                            error=f"AFLOW API error {resp.status}: {text[:200]}",
                        )
                    # AFLOW 返回 JSON (新版本) 或 AFLUX 文本 (老版本), 都试一下
                    try:
                        raw = await resp.json(content_type=None)
                    except Exception:
                        raw = await resp.text()
        except asyncio.TimeoutError:
            return ToolResult(
                data=None, success=False, error="AFLOW request timed out (30s)"
            )
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"AFLOW request failed: {e}")

        records = self._normalize_aflow(raw)
        return ToolResult(
            data={
                "database": "aflow",
                "query_type": args.query_type,
                "count": len(records),
                "records": records,
                "source": "aflow",
            },
            success=True,
        )

    def _normalize_aflow(self, raw: Any) -> list[dict[str, Any]]:
        """把 AFLOW 异构响应归一成 records. AFLOW 可能返回 list/dict/text."""
        records: list[dict[str, Any]] = []
        if isinstance(raw, str):
            # 老版 AFLUX 文本: 逐行解析 "key=value"
            for block in raw.split(">>>"):
                rec: dict[str, Any] = {}
                for line in block.strip().splitlines():
                    if "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip()
                        if k in ("compound", "formula"):
                            rec["formula"] = v
                        elif k in ("aflow_id", "auid"):
                            rec["entry_id"] = v
                        elif k in ("spacegroup", "sg"):
                            rec["spacegroup"] = v
                        elif k == "Egap":
                            try:
                                rec["band_gap"] = float(v)
                            except ValueError:
                                rec["band_gap"] = v
                        elif k in ("enthalpy", "enthalpy_atom"):
                            try:
                                rec["energy"] = float(v)
                            except ValueError:
                                rec["energy"] = v
                if rec:
                    rec["source"] = "aflow"
                    records.append(rec)
        elif isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                records.append({
                    "formula": item.get("compound") or item.get("formula"),
                    "entry_id": item.get("aflow_id") or item.get("auid"),
                    "spacegroup": item.get("spacegroup") or item.get("sg"),
                    "band_gap": item.get("Egap"),
                    "energy": item.get("enthalpy") or item.get("enthalpy_atom"),
                    "source": "aflow",
                })
        elif isinstance(raw, dict) and "data" in raw:
            for item in raw["data"]:
                records.append({
                    "formula": item.get("compound") or item.get("formula"),
                    "entry_id": item.get("aflow_id") or item.get("auid"),
                    "spacegroup": item.get("spacegroup") or item.get("sg"),
                    "band_gap": item.get("Egap"),
                    "energy": item.get("enthalpy") or item.get("enthalpy_atom"),
                    "source": "aflow",
                })
        return records[:50]  # 防爆

    # ── NOMAD ───────────────────────────────────────────────────────
    # NOMAD 公开数据不需要 key, key 只用于私有上传.

    async def _query_nomad(
        self, args: DatabaseToolInput, api_key: str | None
    ) -> ToolResult:
        try:
            import aiohttp
        except ImportError:
            return self._mock_result(args, "aiohttp not installed")

        target = args.formula or args.material_id
        if not target:
            return ToolResult(
                data=None, success=False, error="formula or material_id required for NOMAD"
            )

        base = "https://nomad-lab.eu/prod-1/api/v1/entries/query"
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # NOMAD 用 ES query DSL, 公开数据 owner="public"
        body: dict[str, Any] = {
            "owner": "public",
            "pagination": {"page_size": min(args.limit, 50)},
            "query": {
                "results.material.elements:all": _formula_to_elements(target),
            },
        }
        # 也按化学式模糊匹配
        body["query"]["results.material.chemical_formula_descriptive:contains"] = target

        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                async with session.post(base, json=body) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        return ToolResult(
                            data=None,
                            success=False,
                            error=f"NOMAD API error {resp.status}: {text[:200]}",
                        )
                    raw = await resp.json()
        except asyncio.TimeoutError:
            return ToolResult(
                data=None, success=False, error="NOMAD request timed out (30s)"
            )
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"NOMAD request failed: {e}")

        records = self._normalize_nomad(raw)
        return ToolResult(
            data={
                "database": "nomad",
                "query_type": args.query_type,
                "count": len(records),
                "records": records,
                "source": "nomad",
            },
            success=True,
        )

    def _normalize_nomad(self, raw: Any) -> list[dict[str, Any]]:
        """把 NOMAD /entries/query 响应归一成 records."""
        records: list[dict[str, Any]] = []
        if not isinstance(raw, dict):
            return records
        for item in raw.get("data", []):
            entry_id = item.get("entry_id") or item.get("upload_id")
            results = item.get("results", {})
            material = (results.get("material") or [{}])
            mat = material[0] if isinstance(material, list) and material else material
            formula = (
                mat.get("chemical_formula_descriptive")
                or mat.get("chemical_formula_reduced")
            )
            props = results.get("properties", {})
            records.append({
                "entry_id": entry_id,
                "formula": formula,
                "spacegroup": (mat.get("structure") or {}).get("space_group"),
                "band_gap": props.get("electronic", {}).get("band_gap"),
                "energy": props.get("energetic", {}).get("total_energy"),
                "source": "nomad",
            })
        return records[:50]

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
