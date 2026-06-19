"""Materials database tool — query Materials Project and OQMD.

A read-only tool for retrieving structures, thermodynamic data, and
properties from public materials databases. Requires user-supplied API keys
or environment variables (MP_API_KEY / OQMD_API_KEY).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlencode

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult


class MaterialsDatabaseInput(BaseModel):
    action: Literal[
        "mp_summary",
        "mp_structure",
        "oqmd_query",
        "oqmd_structure",
    ] = Field(..., description="Database action to perform")
    query: str | None = Field(
        default=None,
        description="Formula (e.g. 'SiO2'), material_id (e.g. 'mp-149'), or OQMD filter",
    )
    fields: list[str] | None = Field(
        default=None,
        description="Fields to return. Default depends on action.",
    )
    limit: int = Field(default=10, ge=1, le=100, description="Max records to return")
    output_format: Literal["json", "cif", "poscar"] | None = Field(
        default=None,
        description="If set, save the structure to workspace in this format",
    )
    output_file: str | None = Field(
        default=None,
        description="Filename for saved structure (default auto-generated)",
    )
    api_key: str | None = Field(
        default=None,
        description="API key override. Falls back to MP_API_KEY / OQMD_API_KEY env vars.",
    )


@dataclass
class MaterialsRecord:
    id: str
    formula: str | None = None
    energy_per_atom: float | None = None
    band_gap: float | None = None
    spacegroup: str | None = None
    source: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


class MaterialsDatabaseOutput(BaseModel):
    source: str
    count: int
    records: list[dict[str, Any]]
    saved_file: str | None = None
    warnings: list[str] = []


class MaterialsDatabaseTool(HuginnTool):
    """Query public materials databases (Materials Project, OQMD)."""

    name = "materials_database_tool"
    description = (
        "Query Materials Project or OQMD for structures, energies, band gaps, "
        "and thermodynamic data. Provide an API key or set MP_API_KEY / OQMD_API_KEY."
    )
    input_schema = MaterialsDatabaseInput
    output_schema = MaterialsDatabaseOutput
    read_only = True

    def __init__(self, mp_api_key: str | None = None, oqmd_api_key: str | None = None):
        self._config_mp_key = mp_api_key
        self._config_oqmd_key = oqmd_api_key

    def is_read_only(self, args: MaterialsDatabaseInput) -> bool:
        return True

    async def call(
        self, args: MaterialsDatabaseInput, context: ToolContext
    ) -> ToolResult:
        try:
            if args.action.startswith("mp_"):
                return await self._handle_mp(args, context)
            return await self._handle_oqmd(args, context)
        except (
            Exception
        ) as exc:  # pragma: no cover - broad catch for user-facing errors
            return ToolResult(data=None, success=False, error=str(exc))

    def _mp_key(self, override: str | None) -> str:
        key = override or self._config_mp_key or os.environ.get("MP_API_KEY")
        if not key:
            raise ValueError(
                "Materials Project API key is required. "
                "Pass api_key, set MP_API_KEY, or configure mp_api_key."
            )
        return key

    def _oqmd_key(self, override: str | None) -> str | None:
        return (
            override or self._config_oqmd_key or os.environ.get("OQMD_API_KEY") or None
        )

    async def _handle_mp(
        self, args: MaterialsDatabaseInput, context: ToolContext
    ) -> ToolResult:
        import aiohttp

        api_key = self._mp_key(args.api_key)
        base_url = "https://api.materialsproject.org"

        async with aiohttp.ClientSession() as session:
            if args.action == "mp_summary":
                records, warnings = await self._mp_summary(
                    session, base_url, api_key, args
                )
            elif args.action == "mp_structure":
                records, warnings, saved = await self._mp_structure(
                    session, base_url, api_key, args, context
                )
                output = MaterialsDatabaseOutput(
                    source="materials_project",
                    count=len(records),
                    records=records,
                    saved_file=saved,
                    warnings=warnings,
                )
                return ToolResult(data=output.model_dump(exclude_none=True))
            else:  # pragma: no cover
                raise ValueError(f"Unknown action: {args.action}")

        output = MaterialsDatabaseOutput(
            source="materials_project",
            count=len(records),
            records=records,
            warnings=warnings,
        )
        return ToolResult(data=output.model_dump(exclude_none=True))

    async def _mp_summary(
        self,
        session: Any,
        base_url: str,
        api_key: str,
        args: MaterialsDatabaseInput,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        warnings: list[str] = []
        params: dict[str, Any] = {"limit": args.limit, "API_KEY": api_key}
        if args.query:
            # Heuristic: mp-NNN is a material id, otherwise treat as formula.
            if args.query.lower().startswith("mp-"):
                params["material_ids"] = args.query
            else:
                params["formula"] = args.query
        fields = args.fields or [
            "material_id",
            "formula_pretty",
            "energy_per_atom",
            "band_gap",
            "symmetry",
        ]
        params["fields"] = ",".join(fields)
        url = f"{base_url}/materials/summary/?{urlencode(params, doseq=True)}"
        data = await self._get_json(session, url)
        records = []
        for item in data.get("data", []):
            records.append(self._normalize_mp_summary(item))
        if not records and data.get("meta", {}).get("total", 0) == 0:
            warnings.append(f"No Materials Project results for query: {args.query}")
        return records, warnings

    async def _mp_structure(
        self,
        session: Any,
        base_url: str,
        api_key: str,
        args: MaterialsDatabaseInput,
        context: ToolContext,
    ) -> tuple[list[dict[str, Any]], list[str], str | None]:
        warnings: list[str] = []
        material_id = (args.query or "").strip()
        if not material_id:
            raise ValueError("mp_structure requires a material_id query (e.g. mp-149)")
        params: dict[str, Any] = {
            "API_KEY": api_key,
            "fields": "structure,material_id,formula_pretty",
        }
        url = f"{base_url}/materials/core/{material_id}?{urlencode(params)}"
        data = await self._get_json(session, url)
        item = (
            data.get("data", [None])[0]
            if isinstance(data.get("data"), list)
            else data.get("data")
        )
        if item is None:
            warnings.append(f"No structure found for {material_id}")
            return [], warnings, None

        record = self._normalize_mp_summary(item)
        saved = None
        if args.output_format:
            saved = self._save_structure(
                item.get("structure"),
                material_id,
                args.output_format,
                args.output_file,
                context,
            )
        return [record], warnings, saved

    async def _handle_oqmd(
        self, args: MaterialsDatabaseInput, context: ToolContext
    ) -> ToolResult:
        import aiohttp

        api_key = self._oqmd_key(args.api_key)
        base_url = "http://oqmd.org/oqmdapi"
        headers = {"X-API-KEY": api_key} if api_key else {}

        async with aiohttp.ClientSession(headers=headers) as session:
            if args.action == "oqmd_query":
                records, warnings = await self._oqmd_query(session, base_url, args)
            elif args.action == "oqmd_structure":
                records, warnings, saved = await self._oqmd_structure(
                    session, base_url, args, context
                )
                output = MaterialsDatabaseOutput(
                    source="oqmd",
                    count=len(records),
                    records=records,
                    saved_file=saved,
                    warnings=warnings,
                )
                return ToolResult(data=output.model_dump(exclude_none=True))
            else:  # pragma: no cover
                raise ValueError(f"Unknown action: {args.action}")

        output = MaterialsDatabaseOutput(
            source="oqmd", count=len(records), records=records, warnings=warnings
        )
        return ToolResult(data=output.model_dump(exclude_none=True))

    async def _oqmd_query(
        self,
        session: Any,
        base_url: str,
        args: MaterialsDatabaseInput,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        warnings: list[str] = []
        params: dict[str, Any] = {"limit": args.limit}
        if args.query:
            # Treat plain formula as composition filter.
            if "=" in args.query or "<" in args.query or ">" in args.query:
                params["filter"] = args.query
            else:
                params["composition"] = args.query
        fields = args.fields or ["name", "entry_id", "icsd_id", "band_gap", "delta_e"]
        params["fields"] = ",".join(fields)
        url = f"{base_url}/entry?{urlencode(params, doseq=True)}"
        data = await self._get_json(session, url)
        records = []
        for item in data.get("data", []):
            records.append(self._normalize_oqmd_entry(item))
        if not records:
            warnings.append(f"No OQMD results for query: {args.query}")
        return records, warnings

    async def _oqmd_structure(
        self,
        session: Any,
        base_url: str,
        args: MaterialsDatabaseInput,
        context: ToolContext,
    ) -> tuple[list[dict[str, Any]], list[str], str | None]:
        warnings: list[str] = []
        entry_id = (args.query or "").strip()
        if not entry_id:
            raise ValueError("oqmd_structure requires an entry_id query")
        url = f"{base_url}/entry/{entry_id}?fields=structure,name,entry_id,band_gap,delta_e"
        data = await self._get_json(session, url)
        item = (
            data.get("data", [None])[0]
            if isinstance(data.get("data"), list)
            else data.get("data")
        )
        if item is None:
            warnings.append(f"No OQMD structure found for {entry_id}")
            return [], warnings, None
        record = self._normalize_oqmd_entry(item)
        saved = None
        if args.output_format:
            saved = self._save_structure(
                item.get("structure"),
                entry_id,
                args.output_format,
                args.output_file,
                context,
            )
        return [record], warnings, saved

    async def _get_json(self, session: Any, url: str) -> dict[str, Any]:
        import aiohttp

        async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            text = await resp.text()
            if resp.status != 200:
                raise RuntimeError(
                    f"Database request failed ({resp.status}): {text[:500]}"
                )
            try:
                return json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON response from database: {exc}"
                ) from exc

    def _normalize_mp_summary(self, item: dict[str, Any]) -> dict[str, Any]:
        symmetry = item.get("symmetry") or {}
        record = {
            "id": item.get("material_id") or item.get("task_id"),
            "formula": item.get("formula_pretty") or item.get("formula"),
            "energy_per_atom": item.get("energy_per_atom"),
            "band_gap": item.get("band_gap"),
            "spacegroup": (
                symmetry.get("symbol") if isinstance(symmetry, dict) else None
            ),
            "source": "materials_project",
        }
        record.update({k: v for k, v in item.items() if k not in record})
        return record

    def _normalize_oqmd_entry(self, item: dict[str, Any]) -> dict[str, Any]:
        record = {
            "id": item.get("entry_id") or item.get("id"),
            "formula": item.get("name"),
            "energy_per_atom": item.get("delta_e"),
            "band_gap": item.get("band_gap"),
            "source": "oqmd",
        }
        record.update({k: v for k, v in item.items() if k not in record})
        return record

    def _save_structure(
        self,
        structure_data: dict[str, Any] | None,
        record_id: str,
        fmt: Literal["json", "cif", "poscar"],
        output_file: str | None,
        context: ToolContext,
    ) -> str | None:
        if not structure_data:
            return None
        workspace = Path(context.workspace).expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        filename = output_file or f"{record_id}.{fmt}"
        path = workspace / filename

        if fmt == "json":
            path.write_text(json.dumps(structure_data, indent=2), encoding="utf-8")
            return str(path)

        # Try pymatgen for CIF/POSCAR conversion; fall back to JSON if unavailable.
        try:
            from pymatgen.core import Structure

            structure = Structure.from_dict(structure_data)
            if fmt == "cif":
                structure.to(fmt="cif", filename=str(path))
            elif fmt == "poscar":
                structure.to(fmt="poscar", filename=str(path))
            return str(path)
        except Exception:
            fallback = path.with_suffix(".json")
            fallback.write_text(json.dumps(structure_data, indent=2), encoding="utf-8")
            return str(fallback)
