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

from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile
from huginn.tools.local_structure_db import LocalStructureDB
from huginn.tools.tool_cache import EXTERNAL_API_TTL, cacheable
from huginn.types import ToolContext, ToolResult


class MaterialsDatabaseInput(BaseModel):
    action: Literal[
        "mp_summary",
        "mp_structure",
        "oqmd_query",
        "oqmd_structure",
        "batch_query",
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
    # batch_query 专用: 一次查多个 mp_id 或化学式, 内部并发受 _BATCH_CONCURRENCY 限制
    mp_ids: list[str] | None = Field(
        default=None,
        description="For batch_query: list of material_ids (e.g. ['mp-149', 'mp-13'])",
    )
    formulas: list[str] | None = Field(
        default=None,
        description="For batch_query: list of formulas (e.g. ['SiO2', 'TiO2'])",
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


# batch_query 内部并发查 MP API 的最大并发数, MP 公开 API 限流比较紧,
# 给 5 比较稳妥. 超过这个数的查询自动排队, 不会一次性把 API 打爆.
_BATCH_CONCURRENCY = 5


class MaterialsDatabaseTool(HuginnTool):
    """Query public materials databases (Materials Project, OQMD)."""

    name = "materials_database_tool"
    category = "materials"
    profile = ToolProfile(
        cost_tier="light",
        phases=frozenset({ResearchPhase.LITERATURE, ResearchPhase.HYPOTHESIS}),
    )
    description = (
        "Query Materials Project or OQMD for structures, energies, band gaps, "
        "and thermodynamic data. Provide an API key or set MP_API_KEY / OQMD_API_KEY."
    )
    input_schema = MaterialsDatabaseInput
    output_schema = MaterialsDatabaseOutput
    read_only = True
    _init_kwargs_map = {"mp_api_key": "mp_api_key", "oqmd_api_key": "oqmd_api_key"}

    def __init__(self, mp_api_key: str | None = None, oqmd_api_key: str | None = None):
        self._config_mp_key = mp_api_key
        self._config_oqmd_key = oqmd_api_key

    def is_read_only(self, args: MaterialsDatabaseInput) -> bool:
        return True

    async def call(
        self, args: MaterialsDatabaseInput, context: ToolContext
    ) -> ToolResult:
        return await self._call_cached(args, context)

    @cacheable(
        ttl_seconds=EXTERNAL_API_TTL,
        tool_name="materials_database_tool",
        # api_key 不进 key（敏感且不影响查询结果），output_format 也不进
        # （落盘路径每次可能不同，但查到的数据是一样的）
        # batch_query 把 mp_ids/formulas 拼进 key, 否则不同批次的查询会撞缓存
        key_fn=lambda self, args, ctx: {
            "action": args.action,
            "query": args.query,
            "fields": args.fields,
            "limit": args.limit,
            "mp_ids": args.mp_ids,
            "formulas": args.formulas,
        },
    )
    async def _call_cached(
        self, args: MaterialsDatabaseInput, context: ToolContext
    ) -> ToolResult:
        # batch_query 走单独路径, 内部并发查多个 mp_id/formula
        if args.action == "batch_query":
            return await self._handle_batch_query(args, context)
        # 先查本地结构库，命中就省掉一次外部 API 往返
        local_hit = self._local_lookup(args)
        if local_hit is not None:
            return local_hit
        try:
            if args.action.startswith("mp_"):
                return await self._handle_mp(args, context)
            return await self._handle_oqmd(args, context)
        except (
            Exception
        ) as exc:  # pragma: no cover - broad catch for user-facing errors
            return ToolResult(data=None, success=False, error=str(exc))

    def _local_lookup(self, args: MaterialsDatabaseInput) -> ToolResult | None:
        """查本地结构库，命中返回 ToolResult，没命中返回 None。

        只对 mp_summary / mp_structure 这两个读操作做本地短路，
        OQMD 的数据格式不一样，不走本地库。
        """
        if args.action not in ("mp_summary", "mp_structure"):
            return None
        if not args.query:
            return None
        struct = LocalStructureDB.shared().get(args.query)
        if struct is None:
            return None
        record = {
            "id": struct.get("mp_id"),
            "formula": struct.get("formula_pretty") or struct.get("formula"),
            "energy_per_atom": None,
            "band_gap": struct.get("band_gap"),
            "spacegroup": struct.get("space_group"),
            "source": f"local_db ({struct.get('mp_id')})",
            "lattice_params": struct.get("lattice_params"),
            "atomic_positions": struct.get("atomic_positions"),
            "density": struct.get("density"),
            "volume": struct.get("volume"),
        }
        output = MaterialsDatabaseOutput(
            source="local_db",
            count=1,
            records=[record],
            warnings=[f"from local structure db, not live API: {args.query}"],
        )
        return ToolResult(data=output.model_dump(exclude_none=True))

    async def _handle_batch_query(
        self, args: MaterialsDatabaseInput, context: ToolContext
    ) -> ToolResult:
        """批量查多个 mp_id / formula, 内部并发受 _BATCH_CONCURRENCY 限制.

        每个条目单独走 _call_cached 路径, 这样:
          - 单条命中本地结构库就直接短路, 不打 API
          - 单条命中 tool_cache 就直接返回, 重复 batch_query 不烧 API
          - 没命中的条目走 mp_summary 拿 MP API

        单条失败不影响其它, 错误塞进该条的 record["error"] 返回.
        """
        items: list[str] = []
        items.extend(args.mp_ids or [])
        items.extend(args.formulas or [])
        # 去重保序, 避免 LLM 把同一个 mp_id 写两遍白跑一次
        seen: set[str] = set()
        unique_items: list[str] = []
        for it in items:
            if it not in seen:
                seen.add(it)
                unique_items.append(it)

        if not unique_items:
            return ToolResult(
                data=None,
                success=False,
                error="batch_query requires mp_ids or formulas (non-empty)",
            )

        import asyncio

        sem = asyncio.Semaphore(_BATCH_CONCURRENCY)

        async def _query_one(item: str) -> dict[str, Any]:
            """单条查询, 走 _call_cached 复用本地缓存 + tool_cache."""
            single_args = MaterialsDatabaseInput(
                action="mp_summary",
                query=item,
                fields=args.fields,
                limit=args.limit,
                api_key=args.api_key,
            )
            async with sem:
                try:
                    result = await self._call_cached(single_args, context)
                except Exception as exc:
                    return {
                        "query": item,
                        "error": str(exc),
                        "records": [],
                    }
            if not result.success:
                return {
                    "query": item,
                    "error": result.error or "unknown error",
                    "records": [],
                }
            data = result.data or {}
            records = data.get("records", []) if isinstance(data, dict) else []
            return {
                "query": item,
                "records": records,
                "source": data.get("source") if isinstance(data, dict) else None,
                "warnings": data.get("warnings", []) if isinstance(data, dict) else [],
            }

        # 并发跑所有单条查询, 单条挂掉不影响其它
        per_item = await asyncio.gather(
            *[_query_one(it) for it in unique_items]
        )

        # 把单条结果摊平成 records 列表, 方便上层统一处理
        all_records: list[dict[str, Any]] = []
        all_warnings: list[str] = []
        for entry in per_item:
            if entry.get("error"):
                all_warnings.append(f"{entry['query']}: {entry['error']}")
                # 错误条目也进 records, 让 LLM 知道哪条挂了
                all_records.append({
                    "query": entry["query"],
                    "error": entry["error"],
                })
                continue
            for rec in entry.get("records", []):
                rec_with_query = dict(rec)
                rec_with_query.setdefault("query", entry["query"])
                all_records.append(rec_with_query)
            for w in entry.get("warnings", []):
                all_warnings.append(f"{entry['query']}: {w}")

        output = MaterialsDatabaseOutput(
            source="materials_project_batch",
            count=len(all_records),
            records=all_records,
            warnings=all_warnings,
        )
        return ToolResult(data=output.model_dump(exclude_none=True))

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
