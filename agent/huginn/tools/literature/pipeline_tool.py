"""MinerU 文献流水线工具 (huginn 集成版).

3 个 action:
  - parse_pdf(pdf_path) -> doc_id    调 MinerU API 解析, 入 KB
  - extract_schema(doc_id) -> schema  调 schema_extractor 抽 6 块
  - aggregate_entities(doc_ids) -> agg_result  调 aggregator 跨文献聚合

设计:
  - mineru_api_keys 未配置时 parse_pdf 返回明确错误, agent 自动退回 PyMuPDF+OCR
    (huginn SmartIngester 已有该路径, 这里不重复实现)
  - 复用 LiteratureTool 的 _get_model/_llm_invoke/_parse_json 静态方法 (同包 import)
  - state 用文件路径作 doc_id (stem), 不维护额外状态表
  - KB 入库通过 kb.add_text, 复用现有 ChromaDB 混合检索 + 领域打标

ponytail: 不重写 KB, 不新建状态存储, 不引新依赖.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult

from .aggregator import aggregate, ingest_into_kg
from .mineru_client import (
    FileSpec,
    download_and_extract,
    parsed_dir_for,
    submit_batch,
    wait_batch,
)
from .schema_def import MaterialSchema
from .schema_extractor import extract_from_parsed_dir, parse_document

logger = logging.getLogger(__name__)


# ── 输入 schema ───────────────────────────────────────────────────────────────

class LiteraturePipelineInput(BaseModel):
    action: Literal["parse_pdf", "extract_schema", "aggregate_entities"] = Field(
        ...,
        description="parse_pdf: MinerU VLM 解析 PDF, 入 KB; "
                    "extract_schema: 6 块 Schema 抽取 (Material/Structure/Properties/Synthesis/Application/Metadata); "
                    "aggregate_entities: 跨文献实体聚合 (Union-Find 三级匹配 + KG 集成)",
    )
    pdf_path: str = Field(default="", description="parse_pdf: 本地 PDF 绝对路径")
    doc_id: str = Field(default="", description="extract_schema: 已解析文档 id (parse_pdf 返回值)")
    doc_ids: list[str] = Field(default_factory=list, description="aggregate_entities: 文档 id 列表")
    model_alias: str = Field(
        default="",
        description="extract_schema 用的 LLM alias (空则走 registry.default_alias)",
    )
    ingest_to_kg: bool = Field(
        default=True,
        description="aggregate_entities 完成后是否自动喂给 huginn ProjectKnowledgeGraph",
    )


# ── Tool 实现 ─────────────────────────────────────────────────────────────────

class LiteraturePipelineTool(HuginnTool):
    """MinerU 文献流水线: PDF → 解析 → Schema 抽取 → 跨文献聚合 → KG."""

    name = "literature_pipeline_tool"
    category = "materials"
    description = (
        "MinerU VLM-powered materials literature pipeline: "
        "(1) parse_pdf — submit PDF to MinerU API for VLM layout parsing, ingest full text into local KB; "
        "(2) extract_schema — extract 6-block schema (Material/Structure/Properties/Synthesis/Application/Metadata) "
        "via LLM with 5-level original-text verification; "
        "(3) aggregate_entities — cross-document entity merge via Union-Find (CAS/chemical-name/fuzzy) + "
        "auto-ingest into huginn ProjectKnowledgeGraph. "
        "Requires MINERU_API_KEYS env. Falls back to PyMuPDF+OCR if unconfigured."
    )
    input_schema = LiteraturePipelineInput
    read_only = False  # parse_pdf 写 KB, aggregate 写 KG

    async def call(self, args: LiteraturePipelineInput, context: ToolContext) -> ToolResult:
        try:
            if isinstance(args, dict):
                args = LiteraturePipelineInput(**args)
            if args.action == "parse_pdf":
                return await self._do_parse_pdf(args, context)
            if args.action == "extract_schema":
                return await self._do_extract_schema(args, context)
            if args.action == "aggregate_entities":
                return await self._do_aggregate(args, context)
            return ToolResult(data=None, success=False, error=f"unknown action: {args.action}")
        except Exception as exc:
            logger.exception("literature_pipeline_tool %s failed", getattr(args, "action", "unknown"))
            return ToolResult(data=None, success=False, error=str(exc))

    # ── 共享 helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _parsed_root(context: ToolContext) -> Path:
        """data/parsed 目录: {workspace}/data/parsed."""
        ws = getattr(context, "workspace", None) or "."
        return Path(ws) / "data" / "parsed"

    @staticmethod
    def _aggregated_root(context: ToolContext) -> Path:
        """data/aggregated 目录."""
        ws = getattr(context, "workspace", None) or "."
        return Path(ws) / "data" / "aggregated"

    @staticmethod
    def _extracted_root(context: ToolContext) -> Path:
        """data/extracted 目录 (Schema 抽取结果落盘)."""
        ws = getattr(context, "workspace", None) or "."
        return Path(ws) / "data" / "extracted"

    @staticmethod
    def _get_mineru_keys(context: ToolContext) -> list[str]:
        """从 config 拿 mineru_api_keys. 空表示未配置."""
        cfg = getattr(context, "config", None)
        if cfg is None:
            return []
        return list(getattr(cfg, "mineru_api_keys", []) or [])

    @staticmethod
    def _get_model_for_extract(context: ToolContext, alias: str = "") -> Any:
        """拿 LLM 实例 (走 ModelRegistry, 复用 schema_extractor.resolve_model)."""
        from huginn.models.registry import ModelRegistry
        cfg = getattr(context, "config", None)
        if cfg is None:
            from huginn.config import HuginnConfig
            cfg = HuginnConfig.from_env()
        registry = ModelRegistry.from_config(cfg)
        return LiteraturePipelineTool._resolve_model(registry, alias)

    @staticmethod
    def _resolve_model(registry: Any, alias: str) -> Any:
        """alias 为空走 default_alias."""
        a = alias or registry.default_alias()
        if not a:
            raise ValueError("registry 无可用 alias, 请配置 models 或显式传 model_alias")
        return registry.get(a)

    # ── action 1: parse_pdf ──────────────────────────────────────────────

    async def _do_parse_pdf(self, args: LiteraturePipelineInput, context: ToolContext) -> ToolResult:
        pdf_path_str = (args.pdf_path or "").strip()
        if not pdf_path_str:
            return ToolResult(data=None, success=False, error="pdf_path is required")

        # API key 检查必须在文件存在性检查之前 — 未配置时 agent 应直接退回 PyMuPDF+OCR,
        # 而不是先抱怨文件不存在 (那会误导调用方以为路径问题).
        keys = self._get_mineru_keys(context)
        if not keys:
            return ToolResult(
                data={
                    "error": "MinerU API key 未配置",
                    "hint": "设置 MINERU_API_KEYS 环境变量或 HuginnConfig.mineru_api_keys. "
                            "未配置时 agent 自动退回 PyMuPDF + OCR 路径 (SmartIngester).",
                    "fallback": "pymupdf_ocr",
                },
                success=False,
                error="MinerU API key 未配置",
            )

        pdf_path = Path(pdf_path_str)
        if not pdf_path.exists():
            return ToolResult(data=None, success=False, error=f"PDF not found: {pdf_path}")

        # doc_id 用文件名 stem
        doc_id = pdf_path.stem
        parsed_root = self._parsed_root(context)
        target_dir = parsed_dir_for(doc_id, parsed_root)

        # 已解析过则跳过 MinerU API 调用 (idempotent)
        if (target_dir / "content_list.json").exists():
            logger.info("parse_pdf cache hit: %s", doc_id)
        else:
            # 调 MinerU API (阻塞, 放到 thread 避免阻塞 event loop)
            spec = FileSpec(path=pdf_path, data_id=doc_id, is_ocr=False)
            try:
                batch = await asyncio.to_thread(
                    submit_batch, [spec], api_keys=keys, model_version="vlm",
                )
                results = await asyncio.to_thread(
                    wait_batch, batch.batch_id, api_keys=keys,
                )
            except Exception as exc:
                return ToolResult(
                    data={"error": f"MinerU API 调用失败: {exc}", "doc_id": doc_id},
                    success=False, error=str(exc),
                )

            # 找到 done 的结果, 下载 zip
            ok = next((r for r in results if r.get("state") == "done" and r.get("full_zip_url")), None)
            if not ok:
                failed = [r for r in results if r.get("state") == "failed"]
                err_msg = failed[0].get("err_msg", "unknown") if failed else "no done result"
                return ToolResult(
                    data={"error": f"MinerU 解析失败: {err_msg}", "doc_id": doc_id,
                          "results": results},
                    success=False, error=err_msg,
                )
            try:
                await asyncio.to_thread(download_and_extract, ok["full_zip_url"], target_dir)
            except Exception as exc:
                return ToolResult(
                    data={"error": f"MinerU zip 下载/解压失败: {exc}", "doc_id": doc_id},
                    success=False, error=str(exc),
                )

        # 入 KB: 读 content_list.json 拼全文, 通过 kb.add_text 入 ChromaDB
        kb_written = 0
        try:
            from huginn.knowledge.store import get_knowledge_base
            doc = parse_document(target_dir)
            kb = get_knowledge_base()
            if doc.full_text:
                meta = {
                    "doc_id": doc_id,
                    "source": "mineru_pipeline",
                    "abstract": doc.abstract or "",
                }
                kb.add_text(doc.full_text, filename=doc_id, metadata=meta)
                kb_written = 1
        except Exception as exc:
            logger.warning("parse_pdf KB 入库失败 (doc_id=%s): %s", doc_id, exc)

        return ToolResult(
            data={
                "doc_id": doc_id,
                "parsed_dir": str(target_dir),
                "kb_written": kb_written,
            },
            success=True,
        )

    # ── action 2: extract_schema ─────────────────────────────────────────

    async def _do_extract_schema(self, args: LiteraturePipelineInput, context: ToolContext) -> ToolResult:
        doc_id = (args.doc_id or "").strip()
        if not doc_id:
            return ToolResult(data=None, success=False, error="doc_id is required")
        target_dir = parsed_dir_for(doc_id, self._parsed_root(context))
        if not (target_dir / "content_list.json").exists():
            return ToolResult(
                data=None, success=False,
                error=f"doc_id {doc_id} 未解析, 请先调用 parse_pdf",
            )

        try:
            model = self._get_model_for_extract(context, args.model_alias)
        except Exception as exc:
            return ToolResult(
                data=None, success=False,
                error=f"LLM 初始化失败: {exc}",
            )

        # 调 schema_extractor (同步, 放到 thread)
        try:
            schema: MaterialSchema = await asyncio.to_thread(
                extract_from_parsed_dir, target_dir, model, True,
            )
        except Exception as exc:
            return ToolResult(
                data=None, success=False,
                error=f"Schema 抽取失败: {exc}",
            )

        schema_dict = schema.to_dict()

        # 落盘到 data/extracted/{doc_id}.json (供 aggregate_entities 读)
        try:
            extracted_dir = self._extracted_root(context)
            extracted_dir.mkdir(parents=True, exist_ok=True)
            (extracted_dir / f"{doc_id}.json").write_text(
                json.dumps(schema_dict, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Schema 落盘失败 (doc_id=%s): %s", doc_id, exc)

        # 校验报告: 非法 PropertyCategory + invalid_original_texts 已在 extract 内部处理
        invalid_cats = schema.invalid_property_categories()

        return ToolResult(
            data={
                "doc_id": doc_id,
                "schema": schema_dict,
                "invalid_property_categories": invalid_cats,
            },
            success=True,
        )

    # ── action 3: aggregate_entities ─────────────────────────────────────

    async def _do_aggregate(self, args: LiteraturePipelineInput, context: ToolContext) -> ToolResult:
        doc_ids = args.doc_ids or []
        if not doc_ids:
            return ToolResult(data=None, success=False, error="doc_ids is required (non-empty list)")

        # 读所有 doc_id 的 schema_dict
        extracted_dir = self._extracted_root(context)
        extracted: dict[str, dict] = {}
        missing: list[str] = []
        for did in doc_ids:
            p = extracted_dir / f"{did}.json"
            if not p.exists():
                missing.append(did)
                continue
            try:
                extracted[did] = json.loads(p.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("读取 %s 失败: %s", p, exc)
                missing.append(did)

        if not extracted:
            return ToolResult(
                data=None, success=False,
                error=f"无可用 schema (doc_ids={doc_ids}, missing={missing})",
            )

        # 调 aggregator (同步, 放到 thread)
        agg_dir = self._aggregated_root(context)
        try:
            result = await asyncio.to_thread(aggregate, extracted, agg_dir)
        except Exception as exc:
            return ToolResult(data=None, success=False, error=f"聚合失败: {exc}")

        # 可选: 自动喂给 huginn KG
        kg_stats = {"entities_added": 0, "relations_added": 0}
        if args.ingest_to_kg:
            try:
                from huginn.kg.graph import get_knowledge_graph
                kg = get_knowledge_graph()
                kg_stats = ingest_into_kg(kg, result)
            except Exception as exc:
                # KG 集成失败不阻塞聚合结果返回
                logger.warning("KG 集成失败: %s", exc)
                kg_stats = {"error": str(exc)}

        return ToolResult(
            data={
                "stats": result["stats"],
                "kg_stats": kg_stats,
                "missing_docs": missing,
                "aggregated_dir": str(agg_dir),
                "entities_count": len(result["entities"]),
                "relationships_count": len(result["relationships"]),
            },
            success=True,
        )


if __name__ == "__main__":
    # C6 self-check: action 路由 + 降级路径 + 输入校验. 不调真实 MinerU/LLM.
    import asyncio
    from dataclasses import dataclass

    from huginn.types import ToolResult

    tool = LiteraturePipelineTool()
    assert tool.name == "literature_pipeline_tool"
    assert tool.category == "materials"
    assert tool.read_only is False

    # 构造无 config 的 context (mineru_api_keys 为空)
    @dataclass
    class _Ctx:
        workspace: str = "."
        config: Any = None

    ctx = _Ctx(workspace=".", config=None)

    async def _run():
        # 1. parse_pdf 缺 pdf_path
        r = await tool.call(LiteraturePipelineInput(action="parse_pdf"), ctx)
        assert r.success is False
        assert "pdf_path" in (r.error or "")

        # 2. parse_pdf 未配置 mineru_api_keys → 明确错误 + fallback 提示
        r = await tool.call(LiteraturePipelineInput(action="parse_pdf", pdf_path="x.pdf"), ctx)
        assert r.success is False
        assert "MinerU API key" in (r.error or "")
        assert (r.data or {}).get("fallback") == "pymupdf_ocr"

        # 3. parse_pdf PDF 不存在 (需要先有 keys 才能走到文件检查)
        @dataclass
        class _CfgWithKeys:
            mineru_api_keys: list = None
        _cfg_keys = _CfgWithKeys(mineru_api_keys=["fake_key_for_test"])
        ctx_with_keys = _Ctx(workspace=".", config=_cfg_keys)
        r = await tool.call(
            LiteraturePipelineInput(action="parse_pdf", pdf_path="nonexistent.pdf"),
            ctx_with_keys,
        )
        assert r.success is False
        assert "not found" in (r.error or "").lower()

        # 4. extract_schema 缺 doc_id
        r = await tool.call(LiteraturePipelineInput(action="extract_schema"), ctx)
        assert r.success is False
        assert "doc_id" in (r.error or "")

        # 5. extract_schema doc 未解析
        r = await tool.call(
            LiteraturePipelineInput(action="extract_schema", doc_id="never_parsed"),
            ctx,
        )
        assert r.success is False
        assert "未解析" in (r.error or "")

        # 6. aggregate_entities 缺 doc_ids
        r = await tool.call(LiteraturePipelineInput(action="aggregate_entities"), ctx)
        assert r.success is False
        assert "doc_ids" in (r.error or "")

        # 7. aggregate_entities 空 doc_ids
        r = await tool.call(
            LiteraturePipelineInput(action="aggregate_entities", doc_ids=[]),
            ctx,
        )
        assert r.success is False

        # 8. unknown action
        # pydantic Literal 不允许 unknown action, 走 dict 路径模拟
        r = await tool.call({"action": "unknown"}, ctx)  # type: ignore[arg-type]
        assert r.success is False

    asyncio.run(_run())

    # 路径 helpers
    ctx_ws = _Ctx(workspace="/tmp/huginn_test", config=None)
    assert str(LiteraturePipelineTool._parsed_root(ctx_ws)).endswith("data" + "\\" + "parsed") or \
           str(LiteraturePipelineTool._parsed_root(ctx_ws)).endswith("data/parsed")
    assert str(LiteraturePipelineTool._aggregated_root(ctx_ws)).endswith("data" + "\\" + "aggregated") or \
           str(LiteraturePipelineTool._aggregated_root(ctx_ws)).endswith("data/aggregated")

    # _get_mineru_keys: config=None 时返回空
    assert LiteraturePipelineTool._get_mineru_keys(_Ctx(config=None)) == []

    @dataclass
    class _Cfg:
        mineru_api_keys: list[str]

    assert LiteraturePipelineTool._get_mineru_keys(_Ctx(config=_Cfg(mineru_api_keys=["k1"]))) == ["k1"]

    print("C6 self-check OK")
