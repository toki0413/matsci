"""LiteratureTool 主体: 7 个 action 分发 + LLM 综述 + 文献基准对比.

search/summarize/benchmark_lookup/fetch_pdf/citations/ingest_to_rag/crawl_web.
HTTP 层在 _http, 搜索源在 search_sources, PDF 抓取在 pdf_fetch,
爬虫与订阅源认证在 crawl_web.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import urllib.parse
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult

from ._http import (
    _DISABLED_HINT,
    _disabled,
    _http_get_bytes,
    _http_get_json,
    logger,
)
from .search_sources import (
    _dedup,
    _sort_papers,
    _search_arxiv,
    _search_core,
    _search_crossref,
    _search_datacite,
    _search_doaj,
    _search_europepmc,
    _search_materials_cloud,
    _search_materials_project,
    _search_nomad,
    _search_openalex,
    _search_openaire,
    _search_pubmed,
    _search_s2,
    _search_zenodo,
    _search_cod,
    _opencitations_references,
    _opencitations_citations,
    _enrich_with_crossref,
)
from .pdf_fetch import (
    _scihub_enabled,
    _scihub_pdf_url,
    europepmc_pdf,
    openalex_oa_url,
    split_sections,
    unpaywall_pdf,
)
from .crawl_web import (
    _PROVIDERS,
    _apply_ezproxy,
    _auth_login,
    _auth_logout,
    _detect_provider,
    _list_sessions,
    _save_session_meta,
    _session_dir,
    _session_meta_path,
    crawl_direct,
    crawl_search_engine,
)


# ───────────────────────── LLM prompts ─────────────────────────


_SUMMARY_SYSTEM_PROMPT = """你是材料科学文献综述专家. 基于给定的 N 篇论文标题和摘要, 写一段结构化综述.

输出格式 (Markdown, 不要加代码块标记):

## 关键发现
- 发现1 [1]
- 发现2 [2][3]
...

## 领域共识
(研究者普遍接受的结论)

## 主要分歧
(不同研究矛盾的地方, 标引用编号)

## 数值汇总
| 体系 | 性质 | 数值 | 方法 | 来源 |
|------|------|------|------|------|
| ... | ... | ... | ... | [1] |

## 研究空白
(摘要里没覆盖到的方向, 值得做的下一步)

引用编号 [1][2]... 对应输入论文的顺序. 只引用给定的论文, 不要编造.
如果论文太少或摘要太短不足以综述, 直接说明, 不要硬凑."""


_BENCHMARK_SYSTEM_PROMPT = """你是材料科学数据提取专家. 从给定的论文列表里, 抽出关于指定体系和性质的具体报道值.

每篇论文如果报了具体数值, 提取:
- value: 数值 (float, 不要带单位字符串)
- unit: 单位 (如 "eV", "epsilon", "GPa", "K"; 没明确单位给空字符串)
- method: 计算/实验方法 (如 "DFT-PBE", "MD", "basin-hopping", "experiment"; 没说给空)
- paper_idx: 论文编号 (从 1 开始, 对应输入顺序)
- note: 一句话备注 (可选, 如 "T=0K", "PBE+U")

严格要求:
1. 只抽明确给出的数值, 不要从公式或趋势里推断.
2. 同一篇报多个值的, 每个值一条.
3. 没报数值的论文直接跳过.
4. 数值保留原文精度, 不要四舍五入.

输出严格 JSON, 不要 markdown 代码块, 不要解释:
{"values": [{"value": -44.33, "unit": "epsilon", "method": "basin-hopping", "paper_idx": 1, "note": ""}, ...]}
如果没有任何论文报数值, 返回 {"values": []}"""


# ───────────────────────── Input schema ─────────────────────────


class LiteratureInput(BaseModel):
    action: Literal[
        "search", "summarize", "benchmark_lookup",
        "fetch_pdf", "citations", "ingest_to_rag",
        "crawl_web", "citation_graph", "extract_figures",
    ] = Field(
        ..., description="search/summarize/benchmark_lookup (第一期) + "
                         "fetch_pdf/citations/ingest_to_rag (第二期) + "
                         "crawl_web (第四期, 爬虫补无API的源) + "
                         "citation_graph (BFS 引文图) + "
                         "extract_figures (从PDF提图调image_analysis)"
    )
    query: str = Field(default="", description="搜索/综述 query")
    max_results: int = Field(
        default=10, ge=1, le=50, description="每个源最多取几条 (多路并发后去重)"
    )
    sources: list[str] = Field(
        default_factory=lambda: ["arxiv", "s2", "crossref", "openalex",
                                  "pubmed", "doaj", "core",
                                  "europepmc", "zenodo", "openaire",
                                  "cod", "materials_cloud",
                                  "nomad", "datacite", "materials_project"],
        description="搜索源, 默认十五路全开. 可减到 ['arxiv'] 单源. "
                    "学术文献: arxiv/s2/crossref/openalex/pubmed/doaj/core/"
                    "europepmc/zenodo/openaire; "
                    "材料数据库: cod/materials_cloud/nomad/materials_project; "
                    "数据集: datacite. "
                    "materials_project 需设 HUGINN_MP_API_KEY",
    )
    year_from: int | None = Field(default=None, description="年份下限 (含)")
    year_to: int | None = Field(default=None, description="年份上限 (含)")

    # summarize 专用: 可以直接喂 papers 跳过 search
    papers: list[dict[str, Any]] | None = Field(
        default=None,
        description="summarize/ingest_to_rag 时可直接传 paper 列表 (从上一次 search 拿)",
    )
    focus: str | None = Field(
        default=None, description="summarize 时的重点, 如 'doping efficiency' 或 'low-temp stability'"
    )

    # benchmark_lookup 专用
    system: str | None = Field(
        default=None, description="体系名, 如 'LJ_13 cluster' 或 'GaN wurtzite'"
    )
    property: str | None = Field(
        default=None, description="性质名, 如 'minimum energy' 或 'band gap'"
    )

    # 第二期: fetch_pdf / citations 单篇论文输入
    paper: dict[str, Any] | None = Field(
        default=None,
        description="fetch_pdf/citations 时传单篇 paper dict (从 search 拿), 含 url/doi 等",
    )
    arxiv_id: str | None = Field(
        default=None, description="直接给 arxiv id (如 '1911.08365' 或 'cond-mat/9909087')"
    )
    doi: str | None = Field(
        default=None, description="直接给 DOI (fetch_pdf 走 Unpaywall 找 OA 版; citations 走 S2 lookup)"
    )
    url: str | None = Field(
        default=None, description="直接给 PDF URL (fetch_pdf 直接下载)"
    )

    # fetch_pdf 专用
    max_chars: int = Field(
        default=50000, ge=1000, le=500000,
        description="全文截断长度, 默认 50k 字 (~12k token). 太长撑爆 LLM context",
    )

    # citations 专用
    direction: Literal["forward", "backward", "both"] = Field(
        default="both",
        description="forward=谁引了这篇; backward=这篇引了谁; both=都要",
    )
    max_citations: int = Field(
        default=20, ge=1, le=100, description="每个方向最多取几条引用"
    )

    # citation_graph 专用: BFS 深度 + 节点上限
    max_depth: int = Field(
        default=2, ge=1, le=3,
        description="citation_graph BFS 深度 (1=只取直接引用, 2/3=多跳)",
    )
    max_nodes: int = Field(
        default=50, ge=5, le=200,
        description="citation_graph 最多收集多少个节点, 防止跑飞",
    )

    # 第四期: crawl_web 专用
    engine: Literal["google_scholar", "google_patents", "duckduckgo", "direct"] = Field(
        default="direct",
        description="crawl_web 搜索引擎: google_scholar/google_patents/duckduckgo "
                    "(需配 query); direct=直接爬 url",
    )

    # 第五期: 订阅源认证 (高校非 OA 期刊)
    auth_action: Literal["login", "status", "logout"] | None = Field(
        default=None,
        description="crawl_web 认证动作: login=弹浏览器手动登录存 profile, "
                    "status=列出已存 session, logout=删 profile",
    )
    provider: str | None = Field(
        default=None,
        description="订阅源 provider: cnki/wanfang/cqvip/elsevier/springer/ieee/"
                    "wiley/acs/rsc/nature/wos/tandfonline. 配 auth_action 用",
    )

    model_config = {"protected_namespaces": ()}


# ───────────────────────── Tool ─────────────────────────


class LiteratureTool(HuginnTool):
    """学术文献+材料数据库调研工具. 15 路并发搜索 (arXiv/S2/CrossRef/OpenAlex/PubMed/
    DOAJ/CORE/EuropePMC/Zenodo/OpenAIRE/COD/MaterialsCloud/NOMAD/DataCite/MaterialsProject),
    LLM 综述, 文献基准对比, OA PDF 全文抓取, 引用网络查询, RAG 入库."""

    name = "literature_tool"
    category = "search"
    description = (
        "Search 15 sources in parallel: literature (arXiv/S2/CrossRef/OpenAlex/PubMed/"
        "DOAJ/CORE/EuropePMC/Zenodo/OpenAIRE), materials databases (COD/Materials Cloud/"
        "NOMAD 19M+ entries/Materials Project), and datasets (DataCite). "
        "Generate multi-paper summaries with citations, look up literature-reported "
        "values for a given system+property (complements validate_tool's built-in benchmarks), "
        "fetch OA PDF full text (multi-source: OpenAlex/Unpaywall/Europe PMC/arXiv), "
        "query citation networks, and ingest papers into local RAG. "
        "Use this BEFORE running calculations to find known values, or AFTER to compare."
    )
    input_schema = LiteratureInput
    read_only = True

    async def call(self, args: LiteratureInput, context: ToolContext) -> ToolResult:
        # 兼容 dict 入参, 让 hypothesis_generator_tool._invoke_tool 那种传 dict 的路径也能用
        if isinstance(args, dict):
            args = LiteratureInput(**args)
        if _disabled():
            return ToolResult(
                data={"error": "literature_tool disabled", "hint": _DISABLED_HINT},
                success=False,
                error=_DISABLED_HINT,
            )
        try:
            if args.action == "search":
                return await self._do_search(args)
            if args.action == "summarize":
                return await self._do_summarize(args, context)
            if args.action == "benchmark_lookup":
                return await self._do_benchmark_lookup(args, context)
            if args.action == "fetch_pdf":
                return await self._do_fetch_pdf(args)
            if args.action == "citations":
                return await self._do_citations(args)
            if args.action == "citation_graph":
                return await self._do_citation_graph(args)
            if args.action == "ingest_to_rag":
                return await self._do_ingest_to_rag(args, context)
            if args.action == "crawl_web":
                return await self._do_crawl_web(args, context)
            if args.action == "extract_figures":
                return await self._do_extract_figures(args, context)
            return ToolResult(
                data=None, success=False, error=f"unknown action: {args.action}"
            )
        except Exception as exc:
            logger.exception("literature_tool %s failed", args.action)
            return ToolResult(data=None, success=False, error=str(exc))

    # ── search ──────────────────────────────────────────────

    async def _do_search(self, args: LiteratureInput) -> ToolResult:
        query = (args.query or "").strip()
        if not query:
            return ToolResult(
                data=None, success=False, error="query is required for search"
            )

        tasks: list[tuple[str, asyncio.Task]] = []
        if "arxiv" in args.sources:
            tasks.append(("arxiv", asyncio.create_task(
                _search_arxiv(query, args.max_results, args.year_from, args.year_to)
            )))
        if "s2" in args.sources:
            tasks.append(("s2", asyncio.create_task(
                _search_s2(query, args.max_results, args.year_from, args.year_to)
            )))
        if "crossref" in args.sources:
            tasks.append(("crossref", asyncio.create_task(
                _search_crossref(query, args.max_results, args.year_from, args.year_to)
            )))
        if "openalex" in args.sources:
            tasks.append(("openalex", asyncio.create_task(
                _search_openalex(query, args.max_results, args.year_from, args.year_to)
            )))
        if "pubmed" in args.sources:
            tasks.append(("pubmed", asyncio.create_task(
                _search_pubmed(query, args.max_results, args.year_from, args.year_to)
            )))
        if "doaj" in args.sources:
            tasks.append(("doaj", asyncio.create_task(
                _search_doaj(query, args.max_results, args.year_from, args.year_to)
            )))
        if "core" in args.sources:
            tasks.append(("core", asyncio.create_task(
                _search_core(query, args.max_results, args.year_from, args.year_to)
            )))
        if "europepmc" in args.sources:
            tasks.append(("europepmc", asyncio.create_task(
                _search_europepmc(query, args.max_results, args.year_from, args.year_to)
            )))
        if "zenodo" in args.sources:
            tasks.append(("zenodo", asyncio.create_task(
                _search_zenodo(query, args.max_results, args.year_from, args.year_to)
            )))
        if "openaire" in args.sources:
            tasks.append(("openaire", asyncio.create_task(
                _search_openaire(query, args.max_results, args.year_from, args.year_to)
            )))
        if "cod" in args.sources:
            tasks.append(("cod", asyncio.create_task(
                _search_cod(query, args.max_results, args.year_from, args.year_to)
            )))
        if "materials_cloud" in args.sources:
            tasks.append(("materials_cloud", asyncio.create_task(
                _search_materials_cloud(query, args.max_results, args.year_from, args.year_to)
            )))
        if "nomad" in args.sources:
            tasks.append(("nomad", asyncio.create_task(
                _search_nomad(query, args.max_results, args.year_from, args.year_to)
            )))
        if "datacite" in args.sources:
            tasks.append(("datacite", asyncio.create_task(
                _search_datacite(query, args.max_results, args.year_from, args.year_to)
            )))
        if "materials_project" in args.sources:
            tasks.append(("materials_project", asyncio.create_task(
                _search_materials_project(query, args.max_results, args.year_from, args.year_to)
            )))
        if not tasks:
            return ToolResult(
                data=None, success=False,
                error=f"no valid sources in {args.sources}",
            )

        # 多路并发, 任一失败不阻塞其他
        results = await asyncio.gather(
            *[t for _, t in tasks], return_exceptions=True
        )
        all_papers: list[dict[str, Any]] = []
        source_status: dict[str, Any] = {}
        for (src, _), res in zip(tasks, results):
            if isinstance(res, Exception):
                source_status[src] = {"ok": False, "error": str(res)[:200]}
                logger.warning("source %s failed: %s", src, res)
            else:
                source_status[src] = {"ok": True, "count": len(res)}
                all_papers.extend(res)

        deduped = _dedup(all_papers)
        ranked = _sort_papers(deduped)[: args.max_results * 2]

        return ToolResult(
            data={
                "action": "search",
                "query": query,
                "total": len(ranked),
                "papers": ranked,
                "sources_tried": [s for s, _ in tasks],
                "source_status": source_status,
            },
            success=True,
        )

    # ── summarize ───────────────────────────────────────────

    async def _do_summarize(
        self, args: LiteratureInput, context: ToolContext
    ) -> ToolResult:
        papers = args.papers
        if not papers and args.query:
            search_res = await self._do_search(args)
            if not search_res.success:
                return search_res
            papers = (search_res.data or {}).get("papers", [])
        if not papers or len(papers) == 0:
            return ToolResult(
                data=None, success=False,
                error="no papers to summarize (provide papers or a query)",
            )

        # 太多就截断, 喂 LLM 的 context 不能爆
        papers = papers[:15]
        focus_line = f"\n\n综述重点 (如果给的话): {args.focus}" if args.focus else ""

        # 构造论文清单给 LLM
        paper_block_parts: list[str] = []
        for i, p in enumerate(papers, 1):
            authors_short = ", ".join(p.get("authors", [])[:3])
            if len(p.get("authors", [])) > 3:
                authors_short += " et al."
            year = p.get("year") or ""
            venue = p.get("venue") or ""
            abstract = (p.get("abstract") or "").strip()
            if len(abstract) > 1500:
                abstract = abstract[:1500] + "..."
            paper_block_parts.append(
                f"[{i}] {p.get('title','')}\n"
                f"  Authors: {authors_short}\n"
                f"  Year: {year}  Venue: {venue}  DOI: {p.get('doi') or '-'}\n"
                f"  Abstract: {abstract}"
            )
        paper_block = "\n\n".join(paper_block_parts)

        user_prompt = (
            f"研究 query: {args.query or '(未指定)'}{focus_line}\n\n"
            f"论文列表 ({len(papers)} 篇):\n\n{paper_block}\n\n"
            "请基于以上论文写结构化综述."
        )

        try:
            model = self._get_model(context)
        except Exception as exc:
            return ToolResult(
                data=None, success=False,
                error=f"LLM 初始化失败: {exc}",
            )

        content = await self._llm_invoke(model, _SUMMARY_SYSTEM_PROMPT, user_prompt)

        # 顺带生成 bibtex, 方便用户直接拿去用
        bibtex = self._to_bibtex(papers)

        return ToolResult(
            data={
                "action": "summarize",
                "query": args.query or "",
                "focus": args.focus,
                "n_papers": len(papers),
                "summary_markdown": content,
                "bibtex": bibtex,
                "papers": [
                    {"idx": i + 1, "title": p.get("title", ""),
                     "doi": p.get("doi"), "year": p.get("year")}
                    for i, p in enumerate(papers)
                ],
            },
            success=True,
        )

    # ── benchmark_lookup ────────────────────────────────────

    async def _do_benchmark_lookup(
        self, args: LiteratureInput, context: ToolContext
    ) -> ToolResult:
        if not args.system or not args.property:
            return ToolResult(
                data=None, success=False,
                error="system and property are required for benchmark_lookup",
            )

        # search query 优先用用户给的, 没给就拼 system+property
        query = args.query or f"{args.system} {args.property}"
        # 如果直接传了 papers (可能带 full_text), 就用传的, 不重新 search.
        # 典型场景: 先 fetch_pdf 拿全文, 再喂给 benchmark_lookup 抽精确值
        if args.papers:
            papers = args.papers
        else:
            search_res = await self._do_search(LiteratureInput(
                action="search",
                query=query,
                max_results=max(args.max_results, 15),
                sources=args.sources,
                year_from=args.year_from,
                year_to=args.year_to,
            ))
            if not search_res.success:
                return search_res
            papers = (search_res.data or {}).get("papers", [])
        if not papers:
            return ToolResult(
                data={
                    "action": "benchmark_lookup",
                    "system": args.system,
                    "property": args.property,
                    "query": query,
                    "reported_values": [],
                    "consensus": None,
                    "spread": None,
                    "n_papers_searched": 0,
                    "n_papers_with_values": 0,
                    "kb_written": 0,
                    "message": "没搜到相关文献",
                },
                success=True,
            )

        # 只喂有文本的论文给 LLM; 优先 full_text (fetch_pdf 拿到的), 退回 abstract
        papers_with_text = [p for p in papers if p.get("full_text") or p.get("abstract")]
        # 重新编号, paper_idx 对应这个子集. 全文长, 上限 10 篇避免撑爆 context
        paper_block_parts: list[str] = []
        for i, p in enumerate(papers_with_text[:10], 1):
            text = (p.get("full_text") or p.get("abstract") or "").strip()
            # full_text 可能几万字, 截到 4000 字 (~1000 token) 够 LLM 抽数值
            if len(text) > 4000:
                text = text[:4000] + "..."
            label = "Full text" if p.get("full_text") else "Abstract"
            paper_block_parts.append(
                f"[{i}] {p.get('title','')}\n  {label}: {text}"
            )
        paper_block = "\n\n".join(paper_block_parts)

        user_prompt = (
            f"体系: {args.system}\n"
            f"性质: {args.property}\n\n"
            f"论文列表 ({len(papers_with_text[:10])} 篇, 含全文或 abstract):\n\n"
            f"{paper_block}\n\n"
            f"请抽出关于 {args.system} 的 {args.property} 报道值."
        )

        try:
            model = self._get_model(context)
        except Exception as exc:
            return ToolResult(
                data=None, success=False,
                error=f"LLM 初始化失败: {exc}",
            )

        content = await self._llm_invoke(model, _BENCHMARK_SYSTEM_PROMPT, user_prompt)
        parsed = self._parse_json(content)
        raw_values = parsed.get("values", []) if parsed else []

        # 把 paper_idx 映射回真实 paper 信息
        reported: list[dict[str, Any]] = []
        for v in raw_values:
            if not isinstance(v, dict):
                continue
            try:
                value = float(v.get("value"))
            except (TypeError, ValueError):
                continue
            idx = int(v.get("paper_idx", 0))
            if idx < 1 or idx > len(papers_with_text[:10]):
                continue
            paper = papers_with_text[idx - 1]
            reported.append({
                "value": value,
                "unit": str(v.get("unit", "")),
                "method": str(v.get("method", "")),
                "note": str(v.get("note", "")),
                "source_paper": paper.get("title", ""),
                "doi": paper.get("doi"),
                "year": paper.get("year"),
                "venue": paper.get("venue", ""),
            })

        # 共识 + 离散度
        consensus = None
        spread = None
        if reported:
            vals = [r["value"] for r in reported]
            mean = sum(vals) / len(vals)
            consensus = {
                "mean": round(mean, 6),
                "median": round(sorted(vals)[len(vals) // 2], 6),
                "unit": reported[0]["unit"],
                "n_sources": len(reported),
            }
            if len(vals) >= 2:
                spread = {
                    "min": round(min(vals), 6),
                    "max": round(max(vals), 6),
                    "range": round(max(vals) - min(vals), 6),
                }

        # 把抽到的文献报道值写回知识库，下次同体系查询能直接命中
        kb_written = 0
        try:
            from huginn.knowledge.store import get_knowledge_base
            kb = get_knowledge_base()
            for rv in reported:
                doi = rv.get("doi") or ""
                title = rv.get("source_paper") or ""
                text = (
                    f"{args.system} | {args.property} = {rv['value']} {rv['unit']}\n"
                    f"method: {rv.get('method', '')}\n"
                    f"source: {title}\n"
                    f"doi: {doi}\n"
                    f"year: {rv.get('year', '')}\n"
                )
                meta = {"doi": doi, "title": title, "source": "benchmark_lookup"}
                kb.add_text(text, filename="benchmark_lookup", metadata=meta)
                kb_written += 1
        except Exception as exc:
            logger.warning("benchmark_lookup KB 写回失败: %s", exc)

        return ToolResult(
            data={
                "action": "benchmark_lookup",
                "system": args.system,
                "property": args.property,
                "query": query,
                "reported_values": reported,
                "consensus": consensus,
                "spread": spread,
                "n_papers_searched": len(papers),
                "n_papers_with_values": len(reported),
                "kb_written": kb_written,
            },
            success=True,
        )

    # ── fetch_pdf ───────────────────────────────────────────

    async def _do_fetch_pdf(self, args: LiteratureInput) -> ToolResult:
        """下载 OA PDF + PyMuPDF 抽正文 + 分节. 解决 benchmark_lookup
        从 abstract 抽不到数值的问题 (精确值通常在论文正文表里).

        多源候选: OpenAlex oa_url → Unpaywall → Europe PMC → arxiv.org/pdf.
        arxiv.org/pdf 在国内常超时, 放最后兜底, 前面的 OA 源先试.
        """
        pdf_bytes, used_url, tried = await self._download_pdf_bytes(args)
        if pdf_bytes is None:
            if not tried:
                return ToolResult(
                    data=None, success=False,
                    error="无法解析 PDF URL, 需要 arxiv_id/doi/url 或 paper dict (含 url/doi/oa_url)",
                )
            last_status = tried[-1]["status"] if tried else ""
            err_msg = f"所有 PDF 候选源都失败 (试了 {len(tried)} 个). 最后错误: {last_status[:200]}"
            if any("10060" in t["status"] or "timeout" in t["status"].lower() for t in tried):
                err_msg += " | 多个源超时, 试试配 HTTPS_PROXY 环境变量走代理"
            err_msg += f" | tried: {tried}"
            return ToolResult(
                data=None, success=False,
                error=err_msg,
            )

        # PyMuPDF 抽正文
        try:
            import fitz  # PyMuPDF
        except ImportError:
            return ToolResult(
                data=None, success=False,
                error="PyMuPDF (fitz) 未安装, 无法抽 PDF 正文. pip install pymupdf",
            )

        try:
            # 用 with 保证异常路径也释放文件句柄
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                pages_text: list[str] = []
                for page in doc:
                    pages_text.append(page.get_text())
        except Exception as exc:
            return ToolResult(
                data=None, success=False,
                error=f"PDF 解析失败: {exc}",
            )

        full_text = "\n\n".join(pages_text)
        n_pages = len(pages_text)
        n_chars = len(full_text)
        sections = split_sections(full_text)

        truncated = False
        if n_chars > args.max_chars:
            full_text = full_text[: args.max_chars]
            truncated = True

        return ToolResult(
            data={
                "action": "fetch_pdf",
                "pdf_url": used_url,
                "n_pages": n_pages,
                "n_chars": n_chars,
                "truncated": truncated,
                "n_sections": len(sections),
                "sections": sections,
                "full_text": full_text,
                "candidates_tried": tried,
            },
            success=True,
        )

    async def _resolve_pdf_candidates(self, args: LiteratureInput) -> list[str]:
        """多源解析 PDF URL, 返回候选列表 (优先级高的在前).

        顺序考虑网络可达性 (国内环境):
          1. 直接给的 .pdf url / paper.url 是 .pdf
          2. paper.oa_url (OpenAlex 搜出来的直接给)
          3. OpenAlex 按 DOI 查 oa_url
          4. Unpaywall (DOI → OA PDF)
          5. Europe PMC (DOI → fullTextUrl)
          6. arxiv.org/pdf (arxiv abs url 或 arxiv_id) — 放最后, 国内常超时
          7. Sci-Hub (仅 HUGINN_ENABLE_SCIHUB=1 时启用, 最后兜底, 法律灰色地带)
        """
        candidates: list[str] = []
        doi: str | None = args.doi
        arxiv_id: str | None = args.arxiv_id

        # 从 paper dict 提取信息
        if args.paper:
            p = args.paper
            p_url = p.get("url", "") or ""
            p_doi = p.get("doi")
            if p_doi and not doi:
                doi = p_doi
            # paper dict 自带的 oa_url (OpenAlex 搜出来会有)
            oa_url = p.get("oa_url") or ""
            if oa_url and oa_url.endswith(".pdf"):
                candidates.append(oa_url)
            elif oa_url and "arxiv.org" not in oa_url:
                # 非 arxiv 的 oa_url 也加上, 有些就是直链 PDF
                candidates.append(oa_url)
            # CORE 直接给 download_url (全文 PDF 链接), 优先级高
            core_dl = p.get("download_url") or ""
            if core_dl and core_dl not in candidates:
                candidates.append(core_dl)
            # paper.url 本身就是 .pdf
            if p_url and p_url.endswith(".pdf") and p_url not in candidates:
                candidates.append(p_url)
            # 从 arxiv abs url 提 arxiv_id (放后面拼 arxiv pdf)
            if not arxiv_id and "arxiv.org/abs/" in p_url:
                arxiv_id = p_url.split("/abs/")[-1].strip("/")
            elif not arxiv_id and "arxiv.org/pdf/" in p_url:
                aid = p_url.split("/pdf/")[-1].strip("/")
                arxiv_id = aid[:-4] if aid.endswith(".pdf") else aid

        # 直接给的 url 最优先
        if args.url and args.url not in candidates:
            candidates.insert(0, args.url)

        # OpenAlex 按 DOI 查 oa_url (OpenAlex 在国内一般可达)
        if doi and not any("openalex" in c for c in candidates):
            oa = await openalex_oa_url(doi)
            if oa:
                candidates.append(oa)

        # Unpaywall
        if doi:
            upw = await unpaywall_pdf(doi)
            if upw and upw not in candidates:
                candidates.append(upw)

        # Europe PMC (biomedical 为主, 但也覆盖一些材料/化学)
        if doi:
            epmc = await europepmc_pdf(doi)
            if epmc and epmc not in candidates:
                candidates.append(epmc)

        # arxiv.org/pdf 放最后 (国内常超时)
        if arxiv_id:
            aid = arxiv_id.strip().strip("/")
            aid = aid[:-4] if aid.endswith(".pdf") else aid
            arxiv_pdf = f"https://arxiv.org/pdf/{aid}.pdf"
            if arxiv_pdf not in candidates:
                candidates.append(arxiv_pdf)

        # Sci-Hub: 仅 HUGINN_ENABLE_SCIHUB=1 时启用, 最后兜底
        # 法律灰色地带: Sci-Hub 托管版权论文未获出版商授权, 用户需自行承担合规风险.
        if doi and _scihub_enabled():
            sh_url = await _scihub_pdf_url(doi)
            if sh_url and sh_url not in candidates:
                candidates.append(sh_url)

        # 去重保序
        seen: set[str] = set()
        deduped: list[str] = []
        for c in candidates:
            if c and c not in seen:
                seen.add(c)
                deduped.append(c)
        return deduped

    async def _resolve_pdf_url(self, args: LiteratureInput) -> str | None:
        """单 URL 解析 (向后兼容). 返回第一个候选."""
        cands = await self._resolve_pdf_candidates(args)
        return cands[0] if cands else None

    async def _download_pdf_bytes(
        self, args: LiteratureInput
    ) -> tuple[bytes | None, str, list[dict[str, str]]]:
        """多源下载 PDF, 返回 (pdf_bytes, used_url, tried).

        pdf_bytes 为 None 表示全部候选源失败, tried 列表含各源失败原因.
        _do_fetch_pdf 和 _do_extract_figures 共用这段下载逻辑.
        """
        candidates = await self._resolve_pdf_candidates(args)
        if not candidates:
            return None, "", []

        last_err = ""
        tried: list[dict[str, str]] = []
        pdf_bytes = b""
        used_url = ""
        for cand in candidates:
            try:
                pdf_bytes = await _http_get_bytes(cand)
                # magic bytes 校验: %PDF 开头且够大, OpenAlex 有时返回 HTML 冒充 PDF
                if len(pdf_bytes) < 1000:
                    tried.append({"url": cand, "status": f"too small ({len(pdf_bytes)} bytes)"})
                    continue
                if not pdf_bytes[:5].startswith(b"%PDF"):
                    tried.append({"url": cand, "status": f"not PDF (got {pdf_bytes[:20]!r})"})
                    continue
                used_url = cand
                break
            except Exception as exc:
                err_str = str(exc)
                tried.append({"url": cand, "status": err_str[:120]})
                last_err = err_str
                logger.info("fetch_pdf 候选 %s 失败: %s", cand[:60], exc)

        if not used_url:
            return None, "", tried

        return pdf_bytes, used_url, tried

    # ── citations ───────────────────────────────────────────

    async def _do_citations(self, args: LiteratureInput) -> ToolResult:
        """前/后向引用网络. 先试 S2, 被限速(429)就回退到 OpenCitations.

        forward=谁引了这篇; backward=这篇引了谁.
        S2 给完整元数据 (title/authors/citations), OpenCitations 只给 DOI 边,
        靠 CrossRef 补元数据. 两路都挂才报错.
        """
        paper_id = self._resolve_s2_paper_id(args)
        doi = args.doi
        if not doi and args.paper:
            doi = (args.paper or {}).get("doi")
        if not doi and paper_id and paper_id.startswith("DOI:"):
            doi = paper_id[4:]

        if not paper_id and not doi:
            return ToolResult(
                data=None, success=False,
                error="无法解析 paper_id, 需要 doi/arxiv_id 或 paper dict (含 doi/url)",
            )

        result: dict[str, Any] = {
            "action": "citations",
            "paper_id": paper_id or (f"DOI:{doi}" if doi else ""),
            "direction": args.direction,
            "forward_citations": [],
            "backward_references": [],
            "sources_used": [],
        }

        s2_failed = False

        # ── 前向: 谁引了这篇 ──
        if args.direction in ("forward", "both"):
            # 先试 S2
            if paper_id:
                try:
                    pid_enc = urllib.parse.quote(paper_id, safe="")
                    data = await _http_get_json(
                        f"https://api.semanticscholar.org/graph/v1/paper/{pid_enc}/citations"
                        f"?fields=title,year,authors,citationCount,externalIds&limit={args.max_citations}"
                    )
                    for c in data.get("data", []) or []:
                        cp = c.get("citingPaper", {}) or {}
                        result["forward_citations"].append(self._s2_paper_to_dict(cp))
                    result["sources_used"].append("s2")
                except Exception as exc:
                    s2_failed = True
                    result["forward_error_s2"] = str(exc)[:200]
                    logger.warning("S2 citations (forward) 失败: %s", exc)

            # S2 挂了就回退 OpenCitations
            if s2_failed and doi:
                try:
                    oc_cites = await _opencitations_citations(doi, args.max_citations)
                    if oc_cites:
                        # CrossRef 补元数据
                        oc_cites = await _enrich_with_crossref(oc_cites)
                        result["forward_citations"].extend(oc_cites)
                        result["sources_used"].append("opencitations")
                        result["forward_fallback"] = "opencitations"
                except Exception as exc:
                    result["forward_error_oc"] = str(exc)[:200]
                    logger.warning("OpenCitations citations (forward) 失败: %s", exc)

        # ── 后向: 这篇引了谁 ──
        backward_s2_failed = False
        if args.direction in ("backward", "both"):
            if paper_id:
                try:
                    pid_enc = urllib.parse.quote(paper_id, safe="")
                    data = await _http_get_json(
                        f"https://api.semanticscholar.org/graph/v1/paper/{pid_enc}/references"
                        f"?fields=title,year,authors,citationCount,externalIds&limit={args.max_citations}"
                    )
                    for r in data.get("data", []) or []:
                        cp = r.get("citedPaper", {}) or {}
                        result["backward_references"].append(self._s2_paper_to_dict(cp))
                    if "s2" not in result["sources_used"]:
                        result["sources_used"].append("s2")
                except Exception as exc:
                    backward_s2_failed = True
                    result["backward_error_s2"] = str(exc)[:200]
                    logger.warning("S2 references (backward) 失败: %s", exc)

            # S2 挂了就回退 OpenCitations
            if backward_s2_failed and doi:
                try:
                    oc_refs = await _opencitations_references(doi, args.max_citations)
                    if oc_refs:
                        oc_refs = await _enrich_with_crossref(oc_refs)
                        result["backward_references"].extend(oc_refs)
                        if "opencitations" not in result["sources_used"]:
                            result["sources_used"].append("opencitations")
                        result["backward_fallback"] = "opencitations"
                except Exception as exc:
                    result["backward_error_oc"] = str(exc)[:200]
                    logger.warning("OpenCitations references (backward) 失败: %s", exc)

        result["forward_count"] = len(result["forward_citations"])
        result["backward_count"] = len(result["backward_references"])
        return ToolResult(data=result, success=True)

    @staticmethod
    def _resolve_s2_paper_id(args: LiteratureInput) -> str | None:
        """构造 S2 paper lookup key. DOI:xxx / arXiv:xxx 格式."""
        doi = args.doi
        arxiv_id = args.arxiv_id
        if args.paper:
            p = args.paper
            doi = doi or p.get("doi")
            url = p.get("url", "") or ""
            if not arxiv_id and "arxiv.org/abs/" in url:
                arxiv_id = url.split("/abs/")[-1].strip("/")
        if doi:
            return f"DOI:{doi}"
        if arxiv_id:
            return f"arXiv:{arxiv_id.strip().strip('/')}"
        return None

    @staticmethod
    def _s2_paper_to_dict(p: dict[str, Any]) -> dict[str, Any]:
        return {
            "title": p.get("title", "") or "",
            "authors": [
                a.get("name", "") for a in (p.get("authors") or []) if a.get("name")
            ],
            "year": p.get("year"),
            "citations": p.get("citationCount"),
            "paperId": p.get("paperId"),
            "doi": (p.get("externalIds") or {}).get("DOI"),
        }

    # ── citation_graph ─────────────────────────────────────

    async def _do_citation_graph(self, args: LiteratureInput) -> ToolResult:
        """BFS 引文图: 从种子 paper 出发, 沿 backward references 往上挖.

        每层调 S2 references API, 收集 nodes + edges. S2 被限速(429)时
        自动切到 OpenCitations DOI 边 + CrossRef 元数据补全. 触顶 max_nodes
        或 max_depth 就停. 两路都挂也返回已收集的部分图, agent 能用部分结果.
        """
        seed_id = self._resolve_s2_paper_id(args)
        doi = args.doi
        if not doi and args.paper:
            doi = (args.paper or {}).get("doi")
        if not doi and seed_id and seed_id.startswith("DOI:"):
            doi = seed_id[4:]

        if not seed_id and not doi:
            return ToolResult(
                data=None,
                success=False,
                error="无法解析 paper_id, 需要 doi/arxiv_id 或 paper dict (含 doi/url)",
            )

        visited: set[str] = set()
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, str]] = []
        errors: list[str] = []
        source_mode = "s2"  # 默认用 S2, 限速后切 "opencitations"

        # 种子节点
        seed_node = {
            "paper_id": seed_id or (f"DOI:{doi}" if doi else ""),
            "title": (args.paper or {}).get("title", "") if args.paper else "",
            "year": (args.paper or {}).get("year") if args.paper else None,
            "doi": doi,
            "depth": 0,
        }
        nodes.append(seed_node)
        visited.add(seed_node["paper_id"])

        # BFS: 当前层的 paper_id 列表
        current_layer: list[str] = [seed_node["paper_id"]]
        # 同步维护 current_dois, 用于 OpenCitations fallback
        current_dois: list[str] = [doi] if doi else []
        depth_reached = 0

        for depth in range(1, args.max_depth + 1):
            if len(nodes) >= args.max_nodes or not current_layer:
                break
            next_layer: list[str] = []
            next_dois: list[str] = []

            for idx, parent_id in enumerate(current_layer):
                if len(nodes) >= args.max_nodes:
                    break
                parent_doi = current_dois[idx] if idx < len(current_dois) else None

                # ── S2 模式: 调 references API ──
                if source_mode == "s2" and parent_id:
                    try:
                        pid_enc = urllib.parse.quote(parent_id, safe="")
                        data = await _http_get_json(
                            f"https://api.semanticscholar.org/graph/v1/paper/{pid_enc}/references"
                            f"?fields=title,year,externalIds,paperId&limit={args.max_citations}"
                        )
                        for ref in data.get("data", []) or []:
                            cp = ref.get("citedPaper", {}) or {}
                            child_id = cp.get("paperId") or ""
                            if not child_id:
                                continue
                            edges.append({"from": parent_id, "to": child_id})
                            if child_id not in visited:
                                if len(nodes) >= args.max_nodes:
                                    break
                                child_doi = (cp.get("externalIds") or {}).get("DOI")
                                nodes.append({
                                    "paper_id": child_id,
                                    "title": cp.get("title", "") or "",
                                    "year": cp.get("year"),
                                    "doi": child_doi,
                                    "depth": depth,
                                })
                                visited.add(child_id)
                                next_layer.append(child_id)
                                if child_doi:
                                    next_dois.append(child_doi)
                        depth_reached = depth
                        continue  # S2 成功就跳过 OpenCitations
                    except Exception as exc:
                        errors.append(f"depth={depth} parent={parent_id} s2: {exc}"[:200])
                        logger.warning("S2 BFS depth=%d failed, switching to OpenCitations: %s", depth, exc)
                        source_mode = "opencitations"

                # ── OpenCitations 模式: DOI 边 + CrossRef 补元数据 ──
                if source_mode == "opencitations" and parent_doi:
                    try:
                        oc_refs = await _opencitations_references(parent_doi, args.max_citations)
                        if not oc_refs:
                            continue
                        # 批量补元数据
                        oc_refs = await _enrich_with_crossref(oc_refs)
                        for ref in oc_refs:
                            child_doi = ref.get("doi", "")
                            child_id = f"DOI:{child_doi}" if child_doi else ""
                            if not child_id or child_id in visited:
                                continue
                            edges.append({"from": parent_id, "to": child_id})
                            if len(nodes) >= args.max_nodes:
                                break
                            nodes.append({
                                "paper_id": child_id,
                                "title": ref.get("title", ""),
                                "year": ref.get("year"),
                                "doi": child_doi,
                                "depth": depth,
                                "source": "opencitations",
                            })
                            visited.add(child_id)
                            next_layer.append(child_id)
                            next_dois.append(child_doi)
                        depth_reached = depth
                    except Exception as exc:
                        errors.append(f"depth={depth} parent={parent_doi} oc: {exc}"[:200])

            current_layer = next_layer
            current_dois = next_dois

        # 引文图持久化到 KG — Zotero 启发: 引用关系应持久存储供后续查询.
        # 之前是 ephemeral 的, agent 每次都要重新 BFS. 现在写一次, 后续
        # KG query 和 /graph 可视化都能看到.
        try:
            from huginn.kg.graph import ProjectKnowledgeGraph
            from huginn.kg.entities import EntityType, Relation
            from pathlib import Path
            kg = ProjectKnowledgeGraph(Path("."))
            # pid -> node eid, 供 add_relation 用
            eid_map: dict[str, str] = {}
            for node in nodes:
                pid = node.get("paper_id", "")
                if not pid:
                    continue
                title = node.get("title", "") or pid
                eid = kg.add_entity(
                    title, EntityType.LITERATURE,
                    source="citation_graph",
                    confidence=0.8 if node.get("depth", 99) <= 1 else 0.6,
                    paper_id=pid,
                    doi=node.get("doi", ""),
                    year=node.get("year"),
                    depth=node.get("depth", 0),
                )
                eid_map[pid] = eid
            for edge in edges:
                src = eid_map.get(edge.get("source", ""))
                dst = eid_map.get(edge.get("target", ""))
                if src and dst:
                    kg.add_relation(src, Relation.CITES, dst, source="citation_graph")
            kg.save()
        except Exception:
            pass  # best-effort, 不影响引文图返回

        return ToolResult(
            data={
                "action": "citation_graph",
                "seed_paper_id": seed_node["paper_id"],
                "depth_reached": depth_reached,
                "n_unique_papers": len(nodes),
                "n_edges": len(edges),
                "nodes": nodes,
                "edges": edges,
                "errors": errors,
                "truncated": len(nodes) >= args.max_nodes,
                "source_mode": source_mode,
            },
            success=True,
        )

    # ── ingest_to_rag ───────────────────────────────────────

    async def _do_ingest_to_rag(
        self, args: LiteratureInput, context: ToolContext
    ) -> ToolResult:
        """把搜到的论文 ingest 进 rag_tool, 下次同主题 search 时本地能搜到."""
        from huginn.tools.registry import ToolRegistry

        papers = args.papers
        if not papers and args.query:
            search_res = await self._do_search(args)
            if not search_res.success:
                return search_res
            papers = (search_res.data or {}).get("papers", [])
        if not papers:
            return ToolResult(
                data=None, success=False, error="no papers to ingest (provide papers or query)",
            )

        rag = ToolRegistry.get("rag_tool")
        if rag is None:
            return ToolResult(
                data=None, success=False,
                error="rag_tool 未注册, 无法 ingest. 需要先初始化 rag_tool",
            )

        ingested: list[str] = []
        failed: list[dict[str, str]] = []
        for p in papers:
            title = p.get("title", "")
            authors = ", ".join((p.get("authors") or [])[:5])
            year = p.get("year") or ""
            venue = p.get("venue", "")
            doi = p.get("doi") or ""
            abstract = p.get("abstract", "") or ""
            if not abstract and not title:
                continue
            doc_text = (
                f"Title: {title}\n"
                f"Authors: {authors}\n"
                f"Year: {year}\n"
                f"Venue: {venue}\n"
                f"DOI: {doi}\n"
                f"Abstract: {abstract}"
            )
            doc_id = doi or f"lit_{_norm_title(title)[:40]}"
            try:
                from huginn.rag.rag_tool import RAGToolInput

                res = await rag.call(
                    RAGToolInput(action="ingest", document=doc_text, doc_id=doc_id),
                    context,
                )
                if res.success:
                    ingested.append(doc_id)
                else:
                    failed.append({"doc_id": doc_id, "error": (res.error or "")[:100]})
            except Exception as exc:
                failed.append({"doc_id": doc_id, "error": str(exc)[:100]})

        return ToolResult(
            data={
                "action": "ingest_to_rag",
                "n_ingested": len(ingested),
                "n_failed": len(failed),
                "doc_ids": ingested,
                "failures": failed,
            },
            success=True,
        )

    # ── crawl_web ───────────────────────────────────────────

    async def _do_crawl_web(
        self, args: LiteratureInput, context: ToolContext
    ) -> ToolResult:
        """通用网页爬取 + 搜索引擎桥接 + 订阅源认证.

        三种模式:
          - auth_action=login: 弹非 headless 浏览器让用户手动登录订阅源, 存 profile
          - auth_action=status: 列出所有已存 session
          - auth_action=logout: 删 provider 的 profile
          - engine=direct: 直接爬 args.url (有 session 自动复用 profile)
          - engine=google_scholar/google_patents/duckduckgo: 搜结果链接列表

        crawl4ai 不可用时降级到 urllib + 简单 HTML 抽正文.
        """
        # 认证动作优先
        if args.auth_action == "status":
            sessions = _list_sessions()
            return ToolResult(
                data={
                    "action": "crawl_web",
                    "auth_action": "status",
                    "n_sessions": len(sessions),
                    "sessions": sessions,
                    "supported_providers": list(_PROVIDERS.keys()),
                    "note": "用 auth_action=login provider=<name> 登录新源. "
                            "crawl_web engine=direct 会自动检测 URL 域名复用对应 session.",
                },
                success=True,
            )
        if args.auth_action == "login":
            if not args.provider:
                return ToolResult(
                    data=None, success=False,
                    error="auth_action=login 需要 provider 字段. 可选: "
                          + ", ".join(_PROVIDERS.keys()),
                )
            try:
                res = await _auth_login(args.provider)
            except Exception as exc:
                return ToolResult(
                    data=None, success=False,
                    error=f"auth login 失败: {exc}",
                )
            return ToolResult(
                data={
                    "action": "crawl_web",
                    "auth_action": "login",
                    **res,
                    "note": (
                        "profile 已存盘. 之后 crawl_web engine=direct url=<该源页面> "
                        "会自动 headless 复用此 profile. session 过期会提示重新 login."
                    ) if res["result"] == "success" else
                    "登录未确认成功, profile 仍存盘可试. 换 result 字段看原因.",
                },
                success=True,
            )
        if args.auth_action == "logout":
            if not args.provider:
                return ToolResult(
                    data=None, success=False,
                    error="auth_action=logout 需要 provider 字段",
                )
            try:
                res = await _auth_logout(args.provider)
            except Exception as exc:
                return ToolResult(
                    data=None, success=False,
                    error=f"auth logout 失败: {exc}",
                )
            return ToolResult(
                data={"action": "crawl_web", "auth_action": "logout", **res},
                success=True,
            )

        engine = args.engine
        url = (args.url or "").strip()
        query = (args.query or "").strip()

        # 模式判定
        if engine == "direct":
            if not url:
                return ToolResult(
                    data=None, success=False,
                    error="crawl_web engine=direct 需要 url",
                )
            return await crawl_direct(url, args.max_results)
        # 搜索引擎模式
        if not query:
            return ToolResult(
                data=None, success=False,
                error=f"crawl_web engine={engine} 需要 query",
            )
        return await crawl_search_engine(engine, query, args.max_results, context)

    # ── extract_figures ────────────────────────────────────

    async def _do_extract_figures(
        self, args: LiteratureInput, context: ToolContext
    ) -> ToolResult:
        """从论文 PDF 中提取所有嵌入图片, 逐张调 image_analysis_tool 做 plot_extract.

        流程: 多源下载 PDF -> PyMuPDF 抽 image xobjects -> 存临时文件 ->
        调 image_analysis_tool plot_extract 分析每张图.
        """
        pdf_bytes, used_url, tried = await self._download_pdf_bytes(args)
        if pdf_bytes is None:
            if not tried:
                return ToolResult(
                    data=None, success=False,
                    error="无法解析 PDF URL, 需要 arxiv_id/doi/url 或 paper dict (含 url/doi/oa_url)",
                )
            last_status = tried[-1]["status"] if tried else ""
            return ToolResult(
                data=None, success=False,
                error=f"所有 PDF 候选源都失败 (试了 {len(tried)} 个). 最后错误: {last_status[:200]}",
            )

        try:
            import fitz  # PyMuPDF
        except ImportError:
            return ToolResult(
                data=None, success=False,
                error="PyMuPDF (fitz) 未安装, 无法提取 PDF 图片. pip install pymupdf",
            )

        import tempfile
        from pathlib import Path

        tmp_dir = Path(tempfile.mkdtemp(prefix="lit_figures_"))

        # 用 with 保证异常路径也释放文件句柄
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            extracted: list[dict[str, Any]] = []
            for page_num, page in enumerate(doc):
                for img_index, img in enumerate(page.get_images(full=True)):
                    xref = img[0]
                    try:
                        base_image = doc.extract_image(xref)
                    except Exception as exc:
                        logger.warning("extract_image xref=%s 失败: %s", xref, exc)
                        continue
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    img_filename = f"page{page_num + 1}_img{img_index + 1}.{image_ext}"
                    img_path = tmp_dir / img_filename
                    img_path.write_bytes(image_bytes)
                    extracted.append({
                        "page": page_num + 1,
                        "index": img_index + 1,
                        "path": str(img_path),
                        "ext": image_ext,
                        "size_bytes": len(image_bytes),
                    })

        if not extracted:
            return ToolResult(
                data={
                    "action": "extract_figures",
                    "pdf_url": used_url,
                    "n_images": 0,
                    "images": [],
                    "analyses": [],
                    "message": "PDF 中未找到嵌入图片",
                },
                success=True,
            )

        # 逐张调 image_analysis_tool 做 plot_extract
        from huginn.tools.registry import ToolRegistry

        img_tool = ToolRegistry.get("image_analysis_tool")
        analyses: list[dict[str, Any]] = []
        if img_tool is not None:
            for fig in extracted:
                try:
                    result = await img_tool.call(
                        {
                            "image_path": fig["path"],
                            "action": "plot_extract",
                            "parameters": {},
                        },
                        context,
                    )
                    analyses.append({
                        "image": fig["path"],
                        "page": fig["page"],
                        "success": result.success,
                        "data": result.data if result.success else None,
                        "error": result.error if not result.success else None,
                    })
                except Exception as exc:
                    analyses.append({
                        "image": fig["path"],
                        "page": fig["page"],
                        "success": False,
                        "error": str(exc)[:200],
                    })
        else:
            logger.warning("image_analysis_tool 未注册, 跳过图片分析")

        return ToolResult(
            data={
                "action": "extract_figures",
                "pdf_url": used_url,
                "n_images": len(extracted),
                "images": extracted,
                "analyses": analyses,
                "tmp_dir": str(tmp_dir),
            },
            success=True,
        )

    # ── helpers ─────────────────────────────────────────────

    def _get_model(self, context: ToolContext) -> Any:
        from huginn.llm import get_model

        config = getattr(context, "config", None)
        return get_model(config=config, temperature=0.2, max_tokens=6000)

    async def _llm_invoke(self, model: Any, system_prompt: str, user_prompt: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        if hasattr(model, "ainvoke"):
            response = await model.ainvoke(messages)
        else:
            response = await asyncio.to_thread(model.invoke, messages)
        content = response.content if hasattr(response, "content") else str(response)
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        return content

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any] | None:
        text = text.strip()
        # 容忍 ```json 代码块
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start: end + 1])
                except json.JSONDecodeError:
                    return None
            return None

    @staticmethod
    def _to_bibtex(papers: list[dict[str, Any]]) -> str:
        """把 paper 列表转成 BibTeX 字符串. cite key 用 firstauthor+year."""
        lines: list[str] = []
        for i, p in enumerate(papers, 1):
            authors = p.get("authors") or []
            first_author = authors[0].split()[-1] if authors else "unknown"
            year = p.get("year") or "nd"
            cite_key = f"{first_author.lower()}{year}"
            if len(papers) > 1:
                cite_key += f"_{i}"
            title = p.get("title", "").replace("{", "").replace("}", "")
            venue = p.get("venue", "")
            doi = p.get("doi") or ""
            url = p.get("url", "")
            lines.append(
                f"@article{{{cite_key},\n"
                f"  title = {{{title}}},\n"
                f"  author = {{{' and '.join(authors)}}},\n"
                f"  year = {{{year}}},\n"
                f"  journal = {{{venue}}},\n"
                + (f"  doi = {{{doi}}},\n" if doi else "")
                + (f"  url = {{{url}}},\n" if url else "")
                + "}\n"
            )
        return "\n".join(lines)

    def estimate_cost(self, args: LiteratureInput) -> dict[str, float] | None:
        if args.action == "search":
            return {"cpu_hours": 0.0, "walltime_hours": 0.01}
        if args.action == "fetch_pdf":
            # 多源候选轮询, 可能试 4-5 个 URL 才成
            return {"cpu_hours": 0.0, "walltime_hours": 0.08}
        if args.action == "citations":
            return {"cpu_hours": 0.0, "walltime_hours": 0.02}
        if args.action == "ingest_to_rag":
            return {"cpu_hours": 0.0, "walltime_hours": 0.05}  # embedding 调用
        if args.action == "crawl_web":
            if args.auth_action == "login":
                # 非 headless 等用户操作, 上限 5min
                return {"cpu_hours": 0.0, "walltime_hours": 0.3}
            if args.auth_action in ("status", "logout"):
                return {"cpu_hours": 0.0, "walltime_hours": 0.01}
            # crawl4ai 起 Playwright 浏览器, 比 API 慢得多
            return {"cpu_hours": 0.0, "walltime_hours": 0.15}
        if args.action == "extract_figures":
            # 下载 PDF + 提图 + 逐张调 image_analysis, 图多时偏慢
            return {"cpu_hours": 0.0, "walltime_hours": 0.2}
        return {"cpu_hours": 0.0, "walltime_hours": 0.02}  # summarize/benchmark_lookup 调 LLM
