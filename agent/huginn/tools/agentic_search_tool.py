"""Agentic search tool — multi-hop web research in one tool call.

web_search_tool does one query and returns snippets. crawl_web crawls one URL
or one search-engine page. Neither walks the web: the agent has to call them
repeatedly, read results, decide which links to follow, and stitch findings
together itself. That burns turns and context.

This tool does the whole loop internally:

    question
      -> search (reuses WebSearchTool's Tavily/DDG/urllib stack)
      -> fetch top-N page bodies (lightweight urllib + tag strip)
      -> extract passages most relevant to the question
      -> (hop 2+) derive follow-up queries from what was found, re-search
      -> synthesise a cited answer

Output is a single ToolResult with a synthesised answer string + a findings
list (url, title, passages, hop) so the caller can cite sources.

The search and fetch steps are injectable (__init__(searcher=, fetcher=)) so
tests run hermetically without touching the network. Defaults reuse
WebSearchTool and a bare urllib fetcher.

Cost tier is "none" (not a CPU/GPU-heavy simulation) — gated by phase
(LITERATURE / OPEN) and the autoloop progressive budget instead.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
import re
import socket
import urllib.parse
import urllib.request
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.core_types import ToolContext, ToolResult
from huginn.phases import ResearchPhase
from huginn.tools.base import HuginnTool
from huginn.tools.profile import ToolProfile

logger = logging.getLogger(__name__)

# 离线/CI 环境直接短路, 避免多跳 × 多次 fetch 累计 timeout
_AGENTIC_DISABLED_HINT = (
    "agentic_search disabled by HUGINN_DISABLE_WEB_SEARCH. "
    "Try materials_database_tool / rag_tool for local data."
)

# 停用词: 英文常见词 + 中文单字虚词. 多字中文停用词单独放一组, 抽词前先删.
_STOP_WORDS: set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "of", "in",
    "on", "for", "to", "and", "or", "not", "with", "by", "how", "what",
    "when", "why", "which", "that", "this", "from", "as", "at", "it",
    "again", "more", "also", "just", "very", "much", "many", "some",
    "的", "了", "是", "在", "和", "与", "及", "或", "请",
}
_CN_MULTI_STOP: set[str] = {
    "如何", "什么", "为什么", "怎么", "哪些", "哪个", "帮我", "请问",
}

# HTML 标签清洗: 去掉 script/style 整块, 再剥所有标签, 压空白
_SCRIPT_STYLE_RE = re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _html_to_text(html: str) -> str:
    """粗暴但够用: 砍 script/style, 剥标签, 压空白. 不做 DOM 解析避免依赖."""
    no_script = _SCRIPT_STYLE_RE.sub(" ", html)
    text = _TAG_RE.sub(" ", no_script)
    return _WS_RE.sub(" ", text).strip()


def _sentences(text: str) -> list[str]:
    """切句. 中英文句号/问号/感叹号都认, 丢掉太短的碎片.
    数字里的小数点不切 (1.12 不断句)."""
    raw = re.split(r"(?<!\d)[。.!?！？\n]+(?!\d)", text)
    return [s.strip() for s in raw if len(s.strip()) >= 3]


def _ordered_keywords(text: str) -> list[str]:
    """关键词按出现顺序排列, 去重, 去停用词.
    英文按词切 (len>=2), 中文按单字切 (每字都有意义)."""
    result: list[str] = []
    seen: set[str] = set()
    for t in re.findall(r"[A-Za-z][A-Za-z\-]+", text.lower()):
        if t not in _STOP_WORDS and len(t) >= 2 and t not in seen:
            seen.add(t)
            result.append(t)
    cleaned = text
    for sw in _CN_MULTI_STOP:
        cleaned = cleaned.replace(sw, "")
    for t in re.findall(r"[\u4e00-\u9fff]", cleaned):
        if t not in _STOP_WORDS and t not in seen:
            seen.add(t)
            result.append(t)
    return result


def _keywords(question: str) -> set[str]:
    """从问题里抽关键词. 去停用词, 小写. 返回无序集合."""
    return set(_ordered_keywords(question))


def _score_passage(passage: str, keywords: set[str]) -> int:
    """关键词命中数. 简单但够筛: 命中越多越相关."""
    low = passage.lower()
    return sum(1 for kw in keywords if kw.lower() in low)


def _cosine(a: list[float], b: list[float]) -> float:
    """向量余弦相似度. 空/维度不符返回 0."""
    import math

    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _lazy_embed_fn() -> Callable[[list[str]], list[list[float]]] | None:
    """复用 RAG 的 embedding 模型 (chroma DefaultEF ONNX), 未缓存则 None.

    语义打分是可选增强 (P1-6): embedding 模型没下载时安静降级为关键词打分,
    不阻塞抓取/抽取流程. 与 ``VectorStore._compute_embeddings`` 同一事实来源.
    """
    try:
        from huginn.rag.vector_store import _embedding_model_cached

        if not _embedding_model_cached():
            return None
        from chromadb.utils.embedding_functions import DefaultEF

        ef = DefaultEF()
        _ = ef(["probe"])  # 触发加载, 失败会抛异常

        def embed(texts: list[str]) -> list[list[float]]:
            return ef(texts)

        return embed
    except Exception:
        logger.debug("semantic embed unavailable, keyword scoring only", exc_info=True)
        return None


def _extract_relevant(
    text: str,
    question: str,
    max_passages: int = 3,
    min_score: int = 1,
    embed: Callable[[list[str]], list[list[float]]] | None = None,
) -> list[str]:
    """从正文里抽跟问题最相关的几句. 没有相关内容就返回空.

    ``embed`` 非 None 时叠加 embedding 语义相似度 (余弦 ×3) 与关键词命中 —
    能召回到"换了说法但同义"的句子. embedding 失败自动降级纯关键词.
    """
    kws = _keywords(question)
    sentences = _sentences(text)
    if not kws:
        # 关键词抽空了 (问题全是停用词), 退化为前几句
        return sentences[:max_passages]
    if embed is not None:
        try:
            qemb = embed([question])[0]
            sembs = embed(sentences)
            scored = [
                (s, _score_passage(s, kws) + _cosine(qemb, se) * 3.0)
                for s, se in zip(sentences, sembs)
            ]
            relevant = [(s, sc) for s, sc in scored if sc >= max(min_score, 0.5)]
            relevant.sort(key=lambda x: x[1], reverse=True)
            return [s for s, _ in relevant[:max_passages]]
        except Exception:
            logger.debug("semantic passage scoring failed, fallback to keyword", exc_info=True)
    scored = [
        (s, _score_passage(s, kws)) for s in sentences
    ]
    relevant = [(s, sc) for s, sc in scored if sc >= min_score]
    relevant.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in relevant[:max_passages]]


def _derive_followups(findings: list[dict[str, Any]], question: str, max_n: int = 3) -> list[str]:
    """从已有发现里派生下一跳的查询. 用 title 里的关键词 + 原问题重组.

    简单策略: 收集所有 title, 抽关键词, 跟原问题关键词拼出新 query.
    避免重复原问题. 用有序关键词保证 follow-up query 确定性.
    """
    if not findings:
        return []
    orig_kws_ordered = _ordered_keywords(question)
    seen_words: set[str] = set(orig_kws_ordered)
    followups: list[str] = []
    for f in findings:
        title = f.get("title", "")
        if not title:
            continue
        title_kws = _ordered_keywords(title)
        new_kws = [k for k in title_kws if k not in seen_words]
        if not new_kws:
            continue
        # 新词 + 原问题里的 1-2 个核心词组一个新查询
        new_words = new_kws[:3]
        orig_core = orig_kws_ordered[:2]
        q = " ".join(orig_core + new_words)
        if q and q not in followups:
            followups.append(q)
            seen_words.update(new_kws)
        if len(followups) >= max_n:
            break
    return followups


def _synthesize(findings: list[dict[str, Any]], question: str) -> str:
    """把发现拼成一段带引用的回答. 不调 LLM, 纯拼接, 让调用方自己精炼."""
    if not findings:
        return f"No relevant findings for: {question}"
    lines = [f"Agentic research on: {question}", ""]
    for i, f in enumerate(findings, 1):
        title = f.get("title") or "(untitled)"
        url = f.get("url", "")
        hop = f.get("hop", 0)
        passages = f.get("passages", [])
        lines.append(f"[{i}] (hop {hop}) {title}")
        lines.append(f"    {url}")
        for p in passages:
            lines.append(f"    - {p}")
        lines.append("")
    return "\n".join(lines).strip()


def _is_safe_url(url: str) -> bool:
    """Block non-public URLs to prevent SSRF."""
    try:
        parsed = urllib.parse.urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            return False
        try:
            ip = socket.getaddrinfo(hostname, None)[0][4][0]
            ip_obj = ipaddress.ip_address(ip)
            if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_reserved:
                return False
        except (socket.gaierror, ValueError):
            logger.debug("best-effort op failed", exc_info=True)
            return False
        return True
    except Exception:
        logger.debug("best-effort op failed", exc_info=True)
        return False


class AgenticSearchInput(BaseModel):
    action: Literal["research", "quick"] = Field(
        default="research",
        description='"research" = multi-hop; "quick" = single search, snippets only.',
    )
    question: str = Field(..., description="Research question to investigate.")
    max_hops: int = Field(default=2, ge=1, le=4, description="Max search hops (research only).")
    max_results_per_hop: int = Field(
        default=3, ge=1, le=8, description="Results to fetch per hop."
    )
    max_content_chars: int = Field(
        default=4000, ge=500, le=20000, description="Max chars to extract per page."
    )
    use_llm: bool = Field(
        default=False,
        description="True=用 LLM 做回答综合与下一跳查询派生 (失败自动降级为启发式). "
                    "False=纯启发式 (确定性/可复现, 测试友好).",
    )


class AgenticSearchTool(HuginnTool):
    """Multi-hop web research tool."""

    name = "agentic_search_tool"
    category = "search"
    description = (
        "Multi-hop web research: searches, reads top results, follows links, "
        "and synthesises a cited answer. Use for literature review or any "
        "question needing more than a single search."
    )
    read_only = True
    input_schema = AgenticSearchInput
    profile = ToolProfile(
        cost_tier="none",
        phases=frozenset({ResearchPhase.LITERATURE, ResearchPhase.OPEN}),
    )

    def __init__(
        self,
        searcher: Callable[[str, int], Awaitable[list[dict[str, Any]]]] | None = None,
        fetcher: Callable[[str, int], Awaitable[str | None]] | None = None,
        synthesizer: Callable[..., Awaitable[str]] | None = None,
        followup_deriver: Callable[..., Awaitable[list[str]]] | None = None,
    ) -> None:
        # 注入点: 测试塞 mock, 默认走 web_search_tool + urllib + 启发式综合/派生
        self._searcher = searcher
        self._fetcher = fetcher
        self._synthesizer = synthesizer
        self._followup_deriver = followup_deriver
        # P1-6: 懒加载语义 embedding (模型未缓存则 None, 安静降级关键词打分)
        self._embed = None

    # ── defaults: real network backends ───────────────────────────────

    async def _default_search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        from huginn.tools.web_search_tool import WebSearchTool

        ws = WebSearchTool()
        res = await ws.call(
            {"query": query, "max_results": max_results},
            ToolContext(session_id="agentic_search", workspace=os.getcwd()),
        )
        if not res.success or not res.data:
            return []
        return res.data.get("results", [])

    async def _default_fetch(self, url: str, max_chars: int) -> str | None:
        if not _is_safe_url(url):
            return None
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0 (compatible; HuginnAgent/1.0)"}
            )
            with urllib.request.urlopen(req, timeout=15.0) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
            text = _html_to_text(html)
            return text[:max_chars] if text else None
        except Exception as exc:
            logger.debug("agentic_search fetch failed for %s: %s", url, exc)
            return None

    async def _search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        fn = self._searcher if self._searcher is not None else self._default_search
        return await fn(query, max_results)

    async def _fetch(self, url: str, max_chars: int) -> str | None:
        fn = self._fetcher if self._fetcher is not None else self._default_fetch
        return await fn(url, max_chars)

    # ── P1-5: LLM 综合 + LLM 派生下一跳 (use_llm=True 启用, 失败降级启发式) ──

    def _get_model(self, context: ToolContext | None) -> Any:
        from huginn.llm import get_model

        config = getattr(context, "config", None)
        return get_model(config=config, temperature=0.2, max_tokens=2000)

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

    async def _llm_synthesize(
        self, findings: list[dict[str, Any]], question: str, context: ToolContext | None
    ) -> str:
        """LLM 结构化综合: 关键发现 / 矛盾点 / 证据链+引用. 失败降级纯拼接."""
        if not findings:
            return _synthesize(findings, question)
        evidence = "\n\n".join(
            f"[{i}] ({f.get('title', '')}) {f.get('url', '')}\n"
            + "\n".join(f"- {p}" for p in f.get("passages", []))
            for i, f in enumerate(findings, 1)
        )
        try:
            model = self._get_model(context)
        except Exception as exc:
            logger.debug("llm synthesize skipped (no model): %s", exc)
            return _synthesize(findings, question)
        try:
            sys_p = (
                "你是研究综合助手. 基于多来源证据写一段结构化回答, 输出 Markdown:"
                "\n## 关键发现\n- ... [源编号]\n## 矛盾点 / 证据缺口\n- ... [源编号]"
                "\n## 结论\n- ...\n引用只允许用证据里的 [编号], 不要编造来源."
            )
            user_p = f"研究问题: {question}\n\n证据列表:\n{evidence}"
            content = await self._llm_invoke(model, sys_p, user_p)
            return (content or "").strip() or _synthesize(findings, question)
        except Exception as exc:
            logger.debug("llm synthesize failed, fallback to concatenation: %s", exc)
            return _synthesize(findings, question)

    async def _llm_derive_followups(
        self, findings: list[dict[str, Any]], question: str, max_n: int,
        context: ToolContext | None,
    ) -> list[str]:
        """LLM 决定下一跳查什么. 失败/空结果降级回关键词派生."""
        if not findings:
            return _derive_followups(findings, question, max_n)
        titles = "\n".join(
            f"- {f.get('title', '')}: {' '.join(f.get('passages', []))[:200]}"
            for f in findings
        )
        try:
            model = self._get_model(context)
        except Exception as exc:
            logger.debug("llm followups skipped (no model): %s", exc)
            return _derive_followups(findings, question, max_n)
        try:
            sys_p = (
                f"你是检索规划助手. 基于已有发现, 给出最多 {max_n} 个下一跳搜索查询, "
                "补足未知/矛盾点, 别重复已覆盖的内容.\n"
                f'只输出 JSON: {{"queries": ["q1", "q2"]}}'
            )
            user_p = f"原问题: {question}\n\n已有发现:\n{titles}"
            content = await self._llm_invoke(model, sys_p, user_p)
            parsed = _parse_json_text(content)
            queries = [q for q in (parsed.get("queries", []) if parsed else []) if isinstance(q, str) and q.strip()]
            if queries:
                return queries[:max_n]
        except Exception as exc:
            logger.debug("llm followups failed, fallback to keyword: %s", exc)
        return _derive_followups(findings, question, max_n)

    async def _synthesize_answer(
        self, findings: list[dict[str, Any]], question: str,
        inp: AgenticSearchInput, context: ToolContext | None,
    ) -> str:
        """回答综合: 注入的 synthesizer > use_llm 的 LLM 综合 > 启发式拼接."""
        if self._synthesizer is not None:
            return await self._synthesizer(findings, question, context)
        if inp.use_llm:
            return await self._llm_synthesize(findings, question, context)
        return _synthesize(findings, question)

    async def _derive_next_queries(
        self, findings: list[dict[str, Any]], question: str, max_n: int,
        inp: AgenticSearchInput, context: ToolContext | None,
    ) -> list[str]:
        """下一跳查询派生: 注入的 followup_deriver > use_llm 的 LLM 派生 > 关键词派生."""
        if self._followup_deriver is not None:
            return await self._followup_deriver(findings, question, max_n, context)
        if inp.use_llm:
            return await self._llm_derive_followups(findings, question, max_n, context)
        return _derive_followups(findings, question, max_n)

    # ── entry ──────────────────────────────────────────────────────────

    async def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = AgenticSearchInput(**args)  # schema 错误冒泡, 不吞
        if os.environ.get("HUGINN_DISABLE_WEB_SEARCH", "").lower() in (
            "1", "true", "yes", "on",
        ):
            return ToolResult(
                data={"question": input_data.question, "findings": [], "answer": ""},
                success=False,
                error=_AGENTIC_DISABLED_HINT,
            )
        try:
            if input_data.action == "quick":
                return await self._quick(input_data, context)
            return await self._research(input_data, context)
        except Exception as exc:
            return ToolResult(
                data={"question": input_data.question, "findings": [], "answer": ""},
                success=False,
                error=f"agentic_search failed: {exc}",
            )

    # ── actions ────────────────────────────────────────────────────────

    async def _quick(
        self, inp: AgenticSearchInput, context: ToolContext | None = None
    ) -> ToolResult:
        results = await self._search(inp.question, inp.max_results_per_hop)
        findings = [
            {
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "passages": [r.get("snippet", "")] if r.get("snippet") else [],
                "hop": 0,
            }
            for r in results
        ]
        answer = await self._synthesize_answer(findings, inp.question, inp, context)
        return ToolResult(
            data={
                "action": "quick",
                "question": inp.question,
                "n_findings": len(findings),
                "findings": findings,
                "answer": answer,
            },
            success=True,
        )

    async def _research(
        self, inp: AgenticSearchInput, context: ToolContext | None = None
    ) -> ToolResult:
        findings: list[dict[str, Any]] = []
        visited: set[str] = set()
        queries: list[str] = [inp.question]
        # P1-6: 懒加载语义 embedding (模型未缓存则为 None)
        if self._embed is None:
            self._embed = _lazy_embed_fn()

        for hop in range(inp.max_hops):
            new_findings: list[dict[str, Any]] = []
            for q in queries:
                results = await self._search(q, inp.max_results_per_hop)
                for r in results:
                    url = r.get("url", "")
                    if not url or url in visited:
                        continue
                    visited.add(url)
                    content = await self._fetch(url, inp.max_content_chars)
                    passages = (
                        _extract_relevant(content, inp.question, embed=self._embed)
                        if content else
                        ([r.get("snippet", "")] if r.get("snippet") else [])
                    )
                    if not passages:
                        continue
                    new_findings.append({
                        "url": url,
                        "title": r.get("title", ""),
                        "passages": passages,
                        "hop": hop,
                    })
            findings.extend(new_findings)
            # 下一跳查询: 从本轮发现派生. 没有新查询就停, 不硬撑 max_hops.
            queries = await self._derive_next_queries(
                new_findings, inp.question, inp.max_results_per_hop, inp, context
            )
            if not queries:
                break

        answer = await self._synthesize_answer(findings, inp.question, inp, context)
        return ToolResult(
            data={
                "action": "research",
                "question": inp.question,
                "n_hops": min(inp.max_hops, hop + 1) if findings else 0,
                "n_findings": len(findings),
                "findings": findings,
                "answer": answer,
            },
            success=True,
        )


def _parse_json_text(text: str) -> dict[str, Any] | None:
    """容错 JSON 解析 (容忍 ```json 代码块 / 首尾花括号外噪声)."""
    import json

    text = text.strip()
    if text.startswith("```"):
        import re as _re

        text = _re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = _re.sub(r"\n?```\s*$", "", text)
    try:
        return json.loads(text)
    except Exception:
        import re as _re

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start: end + 1])
            except Exception:
                return None
        return None


__all__ = ["AgenticSearchTool", "AgenticSearchInput"]
