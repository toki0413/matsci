"""联网搜索工具 —— 支持 Tavily / DuckDuckGo 多后端降级。

只读工具，无副作用，可以自动执行。
后端优先级：
  1. Tavily（需要 TAVILY_API_KEY）
  2. duckduckgo_search 库（免费）
  3. urllib 直接抓 DuckDuckGo HTML（最后兜底）
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import urllib.parse
import urllib.request
from typing import Any

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult

logger = logging.getLogger(__name__)


def _search_timeout() -> float:
    """网络请求超时 (秒). 默认 15s, 用 HUGINN_WEB_SEARCH_TIMEOUT 覆盖.

    网络不通的环境里 15s × N 次累计延迟很可观, 调小可以快速 fail;
    网络好但响应慢时可以调大. 0 或负数走默认.
    """
    raw = os.environ.get("HUGINN_WEB_SEARCH_TIMEOUT", "15")
    try:
        v = float(raw)
        return v if v > 0 else 15.0
    except (TypeError, ValueError):
        return 15.0


def _web_search_disabled() -> bool:
    """离线 / CI 环境里直接禁用 web_search, 避免 15s × N 次兜底 timeout 累计."""
    return os.environ.get("HUGINN_DISABLE_WEB_SEARCH", "").lower() in (
        "1", "true", "yes", "on",
    )


# 网络不通时给 LLM 的引导, 避免它死磕 web_search 反复重试
_SEARCH_DISABLED_HINT = (
    "web_search disabled by HUGINN_DISABLE_WEB_SEARCH. "
    "Try materials_database_tool / rag_tool / code_tool for local data."
)
_SEARCH_FAIL_HINT = (
    "all search backends failed (network issue). "
    "Consider materials_database_tool / rag_tool / code_tool instead of retrying."
)


class WebSearchInput(BaseModel):
    query: str = Field(..., description="Search query string")
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of results to return (default 5)",
    )


class WebSearchTool(HuginnTool):
    """联网搜索工具，支持网页搜索和内容提取。"""

    # 搜索只读，没有副作用，可以直接自动执行
    category = "search"
    read_only = True

    input_schema = WebSearchInput

    @property
    def name(self) -> str:
        return "web_search_tool"

    @property
    def description(self) -> str:
        return (
            "Search the web for up-to-date information. "
            "Returns search results with titles, URLs, and snippets. "
            "Use for finding recent papers, material properties, "
            "or any information not in local knowledge base."
        )

    def get_schema(self) -> dict:
        """返回工具参数的 JSON schema。"""
        return WebSearchInput.model_json_schema()

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        """执行搜索。按优先级降级：Tavily → duckduckgo_search → urllib。"""
        query = (args.get("query") or "").strip()
        if not query:
            return ToolResult(
                data={"error": "query is required", "results": []},
                success=False,
                error="query is required",
            )
        # 离线 / CI 环境: 直接返回, 不走任何后端, 避免兜底 timeout 累计
        if _web_search_disabled():
            return ToolResult(
                data={
                    "query": query,
                    "results": [],
                    "error": "web_search disabled",
                    "hint": _SEARCH_DISABLED_HINT,
                    "search_engine": "disabled",
                },
                success=False,
                error=_SEARCH_DISABLED_HINT,
            )
        try:
            max_results = int(args.get("max_results", 5) or 5)
        except (TypeError, ValueError):
            max_results = 5

        # 1) Tavily（有 API key 才走这条）
        if os.environ.get("TAVILY_API_KEY"):
            result = self._search_tavily(query, max_results)
            if result is not None:
                return result
            # Tavily 挂了就继续往下走

        # 2) duckduckgo_search 库
        result = self._search_ddgs(query, max_results)
        if result is not None:
            return result

        # 3) urllib 兜底
        return self._search_fallback(query, max_results)

    async def call(self, args: dict, context: ToolContext) -> ToolResult:
        """HuginnTool 入口。放到线程池里跑，避免阻塞事件循环。"""
        return await asyncio.to_thread(self.execute, args, context)

    # ── Tavily ───────────────────────────────────────────────────────

    def _search_tavily(
        self, query: str, max_results: int
    ) -> ToolResult | None:
        try:
            from tavily import TavilyClient
        except ImportError:
            return None

        try:
            client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
            response = client.search(query, max_results=max_results)
            results = []
            for item in response.get("results", []):
                results.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "snippet": item.get("content", ""),
                    }
                )
            return ToolResult(
                data={
                    "query": query,
                    "results": results,
                    "search_engine": "tavily",
                },
                success=True,
            )
        except Exception as exc:
            logger.warning("Tavily 搜索失败，降级到下一个后端: %s", exc)
            return None

    # ── duckduckgo_search ───────────────────────────────────────────

    def _search_ddgs(
        self, query: str, max_results: int
    ) -> ToolResult | None:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return None

        try:
            results = []
            # 新旧版本字段名不太一样，都兼容一下
            with DDGS() as ddgs:
                for item in ddgs.text(query, max_results=max_results):
                    results.append(
                        {
                            "title": item.get("title", ""),
                            "url": item.get("href", "") or item.get("url", ""),
                            "snippet": item.get("body", "") or item.get("content", ""),
                        }
                    )
            return ToolResult(
                data={
                    "query": query,
                    "results": results,
                    "search_engine": "duckduckgo",
                },
                success=True,
            )
        except Exception as exc:
            logger.warning("duckduckgo_search 失败，降级到 urllib: %s", exc)
            return None

    # ── urllib 兜底（直接抓 DDG HTML）──────────────────────────────

    def _search_fallback(
        self, query: str, max_results: int
    ) -> ToolResult:
        try:
            url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; HuginnAgent/1.0)"
                },
            )
            with urllib.request.urlopen(req, timeout=_search_timeout()) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
            results = self._parse_ddg_html(html, max_results)
            return ToolResult(
                data={
                    "query": query,
                    "results": results,
                    "search_engine": "fallback",
                },
                success=True,
            )
        except Exception as exc:
            logger.warning("urllib 兜底搜索也失败了: %s", exc)
            # 加 hint 引导 LLM 换路径, 不然它看到 error 会反复重试 web_search,
            # 每次 15s 兜底 timeout 累计大量延迟 (v14 V1 一次测试调 20 次全失败 = 300s+)
            return ToolResult(
                data={
                    "query": query,
                    "results": [],
                    "error": str(exc),
                    "hint": _SEARCH_FAIL_HINT,
                    "search_engine": "fallback",
                },
                success=False,
                error=_SEARCH_FAIL_HINT,
            )

    @staticmethod
    def _parse_ddg_html(html: str, max_results: int) -> list[dict[str, Any]]:
        """从 DuckDuckGo HTML 结果页里抠出标题/链接/摘要。

        结构比较脆，DDG 改版可能就要跟着改，但作为最后兜底够用了。
        """
        # 标题和链接在 result__a 这个锚点里
        link_re = re.compile(
            r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            re.DOTALL,
        )
        # 摘要单独拎出来，标签可能是 a/td/span
        snippet_re = re.compile(
            r'class="result__snippet"[^>]*>(.*?)</(?:a|td|span|div)>',
            re.DOTALL,
        )

        links = link_re.findall(html)
        snippets = snippet_re.findall(html)

        results: list[dict[str, Any]] = []
        for i, (href, title) in enumerate(links[:max_results]):
            clean_title = re.sub(r"<[^>]+>", "", title).strip()
            snippet = ""
            if i < len(snippets):
                snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip()
            # DDG 的链接会包一层 redirect，真实地址在 uddg= 参数里
            real_url = href
            m = re.search(r"[?&]uddg=([^&]+)", href) or re.search(
                r"[?&]u=([^&]+)", href
            )
            if m:
                real_url = urllib.parse.unquote(m.group(1))
            results.append(
                {
                    "title": clean_title,
                    "url": real_url,
                    "snippet": snippet,
                }
            )
        return results
