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

from huginn.core_types import ToolContext, ToolResult
from huginn.tools.base import HuginnTool, ToolProfile

logger = logging.getLogger(__name__)

# DDG HTML 解析正则; 提到模块级避免每次重编译
_DDG_LINK_RE = re.compile(
    r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
    re.DOTALL,
)
# 摘要单独拎出来, 标签可能是 a/td/span
_DDG_SNIPPET_RE = re.compile(
    r'class="result__snippet"[^>]*>(.*?)</(?:a|td|span|div)>',
    re.DOTALL,
)


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
        logger.debug("best-effort op failed", exc_info=True)
        return 15.0


def _web_search_disabled() -> bool:
    """离线 / CI 环境里直接禁用 web_search, 避免 15s × N 次兜底 timeout 累计."""
    return os.environ.get("HUGINN_DISABLE_WEB_SEARCH", "").lower() in (
        "1", "true", "yes", "on",
    )


def _is_ssrf_blocked(url: str) -> tuple[bool, str]:
    """SSRF 防护: 拦截指向内网/本机的 fetch 请求.

    只放行 http(s), 解析主机名拿到 IP 后拒绝 loopback / 私有 / 链路本地 /
    保留地址. DNS 解析存在 TOCTOU (解析时 OK、请求时变), 这里做第一道防线
    已经挡住绝大多数内网探测; 严格环境应在出网代理侧再校验一次.
    """
    import ipaddress
    import socket

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return True, f"scheme '{parsed.scheme}' not allowed"
    host = parsed.hostname
    if not host:
        return True, "missing host"
    # 字面量 IP 直接判, 主机名走 DNS 解析
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return True, f"cannot resolve host '{host}'"
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return True, f"host '{host}' resolves to non-public {ip}"
    return False, ""


# 网络不通时给 LLM 的引导, 避免它死磕 web_search 反复重试
_SEARCH_DISABLED_HINT = (
    "web_search disabled by HUGINN_DISABLE_WEB_SEARCH. "
    "Try materials_database_tool / rag_tool / code_tool for local data."
)
_SEARCH_FAIL_HINT = (
    "all search backends failed (network issue). "
    "Consider materials_database_tool / rag_tool / code_tool instead of retrying."
)


def web_search_health_check(timeout: float = 8.0) -> tuple[bool, str]:
    """C4: bench 前检索健康检查 — 轻量探测 arxiv API (免费/稳定/无 rate limit).

    返回 (ok, msg). ok=True 说明至少一个后端可达, agent 检索链能用.
    ok=False 时 bench 仍可跑, 但 agent 检索会走降级路径 (materials_database_tool 等).

    ponytail: 只测 arxiv — Tavily 要 key, DDG 有 rate limit, 频繁探测会被 ban.
      arxiv API 无 key 无 limit, 是最稳的探测点.
      ceiling: arxiv 可达不代表 DDG/Tavily 可达, 但 arxiv 挂了基本等于断网.
    """
    if _web_search_disabled():
        return False, "web_search disabled by HUGINN_DISABLE_WEB_SEARCH"
    try:
        url = "http://export.arxiv.org/api/query?search_query=all:test&max_results=1"
        req = urllib.request.Request(url, headers={"User-Agent": "huginn-healthcheck/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            _ = resp.read(512)  # 只读前 512 字节, 够确认连通
        return True, "arxiv API reachable"
    except Exception as exc:
        return False, f"arxiv API unreachable: {exc}"


class WebSearchInput(BaseModel):
    action: str = Field(
        default="search",
        description="search | fetch. search 跑搜索; fetch 抓取单个 URL 正文",
    )
    query: str = Field(default="", description="搜索 query (action=search 时用)")
    url: str = Field(default="", description="要抓取的页面 URL (action=fetch 时用)")
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of results to return (default 5)",
    )
    compact: bool = Field(
        default=True,
        description=(
            "True: 返回索引化文本块 [0] 标题: 摘要..., 后续用 [0]/[1] 引用; "
            "False: 返回原始结构化结果"
        ),
    )


class WebSearchTool(HuginnTool):
    """联网搜索工具，支持网页搜索和内容提取。"""

    # 搜索只读，没有副作用，可以直接自动执行
    category = "search"
    read_only = True
    profile = ToolProfile(cost_tier="light")

    input_schema = WebSearchInput

    # 失败熔断: 同一 session 连续失败 N 次后直接拒绝, 避免死循环重试
    _MAX_CONSECUTIVE_FAILURES = 3
    _consecutive_failures: int = 0
    _circuit_broken: bool = False

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
        """执行搜索/抓取。action=search 按优先级降级：Tavily → arxiv → duckduckgo_search → urllib。

        compact=True (默认) 时把结果压成索引化文本块, 方便 LLM 用 [0]/[1] 引用。
        连续失败 3 次后熔断, 直接返回错误, 避免死循环重试。
        """
        action = (args.get("action") or "search").strip().lower()

        # 熔断检查: 连续失败太多直接拒绝
        if self._circuit_broken:
            return ToolResult(
                data={
                    "error": "web_search circuit broken (连续失败过多)",
                    "hint": "搜索不可用, 请用 code_tool / materials_database_tool / 已有知识回答.",
                    "results": [],
                    # P1-C4: 显式不可用信号, agent 据此切换策略而非死循环重试
                    "search_unavailable": True,
                },
                success=False,
                error="web_search circuit broken",
            )

        if action == "fetch":
            return self._fetch(args)

        # ── search ──
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
            logger.debug("best-effort op failed", exc_info=True)
            max_results = 5

        # 1) Tavily（有 API key 才走这条）
        if os.environ.get("TAVILY_API_KEY"):
            result = self._search_tavily(query, max_results)
            if result is not None:
                self._on_success()
                return self._maybe_compact(result, args)

        # 2) arxiv API — 学术搜索, 稳定免费, 对 paper/physics query 尤其有效
        result = self._search_arxiv(query, max_results)
        if result is not None:
            self._on_success()
            return self._maybe_compact(result, args)

        # 3) duckduckgo_search 库
        result = self._search_ddgs(query, max_results)
        if result is not None:
            self._on_success()
            return self._maybe_compact(result, args)

        # 4) urllib 兜底 — 到这里说明前面全失败
        self._on_failure()
        return self._maybe_compact(self._search_fallback(query, max_results), args)

    @classmethod
    def _on_success(cls) -> None:
        """搜索成功, 重置失败计数."""
        cls._consecutive_failures = 0
        cls._circuit_broken = False

    @classmethod
    def _on_failure(cls) -> None:
        """搜索失败, 累计计数, 达到阈值后熔断."""
        cls._consecutive_failures += 1
        if cls._consecutive_failures >= cls._MAX_CONSECUTIVE_FAILURES:
            cls._circuit_broken = True
            logger.warning(
                "web_search 熔断: 连续失败 %d 次, 后续调用直接拒绝",
                cls._consecutive_failures,
            )

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
            logger.debug("best-effort op failed", exc_info=True)
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

    # ── arxiv API ────────────────────────────────────────────────────

    def _search_arxiv(
        self, query: str, max_results: int
    ) -> ToolResult | None:
        """arxiv API 搜索, 返回 None 表示无结果/失败, 降级到下一个后端.

        arxiv API 稳定免费, 无需 key, 对论文/物理/材料 query 尤其有效.
        对非学术 query (如"铜的电导率") 通常返回空, 自然降级到 DDG.
        """
        try:
            import xml.etree.ElementTree as ET
        except ImportError:
            logger.debug("best-effort op failed", exc_info=True)
            return None

        try:
            api_url = (
                f"http://export.arxiv.org/api/query?"
                f"search_query=all:{urllib.parse.quote(query)}"
                f"&max_results={max_results}"
            )
            req = urllib.request.Request(
                api_url,
                headers={"User-Agent": "HuginnAgent/1.0"},
            )
            with urllib.request.urlopen(req, timeout=_search_timeout()) as resp:
                xml_data = resp.read().decode("utf-8", errors="ignore")

            root = ET.fromstring(xml_data)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            results: list[dict[str, Any]] = []
            for entry in root.findall("atom:entry", ns):
                title_el = entry.find("atom:title", ns)
                summary_el = entry.find("atom:summary", ns)
                id_el = entry.find("atom:id", ns)
                title = (title_el.text or "").strip().replace("\n", " ") if title_el is not None else ""
                summary = (summary_el.text or "").strip().replace("\n", " ") if summary_el is not None else ""
                arxiv_id = (id_el.text or "").strip() if id_el is not None else ""
                if len(summary) > 300:
                    summary = summary[:297] + "..."
                results.append({
                    "title": title,
                    "url": arxiv_id,
                    "snippet": summary,
                })
            if not results:
                return None
            return ToolResult(
                data={
                    "query": query,
                    "results": results,
                    "search_engine": "arxiv",
                },
                success=True,
            )
        except Exception as exc:
            logger.warning("arxiv 搜索失败, 降级到下一个后端: %s", exc)
            return None

    # ── duckduckgo_search ───────────────────────────────────────────

    def _search_ddgs(
        self, query: str, max_results: int
    ) -> ToolResult | None:
        # duckduckgo_search 已重命名为 ddgs, 两个都试一下
        DDGS = None
        try:
            from ddgs import DDGS
        except ImportError:
            try:
                from duckduckgo_search import DDGS
            except ImportError:
                logger.debug("best-effort op failed", exc_info=True)
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
            logger.warning("ddgs/duckduckgo_search 失败，降级到 urllib: %s", exc)
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
        links = _DDG_LINK_RE.findall(html)
        snippets = _DDG_SNIPPET_RE.findall(html)

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

    # ── 索引化输出 (BrowserAct 启发) ──────────────────────────────────

    @staticmethod
    def _maybe_compact(result: ToolResult, args: dict) -> ToolResult:
        """compact=True 时把搜索结果压成 '[i] 标题: 摘要' 索引文本块.

        原始 results 仍保留 (但精简成 index/title/url, 去掉 snippet 省 token),
        后续 LLM/调用方可以用 [0]/[1] 直接引用某条结果. compact=False 时原样返回.
        """
        if not args.get("compact", True):
            return result
        # 浅拷一份, 不就地改调用方持有的 data
        data = dict(result.data) if isinstance(result.data, dict) else {}
        results = data.get("results") or []
        lines = []
        slim: list[dict[str, Any]] = []
        for i, item in enumerate(results):
            title = (item.get("title") or "").strip()
            snippet = (item.get("snippet") or "").strip()
            # 一行一条, 截断超长摘要避免刷屏
            if len(snippet) > 300:
                snippet = snippet[:297] + "..."
            lines.append(f"[{i}] {title}: {snippet}" if snippet else f"[{i}] {title}")
            slim.append({"index": i, "title": title, "url": item.get("url", "")})
        data["indexed"] = "\n".join(lines)
        data["results"] = slim
        data["n_results"] = len(slim)
        return ToolResult(
            data=data,
            success=result.success,
            error=result.error,
            new_messages=result.new_messages,
            side_effects=result.side_effects,
        )

    # ── fetch: 抓取单个页面, 切块索引 ────────────────────────────────

    def _fetch(self, args: dict) -> ToolResult:
        """下载一个 URL 的正文, 去标签后切成索引化文本块返回."""
        url = (args.get("url") or "").strip()
        if not url:
            return ToolResult(
                data={"error": "url is required for fetch", "chunks": []},
                success=False,
                error="url is required for fetch",
            )
        if _web_search_disabled():
            return ToolResult(
                data={"url": url, "chunks": [], "error": "web_search disabled",
                      "hint": _SEARCH_DISABLED_HINT},
                success=False, error=_SEARCH_DISABLED_HINT,
            )

        blocked, reason = _is_ssrf_blocked(url)
        if blocked:
            logger.warning("fetch blocked by SSRF guard: %s (%s)", url, reason)
            return ToolResult(
                data={"url": url, "chunks": [], "error": f"blocked: {reason}"},
                success=False, error=f"fetch blocked (SSRF guard): {reason}",
            )

        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; HuginnAgent/1.0)",
                    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
                },
            )
            with urllib.request.urlopen(req, timeout=_search_timeout()) as resp:
                raw = resp.read()
            # 大页面截断, 避免内存爆掉
            if len(raw) > 2_000_000:
                raw = raw[:2_000_000]
            charset = "utf-8"
            ctype = resp.headers.get("Content-Type", "")
            m = re.search(r"charset=([\w-]+)", ctype, re.IGNORECASE)
            if m:
                charset = m.group(1)
            html = raw.decode(charset, errors="ignore")
        except Exception as exc:
            logger.warning("fetch %s 失败: %s", url, exc)
            return ToolResult(
                data={"url": url, "chunks": [], "error": str(exc),
                      "hint": _SEARCH_FAIL_HINT},
                success=False, error=f"fetch failed: {exc}",
            )

        text = self._html_to_text(html)
        compact = args.get("compact", True)
        chunks = self._chunk_text(text, size=600)
        data: dict[str, Any] = {
            "url": url,
            "n_chunks": len(chunks),
            "title": self._extract_title(html),
        }
        if compact:
            # 索引化: [0] 文本块...
            data["indexed"] = "\n".join(f"[{i}] {c}" for i, c in enumerate(chunks))
            data["chunks"] = [{"index": i, "preview": c[:80]} for i, c in enumerate(chunks)]
        else:
            data["chunks"] = [{"index": i, "text": c} for i, c in enumerate(chunks)]
        return ToolResult(data=data, success=True)

    @staticmethod
    def _html_to_text(html: str) -> str:
        """去导航/页脚噪音 + 剥标签 + 解码实体, 拿到可读正文. 不做完整 DOM 解析, 够 LLM 读就行."""
        # 先抠 <script>/<style>/<noscript> 整块扔掉, 否则 JS 会污染正文
        html = re.sub(r"<(script|style|noscript)\b[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
        # 再抠导航/页脚/侧栏等噪音容器 (P1-6): 不承载正文, 还常把"下一页/相关链接"
        # 之类灌进正文, 去标签前整块扔掉
        html = re.sub(r"<(nav|header|footer|aside)\b[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
        # 块级标签换行
        html = re.sub(r"</(p|div|li|h[1-6]|tr|br)\s*>", "\n", html, flags=re.IGNORECASE)
        html = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
        # 去剩余标签
        text = re.sub(r"<[^>]+>", "", html)
        # 解码常见 HTML 实体
        text = (text.replace("&nbsp;", " ").replace("&amp;", "&")
                    .replace("&lt;", "<").replace("&gt;", ">")
                    .replace("&quot;", '"').replace("&#39;", "'"))
        # 压缩空白
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        return text.strip()

    @staticmethod
    def _extract_title(html: str) -> str:
        m = re.search(r"<title[^>]*>(.*?)</title>", html, re.DOTALL | re.IGNORECASE)
        if not m:
            return ""
        return re.sub(r"\s+", " ", m.group(1)).strip()

    @staticmethod
    def _chunk_text(text: str, size: int = 600) -> list[str]:
        """按段落/句子边界切块, 每块不超过 size 字符 (P1-6).

        旧实现按句号硬切, 会在段落中间断开, 后续检索/喂 LLM 会丢上下文.
        新实现: 段落是语义边界 — 段内句子拼到 size 附近; 段尾若已积攒一半以上
        size 就封块 (段落语义独立, 跨段拼块会混入无关上下文); 超长单句才硬切.
        """
        if not text:
            return []
        # _html_to_text 已把块级边界转成 \n, 按空行分段 (语义边界)
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
        if not paragraphs:
            return []
        chunks: list[str] = []
        cur = ""
        for para in paragraphs:
            # 段内按句子软拼
            sentences = [s.strip() for s in re.split(r"(?<=[。.!?])\s+", para) if s.strip()]
            if not sentences:
                sentences = [para]
            for s in sentences:
                if len(cur) + len(s) + 1 <= size:
                    cur = (cur + " " + s).strip()
                    continue
                if cur:
                    chunks.append(cur)
                    cur = ""
                # 超长单句硬切
                while len(s) > size:
                    chunks.append(s[:size])
                    s = s[size:]
                cur = s
            # 段落末尾: 段尾足量才封块, 否则继续拼下一段 (避免碎块)
            if cur and len(cur) >= size // 2:
                chunks.append(cur)
                cur = ""
        if cur:
            chunks.append(cur)
        return chunks


def _c4_self_check() -> int:
    """C4 self-check: 验证检索健康检查 + search_unavailable flag + 多源降级链.

    覆盖:
    1. web_search_health_check 返回 (bool, str) 元组, 离线模式返回 False
    2. 熔断时 execute 返回 search_unavailable=True
    3. 多源降级链存在 (Tavily/arxiv/ddgs/urllib 4 个 _search_* 方法存在)

    ponytail: 不真发网络请求 (CI 不稳定), 只验函数签名 + 离线模式 + 熔断 flag.
      ceiling: 不验真实网络连通性, 那是 integration test 的活.
    """
    import os as _os

    # 1. health_check 函数签名 + 离线模式
    _os.environ["HUGINN_DISABLE_WEB_SEARCH"] = "1"
    ok, msg = web_search_health_check(timeout=1.0)
    assert ok is False, f"disabled mode should return False, got {ok}"
    assert "disabled" in msg.lower(), f"msg should mention disabled, got: {msg}"
    print(f"[CHECK C4.1] health_check disabled mode OK: {msg}")
    _os.environ.pop("HUGINN_DISABLE_WEB_SEARCH", None)

    # 2. 熔断时 search_unavailable flag
    tool = WebSearchTool()
    # 强制熔断状态
    WebSearchTool._circuit_broken = True
    result = tool.execute({"action": "search", "query": "test"}, None)  # type: ignore[arg-type]
    assert result.success is False, "circuit broken should fail"
    assert result.data.get("search_unavailable") is True, \
        "circuit broken should set search_unavailable=True"
    print("[CHECK C4.2] circuit broken sets search_unavailable=True OK")
    # 复位
    WebSearchTool._circuit_broken = False
    WebSearchTool._consecutive_failures = 0

    # 3. 多源降级链: 4 个 _search_* 方法都存在
    for name in ("_search_tavily", "_search_arxiv", "_search_ddgs", "_search_fallback"):
        assert hasattr(WebSearchTool, name), f"missing {name} — 降级链断"
    print("[CHECK C4.3] 4-backend degradation chain exists OK")

    # 4. ddgs 兼容: ddgs + duckduckgo_search 双导入路径 (C4 迁移)
    # 不真导入 (可能没装), 只验 _search_ddgs 里有双 try
    import inspect
    src = inspect.getsource(WebSearchTool._search_ddgs)
    assert "from ddgs import DDGS" in src, "ddgs import path missing"
    assert "from duckduckgo_search import DDGS" in src, "duckduckgo_search fallback missing"
    print("[CHECK C4.4] ddgs/duckduckgo_search dual import path OK")

    print("[CHECK C4] ALL ASSERTS PASSED")
    return 0


if __name__ == "__main__":
    import sys
    if "--self-check-c4" in sys.argv:
        sys.exit(_c4_self_check())
