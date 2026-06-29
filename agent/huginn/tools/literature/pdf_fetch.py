"""PDF 全文抓取层.

crawl4ai / Playwright / urllib 三级降级抓页面, Sci-Hub 兜底 (opt-in),
4 个 DOI → OA URL 辅助函数 (OpenAlex/Unpaywall/Europe PMC) + 章节切分.
"""

from __future__ import annotations

import asyncio
import os
import re
import urllib.parse
from typing import Any

from ._http import (
    _USER_AGENT,
    _http_get_json,
    _http_get_text,
    _http_get_bytes,
    _timeout,
    logger,
)


# ───────────────────────── crawl4ai 集成 ─────────────────────────


class _Crawl4aiUnavailable(RuntimeError):
    """crawl4ai 没装或初始化失败. 调用方走降级路径."""


_crawl4ai_warned = False  # 模块级标记, 缺库只警告一次


async def _crawl4ai_fetch(
    url: str, max_links: int = 50, user_data_dir: str | None = None,
    headless: bool = True,
) -> tuple[str, str, list[str]]:
    """用 crawl4ai 抓单页, 返回 (markdown, title, links).

    crawl4ai 内部用 Playwright 做 JS 渲染, 适合 Google Scholar / Patents 这种
    重 JS 站. 输出 fit_markdown 已经去噪, 适合喂 LLM.

    user_data_dir 不为 None 时复用 Playwright 持久化 profile (订阅源认证 session).
    不可用时抛 _Crawl4aiUnavailable, 让调用方降级.
    """
    global _crawl4ai_warned
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
        from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
    except ImportError:
        if not _crawl4ai_warned:
            logger.info("crawl4ai 未安装, crawl_web 走 urllib 降级. 装: pip install crawl4ai-skill")
            _crawl4ai_warned = True
        raise _Crawl4aiUnavailable("crawl4ai not installed")

    # 一次性配置: headless, 禁图, UA 伪装; 有 user_data_dir 时复用 session
    browser_cfg_kwargs: dict[str, Any] = {
        "headless": headless,
        "light_mode": True,  # 关掉一些重资源加载
        "user_agent": _USER_AGENT,
    }
    if user_data_dir:
        browser_cfg_kwargs["user_data_dir"] = user_data_dir
    browser_cfg = BrowserConfig(**browser_cfg_kwargs)
    run_cfg = CrawlerRunConfig(
        markdown_generator=DefaultMarkdownGenerator(),
        word_count_threshold=10,  # 过滤太短的段落
        page_timeout=int(_timeout() * 1000 * 2),  # 毫秒
        verbose=False,
    )

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result = await crawler.arun(url=url, config=run_cfg)
        if not result.success:
            raise RuntimeError(f"crawl4ai fetch failed: {result.error_message}")

        md = result.markdown.raw_markdown or ""
        # fit_markdown 是去噪后的, 优先用
        fit_md = getattr(result.markdown, "fit_markdown", None)
        if fit_md:
            md = fit_md

        title = ""
        if result.metadata:
            title = result.metadata.get("title", "") or ""

        # 抽页面内链接
        links: list[str] = []
        seen: set[str] = set()
        for link in (result.links.get("internal", []) or []) + (result.links.get("external", []) or []):
            href = link.get("href", "") if isinstance(link, dict) else ""
            if href and href.startswith("http") and href not in seen:
                seen.add(href)
                links.append(href)
            if len(links) >= max_links:
                break

        return md, title, links


async def _playwright_fetch(
    url: str, user_data_dir: str | None = None, max_links: int = 50,
    wait_sec: float = 3.0, headless: bool = True,
) -> tuple[str, str, list[str]]:
    """原生 Playwright 抓页. crawl4ai anti-bot 误判时的降级.

    比 crawl4ai 更可控: 显式 wait_for_load_state("networkidle") + 额外 delay,
    能拿到 JS 渲染后的内容. 不做 fit_markdown 去噪, 返回 raw text.

    headless=False 用于订阅源 Cloudflare 复用场景: cf_clearance cookie 绑定
    浏览器指纹, headless 指纹和登录时不同会被重新挑战, 用非 headless 复用
    同一 profile 指纹一致才能过.
    """
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        if user_data_dir:
            context = await pw.chromium.launch_persistent_context(
                user_data_dir=user_data_dir,
                headless=headless,
                user_agent=_USER_AGENT,
                viewport={"width": 1280, "height": 800},
                args=["--disable-blink-features=AutomationControlled"],
            )
            browser = None
        else:
            browser = await pw.chromium.launch(
                headless=headless,
                args=["--disable-blink-features=AutomationControlled"],
            )
            context = await browser.new_context(
                user_agent=_USER_AGENT,
                viewport={"width": 1280, "height": 800},
            )
        try:
            page = await context.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            # 等 JS 渲染: networkidle + 额外 delay
            try:
                await page.wait_for_load_state("networkidle", timeout=15000)
            except Exception:
                pass  # 有些站永远不 idle, 不强求
            await asyncio.sleep(wait_sec)

            # Cloudflare 自动挑战等待: 标题含 "just a moment"/"请稍候" 说明在挑战,
            # 很多 Cloudflare JS 挑战会自动过, 给最多 30s 缓冲.
            cf_title_kw = ("just a moment", "请稍候", "attention required",
                           "checking your browser")
            for _ in range(10):  # 10 次 × 3s = 30s 上限
                try:
                    cur_title = ((await page.title()) or "").lower()
                except Exception:
                    break
                if not any(k in cur_title for k in cf_title_kw):
                    break
                # 还在挑战页, 再等 3s
                await asyncio.sleep(3.0)

            title = await page.title()
            # 报告 429 或其他错误
            html = await page.content()
            # 简单 HTML → text
            import re as _re
            text = _re.sub(r"<script[^>]*>.*?</script>", "", html, flags=_re.IGNORECASE | _re.DOTALL)
            text = _re.sub(r"<style[^>]*>.*?</style>", "", text, flags=_re.IGNORECASE | _re.DOTALL)
            text = _re.sub(r"<[^>]+>", " ", text)
            text = _re.sub(r"&nbsp;", " ", text)
            text = _re.sub(r"&amp;", "&", text)
            text = _re.sub(r"&lt;", "<", text)
            text = _re.sub(r"&gt;", ">", text)
            text = _re.sub(r"&quot;", '"', text)
            text = _re.sub(r"\s+", " ", text).strip()

            # 抽链接
            links_raw = await page.eval_on_selector_all(
                "a[href]", "els => els.map(e => e.href)"
            )
            links: list[str] = []
            seen: set[str] = set()
            for href in links_raw or []:
                if href.startswith("http") and href not in seen:
                    seen.add(href)
                    links.append(href)
                if len(links) >= max_links:
                    break
            return text, title, links
        finally:
            await context.close()
            if browser:
                await browser.close()


async def _fallback_html_fetch(url: str) -> tuple[str, str]:
    """crawl4ai 不可用时的降级: urllib + 简单 HTML 抽取.

    只能处理静态 HTML, JS 渲染页面会丢内容. 但总比啥都没有强.
    返回 (近似 markdown, title).
    """
    try:
        html = await _http_get_text(url)
    except Exception as exc:
        raise RuntimeError(f"fallback fetch failed: {exc}") from exc

    # 抽 title
    title = ""
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if m:
        title = re.sub(r"\s+", " ", m.group(1)).strip()

    # 简单 HTML → 文本: 去 script/style/标签, 压空白
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"\s+", " ", text).strip()
    return text, title


# ───────────────────────── Sci-Hub (opt-in, 法律灰色) ───────────────────
# Sci-Hub 托管版权论文未获出版商授权, 在多数司法管辖区属法律灰色地带.
# 默认关闭. 用户设 HUGINN_ENABLE_SCIHUB=1 显式启用, 自行承担合规风险.
# 仅在所有合法 OA 源 (OpenAlex/Unpaywall/EuropePMC/arxiv/CORE) 都失败后兜底.

# 常用镜像, 按稳定性排序. 逐个尝试, 第一个返回 PDF 的就用.
_SCIHUB_MIRRORS = (
    "https://sci-hub.se",
    "https://sci-hub.ru",
    "https://sci-hub.st",
    "https://sci-hub.ee",
)


def _scihub_enabled() -> bool:
    """是否启用 Sci-Hub 候选源. 默认关闭, 需 HUGINN_ENABLE_SCIHUB=1."""
    return os.environ.get("HUGINN_ENABLE_SCIHUB", "").lower() in (
        "1", "true", "yes", "on",
    )


async def _scihub_pdf_url(doi: str) -> str | None:
    """查 Sci-Hub 拿 PDF 直链.

    Sci-Hub 页面结构: 返回 HTML, 里有 <embed id="pdf" src="..."> 或
    <iframe src="..."> 指向真实 PDF. src 可能是相对路径 (//... 或 /downloads/...).

    逐个镜像试, 第一个成功的返回. 全失败返回 None.
    """
    if not doi:
        return None
    logger.warning(
        "Sci-Hub 兜底启用 (HUGINN_ENABLE_SCIHUB=1). 法律灰色地带, 用户自负风险. "
        "DOI: %s", doi
    )
    for mirror in _SCIHUB_MIRRORS:
        url = f"{mirror}/{doi}"
        try:
            html = await _http_get_text(url, timeout=_timeout() * 1.5)
        except Exception as exc:
            logger.debug("Sci-Hub mirror %s 失败: %s", mirror, exc)
            continue

        # 抽 <embed id="pdf" src="..."> 或 <iframe src="...">
        # src 可能是 //foo/bar (协议相对) 或 /downloads/foo (路径相对) 或 https://...
        patterns = [
            r'<embed[^>]+id="pdf"[^>]+src="([^"]+)"',
            r'<iframe[^>]+src="([^"]+)"',
            r'<embed[^>]+src="([^"]+)"',
        ]
        for pat in patterns:
            m = re.search(pat, html, re.IGNORECASE)
            if m:
                src = m.group(1).strip()
                # 补全相对路径
                if src.startswith("//"):
                    src = "https:" + src
                elif src.startswith("/"):
                    src = mirror + src
                elif not src.startswith("http"):
                    src = f"{mirror}/{src.lstrip('/')}"
                return src
    return None


# ───────────────────────── DOI → OA PDF URL 辅助 ───────────────────
# 4 个独立函数, 不依赖 LiteratureTool 实例状态, 从原方法剥离 self 即可.


async def openalex_oa_url(doi: str) -> str | None:
    """OpenAlex 按 DOI 查 open_access.oa_url. OpenAlex 在国内一般可达."""
    if not doi:
        return None
    url = f"https://api.openalex.org/works/doi:{urllib.parse.quote(doi)}?select=open_access"
    try:
        data = await _http_get_json(url)
    except Exception as exc:
        logger.warning("OpenAlex DOI 查询失败: %s", exc)
        return None
    oa = data.get("open_access") or {}
    oa_url = oa.get("oa_url") or ""
    # 只要 pdf 或 html 都收, html 也能抽数值; 优先 pdf
    return oa_url or None


async def unpaywall_pdf(doi: str) -> str | None:
    """Unpaywall 找 DOI 的 OA PDF URL. 免 key, 要 email."""
    email = os.environ.get("HUGINN_UNPAYWALL_EMAIL", "user@example.com")
    url = f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}?email={email}"
    try:
        data = await _http_get_json(url)
    except Exception as exc:
        logger.warning("Unpaywall 请求失败: %s", exc)
        return None
    # oa_locations 里优先找 url_for_pdf
    for loc in data.get("oa_locations", []) or []:
        pdf_url = loc.get("url_for_pdf")
        if pdf_url:
            return pdf_url
    best = data.get("best_oa_location") or {}
    return best.get("url_for_pdf") or best.get("url")


async def europepmc_pdf(doi: str) -> str | None:
    """Europe PMC 按 DOI 查 fullTextUrlList 里的 PDF 链接.

    Europe PMC 覆盖生物医学为主, 但也收了部分化学/材料期刊的 OA 全文.
    ebi.ac.uk 在国内一般可达, 是 arxiv 之外的好补充.
    """
    if not doi:
        return None
    q = urllib.parse.quote(f"DOI:{doi}")
    url = (
        f"https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        f"?query={q}&format=json&resultType=core"
    )
    try:
        data = await _http_get_json(url)
    except Exception as exc:
        logger.warning("Europe PMC 请求失败: %s", exc)
        return None
    hits = (data.get("resultList") or {}).get("result", []) or []
    if not hits:
        return None
    for ft_url in hits[0].get("fullTextUrlList", {}).get("fullTextUrl", []) or []:
        style = (ft_url.get("documentStyle") or "").lower()
        if style == "pdf":
            return ft_url.get("url")
    # 没 pdf 就退 html
    for ft_url in hits[0].get("fullTextUrlList", {}).get("fullTextUrl", []) or []:
        if ft_url.get("url"):
            return ft_url.get("url")
    return None


def split_sections(text: str) -> list[dict[str, Any]]:
    """简单按章节标题切. 找 Introduction/Methods/Results/Discussion/Conclusion.

    启发式: 短行 (< 60 字符) + 匹配常见标题词. 不完美但够用,
    让 LLM 能定位到 Results/Tables 章节抽数值.
    """
    section_titles = (
        "introduction", "abstract", "methods", "methodology", "method",
        "results", "discussion", "conclusion", "conclusions",
        "references", "acknowledg", "appendix", "table", "tables",
        "experiment", "experimental", "theory", "computational",
    )
    sections: list[dict[str, Any]] = []
    current_title = "preamble"
    current_lines: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip().lower().rstrip(".:")
        if 0 < len(stripped) < 60 and any(stripped == t or stripped.startswith(t) for t in section_titles):
            if current_lines:
                sections.append({
                    "title": current_title,
                    "text": "\n".join(current_lines).strip()[:8000],
                })
            current_title = stripped
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append({
            "title": current_title,
            "text": "\n".join(current_lines).strip()[:8000],
        })
    return sections
