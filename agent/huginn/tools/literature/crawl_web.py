"""crawl_web 子动作: 搜索引擎桥接 + 直接爬单页 + 订阅源认证 session.

12 个学术 provider 的登录/检测配置, Playwright 持久化 profile 复用,
EZproxy URL 重写, crawl4ai → Playwright → urllib 三级降级.
3 个原 LiteratureTool 方法 (_crawl_direct / _crawl_search_engine /
_fallback_web_search) 转模块级函数, 不依赖 self 状态.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import urllib.parse
from pathlib import Path
from typing import Any

from huginn.types import ToolContext, ToolResult

from ._http import _USER_AGENT, logger
from .pdf_fetch import (
    _Crawl4aiUnavailable,
    _crawl4ai_fetch,
    _playwright_fetch,
)


# ───────────────────────── Authenticated sessions (高校订阅源) ──────────
# 高校师生有合法订阅 (Elsevier/IEEE/CNKI/万方等), 但这些站有 Cloudflare/
# CARSI/二维码登录, headless 过不去. 方案: Playwright 持久化 profile,
# 用户首次手动登录 (非 headless), profile 存盘后续 headless 复用.
# session 过期 (检测到登录跳转) 提示重新 auth.

# Provider 注册表: 域名 → 登录/成功检测配置
# success_pattern: 登录成功后 URL 匹配的正则; None 表示不自动检测, 靠用户关页
_PROVIDERS: dict[str, dict[str, Any]] = {
    "cnki": {
        "domains": ["cnki.net", "cnki.com.cn", "kns.cnki.net", "kns8.cnki.net"],
        "login_url": "https://kns.cnki.net/kns8/defaultresult/index",
        "success_pattern": r"my\.cnki\.net|kns\.cnki\.net.*loginmode=1",
        "label": "中国知网 CNKI",
    },
    "wanfang": {
        "domains": ["wanfangdata.com.cn", "s.wanfangdata.com.cn"],
        "login_url": "https://s.wanfangdata.com.cn/",
        "success_pattern": r"login\.wanfangdata\.com\.cn.*loginto|wanfangdata.*islogin=1",
        "label": "万方数据",
    },
    "cqvip": {
        "domains": ["cqvip.com", "qikan.cqvip.com"],
        "login_url": "https://qikan.cqvip.com/",
        "success_pattern": None,
        "label": "维普 VIP",
    },
    "elsevier": {
        "domains": ["sciencedirect.com", "elsevier.com"],
        "login_url": "https://www.sciencedirect.com/user/login",
        "success_pattern": r"sciencedirect\.com.*loggedIn|user/view",
        "label": "Elsevier ScienceDirect",
    },
    "springer": {
        "domains": ["link.springer.com", "springer.com"],
        "login_url": "https://link.springer.com/signup-login",
        "success_pattern": r"link\.springer\.com.*loggedIn",
        "label": "Springer Link",
    },
    "ieee": {
        "domains": ["ieeexplore.ieee.org"],
        "login_url": "https://ieeexplore.ieee.org/servlet/Login",
        "success_pattern": r"ieeexplore.*personalLogin|ieee.*accountId",
        "label": "IEEE Xplore",
    },
    "wiley": {
        "domains": ["onlinelibrary.wiley.com"],
        "login_url": "https://onlinelibrary.wiley.com/action/ssostart",
        "success_pattern": r"onlinelibrary\.wiley\.com.*loggedIn",
        "label": "Wiley Online",
    },
    "acs": {
        "domains": ["pubs.acs.org"],
        "login_url": "https://pubs.acs.org/action/ssostart",
        "success_pattern": r"pubs\.acs\.org.*loggedIn",
        "label": "ACS Publications",
    },
    "rsc": {
        "domains": ["pubs.rsc.org"],
        "login_url": "https://pubs.rsc.org/en/account/login",
        "success_pattern": r"pubs\.rsc\.org.*signin",
        "label": "RSC Publishing",
    },
    "nature": {
        "domains": ["nature.com"],
        "login_url": "https://id.nature.com/login",
        "success_pattern": r"nature\.com.*loggedIn",
        "label": "Nature",
    },
    "wos": {
        "domains": ["webofscience.com", "apps.webofknowledge.com"],
        "login_url": "https://www.webofscience.com/wos/alloy/basicSearch",
        "success_pattern": r"webofscience\.com.*authorized",
        "label": "Web of Science",
    },
    "tandfonline": {
        "domains": ["tandfonline.com"],
        "login_url": "https://www.tandfonline.com/action/ssostart",
        "success_pattern": r"tandfonline\.com.*loggedIn",
        "label": "Taylor & Francis",
    },
}


def _session_base() -> Path:
    """session 存储根目录: ~/.huginn/sessions/"""
    return Path.home() / ".huginn" / "sessions"


def _session_dir(provider: str) -> Path:
    """单个 provider 的 profile 目录."""
    return _session_base() / provider / "profile"


def _session_meta_path(provider: str) -> Path:
    return _session_base() / provider / "meta.json"


def _detect_provider(url: str) -> str | None:
    """从 URL 域名反查 provider. 没匹配返回 None."""
    try:
        host = urllib.parse.urlparse(url).hostname or ""
    except Exception:
        return None
    host = host.lower()
    for provider, cfg in _PROVIDERS.items():
        for domain in cfg["domains"]:
            # 子域名也算 (如 www.sciencedirect.com 匹配 sciencedirect.com)
            if host == domain or host.endswith("." + domain):
                return provider
    return None


def _list_sessions() -> list[dict[str, Any]]:
    """列出所有有 profile 的 provider 及元信息."""
    out: list[dict[str, Any]] = []
    base = _session_base()
    if not base.exists():
        return out
    for provider_dir in base.iterdir():
        if not provider_dir.is_dir():
            continue
        provider = provider_dir.name
        if provider not in _PROVIDERS:
            continue
        meta_path = _session_meta_path(provider)
        meta: dict[str, Any] = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        out.append({
            "provider": provider,
            "label": _PROVIDERS[provider]["label"],
            "has_profile": _session_dir(provider).exists(),
            "last_auth_at": meta.get("last_auth_at"),
            "last_url": meta.get("last_url"),
            "n_crawls": meta.get("n_crawls", 0),
        })
    return out


def _save_session_meta(provider: str, **fields: Any) -> None:
    """更新 provider 的 meta.json (合并写)."""
    meta_path = _session_meta_path(provider)
    meta: dict[str, Any] = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    meta.update(fields)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _apply_ezproxy(url: str) -> str:
    """EZproxy URL 重写. 读 HUGINN_EZPROXY_PREFIX 环境变量.

    典型 EZproxy 前缀 (各校不同, 用户填自家的):
      https://vpn.uni.edu.cn/http/77726476706e69737468656265737421xxxx/
    重写后: {prefix}{原域名}{原路径}

    只重写命中 _PROVIDERS 学术域的 URL, 避免把所有网页都走代理.
    HUGINN_EZPROXY_DOMAINS 可限制范围 (逗号分隔, 默认全部学术域).
    """
    prefix = os.environ.get("HUGINN_EZPROXY_PREFIX", "").strip()
    if not prefix:
        return url
    provider = _detect_provider(url)
    if not provider:
        return url  # 非学术域不重写
    # 可选: 限定重写域名白名单
    whitelist = os.environ.get("HUGINN_EZPROXY_DOMAINS", "").strip()
    if whitelist:
        allowed = {d.strip().lower() for d in whitelist.split(",") if d.strip()}
        host = (urllib.parse.urlparse(url).hostname or "").lower()
        if not any(host == d or host.endswith("." + d) for d in allowed):
            return url
    # 拆 URL: scheme://host/path → prefix + host + path
    parsed = urllib.parse.urlparse(url)
    reconstructed = parsed.netloc + parsed.path
    if parsed.query:
        reconstructed += "?" + parsed.query
    if parsed.fragment:
        reconstructed += "#" + parsed.fragment
    # prefix 末尾保证有 /
    if not prefix.endswith("/"):
        prefix += "/"
    return prefix + reconstructed


async def _auth_login(provider: str, timeout_sec: int = 300) -> dict[str, Any]:
    """弹非 headless 浏览器让用户手动登录, 存 Playwright profile.

    流程:
      1. launch_persistent_context(user_data_dir=session_dir, headless=False)
      2. 打开 provider 的 login_url
      3. 轮询当前页 URL 是否匹配 success_pattern (2s 间隔, timeout_sec 上限)
      4. 匹配成功 → 关浏览器, 存 meta, 返回成功
      5. 用户手动关浏览器 → 当作取消
      6. 超时 → 返回 timeout

    provider 不存在或 Playwright 不可用抛异常.
    """
    if provider not in _PROVIDERS:
        raise ValueError(f"未知 provider: {provider}. 可选: {list(_PROVIDERS.keys())}")
    cfg = _PROVIDERS[provider]
    login_url = cfg["login_url"]
    success_pattern = cfg.get("success_pattern")

    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright 未安装. 装: pip install playwright && playwright install chromium"
        ) from exc

    profile_dir = _session_dir(provider)
    profile_dir.mkdir(parents=True, exist_ok=True)

    import time
    from datetime import datetime, timezone

    async with async_playwright() as pw:
        # headless=False 让用户看到浏览器窗口操作
        context = await pw.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            headless=False,
            user_agent=_USER_AGENT,
            viewport={"width": 1280, "height": 800},
            args=["--disable-blink-features=AutomationControlled"],  # 降低被检测概率
        )
        page = context.pages[0] if context.pages else await context.new_page()
        await page.goto(login_url, wait_until="domcontentloaded", timeout=30000)

        start = time.time()
        result = "timeout"
        last_url = login_url
        # 机器人验证页特征 (与 crawl_direct 的 bot_indicators 对齐)
        # 校园网场景下用户不需要真登录账号, 过了 Cloudflare/滑块就算成功
        bot_keywords = (
            "are you a robot", "please confirm you are not a robot",
            "verify you are human", "checking your browser",
            "captcha", "enable javascript and cookies to continue",
            "安全验证", "向右滑动", "滑动验证",
        )
        cloudflare_title_keywords = ("just a moment", "请稍候", "attention required")
        while time.time() - start < timeout_sec:
            # 检测 context 是否被用户关掉
            try:
                _ = context.pages  # 抛异常说明关了
            except Exception:
                result = "user_closed"
                break
            try:
                current_url = page.url
                last_url = current_url
            except Exception:
                # page 可能被关, 看看还有没有其他页
                if not context.pages:
                    result = "user_closed"
                    break
                page = context.pages[0]
                continue

            # 抓页面 title 和正文前 3KB, 判断是否还在机器人验证页
            page_title = ""
            page_text_lower = ""
            try:
                page_title = (await page.title()) or ""
                page_text_lower = ((await page.content()) or "").lower()[:3000]
            except Exception:
                pass
            title_lower = page_title.lower()
            is_bot_page = (
                any(k in page_text_lower for k in bot_keywords)
                or any(k in title_lower for k in cloudflare_title_keywords)
                or "cdn-cgi/challenge" in current_url.lower()
            )

            # 成功检测 1: URL 命中 success_pattern (用户真登录了账号)
            if success_pattern and re.search(success_pattern, current_url):
                result = "success"
                break

            # 成功检测 2: 机器人验证页已过 + 停留 15s 以上
            # 校园网场景: IP 已认证, 用户过了 Cloudflare 就能看到真实页面,
            # 不需要真登录. 给 15s 让用户手动过验证.
            if not is_bot_page and time.time() - start > 15:
                result = "success"
                break

            # 成功检测 3: 没 success_pattern 的 provider, 停在非登录页 10s 也算成功
            if not success_pattern and time.time() - start > 10:
                if "login" not in current_url.lower() and "signin" not in current_url.lower():
                    result = "success"
                    break

            await asyncio.sleep(2)

        # 存 meta 不管结果
        _save_session_meta(
            provider,
            last_auth_at=datetime.now(timezone.utc).isoformat(),
            last_url=last_url,
            auth_result=result,
        )

        try:
            await context.close()
        except Exception:
            pass

    return {
        "provider": provider,
        "label": cfg["label"],
        "result": result,
        "last_url": last_url,
        "profile_dir": str(profile_dir),
    }


async def _auth_logout(provider: str) -> dict[str, Any]:
    """删 provider 的 profile 目录和 meta."""
    import shutil
    if provider not in _PROVIDERS:
        raise ValueError(f"未知 provider: {provider}")
    profile = _session_dir(provider)
    meta = _session_meta_path(provider)
    removed = []
    if profile.exists():
        shutil.rmtree(profile, ignore_errors=True)
        removed.append("profile")
    if meta.exists():
        meta.unlink()
        removed.append("meta")
    return {"provider": provider, "removed": removed or ["nothing"]}


# ───────────────────────── crawl_direct / crawl_search_engine ──────────
# 从 LiteratureTool 剥离的 3 个方法, 不访问 self 状态, 转模块级函数.


async def crawl_direct(url: str, max_results: int) -> ToolResult:
    """直接爬单页. 优先 crawl4ai (可复用 session profile), 不可用降级 urllib.

    流程:
      1. EZproxy URL 重写 (若配了 HUGINN_EZPROXY_PREFIX)
      2. 检测 URL 属于哪个订阅源 provider
      3. 有 session profile → crawl4ai 带 user_data_dir headless 复用
      4. 无 session → 普通 crawl4ai headless
      5. 检测响应是否是登录跳转 → 提示重新 auth
    """
    # EZproxy 重写
    final_url = _apply_ezproxy(url)
    ezproxy_applied = final_url != url

    # provider 检测 + session 复用
    provider = _detect_provider(url)
    session_dir = _session_dir(provider) if provider else None
    used_session = bool(provider and session_dir and session_dir.exists())

    fetcher = "crawl4ai"
    try:
        md, title, links = await _crawl4ai_fetch(
            final_url, max_links=max_results,
            user_data_dir=str(session_dir) if used_session else None,
            headless=True,
        )
    except _Crawl4aiUnavailable:
        # crawl4ai 没装: 直接走 Playwright (比 urllib 强, 能渲染 JS)
        try:
            md, title, links = await _playwright_fetch(
                final_url,
                user_data_dir=str(session_dir) if used_session else None,
                max_links=max_results,
            )
            fetcher = "playwright"
        except Exception as exc2:
            return ToolResult(
                data=None, success=False,
                error=f"crawl_web direct 失败 (crawl4ai 和 playwright 都失败): {exc2}",
            )
    except Exception as exc:
        # crawl4ai 报错: 看是不是 anti-bot 误判, 是就降级 Playwright
        err_str = str(exc).lower()
        is_antibot_false_positive = any(
            k in err_str for k in ("anti-bot", "minimal_text", "no_content_elements",
                                    "script_heavy_shell", "blocked by")
        )
        if is_antibot_false_positive:
            logger.info("crawl4ai 判 anti-bot, 降级 Playwright: %s", exc)
            try:
                md, title, links = await _playwright_fetch(
                    final_url,
                    user_data_dir=str(session_dir) if used_session else None,
                    max_links=max_results,
                    wait_sec=4.0,  # JS 重的多等会
                )
                fetcher = "playwright_fallback"
            except Exception as exc2:
                return ToolResult(
                    data=None, success=False,
                    error=f"crawl_web direct 失败: crawl4ai='{exc}' playwright='{exc2}'",
                )
        else:
            logger.warning("crawl_web direct 失败: %s", exc)
            return ToolResult(
                data=None, success=False,
                error=f"crawl_web direct 失败: {exc}",
            )

    # 登录跳转检测: 命中订阅源但内容像登录页 → session 过期
    login_indicators = (
        "please log in", "please sign in", "login to continue",
        "sign in to continue", "登录", "请登录", "用户登录",
        'id="username"', 'name="password"', "shibboleth", "carsi",
    )
    # 机器人检测页特征 (Cloudflare/reCAPTCHA/知网滑块)
    bot_indicators = (
        "are you a robot", "please confirm you are not a robot",
        "verify you are human", "checking your browser",
        "安全验证", "向右滑动", "滑动验证", "captcha",
        "enable javascript and cookies to continue",
    )
    md_lower = md.lower()[:3000]
    is_login_page = any(ind in md_lower for ind in login_indicators) and len(md) < 4000
    is_bot_block = any(ind in md_lower for ind in bot_indicators) and len(md) < 6000
    if used_session and is_login_page:
        return ToolResult(
            data=None, success=False,
            error=(
                f"订阅源 {provider} session 可能过期 (页面是登录表单). "
                f"重新登录: crawl_web auth_action=login provider={provider}"
            ),
        )
    # 没 session 但命中订阅源且被墙 → 提示先 auth
    if provider and not used_session and is_login_page:
        return ToolResult(
            data=None, success=False,
            error=(
                f"URL 属于订阅源 {provider} (_PROVIDERS[\"{provider}\"][\"label\"]), "
                f"但没存 session. 先登录: "
                f"crawl_web auth_action=login provider={provider}"
            ),
        )
    # 机器人检测页 (Cloudflare/CAPTCHA/滑块): headless 过不去
    if is_bot_block:
        prov = provider or _detect_provider(url)
        if prov:
            hint = (
                f"页面命中反机器人检测 (Cloudflare/CAPTCHA/滑块验证). "
                f"该站用 Cloudflare 托管挑战 (需交互式点击 I'm not a robot), "
                f"cf_clearance cookie 绑定浏览器指纹无法跨 session 复用, "
                f"auth_action=login 也救不了.\n"
                f"三条出路:\n"
                f"  1. 配 EZproxy (推荐): 设 HUGINN_EZPROXY_PREFIX=<本校 EZproxy URL>, "
                f"crawl_web 会自动把 {prov} 域名重写到 EZproxy 走机构 IP 白名单, "
                f"绕过 Cloudflare.\n"
                f"  2. fetch_pdf 走 OA 兜底: 该文章可能在 arXiv/Europe PMC/Unpaywall "
                f"有 OA 版本, fetch_pdf doi=<doi> 会自动找.\n"
                f"  3. 手动下载: 浏览器打开 {url} 下载 PDF, "
                f"再 fetch_pdf 用本地文件路径."
            )
        else:
            hint = (
                "页面命中反机器人检测. 该 URL 不在已知 provider 列表, "
                "无法自动存 session. 换个 URL 或手动用浏览器访问."
            )
        return ToolResult(
            data=None, success=False,
            error=hint,
        )

    # 更新 crawl 计数
    if used_session and provider:
        meta_path = _session_meta_path(provider)
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                _save_session_meta(provider, n_crawls=meta.get("n_crawls", 0) + 1)
            except Exception:
                pass

    data: dict[str, Any] = {
        "action": "crawl_web",
        "engine": "direct",
        "url": url,
        "final_url": final_url,
        "title": title,
        "markdown": md,
        "links": links,
        "n_links": len(links),
        "fetcher": fetcher,
    }
    if used_session:
        data["session_provider"] = provider
        data["session_used"] = True
    if ezproxy_applied:
        data["ezproxy_applied"] = True
    if provider and not used_session:
        data["provider_detected"] = provider
        data["note"] = (
            f"检测到订阅源 {provider} 但无 session, 走匿名访问. "
            f"被墙/要登录时: crawl_web auth_action=login provider={provider}"
        )
    if fetcher == "urllib_fallback":
        data["note"] = (data.get("note", "") +
            " crawl4ai 不可用, urllib 简单抽取, JS 页面会丢内容.").strip()
    return ToolResult(data=data, success=True)


async def crawl_search_engine(
    engine: str,
    query: str,
    max_results: int,
    context: ToolContext,
) -> ToolResult:
    """搜索引擎桥接. 返回结果链接列表, 不爬正文 (正文要再调 direct)."""
    # 构造搜索 URL
    q = urllib.parse.quote(query)
    if engine == "google_scholar":
        search_url = f"https://scholar.google.com/scholar?q={q}&hl=en"
        site_name = "Google Scholar"
    elif engine == "google_patents":
        search_url = f"https://patents.google.com/?q={q}&oq={q}"
        site_name = "Google Patents"
    elif engine == "duckduckgo":
        search_url = f"https://duckduckgo.com/html/?q={q}"
        site_name = "DuckDuckGo"
    else:
        return ToolResult(
            data=None, success=False,
            error=f"unknown engine: {engine}",
        )

    # 先抓搜索结果页
    try:
        md, title, links = await _crawl4ai_fetch(search_url, max_links=max_results * 3)
    except _Crawl4aiUnavailable:
        # crawl4ai 没装: duckduckgo 走 web_search_tool 兜底, 其他引擎没法降级
        if engine == "duckduckgo":
            return await fallback_web_search(query, max_results, site_name)
        return ToolResult(
            data=None, success=False,
            error=f"crawl4ai 不可用, engine={engine} 需要 JS 渲染, 无法降级. "
                  f"装 crawl4ai: pip install crawl4ai-skill",
        )
    except Exception as exc:
        # crawl4ai 装了但抓取失败 (网络超时/反爬/JS 错误)
        logger.warning("crawl_web %s 抓取失败: %s", engine, exc)
        if engine == "duckduckgo":
            # duckduckgo 直连超时常见 (国内被墙), 走 web_search_tool 兜底
            return await fallback_web_search(query, max_results, site_name)
        return ToolResult(
            data=None, success=False,
            error=f"crawl_web {engine} 失败: {exc}. "
                  f"可能是网络问题或反爬, 重试或换 engine=duckduckgo",
        )

    # 从 markdown 里抽结果链接 (搜索引擎结果页链接密度高)
    # 简单策略: 取所有 http(s) 链接, 去重, 截断到 max_results
    # 命中学术域的链接自动 EZproxy 重写
    seen: set[str] = set()
    results: list[dict[str, str]] = []
    for link in links:
        if link in seen:
            continue
        # 过滤掉搜索引擎自己的导航链接
        if any(d in link for d in (
            "google.com/scholar?", "google.com/?q=", "duckduckgo.com/y.js",
            "duckduckgo.com/l/?uddg=", "google.com/search?",
        )):
            continue
        seen.add(link)
        # EZproxy 重写 (命中学术域才改)
        final_link = _apply_ezproxy(link)
        results.append({"url": final_link, "original_url": link, "title": ""})
        if len(results) >= max_results:
            break

    return ToolResult(
        data={
            "action": "crawl_web",
            "engine": engine,
            "query": query,
            "site": site_name,
            "search_url": search_url,
            "n_results": len(results),
            "results": results,
            "page_markdown": md[:2000],  # 截一段给 agent 看上下文
            "note": (
                "搜索引擎只给链接. 要正文用 crawl_web engine=direct url=<具体链接>."
            ),
        },
        success=True,
    )


async def fallback_web_search(
    query: str, max_results: int, site_name: str
) -> ToolResult:
    """crawl4ai 抓 duckduckgo 失败时, 走 web_search_tool 兜底.

    web_search_tool 有 Tavily / duckduckgo_search / urllib 三路降级,
    比 crawl4ai 单路更可能成功 (尤其 duckduckgo_search 库走 API 不是网页).
    """
    try:
        from huginn.tools.web_search_tool import WebSearchTool
        # 直接实例化, 不依赖 ToolRegistry (smoke test 场景没注册)
        ws = WebSearchTool()
        # WebSearchTool.call 收 dict 不收 pydantic model
        res = await ws.call(
            {"query": query, "max_results": min(max_results, 20)},
            ToolContext(session_id="literature", workspace=os.getcwd()),
        )
        if not res.success:
            return ToolResult(
                data=None, success=False,
                error=f"crawl4ai 和 web_search_tool 都失败: {res.error}",
            )
        return ToolResult(
            data={
                "action": "crawl_web",
                "engine": "duckduckgo",
                "query": query,
                "site": site_name,
                "n_results": len((res.data or {}).get("results", [])),
                "results": (res.data or {}).get("results", []),
                "fetcher": "web_search_tool_fallback",
                "note": "crawl4ai 抓取失败, 走 web_search_tool 兜底.",
            },
            success=True,
        )
    except Exception as exc:
        return ToolResult(
            data=None, success=False,
            error=f"fallback web_search 也失败: {exc}",
        )
