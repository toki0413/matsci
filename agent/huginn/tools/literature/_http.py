"""HTTP 层 + 配置开关. 进程级 opener 单例, 所有模块共用连接池."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

_USER_AGENT = "HuginnAgent/1.0 (materials-science-research; mailto:user@example.com)"


def _timeout() -> float:
    """单次 API 请求超时. 默认 20s, 大了挡事件循环, 小了 S2 偶尔慢会误伤."""
    raw = os.environ.get("HUGINN_LITERATURE_TIMEOUT", "20")
    try:
        v = float(raw)
        return v if v > 0 else 20.0
    except (TypeError, ValueError):
        return 20.0


def _disabled() -> bool:
    """离线/CI 环境直接禁用, 跟 web_search_tool 用同一个开关."""
    return os.environ.get("HUGINN_DISABLE_WEB_SEARCH", "").lower() in (
        "1", "true", "yes", "on",
    )


_DISABLED_HINT = (
    "literature_tool disabled by HUGINN_DISABLE_WEB_SEARCH. "
    "Only local rag_tool / materials_database_tool available."
)


def _build_opener() -> urllib.request.OpenerDirector:
    """构造带代理的 URL opener. 读 HTTPS_PROXY/HTTP_PROXY/ALL_PROXY 环境变量.

    urllib 默认会读 *_PROXY 环境变量, 但显式构造 opener 更可控, 也方便
    后续加自定义 header (polite pool mailto 之类).
    """
    handlers: list[urllib.request.BaseHandler] = [
        urllib.request.ProxyHandler(),  # 默认会读环境变量
    ]
    # 显式覆盖, 避免 urllib 在某些环境下读不到环境变量
    proxy = (
        os.environ.get("HTTPS_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("http_proxy")
        or os.environ.get("ALL_PROXY")
        or os.environ.get("all_proxy")
    )
    if proxy:
        handlers = [urllib.request.ProxyHandler({
            "http": proxy,
            "https": proxy,
        })]
    return urllib.request.build_opener(*handlers)


# 模块级 opener, 进程内复用连接
_OPENER = _build_opener()


async def _http_get_json(url: str, timeout: float | None = None) -> dict[str, Any]:
    """GET JSON. 失败抛异常, 让调用方决定怎么降级."""
    t = timeout if timeout is not None else _timeout()

    def _fetch() -> dict[str, Any]:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with _OPENER.open(req, timeout=t) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))

    return await asyncio.to_thread(_fetch)


async def _http_get_text(url: str, timeout: float | None = None) -> str:
    t = timeout if timeout is not None else _timeout()

    def _fetch() -> str:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with _OPENER.open(req, timeout=t) as resp:
            return resp.read().decode("utf-8", errors="replace")

    return await asyncio.to_thread(_fetch)


async def _http_get_bytes(url: str, timeout: float | None = None) -> bytes:
    """下载二进制 (PDF). fetch_pdf 用."""
    t = timeout if timeout is not None else _timeout() * 2  # PDF 大, 给双倍超时

    def _fetch() -> bytes:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with _OPENER.open(req, timeout=t) as resp:
            return resp.read()

    return await asyncio.to_thread(_fetch)
