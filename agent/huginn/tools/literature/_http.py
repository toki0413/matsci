"""HTTP 层 + 配置开关. 进程级 opener 单例, 所有模块共用连接池.

带 per-host QPS 限速 + 自动重试 + gzip. 三个公开函数签名不变, 调用方无感.
"""

from __future__ import annotations

import asyncio
import email.utils
import gzip
import json
import logging
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
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
        or os.environ.get("all_proxy")  # noqa: SIM112  # 大小写都试, 保持兼容
    )
    if proxy:
        handlers = [urllib.request.ProxyHandler({
            "http": proxy,
            "https": proxy,
        })]
    return urllib.request.build_opener(*handlers)


# 模块级 opener, 进程内复用连接
_OPENER = _build_opener()


# ---- 限速 / 重试 配置 ----

def _qps() -> float:
    """每秒每 host 最大请求数. 默认 5, 想更保守就 HUGINN_HTTP_QPS=2."""
    raw = os.environ.get("HUGINN_HTTP_QPS", "5")
    try:
        v = float(raw)
        return v if v > 0 else 5.0
    except (TypeError, ValueError):
        return 5.0


# ponytail: 单锁串行所有 host 的限速检查, 高并发下不同 host 会互相等.
# 升级路径: 换成 per-host 的 asyncio.Lock 字典, 让不同 host 并行等待.
_last_request: dict[str, float] = {}
_qps_lock = asyncio.Lock()

# 5xx 和 429 才重试, 4xx 是调用方的锅 (key 错了 / DOI 不存在)
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_MAX_RETRIES = 3
_BACKOFF_BASE = 3.0
_BACKOFF_MAX = 60.0
_JITTER = 0.5  # ±50% 抖动, 防止重试风暴


class HttpError(Exception):
    """非 2xx 且重试耗尽, 或网络层彻底失败时抛出.

    status_code=0 表示网络层错误 (DNS / 连接拒绝 / 超时), 拿不到 HTTP 状态.
    """

    def __init__(self, status_code: int, body: Any, url: str) -> None:
        self.status_code = status_code
        self.url = url
        if isinstance(body, bytes):
            body = body.decode("utf-8", errors="replace")
        elif not isinstance(body, str):
            body = str(body)
        self.body = body[:500]
        super().__init__(f"HTTP {status_code} {url}: {self.body[:200]}")


def _parse_retry_after(headers: Any) -> float | None:
    """解析 Retry-After 头. 支持 delta-seconds 和 HTTP-date 两种格式."""
    val = headers.get("Retry-After") if headers else None
    if not val:
        return None
    # 纯数字 = 等待秒数
    try:
        return max(0.0, float(val))
    except (TypeError, ValueError):
        pass
    # HTTP-date (RFC 7231), 比如 "Wed, 21 Oct 2015 07:28:00 GMT"
    try:
        dt = email.utils.parsedate_to_datetime(val)
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return max(0.0, (dt - datetime.now(UTC)).total_seconds())
    except (TypeError, ValueError):
        return None


def _backoff_delay(attempt: int, retry_after: float | None) -> float:
    """指数退避 + jitter. 有 Retry-After 就取 max(服务端要求, 退避)."""
    base = min(_BACKOFF_BASE * (2 ** attempt), _BACKOFF_MAX)
    base *= random.uniform(1 - _JITTER, 1 + _JITTER)
    if retry_after is not None:
        return max(retry_after, base)
    return base


async def _throttle(host: str) -> None:
    """per-host QPS 限速. 保证同 host 两次请求间隔 >= 1/QPS 秒."""
    if not host:
        return
    qps = _qps()
    if qps <= 0:
        return
    async with _qps_lock:
        now = time.monotonic()
        last = _last_request.get(host, 0.0)
        wait = (1.0 / qps) - (now - last)
        if wait > 0:
            await asyncio.sleep(wait)
        _last_request[host] = time.monotonic()


def _maybe_gunzip(body: bytes, headers: Any) -> bytes:
    """Content-Encoding: gzip 时解压. 非 gzip 原样返回."""
    enc = (headers.get("Content-Encoding") if headers else "") or ""
    if "gzip" in enc.lower() and body:
        try:
            return gzip.decompress(body)
        except (OSError, EOFError):
            # 解压失败退回原始 body, 别让一个坏响应把整个请求搞挂
            logger.debug("gzip 解压失败, 退回原始 body")
    return body


def _fetch_sync(url: str, timeout: float, accept: str) -> tuple[int, Any, bytes]:
    """同步执行单次 GET. 返回 (status, headers, body_bytes).

    HTTP 错误状态 (4xx/5xx) 不抛异常, 转成 (code, headers, body) 让上层
    决定要不要重试. 只有网络层错误 (URLError / 超时) 才往外抛.
    """
    headers = {
        "User-Agent": _USER_AGENT,
        "Accept-Encoding": "gzip",
    }
    if accept:
        headers["Accept"] = accept
    req = urllib.request.Request(url, headers=headers)
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            body = resp.read()
            return resp.getcode(), resp.headers, _maybe_gunzip(body, resp.headers)
    except urllib.error.HTTPError as exc:
        # urllib 把 >= 400 当异常抛, 但 body 还能读, 拿来给重试逻辑判断
        try:
            body = exc.read()
        except Exception:
            body = b""
        return exc.code, exc.headers, _maybe_gunzip(body, exc.headers)


async def _http_request(
    url: str, timeout: float, accept: str = ""
) -> tuple[int, Any, bytes]:
    """统一入口: 限速 -> 请求 -> 重试. 返回 (status, headers, body_bytes).

    重试策略: 5xx/429 + 网络错误重试最多 _MAX_RETRIES 次, 指数退避 +
    Retry-After. 4xx 直接抛 HttpError 不重试.
    """
    host = urllib.parse.urlparse(url).hostname or ""

    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        await _throttle(host)
        try:
            status, headers, body = await asyncio.to_thread(
                _fetch_sync, url, timeout, accept
            )
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            # 网络层错误 (DNS / 拒绝连接 / 超时), 也值得重试
            last_exc = exc
            if attempt >= _MAX_RETRIES:
                raise HttpError(0, str(exc), url) from exc
            logger.debug("%s 网络错误 %s, 第 %d 次重试", host, exc, attempt + 1)
            await asyncio.sleep(_backoff_delay(attempt, None))
            continue
        except Exception as exc:
            # 别的异常 (编码问题之类) 不重试, 直接抛
            raise HttpError(0, str(exc), url) from exc

        if status in RETRYABLE_STATUS_CODES and attempt < _MAX_RETRIES:
            retry_after = _parse_retry_after(headers)
            logger.debug(
                "%s 返回 %d, 第 %d 次重试 (等 %.1fs)",
                host, status, attempt + 1, _backoff_delay(attempt, retry_after),
            )
            await asyncio.sleep(_backoff_delay(attempt, retry_after))
            continue

        if status >= 400:
            raise HttpError(status, body, url)

        return status, headers, body

    # 兜底, 正常控制流走不到这
    raise HttpError(0, str(last_exc) if last_exc else "retries exhausted", url)


async def _http_get_json(url: str, timeout: float | None = None) -> dict[str, Any]:
    """GET JSON. 失败抛 HttpError, 让调用方决定怎么降级."""
    t = timeout if timeout is not None else _timeout()
    _, _, body = await _http_request(url, t, "application/json")
    return json.loads(body.decode("utf-8", errors="replace"))


async def _http_get_text(url: str, timeout: float | None = None) -> str:
    t = timeout if timeout is not None else _timeout()
    _, _, body = await _http_request(url, t, "text/html, text/plain, */*")
    return body.decode("utf-8", errors="replace")


async def _http_get_bytes(url: str, timeout: float | None = None) -> bytes:
    """下载二进制 (PDF). fetch_pdf 用."""
    t = timeout if timeout is not None else _timeout() * 2  # PDF 大, 给双倍超时
    _, _, body = await _http_request(url, t, "*/*")
    return body
