"""LLM 调用的重试与降级机制.

参考 Claude Code 的 services/api/withRetry.ts, 给 agent 加一层统一的容错:
- 429 限流: 按 retry-after 头等待
- 529 过载: 指数退避, 连续 N 次后触发降级信号
- 上下文溢出: 自动收敛 max_tokens
- 网络抖动: 退避 + 抖动
- 无人值守模式: persistent_retry 长时间重试

不绑死任何一家 SDK, 靠异常的 status_code 属性 + 消息文本判断类型.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# 529 连续触发多少次后切换到更便宜的模型
MAX_529_RETRIES = 3

# 退避参数 (秒)
INITIAL_BACKOFF = 1.0
MAX_BACKOFF = 60.0

# 上下文溢出时, 新 max_tokens 占上下文窗口的比例, 给 prompt 留点余量
CONTEXT_OVERFLOW_SHRINK = 0.5
# max_tokens 兜底下限, 别压到没法生成
MIN_MAX_TOKENS = 256

# 主模型 -> 便宜模型的映射, 命中 529 风暴时切换
FALLBACK_MODELS: dict[str, str] = {
    "claude-sonnet-4-6": "claude-haiku-4",
    "gpt-4o": "gpt-4o-mini",
    "deepseek-chat": "deepseek-coder",
    "moonshot-v1-128k": "moonshot-v1-8k",
}

__all__ = [
    "MAX_529_RETRIES",
    "FALLBACK_MODELS",
    "FallbackTriggeredError",
    "with_retry",
    "call_with_fallback",
    "parse_context_overflow",
    "persistent_retry",
]


class FallbackTriggeredError(Exception):
    """529 连续触发时抛出, 信号上层切到更便宜的模型."""

    def __init__(self, message: str = "529 overloaded, fallback to cheaper model") -> None:
        super().__init__(message)
        self.reason = "529_overloaded"


# ---- 异常属性探测 -----------------------------------------------------------

def _get_status_code(exc: BaseException) -> int | None:
    """尽量从各种 SDK 异常里挖出 HTTP status code.

    不同库挂的位置不一样: openai 错误有 status_code, httpx 有 response.status_code,
    有的自定义异常直接叫 status. 都试一遍.
    """
    code = getattr(exc, "status_code", None)
    if code is None:
        code = getattr(exc, "status", None)
    if code is None:
        resp = getattr(exc, "response", None)
        if resp is not None:
            code = getattr(resp, "status_code", None) or getattr(resp, "status", None)
    try:
        return int(code) if code is not None else None
    except (TypeError, ValueError):
        return None


def _get_retry_after(exc: BaseException) -> float | None:
    """从异常或 response header 里挖 retry-after (秒).

    只处理数字形式 (秒), HTTP-date 格式不解析, 调用方自己退避就行.
    """
    raw = getattr(exc, "retry_after", None)
    if raw is None:
        resp = getattr(exc, "response", None)
        if resp is not None:
            headers = getattr(resp, "headers", None) or {}
            # httpx / requests / aiohttp 的 headers 大小写不敏感, 但兜底一下
            try:
                raw = headers.get("retry-after") or headers.get("Retry-After")
            except AttributeError:
                for k, v in headers.items():
                    if k.lower() == "retry-after":
                        raw = v
                        break
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


# ---- 上下文溢出解析 ---------------------------------------------------------

# 各家 SDK 溢出报错的常见文本模式, 抓出上下文窗口大小
_OVERFLOW_PATTERNS = [
    # openai: "maximum context length is 8192 tokens"
    re.compile(r"maximum context length is (\d+) tokens", re.IGNORECASE),
    # "maximum allowed 8192 tokens"
    re.compile(r"maximum(?:\s+allowed)?\s+(\d+)\s+tokens", re.IGNORECASE),
    # "max is 8192"
    re.compile(r"max(?:imum)?\s+is\s+(\d+)", re.IGNORECASE),
    # anthropic: "context window is 200000 tokens"
    re.compile(r"context\s+window\s+(?:is\s+)?(\d+)", re.IGNORECASE),
]


def parse_context_overflow(error: BaseException) -> int | None:
    """从溢出错误里推断出建议的 max_tokens.

    匹配各家 SDK 的常见报错文本, 抓出上下文窗口大小, 按
    ``CONTEXT_OVERFLOW_SHRINK`` 比例缩一下作为新的 max_tokens.

    Returns
    -------
    int | None
        建议的 max_tokens; 返回 ``None`` 表示这个错误不像上下文溢出.
    """
    text = str(error)
    text_lower = text.lower()
    cls_name = type(error).__name__.lower()
    err_code = str(getattr(error, "code", "") or "").lower()

    # 先试着匹配具体模式, 命中就直接判定为溢出
    for pat in _OVERFLOW_PATTERNS:
        m = pat.search(text)
        if m:
            context_size = int(m.group(1))
            return max(MIN_MAX_TOKENS, int(context_size * CONTEXT_OVERFLOW_SHRINK))

    # 没匹配到模式, 再用类名/错误码/关键词粗筛
    looks_like_overflow = (
        "context" in cls_name
        or "overflow" in cls_name
        or "context_length" in err_code
        or "context_length" in text_lower
        or "context length" in text_lower  # 消息文本里常见带空格的写法
        or ("context" in text_lower and "exceed" in text_lower)
        or ("context" in text_lower and "overflow" in text_lower)
    )
    if not looks_like_overflow:
        return None

    # 看着像溢出但没挖到数字, 给个保守默认值
    return max(MIN_MAX_TOKENS, int(4096 * CONTEXT_OVERFLOW_SHRINK))


# ---- 异常分类 ---------------------------------------------------------------

def _is_rate_limit(exc: BaseException) -> bool:
    """429 限流."""
    if _get_status_code(exc) == 429:
        return True
    name = type(exc).__name__.lower()
    return "ratelimit" in name or "rate_limit" in name


def _is_overloaded(exc: BaseException) -> bool:
    """529 过载 (Anthropic) 或类似的 '暂时不可用'."""
    if _get_status_code(exc) == 529:
        return True
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return "overloaded" in name or "overloaded" in text or "service unavailable" in text


def _is_context_overflow(exc: BaseException) -> bool:
    """上下文窗口溢出."""
    return parse_context_overflow(exc) is not None


def _is_transient_network(exc: BaseException) -> bool:
    """可重试的网络错误 (超时 / 连接重置 / 5xx 除 529)."""
    code = _get_status_code(exc)
    if code is not None and 500 <= code < 600 and code != 529:
        return True
    name = type(exc).__name__.lower()
    return any(
        kw in name
        for kw in ("timeout", "connection", "network", "reset", "brokenpipe")
    )


def _is_auth_error(exc: BaseException) -> bool:
    """401 认证失败, 可能是 token 过期, 刷新后可重试."""
    code = _get_status_code(exc)
    if code == 401:
        return True
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return (
        "authentication" in name
        or "unauthorized" in name
        or "invalid api key" in text
        or "invalid_authentication" in text
    )


# ---- 退避计算 ---------------------------------------------------------------

def _exponential_backoff(
    attempt: int, base: float = INITIAL_BACKOFF, cap: float = MAX_BACKOFF
) -> float:
    """指数退避, 2^(n-1) * base, 上限 cap."""
    return min(base * (2 ** (attempt - 1)), cap)


def _jitter(seconds: float, jitter_ratio: float = 0.25) -> float:
    """加一点抖动, 避免雷鸣群效应."""
    delta = seconds * jitter_ratio
    return max(0.0, seconds + random.uniform(-delta, delta))


async def _sleep_with_log(seconds: float, reason: str, attempt: int) -> None:
    """带日志的 sleep, 方便排障."""
    logger.info("retry sleep %.2fs (attempt=%d, reason=%s)", seconds, attempt, reason)
    await asyncio.sleep(seconds)


# ---- 分级重试 ---------------------------------------------------------------

async def with_retry(
    coro_factory: Callable[[], Awaitable[T]],
    source: str = "foreground",
    max_attempts: int = 5,
    max_tokens_adjuster: Callable[[int], int] | None = None,
) -> T:
    """分级重试包装器.

    Parameters
    ----------
    coro_factory
        每次重试时调用的工厂, 返回新的 coroutine. 注意传函数本身, 不是已构造的 coroutine.
    source
        调用来源, 仅用于日志 ("foreground" / "background" / "batch").
    max_attempts
        最大尝试次数 (含首次).
    max_tokens_adjuster
        可选回调, 收到溢出建议时把新的 max_tokens 喂回上层. 比如
        ``lambda new: settings.set_max_tokens(new)``.

    Raises
    ------
    FallbackTriggeredError
        529 连续触发 ``MAX_529_RETRIES`` 次, 提示上层切模型.
    """
    consecutive_529 = 0
    last_error: BaseException | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return await coro_factory()
        except Exception as exc:  # noqa: BLE001 - 重试逻辑就是要吞所有错误再判断
            last_error = exc

            # 429 限流: 按 retry-after 等待, 没有就走指数退避
            if _is_rate_limit(exc):
                consecutive_529 = 0
                wait = _get_retry_after(exc)
                if wait is None:
                    wait = _jitter(_exponential_backoff(attempt))
                else:
                    # 服务端给的 retry-after 我们尊重, 只加一点点抖动
                    wait = _jitter(wait, jitter_ratio=0.1)
                logger.warning(
                    "[%s] 429 rate limit, retry in %.2fs (attempt %d/%d)",
                    source, wait, attempt, max_attempts,
                )
                if attempt >= max_attempts:
                    break
                await _sleep_with_log(wait, "rate_limit", attempt)
                continue

            # 529 过载: 连续 N 次后切模型
            if _is_overloaded(exc):
                consecutive_529 += 1
                if consecutive_529 >= MAX_529_RETRIES:
                    logger.error(
                        "[%s] 529 overloaded %d times in a row, triggering fallback",
                        source, consecutive_529,
                    )
                    raise FallbackTriggeredError(
                        f"529 overloaded {consecutive_529} times in a row"
                    ) from exc
                wait = _jitter(_exponential_backoff(consecutive_529))
                logger.warning(
                    "[%s] 529 overloaded, retry in %.2fs (streak=%d, attempt %d/%d)",
                    source, wait, consecutive_529, attempt, max_attempts,
                )
                if attempt >= max_attempts:
                    break
                await _sleep_with_log(wait, "overloaded", attempt)
                continue

            # 上下文溢出: 收缩 max_tokens 再试
            if _is_context_overflow(exc):
                consecutive_529 = 0
                suggested = parse_context_overflow(exc) or MIN_MAX_TOKENS
                logger.warning(
                    "[%s] context overflow, shrink max_tokens to %d (attempt %d/%d)",
                    source, suggested, attempt, max_attempts,
                )
                if max_tokens_adjuster is not None:
                    max_tokens_adjuster(suggested)
                if attempt >= max_attempts:
                    break
                # 改了参数大概率能成, 不用退避太久
                await _sleep_with_log(_jitter(0.5), "context_overflow", attempt)
                continue

            # 普通 5xx / 网络错误: 指数退避 + 抖动
            if _is_transient_network(exc):
                consecutive_529 = 0
                wait = _jitter(_exponential_backoff(attempt))
                logger.warning(
                    "[%s] transient error (%s), retry in %.2fs (attempt %d/%d)",
                    source, type(exc).__name__, wait, attempt, max_attempts,
                )
                if attempt >= max_attempts:
                    break
                await _sleep_with_log(wait, "transient", attempt)
                continue

            # 401 认证错误: 尝试刷新 token, 成功则重试
            # 用 auth 模块的 handle_401_error 做去重, 同一 token 的并发
            # 401 只触发一次刷新
            if _is_auth_error(exc):
                try:
                    # 延迟导入避免循环依赖
                    from huginn.security.auth import handle_401_error

                    refreshed = await handle_401_error(
                        str(getattr(exc, "response", "") or "")
                    )
                except Exception:
                    refreshed = False
                if refreshed and attempt < max_attempts:
                    logger.info(
                        "[%s] 401 auth refreshed, retrying (attempt %d/%d)",
                        source, attempt, max_attempts,
                    )
                    continue
                # 刷新失败或不支持刷新, 直接抛
                logger.error("[%s] 401 auth error, not retryable: %r", source, exc)
                raise

            # 其它异常不重试, 直接抛
            logger.error("[%s] non-retryable error: %r", source, exc)
            raise

    assert last_error is not None
    raise last_error


# ---- 模型降级 ---------------------------------------------------------------

async def call_with_fallback(
    prompt: Any,  # noqa: ANN401 - prompt 类型由 llm_call_fn 决定, 透传即可
    primary_model: str,
    llm_call_fn: Callable[[Any, str], Awaitable[T]],
) -> T:
    """先用主模型试, 529 风暴时切到便宜模型.

    Parameters
    ----------
    prompt
        喂给 LLM 的 prompt (字符串 / 消息列表都行, 透传给 ``llm_call_fn``).
    primary_model
        首选模型名, 比如 ``"claude-sonnet-4-6"``.
    llm_call_fn
        实际的调用函数, 签名 ``(prompt, model_name) -> awaitable``.
    """
    fallback_model = FALLBACK_MODELS.get(primary_model)
    if fallback_model is None:
        # 没有便宜档可切, 直接调
        return await llm_call_fn(prompt, primary_model)

    async def _primary_call() -> T:
        return await llm_call_fn(prompt, primary_model)

    try:
        return await with_retry(_primary_call, source="primary")
    except FallbackTriggeredError:
        logger.warning(
            "primary model %s overloaded, falling back to %s",
            primary_model, fallback_model,
        )

    async def _fallback_call() -> T:
        return await llm_call_fn(prompt, fallback_model)

    # 便宜模型也走一遍重试, 529 再触发就直接抛上去
    return await with_retry(_fallback_call, source="fallback")


# ---- 无人值守长时重试 -------------------------------------------------------

async def persistent_retry(
    coro_factory: Callable[[], Awaitable[T]],
    max_hours: float = 6.0,
    heartbeat_interval: float = 30.0,
) -> T:
    """无人值守模式的长时重试.

    适合后台批处理: 只要不是 fatal 错误就一直试, 直到 ``max_hours`` 用完.
    每隔 ``heartbeat_interval`` 秒打一次心跳日志, 方便监控.

    注意: ``FallbackTriggeredError`` 在这里被吞掉并退避重试 —— 无人值守场景
    没有上层接降级信号, 适合已经选好模型、不想再切的情况.

    Parameters
    ----------
    coro_factory
        每次重试时调用的工厂.
    max_hours
        最长重试时长 (小时).
    heartbeat_interval
        心跳日志间隔 (秒).
    """
    deadline = time.monotonic() + max_hours * 3600.0
    attempt = 0
    last_heartbeat = time.monotonic()

    while time.monotonic() < deadline:
        attempt += 1
        try:
            return await coro_factory()
        except FallbackTriggeredError:
            # 无人值守没有 fallback 上层, 退避后继续试
            wait = _jitter(_exponential_backoff(min(attempt, 8)))
            logger.warning(
                "[batch] overloaded (fallback signal swallowed), retry in %.2fs",
                wait,
            )
            await asyncio.sleep(min(wait, max(0.0, deadline - time.monotonic())))
            continue
        except Exception as exc:  # noqa: BLE001
            if not (
                _is_rate_limit(exc)
                or _is_overloaded(exc)
                or _is_transient_network(exc)
                or _is_context_overflow(exc)
            ):
                # 不可重试, 立即抛
                raise

            if _is_rate_limit(exc):
                wait = _get_retry_after(exc) or _exponential_backoff(min(attempt, 8))
            else:
                wait = _exponential_backoff(min(attempt, 8))
            wait = _jitter(wait)

            now = time.monotonic()
            if now - last_heartbeat >= heartbeat_interval:
                logger.info(
                    "[batch] still retrying, attempt=%d, next wait=%.2fs, %.2fh left",
                    attempt, wait, (deadline - now) / 3600.0,
                )
                last_heartbeat = now

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            await asyncio.sleep(min(wait, remaining))

    raise TimeoutError(
        f"persistent_retry exceeded {max_hours}h after {attempt} attempts"
    )
