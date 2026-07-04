"""LLM 调用限流护栏 —— 防 agent 陷入无限生成烧 token.

设计参考 Moonshine Voice 的 max_tokens_per_second 限流机制. 核心思路:
不让 LLM 在单轮 / 秒级 / 总量三个维度上失控, 在 token 还没花出去之前先挡一刀.

三层限流:
  1. 单轮上限 (max_tokens_per_turn): 一次 agent turn 累计 token 不能超
  2. 秒级速率 (max_tokens_per_second): 滑动窗口 1s 内消费的 token 数有上限,
     专治 LLM 陷入循环反复生成的场景 (Moonshine Voice 就是靠这个拦住无限朗读的)
  3. 总成本上限 (max_total_cost_usd): 累计花费到阈值就停, 兜底防破产

用法上, RateLimitMiddleware (在 agent.py 里) 包一层 LangChain model 的
invoke/stream, 调用前 check_allowed 拦截, 调用后 record_usage 记账.
线程安全, 纯标准库实现.

兼容 Python 3.10+.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "RateLimitConfig",
    "RateLimitExceeded",
    "TokenRateLimiter",
    "get_rate_limiter",
]


class RateLimitExceeded(Exception):
    """限流触发时抛出, 上层捕获后可以决定降级或中断.

    reason 属性标明是哪道闸门挡的:
      - turn_limit     单轮 token 超限
      - second_limit   秒级 token 超限
      - cost_limit      总成本超限
      - limit_exceeded  兜底 (没细分)
    """

    def __init__(self, message: str, reason: str = "limit_exceeded") -> None:
        super().__init__(message)
        self.reason = reason


@dataclass
class RateLimitConfig:
    """限流配置, 默认值按经验给的, 大部分场景不用改.

    可以通过环境变量 HUGINN_RATE_LIMIT_* 覆盖 (见 get_rate_limiter).
    """

    # 单轮 (一个 agent turn) 累计 token 上限
    max_tokens_per_turn: int = 100_000

    # 1s 滑动窗口内消费 token 上限, 这是拦无限循环的关键闸门
    max_tokens_per_second: float = 5000.0

    # 全局累计花费上限 (美元), 到了就停
    max_total_cost_usd: float = 10.0

    # 到达限额的百分之多少时发 warning, 0 表示不发
    warning_threshold: float = 0.8

    # 总开关, False 时所有检查直接放行
    enabled: bool = True


class TokenRateLimiter:
    """线程安全的 token 限流器, 三道闸门: 单轮 / 秒级 / 总成本.

    典型流程::

        limiter = TokenRateLimiter()
        ok, reason = limiter.check_allowed("claude-sonnet-4", 5000)
        if not ok:
            raise RateLimitExceeded(reason)
        # ... 调 LLM ...
        limiter.record_usage("claude-sonnet-4",
                             input_tokens=5200, output_tokens=800, cost=0.024)

    一个 turn 结束后调 reset_turn() 清单轮计数, 全局累计 (token + cost)
    不动. 要彻底清零用 reset_all().
    """

    def __init__(self, config: RateLimitConfig | None = None) -> None:
        self.config = config or RateLimitConfig()
        self._lock = threading.Lock()

        # 全局累计 (跨 turn, 只有 reset_all 才清) — 总成本是全局预算
        self._total_tokens: int = 0
        self._total_cost: float = 0.0

        # 按模型记账, 方便事后排查哪个模型烧得多
        self._per_model: dict[str, dict[str, float]] = {}

        # Per-session state: turn tokens, second window, warnings are
        # isolated per thread_id so one user's burst can't throttle another.
        self._sessions: dict[str, dict[str, Any]] = {}

        self._start_time: float = time.time()

    def _get_session(self, thread_id: str) -> dict[str, Any]:
        """Lazily create per-session counters. Each session gets its own
        turn token counter and sliding window."""
        s = self._sessions.get(thread_id)
        if s is None:
            s = {
                "turn_tokens": 0,
                "turn_cost": 0.0,
                "second_window": deque(),
                "warned": set(),
            }
            self._sessions[thread_id] = s
        return s

    def _prune_session_window(self, s: dict, now: float) -> int:
        """Drop entries older than 1s from the session's sliding window."""
        cutoff = now - 1.0
        win = s["second_window"]
        while win and win[0][0] < cutoff:
            win.popleft()
        return sum(tok for _, tok in win)

    # ---- 内部辅助 ----------------------------------------------------------

    def _maybe_warn(self, s: dict, key: str, ratio: float, label: str) -> None:
        """到阈值发一次 warning, 同一轮内同维度不重复发."""
        thr = self.config.warning_threshold
        if thr <= 0 or ratio < thr:
            return
        if key in s["warned"]:
            return
        s["warned"].add(key)
        logger.warning(
            "限流预警: %s 已用 %.1f%% (阈值 %.0f%%)",
            label, ratio * 100, thr * 100,
        )

    # ---- 公开接口 ----------------------------------------------------------

    def check_allowed(
        self,
        model_name: str,
        estimated_input_tokens: int,
        thread_id: str = "default",
    ) -> tuple[bool, str]:
        """调用前检查这次调用会不会超限.

        返回 (是否放行, 原因). 放行返回 (True, ""), 超限返回 (False, 详细原因).

        注意: 这里只能预估 input token, output 还没产生没法预估, 所以
        秒级/单轮的判定偏宽松 —— 真正的硬上限靠 record_usage 累加后,
        下次 check_allowed 把后续调用拦住.

        thread_id 用于按会话隔离: 单轮上限和秒级速率是 per-session 的,
        一个用户的 burst 不会卡住另一个用户; 总成本是全局共享的.
        """
        if not self.config.enabled:
            return True, ""

        with self._lock:
            now = time.time()
            s = self._get_session(thread_id)
            sec_tokens = self._prune_session_window(s, now)
            est = max(estimated_input_tokens, 0)

            # 闸门 1: 单轮 token 上限 (per-session)
            if s["turn_tokens"] + est > self.config.max_tokens_per_turn:
                return False, (
                    f"单轮 token 超限: 已用 {s['turn_tokens']} + 预估 "
                    f"{est} > 上限 {self.config.max_tokens_per_turn} "
                    f"(model={model_name}, thread={thread_id})"
                )

            # 闸门 2: 秒级速率 (per-session)
            if sec_tokens + est > self.config.max_tokens_per_second:
                return False, (
                    f"秒级 token 超限: 近 1s 已用 {sec_tokens} + 预估 "
                    f"{est} > 上限 {self.config.max_tokens_per_second} "
                    f"(model={model_name}, thread={thread_id})"
                )

            # 闸门 3: 总成本兜底 (全局, 跨所有会话)
            if self._total_cost >= self.config.max_total_cost_usd:
                return False, (
                    f"总成本超限: 已花 ${self._total_cost:.4f} >= "
                    f"上限 ${self.config.max_total_cost_usd:.2f} "
                    f"(model={model_name})"
                )

            # 预警
            self._maybe_warn(
                s, "turn",
                (s["turn_tokens"] + est) / self.config.max_tokens_per_turn,
                f"单轮 token ({model_name})",
            )
            self._maybe_warn(
                s, "second",
                (sec_tokens + est) / self.config.max_tokens_per_second,
                f"秒级 token ({model_name})",
            )
            if self.config.max_total_cost_usd > 0:
                self._maybe_warn(
                    s, "cost",
                    self._total_cost / self.config.max_total_cost_usd,
                    f"总成本 ({model_name})",
                )

            return True, ""

    def record_usage(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        cost: float = 0.0,
        thread_id: str = "default",
    ) -> None:
        """调用后记账, 把实际用量加到各计数器上.

        input/output/cost 都是这次调用真实产生的. 如果拿不到 usage
        (比如 provider 不返回), 传 0 也行, 至少 turn 调用次数能记上.

        turn tokens 和秒级窗口记到 per-session 计数器; 总 token 和总成本
        记到全局计数器 (跨所有会话).
        """
        if not self.config.enabled:
            return

        total = max(input_tokens, 0) + max(output_tokens, 0)
        now = time.time()

        with self._lock:
            s = self._get_session(thread_id)
            s["turn_tokens"] += total
            s["turn_cost"] += cost
            self._total_tokens += total
            self._total_cost += cost

            # 写进 per-session 滑动窗口
            s["second_window"].append((now, total))
            self._prune_session_window(s, now)

            # 按模型汇总 (全局)
            bucket = self._per_model.setdefault(
                model_name,
                {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost": 0.0,
                    "calls": 0,
                },
            )
            bucket["input_tokens"] += input_tokens
            bucket["output_tokens"] += output_tokens
            bucket["cost"] += cost
            bucket["calls"] += 1

    def get_stats(self) -> dict[str, Any]:
        """返回当前各维度统计快照, 给监控 / 日志 / debug 用.

        汇总所有活跃 session 的 turn 计数 + 全局累计.
        """
        with self._lock:
            now = time.time()
            total_turn_tokens = 0
            total_turn_cost = 0.0
            active_sessions = {}
            for tid, s in self._sessions.items():
                sec = self._prune_session_window(s, now)
                total_turn_tokens += s["turn_tokens"]
                total_turn_cost += s["turn_cost"]
                active_sessions[tid] = {
                    "turn_tokens": s["turn_tokens"],
                    "turn_cost": round(s["turn_cost"], 6),
                    "tokens_per_second": sec,
                }
            return {
                "turn_tokens": total_turn_tokens,
                "turn_cost": round(total_turn_cost, 6),
                "total_tokens": self._total_tokens,
                "total_cost": round(self._total_cost, 6),
                "active_sessions": active_sessions,
                "uptime_sec": round(now - self._start_time, 2),
                "per_model": {
                    k: {
                        kk: (round(vv, 6) if isinstance(vv, float) else vv)
                        for kk, vv in v.items()
                    }
                    for k, v in self._per_model.items()
                },
                "limits": {
                    "max_tokens_per_turn": self.config.max_tokens_per_turn,
                    "max_tokens_per_second": self.config.max_tokens_per_second,
                    "max_total_cost_usd": self.config.max_total_cost_usd,
                },
            }

    def reset_turn(self, thread_id: str = "default") -> None:
        """新一轮 turn 开始时调, 清该 session 的单轮计数和预警标记,
        全局累计不动."""
        with self._lock:
            s = self._get_session(thread_id)
            s["turn_tokens"] = 0
            s["turn_cost"] = 0.0
            s["warned"].clear()

    def reset_all(self) -> None:
        """彻底清零 —— 把全局累计和所有 session 状态都清了, 慎用."""
        with self._lock:
            self._total_tokens = 0
            self._total_cost = 0.0
            self._sessions.clear()
            self._per_model.clear()
            self._start_time = time.time()


# ---------------------------------------------------------------------------
# 用量提取辅助 —— 不绑死任何一家 SDK, 靠鸭子类型探测
# ---------------------------------------------------------------------------

def _detect_model_name(model: Any) -> str:
    """尽量从 model 对象上挖出模型名, 挖不到就退回到类名."""
    for attr in ("model_name", "model", "deployment_name", "name"):
        val = getattr(model, attr, None)
        if val and isinstance(val, str):
            return val
    return type(model).__name__


def _estimate_input_tokens(input: Any) -> int:
    """粗估 input 的 token 数, 不用 tokenizer, 按 ~4 字符 / token 算.

    支持 str / list[str] / list[BaseMessage] / tuple 这些常见入参形态.
    估大了不影响正确性 (顶多多拦一次), 估小了也没事 —— record_usage 会
    把真实值记上, 下次 check_allowed 就准了.
    """
    total_chars = 0
    # 拆开看, 可能是字符串、消息列表、prompt value 等
    items = input if isinstance(input, (list, tuple)) else [input]
    for item in items:
        if item is None:
            continue
        if isinstance(item, str):
            total_chars += len(item)
            continue
        # LangChain BaseMessage 有 content 属性; prompt 模板有 .format()
        content = getattr(item, "content", None)
        if content is None:
            # 有些 prompt 对象把内容藏在 messages 里
            messages = getattr(item, "messages", None)
            if messages is not None:
                for m in messages:
                    c = getattr(m, "content", None)
                    if isinstance(c, str):
                        total_chars += len(c)
                    elif c is not None:
                        total_chars += len(str(c))
                continue
            content = str(item)
        elif not isinstance(content, str):
            # content 可能是 list (multimodal) 或 dict
            content = str(content)
        total_chars += len(content)
    # 4 字符约 1 token, 至少算 1, 别让空输入直接 0
    return max(total_chars // 4, 1)


def _extract_usage(result: Any) -> tuple[int, int]:
    """从 LLM 返回结果里挖 input/output token 用量.

    兼容多种返回形态:
      - LangChain AIMessage: usage_metadata / response_metadata
      - Anthropic: response_metadata 顶层的 input_tokens / output_tokens
      - OpenAI: response_metadata.token_usage / usage 子 dict
      - 老 LangChain: llm_output.usage / token_usage

    返回 (input_tokens, output_tokens), 挖不到就 (0, 0).
    """
    in_tok, out_tok = _extract_from_message(result)
    if in_tok or out_tok:
        return in_tok, out_tok

    # 可能是 ChatResult (老版 invoke 返回), 带 llm_output
    llm_output = getattr(result, "llm_output", None)
    if isinstance(llm_output, dict):
        in_tok, out_tok = _extract_from_usage_dict(llm_output)
        if in_tok or out_tok:
            return in_tok, out_tok

    # 最后兜底: result 本身可能就是个 dict
    if isinstance(result, dict):
        return _extract_from_usage_dict(result)

    return 0, 0


def _extract_from_message(msg: Any) -> tuple[int, int]:
    """从单个 message / chunk 对象上提取 usage."""
    # 优先看 LangChain 标准化的 usage_metadata (最靠谱)
    usage_meta = getattr(msg, "usage_metadata", None)
    if isinstance(usage_meta, dict):
        in_tok = _safe_int(usage_meta.get("input_tokens"))
        out_tok = _safe_int(usage_meta.get("output_tokens"))
        if in_tok or out_tok:
            return in_tok, out_tok

    # 再看 response_metadata (provider 原始格式, 各家塞法不一样)
    resp_meta = getattr(msg, "response_metadata", None)
    if isinstance(resp_meta, dict):
        # Anthropic 风格: 顶层直接有 input_tokens / output_tokens
        in_tok = _safe_int(resp_meta.get("input_tokens"))
        out_tok = _safe_int(resp_meta.get("output_tokens"))
        if in_tok or out_tok:
            return in_tok, out_tok

        # OpenAI 风格: 塞在 token_usage 或 usage 子 dict 里
        for key in ("token_usage", "usage"):
            sub = resp_meta.get(key)
            if isinstance(sub, dict):
                in_tok, out_tok = _extract_from_usage_dict(sub)
                if in_tok or out_tok:
                    return in_tok, out_tok

    return 0, 0


def _extract_from_usage_dict(d: dict[str, Any]) -> tuple[int, int]:
    """从 usage dict 里挖 token 数, 兼容各家字段命名."""
    in_tok = _safe_int(
        d.get("input_tokens")
        or d.get("prompt_tokens")
        or d.get("inputTokens")
    )
    out_tok = _safe_int(
        d.get("output_tokens")
        or d.get("completion_tokens")
        or d.get("outputTokens")
    )
    return in_tok, out_tok


def _safe_int(val: Any) -> int:
    """把各种类型的值安全转成 int, 转不了返回 0."""
    if val is None:
        return 0
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# 模块级单例 —— 读环境变量配置
# ---------------------------------------------------------------------------

_singleton: TokenRateLimiter | None = None
_singleton_lock = threading.Lock()


def _build_from_env() -> TokenRateLimiter:
    """从 HUGINN_RATE_LIMIT_* 环境变量构建限流器."""
    cfg = RateLimitConfig(
        max_tokens_per_turn=int(
            os.environ.get("HUGINN_RATE_LIMIT_TOKENS_PER_TURN", "100000")
        ),
        max_tokens_per_second=float(
            os.environ.get("HUGINN_RATE_LIMIT_TOKENS_PER_SECOND", "5000")
        ),
        max_total_cost_usd=float(
            os.environ.get("HUGINN_RATE_LIMIT_TOTAL_COST_USD", "10.0")
        ),
        warning_threshold=float(
            os.environ.get("HUGINN_RATE_LIMIT_WARNING_THRESHOLD", "0.8")
        ),
        enabled=os.environ.get("HUGINN_RATE_LIMIT_ENABLED", "1")
        not in ("0", "false", "no", "False"),
    )
    return TokenRateLimiter(cfg)


def get_rate_limiter() -> TokenRateLimiter:
    """获取模块级单例限流器, 读 HUGINN_RATE_LIMIT_* 环境变量.

    第一次调用时构建, 之后复用同一个实例. 线程安全.

    支持的环境变量:
      HUGINN_RATE_LIMIT_TOKENS_PER_TURN      单轮 token 上限 (默认 100000)
      HUGINN_RATE_LIMIT_TOKENS_PER_SECOND    秒级 token 上限 (默认 5000)
      HUGINN_RATE_LIMIT_TOTAL_COST_USD       总成本上限 (默认 10.0)
      HUGINN_RATE_LIMIT_WARNING_THRESHOLD    预警阈值 (默认 0.8)
      HUGINN_RATE_LIMIT_ENABLED              总开关 (默认 1, 设 0/false 关闭)
    """
    global _singleton
    if _singleton is not None:
        return _singleton
    with _singleton_lock:
        if _singleton is not None:
            return _singleton
        _singleton = _build_from_env()
        return _singleton
