"""主动降级协调器 — 工具熔断时不依赖 LLM 即兴发挥，按声明的降级链自动尝试替代。

核心设计:
  工具 A 熔断 → 查 ToolProfile.degradation_chain → [B, C, D]
  → 自动尝试 B → 成功则返回结果 + degraded_from 标记
  → B 也熔断 → 尝试 C → ...
  → 全部熔断 → 返回结构化降级报告，LLM 拿到的是"已尝试全部替代"
    而非裸 error，决策上下文充分

与 CircuitBreaker 的关系:
  CircuitBreaker 是"门卫" — 决定能不能调
  DegradationCoordinator 是"备选方案执行者" — 门关了就走后门

材料基因工程启发:
  MGE 的质量层次: DFT HSE06 > DFT PBE > ML surrogate > database lookup
  降级链按此层次声明，结果标记 quality_tier 让下游知道数据可信度
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def _try_fallback(
    tool_name: str,
    tool_input: dict[str, Any],
    context: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """尝试调用一个替代工具，成功返回结果 dict，失败返回 None。

    ponytail: 复用 ToolRegistry + tool.call()，不新建执行器。
    tool.call() 是 async 的，这里用 asyncio.run 桥接同步调用方。
    """
    try:
        from huginn.tools.registry import ToolRegistry

        tool = ToolRegistry.get(tool_name)
        if tool is None:
            return None

        # 熔断器检查 — 替代工具也可能熔断
        from huginn.tools.adapter import _breaker_blocked

        blocked = _breaker_blocked(tool_name)
        if blocked is not None:
            logger.debug("degradation: fallback %s also blocked", tool_name)
            return None

        # 构造 ToolContext — 复用主调用的 context，没有就建一个最小的
        from huginn.types import ToolContext

        if isinstance(context, ToolContext):
            ctx = context
        elif isinstance(context, dict):
            ctx = ToolContext(
                session_id=context.get("session_id", "degradation"),
                workspace=context.get("workspace", "."),
            )
        else:
            ctx = ToolContext(session_id="degradation", workspace=".")

        # tool.call() 是 async，用 _run_sync 桥接
        result = _run_sync(tool.call, tool_input, ctx)

        # ToolResult → dict
        if hasattr(result, "to_dict"):
            d = result.to_dict()
            if result.success:
                return d.get("data", d) if isinstance(d.get("data"), dict) else d
            else:
                return None  # 工具执行失败
        if isinstance(result, dict):
            return result
        return {"result": result}
    except Exception as e:
        logger.debug("degradation: fallback %s failed: %s", tool_name, e)
        return None


def _run_sync(coro_func, *args, **kwargs):
    """同步调用 async 函数 — 复用 event loop 或新建一个."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        # 已在 event loop 中 — 用 to_thread 避免阻塞
        import concurrent.futures
        import threading

        # ponytail: 新建独立 event loop 在子线程中跑，不阻塞主 loop
        result_holder: dict[str, Any] = {}

        def _run_in_new_loop():
            new_loop = asyncio.new_event_loop()
            try:
                result_holder["value"] = new_loop.run_until_complete(
                    coro_func(*args, **kwargs)
                )
            except Exception as e:
                result_holder["error"] = e
            finally:
                new_loop.close()

        t = threading.Thread(target=_run_in_new_loop)
        t.start()
        t.join()
        if "error" in result_holder:
            raise result_holder["error"]
        return result_holder.get("value")
    else:
        return asyncio.run(coro_func(*args, **kwargs))


def try_with_degradation(
    primary_tool: str,
    tool_input: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """主工具熔断时，按声明的降级链自动尝试替代方案。

    返回值结构:
      - 主工具可用: 直接返回 {"_degraded": False, ...原结果}
      - 降级成功:   {"_degraded": True, "_degraded_from": "primary",
                       "_degraded_to": "fallback", "_quality_tier": "...", ...结果}
      - 全部失败:   {"error": "all_tools_blocked", "primary": primary,
                       "tried": ["B", "C"], "_circuit_open": True}

    这个函数不替代正常工具调用 — 只在 _breaker_blocked 返回非 None 时被调用。
    """
    from huginn.tools.registry import ToolRegistry

    # 查主工具的降级链
    primary = ToolRegistry.get(primary_tool)
    chain: tuple[str, ...] = ()
    quality_tier = ""
    if primary and primary.profile:
        chain = primary.profile.degradation_chain
        quality_tier = primary.profile.quality_tier

    if not chain:
        # 没有降级链 — 回退到旧行为（返回 circuit_open 错误）
        return {
            "error": "circuit_open",
            "tool": primary_tool,
            "_circuit_open": True,
            "_no_degradation_chain": True,
        }

    tried: list[str] = []
    for fallback_name in chain:
        tried.append(fallback_name)
        logger.info(
            "degradation: %s blocked, trying fallback %s", primary_tool, fallback_name
        )
        result = _try_fallback(fallback_name, tool_input, context)
        if result is not None and "error" not in result:
            # 降级成功 — 标记结果质量
            result["_degraded"] = True
            result["_degraded_from"] = primary_tool
            result["_degraded_to"] = fallback_name
            # 取替代工具的 quality_tier
            fallback_tool = ToolRegistry.get(fallback_name)
            if fallback_tool and fallback_tool.profile:
                result["_quality_tier"] = fallback_tool.profile.quality_tier or "unknown"
            else:
                result["_quality_tier"] = "unknown"
            logger.info(
                "degradation: %s → %s succeeded (quality=%s)",
                primary_tool,
                fallback_name,
                result["_quality_tier"],
            )
            return result

    # 全部替代也熔断/失败 — 返回结构化报告
    return {
        "error": "all_tools_blocked",
        "primary": primary_tool,
        "tried": tried,
        "_circuit_open": True,
        "_degradation_exhausted": True,
        "quality_tier_original": quality_tier,
        # 给 LLM 的上下文：不是"工具挂了你看着办"，
        # 而是"主工具和 N 个替代都不可用，这是已尝试的清单"
        "message": (
            f"Primary tool '{primary_tool}' and {len(tried)} fallback(s) "
            f"({', '.join(tried)}) are all unavailable (circuit open). "
            f"Consider: 1) wait for circuit reset, 2) use a different approach, "
            f"3) use knowledge base or literature search for approximate values."
        ),
    }
