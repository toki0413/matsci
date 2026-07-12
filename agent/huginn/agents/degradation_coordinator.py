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

    ponytail: 直接复用 ToolRegistry 的 call 机制，不新建执行器。
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
        # 调用替代工具
        result = tool.run(tool_input, context)
        return result if isinstance(result, dict) else {"result": result}
    except Exception as e:
        logger.debug("degradation: fallback %s failed: %s", tool_name, e)
        return None


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
