"""视觉渲染链的空间可组合性 (Cordis 论文 spatial composability).

视觉工作流是一条明确的依赖链: 生成(visualize_tool) → QA(visualize_qa)
→ 一致性(visualize_check) → 门禁(visualize_gate). 任一环节缺失
(如 matplotlib/vision 编码器不可用), 下游应自动停用, 而不是继续跑
无依据的门禁判定.

本模块用 ``CoEffectRegistry`` 声明这条链, 暴露一个惰性单例 registry,
并给 ``visualize_gate`` 提供 ``gate_available()`` 查询 — 门禁组件是否激活
由依赖推导 (声明即保证), 不靠手动开关.
"""

from __future__ import annotations

import logging
from typing import Any

from huginn.security.coeffect import CoEffectRegistry

logger = logging.getLogger(__name__)

# 依赖链 produce 键 (registry 里声明为组件 provides/requires).
K_GENERATE = "visual.generate"      # 生成器可用 (matplotlib 等渲染后端)
K_QA = "visual.qa"                  # QA 可用 (PIL 可读图)
K_CONSISTENCY = "visual.consistency"  # 一致性校验可用 (vision 编码器)
K_GATE = "visual.gate"              # 门禁可用 (依赖 QA + 一致性)

# 门禁组件的便捷查询: 链是否完整可用.
def gate_available() -> bool:
    """门禁是否激活 — 由依赖链推导 (QA + 一致性都可用才为真)."""
    return _registry().is_active("visual_gate")


def availability() -> dict[str, bool]:
    """当前整条视觉链的可用性快照 (排障/审计用)."""
    reg = _registry()
    return {
        "generate": reg.is_available(K_GENERATE),
        "qa": reg.is_available(K_QA),
        "consistency": reg.is_available(K_CONSISTENCY),
        "gate": reg.is_active("visual_gate"),
    }


def set_dependency_available(key: str, available: bool) -> None:
    """声明某个视觉环节后端可用性 (运行时由接入方调用)."""
    _registry().set_available(key, available)


def _registry() -> CoEffectRegistry:
    """惰性构建并接线视觉链的单例 registry."""
    global _VISUAL_REGISTRY
    if _VISUAL_REGISTRY is None:
        reg = CoEffectRegistry()
        # 生成器: 无 requires (后端本就绪即为激活), produces generate.
        reg.declare("visual_generator", provides=(K_GENERATE,))
        # QA: requires 生成产物可见, produces qa.
        reg.declare("visual_qa_comp", requires=(K_GENERATE,), provides=(K_QA,))
        # 一致性: requires 生成产物, produces consistency.
        reg.declare(
            "visual_consistency_comp",
            requires=(K_GENERATE,),
            provides=(K_CONSISTENCY,),
        )
        # 门禁: requires QA + 一致性, produces gate.
        reg.declare(
            "visual_gate",
            requires=(K_QA, K_CONSISTENCY),
            provides=(K_GATE,),
            on_change=_on_gate_change,
        )
        _VISUAL_REGISTRY = reg
    return _VISUAL_REGISTRY


def _on_gate_change(key: str, action: str) -> None:
    """门禁激活/停用事件 — 记录日志供审计 (不改变调用方行为)."""
    logger.info(
        "visual chain: gate %s (dep=%s)", action, key,
    )


# 模块级单例 (惰性初始化).
_VISUAL_REGISTRY: CoEffectRegistry | None = None


def _reset_for_tests() -> None:
    """测试用: 重置单例, 便于隔离验证."""
    global _VISUAL_REGISTRY
    _VISUAL_REGISTRY = None