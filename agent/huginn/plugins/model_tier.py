"""模型档位 profile —— 极简模式的落地 (M1).

设计背景: 我们的工作流 (phase 机 / plan gating / 认知纪律 / 重 compaction /
记忆) 本质是对弱本地模型的补偿。对顶尖大模型, 这些补偿的边际收益骤降,
而代价真实 (token 开销 / 认知摩擦 / 延迟)。oh-my-pi 的结论是: 束缚强模型的
不是"结构本身", 而是"常驻的开销 + 反直觉的摩擦"。

因此极简模式不是"砍功能", 而是把"认知编排"从常驻改为事件驱动 + 按模型档位
聚合。本模块定义三档 profile, 每个档位聚合一组开关:

- full     : 本地弱模型, 保留全部认知编排 (常驻纪律 + phase 门控)
- balanced : 中等模型, 认知纪律降级为事件驱动 (只偏离才注入)
- minimal  : 顶尖大模型, 跳过 phase/plan 门控, 事件驱动守护, 轻 compaction

安全层 (命令校验 / 物理 sanity check / 资源预算警告) 在所有档位都保留 —
裁剪的是"管模型的架构", 不是"管安全的架构"。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum


class ModelTier(StrEnum):
    """模型档位."""

    FULL = "full"
    BALANCED = "balanced"
    MINIMAL = "minimal"


@dataclass(frozen=True)
class TierProfile:
    """一个模型档位的聚合配置."""

    tier: ModelTier
    # 认知编排开关
    use_phase_machine: bool = True  # False 时 phase 映射到 OPEN (无约束)
    use_plan_gating: bool = True  # False 时跳过 plan 门控, 模型自行规划
    # cognitive discipline: "always" 常驻 prompt / "event" 事件驱动守护
    cognitive_discipline: str = "always"
    # compaction/记忆档位: heavy / medium / light
    compaction_tier: str = "heavy"
    # external thinking 显式开关 (M4)
    external_thinking: bool = True


# 三档内置 profile. 以 frozen dataclass 常量声明, 避免外部改写.
_TIERS: dict[ModelTier, TierProfile] = {
    ModelTier.FULL: TierProfile(
        tier=ModelTier.FULL,
        use_phase_machine=True,
        use_plan_gating=True,
        cognitive_discipline="always",
        compaction_tier="heavy",
        external_thinking=True,
    ),
    ModelTier.BALANCED: TierProfile(
        tier=ModelTier.BALANCED,
        use_phase_machine=True,
        use_plan_gating=True,
        cognitive_discipline="event",
        compaction_tier="medium",
        external_thinking=True,
    ),
    ModelTier.MINIMAL: TierProfile(
        tier=ModelTier.MINIMAL,
        use_phase_machine=False,  # -> OPEN
        use_plan_gating=False,
        cognitive_discipline="event",
        compaction_tier="light",
        external_thinking=False,  # 默认关, 可按需开 (M4)
    ),
}


class TierProfileStore:
    """当前模型档位的持有者 (单例, 运行时/前端可切换)."""

    def __init__(self, initial: ModelTier | None = None) -> None:
        self._tier = initial or self._default_tier()

    @staticmethod
    def _default_tier() -> ModelTier:
        """从环境变量取默认档位, 无效值回退 FULL."""
        raw = os.environ.get("HUGINN_MODEL_TIER", "full").strip().lower()
        try:
            return ModelTier(raw)
        except ValueError:
            return ModelTier.FULL

    @property
    def tier(self) -> ModelTier:
        return self._tier

    def set_tier(self, tier: ModelTier) -> None:
        self._tier = tier

    def profile(self) -> TierProfile:
        return _TIERS[self._tier]


_shared: TierProfileStore | None = None


def get_store() -> TierProfileStore:
    """获取全局档位 store (惰性单例)."""
    global _shared
    if _shared is None:
        _shared = TierProfileStore()
    return _shared


def get_tier() -> ModelTier:
    """当前激活的模型档位."""
    return get_store().tier


def set_tier(tier: ModelTier) -> None:
    """运行时切换模型档位 (前端设置项会调用这里).

    M4: 切换档位时联动 external_thinking 功能开关 — prompt 注入点读的是
    FeatureFlags, 这里按档位 profile 同步, 保证 tier 与 flag 一致.
    """
    get_store().set_tier(tier)
    try:
        from huginn.feature_flags import FeatureFlags

        ff = FeatureFlags.shared()
        if get_profile().external_thinking:
            ff.enable(_PROFILE_EXTERNAL_THINKING_FLAG)
        else:
            ff.disable(_PROFILE_EXTERNAL_THINKING_FLAG)
    except Exception:
        pass  # flag 同步失败不回滚 tier 切换


# external_thinking 在 FeatureFlags 里的登记名 (T4 联动用).
_PROFILE_EXTERNAL_THINKING_FLAG = "external_thinking"


def get_profile() -> TierProfile:
    """当前档位的聚合 profile."""
    return get_store().profile()


@dataclass(frozen=True)
class CompactionKnobs:
    """按档位映射的 compaction 力度旋钮 (M3).

    以"乘子"形式提供, 作用于 streaming.py 里已有的 keep_last_n / keep_root_n
    决策 (adaptive / CSM / trace 逻辑保持原样, 只按档位整体放宽或收紧):

    - ``keep_multiplier``: 乘到 keep_last_n 上 (>1 保留更多原始消息).
    - ``root_multiplier``: 乘到 keep_root_n 上 (>1 保留更多稳定前缀).
    - ``summarize``: 预算超限时是否优先用 LLM 摘要 (light 档留给强模型原样保留).
    """

    keep_multiplier: float
    root_multiplier: float
    summarize: bool


def compaction_knobs() -> CompactionKnobs:
    """当前档位的 compaction 旋钮.

    heavy/medium 保持默认激进裁剪 (乘子 1.0, 照顾弱模型的小上下文); light
    (minimal 档) 整体放宽 2 倍, 强模型上下文大, 不必为省 token 牺牲细粒度.
    安全层不受影响.
    """
    tier = get_tier()
    if tier is ModelTier.MINIMAL:
        return CompactionKnobs(keep_multiplier=2.0, root_multiplier=1.5, summarize=True)
    return CompactionKnobs(keep_multiplier=1.0, root_multiplier=1.0, summarize=True)


__all__ = [
    "ModelTier",
    "TierProfile",
    "TierProfileStore",
    "get_profile",
    "get_store",
    "get_tier",
    "set_tier",
]