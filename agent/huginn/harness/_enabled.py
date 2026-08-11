"""harness 实验性功能统一开关 (enable 条件集中登记).

harness 各模块 (significance_gate / ood_holdout / phase_spec / joint_optimizer /
prompt_patch / bandit H2) 之前各自维护一份 `_harness_enabled`, 读
cfg.feature_flags.<key>. 这里收拢成单一 helper, 让开启方式统一:

  1. huginn.toml 的 [feature_flags] 字段:
       [feature_flags]
       harness_ood_holdout = true
  2. 环境变量 HUGINN_FEATURE_<NAME>=true (FeatureFlags 单例, 见 feature_flags.py):
       HUGINN_FEATURE_HARNESS_OOD_HOLDOUT=true

优先级: cfg.feature_flags 显式值 > FeatureFlags (env/runtime) > default.
默认全部 off — 这些是实验性栅栏, 默认关避免 CI 不稳定, 显式开启才生效.
"""
from __future__ import annotations

from typing import Any


def _harness_enabled(key: str, default: bool = False) -> bool:
    """统一读 harness 实验性开关. 默认 off."""
    try:
        from huginn.config import get_config

        cfg = get_config()
        ff: dict[str, Any] = getattr(cfg, "feature_flags", None) or {}
        if key in ff:
            return bool(ff[key])
    except Exception:
        pass
    # 环境变量 / FeatureFlags 运行时覆盖 (huginn.toml 未显式设时生效)
    try:
        from huginn.feature_flags import FeatureFlags

        return bool(FeatureFlags.shared().is_enabled(key))
    except Exception:
        return default


def harness_enabled(key: str) -> bool:
    """读 harness 实验性开关 (默认 off). 语义别名, 供模块语义化调用."""
    return _harness_enabled(key, default=False)
