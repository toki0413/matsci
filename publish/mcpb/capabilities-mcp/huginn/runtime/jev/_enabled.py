"""JEV 接入的统一开关 — 默认全关 + 叠加隐私外发闸.

参照 harness 实验栅栏 (huginn/harness/_enabled.py) 的三级开关
(cfg.feature_flags 优先 → HUGINN_FEATURE_* env → 默认 off), 但多一条硬约束:
**JEV 是外部托管 API, 材料/研究数据外发必须先过隐私档**. 因此:

  - ``jev_enabled(key)`` = 隐私允许外发 AND 对应 JEV 开关开启.
  - ``jev_egress_allowed()`` = privacy_local_only 时为 False (完全本地, 禁止外发);
    privacy_redact 时也禁 (外发前脱敏), 只有 privacy_off (默认) 才允许.
    这三个档互斥, 见 FeatureFlags._DEFAULTS.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _flag_enabled(key: str, default: bool = False) -> bool:
    """读 JEV 开关 (对齐 harness 的 _harness_enabled). 默认全 off."""
    try:
        from huginn.config import get_config

        cfg = get_config()
        ff: dict[str, object] = getattr(cfg, "feature_flags", None) or {}
        if key in ff:
            return bool(ff[key])
    except Exception:  # noqa: BLE001 — 配置不可读回退 env
        logger.debug("read config feature_flags failed, falling back", exc_info=True)
    try:
        from huginn.feature_flags import FeatureFlags

        return bool(FeatureFlags.shared().is_enabled(key))
    except Exception:  # noqa: BLE001 — FeatureFlags 不可用按默认 off
        return default


def _privacy_level() -> str:
    """读互斥隐私档: off / redact / local_only. 默认 off."""
    try:
        from huginn.feature_flags import FeatureFlags

        shared = FeatureFlags.shared()
        if shared.is_enabled("privacy_local_only"):
            return "local_only"
        if shared.is_enabled("privacy_redact"):
            return "redact"
    except Exception:  # noqa: BLE001 — 读失败按默认 off (允许外发)
        pass
    return "off"


def jev_egress_allowed() -> bool:
    """是否允许把研究数据外发给 JEV 托管端点.

    local_only / redact → False (完全本地 / 需脱敏), 只有 off → True.
    这是 JEV 请求的硬前置, 不满足时任何 JEV 调用都必须短路.
    """
    return _privacy_level() == "off"


def jev_enabled(key: str) -> bool:
    """JEV 对应能力是否可用 = 隐私允许外发 AND 该 key 开关开启.

    key 例如 "jev_tool_router" / "jev_guardrail". 默认 false, 显式开启才生效.
    """
    if not jev_egress_allowed():
        logger.debug("jev: egress blocked by privacy tier, short-circuit %s", key)
        return False
    return _flag_enabled(key, default=False)