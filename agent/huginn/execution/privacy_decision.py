"""P1(修正版) — 逐调用数据敏感度判定: 私有化不阻碍正常任务.

原则 (对齐"私有化的前提是不妨碍正常任务"):
- 拦截点是**单条数据的敏感度**, 不是全局开关.
- 命中硬信号 (local_only_tag / ephemeral 级 / never_cloud) → sensitive=True → 必须本地.
- 未命中或无法判定 → sensitive=False → 默认放行 (正常任务不被妨碍, 只落审计提示).
- ``conservative_temporary`` 仅为"私密设备"可选的更严档: 把 experimental 原始数据
  (temporary 级) 也留在本地; 默认 False (普通任务照常).

判定完全复用 PrivacyGuard (classify_data / is_tagged_local / get_retention_policy),
不重复造表; 独立小模块便于 orchestrator 与后端门禁共用同一把尺.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PrivacyDecision:
    """一次判定的结果."""

    sensitive: bool
    reason: str
    # local_only_tag / ephemeral / never_cloud / private_device_temporary / none
    signal: str | None = None


def decide_privacy(
    content: str,
    content_type: str = "auto",
    *,
    privacy: Any = None,
    conservative_temporary: bool = False,
    override: str | None = None,
) -> PrivacyDecision:
    """判定一段内容是否敏感 (需强制本地). 默认放行; 命中硬信号才敏感.

    Parameters
    ----------
    content : 需要判定的内容 (工作参数/路径/原文).
    content_type : 数据形态提示, 默认 auto (走 classify_data 自动检测).
    privacy : 可选注入的 PrivacyGuard; None → 用进程级单例.
    conservative_temporary : 私密设备档. True 时 temporary (实验原始数据) 也留本地.
    override : 用户自由度出口 — 显式覆盖本次判定:
        "allow" → 用户明确同意该调用可外发 (敏感判定被放行, 仍审计);
        "local" → 用户明确要求留本地 (即便不敏感也强制本地);
        None   → 默认规则 (敏感→本地, 否则放行).
    """
    # 0) 用户显式覆盖优先 (自由度): 默认安全兜底, 但用户有最终决定权.
    if override == "allow":
        return PrivacyDecision(
            False, "explicit user override: allow remote for this call",
            signal="user_override_allow",
        )
    if override == "local":
        return PrivacyDecision(
            True, "explicit user override: keep local for this call",
            signal="user_override_local",
        )

    if privacy is None:
        from huginn.privacy_guard import PrivacyGuard

        privacy = PrivacyGuard.shared()

    # 1) 硬信号: 用户标记"始终本地"
    if privacy.is_tagged_local(content):
        return PrivacyDecision(
            True, "matched local_only tag", signal="local_only_tag"
        )

    # 2) 分级: ephemeral (凭证/密钥/内部路径) → 绝不发云端
    tier = privacy.classify_data(content, content_type)
    if tier == "ephemeral":
        return PrivacyDecision(
            True, "ephemeral data (secrets/internal paths)", signal="ephemeral"
        )

    # 3) 保留策略永不发云端
    policy = privacy.get_retention_policy(tier)
    if policy.get("never_cloud"):
        return PrivacyDecision(
            True, "retention policy never_cloud", signal="never_cloud"
        )

    # 4) 可选更严档: 私密设备把实验原始数据也留本地 (不阻碍任务, 只改落点)
    if conservative_temporary and tier == "temporary":
        return PrivacyDecision(
            True, "private device keeps temporary data local",
            signal="private_device_temporary",
        )

    # 5) 默认放行 (normal / unknown)。核心: 无法判定不拦截, 正常任务不被妨碍.
    return PrivacyDecision(False, "not sensitive (default allow)", signal=None)
