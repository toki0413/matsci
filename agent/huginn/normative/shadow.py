"""准入 shadow 评估 — 在真实工具调用路径上评估规范门，但默认不阻断.

这是 P0-b 的 rollout 阶段：把 ``admit()`` 的判定接到工具执行前，记录
``allow / ask / deny / degrade``，用于校准；只有显式设置
``HUGINN_NORMATIVE_ENFORCE=1`` 时，才在高风险动作上真正阻断。

设计约束：
- 不调 LLM；只读规范账投影。
- 评估失败绝不阻断工具（fail-open for availability）；审计可见。
- 读投影按 (tenant, session) 做短 TTL 缓存，避免每次工具调用都开库查询。
"""
from __future__ import annotations

import logging
import time
from typing import Any

from huginn.core_types import RiskLevel
from huginn.normative.admission import AdmissionDecision, AdmissionRequest, admit
from huginn.normative.ledger import NormativeLedger, register_append_listener

logger = logging.getLogger(__name__)

_ledger: NormativeLedger | None = None
_view_cache: dict[tuple[str, str], tuple[float, Any]] = {}
_VIEW_TTL_S = 1.0

# 事件写入后立刻失效视图缓存，保证审批后的下一次调用能看到新授权。
register_append_listener(_view_cache.clear)


def _enabled() -> bool:
    from huginn.env_access import env_bool

    return env_bool("HUGINN_NORMATIVE_SHADOW")


def _enforcing() -> bool:
    from huginn.env_access import env_bool

    return env_bool("HUGINN_NORMATIVE_ENFORCE")


def _get_ledger() -> NormativeLedger:
    global _ledger
    if _ledger is None:
        _ledger = NormativeLedger()
    return _ledger


def _view(ledger: NormativeLedger, tenant_id: str, session_id: str):
    key = (tenant_id, session_id)
    now = time.monotonic()
    ttl = 0.0 if _enforcing() else _VIEW_TTL_S
    cached = _view_cache.get(key)
    if cached is not None and now - cached[0] <= ttl:
        return cached[1]
    view = ledger.project(tenant_id, session_id)
    _view_cache[key] = (now, view)
    return view


def evaluate_tool_call(
    *,
    tenant_id: str,
    session_id: str,
    tool_name: str,
    resources: tuple[str, ...] = (),
    risk: RiskLevel = RiskLevel.MEDIUM,
    requires_authorization: bool = True,
    claim_statuses: dict[str, str] | None = None,
    ledger: NormativeLedger | None = None,
    enabled: bool | None = None,
) -> AdmissionDecision | None:
    """评估一次工具调用的准入；shadow 关闭时返回 None。"""
    if enabled is None:
        enabled = _enabled()
    if not enabled:
        return None
    try:
        _l = ledger or _get_ledger()
        request = AdmissionRequest(
            tenant_id=tenant_id,
            session_id=session_id,
            action_name=tool_name,
            resources=resources,
            risk=risk,
            requires_authorization=requires_authorization,
        )
        decision = admit(
            request,
            _view(_l, tenant_id, session_id),
            claim_statuses=claim_statuses,
        )
        if not decision.allow:
            logger.debug(
                "normative shadow %s/%s tool=%s mode=%s required=%s",
                tenant_id,
                session_id,
                tool_name,
                decision.mode,
                decision.required_acts,
            )
        return decision
    except Exception:  # 防御: 评估失败不阻断工具
        logger.debug("normative shadow evaluation failed", exc_info=True)
        return None


__all__ = ["evaluate_tool_call"]
