"""记忆双层治理：保留闸（Retention）与影响闸（Influence）.

原则：**记住一件事，不等于授权它此刻塑造推理**。检索返回候选者；候选者是否
进入上下文、以多大权威进入，由本模块逐条判定。

本模块是纯函数（无 I/O、无 LLM）：
- `evaluate_retention`: 可用 / 归档 / 巩固 / 修剪；
- `evaluate_influence`: authorize / uncertain / low_impact / arbitrate /
  quarantine / consolidate / block；
- `assemble_context`: 按 influence 排序 + 预算装配，带来源标签。

权威上限（spec §18.6）：
  direct 1.0 / derived 0.6 / hypothesis 0.4；quarantine|block 0.0。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Route = Literal[
    "authorize",
    "uncertain",
    "low_impact",
    "arbitrate",
    "quarantine",
    "consolidate",
    "block",
]
RetentionAction = Literal["keep_available", "archive", "consolidate", "prune"]
AuthorityClass = Literal["direct", "derived", "hypothesis"]

_ROUTE_CAPS: dict[str, float] = {
    "authorize": 1.0,
    "uncertain": 0.4,
    "low_impact": 0.3,
    "arbitrate": 0.0,
    "quarantine": 0.0,
    "consolidate": 0.0,
    "block": 0.0,
}
_AUTHORITY_CAPS: dict[str, float] = {
    "direct": 1.0,
    "derived": 0.6,
    "hypothesis": 0.4,
}
_CONTEXT_ROUTES = frozenset({"authorize", "uncertain", "low_impact"})
_LABELS = {
    "direct": "[DIRECT]",
    "derived": "[DERIVED]",
    "hypothesis": "[DREAM]",
}


@dataclass(frozen=True)
class HygienePolicy:
    """可配置权重；默认值只作起点，生产应由策略注册表提供."""

    w_relevance: float = 0.30
    w_confidence: float = 0.25
    w_grounding: float = 0.15
    w_provenance: float = 0.15
    w_recency: float = 0.15
    w_uncertainty: float = 0.20
    w_contradiction: float = 0.35
    authorize_threshold: float = 0.55
    confidence_floor: float = 0.35
    contradiction_high: float = 0.80
    provenance_floor: float = 0.30
    prune_importance: float = 0.20


@dataclass(frozen=True)
class RetentionDecision:
    memory_id: str
    action: RetentionAction
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class InfluenceDecision:
    memory_id: str
    route: Route
    influence: float
    caps_applied: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    authority_class: str = "direct"

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "route": self.route,
            "influence": round(self.influence, 4),
            "caps_applied": list(self.caps_applied),
            "reasons": list(self.reasons),
            "authority_class": self.authority_class,
        }


@dataclass(frozen=True)
class ContextEntry:
    memory_id: str
    text: str
    label: str
    influence: float


@dataclass(frozen=True)
class ContextPack:
    entries: tuple[ContextEntry, ...] = ()
    no_evidence: bool = False
    total_influence: float = 0.0


def _f(memory: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(memory.get(key, default))
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def evaluate_retention(
    memory: dict[str, Any],
    *,
    policy: HygienePolicy | None = None,
    allow_prune: bool = False,
) -> RetentionDecision:
    """只判断「还值不值得留」，不判断「此刻能不能影响推理」."""
    policy = policy or HygienePolicy()
    memory_id = str(memory.get("memory_id") or memory.get("id") or "")
    if memory.get("duplicate_of"):
        return RetentionDecision(memory_id, "consolidate", ("近似重复，合并摘要",))
    if memory.get("revoked"):
        return RetentionDecision(memory_id, "archive", ("已撤销，留档但不参与召回",))
    if memory.get("expired"):
        # 硬修剪需要两个独立信号 + 显式允许；否则只归档。
        if allow_prune and memory.get("predicted_failure"):
            return RetentionDecision(memory_id, "prune", ("过期 + 预测失败，允许硬修剪",))
        return RetentionDecision(memory_id, "archive", ("过期，默认归档而非删除",))
    importance = _f(memory, "importance", 0.5)
    if memory.get("archived") or importance < policy.prune_importance:
        return RetentionDecision(memory_id, "archive", ("低重要性，归档留查",))
    return RetentionDecision(memory_id, "keep_available", ("仍可用",))


def evaluate_influence(
    memory: dict[str, Any],
    *,
    query: dict[str, Any] | None = None,
    policy: HygienePolicy | None = None,
) -> InfluenceDecision:
    """逐查询判定「是否、以多大权威进入当前推理上下文」."""
    policy = policy or HygienePolicy()
    query = query or {}
    memory_id = str(memory.get("memory_id") or memory.get("id") or "")
    authority = str(memory.get("authority_class") or "direct")
    if authority not in _AUTHORITY_CAPS:
        authority = "hypothesis"
    authority_cap = _AUTHORITY_CAPS[authority]

    # 硬约束：租户/撤销/过期/来源
    if (
        query.get("tenant_id")
        and memory.get("tenant_id")
        and str(query["tenant_id"]) != str(memory["tenant_id"])
    ):
        return InfluenceDecision(
            memory_id, "block", 0.0, ("tenant_isolation",), ("跨租户记忆禁止进入",), authority
        )
    if memory.get("revoked") or memory.get("expired"):
        return InfluenceDecision(
            memory_id, "block", 0.0, ("validity",), ("已撤销/过期",), authority
        )

    confidence = _f(memory, "confidence", _f(memory, "importance", 0.5))
    uncertainty = _f(memory, "uncertainty", 0.0)
    contradiction = _f(memory, "contradiction_pressure", 0.0)
    provenance = _f(memory, "provenance_completeness", 0.5)
    risk = str(memory.get("risk") or "low").lower()

    if contradiction >= policy.contradiction_high:
        return InfluenceDecision(
            memory_id, "arbitrate", 0.0, ("contradiction",),
            ("矛盾压力高，先仲裁而非静默选边",), authority
        )
    if provenance < policy.provenance_floor and risk in ("high", "critical"):
        return InfluenceDecision(
            memory_id, "quarantine", 0.0, ("provenance", "risk"),
            ("高风险 + 溯源不完整，隔离",), authority
        )

    relevance = _f(memory, "relevance", 0.5)
    grounding = _f(memory, "grounding_status", 0.5)
    recency = _f(memory, "recency_validity", 1.0)
    raw = (
        policy.w_relevance * relevance
        + policy.w_confidence * confidence
        + policy.w_grounding * grounding
        + policy.w_provenance * provenance
        + policy.w_recency * recency
        - policy.w_uncertainty * uncertainty
        - policy.w_contradiction * contradiction
    )
    raw = _clamp(raw)

    reasons: list[str] = []
    route: Route
    if confidence < policy.confidence_floor:
        route = "uncertain"
        reasons.append("置信度低于地板")
    elif raw >= policy.authorize_threshold:
        route = "authorize"
    else:
        route = "low_impact"
        reasons.append("相关但权威不足")

    route_cap = _ROUTE_CAPS[route]
    influence = min(raw, authority_cap, route_cap)
    caps: list[str] = []
    if influence < raw:
        caps.append(f"cap={min(authority_cap, route_cap):.2f}")
    return InfluenceDecision(
        memory_id=memory_id,
        route=route,
        influence=influence,
        caps_applied=tuple(caps),
        reasons=tuple(reasons),
        authority_class=authority,
    )


def assemble_context(
    memories: list[dict[str, Any]],
    *,
    query: dict[str, Any] | None = None,
    budget: float = 1.0,
    policy: HygienePolicy | None = None,
) -> ContextPack:
    """按影响权威装配上下文；全被 block/quarantine 时显式声明无证据."""
    decisions: list[tuple[dict[str, Any], InfluenceDecision]] = []
    for memory in memories:
        decision = evaluate_influence(memory, query=query, policy=policy)
        if decision.route in _CONTEXT_ROUTES:
            decisions.append((memory, decision))
    decisions.sort(key=lambda pair: pair[1].influence, reverse=True)

    entries: list[ContextEntry] = []
    total = 0.0
    for memory, decision in decisions:
        if total + decision.influence > budget and entries:
            break
        label = _LABELS.get(decision.authority_class, "[DIRECT]")
        if decision.route == "uncertain":
            label += "[UNCERTAIN]"
        elif decision.route == "low_impact":
            label += "[LOW]"
        entries.append(
            ContextEntry(
                memory_id=decision.memory_id,
                text=str(memory.get("text") or memory.get("content") or "")[:400],
                label=label,
                influence=decision.influence,
            )
        )
        total += decision.influence

    if not entries:
        return ContextPack(entries=(), no_evidence=True, total_influence=0.0)
    return ContextPack(
        entries=tuple(entries),
        no_evidence=False,
        total_influence=round(total, 4),
    )


__all__ = [
    "AuthorityClass",
    "ContextEntry",
    "ContextPack",
    "HygienePolicy",
    "InfluenceDecision",
    "RetentionAction",
    "RetentionDecision",
    "Route",
    "assemble_context",
    "evaluate_influence",
    "evaluate_retention",
]
