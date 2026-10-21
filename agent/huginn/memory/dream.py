"""梦境周期核心（纯函数）— SleepClock + Dreamer / Analyst / Consolidator.

边界（spec §18.4 / §18.6）：
- 睡眠状态只门控**后台巩固**，不门控用户请求；
- Dreamer 只生成 finding（高召回、可错）；
- Analyst 只归因并给 confidence，可弃权，不执行；
- Consolidator 只产出可逆修复动作，不删除 direct 证据；
- 梦产物默认 derived/hypothesis，不能授权动作，不能改 PersonaKernel。

本模块不做 I/O、不调 LLM；执行/落库由调用方接。
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Literal
from zoneinfo import ZoneInfo

SleepState = Literal["active", "sleep"]
ProbeType = Literal["structure", "demand", "surprise"]
CauseClass = Literal[
    "missing_link",
    "duplicate",
    "stale",
    "overgeneralized",
    "contradiction",
    "coverage_gap",
    "provenance_gap",
    "authority_mismatch",
    "goal_drift",
]
RepairName = Literal[
    "add_link",
    "split_cluster",
    "merge_duplicate",
    "re_anchor",
    "soft_decay",
    "quarantine",
    "summarize",
]
Risk = Literal["low", "medium", "high"]
DreamStatus = Literal["candidate", "promoted", "quarantined", "expired"]

_WEEKDAYS = {
    "mon": 0, "monday": 0,
    "tue": 1, "tuesday": 1,
    "wed": 2, "wednesday": 2,
    "thu": 3, "thursday": 3,
    "fri": 4, "friday": 4,
    "sat": 5, "saturday": 5,
    "sun": 6, "sunday": 6,
}


@dataclass(frozen=True)
class SleepClock:
    persona_id: str
    timezone: str = "UTC"
    # (weekday, start HH:MM, end HH:MM)；支持跨午夜
    weekly_windows: tuple[tuple[str, str, str], ...] = ()
    state: SleepState = "active"
    last_transition_at: str = ""


def _parse_now(now: str | datetime | None) -> datetime:
    if now is None:
        return datetime.now(UTC)
    if isinstance(now, datetime):
        return now if now.tzinfo else now.replace(tzinfo=UTC)
    text = str(now).strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _weekday_index(value: Any) -> int | None:
    if isinstance(value, int):
        return value if 0 <= value <= 6 else None
    key = str(value).strip().lower()
    if key.isdigit():
        idx = int(key)
        return idx if 0 <= idx <= 6 else None
    return _WEEKDAYS.get(key)


def _minutes(hhmm: str) -> int | None:
    try:
        hh, mm = str(hhmm).strip().split(":")[:2]
        return int(hh) * 60 + int(mm)
    except (TypeError, ValueError):
        return None


def _in_window(now_local: datetime, windows: tuple[tuple[str, str, str], ...]) -> bool:
    if not windows:
        return False
    current_min = now_local.hour * 60 + now_local.minute
    current_day = now_local.weekday()
    for window in windows:
        if len(window) != 3:
            continue
        day = _weekday_index(window[0])
        start = _minutes(window[1])
        end = _minutes(window[2])
        if day is None or start is None or end is None:
            continue
        if start <= end:
            if current_day == day and start <= current_min < end:
                return True
        else:  # 跨午夜：当天 start→24:00，次日 00:00→end
            if current_day == day and current_min >= start:
                return True
            if current_day == (day + 1) % 7 and current_min < end:
                return True
    return False


def tick(clock: SleepClock, now: str | datetime | None = None) -> SleepClock:
    """按人格真实作息推进 active/sleep；无窗口时保持 active。"""
    dt = _parse_now(now)
    try:
        local = dt.astimezone(ZoneInfo(clock.timezone))
    except Exception:
        # 时区数据不可用时按 UTC 判定，避免静默变成永久 sleep
        local = dt.astimezone(UTC)
    if not clock.weekly_windows:
        return clock
    new_state: SleepState = (
        "sleep" if _in_window(local, clock.weekly_windows) else "active"
    )
    if new_state == clock.state:
        return clock
    return replace(
        clock,
        state=new_state,
        last_transition_at=local.isoformat(),
    )


@dataclass(frozen=True)
class DreamFinding:
    finding_id: str
    probe_type: ProbeType
    subject: str
    evidence_ids: tuple[str, ...] = ()
    surprise: float = 0.0
    expected_impact: float = 0.0
    cost: float = 0.0


@dataclass(frozen=True)
class Diagnosis:
    diagnosis_id: str
    finding_id: str
    cause_class: CauseClass
    confidence: float
    affected_ids: tuple[str, ...]
    repair_class: RepairName
    expected_gain: float
    risk: Risk = "medium"
    counterfactual_passed: bool = False
    abstain: bool = False


@dataclass(frozen=True)
class RepairAction:
    action: RepairName
    target_ids: tuple[str, ...]
    risk: Risk
    reversible: bool = True
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DreamObject:
    dream_id: str
    persona_id: str
    sleep_window: tuple[str, str]
    seed_event_ids: tuple[str, ...]
    # §13: 跨设备同步只同步投影对象；租户/部署用于隔离与溯源.
    tenant_id: str = "default"
    deployment_id: str = "local"
    highlights: tuple[str, ...] = ()
    decisions: tuple[str, ...] = ()
    risks: tuple[str, ...] = ()
    followups: tuple[str, ...] = ()
    affect: dict[str, float] = field(default_factory=dict)
    abstraction_authority: float = 0.7
    evidential_authority: float = 0.4
    influence_authority: float = 0.6
    provenance: tuple[str, ...] = ()
    soul_priority: str = "agent"
    status: DreamStatus = "candidate"


@dataclass(frozen=True)
class DreamCycleResult:
    status: Literal["skipped_active", "completed"]
    findings: tuple[DreamFinding, ...] = ()
    diagnoses: tuple[Diagnosis, ...] = ()
    repairs: tuple[RepairAction, ...] = ()
    dream_object: DreamObject | None = None
    skipped_reason: str = ""


def probe_memory_graph(
    memories: list[dict[str, Any]],
    *,
    probes: list[dict[str, Any]] | None = None,
) -> list[DreamFinding]:
    """Dreamer：从结构/需求/惊奇信号生成 findings（高召回，可错）."""
    findings: dict[tuple[str, str], DreamFinding] = {}

    def _add(finding: DreamFinding) -> None:
        findings.setdefault((finding.probe_type, finding.subject), finding)

    for memory in memories:
        mid = str(memory.get("memory_id") or memory.get("id") or "")
        if not mid:
            continue
        degree = int(memory.get("degree", memory.get("links", 0)) or 0)
        if degree == 0 and not memory.get("duplicate_of"):
            _add(DreamFinding(
                finding_id=f"isolated:{mid}", probe_type="structure",
                subject=mid, evidence_ids=(mid,), surprise=0.4,
                expected_impact=0.5, cost=0.1,
            ))
        if memory.get("stale") or memory.get("age_days", 0) > 90:
            _add(DreamFinding(
                finding_id=f"stale:{mid}", probe_type="structure",
                subject=mid, evidence_ids=(mid,), surprise=0.3,
                expected_impact=0.4, cost=0.1,
            ))
        if memory.get("duplicate_of"):
            _add(DreamFinding(
                finding_id=f"duplicate:{mid}", probe_type="structure",
                subject=mid, evidence_ids=(mid, str(memory["duplicate_of"])),
                surprise=0.2, expected_impact=0.4, cost=0.1,
            ))
        if float(memory.get("contradiction_pressure", 0.0) or 0.0) >= 0.8:
            _add(DreamFinding(
                finding_id=f"contradiction:{mid}", probe_type="structure",
                subject=mid, evidence_ids=(mid,), surprise=0.9,
                expected_impact=0.8, cost=0.2,
            ))

    for probe in probes or []:
        pid = str(probe.get("probe_id") or probe.get("query") or "")
        if not pid or not probe.get("failed"):
            continue
        _add(DreamFinding(
            finding_id=f"coverage:{pid}", probe_type="demand",
            subject=pid, evidence_ids=tuple(probe.get("evidence_ids") or ()),
            surprise=float(probe.get("surprise", 0.6) or 0.6),
            expected_impact=0.7, cost=0.1,
        ))

    return list(findings.values())


def diagnose(
    finding: DreamFinding,
    *,
    counterfactual: Callable[[DreamFinding, str], bool] | None = None,
) -> Diagnosis:
    """Analyst：归因 + confidence；证据不足则弃权，不猜."""
    subject = finding.subject
    if finding.probe_type == "demand":
        cause: CauseClass = "coverage_gap"
        repair: RepairName = "summarize"
    elif finding.finding_id.startswith("isolated"):
        cause, repair = "missing_link", "add_link"
    elif finding.finding_id.startswith("stale"):
        cause, repair = "stale", "re_anchor"
    elif finding.finding_id.startswith("duplicate"):
        cause, repair = "duplicate", "merge_duplicate"
    elif finding.finding_id.startswith("contradiction"):
        cause, repair = "contradiction", "quarantine"
    else:
        cause, repair = "coverage_gap", "summarize"

    if cause == "duplicate" and len(finding.evidence_ids) >= 2:
        affected_ids = tuple(finding.evidence_ids[:2])  # (duplicate, original)
    else:
        affected_ids = (subject,)

    confidence = 0.5 + 0.3 * min(1.0, finding.surprise)
    passed = False
    if counterfactual is not None:
        try:
            passed = bool(counterfactual(finding, repair))
        except Exception:
            passed = False
        confidence = 0.85 if passed else 0.35

    abstain = confidence < 0.5
    risk: Risk = "high" if cause == "contradiction" else "medium"
    return Diagnosis(
        diagnosis_id=f"dx:{finding.finding_id}",
        finding_id=finding.finding_id,
        cause_class=cause,
        confidence=round(confidence, 3),
        affected_ids=affected_ids,
        repair_class=repair,
        expected_gain=finding.expected_impact,
        risk=risk,
        counterfactual_passed=passed,
        abstain=abstain,
    )


def plan_repairs(
    diagnoses: list[Diagnosis],
    *,
    max_risk: Risk = "medium",
) -> list[RepairAction]:
    """Consolidator：只产出可逆修复；不删除 direct 证据."""
    allowed = {"low": 0, "medium": 1, "high": 2}
    repairs: list[RepairAction] = []
    for dx in diagnoses:
        if dx.abstain:
            continue
        if allowed.get(dx.risk, 2) > allowed.get(max_risk, 1):
            continue
        repairs.append(
            RepairAction(
                action=dx.repair_class,
                target_ids=dx.affected_ids,
                risk=dx.risk,
                reversible=True,
                evidence={
                    "diagnosis_id": dx.diagnosis_id,
                    "cause_class": dx.cause_class,
                    "confidence": dx.confidence,
                },
            )
        )
    return repairs


def synthesize_dream_object(
    *,
    dream_id: str,
    persona_id: str,
    sleep_window: tuple[str, str],
    findings: list[DreamFinding],
    repairs: list[RepairAction],
    tenant_id: str = "default",
    deployment_id: str = "local",
) -> DreamObject:
    """把 findings/repairs 合成梦对象；默认 derived 权威，不授权动作."""
    ordered = sorted(findings, key=lambda f: f.surprise, reverse=True)
    return DreamObject(
        dream_id=dream_id,
        persona_id=persona_id,
        sleep_window=sleep_window,
        tenant_id=str(tenant_id or "default"),
        deployment_id=str(deployment_id or "local"),
        seed_event_ids=tuple(f.finding_id for f in ordered[:10]),
        highlights=tuple(f.subject for f in ordered[:3]),
        decisions=tuple(f"{r.action}:{','.join(r.target_ids)}" for r in repairs[:5]),
        risks=tuple(
            f.subject for f in findings if f.finding_id.startswith("contradiction")
        ),
        followups=tuple(f.subject for f in findings if f.probe_type == "demand")[:3],
        provenance=tuple(f.finding_id for f in ordered),
        status="candidate",
    )


def run_dream_cycle(
    *,
    clock: SleepClock,
    dream_id: str,
    memories: list[dict[str, Any]],
    probes: list[dict[str, Any]] | None = None,
    now: str | datetime | None = None,
    counterfactual: Callable[[DreamFinding, str], bool] | None = None,
    tenant_id: str = "default",
    deployment_id: str = "local",
) -> DreamCycleResult:
    """睡眠门控 + Dreamer→Analyst→Consolidator 的纯核心."""
    current = tick(clock, now)
    if current.state != "sleep":
        return DreamCycleResult(
            status="skipped_active",
            skipped_reason="consolidation only runs during sleep",
        )
    findings = probe_memory_graph(memories, probes=probes)
    diagnoses = [diagnose(f, counterfactual=counterfactual) for f in findings]
    repairs = plan_repairs(diagnoses)
    obj = synthesize_dream_object(
        dream_id=dream_id,
        persona_id=clock.persona_id,
        sleep_window=(current.last_transition_at, str(now or "")),
        findings=findings,
        repairs=repairs,
        tenant_id=tenant_id,
        deployment_id=deployment_id,
    )
    return DreamCycleResult(
        status="completed",
        findings=tuple(findings),
        diagnoses=tuple(diagnoses),
        repairs=tuple(repairs),
        dream_object=obj,
    )


def promote_dream_object(
    obj: DreamObject,
    *,
    corroboration: int = 0,
    provenance_complete: bool = False,
    has_unresolved_contradiction: bool = False,
) -> DreamObject:
    """写入侧晋升门：hypothesis/derived → derived/promoted；矛盾则隔离."""
    if obj.status != "candidate":
        return obj
    if has_unresolved_contradiction:
        return replace(
            obj,
            status="quarantined",
            evidential_authority=0.1,
            influence_authority=0.0,
        )
    if provenance_complete and corroboration >= 1:
        return replace(
            obj,
            status="promoted",
            evidential_authority=0.6,
            influence_authority=0.6,
        )
    return obj


def apply_soft_decay(score: float, factor: float) -> tuple[float, dict[str, float]]:
    """可逆软衰减：返回衰减后分数 + 恢复句柄."""
    factor = max(0.0, min(1.0, float(factor)))
    before = float(score)
    return before * factor, {"before": before, "factor": factor}


def restore_soft_decay(decayed: float, handle: dict[str, float]) -> float:
    """按软衰减句柄恢复原分数（不依赖 decayed 值）."""
    _ = decayed
    return float(handle.get("before", 0.0))


__all__ = [
    "CauseClass",
    "Diagnosis",
    "DreamCycleResult",
    "DreamFinding",
    "DreamObject",
    "DreamStatus",
    "ProbeType",
    "RepairAction",
    "RepairName",
    "Risk",
    "SleepClock",
    "SleepState",
    "apply_soft_decay",
    "diagnose",
    "plan_repairs",
    "probe_memory_graph",
    "promote_dream_object",
    "restore_soft_decay",
    "run_dream_cycle",
    "synthesize_dream_object",
    "tick",
]
