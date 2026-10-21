"""行为指纹 — 跨会话比较同一 persona 的行为分布（spec §7.3，P3）.

替代「人格描述是否一致」这种不可测判据：用可数的行为分布（工具族、澄清率、
证据门通过率、风险分布、授权边界命中）计算 JS 散度，作为漂移监控信号。
"""
from __future__ import annotations

import math
import threading
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BehaviorFingerprint:
    persona_id: str
    window_start: str
    window_end: str
    tool_family_dist: dict[str, float] = field(default_factory=dict)
    clarification_rate: float = 0.0
    evidence_gate_pass_rate: float = 0.0
    risk_level_hist: dict[str, int] = field(default_factory=dict)
    authorization_boundary_hits: int = 0
    n_events: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "persona_id": self.persona_id,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "tool_family_dist": dict(self.tool_family_dist),
            "clarification_rate": self.clarification_rate,
            "evidence_gate_pass_rate": self.evidence_gate_pass_rate,
            "risk_level_hist": dict(self.risk_level_hist),
            "authorization_boundary_hits": self.authorization_boundary_hits,
            "n_events": self.n_events,
        }


def _tool_family(tool: str) -> str:
    text = str(tool or "").strip().lower()
    for prefix in ("vasp", "lammps", "qe", "cp2k", "gaussian", "orca", "abacus",
                   "comsol", "openfoam", "fenics", "elmer", "pybullet"):
        if text.startswith(prefix):
            return prefix
    return "other"


def fingerprint_from_events(
    events: list[dict[str, Any]],
    *,
    persona_id: str,
    window_start: str = "",
    window_end: str = "",
) -> BehaviorFingerprint:
    """从执行事件构建指纹；空事件返回全零指纹（不臆造基线）."""
    rows = [e for e in events if isinstance(e, dict)]
    n = len(rows)
    if n == 0:
        return BehaviorFingerprint(
            persona_id=persona_id,
            window_start=window_start,
            window_end=window_end,
        )
    families = Counter(_tool_family(str(e.get("tool") or "")) for e in rows)
    tool_dist = {name: count / n for name, count in families.items()}
    clarified = sum(1 for e in rows if e.get("clarified"))
    gate_passed = sum(1 for e in rows if e.get("evidence_gate_passed"))
    risk_hist = Counter(str(e.get("risk") or "low") for e in rows)
    boundary_hits = sum(1 for e in rows if e.get("boundary_hit"))
    return BehaviorFingerprint(
        persona_id=persona_id,
        window_start=window_start,
        window_end=window_end,
        tool_family_dist=tool_dist,
        clarification_rate=clarified / n,
        evidence_gate_pass_rate=gate_passed / n,
        risk_level_hist=dict(risk_hist),
        authorization_boundary_hits=boundary_hits,
        n_events=n,
    )


def _dist_vector(dist: dict[str, float], keys: list[str]) -> list[float]:
    values = [max(1e-9, float(dist.get(key, 0.0) or 0.0)) for key in keys]
    total = sum(values)
    return [value / total for value in values]


def js_divergence(
    p: dict[str, float],
    q: dict[str, float],
) -> float:
    """Jensen-Shannon divergence（base 2，∈[0,1]）；两分布都为空返回 0."""
    keys = sorted(set(p) | set(q))
    if not keys or (not p and not q):
        return 0.0
    pv = _dist_vector(p, keys)
    qv = _dist_vector(q, keys)
    mid = [(a + b) / 2 for a, b in zip(pv, qv, strict=True)]

    def _kl(a: list[float], m: list[float]) -> float:
        return sum(
            ai * math.log(ai / mi, 2)
            for ai, mi in zip(a, m, strict=True)
            if ai > 0 and mi > 0
        )

    return max(0.0, (_kl(pv, mid) + _kl(qv, mid)) / 2)


def behavior_drift(
    baseline: BehaviorFingerprint,
    current: BehaviorFingerprint,
) -> dict[str, Any]:
    """返回可观测的漂移分；阈值应走策略注册表，不在这里硬编码告警."""
    tool_js = js_divergence(
        baseline.tool_family_dist, current.tool_family_dist
    )
    risk_js = js_divergence(
        {k: float(v) for k, v in baseline.risk_level_hist.items()},
        {k: float(v) for k, v in current.risk_level_hist.items()},
    )
    return {
        "persona_id": current.persona_id,
        "tool_family_js": round(tool_js, 6),
        "risk_js": round(risk_js, 6),
        "clarification_delta": round(
            current.clarification_rate - baseline.clarification_rate, 6
        ),
        "evidence_gate_delta": round(
            current.evidence_gate_pass_rate - baseline.evidence_gate_pass_rate, 6
        ),
        "boundary_hits": current.authorization_boundary_hits,
        "n_events": current.n_events,
    }


class BehaviorMonitor:
    """按 (tenant, persona) 累积近期行为事件；基线漂移写成 drift_events.

    只做可观测，不改变任何授权/检索行为；阈值默认仅作起点。
    """

    def __init__(self, *, window: int = 200, threshold: float = 0.2) -> None:
        self.window = max(10, int(window))
        self.threshold = float(threshold)
        self._lock = threading.RLock()
        self._events: dict[tuple[str, str], deque[dict[str, Any]]] = {}
        self._baseline: dict[tuple[str, str], BehaviorFingerprint] = {}

    def record_event(
        self,
        *,
        tenant_id: str,
        persona_id: str,
        event: dict[str, Any],
    ) -> None:
        key = (str(tenant_id), str(persona_id))
        with self._lock:
            queue = self._events.setdefault(key, deque(maxlen=self.window))
            queue.append(dict(event))

    def current(
        self, *, tenant_id: str, persona_id: str
    ) -> BehaviorFingerprint:
        key = (str(tenant_id), str(persona_id))
        with self._lock:
            events = list(self._events.get(key, ()))
        return fingerprint_from_events(
            events, persona_id=str(persona_id)
        )

    def set_baseline(self, *, tenant_id: str, persona_id: str) -> BehaviorFingerprint:
        key = (str(tenant_id), str(persona_id))
        baseline = self.current(tenant_id=tenant_id, persona_id=persona_id)
        with self._lock:
            self._baseline[key] = baseline
        return baseline

    def drift(
        self, *, tenant_id: str, persona_id: str
    ) -> dict[str, Any] | None:
        key = (str(tenant_id), str(persona_id))
        with self._lock:
            baseline = self._baseline.get(key)
        if baseline is None or baseline.n_events == 0:
            return None
        current = self.current(tenant_id=tenant_id, persona_id=persona_id)
        if current.n_events == 0:
            return None
        report = behavior_drift(baseline, current)
        report["drifted"] = (
            float(report["tool_family_js"]) > self.threshold
            or float(report["risk_js"]) > self.threshold
        )
        return report

    def maybe_record_drift(
        self,
        *,
        tenant_id: str = "default",
        persona_id: str = "default",
        ledger: Any = None,
    ) -> str | None:
        """基线缺失时先建基线；漂移超阈值才写 drift_events（不阻断）."""
        key = (str(tenant_id), str(persona_id))
        with self._lock:
            has_baseline = key in self._baseline
        if not has_baseline:
            self.set_baseline(tenant_id=tenant_id, persona_id=persona_id)
            return None
        report = self.drift(tenant_id=tenant_id, persona_id=persona_id)
        if not report or not report.get("drifted"):
            return None
        from huginn.metacog.drift_ledger import DriftEvent, get_drift_ledger

        target = ledger or get_drift_ledger()
        observed = max(float(report["tool_family_js"]), float(report["risk_js"]))
        event = DriftEvent(
            tenant_id=str(tenant_id),
            session_id=str(persona_id),
            detector="behavior_fingerprint",
            baseline_id=f"behavior:{persona_id}",
            observed=observed,
            threshold=self.threshold,
            level="review",
            evidence=report,
        )
        return target.record(event)


_monitor: BehaviorMonitor | None = None


def get_behavior_monitor() -> BehaviorMonitor:
    global _monitor
    if _monitor is None:
        _monitor = BehaviorMonitor()
    return _monitor


__all__ = [
    "BehaviorFingerprint",
    "BehaviorMonitor",
    "behavior_drift",
    "fingerprint_from_events",
    "get_behavior_monitor",
    "js_divergence",
]
