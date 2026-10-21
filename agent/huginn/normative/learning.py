"""学习回路 — 轨迹索引 + ActClassifier + AdmissionPolicy + MemoryOpPolicy + OPE（§11）.

硬约束：
- 热路径不调 LLM；分类器只做**建议**，不用于放行；
- 授权/撤销/跨租户是 **action mask**，不进 reward —— 未授权动作在结构上不可选；
- 策略只做离线训练 / OPE / 影子模式，禁止直接上线；
- 所有统计可回放、可审计；不改模型权重。
"""
from __future__ import annotations

import json
import random
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from huginn.core_types import RiskLevel

DecisionArm = Literal["ask", "auto_allow", "degrade", "deny"]
MemoryArm = Literal["store", "retrieve", "summarize", "discard"]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class AdmissionQuery:
    intent: str
    goal: str = ""
    persona_params: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class AdmissionSample:
    query: AdmissionQuery
    context_features: dict[str, Any]
    decision: dict[str, Any]
    outcome: dict[str, Any]
    tenant_id: str
    ts: str = field(default_factory=_now_iso)
    behaviour_prob: float = 1.0
    sample_id: str = ""

    def reward(self) -> float:
        """未授权执行是硬失败（reward 0），不是低奖励；其余用 outcome.reward."""
        if self.outcome.get("unauthorized"):
            return 0.0
        return float(self.outcome.get("reward", 0.0))

    def action(self) -> str:
        mode = str(self.decision.get("mode") or "")
        return {"allow": "auto_allow"}.get(mode, mode)


class TrajectoryIndex:
    """准入轨迹索引；按 tenant 分区，追加式，供离线 OPE / 审计."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            from huginn.utils.runtime import get_runtime_home

            db_path = get_runtime_home() / "admission_trajectories.db"
        self.db_path = str(db_path)
        self._lock = threading.RLock()
        self._memory_conn: sqlite3.Connection | None = None
        if self.db_path == ":memory:":
            self._memory_conn = sqlite3.connect(":memory:")
            self._memory_conn.row_factory = sqlite3.Row
        else:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            conn = self._connect()
            conn.execute(
                "CREATE TABLE IF NOT EXISTS admission_samples ("
                "sample_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, "
                "payload_json TEXT NOT NULL, created_at TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_adm_tenant "
                "ON admission_samples(tenant_id, created_at)"
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        if self._memory_conn is not None:
            return self._memory_conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def record(self, sample: AdmissionSample) -> str:
        sample_id = sample.sample_id or f"adm:{sample.tenant_id}:{_now_iso()}"
        payload = {
            "query": {
                "intent": sample.query.intent,
                "goal": sample.query.goal,
                "persona_params": sample.query.persona_params,
            },
            "context_features": sample.context_features,
            "decision": sample.decision,
            "outcome": sample.outcome,
            "behaviour_prob": sample.behaviour_prob,
            "ts": sample.ts,
        }
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO admission_samples "
                "(sample_id, tenant_id, payload_json, created_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    sample_id,
                    sample.tenant_id,
                    json.dumps(payload, ensure_ascii=False, default=str),
                    sample.ts,
                ),
            )
            conn.commit()
        return sample_id

    def list_samples(self, tenant_id: str, limit: int = 1000) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM admission_samples WHERE tenant_id = ? "
                "ORDER BY created_at ASC LIMIT ?",
                (tenant_id, int(limit)),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                item["payload"] = json.loads(item.pop("payload_json") or "{}")
            except (json.JSONDecodeError, TypeError):
                item["payload"] = {}
            out.append(item)
        return out

    def count(self, tenant_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM admission_samples WHERE tenant_id = ?",
                (tenant_id,),
            ).fetchone()
        return int(row["n"]) if row else 0


# ── ActClassifier（规则基线 + 监督校准；只做建议） ─────────────────────

_ACT_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("do not authorize", "don't authorize", "revoke", "withdraw permission"), "revoke"),
    (("authorize", "grant permission", "approve", "permission to"), "authorize"),
    (("do not accept", "don't accept", "reject", "refuse"), "reject"),
    (("accept", "agree", "approve the proposal"), "accept"),
    (("acknowledge", "noted", "consider", "mention"), "acknowledge"),
    (("withdraw", "retract my proposal"), "withdraw"),
    (("i will", "commit", "promise to"), "commit"),
    (("propose", "suggest", "recommend", "plan to"), "propose"),
)

_ACT_TYPES = (
    "propose", "acknowledge", "accept", "reject",
    "authorize", "commit", "revoke", "withdraw",
)


class ActClassifier:
    """utterance → act_type + scope；**只产生建议，不能用于放行**."""

    def __init__(self) -> None:
        self._accuracy: dict[str, float] = {}

    def classify(self, utterance: str) -> dict[str, Any]:
        text = str(utterance or "").strip().lower()
        act: str = "propose"
        matched = ""
        for keywords, candidate in _ACT_RULES:
            if any(keyword in text for keyword in keywords):
                act, matched = candidate, keywords[0]
                break
        base = 0.9 if matched else 0.3
        confidence = round(base * self._accuracy.get(act, 1.0), 4)
        return {
            "act_type": act,
            "scope": {"actions": [], "resources": []},
            "confidence": confidence,
            "advisory_only": True,
        }

    def fit(self, labelled: list[tuple[str, str]]) -> dict[str, float]:
        """按类准确率校准置信度；空样本不改变校准."""
        hits: dict[str, int] = {}
        totals: dict[str, int] = {}
        for utterance, expected in labelled:
            predicted = self.classify(utterance)["act_type"]
            totals[expected] = totals.get(expected, 0) + 1
            hits[expected] = hits.get(expected, 0) + int(predicted == expected)
        self._accuracy = {
            act: hits.get(act, 0) / total
            for act, total in totals.items()
            if total
        }
        return dict(self._accuracy)

    def sensitivity_specificity(
        self, labelled: list[tuple[str, str]], *, positive: str = "authorize"
    ) -> dict[str, float]:
        tp = fp = fn = tn = 0
        for utterance, expected in labelled:
            predicted = self.classify(utterance)["act_type"]
            if expected == positive and predicted == positive:
                tp += 1
            elif expected != positive and predicted == positive:
                fp += 1
            elif expected == positive and predicted != positive:
                fn += 1
            else:
                tn += 1
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        return {
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }


# ── AdmissionPolicy（上下文 bandit + 硬 action mask） ────────────────

_ADM_ARMS: tuple[str, ...] = ("ask", "auto_allow", "degrade", "deny")


class AdmissionPolicy:
    """风险 × 上下文 → ask/auto_allow/degrade；mask 由规范状态决定."""

    def __init__(self, *, epsilon: float = 0.0, seed: int = 7) -> None:
        self.epsilon = max(0.0, min(1.0, float(epsilon)))
        self._rng = random.Random(int(seed))
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._regret = 0.0

    def allowed_arms(
        self,
        *,
        authorized: bool,
        revoked: bool = False,
        cross_tenant: bool = False,
    ) -> tuple[str, ...]:
        """硬 mask：未授权/已撤销/跨租户时 auto_allow 结构上不可选."""
        if cross_tenant:
            return ("deny",)
        allowed: list[str] = []
        if not revoked:
            allowed.append("ask")
            allowed.append("degrade")
        if authorized and not revoked:
            allowed.insert(0, "auto_allow")
        if revoked:
            allowed = ["ask", "degrade"]
        return tuple(allowed)

    def select(
        self,
        *,
        authorized: bool,
        revoked: bool = False,
        cross_tenant: bool = False,
        risk: RiskLevel = RiskLevel.LOW,
    ) -> str:
        allowed = self.allowed_arms(
            authorized=authorized, revoked=revoked, cross_tenant=cross_tenant
        )
        if not allowed:
            return "deny"
        if risk in (RiskLevel.HIGH, RiskLevel.CRITICAL) and "auto_allow" in allowed:
            allowed = tuple(arm for arm in allowed if arm != "auto_allow")
            if not allowed:
                return "ask"
        if self.epsilon > 0 and self._rng.random() < self.epsilon:
            return self._rng.choice(list(allowed))
        return max(allowed, key=lambda arm: (self.mean(arm), -_ADM_ARMS.index(arm)))

    def mean(self, arm: str) -> float:
        count = self._counts.get(str(arm), 0)
        return self._sums[str(arm)] / count if count else 0.5

    def update(self, arm: str, reward: float, *, baseline_reward: float = 0.0) -> None:
        arm = str(arm)
        if arm not in _ADM_ARMS:
            return
        self._sums[arm] = self._sums.get(arm, 0.0) + float(reward)
        self._counts[arm] = self._counts.get(arm, 0) + 1
        self._regret += max(0.0, float(baseline_reward) - float(reward))

    def snapshot(self) -> dict[str, Any]:
        return {
            "means": {arm: self.mean(arm) for arm in _ADM_ARMS},
            "counts": dict(self._counts),
            "regret": round(self._regret, 6),
        }


# ── MemoryOpPolicy（store/retrieve/summarize/discard） ─────────────────


class MemoryOpPolicy:
    """何时 store / retrieve / summarize / discard；discard 对 direct 证据不可选."""

    _ARMS: tuple[str, ...] = ("store", "retrieve", "summarize", "discard")

    def __init__(self, *, epsilon: float = 0.0, seed: int = 11) -> None:
        self.epsilon = max(0.0, min(1.0, float(epsilon)))
        self._rng = random.Random(int(seed))
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def allowed_arms(self, *, is_direct: bool) -> tuple[str, ...]:
        return tuple(arm for arm in self._ARMS if not (is_direct and arm == "discard"))

    def select(self, *, is_direct: bool = False) -> str:
        allowed = self.allowed_arms(is_direct=is_direct)
        if self.epsilon > 0 and self._rng.random() < self.epsilon:
            return self._rng.choice(list(allowed))
        return max(
            allowed,
            key=lambda arm: (
                (self._sums.get(arm, 0.0) / self._counts[arm])
                if self._counts.get(arm) else 0.5,
                -self._ARMS.index(arm),
            ),
        )

    def update(self, arm: str, reward: float) -> None:
        arm = str(arm)
        if arm not in self._ARMS:
            return
        self._sums[arm] = self._sums.get(arm, 0.0) + float(reward)
        self._counts[arm] = self._counts.get(arm, 0) + 1


# ── Off-policy evaluation ─────────────────────────────────────────────


def _policy_prob(
    sample: AdmissionSample,
    target_policy: Callable[[AdmissionSample], str],
) -> float:
    chosen = target_policy(sample)
    return 1.0 if chosen == sample.action() else 0.0


def ope_report(
    samples: list[AdmissionSample],
    target_policy: Callable[[AdmissionSample], str],
    *,
    policy_id: str = "candidate",
) -> dict[str, Any]:
    """IPS / SNIPS / DR + 影子模式安全指标；不上线，只出报告."""
    usable: list[tuple[AdmissionSample, float, float]] = []
    per_arm: dict[str, list[float]] = {}
    unauthorized = false_pause = 0
    for sample in samples:
        action = sample.action()
        if action not in _ADM_ARMS:
            continue
        reward = sample.reward()
        mu = max(1e-6, float(sample.behaviour_prob))
        pi = _policy_prob(sample, target_policy)
        usable.append((sample, pi / mu, reward))
        per_arm.setdefault(action, []).append(reward)
        unauthorized += int(bool(sample.outcome.get("unauthorized")))
        false_pause += int(bool(sample.outcome.get("false_pause")))

    if not usable:
        return {
            "policy_id": policy_id,
            "n": 0,
            "ips": 0.0,
            "snips": 0.0,
            "dr": 0.0,
            "unauthorized_rate": 0.0,
            "false_pause_rate": 0.0,
            "note": "no usable samples",
        }

    ips = sum(weight * reward for _, weight, reward in usable) / len(usable)
    total_weight = sum(weight for _, weight, _ in usable)
    snips = (
        sum(weight * reward for _, weight, reward in usable) / total_weight
        if total_weight > 0
        else 0.0
    )

    # DR：用每臂经验均值做 reward model；目标动作预测值 + IPS 残差修正
    arm_mean = {
        arm: sum(values) / len(values) for arm, values in per_arm.items() if values
    }
    dr_terms: list[float] = []
    for sample, weight, reward in usable:
        action = sample.action()
        predicted = arm_mean.get(action, 0.5)
        dr_terms.append(predicted + weight * (reward - predicted))
    dr = sum(dr_terms) / len(dr_terms) if dr_terms else 0.0

    n = len(usable)
    return {
        "policy_id": policy_id,
        "n": n,
        "ips": round(ips, 6),
        "snips": round(snips, 6),
        "dr": round(dr, 6),
        "unauthorized_rate": round(unauthorized / n, 6),
        "false_pause_rate": round(false_pause / n, 6),
        "behaviour_value": round(
            sum(reward for _, _, reward in usable) / n, 6
        ),
    }


def shadow_evaluate(
    samples: list[AdmissionSample],
    policy: AdmissionPolicy,
) -> dict[str, Any]:
    """影子模式：按当前规范状态调用 policy，统计未授权执行（必须为 0）."""
    decisions: list[str] = []
    unauthorized = 0
    for sample in samples:
        features = sample.context_features
        arm = policy.select(
            authorized=bool(features.get("authorized")),
            revoked=bool(features.get("revoked")),
            cross_tenant=bool(features.get("tenant_id", sample.tenant_id) != sample.tenant_id),
            risk=RiskLevel(str(features.get("risk", "low"))),
        )
        decisions.append(arm)
        if arm == "auto_allow" and not features.get("authorized"):
            unauthorized += 1
    return {
        "n": len(samples),
        "unauthorized_executions": unauthorized,
        "decisions": decisions,
        "policy": policy.snapshot(),
    }


__all__ = [
    "ActClassifier",
    "AdmissionPolicy",
    "AdmissionQuery",
    "AdmissionSample",
    "DecisionArm",
    "MemoryArm",
    "MemoryOpPolicy",
    "TrajectoryIndex",
    "ope_report",
    "shadow_evaluate",
]
