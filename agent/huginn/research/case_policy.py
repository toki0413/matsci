"""案例策略 — 成功/失败案例库 + 可回放的检索策略（spec §19.5，I31）.

明确边界：
- 只更新**案例统计**（每 action 的均值/计数），不改模型权重；
- 决策是案例统计的确定性函数，可回放、可审计；
- 策略版本随每次更新递增，便于灰度/回滚。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Case:
    case_id: str
    query: str
    action: str
    reward: float
    source: str = ""
    created_at: str = ""


@dataclass
class PolicySnapshot:
    version: int
    counts: dict[str, int]
    means: dict[str, float]
    decisions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "counts": dict(self.counts),
            "means": dict(self.means),
            "decisions": list(self.decisions),
            "model_weights": None,
        }


class CasePolicy:
    """案例均值策略：选择历史均值最高的动作；同分用 action 名稳定排序."""

    def __init__(self) -> None:
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._version = 0
        self._decisions: list[str] = []

    def update(self, case: Case) -> None:
        action = str(case.action)
        self._sums[action] = self._sums.get(action, 0.0) + float(case.reward)
        self._counts[action] = self._counts.get(action, 0) + 1
        self._version += 1

    def mean(self, action: str) -> float:
        count = self._counts.get(str(action), 0)
        return self._sums[str(action)] / count if count else 0.0

    def select(self, query: str = "") -> str | None:  # noqa: ARG002
        """确定性选择：最高均值，同分按字典序；无案例返回 None."""
        if not self._counts:
            return None
        action = max(
            self._counts, key=lambda name: (self.mean(name), name)
        )
        self._decisions.append(action)
        return action

    def snapshot(self) -> PolicySnapshot:
        return PolicySnapshot(
            version=self._version,
            counts=dict(self._counts),
            means={name: self.mean(name) for name in self._counts},
            decisions=list(self._decisions),
        )

    def export_events(self) -> list[dict[str, Any]]:
        return [
            {"action": name, "count": count, "sum": self._sums[name]}
            for name, count in sorted(self._counts.items())
        ]

    def replay(self, events: list[dict[str, Any]]) -> None:
        """从事件重建统计；不触碰任何模型权重."""
        self._sums = {}
        self._counts = {}
        self._version = 0
        for event in events:
            action = str(event["action"])
            self._sums[action] = float(event.get("sum", 0.0))
            self._counts[action] = int(event.get("count", 0))
            self._version += int(event.get("count", 0))
        self._decisions = []


def load_events(text: str) -> list[dict[str, Any]]:
    return json.loads(text) if text else []


__all__ = ["Case", "CasePolicy", "PolicySnapshot", "load_events"]
