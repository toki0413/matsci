"""共享观测状态收口 —— ObsVector / StateSnapshot / StateEstimator / ForwardPredictor.

对应 ``docs/research-notes/world-representation-inventory.md`` 标注的"真缺"清单，
是把散落在 ``physics_schema``(动作参数零填充) / ``ToolSpec.schema``(状态空间+观测)
里的世界表征**收拢成主循环可用的一条链**，而不是再造物理模型：

- ``ObsVector``          : 定长槽观测向量 + 缺失槽零填充 + ``contract_version`` 握手。
- ``StateSnapshot``      : "物理状态 dict"升级为带不确定性的状态视图(观察 σ / 不可估 None)。
- ``StateEstimator``     : 由 ``ToolSpec.schema.space.state/observables`` 生成状态分布快照，
                           含可辨识档位启发式(真实辨识上限见 ``validation.identifiability``)。
- ``ForwardPredictor``   : 逐轮前向投影入口：优先用已注册解析前向真值(或外部 predictor)，
                           产出 ``[WORLD PREDICTION]``；命中可回填 ``StateSnapshot.predicted``，
                           供后续奖励/记忆回流(本轮仅保留接口)。

原则：只依赖 schema 契约，不强绑定具体物理后端；不私自发明字段，全部对齐已有契约。
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from huginn.security import physics_schema

# 可辨识档位名(对齐 validation/identifiability 的三档命名)
_ADEOUATE = "adequate"
_LIMITED = "limited"
_DEFICIENT = "deficient"

# 已观测量的启发式不确定度(SI 相对量). 真实标定应来自 sensor_view/标定, 这里仅缺省.
_OBSERVED_SIGMA = 0.01


def _space_state(schema: Mapping[str, Any]) -> tuple[str, ...]:
    space = schema.get("space") or {}
    return tuple(space.get("state") or ())


def _observables(schema: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(schema.get("observables") or ())


@dataclass
class ObsVector:
    """定长槽观测向量: 缺失/未用槽零填充, 不删字段(与 microduck 契约同理)."""

    slots: tuple[str, ...] = field(default_factory=tuple)
    values: tuple[float, ...] = field(default_factory=tuple)
    contract_version: int = field(
        default_factory=lambda: physics_schema.SHARED_CONTRACT_VERSION
    )

    @classmethod
    def from_schema(
        cls, schema: Mapping[str, Any], *, version: int | None = None
    ) -> ObsVector:
        """由 schema 的状态空间申明生成全零向量(零填充骨架)."""
        states = _space_state(schema)
        ver = (
            version
            if version is not None
            else schema.get("contract_version")
            or physics_schema.SHARED_CONTRACT_VERSION
        )
        return cls(
            slots=states,
            values=tuple(0.0 for _ in states),
            contract_version=int(ver),
        )

    def pack(self, state: Mapping[str, Any]) -> ObsVector:
        """按槽序取数值; 缺失槽补 0, 未登记的键忽略(不 drop、不臆造)."""
        return ObsVector(
            slots=self.slots,
            values=tuple(float(state.get(s, 0.0)) for s in self.slots),
            contract_version=self.contract_version,
        )

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.slots, self.values))


@dataclass
class StateSnapshot:
    """带不确定性的状态视图: 观测项给 σ, 不可估项给 None, 可辨识档位标出哪些状态其实不可估."""

    state: dict[str, float]
    observables: tuple[str, ...] = ()
    uncertainty: dict[str, float | None] = field(default_factory=dict)
    identifiable_level: str = _DEFICIENT
    contract_version: int = physics_schema.SHARED_CONTRACT_VERSION
    # 前向投影结果(可选回填), 供奖励/记忆回流占位
    predicted: dict[str, float] | None = None
    ts: float = field(default_factory=time.time)

    def to_prompt(self) -> str:
        parts = [
            f"state={self.state}",
            f"identifiability={self.identifiable_level}",
        ]
        if self.uncertainty:
            parts.append(f"uncertainty={self.uncertainty}")
        if self.predicted:
            parts.append(f"prediction={self.predicted}")
        return "\n[STATE] " + "; ".join(parts)


class StateEstimator:
    """把"物理状态 dict"收拢成状态分布快照(对齐共享观测契约的零填充观)."""

    def __init__(self, schema: Mapping[str, Any]) -> None:
        self.schema = schema
        self.states = _space_state(schema) or _observables(schema)
        self.obs = _observables(schema)
        self.contract_version = int(
            schema.get("contract_version") or physics_schema.SHARED_CONTRACT_VERSION
        )

    def estimate(self, state: Mapping[str, Any]) -> StateSnapshot:
        obs_set = set(self.obs)
        uncertainty: dict[str, float | None] = {
            s: (_OBSERVED_SIGMA if s in obs_set else None) for s in self.states
        }
        covered = sum(1 for s in self.states if s in obs_set)
        if not self.states:
            level = _DEFICIENT
        elif covered == len(self.states):
            level = _ADEOUATE
        elif covered > 0:
            level = _LIMITED
        else:
            level = _DEFICIENT
        return StateSnapshot(
            state={s: float(state.get(s, 0.0)) for s in self.states},
            observables=tuple(self.obs),
            uncertainty=uncertainty,
            identifiable_level=level,
            contract_version=self.contract_version,
        )


class ForwardPredictor:
    """最小前向投影入口: 优先用解析前向真值/外部 predictor 对状态做一次前向预测."""

    def __init__(self, schema: Mapping[str, Any]) -> None:
        self.schema = schema
        self.contract_version = int(
            schema.get("contract_version") or physics_schema.SHARED_CONTRACT_VERSION
        )

    def predict(
        self,
        snapshot: StateSnapshot,
        *,
        predictor: Callable[[dict, Any | None], dict] | Any | None = None,
        action: Any | None = None,
    ) -> dict[str, Any] | None:
        """前向预测; predictor 缺省或失败时返回 None(不硬阻断, 由上层决定走 self-correct)."""
        if predictor is None or not hasattr(predictor, "predict"):
            return None
        try:
            pred = predictor.predict(snapshot.state)
        except TypeError:
            try:
                pred = predictor.predict(snapshot.state, action)
            except Exception:
                return None
        except Exception:
            return None
        return dict(pred) if isinstance(pred, Mapping) else None

    @staticmethod
    def to_prompt(prediction: Mapping[str, Any] | None) -> str:
        if not prediction:
            return ""
        return "\n[WORLD PREDICTION] 前向投影(仅供参考): " + json.dumps(
            dict(prediction), ensure_ascii=False
        )


def snapshot_from_schema(
    schema: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    estimator: StateEstimator | None = None,
) -> StateSnapshot:
    """一站入口: schema → estimator → 状态快照(供主循环低耦合调用)."""
    est = estimator or StateEstimator(schema)
    return est.estimate(state)


__all__ = [
    "ObsVector",
    "StateSnapshot",
    "StateEstimator",
    "ForwardPredictor",
    "snapshot_from_schema",
]
