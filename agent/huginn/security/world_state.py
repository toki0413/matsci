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
import math
import time
from collections.abc import Callable, Mapping, Sequence
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


class WorldStateTracker:
    """逐轮世界状态跟踪器 — 把 StateEstimator + ForwardPredictor 接进物理主循环.

    每执行一个动作 (前状态 → 后状态) 收口一次世界表征闭环:

    1. ``StateEstimator`` 由 ``schema`` 生成当前状态快照 (含可辨识档位/不确定度);
    2. ``ForwardPredictor`` 复用解析前向真值 (``world_model.predict``) 对"后状态"
       做前向投影, 命中回填 ``snapshot.predicted``;
    3. ``observe(state_after)`` 把"预测的后状态" vs "实测后状态" 的 RMS 差当成
       **预测误差** —— 这是回流到奖励/记忆的过程信号 (命中 → 误差小 → 正信号).

    世界表征是 advisory: 任何一步失败都吞掉, 不阻塞实验执行.
    """

    def __init__(self, schema: Mapping[str, Any], world_model: Any) -> None:
        self.schema = schema
        self.estimator = StateEstimator(schema)
        self.predictor = ForwardPredictor(schema)
        self.world_model = world_model
        self.last_snapshot: StateSnapshot | None = None
        self.last_prediction: dict[str, float] | None = None
        self.last_prediction_error: float | None = None
        self.last_prediction_reward: float | None = None
        # 运行期累计预测命中奖励 (每次 observe 追加), 供 runner 求均作 r_phys 贡献.
        self._rewards: list[float] = []

    def step(
        self, state_before: Mapping[str, Any], action: Any = None
    ) -> StateSnapshot:
        """前向一步: 对状态做估计, 再用解析真值把后状态投影出来 (回填 predicted)."""
        snapshot = self.estimator.estimate(state_before)
        pred = self.predictor.predict(
            snapshot, predictor=self.world_model, action=action
        )
        if isinstance(pred, Mapping):
            snapshot.predicted = {str(k): float(v) for k, v in pred.items()}
        self.last_snapshot = snapshot
        self.last_prediction = snapshot.predicted
        return snapshot

    def observe(self, state_after: Mapping[str, Any]) -> float | None:
        """执行后观测: 算"预测 vs 实测"的 RMS 误差 (回流信号).

        无预测 / 二者无共享维度时返回 None (不误报 0 错).
        """
        pred = self.last_prediction
        if not pred:
            return None
        # 数值量级可能悬殊 (p ~1e5, n ~1), 用相对误差归一, 避免高压压倒低压.
        errs: list[float] = []
        for k, v in pred.items():
            if k in state_after:
                actual = float(state_after[k])
                scale = max(abs(float(v)), abs(actual), 1e-12)
                errs.append(((float(v) - actual) / scale) ** 2)
        if not errs:
            return None
        self.last_prediction_error = math.sqrt(sum(errs) / len(errs))
        self.last_prediction_reward = prediction_error_to_reward(
            self.last_prediction_error
        )
        if self.last_prediction_reward is not None:
            self._rewards.append(self.last_prediction_reward)
        return self.last_prediction_error

    def avg_reward(self) -> float | None:
        """运行期平均预测命中奖励: runner 求整次协议的 r_phys 贡献. 无样本返回 None."""
        if not self._rewards:
            return None
        return sum(self._rewards) / len(self._rewards)


def prediction_error_to_reward(
    error: float | None,
    *,
    half_life: float = 1.0,
) -> float | None:
    """预测误差 → 过程奖励 (回流给 bandit/episodic 的稀疏"预测命中"信号).

    把相对 RMS 预测误差映射到 [0, 1]: 命中 (误差→0) 得高奖励, 漂移 (误差大) 趋 0.
    ``reward = exp(-error / half_life)``, 单调递减、无除零、对量级稳健.
    ``error`` 为 None/NaN 时返回 None (没有预测就谈不上"命中", 不给奖励, 不误报 0).
    """
    if error is None:
        return None
    if isinstance(error, float) and math.isnan(error):
        return None
    return float(math.exp(-float(error) / float(half_life)))


class SparseRewardTuner:
    """用"预测命中→过程奖励"驱动梯度学习的最小残差自校准器.

    对每个观测键学习一个加性残差 offset, 在线 SGD 最小化 (predicted+offset - observed)²,
    即最大化命中奖励 exp(-mse). 适用: sensor 读数带未知系统偏置 / 外部工具输出与解析
    预演有稳定偏差 —— 从 (预测, 实测) 数据对把偏置学出来. 单参数梯度 (ponytail);
    真深度 RL (网络+策略) 属 P2 learned-surrogate 路径, 不在此冒充.
    """

    def __init__(
        self, keys: Sequence[str], *, lr: float = 0.05, reward_exponent: float = 1.0
    ) -> None:
        self.offset: dict[str, float] = dict.fromkeys(keys, 0.0)
        self.lr = lr
        self.reward_exponent = reward_exponent

    def update(
        self, predicted: Mapping[str, Any], observed: Mapping[str, Any]
    ) -> float | None:
        """在线一步: 用预测误差更新 offset, 返回本步命中奖励 (∈(0,1]).

        误差 ~ (predicted+offset - observed). offset 沿 -误差 方向走 = 让误差
        变小 = 让命中奖励变大 (因为 reward = exp(-mse), 单调反比于误差).
        """
        sq = 0.0
        n = 0
        for k in self.offset:
            if k not in predicted or k not in observed:
                continue
            p = float(predicted[k])
            o = float(observed[k])
            err = (p + self.offset[k]) - o
            self.offset[k] -= self.lr * err
            sq += err * err
            n += 1
        if not n:
            return None
        return math.exp(-self.reward_exponent * sq / float(n))


def reconcile_r_phys(
    base: float,
    world_reward: float | None = None,
    *,
    graft: float = 0.5,
) -> float:
    """把"世界预测命中奖励"并进底层 r_phys 聚合 (单一权威实现).

    ``reconciled = graft*base + (1-graft)*world_reward``.
    ``world_reward`` 为 None (无世界跟踪/无预测样本) 时原样返回 ``base``,
    不因缺世界模型而扣分. 这是物理 runner / validate tool 默认消费
    `last_prediction_reward` 的出口 — 下游统一用它喂 bandit.
    """
    if world_reward is None:
        return float(base)
    return float(graft) * float(base) + (1.0 - float(graft)) * float(world_reward)


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
    "WorldStateTracker",
    "SparseRewardTuner",
    "prediction_error_to_reward",
    "reconcile_r_phys",
    "snapshot_from_schema",
]
