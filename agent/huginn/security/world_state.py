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

import contextlib
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

    def __init__(
        self,
        schema: Mapping[str, Any],
        world_model: Any,
        *,
        surrogate: LearnableForwardModel | None = None,
    ) -> None:
        self.schema = schema
        self.estimator = StateEstimator(schema)
        self.predictor = ForwardPredictor(schema)
        self.world_model = world_model
        # P2 线: 可选学代理, observe 时把 (前状态, 动作, 实测后状态) 喂给它积累训练数据.
        self.surrogate = surrogate
        self._last_state_before: dict[str, float] | None = None
        self._last_action: Any = None
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
        self._last_state_before = {str(k): float(v) for k, v in state_before.items()}
        self._last_action = action
        return snapshot

    def observe(self, state_after: Mapping[str, Any]) -> float | None:
        """执行后观测: 算"预测 vs 实测"的 RMS 误差 (回流信号).

        无预测 / 二者无共享维度时返回 None (不误报 0 错). 学代理喂养独立于预测成功 —
        只要有 (前状态, 动作, 实测后状态) 就喂, 让代理从实际执行里积累数据.
        """
        # P2 线: 本轮 (前状态, 动作, 实测后状态) 喂给学代理积累真实运行数据.
        # 有前状态/动作/代理三件套才喂; 失败吞掉 (advisory, 不阻塞实验).
        if (
            self.surrogate is not None
            and self._last_state_before is not None
            and self._last_action is not None
        ):
            with contextlib.suppress(Exception):
                self.surrogate.fit(
                    self._last_state_before, self._last_action, state_after
                )
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

    def surrogate_samples(self) -> int:
        """学代理已积累的观测对数 (跨动作类型). 无代理 → 0.
        供调用方确认"真实运行数据确实喂进代理" (P2 快预演积累的证据)."""
        if self.surrogate is None:
            return 0
        return sum(len(rows) for rows, _ in getattr(self.surrogate, "_samples", {}).values())

    def surrogate_predict(
        self, state: Mapping[str, Any], action: Any
    ) -> dict[str, float] | None:
        """快预演入口: 让学代理先试着预测 (有样本才预测; 无代理/无样本 → None).
        ``world_model`` 解析投影是慢而准的真值, 代理是基于真实运行学来的快近似 —
        这里把代理预测作为可选项给出, 供调用方权衡 (不必替换解析真值)."""
        if self.surrogate is None:
            return None
        try:
            pred = self.surrogate.predict(state, action)
        except Exception:
            return None
        return pred if isinstance(pred, Mapping) else None


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


class LearnableForwardModel:
    """P2 learned-surrogate 最小种子: 数据驱动的线性前向代理.

    用 (state_before, action_params) 拼特征, 按 ``action.type`` 分别做多输出最小
    二乘, 拟合 ``observed_after`` 的每个状态键. ``predict(state, action)`` 契约与
    ``ForwardPredictor`` 打通 —— 可直接替换解析前向真值当 predictor 用
    (这正是"把解析真值换成可学代理"的那一步). 样本不足/该动作类型无样本时返回
    None, 不硬编造. 解析前向真值 → 代理可学后, 接策略梯度属更后续 (P2 深).
    """

    def __init__(self, state_keys: Sequence[str]) -> None:
        self.state_keys = list(state_keys)
        # action_type -> (特征行列表, {输出键: [真实值]}); 系数按动作分桶缓存.
        self._samples: dict[str, tuple[list[list[float]], dict[str, list[float]]]] = {}
        self._coef: dict[str, dict[str, list[float]]] = {}

    def fit(
        self,
        state_before: Mapping[str, Any],
        action: Any,
        observed_after: Mapping[str, Any],
    ) -> LearnableForwardModel:
        """喂一条 (前状态, 动作, 后状态) 数据对并惰性重解该动作的最小二乘."""
        act_type = getattr(action, "type", None)
        if act_type is None:
            return self
        feat = _forward_features(state_before, self.state_keys, action)
        rows, ys = self._samples.setdefault(act_type, ([], {}))
        rows.append(feat)
        for k in self.state_keys:
            if k in observed_after:
                ys.setdefault(k, []).append(float(observed_after[k]))
        w = _solve_bucket(rows, ys)
        if w is not None:
            self._coef[act_type] = w
        return self

    def predict(self, state: Mapping[str, Any], action: Any) -> dict[str, float] | None:
        """代理前向预测. 契约与 ForwardPredictor 兼容 (predict(state, action))."""
        act_type = getattr(action, "type", None)
        coef = self._coef.get(act_type)
        if not coef:
            return None
        feat = _forward_features(state, self.state_keys, action)
        return {k: _dot(w, feat) for k, w in coef.items()}


def _forward_features(
    state: Mapping[str, Any], state_keys: Sequence[str], action: Any
) -> list[float]:
    """特征: [1] + 各状态键值 + 动作参数量(可转 float 的)."""
    feat = [1.0]
    for k in state_keys:
        try:
            feat.append(float(state.get(k, 0.0)))
        except (TypeError, ValueError):
            feat.append(0.0)
    params = getattr(action, "params", {}) or {}
    for _k, _v in params.items():
        try:
            feat.append(float(_v))
        except (TypeError, ValueError):
            continue
    return feat


def _solve_bucket(
    rows: list[list[float]], ys: dict[str, list[float]]
) -> dict[str, list[float]] | None:
    """对每个输出键解正规方程 (A w = b), 样本不足返回 None.

    特征矩阵可能列秩不足: 真实运行里某状态键恒定 (如等容加热 V/n 不变) 会让 Gram 矩阵
    奇异 -> ``_gauss_solve`` 无解. 此时对每个键内做 ridge 兜底 (A + λI), 把最小二乘
    解稳定出来, 而不是整体 drop. 满秩时走原精确路径, 不引入扰动.
    """
    if not rows or len(rows) < len(rows[0]):
        return None
    d = len(rows[0])
    A = [[sum(r[i] * r[j] for r in rows) for j in range(d)] for i in range(d)]
    # ridge 系数相对对角量级取极小值, 只用来稳住奇异 Gram, 不污染满秩解.
    lam = 1e-6 * max(sum(A[i][i] for i in range(d)) / d, 1.0)
    out: dict[str, list[float]] = {}
    for k, ylist in ys.items():
        if len(ylist) < len(rows):
            continue
        b = [sum(r[i] * y for r, y in zip(rows, ylist)) for i in range(d)]
        # _gauss_solve 内联消元会就地改写传入矩阵, 每键都须用 A 的拷贝.
        w = _gauss_solve([list(Ai) for Ai in A], b)
        if w is None:
            # 奇异 -> ridge 兜底 (仅该键), 不影响其他键/满秩场景.
            Ar = [list(Ai) for Ai in A]
            for i in range(d):
                Ar[i][i] += lam
            w = _gauss_solve(Ar, b)
        if w is not None:
            out[k] = w
    return out or None


def _gauss_solve(A: list[list[float]], b: list[float]) -> list[float] | None:
    """高斯消去解小方阵 A w = b; 奇异返回 None."""
    n = len(A)
    for i in range(n):
        piv = max(range(i, n), key=lambda r: abs(A[r][i]))
        if abs(A[piv][i]) < 1e-12:
            return None
        A[i], A[piv] = A[piv], A[i]
        b[i], b[piv] = b[piv], b[i]
        for r in range(i + 1, n):
            m = A[r][i] / A[i][i]
            for c in range(i, n):
                A[r][c] -= m * A[i][c]
            b[r] -= m * b[i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(A[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (b[i] - s) / A[i][i]
    return x


def _dot(w: Sequence[float], x: Sequence[float]) -> float:
    return sum(float(w[i]) * float(x[i]) for i in range(len(w)))


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
    "LearnableForwardModel",
    "prediction_error_to_reward",
    "reconcile_r_phys",
    "snapshot_from_schema",
]
