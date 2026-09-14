"""物理世界接入核心结构: WorldModel 作为逆生成器 (Cordis 时空可组合的物理实例化).

核心结构 (revertible/coeffect) 是领域无关的抽象; 本模块把核心落到物理世界
(世界模型 / VLA / 机器人 / 实验室实验), 成为它的第三次实例化 (前两次:
软件沙箱 SandboxExecutor、视觉链 visual_chain).

与软件世界的差异: 软件逆可显式枚举 (删文件/恢复 env), 物理世界的逆不天然
已知 — 撤销一个动作需要知道"如何从新状态回到旧状态". 本模块提供:

- ``PhysicalAction``: 一个可序列化的物理动作 (type + params), 可 journal.
- ``WorldModel``: 逆生成器接口. 从 (前状态, 动作) 推断逆动作.
- ``NaiveWorldModel``: 朴素逆规则 (move 逆=反向 move, grasp 逆=release ...).
  先验证"核心机制可用", 后续可替换为数据驱动的真实世界模型.

逆动作由 WorldModel 推断后, 经 :meth:`RevertibleContext.track_world_action`
登记为 OP_ACTION 数据驱动逆, 由物理执行器 (register_physical_executor) 执行.
"""

from __future__ import annotations

import contextlib
from typing import Any, Protocol


class PhysicalAction:
    """一个可序列化的物理动作."""

    __slots__ = ("type", "params")

    def __init__(self, type: str, params: dict[str, Any] | None = None) -> None:
        self.type = type
        self.params = dict(params or {})

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "params": self.params}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PhysicalAction:
        return cls(d.get("type", ""), d.get("params") or {})

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, PhysicalAction)
            and self.type == other.type
            and self.params == other.params
        )

    def __repr__(self) -> str:
        return f"PhysicalAction({self.type!r}, {self.params!r})"


# ── 前向转移规则 (单一事实来源) ───────────────────────────────────
# 与 SimExecutor (workspace.py) 的 _EFFECTS 语义一致, 集中在此供 world model
# 预演复用, 避免两处前向逻辑漂移. type -> (dec_key, inc_key, vol_param).
FORWARD_EFFECTS: dict[str, tuple[str, str, str]] = {
    "aspirate": ("reagent_vol", "sample_vol", "vol"),
    "dispense": ("sample_vol", "tube_vol", "vol"),
    "mix": ("", "mixed", ""),
    "aliquot": ("", "aliquot_count", ""),
}


def apply_forward(state_before: dict[str, Any], action: PhysicalAction) -> dict[str, Any]:
    """纯函数前向预测: 从 ``state_before`` 推出 ``state_after`` (无副作用).

    未知动作类型返回输入状态的拷贝 (不做任何转移). 不修改入参.
    """
    state = dict(state_before)
    effect = FORWARD_EFFECTS.get(action.type)
    if effect is None:
        return state
    dec_key, inc_key, vol_param = effect
    if dec_key:
        v = float(action.params.get(vol_param, 0) or 0)
        state[dec_key] = max(0.0, float(state.get(dec_key, 0.0)) - v)
    if inc_key == "mixed":
        state["mixed"] = True
    elif inc_key == "aliquot_count":
        state["aliquot_count"] = int(state.get("aliquot_count", 0)) + 1
    elif inc_key:
        v = float(action.params.get(vol_param, 0) or 0)
        state[inc_key] = float(state.get(inc_key, 0.0)) + v
    return state


def check_constraints(state_before: dict[str, Any], action: PhysicalAction) -> list[str]:
    """第一性原理物理约束校验: 返回违规描述列表 (空列表 = 合法).

    不依赖硬件的守恒/边界约束: 体积非负、源量充足、分装次数合理.
    供执行前预演 (preflight) 用, 命中即应阻止动作.
    """
    issues: list[str] = []
    t = action.type
    if t in ("aspirate", "dispense"):
        v = float(action.params.get("vol", 0) or 0)
        if v < 0:
            issues.append(f"{t} vol<0: {v}")
    if t == "aspirate":
        needed = float(action.params.get("vol", 0) or 0)
        have = float(state_before.get("reagent_vol", 0.0))
        if needed > have:
            issues.append(f"aspirate 源量不足: 需{needed} 有{have}")
    if t == "dispense":
        needed = float(action.params.get("vol", 0) or 0)
        have = float(state_before.get("sample_vol", 0.0))
        if needed > have:
            issues.append(f"dispense 样液不足: 需{needed} 有{have}")
    if t == "aliquot":
        n = int(action.params.get("n", 0) or 0)
        if n < 0:
            issues.append(f"aliquot n<0: {n}")
    return issues


class WorldModel(Protocol):
    """世界模型: 逆生成器 (后向) + 前向预测 (预演).

    - ``infer_inverse``: 从 (前状态, 动作) 推断逆动作. 可逆返回逆动作, 不可逆
      返回 ``None`` (无法自动回滚, 调用方自行处理).
    - ``predict``: 从 (前状态, 动作) 预测后状态 (无副作用). 供执行前预演 /
      感知确认对比预期用.
    """

    def infer_inverse(
        self,
        state_before: dict[str, Any],
        action: PhysicalAction,
    ) -> PhysicalAction | None:
        ...

    def predict(
        self,
        state_before: dict[str, Any],
        action: PhysicalAction,
    ) -> dict[str, Any]:
        ...


class NaiveWorldModel:
    """朴素世界模型 — 逆规则 + 确定性前向规则, 用于验证核心机制, 非真实物理模型.

    依据动作类型和参数推断对称逆, 并用 ``FORWARD_EFFECTS`` 做前向预测
    (与 SimExecutor 语义一致):
    - ``move``   : 逆 = 反向 move (start/target 互换)
    - ``grasp``  : 逆 = ``release`` (同参数)
    - ``release``: 逆 = ``grasp`` (同参数)
    - ``dispense``: 逆 = ``aspirate`` (同参数)
    - ``aspirate``: 逆 = ``dispense`` (同参数)
    其余动作视为不可逆 (返回 None).
    """

    _INVERSE_TYPE: dict[str, str] = {
        "grasp": "release",
        "release": "grasp",
        "dispense": "aspirate",
        "aspirate": "dispense",
    }

    def infer_inverse(
        self,
        state_before: dict[str, Any],
        action: PhysicalAction,
    ) -> PhysicalAction | None:
        if action.type == "move":
            params = dict(action.params)
            start = params.get("start")
            target = params.get("target")
            if start is not None and target is not None:
                return PhysicalAction("move", {"start": target, "target": start})
            return None
        inverse_type = self._INVERSE_TYPE.get(action.type)
        if inverse_type is None:
            return None
        return PhysicalAction(inverse_type, dict(action.params))

    def predict(
        self,
        state_before: dict[str, Any],
        action: PhysicalAction,
    ) -> dict[str, Any]:
        """前向预测: 从 (前状态, 动作) 推出后状态 (纯函数, 复用 FORWARD_EFFECTS).

        与 SimExecutor._apply 语义一致, 供执行前预演与感知确认对比预期.
        """
        return apply_forward(state_before, action)


class ConstraintViolationError(Exception):
    """物理约束校验失败 — 动作违背第一性原理边界, 执行前被阻止."""

    def __init__(self, action: PhysicalAction, issues: list[str]) -> None:
        self.action = action
        self.issues = issues
        super().__init__(f"物理约束违规 ({action.type}): " + "; ".join(issues))


# ── 数据驱动前向世界模型 (dual-axis RSI 的"第二根轴"雏形) ────────────────────
# 硬编码 FORWARD_EFFECTS 是"先验规则"; 本类让**前向预测**从 agent 真实执行数据
# (state_before, action, state_after) 学习, 替代/补强静态规则. 校验约束不变
# (物理不变量不该被学), 只学"世界如何变换"这一可学部分.
_MIN_LEARN_SAMPLES = 3


class LearnedWorldModel:
    """数据驱动前向世界模型: 从真实执行观测学习每个动作类型对状态键的效应.

    - ``learn(state_before, action, state_after)``: 累加 (新增键→sets, 数值键→delta 均值).
    - ``predict(state_before, action)``: 应用已学效应的前向预测; 无模型/未知动作 → 原样拷贝.
    - ``has_model(action_type)``: 是否学到足够样本 (>= ``_MIN_LEARN_SAMPLES``).
    持久化到 ``.huginn/world_model/learned.json``, 供跨 session 复用.
    """

    def __init__(self, path: Any | None = None) -> None:
        if path is None:
            from pathlib import Path
            from huginn.utils.runtime import get_runtime_home

            path = Path(get_runtime_home()) / "world_model" / "learned.json"
        self._path = path
        # action_type -> {"sets": {key: val}, "deltas": {key: {"sum": float, "n": int}}, "count": n}
        self._effects: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        with contextlib.suppress(Exception):
            if self._path.exists():
                import json

                self._effects = json.loads(self._path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        with contextlib.suppress(Exception):
            import os

            os.makedirs(self._path.parent, exist_ok=True)
            import json

            self._path.write_text(json.dumps(self._effects, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _as_float(v: Any) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    def learn(self, state_before: dict[str, Any], action: PhysicalAction,
              state_after: dict[str, Any]) -> None:
        """学习一次真实转移: 新增键记入 sets, 数值键记入 delta 均值."""
        eff = self._effects.setdefault(
            action.type, {"sets": {}, "deltas": {}, "count": 0})
        for k, v in state_after.items():
            if k in state_before:
                delta = self._as_float(v) - self._as_float(state_before[k])
                if abs(delta) > 1e-9:
                    d = eff["deltas"].setdefault(k, {"sum": 0.0, "n": 0})
                    d["sum"] += delta
                    d["n"] += 1
            else:
                # 新增键 (如 mixed/flag): 记录其出现值
                eff["sets"][k] = bool(v) if isinstance(v, bool) else v
        eff["count"] += 1
        self._save()

    def has_model(self, action_type: str) -> bool:
        eff = self._effects.get(action_type)
        return bool(eff and eff["count"] >= _MIN_LEARN_SAMPLES)

    def predict(self, state_before: dict[str, Any], action: PhysicalAction) -> dict[str, Any]:
        """前向预测. 无该动作模型 → 原样拷贝 (保守); 有 → 应用 sets + delta 均值."""
        state = dict(state_before)
        eff = self._effects.get(action.type)
        if not eff or eff["count"] < _MIN_LEARN_SAMPLES:
            return state
        for k, v in eff["sets"].items():
            state[k] = v
        for k, d in eff["deltas"].items():
            delta = d["sum"] / max(1, d["n"])
            state[k] = self._as_float(state.get(k, 0.0)) + delta
        for key in list(state.keys()):
            if key not in state_before and key not in eff["sets"]:
                state.pop(key, None)
        return state

    def snapshot(self) -> dict[str, Any]:
        return {t: {"sets": dict(ef["sets"]),
                    "deltas": {k: round(d["sum"] / max(1, d["n"]), 4)
                               for k, d in ef["deltas"].items()},
                    "count": ef["count"]}
                for t, ef in self._effects.items()}


_LEARNED_INSTANCE: LearnedWorldModel | None = None


def learned_world_model(path: Any | None = None) -> LearnedWorldModel:
    """Returns the shared data-driven forward world model (可注入 path 供测试隔离)."""
    global _LEARNED_INSTANCE
    if path is not None or _LEARNED_INSTANCE is None:
        if path is not None:
            _LEARNED_INSTANCE = LearnedWorldModel(path)
        else:
            _LEARNED_INSTANCE = LearnedWorldModel()
    return _LEARNED_INSTANCE


# ── 仿真世界 + 协调净评价器 (world-closure 目标域) ───────────────────────────
# 给 RSI 干净跑一个"真实"目标域: agent 在 SimWorld 里执行动作, 世界返回真实观测,
# 评价器根据真实状态产出 r_phys. 由此 RPhys 归因 / VerifiableGate 真实实验 /
# LearnedWorldModel 学习 / RandomControl 采样 都对着真实世界产物闭合, 而非空转.


class SimWorld:
    """确定性液相传移仿真世界 + 协调净评价器.

    - ``step(action)``: 应用"真实"转移效果 (与 FORWARD_EFFECTS 同构) 并返回观测.
      含小幅确定性噪声, 让数据驱动世界模型必须从多次执行学习均值.
    - ``r_phys()``: 协调净评价器 — 越接近目标态 (reagent 逼近 target) r_phys 越高.
    - ``snapshot``/``reset``: 状态管理.
    """

    def __init__(self, target_reagent: float = 2.0, noise: float = 0.05,
                 seed: int = 7) -> None:
        import random

        self._rng = random.Random(seed)
        self.target_reagent = target_reagent
        self.noise = noise
        self.state: dict[str, float] = {"reagent_vol": 10.0, "sample_vol": 0.0,
                                        "tube_vol": 0.0}
        self._moves = 0

    def step(self, action: PhysicalAction) -> dict[str, float]:
        """执行动作并返回观测世界状态 (含噪声)."""
        st = dict(self.state)
        eff = FORWARD_EFFECTS.get(action.type)
        if eff is not None:
            dec_key, inc_key, vol_param = eff
            v = self._as_maybe(action.params.get(vol_param, 0) or 0)
            n = self._rng.uniform(-self.noise, self.noise)
            if dec_key:
                st[dec_key] = max(0.0, self._as_maybe(st.get(dec_key, 0.0)) - v)
            if inc_key == "mixed":
                st["mixed"] = True
            elif inc_key == "aliquot_count":
                st["aliquot_count"] = int(st.get("aliquot_count", 0)) + 1
            elif inc_key:
                st[inc_key] = self._as_maybe(st.get(inc_key, 0.0)) + v + n
        self.state = st
        self._moves += 1
        return dict(st)

    def r_phys(self) -> float:
        """协调净评价器: 离目标态越近 r_phys 越高 (真实世界评分, 非代理)."""
        err = abs(self._as_maybe(self.state.get("reagent_vol", 0.0)) - self.target_reagent)
        return max(0.0, min(1.0, 1.0 - err / 10.0))

    def snapshot(self) -> dict[str, float]:
        return dict(self.state)

    @staticmethod
    def _as_maybe(v: Any) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0
