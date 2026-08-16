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
