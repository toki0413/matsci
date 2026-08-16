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


class WorldModel(Protocol):
    """逆生成器: 从 (前状态, 动作) 推断逆动作.

    一个动作是可逆的 (返回逆动作) 还是不可逆的 (返回 ``None``), 由世界模型
    决定. 不可逆动作说明无法自动回滚, 调用方需自行处理 (如人工介入 / 保留).
    """

    def infer_inverse(
        self,
        state_before: dict[str, Any],
        action: PhysicalAction,
    ) -> PhysicalAction | None:
        ...


class NaiveWorldModel:
    """朴素逆规则 — 用于验证核心机制, 非真实物理模型.

    依据动作类型和参数推断对称逆:
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
