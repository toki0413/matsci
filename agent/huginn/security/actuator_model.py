"""执行器/传感器模型基类 — "前向真值 + 传感器读数视图 + 域随机化 + 运行时标定".

microduck sim2real 的核心抽象落到 Huginn: 一个物理执行器由三件事组成

1. **前向真值引擎** ``_forward(state, action)``: 动作如何推进世界状态。
   对 duck 是 MuJoCo 仿真器; 对 Huginn 是规则表 / 计算工具 / 未来学到的高保真
   仿真器 — 这是阶段 4 "世界模型事实来源" 的唯一替换点.
2. **传感器读数视图** ``sensor_view(state, action_type)``: 真实传感器在**输出侧**
   读到"实测" = 世界状态 + 系统偏置(systematic) + 零均值噪声(sigma). 感知确认
   必须在该视图下进行, 否则会惩罚"偏置但正确"的动作 (编码器读穿齿轮输出侧).
3. **运行时标定** ``calibrate(bias)``: systematic 只能标定消除, 不能让 DR /
   放宽容差去吞掉它 (DR 只覆盖零均值, 系统(安装/刻度)偏置 DR 治不了).

新的物理工具只需声明三样东西即可套用本骨架: ``forward``(状态推进)、
``sensor_keys``(每个动作影响哪些量, 决定偏置施加在哪)、``error_model``(该工具
的 systematic/sigma). 相同抽象让多工具共享一致的 execute/observe/确认/回滚语义.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from huginn.security.world_model import PhysicalAction


@dataclass
class ErrorModel:
    """硬件执行误差模型 — 施加在体积相关量的实测值上.

    两类误差的处理必须不同 (microduck sim2real 核心教训):
    - ``systematic``: **系统(刻度/安装)偏置**, 每次固定叠加, 是运行时标定
      (``SensorModelExecutor.calibrate``) 要补偿的东西, 不是 DR/容差要容忍的.
    - ``sigma``: 零均值随机误差标准差, 是"域随机化"让确认去训练容忍的东西.

    仅对这类带体积/量化参数的动作注入; 离散量 (如 mix/aliquot) 不受影响.
    """

    systematic: float = 0.0
    sigma: float = 0.0


class SensorModelExecutor(ABC):
    """可复用物理执行器骨架: execute / observe / sensor_view / calibrate 全实现,
    子类只声明 前向真值(``_forward``) 与 传感器键(``_sensor_keys``)."""

    def __init__(
        self,
        fail_on: set[str] | None = None,
        *,
        initial: dict[str, Any] | None = None,
        error_model: ErrorModel | None = None,
        seed: int | None = None,
    ) -> None:
        self.log: list[PhysicalAction] = []
        self.fail_on = set(fail_on or ())
        self.state: dict[str, Any] = dict(initial or {})
        self.error_model = error_model
        self._rng = random.Random(seed)

    # ── 子类需实现 ──────────────────────────────────────────────
    @abstractmethod
    def _forward(self, state: dict[str, Any], action: PhysicalAction) -> dict[str, Any]:
        """前向真值: 返回动作执行后的世界状态 (纯函数, 不修改入参)."""

    @abstractmethod
    def _sensor_keys(self, action_type: str) -> tuple[str, str]:
        """返回该动作影响的 (dec_key, inc_key). 系统偏置按"stx等幅反相"施于其上,
        保持量守恒; 无影响键返回 ("", "")."""

    def _noise_eligible(self, action: PhysicalAction) -> bool:
        """是否对本动作注入执行误差 (默认总是; 移液等按需覆盖)."""
        return bool(
            self._sensor_keys(action.type)[0] or self._sensor_keys(action.type)[1]
        )

    # ── 对外公共接口 (全实现) ───────────────────────────────────
    def execute(self, action: PhysicalAction) -> None:
        if action.type in self.fail_on:
            raise RuntimeError(
                f"{self.__class__.__name__}: action {action.type} failed"
            )
        self.state = self._forward(self.state, action)
        self._inject_noise(action)
        self.log.append(action)

    def observe(self) -> dict[str, Any]:
        return dict(self.state)

    def calibrate(self, bias: float | None = None) -> float:
        """运行时标定: 把系统性偏置 systematic 从读数里补偿掉.

        microduck 规则: 系统偏置 DR 治不了, 只能标定 — 别靠放宽容差去训练容忍.
        调用后, 后续感知确认在同一容差下即可通过. 返回本次消除的偏置量.
        """
        if self.error_model is None:
            return 0.0
        take = self.error_model.systematic if bias is None else bias
        self.error_model.systematic -= take
        return take

    def sensor_view(self, state: dict[str, Any], action_type: str) -> dict[str, Any]:
        """把 action_type 执行后的"理想世界状态"投影到与实测量同一读数视角.

        只复刻 systematic (确定、幂等); 零均值 sigma 属 execute 时的真实注入, 确认
        时保留为真随机 DR — 正是 microduck "编码器读穿输出侧" 的语义.
        """
        if self.error_model is None or self.error_model.systematic == 0.0:
            return dict(state)
        bias = self.error_model.systematic
        view = dict(state)
        dec_key, inc_key = self._sensor_keys(action_type)
        if (
            inc_key
            and inc_key not in ("mixed", "aliquot_count")
            and isinstance(view.get(inc_key), int | float)
        ):
            view[inc_key] = float(view.get(inc_key, 0.0)) + bias
        if dec_key and isinstance(view.get(dec_key), int | float):
            view[dec_key] = float(view.get(dec_key, 0.0)) - bias
        return view

    # ── 内部 ───────────────────────────────────────────────────
    def _inject_noise(self, action: PhysicalAction) -> None:
        """域随机化注入: systematic + 零均值 sigma, 等幅反相保持量守恒."""
        if self.error_model is None or not self._noise_eligible(action):
            return
        dec_key, inc_key = self._sensor_keys(action.type)
        noise = self.error_model.systematic + self._rng.gauss(
            0.0, self.error_model.sigma
        )
        if (
            inc_key
            and inc_key not in ("mixed", "aliquot_count")
            and isinstance(self.state.get(inc_key), int | float)
        ):
            self.state[inc_key] = float(self.state.get(inc_key, 0.0)) + noise
        if dec_key and isinstance(self.state.get(dec_key), int | float):
            self.state[dec_key] = float(self.state.get(dec_key, 0.0)) - noise
