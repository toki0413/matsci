"""物理动作/状态 Schema 显式化 — 硬件执行器的契约层.

物理世界的动作参数 (volume / position / tolerance) 目前都是自由 dict
(PhysicalAction.params / Executor.state), 接真实移液站 / 机械臂时字段语义
靠猜, 且难以表达"实测多少才算通过". 本模块把这两者声明式化, 作为硬件 adapter
的统一契约:

- ``StepResult``: 对 execute 后某个状态量的**量化断言** — 期望值 + 相对/绝对容差.
  真实移液站规格是相对误差 (标称 ±1%), 故 ``tolerance`` 默认按**相对**解释,
  ``tol_abs`` 兜底 (量接近 0 时相对无意义).
- ``ActionSpec``: 一个物理动作的声明式规格 — 前置依赖(替代 build 期 is_active
  判读) + 参数 + 期望的后置状态 (转成状态级感知确认).

本层不修改 PhysicalAction / SimExecutor 自身; 通过 ``PhysicalWorkspace.execute
(spec=...)`` 消费, ``spec=None`` 时行为与旧完全一致, 保证向后兼容.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StepResult(BaseModel):
    """对 execute 后实测状态的量化断言 (相对容差为主, 绝对兜底).

    校验时: 偏差经 ``tol_abs`` / 相对 ``tolerance`` 判定, 落在界限内即通过.
    """

    key: str = Field(..., description="状态字段名, 如 sample_vol")
    expected: float = Field(..., description="期望值 (统一物理单位, 见单位约定)")
    tolerance: float = Field(
        default=0.01, ge=0, le=1,
        description="相对容差 (比例, 0.01=±1%). 对 vol/pos 等有量纲量为主体判据",
    )
    tol_abs: float = Field(
        default=1e-9, ge=0,
        description="绝对容差兜底: 当 |expected| 极小 (相对无意义) 时改用此值",
    )

    def delta_allowed(self) -> float:
        """返回允许的最大偏差: 取绝对容差与相对容差中较大者."""
        rel = abs(self.expected) * self.tolerance
        return max(self.tol_abs, rel)


class ActionSpec(BaseModel):
    """一个物理动作的声明式规格.

    - ``preconditions``: 依赖 key, 全部满足才执行. 替代 build 期 is_active 判读,
      让"能否执行"成为可运行时断言的后置约束.
    - ``expect``: 执行后应达到的状态断言 (状态级感知确认的声明式来源).
    - ``params``: 透传给 PhysicalAction 的参数字典.
    """

    id: str = Field(..., description="动作 id, 如 aspirate_step")
    action_type: str = Field(..., description="物理动作类型, 如 aspirate")
    params: dict[str, Any] = Field(default_factory=dict)
    preconditions: list[str] = Field(
        default_factory=list, description="前置依赖 key, 全部满足才可执行"
    )
    expect: list[StepResult] = Field(default_factory=list)

    def to_action_params(self) -> dict[str, Any]:
        """转成 PhysicalAction 期望的参数字典 (透传)."""
        return dict(self.params)


# 移液-混合-分装协议: 声明式规格 (与 run_pipette_protocol 硬编码等价).
PIPETTING_SPEC: list[ActionSpec] = [
    ActionSpec(
        id="aspirate_step",
        action_type="aspirate",
        params={"vol": 10},
        preconditions=["reagent.ready", "pipette.ready"],
        expect=[
            StepResult(key="reagent_vol", expected=90.0),
            StepResult(key="sample_vol", expected=10.0),
        ],
    ),
    ActionSpec(
        id="dispense_step",
        action_type="dispense",
        params={"vol": 10},
        preconditions=["sample.aspirated", "tube.present"],
        expect=[
            StepResult(key="sample_vol", expected=0.0, tol_abs=0.0),
            StepResult(key="tube_vol", expected=10.0),
        ],
    ),
    ActionSpec(
        id="mix_step",
        action_type="mix",
        params={"mode": "vortex"},
        preconditions=["tube.filled", "mixer.ready"],
        expect=[StepResult(key="mixed", expected=1.0)],
    ),
    ActionSpec(
        id="aliquot_step",
        action_type="aliquot",
        params={"n": 3},
        preconditions=["mixture.mixed"],
        expect=[StepResult(key="aliquot_count", expected=1)],
    ),
]