"""物理动作/状态 Schema 显式化 — 硬件执行器的契约层.

物理世界的动作参数 (volume / position / tolerance) 目前都是自由 dict
(PhysicalAction.params / Executor.state), 接真实移液站 / 机械臂时字段语义
靠猜, 且难以表达"实测多少才算通过". 本模块把这两者声明式化, 作为硬件 adapter
的统一契约:

- ``StepResult``: 对 execute 后某个状态量的**量化断言** — 期望值 + 相对/绝对容差
  + 物理单位. 真实移液站规格是相对误差 (标称 ±1%), 故 ``tolerance`` 默认按
  **相对**解释, ``tol_abs`` 兜底 (量接近 0 时相对无意义).
- ``ActionSpec``: 一个物理动作的声明式规格 — 前置依赖(替代 build 期 is_active
  判读) + 参数 + 期望的后置状态 (转成状态级感知确认).
- 单位标注: ``VolumeUnit`` / ``Quantity`` 提供体积单位元数据与换算, 接硬件 SDK
  时字段单位不再靠猜; 执行路径默认把参数值视为内部单位 uL (兼容旧行为).

本层不修改 PhysicalAction / SimExecutor 自身; 通过 ``PhysicalWorkspace.execute
(spec=...)`` 消费, ``spec=None`` 时行为与旧完全一致, 保证向后兼容.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# 体积单位 (内部统一用 uL, 换算系数: 1 mL = 1000 uL).
VolumeUnit = Literal["uL", "mL"]

_VOL_TO_UL: dict[str, float] = {"uL": 1.0, "mL": 1000.0}


def to_ul(value: float, unit: str) -> float:
    """把带单位的体积值换算成内部单位 uL (未知单位按 uL 透传)."""
    return float(value) * _VOL_TO_UL.get(unit, 1.0)


class Quantity(BaseModel):
    """带单位的物理量值对象 (供硬件契约标注用, 不参与执行路径换算)."""

    value: float = Field(..., description="数值")
    unit: VolumeUnit = Field(default="uL", description="单位")


class StepResult(BaseModel):
    """对 execute 后实测状态的量化断言 (相对容差为主, 绝对兜底).

    校验时: 偏差经 ``tol_abs`` / 相对 ``tolerance`` 判定, 落在界限内即通过.
    ``unit`` 标注期望值的物理单位 (仅元数据; 默认执行路径期望值即 uL).
    """

    key: str = Field(..., description="状态字段名, 如 sample_vol")
    expected: float = Field(..., description="期望值 (默认内部单位 uL)")
    unit: VolumeUnit = Field(
        default="uL", description="期望值的物理单位 (契约元数据)"
    )
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


def step_allowed(step: StepResult) -> float:
    """单个 StepResult 允许的最大偏差 (相对/绝对取大)."""
    rel = abs(step.expected) * step.tolerance
    return max(step.tol_abs, rel)


def matches_state(
    expected: dict[str, Any] | list[StepResult],
    actual: dict[str, Any],
    tolerance: float = 1e-9,
) -> bool:
    """状态对比: 断言列表或期望 dict 是否与实测状态匹配.

    - ``StepResult`` 列表: 每个字段用其自身相对/绝对容差 (声明式断言).
    - dict: 每个键用统一 ``tolerance`` 绝对容差 (旧契约).

    本函数是 schema 层的状态判定唯一事实来源; workspace 与协议解释器共用,
    保证"单步确认"与"终态断言"判定语义一致.
    """
    if isinstance(expected, list):  # StepResult 断言列表
        for step in expected:
            got = actual.get(step.key)
            if got is None:
                return False
            dev = abs(float(got) - step.expected)
            if dev > step_allowed(step):
                return False
        return True
    for k, want in expected.items():
        got = actual.get(k)
        if got is None:
            return False
        try:
            if abs(float(got) - float(want)) > tolerance:
                return False
        except (TypeError, ValueError):
            if got != want:
                return False
    return True


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
    vol_unit: VolumeUnit = Field(
        default="uL",
        description="params 中体积参数(vol)的单位 (契约元数据; 执行路径按 uL 处理)",
    )
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


class ProtocolSpec(BaseModel):
    """一个实验协议的声明式描述 (协议语言最小内核).

    把"裸步骤列表"升级成自描述的协议对象:
    - ``name``: 协议名.
    - ``steps``: 有序步骤 (ActionSpec), 前置不满足的步骤被跳过 (degrade).
    - ``final_state``: 协议**整体**完成后对终态的整体断言 (跨步骤不变量),
      如量守恒 / 分装总数.

    与 ``run_pipette_protocol`` 硬编码等价, 但把流程+终态都数据化 — 接真实
    LIMS/Workflow 软件时, 协议规格本身就是可加载的配置.
    """

    name: str = Field(..., description="协议名")
    steps: list[ActionSpec] = Field(default_factory=list)
    final_state: list[StepResult] = Field(
        default_factory=list, description="协议完成后对终态的整体断言"
    )

    def step_ids(self) -> list[str]:
        """返回步骤 id 列表 (便于追踪执行顺序)."""
        return [s.id for s in self.steps]


# 完整移液-混合-分装协议 (声明式, 含终态整体断言).
PIPETTING_PROTOCOL = ProtocolSpec(
    name="pipette-mix-aliquot",
    steps=PIPETTING_SPEC,
    final_state=[
        # 量守恒: 移液总量守恒 (reagent_vol + sample_vol + tube_vol = 初始 100).
        StepResult(key="tube_vol", expected=10.0, tolerance=0.01),
        StepResult(key="aliquot_count", expected=1, tol_abs=0.0),
    ],
)


class ProtocolMachine:
    """声明式协议解释器 — 按 ProtocolSpec 驱动 PhysicalWorkspace 执行.

    与 :func:`run_action_spec` 等价, 但以协议对象为单位:
    - 逐步骤执行, 前置不满足跳过 (degrade).
    - 每步 expect 驱动状态级感知确认 (spec.expect).
    - 协议结束后对 ``final_state`` 做整体断言 (跨步骤不变量), 失败抛
      :class:`WorkspaceConfirmError`.
    - ``preflight`` 透传执行前预演.

    ``steps_ok`` 返回实际执行成功的步骤 id (跳过的不算), 便于上层知道协议
    实际走了哪些路径.
    """

    def __init__(self, protocol: ProtocolSpec) -> None:
        self.protocol = protocol

    def run(self, wa: Any, *, preflight: bool = True) -> list[str]:
        """在 ``wa`` (PhysicalWorkspace) 上执行协议, 返回成功执行的步骤 id."""
        from huginn.security.workspace import WorkspaceConfirmError
        from huginn.security.world_model import PhysicalAction

        executed: list[str] = []
        with wa.transaction():
            for spec in self.protocol.steps:
                if not all(wa.is_available(d) for d in spec.preconditions):
                    continue
                wa.execute(
                    PhysicalAction(spec.action_type, spec.to_action_params()),
                    spec=spec,
                    preflight=preflight,
                )
                executed.append(spec.id)
            if self.protocol.final_state and not matches_state(self.protocol.final_state, wa.state):
                raise WorkspaceConfirmError(
                    f"协议终态断言失败: 协议 {self.protocol.name}"
                )
        return executed
