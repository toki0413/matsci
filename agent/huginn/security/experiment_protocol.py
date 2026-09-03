"""实验协议演示 — "移液—混合—分装" (pipetting—mixing—aliquoting).

展示核心结构的物理三次实例化如何表达一个真实实验协议:

- **空间可组合性**: 协议被拆成依赖链组件, 任一环节后端缺失 (如离心机/
  混合器不可用) 时, 下游步骤自动停用 (degrade), 而不是无依据地继续.
- **时间可组合性**: 整个协议作为一个事务运行; 任一步失败 (执行失败或感知
  确认失败) → 物理逆按 LIFO 执行, 工作台恢复到协议前.

组件依赖链:
  reagent.ready ─┐
                 ├→ aspirate → sample.aspirated → dispense → tube.filled
  pipette.ready ─┘                                          │
                                                             ↓
                    mixer.ready → mix → mixture.mixed → aliquot → aliquots.done
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from huginn.security.physics_schema import ActionSpec
from huginn.security.workspace import ActionExecutor, PhysicalWorkspace
from huginn.security.world_model import NaiveWorldModel, PhysicalAction

logger = logging.getLogger(__name__)

# 空间依赖链 produce 键.
K_REAGENT = "reagent.ready"
K_PIPETTE = "pipette.ready"
K_MIXER = "mixer.ready"
K_TUBE = "tube.present"
K_ASPIRATED = "sample.aspirated"
K_FILLED = "tube.filled"
K_MIXED = "mixture.mixed"
K_ALIQUOTS = "aliquots.done"

# 协议步骤组件 id.
C_ASPIRATE = "aspirate_step"
C_DISPENSE = "dispense_step"
C_MIX = "mix_step"
C_ALIQUOT = "aliquot_step"


def build_pipette_workflow(
    executor: ActionExecutor,
    world_model: NaiveWorldModel | None = None,
    *,
    mixer_available: bool = True,
    revertible: Any | None = None,
    schema: Mapping[str, Any] | None = None,
    surrogate: Any = None,
) -> PhysicalWorkspace:
    """接线一个"移液—混合—分装"工作台: 声明依赖链 + 感知确认器.

    ``mixer_available=False`` 模拟混合器缺失 → mix/aliquot 自动停用.
    ``revertible`` 可传入外部逆上下文 (如 ToolContext.revertible), 让物理逆
    进入 agent 统一逆栈; 否则工作台自建.
    ``schema``: 传入 ToolSpec 机器可读描述时启用世界表征跟踪, runner 可用
    ``wa.reconcile_r_phys(base)`` 把预测命中奖励并进 r_phys (阶段3默认消费).
    ``surrogate``: 可选学代理 (``LearnableForwardModel``), 配 ``schema`` 一起用 —
    每次真实执行 (sim/真工具) 的观测对会喂进代理积累样本, 随后可用
    ``wa.surrogate_predict(state, action)`` 做快预演 (P2 线).
    """
    wa = PhysicalWorkspace(
        world_model or NaiveWorldModel(),
        executor,
        revertible=revertible,
        schema=schema,
        surrogate=surrogate,
    )

    # 资源后端 (无 requires, 后端就绪即激活).
    wa.declare("reagent_backend", provides=(K_REAGENT,))
    wa.declare("pipette_backend", provides=(K_PIPETTE,))
    wa.declare("mixer_backend", provides=(K_MIXER,))
    wa.declare("tube_backend", provides=(K_TUBE,))

    # 协议步骤依赖链.
    wa.declare(C_ASPIRATE, requires=(K_REAGENT, K_PIPETTE), provides=(K_ASPIRATED,))
    wa.declare(C_DISPENSE, requires=(K_ASPIRATED, K_TUBE), provides=(K_FILLED,))
    wa.declare(C_MIX, requires=(K_FILLED, K_MIXER), provides=(K_MIXED,))
    wa.declare(C_ALIQUOT, requires=(K_MIXED,), provides=(K_ALIQUOTS,))

    # 资源就绪.
    wa.set_available(K_REAGENT, True)
    wa.set_available(K_PIPETTE, True)
    wa.set_available(K_TUBE, True)
    wa.set_available(K_MIXER, mixer_available)

    # 感知确认器: 基于执行器日志判断动作确实完成了 (模拟视觉/力觉闭环).
    def done(*types: str) -> bool:
        return any(a.type in types for a in executor.log)

    wa.confirm(K_ASPIRATED, lambda: done("aspirate"))
    wa.confirm(K_FILLED, lambda: done("dispense"))
    wa.confirm(K_MIXED, lambda: done("mix"))
    wa.confirm(K_ALIQUOTS, lambda: done("aliquot"))

    return wa


def run_pipette_protocol(
    wa: PhysicalWorkspace, *, aliquot: bool = True, sim: bool = False
) -> None:
    """在一个事务里执行完整协议 (失败即整体回滚).

    只执行当前激活的步骤 (依赖缺失的步骤被空间链自动停用).
    ``sim=True`` 时启用执行前预演 (约束校验 + 世界模型前向预测对比实测状态).
    """
    with wa.transaction():
        if wa.is_active(C_ASPIRATE):
            wa.execute(
                PhysicalAction("aspirate", {"vol": 10}),
                confirm_key=K_ASPIRATED,
                preflight=sim,
            )
        if wa.is_active(C_DISPENSE):
            wa.execute(
                PhysicalAction("dispense", {"vol": 10}),
                confirm_key=K_FILLED,
                preflight=sim,
            )
        if wa.is_active(C_MIX):
            wa.execute(
                PhysicalAction("mix", {"mode": "vortex"}),
                confirm_key=K_MIXED,
                preflight=sim,
            )
        if aliquot and wa.is_active(C_ALIQUOT):
            wa.execute(
                PhysicalAction("aliquot", {"n": 3}),
                confirm_key=K_ALIQUOTS,
                preflight=sim,
            )


def run_action_spec(
    wa: PhysicalWorkspace,
    spec_list: list[ActionSpec] | None = None,
    *,
    sim: bool = True,
) -> None:
    """声明式协议执行器 — 加载 ActionSpec 列表并逐一执行 (规格驱动).

    与 :func:`run_pipette_protocol` 的硬编码顺序等价, 但把流程从"手写 if/链"
    提升为"数据驱动": 每个步骤的前置依赖 (preconditions) 与后置断言 (expect)
    都来自规格本身.

    - 前置依赖不满足的步骤被**跳过** (degrade, 与硬编码 ``is_active`` 语义一致),
      整个协议继续跑完能跑的部分, 不崩溃.
    - ``expect`` 驱动状态级感知确认 (每字段相对/绝对容差), 替代旧版仅看执行日志
      的 verifier.
    - ``sim=True`` 额外启用执行前预演 (约束校验 + 前向预测).

    ``spec_list`` 缺省使用 :data:`PIPETTING_SPEC` (等价移液-混合-分装协议).
    """
    from huginn.security.physics_schema import PIPETTING_SPEC as _DEFAULT_SPEC

    specs = spec_list if spec_list is not None else _DEFAULT_SPEC
    with wa.transaction():
        for spec in specs:
            # 前置依赖不满足 → 跳过该步, 不执行 (空间 degrade).
            if not all(wa.is_available(d) for d in spec.preconditions):
                continue
            wa.execute(
                PhysicalAction(spec.action_type, spec.to_action_params()),
                spec=spec,
                preflight=sim,
            )
