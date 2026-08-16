"""实验协议工具 — 在 agent 主循环里执行可回滚、可降级的实验室实验协议.

把核心的物理世界实例化 (PhysicalWorkspace) 封装成 agent 可调用的工具:

- **时间可组合**: 整个协议作为一个事务运行; 任一步失败 (执行失败或感知确认
  失败) → 物理逆按 LIFO 执行, 工作台恢复到协议前. 逆登记进
  ``ToolContext.revertible`` (若存在), 上层整体回滚会一并撤销实验副作用.
- **空间可组合**: 协议步骤声明依赖链; 后端缺失 (如混合器不可用) → 下游步骤
  自动停用 (degrade), 只执行激活的步骤.

当前实现基于 MockExecutor (物理后端为 mock), 用于验证时空可组合机制;
真实 VLA/仿真执行器可从 ``executor_backend`` 扩展.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.core_types import ToolContext, ToolResult
from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile

logger = logging.getLogger(__name__)

# 协议步骤 id (与 experiment_protocol 对齐).
_STEP_IDS = ("aspirate_step", "dispense_step", "mix_step", "aliquot_step")


class ExperimentProtocolInput(BaseModel):
    action: Literal["run", "status"] = Field(
        default="run", description="run=执行一个协议批次, status=查询步骤激活状态"
    )
    protocol: Literal["pipetting"] = Field(
        default="pipetting", description="实验协议 (当前仅 pipetting 移液-混合-分装)"
    )
    mixer_available: bool = Field(
        default=True,
        description="混合器是否可用; False 模拟后端缺失 → mix/aliquot 自动停用",
    )
    executor_backend: Literal[
        "mock", "sim"
    ] = Field(
        default="sim",
        description=(
            "物理执行器后端: mock=内存日志 (无状态), "
            "sim=状态转移仿真 (确定性规则, 感知确认可校验量变化)"
        ),
    )


class ExperimentProtocolTool(HuginnTool):
    """执行一个可回滚、可降级的实验室实验协议 (时空可组合性的物理实例化)."""

    name = "experiment_protocol_tool"
    category = "sci"
    profile = ToolProfile(phases=frozenset({ResearchPhase.EXECUTION}))
    description = (
        "Execute a composable wet-lab experiment protocol (pipetting-mixing-aliquoting) "
        "with spatiotemporal composability: the whole protocol runs as a transaction so "
        "any failed/incorrect step rolls back the physical actions in LIFO order, and "
        "steps whose backend is missing are auto-disabled (degrade). Returns the executed "
        "steps, disabled steps, and inverse count."
    )
    input_schema = ExperimentProtocolInput

    def is_read_only(self, args: ExperimentProtocolInput) -> bool:
        return args.action == "status"

    def _build_workspace(self, args: ExperimentProtocolInput, context: ToolContext) -> Any:
        from huginn.security.experiment_protocol import build_pipette_workflow
        from huginn.security.workspace import MockExecutor, SimExecutor
        from huginn.security.world_model import NaiveWorldModel

        if args.executor_backend == "sim":
            executor = SimExecutor()
        else:
            executor = MockExecutor()
        rv = getattr(context, "revertible", None)
        wa = build_pipette_workflow(
            executor,
            world_model=NaiveWorldModel(),
            mixer_available=args.mixer_available,
            revertible=rv,
        )
        return wa, executor, rv

    @staticmethod
    def _steps_status(wa: Any) -> dict[str, bool]:
        return {sid: wa.is_active(sid) for sid in _STEP_IDS}

    async def call(
        self, args: ExperimentProtocolInput | dict, context: ToolContext
    ) -> ToolResult:
        # 兼容 dict 与模型两种输入: 统一分派入口 dispatch_tool / HTTP call_tool
        # 传 dict (工具内部 schema 校验), 直接工具调用传模型. 见 file_read_tool 同款约定.
        if not isinstance(args, ExperimentProtocolInput):
            args = ExperimentProtocolInput.model_validate(args)
        try:
            wa, executor, rv = self._build_workspace(args, context)

            if args.action == "status":
                return ToolResult(
                    success=True,
                    data={
                        "protocol": args.protocol,
                        "steps": self._steps_status(wa),
                    },
                )

            from huginn.security.experiment_protocol import run_pipette_protocol

            run_pipette_protocol(wa, sim=(args.executor_backend == "sim"))
            executed = [a.type for a in executor.log]
            degraded = [
                sid for sid, active in self._steps_status(wa).items() if not active
            ]
            return ToolResult(
                success=True,
                data={
                    "protocol": args.protocol,
                    "executor_backend": args.executor_backend,
                    "executed_steps": executed,
                    "degraded_steps": degraded,
                    "final_state": executor.observe() if args.executor_backend == "sim" else None,
                    "inverse_count": wa.revertible.depth,
                    "revertible_shared_with_agent": rv is not None,
                },
            )
        except Exception as exc:
            # 执行失败/感知确认失败 → 事务已回滚, 物理逆已执行. 透传失败语义.
            return ToolResult(data=None, success=False, error=f"{type(exc).__name__}: {exc}")
