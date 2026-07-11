"""PRE_TOOL_USE 钩子: 未确认 plan 时拦截执行类工具.

配合 design_plan_tool 使用. agent 调用 vasp_tool / lammps_tool 等重型
工具前, 这个钩子先检查 _PlanStore 里是否有 confirmed 的 plan.
- 有 confirmed plan: 放行
- 无 confirmed plan:
  - autoloop 模式: blocked=True 硬阻塞, 提示 agent 先 propose
  - chat 模式: advisory only, 附带提示但放行 (避免堵点)

这样保证"先计划再执行"的 gate 不依赖 LLM 自觉, 由系统强制 (autoloop)
或建议 (chat).
"""

from __future__ import annotations

import logging

from huginn.hooks import HookContext
from huginn.tools.design_plan_tool import GATED_TOOLS, _PlanStore

logger = logging.getLogger(__name__)


class DesignPlanGateHook:
    """PRE_TOOL_USE 钩子: 拦截执行类工具, 要求先有用户确认的 plan.

    mode="strict": 硬阻塞 (autoloop 用)
    mode="advisory": 仅注入提示, 不阻塞 (chat 用)
    """

    def __init__(self, mode: str = "strict") -> None:
        self._store = _PlanStore.instance()
        self._mode = mode

    async def __call__(self, ctx: HookContext) -> HookContext | None:
        try:
            # 只拦截 gated 工具
            if ctx.tool_name not in GATED_TOOLS:
                return None

            # thread_id 从 metadata 拿 (run_pre 调用前由 agent 注入).
            # 不同 thread 的 confirm 状态是隔离的, 不能让 A thread confirm
            # 后 B thread 的 gate 也跟着放行.
            thread_id = ctx.metadata.get("thread_id")
            # 有对应 thread 的 confirmed plan 直接放行
            if self._store.has_confirmed(thread_id):
                return None

            # 无 confirmed plan
            reason = (
                f"工具 {ctx.tool_name} 属于执行类, 需要先有用户确认的 design plan. "
                "请先调用 design_plan_tool action=propose 提交计划, "
                "等用户 action=confirm 确认后再执行."
            )

            if self._mode == "advisory":
                # Chat 模式: 不阻塞, 仅注入提示, 让 agent 自行判断
                ctx.metadata["design_plan_advisory"] = reason
                logger.info(
                    "DesignPlanGate advisory (not blocked) %s — no confirmed plan",
                    ctx.tool_name,
                )
            else:
                # Autoloop / 严格模式: 硬阻塞
                ctx.blocked = True
                ctx.metadata["block_reason"] = reason
                logger.info(
                    "DesignPlanGate blocked %s (no confirmed plan)",
                    ctx.tool_name,
                )
        except Exception:
            # 钩子挂了不能阻塞主流程, 默认放行
            logger.warning("DesignPlanGateHook raised", exc_info=True)
        return ctx
