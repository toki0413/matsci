"""Research 模式安全门 — 拦截昂贵仿真工具, 标记需要用户确认.

research 模式下 agent 有更大自主权, 一次误判可能跑掉几十个 CPU 小时.
这个 PRE_TOOL_USE hook 不直接 block, 只在 metadata 里标记 requires_confirmation,
让上层 (CLI / WebSocket / adapter) 决定怎么跟用户交互.
"""

from __future__ import annotations

import logging

from huginn.hooks import PRE_TOOL_USE, HookContext, HookManager

logger = logging.getLogger(__name__)

# 这些工具单次调用就可能消耗大量计算资源 (HPC 排队 / GPU 时间 / 长时间运行)
_EXPENSIVE_TOOLS = frozenset({
    "vasp_tool", "lammps_tool", "cp2k_tool", "qe_tool",
    "gaussian_tool", "orca_tool", "gromacs_tool",
    "transolver_tool", "autoloop_tool",
})


def _is_expensive_call(ctx: HookContext) -> bool:
    """检查是否是 research 模式下的昂贵工具调用."""
    agent = getattr(ctx, "agent", None)
    if agent is None or not hasattr(agent, "is_research_mode"):
        return False
    if not agent.is_research_mode():
        return False
    return ctx.tool_name in _EXPENSIVE_TOOLS


async def research_safety_hook(ctx: HookContext) -> HookContext | None:
    """research 模式下昂贵工具调用前标记需要确认."""
    if not _is_expensive_call(ctx):
        return None
    ctx.metadata["requires_confirmation"] = True
    ctx.metadata["confirmation_reason"] = (
        f"Research mode: {ctx.tool_name} is a high-cost simulation tool. "
        "Confirm to proceed, or cancel to skip."
    )
    return ctx


def register_research_safety_hooks(hm: HookManager) -> None:
    """注册研究安全 hook (幂等)."""
    if getattr(hm, "_research_safety_registered", False):
        return
    hm.register(PRE_TOOL_USE, research_safety_hook)
    hm._research_safety_registered = True


if __name__ == "__main__":
    # 快速自检: 非 research 模式不拦, research 模式 + 昂贵工具才拦
    import asyncio

    class _FakeAgent:
        def __init__(self, research: bool) -> None:
            self._research = research

        def is_research_mode(self) -> bool:
            return self._research

    async def _check():
        # research 模式 + 昂贵工具 -> 标记确认
        ctx = HookContext(tool_name="vasp_tool", args={})
        ctx.agent = _FakeAgent(research=True)
        ret = await research_safety_hook(ctx)
        assert ret is not None, "research + vasp should be flagged"
        assert ret.metadata["requires_confirmation"] is True

        # 非 research 模式 -> 放行
        ctx2 = HookContext(tool_name="vasp_tool", args={})
        ctx2.agent = _FakeAgent(research=False)
        ret2 = await research_safety_hook(ctx2)
        assert ret2 is None, "chat mode should not flag"

        # research 模式 + 便宜工具 -> 放行
        ctx3 = HookContext(tool_name="web_search", args={})
        ctx3.agent = _FakeAgent(research=True)
        ret3 = await research_safety_hook(ctx3)
        assert ret3 is None, "cheap tool should not flag"

        # 没有 agent -> 放行 (不崩)
        ctx4 = HookContext(tool_name="vasp_tool", args={})
        ret4 = await research_safety_hook(ctx4)
        assert ret4 is None, "no agent should not flag"

        print("research_safety_hook self-check passed")

    asyncio.run(_check())
