"""材料科学内置 hook — 使用 huginn.hooks.HookManager 的 async API.

这些 hook 通过 register_science_hooks(hook_manager) 注册, 复用现有的
HookContext + run_pre/run_post 机制, 不创建并行系统.

与 __init__.py 里的 AnomalyDetectionHook (登记异常) 互补:
  - AnomalyDetectionHook: 只登记, 不阻断
  - 这里的 hook: 可以 block (如 VASP 未收敛), 保护 agent 不基于坏数据继续走
"""

from __future__ import annotations

import logging
from typing import Any

from huginn.hooks import POST_TOOL_USE, PRE_TOOL_USE, HookContext, HookManager

logger = logging.getLogger(__name__)


# ── VASP 收敛性检查 ─────────────────────────────────────────────

async def vasp_convergence_hook(ctx: HookContext) -> HookContext | None:
    """POST_TOOL_USE: 检查 VASP 输出是否收敛. 不收敛时 block.

    block 方式: ctx.metadata['block_reason'] = '...', 上层的 adapter
    检查这个字段并替换输出为错误消息.
    """
    if ctx.tool_name != "vasp_tool":
        return None

    result = ctx.result if isinstance(ctx.result, dict) else {}
    text = ""
    if "result" in result and isinstance(result["result"], dict):
        data = result["result"]
        converged = data.get("converged")
        if converged is True:
            return None  # 明确收敛
        if converged is False:
            ctx.metadata["block_reason"] = (
                "VASP calculation did not converge. Consider increasing "
                "NSW (max ionic steps) or adjusting EDIFF/EDIFFG."
            )
            ctx.metadata["blocked_by_hook"] = True
            return ctx
        # converged 字段不存在, 从输出文本判断
        text = str(data)
    else:
        text = str(result)

    text_lower = text.lower()
    if "reached required accuracy" in text_lower:
        return None  # 收敛了
    if "aborting loop" in text_lower or "brions" in text_lower:
        ctx.metadata["block_reason"] = (
            "VASP ionic relaxation did not reach required accuracy. "
            "Consider increasing NSW or adjusting EDIFFG."
        )
        ctx.metadata["blocked_by_hook"] = True
        return ctx
    return None


# ── LAMMPS 稳定性检查 ───────────────────────────────────────────

async def lammps_stability_hook(ctx: HookContext) -> HookContext | None:
    """POST_TOOL_USE: 检查 LAMMPS 是否丢原子或不稳定."""
    if ctx.tool_name != "lammps_tool":
        return None

    result = ctx.result if isinstance(ctx.result, dict) else {}
    text = str(result.get("result", result)).lower()

    if "lost atoms" in text:
        ctx.metadata["block_reason"] = (
            "LAMMPS simulation lost atoms — system became unstable. "
            "Consider reducing timestep or using a softer potential."
        )
        ctx.metadata["blocked_by_hook"] = True
        return ctx
    return None


# ── 结构合理性检查 ───────────────────────────────────────────────

async def structure_sanity_hook(ctx: HookContext) -> HookContext | None:
    """POST_TOOL_USE: 检查结构工具返回的原子间距是否物理合理."""
    if ctx.tool_name != "structure_tool":
        return None

    result = ctx.result if isinstance(ctx.result, dict) else {}
    data = result.get("result", {})
    if not isinstance(data, dict):
        return None

    positions = data.get("positions") or data.get("frac_positions")
    if not positions or not isinstance(positions, (list, tuple)) or len(positions) < 2:
        return None

    import math

    for i in range(len(positions)):
        for j in range(i + 1, min(i + 5, len(positions))):
            p1, p2 = positions[i], positions[j]
            if not (isinstance(p1, (list, tuple)) and len(p1) >= 3):
                continue
            dx = float(p1[0]) - float(p2[0])
            dy = float(p1[1]) - float(p2[1])
            dz = float(p1[2]) - float(p2[2])
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if 0 < dist < 0.5:
                ctx.metadata["block_reason"] = (
                    f"Structure sanity check: atoms {i} and {j} are "
                    f"only {dist:.2f} Å apart (min 0.5 Å). Check coordinates."
                )
                ctx.metadata["blocked_by_hook"] = True
                return ctx
    return None


# ── 注册函数 ─────────────────────────────────────────────────────

def register_science_hooks(hm: HookManager) -> None:
    """把材料科学 hook 注册到 HookManager. 在 agent 初始化时调用.

    幂等: 检查是否已注册, 避免重复.
    """
    # 用 attribute 标记已注册, 避免重复
    if getattr(hm, "_science_hooks_registered", False):
        return

    hm.register(POST_TOOL_USE, vasp_convergence_hook)
    hm.register(POST_TOOL_USE, lammps_stability_hook)
    hm.register(POST_TOOL_USE, structure_sanity_hook)
    hm._science_hooks_registered = True
    logger.info("Science hooks registered (vasp_convergence, lammps_stability, structure_sanity)")
