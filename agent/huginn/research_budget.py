"""Research mode 预算追踪 — 限制单次研究会话中昂贵工具的调用次数.

autoloop 的 ProgressiveBudget 按迭代限制 plan mode, 这里按会话限制昂贵工具调用.
故意做得很简单: 一个 dict 计数器 + 阈值检查.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from huginn.hooks import HookContext

logger = logging.getLogger(__name__)

# 单次研究会话中昂贵工具的最大调用次数
_DEFAULT_MAX_EXPENSIVE_CALLS = 10
# 达到此比例时发出警告
_WARN_THRESHOLD_RATIO = 0.8

_EXPENSIVE_TOOLS = frozenset({
    "vasp_tool", "lammps_tool", "cp2k_tool", "qe_tool",
    "gaussian_tool", "orca_tool", "gromacs_tool",
    "transolver_tool", "autoloop_tool",
})


@dataclass
class ResearchBudget:
    """按会话追踪昂贵工具调用次数."""

    max_calls: int = _DEFAULT_MAX_EXPENSIVE_CALLS
    _calls: dict[str, int] = field(default_factory=dict)

    def record_call(self, tool_name: str) -> dict[str, str] | None:
        """记录一次调用, 返回警告信息 dict 如果超限."""
        if tool_name not in _EXPENSIVE_TOOLS:
            return None
        count = self._calls.get(tool_name, 0) + 1
        self._calls[tool_name] = count
        total = sum(self._calls.values())

        if total >= self.max_calls:
            return {
                "severity": "blocking",
                "message": (
                    f"Research budget exhausted: {total}/{self.max_calls} "
                    f"expensive tool calls used. Further calls will be blocked."
                ),
            }
        if total >= int(self.max_calls * _WARN_THRESHOLD_RATIO):
            return {
                "severity": "major",
                "message": (
                    f"Research budget warning: {total}/{self.max_calls} "
                    f"expensive tool calls used."
                ),
            }
        return None

    @property
    def total_calls(self) -> int:
        return sum(self._calls.values())

    @property
    def remaining(self) -> int:
        return max(0, self.max_calls - self.total_calls)

    def reset(self) -> None:
        self._calls.clear()

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_calls": self.max_calls,
            "total_calls": self.total_calls,
            "remaining": self.remaining,
            "per_tool": dict(self._calls),
        }


# 全局单例 (一个 agent 进程一个)
_budget: ResearchBudget | None = None


def get_research_budget() -> ResearchBudget:
    global _budget
    if _budget is None:
        _budget = ResearchBudget()
    return _budget


def reset_research_budget() -> None:
    global _budget
    if _budget is not None:
        _budget.reset()


async def research_budget_hook(ctx: HookContext) -> HookContext | None:
    """POST_TOOL_USE: 记录昂贵工具调用, 超限时 warn/block."""
    agent = getattr(ctx, "agent", None)
    if agent is None or not hasattr(agent, "is_research_mode"):
        return None
    if not agent.is_research_mode():
        return None
    if ctx.tool_name not in _EXPENSIVE_TOOLS:
        return None

    budget = get_research_budget()
    warning = budget.record_call(ctx.tool_name)
    if warning:
        warnings = ctx.metadata.get("warnings", [])
        if not isinstance(warnings, list):
            warnings = []
            ctx.metadata["warnings"] = warnings
        warnings.append(warning)
        if warning["severity"] == "blocking":
            ctx.metadata["blocked_by_hook"] = True
            ctx.metadata["block_reason"] = warning["message"]
            ctx.metadata["severity"] = "blocking"
            return ctx
    return None


if __name__ == "__main__":
    # 自检: 计数 / 警告 / 阻断 / 重置
    import asyncio

    class _FakeAgent:
        def is_research_mode(self) -> bool:
            return True

    async def _check():
        # 用小阈值方便测
        b = ResearchBudget(max_calls=5)
        assert b.total_calls == 0
        assert b.remaining == 5

        # 前 3 次不警告 (3 < 5*0.8=4)
        for i in range(3):
            w = b.record_call("vasp_tool")
            assert w is None, f"call {i+1} should not warn"

        # 第 4 次触发 major 警告 (4 >= 4)
        w = b.record_call("lammps_tool")
        assert w is not None and w["severity"] == "major"
        assert b.total_calls == 4

        # 第 5 次触发 blocking
        w = b.record_call("qe_tool")
        assert w is not None and w["severity"] == "blocking"
        assert b.remaining == 0

        # 非昂贵工具不计
        w = b.record_call("web_search")
        assert w is None
        assert b.total_calls == 5

        # reset
        b.reset()
        assert b.total_calls == 0
        assert b.remaining == 5

        # hook 集成测试
        reset_research_budget()
        ctx = HookContext(tool_name="vasp_tool", args={})
        ctx.agent = _FakeAgent()
        ret = await research_budget_hook(ctx)
        assert ret is None, "first call should not warn"

        # 把 budget 调小再塞一次, 凑到超额
        budget = get_research_budget()
        budget.max_calls = 2
        budget.record_call("vasp_tool")  # hook 已经记了 1 次, 这是第 2 次
        ctx2 = HookContext(tool_name="lammps_tool", args={})
        ctx2.agent = _FakeAgent()
        ret2 = await research_budget_hook(ctx2)
        assert ret2 is not None, "budget exceeded should block"
        assert ret2.metadata.get("blocked_by_hook") is True

        reset_research_budget()
        print("research_budget self-check passed")

    asyncio.run(_check())
