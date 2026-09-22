"""Pi 极简内核模式 —— 让 huginn 更像 Pi (极简) 而非 OpenClaw (肥大).

设计对齐 Pi 的核心哲学:
  - **极简 system prompt**: 几百 token, 不塞 180 工具的行为说明, 把上下文留给真实工作.
  - **默认隐藏工具动物园**: ``apply_pi_mode`` 把非原语工具置 ``active=False``
    (``get_all_schemas`` 据此从 LLM 可见 schema 里滤除), 但工具仍经
    ``ToolRegistry.get`` 可调 —— 需要某能力时用 ``move_tool``/``capability_tool``/
    ``tool_search`` 按需找回来, 或让模型 ``make_tool`` 当场写一个.
  - **可逆**: ``exit_pi_mode`` 恢复全部工具可见.

Pi 原语集 (对应 Pi 的 read/write/edit/bash + 自扩展):
  file_read_tool / file_write_tool / multi_edit_tool / bash_tool / code_tool
  + tool_search (按需发现) + capability_tool (整箱能力) + make_tool (自写工具).
"""
from __future__ import annotations

from typing import Any

from huginn.tools.registry import ToolRegistry

# Pi 原语集 —— 极简模式下仍暴露给 LLM 的少量工具
PRIMITIVES: frozenset[str] = frozenset({
    "file_read_tool",
    "file_write_tool",
    "multi_edit_tool",
    "bash_tool",
    "code_tool",
    "tool_search",
    "capability_tool",
    "make_tool",
})

# 极简系统提示 —— 刻意简短, 把上下文让给真实工作/自扩展
PI_SYSTEM_PROMPT: str = (
    "你是 Huginn, 一个普适科研 agent，处于 pi 极简模式。\n"
    "原则：\n"
    "1. 只暴露少量原语（读写编辑/bash/搜索/能力/自写工具）。没有你要的能力，就用\n"
    "   make_tool 当场写一个：给 name 和定义 HuginnTool 子类的 Python 源码，下一步即可调用。\n"
    "2. 上下文纪律：只把对当前任务必要的信息放进上下文，不塞无关工具说明。\n"
    "3. 科学诚实：任何声称都要可证伪（可执行、可对账）；不可证伪的预判不冒充结论。\n"
    "4. 你掌控你造的工具：make_tool 按需经权限门禁放行。\n"
)


def apply_pi_mode() -> dict[str, Any]:
    """进入 pi 模式: 隐藏非原语工具 (active=False), 强制原语可见. 可逆."""
    primitives_ok = 0
    hidden = 0
    total = 0
    for name in ToolRegistry.list_tools():
        tool = ToolRegistry.get(name)
        if tool is None:
            continue
        total += 1
        if name in PRIMITIVES:
            if not tool.active:
                tool.active = True
            primitives_ok += 1
        else:
            if tool.active:
                tool.active = False
                hidden += 1
    return {
        "mode": "pi",
        "total_tools": total,
        "exposed_primitives": primitives_ok,
        "hidden": hidden,
        "primitives": sorted(PRIMITIVES),
    }


def exit_pi_mode() -> dict[str, Any]:
    """退出 pi 模式: 恢复全部已注册工具可见."""
    restored = 0
    for name in ToolRegistry.list_tools():
        tool = ToolRegistry.get(name)
        if tool is not None and not tool.active:
            tool.active = True
            restored += 1
    return {"mode": "default", "restored": restored}


def pi_active() -> bool:
    """当前是否 pi 模式: 任一原语 active 为 True 且存在隐藏工具即视为模式生效."""
    hidden = any(
        (ToolRegistry.get(n) is not None) and not ToolRegistry.get(n).active  # type: ignore[union-attr]
        for n in ToolRegistry.list_tools()
    )
    return hidden


def pi_system_prompt() -> str:
    """返回 pi 极简系统提示 (供 persona/prompt 构建处注入)."""
    return PI_SYSTEM_PROMPT


__all__ = [
    "PRIMITIVES", "PI_SYSTEM_PROMPT",
    "apply_pi_mode", "exit_pi_mode", "pi_active", "pi_system_prompt",
]
