"""Agent 顶层 mode 词表 — 单一事实来源 (A3).

此前 mode 词表散在六处: prompt 段键集、session 快照恢复白名单、
``metacog.critique`` 校验集、``core._LONG_HORIZON_MODES``、``set_mode()``
校验实参、``_mode`` 字段注解. 各写一份字面量, 增删 mode 时必然漏改一处.

本模块只承载词表, 不承载行为:

- ``AgentMode``: ``Literal`` (静态检查用)
- ``ALL_MODES``: 全集 (运行期遍历/校验/测试用)
- ``SETTABLE_MODES``: ``set_mode()`` 可切换的公开子集
- ``LONG_HORIZON_MODES``: 需要长程上下文管理的子集
- ``CRITIQUE_MODES``: metacog critique 校验的 mode 子集
- ``PROMPT_MODES``: 有 prompt 段引导的 mode 集合

ponytail: 纯 stdlib, 无运行期依赖, 任意模块可安全导入 (不引入循环).
"""

from __future__ import annotations

from typing import Literal

AgentMode = Literal["chat", "plan", "research", "code", "extreme", "fusion"]

ALL_MODES: tuple[AgentMode, ...] = (
    "chat",
    "plan",
    "research",
    "code",
    "extreme",
    "fusion",
)

SETTABLE_MODES: tuple[AgentMode, ...] = ("chat", "research", "plan", "extreme")

LONG_HORIZON_MODES: tuple[AgentMode, ...] = ("research", "extreme")

CRITIQUE_MODES: frozenset[AgentMode] = frozenset({"chat", "plan", "research"})

PROMPT_MODES: tuple[AgentMode, ...] = ("chat", "research", "extreme", "code", "fusion")
