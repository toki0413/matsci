"""三类能力实现 — 原子 / 组合 / 外部 (集装箱的三种"装载物").

- :class:`AtomicCapability`: 包装单个 HuginnTool, 把一次工具调用封装成能力。
- :class:`CompositeCapability`: 编排多个子能力为一个流程 (DAG 顺序执行), 是
  "组合能力复用"的核心 — 把多步流程沉淀成可一键调用的一体能力。
- :class:`ExternalCapability`: 包装外部服务 (如 MCPToolAdapter), 让外部能力
  经同一契约接入能力清单。
"""

from __future__ import annotations

import logging
from typing import Any

from huginn.capabilities.base import (
    Capability,
    CapabilityResult,
    InputT,
)
from huginn.core_types import ToolContext
from huginn.tools.base import HuginnTool

logger = logging.getLogger(__name__)


class AtomicCapability(Capability[InputT]):
    """把单个 HuginnTool 包装成原子能力。

    ``degradation_chain`` / ``read_only`` / ``is_available`` 会自动代理到底层
    tool 对应的 profile 语义, 保证装箱后行为与直接调 tool 一致。
    """

    def __init__(self, tool: HuginnTool) -> None:
        self.tool = tool
        self.name = tool.name
        self.category = getattr(tool, "category", "misc")
        self.description = tool.description
        self.input_schema = tool.input_schema  # type: ignore[assignment]
        self.read_only = tool.read_only
        self.destructive = tool.destructive
        # profile 里的降级链自动透传 (若已声明)
        profile = getattr(tool, "profile", None)
        if profile is not None:
            self.degradation_chain = tuple(profile.degradation_chain or ())

    def is_available(self) -> bool:
        return self.tool.is_available()

    async def _run(self, args: InputT, context: ToolContext | None) -> CapabilityResult:
        if context is None:
            context = ToolContext(session_id="capability", workspace=".")
        result = await self.tool.call(args, context)
        if result.success:
            return CapabilityResult.ok(result.data)
        return CapabilityResult.fail(
            result.error or "tool failed", code="tool_error"
        )


def capability_from_tool(tool: HuginnTool) -> AtomicCapability:
    """便捷工厂：从 HuginnTool 构造原子能力。"""
    return AtomicCapability(tool)


class CompositeCapability(Capability[InputT]):
    """组合能力 — 把多个子能力按顺序编排成一个流程。

    ``steps`` 是 (步骤名, 子能力名, 输入转换或 None) 的列表:
        steps = [
            ("search",  "literature_search_cap", None),
            ("verify",  "literature_verify_cap",  lambda prev, inp: {...}),
            ...
        ]

    每步的输出存在 ``state`` 里 (``state["<step>"]``), 供后续步骤用
    ``input_transform(prev, inp)`` 引用。任一子能力失败即整体失败 (fail-fast),
    除非它在 ``best_effort`` 里 (失败记 warning 不中断)。
    """

    def __init__(
        self,
        name: str,
        steps: list[tuple[str, Capability, Any]],
        *,
        description: str = "",
        category: str = "composite",
        best_effort: set[str] | None = None,
        input_schema: type[Any] | None = None,
        degradation_chain: tuple[str, ...] = (),
    ) -> None:
        self.name = name
        self.category = category
        self.description = description
        self._steps = steps
        self._best_effort = best_effort or set()
        self.input_schema = input_schema
        # sub_capabilities 记录"依赖的原子能力名" (对清单/编排更有意义),
        # 而非步骤名 — 步骤名只用于 state 里定位, 单独存 _steps.
        deps: list[str] = []
        for _name, cap, _t in steps:
            if cap.name and cap.name not in deps:
                deps.append(cap.name)
        self.sub_capabilities = tuple(deps)
        self.degradation_chain = degradation_chain

    def is_available(self) -> bool:
        return all(s[1].is_available() for s in self._steps)

    async def _run(self, args: InputT, context: ToolContext | None) -> CapabilityResult:
        state: dict[str, Any] = {"_input": args}
        step_log: list[dict[str, Any]] = []
        for step_name, cap, transform in self._steps:
            # 输入: 无 transform -> 原 args; 有 -> transform(前步 state, 原 args)
            step_input = args if transform is None else transform(state, args)
            cap_result = await cap.run(step_input, context)
            state[step_name] = cap_result
            step_log.append({
                "step": step_name,
                "capability": cap.name,
                "success": cap_result.success,
                "error": cap_result.error,
            })
            if not cap_result.success:
                if step_name in self._best_effort:
                    logger.warning(
                        "capability %s: best_effort step '%s' failed: %s",
                        self.name, step_name, cap_result.error,
                    )
                    continue
                return CapabilityResult.fail(
                    f"step '{step_name}' ({cap.name}) failed: {cap_result.error}",
                    code="step_failed",
                    steps=step_log,
                )
        return CapabilityResult.ok(
            {"steps": step_log, "state": state},
        )


class ExternalCapability(Capability[InputT]):
    """外部能力 — 包装一个外部服务 (通常是 MCPToolAdapter)。

    与 AtomicCapability 唯一的区别是: 它强调"来自外部", 且不强制要求底层是
    注册表里的 HuginnTool, 只要实现 ``call(args, context) -> ToolResult`` 即可。
    """

    def __init__(
        self,
        name: str,
        *,
        description: str,
        backend: Any,
        category: str = "external",
        read_only: bool = False,
    ) -> None:
        self.name = name
        self.category = category
        self.description = description
        self._backend = backend
        self.read_only = read_only
        # 若 backend 有 input_schema (如 MCP adapter), 透传
        self.input_schema = getattr(backend, "input_schema", None) or None  # type: ignore[assignment]

    def is_available(self) -> bool:
        backend = self._backend
        if isinstance(backend, HuginnTool):
            return backend.is_available()
        check = getattr(backend, "is_available", None)
        return check() if callable(check) else True

    async def _run(self, args: InputT, context: ToolContext | None) -> CapabilityResult:
        if context is None:
            context = ToolContext(session_id="capability", workspace=".")
        result = await self._backend.call(args, context)
        if result.success:
            return CapabilityResult.ok(result.data)
        return CapabilityResult.fail(
            result.error or "external capability failed", code="external_error"
        )
