"""能力统一契约基类 — "集装箱"的标准箱体.

任何能力 (原子工具 / 组合流程 / 外部 MCP 服务) 都实现同一个接口, 从而能在
同一份"能力清单"里被发现、被装配、被编排, 也能在调用方视角统一使用.

与既有 ``HuginnTool`` 的区别:
- ``HuginnTool`` 是"单次调用"的原子单元;
- ``Capability`` 是"可组合的箱体" — 它可以是原子工具, 也可以是多个子能力
  编排成的复合能力, 对外暴露一致的 ``run()`` 契约.

设计原则 (集装箱隐喻):
- 标准尺寸: 统一 ``run(input, context) -> CapabilityResult`` 契约
- 箱体贴标: ``name/category/description/input_schema/output_schema``
- 健康检查: ``is_available()`` 决定能否装载上船 (对 LLM 不可见时静默跳过)
- 可组合: ``sub_capabilities`` 声明依赖; ``degradation_chain`` 声明降级备选
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from huginn.core_types import ToolContext, ToolResult

InputT = TypeVar("InputT", bound=BaseModel)


class CapabilityError(Exception):
    """能力执行失败。包含可读 message 与机器可读 code。"""

    def __init__(self, message: str, code: str = "capability_error") -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class CapabilityResult:
    """能力调用的统一返回值 (集装箱卸下的货)。

    与 ToolResult 对齐但更轻: 固定含 ``success/data/error`` 三要素,
    便于组合能力在各步骤间流转。
    """

    success: bool = True
    data: Any = None
    error: str | None = None
    code: str = "ok"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_tool_result(self) -> ToolResult:
        """转回 ToolResult, 供既有调度/LLM 层消费。"""
        return ToolResult(
            data=self.data,
            success=self.success,
            error=self.error,
            metadata=self.metadata,
        )

    @classmethod
    def ok(cls, data: Any = None, **meta: Any) -> CapabilityResult:
        return cls(success=True, data=data, metadata=meta)

    @classmethod
    def fail(cls, error: str, code: str = "capability_error", **meta: Any) -> CapabilityResult:
        return cls(success=False, error=error, code=code, metadata=meta)


class Capability(ABC, Generic[InputT]):
    """能力集装箱的基类。子类实现 ``_run`` 即可, 得到统一契约。"""

    name: str = ""
    category: str = "misc"
    description: str = ""
    # input_schema 是可选的; 严格校验时用 Pydantic model
    input_schema: type[InputT] | None = None

    # 本能力依赖的子能力名 — 用于能力清单展示依赖图, 也供编排器做拓扑排序
    sub_capabilities: tuple[str, ...] = ()
    # 降级链: 本能力不可用/失败时, 依次尝试的后备能力名 (同契约)
    degradation_chain: tuple[str, ...] = ()

    # 只读能力可自动执行; 写能力需显式确认 (对齐 HuginnTool.read_only/destructive)
    read_only: bool = False
    destructive: bool = False

    def is_available(self) -> bool:
        """能否装载上船。外部资源断连 / 可选依赖缺失时返回 False。

        默认 True; 依赖 HuginnTool 的能力基于底层 tool.is_available()。
        """
        return True

    def _validate_input(self, args: Any) -> InputT:
        """把入参规整成 input_schema 实例。不校验时, 原样透传。"""
        if self.input_schema is not None:
            if isinstance(args, self.input_schema):
                return args
            if isinstance(args, dict):
                try:
                    return self.input_schema(**args)
                except Exception as exc:
                    # pydantic validation / type coercion failure -> 标准化入参错误
                    raise CapabilityError(
                        f"{self.name}: invalid input for {self.input_schema.__name__}: {exc}",
                        code="invalid_input",
                    ) from exc
            raise CapabilityError(
                f"{self.name}: input must be dict or {self.input_schema.__name__}",
                code="invalid_input",
            )
        if isinstance(args, dict):
            # 无 schema 时用宽松 dict 包一层轻量校验对象
            return args  # type: ignore[return-value]
        return args  # type: ignore[return-value]

    async def run(self, args: Any, context: ToolContext | None = None) -> CapabilityResult:
        """统一执行入口。子类实现 ``_run``, 这里负责校验 + 异常收敛。"""
        try:
            validated = self._validate_input(args)
        except CapabilityError as exc:
            return CapabilityResult.fail(exc.message, code=exc.code)
        try:
            return await self._run(validated, context)
        except CapabilityError as exc:
            return CapabilityResult.fail(exc.message, code=exc.code)
        except Exception as exc:  # noqa: BLE001 — 能力边界收敛, 不上抛
            return CapabilityResult.fail(
                f"{type(exc).__name__}: {exc}", code="execution_error"
            )

    @abstractmethod
    async def _run(self, args: InputT, context: ToolContext | None) -> CapabilityResult:
        """执行本体。抛 CapabilityError 或用 return 的 CapabilityResult 表达结果。"""

    @property
    def input_json_schema(self) -> dict[str, Any] | None:
        if self.input_schema is not None:
            return self.input_schema.model_json_schema()
        return None
