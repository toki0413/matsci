"""工具注册/交换机制 — M5"同一接口换后端"的软件兑现.

一个物理工具 = (执行器 + 世界模型 + 物理健康门控). 通过 ``get_tool(name)`` 解析,
安装时按制品其 ``tool`` 身份走对应健康门控与后端构建. 换后端 = 换制品, 不换
上层 (PhysicalWorkspace / BehaviorLifecycle / ExecutionGuard).

真实 HPC 工具 (VASP/DFT) 未来以同样的 ``ToolSpec`` 注册, 只需把外部/异步执行与
输出解析接进 ``build_executor`` 的 ``_forward`` — 其余层次完全复用.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from huginn.security.actuator_model import SensorModelExecutor
from huginn.security.behavior_lifecycle import (
    BehaviorArtifact,
    BehaviorLifecycle,
    InstallResult,
)
from huginn.security.world_model import WorldModel


class UnknownToolError(Exception):
    """未注册的工具名."""


@dataclass(frozen=True)
class ToolSpec:
    """一个物理工具的后端构建与健康门控规范."""

    name: str
    build_executor: Callable[[BehaviorArtifact], SensorModelExecutor]
    build_world_model: Callable[[], WorldModel]
    health_check: Callable[[BehaviorArtifact], bool]


_TOOLS: dict[str, ToolSpec] = {}


def register_tool(spec: ToolSpec) -> None:
    _TOOLS[spec.name] = spec


def get_tool(name: str) -> ToolSpec:
    try:
        return _TOOLS[name]
    except KeyError:
        raise UnknownToolError(f"unknown tool: {name!r}") from None


def make_components(
    tool: str, artifact: BehaviorArtifact
) -> tuple[SensorModelExecutor, WorldModel]:
    """由工具名 + 制品解析 (执行器, 世界模型) — 同一接口, 不同后端."""
    spec = get_tool(tool)
    return spec.build_executor(artifact), spec.build_world_model()


def install_tool(
    lifecycle: BehaviorLifecycle,
    tool: str,
    artifact: BehaviorArtifact,
) -> InstallResult:
    """安装某工具的制品, 用该工具自身的物理健康门控做安装门控."""
    spec = get_tool(tool)
    return lifecycle.install(
        artifact, health_check=lambda _v: spec.health_check(artifact)
    )


def registered_tools() -> tuple[str, ...]:
    return tuple(sorted(_TOOLS))


# ── 内置工具注册 (导入时完成) ──────────────────────────────────
def _register_builtins() -> None:
    from huginn.security.mechanics_oscillator import (
        OscillatorWorldModel,
        osc_executor_from_artifact,
        osc_health_check,
    )
    from huginn.security.thermo_system import (
        IdealGasWorldModel,
        executor_from_artifact,
        thermo_health_check,
    )
    from huginn.security.van_der_waals import (
        VanDerWaalsWorldModel,
        vdw_executor_from_artifact,
        vdw_health_check,
    )

    # 热力学域 (理想气 / 范德华) + 力学域 (一维振子): 领域无关, 只是各自 ToolSpec.
    register_tool(
        ToolSpec(
            name="ideal_gas",
            build_executor=executor_from_artifact,
            build_world_model=IdealGasWorldModel,
            health_check=thermo_health_check,
        )
    )
    register_tool(
        ToolSpec(
            name="van_der_waals",
            build_executor=vdw_executor_from_artifact,
            build_world_model=VanDerWaalsWorldModel,
            health_check=vdw_health_check,
        )
    )
    register_tool(
        ToolSpec(
            name="oscillator",
            build_executor=osc_executor_from_artifact,
            build_world_model=OscillatorWorldModel,
            health_check=osc_health_check,
        )
    )


_register_builtins()
