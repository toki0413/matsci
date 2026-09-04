"""工具注册/交换机制 — M5"同一接口换后端"的软件兑现.

一个物理工具 = (执行器 + 世界模型 + 物理健康门控). 通过 ``get_tool(name)`` 解析,
安装时按制品其 ``tool`` 身份走对应健康门控与后端构建. 换后端 = 换制品, 不换
上层 (PhysicalWorkspace / BehaviorLifecycle / ExecutionGuard).

借鉴 Fable 5.1 那套"46 工具 JSON schema"的**声明式 + 版本化**手法:
- 每个 ``ToolSpec`` 携带机器可读 ``schema`` (物理域 / 状态空间 / 动作 / 观测),
  供 agent 自我路由 — 是 Fable 工具 schema 的松耦合版 (代码而非文字承载).
- 每个 ``ToolSpec`` 声明 ``contract_version``; ``install_tool`` 做**工具级契约握手**,
  制品契约版本必须与工具后端一致, 否则拒绝 — "换后端"走换 ToolSpec/制品,
  而不是原地改写 (microduck releases-swapped-not-patched).

真实 HPC 工具 (VASP/DFT) 未来以同样的 ``ToolSpec`` 注册, 只需把外部/异步执行与
输出解析接进 ``build_executor`` 的 ``_forward`` — 其余层次完全复用.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from huginn.security.actuator_model import SensorModelExecutor
from huginn.security.behavior_lifecycle import (
    BehaviorArtifact,
    BehaviorLifecycle,
    InstallResult,
)
from huginn.security.world_model import WorldModel


class UnknownToolError(Exception):
    """未注册的工具名."""


class ToolContractError(Exception):
    """制品契约版本与工具后端不一致 — 换后端须换 ToolSpec/制品, 不原地改写."""


@dataclass(frozen=True)
class ToolSpec:
    """一个物理工具的后端构建、健康门控、契约与机器可读描述.

    - ``schema``          : 机器可读 JSON 描述 (物理域/状态空间/动作/观测), agent 自我路由.
    - ``contract_version``: 工具级契约版本; 制品安装须与之握手一致, 升级走换后端.
    """

    name: str
    build_executor: Callable[[BehaviorArtifact], SensorModelExecutor]
    build_world_model: Callable[[], WorldModel]
    health_check: Callable[[BehaviorArtifact], bool]
    contract_version: int = 1
    schema: Mapping[str, Any] = field(default_factory=dict)


_TOOLS: dict[str, ToolSpec] = {}


def register_tool(spec: ToolSpec) -> None:
    _TOOLS[spec.name] = spec


def get_tool(name: str) -> ToolSpec:
    try:
        return _TOOLS[name]
    except KeyError:
        raise UnknownToolError(f"unknown tool: {name!r}") from None


def tool_schema(name: str) -> Mapping[str, Any]:
    """某工具的机器可读描述 — agent 据此自我路由 (Fable 46 工具 schema 的等价物)."""
    return _TOOLS[name].schema


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
    """安装某工具的制品, 用该工具自身的物理健康门控做安装门控.

    先做**工具级契约握手**: 制品契约版本须等于该 ToolSpec 的 ``contract_version``,
    不符即拒绝 (换后端 = 换 ToolSpec/制品, 不是原地改写).
    """
    spec = get_tool(tool)
    if artifact.contract_version != spec.contract_version:
        raise ToolContractError(
            f"tool '{tool}' contract mismatch: "
            f"artifact={artifact.contract_version}, spec={spec.contract_version}"
        )
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
            schema={
                "domain": "thermodynamics",
                "space": {"state": ["p", "V", "T", "n"], "action": ["heat", "move"]},
                "observables": ["p", "T"],
                "forward": "pV=nRT, isochoric heat / isothermal move",
            },
        )
    )
    register_tool(
        ToolSpec(
            name="van_der_waals",
            build_executor=vdw_executor_from_artifact,
            build_world_model=VanDerWaalsWorldModel,
            health_check=vdw_health_check,
            schema={
                "domain": "thermodynamics",
                "space": {"state": ["p", "V", "T", "n"], "action": ["heat", "move"]},
                "observables": ["p", "T"],
                "forward": "van der Waals EOS (a, b corrections)",
            },
        )
    )
    register_tool(
        ToolSpec(
            name="oscillator",
            build_executor=osc_executor_from_artifact,
            build_world_model=OscillatorWorldModel,
            health_check=osc_health_check,
            schema={
                "domain": "mechanics",
                "space": {"state": ["x", "v"], "action": ["kick", "displace"]},
                "observables": ["x"],
                "forward": "1-D harmonic oscillator",
            },
        )
    )
    # 外部计算工具占位: 真跑子进程 (compute_adapter). 真实 HPC 工具以同种 ToolSpec 接入.
    from huginn.security.compute_adapter import (
        ShellComputeWorldModel,
        shell_executor_from_artifact,
        shell_health_check,
    )

    register_tool(
        ToolSpec(
            name="external_shell_compute",
            build_executor=shell_executor_from_artifact,
            build_world_model=ShellComputeWorldModel,
            health_check=shell_health_check,
            schema={
                "domain": "external_compute",
                "space": {"state": ["energy"], "action": ["shell_compute"]},
                "observables": ["energy"],
                "backend": "subprocess",
                "forward": "E = n·Cv·T via external process",
            },
        )
    )


_register_builtins()
