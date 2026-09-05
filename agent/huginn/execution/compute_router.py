"""Compute Router — decides where a computation should run (local vs HPC).

Routing is purely heuristic: problem size (n_atoms) and expected walltime
are checked against per-tool thresholds. The user can always override by
setting ``execution_backend`` in params.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# DFT / MD tools — scale O(n^3) or worse, 100 atoms is the typical cutoff
# where a laptop stops being fun.
_DFT_MD_TOOLS = frozenset(
    {
        "vasp",
        "vasp_tool",
        "lammps",
        "lammps_tool",
        "qe",
        "quantum_espresso",
        "qe_tool",
        "cp2k",
        "cp2k_tool",
    }
)

# Quantum chemistry packages — even steeper scaling, lower threshold.
_QC_TOOLS = frozenset(
    {
        "gaussian",
        "gaussian_tool",
        "orca",
        "orca_tool",
    }
)

_DFT_MD_ATOM_THRESHOLD = 100
_QC_ATOM_THRESHOLD = 50
_WALLTIME_HPC_SECONDS = 3600  # 1 hour


@dataclass(frozen=True)
class ComputeToolSpec:
    """每工具计算路由元数据 — 替代硬编码阈值表 (M1 分流审计: 配置驱动).

    - ``scaling``      : dft_md / qc / generic。决定默认标签与兜底策略。
    - ``atom_threshold``: n_atoms 超过即投 hpc; ``None`` = 不看原子数。
    - ``walltime_hpc`` : 墙钟(秒)超过即投 hpc; ``None`` = 不看墙钟。

    默认规格复刻历史硬编码行为 (DFT/MD 100 原子 / 1h, QC 50 原子); 未登记工具
    走 generic (恒 local)。真实阈值最终应对齐 ToolSpec/权限成本 (M2), 这里先立
    数据驱动 + 可注册接缝。
    """

    scaling: str = "generic"
    atom_threshold: int | None = None
    walltime_hpc: int | None = None


_DFT_MD_SPEC = ComputeToolSpec(
    scaling="dft_md", atom_threshold=_DFT_MD_ATOM_THRESHOLD, walltime_hpc=_WALLTIME_HPC_SECONDS
)
_QC_SPEC = ComputeToolSpec(scaling="qc", atom_threshold=_QC_ATOM_THRESHOLD, walltime_hpc=None)
_GENERIC_SPEC = ComputeToolSpec(scaling="generic")

# 默认登记表: 复刻原 _DFT_MD_TOOLS / _QC_TOOLS 的硬编码归属.
_DEFAULT_TOOL_SPECS: dict[str, ComputeToolSpec] = dict.fromkeys(_DFT_MD_TOOLS, _DFT_MD_SPEC)
_DEFAULT_TOOL_SPECS.update(dict.fromkeys(_QC_TOOLS, _QC_SPEC))

# 缩放族 → 路由标签 (保持历史 reason 文案以兼容下游).
_SCALING_LABELS: dict[str, str] = {
    "dft_md": "DFT/MD tool",
    "qc": "QC tool",
    "generic": "tool",
}


@dataclass
class RouteDecision:
    """Where to run and why."""

    target: str  # "local" or "hpc"
    reason: str


class ComputeRouter:
    """Routes a tool execution to local or HPC based on problem size.

    数据驱动: 每工具规格来自 :class:`ComputeToolSpec` (默认复刻历史硬编码), 可经
    构造参数 ``tool_specs`` 或 :meth:`register_tool` 覆盖/新增, 无需改代码即换阈值.
    """

    def __init__(
        self, tool_specs: dict[str, ComputeToolSpec] | None = None
    ) -> None:
        merged = dict(_DEFAULT_TOOL_SPECS)
        if tool_specs:
            for k, v in tool_specs.items():
                merged[k.lower()] = v
        self._tool_specs = merged

    def register_tool(self, tool_name: str, spec: ComputeToolSpec) -> None:
        """按名登记/覆盖某工具的计算路由规格."""
        self._tool_specs[tool_name.lower()] = spec

    def spec_for(self, tool_name: str) -> ComputeToolSpec:
        """某工具的路由规格, 未登记返回 generic."""
        return self._tool_specs.get(tool_name.lower(), _GENERIC_SPEC)

    def route(
        self, tool_name: str, action: str, params: dict[str, Any], *, local_only: bool = False,
    ) -> RouteDecision:
        """路由决策; ``local_only`` (M3 设备私有化) 强制任何远程目标回落到本地.

        Parameters
        ----------
        local_only : 设备私有化开关. True 时 hpc/remote 目标 (无论启发式还是用户
            execution_backend 偏好) 一律改投 local, 数据不离开设备. 默认 False
            (不改变现有行为).
        """
        tool_lower = tool_name.lower()

        # Explicit user preference always wins (除非 local_only 强制本地).
        backend = params.get("execution_backend")
        if backend in ("local", "hpc", "remote", "device"):
            target = {"remote": "hpc", "device": "device"}.get(backend, backend)
            if local_only and target not in ("local", "device"):
                return RouteDecision(
                    target="local",
                    reason=(
                        f"execution_backend '{backend}' overridden "
                        "by execution_privacy=local_only"
                    ),
                )
            return RouteDecision(target=target, reason="user_preference")

        spec = self._tool_specs.get(tool_lower, _GENERIC_SPEC)
        n_atoms = _extract_n_atoms(params)
        walltime_s = _extract_walltime_seconds(params)

        target: str = "local"
        reason = "default local routing"
        if spec.atom_threshold is not None and n_atoms is not None and n_atoms > spec.atom_threshold:
            target, reason = "hpc", (
                f"n_atoms={n_atoms} > {spec.atom_threshold} "
                f"for {_SCALING_LABELS.get(spec.scaling, 'tool')}"
            )
        elif spec.walltime_hpc is not None and walltime_s is not None and walltime_s > spec.walltime_hpc:
            target, reason = "hpc", (
                f"walltime={walltime_s}s > {spec.walltime_hpc}s "
                f"for {_SCALING_LABELS.get(spec.scaling, 'tool')}"
            )
        elif spec.scaling == "dft_md":
            target, reason = "local", "below DFT/MD HPC thresholds"
        elif spec.scaling == "qc":
            target, reason = "local", "below QC HPC threshold"

        # M3 设备私有化: 远程目标强制本地, 数据不出设备.
        if local_only and target not in ("local", "device"):
            return RouteDecision(
                target="local",
                reason="hpc target overridden by execution_privacy=local_only",
            )
        return RouteDecision(target=target, reason=reason)


# ── helpers ─────────────────────────────────────────────────────────


def _extract_n_atoms(params: dict[str, Any]) -> int | None:
    val = params.get("n_atoms")
    if isinstance(val, int):
        return val
    if isinstance(val, str) and val.isdigit():
        return int(val)
    return None


def _extract_walltime_seconds(params: dict[str, Any]) -> float | None:
    val = params.get("walltime")
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        s = val.strip().lower()
        if s.endswith("h"):
            try:
                return float(s[:-1]) * 3600
            except ValueError:
                return None
        if s.endswith("s"):
            s = s[:-1]
        try:
            return float(s)
        except ValueError:
            return None
    return None
