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


@dataclass
class RouteDecision:
    """Where to run and why."""

    target: str  # "local" or "hpc"
    reason: str


class ComputeRouter:
    """Routes a tool execution to local or HPC based on problem size."""

    def route(
        self, tool_name: str, action: str, params: dict[str, Any]
    ) -> RouteDecision:
        tool_lower = tool_name.lower()

        # Explicit user preference always wins.
        backend = params.get("execution_backend")
        if backend in ("local", "hpc", "remote"):
            return RouteDecision(
                target="hpc" if backend == "remote" else backend,
                reason="user_preference",
            )

        n_atoms = _extract_n_atoms(params)
        walltime_s = _extract_walltime_seconds(params)

        if tool_lower in _DFT_MD_TOOLS:
            if n_atoms is not None and n_atoms > _DFT_MD_ATOM_THRESHOLD:
                return RouteDecision(
                    target="hpc",
                    reason=f"n_atoms={n_atoms} > {_DFT_MD_ATOM_THRESHOLD} for DFT/MD tool",
                )
            if walltime_s is not None and walltime_s > _WALLTIME_HPC_SECONDS:
                return RouteDecision(
                    target="hpc",
                    reason=f"walltime={walltime_s}s > 1h for DFT/MD tool",
                )
            return RouteDecision(
                target="local", reason="below DFT/MD HPC thresholds"
            )

        if tool_lower in _QC_TOOLS:
            if n_atoms is not None and n_atoms > _QC_ATOM_THRESHOLD:
                return RouteDecision(
                    target="hpc",
                    reason=f"n_atoms={n_atoms} > {_QC_ATOM_THRESHOLD} for QC tool",
                )
            return RouteDecision(
                target="local", reason="below QC HPC threshold"
            )

        return RouteDecision(target="local", reason="default local routing")


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
