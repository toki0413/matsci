"""ComputeRouter — auto-selects local vs HPC execution for computational tasks.

Symmetric with ModelRouter: ModelRouter routes LLM calls, ComputeRouter routes
compute jobs. Plugged into ExecutionOrchestrator._execute_stage() before the
tool call, sets params["execution_target"] so tools know where to run.

Routing factors:
  - config.execution_backend user preference (local/remote/auto)
  - tool_name + action (e.g. vasp_tool relax vs scf)
  - n_atoms from params (if available)
  - estimated walltime from params

When target is "hpc", the tool should use HPCClient to submit. When "local",
run via sandbox.run() as before.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)

Target = Literal["local", "hpc"]


@dataclass
class ComputeRoute:
    """Result of a routing decision."""
    target: Target
    reason: str
    estimated_walltime_hours: float = 0.0
    estimated_gpus: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "reason": self.reason,
            "estimated_walltime_hours": self.estimated_walltime_hours,
            "estimated_gpus": self.estimated_gpus,
        }


# ── Routing rules ────────────────────────────────────────────────────────────
# Each rule: (tool_name, action_pattern) -> (max_local_atoms, max_local_walltime_h, gpu_required)
# If n_atoms or walltime exceeds local thresholds, route to HPC.

_ROUTING_RULES: dict[tuple[str, str], tuple[int, float, bool]] = {
    # VASP: scf/dos/band are quick, relax/md/phonon are heavy
    ("vasp_tool", "scf"): (100, 2.0, False),
    ("vasp_tool", "dos"): (100, 1.0, False),
    ("vasp_tool", "band"): (100, 1.0, False),
    ("vasp_tool", "relax"): (50, 4.0, False),
    ("vasp_tool", "md"): (20, 1.0, False),
    ("vasp_tool", "phonon"): (50, 8.0, False),
    # LAMMPS: MD is parallel-friendly, large systems go to HPC
    ("lammps_tool", "run"): (1000, 1.0, False),
    ("lammps_tool", "minimize"): (500, 0.5, False),
    ("lammps_tool", "equilibrate"): (1000, 2.0, False),
    # QE/CP2K/GROMACS: generally heavy, lower local thresholds
    ("qe_tool", "scf"): (80, 2.0, False),
    ("qe_tool", "relax"): (40, 4.0, False),
    ("cp2k_tool", "energy"): (80, 2.0, False),
    ("cp2k_tool", "optimize"): (40, 4.0, False),
    ("gromacs_tool", "md"): (5000, 1.0, False),
    # Abaqus/FEniCS/Elmer: FEM, mesh size matters more than atoms
    ("abaqus_tool", "*"): (0, 0.5, False),  # FEM almost always needs HPC
    ("fenics_tool", "*"): (0, 0.5, False),
    # Gaussian: small molecules, usually fine locally
    ("gaussian_tool", "*"): (0, 1.0, False),
}

# Default rule for unknown tools
_DEFAULT_RULE: tuple[int, float, bool] = (50, 1.0, False)


def _match_rule(tool_name: str, action: str) -> tuple[int, float, bool]:
    """Find the matching routing rule, falling back to wildcard action."""
    key = (tool_name, action)
    if key in _ROUTING_RULES:
        return _ROUTING_RULES[key]
    # Try wildcard action
    key_wc = (tool_name, "*")
    if key_wc in _ROUTING_RULES:
        return _ROUTING_RULES[key_wc]
    return _DEFAULT_RULE


class ComputeRouter:
    """Routes compute tasks to local or HPC based on resource estimation.

    Usage:
        router = ComputeRouter(config)
        route = router.route("vasp_tool", "relax", params={"n_atoms": 350})
        if route.target == "hpc":
            # submit via HPCClient
    """

    def __init__(self, config: Any = None):
        """config should have execution_backend, hpc_host, etc."""
        self._config = config
        # Extract user preference
        self._user_pref = "auto"
        if config is not None:
            pref = getattr(config, "execution_backend", "local")
            # "local" / "remote" are explicit; "auto" means router decides
            self._user_pref = pref if pref in ("local", "remote") else "auto"
            self._hpc_host = getattr(config, "hpc_host", None)
        else:
            self._hpc_host = None

    def route(
        self,
        tool_name: str,
        action: str,
        params: dict[str, Any] | None = None,
    ) -> ComputeRoute:
        """Decide where to run this task.

        params may contain:
          - n_atoms: int (atom count, used for DFT/MD)
          - walltime_hours: float (user-specified walltime)
          - queue: str (user-specified queue)
        """
        params = params or {}
        n_atoms = self._extract_n_atoms(tool_name, params)
        walltime = params.get("walltime_hours", 0.0)

        # User explicitly set execution_backend — respect it
        if self._user_pref == "remote":
            return ComputeRoute(
                target="hpc",
                reason="execution_backend=remote (user preference)",
                estimated_walltime_hours=walltime,
            )
        if self._user_pref == "local":
            return ComputeRoute(
                target="local",
                reason="execution_backend=local (user preference)",
                estimated_walltime_hours=walltime,
            )

        # Auto mode: decide based on rules
        max_atoms, max_walltime, gpu_required = _match_rule(tool_name, action)

        # Check GPU requirement
        if gpu_required:
            return ComputeRoute(
                target="hpc",
                reason=f"{tool_name}/{action} requires GPU",
                estimated_walltime_hours=walltime,
                estimated_gpus=1,
            )

        # Check HPC availability
        if not self._hpc_host:
            # No HPC configured, must run locally
            return ComputeRoute(
                target="local",
                reason="no HPC host configured",
                estimated_walltime_hours=walltime,
            )

        # Check atom count threshold
        if n_atoms > max_atoms:
            return ComputeRoute(
                target="hpc",
                reason=f"n_atoms={n_atoms} > local threshold {max_atoms} for {tool_name}/{action}",
                estimated_walltime_hours=walltime,
            )

        # Check walltime threshold
        if walltime > max_walltime:
            return ComputeRoute(
                target="hpc",
                reason=f"walltime={walltime}h > local threshold {max_walltime}h for {tool_name}/{action}",
                estimated_walltime_hours=walltime,
            )

        # Default: run locally
        return ComputeRoute(
            target="local",
            reason=f"within local limits (atoms={n_atoms}/{max_atoms}, walltime={walltime}/{max_walltime}h)",
            estimated_walltime_hours=walltime,
        )

    def _extract_n_atoms(self, tool_name: str, params: dict[str, Any]) -> int:
        """Try to get atom count from params, varies by tool."""
        # Direct field
        if "n_atoms" in params:
            try:
                return int(params["n_atoms"])
            except (ValueError, TypeError):
                pass
        # Structure string — count element lines in XYZ format
        struct = params.get("structure") or params.get("poscar") or params.get("xyz")
        if isinstance(struct, str) and len(struct) > 0:
            lines = struct.strip().split("\n")
            if len(lines) > 2 and lines[0].strip().isdigit():
                # XYZ format: first line is atom count
                try:
                    return int(lines[0].strip())
                except ValueError:
                    pass
            # POSCAR: count element count line (second line)
            if len(lines) > 6:
                counts_line = lines[6].strip().split()
                try:
                    return sum(int(x) for x in counts_line)
                except ValueError:
                    pass
        # Number of atoms in a list
        atoms = params.get("atoms")
        if isinstance(atoms, list):
            return len(atoms)
        return 0

    def route_stage(self, stage: dict[str, Any]) -> ComputeRoute:
        """Convenience: route from a workflow stage dict."""
        return self.route(
            tool_name=stage.get("tool", "unknown"),
            action=stage.get("action", ""),
            params=stage.get("params", {}),
        )
