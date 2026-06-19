"""AutoFix Loop — automatically diagnose and fix failed calculations.

When a calculation fails, this module:
  1. Analyzes the error message and output files
  2. Matches against known failure patterns
  3. Applies heuristic fixes to input parameters
  4. Returns the fixed parameters for retry

This closes the loop: execute → fail → diagnose → fix → retry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class AutoFixLoop:
    """Automatic failure diagnosis and repair for computational calculations.

    Usage:
        fixer = AutoFixLoop()
        fixed_params = fixer.apply_fix(
            tool_name="vasp_tool",
            error="ZBRENT: fatal error in bracketing",
            current_params={"ALGO": "Fast", "NELM": 60}
        )
        if fixed_params:
            # Retry with fixed parameters
            result = vasp_tool.run(**fixed_params)
    """

    def __init__(self, rules_path: str | None = None):
        self.rules_path = Path(rules_path) if rules_path else None
        self._rules: list[dict[str, Any]] = self._load_builtin_rules()

    def _load_builtin_rules(self) -> list[dict[str, Any]]:
        """Built-in heuristic fix rules."""
        return [
            # VASP rules
            {
                "tools": ["vasp_tool"],
                "patterns": [
                    "ZBRENT",
                    "EDDDAV",
                    "scf convergence",
                    "electronic convergence",
                ],
                "fixes": {"ALGO": "Normal", "NELMIN": 6, "ISMEAR": 0, "SIGMA": 0.05},
                "description": "Switch to more robust SCF algorithm",
            },
            {
                "tools": ["vasp_tool"],
                "patterns": ["ionic", "relaxation", "too many steps"],
                "fixes": {"IBRION": 2, "POTIM": 0.1, "NSW": 200},
                "description": "Adjust ionic relaxation parameters",
            },
            {
                "tools": ["vasp_tool"],
                "patterns": ["memory", "out of memory", "allocation"],
                "fixes": {"NCORE": 4, "KPAR": 2},
                "description": "Increase parallelization to reduce per-core memory",
            },
            {
                "tools": ["vasp_tool"],
                "patterns": [" POTIM ", "step size"],
                "fixes": {"POTIM": 0.05},
                "description": "Reduce ionic step size",
            },
            {
                "tools": ["vasp_tool"],
                "patterns": ["ISIF", "stress tensor"],
                "fixes": {"ISIF": 2},
                "description": "Relax ions only, fix cell shape/volume",
            },
            # Gaussian rules
            {
                "tools": ["gaussian_tool"],
                "patterns": ["scf", "convergence", "Convergence failure"],
                "fixes": {"scf": "xqc", "integral": "ultrafine"},
                "description": "Enable SCF=XQC and ultrafine grid",
            },
            {
                "tools": ["gaussian_tool"],
                "patterns": ["optimization", "Opt", "stationary point"],
                "fixes": {"opt": "calcfc", "maxcycle": 200},
                "description": "Calculate force constants at start, increase max cycles",
            },
            {
                "tools": ["gaussian_tool"],
                "patterns": ["basis set", "diffuse", "Missing basis"],
                "fixes": {"basis_check": True, "diffuse": "add_for_anions"},
                "description": "Check basis set completeness",
            },
            # LAMMPS rules
            {
                "tools": ["lammps_tool"],
                "patterns": ["lost atoms"],
                "fixes": {"timestep": "halve", "neighbor_skin": "increase"},
                "description": "Reduce timestep and increase neighbor skin",
            },
            {
                "tools": ["lammps_tool"],
                "patterns": ["bond", "angle", "improper"],
                "fixes": {"fix_shake": True, "bond_style_check": True},
                "description": "Apply SHAKE constraints and verify bond style",
            },
            {
                "tools": ["lammps_tool"],
                "patterns": ["thermo", "temperature", "T = "],
                "fixes": {"fix_nvt_damping": "increase", "timestep": "reduce"},
                "description": "Adjust thermostat damping and reduce timestep",
            },
            # ABAQUS rules
            {
                "tools": ["abaqus_tool"],
                "patterns": ["too many attempts", "convergence", "equilibrium"],
                "fixes": {"increment_size": "halve", "line_search": "enable"},
                "description": "Reduce increment size and enable line search",
            },
            {
                "tools": ["abaqus_tool"],
                "patterns": ["negative eigenvalues", "buckling"],
                "fixes": {"nlgeom": True, "stabilization": "enable"},
                "description": "Enable geometric nonlinearity and stabilization",
            },
            {
                "tools": ["abaqus_tool"],
                "patterns": ["excessive distortion", "element distortion"],
                "fixes": {"element_type": "C3D10", "mesh_refinement": "increase"},
                "description": "Switch to higher-order elements and refine mesh",
            },
            # OpenFOAM rules
            {
                "tools": ["openfoam_tool"],
                "patterns": ["floating point exception", "SIGFPE"],
                "fixes": {"boundary_check": True, "initial_fields": "verify"},
                "description": "Check boundary conditions and initial field consistency",
            },
            {
                "tools": ["openfoam_tool"],
                "patterns": ["continuity error", "mass imbalance"],
                "fixes": {"under_relaxation_p": 0.2, "under_relaxation_U": 0.5},
                "description": "Reduce under-relaxation factors",
            },
            {
                "tools": ["openfoam_tool"],
                "patterns": ["divergence", "residual", "not converging"],
                "fixes": {
                    "under_relaxation": "reduce_all",
                    "nNonOrthogonalCorrectors": 2,
                },
                "description": "Reduce under-relaxation and add non-orthogonal correctors",
            },
            # Generic rules
            {
                "tools": ["*"],
                "patterns": ["timeout", "time limit", "walltime"],
                "fixes": {"walltime_hours": "double", "checkpoint": True},
                "description": "Double walltime and enable checkpointing",
            },
            {
                "tools": ["*"],
                "patterns": ["permission", "access denied"],
                "fixes": {"permissions_check": True, "path_verify": True},
                "description": "Check file permissions and paths",
            },
            {
                "tools": ["*"],
                "patterns": ["file not found", "missing", "no such file"],
                "fixes": {"file_check": True, "dependency_check": True},
                "description": "Verify all required input files exist",
            },
        ]

    def apply_fix(
        self,
        tool_name: str,
        error: str,
        current_params: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Apply the best matching fix rule to the current parameters.

        Returns:
            New parameters dict if a fix was applied, None otherwise.
        """
        error_lower = error.lower()
        best_rule = None
        best_score = 0

        for rule in self._rules:
            # Check tool match
            if not self._tool_matches(rule["tools"], tool_name):
                continue

            # Score pattern matches
            score = sum(1 for pat in rule["patterns"] if pat.lower() in error_lower)
            if score > best_score:
                best_score = score
                best_rule = rule

        if best_rule is None or best_score == 0:
            return None

        # Apply fixes
        new_params = dict(current_params)
        for key, val in best_rule["fixes"].items():
            if val == "halve" and isinstance(current_params.get(key), (int, float)):
                new_params[key] = current_params[key] / 2
            elif val == "double" and isinstance(current_params.get(key), (int, float)):
                new_params[key] = current_params[key] * 2
            elif val == "increase" and isinstance(
                current_params.get(key), (int, float)
            ):
                new_params[key] = current_params[key] * 1.5
            else:
                new_params[key] = val

        # Record the fix
        new_params["__auto_fix"] = best_rule["description"]
        new_params["__auto_fix_patterns_matched"] = best_score

        return new_params

    def _tool_matches(self, rule_tools: list[str], tool_name: str) -> bool:
        """Check if a tool name matches the rule's tool list."""
        if "*" in rule_tools:
            return True
        return any(t.lower() == tool_name.lower() for t in rule_tools)

    def add_rule(self, rule: dict[str, Any]) -> None:
        """Add a custom fix rule at runtime."""
        self._rules.append(rule)

    def list_rules(self, tool_name: str | None = None) -> list[dict[str, Any]]:
        """List all rules, optionally filtered by tool."""
        if tool_name is None:
            return list(self._rules)
        return [r for r in self._rules if self._tool_matches(r["tools"], tool_name)]
