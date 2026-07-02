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
            # Quantum ESPRESSO rules
            {
                "tools": ["qe_tool"],
                "patterns": ["convergence not achieved", "scf convergence", "electron"],
                "fixes": {"conv_thr": 1e-8, "mixing_beta": 0.4, "mixing_type": "pulay"},
                "description": "Reduce mixing beta and switch to Pulay mixing for SCF",
            },
            {
                "tools": ["qe_tool"],
                "patterns": ["bravais", "k-point", "k point", "mesh"],
                "fixes": {"k_points": "automatic", "k_spacing": "increase"},
                "description": "Switch to automatic k-point generation with coarser spacing",
            },
            {
                "tools": ["qe_tool"],
                "patterns": ["too many bands", "highest band", "occupation"],
                "fixes": {"nbnd": "increase_20pct", "smearing": "gauss", "degauss": 0.01},
                "description": "Add more bands and apply Gaussian smearing",
            },
            {
                "tools": ["qe_tool"],
                "patterns": ["memory", "allocation", "out of memory"],
                "fixes": {"nproc_image": "increase", "nplane": "increase", "ntask_groups": 2},
                "description": "Increase parallel image / plane / task group distribution",
            },
            # CP2K rules
            {
                "tools": ["cp2k_tool"],
                "patterns": ["SCF", "convergence", "not converged", "OT"],
                "fixes": {"OT_MINIMIZER": "DIIS", "OT_N_DIIS": 7, "EPS_SCF": 1e-6},
                "description": "Switch OT minimizer to DIIS with larger history",
            },
            {
                "tools": ["cp2k_tool"],
                "patterns": [" preconditioner", "cholesky", "inverse"],
                "fixes": {"OT_PRECONDITIONER": "FULL_SINGLE_INVERSE", "OT_SCALING": 100},
                "description": "Use full single inverse preconditioner with larger scaling",
            },
            {
                "tools": ["cp2k_tool"],
                "patterns": ["force_eval", "energy", "imaginary frequency"],
                "fixes": {"FORCE_EVAL_METHOD": "FIST", "INTERFACE": "CHARMM"},
                "description": "Fall back to classical force field evaluation",
            },
            # GROMACS rules
            {
                "tools": ["gromacs_tool"],
                "patterns": ["LINCS warning", "constraint", "shake"],
                "fixes": {"dt": "halve", "constraints": "h-bonds", "lincs_iter": 2},
                "description": "Halve timestep, constrain H-bonds, increase LINCS iterations",
            },
            {
                "tools": ["gromacs_tool"],
                "patterns": ["segmentation fault", "crash", "bond"],
                "fixes": {"dt": "halve", "nsteps": "halve", "em_steps": 1000},
                "description": "Halve timestep and run energy minimization first",
            },
            {
                "tools": ["gromacs_tool"],
                "patterns": ["temperature", "thermostat", "NVT", "NPT"],
                "fixes": {"tcoupl": "V-rescale", "tau_t": 0.5, "ref_t": "verify"},
                "description": "Switch to V-rescale thermostat with larger coupling time",
            },
            {
                "tools": ["gromacs_tool"],
                "patterns": ["pressure", "barostat", "density"],
                "fixes": {"pcoupl": "C-rescale", "tau_p": 2.0, "compressibility": 4.5e-5},
                "description": "Use C-rescale barostat with standard water compressibility",
            },
            # FEniCS rules
            {
                "tools": ["fenics_tool"],
                "patterns": ["unable to solve linear system", "singular", "condition number"],
                "fixes": {"linear_solver": "mumps", "lu_solver": True, "mesh_refine": True},
                "description": "Switch to MUMPS direct solver and refine mesh",
            },
            {
                "tools": ["fenics_tool"],
                "patterns": ["Newton", "not converge", "nonlinear"],
                "fixes": {"newton_relaxation": 0.7, "newton_max_iter": 50, "newton_tol": 1e-8},
                "description": "Apply Newton relaxation and increase max iterations",
            },
            {
                "tools": ["fenics_tool"],
                "patterns": ["diverged", "nan", "inf in solution"],
                "fixes": {"mesh_refine": True, "element_order": "increase", "stabilization": "SUPG"},
                "description": "Refine mesh, increase element order, add SUPG stabilization",
            },
            # Elmer rules
            {
                "tools": ["elmer_tool"],
                "patterns": ["Iteration", "not converge", "linear system"],
                "fixes": {"linear_solver": "BiCGStab", "max_iterations": 1000, "tolerance": 1e-8},
                "description": "Switch to BiCGStab with more iterations and tighter tolerance",
            },
            {
                "tools": ["elmer_tool"],
                "patterns": ["mesh", "element", "Jacobian", "negative"],
                "fixes": {"mesh_quality": "improve", "element_type": "tetra", "refine": True},
                "description": "Improve mesh quality and switch to tetrahedral elements",
            },
            {
                "tools": ["elmer_tool"],
                "patterns": ["coupled", "multi-physics", "interface"],
                "fixes": {"relaxation_factor": 0.5, "max_coupling_iter": 50, "stabilization": True},
                "description": "Add relaxation for coupled multi-physics iterations",
            },
            # COMSOL rules
            {
                "tools": ["comsol_tool"],
                "patterns": ["not converge", "nonlinear", "segregated"],
                "fixes": {"solver_type": "segregated", "relaxation_factor": 0.5, "max_iter": 100},
                "description": "Switch to segregated solver with relaxation",
            },
            {
                "tools": ["comsol_tool"],
                "patterns": ["mesh", "element", "Jacobian", "singular"],
                "fixes": {"mesh_refinement": "increase", "element_type": "quadratic", "quality_threshold": 0.1},
                "description": "Refine mesh and use quadratic elements",
            },
            {
                "tools": ["comsol_tool"],
                "patterns": ["memory", "out of", "allocation"],
                "fixes": {"solver_type": "iterative", "memory_saving": True, "mesh_coarsen": "far_field"},
                "description": "Switch to iterative solver and coarsen far-field mesh",
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
