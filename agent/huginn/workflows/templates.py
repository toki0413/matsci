"""Preset workflow templates for common material science calculations.

Each template defines a multi-stage computational pipeline that the Agent
can execute with minimal user input.
"""

from __future__ import annotations

from huginn.workflows.engine import ComputationalStage, ValidationRule, RetryPolicy


def standard_dft_workflow(structure_path: str, engine: str = "vasp") -> list[ComputationalStage]:
    """Standard DFT workflow: relax → SCF → properties.
    
    Args:
        structure_path: Path to initial structure file
        engine: Computational engine (vasp, qe, etc.)
    """
    return [
        ComputationalStage(
            id="relax",
            name="Structure Relaxation",
            tool=f"{engine}_tool",
            tool_input={
                "action": "relax",
                "structure_file": structure_path,
                "params": {"ISIF": 3, "IBRION": 2, "EDIFFG": -0.01}
            },
            validation=ValidationRule(check="convergence"),
            retry_policy=RetryPolicy(max_retries=2, retry_on=["convergence_fail"]),
        ),
        ComputationalStage(
            id="scf",
            name="Self-Consistent Field",
            tool=f"{engine}_tool",
            tool_input={
                "action": "scf",
                "structure_file": "${relax.output_structure}",
                "params": {"ISTART": 1}
            },
            dependencies=["relax"],
            validation=ValidationRule(check="convergence"),
        ),
        ComputationalStage(
            id="band",
            name="Band Structure",
            tool=f"{engine}_tool",
            tool_input={
                "action": "band",
                "structure_file": "${relax.output_structure}",
                "charge_density": "${scf.chgcar}",
                "params": {"ICHARG": 11, "LORBIT": 11}
            },
            dependencies=["scf"],
            validation=ValidationRule(check="energy_sign"),
        ),
        ComputationalStage(
            id="dos",
            name="Density of States",
            tool=f"{engine}_tool",
            tool_input={
                "action": "dos",
                "structure_file": "${relax.output_structure}",
                "charge_density": "${scf.chgcar}",
            },
            dependencies=["scf"],
        ),
    ]


def aimd_workflow(structure_path: str, temperature: float = 300.0, 
                  timestep_fs: float = 1.0, steps: int = 10000) -> list[ComputationalStage]:
    """Ab-initio MD workflow: minimization → equilibration → production.
    
    Args:
        structure_path: Path to initial structure
        temperature: Target temperature in K
        timestep_fs: Time step in femtoseconds
        steps: Number of production steps
    """
    return [
        ComputationalStage(
            id="minimize",
            name="Energy Minimization",
            tool="vasp_tool",
            tool_input={
                "action": "relax",
                "structure_file": structure_path,
                "params": {"ISIF": 2, "IBRION": 2, "NSW": 100}
            },
            validation=ValidationRule(check="force_convergence", threshold=0.05),
        ),
        ComputationalStage(
            id="equil_nvt",
            name="NVT Equilibration",
            tool="vasp_tool",
            tool_input={
                "action": "md",
                "structure_file": "${minimize.output_structure}",
                "params": {
                    "MDALGO": 2,  # Nosé-Hoover
                    "SMASS": 0,
                    "TEBEG": 0,
                    "TEEND": temperature,
                    "NSW": 1000,
                    "POTIM": timestep_fs,
                }
            },
            dependencies=["minimize"],
            validation=ValidationRule(check="convergence"),
        ),
        ComputationalStage(
            id="production",
            name="NVE Production",
            tool="vasp_tool",
            tool_input={
                "action": "md",
                "structure_file": "${equil_nvt.output_structure}",
                "params": {
                    "MDALGO": 0,  # NVE
                    "NSW": steps,
                    "POTIM": timestep_fs,
                }
            },
            dependencies=["equil_nvt"],
            validation=ValidationRule(check="energy_conservation"),
            retry_policy=RetryPolicy(max_retries=1, retry_on=["convergence_fail"]),
        ),
    ]


def defect_workflow(pristine_path: str, defect_type: str, 
                    site_index: int | None = None) -> list[ComputationalStage]:
    """Defect calculation workflow.
    
    Args:
        pristine_path: Path to pristine structure
        defect_type: "vacancy", "substitution", or "interstitial"
        site_index: Site index for defect (auto-detected if None)
    """
    return [
        ComputationalStage(
            id="pristine_relax",
            name="Pristine Relaxation",
            tool="vasp_tool",
            tool_input={
                "action": "relax",
                "structure_file": pristine_path,
            },
            validation=ValidationRule(check="convergence"),
        ),
        ComputationalStage(
            id="create_defect",
            name="Create Defect Structure",
            tool="structure_tool",
            tool_input={
                "action": "create_defect",
                "file_path": pristine_path,
                "defect_type": defect_type,
                "site_index": site_index,
            },
            dependencies=["pristine_relax"],
        ),
        ComputationalStage(
            id="defect_relax",
            name="Defect Relaxation",
            tool="vasp_tool",
            tool_input={
                "action": "relax",
                "structure_file": "${create_defect.output_path}",
            },
            dependencies=["create_defect"],
            validation=ValidationRule(check="convergence"),
        ),
        ComputationalStage(
            id="formation_energy",
            name="Formation Energy Calculation",
            tool="vasp_tool",
            tool_input={
                "action": "formation_energy",
                "pristine_structure": "${pristine_relax.output_structure}",
                "defect_structure": "${defect_relax.output_structure}",
            },
            dependencies=["pristine_relax", "defect_relax"],
        ),
    ]


def surface_workflow(bulk_path: str, miller_index: str = "111", 
                     layers: int = 6, vacuum: float = 15.0) -> list[ComputationalStage]:
    """Surface calculation workflow.
    
    Args:
        bulk_path: Path to bulk structure
        miller_index: Miller index of surface (e.g., "111", "100")
        layers: Number of atomic layers
        vacuum: Vacuum thickness in Å
    """
    return [
        ComputationalStage(
            id="bulk_relax",
            name="Bulk Relaxation",
            tool="vasp_tool",
            tool_input={
                "action": "relax",
                "structure_file": bulk_path,
            },
            validation=ValidationRule(check="convergence"),
        ),
        ComputationalStage(
            id="cut_surface",
            name="Cut Surface",
            tool="structure_tool",
            tool_input={
                "action": "cut_surface",
                "file_path": "${bulk_relax.output_structure}",
                "miller_index": miller_index,
                "layers": layers,
                "vacuum": vacuum,
            },
            dependencies=["bulk_relax"],
        ),
        ComputationalStage(
            id="surface_relax",
            name="Surface Relaxation",
            tool="vasp_tool",
            tool_input={
                "action": "relax",
                "structure_file": "${cut_surface.output_path}",
                "params": {"ISIF": 2}  # Only relax ions, keep cell fixed
            },
            dependencies=["cut_surface"],
            validation=ValidationRule(check="convergence"),
        ),
    ]


def ml_potential_workflow(training_structures: list[str], 
                          potential_type: str = "nep") -> list[ComputationalStage]:
    """ML potential training workflow.
    
    Args:
        training_structures: List of DFT-calculated structure paths
        potential_type: "nep", "snap", "gap", or "ace"
    """
    return [
        ComputationalStage(
            id="prepare_dataset",
            name="Prepare Training Dataset",
            tool="potential_tool",
            tool_input={
                "action": "prepare_dataset",
                "structures": training_structures,
                "format": "extxyz",
            },
        ),
        ComputationalStage(
            id="train",
            name="Train Potential",
            tool="potential_tool",
            tool_input={
                "action": "train",
                "dataset": "${prepare_dataset.dataset_path}",
                "potential_type": potential_type,
            },
            dependencies=["prepare_dataset"],
            validation=ValidationRule(check="convergence"),
            retry_policy=RetryPolicy(max_retries=1),
        ),
        ComputationalStage(
            id="validate",
            name="Validate Potential",
            tool="potential_tool",
            tool_input={
                "action": "validate",
                "potential": "${train.potential_path}",
                "test_set": "${prepare_dataset.test_path}",
            },
            dependencies=["train"],
        ),
    ]


def symbolic_verify_workflow(
    verify_type: str = "derivative",
    expression: str | None = None,
    symbols: list[str] | None = None,
    variable: str | None = None,
    free_energy: str | None = None,
    matrix: list[list[str]] | None = None,
    equations: list[str] | None = None,
    lean_project: str = "HuginnLean",
) -> list[ComputationalStage]:
    """Symbolic derivation followed by Lean formal verification.

    Executes a two-stage pipeline:
      1. symbolic_derive — run SymPy to obtain an analytical result
      2. lean_verify     — feed the result into LeanTool.auto_verify

    Supported verify_type values (align with both symbolic_math_tool
    and lean_tool auto_verify_action):
      derivative | constitutive | weak_form | eigenvalue | tensor_ops | solve
    """
    # Map unified verify_type to symbolic_math_tool action.
    # LeanTool auto_verify_action accepts the same names directly.
    symbolic_action = verify_type
    if symbolic_action == "derivative":
        symbolic_action = "differentiate"

    tool_input: dict[str, Any] = {"action": symbolic_action}
    if expression is not None:
        tool_input["expression"] = expression
    if symbols is not None:
        tool_input["symbols"] = symbols
    if variable is not None:
        tool_input["variable"] = variable
    if free_energy is not None:
        tool_input["free_energy"] = free_energy
    if matrix is not None:
        tool_input["matrix"] = matrix
    if equations is not None:
        tool_input["equations"] = equations

    return [
        ComputationalStage(
            id="symbolic_derive",
            name="Symbolic Derivation",
            tool="symbolic_math_tool",
            tool_input=tool_input,
            validation=ValidationRule(check="custom", custom_fn="has_result"),
        ),
        ComputationalStage(
            id="lean_verify",
            name="Lean Formal Verification",
            tool="lean_tool",
            tool_input={
                "action": "auto_verify",
                "auto_verify_action": verify_type,
                "symbolic_result": "${symbolic_derive}",
            },
            dependencies=["symbolic_derive"],
            validation=ValidationRule(check="custom", custom_fn="lean_compiles"),
        ),
    ]


def tensor_calculus_verify_workflow(
    voigt_vector: list[float],
    operation: str = "invariants",
    tensor_type: str = "stress",
    rotation_matrix: list[list[float]] | None = None,
) -> list[ComputationalStage]:
    """Tensor calculus derivation followed by Lean formal verification.

    Args:
        voigt_vector: 6 components for 2nd-order or 21 for 4th-order stiffness
        operation: invariants | deviatoric | principal | von_mises | rotate | apply_to_strain
        tensor_type: stress | strain | stiffness | compliance
        rotation_matrix: Optional 3×3 rotation matrix
    """
    tool_input: dict[str, Any] = {
        "action": "tensor_calculus",
        "expression": operation,
        "tensor_type": tensor_type,
        "voigt_vector": voigt_vector,
    }
    if rotation_matrix is not None:
        tool_input["rotation_matrix"] = rotation_matrix

    return [
        ComputationalStage(
            id="symbolic_derive",
            name="Tensor Calculus Derivation",
            tool="symbolic_math_tool",
            tool_input=tool_input,
            validation=ValidationRule(check="custom", custom_fn="has_result"),
        ),
        ComputationalStage(
            id="lean_verify",
            name="Lean Formal Verification",
            tool="lean_tool",
            tool_input={
                "action": "auto_verify",
                "auto_verify_action": "tensor_calculus",
                "symbolic_result": "${symbolic_derive}",
            },
            dependencies=["symbolic_derive"],
            validation=ValidationRule(check="custom", custom_fn="lean_compiles"),
        ),
    ]


def fem_verify_workflow(
    element_type: str = "bar",
    symbols: list[str] | None = None,
    expression: str | None = None,
) -> list[ComputationalStage]:
    """FEM weak-form derivation and element matrix assembly with Lean verification.

    Args:
        element_type: bar | poisson_tri | elasticity_tri | linear_elasticity | heat_conduction
        symbols: Symbol names (e.g. ["u", "v", "x", "E", "A", "h"])
        expression: Extra expression string (e.g. element type for assemble_element_matrix)
    """
    tool_input: dict[str, Any] = {
        "action": "weak_form",
        "target": element_type,
    }
    if symbols is not None:
        tool_input["symbols"] = symbols
    if expression is not None:
        tool_input["expression"] = expression

    return [
        ComputationalStage(
            id="symbolic_derive",
            name="FEM Weak Form Derivation",
            tool="symbolic_math_tool",
            tool_input=tool_input,
            validation=ValidationRule(check="custom", custom_fn="has_result"),
        ),
        ComputationalStage(
            id="lean_verify",
            name="Lean Formal Verification",
            tool="lean_tool",
            tool_input={
                "action": "auto_verify",
                "auto_verify_action": "fem",
                "symbolic_result": "${symbolic_derive}",
            },
            dependencies=["symbolic_derive"],
            validation=ValidationRule(check="custom", custom_fn="lean_compiles"),
        ),
    ]


def la_verify_workflow(
    target: str = "lu_decompose",
    matrix: list[list[str]] | None = None,
    expression: str | None = None,
    symbols: list[str] | None = None,
) -> list[ComputationalStage]:
    """Linear algebra computation followed by Lean formal verification.

    Args:
        target: lu_decompose | cholesky | jacobi_solve | cg_solve | mat_vec_mul | cond_number
        matrix: Square matrix as nested lists of strings
        expression: Optional vector or extra expression (comma-separated for vectors)
        symbols: Symbol names for parsing
    """
    tool_input: dict[str, Any] = {
        "action": "linear_algebra",
        "target": target,
    }
    if matrix is not None:
        tool_input["matrix"] = matrix
    if expression is not None:
        tool_input["expression"] = expression
    if symbols is not None:
        tool_input["symbols"] = symbols

    return [
        ComputationalStage(
            id="symbolic_derive",
            name="Linear Algebra Computation",
            tool="symbolic_math_tool",
            tool_input=tool_input,
            validation=ValidationRule(check="custom", custom_fn="has_result"),
        ),
        ComputationalStage(
            id="lean_verify",
            name="Lean Formal Verification",
            tool="lean_tool",
            tool_input={
                "action": "auto_verify",
                "auto_verify_action": "linear_algebra",
                "symbolic_result": "${symbolic_derive}",
            },
            dependencies=["symbolic_derive"],
            validation=ValidationRule(check="custom", custom_fn="lean_compiles"),
        ),
    ]


def dft_verify_workflow(
    target: str = "fermi_energy",
    expression: str | None = None,
) -> list[ComputationalStage]:
    """DFT computation followed by Lean formal verification.

    Args:
        target: fermi_energy | free_electron_dos | particle_in_box | tight_binding_band | lda_xc_energy
        expression: Parameter string like \"n=0.05\" or \"L=10.0,N=3\"
    """
    tool_input: dict[str, Any] = {
        "action": "dft",
        "target": target,
    }
    if expression is not None:
        tool_input["expression"] = expression

    return [
        ComputationalStage(
            id="symbolic_derive",
            name="DFT Computation",
            tool="symbolic_math_tool",
            tool_input=tool_input,
            validation=ValidationRule(check="custom", custom_fn="has_result"),
        ),
        ComputationalStage(
            id="lean_verify",
            name="Lean Formal Verification",
            tool="lean_tool",
            tool_input={
                "action": "auto_verify",
                "auto_verify_action": "dft",
                "symbolic_result": "${symbolic_derive}",
            },
            dependencies=["symbolic_derive"],
            validation=ValidationRule(check="custom", custom_fn="lean_compiles"),
        ),
    ]


def thermo_verify_workflow(
    target: str = "ideal_gas",
    expression: str | None = None,
) -> list[ComputationalStage]:
    """Thermodynamics computation followed by Lean formal verification.

    Args:
        target: ideal_gas | van_der_waals | helmholtz_energy | gibbs_energy
                | chemical_potential | clausius_clapeyron | partition_function
        expression: Parameter string like \"n=1.0,T=273.15,V=0.022414\"
    """
    tool_input: dict[str, Any] = {
        "action": "thermodynamics",
        "target": target,
    }
    if expression is not None:
        tool_input["expression"] = expression

    return [
        ComputationalStage(
            id="symbolic_derive",
            name="Thermodynamics Computation",
            tool="symbolic_math_tool",
            tool_input=tool_input,
            validation=ValidationRule(check="custom", custom_fn="has_result"),
        ),
        ComputationalStage(
            id="lean_verify",
            name="Lean Formal Verification",
            tool="lean_tool",
            tool_input={
                "action": "auto_verify",
                "auto_verify_action": "thermodynamics",
                "symbolic_result": "${symbolic_derive}",
            },
            dependencies=["symbolic_derive"],
            validation=ValidationRule(check="custom", custom_fn="lean_compiles"),
        ),
    ]


def probability_verify_workflow(
    target: str = "normal_pdf",
    expression: str | None = None,
    equations: list[str] | None = None,
) -> list[ComputationalStage]:
    """Probability computation followed by Lean formal verification.

    Args:
        target: normal_pdf | normal_cdf | gp_kernel | monte_carlo_integral | bayesian_update_normal
        expression: Parameter string like \"mu=0.0,sigma=1.0,x=0.0\"
        equations: Optional list with kernel type for gp_kernel (e.g. [\"rbf\"])
    """
    tool_input: dict[str, Any] = {
        "action": "probability",
        "target": target,
    }
    if expression is not None:
        tool_input["expression"] = expression
    if equations is not None:
        tool_input["equations"] = equations

    return [
        ComputationalStage(
            id="symbolic_derive",
            name="Probability Computation",
            tool="symbolic_math_tool",
            tool_input=tool_input,
            validation=ValidationRule(check="custom", custom_fn="has_result"),
        ),
        ComputationalStage(
            id="lean_verify",
            name="Lean Formal Verification",
            tool="lean_tool",
            tool_input={
                "action": "auto_verify",
                "auto_verify_action": "probability",
                "symbolic_result": "${symbolic_derive}",
            },
            dependencies=["symbolic_derive"],
            validation=ValidationRule(check="custom", custom_fn="lean_compiles"),
        ),
    ]


# Registry of all templates
WORKFLOW_TEMPLATES = {
    "standard_dft": standard_dft_workflow,
    "aimd": aimd_workflow,
    "defect": defect_workflow,
    "surface": surface_workflow,
    "ml_potential": ml_potential_workflow,
    "symbolic_verify": symbolic_verify_workflow,
    "tensor_calculus_verify": tensor_calculus_verify_workflow,
    "fem_verify": fem_verify_workflow,
    "la_verify": la_verify_workflow,
    "dft_verify": dft_verify_workflow,
    "thermo_verify": thermo_verify_workflow,
    "probability_verify": probability_verify_workflow,
}


def list_templates() -> list[str]:
    """List available workflow templates."""
    return list(WORKFLOW_TEMPLATES.keys())


def get_template(name: str):
    """Get a workflow template by name."""
    return WORKFLOW_TEMPLATES.get(name)


def register_template(name: str, template_fn):
    """Register a workflow template."""
    WORKFLOW_TEMPLATES[name] = template_fn


# Import and register quantum chemistry templates from Sobko knowledge base
# This registers: wavefunction_analysis, reactivity_prediction, weak_interaction,
# excited_state, charge_analysis
try:
    from huginn.workflows import templates_qc
except ImportError:
    pass  # templates_qc not available
