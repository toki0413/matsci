"""Preset workflow templates for common material science calculations.

Each template defines a multi-stage computational pipeline that the Agent
can execute with minimal user input.
"""

from __future__ import annotations

from typing import Any

from huginn.workflows.engine import ComputationalStage, RetryPolicy, ValidationRule


def _dft_relax_stage(structure_path: str, engine: str) -> ComputationalStage:
    """relax 阶段: 结构弛豫."""
    return ComputationalStage(
        id="relax",
        name="Structure Relaxation",
        tool=f"{engine}_tool",
        tool_input={
            "action": "relax",
            "structure_file": structure_path,
            "params": {"ISIF": 3, "IBRION": 2, "EDIFFG": -0.01},
        },
        validation=ValidationRule(check="convergence"),
        retry_policy=RetryPolicy(max_retries=2, retry_on=["convergence_fail"]),
    )


def _dft_scf_stage(engine: str) -> ComputationalStage:
    """scf 阶段: 自洽场."""
    return ComputationalStage(
        id="scf",
        name="Self-Consistent Field",
        tool=f"{engine}_tool",
        tool_input={
            "action": "scf",
            "structure_file": "${relax.output_structure}",
            "params": {"ISTART": 1},
        },
        dependencies=["relax"],
        validation=ValidationRule(check="convergence"),
    )


def _dft_band_stage(engine: str) -> ComputationalStage:
    """band 阶段: 能带结构."""
    return ComputationalStage(
        id="band",
        name="Band Structure",
        tool=f"{engine}_tool",
        tool_input={
            "action": "band",
            "structure_file": "${relax.output_structure}",
            "charge_density": "${scf.chgcar}",
            "params": {"ICHARG": 11, "LORBIT": 11},
        },
        dependencies=["scf"],
        validation=ValidationRule(check="energy_sign"),
    )


def _dft_dos_stage(engine: str) -> ComputationalStage:
    """dos 阶段: 态密度."""
    return ComputationalStage(
        id="dos",
        name="Density of States",
        tool=f"{engine}_tool",
        tool_input={
            "action": "dos",
            "structure_file": "${relax.output_structure}",
            "charge_density": "${scf.chgcar}",
        },
        dependencies=["scf"],
    )


def standard_dft_workflow(
    structure_path: str, engine: str = "vasp"
) -> list[ComputationalStage]:
    """Standard DFT workflow: relax → SCF → properties.

    Args:
        structure_path: Path to initial structure file
        engine: Computational engine (vasp, qe, etc.)
    """
    return [
        _dft_relax_stage(structure_path, engine),
        _dft_scf_stage(engine),
        _dft_band_stage(engine),
        _dft_dos_stage(engine),
    ]


def _aimd_minimize_stage(structure_path: str) -> ComputationalStage:
    """minimize 阶段: 能量最小化."""
    return ComputationalStage(
        id="minimize",
        name="Energy Minimization",
        tool="vasp_tool",
        tool_input={
            "action": "relax",
            "structure_file": structure_path,
            "params": {"ISIF": 2, "IBRION": 2, "NSW": 100},
        },
        validation=ValidationRule(check="force_convergence", threshold=0.05),
    )


def _aimd_equil_stage(temperature: float, timestep_fs: float) -> ComputationalStage:
    """equil_nvt 阶段: NVT 平衡."""
    return ComputationalStage(
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
            },
        },
        dependencies=["minimize"],
        validation=ValidationRule(check="convergence"),
    )


def _aimd_production_stage(steps: int, timestep_fs: float) -> ComputationalStage:
    """production 阶段: NVE 采样."""
    return ComputationalStage(
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
            },
        },
        dependencies=["equil_nvt"],
        validation=ValidationRule(check="energy_conservation"),
        retry_policy=RetryPolicy(max_retries=1, retry_on=["convergence_fail"]),
    )


def aimd_workflow(
    structure_path: str,
    temperature: float = 300.0,
    timestep_fs: float = 1.0,
    steps: int = 10000,
) -> list[ComputationalStage]:
    """Ab-initio MD workflow: minimization → equilibration → production.

    Args:
        structure_path: Path to initial structure
        temperature: Target temperature in K
        timestep_fs: Time step in femtoseconds
        steps: Number of production steps
    """
    return [
        _aimd_minimize_stage(structure_path),
        _aimd_equil_stage(temperature, timestep_fs),
        _aimd_production_stage(steps, timestep_fs),
    ]


def _defect_pristine_stage(pristine_path: str) -> ComputationalStage:
    """pristine_relax 阶段: 完美晶格弛豫."""
    return ComputationalStage(
        id="pristine_relax",
        name="Pristine Relaxation",
        tool="vasp_tool",
        tool_input={
            "action": "relax",
            "structure_file": pristine_path,
        },
        validation=ValidationRule(check="convergence"),
    )


def _defect_create_stage(
    pristine_path: str, defect_type: str, site_index: int | None
) -> ComputationalStage:
    """create_defect 阶段: 构造缺陷结构."""
    return ComputationalStage(
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
    )


def _defect_relax_stage() -> ComputationalStage:
    """defect_relax 阶段: 缺陷结构弛豫."""
    return ComputationalStage(
        id="defect_relax",
        name="Defect Relaxation",
        tool="vasp_tool",
        tool_input={
            "action": "relax",
            "structure_file": "${create_defect.output_path}",
        },
        dependencies=["create_defect"],
        validation=ValidationRule(check="convergence"),
    )


def _defect_formation_energy_stage() -> ComputationalStage:
    """formation_energy 阶段: 形成能计算."""
    return ComputationalStage(
        id="formation_energy",
        name="Formation Energy Calculation",
        tool="vasp_tool",
        tool_input={
            "action": "formation_energy",
            "pristine_structure": "${pristine_relax.output_structure}",
            "defect_structure": "${defect_relax.output_structure}",
        },
        dependencies=["pristine_relax", "defect_relax"],
    )


def defect_workflow(
    pristine_path: str, defect_type: str, site_index: int | None = None
) -> list[ComputationalStage]:
    """Defect calculation workflow.

    Args:
        pristine_path: Path to pristine structure
        defect_type: "vacancy", "substitution", or "interstitial"
        site_index: Site index for defect (auto-detected if None)
    """
    return [
        _defect_pristine_stage(pristine_path),
        _defect_create_stage(pristine_path, defect_type, site_index),
        _defect_relax_stage(),
        _defect_formation_energy_stage(),
    ]


def surface_workflow(
    bulk_path: str, miller_index: str = "111", layers: int = 6, vacuum: float = 15.0
) -> list[ComputationalStage]:
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
                "params": {"ISIF": 2},  # Only relax ions, keep cell fixed
            },
            dependencies=["cut_surface"],
            validation=ValidationRule(check="convergence"),
        ),
    ]


def ml_potential_workflow(
    training_structures: list[str], potential_type: str = "nep"
) -> list[ComputationalStage]:
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


# symbolic_derive 阶段的可选 tool_input 字段: (参数名, 值)
_SV_OPTIONAL_INPUTS = (
    "expression",
    "symbols",
    "variable",
    "free_energy",
    "matrix",
    "equations",
)


def _sv_tool_input(verify_type: str, **optional: Any) -> dict[str, Any]:
    """构造 symbolic_derive 阶段的 tool_input."""
    # Map unified verify_type to symbolic_math_tool action.
    # LeanTool auto_verify_action accepts the same names directly.
    symbolic_action = "differentiate" if verify_type == "derivative" else verify_type
    tool_input: dict[str, Any] = {"action": symbolic_action}
    for field_name in _SV_OPTIONAL_INPUTS:
        value = optional.get(field_name)
        if value is not None:
            tool_input[field_name] = value
    return tool_input


def _sv_stages(
    tool_input: dict[str, Any], verify_type: str
) -> list[ComputationalStage]:
    """symbolic_derive → lean_verify 两段流水线."""
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
    tool_input = _sv_tool_input(
        verify_type,
        expression=expression,
        symbols=symbols,
        variable=variable,
        free_energy=free_energy,
        matrix=matrix,
        equations=equations,
    )
    return _sv_stages(tool_input, verify_type)


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


def _rv_default_compute_input() -> dict[str, Any]:
    """默认示例: 对 x**2 求导, 跑通整个 reviewer 流水线."""
    return {
        "action": "differentiate",
        "expression": "x**2",
        "symbols": ["x"],
        "variable": "x",
    }


def _rv_compute_stage(
    compute_tool: str, compute_input: dict[str, Any]
) -> ComputationalStage:
    """compute 阶段: 先把结果算出来."""
    return ComputationalStage(
        id="compute",
        name="Compute",
        tool=compute_tool,
        tool_input=compute_input,
        persona="dft_expert",
        validation=ValidationRule(check="custom", custom_fn="has_result"),
    )


def _rv_validate_stage() -> ComputationalStage:
    """validate 阶段: 用审稿人眼光校验结果."""
    return ComputationalStage(
        id="validate",
        name="Validate Result",
        tool="validate_tool",
        tool_input={
            "result_type": "auto",
            "result_data": "${compute}",
        },
        dependencies=["compute"],
        persona="reviewer",
    )


def _rv_review_stage() -> ComputationalStage:
    """review 阶段: 走学术预审的理论审查."""
    return ComputationalStage(
        id="review",
        name="Peer Review",
        tool="rag_tool",
        tool_input={
            "action": "search",
            "query": "peer review checklist for computational materials science",
            "top_k": 5,
        },
        dependencies=["validate"],
        persona="reviewer_1_theory",
    )


def _rv_report_stage() -> ComputationalStage:
    """report 阶段: 用教学口吻写最终报告."""
    return ComputationalStage(
        id="report",
        name="Final Report",
        tool="report_tool",
        tool_input={
            "action": "render",
            "compute_output": "${compute}",
            "validation_output": "${validate}",
            "review_output": "${review}",
        },
        dependencies=["review"],
        persona="tutor",
    )


def reviewer_workflow(
    compute_tool: str = "symbolic_math_tool",
    compute_input: dict[str, Any] | None = None,
) -> list[ComputationalStage]:
    """Reviewer-driven 4-stage pipeline: compute → validate → review → report.

    每个阶段钉一个 persona, 同样的物理工具会被不同专家视角解读:
      - compute  : dft_expert            先把结果算出来
      - validate : reviewer              用审稿人眼光校验结果
      - review   : reviewer_1_theory     走 academic-pre-review-committee 的理论审查
      - report   : tutor                 用教学口吻写最终报告

    compute_tool / compute_input 可覆盖, 默认跑一个 symbolic_math 求导当示例.
    后续阶段的 tool_input 用 ${stage_id} 引用上游输出, 跟其他模板保持一致.

    Args:
        compute_tool: compute 阶段调用的工具名
        compute_input: compute 阶段的 tool_input; 不传走默认示例
    """
    if compute_input is None:
        compute_input = _rv_default_compute_input()

    return [
        _rv_compute_stage(compute_tool, compute_input),
        _rv_validate_stage(),
        _rv_review_stage(),
        _rv_report_stage(),
    ]


# simulation_type → plasma_tool 主仿真动作
_PLASMA_MAIN_ACTIONS = {
    "pic": "pic_simulation",
    "fluid": "fluid_simulation",
    "arc": "arc_plasma",
}


def _ps_main_action(simulation_type: str) -> str:
    """校验 simulation_type 并返回对应的主仿真动作."""
    if simulation_type not in _PLASMA_MAIN_ACTIONS:
        raise ValueError(
            f"simulation_type 必须是 {set(_PLASMA_MAIN_ACTIONS)} 之一, "
            f"收到 {simulation_type}"
        )
    return _PLASMA_MAIN_ACTIONS[simulation_type]


def _ps_setup_stage(
    plasma_density: float, electron_temp_eV: float
) -> ComputationalStage:
    """setup 阶段: 算基础等离子体参数."""
    return ComputationalStage(
        id="setup",
        name="Plasma Parameter Setup",
        tool="plasma_tool",
        tool_input={
            "action": "sheath_model",
            "plasma_density": plasma_density,
            "electron_temp": electron_temp_eV,
        },
        validation=ValidationRule(check="custom", custom_fn="has_result"),
    )


def _ps_simulate_stage(
    plasma_density: float,
    electron_temp_eV: float,
    simulation_type: str,
    main_action: str,
    num_steps: int,
) -> ComputationalStage:
    """simulate 阶段: 主仿真."""
    return ComputationalStage(
        id="simulate",
        name=f"Plasma {simulation_type.upper()} Simulation",
        tool="plasma_tool",
        tool_input={
            "action": main_action,
            "plasma_density": plasma_density,
            "electron_temp": electron_temp_eV,
            "num_steps": num_steps,
        },
        dependencies=["setup"],
        validation=ValidationRule(check="custom", custom_fn="has_result"),
        retry_policy=RetryPolicy(max_retries=1, retry_on=["convergence_fail"]),
    )


def _ps_analyze_stage(
    plasma_density: float, electron_temp_eV: float
) -> ComputationalStage:
    """analyze 阶段: 后处理 — 鞘层 + 输运系数 + 波色散."""
    return ComputationalStage(
        id="analyze",
        name="Post-Processing (Transport + Waves)",
        tool="plasma_tool",
        tool_input={
            "action": "transport_coefficients",
            "plasma_density": plasma_density,
            "electron_temp": electron_temp_eV,
        },
        dependencies=["simulate"],
    )


def _ps_report_stage() -> ComputationalStage:
    """report 阶段: 汇总报告."""
    return ComputationalStage(
        id="report",
        name="Final Report",
        tool="report_tool",
        tool_input={
            "action": "render",
            "setup_output": "${setup}",
            "simulate_output": "${simulate}",
            "analyze_output": "${analyze}",
        },
        dependencies=["analyze"],
    )


def plasma_simulation_workflow(
    plasma_density: float = 1e18,
    electron_temp_eV: float = 1.0,
    simulation_type: str = "pic",
    num_steps: int = 200,
) -> list[ComputationalStage]:
    """等离子体仿真 workflow: setup → simulate → analyze → report.

    借鉴 ai4plasma 的方法学, 把等离子体计算拆成 4 段:
      1. setup    : 用 plasma_tool 算基础等离子体参数 (Debye 长度/频率/Bohm 速度)
      2. simulate : 主仿真 (pic / fluid / arc, 由 simulation_type 决定)
      3. analyze  : 后处理 — 鞘层 + 输运系数 + 波色散
      4. report   : report_tool 汇总

    Args:
        plasma_density: 等离子体数密度 n (m^-3)
        electron_temp_eV: 电子温度 (eV)
        simulation_type: pic | fluid | arc, 选主仿真动作
        num_steps: 仿真步数
    """
    main_action = _ps_main_action(simulation_type)

    return [
        _ps_setup_stage(plasma_density, electron_temp_eV),
        _ps_simulate_stage(
            plasma_density,
            electron_temp_eV,
            simulation_type,
            main_action,
            num_steps,
        ),
        _ps_analyze_stage(plasma_density, electron_temp_eV),
        _ps_report_stage(),
    ]


def _rp_relax_stage(
    stage_id: str,
    stage_name: str,
    relax_tool: str,
    structure_path: str,
) -> ComputationalStage:
    """弛豫阶段 (relax_initial / relax_final 共用)."""
    return ComputationalStage(
        id=stage_id,
        name=stage_name,
        tool=relax_tool,
        tool_input={
            "action": "relax",
            "structure_file": structure_path,
        },
        validation=ValidationRule(check="convergence"),
        retry_policy=RetryPolicy(max_retries=2, retry_on=["convergence_fail"]),
    )


def _rp_neb_stage(
    neb_evaluator: str,
    n_images: int,
    max_iter: int,
    climbing_image: bool,
) -> ComputationalStage:
    """neb 阶段: CI-NEB 找最小能量路径."""
    return ComputationalStage(
        id="neb",
        name="Nudged Elastic Band",
        tool="neb_tool",
        tool_input={
            "action": "neb",
            "initial_structure": "${relax_initial.output_path}",
            "final_structure": "${relax_final.output_path}",
            "n_images": n_images,
            "max_iter": max_iter,
            "climbing_image": climbing_image,
            "energy_evaluator": neb_evaluator,
        },
        dependencies=["relax_initial", "relax_final"],
        validation=ValidationRule(check="custom", custom_fn="has_result"),
        retry_policy=RetryPolicy(max_retries=1, retry_on=["convergence_fail"]),
    )


def _rp_analyze_stage() -> ComputationalStage:
    """analyze 阶段: 算势垒 / 能量剖面."""
    return ComputationalStage(
        id="analyze",
        name="MEP Analysis",
        tool="neb_tool",
        tool_input={
            "action": "mep_analyze",
            "neb_result": "${neb}",
            "analysis_type": "energy_profile",
        },
        dependencies=["neb"],
    )


def _rp_report_stage() -> ComputationalStage:
    """report 阶段: 汇总反应路径报告."""
    return ComputationalStage(
        id="report",
        name="Reaction Pathway Report",
        tool="report_tool",
        tool_input={
            "action": "render",
            "neb_output": "${neb}",
            "analyze_output": "${analyze}",
        },
        dependencies=["analyze"],
    )


def reaction_pathway_workflow(
    initial_structure: str,
    final_structure: str,
    engine: str = "ml_potential",
    n_images: int = 7,
    max_iter: int = 300,
    climbing_image: bool = True,
) -> list[ComputationalStage]:
    """反应路径 workflow: 弛豫初末态 → NEB → MEP 分析 → 报告.

    典型用途: 找扩散势垒 / 化学反应最小能量路径 / 相变路径.
    五段式:
      1. relax_initial : 弛豫初始结构 (vasp_tool 或 ml_potential_tool)
      2. relax_final   : 弛豫末态结构
      3. neb           : neb_tool 跑 CI-NEB 找最小能量路径
      4. analyze       : neb_tool.mep_analyze 算势垒 / 能量剖面
      5. report        : report_tool 汇总反应路径报告

    Args:
        initial_structure: 初始结构文件路径
        final_structure: 末态结构文件路径
        engine: 弛豫引擎, "ml_potential" (默认, 快) 或 "vasp" (精确)
        n_images: NEB 图像数 (含首尾)
        max_iter: NEB 最大迭代步数
        climbing_image: 是否启用 CI-NEB
    """
    if engine not in ("ml_potential", "vasp"):
        raise ValueError(f"engine 必须是 'ml_potential' 或 'vasp', 收到 {engine}")

    relax_tool = f"{engine}_tool"

    # NEB 评估后端跟弛豫引擎保持一致: ml_potential 走 ML 势, vasp 走 DFT
    neb_evaluator = "ml_potential" if engine == "ml_potential" else "vasp"

    return [
        _rp_relax_stage(
            "relax_initial", "Relax Initial Structure", relax_tool, initial_structure
        ),
        _rp_relax_stage(
            "relax_final", "Relax Final Structure", relax_tool, final_structure
        ),
        _rp_neb_stage(neb_evaluator, n_images, max_iter, climbing_image),
        _rp_analyze_stage(),
        _rp_report_stage(),
    ]


# Registry of all templates
WORKFLOW_TEMPLATES = {
    "standard_dft": standard_dft_workflow,
    "aimd": aimd_workflow,
    "defect": defect_workflow,
    "surface": surface_workflow,
    "symbolic_verify": symbolic_verify_workflow,
    "tensor_calculus_verify": tensor_calculus_verify_workflow,
    "fem_verify": fem_verify_workflow,
    "la_verify": la_verify_workflow,
    "dft_verify": dft_verify_workflow,
    "thermo_verify": thermo_verify_workflow,
    "probability_verify": probability_verify_workflow,
    "reviewer": reviewer_workflow,
    "plasma_simulation": plasma_simulation_workflow,
    "reaction_pathway": reaction_pathway_workflow,
}

# 模块级别别名, 方便 `from huginn.workflows.templates import REVIEWER_WORKFLOW`
REVIEWER_WORKFLOW = reviewer_workflow
PLASMA_SIMULATION_WORKFLOW = plasma_simulation_workflow
REACTION_PATHWAY_WORKFLOW = reaction_pathway_workflow


def list_templates() -> list[str]:
    """List available workflow templates."""
    return list(WORKFLOW_TEMPLATES.keys())


def get_template(name: str):
    """Get a workflow template by name."""
    return WORKFLOW_TEMPLATES.get(name)


def register_template(name: str, template_fn):
    """Register a workflow template."""
    WORKFLOW_TEMPLATES[name] = template_fn
