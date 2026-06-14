"""Preset material science skills.

Pre-defined workflow templates for common computational tasks.
"""

from __future__ import annotations

from huginn.skills.base import SkillDefinition, SkillParameter, SkillStep
from huginn.skills.registry import register_skill


# --- Computational Skills ---

STANDARD_DFT = register_skill(SkillDefinition(
    name="standard_dft",
    description="Standard DFT workflow: relaxation → SCF → band structure + DOS",
    category="computation",
    parameters=[
        SkillParameter("structure_file", "str", "Path to POSCAR/CONTCAR or CIF file", required=True),
        SkillParameter("functional", "str", "Exchange-correlation functional", default="PBE"),
        SkillParameter("kpoints", "str", "K-point mesh density", default="gamma 4 4 4"),
        SkillParameter("encut", "float", "Plane-wave cutoff energy (eV)", default=520.0),
        SkillParameter("calculate_bands", "bool", "Compute band structure", default=True),
        SkillParameter("calculate_dos", "bool", "Compute DOS", default=True),
    ],
    steps=[
        SkillStep(
            name="validate_structure",
            tool="structure_tool",
            input_mapping={"action": "'analyze'", "file_path": "$structure_file"},
            output_key="structure_info",
        ),
        SkillStep(
            name="relax_structure",
            tool="vasp_tool",
            input_mapping={
                "action": "'relax'",
                "structure_file": "$structure_file",
                "encut": "$encut",
                "kpoints": "$kpoints",
                "functional": "$functional",
            },
            output_key="relax_result",
            on_failure="abort",
        ),
        SkillStep(
            name="scf_calculation",
            tool="vasp_tool",
            input_mapping={
                "action": "'scf'",
                "structure_file": "$relax_result.relaxed_structure",
                "encut": "$encut",
                "kpoints": "$kpoints",
            },
            output_key="scf_result",
            on_failure="abort",
        ),
        SkillStep(
            name="validate_scf",
            tool="validate_tool",
            input_mapping={
                "check_type": "'dft'",
                "data": "$scf_result",
            },
            output_key="validation",
            on_failure="abort",
        ),
    ],
    required_tools=["structure_tool", "vasp_tool", "validate_tool"],
    tags=["dft", "electronic_structure", "standard"],
))

AIMD_WORKFLOW = register_skill(SkillDefinition(
    name="aimd_workflow",
    description="Ab-initio MD: relaxation → thermalization → production run",
    category="computation",
    parameters=[
        SkillParameter("structure_file", "str", "Initial structure", required=True),
        SkillParameter("temperature", "float", "Target temperature (K)", default=300.0),
        SkillParameter("ensemble", "str", "NVT or NVE", default="NVT"),
        SkillParameter("timestep_fs", "float", "MD timestep in fs", default=1.0),
        SkillParameter("n_steps", "int", "Number of MD steps", default=1000),
        SkillParameter("functional", "str", "XC functional", default="PBE"),
    ],
    steps=[
        SkillStep(
            name="relax",
            tool="vasp_tool",
            input_mapping={
                "action": "'relax'",
                "structure_file": "$structure_file",
                "functional": "$functional",
            },
            output_key="relax",
        ),
        SkillStep(
            name="md_run",
            tool="vasp_tool",
            input_mapping={
                "action": "'aimd'",
                "structure_file": "$relax.relaxed_structure",
                "temperature": "$temperature",
                "ensemble": "$ensemble",
                "timestep": "$timestep_fs",
                "n_steps": "$n_steps",
            },
            output_key="md",
        ),
        SkillStep(
            name="validate_md",
            tool="validate_tool",
            input_mapping={
                "check_type": "'md'",
                "trajectory": "$md.trajectory",
            },
            output_key="validation",
        ),
    ],
    required_tools=["vasp_tool", "validate_tool"],
    tags=["md", "aimd", "dynamics"],
))

DEFECT_CALCULATION = register_skill(SkillDefinition(
    name="defect_calculation",
    description="Point defect workflow: perfect → defect → formation energy",
    category="computation",
    parameters=[
        SkillParameter("structure_file", "str", "Perfect crystal structure", required=True),
        SkillParameter("defect_type", "str", "vacancy | substitution | interstitial", required=True),
        SkillParameter("species", "str", "Defect species (e.g., 'V_O', 'Mg_Ca')", required=True),
        SkillParameter("charge_states", "list", "Charge states to calculate", default=[0]),
    ],
    steps=[
        SkillStep(
            name="perfect_relax",
            tool="vasp_tool",
            input_mapping={
                "action": "'relax'",
                "structure_file": "$structure_file",
            },
            output_key="perfect",
        ),
        SkillStep(
            name="create_defect",
            tool="structure_tool",
            input_mapping={
                "action": "'create_defect'",
                "structure_file": "$perfect.relaxed_structure",
                "defect_type": "$defect_type",
                "species": "$species",
            },
            output_key="defect_structure",
        ),
        SkillStep(
            name="defect_relax",
            tool="vasp_tool",
            input_mapping={
                "action": "'relax'",
                "structure_file": "$defect_structure.file",
            },
            output_key="defect",
        ),
    ],
    required_tools=["vasp_tool", "structure_tool"],
    tags=["defect", "point_defect", "formation_energy"],
))

SURFACE_CALCULATION = register_skill(SkillDefinition(
    name="surface_calculation",
    description="Surface slab workflow: bulk → cleave → relax → adsorption",
    category="computation",
    parameters=[
        SkillParameter("structure_file", "str", "Bulk structure", required=True),
        SkillParameter("miller_indices", "list", "Surface Miller indices", required=True),
        SkillParameter("slab_thickness", "int", "Number of atomic layers", default=6),
        SkillParameter("vacuum_angstrom", "float", "Vacuum thickness (Å)", default=15.0),
        SkillParameter("adsorbate", "str", "Optional adsorbate species", default=None),
    ],
    steps=[
        SkillStep(
            name="bulk_relax",
            tool="vasp_tool",
            input_mapping={"action": "'relax'", "structure_file": "$structure_file"},
            output_key="bulk",
        ),
        SkillStep(
            name="create_slab",
            tool="structure_tool",
            input_mapping={
                "action": "'create_slab'",
                "structure_file": "$bulk.relaxed_structure",
                "miller_indices": "$miller_indices",
                "slab_thickness": "$slab_thickness",
                "vacuum": "$vacuum_angstrom",
            },
            output_key="slab",
        ),
        SkillStep(
            name="slab_relax",
            tool="vasp_tool",
            input_mapping={"action": "'relax'", "structure_file": "$slab.file"},
            output_key="relaxed_slab",
        ),
    ],
    required_tools=["vasp_tool", "structure_tool"],
    tags=["surface", "slab", "adsorption"],
))

LAMMPS_MELT_QUENCH = register_skill(SkillDefinition(
    name="lammps_melt_quench",
    description="Classical MD melt-quench for amorphous structure generation",
    category="computation",
    parameters=[
        SkillParameter("structure_file", "str", "Initial crystal structure", required=True),
        SkillParameter("potential_file", "str", "Interatomic potential", required=True),
        SkillParameter("melt_temp", "float", "Melting temperature (K)", default=3000.0),
        SkillParameter("quench_rate", "float", "Quench rate (K/ps)", default=10.0),
        SkillParameter("final_temp", "float", "Final temperature (K)", default=300.0),
    ],
    steps=[
        SkillStep(
            name="melt",
            tool="lammps_tool",
            input_mapping={
                "action": "'melt'",
                "structure_file": "$structure_file",
                "potential": "$potential_file",
                "temperature": "$melt_temp",
            },
            output_key="melt",
        ),
        SkillStep(
            name="quench",
            tool="lammps_tool",
            input_mapping={
                "action": "'quench'",
                "structure_file": "$melt.final_structure",
                "potential": "$potential_file",
                "start_temp": "$melt_temp",
                "end_temp": "$final_temp",
                "rate": "$quench_rate",
            },
            output_key="quench",
        ),
        SkillStep(
            name="validate_glass",
            tool="validate_tool",
            input_mapping={
                "check_type": "'structure'",
                "data": "$quench.final_structure",
            },
            output_key="validation",
        ),
    ],
    required_tools=["lammps_tool", "validate_tool"],
    tags=["md", "lammps", "amorphous", "glass"],
))

ML_POTENTIAL_TRAINING = register_skill(SkillDefinition(
    name="ml_potential_training",
    description="Train a machine-learning potential: sampling → training → validation",
    category="computation",
    parameters=[
        SkillParameter("training_structures", "list", "List of training structure files", required=True),
        SkillParameter("potential_type", "str", "NEP | SNAP | GAP | ACE", default="NEP"),
        SkillParameter("test_split", "float", "Fraction for test set", default=0.2),
        SkillParameter("max_iterations", "int", "Training iterations", default=10000),
    ],
    steps=[
        SkillStep(
            name="train_potential",
            tool="potential_tool",
            input_mapping={
                "action": "'train'",
                "structures": "$training_structures",
                "potential_type": "$potential_type",
                "test_split": "$test_split",
                "max_iterations": "$max_iterations",
            },
            output_key="potential",
        ),
        SkillStep(
            name="validate_potential",
            tool="validate_tool",
            input_mapping={
                "check_type": "'potential'",
                "data": "$potential",
            },
            output_key="validation",
        ),
    ],
    required_tools=["potential_tool", "validate_tool"],
    tags=["ml_potential", "nep", "snap", "gap", "ace"],
))

# --- Analysis Skills ---

BAND_GAP_ANALYSIS = register_skill(SkillDefinition(
    name="band_gap_analysis",
    description="Extract and validate band gap from DFT calculations",
    category="analysis",
    parameters=[
        SkillParameter("vasprun_path", "str", "Path to vasprun.xml", required=True),
        SkillParameter("method", "str", "Direct | HSE | GW approximation", default="Direct"),
    ],
    steps=[
        SkillStep(
            name="extract_bands",
            tool="structure_tool",
            input_mapping={
                "action": "'extract_band_gap'",
                "file_path": "$vasprun_path",
            },
            output_key="band_gap",
        ),
        SkillStep(
            name="validate_gap",
            tool="validate_tool",
            input_mapping={
                "check_type": "'band_gap'",
                "value": "$band_gap.value",
            },
            output_key="validation",
        ),
    ],
    required_tools=["structure_tool", "validate_tool"],
    tags=["band_gap", "electronic", "analysis"],
))

ELASTIC_CONSTANTS = register_skill(SkillDefinition(
    name="elastic_constants",
    description="Calculate elastic tensor and mechanical stability",
    category="analysis",
    parameters=[
        SkillParameter("structure_file", "str", "Relaxed structure", required=True),
        SkillParameter("deformations", "int", "Number of strain deformations", default=6),
    ],
    steps=[
        SkillStep(
            name="calculate_elastic",
            tool="vasp_tool",
            input_mapping={
                "action": "'elastic'",
                "structure_file": "$structure_file",
                "n_deformations": "$deformations",
            },
            output_key="elastic",
        ),
        SkillStep(
            name="check_stability",
            tool="validate_tool",
            input_mapping={
                "check_type": "'elastic'",
                "tensor": "$elastic.tensor",
            },
            output_key="stability",
        ),
    ],
    required_tools=["vasp_tool", "validate_tool"],
    tags=["elastic", "mechanical", "stability"],
))

PHONON_CALCULATION = register_skill(SkillDefinition(
    name="phonon_calculation",
    description="Phonon dispersion and density of states",
    category="computation",
    parameters=[
        SkillParameter("structure_file", "str", "Relaxed structure", required=True),
        SkillParameter("supercell", "list", "Supercell dimensions", default=[2, 2, 2]),
        SkillParameter("method", "str", "DFPT | Finite differences", default="DFPT"),
    ],
    steps=[
        SkillStep(
            name="calculate_phonons",
            tool="vasp_tool",
            input_mapping={
                "action": "'phonon'",
                "structure_file": "$structure_file",
                "supercell": "$supercell",
                "method": "$method",
            },
            output_key="phonon",
        ),
        SkillStep(
            name="validate_phonons",
            tool="validate_tool",
            input_mapping={
                "check_type": "'phonon'",
                "data": "$phonon",
            },
            output_key="validation",
        ),
    ],
    required_tools=["vasp_tool", "validate_tool"],
    tags=["phonon", "vibrational", "dynamics"],
))

CONVERGENCE_DIAGNOSIS = register_skill(SkillDefinition(
    name="convergence_diagnosis",
    description="Diagnose and fix convergence failures in DFT/MD calculations",
    category="diagnostics",
    parameters=[
        SkillParameter("log_file", "str", "Path to calculation log (OUTCAR or log.lammps)", required=True),
        SkillParameter("engine", "str", "vasp | lammps", required=True),
        SkillParameter("auto_fix", "bool", "Automatically apply fixes", default=False),
    ],
    steps=[
        SkillStep(
            name="diagnose",
            tool="diagnose_tool",
            input_mapping={
                "engine": "$engine",
                "log_file": "$log_file",
            },
            output_key="diagnosis",
        ),
        SkillStep(
            name="apply_fix",
            tool="diagnose_tool",
            input_mapping={
                "action": "'fix'",
                "diagnosis": "$diagnosis",
                "auto_apply": "$auto_fix",
            },
            output_key="fix_result",
            on_failure="skip",
        ),
    ],
    required_tools=["diagnose_tool"],
    tags=["convergence", "diagnostics", "troubleshooting"],
))

HT_SCREENING = register_skill(SkillDefinition(
    name="ht_screening",
    description="High-throughput screening with multi-criteria decision analysis",
    category="analysis",
    parameters=[
        SkillParameter("candidates", "list", "List of structure files or IDs", required=True),
        SkillParameter("properties", "list", "Properties to calculate", default=["energy", "band_gap"]),
        SkillParameter("criteria", "dict", "MCDA criteria weights", default={}),
    ],
    steps=[
        SkillStep(
            name="batch_calculate",
            tool="job_tool",
            input_mapping={
                "action": "'batch_submit'",
                "structures": "$candidates",
                "properties": "$properties",
            },
            output_key="batch",
        ),
        SkillStep(
            name="evaluate",
            tool="evaluation_tool",
            input_mapping={
                "action": "'mcda'",
                "data": "$batch.results",
                "criteria": "$criteria",
            },
            output_key="ranking",
        ),
    ],
    required_tools=["job_tool", "evaluation_tool"],
    tags=["high_throughput", "screening", "mcda"],
))

SYMBOLIC_REGRESSION = register_skill(SkillDefinition(
    name="symbolic_regression",
    description="Discover analytical laws from simulation/experimental data using PSE/PSRN",
    category="analysis",
    parameters=[
        SkillParameter("data_file", "str", "Path to CSV with features and target", required=True),
        SkillParameter("target_column", "str", "Name of target variable", required=True),
        SkillParameter("feature_columns", "list", "Feature column names", default=None),
        SkillParameter("operators", "list", "Allowed operators", default=["Add", "Mul", "Identity", "Sin", "Cos", "Exp", "Log"]),
        SkillParameter("time_limit", "int", "Search time in seconds", default=300),
        SkillParameter("use_const", "bool", "Fit numerical constants", default=True),
    ],
    steps=[
        SkillStep(
            name="run_sr",
            tool="symbolic_regression_tool",
            input_mapping={
                "data_file": "$data_file",
                "target_column": "$target_column",
                "feature_columns": "$feature_columns",
                "operators": "$operators",
                "time_limit": "$time_limit",
                "use_const": "$use_const",
            },
            output_key="sr_result",
        ),
        SkillStep(
            name="validate_expression",
            tool="validate_tool",
            input_mapping={
                "check_type": "'expression'",
                "data": "$sr_result.best_expression",
            },
            output_key="validation",
            on_failure="skip",
        ),
    ],
    required_tools=["symbolic_regression_tool", "validate_tool"],
    tags=["symbolic_regression", "pse", "psrn", "data_mining", "law_discovery"],
))

SYMBOLIC_VERIFY = register_skill(SkillDefinition(
    name="symbolic_verify",
    description="Symbolic derivation followed by Lean 4 formal verification",
    category="verification",
    parameters=[
        SkillParameter("verify_type", "str", "derivative | constitutive | weak_form | eigenvalue | tensor_ops | solve", required=True),
        SkillParameter("expression", "str", "Mathematical expression (for derivative/weak_form)", default=None),
        SkillParameter("symbols", "list", "Symbol names", default=[]),
        SkillParameter("variable", "str", "Differentiation variable (for derivative)", default=None),
        SkillParameter("free_energy", "str", "Free energy expression (for constitutive)", default=None),
        SkillParameter("matrix", "list", "Matrix entries as nested lists of strings (for eigenvalue/tensor_ops)", default=None),
        SkillParameter("equations", "list", "Equations to solve (for solve)", default=None),
        SkillParameter("lean_project", "str", "Lean project name", default="HuginnLean"),
    ],
    steps=[
        SkillStep(
            name="symbolic_derive",
            tool="symbolic_math_tool",
            input_mapping={
                "action": "$verify_type",
                "expression": "$expression",
                "symbols": "$symbols",
                "variable": "$variable",
                "free_energy": "$free_energy",
                "matrix": "$matrix",
                "equations": "$equations",
            },
            output_key="symbolic_result",
        ),
        SkillStep(
            name="lean_verify",
            tool="lean_tool",
            input_mapping={
                "action": "'auto_verify'",
                "auto_verify_action": "$verify_type",
                "symbolic_result": "$symbolic_result",
            },
            output_key="verification",
            on_failure="abort",
        ),
    ],
    required_tools=["symbolic_math_tool", "lean_tool"],
    tags=["symbolic", "lean", "verification", "formal_methods", "cross_goal"],
))

TENSOR_VERIFY = register_skill(SkillDefinition(
    name="tensor_verify",
    description="Tensor calculus derivation followed by Lean 4 formal verification",
    category="verification",
    parameters=[
        SkillParameter("voigt_vector", "list", "Voigt components [v11,v22,v33,v23,v13,v12] or 21 stiffness params", required=True),
        SkillParameter("operation", "str", "invariants | deviatoric | principal | von_mises | rotate | apply_to_strain", default="invariants"),
        SkillParameter("tensor_type", "str", "stress | strain | stiffness | compliance", default="stress"),
        SkillParameter("rotation_matrix", "list", "Optional 3×3 rotation matrix", default=None),
    ],
    steps=[
        SkillStep(
            name="tensor_calculus",
            tool="symbolic_math_tool",
            input_mapping={
                "action": "'tensor_calculus'",
                "expression": "$operation",
                "tensor_type": "$tensor_type",
                "voigt_vector": "$voigt_vector",
                "rotation_matrix": "$rotation_matrix",
            },
            output_key="tensor_result",
        ),
        SkillStep(
            name="lean_verify",
            tool="lean_tool",
            input_mapping={
                "action": "'auto_verify'",
                "auto_verify_action": "'tensor_calculus'",
                "symbolic_result": "$tensor_result",
            },
            output_key="verification",
            on_failure="abort",
        ),
    ],
    required_tools=["symbolic_math_tool", "lean_tool"],
    tags=["tensor", "continuum_mechanics", "lean", "verification", "cross_goal"],
))

FEM_VERIFY = register_skill(SkillDefinition(
    name="fem_verify",
    description="FEM weak-form derivation and element matrix assembly with Lean 4 formal verification",
    category="verification",
    parameters=[
        SkillParameter("element_type", "str", "bar | poisson_tri | elasticity_tri | linear_elasticity | heat_conduction", required=True),
        SkillParameter("symbols", "list", "Symbol names (e.g. [\"u\", \"v\", \"x\", \"E\", \"A\", \"h\"])", default=[]),
        SkillParameter("expression", "str", "Extra expression (e.g. element type for assemble_element_matrix)", default=None),
    ],
    steps=[
        SkillStep(
            name="fem_derive",
            tool="symbolic_math_tool",
            input_mapping={
                "action": "'weak_form'",
                "target": "$element_type",
                "symbols": "$symbols",
                "expression": "$expression",
            },
            output_key="fem_result",
        ),
        SkillStep(
            name="lean_verify",
            tool="lean_tool",
            input_mapping={
                "action": "'auto_verify'",
                "auto_verify_action": "'fem'",
                "symbolic_result": "$fem_result",
            },
            output_key="verification",
            on_failure="abort",
        ),
    ],
    required_tools=["symbolic_math_tool", "lean_tool"],
    tags=["fem", "finite_element", "lean", "verification", "cross_goal"],
))

LA_VERIFY = register_skill(SkillDefinition(
    name="la_verify",
    description="Linear algebra computation followed by Lean 4 formal verification",
    category="verification",
    parameters=[
        SkillParameter("target", "str", "lu_decompose | cholesky | jacobi_solve | cg_solve | mat_vec_mul | cond_number", required=True),
        SkillParameter("matrix", "list", "Square matrix as nested lists of strings", default=None),
        SkillParameter("expression", "str", "Optional vector or extra expression (comma-separated for vectors)", default=None),
        SkillParameter("symbols", "list", "Symbol names for parsing", default=[]),
    ],
    steps=[
        SkillStep(
            name="la_compute",
            tool="symbolic_math_tool",
            input_mapping={
                "action": "'linear_algebra'",
                "target": "$target",
                "matrix": "$matrix",
                "expression": "$expression",
                "symbols": "$symbols",
            },
            output_key="la_result",
        ),
        SkillStep(
            name="lean_verify",
            tool="lean_tool",
            input_mapping={
                "action": "'auto_verify'",
                "auto_verify_action": "'linear_algebra'",
                "symbolic_result": "$la_result",
            },
            output_key="verification",
            on_failure="abort",
        ),
    ],
    required_tools=["symbolic_math_tool", "lean_tool"],
    tags=["linear_algebra", "numerical", "lean", "verification", "cross_goal"],
))

DFT_VERIFY = register_skill(SkillDefinition(
    name="dft_verify",
    description="DFT computation followed by Lean 4 formal verification",
    category="verification",
    parameters=[
        SkillParameter("target", "str", "fermi_energy | free_electron_dos | particle_in_box | tight_binding_band | lda_xc_energy", required=True),
        SkillParameter("expression", "str", "Parameter string like 'n=0.05' or 'L=10.0,N=3'", default=None),
    ],
    steps=[
        SkillStep(
            name="dft_compute",
            tool="symbolic_math_tool",
            input_mapping={
                "action": "'dft'",
                "target": "$target",
                "expression": "$expression",
            },
            output_key="dft_result",
        ),
        SkillStep(
            name="lean_verify",
            tool="lean_tool",
            input_mapping={
                "action": "'auto_verify'",
                "auto_verify_action": "'dft'",
                "symbolic_result": "$dft_result",
            },
            output_key="verification",
            on_failure="abort",
        ),
    ],
    required_tools=["symbolic_math_tool", "lean_tool"],
    tags=["dft", "electronic_structure", "lean", "verification", "cross_goal"],
))

THERMO_VERIFY = register_skill(SkillDefinition(
    name="thermo_verify",
    description="Thermodynamics computation followed by Lean 4 formal verification",
    category="verification",
    parameters=[
        SkillParameter("target", "str", "ideal_gas | van_der_waals | helmholtz_energy | gibbs_energy | chemical_potential | clausius_clapeyron | partition_function", required=True),
        SkillParameter("expression", "str", "Parameter string like 'n=1.0,T=273.15,V=0.022414'", default=None),
    ],
    steps=[
        SkillStep(
            name="thermo_compute",
            tool="symbolic_math_tool",
            input_mapping={
                "action": "'thermodynamics'",
                "target": "$target",
                "expression": "$expression",
            },
            output_key="thermo_result",
        ),
        SkillStep(
            name="lean_verify",
            tool="lean_tool",
            input_mapping={
                "action": "'auto_verify'",
                "auto_verify_action": "'thermodynamics'",
                "symbolic_result": "$thermo_result",
            },
            output_key="verification",
            on_failure="abort",
        ),
    ],
    required_tools=["symbolic_math_tool", "lean_tool"],
    tags=["thermodynamics", "statistical_mechanics", "lean", "verification", "cross_goal"],
))

PROBABILITY_VERIFY = register_skill(SkillDefinition(
    name="probability_verify",
    description="Probability and Gaussian process computation followed by Lean 4 formal verification",
    category="verification",
    parameters=[
        SkillParameter("target", "str", "normal_pdf | normal_cdf | gp_kernel | monte_carlo_integral | bayesian_update_normal", required=True),
        SkillParameter("expression", "str", "Parameter string like 'mu=0.0,sigma=1.0,x=0.0'", default=None),
        SkillParameter("equations", "list", "Optional list with kernel type for gp_kernel (e.g. ['rbf'])", default=None),
    ],
    steps=[
        SkillStep(
            name="probability_compute",
            tool="symbolic_math_tool",
            input_mapping={
                "action": "'probability'",
                "target": "$target",
                "expression": "$expression",
                "equations": "$equations",
            },
            output_key="probability_result",
        ),
        SkillStep(
            name="lean_verify",
            tool="lean_tool",
            input_mapping={
                "action": "'auto_verify'",
                "auto_verify_action": "'probability'",
                "symbolic_result": "$probability_result",
            },
            output_key="verification",
            on_failure="abort",
        ),
    ],
    required_tools=["symbolic_math_tool", "lean_tool"],
    tags=["probability", "gaussian_process", "bayesian", "lean", "verification", "cross_goal"],
))


# --- Uncertainty Quantification & Gaussian Process Skills ---

UNCERTAINTY_PROPAGATION = register_skill(SkillDefinition(
    name="uncertainty_propagation",
    description="Monte Carlo uncertainty propagation for a symbolic model",
    category="analysis",
    parameters=[
        SkillParameter("expression", "str", "SymPy-compatible expression", required=True),
        SkillParameter("variables", "list", "Variable specifications (list of dicts)", required=True),
        SkillParameter("n_samples", "int", "Number of Monte Carlo samples", default=1000),
        SkillParameter("seed", "int", "Random seed", default=None, required=False),
    ],
    steps=[
        SkillStep(
            name="run_monte_carlo",
            tool="uq_tool",
            input_mapping={
                "action": "monte_carlo",
                "expression": "$expression",
                "variables": "$variables",
                "n_samples": "$n_samples",
                "seed": "$seed",
            },
            output_key="uq_result",
        ),
    ],
    required_tools=["uq_tool"],
    tags=["uq", "monte_carlo", "uncertainty"],
))

SENSITIVITY_ANALYSIS = register_skill(SkillDefinition(
    name="sensitivity_analysis",
    description="Local or Sobol global sensitivity analysis for a symbolic model",
    category="analysis",
    parameters=[
        SkillParameter("expression", "str", "SymPy-compatible expression", required=True),
        SkillParameter("variables", "list", "Variable specifications (list of dicts)", required=True),
        SkillParameter("method", "str", "sensitivity | sobol", default="sensitivity"),
        SkillParameter("n_samples", "int", "Number of samples for Sobol", default=1000),
        SkillParameter("seed", "int", "Random seed", default=None, required=False),
    ],
    steps=[
        SkillStep(
            name="run_sensitivity",
            tool="uq_tool",
            input_mapping={
                "action": "$method",
                "expression": "$expression",
                "variables": "$variables",
                "n_samples": "$n_samples",
                "seed": "$seed",
            },
            output_key="sensitivity_result",
        ),
    ],
    required_tools=["uq_tool"],
    tags=["uq", "sensitivity", "sobol"],
))

GP_PREDICTION = register_skill(SkillDefinition(
    name="gp_prediction",
    description="Fit a Gaussian process surrogate and predict at new points",
    category="analysis",
    parameters=[
        SkillParameter("X", "list", "Training inputs (list of lists)", required=True),
        SkillParameter("y", "list", "Training targets (list of floats)", required=True),
        SkillParameter("X_new", "list", "Prediction inputs (list of lists)", required=True),
        SkillParameter("length_scale", "float", "Kernel length scale", default=1.0),
        SkillParameter("sigma_f", "float", "Signal variance", default=1.0),
        SkillParameter("sigma_n", "float", "Observation noise", default=1e-5),
    ],
    steps=[
        SkillStep(
            name="gp_predict",
            tool="gp_tool",
            input_mapping={
                "action": "predict",
                "X": "$X",
                "y": "$y",
                "X_new": "$X_new",
                "length_scale": "$length_scale",
                "sigma_f": "$sigma_f",
                "sigma_n": "$sigma_n",
            },
            output_key="gp_result",
        ),
    ],
    required_tools=["gp_tool"],
    tags=["gp", "surrogate", "prediction"],
))

BAYESIAN_CALIBRATION = register_skill(SkillDefinition(
    name="bayesian_calibration",
    description="Bayesian optimization / calibration loop against a symbolic objective",
    category="analysis",
    parameters=[
        SkillParameter("objective_expression", "str", "SymPy expression to optimize", required=True),
        SkillParameter("calibration_variables", "list", "Variable bounds (list of dicts with name/low/high)", required=True),
        SkillParameter("n_initial", "int", "Initial random samples", default=5),
        SkillParameter("n_iterations", "int", "Bayesian optimization iterations", default=10),
        SkillParameter("maximize", "bool", "Maximize (True) or minimize (False)", default=False),
        SkillParameter("seed", "int", "Random seed", default=None, required=False),
    ],
    steps=[
        SkillStep(
            name="gp_calibrate",
            tool="gp_tool",
            input_mapping={
                "action": "calibrate",
                "objective_expression": "$objective_expression",
                "calibration_variables": "$calibration_variables",
                "n_initial": "$n_initial",
                "n_iterations": "$n_iterations",
                "maximize": "$maximize",
                "seed": "$seed",
            },
            output_key="calibration_result",
        ),
    ],
    required_tools=["gp_tool"],
    tags=["gp", "bayesian", "calibration", "optimization"],
))
