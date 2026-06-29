"""Preset material science skills.

Pre-defined workflow templates for common computational tasks.
"""

from __future__ import annotations

from huginn.skills.base import SkillDefinition, SkillParameter, SkillStep
from huginn.skills.registry import register_skill

# --- Computational Skills ---

STANDARD_DFT = register_skill(
    SkillDefinition(
        name="standard_dft",
        description="Standard DFT workflow: relaxation → SCF → band structure + DOS",
        category="computation",
        parameters=[
            SkillParameter(
                "structure_file",
                "str",
                "Path to POSCAR/CONTCAR or CIF file",
                required=True,
            ),
            SkillParameter(
                "functional", "str", "Exchange-correlation functional", default="PBE"
            ),
            SkillParameter(
                "kpoints", "str", "K-point mesh density", default="gamma 4 4 4"
            ),
            SkillParameter(
                "encut", "float", "Plane-wave cutoff energy (eV)", default=520.0
            ),
            SkillParameter(
                "calculate_bands", "bool", "Compute band structure", default=True
            ),
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
    )
)

AIMD_WORKFLOW = register_skill(
    SkillDefinition(
        name="aimd_workflow",
        description="Ab-initio MD: relaxation → thermalization → production run",
        category="computation",
        parameters=[
            SkillParameter("structure_file", "str", "Initial structure", required=True),
            SkillParameter(
                "temperature", "float", "Target temperature (K)", default=300.0
            ),
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
    )
)

DEFECT_CALCULATION = register_skill(
    SkillDefinition(
        name="defect_calculation",
        description="Point defect workflow: perfect → defect → formation energy",
        category="computation",
        parameters=[
            SkillParameter(
                "structure_file", "str", "Perfect crystal structure", required=True
            ),
            SkillParameter(
                "defect_type",
                "str",
                "vacancy | substitution | interstitial",
                required=True,
            ),
            SkillParameter(
                "species", "str", "Defect species (e.g., 'V_O', 'Mg_Ca')", required=True
            ),
            SkillParameter(
                "charge_states", "list", "Charge states to calculate", default=[0]
            ),
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
    )
)

SURFACE_CALCULATION = register_skill(
    SkillDefinition(
        name="surface_calculation",
        description="Surface slab workflow: bulk → cleave → relax → adsorption",
        category="computation",
        parameters=[
            SkillParameter("structure_file", "str", "Bulk structure", required=True),
            SkillParameter(
                "miller_indices", "list", "Surface Miller indices", required=True
            ),
            SkillParameter(
                "slab_thickness", "int", "Number of atomic layers", default=6
            ),
            SkillParameter(
                "vacuum_angstrom", "float", "Vacuum thickness (Å)", default=15.0
            ),
            SkillParameter(
                "adsorbate", "str", "Optional adsorbate species", default=None
            ),
        ],
        steps=[
            SkillStep(
                name="bulk_relax",
                tool="vasp_tool",
                input_mapping={
                    "action": "'relax'",
                    "structure_file": "$structure_file",
                },
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
    )
)

LAMMPS_MELT_QUENCH = register_skill(
    SkillDefinition(
        name="lammps_melt_quench",
        description="Classical MD melt-quench for amorphous structure generation",
        category="computation",
        parameters=[
            SkillParameter(
                "structure_file", "str", "Initial crystal structure", required=True
            ),
            SkillParameter(
                "potential_file", "str", "Interatomic potential", required=True
            ),
            SkillParameter(
                "melt_temp", "float", "Melting temperature (K)", default=3000.0
            ),
            SkillParameter("quench_rate", "float", "Quench rate (K/ps)", default=10.0),
            SkillParameter(
                "final_temp", "float", "Final temperature (K)", default=300.0
            ),
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
    )
)

ML_POTENTIAL_TRAINING = register_skill(
    SkillDefinition(
        name="ml_potential_training",
        description="Train a machine-learning potential: sampling → training → validation",
        category="computation",
        parameters=[
            SkillParameter(
                "training_structures",
                "list",
                "List of training structure files",
                required=True,
            ),
            SkillParameter(
                "potential_type", "str", "NEP | SNAP | GAP | ACE", default="NEP"
            ),
            SkillParameter("test_split", "float", "Fraction for test set", default=0.2),
            SkillParameter(
                "max_iterations", "int", "Training iterations", default=10000
            ),
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
    )
)

# --- Analysis Skills ---

BAND_GAP_ANALYSIS = register_skill(
    SkillDefinition(
        name="band_gap_analysis",
        description="Extract and validate band gap from DFT calculations",
        category="analysis",
        parameters=[
            SkillParameter("vasprun_path", "str", "Path to vasprun.xml", required=True),
            SkillParameter(
                "method", "str", "Direct | HSE | GW approximation", default="Direct"
            ),
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
    )
)

ELASTIC_CONSTANTS = register_skill(
    SkillDefinition(
        name="elastic_constants",
        description="Calculate elastic tensor and mechanical stability",
        category="analysis",
        parameters=[
            SkillParameter("structure_file", "str", "Relaxed structure", required=True),
            SkillParameter(
                "deformations", "int", "Number of strain deformations", default=6
            ),
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
    )
)

PHONON_CALCULATION = register_skill(
    SkillDefinition(
        name="phonon_calculation",
        description="Phonon dispersion and density of states",
        category="computation",
        parameters=[
            SkillParameter("structure_file", "str", "Relaxed structure", required=True),
            SkillParameter(
                "supercell", "list", "Supercell dimensions", default=[2, 2, 2]
            ),
            SkillParameter(
                "method", "str", "DFPT | Finite differences", default="DFPT"
            ),
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
    )
)

CONVERGENCE_DIAGNOSIS = register_skill(
    SkillDefinition(
        name="convergence_diagnosis",
        description="Diagnose and fix convergence failures in DFT/MD calculations",
        category="diagnostics",
        parameters=[
            SkillParameter(
                "log_file",
                "str",
                "Path to calculation log (OUTCAR or log.lammps)",
                required=True,
            ),
            SkillParameter("engine", "str", "vasp | lammps", required=True),
            SkillParameter(
                "auto_fix", "bool", "Automatically apply fixes", default=False
            ),
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
    )
)

HT_SCREENING = register_skill(
    SkillDefinition(
        name="ht_screening",
        description="High-throughput screening with multi-criteria decision analysis",
        category="analysis",
        parameters=[
            SkillParameter(
                "candidates", "list", "List of structure files or IDs", required=True
            ),
            SkillParameter(
                "properties",
                "list",
                "Properties to calculate",
                default=["energy", "band_gap"],
            ),
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
    )
)

SYMBOLIC_REGRESSION = register_skill(
    SkillDefinition(
        name="symbolic_regression",
        description="Discover analytical laws from simulation/experimental data using PSE/PSRN",
        category="analysis",
        parameters=[
            SkillParameter(
                "data_file",
                "str",
                "Path to CSV with features and target",
                required=True,
            ),
            SkillParameter(
                "target_column", "str", "Name of target variable", required=True
            ),
            SkillParameter(
                "feature_columns", "list", "Feature column names", default=None
            ),
            SkillParameter(
                "operators",
                "list",
                "Allowed operators",
                default=["Add", "Mul", "Identity", "Sin", "Cos", "Exp", "Log"],
            ),
            SkillParameter("time_limit", "int", "Search time in seconds", default=300),
            SkillParameter(
                "use_const", "bool", "Fit numerical constants", default=True
            ),
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
    )
)

SYMBOLIC_VERIFY = register_skill(
    SkillDefinition(
        name="symbolic_verify",
        description="Symbolic derivation followed by Lean 4 formal verification",
        category="verification",
        parameters=[
            SkillParameter(
                "verify_type",
                "str",
                "derivative | constitutive | weak_form | eigenvalue | tensor_ops | solve",
                required=True,
            ),
            SkillParameter(
                "expression",
                "str",
                "Mathematical expression (for derivative/weak_form)",
                default=None,
            ),
            SkillParameter("symbols", "list", "Symbol names", default=[]),
            SkillParameter(
                "variable",
                "str",
                "Differentiation variable (for derivative)",
                default=None,
            ),
            SkillParameter(
                "free_energy",
                "str",
                "Free energy expression (for constitutive)",
                default=None,
            ),
            SkillParameter(
                "matrix",
                "list",
                "Matrix entries as nested lists of strings (for eigenvalue/tensor_ops)",
                default=None,
            ),
            SkillParameter(
                "equations", "list", "Equations to solve (for solve)", default=None
            ),
            SkillParameter(
                "lean_project", "str", "Lean project name", default="HuginnLean"
            ),
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
    )
)

TENSOR_VERIFY = register_skill(
    SkillDefinition(
        name="tensor_verify",
        description="Tensor calculus derivation followed by Lean 4 formal verification",
        category="verification",
        parameters=[
            SkillParameter(
                "voigt_vector",
                "list",
                "Voigt components [v11,v22,v33,v23,v13,v12] or 21 stiffness params",
                required=True,
            ),
            SkillParameter(
                "operation",
                "str",
                "invariants | deviatoric | principal | von_mises | rotate | apply_to_strain",
                default="invariants",
            ),
            SkillParameter(
                "tensor_type",
                "str",
                "stress | strain | stiffness | compliance",
                default="stress",
            ),
            SkillParameter(
                "rotation_matrix", "list", "Optional 3×3 rotation matrix", default=None
            ),
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
    )
)

FEM_VERIFY = register_skill(
    SkillDefinition(
        name="fem_verify",
        description="FEM weak-form derivation and element matrix assembly with Lean 4 formal verification",
        category="verification",
        parameters=[
            SkillParameter(
                "element_type",
                "str",
                "bar | poisson_tri | elasticity_tri | linear_elasticity | heat_conduction",
                required=True,
            ),
            SkillParameter(
                "symbols",
                "list",
                'Symbol names (e.g. ["u", "v", "x", "E", "A", "h"])',
                default=[],
            ),
            SkillParameter(
                "expression",
                "str",
                "Extra expression (e.g. element type for assemble_element_matrix)",
                default=None,
            ),
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
    )
)

LA_VERIFY = register_skill(
    SkillDefinition(
        name="la_verify",
        description="Linear algebra computation followed by Lean 4 formal verification",
        category="verification",
        parameters=[
            SkillParameter(
                "target",
                "str",
                "lu_decompose | cholesky | jacobi_solve | cg_solve | mat_vec_mul | cond_number",
                required=True,
            ),
            SkillParameter(
                "matrix",
                "list",
                "Square matrix as nested lists of strings",
                default=None,
            ),
            SkillParameter(
                "expression",
                "str",
                "Optional vector or extra expression (comma-separated for vectors)",
                default=None,
            ),
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
    )
)

DFT_VERIFY = register_skill(
    SkillDefinition(
        name="dft_verify",
        description="DFT computation followed by Lean 4 formal verification",
        category="verification",
        parameters=[
            SkillParameter(
                "target",
                "str",
                "fermi_energy | free_electron_dos | particle_in_box | tight_binding_band | lda_xc_energy",
                required=True,
            ),
            SkillParameter(
                "expression",
                "str",
                "Parameter string like 'n=0.05' or 'L=10.0,N=3'",
                default=None,
            ),
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
    )
)

THERMO_VERIFY = register_skill(
    SkillDefinition(
        name="thermo_verify",
        description="Thermodynamics computation followed by Lean 4 formal verification",
        category="verification",
        parameters=[
            SkillParameter(
                "target",
                "str",
                "ideal_gas | van_der_waals | helmholtz_energy | gibbs_energy | chemical_potential | clausius_clapeyron | partition_function",
                required=True,
            ),
            SkillParameter(
                "expression",
                "str",
                "Parameter string like 'n=1.0,T=273.15,V=0.022414'",
                default=None,
            ),
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
        tags=[
            "thermodynamics",
            "statistical_mechanics",
            "lean",
            "verification",
            "cross_goal",
        ],
    )
)

PROBABILITY_VERIFY = register_skill(
    SkillDefinition(
        name="probability_verify",
        description="Probability and Gaussian process computation followed by Lean 4 formal verification",
        category="verification",
        parameters=[
            SkillParameter(
                "target",
                "str",
                "normal_pdf | normal_cdf | gp_kernel | monte_carlo_integral | bayesian_update_normal",
                required=True,
            ),
            SkillParameter(
                "expression",
                "str",
                "Parameter string like 'mu=0.0,sigma=1.0,x=0.0'",
                default=None,
            ),
            SkillParameter(
                "equations",
                "list",
                "Optional list with kernel type for gp_kernel (e.g. ['rbf'])",
                default=None,
            ),
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
        tags=[
            "probability",
            "gaussian_process",
            "bayesian",
            "lean",
            "verification",
            "cross_goal",
        ],
    )
)


# --- Uncertainty Quantification & Gaussian Process Skills ---

UNCERTAINTY_PROPAGATION = register_skill(
    SkillDefinition(
        name="uncertainty_propagation",
        description="Monte Carlo uncertainty propagation for a symbolic model",
        category="analysis",
        parameters=[
            SkillParameter(
                "expression", "str", "SymPy-compatible expression", required=True
            ),
            SkillParameter(
                "variables",
                "list",
                "Variable specifications (list of dicts)",
                required=True,
            ),
            SkillParameter(
                "n_samples", "int", "Number of Monte Carlo samples", default=1000
            ),
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
    )
)

SENSITIVITY_ANALYSIS = register_skill(
    SkillDefinition(
        name="sensitivity_analysis",
        description="Local or Sobol global sensitivity analysis for a symbolic model",
        category="analysis",
        parameters=[
            SkillParameter(
                "expression", "str", "SymPy-compatible expression", required=True
            ),
            SkillParameter(
                "variables",
                "list",
                "Variable specifications (list of dicts)",
                required=True,
            ),
            SkillParameter(
                "method", "str", "sensitivity | sobol", default="sensitivity"
            ),
            SkillParameter(
                "n_samples", "int", "Number of samples for Sobol", default=1000
            ),
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
    )
)

GP_PREDICTION = register_skill(
    SkillDefinition(
        name="gp_prediction",
        description="Fit a Gaussian process surrogate and predict at new points",
        category="analysis",
        parameters=[
            SkillParameter(
                "X", "list", "Training inputs (list of lists)", required=True
            ),
            SkillParameter(
                "y", "list", "Training targets (list of floats)", required=True
            ),
            SkillParameter(
                "X_new", "list", "Prediction inputs (list of lists)", required=True
            ),
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
    )
)

BAYESIAN_CALIBRATION = register_skill(
    SkillDefinition(
        name="bayesian_calibration",
        description="Bayesian optimization / calibration loop against a symbolic objective",
        category="analysis",
        parameters=[
            SkillParameter(
                "objective_expression",
                "str",
                "SymPy expression to optimize",
                required=True,
            ),
            SkillParameter(
                "calibration_variables",
                "list",
                "Variable bounds (list of dicts with name/low/high)",
                required=True,
            ),
            SkillParameter("n_initial", "int", "Initial random samples", default=5),
            SkillParameter(
                "n_iterations", "int", "Bayesian optimization iterations", default=10
            ),
            SkillParameter(
                "maximize", "bool", "Maximize (True) or minimize (False)", default=False
            ),
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
    )
)

HPC_REMOTE_RUN = register_skill(
    SkillDefinition(
        name="hpc_remote_run",
        description="Run a command remotely on an HPC cluster via SSH/Slurm/PBS",
        category="computation",
        parameters=[
            SkillParameter(
                "command",
                "str",
                "Shell command to execute on the remote host",
                required=True,
            ),
            SkillParameter("hpc_host", "str", "HPC hostname", required=True),
            SkillParameter("hpc_username", "str", "HPC username", required=True),
            SkillParameter(
                "queue", "str", "Scheduler partition/queue", default="normal"
            ),
            SkillParameter("walltime_hours", "int", "Walltime in hours", default=24),
            SkillParameter("cores", "int", "Number of CPU cores", default=4),
            SkillParameter("memory_gb", "int", "Memory in GB", default=16),
            SkillParameter("modules", "list", "Modules to load", default=[]),
            SkillParameter(
                "remote_work_dir",
                "str",
                "Remote working directory",
                default="~/huginn_jobs",
            ),
        ],
        steps=[
            SkillStep(
                name="submit_remote",
                tool="job_tool",
                input_mapping={
                    "action": "'submit_remote'",
                    "command": "$command",
                    "hpc_host": "$hpc_host",
                    "hpc_username": "$hpc_username",
                    "queue": "$queue",
                    "walltime_hours": "$walltime_hours",
                    "cores": "$cores",
                    "memory_gb": "$memory_gb",
                    "modules": "$modules",
                    "remote_work_dir": "$remote_work_dir",
                },
                output_key="job",
            ),
            SkillStep(
                name="poll_remote",
                tool="job_tool",
                input_mapping={
                    "action": "'poll_remote'",
                    "job_id": "$job.job_id",
                    "hpc_host": "$hpc_host",
                    "hpc_username": "$hpc_username",
                },
                output_key="status",
            ),
        ],
        required_tools=["job_tool"],
        tags=["hpc", "remote", "slurm", "pbs"],
    )
)

ACTIVE_LEARNING_SAMPLING = register_skill(
    SkillDefinition(
        name="active_learning_sampling",
        description="Iteratively expand a training set for ML potentials using uncertainty sampling",
        category="computation",
        parameters=[
            SkillParameter(
                "initial_structures",
                "list",
                "Starting training structure files",
                required=True,
            ),
            SkillParameter(
                "potential_type", "str", "NEP | SNAP | GAP | ACE | MACE", default="NEP"
            ),
            SkillParameter(
                "md_temperature", "float", "Exploration temperature (K)", default=300.0
            ),
            SkillParameter(
                "max_iterations", "int", "Active-learning iterations", default=5
            ),
            SkillParameter(
                "uncertainty_threshold",
                "float",
                "Force uncertainty threshold for re-labeling",
                default=0.5,
            ),
        ],
        steps=[
            SkillStep(
                name="train_initial",
                tool="potential_tool",
                input_mapping={
                    "action": "'train'",
                    "structures": "$initial_structures",
                    "potential_type": "$potential_type",
                },
                output_key="potential",
            ),
            SkillStep(
                name="run_exploratory_md",
                tool="lammps_tool",
                input_mapping={
                    "action": "'md'",
                    "structure_file": "$initial_structures[0]",
                    "potential": "$potential.file",
                    "temperature": "$md_temperature",
                },
                output_key="trajectory",
            ),
            SkillStep(
                name="identify_uncertain_frames",
                tool="validate_tool",
                input_mapping={
                    "check_type": "'uncertainty'",
                    "data": "$trajectory.frames",
                    "threshold": "$uncertainty_threshold",
                },
                output_key="uncertain_frames",
            ),
        ],
        required_tools=["potential_tool", "lammps_tool", "validate_tool"],
        tags=["ml_potential", "active_learning", "sampling"],
    )
)


CONVERGENCE_TEST = register_skill(
    SkillDefinition(
        name="convergence_test",
        description="Systematic ENCUT and k-point convergence for DFT total energies",
        category="computation",
        parameters=[
            SkillParameter("structure_file", "str", "Initial structure", required=True),
            SkillParameter(
                "encut_values",
                "list",
                "List of ENCUT values to test (eV)",
                default=[400, 500, 600, 700, 800],
            ),
            SkillParameter(
                "kpoint_densities",
                "list",
                "List of k-point grid densities (e.g. [2,3,4,5])",
                default=[2, 3, 4, 5],
            ),
            SkillParameter("functional", "str", "XC functional", default="PBE"),
            SkillParameter(
                "threshold_meV",
                "float",
                "Convergence threshold per atom (meV)",
                default=1.0,
            ),
        ],
        steps=[
            SkillStep(
                name="encut_scan",
                tool="vasp_tool",
                input_mapping={
                    "action": "'convergence_encut'",
                    "structure_file": "$structure_file",
                    "encut_values": "$encut_values",
                    "functional": "$functional",
                    "threshold_meV": "$threshold_meV",
                },
                output_key="encut_result",
            ),
            SkillStep(
                name="kpoint_scan",
                tool="vasp_tool",
                input_mapping={
                    "action": "'convergence_kpoints'",
                    "structure_file": "$structure_file",
                    "kpoint_densities": "$kpoint_densities",
                    "functional": "$functional",
                    "threshold_meV": "$threshold_meV",
                },
                output_key="kpoint_result",
            ),
            SkillStep(
                name="validate_convergence",
                tool="validate_tool",
                input_mapping={
                    "check_type": "'convergence'",
                    "data": {
                        "encut": "$encut_result",
                        "kpoints": "$kpoint_result",
                    },
                },
                output_key="validation",
            ),
        ],
        required_tools=["vasp_tool", "validate_tool"],
        tags=["dft", "convergence", "encut", "kpoints"],
    )
)

BATTERY_IONIC_CONDUCTIVITY = register_skill(
    SkillDefinition(
        name="battery_ionic_conductivity",
        description="Estimate Li/Na ionic conductivity from AIMD trajectories",
        category="computation",
        parameters=[
            SkillParameter(
                "structure_file",
                "str",
                "Initial electrode/electrolyte structure",
                required=True,
            ),
            SkillParameter(
                "temperatures",
                "list",
                "Temperatures (K) to simulate",
                default=[300, 400, 500],
            ),
            SkillParameter("md_steps", "int", "Production MD steps", default=5000),
            SkillParameter("timestep_fs", "float", "MD timestep (fs)", default=2.0),
            SkillParameter("mobile_ion", "str", "Mobile species symbol", default="Li"),
        ],
        steps=[
            SkillStep(
                name="equilibrate",
                tool="vasp_tool",
                input_mapping={
                    "action": "'relax'",
                    "structure_file": "$structure_file",
                },
                output_key="relaxed",
            ),
            SkillStep(
                name="aimd_runs",
                tool="vasp_tool",
                input_mapping={
                    "action": "'aimd'",
                    "structure_file": "$relaxed.relaxed_structure",
                    "temperature": "$temperatures",
                    "n_steps": "$md_steps",
                    "timestep": "$timestep_fs",
                },
                output_key="trajectories",
            ),
            SkillStep(
                name="diffusivity_analysis",
                tool="structure_tool",
                input_mapping={
                    "action": "'msd_analysis'",
                    "trajectories": "$trajectories",
                    "mobile_ion": "$mobile_ion",
                },
                output_key="diffusivity",
            ),
            SkillStep(
                name="arrhenius_fit",
                tool="evaluation_tool",
                input_mapping={
                    "action": "'arrhenius_fit'",
                    "temperatures": "$temperatures",
                    "diffusivities": "$diffusivity.values",
                },
                output_key="conductivity",
            ),
        ],
        required_tools=["vasp_tool", "structure_tool", "evaluation_tool"],
        tags=["battery", "ionic_conductivity", "aimd", "diffusion"],
    )
)

CATALYSIS_SCREENING = register_skill(
    SkillDefinition(
        name="catalysis_screening",
        description="Screen adsorption energies on surface slabs for catalysis",
        category="computation",
        parameters=[
            SkillParameter(
                "bulk_structure", "str", "Bulk catalyst structure", required=True
            ),
            SkillParameter(
                "miller_indices", "list", "Surface facets to screen", required=True
            ),
            SkillParameter(
                "adsorbates", "list", "Adsorbate species/formulas", required=True
            ),
            SkillParameter(
                "vacuum_angstrom", "float", "Vacuum thickness", default=15.0
            ),
            SkillParameter(
                "slab_thickness", "int", "Number of atomic layers", default=6
            ),
        ],
        steps=[
            SkillStep(
                name="bulk_relax",
                tool="vasp_tool",
                input_mapping={
                    "action": "'relax'",
                    "structure_file": "$bulk_structure",
                },
                output_key="bulk",
            ),
            SkillStep(
                name="create_slabs",
                tool="structure_tool",
                input_mapping={
                    "action": "'create_slabs'",
                    "structure_file": "$bulk.relaxed_structure",
                    "miller_indices": "$miller_indices",
                    "slab_thickness": "$slab_thickness",
                    "vacuum": "$vacuum_angstrom",
                },
                output_key="slabs",
            ),
            SkillStep(
                name="adsorption_screen",
                tool="vasp_tool",
                input_mapping={
                    "action": "'adsorption'",
                    "slabs": "$slabs.files",
                    "adsorbates": "$adsorbates",
                },
                output_key="adsorption",
            ),
            SkillStep(
                name="rank_sites",
                tool="evaluation_tool",
                input_mapping={
                    "action": "'rank'",
                    "data": "$adsorption.energies",
                    "criteria": {
                        "lower_is_better": True,
                        "property": "adsorption_energy",
                    },
                },
                output_key="ranking",
            ),
        ],
        required_tools=["vasp_tool", "structure_tool", "evaluation_tool"],
        tags=["catalysis", "surface", "adsorption", "screening"],
    )
)

PHASE_DIAGRAM_CONSTRUCTION = register_skill(
    SkillDefinition(
        name="phase_diagram_construction",
        description="Build a zero-Kelvin convex hull phase diagram from DFT energies",
        category="analysis",
        parameters=[
            SkillParameter(
                "structure_files", "list", "Candidate structure files", required=True
            ),
            SkillParameter(
                "composition_reference",
                "dict",
                "Elemental reference energies",
                required=True,
            ),
            SkillParameter("functional", "str", "XC functional", default="PBE"),
            SkillParameter("encut", "float", "Plane-wave cutoff", default=520.0),
        ],
        steps=[
            SkillStep(
                name="relax_candidates",
                tool="vasp_tool",
                input_mapping={
                    "action": "'relax_batch'",
                    "structure_files": "$structure_files",
                    "functional": "$functional",
                    "encut": "$encut",
                },
                output_key="relaxed",
            ),
            SkillStep(
                name="build_phase_diagram",
                tool="structure_tool",
                input_mapping={
                    "action": "'phase_diagram'",
                    "structures": "$relaxed.energies",
                    "references": "$composition_reference",
                },
                output_key="phase_diagram",
            ),
            SkillStep(
                name="validate_hull",
                tool="validate_tool",
                input_mapping={
                    "check_type": "'phase_diagram'",
                    "data": "$phase_diagram",
                },
                output_key="validation",
            ),
        ],
        required_tools=["vasp_tool", "structure_tool", "validate_tool"],
        tags=["phase_diagram", "convex_hull", "thermodynamics"],
    )
)


XRD_STRUCTURE_SOLUTION = register_skill(
    SkillDefinition(
        name="xrd_structure_solution",
        description="Solve a crystal structure from powder XRD data",
        category="analysis",
        parameters=[
            SkillParameter(
                "xrd_file", "str", "Path to powder XRD pattern", required=True
            ),
            SkillParameter(
                "wavelength", "float", "X-ray wavelength (Å)", default=1.5418
            ),
            SkillParameter("symmetry", "str", "Crystal system hint", default=None),
            SkillParameter(
                "chemical_formula", "str", "Nominal composition", default=None
            ),
        ],
        steps=[
            SkillStep(
                name="peak_search",
                tool="structure_tool",
                input_mapping={
                    "action": "'xrd_peak_search'",
                    "file_path": "$xrd_file",
                    "wavelength": "$wavelength",
                },
                output_key="peaks",
            ),
            SkillStep(
                name="index_pattern",
                tool="structure_tool",
                input_mapping={
                    "action": "'index_xrd'",
                    "peaks": "$peaks",
                    "symmetry": "$symmetry",
                },
                output_key="unit_cell",
            ),
            SkillStep(
                name="solve_structure",
                tool="structure_tool",
                input_mapping={
                    "action": "'solve_structure'",
                    "xrd_file": "$xrd_file",
                    "unit_cell": "$unit_cell",
                    "composition": "$chemical_formula",
                },
                output_key="structure",
            ),
            SkillStep(
                name="rietveld_refinement",
                tool="validate_tool",
                input_mapping={
                    "check_type": "'rietveld'",
                    "data": "$structure",
                },
                output_key="validation",
            ),
        ],
        required_tools=["structure_tool", "validate_tool"],
        tags=["xrd", "crystallography", "structure_solution"],
    )
)

PHONON_SPECTROSCOPY_WORKFLOW = register_skill(
    SkillDefinition(
        name="phonon_spectroscopy_workflow",
        description="Compute phonons and compare IR/Raman spectra with experiment",
        category="computation",
        parameters=[
            SkillParameter("structure_file", "str", "Relaxed structure", required=True),
            SkillParameter(
                "supercell", "list", "Supercell dimensions", default=[2, 2, 2]
            ),
            SkillParameter(
                "method", "str", "DFPT | finite_differences", default="DFPT"
            ),
            SkillParameter(
                "experimental_spectrum", "str", "Optional spectrum file", default=None
            ),
        ],
        steps=[
            SkillStep(
                name="phonon_calculation",
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
                name="spectrum_analysis",
                tool="structure_tool",
                input_mapping={
                    "action": "'ir_raman_spectrum'",
                    "phonon_data": "$phonon",
                    "experimental_spectrum": "$experimental_spectrum",
                },
                output_key="spectrum",
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
        required_tools=["vasp_tool", "structure_tool", "validate_tool"],
        tags=["phonon", "ir", "raman", "spectroscopy"],
    )
)

CALPHAD_PHASE_DIAGRAM = register_skill(
    SkillDefinition(
        name="calphad_phase_diagram",
        description="Compute a CALPHAD phase diagram for an alloy system",
        category="analysis",
        parameters=[
            SkillParameter("elements", "list", "List of elements", required=True),
            SkillParameter(
                "temperature_range", "list", "[T_min, T_max] in K", required=True
            ),
            SkillParameter("database", "str", "Thermodynamic database", default="TCFE"),
            SkillParameter("pressure_pa", "float", "Pressure in Pa", default=1e5),
        ],
        steps=[
            SkillStep(
                name="database_query",
                tool="database_tool",
                input_mapping={
                    "action": "'calphad_lookup'",
                    "elements": "$elements",
                    "database": "$database",
                },
                output_key="thermo_data",
            ),
            SkillStep(
                name="compute_phase_diagram",
                tool="structure_tool",
                input_mapping={
                    "action": "'calphad_diagram'",
                    "elements": "$elements",
                    "temperature_range": "$temperature_range",
                    "pressure": "$pressure_pa",
                    "database": "$thermo_data",
                },
                output_key="phase_diagram",
            ),
            SkillStep(
                name="validate_diagram",
                tool="validate_tool",
                input_mapping={
                    "check_type": "'phase_diagram'",
                    "data": "$phase_diagram",
                },
                output_key="validation",
            ),
        ],
        required_tools=["database_tool", "structure_tool", "validate_tool"],
        tags=["calphad", "phase_diagram", "thermodynamics"],
    )
)

DEFECT_FORMATION_ENERGY = register_skill(
    SkillDefinition(
        name="defect_formation_energy",
        description="Calculate point-defect formation energies and transition levels",
        category="computation",
        parameters=[
            SkillParameter(
                "structure_file", "str", "Perfect crystal structure", required=True
            ),
            SkillParameter(
                "defect_type",
                "str",
                "vacancy | substitution | interstitial",
                required=True,
            ),
            SkillParameter("species", "str", "Defect species label", required=True),
            SkillParameter("charge_states", "list", "Charge states", default=[0]),
            SkillParameter(
                "supercell", "list", "Supercell dimensions", default=[3, 3, 3]
            ),
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
                    "supercell": "$supercell",
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
            SkillStep(
                name="formation_energy",
                tool="structure_tool",
                input_mapping={
                    "action": "'defect_formation_energy'",
                    "perfect_energy": "$perfect.energy",
                    "defect_energy": "$defect.energy",
                    "charge_states": "$charge_states",
                },
                output_key="formation_energy",
            ),
        ],
        required_tools=["vasp_tool", "structure_tool"],
        tags=["defect", "formation_energy", "point_defect"],
    )
)


ELECTROCHEMISTRY_POURBAIX = register_skill(
    SkillDefinition(
        name="electrochemistry_pourbaix",
        description="Construct a Pourbaix diagram for an aqueous electrochemical system",
        category="computation",
        parameters=[
            SkillParameter("elements", "list", "Elements in the system", required=True),
            SkillParameter("ph_range", "list", "[pH_min, pH_max]", default=[0, 14]),
            SkillParameter(
                "potential_range", "list", "[V_min, V_max] vs SHE", default=[-2, 3]
            ),
            SkillParameter("temperature", "float", "Temperature (K)", default=298.15),
        ],
        steps=[
            SkillStep(
                name="query_thermo_data",
                tool="database_tool",
                input_mapping={
                    "action": "'pourbaix_lookup'",
                    "elements": "$elements",
                    "temperature": "$temperature",
                },
                output_key="thermo_data",
            ),
            SkillStep(
                name="build_pourbaix",
                tool="structure_tool",
                input_mapping={
                    "action": "'pourbaix_diagram'",
                    "elements": "$elements",
                    "ph_range": "$ph_range",
                    "potential_range": "$potential_range",
                    "thermo_data": "$thermo_data",
                },
                output_key="pourbaix",
            ),
            SkillStep(
                name="validate_diagram",
                tool="validate_tool",
                input_mapping={
                    "check_type": "'phase_diagram'",
                    "data": "$pourbaix",
                },
                output_key="validation",
            ),
        ],
        required_tools=["database_tool", "structure_tool", "validate_tool"],
        tags=["electrochemistry", "pourbaix", "aqueous"],
    )
)

POLYMER_GLASS_TRANSITION = register_skill(
    SkillDefinition(
        name="polymer_glass_transition",
        description="Estimate polymer glass transition temperature from MD simulations",
        category="computation",
        parameters=[
            SkillParameter(
                "polymer_file",
                "str",
                "Polymer structure or monomer file",
                required=True,
            ),
            SkillParameter(
                "force_field", "str", "OPLS | GAFF | TraPPE | MARTINI", default="OPLS"
            ),
            SkillParameter(
                "temperatures",
                "list",
                "Temperatures to sample (K)",
                default=[250, 300, 350, 400, 450],
            ),
            SkillParameter("n_steps", "int", "MD steps per temperature", default=50000),
        ],
        steps=[
            SkillStep(
                name="build_amorphous_cell",
                tool="structure_tool",
                input_mapping={
                    "action": "'build_polymer_cell'",
                    "polymer_file": "$polymer_file",
                    "force_field": "$force_field",
                },
                output_key="cell",
            ),
            SkillStep(
                name="md_temperature_scan",
                tool="lammps_tool",
                input_mapping={
                    "action": "'temperature_scan'",
                    "structure_file": "$cell.file",
                    "force_field": "$force_field",
                    "temperatures": "$temperatures",
                    "n_steps": "$n_steps",
                },
                output_key="md",
            ),
            SkillStep(
                name="fit_tg",
                tool="evaluation_tool",
                input_mapping={
                    "action": "'tg_fit'",
                    "temperatures": "$temperatures",
                    "volumes": "$md.volumes",
                },
                output_key="tg",
            ),
        ],
        required_tools=["structure_tool", "lammps_tool", "evaluation_tool"],
        tags=["polymer", "glass_transition", "md"],
    )
)

MAGNETIC_ANISOTROPY = register_skill(
    SkillDefinition(
        name="magnetic_anisotropy",
        description="Calculate magnetocrystalline anisotropy energy (MAE) with SOC",
        category="computation",
        parameters=[
            SkillParameter(
                "structure_file", "str", "Magnetic structure", required=True
            ),
            SkillParameter(
                "spin_state",
                "str",
                "Ferromagnetic | antiferromagnetic",
                default="Ferromagnetic",
            ),
            SkillParameter("u_value", "float", "DFT+U Hubbard U", default=None),
            SkillParameter("soc", "bool", "Include spin-orbit coupling", default=True),
        ],
        steps=[
            SkillStep(
                name="relax_magnetic",
                tool="vasp_tool",
                input_mapping={
                    "action": "'relax'",
                    "structure_file": "$structure_file",
                    "spin_state": "$spin_state",
                    "hubbard_u": "$u_value",
                },
                output_key="relaxed",
            ),
            SkillStep(
                name="mae_calculation",
                tool="vasp_tool",
                input_mapping={
                    "action": "'mae'",
                    "structure_file": "$relaxed.relaxed_structure",
                    "soc": "$soc",
                },
                output_key="mae",
            ),
            SkillStep(
                name="validate_mae",
                tool="validate_tool",
                input_mapping={
                    "check_type": "'mae'",
                    "data": "$mae",
                },
                output_key="validation",
            ),
        ],
        required_tools=["vasp_tool", "validate_tool"],
        tags=["magnetism", "mae", "soc", "anisotropy"],
    )
)

CATALYSIS_VOLCANO = register_skill(
    SkillDefinition(
        name="catalysis_volcano",
        description="Build a volcano plot from adsorption energies for a catalytic reaction",
        category="analysis",
        parameters=[
            SkillParameter(
                "candidates", "list", "List of surface structure files", required=True
            ),
            SkillParameter("adsorbate", "str", "Adsorbate species", required=True),
            SkillParameter("descriptor", "str", "CO | OH | O | N", default="O"),
            SkillParameter("reaction", "str", "ORR | OER | HER | CO2RR", default="ORR"),
        ],
        steps=[
            SkillStep(
                name="adsorption_energies",
                tool="vasp_tool",
                input_mapping={
                    "action": "'adsorption_batch'",
                    "slabs": "$candidates",
                    "adsorbate": "$adsorbate",
                    "descriptor": "$descriptor",
                },
                output_key="adsorption",
            ),
            SkillStep(
                name="volcano_plot",
                tool="evaluation_tool",
                input_mapping={
                    "action": "'volcano_plot'",
                    "descriptor_energies": "$adsorption.descriptor_energies",
                    "reaction": "$reaction",
                },
                output_key="volcano",
            ),
            SkillStep(
                name="rank_candidates",
                tool="evaluation_tool",
                input_mapping={
                    "action": "'rank'",
                    "data": "$volcano.near_peak",
                    "criteria": {"lower_is_better": True, "property": "overpotential"},
                },
                output_key="ranking",
            ),
        ],
        required_tools=["vasp_tool", "evaluation_tool"],
        tags=["catalysis", "volcano", "adsorption", "screening"],
    )
)

FETCH_REFERENCE_STRUCTURE = register_skill(
    SkillDefinition(
        name="fetch_reference_structure",
        description="Fetch a reference crystal structure from Materials Project or OQMD",
        category="data",
        parameters=[
            SkillParameter(
                "query",
                "str",
                "Formula (e.g. 'SiO2') or database id (e.g. 'mp-149', OQMD entry id)",
                required=True,
            ),
            SkillParameter(
                "action",
                "str",
                "Tool action: 'mp_structure' or 'oqmd_structure'",
                default="mp_structure",
            ),
            SkillParameter(
                "output_format",
                "str",
                "Save format: json, cif, poscar",
                default="poscar",
            ),
            SkillParameter(
                "output_file",
                "str",
                "Optional output filename",
                default=None,
            ),
        ],
        steps=[
            SkillStep(
                name="query_database",
                tool="materials_database_tool",
                input_mapping={
                    "action": "$action",
                    "query": "$query",
                    "output_format": "$output_format",
                    "output_file": "$output_file",
                },
                output_key="db_result",
            ),
        ],
        required_tools=["materials_database_tool"],
        tags=["database", "materials_project", "oqmd", "structure"],
    )
)

ACTIVE_LEARNING_SCREENING = register_skill(
    SkillDefinition(
        name="active_learning_screening",
        description="Bayesian active-learning loop: fit GP, predict candidates, suggest next experiment",
        category="computation",
        parameters=[
            SkillParameter(
                "training_X",
                "list",
                "Training descriptor matrix (list of feature vectors)",
                required=True,
            ),
            SkillParameter(
                "training_y", "list", "Training target values", required=True
            ),
            SkillParameter(
                "candidates_X",
                "list",
                "Candidate descriptor matrix",
                required=True,
            ),
            SkillParameter(
                "maximize",
                "bool",
                "If True, maximize the target; otherwise minimize",
                default=False,
            ),
            SkillParameter("length_scale", "float", "GP length scale", default=1.0),
            SkillParameter("sigma_f", "float", "GP signal variance", default=1.0),
            SkillParameter("sigma_n", "float", "GP observation noise", default=1e-5),
        ],
        steps=[
            SkillStep(
                name="fit_gp",
                tool="gp_tool",
                input_mapping={
                    "action": "'fit'",
                    "X": "$training_X",
                    "y": "$training_y",
                    "length_scale": "$length_scale",
                    "sigma_f": "$sigma_f",
                    "sigma_n": "$sigma_n",
                },
                output_key="gp_fit",
            ),
            SkillStep(
                name="predict_candidates",
                tool="gp_tool",
                input_mapping={
                    "action": "'predict'",
                    "X": "$training_X",
                    "y": "$training_y",
                    "X_new": "$candidates_X",
                    "length_scale": "$length_scale",
                    "sigma_f": "$sigma_f",
                    "sigma_n": "$sigma_n",
                },
                output_key="predictions",
            ),
            SkillStep(
                name="suggest_next",
                tool="gp_tool",
                input_mapping={
                    "action": "'suggest'",
                    "X": "$training_X",
                    "y": "$training_y",
                    "X_new": "$candidates_X",
                    "maximize": "$maximize",
                    "length_scale": "$length_scale",
                    "sigma_f": "$sigma_f",
                    "sigma_n": "$sigma_n",
                },
                output_key="suggestion",
            ),
        ],
        required_tools=["gp_tool"],
        tags=["active_learning", "bayesian_optimization", "screening", "ml"],
    )
)

TOPOLOGICAL_GEOMETRY_ANALYSIS = register_skill(
    SkillDefinition(
        name="topological_geometry_analysis",
        description=(
            "Compute composition descriptors and generate a topological/geometric "
            "interpretation report"
        ),
        category="analysis",
        parameters=[
            SkillParameter(
                "formula",
                "str",
                "Chemical formula (e.g. 'SiO2')",
                required=True,
            ),
            SkillParameter(
                "output_path",
                "str",
                "Path for the generated report",
                default="topology_report.md",
            ),
        ],
        steps=[
            SkillStep(
                name="compute_descriptors",
                tool="descriptor_tool",
                input_mapping={
                    "action": "'composition'",
                    "formula": "$formula",
                },
                output_key="descriptors",
            ),
            SkillStep(
                name="search_topology_knowledge",
                tool="rag_tool",
                input_mapping={
                    "action": "'search'",
                    "query": "'topological data analysis geometric invariants materials'",
                    "top_k": "3",
                },
                output_key="topology_hints",
                on_failure="skip",
            ),
            SkillStep(
                name="generate_report",
                tool="report_tool",
                input_mapping={
                    "action": "'generate'",
                    "workflow_results": "$descriptors",
                    "style": "'brief'",
                    "output_path": "$output_path",
                },
                output_key="report",
            ),
        ],
        required_tools=["descriptor_tool", "rag_tool", "report_tool"],
        tags=["topology", "geometry", "descriptors", "report"],
    )
)

VISUALIZE_RESULTS = register_skill(
    SkillDefinition(
        name="visualize_results",
        description=(
            "Generate a figure from a benchmark, evolution, or exploration report"
        ),
        category="reporting",
        parameters=[
            SkillParameter(
                "report_path",
                "str",
                "Path to a JSON report file",
                required=True,
            ),
            SkillParameter(
                "action",
                "str",
                "Report type: benchmark, evolution, or exploration",
                required=True,
            ),
            SkillParameter(
                "output_path",
                "str",
                "Figure output path",
                required=True,
            ),
            SkillParameter(
                "plot_type",
                "str",
                "Plot subtype (action-specific)",
                default="auto",
            ),
        ],
        steps=[
            SkillStep(
                name="plot_report",
                tool="visualize_tool",
                input_mapping={
                    "action": "$action",
                    "report_path": "$report_path",
                    "output_path": "$output_path",
                    "plot_type": "$plot_type",
                },
                output_key="figure",
            ),
        ],
        required_tools=["visualize_tool"],
        tags=["visualization", "plot", "report", "multi_modal"],
    )
)

SYNTHESIS_PLANNING = register_skill(
    SkillDefinition(
        name="synthesis_planning",
        description=(
            "Load experimental data and recommend the next synthesis conditions "
            "to optimize a target property using GP active learning"
        ),
        category="experiment",
        parameters=[
            SkillParameter(
                "data_path",
                "str",
                "Path to CSV with historical experiments",
                required=True,
            ),
            SkillParameter(
                "target_column",
                "str",
                "Column name of the property to optimize",
                default="target",
            ),
            SkillParameter(
                "feature_columns",
                "list",
                "Input parameter column names (inferred if omitted)",
                default=None,
            ),
            SkillParameter(
                "bounds",
                "dict",
                "Parameter bounds for generating candidates",
                default=None,
            ),
            SkillParameter(
                "n_recommendations",
                "int",
                "Number of experiments to recommend",
                default=3,
            ),
            SkillParameter(
                "maximize",
                "bool",
                "If True, maximize the target; otherwise minimize",
                default=False,
            ),
            SkillParameter(
                "output_path",
                "str",
                "Path for the recommendation report",
                default="synthesis_recommendations.md",
            ),
        ],
        steps=[
            SkillStep(
                name="load_data",
                tool="active_learning_tool",
                input_mapping={
                    "action": "'load_csv'",
                    "data_path": "$data_path",
                    "target_column": "$target_column",
                    "feature_columns": "$feature_columns",
                },
                output_key="data_summary",
            ),
            SkillStep(
                name="recommend_experiments",
                tool="active_learning_tool",
                input_mapping={
                    "action": "'recommend'",
                    "data_path": "$data_path",
                    "target_column": "$target_column",
                    "feature_columns": "$feature_columns",
                    "bounds": "$bounds",
                    "n_recommendations": "$n_recommendations",
                    "maximize": "$maximize",
                },
                output_key="recommendations",
            ),
            SkillStep(
                name="generate_report",
                tool="report_tool",
                input_mapping={
                    "action": "'generate'",
                    "workflow_results": "$recommendations",
                    "style": "'brief'",
                    "output_path": "$output_path",
                },
                output_key="report",
            ),
        ],
        required_tools=["active_learning_tool", "report_tool"],
        tags=["active_learning", "synthesis", "experiment", "bayesian_optimization"],
    )
)

ML_POTENTIAL_PREDICTION = register_skill(
    SkillDefinition(
        name="ml_potential_prediction",
        description=(
            "Predict energy, forces, and stress for a structure using a "
            "machine-learning potential (MACE, CHGNet, or NEP)"
        ),
        category="computation",
        parameters=[
            SkillParameter(
                "backend",
                "str",
                "ML potential backend: mace, chgnet, or nep",
                required=True,
            ),
            SkillParameter(
                "structure_file",
                "str",
                "Path to structure file (POSCAR, CIF, XYZ)",
                required=True,
            ),
            SkillParameter(
                "model_path",
                "str",
                "Path to trained model; omit to use pretrained/default",
                default=None,
            ),
            SkillParameter(
                "action",
                "str",
                "predict or relax",
                default="predict",
            ),
            SkillParameter(
                "output_path",
                "str",
                "Path to save relaxed structure",
                default=None,
            ),
        ],
        steps=[
            SkillStep(
                name="run_ml_potential",
                tool="ml_potential_tool",
                input_mapping={
                    "backend": "$backend",
                    "action": "$action",
                    "structure_file": "$structure_file",
                    "model_path": "$model_path",
                    "output_path": "$output_path",
                },
                output_key="ml_result",
            ),
        ],
        required_tools=["ml_potential_tool"],
        tags=["ml_potential", "mace", "chgnet", "nep", "md", "surrogate"],
    )
)

CHARACTERIZATION_ANALYSIS = register_skill(
    SkillDefinition(
        name="characterization_analysis",
        description=(
            "Analyze experimental characterization data (XRD, Raman/IR, PDF, "
            "TEM, XPS) to extract peaks and structural information"
        ),
        category="analysis",
        parameters=[
            SkillParameter(
                "action",
                "str",
                "Analysis type: xrd_peak_detect, spectroscopy_peak_fit, xps_peak_fit, pdf_fit, or tem_image_analysis",
                required=True,
            ),
            SkillParameter(
                "data_path",
                "str",
                "Path to CSV/JSON data file",
                required=True,
            ),
            SkillParameter(
                "output_path",
                "str",
                "Path to save annotated peak list JSON",
                default="characterization_results.json",
            ),
        ],
        steps=[
            SkillStep(
                name="analyze_data",
                tool="characterization_tool",
                input_mapping={
                    "action": "$action",
                    "data_path": "$data_path",
                    "output_path": "$output_path",
                    "parameters": "{}",
                },
                output_key="peaks",
            ),
            SkillStep(
                name="generate_report",
                tool="report_tool",
                input_mapping={
                    "action": "'generate'",
                    "workflow_results": "$peaks",
                    "style": "'brief'",
                    "output_path": "$output_path",
                },
                output_key="report",
                on_failure="skip",
            ),
        ],
        required_tools=["characterization_tool"],
        tags=["characterization", "xrd", "spectroscopy", "peaks", "experiment"],
    )
)

AUTORESEARCH_WORKFLOW = register_skill(
    SkillDefinition(
        name="autoresearch_workflow",
        description=(
            "Drive an AutoResearch workspace: initialize, prepare data, establish "
            "a baseline, then run an autonomous edit/train/ratchet loop."
        ),
        category="automation",
        parameters=[
            SkillParameter(
                "workspace",
                "str",
                "Path to the autoresearch workspace",
                required=True,
            ),
            SkillParameter(
                "branch",
                "str",
                "Git branch for the experiment run",
                default="",
            ),
            SkillParameter(
                "iterations",
                "int",
                "Number of autonomous loop iterations",
                default=3,
            ),
            SkillParameter(
                "timeout",
                "int",
                "Seconds to wait for each training run",
                default=600,
            ),
            SkillParameter(
                "user_hint",
                "str",
                "Hint to guide the agent's edits",
                default="",
            ),
        ],
        steps=[
            SkillStep(
                name="init_workspace",
                tool="autoresearch_tool",
                input_mapping={
                    "action": "'init_workspace'",
                    "workspace": "$workspace",
                    "branch": "$branch",
                },
                output_key="workspace_info",
                on_failure="abort",
            ),
            SkillStep(
                name="prepare_data",
                tool="autoresearch_tool",
                input_mapping={
                    "action": "'prepare'",
                    "workspace": "$workspace",
                },
                output_key="prepare_result",
                on_failure="abort",
            ),
            SkillStep(
                name="baseline",
                tool="autoresearch_tool",
                input_mapping={
                    "action": "'run_experiment'",
                    "workspace": "$workspace",
                    "timeout": "$timeout",
                },
                output_key="baseline_result",
                on_failure="abort",
            ),
            SkillStep(
                name="autonomous_loop",
                tool="autoresearch_tool",
                input_mapping={
                    "action": "'loop'",
                    "workspace": "$workspace",
                    "max_iterations": "$iterations",
                    "timeout": "$timeout",
                    "user_hint": "$user_hint",
                },
                output_key="loop_result",
                on_failure="skip",
            ),
        ],
        required_tools=["autoresearch_tool"],
        tags=["autoresearch", "ml", "experiment", "loop", "agent"],
    )
)


# --- Research Skills ---

# 声明式壳: 真正的 LLM 推理 (假设生成 / 模板映射) 在 hypothesis_generator_tool 里.
# 这里的 steps 把"文献综述 → 科学假设 → 可执行 workflow"的链路显式声明出来,
# 非关键步骤 on_failure=continue, 一处失败不阻断整条链.
HYPOTHESIS_GENERATOR = register_skill(
    SkillDefinition(
        name="hypothesis_generator",
        description=(
            "From literature review to executable workflow: generate scientific "
            "hypotheses from literature gaps and map to workflow templates"
        ),
        category="research",
        parameters=[
            SkillParameter(
                "research_topic",
                "str",
                "研究主题, 如 'GaN p-type doping efficiency'",
                required=True,
            ),
            SkillParameter(
                "literature_query",
                "str",
                "文献检索 query, 留空则用 research_topic",
                default=None,
            ),
            SkillParameter(
                "max_hypotheses", "int", "最多生成几个假设", default=3
            ),
            SkillParameter(
                "target_workflow",
                "str",
                "指定 workflow 模板名 (standard_dft/aimd/defect/surface/"
                "ml_potential/...), 不指定则自动选",
                default=None,
            ),
            SkillParameter(
                "auto_execute",
                "bool",
                "是否自动执行生成的 workflow",
                default=False,
            ),
        ],
        steps=[
            # 1. 检索文献. 壳里直接拿 research_topic 当 query;
            #    hypothesis_generator_tool 会尊重 literature_query.
            SkillStep(
                name="search_literature",
                tool="web_search_tool",
                input_mapping={"query": "$research_topic", "max_results": "10"},
                output_key="literature_summary",
                on_failure="continue",
            ),
            # 2. 识别研究空白, 喂规则分析
            SkillStep(
                name="identify_gaps",
                tool="gap_analysis_tool",
                input_mapping={
                    "action": "'analyze_gaps'",
                    "topic": "$research_topic",
                    "papers": "$literature_summary.results",
                },
                output_key="research_gaps",
                on_failure="continue",
            ),
            # 3. 生成假设. 壳里退化为 design_plan_tool 提一个计划占位,
            #    真正的 statement/rationale/testable_prediction/required_data
            #    由 hypothesis_generator_tool 的 LLM 推理产出.
            SkillStep(
                name="generate_hypotheses",
                tool="design_plan_tool",
                input_mapping={
                    "action": "'propose'",
                    "goal": "$research_topic",
                    "expected_output": "$research_gaps.summary",
                },
                output_key="hypotheses_list",
                on_failure="continue",
            ),
            # 4. 映射到 workflow 模板. 壳里用 code_tool 占位,
            #    真正的 template_name/args/expected_observable/falsification_criterion
            #    由 hypothesis_generator_tool 产出.
            SkillStep(
                name="map_to_workflow",
                tool="code_tool",
                input_mapping={
                    "action": "'generate'",
                    "code": "'# map hypotheses to workflow template'",
                },
                output_key="workflow_proposals",
                on_failure="continue",
            ),
            # 5. (可选) 自动执行第一个 workflow; auto_execute=False 时跳过, 留给用户选
            SkillStep(
                name="execute_workflow",
                tool="bash_tool",
                input_mapping={
                    "action": "'run'",
                    "command": "['echo', 'pending workflow execution']",
                },
                output_key="execution_result",
                condition="auto_execute == True",
                on_failure="continue",
            ),
        ],
        required_tools=[
            "web_search_tool",
            "gap_analysis_tool",
            "design_plan_tool",
            "code_tool",
            "bash_tool",
        ],
        tags=["hypothesis", "literature", "research", "workflow_design"],
    )
)


# 材料科研版 autoresearch —— 仿 Karpathy autoresearch, 但 ratchet 对象是物理量.
# 声明式壳只有一个 step: 全部逻辑 (LLM 提参 → 跑 VASP → 抠指标 → ratchet) 在
# materials_autoresearch_tool 里. 这里只负责把参数透传过去.
MATERIALS_AUTORESEARCH = register_skill(
    SkillDefinition(
        name="materials_autoresearch",
        description=(
            "Materials science research loop: LLM proposes parameters → run DFT/MD "
            "→ extract metric → ratchet"
        ),
        category="research",
        parameters=[
            SkillParameter(
                "research_goal",
                "str",
                "研究目标, 如 'minimize formation energy of Li7La3Zr2O12'",
                required=True,
            ),
            SkillParameter(
                "ratchet_metric",
                "str",
                "ratchet 指标: formation_energy / band_gap / conductivity / "
                "defect_formation_energy / elastic_modulus",
                required=True,
            ),
            SkillParameter(
                "ratchet_direction",
                "str",
                "minimize 或 maximize",
                default="minimize",
            ),
            SkillParameter(
                "initial_structure",
                "str",
                "初始结构文件路径 (POSCAR/CIF), 留空则用工作目录下现成的 POSCAR",
                default=None,
            ),
            SkillParameter(
                "workflow_template",
                "str",
                "用哪个 workflow 模板跑实验 (standard_dft/aimd/defect/surface/ml_potential/...)",
                default="standard_dft",
            ),
            SkillParameter(
                "max_iterations", "int", "最多迭代几轮", default=10
            ),
            SkillParameter(
                "convergence_threshold",
                "float",
                "收敛阈值, 最近 3 轮指标波动小于该值就停; 留空则跑到 max_iterations",
                default=None,
            ),
            SkillParameter(
                "parameter_space",
                "dict",
                "LLM 可以调的参数空间, 如 "
                "{'encut': [400, 500, 600], 'kpoints': ['4 4 4', '6 6 6'], 'ismear': [0, -5]}",
                default=None,
            ),
            SkillParameter(
                "record_history", "bool", "是否记录迭代历史", default=True
            ),
            SkillParameter(
                "work_dir",
                "str",
                "工作根目录, 留空则用 context.workspace 下的 materials_autoresearch/",
                default=None,
            ),
            SkillParameter(
                "walltime_hours", "int", "单轮计算墙钟上限 (小时)", default=24
            ),
        ],
        steps=[
            SkillStep(
                name="materials_research_loop",
                tool="materials_autoresearch_tool",
                input_mapping={
                    "research_goal": "$research_goal",
                    "ratchet_metric": "$ratchet_metric",
                    "ratchet_direction": "$ratchet_direction",
                    "initial_structure": "$initial_structure",
                    "workflow_template": "$workflow_template",
                    "max_iterations": "$max_iterations",
                    "convergence_threshold": "$convergence_threshold",
                    "parameter_space": "$parameter_space",
                    "record_history": "$record_history",
                    "work_dir": "$work_dir",
                    "walltime_hours": "$walltime_hours",
                },
                output_key="research_loop_result",
                on_failure="abort",
            ),
        ],
        required_tools=["materials_autoresearch_tool"],
        tags=["autoresearch", "materials", "ratchet", "research", "loop"],
    )
)


# --- Meta Skills ---

# 场景工具选择器: LLM 识别用户意图后, 调一次 scenario_tool 拿到该场景的
# 推荐工具集 + 调用链 + workflow 模板, 不用逐个挑工具, 减少 token 消耗.
# 壳只做参数透传, 真正的 LLM 匹配 + 关键词兜底逻辑在 scenario_tool 里.
SCENARIO_TOOL_SELECTOR = register_skill(
    SkillDefinition(
        name="scenario_tool_selector",
        description="Auto-select tools based on scenario description",
        category="meta",
        parameters=[
            SkillParameter(
                "scenario",
                "str",
                "用户场景描述, 如 '优化 Si 结构' / '调研高熵合金文献' / '审查论文'",
                required=True,
            ),
        ],
        steps=[
            SkillStep(
                name="select_scenario_tools",
                tool="scenario_tool",
                input_mapping={"scenario": "$scenario"},
                output_key="scenario_tools",
            ),
        ],
        required_tools=["scenario_tool"],
        tags=["meta", "scenario", "tool_selection", "routing"],
    )
)
