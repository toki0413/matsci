"""Composite skills — multi-tool research recipes.

These chain individual tool calls into reusable workflows, like
'band structure analysis' which runs relax -> SCF -> band -> DOS.
Each composite skill carries default parameters tuned for common cases
and tags that describe the material system it applies to.
"""

from __future__ import annotations

from huginn.skills.base import (
    SkillDefinition,
    SkillParameter,
    SkillStep,
)
from huginn.skills.registry import register_skill


def _cond(output_key: str, field: str = "converged") -> str:
    # safe_eval rejects .get() and attribute access, so we use subscript with
    # an in-guard. IfExp short-circuits (unlike BoolOp), so the subscript only
    # fires when the key is actually present — no KeyError.
    return f"{output_key}['{field}'] if '{field}' in {output_key} else False"


def _make_band_structure_analysis() -> SkillDefinition:
    return SkillDefinition(
        name="band_structure_analysis",
        description="Complete band structure + DOS workflow: relax -> SCF -> band -> DOS",
        category="computation",
        parameters=[
            SkillParameter("structure_file", "str", "Path to input structure (POSCAR/cif)"),
            SkillParameter("encut", "float", "Plane-wave cutoff energy (eV)", default=520, required=False),
            SkillParameter("kpoints_mesh", "str", "K-point mesh for SCF", default="Gamma 12 12 12", required=False),
            SkillParameter("kpoints_band", "str", "K-point path for band structure", default="Line-mode G-X-W-K-G-L-U-W-L", required=False),
            SkillParameter("ediff", "float", "Electronic convergence", default=1e-6, required=False),
            SkillParameter("ediffg", "float", "Ionic convergence (eV/Å)", default=-0.01, required=False),
            SkillParameter("ispin", "int", "Spin polarization (1=no, 2=yes)", default=1, required=False),
            SkillParameter("functional", "str", "XC functional", default="PBE", required=False),
        ],
        steps=[
            SkillStep(
                name="structure_relaxation",
                tool="vasp_tool",
                input_mapping={
                    "action": "relax",
                    "structure": "$structure_file",
                    "encut": "$encut",
                    "ediff": "$ediff",
                    "ediffg": "$ediffg",
                    "kpoints": "$kpoints_mesh",
                    "ispin": "$ispin",
                    "functional": "$functional",
                },
                output_key="relax_result",
                validation=_cond("relax_result"),
                on_failure="abort",
            ),
            SkillStep(
                name="scf_calculation",
                tool="vasp_tool",
                input_mapping={
                    "action": "static",
                    "structure": "$relax_result.convc",
                    "encut": "$encut",
                    "ediff": "$ediff",
                    "kpoints": "$kpoints_mesh",
                    "ispin": "$ispin",
                    "functional": "$functional",
                },
                output_key="scf_result",
                condition=_cond("relax_result"),
                validation=_cond("scf_result"),
                on_failure="abort",
            ),
            SkillStep(
                name="band_calculation",
                tool="vasp_tool",
                input_mapping={
                    "action": "band",
                    "structure": "$scf_result.contin",
                    "encut": "$encut",
                    "kpoints": "$kpoints_band",
                    "icharg": "11",
                    "functional": "$functional",
                },
                output_key="band_result",
                condition=_cond("scf_result"),
            ),
            SkillStep(
                name="dos_calculation",
                tool="vasp_tool",
                input_mapping={
                    "action": "dos",
                    "structure": "$scf_result.contin",
                    "encut": "$encut",
                    "kpoints": "$kpoints_mesh",
                    "icharg": "11",
                    "functional": "$functional",
                },
                output_key="dos_result",
                condition=_cond("scf_result"),
            ),
        ],
        required_tools=["vasp_tool"],
        estimated_cost={"cpu_hours": 8, "memory_gb": 16, "disk_gb": 5},
        tags=["bandgap", "dos", "electronic", "vasp", "dft"],
        metadata={"applicable_systems": ["crystalline", "bulk", "2d-material"]},
    )


def _make_mechanical_properties() -> SkillDefinition:
    return SkillDefinition(
        name="mechanical_properties",
        description="Elastic constants and polycrystalline moduli: relax -> elastic tensor -> bulk/shear/Young",
        category="computation",
        parameters=[
            SkillParameter("structure_file", "str", "Path to input structure (POSCAR/cif)"),
            SkillParameter("encut", "float", "Plane-wave cutoff energy (eV)", default=520, required=False),
            SkillParameter("kpoints_mesh", "str", "K-point mesh for SCF/elastic", default="Gamma 8 8 8", required=False),
            SkillParameter("ediff", "float", "Electronic convergence", default=1e-6, required=False),
            SkillParameter("ediffg", "float", "Ionic convergence (eV/Å)", default=-0.01, required=False),
            SkillParameter("ispin", "int", "Spin polarization (1=no, 2=yes)", default=1, required=False),
            SkillParameter("functional", "str", "XC functional", default="PBE", required=False),
            SkillParameter("n_deformations", "int", "Number of strain deformations", default=6, required=False),
            SkillParameter("d_strain", "float", "Strain magnitude (fraction)", default=0.005, required=False),
        ],
        steps=[
            SkillStep(
                name="structure_relaxation",
                tool="vasp_tool",
                input_mapping={
                    "action": "relax",
                    "structure": "$structure_file",
                    "encut": "$encut",
                    "ediff": "$ediff",
                    "ediffg": "$ediffg",
                    "kpoints": "$kpoints_mesh",
                    "ispin": "$ispin",
                    "functional": "$functional",
                },
                output_key="relax_result",
                validation=_cond("relax_result"),
                on_failure="abort",
            ),
            SkillStep(
                name="elastic_constants",
                tool="vasp_tool",
                input_mapping={
                    "action": "elastic",
                    "structure": "$relax_result.convc",
                    "encut": "$encut",
                    "ediff": "$ediff",
                    "kpoints": "$kpoints_mesh",
                    "ispin": "$ispin",
                    "functional": "$functional",
                    "n_deformations": "$n_deformations",
                    "d_strain": "$d_strain",
                },
                output_key="elastic_result",
                condition=_cond("relax_result"),
                validation=_cond("elastic_result", "tensor"),
                on_failure="abort",
            ),
            SkillStep(
                name="modulus_analysis",
                tool="mechanical_tool",
                input_mapping={
                    "action": "modulus",
                    "tensor": "$elastic_result.tensor",
                },
                output_key="modulus_result",
                condition=_cond("elastic_result", "tensor"),
                validation=_cond("modulus_result", "bulk_modulus"),
            ),
        ],
        required_tools=["vasp_tool", "mechanical_tool"],
        estimated_cost={"cpu_hours": 6, "memory_gb": 8, "disk_gb": 3},
        tags=["elastic", "bulk_modulus", "shear_modulus", "youngs_modulus", "mechanical", "vasp"],
        metadata={"applicable_systems": ["crystalline", "bulk", "alloy"]},
    )


def _make_md_pipeline() -> SkillDefinition:
    return SkillDefinition(
        name="md_pipeline",
        description="Classical MD pipeline: pack -> minimize -> NPT equilibrate -> production -> analyze",
        category="computation",
        parameters=[
            SkillParameter("input_molecules", "str", "Molecule SMILES or composition spec"),
            SkillParameter("potential_file", "str", "LAMMPS potential file path"),
            SkillParameter("n_molecules", "int", "Number of molecules in box", default=1000, required=False),
            SkillParameter("box_density", "float", "Target density (g/cm³)", default=1.0, required=False),
            SkillParameter("temperature", "float", "Target temperature (K)", default=300.0, required=False),
            SkillParameter("pressure", "float", "Target pressure (atm)", default=1.0, required=False),
            SkillParameter("timestep_fs", "float", "MD timestep (fs)", default=1.0, required=False),
            SkillParameter("n_steps_minimize", "int", "Minimization steps", default=10000, required=False),
            SkillParameter("n_steps_equilibration", "int", "NPT equilibration steps", default=100000, required=False),
            SkillParameter("n_steps_production", "int", "Production MD steps", default=500000, required=False),
        ],
        steps=[
            SkillStep(
                name="build_initial_config",
                tool="packing_tool",
                input_mapping={
                    "action": "pack",
                    "molecules": "$input_molecules",
                    "n_molecules": "$n_molecules",
                    "density": "$box_density",
                },
                output_key="packed",
                validation=_cond("packed", "structure"),
                on_failure="abort",
            ),
            SkillStep(
                name="energy_minimization",
                tool="lammps_tool",
                input_mapping={
                    "action": "minimize",
                    "structure": "$packed.structure",
                    "potential": "$potential_file",
                    "max_steps": "$n_steps_minimize",
                },
                output_key="min_result",
                condition=_cond("packed", "structure"),
                validation=_cond("min_result"),
                on_failure="abort",
            ),
            SkillStep(
                name="npt_equilibration",
                tool="lammps_tool",
                input_mapping={
                    "action": "npt",
                    "structure": "$min_result.structure",
                    "potential": "$potential_file",
                    "temperature": "$temperature",
                    "pressure": "$pressure",
                    "timestep": "$timestep_fs",
                    "n_steps": "$n_steps_equilibration",
                },
                output_key="npt_result",
                condition=_cond("min_result"),
                validation=_cond("npt_result", "equilibrated"),
                on_failure="abort",
            ),
            SkillStep(
                name="production_md",
                tool="lammps_tool",
                input_mapping={
                    "action": "production",
                    "structure": "$npt_result.structure",
                    "potential": "$potential_file",
                    "temperature": "$temperature",
                    "pressure": "$pressure",
                    "timestep": "$timestep_fs",
                    "n_steps": "$n_steps_production",
                },
                output_key="prod_result",
                condition=_cond("npt_result", "equilibrated"),
                validation=_cond("prod_result", "completed"),
                on_failure="abort",
            ),
            SkillStep(
                name="trajectory_analysis",
                tool="evaluation_tool",
                input_mapping={
                    "action": "md_analysis",
                    "trajectory": "$prod_result.trajectory",
                    "temperature": "$temperature",
                },
                output_key="analysis_result",
                condition=_cond("prod_result", "completed"),
            ),
        ],
        required_tools=["packing_tool", "lammps_tool", "evaluation_tool"],
        estimated_cost={"cpu_hours": 4, "memory_gb": 8, "disk_gb": 10},
        tags=["md", "lammps", "equilibration", "trajectory", "classical", "npt"],
        metadata={"applicable_systems": ["liquid", "polymer", "melt", "solution"]},
    )


def _make_molecule_screening() -> SkillDefinition:
    return SkillDefinition(
        name="molecule_screening",
        description="Molecular property screening: optimize -> single point -> HOMO/LUMO -> solubility",
        category="computation",
        parameters=[
            SkillParameter("smiles", "str", "Molecule SMILES string"),
            SkillParameter("basis_set", "str", "Basis set", default="6-31G*", required=False),
            SkillParameter("functional", "str", "DFT functional", default="B3LYP", required=False),
            SkillParameter("solvent", "str", "Implicit solvent (none = gas phase)", default="water", required=False),
            SkillParameter("charge", "int", "Net charge", default=0, required=False),
            SkillParameter("multiplicity", "int", "Spin multiplicity", default=1, required=False),
        ],
        steps=[
            SkillStep(
                name="molecule_optimization",
                tool="rdkit_tool",
                input_mapping={
                    "action": "optimize",
                    "smiles": "$smiles",
                },
                output_key="opt_result",
                validation=_cond("opt_result", "optimized"),
                on_failure="abort",
            ),
            SkillStep(
                name="single_point_energy",
                tool="gaussian_tool",
                input_mapping={
                    "action": "energy",
                    "geometry": "$opt_result.geometry",
                    "basis_set": "$basis_set",
                    "functional": "$functional",
                    "charge": "$charge",
                    "multiplicity": "$multiplicity",
                },
                output_key="sp_result",
                condition=_cond("opt_result", "optimized"),
                validation=_cond("sp_result", "energy"),
                on_failure="abort",
            ),
            SkillStep(
                name="orbital_analysis",
                tool="gaussian_tool",
                input_mapping={
                    "action": "orbitals",
                    "checkpoint": "$sp_result.checkpoint",
                    "basis_set": "$basis_set",
                    "functional": "$functional",
                },
                output_key="orbital_result",
                condition=_cond("sp_result", "energy"),
                validation=_cond("orbital_result", "homo"),
            ),
            SkillStep(
                name="solubility_prediction",
                tool="rdkit_tool",
                input_mapping={
                    "action": "solubility",
                    "smiles": "$smiles",
                    "solvent": "$solvent",
                },
                output_key="sol_result",
                condition=_cond("opt_result", "optimized"),
                validation=_cond("sol_result", "logp"),
            ),
        ],
        required_tools=["rdkit_tool", "gaussian_tool"],
        estimated_cost={"cpu_hours": 2, "memory_gb": 4, "disk_gb": 1},
        tags=["homo", "lumo", "bandgap", "solubility", "molecule", "screening", "dft"],
        metadata={"applicable_systems": ["molecular", "organic", "drug-like"]},
    )


def _make_phonon_analysis() -> SkillDefinition:
    return SkillDefinition(
        name="phonon_analysis",
        description="Phonon + thermal properties: relax -> DFPT force constants -> thermal analysis",
        category="computation",
        parameters=[
            SkillParameter("structure_file", "str", "Path to input structure (POSCAR/cif)"),
            SkillParameter("encut", "float", "Plane-wave cutoff energy (eV)", default=520, required=False),
            SkillParameter("kpoints_mesh", "str", "K-point mesh for SCF", default="Gamma 8 8 8", required=False),
            SkillParameter("qpoints_mesh", "str", "Q-point mesh for phonon", default="4 4 4", required=False),
            SkillParameter("ediff", "float", "Electronic convergence (tighter for DFPT)", default=1e-8, required=False),
            SkillParameter("ediffg", "float", "Ionic convergence (eV/Å)", default=-0.001, required=False),
            SkillParameter("ispin", "int", "Spin polarization (1=no, 2=yes)", default=1, required=False),
            SkillParameter("functional", "str", "XC functional", default="PBE", required=False),
            SkillParameter("t_min", "float", "Min temperature for thermal props (K)", default=0.0, required=False),
            SkillParameter("t_max", "float", "Max temperature for thermal props (K)", default=1000.0, required=False),
            SkillParameter("t_step", "float", "Temperature step (K)", default=10.0, required=False),
        ],
        steps=[
            SkillStep(
                name="structure_relaxation",
                tool="vasp_tool",
                input_mapping={
                    "action": "relax",
                    "structure": "$structure_file",
                    "encut": "$encut",
                    "ediff": "$ediff",
                    "ediffg": "$ediffg",
                    "kpoints": "$kpoints_mesh",
                    "ispin": "$ispin",
                    "functional": "$functional",
                },
                output_key="relax_result",
                validation=_cond("relax_result"),
                on_failure="abort",
            ),
            SkillStep(
                name="dfpt_phonon",
                tool="vasp_tool",
                input_mapping={
                    "action": "dfpt",
                    "structure": "$relax_result.convc",
                    "encut": "$encut",
                    "ediff": "$ediff",
                    "kpoints": "$kpoints_mesh",
                    "qpoints": "$qpoints_mesh",
                    "ispin": "$ispin",
                    "functional": "$functional",
                },
                output_key="phonon_result",
                condition=_cond("relax_result"),
                validation=_cond("phonon_result", "force_constants"),
                on_failure="abort",
            ),
            SkillStep(
                name="thermal_properties",
                tool="evaluation_tool",
                input_mapping={
                    "action": "thermal",
                    "force_constants": "$phonon_result.force_constants",
                    "structure": "$relax_result.convc",
                    "t_min": "$t_min",
                    "t_max": "$t_max",
                    "t_step": "$t_step",
                },
                output_key="thermal_result",
                condition=_cond("phonon_result", "force_constants"),
                validation=_cond("thermal_result", "free_energy"),
            ),
        ],
        required_tools=["vasp_tool", "evaluation_tool"],
        estimated_cost={"cpu_hours": 12, "memory_gb": 16, "disk_gb": 8},
        tags=["phonon", "dfpt", "thermal", "lattice_dynamics", "heat_capacity", "vasp"],
        metadata={"applicable_systems": ["crystalline", "bulk", "thermoelectric"]},
    )


def _make_fracture_assessment() -> SkillDefinition:
    """SEM crack detection → LEFM fracture assessment.

    Chains image_analysis_tool (defect_detect) to find cracks in microscopy
    images, then feeds the crack length into specialty_analysis_tool
    (fracture_lefm) to compute K_I / J / G and compare against K_IC.
    """
    return SkillDefinition(
        name="fracture_assessment",
        description=(
            "Fracture assessment from SEM imagery: detect cracks → "
            "compute stress intensity factor K_I and compare with K_IC"
        ),
        category="analysis",
        parameters=[
            SkillParameter("image_path", "str", "Path to SEM/OM image (PNG/JPG/TIFF)"),
            SkillParameter("pixel_size_nm", "float", "Pixel size in nm", default=1.0, required=False),
            SkillParameter("sensitivity", "float", "Detection sensitivity 0-1", default=0.5, required=False),
            SkillParameter("applied_stress", "float", "Applied stress σ (Pa)"),
            SkillParameter("youngs_modulus", "float", "Young's modulus E (Pa)", default=210e9, required=False),
            SkillParameter("poissons_ratio", "float", "Poisson's ratio ν", default=0.3, required=False),
            SkillParameter("k_ic", "float", "Fracture toughness K_IC (Pa·√m)", required=False),
            SkillParameter("crack_type", "str", "Crack type: edge/interior/surface", default="edge", required=False),
        ],
        steps=[
            SkillStep(
                name="crack_detection",
                tool="image_analysis_tool",
                input_mapping={
                    "image_path": "$image_path",
                    "action": "defect_detect",
                    "parameters": {
                        "defect_type": "crack",
                        "sensitivity": "$sensitivity",
                        "pixel_size_nm": "$pixel_size_nm",
                    },
                },
                output_key="defect_result",
                validation=_cond("defect_result", "measurements"),
                on_failure="abort",
            ),
            SkillStep(
                name="fracture_lefm",
                tool="specialty_analysis_tool",
                input_mapping={
                    "action": "fracture_lefm",
                    "crack_type": "$crack_type",
                    "crack_length": "$defect_result.measurements.defects[0].area_nm2",
                    "applied_stress": "$applied_stress",
                    "youngs_modulus": "$youngs_modulus",
                    "poissons_ratio": "$poissons_ratio",
                    "k_ic": "$k_ic",
                },
                output_key="fracture_result",
                condition=_cond("defect_result", "measurements"),
                validation=_cond("fracture_result", "stress_intensity_factor_ki"),
            ),
        ],
        required_tools=["image_analysis_tool", "specialty_analysis_tool"],
        estimated_cost={"cpu_hours": 0.1, "memory_gb": 2, "disk_gb": 1},
        tags=["fracture", "lefm", "crack", "sem", "kic", "safety_factor"],
        metadata={"applicable_systems": ["metallurgy", "ceramics", "composites"]},
    )


# Register at import time so SkillRegistry sees them as soon as the
# skills package is loaded (same pattern as presets.py).
BAND_STRUCTURE_ANALYSIS = register_skill(_make_band_structure_analysis())
MECHANICAL_PROPERTIES = register_skill(_make_mechanical_properties())
MD_PIPELINE = register_skill(_make_md_pipeline())
MOLECULE_SCREENING = register_skill(_make_molecule_screening())
PHONON_ANALYSIS = register_skill(_make_phonon_analysis())
FRACTURE_ASSESSMENT = register_skill(_make_fracture_assessment())

__all__ = [
    "BAND_STRUCTURE_ANALYSIS",
    "MECHANICAL_PROPERTIES",
    "MD_PIPELINE",
    "MOLECULE_SCREENING",
    "PHONON_ANALYSIS",
    "FRACTURE_ASSESSMENT",
]
