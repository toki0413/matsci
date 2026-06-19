"""
Solid Mechanics / FEA Workflow Templates for Huginn.

Finite element analysis workflows for structural mechanics,
crystal plasticity, and multiscale deformation modeling.
"""

from __future__ import annotations

from typing import Any

from huginn.workflows.templates import register_template

# ---------------------------------------------------------------------------
# Workflow 1: Structural Analysis Pipeline
# ---------------------------------------------------------------------------


def structural_analysis_pipeline(
    geometry_file: str,
    software: str = "ABAQUS",
    analysis_type: str = "static",
    material_model: str = "elastic",
) -> list[dict[str, Any]]:
    """Complete structural analysis pipeline: preprocess → solve → post-process.

    Args:
        geometry_file: CAD or mesh file (.stp, .inp, .cae)
        software: FEA software (ABAQUS, ANSYS, COMSOL)
        analysis_type: "static", "dynamic", "modal", "buckling", "thermal"
        material_model: "elastic", "plastic", "hyperelastic", "creep"
    """
    stages = []

    # Stage 1: Mesh generation and quality check
    stages.append(
        {
            "name": "mesh_generation",
            "tool": "abaqus_tool" if software.lower() == "abaqus" else "fea_tool",
            "action": "mesh",
            "params": {
                "geometry_file": geometry_file,
                "software": software,
                "element_type": "C3D8R" if analysis_type == "static" else "C3D10",
                "mesh_density": "convergence_study",
                "quality_thresholds": {
                    "aspect_ratio": 10.0,
                    "skewness": 0.6,
                    "min_angle": 15.0,
                },
            },
        }
    )

    # Stage 2: Material property assignment
    mat_params = {
        "elastic": {"type": "linear_elastic", "E": 210e9, "nu": 0.3},
        "plastic": {
            "type": "von_mises",
            "E": 210e9,
            "nu": 0.3,
            "yield_stress": 250e6,
            "hardening": "isotropic",
        },
        "hyperelastic": {"type": "neo_hookean", "C10": 0.5e6, "D1": 2e-9},
        "creep": {"type": "norton", "A": 1e-35, "n": 5.0, "m": 0.0},
    }
    stages.append(
        {
            "name": "material_assignment",
            "tool": "abaqus_tool" if software.lower() == "abaqus" else "fea_tool",
            "action": "assign_material",
            "params": {
                "software": software,
                "material_model": material_model,
                "properties": mat_params.get(material_model, mat_params["elastic"]),
            },
            "depends_on": ["mesh_generation"],
        }
    )

    # Stage 3: Boundary conditions and loading
    stages.append(
        {
            "name": "boundary_conditions",
            "tool": "abaqus_tool" if software.lower() == "abaqus" else "fea_tool",
            "action": "apply_bc",
            "params": {
                "software": software,
                "displacement_bc": {"surface": "fixed_end", "DOF": "ALL"},
                "load": {
                    "type": "pressure",
                    "magnitude": 1e6,
                    "surface": "loaded_surface",
                },
            },
            "depends_on": ["material_assignment"],
        }
    )

    # Stage 4: Solve
    solver_settings = {
        "static": {"procedure": "Static", "nlgeom": False, "max_inc": 100},
        "dynamic": {
            "procedure": "Dynamic, Explicit",
            "time_period": 1.0,
            "mass_scaling": None,
        },
        "modal": {"procedure": "Frequency", "n_modes": 10},
        "buckling": {"procedure": "Buckle", "n_modes": 5},
        "thermal": {"procedure": "Heat transfer", "steady_state": True},
    }
    stages.append(
        {
            "name": "solve",
            "tool": "abaqus_tool" if software.lower() == "abaqus" else "fea_tool",
            "action": "solve",
            "params": {
                "software": software,
                "analysis_type": analysis_type,
                "solver_settings": solver_settings.get(
                    analysis_type, solver_settings["static"]
                ),
                "parallel": {"cpus": 4, "domains": 4},
            },
            "depends_on": ["boundary_conditions"],
        }
    )

    # Stage 5: Post-processing
    stages.append(
        {
            "name": "post_process",
            "tool": "rag_tool",
            "action": "search",
            "params": {
                "query": f"{software} {analysis_type} post-processing stress strain extraction",
                "top_k": 3,
            },
            "depends_on": ["solve"],
        }
    )

    return stages


# ---------------------------------------------------------------------------
# Workflow 2: Crystal Plasticity FEM Pipeline
# ---------------------------------------------------------------------------


def crystal_plasticity_pipeline(
    microstructure_file: str,
    software: str = "DAMASK",
    loading_mode: str = "tensile",
) -> list[dict[str, Any]]:
    """Crystal plasticity finite element method (CPFEM) workflow.

    Args:
        microstructure_file: Microstructure descriptor (Voronoi tessellation, EBSD data, etc.)
        software: "DAMASK", "ABAQUS+UMAT", "CPFEM_custom"
        loading_mode: "tensile", "compressive", "shear", "cyclic"
    """
    stages = []

    # Stage 1: Microstructure preparation
    stages.append(
        {
            "name": "microstructure_prep",
            "tool": "structure_tool",
            "action": "process_microstructure",
            "params": {
                "input_file": microstructure_file,
                "output_format": (
                    "damask_grid" if software.lower() == "damask" else "abaqus_inp"
                ),
                "resolution": 64,  # Grid points per dimension
                "phases": ["fcc", "bcc"],  # Detect from input
            },
        }
    )

    # Stage 2: Crystal orientation assignment (texture)
    stages.append(
        {
            "name": "texture_assignment",
            "tool": "structure_tool",
            "action": "assign_orientations",
            "params": {
                "texture_type": "random",  # or "rolling", "recrystallized"
                "orientation_format": "euler_bunge",
            },
            "depends_on": ["microstructure_prep"],
        }
    )

    # Stage 3: Constitutive model setup
    stages.append(
        {
            "name": "constitutive_setup",
            "tool": "abaqus_tool" if "abaqus" in software.lower() else "cpfem_tool",
            "action": "setup_crystal_plasticity",
            "params": {
                "software": software,
                "slip_systems": {"fcc": "{111}<110>", "bcc": "{110}<111>"},
                "hardening_model": "voce",
                "hardening_params": {
                    "tau0": 50e6,
                    "tau_sat": 200e6,
                    "h0": 500e6,
                    "a": 2.0,
                },
            },
            "depends_on": ["texture_assignment"],
        }
    )

    # Stage 4: Boundary conditions (periodic or Dirichlet)
    stages.append(
        {
            "name": "cpfem_bc",
            "tool": "abaqus_tool" if "abaqus" in software.lower() else "cpfem_tool",
            "action": "apply_periodic_bc",
            "params": {
                "software": software,
                "loading": {
                    "mode": loading_mode,
                    "strain_rate": 1e-3,
                    "max_strain": (
                        0.3 if loading_mode in ("tensile", "compressive") else 0.1
                    ),
                },
            },
            "depends_on": ["constitutive_setup"],
        }
    )

    # Stage 5: Solve
    stages.append(
        {
            "name": "cpfem_solve",
            "tool": "abaqus_tool" if "abaqus" in software.lower() else "cpfem_tool",
            "action": "solve",
            "params": {
                "software": software,
                "solver": "spectral" if software.lower() == "damask" else "implicit",
                "max_iter": 100,
                "tolerance": 1e-6,
            },
            "depends_on": ["cpfem_bc"],
        }
    )

    # Stage 6: Post-processing: stress-strain, texture evolution, slip activity
    stages.append(
        {
            "name": "cpfem_post",
            "tool": "rag_tool",
            "action": "search",
            "params": {
                "query": f"{software} crystal plasticity post-processing stress strain texture slip activity",
                "top_k": 5,
            },
            "depends_on": ["cpfem_solve"],
        }
    )

    return stages


# ---------------------------------------------------------------------------
# Workflow 3: Fracture Mechanics Analysis
# ---------------------------------------------------------------------------


def fracture_mechanics_pipeline(
    cracked_geometry_file: str,
    software: str = "ABAQUS",
    analysis_type: str = "J_integral",
) -> list[dict[str, Any]]:
    """Fracture mechanics analysis: J-integral, CTOD, or stress intensity.

    Args:
        cracked_geometry_file: Geometry with crack (pre-meshed or CAD)
        software: FEA software
        analysis_type: "J_integral", "CTOD", "SIF", "cohesive_zone"
    """
    stages = []

    # Stage 1: Crack tip mesh refinement
    stages.append(
        {
            "name": "crack_tip_mesh",
            "tool": "abaqus_tool" if software.lower() == "abaqus" else "fea_tool",
            "action": "mesh_crack",
            "params": {
                "software": software,
                "crack_tip_elements": (
                    "CPE6" if analysis_type in ("J_integral", "SIF") else "COH2D4"
                ),
                "refinement_radius": 0.5,
                "collapse_nodes": analysis_type in ("J_integral", "SIF"),
            },
        }
    )

    # Stage 2: Material (elastic-plastic for J-integral/CTOD)
    stages.append(
        {
            "name": "fracture_material",
            "tool": "abaqus_tool" if software.lower() == "abaqus" else "fea_tool",
            "action": "assign_material",
            "params": {
                "software": software,
                "material_model": "elastic_plastic",
                "E": 210e9,
                "nu": 0.3,
                "yield_stress": 350e6,
                "fracture_toughness": 50e6,  # K_IC in Pa·sqrt(m)
            },
            "depends_on": ["crack_tip_mesh"],
        }
    )

    # Stage 3: Apply loading and solve
    stages.append(
        {
            "name": "fracture_solve",
            "tool": "abaqus_tool" if software.lower() == "abaqus" else "fea_tool",
            "action": "solve",
            "params": {
                "software": software,
                "analysis_type": analysis_type,
                "contour_integrals": (
                    [1, 2, 3, 4, 5] if analysis_type == "J_integral" else None
                ),
            },
            "depends_on": ["fracture_material"],
        }
    )

    # Stage 4: Extract fracture parameters
    stages.append(
        {
            "name": "fracture_post",
            "tool": "rag_tool",
            "action": "search",
            "params": {
                "query": f"{software} {analysis_type} extraction crack tip field",
                "top_k": 3,
            },
            "depends_on": ["fracture_solve"],
        }
    )

    return stages


# Register all FEA templates
register_template("structural_analysis", structural_analysis_pipeline)
register_template("crystal_plasticity", crystal_plasticity_pipeline)
register_template("fracture_mechanics", fracture_mechanics_pipeline)
