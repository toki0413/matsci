"""
Computational Fluid Dynamics Workflow Templates for MatSci-Agent.

CFD workflows for turbulent flows, multiphase systems, and
conjugate heat transfer in materials processing.
"""

from __future__ import annotations

from typing import Any

from matsci_agent.workflows.templates import register_template


# ---------------------------------------------------------------------------
# Workflow 1: Turbulent Flow Simulation
# ---------------------------------------------------------------------------

def turbulent_flow_pipeline(
    geometry_file: str,
    software: str = "OpenFOAM",
    turbulence_model: str = "kOmegaSST",
    reynolds_number: float = 1e5,
) -> list[dict[str, Any]]:
    """RANS/LES turbulent flow simulation pipeline.

    Args:
        geometry_file: CAD or mesh file
        software: "OpenFOAM", "Fluent", "COMSOL"
        turbulence_model: RANS or LES model name
        reynolds_number: Characteristic Reynolds number
    """
    stages = []

    # Stage 1: Mesh generation with inflation layers
    stages.append({
        "name": "cfd_mesh",
        "tool": "openfoam_tool" if software.lower() == "openfoam" else "cfd_tool",
        "action": "mesh",
        "params": {
            "software": software,
            "geometry_file": geometry_file,
            "mesh_type": "polyhedral" if software.lower() == "fluent" else "hex_dominant",
            "inflation_layers": 10 if reynolds_number > 1e4 else 5,
            "first_layer_height": "auto_yplus",
            "target_yplus": 1.0 if turbulence_model.startswith(("LES", "DES")) else 30.0,
            "growth_rate": 1.2,
        },
    })

    # Stage 2: Turbulence model and boundary condition setup
    turb_settings = {
        "kOmegaSST": {
            "model_type": "RAS",
            "RASModel": "kOmegaSST",
            "turbulence": "on",
            "printCoeffs": "on",
        },
        "kEpsilon": {
            "model_type": "RAS",
            "RASModel": "kEpsilon",
            "turbulence": "on",
        },
        "SpalartAllmaras": {
            "model_type": "RAS",
            "RASModel": "SpalartAllmaras",
            "turbulence": "on",
        },
        "Smagorinsky": {
            "model_type": "LES",
            "LESModel": "Smagorinsky",
            "delta": "cubeRootVol",
        },
        "WALE": {
            "model_type": "LES",
            "LESModel": "WALE",
            "delta": "cubeRootVol",
        },
    }

    stages.append({
        "name": "turbulence_setup",
        "tool": "openfoam_tool" if software.lower() == "openfoam" else "cfd_tool",
        "action": "setup_turbulence",
        "params": {
            "software": software,
            "turbulence_model": turbulence_model,
            "settings": turb_settings.get(turbulence_model, turb_settings["kOmegaSST"]),
            "inlet_turbulence": {
                "turbulence_intensity": 0.05,
                "turbulent_length_scale": 0.01,
            },
        },
        "depends_on": ["cfd_mesh"],
    })

    # Stage 3: Boundary conditions
    stages.append({
        "name": "cfd_boundary_conditions",
        "tool": "openfoam_tool" if software.lower() == "openfoam" else "cfd_tool",
        "action": "apply_bc",
        "params": {
            "software": software,
            "velocity_inlet": {"U": 10.0, "turbulence": "from_inlet_turbulence"},
            "pressure_outlet": {"p": 0.0, "type": "zeroGradient"},
            "wall": {"type": "noSlip", "roughness": 0.0},
        },
        "depends_on": ["turbulence_setup"],
    })

    # Stage 4: Solver settings and run
    solver_map = {
        "OpenFOAM": "simpleFoam" if reynolds_number > 1e3 else "icoFoam",
        "Fluent": "pressure_based",
        "COMSOL": "stationary",
    }
    stages.append({
        "name": "cfd_solve",
        "tool": "openfoam_tool" if software.lower() == "openfoam" else "cfd_tool",
        "action": "solve",
        "params": {
            "software": software,
            "solver": solver_map.get(software, "simpleFoam"),
            "steady_state": True,
            "max_iterations": 1000,
            "residual_threshold": 1e-6,
            "courant_number_max": 5.0 if software.lower() == "openfoam" else None,
        },
        "depends_on": ["cfd_boundary_conditions"],
    })

    # Stage 5: Post-processing and y+ check
    stages.append({
        "name": "cfd_post",
        "tool": "rag_tool",
        "action": "search",
        "params": {
            "query": f"{software} {turbulence_model} post-processing yplus velocity profiles",
            "top_k": 3,
        },
        "depends_on": ["cfd_solve"],
    })

    return stages


# ---------------------------------------------------------------------------
# Workflow 2: Multiphase Flow (VOF / Euler-Euler)
# ---------------------------------------------------------------------------

def multiphase_flow_pipeline(
    geometry_file: str,
    software: str = "OpenFOAM",
    multiphase_model: str = "VOF",
    phases: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Multiphase flow simulation: free surface or particle-laden flow.

    Args:
        geometry_file: Domain geometry
        software: CFD software
        multiphase_model: "VOF", "EulerEuler", "DPM"
        phases: List of phase properties [{"name": "water", "rho": 1000, "mu": 1e-3}, ...]
    """
    phases = phases or [
        {"name": "air", "rho": 1.225, "mu": 1.8e-5, "alpha": 0.5},
        {"name": "water", "rho": 998, "mu": 1e-3, "alpha": 0.5},
    ]
    stages = []

    # Stage 1: Mesh
    stages.append({
        "name": "multiphase_mesh",
        "tool": "openfoam_tool" if software.lower() == "openfoam" else "cfd_tool",
        "action": "mesh",
        "params": {
            "software": software,
            "geometry_file": geometry_file,
            "refinement_regions": ["interface"],
            "max_cell_size": 0.001,
        },
    })

    # Stage 2: Phase properties and interfacial models
    stages.append({
        "name": "phase_setup",
        "tool": "openfoam_tool" if software.lower() == "openfoam" else "cfd_tool",
        "action": "setup_phases",
        "params": {
            "software": software,
            "multiphase_model": multiphase_model,
            "phases": phases,
            "surface_tension": 0.072,  # N/m for air-water
            "interface_compression": 1.0 if multiphase_model == "VOF" else None,
        },
        "depends_on": ["multiphase_mesh"],
    })

    # Stage 3: Solve
    solver = {
        "VOF": "interFoam",
        "EulerEuler": "multiphaseEulerFoam",
        "DPM": "DPMFoam",
    }.get(multiphase_model, "interFoam")

    stages.append({
        "name": "multiphase_solve",
        "tool": "openfoam_tool" if software.lower() == "openfoam" else "cfd_tool",
        "action": "solve",
        "params": {
            "software": software,
            "solver": solver,
            "transient": True,
            "max_co": 1.0,
            "delta_t": 1e-4,
            "end_time": 1.0,
        },
        "depends_on": ["phase_setup"],
    })

    # Stage 4: Post-processing
    stages.append({
        "name": "multiphase_post",
        "tool": "rag_tool",
        "action": "search",
        "params": {
            "query": f"{software} {multiphase_model} interface tracking volume fraction post-processing",
            "top_k": 3,
        },
        "depends_on": ["multiphase_solve"],
    })

    return stages


# ---------------------------------------------------------------------------
# Workflow 3: Conjugate Heat Transfer
# ---------------------------------------------------------------------------

def conjugate_heat_transfer_pipeline(
    solid_geometry: str,
    fluid_geometry: str,
    software: str = "OpenFOAM",
    heat_source: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Conjugate heat transfer: coupled solid-fluid thermal analysis.

    Args:
        solid_geometry: Solid domain mesh
        fluid_geometry: Fluid domain mesh
        software: CFD software with CHT capability
        heat_source: {"power": W, "location": "surface_name"}
    """
    stages = []

    # Stage 1: Coupled mesh setup
    stages.append({
        "name": "cht_mesh",
        "tool": "openfoam_tool" if software.lower() == "openfoam" else "cfd_tool",
        "action": "mesh_cht",
        "params": {
            "software": software,
            "solid_geometry": solid_geometry,
            "fluid_geometry": fluid_geometry,
            "interface_coupling": "temperature_and_heat_flux",
        },
    })

    # Stage 2: Material properties for solid and fluid
    stages.append({
        "name": "cht_materials",
        "tool": "openfoam_tool" if software.lower() == "openfoam" else "cfd_tool",
        "action": "assign_materials",
        "params": {
            "software": software,
            "solid": {"rho": 2700, "cp": 900, "k": 200},  # Aluminum
            "fluid": {"rho": 1.225, "cp": 1005, "k": 0.026, "mu": 1.8e-5, "Pr": 0.71},
        },
        "depends_on": ["cht_mesh"],
    })

    # Stage 3: Thermal boundary conditions
    bc_params = {
        "software": software,
        "fluid_inlet": {"T": 300, "U": 5.0},
        "fluid_outlet": {"p": 0, "T": "zeroGradient"},
        "solid_heat_source": heat_source or {"power": 1000, "location": "heated_surface"},
        "external_wall": {"type": "convective", "h": 10, "T_inf": 300},
    }
    stages.append({
        "name": "cht_bc",
        "tool": "openfoam_tool" if software.lower() == "openfoam" else "cfd_tool",
        "action": "apply_bc",
        "params": bc_params,
        "depends_on": ["cht_materials"],
    })

    # Stage 4: Solve
    stages.append({
        "name": "cht_solve",
        "tool": "openfoam_tool" if software.lower() == "openfoam" else "cfd_tool",
        "action": "solve",
        "params": {
            "software": software,
            "solver": "chtMultiRegionFoam" if software.lower() == "openfoam" else "conjugate_heat_transfer",
            "steady_state": True,
            "max_iterations": 500,
        },
        "depends_on": ["cht_bc"],
    })

    # Stage 5: Post-processing
    stages.append({
        "name": "cht_post",
        "tool": "rag_tool",
        "action": "search",
        "params": {
            "query": f"{software} conjugate heat transfer temperature distribution Nusselt number",
            "top_k": 3,
        },
        "depends_on": ["cht_solve"],
    })

    return stages


# Register all CFD templates
register_template("turbulent_flow", turbulent_flow_pipeline)
register_template("multiphase_flow", multiphase_flow_pipeline)
register_template("conjugate_heat_transfer", conjugate_heat_transfer_pipeline)
