"""
Quantum Chemistry Workflow Templates for Huginn.

Derived from Sobko database best practices for molecular-level
computational chemistry analyses.
"""

from __future__ import annotations

from typing import Any

from huginn.workflows.templates import register_template

# ---------------------------------------------------------------------------
# Workflow 1: Wavefunction Analysis Pipeline
# ---------------------------------------------------------------------------


def _geometry_optimization_stage(
    structure_file: str, software: str
) -> dict[str, Any] | None:
    """Build the geometry-optimization stage for the selected QM software."""
    if software.lower() == "gaussian":
        return {
            "name": "geometry_optimization",
            "tool": "vasp_tool",  # or a generic QM tool
            "action": "optimize",
            "params": {
                "structure_file": structure_file,
                "method": "B3LYP",
                "basis": "6-31G(d)",
                "job_type": "opt",
            },
        }
    if software.lower() == "orca":
        return {
            "name": "geometry_optimization",
            "tool": "vasp_tool",
            "action": "optimize",
            "params": {
                "structure_file": structure_file,
                "method": "B3LYP",
                "basis": "def2-SVP",
                "job_type": "opt",
            },
        }
    return None


def _single_point_stage(software: str) -> dict[str, Any]:
    """Build the single-point stage with a better basis set."""
    return {
        "name": "single_point",
        "tool": "vasp_tool",
        "action": "single_point",
        "params": {
            "structure_file": "optimized_geometry",
            "method": "B3LYP",
            "basis": (
                "6-311+G(d,p)" if software.lower() == "gaussian" else "def2-TZVP"
            ),
            "job_type": "sp",
        },
        "depends_on": ["geometry_optimization"],
    }


def _method_analysis_stage(method: str) -> dict[str, Any] | None:
    """Build the Multiwfn post-processing stage for one analysis method."""
    if method.upper() == "ESP":
        return {
            "name": "esp_analysis",
            "tool": "rag_tool",
            "action": "search",
            "params": {
                "query": "ESP electrostatic potential analysis Multiwfn procedure",
                "top_k": 3,
            },
            "depends_on": ["single_point"],
        }
    if method.upper() in ("NCI", "IGMH"):
        return {
            "name": f"{method.lower()}_analysis",
            "tool": "rag_tool",
            "action": "search",
            "params": {
                "query": f"{method} weak interaction visualization Multiwfn",
                "top_k": 3,
            },
            "depends_on": ["single_point"],
        }
    if method.upper() == "NTO":
        return {
            "name": "nto_analysis",
            "tool": "rag_tool",
            "action": "search",
            "params": {
                "query": "NTO natural transition orbital excited state analysis",
                "top_k": 3,
            },
            "depends_on": ["single_point"],
        }
    return None


def wavefunction_analysis_pipeline(
    structure_file: str,
    software: str = "Gaussian",
    analysis_methods: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Complete wavefunction analysis pipeline: optimize → calculate → post-process.

    Args:
        structure_file: Input molecular structure (xyz, pdb, etc.)
        software: QM software to use (Gaussian, ORCA)
        analysis_methods: List of analyses to perform (ESP, NCI, IGMH, NTO, etc.)
    """
    methods = analysis_methods or ["ESP", "NCI"]
    stages = []

    # Stage 1: Geometry optimization
    geometry_stage = _geometry_optimization_stage(structure_file, software)
    if geometry_stage is not None:
        stages.append(geometry_stage)

    # Stage 2: Single-point with better basis
    stages.append(_single_point_stage(software))

    # Stage 3: Post-processing with Multiwfn
    for method in methods:
        analysis_stage = _method_analysis_stage(method)
        if analysis_stage is not None:
            stages.append(analysis_stage)

    return stages


# ---------------------------------------------------------------------------
# Workflow 2: Reactivity Prediction Pipeline
# ---------------------------------------------------------------------------


def _optimize_neutral_stage(structure_file: str) -> dict[str, Any]:
    """Build the neutral-state optimization stage."""
    return {
        "name": "optimize_neutral",
        "tool": "vasp_tool",
        "action": "optimize",
        "params": {
            "structure_file": structure_file,
            "method": "B3LYP",
            "basis": "6-31G(d)",
        },
    }


def _fukui_state_stage(name: str, charge_delta: int) -> dict[str, Any]:
    """Build the N±1 single-point stage needed for the Fukui function."""
    return {
        "name": name,
        "tool": "vasp_tool",
        "action": "single_point",
        "params": {
            "structure_file": "optimized_geometry",
            "method": "B3LYP",
            "basis": "6-31G(d)",
            "charge_delta": charge_delta,
        },
        "depends_on": ["optimize_neutral"],
    }


def _multiwfn_fukui_stage() -> dict[str, Any]:
    """Build the Multiwfn Fukui post-processing stage."""
    return {
        "name": "multiwfn_fukui",
        "tool": "rag_tool",
        "action": "search",
        "params": {
            "query": "Fukui function dual descriptor reaction site prediction Multiwfn CDFT",
            "top_k": 5,
        },
        "depends_on": ["calculate_n_minus_1", "calculate_n_plus_1"],
    }


def reactivity_prediction_pipeline(
    structure_file: str,
    prediction_type: str = "both",
) -> list[dict[str, Any]]:
    """Predict electrophilic and nucleophilic reaction sites using conceptual DFT.

    Args:
        structure_file: Input molecular structure
        prediction_type: "electrophilic", "nucleophilic", or "both"
    """
    stages = []

    # Stage 1: Optimize neutral state
    stages.append(_optimize_neutral_stage(structure_file))

    # Stage 2: Calculate N+1 and N-1 states for Fukui function
    if prediction_type in ("electrophilic", "both"):
        stages.append(_fukui_state_stage("calculate_n_minus_1", -1))  # N-1 state

    if prediction_type in ("nucleophilic", "both"):
        stages.append(_fukui_state_stage("calculate_n_plus_1", +1))  # N+1 state

    # Stage 3: Post-process with Multiwfn
    stages.append(_multiwfn_fukui_stage())

    return stages


# ---------------------------------------------------------------------------
# Workflow 3: Weak Interaction Analysis Pipeline
# ---------------------------------------------------------------------------


def _weak_single_point_stage(structure_file: str) -> dict[str, Any]:
    """Build the single-point stage that produces the wavefunction."""
    return {
        "name": "single_point",
        "tool": "vasp_tool",
        "action": "single_point",
        "params": {
            "structure_file": structure_file,
            "method": "B3LYP",
            "basis": "6-31G(d)",
        },
    }


def _weak_procedure_stage(method: str) -> dict[str, Any]:
    """Build the retrieval stage for the weak-interaction procedure."""
    query_map = {
        "IGMH": "IGMH independent gradient model Hirshfeld weak interaction Multiwfn",
        "NCI": "NCI non-covalent interaction RDG weak interaction Multiwfn",
        "MIGM": "mIGM geometry weak interaction fast visualization",
    }

    return {
        "name": f"{method.lower()}_procedure",
        "tool": "rag_tool",
        "action": "search",
        "params": {
            "query": query_map.get(
                method.upper(), f"{method} weak interaction analysis"
            ),
            "top_k": 5,
        },
        "depends_on": ["single_point"] if method.upper() in ("IGMH", "NCI") else [],
    }


def weak_interaction_pipeline(
    structure_file: str,
    fragments: list[list[int]] | None = None,
    method: str = "IGMH",
) -> list[dict[str, Any]]:
    """Analyze weak interactions in a molecular system.

    Args:
        structure_file: Input structure
        fragments: List of atom index lists defining fragments
        method: "IGMH", "NCI", or "mIGM"
    """
    stages = []

    # Stage 1: Single point calculation (if wavefunction needed)
    if method.upper() in ("IGMH", "NCI"):
        stages.append(_weak_single_point_stage(structure_file))

    # Stage 2: Retrieve analysis procedure
    stages.append(_weak_procedure_stage(method))

    return stages


# ---------------------------------------------------------------------------
# Workflow 4: Excited State Analysis Pipeline
# ---------------------------------------------------------------------------


def _tddft_stage(
    structure_file: str, n_states: int, software: str
) -> dict[str, Any] | None:
    """Build the TDDFT stage for the selected QM software."""
    if software.lower() == "gaussian":
        return {
            "name": "tddft_calculation",
            "tool": "vasp_tool",
            "action": "single_point",
            "params": {
                "structure_file": structure_file,
                "method": "CAM-B3LYP",
                "basis": "6-311+G(d,p)",
                "job_type": "td",
                "n_states": n_states,
                "keywords": "IOp(9/40=4)",  # Print all CI coefficients
            },
        }
    if software.lower() == "orca":
        return {
            "name": "tddft_calculation",
            "tool": "vasp_tool",
            "action": "single_point",
            "params": {
                "structure_file": structure_file,
                "method": "CAM-B3LYP",
                "basis": "def2-TZVP",
                "job_type": "td",
                "n_states": n_states,
                "keywords": "TPrint",
            },
        }
    return None


def excited_state_pipeline(
    structure_file: str,
    n_states: int = 10,
    software: str = "Gaussian",
) -> list[dict[str, Any]]:
    """Perform excited state calculation and analysis.

    Args:
        structure_file: Input structure
        n_states: Number of excited states to calculate
        software: "Gaussian" or "ORCA"
    """
    stages = []

    # Stage 1: TDDFT calculation
    tddft_stage = _tddft_stage(structure_file, n_states, software)
    if tddft_stage is not None:
        stages.append(tddft_stage)

    # Stage 2: Retrieve post-processing procedures
    stages.append(
        {
            "name": "excited_state_analysis",
            "tool": "rag_tool",
            "action": "search",
            "params": {
                "query": "hole-electron analysis NTO natural transition orbital excited state Multiwfn",
                "top_k": 5,
            },
            "depends_on": ["tddft_calculation"],
        }
    )

    return stages


# ---------------------------------------------------------------------------
# Workflow 5: Charge Analysis Pipeline
# ---------------------------------------------------------------------------


def charge_analysis_pipeline(
    structure_file: str,
    charge_methods: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Perform atomic charge analysis using multiple methods.

    Args:
        structure_file: Input structure
        charge_methods: Methods to use (RESP, Hirshfeld, ADCH, NPA, etc.)
    """
    methods = charge_methods or ["RESP", "Hirshfeld"]
    stages = []

    # Stage 1: Single point with high-quality wavefunction
    stages.append(
        {
            "name": "single_point",
            "tool": "vasp_tool",
            "action": "single_point",
            "params": {
                "structure_file": structure_file,
                "method": "B3LYP",
                "basis": "6-311+G(d,p)",
            },
        }
    )

    # Stage 2: Retrieve charge analysis procedures
    for method in methods:
        stages.append(
            {
                "name": f"{method.lower()}_charge",
                "tool": "rag_tool",
                "action": "search",
                "params": {
                    "query": f"{method} atomic charge analysis Multiwfn procedure",
                    "top_k": 3,
                },
                "depends_on": ["single_point"],
            }
        )

    return stages


# Register all templates
register_template("wavefunction_analysis", wavefunction_analysis_pipeline)
register_template("reactivity_prediction", reactivity_prediction_pipeline)
register_template("weak_interaction", weak_interaction_pipeline)
register_template("excited_state", excited_state_pipeline)
register_template("charge_analysis", charge_analysis_pipeline)
