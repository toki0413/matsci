"""Skills package for material science workflows."""

from huginn.skills.base import (
    DeclarativeSkillExecutor,
    SkillDefinition,
    SkillExecutor,
    SkillParameter,
    SkillStep,
)
from huginn.skills.presets import (
    AIMD_WORKFLOW,
    AUTORESEARCH_WORKFLOW,
    BAND_GAP_ANALYSIS,
    CONVERGENCE_DIAGNOSIS,
    DEFECT_CALCULATION,
    ELASTIC_CONSTANTS,
    HPC_REMOTE_RUN,
    HT_SCREENING,
    LAMMPS_MELT_QUENCH,
    PHONON_CALCULATION,
    STANDARD_DFT,
    SURFACE_CALCULATION,
    SYMBOLIC_REGRESSION,
)
from huginn.skills.registry import SkillRegistry, register_skill

__all__ = [
    "SkillDefinition",
    "SkillParameter",
    "SkillStep",
    "SkillExecutor",
    "DeclarativeSkillExecutor",
    "SkillRegistry",
    "register_skill",
    # Preset skills
    "STANDARD_DFT",
    "AIMD_WORKFLOW",
    "DEFECT_CALCULATION",
    "SURFACE_CALCULATION",
    "LAMMPS_MELT_QUENCH",
    "BAND_GAP_ANALYSIS",
    "ELASTIC_CONSTANTS",
    "PHONON_CALCULATION",
    "CONVERGENCE_DIAGNOSIS",
    "HT_SCREENING",
    "SYMBOLIC_REGRESSION",
    "HPC_REMOTE_RUN",
    "AUTORESEARCH_WORKFLOW",
]
