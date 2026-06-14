"""Skills package for material science workflows."""

from huginn.skills.base import (
    SkillDefinition,
    SkillParameter,
    SkillStep,
    SkillExecutor,
    DeclarativeSkillExecutor,
)
from huginn.skills.registry import SkillRegistry, register_skill
from huginn.skills.presets import (
    STANDARD_DFT,
    AIMD_WORKFLOW,
    DEFECT_CALCULATION,
    SURFACE_CALCULATION,
    LAMMPS_MELT_QUENCH,
    ML_POTENTIAL_TRAINING,
    BAND_GAP_ANALYSIS,
    ELASTIC_CONSTANTS,
    PHONON_CALCULATION,
    CONVERGENCE_DIAGNOSIS,
    HT_SCREENING,
    SYMBOLIC_REGRESSION,
)

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
    "ML_POTENTIAL_TRAINING",
    "BAND_GAP_ANALYSIS",
    "ELASTIC_CONSTANTS",
    "PHONON_CALCULATION",
    "CONVERGENCE_DIAGNOSIS",
    "HT_SCREENING",
    "SYMBOLIC_REGRESSION",
]
