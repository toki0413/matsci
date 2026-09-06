"""Skills package for material science workflows."""

from typing import Any

from huginn.skills.base import (
    DeclarativeSkillExecutor,
    SkillDefinition,
    SkillExecutor,
    SkillParameter,
    SkillStep,
)
from huginn.skills.composite import (
    BAND_STRUCTURE_ANALYSIS,
    FRACTURE_ASSESSMENT,
    MD_PIPELINE,
    MECHANICAL_PROPERTIES,
    MOLECULE_SCREENING,
    PHONON_ANALYSIS,
)
from huginn.skills.evolution import SkillEvolutionLayer, ToolBelief
from huginn.skills.registry import SkillRegistry, register_skill

__all__ = [
    "SkillDefinition",
    "SkillParameter",
    "SkillStep",
    "SkillExecutor",
    "DeclarativeSkillExecutor",
    "SkillRegistry",
    "register_skill",
    # Bayesian evolution
    "SkillEvolutionLayer",
    "ToolBelief",
    # Preset skills — verification
    "SYMBOLIC_VERIFY",
    "TENSOR_VERIFY",
    "FEM_VERIFY",
    "LA_VERIFY",
    "DFT_VERIFY",
    "THERMO_VERIFY",
    "PROBABILITY_VERIFY",
    # Preset skills — analysis
    "UNCERTAINTY_PROPAGATION",
    "SENSITIVITY_ANALYSIS",
    "GP_PREDICTION",
    "BAYESIAN_CALIBRATION",
    "CONVERGENCE_TEST",
    # Preset skills — domain workflows
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
    "BATTERY_IONIC_CONDUCTIVITY",
    "CATALYSIS_SCREENING",
    "PHASE_DIAGRAM_CONSTRUCTION",
    "XRD_STRUCTURE_SOLUTION",
    "PHONON_SPECTROSCOPY_WORKFLOW",
    "CALPHAD_PHASE_DIAGRAM",
    "DEFECT_FORMATION_ENERGY",
    "ELECTROCHEMISTRY_POURBAIX",
    "POLYMER_GLASS_TRANSITION",
    "MAGNETIC_ANISOTROPY",
    "CATALYSIS_VOLCANO",
    "FETCH_REFERENCE_STRUCTURE",
    "ACTIVE_LEARNING_SCREENING",
    "TOPOLOGICAL_GEOMETRY_ANALYSIS",
    "VISUALIZE_RESULTS",
    "SYNTHESIS_PLANNING",
    "ML_POTENTIAL_PREDICTION",
    "CHARACTERIZATION_ANALYSIS",
    "HYPOTHESIS_GENERATOR",
    "MATERIALS_AUTORESEARCH",
    "MD_DFT_CROSS_VALIDATION",
    "SCENARIO_TOOL_SELECTOR",
    # Composite skills
    "BAND_STRUCTURE_ANALYSIS",
    "MECHANICAL_PROPERTIES",
    "MD_PIPELINE",
    "MOLECULE_SCREENING",
    "PHONON_ANALYSIS",
    "FRACTURE_ASSESSMENT",
]

# 懒加载 (设备端/小模型): ``import huginn.skills`` 不再副作用注册 ~45 个预设技能.
# 预设 SkillDefinition 只在其名字被访问时 (或 SkillRegistry 查询触发 ensure_presets)
# 才导入 ``huginn.skills.presets`` 注册. base/registry/composite 保持轻量 eager.
# ``from huginn.skills import presets`` 走子模块导入 (__getattr__ 抛 AttributeError →
# 包导入机制顶回落), 与 routes/skill_import 既有用法兼容.
_LAZY_SKILL_NAMES: frozenset[str] = frozenset(__all__)


def __getattr__(name: str) -> Any:
    if name in _LAZY_SKILL_NAMES:
        import huginn.skills.presets as _presets
        if hasattr(_presets, name):
            return getattr(_presets, name)
    raise AttributeError(f"module 'huginn.skills' has no attribute {name!r}")
