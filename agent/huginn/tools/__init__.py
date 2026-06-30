"""Huginn tools package."""

from __future__ import annotations

import os
import sys
from typing import Any

from huginn.tools.base import HuginnTool
from huginn.tools.registry import ToolRegistry, register_tool

__all__ = ["HuginnTool", "ToolRegistry", "register_tool", "register_all_tools"]


def register_all_tools(config: Any | None = None) -> list[str]:
    """Register every built-in tool to the global registry.

    If ``config`` is provided, the execution backend (local sandbox or remote
    HPC) and per-tool executable paths are wired into the simulation tools.
    Calling this function multiple times is safe; subsequent calls are no-ops.

    Optional tools whose dependencies are missing are silently skipped so the
    agent can still start in a minimal environment (e.g. PyInstaller build).
    """
    if ToolRegistry.list_tools():
        return ToolRegistry.list_tools()

    import importlib
    import logging

    logger = logging.getLogger(__name__)

    from huginn.config import HuginnConfig
    from huginn.execution.remote_executor import build_executor

    resolved_config = config if config is not None else HuginnConfig.from_env()

    # Sync allow_local_bash from config to env so SandboxExecutor and
    # get_executor() pick it up. This lets users set it in huginn.toml
    # instead of only via environment variables.
    if getattr(resolved_config, "allow_local_bash", False):
        os.environ.setdefault("HUGINN_ALLOW_LOCAL_BASH", "1")

    executor = build_executor(resolved_config)

    # Ensure Bourbaki (math_anything) is discoverable if installed externally.
    # Path is configurable via HUGINN_BOURBAKI_PATH env var.
    _bourbaki_path = os.environ.get(
        "HUGINN_BOURBAKI_PATH", ""
    )
    if _bourbaki_path and _bourbaki_path not in sys.path:
        sys.path.insert(0, _bourbaki_path)

    def _tool_kwargs(cls: type) -> dict[str, Any]:
        """Build init kwargs for tool classes that accept sandbox/executables."""
        import inspect

        sig = inspect.signature(cls.__init__)
        params = sig.parameters
        kwargs: dict[str, Any] = {}

        # 声明式注入: 子类用 _init_kwargs_map = {param: config_field} 声明需求
        for param_name, config_field in cls._init_kwargs_map.items():
            if param_name in params:
                kwargs[param_name] = getattr(resolved_config, config_field, None)

        if "sandbox" in params:
            kwargs["sandbox"] = executor
        return kwargs

    # Core tools (always expected to be available)
    # 分组注释对应阶段 4 子包划分: core/ search/ meta/ + 外部包
    core_modules = [
        # ── core/ ──
        ("huginn.tools.bash_tool", "BashTool"),
        ("huginn.tools.code_tool", "CodeTool"),
        ("huginn.tools.file_edit_tool", "FileEditTool"),
        ("huginn.tools.file_read_tool", "FileReadTool"),
        ("huginn.tools.file_write_tool", "FileWriteTool"),
        ("huginn.tools.git_tool", "GitTool"),
        ("huginn.tools.bourbaki_tool", "BourbakiTool"),
        ("huginn.tools.diff_tool", "DiffTool"),
        ("huginn.tools.validate_tool", "ValidateTool"),
        ("huginn.tools.diagnose_tool", "DiagnoseTool"),
        ("huginn.tools.system_diagnostic_tool", "SystemDiagnosticTool"),
        ("huginn.tools.extract_tool", "ExtractTool"),
        ("huginn.tools.job_tool", "JobTool"),
        ("huginn.tools.database_tool", "DatabaseTool"),
        ("huginn.tools.report_tool", "ReportTool"),
        ("huginn.tools.lean_tool", "LeanTool"),
        ("huginn.tools.structure_tool", "StructureTool"),
        # ── search/ ──
        ("huginn.tools.web_search_tool", "WebSearchTool"),
        ("huginn.tools.literature_tool", "LiteratureTool"),
        # ── meta/ ──
        ("huginn.tools.orchestrate_tool", "OrchestrateTool"),
        ("huginn.tools.skill_tool", "SkillTool"),
        ("huginn.tools.memory_tool", "RememberTool"),
        ("huginn.tools.memory_tool", "RecallTool"),
        ("huginn.tools.scenario_tool", "ScenarioTool"),
        ("huginn.tools.simple_path_tool", "SimplePathTool"),
        ("huginn.tools.personalization_tool", "PersonalizationTool"),
        ("huginn.tools.config_wizard_tool", "ConfigWizardTool"),
        ("huginn.tools.clarification_tool", "ClarificationTool"),
        # ── 外部包 (evaluation/rag/plugins, 不在 tools/ 下) ──
        ("huginn.evaluation.evaluation_tool", "EvaluationTool"),
        ("huginn.rag.rag_tool", "RAGTool"),
        ("huginn.plugins.autoresearch", "AutoresearchTool"),
    ]

    # Optional simulation / science tools (skip if deps missing)
    # 分组注释对应阶段 4 子包划分: sim/ sci/ design/ cv/ search/ meta/ materials/
    optional_modules = [
        # ── sim/ ──
        ("huginn.tools.vasp_tool", "VaspTool"),
        ("huginn.tools.lammps_tool", "LammpsTool"),
        ("huginn.tools.comsol_tool", "ComsolTool"),
        ("huginn.tools.qe_tool", "QuantumEspressoTool"),
        ("huginn.tools.cp2k_tool", "Cp2kTool"),
        ("huginn.tools.openfoam_tool", "OpenFoamTool"),
        ("huginn.tools.packing_tool", "PackingTool"),
        ("huginn.tools.abaqus_tool", "AbaqusTool"),
        ("huginn.tools.plasma_tool", "PlasmaTool"),
        ("huginn.tools.neb_tool", "NEBTool"),
        ("huginn.tools.structural_analytical_tool", "StructuralAnalyticalTool"),
        ("huginn.tools.specialty_analysis_tool", "SpecialtyAnalysisTool"),
        ("huginn.tools.fem_tool", "FEMTool"),
        # ── sci/ ──
        ("huginn.tools.symbolic_regression_tool", "SymbolicRegressionTool"),
        ("huginn.tools.symbolic_math_tool", "SymbolicMathTool"),
        ("huginn.tools.autodiff_tool", "AutoDiffTool"),
        ("huginn.tools.numerical_tool", "NumericalTool"),
        ("huginn.tools.unit_tool", "UnitTool"),
        ("huginn.tools.symmetry_tool", "SymmetryTool"),
        ("huginn.tools.tda_tool", "TDATool"),
        ("huginn.tools.uq_tool", "UQTool"),
        ("huginn.tools.gp_tool", "GPTool"),
        ("huginn.tools.descriptor_tool", "DescriptorTool"),
        ("huginn.tools.evidence_fusion_tool", "EvidenceFusionTool"),
        ("huginn.tools.active_learning_tool", "ActiveLearningTool"),
        ("huginn.tools.ml_potential_tool", "MLPotentialTool"),
        ("huginn.tools.high_throughput_tool", "HighThroughputTool"),
        ("huginn.tools.xrd_sim_tool", "XrdSimTool"),
        # ── design/ ──
        ("huginn.tools.gap_analysis_tool", "GapAnalysisTool"),
        ("huginn.tools.doe_tool", "DOETool"),
        ("huginn.tools.debugger_tool", "DebuggerTool"),
        ("huginn.tools.design_plan_tool", "DesignPlanTool"),
        ("huginn.tools.nudge_tool", "NudgeTool"),
        ("huginn.tools.design_atom_tool", "DesignAtomTool"),
        ("huginn.tools.generative_design_tool", "GenerativeDesignTool"),
        # ── cv/ ──
        ("huginn.tools.image_analysis_tool", "ImageAnalysisTool"),
        ("huginn.tools.image_design_tool", "ImageDesignTool"),
        ("huginn.tools.visualize_tool", "VisualizeTool"),
        ("huginn.tools.characterization_tool", "CharacterizationTool"),
        # ── search/ (可选检索类) ──
        ("huginn.tools.browser_tool", "BrowserTool"),
        ("huginn.tools.review_committee_tool", "ReviewCommitteeTool"),
        ("huginn.tools.hypothesis_generator_tool", "HypothesisGeneratorTool"),
        ("huginn.tools.materials_autoresearch_tool", "MaterialsAutoResearchTool"),
        # ── meta/ (可选) ──
        ("huginn.tools.nuwa_persona_tool", "NuwaPersonaTool"),
        # ── materials/ ──
        ("huginn.tools.materials_database_tool", "MaterialsDatabaseTool"),
        ("huginn.tools.experimental_data_tool", "ExperimentalDataTool"),
    ]

    registered: list[str] = []
    skipped: list[str] = []

    for module_name, class_name in core_modules + optional_modules:
        try:
            mod = importlib.import_module(module_name)
            cls = getattr(mod, class_name)
            ToolRegistry.register(cls(**_tool_kwargs(cls)))
            registered.append(class_name)
        except ImportError as exc:
            skipped.append(f"{class_name} ({exc.name or module_name})")
        except Exception as exc:
            logger.warning(f"Tool {class_name} registration failed: {exc}")
            skipped.append(class_name)

    if skipped:
        logger.info(f"Skipped {len(skipped)} optional tools (missing deps): {', '.join(skipped[:5])}")

    # Science-skills bridge (google-deepmind/science-skills plugin)
    try:
        from huginn.plugins.science_skills_bridge import register_science_skills
        science_names = register_science_skills()
        registered.extend(science_names)
        logger.info(f"Registered {len(science_names)} science-skills bridge tools")
    except Exception as exc:
        logger.warning(f"Science-skills bridge registration failed: {exc}")

    # Rebuild dispatch tables from ToolProfile metadata so the phase
    # filters, router, and constraint scopes track the registered tools'
    # declared profiles instead of hand-maintained dicts.
    from huginn.agents.tool_call_router import _rebuild_router_tables
    from huginn.phases import _rebuild_phase_tools
    from huginn.tools.adapter import _rebuild_constraint_scopes

    _rebuild_phase_tools()
    _rebuild_router_tables()
    _rebuild_constraint_scopes()

    return ToolRegistry.list_tools()
