"""Huginn tools package."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from huginn.tools.base import HuginnTool
from huginn.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

__all__ = [
    "HuginnTool",
    "ToolRegistry",
    "register_all_tools",
    "register_core_tools",
    "register_optional_tools",
]


# ── Tool module lists ──────────────────────────────────────────────

# Core tools — always available, fast to import (~35 tools)
_CORE_MODULES = [
    ("huginn.tools.bash_tool", "BashTool"),
    ("huginn.tools.code_tool", "CodeTool"),
    ("huginn.tools.file_edit_tool", "FileEditTool"),
    ("huginn.tools.multi_edit_tool", "MultiEditTool"),
    ("huginn.tools.file_read_tool", "FileReadTool"),
    ("huginn.tools.file_write_tool", "FileWriteTool"),
    ("huginn.tools.glob_tool", "GlobTool"),
    ("huginn.tools.grep_tool", "GrepTool"),
    ("huginn.tools.eval_tool", "EvalTool"),
    ("huginn.tools.git_tool", "GitTool"),
    ("huginn.tools.github_tool", "GithubTool"),
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
    ("huginn.tools.web_search_tool", "WebSearchTool"),
    ("huginn.tools.agentic_search_tool", "AgenticSearchTool"),
    ("huginn.tools.literature_tool", "LiteratureTool"),
    ("huginn.tools.orchestrate_tool", "OrchestrateTool"),
    ("huginn.tools.skill_tool", "SkillTool"),
    ("huginn.tools.memory_tool", "RememberTool"),
    ("huginn.tools.memory_tool", "RecallTool"),
    ("huginn.tools.scenario_tool", "ScenarioTool"),
    ("huginn.tools.simple_path_tool", "SimplePathTool"),
    ("huginn.tools.personalization_tool", "PersonalizationTool"),
    ("huginn.tools.onboarding_tool", "OnboardingTool"),
    ("huginn.tools.phase_tool", "PhaseTool"),
    ("huginn.tools.workflow_tool", "WorkflowTool"),
    ("huginn.tools.config_wizard_tool", "ConfigWizardTool"),
    ("huginn.tools.clarification_tool", "ClarificationTool"),
    ("huginn.evaluation.evaluation_tool", "EvaluationTool"),
    ("huginn.rag.rag_tool", "RAGTool"),
    ("huginn.plugins.autoresearch", "AutoresearchTool"),
    ("huginn.academic.paper_tool", "PaperTool"),
    ("huginn.academic.deli_research", "DeliAutoResearchTool"),
    ("huginn.tools.tool_search_tool", "ToolSearchTool"),
]

# Optional tools — heavy imports (numpy/scipy/simulation), safe to defer
_OPTIONAL_MODULES = [
    ("huginn.tools.vasp_tool", "VaspTool"),
    ("huginn.tools.lammps_tool", "LammpsTool"),
    ("huginn.tools.comsol_tool", "ComsolTool"),
    ("huginn.tools.qe_tool", "QuantumEspressoTool"),
    ("huginn.tools.cp2k_tool", "Cp2kTool"),
    ("huginn.tools.gaussian_tool", "GaussianTool"),
    ("huginn.tools.orca_tool", "OrcaTool"),
    ("huginn.tools.openfoam_tool", "OpenFoamTool"),
    ("huginn.tools.packing_tool", "PackingTool"),
    ("huginn.tools.abaqus_tool", "AbaqusTool"),
    ("huginn.tools.fenics_tool", "FenicsTool"),
    ("huginn.tools.elmer_tool", "ElmerTool"),
    ("huginn.tools.gromacs_tool", "GromacsTool"),
    ("huginn.tools.plasma_tool", "PlasmaTool"),
    ("huginn.tools.neb_tool", "NEBTool"),
    ("huginn.tools.structural_analytical_tool", "StructuralAnalyticalTool"),
    ("huginn.tools.specialty_analysis_tool", "SpecialtyAnalysisTool"),
    ("huginn.tools.fem_tool", "FEMTool"),
    ("huginn.tools.sim.transolver_tool", "TransolverTool"),
    ("huginn.tools.sim.mechanical_tool", "MechanicalTool"),
    ("huginn.tools.sim.convergence_test_tool", "ConvergenceTestTool"),
    ("huginn.tools.sim.resolve_executable_tool", "ResolveExecutableTool"),
    ("huginn.tools.vina_tool", "VinaTool"),
    ("huginn.tools.openmm_tool", "OpenMMTool"),
    ("huginn.tools.symbolic_regression_tool", "SymbolicRegressionTool"),
    ("huginn.tools.symbolic_math_tool", "SymbolicMathTool"),
    ("huginn.tools.dynamics_discovery_tool", "DynamicsDiscoveryTool"),
    ("huginn.tools.sci.interpretable_ml_tool", "InterpretableMLTool"),
    ("huginn.tools.autodiff_tool", "AutoDiffTool"),
    ("huginn.tools.numerical_tool", "NumericalTool"),
    ("huginn.tools.unit_tool", "UnitTool"),
    ("huginn.tools.symmetry_tool", "SymmetryTool"),
    ("huginn.tools.tda_tool", "TDATool"),
    ("huginn.tools.uq_tool", "UQTool"),
    ("huginn.tools.gp_tool", "GPTool"),
    ("huginn.tools.descriptor_tool", "DescriptorTool"),
    ("huginn.tools.rdkit_tool", "RDKitTool"),
    ("huginn.tools.fep_tool", "FEPTool"),
    ("huginn.tools.enhanced_sampling_tool", "EnhancedSamplingTool"),
    ("huginn.tools.msm_tool", "MSMTool"),
    ("huginn.tools.inverse_design_tool", "InverseDesignTool"),
    ("huginn.tools.motif_mining_tool", "MotifMiningTool"),
    ("huginn.tools.consensus_scoring_tool", "ConsensusScoringTool"),
    ("huginn.tools.evidence_fusion_tool", "EvidenceFusionTool"),
    ("huginn.tools.active_learning_tool", "ActiveLearningTool"),
    ("huginn.tools.ml_potential_tool", "MLPotentialTool"),
    ("huginn.tools.high_throughput_tool", "HighThroughputTool"),
    ("huginn.tools.multi_fidelity_tool", "MultiFidelityTool"),
    ("huginn.tools.xrd_sim_tool", "XrdSimTool"),
    ("huginn.tools.gap_analysis_tool", "GapAnalysisTool"),
    ("huginn.tools.doe_tool", "DOETool"),
    ("huginn.tools.debugger_tool", "DebuggerTool"),
    ("huginn.tools.design_plan_tool", "DesignPlanTool"),
    ("huginn.tools.nudge_tool", "NudgeTool"),
    ("huginn.tools.design_atom_tool", "DesignAtomTool"),
    ("huginn.tools.generative_design_tool", "GenerativeDesignTool"),
    ("huginn.tools.plan_store_tool", "PlanStoreTool"),
    ("huginn.tools.image_analysis_tool", "ImageAnalysisTool"),
    ("huginn.tools.image_design_tool", "ImageDesignTool"),
    ("huginn.tools.visualize_tool", "VisualizeTool"),
    ("huginn.tools.characterization_tool", "CharacterizationTool"),
    ("huginn.tools.model3d_tool", "Model3DTool"),
    ("huginn.tools.browser_tool", "BrowserTool"),
    ("huginn.tools.review_committee_tool", "ReviewCommitteeTool"),
    ("huginn.tools.hypothesis_generator_tool", "HypothesisGeneratorTool"),
    ("huginn.tools.materials_autoresearch_tool", "MaterialsAutoResearchTool"),
    ("huginn.tools.nuwa_persona_tool", "NuwaPersonaTool"),
    ("huginn.tools.subagent_tool", "SubagentTool"),
    ("huginn.tools.materials_database_tool", "MaterialsDatabaseTool"),
    ("huginn.tools.experimental_data_tool", "ExperimentalDataTool"),
    ("huginn.tools.thermo_tool", "ThermoTool"),
    ("huginn.tools.wetlab_rpc_tool", "WetlabRpcTool"),
    # bench_infra — 预置 benchmark 工具, 治 ζ_* (agent 不再从零写训练循环/画图/C2ST/MCMC/CSV)
    ("huginn.tools.bench_infra.plot_tool", "PlotTool"),
    ("huginn.tools.bench_infra.matrix_tool", "TrainingMatrixTool"),
    ("huginn.tools.bench_infra.c2st_tool", "C2STEvaluatorTool"),
    ("huginn.tools.bench_infra.mcmc_tool", "MCMCSamplerTool"),
    ("huginn.tools.bench_infra.kaggle_tool", "KaggleSubmitTool"),
]


def _resolve_config(config: Any | None = None):
    from huginn.config import HuginnConfig
    resolved = config if config is not None else HuginnConfig.from_env()
    if getattr(resolved, "allow_local_bash", False):
        os.environ.setdefault("HUGINN_ALLOW_LOCAL_BASH", "1")
    _bourbaki_path = os.environ.get("HUGINN_BOURBAKI_PATH", "")
    if _bourbaki_path and _bourbaki_path not in sys.path:
        sys.path.insert(0, _bourbaki_path)
    return resolved


def _make_tool_kwargs(resolved_config, executor):
    """Build the kwargs factory for tool instantiation."""
    def _tool_kwargs(cls: type) -> dict[str, Any]:
        import inspect
        sig = inspect.signature(cls.__init__)
        params = sig.parameters
        kwargs: dict[str, Any] = {}
        for param_name, config_field in cls._init_kwargs_map.items():
            if param_name in params:
                kwargs[param_name] = getattr(resolved_config, config_field, None)
        if "sandbox" in params:
            kwargs["sandbox"] = executor
        return kwargs
    return _tool_kwargs


def _do_register(modules_list, _tool_kwargs) -> tuple[list[str], list[str]]:
    """Import and register a list of (module, class) tuples."""
    import importlib
    registered: list[str] = []
    skipped: list[str] = []

    for module_name, class_name in modules_list:
        try:
            mod = importlib.import_module(module_name)
            cls = getattr(mod, class_name)
            kwargs = _tool_kwargs(cls)

            # RAGTool shares the KB collection if available
            if class_name == "RAGTool":
                import inspect
                sig = inspect.signature(cls.__init__)
                if "kb" in sig.parameters:
                    try:
                        from huginn import server_context as _sc
                        _ctx = _sc._server_context
                        if _ctx is not None and _ctx.kb is not None:
                            kwargs["kb"] = _ctx.kb
                    except Exception:
                        logger.debug("tool kwargs failed", exc_info=True)

            ToolRegistry.register(cls(**kwargs))
            registered.append(class_name)
        except ImportError as exc:
            skipped.append(f"{class_name} ({exc.name or module_name})")
        except Exception as exc:
            logger.warning(f"Tool {class_name} registration failed: {exc}")
            skipped.append(class_name)
    return registered, skipped


def _rebuild_dispatch_tables() -> None:
    """Rebuild phase/router/constraint tables after tool registration."""
    from huginn.agents.tool_call_router import _rebuild_router_tables
    from huginn.phases import _rebuild_phase_tools
    from huginn.tools.adapter import _rebuild_constraint_scopes
    _rebuild_phase_tools()
    _rebuild_router_tables()
    _rebuild_constraint_scopes()


def register_core_tools(config: Any | None = None) -> list[str]:
    """Register only the core tools (fast, ~35 tools, no heavy deps).

    This is safe to call synchronously at startup. Optional tools should
    be registered via register_optional_tools() in the background.
    """
    if ToolRegistry.list_tools():
        return ToolRegistry.list_tools()

    from huginn.execution.remote_executor import build_executor
    resolved_config = _resolve_config(config)
    executor = build_executor(resolved_config)
    _tool_kwargs = _make_tool_kwargs(resolved_config, executor)

    registered, skipped = _do_register(_CORE_MODULES, _tool_kwargs)
    if skipped:
        logger.info(f"Skipped {len(skipped)} core tools (missing deps): {', '.join(skipped[:5])}")
    _rebuild_dispatch_tables()
    logger.info(f"[tools] registered {len(registered)} core tools")
    return ToolRegistry.list_tools()


def register_optional_tools(config: Any | None = None) -> list[str]:
    """Register optional simulation/science tools (slow, heavy imports).

    Call this in the background after core tools are registered.
    """
    existing = set(ToolRegistry.list_tools())
    pending = [(m, c) for m, c in _OPTIONAL_MODULES if c not in existing]
    if not pending:
        return ToolRegistry.list_tools()

    from huginn.execution.remote_executor import build_executor
    resolved_config = _resolve_config(config)
    executor = build_executor(resolved_config)
    _tool_kwargs = _make_tool_kwargs(resolved_config, executor)

    registered, skipped = _do_register(pending, _tool_kwargs)
    if skipped:
        logger.info(f"Skipped {len(skipped)} optional tools (missing deps): {', '.join(skipped[:5])}")

    # Science-skills bridge
    try:
        from huginn.plugins.science_skills_bridge import register_science_skills
        science_names = register_science_skills()
        registered.extend(science_names)
        logger.info(f"Registered {len(science_names)} science-skills bridge tools")
    except Exception as exc:
        logger.warning(f"Science-skills bridge registration failed: {exc}")

    _rebuild_dispatch_tables()

    # Start system health monitor if enabled
    try:
        from huginn.feature_flags import FeatureFlags
        if FeatureFlags.shared().is_enabled("system_health_monitor"):
            from huginn.diagnostics.system_health import SystemHealthMonitor
            SystemHealthMonitor.shared().start()
    except Exception as exc:
        logger.warning(f"System health monitor failed to start: {exc}")

    logger.info(f"[tools] registered {len(registered)} optional tools (total: {len(ToolRegistry.list_tools())})")
    return ToolRegistry.list_tools()


def register_all_tools(config: Any | None = None) -> list[str]:
    """Register every built-in tool to the global registry.

    Calls register_core_tools() then register_optional_tools() synchronously.
    For faster startup, call register_core_tools() and schedule
    register_optional_tools() in the background instead.
    """
    if ToolRegistry.list_tools():
        return ToolRegistry.list_tools()

    register_core_tools(config)
    register_optional_tools(config)
    return ToolRegistry.list_tools()
