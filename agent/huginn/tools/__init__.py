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
    "validate_tool_specs",
]


# ── Tool module lists ──────────────────────────────────────────────

# Core tools — always available, fast to import (~35 tools)
_CORE_MODULES = [
    ("huginn.tools.bash_tool", "BashTool"),
    ("huginn.tools.code_tool", "CodeTool"),
    ("huginn.tools.file_edit_tool", "FileEditTool"),
    ("huginn.tools.multi_edit_tool", "MultiEditTool"),
    ("huginn.tools.lsp_tool", "LspTool"),
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
    ("huginn.tools.literature_tool", "LiteraturePipelineTool"),
    ("huginn.tools.orchestrate_tool", "OrchestrateTool"),
    ("huginn.tools.skill_tool", "SkillTool"),
    ("huginn.tools.eco_tool", "EcoTool"),
    ("huginn.tools.memory_tool", "RememberTool"),
    ("huginn.tools.memory_tool", "RecallTool"),
    ("huginn.tools.prospective_tool", "ScheduleIntentionTool"),
    ("huginn.tools.prospective_tool", "ListPendingIntentionsTool"),
    ("huginn.tools.prospective_tool", "CancelIntentionTool"),
    ("huginn.tools.self_observe_tool", "SelfObserveTool"),
    ("huginn.tools.deep_think_tool", "DeepThinkTool"),
    ("huginn.tools.recall_context_tool", "RecallContextTool"),
    ("huginn.tools.todo_tool", "TodoWriteTool"),
    ("huginn.tools.todo_tool", "TodoReadTool"),
    ("huginn.tools.notebook_tool", "NotebookEditTool"),
    ("huginn.tools.scenario_tool", "ScenarioTool"),
    ("huginn.tools.simple_path_tool", "SimplePathTool"),
    ("huginn.tools.personalization_tool", "PersonalizationTool"),
    ("huginn.tools.onboarding_tool", "OnboardingTool"),
    ("huginn.tools.phase_tool", "PhaseTool"),
    ("huginn.tools.workflow_tool", "WorkflowTool"),
    ("huginn.tools.config_wizard_tool", "ConfigWizardTool"),
    ("huginn.tools.config_domain_tool", "ConfigDomainTool"),
    ("huginn.tools.clarification_tool", "ClarificationTool"),
    ("huginn.evaluation.evaluation_tool", "EvaluationTool"),
    ("huginn.rag.rag_tool", "RAGTool"),
    ("huginn.plugins.autoresearch", "AutoresearchTool"),
    ("huginn.academic.paper_tool", "PaperTool"),
    ("huginn.academic.deli_research", "DeliAutoResearchTool"),
    ("huginn.tools.tool_search_tool", "ToolSearchTool"),
    ("huginn.tools.prompt_optimize_tool", "PromptOptimizeTool"),
]

# Optional tools — heavy imports (numpy/scipy/simulation), safe to defer
_OPTIONAL_MODULES = [
    ("huginn.tools.vasp_tool", "VaspTool"),
    ("huginn.tools.sim.lammps_tool", "LammpsTool"),
    ("huginn.tools.sim.comsol_tool", "ComsolTool"),
    ("huginn.tools.sim.qe_tool", "QuantumEspressoTool"),
    ("huginn.tools.sim.cp2k_tool", "Cp2kTool"),
    ("huginn.tools.gaussian_tool", "GaussianTool"),
    ("huginn.tools.orca_tool", "OrcaTool"),
    ("huginn.tools.sim.openfoam_tool", "OpenFoamTool"),
    ("huginn.tools.sim.packing_tool", "PackingTool"),
    ("huginn.tools.sim.abaqus_tool", "AbaqusTool"),
    ("huginn.tools.fenics_tool", "FenicsTool"),
    ("huginn.tools.elmer_tool", "ElmerTool"),
    ("huginn.tools.gromacs_tool", "GromacsTool"),
    ("huginn.tools.sim.plasma_tool", "PlasmaTool"),
    ("huginn.tools.neb_tool", "NEBTool"),
    ("huginn.tools.structural_analytical.tool", "StructuralAnalyticalTool"),
    ("huginn.tools.specialty_analysis.tool", "SpecialtyAnalysisTool"),
    ("huginn.tools.fem_tool", "FEMTool"),
    ("huginn.tools.sim.transolver_tool", "TransolverTool"),
    ("huginn.tools.sim.mechanical_tool", "MechanicalTool"),
    ("huginn.tools.sim.convergence_test_tool", "ConvergenceTestTool"),
    ("huginn.tools.sim.resolve_executable_tool", "ResolveExecutableTool"),
    ("huginn.tools.vina_tool", "VinaTool"),
    ("huginn.tools.openmm_tool", "OpenMMTool"),
    ("huginn.tools.sci.symbolic_regression_tool", "SymbolicRegressionTool"),
    ("huginn.tools.symbolic_math_tool", "SymbolicMathTool"),
    ("huginn.tools.discrete_smt_tool", "DiscreteSMTTool"),
    ("huginn.tools.discrete_group_tool", "DiscreteGroupTool"),
    ("huginn.tools.discrete_oeis_tool", "DiscreteOEISTool"),
    ("huginn.tools.discrete_additive_tool", "DiscreteAdditiveTool"),
    ("huginn.tools.dynamics_discovery_tool", "DynamicsDiscoveryTool"),
    ("huginn.tools.sci.interpretable_ml_tool", "InterpretableMLTool"),
    ("huginn.tools.sci.sklearn_tool", "SklearnTool"),
    ("huginn.tools.sci.transformer_tool", "TransformerTool"),
    ("huginn.tools.sci.vae_tool", "VAETool"),
    ("huginn.tools.sci.autodiff_tool", "AutoDiffTool"),
    ("huginn.tools.numerical_tool", "NumericalTool"),
    ("huginn.tools.sci.unit_tool", "UnitTool"),
    ("huginn.tools.sci.symmetry_tool", "SymmetryTool"),
    ("huginn.tools.sci.tda_tool", "TDATool"),
    ("huginn.tools.sci.uq_tool", "UQTool"),
    ("huginn.tools.gp_tool", "GPTool"),
    ("huginn.tools.sci.pytorch_train_tool", "PyTorchTrainTool"),
    ("huginn.tools.sci.gnn_tool", "GNNTool"),
    ("huginn.tools.sci.pybamm_tool", "PyBaMMTool"),
    ("huginn.tools.sci.stat_tests_tool", "StatTestsTool"),
    ("huginn.tools.sci.descriptor_tool", "DescriptorTool"),
    ("huginn.tools.rdkit_tool", "RDKitTool"),
    ("huginn.tools.fep_tool", "FEPTool"),
    ("huginn.tools.enhanced_sampling_tool", "EnhancedSamplingTool"),
    ("huginn.tools.msm_tool", "MSMTool"),
    ("huginn.tools.inverse_design_tool", "InverseDesignTool"),
    ("huginn.tools.motif_mining_tool", "MotifMiningTool"),
    ("huginn.tools.consensus_scoring_tool", "ConsensusScoringTool"),
    ("huginn.tools.sci.evidence_fusion_tool", "EvidenceFusionTool"),
    ("huginn.tools.sci.active_learning_tool", "ActiveLearningTool"),
    ("huginn.tools.sci.ml_potential_tool", "MLPotentialTool"),
    ("huginn.tools.sci.high_throughput_tool", "HighThroughputTool"),
    ("huginn.tools.sci.multi_fidelity_tool", "MultiFidelityTool"),
    ("huginn.tools.sci.xrd_sim_tool", "XrdSimTool"),
    ("huginn.tools.design.gap_analysis_tool", "GapAnalysisTool"),
    ("huginn.tools.design.doe_tool", "DOETool"),
    ("huginn.tools.design.debugger_tool", "DebuggerTool"),
    ("huginn.tools.design_plan_tool", "DesignPlanTool"),
    ("huginn.tools.design.nudge_tool", "NudgeTool"),
    ("huginn.tools.design_atom_tool", "DesignAtomTool"),
    ("huginn.tools.design.generative_design_tool", "GenerativeDesignTool"),
    ("huginn.tools.plan_store_tool", "PlanStoreTool"),
    ("huginn.tools.image_analysis_tool", "ImageAnalysisTool"),
    ("huginn.tools.image_design_tool", "ImageDesignTool"),
    ("huginn.tools.visualize_tool", "VisualizeTool"),
    ("huginn.tools.vision_describe_tool", "VisionDescribeTool"),
    ("huginn.tools.vision_locate_tool", "VisionLocateTool"),
    ("huginn.causal.predict_intervention", "PredictInterventionTool"),
    ("huginn.causal.visual_causal_chain", "FitSCMFromObservationsTool"),
    ("huginn.causal.counterfactual_render", "CounterfactualRenderTool"),
    ("huginn.tools.characterization_tool", "CharacterizationTool"),
    ("huginn.tools.model3d_tool", "Model3DTool"),
    ("huginn.tools.browser_tool", "BrowserTool"),
    ("huginn.tools.review_committee_tool", "ReviewCommitteeTool"),
    ("huginn.academic.thesis_audit_tool", "ThesisAuditTool"),
    ("huginn.tools.hypothesis_generator_tool", "HypothesisGeneratorTool"),
    ("huginn.tools.materials_autoresearch_tool", "MaterialsAutoResearchTool"),
    ("huginn.tools.nuwa_persona_tool", "NuwaPersonaTool"),
    ("huginn.tools.subagent_tool", "SubagentTool"),
    ("huginn.tools.materials_database_tool", "MaterialsDatabaseTool"),
    ("huginn.tools.experimental_data_tool", "ExperimentalDataTool"),
    ("huginn.tools.thermo_tool", "ThermoTool"),
    ("huginn.tools.wetlab_rpc_tool", "WetlabRpcTool"),
    ("huginn.tools.experiment_protocol_tool", "ExperimentProtocolTool"),
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


def validate_tool_specs() -> list[str]:
    """Validate every (module, class) spec in the builtin tool lists.

    Each entry is a fully-qualified string ``"huginn.<module>.<path>", "ClassName"``.
    A typo'd module path or class name here fails silently at registration time
    (the tool simply never appears), which is a classic source of "missing tool"
    bugs. This helper resolves every spec eagerly and surfaces structural errors:

    - A spec whose module path or class name is broken **inside huginn** raises
      ``ImportError`` / ``AttributeError`` immediately (a real bug — the tool
      would silently vanish at runtime).
    - A spec that only fails because an optional third-party dependency
      (e.g. ``rdkit``) is absent is reported as skipped, matching the lenient
      behaviour of ``_do_register``.

    Returns the list of resolved class names.
    """
    import importlib

    resolved: list[str] = []
    for module_name, class_name in [*_CORE_MODULES, *_OPTIONAL_MODULES]:
        try:
            mod = importlib.import_module(module_name)
        except ImportError as exc:
            # A huginn-internal module that cannot be imported is a spec bug.
            # A missing third-party dependency is a normal "skip" case.
            if module_name.startswith("huginn."):
                raise
            logger.warning(
                "validate_tool_specs: skipping %s.%s (missing dep: %s)",
                module_name,
                class_name,
                exc.name or exc,
            )
            continue
        cls = getattr(mod, class_name)
        resolved.append(f"{module_name}.{class_name}:{cls.__name__}")
    return resolved


def _collect_top_level_imports(path: str, _depth: int = 0) -> set[str]:
    """Collect the set of third-party packages a module depends on.

    Covers (a) module-level imports and (b) imports inside function/class
    bodies — the common ``try: import torch`` lazy pattern. Additionally, if
    the module is a *pure shim* (only re-export statements, e.g.
    ``huginn/tools/rdkit_tool.py`` forwarding to ``sci/rdkit_tool.py``), it is
    followed to its real implementation so the shim's heavy dependencies are
    surfaced instead of silently skipped.

    Non-shim modules (which contain logic) are NOT followed into their
    ``huginn.*`` imports — otherwise a tool importing a broad facade like
    ``huginn.llm`` would inherit that facade's unrelated lazy dependencies.

    Returns raw top-level names (stdlib / huginn / third-party); the caller
    filters out stdlib and huginn-internal.
    """
    import ast
    import importlib.util
    import tokenize

    if _depth > 4:  # guard against pathological shim chains
        return set()

    with tokenize.open(path) as f:
        tree = ast.parse(f.read(), filename=path)

    deps: set[str] = set()
    internal_targets: list[str] = []
    has_logic = False

    def _collect(node: ast.AST) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top.startswith("huginn"):
                    internal_targets.append(alias.name)
                else:
                    deps.add(top)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            top = node.module.split(".")[0]
            if top.startswith("huginn"):
                internal_targets.append(node.module)
            else:
                deps.add(top)

    def _walk_body(node: ast.AST) -> None:
        """Collect imports nested inside a function/class body (lazy deps)."""
        for child in ast.iter_child_nodes(node):
            _collect(child)
            _walk_body(child)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            has_logic = True
            _walk_body(node)
        else:
            _collect(node)

    # Only follow re-export targets for pure shims (no logic). This surfaces a
    # shim's real deps without letting a tool inherit a broad facade's params.
    if not has_logic:
        for target in internal_targets:
            spec = importlib.util.find_spec(target)
            if spec is None or not spec.origin or spec.origin.endswith("__init__.py"):
                continue
            deps |= _collect_top_level_imports(spec.origin, _depth + 1)

    return deps


def probe_tool_dependencies(
    tool_lists: list[tuple[str, str]] | None = None,
) -> dict[str, list[str]]:
    """Detect optional tools whose third-party deps are missing at runtime.

    Some optional tools import heavy libs (``torch``, ``pymatgen``, ``rdkit``,
    ...) at module top-level but wrap them in ``try/except`` so the module
    still *imports* when the lib is absent — the tool registers with a name
    but has no real capability. This helper statically scans each tool
    module's module-level imports and reports which third-party deps are
    missing, so operators get an explicit warning instead of a silent no-op.

    Returns a mapping of ``{module_name: [missing_dep, ...]}`` for affected
    modules (empty dict when everything is satisfied).
    """
    import importlib.util

    tool_lists = tool_lists if tool_lists is not None else _OPTIONAL_MODULES
    stdlib = sys.stdlib_module_names
    report: dict[str, list[str]] = {}

    for module_name, _class_name in tool_lists:
        try:
            mod = importlib.import_module(module_name)
        except ImportError:
            # Module itself cannot be imported (a hard dep is missing) — that
            # case is already surfaced by _do_register / validate_tool_specs.
            continue
        path = getattr(mod, "__file__", None)
        if not path:
            continue
        deps = _collect_top_level_imports(path)
        if not deps:
            continue
        missing: list[str] = []
        for dep in sorted(deps):
            if dep.startswith("huginn") or dep in stdlib:
                continue
            if importlib.util.find_spec(dep) is None:
                missing.append(dep)
        if missing:
            report[module_name] = missing
            logger.warning(
                "Tool %s registered but its deps are missing — capability degraded: %s",
                module_name,
                ", ".join(missing),
            )
    return report


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
        logger.info(
            f"Skipped {len(skipped)} core tools (missing deps): {', '.join(skipped[:5])}"
        )
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
        logger.info(
            f"Skipped {len(skipped)} optional tools (missing deps): {', '.join(skipped[:5])}"
        )

    # Science-skills bridge
    try:
        from huginn.plugins.science_skills_bridge import register_science_skills

        science_names = register_science_skills()
        registered.extend(science_names)
        logger.info(f"Registered {len(science_names)} science-skills bridge tools")
    except Exception as exc:
        logger.warning(f"Science-skills bridge registration failed: {exc}")

    _rebuild_dispatch_tables()

    # Static probe of optional tools' third-party deps — warn loudly when a
    # tool registers with a name but its backing libraries are absent, so
    # capability degradation isn't silent.
    try:
        probe_tool_dependencies(pending)
    except Exception as exc:
        logger.warning(f"Tool dependency probe failed: {exc}")

    # Start system health monitor if enabled
    try:
        from huginn.feature_flags import FeatureFlags

        if FeatureFlags.shared().is_enabled("system_health_monitor"):
            from huginn.diagnostics.system_health import SystemHealthMonitor

            SystemHealthMonitor.shared().start()
    except Exception as exc:
        logger.warning(f"System health monitor failed to start: {exc}")

    logger.info(
        f"[tools] registered {len(registered)} optional tools (total: {len(ToolRegistry.list_tools())})"
    )
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
