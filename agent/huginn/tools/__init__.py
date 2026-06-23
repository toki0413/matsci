"""Huginn tools package."""

from __future__ import annotations

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
    executor = build_executor(resolved_config)

    # Ensure Bourbaki (math_anything) is discoverable if installed externally
    _bourbaki_path = r"C:\Users\wanzh\Desktop\math-anything\math-anything"
    if _bourbaki_path not in sys.path:
        sys.path.insert(0, _bourbaki_path)

    def _tool_kwargs(cls: type) -> dict[str, Any]:
        """Build init kwargs for tool classes that accept sandbox/executables."""
        import inspect

        sig = inspect.signature(cls.__init__)
        params = sig.parameters
        kwargs: dict[str, Any] = {}

        if cls.__name__ == "VaspTool" and "vasp_executable" in params:
            kwargs["vasp_executable"] = resolved_config.vasp_executable
        if cls.__name__ == "LammpsTool" and "lammps_executable" in params:
            kwargs["lammps_executable"] = resolved_config.lammps_executable

        if cls.__name__ == "MaterialsDatabaseTool":
            if "mp_api_key" in params:
                kwargs["mp_api_key"] = resolved_config.mp_api_key
            if "oqmd_api_key" in params:
                kwargs["oqmd_api_key"] = resolved_config.oqmd_api_key

        if "sandbox" in params:
            kwargs["sandbox"] = executor
        return kwargs

    # Core tools (always expected to be available)
    core_modules = [
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
        ("huginn.tools.extract_tool", "ExtractTool"),
        ("huginn.tools.job_tool", "JobTool"),
        ("huginn.tools.database_tool", "DatabaseTool"),
        ("huginn.tools.potential_tool", "PotentialTool"),
        ("huginn.tools.structure_tool", "StructureTool"),
        ("huginn.tools.report_tool", "ReportTool"),
        ("huginn.tools.orchestrate_tool", "OrchestrateTool"),
        ("huginn.tools.memory_tool", "RememberTool"),
        ("huginn.tools.memory_tool", "RecallTool"),
        ("huginn.tools.lean_tool", "LeanTool"),
        ("huginn.evaluation.evaluation_tool", "EvaluationTool"),
        ("huginn.rag.rag_tool", "RAGTool"),
        ("huginn.plugins.autoresearch", "AutoresearchTool"),
    ]

    # Optional simulation / science tools (skip if deps missing)
    optional_modules = [
        ("huginn.tools.vasp_tool", "VaspTool"),
        ("huginn.tools.lammps_tool", "LammpsTool"),
        ("huginn.tools.symbolic_regression_tool", "SymbolicRegressionTool"),
        ("huginn.tools.symbolic_math_tool", "SymbolicMathTool"),
        ("huginn.tools.autodiff_tool", "AutoDiffTool"),
        ("huginn.tools.comsol_tool", "ComsolTool"),
        ("huginn.tools.qe_tool", "QuantumEspressoTool"),
        ("huginn.tools.cp2k_tool", "Cp2kTool"),
        ("huginn.tools.openfoam_tool", "OpenFoamTool"),
        ("huginn.tools.packing_tool", "PackingTool"),
        ("huginn.tools.abaqus_tool", "AbaqusTool"),
        ("huginn.tools.uq_tool", "UQTool"),
        ("huginn.tools.gp_tool", "GPTool"),
        ("huginn.tools.materials_database_tool", "MaterialsDatabaseTool"),
        ("huginn.tools.experimental_data_tool", "ExperimentalDataTool"),
        ("huginn.tools.descriptor_tool", "DescriptorTool"),
        ("huginn.tools.visualize_tool", "VisualizeTool"),
        ("huginn.tools.active_learning_tool", "ActiveLearningTool"),
        ("huginn.tools.ml_potential_tool", "MLPotentialTool"),
        ("huginn.tools.characterization_tool", "CharacterizationTool"),
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

    return ToolRegistry.list_tools()
