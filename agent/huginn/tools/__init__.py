"""Huginn tools package."""

from __future__ import annotations

from typing import Any

from huginn.tools.base import HuginnTool
from huginn.tools.registry import ToolRegistry, register_tool

__all__ = ["HuginnTool", "ToolRegistry", "register_tool", "register_all_tools"]


def register_all_tools(config: Any | None = None) -> list[str]:
    """Register every built-in tool to the global registry.

    If ``config`` is provided, the execution backend (local sandbox or remote
    HPC) and per-tool executable paths are wired into the simulation tools.
    Calling this function multiple times is safe; subsequent calls are no-ops.
    """
    if ToolRegistry.list_tools():
        return ToolRegistry.list_tools()

    # Local imports avoid heavy dependencies at package-import time.
    from huginn.config import HuginnConfig
    from huginn.evaluation.evaluation_tool import EvaluationTool
    from huginn.execution.remote_executor import build_executor
    from huginn.plugins.autoresearch import AutoresearchTool
    from huginn.rag.rag_tool import RAGTool
    from huginn.tools.abaqus_tool import AbaqusTool
    from huginn.tools.active_learning_tool import ActiveLearningTool
    from huginn.tools.autodiff_tool import AutoDiffTool
    from huginn.tools.bash_tool import BashTool
    from huginn.tools.characterization_tool import CharacterizationTool
    from huginn.tools.code_tool import CodeTool
    from huginn.tools.comsol_tool import ComsolTool
    from huginn.tools.cp2k_tool import Cp2kTool
    from huginn.tools.database_tool import DatabaseTool
    from huginn.tools.descriptor_tool import DescriptorTool
    from huginn.tools.diagnose_tool import DiagnoseTool
    from huginn.tools.diff_tool import DiffTool
    from huginn.tools.experimental_data_tool import ExperimentalDataTool
    from huginn.tools.extract_tool import ExtractTool
    from huginn.tools.file_edit_tool import FileEditTool
    from huginn.tools.file_read_tool import FileReadTool
    from huginn.tools.file_write_tool import FileWriteTool
    from huginn.tools.git_tool import GitTool
    from huginn.tools.gp_tool import GPTool
    from huginn.tools.job_tool import JobTool
    from huginn.tools.lammps_tool import LammpsTool
    from huginn.tools.lean_tool import LeanTool
    from huginn.tools.materials_database_tool import MaterialsDatabaseTool
    from huginn.tools.memory_tool import RecallTool, RememberTool
    from huginn.tools.ml_potential_tool import MLPotentialTool
    from huginn.tools.openfoam_tool import OpenFoamTool
    from huginn.tools.orchestrate_tool import OrchestrateTool
    from huginn.tools.packing_tool import PackingTool
    from huginn.tools.potential_tool import PotentialTool
    from huginn.tools.qe_tool import QuantumEspressoTool
    from huginn.tools.report_tool import ReportTool
    from huginn.tools.structure_tool import StructureTool
    from huginn.tools.symbolic_math_tool import SymbolicMathTool
    from huginn.tools.symbolic_regression_tool import SymbolicRegressionTool
    from huginn.tools.uq_tool import UQTool
    from huginn.tools.validate_tool import ValidateTool
    from huginn.tools.vasp_tool import VaspTool
    from huginn.tools.visualize_tool import VisualizeTool

    resolved_config = config if config is not None else HuginnConfig.from_env()
    executor = build_executor(resolved_config)

    def _tool_kwargs(cls: type) -> dict[str, Any]:
        """Build init kwargs for tool classes that accept sandbox/executables."""
        import inspect

        sig = inspect.signature(cls.__init__)
        params = sig.parameters
        kwargs: dict[str, Any] = {}

        if cls is VaspTool and "vasp_executable" in params:
            kwargs["vasp_executable"] = resolved_config.vasp_executable
        if cls is LammpsTool and "lammps_executable" in params:
            kwargs["lammps_executable"] = resolved_config.lammps_executable

        if cls is MaterialsDatabaseTool:
            if "mp_api_key" in params:
                kwargs["mp_api_key"] = resolved_config.mp_api_key
            if "oqmd_api_key" in params:
                kwargs["oqmd_api_key"] = resolved_config.oqmd_api_key

        if "sandbox" in params:
            kwargs["sandbox"] = executor
        return kwargs

    tool_classes = [
        StructureTool,
        ExtractTool,
        JobTool,
        DatabaseTool,
        PotentialTool,
        DiffTool,
        ValidateTool,
        DiagnoseTool,
        VaspTool,
        LammpsTool,
        SymbolicRegressionTool,
        ReportTool,
        LeanTool,
        SymbolicMathTool,
        AutoDiffTool,
        ComsolTool,
        QuantumEspressoTool,
        Cp2kTool,
        OpenFoamTool,
        PackingTool,
        AbaqusTool,
        CodeTool,
        FileReadTool,
        FileWriteTool,
        FileEditTool,
        BashTool,
        GitTool,
        UQTool,
        GPTool,
        RememberTool,
        RecallTool,
        OrchestrateTool,
        RAGTool,
        EvaluationTool,
        MaterialsDatabaseTool,
        ExperimentalDataTool,
        DescriptorTool,
        VisualizeTool,
        ActiveLearningTool,
        MLPotentialTool,
        CharacterizationTool,
        AutoresearchTool,
    ]

    for cls in tool_classes:
        ToolRegistry.register(cls(**_tool_kwargs(cls)))

    return ToolRegistry.list_tools()
