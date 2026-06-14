"""Huginn tools package."""

from huginn.tools.base import HuginnTool
from huginn.tools.registry import ToolRegistry, register_tool

__all__ = ["HuginnTool", "ToolRegistry", "register_tool", "register_all_tools"]


def register_all_tools() -> list[str]:
    """Register every built-in tool to the global registry.

    Returns the list of registered tool names. Calling this function multiple
    times is safe; subsequent calls are no-ops.
    """
    if ToolRegistry.list_tools():
        return ToolRegistry.list_tools()

    # Local imports avoid heavy dependencies at package-import time.
    from huginn.tools.structure_tool import StructureTool
    from huginn.tools.extract_tool import ExtractTool
    from huginn.tools.job_tool import JobTool
    from huginn.tools.database_tool import DatabaseTool
    from huginn.tools.potential_tool import PotentialTool
    from huginn.tools.diff_tool import DiffTool
    from huginn.tools.validate_tool import ValidateTool
    from huginn.tools.diagnose_tool import DiagnoseTool
    from huginn.tools.vasp_tool import VaspTool
    from huginn.tools.lammps_tool import LammpsTool
    from huginn.tools.symbolic_regression_tool import SymbolicRegressionTool
    from huginn.tools.report_tool import ReportTool
    from huginn.tools.lean_tool import LeanTool
    from huginn.tools.symbolic_math_tool import SymbolicMathTool
    from huginn.tools.autodiff_tool import AutoDiffTool
    from huginn.tools.comsol_tool import ComsolTool
    from huginn.tools.qe_tool import QuantumEspressoTool
    from huginn.tools.cp2k_tool import Cp2kTool
    from huginn.tools.openfoam_tool import OpenFoamTool
    from huginn.tools.packing_tool import PackingTool
    from huginn.tools.abaqus_tool import AbaqusTool
    from huginn.tools.code_tool import CodeTool
    from huginn.tools.file_read_tool import FileReadTool
    from huginn.tools.file_write_tool import FileWriteTool
    from huginn.tools.file_edit_tool import FileEditTool
    from huginn.tools.bash_tool import BashTool
    from huginn.tools.git_tool import GitTool
    from huginn.tools.uq_tool import UQTool
    from huginn.tools.gp_tool import GPTool
    from huginn.tools.memory_tool import RememberTool, RecallTool
    from huginn.tools.orchestrate_tool import OrchestrateTool
    from huginn.rag.rag_tool import RAGTool
    from huginn.evaluation.evaluation_tool import EvaluationTool

    tools = [
        StructureTool(),
        ExtractTool(),
        JobTool(),
        DatabaseTool(),
        PotentialTool(),
        DiffTool(),
        ValidateTool(),
        DiagnoseTool(),
        VaspTool(),
        LammpsTool(),
        SymbolicRegressionTool(),
        ReportTool(),
        LeanTool(),
        SymbolicMathTool(),
        AutoDiffTool(),
        ComsolTool(),
        QuantumEspressoTool(),
        Cp2kTool(),
        OpenFoamTool(),
        PackingTool(),
        AbaqusTool(),
        CodeTool(),
        FileReadTool(),
        FileWriteTool(),
        FileEditTool(),
        BashTool(),
        GitTool(),
        UQTool(),
        GPTool(),
        RememberTool(),
        RecallTool(),
        OrchestrateTool(),
        RAGTool(),
        EvaluationTool(),
    ]

    for tool in tools:
        ToolRegistry.register(tool)

    return ToolRegistry.list_tools()
