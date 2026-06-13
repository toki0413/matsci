"""Execution layer — turns workflow descriptions into real actions.

This is the bridge between "Agent tells you what to do" and "Agent does it for you".

Components:
  - InputFileGenerator: Creates calculation input files from high-level specs
  - ExecutionOrchestrator: Runs workflow stages, handles dependencies, monitors progress
  - ResultParser: Extracts physical insights from raw output files
  - AutoFixLoop: Detects failures, applies fixes, and retries automatically
"""

from matsci_agent.execution.input_generator import InputFileGenerator
from matsci_agent.execution.orchestrator import ExecutionOrchestrator, StageResult
from matsci_agent.execution.result_parser import ResultParser
from matsci_agent.execution.autofix import AutoFixLoop

__all__ = [
    "InputFileGenerator",
    "ExecutionOrchestrator",
    "StageResult",
    "ResultParser",
    "AutoFixLoop",
]
