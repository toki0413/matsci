"""Execution layer — turns workflow descriptions into real actions.

This is the bridge between "Agent tells you what to do" and "Agent does it for you".

Components:
  - ExecutionOrchestrator: Runs workflow stages, handles dependencies, monitors progress
  - AutoFixLoop: Detects failures, applies fixes, and retries automatically
"""

from huginn.execution.autofix import AutoFixLoop
from huginn.execution.orchestrator import ExecutionOrchestrator, StageResult

__all__ = [
    "ExecutionOrchestrator",
    "StageResult",
    "AutoFixLoop",
]
