"""Execution layer — turns workflow descriptions into real actions.

Components:
  - ExecutionOrchestrator: Runs workflow stages, handles dependencies, monitors progress
  - AutoFixLoop: Detects failures, applies fixes, and retries automatically
  - KernelSession / KernelSessionManager: 有状态 ipykernel 会话, 跨 execute 保留变量
"""

from huginn.execution.autofix import AutoFixLoop
from huginn.execution.kernel_session import (
    KernelExecResult,
    KernelSession,
    KernelSessionManager,
)
from huginn.execution.orchestrator import ExecutionOrchestrator, StageResult

__all__ = [
    "ExecutionOrchestrator",
    "StageResult",
    "AutoFixLoop",
    "KernelSession",
    "KernelSessionManager",
    "KernelExecResult",
]
