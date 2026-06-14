"""Workflow engine package."""

from huginn.workflows.engine import (
    WorkflowEngine,
    ComputationalStage,
    WorkflowResult,
    ValidationRule,
    RetryPolicy,
)

# Import submodules to auto-register workflow templates
from huginn.workflows import templates_qc, templates_fea, templates_cfd, templates_symbolic

__all__ = [
    "WorkflowEngine",
    "ComputationalStage",
    "WorkflowResult",
    "ValidationRule",
    "RetryPolicy",
]
