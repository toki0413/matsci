"""Workflow engine package."""

# Import submodules to auto-register workflow templates
from huginn.workflows import (
    templates_cfd,
    templates_fea,
    templates_qc,
    templates_symbolic,
)
from huginn.workflows.engine import (
    ComputationalStage,
    RetryPolicy,
    ValidationRule,
    WorkflowEngine,
    WorkflowResult,
)

__all__ = [
    "WorkflowEngine",
    "ComputationalStage",
    "WorkflowResult",
    "ValidationRule",
    "RetryPolicy",
    "templates_cfd",
    "templates_fea",
    "templates_qc",
    "templates_symbolic",
]
