"""Workflow engine package."""

from matsci_agent.workflows.engine import (
    WorkflowEngine,
    ComputationalStage,
    WorkflowResult,
    ValidationRule,
    RetryPolicy,
)

# Import submodules to auto-register workflow templates
from matsci_agent.workflows import templates_qc, templates_fea, templates_cfd, templates_symbolic

__all__ = [
    "WorkflowEngine",
    "ComputationalStage",
    "WorkflowResult",
    "ValidationRule",
    "RetryPolicy",
]
