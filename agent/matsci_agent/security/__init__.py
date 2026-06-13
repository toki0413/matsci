"""Security layer for MatSci-Agent.

Provides command execution sandboxing, audit logging, and credential management.
"""

from matsci_agent.security.sandbox import SandboxExecutor, SandboxConfig, SandboxError
from matsci_agent.security.audit import AuditLogger, AuditEvent
from matsci_agent.security.safe_eval import safe_eval, SafeEvalError
from matsci_agent.security.restricted_python import validate_code, RestrictedPythonError

__all__ = [
    "SandboxExecutor",
    "SandboxConfig",
    "SandboxError",
    "AuditLogger",
    "AuditEvent",
    "safe_eval",
    "SafeEvalError",
    "validate_code",
    "RestrictedPythonError",
]
