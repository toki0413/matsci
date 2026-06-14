"""Security layer for Huginn.

Provides command execution sandboxing, audit logging, and credential management.
"""

from huginn.security.sandbox import SandboxExecutor, SandboxConfig, SandboxError
from huginn.security.audit import AuditLogger, AuditEvent
from huginn.security.safe_eval import safe_eval, SafeEvalError
from huginn.security.restricted_python import validate_code, RestrictedPythonError

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
