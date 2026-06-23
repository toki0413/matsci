"""Security layer for Huginn.

Provides command execution sandboxing, audit logging, and credential management.
"""

from huginn.security.audit import AuditEvent, AuditLogger
from huginn.security.container_executor import ContainerExecutor
from huginn.security.execution import allow_local_bash, get_executor
from huginn.security.restricted_python import RestrictedPythonError, validate_code
from huginn.security.safe_eval import SafeEvalError, safe_eval
from huginn.security.sandbox import SandboxConfig, SandboxError, SandboxExecutor, SandboxResult
from huginn.security.script_runner import ScriptResult, ScriptRunner
from huginn.security.secrets import (
    EnvSecretBackend,
    SecretBackend,
    VaultSecretBackend,
    get_secret_backend,
)

__all__ = [
    "SandboxExecutor",
    "SandboxConfig",
    "SandboxError",
    "SandboxResult",
    "ContainerExecutor",
    "get_executor",
    "allow_local_bash",
    "SecretBackend",
    "EnvSecretBackend",
    "VaultSecretBackend",
    "get_secret_backend",
    "AuditLogger",
    "AuditEvent",
    "safe_eval",
    "SafeEvalError",
    "validate_code",
    "RestrictedPythonError",
    "ScriptRunner",
    "ScriptResult",
]
