"""Security layer for Huginn.

Provides command execution sandboxing, audit logging, and credential management.
"""

from huginn.security.audit import AuditEvent, AuditLogger
from huginn.security.container_executor import ContainerExecutor
from huginn.security.execution import allow_local_bash, get_executor
from huginn.security.math_eval import safe_math_eval
from huginn.security.policy_engine import (
    PolicyDecision,
    PolicyEngine,
    PolicyRule,
    evaluate_command_hook,
)
from huginn.security.prompt_security import (
    untrusted_context_message,
    wrap_rag_chunks,
)
from huginn.security.rate_limiter import (
    RateLimitConfig,
    RateLimitExceeded,
    TokenRateLimiter,
    get_rate_limiter,
)
from huginn.security.restricted_python import RestrictedPythonError, validate_code
from huginn.security.safe_eval import SafeEvalError, safe_eval
from huginn.security.sandbox import (
    SandboxConfig,
    SandboxError,
    SandboxExecutor,
    SandboxResult,
    create_sandbox,
)
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
    "create_sandbox",
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
    "safe_math_eval",
    "SafeEvalError",
    "validate_code",
    "RestrictedPythonError",
    "PolicyEngine",
    "PolicyRule",
    "PolicyDecision",
    "evaluate_command_hook",
    "ScriptRunner",
    "ScriptResult",
    "TokenRateLimiter",
    "RateLimitConfig",
    "RateLimitExceeded",
    "get_rate_limiter",
    "untrusted_context_message",
    "wrap_rag_chunks",
]
