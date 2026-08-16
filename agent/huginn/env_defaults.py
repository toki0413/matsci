"""Centralized registry for HUGINN_ environment variables.

This module provides:

1. **Typed accessors** — ``get_str``, ``get_bool``, ``get_int``,
   ``get_float``, ``get_path`` that standardize the parsing of
   environment variables.  Before this module, the codebase used at
   least four different patterns for boolean env vars (``== "1"``,
   ``!= "0"``, ``.lower() in ("1","true","yes")``, ``.lower() != "false"``),
   which made it hard to reason about what a given flag actually accepted.

2. **A registry** — ``ENV_REGISTRY`` documents the known HUGINN_
   environment variables, their default values, types, and the module(s)
   that consume them.  This is a *living reference*, not an exhaustive
   enforcement layer — call sites migrate to the typed accessors
   gradually.

Usage::

    from huginn.env_defaults import get_bool, get_int

    if get_bool("HUGINN_DEV_MODE", default=False):
        ...
    timeout = get_int("HUGINN_STREAM_IDLE_TIMEOUT", default=60)

Design notes
------------

- The accessors read ``os.environ`` on every call (no caching).  This
  matches the existing pattern where config can be mutated at runtime
  via ``os.environ[...] = ...`` (e.g. ``routes/config.py`` applies
  POSTed params to env).  A read-through cache would break that.
- ``get_bool`` accepts ``1/true/yes/on`` (case-insensitive) as True
  and everything else as False.  This is a superset of all four
  legacy patterns, so migrating call sites does not change behaviour.
- The registry is intentionally a plain dict, not a class hierarchy.
  ponytail: no one wants to instantiate ``EnvVar`` objects to read a
  flag.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

__all__ = [
    "get_str",
    "get_bool",
    "get_int",
    "get_float",
    "get_path",
    "ENV_REGISTRY",
    "EnvCategory",
]

# ── Typed accessors ────────────────────────────────────────────────

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off", "", "none"})


def get_str(key: str, *, default: str = "") -> str:
    """Read a string env var, stripped of surrounding whitespace."""
    val = os.environ.get(key)
    if val is None:
        return default
    return val.strip()


def get_bool(key: str, *, default: bool = False) -> bool:
    """Read a boolean env var.

    Accepts ``1/true/yes/on`` (case-insensitive) as True,
    ``0/false/no/off/""/none`` as False.
    Unset falls back to *default*; unrecognized values fall back to
    *default* as well (matching the most lenient legacy pattern).
    """
    val = os.environ.get(key)
    if val is None:
        return default
    lowered = val.strip().lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    return default


def get_int(key: str, *, default: int = 0) -> int:
    """Read an integer env var. Unset or unparseable → *default*."""
    val = os.environ.get(key)
    if val is None or not val.strip():
        return default
    try:
        return int(val.strip())
    except ValueError:
        return default


def get_float(key: str, *, default: float = 0.0) -> float:
    """Read a float env var. Unset or unparseable → *default*."""
    val = os.environ.get(key)
    if val is None or not val.strip():
        return default
    try:
        return float(val.strip())
    except ValueError:
        return default


def get_path(key: str, *, default: Path | None = None) -> Path | None:
    """Read a path env var. Unset → *default* (or None)."""
    val = os.environ.get(key)
    if val is None or not val.strip():
        return default
    return Path(val.strip()).expanduser()


# ── Registry ───────────────────────────────────────────────────────


class EnvCategory:
    """Category labels for grouping env vars in docs/diagnostics."""

    RUNTIME = "runtime"
    AUTH = "auth"
    ENCRYPTION = "encryption"
    SANDBOX = "sandbox"
    RATE_LIMIT = "rate_limit"
    LLM = "llm"
    AGENT = "agent"
    MEMORY = "memory"
    FEATURE_FLAG = "feature_flag"
    METACOG = "metacog"
    HPC = "hpc"
    LITERATURE = "literature"
    MIDDLEWARE = "middleware"
    LOGGING = "logging"
    GOVERNANCE = "governance"
    RCB = "rcb"


# Each entry: (category, type, default, description, primary_consumer)
# ``type`` is one of: str, bool, int, float, path, json
# ``primary_consumer`` is the module that owns the variable (not
# necessarily the only reader).
ENV_REGISTRY: dict[str, dict[str, Any]] = {
    # ── Runtime & workspace ─────────────────────────────────────────
    "HUGINN_CACHE_DIR": {
        "category": EnvCategory.RUNTIME,
        "type": "path",
        "default": "~/.huginn",
        "description": "Runtime home directory (audit log, memory, credentials, …).",
        "consumer": "huginn.utils.runtime.get_runtime_home",
    },
    "HUGINN_WORKSPACE": {
        "category": EnvCategory.RUNTIME,
        "type": "path",
        "default": ".",
        "description": "Current workspace path for sandbox scoping and HPC jobs.",
        "consumer": "huginn.config",
    },
    "HUGINN_CONFIG_FILE": {
        "category": EnvCategory.RUNTIME,
        "type": "path",
        "default": "<workspace>/huginn.toml",
        "description": "Override path for the Huginn TOML config file.",
        "consumer": "huginn.config.get_config",
    },
    "HUGINN_CHECKPOINTER_PATH": {
        "category": EnvCategory.RUNTIME,
        "type": "path",
        "default": None,
        "description": "Checkpoint storage path. None = in-memory checkpoints.",
        "consumer": "huginn.persistence.checkpointer",
    },
    "HUGINN_TRANSCRIPT_DIR": {
        "category": EnvCategory.RUNTIME,
        "type": "path",
        "default": None,
        "description": "Directory for conversation transcripts.",
        "consumer": "huginn.events.transcript",
    },
    # ── Auth & security ─────────────────────────────────────────────
    "HUGINN_API_KEY": {
        "category": EnvCategory.AUTH,
        "type": "str",
        "default": None,
        "description": "Shared API key (legacy mode). Also used as JWT secret fallback.",
        "consumer": "huginn.security.auth",
    },
    "HUGINN_ADMIN_API_KEY": {
        "category": EnvCategory.AUTH,
        "type": "str",
        "default": None,
        "description": "Admin endpoint API key.",
        "consumer": "huginn.security.auth",
    },
    "HUGINN_JWT_SECRET": {
        "category": EnvCategory.AUTH,
        "type": "str",
        "default": None,
        "description": "JWT signing secret. Falls back to HUGINN_API_KEY.",
        "consumer": "huginn.security.auth",
    },
    "HUGINN_DEV_MODE": {
        "category": EnvCategory.AUTH,
        "type": "bool",
        "default": False,
        "description": "Dev mode bypasses authentication checks.",
        "consumer": "huginn.security.auth",
    },
    "HUGINN_ENV": {
        "category": EnvCategory.AUTH,
        "type": "str",
        "default": "",
        "description": "Environment label. 'production' triggers docs hiding.",
        "consumer": "huginn.security.auth",
    },
    "HUGINN_ENFORCE_WRITE_CAPABILITY": {
        "category": EnvCategory.AUTH,
        "type": "bool",
        "default": True,
        "description": "Enforce write capability check on mutating endpoints.",
        "consumer": "huginn.security.auth",
    },
    "HUGINN_AUDIT_SIGNING_KEY": {
        "category": EnvCategory.AUTH,
        "type": "str",
        "default": None,
        "description": "HMAC signing key for audit log tamper evidence.",
        "consumer": "huginn.security.audit",
    },
    # ── Encryption ──────────────────────────────────────────────────
    "HUGINN_ENCRYPTION_PASSWORD": {
        "category": EnvCategory.ENCRYPTION,
        "type": "str",
        "default": None,
        "description": "Master encryption password (config + credentials).",
        "consumer": "huginn.config",
    },
    "HUGINN_ENCRYPTION_KEY_FILE": {
        "category": EnvCategory.ENCRYPTION,
        "type": "path",
        "default": None,
        "description": "Path to Fernet base64-urlsafe key file.",
        "consumer": "huginn.config",
    },
    "HUGINN_ENCRYPTION_ENABLED": {
        "category": EnvCategory.ENCRYPTION,
        "type": "bool",
        "default": False,
        "description": "Enable at-rest encryption.",
        "consumer": "huginn.config",
    },
    "HUGINN_ENCRYPT_RAG_DOCS": {
        "category": EnvCategory.ENCRYPTION,
        "type": "bool",
        "default": True,
        "description": "Encrypt RAG document content at rest.",
        "consumer": "huginn.config",
    },
    "HUGINN_ENCRYPT_RAG_META": {
        "category": EnvCategory.ENCRYPTION,
        "type": "bool",
        "default": True,
        "description": "Encrypt RAG metadata at rest.",
        "consumer": "huginn.config",
    },
    # ── Sandbox & execution ─────────────────────────────────────────
    "HUGINN_ALLOW_LOCAL_BASH": {
        "category": EnvCategory.SANDBOX,
        "type": "bool",
        "default": False,
        "description": "Allow local bash execution without container isolation.",
        "consumer": "huginn.security.execution",
    },
    "HUGINN_CONTAINER_RUNTIME": {
        "category": EnvCategory.SANDBOX,
        "type": "str",
        "default": "none",
        "description": "Container runtime: none/docker/podman/apptainer/singularity.",
        "consumer": "huginn.security.execution",
    },
    "HUGINN_CONTAINER_IMAGE": {
        "category": EnvCategory.SANDBOX,
        "type": "str",
        "default": None,
        "description": "Container image for sandboxed execution.",
        "consumer": "huginn.security.execution",
    },
    "HUGINN_CODEACT_MEM_CAP": {
        "category": EnvCategory.SANDBOX,
        "type": "int",
        "default": 2147483648,
        "description": "CodeAct memory cap in bytes (default 2 GiB).",
        "consumer": "huginn.security.code_act_sandbox",
    },
    "HUGINN_RESTRICTED_PYTHON": {
        "category": EnvCategory.SANDBOX,
        "type": "bool",
        "default": True,
        "description": "Use restricted Python exec in CodeAct sandbox.",
        "consumer": "huginn.cli.rcb_runner",
    },
    # ── Rate limiting ───────────────────────────────────────────────
    "HUGINN_RATE_LIMIT_ENABLED": {
        "category": EnvCategory.RATE_LIMIT,
        "type": "bool",
        "default": True,
        "description": "Master switch for rate limiting.",
        "consumer": "huginn.security.rate_limiter",
    },
    "HUGINN_RATE_LIMIT_TOKENS_PER_TURN": {
        "category": EnvCategory.RATE_LIMIT,
        "type": "int",
        "default": 100000,
        "description": "Max tokens per turn.",
        "consumer": "huginn.security.rate_limiter",
    },
    "HUGINN_RATE_LIMIT_TOKENS_PER_SECOND": {
        "category": EnvCategory.RATE_LIMIT,
        "type": "int",
        "default": 5000,
        "description": "Max tokens per second.",
        "consumer": "huginn.security.rate_limiter",
    },
    "HUGINN_RATE_LIMIT_TOTAL_COST_USD": {
        "category": EnvCategory.RATE_LIMIT,
        "type": "float",
        "default": 10.0,
        "description": "Total cost cap in USD.",
        "consumer": "huginn.security.rate_limiter",
    },
    # ── LLM configuration ───────────────────────────────────────────
    "HUGINN_PROVIDER": {
        "category": EnvCategory.LLM,
        "type": "str",
        "default": "default",
        "description": "LLM provider name.",
        "consumer": "huginn.config",
    },
    "HUGINN_MODEL": {
        "category": EnvCategory.LLM,
        "type": "str",
        "default": None,
        "description": "LLM model name.",
        "consumer": "huginn.config",
    },
    "HUGINN_BASE_URL": {
        "category": EnvCategory.LLM,
        "type": "str",
        "default": None,
        "description": "LLM API base URL.",
        "consumer": "huginn.config",
    },
    "HUGINN_THINKING": {
        "category": EnvCategory.LLM,
        "type": "str",
        "default": "",
        "description": "Thinking intensity: low/medium/high or JSON.",
        "consumer": "huginn.config",
    },
    "HUGINN_MAX_TOKENS": {
        "category": EnvCategory.LLM,
        "type": "int",
        "default": None,
        "description": "Max response tokens.",
        "consumer": "huginn.config",
    },
    "HUGINN_TORCH_DEVICE": {
        "category": EnvCategory.LLM,
        "type": "str",
        "default": "cpu",
        "description": "Torch device (cpu/cuda/mps).",
        "consumer": "huginn.tools.sci",
    },
    # ── Agent behaviour ─────────────────────────────────────────────
    "HUGINN_AUTO_APPROVE": {
        "category": EnvCategory.AGENT,
        "type": "bool",
        "default": False,
        "description": "Auto-approve tool execution without human confirmation.",
        "consumer": "huginn.tools.adapter",
    },
    "HUGINN_PERM_COST_BUDGET_HOURS": {
        "category": EnvCategory.AGENT,
        "type": "float",
        "default": None,
        "description": "Permission cost budget (CPU hours). Tools exceeding it escalate to ASK.",
        "consumer": "huginn.permissions.PermissionConfig.cost_budget_hours",
    },
    "HUGINN_PERM_TRUST_ADAPTIVE": {
        "category": EnvCategory.AGENT,
        "type": "bool",
        "default": False,
        "description": "Enable trust-adaptive permission: low trust forces ASK, high trust auto-approves medium risk.",
        "consumer": "huginn.permissions.PermissionConfig.trust_adaptive",
    },
    "HUGINN_MAX_TOOL_OUTPUT_TOKENS": {
        "category": EnvCategory.AGENT,
        "type": "int",
        "default": 25000,
        "description": "Max tokens in tool output before truncation.",
        "consumer": "huginn.config",
    },
    "HUGINN_MAX_TOOL_CALLS": {
        "category": EnvCategory.AGENT,
        "type": "int",
        "default": 15,
        "description": "Max tool calls per turn.",
        "consumer": "huginn.agent_config",
    },
    "HUGINN_EXTREME_DISPATCH": {
        "category": EnvCategory.AGENT,
        "type": "bool",
        "default": False,
        "description": "Extreme dispatch mode for aggressive parallelism.",
        "consumer": "huginn.agent.core",
    },
    "HUGINN_TELEMETRY_ENABLED": {
        "category": EnvCategory.AGENT,
        "type": "bool",
        "default": True,
        "description": "Enable telemetry collection.",
        "consumer": "huginn.config",
    },
    "HUGINN_PRIVACY_REDACT_SECRETS": {
        "category": EnvCategory.AGENT,
        "type": "bool",
        "default": True,
        "description": "Redact detected secrets from tool I/O.",
        "consumer": "huginn.config",
    },
    "HUGINN_PRIVACY_BLOCK_ON_SECRETS": {
        "category": EnvCategory.AGENT,
        "type": "bool",
        "default": False,
        "description": "Block execution when secrets are detected.",
        "consumer": "huginn.config",
    },
    # ── Memory ──────────────────────────────────────────────────────
    "HUGINN_MEMORY_DECAY_ENABLED": {
        "category": EnvCategory.MEMORY,
        "type": "bool",
        "default": False,
        "description": "Enable memory decay.",
        "consumer": "huginn.config",
    },
    "HUGINN_WM_TOKEN_BUDGET": {
        "category": EnvCategory.MEMORY,
        "type": "int",
        "default": 8192,
        "description": "Working memory sliding window token capacity.",
        "consumer": "huginn.config",
    },
    "HUGINN_EM_RECALL_TOP_K": {
        "category": EnvCategory.MEMORY,
        "type": "int",
        "default": 5,
        "description": "Episodic memory recall top-k.",
        "consumer": "huginn.config",
    },
    # ── Middleware ──────────────────────────────────────────────────
    "HUGINN_MAX_BODY_SIZE_MB": {
        "category": EnvCategory.MIDDLEWARE,
        "type": "int",
        "default": 16,
        "description": "Max request body size in MB.",
        "consumer": "huginn.middleware.limits",
    },
    "HUGINN_REQUEST_TIMEOUT_SEC": {
        "category": EnvCategory.MIDDLEWARE,
        "type": "int",
        "default": 300,
        "description": "Request timeout in seconds.",
        "consumer": "huginn.middleware.limits",
    },
    "HUGINN_WS_MAX_CONNECTIONS": {
        "category": EnvCategory.MIDDLEWARE,
        "type": "int",
        "default": 50,
        "description": "Max concurrent WebSocket connections.",
        "consumer": "huginn.middleware.ws_governance",
    },
    # ── Logging ─────────────────────────────────────────────────────
    "HUGINN_JSON_LOGS": {
        "category": EnvCategory.LOGGING,
        "type": "bool",
        "default": True,
        "description": "Emit structured JSON logs.",
        "consumer": "huginn.utils.json_logging",
    },
    "HUGINN_LOG_LEVEL": {
        "category": EnvCategory.LOGGING,
        "type": "str",
        "default": "INFO",
        "description": "Log level (DEBUG/INFO/WARNING/ERROR).",
        "consumer": "huginn.utils.json_logging",
    },
    # ── Governance ──────────────────────────────────────────────────
    "HUGINN_GOVERNANCE_DEFAULT_DECISION": {
        "category": EnvCategory.GOVERNANCE,
        "type": "str",
        "default": "deny",
        "description": "Default governance decision: allow or deny.",
        "consumer": "huginn.governance",
    },
    "HUGINN_MAINTENANCE": {
        "category": EnvCategory.GOVERNANCE,
        "type": "bool",
        "default": False,
        "description": "Maintenance mode (returns 503).",
        "consumer": "huginn.middleware.maintenance",
    },
}
