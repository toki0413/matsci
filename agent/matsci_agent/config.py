"""Configuration management for MatSci-Agent.

Supports environment variables, .env files, and config files.
"""

from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class ModelConfig:
    """A single LLM provider/model entry in the model pool."""
    alias: str  # e.g. "gpt4o", "claude-sonnet"
    provider: Literal[
        "anthropic", "openai", "ollama", "deepseek",
        "google-genai", "openrouter", "nvidia", "vllm", "local", "default"
    ]
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.7
    enabled: bool = True


@dataclass
class AgentProfileConfig:
    """A reusable agent profile that maps to a model alias + persona + tools."""
    id: str  # e.g. "lead", "coder", "reviewer"
    name: str = ""
    model_alias: str = ""
    persona: str = "default"
    tools: list[str] = field(default_factory=list)
    enabled: bool = True
    max_steps: int = 10


@dataclass
class MatSciConfig:
    """MatSci-Agent configuration."""

    # Legacy single-model settings (kept for backward compatibility)
    provider: Literal[
        "anthropic", "openai", "ollama", "deepseek",
        "google-genai", "openrouter", "nvidia", "vllm", "local", "default"
    ] = "default"
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None

    # Multi-LLM / multi-agent settings
    models: list[ModelConfig] = field(default_factory=list)
    agents: list[AgentProfileConfig] = field(default_factory=list)
    team_mode_enabled: bool = False
    max_concurrent_subagents: int = 3

    # Ollama specific
    ollama_host: str = "http://localhost:11434"

    # Computational tools
    vasp_executable: str | None = None
    lammps_executable: str | None = None

    # HPC settings
    hpc_scheduler: Literal["slurm", "pbs", "local"] = "local"
    hpc_host: str | None = None
    hpc_username: str | None = None

    # MCP server paths
    abaqus_mcp_server: str | None = None

    # Workspace
    workspace: str = "."

    # Agent behavior
    auto_approve: bool = False
    enable_exploration: bool = True
    max_parallel_branches: int = 5
    persona: str = "default"
    rag_enabled: bool = False

    # Encryption settings
    encryption_enabled: bool = False
    encryption_password: str | None = None
    encryption_key_file: str | None = None
    encrypt_rag_documents: bool = True
    encrypt_rag_metadata: bool = True
    encrypt_config: bool = False

    # Privacy / data leakage controls
    privacy_redact_secrets: bool = True
    privacy_block_on_secrets: bool = False

    @classmethod
    def from_env(cls) -> MatSciConfig:
        """Load configuration from environment variables.

        API key resolution priority:
        1. MATSCI_API_KEY (generic)
        2. Provider-specific env vars (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)
        """
        provider = os.environ.get("MATSCI_PROVIDER", "default").lower().strip()

        # Resolve API key with provider-specific fallback
        api_key = os.environ.get("MATSCI_API_KEY")
        if not api_key:
            provider_key_map = {
                "anthropic": "ANTHROPIC_API_KEY",
                "openai": "OPENAI_API_KEY",
                "deepseek": "DEEPSEEK_API_KEY",
                "google-genai": "GOOGLE_API_KEY",
                "openrouter": "OPENROUTER_API_KEY",
                "nvidia": "NVIDIA_API_KEY",
                "vllm": "OPENAI_API_KEY",
                "local": "OPENAI_API_KEY",
            }
            env_key = provider_key_map.get(provider)
            if env_key:
                api_key = os.environ.get(env_key)

        models = cls._parse_models_env()
        agents = cls._parse_agents_env()

        # If no model pool but legacy provider is set, synthesize a default model entry.
        if not models and provider != "default":
            models = [ModelConfig(
                alias="default",
                provider=provider,  # type: ignore[arg-type]
                model=os.environ.get("MATSCI_MODEL"),
                api_key=api_key,
                base_url=os.environ.get("MATSCI_BASE_URL"),
            )]

        # If no agent profiles, synthesize a default lead agent pointing at the default/only model.
        if not agents:
            model_alias = models[0].alias if models else "default"
            agents = [AgentProfileConfig(
                id="lead",
                name="Lead",
                model_alias=model_alias,
                persona=os.environ.get("MATSCI_PERSONA", "default").strip(),
            )]

        return cls(
            provider=provider,
            model=os.environ.get("MATSCI_MODEL"),
            api_key=api_key,
            base_url=os.environ.get("MATSCI_BASE_URL"),
            models=models,
            agents=agents,
            team_mode_enabled=os.environ.get("MATSCI_TEAM_MODE", "").lower() == "true",
            max_concurrent_subagents=int(os.environ.get("MATSCI_MAX_CONCURRENT_SUBAGENTS", "3")),
            ollama_host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
            vasp_executable=os.environ.get("VASP_EXECUTABLE"),
            lammps_executable=os.environ.get("LAMMPS_EXECUTABLE"),
            hpc_scheduler=os.environ.get("HPC_SCHEDULER", "local"),
            hpc_host=os.environ.get("HPC_HOST"),
            hpc_username=os.environ.get("HPC_USERNAME"),
            abaqus_mcp_server=os.environ.get("ABAQUS_MCP_SERVER_PATH"),
            workspace=os.environ.get("MATSCI_WORKSPACE", "."),
            auto_approve=os.environ.get("MATSCI_AUTO_APPROVE", "").lower() == "true",
            enable_exploration=os.environ.get("MATSCI_ENABLE_EXPLORATION", "true").lower() == "true",
            max_parallel_branches=int(os.environ.get("MATSCI_MAX_BRANCHES", "5")),
            encryption_enabled=os.environ.get("MATSCI_ENCRYPTION_ENABLED", "").lower() == "true",
            encryption_password=os.environ.get("MATSCI_ENCRYPTION_PASSWORD") or None,
            encryption_key_file=os.environ.get("MATSCI_ENCRYPTION_KEY_FILE") or None,
            encrypt_rag_documents=os.environ.get("MATSCI_ENCRYPT_RAG_DOCS", "true").lower() == "true",
            encrypt_rag_metadata=os.environ.get("MATSCI_ENCRYPT_RAG_META", "true").lower() == "true",
            encrypt_config=os.environ.get("MATSCI_ENCRYPT_CONFIG", "").lower() == "true",
            persona=os.environ.get("MATSCI_PERSONA", "default").strip(),
            rag_enabled=os.environ.get("MATSCI_RAG_ENABLED", "").lower() == "true",
        )

    @staticmethod
    def _parse_models_env() -> list[ModelConfig]:
        raw = os.environ.get("MATSCI_MODELS", "")
        if not raw:
            return []
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return [ModelConfig(**item) for item in data]
        except Exception:
            pass
        return []

    @staticmethod
    def _parse_agents_env() -> list[AgentProfileConfig]:
        raw = os.environ.get("MATSCI_AGENTS", "")
        if not raw:
            return []
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return [AgentProfileConfig(**item) for item in data]
        except Exception:
            pass
        return []
    

    @staticmethod
    def resolve_key(raw: str | None) -> str | None:
        """Resolve a credential string supporting env: and keyring: prefixes.

        Supported formats:
        - "plain-secret"               -> returned as-is
        - "env:OPENAI_API_KEY"         -> read from environment variable
        - "keyring:matsci:openai"      -> read from OS keyring (requires 'keyring' package)
        - None                           -> None
        """
        if raw is None:
            return None
        raw = raw.strip()
        if raw.startswith("env:"):
            var_name = raw[4:]
            return os.environ.get(var_name) or None
        if raw.startswith("keyring:"):
            try:
                import keyring
            except ImportError:
                raise ImportError("pip install keyring")
            parts = raw[8:].split(":", 1)
            service = parts[0] if parts else "matsci"
            username = parts[1] if len(parts) > 1 else "default"
            return keyring.get_password(service, username)
        return raw

    @property
    def resolved_api_key(self) -> str | None:
        """Return the resolved API key (supports env:/keyring: prefixes)."""
        return self.resolve_key(self.api_key)

    def to_dict(self, mask_key: bool = True) -> dict[str, Any]:
        def mask(k: str | None) -> str | None:
            return "***" if mask_key and k else k

        return {
            "provider": self.provider,
            "model": self.model,
            "api_key": mask(self.api_key),
            "base_url": self.base_url,
            "models": [
                {
                    "alias": m.alias,
                    "provider": m.provider,
                    "model": m.model,
                    "api_key": mask(m.api_key),
                    "base_url": m.base_url,
                    "temperature": m.temperature,
                    "enabled": m.enabled,
                }
                for m in self.models
            ],
            "agents": [
                {
                    "id": a.id,
                    "name": a.name or a.id,
                    "model_alias": a.model_alias,
                    "persona": a.persona,
                    "tools": a.tools,
                    "enabled": a.enabled,
                    "max_steps": a.max_steps,
                }
                for a in self.agents
            ],
            "team_mode_enabled": self.team_mode_enabled,
            "max_concurrent_subagents": self.max_concurrent_subagents,
            "ollama_host": self.ollama_host,
            "vasp_executable": self.vasp_executable,
            "lammps_executable": self.lammps_executable,
            "hpc_scheduler": self.hpc_scheduler,
            "workspace": self.workspace,
            "auto_approve": self.auto_approve,
            "enable_exploration": self.enable_exploration,
            "max_parallel_branches": self.max_parallel_branches,
            "encryption_enabled": self.encryption_enabled,
            "encryption_password": self.encryption_password,
            "encryption_key_file": self.encryption_key_file,
            "encrypt_rag_documents": self.encrypt_rag_documents,
            "encrypt_rag_metadata": self.encrypt_rag_metadata,
            "encrypt_config": self.encrypt_config,
            "persona": self.persona,
            "rag_enabled": self.rag_enabled,
            "privacy_redact_secrets": self.privacy_redact_secrets,
            "privacy_block_on_secrets": self.privacy_block_on_secrets,
        }

    @classmethod
    def from_dict(cls, data: dict) -> MatSciConfig:
        """Restore config from a plain dict."""
        models_raw = data.get("models")
        agents_raw = data.get("agents")
        kwargs = {k: v for k, v in data.items() if k not in ("models", "agents")}

        if models_raw is not None:
            kwargs["models"] = [ModelConfig(**m) for m in models_raw if isinstance(m, dict)]
        if agents_raw is not None:
            kwargs["agents"] = [AgentProfileConfig(**a) for a in agents_raw if isinstance(a, dict)]

        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in kwargs.items() if k in known}
        return cls(**filtered)

    def save(self, path: str | pathlib.Path, format: Literal["toml", "json"] = "toml") -> None:
        """Persist configuration to a file.

        API keys are written in plain text; ensure the file has restricted permissions.
        """
        target = pathlib.Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict(mask_key=False)
        if format == "toml":
            try:
                import toml
            except ImportError:
                raise ImportError("pip install toml")
            target.write_text(toml.dumps(data), encoding="utf-8")
        else:
            import json
            target.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | pathlib.Path, format: Literal["toml", "json"] | None = None) -> MatSciConfig:
        """Load configuration from a file."""
        target = pathlib.Path(path)
        if not target.exists():
            raise FileNotFoundError(f"Config file not found: {target}")
        fmt = format or ("toml" if target.suffix in (".toml", ".tml") else "json")
        text = target.read_text(encoding="utf-8")
        if fmt == "toml":
            try:
                import toml
            except ImportError:
                raise ImportError("pip install toml")
            data = toml.loads(text)
        else:
            import json
            data = json.loads(text)
        return cls.from_dict(data)


@dataclass
class CoderSettings:
    """Settings for autonomous coder mode."""

    max_iterations: int = 20
    done_marker: str = "[DONE]"

    @classmethod
    def from_env(cls) -> "CoderSettings":
        """Load coder settings from environment variables."""
        return cls(
            max_iterations=int(os.environ.get("MATSCI_CODER_MAX_ITER", "20")),
            done_marker=os.environ.get("MATSCI_CODER_DONE_MARKER", "[DONE]"),
        )


@dataclass
class Settings:
    """Top-level application settings."""

    config: MatSciConfig = field(default_factory=MatSciConfig.from_env)
    coder: CoderSettings = field(default_factory=CoderSettings.from_env)


def get_settings() -> Settings:
    """Load application settings from environment."""
    return Settings()
